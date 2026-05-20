"""
train.py - Validacion completa de Agent J (ETH 1D rescaled-A).

4 capas de validacion (obligatorias por el prompt):
  1. Real ETH 2020-2025: WF 12 semestres con purga >=2 semanas + bootstrap p
  2. 20 series sinteticas via block bootstrap del ETH 1D (bloques 30 dias)
  3. Null hypothesis: shuffle aleatorio -> debe dar ~0%
  4. Comparativa con ETH-A 4h (p=0.103, +11.3% annual)

Uso:
  C:/Python/python.exe experiments/agent_J/train.py
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from strategy import (
    PARAMS, prepare_data, signal, simulate, run_backtest, metrics
)

ROOT = HERE.parent.parent
DATA = ROOT / 'data'


# =============================================================================
# CARGA DE DATOS
# =============================================================================
def load_eth_1d(cutoff: str = '2025-12-31') -> pd.DataFrame:
    df = pd.read_parquet(DATA / 'ETH_USDT_1d_history.parquet')
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    cut = pd.Timestamp(cutoff)
    df = df[df.index <= cut].sort_index()
    assert df.index.max() <= cut, "VIOLACION cutoff"
    return df[['open', 'high', 'low', 'close', 'volume']].copy()


# =============================================================================
# WALK-FORWARD
# =============================================================================
WF_SEMESTERS = [
    ('2020-01-01', '2020-06-30'),
    ('2020-07-01', '2020-12-31'),
    ('2021-01-01', '2021-06-30'),
    ('2021-07-01', '2021-12-31'),
    ('2022-01-01', '2022-06-30'),
    ('2022-07-01', '2022-12-31'),
    ('2023-01-01', '2023-06-30'),
    ('2023-07-01', '2023-12-31'),
    ('2024-01-01', '2024-06-30'),
    ('2024-07-01', '2024-12-31'),
    ('2025-01-01', '2025-06-30'),
    ('2025-07-01', '2025-12-31'),
]
PURGE_DAYS = 14


def walk_forward(df_features: pd.DataFrame, params: dict) -> dict:
    fold_results = []
    for start_s, end_s in WF_SEMESTERS:
        start = pd.Timestamp(start_s)
        end = pd.Timestamp(end_s)
        purge_until = start + pd.Timedelta(days=PURGE_DAYS)

        in_window = (df_features.index >= start) & (df_features.index <= end)
        idxs = np.where(in_window)[0]
        # Para 1D: ~180 dias en semestre, exigir minimo razonable
        if len(idxs) < 100:
            fold_results.append({'period': start_s[:7], 'n': 0, 'pf': 0.0,
                                 'wr': 0.0, 'total': 0.0, 'monthly': 0.0,
                                 'max_dd': 0.0, 'ok': False,
                                 'status': 'no_data', 'no_data': True})
            continue
        start_i, end_i = int(idxs[0]), int(idxs[-1] + 1)
        trades_all = run_backtest(df_features, params, start_i=start_i, end_i=end_i)
        trades = [t for t in trades_all
                  if pd.Timestamp(t['entry_ts']) >= purge_until]
        m = metrics(trades)
        if m['n'] == 0:
            status = 'no_signal'
            ok = False
        elif m['n'] < 3:
            status = 'small_sample'
            ok = False
        else:
            status = 'evaluated'
            ok = (np.isfinite(m['pf']) and m['pf'] >= 1.2
                  and m['total_return'] > 0)
        fold_results.append({
            'period': start_s[:7], 'n': m['n'], 'pf': m['pf'],
            'wr': m['wr'], 'total': m['total_return'],
            'monthly': m['monthly_return'], 'max_dd': m['max_dd'],
            'ok': ok, 'status': status, 'no_data': False,
        })
    folds_ok = sum(1 for r in fold_results if r['ok'])
    folds_eval = sum(1 for r in fold_results
                     if not r['no_data'] and r.get('status') == 'evaluated')
    folds_total = sum(1 for r in fold_results if not r['no_data'])
    return {'folds': fold_results, 'folds_ok': folds_ok,
            'folds_evaluated': folds_eval, 'folds_total': folds_total}


# =============================================================================
# BOOTSTRAP DE SIGNIFICANCIA
# =============================================================================
def bootstrap_pvalue(trades: list, n_iter: int = 3000, seed: int = 42) -> dict | None:
    if len(trades) < 5:
        return None
    rng = np.random.default_rng(seed)
    pnls = np.array([t['pnl_pct'] for t in trades])
    k = len(pnls)
    totals = np.empty(n_iter)
    for j in range(n_iter):
        sample = rng.choice(pnls, size=k, replace=True)
        totals[j] = float(np.prod(1.0 + sample) - 1.0)
    p_value = float(np.mean(totals <= 0))
    return {'p_value': p_value,
            'pctl_5': float(np.percentile(totals, 5)),
            'pctl_50': float(np.percentile(totals, 50)),
            'pctl_95': float(np.percentile(totals, 95)),
            'n_iter': n_iter, 'n_trades': k}


# =============================================================================
# SINTETICOS via block bootstrap del ETH 1D real
# =============================================================================
def block_bootstrap_ohlcv_1d(df: pd.DataFrame, block_size: int = 30,
                             n_bars: int | None = None, seed: int = 0) -> pd.DataFrame:
    """
    Bloque-bootstrap para 1D: bloques de 30 dias (mes calendario).
    Preserva micro-estructura intra-bloque, randomiza orden de bloques.
    Re-escala cada bloque para empalmar cierres.
    """
    if n_bars is None:
        n_bars = len(df)
    rng = np.random.default_rng(seed)
    max_start = len(df) - block_size
    if max_start <= 0:
        raise ValueError("block_size demasiado grande")
    n_blocks = (n_bars // block_size) + 2
    starts = rng.integers(0, max_start, size=n_blocks)
    parts = []
    last_close = float(df['close'].iloc[0])
    for s in starts:
        block = df.iloc[int(s):int(s) + block_size].copy()
        scale = last_close / float(block['open'].iloc[0])
        block[['open', 'high', 'low', 'close']] *= scale
        parts.append(block)
        last_close = float(block['close'].iloc[-1])
    out = pd.concat(parts).iloc[:n_bars].copy()
    out.index = pd.date_range(start=df.index[0], periods=len(out), freq='1D')
    return out


def shuffle_returns_1d(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """
    Null hypothesis: permuta log-returns -> destruye estructura temporal,
    preserva distribucion marginal.
    """
    rng = np.random.default_rng(seed)
    df_c = df.copy()
    log_ret = np.log(df_c['close'].values[1:] / df_c['close'].values[:-1])
    perm = rng.permutation(len(log_ret))
    log_ret_shuf = log_ret[perm]
    new_close = np.empty(len(df_c))
    new_close[0] = df_c['close'].iloc[0]
    new_close[1:] = df_c['close'].iloc[0] * np.exp(np.cumsum(log_ret_shuf))
    # Preservar ratios h/l/o relativos al close
    ratio_h_c = (df_c['high'] / df_c['close']).values[1:][perm]
    ratio_l_c = (df_c['low'] / df_c['close']).values[1:][perm]
    ratio_o_c = (df_c['open'] / df_c['close']).values[1:][perm]
    vol_shuf = df_c['volume'].values[1:][perm]
    out = pd.DataFrame({
        'open':  np.r_[df_c['open'].iloc[0], new_close[1:] * ratio_o_c],
        'high':  np.r_[df_c['high'].iloc[0], new_close[1:] * ratio_h_c],
        'low':   np.r_[df_c['low'].iloc[0], new_close[1:] * ratio_l_c],
        'close': new_close,
        'volume': np.r_[df_c['volume'].iloc[0], vol_shuf],
    }, index=df_c.index)
    return out


def run_J_on_synth(df_synth: pd.DataFrame, params: dict = PARAMS) -> dict:
    """Corre J sobre un DF sintetico/null y devuelve metrics."""
    p = dict(params)
    p['cutoff_date'] = '2099-01-01'  # no cortar el sintetico
    df_feat = prepare_data(df_synth, p)
    trades = run_backtest(df_feat, p)
    return metrics(trades)


# =============================================================================
# MAIN
# =============================================================================
def fmt_pf(pf):
    return 'inf' if not np.isfinite(pf) else f'{pf:.2f}'


def main():
    print('=' * 78)
    print('AGENT J - ETH 1D RESCALED-A (Donchian-10 + EMA50/200 daily + ATR x 2.5)')
    print('=' * 78)

    print('\n[1/4] CARGA DE DATOS')
    df_eth = load_eth_1d(cutoff=PARAMS['cutoff_date'])
    print(f'  ETH 1D: {len(df_eth)} bars, '
          f'{df_eth.index.min().date()} -> {df_eth.index.max().date()}')

    print('\n[2/4] PREPARACION DE FEATURES')
    df_feat = prepare_data(df_eth, PARAMS)
    print(f'  features ready: {len(df_feat)} bars, '
          f'{df_feat.index.min().date()} -> {df_feat.index.max().date()}')
    bull_frac = df_feat['bull_1d'].mean()
    print(f'  fraccion de bars con bull_1d=1: {bull_frac:.1%}')

    # =====================================================================
    # CAPA 1: Walk-forward en real ETH
    # =====================================================================
    print('\n' + '=' * 78)
    print(f'CAPA 1 - WALK-FORWARD ETH 1D ({len(WF_SEMESTERS)} semestres, '
          f'purga {PURGE_DAYS}d)')
    print('=' * 78)
    wf = walk_forward(df_feat, PARAMS)
    print(f'\nFolds OK (PF>=1.2, total>0, n>=3): '
          f'{wf["folds_ok"]}/{wf["folds_evaluated"]} evaluados '
          f'({wf["folds_total"]} con datos)')
    pf_eval = [f['pf'] for f in wf['folds']
               if f.get('status') == 'evaluated' and np.isfinite(f['pf'])]
    if pf_eval:
        print(f'PF mediano (folds evaluados): {np.median(pf_eval):.2f}')
        print(f'PF medio (folds evaluados):   {np.mean(pf_eval):.2f}')
    print(f'\n  {"period":<10} {"n":>4} {"wr":>6} {"pf":>7} {"total":>9} '
          f'{"monthly":>9} {"dd":>7} ok')
    for f in wf['folds']:
        if f['no_data']:
            print(f"  {f['period']:<10} sin datos")
            continue
        st = f.get('status', '?')
        if st == 'no_signal':
            flag = '[no-signal]'
        elif st == 'small_sample':
            flag = '[small-n]'
        else:
            flag = '[+]' if f['ok'] else '[-]'
        print(f"  {f['period']:<10} {f['n']:>4} {f['wr']*100:>5.1f}% "
              f"{fmt_pf(f['pf']):>7} {f['total']*100:>+8.1f}% "
              f"{f['monthly']*100:>+8.2f}% {f['max_dd']*100:>6.1f}% {flag}")

    # Backtest global + bootstrap
    print('\nBacktest GLOBAL 2020-01-01 -> cutoff...')
    start_full = pd.Timestamp('2020-01-01')
    idxs = np.where(df_feat.index >= start_full)[0]
    start_i = int(idxs[0]) if len(idxs) else 0
    all_trades = run_backtest(df_feat, PARAMS, start_i=start_i, end_i=len(df_feat))
    M = metrics(all_trades)
    print(f"  N={M['n']}  WR={M['wr']*100:.1f}%  PF={fmt_pf(M['pf'])}  "
          f"total={M['total_return']*100:+.1f}%  "
          f"annual={M['annual_return']*100:+.2f}%  "
          f"DD={M['max_dd']*100:.1f}%  months={M['months']:.1f}")

    print('\nBootstrap (3000 iter)...')
    boot = bootstrap_pvalue(all_trades, n_iter=3000)
    if boot is None:
        print('  insuficientes trades')
    else:
        sig = 'SIGNIFICATIVO' if boot['p_value'] < 0.05 else 'NO significativo'
        print(f"  p-value: {boot['p_value']:.4f} -> {sig}")
        print(f"  retorno mediano resampled: {boot['pctl_50']*100:+.1f}%")
        print(f"  p5 / p95: {boot['pctl_5']*100:+.1f}% / "
              f"{boot['pctl_95']*100:+.1f}%")

    # =====================================================================
    # CAPA 2: 20 series sinteticas
    # =====================================================================
    N_SYNTH = 20
    print('\n' + '=' * 78)
    print(f'CAPA 2 - {N_SYNTH} SERIES SINTETICAS (block bootstrap, bloques 30d)')
    print('=' * 78)
    synth = []
    for seed in range(N_SYNTH):
        df_s = block_bootstrap_ohlcv_1d(df_eth, block_size=30, seed=seed)
        m = run_J_on_synth(df_s, PARAMS)
        synth.append(m)
        print(f"  serie {seed:2d}: N={m['n']:3d}  WR={m['wr']:.1%}  "
              f"PF={fmt_pf(m['pf']):>6}  annual={m['annual_return']*100:+6.1f}%  "
              f"DD={m['max_dd']*100:5.1f}%")
    annuals = [r['annual_return'] for r in synth]
    pfs = [r['pf'] for r in synth if np.isfinite(r['pf']) and r['n'] >= 5]
    print(f'\n  Annual distribucion:')
    print(f'    mediana = {np.median(annuals)*100:+.1f}%')
    print(f'    media   = {np.mean(annuals)*100:+.1f}%')
    print(f'    p25-p75 = [{np.percentile(annuals,25)*100:+.1f}%, '
          f'{np.percentile(annuals,75)*100:+.1f}%]')
    print(f'    p5-p95  = [{np.percentile(annuals,5)*100:+.1f}%, '
          f'{np.percentile(annuals,95)*100:+.1f}%]')
    n_pos = sum(1 for a in annuals if a > 0)
    n_pos_10 = sum(1 for a in annuals if a > 0.10)
    print(f'    sintéticas con annual>0:   {n_pos}/{N_SYNTH}')
    print(f'    sintéticas con annual>10%: {n_pos_10}/{N_SYNTH}')
    inside = np.percentile(annuals, 5) <= M['annual_return'] <= np.percentile(annuals, 95)
    print(f'    Real ETH 1D ({M["annual_return"]*100:+.1f}%) '
          f'{"dentro" if inside else "FUERA"} de p5-p95')

    # =====================================================================
    # CAPA 3: Null hypothesis (shuffle)
    # =====================================================================
    print('\n' + '=' * 78)
    print('CAPA 3 - NULL HYPOTHESIS (shuffle de log-returns)')
    print('=' * 78)
    null_annuals = []
    null_n = []
    for seed in range(10):
        df_n = shuffle_returns_1d(df_eth, seed=seed)
        m = run_J_on_synth(df_n, PARAMS)
        null_annuals.append(m['annual_return'])
        null_n.append(m['n'])
        print(f"  seed {seed}: N={m['n']:3d}  WR={m['wr']:.1%}  "
              f"PF={fmt_pf(m['pf']):>6}  annual={m['annual_return']*100:+6.1f}%")
    print(f'\n  Null mediana annual: {np.median(null_annuals)*100:+.1f}%')
    print(f'  Null media annual:   {np.mean(null_annuals)*100:+.1f}%')
    edge = np.median(annuals) - np.median(null_annuals)
    print(f'  Edge real (sintetico - null): {edge*100:+.1f}%')

    # =====================================================================
    # CAPA 4: Comparativa con ETH-A 4h
    # =====================================================================
    print('\n' + '=' * 78)
    print('CAPA 4 - COMPARATIVA CON ETH-A 4h')
    print('=' * 78)
    eth_4h_ref = {
        'annual': 0.113, 'pf': 1.53, 'wr': 0.460, 'max_dd': 0.182,
        'bootstrap_p': 0.103, 'n_trades': 87,
    }
    print(f"  {'Metrica':<22} {'ETH-J 1D':>12} {'ETH-A 4h':>12} {'Delta':>10}")
    print(f"  {'N trades':<22} {M['n']:>12} {eth_4h_ref['n_trades']:>12} "
          f"{M['n']-eth_4h_ref['n_trades']:>+10}")
    print(f"  {'WR':<22} {M['wr']*100:>11.1f}% {eth_4h_ref['wr']*100:>11.1f}% "
          f"{(M['wr']-eth_4h_ref['wr'])*100:>+9.1f}pp")
    print(f"  {'PF':<22} {fmt_pf(M['pf']):>12} {eth_4h_ref['pf']:>12.2f} "
          f"{M['pf']-eth_4h_ref['pf']:>+10.2f}")
    print(f"  {'Annual':<22} {M['annual_return']*100:>11.1f}% "
          f"{eth_4h_ref['annual']*100:>11.1f}% "
          f"{(M['annual_return']-eth_4h_ref['annual'])*100:>+9.1f}pp")
    print(f"  {'Max DD':<22} {M['max_dd']*100:>11.1f}% "
          f"{eth_4h_ref['max_dd']*100:>11.1f}% "
          f"{(M['max_dd']-eth_4h_ref['max_dd'])*100:>+9.1f}pp")
    p_j = boot['p_value'] if boot else float('nan')
    print(f"  {'Bootstrap p':<22} {p_j:>12.3f} {eth_4h_ref['bootstrap_p']:>12.3f} "
          f"{p_j-eth_4h_ref['bootstrap_p']:>+10.3f}")

    # =====================================================================
    # VEREDICTO FINAL
    # =====================================================================
    print('\n' + '=' * 78)
    print('VEREDICTO FINAL')
    print('=' * 78)
    crit_p = boot is not None and boot['p_value'] < 0.05
    crit_med = np.median(annuals) > 0
    crit_pos = n_pos >= 14
    crit_edge = edge > 0.05
    print(f"  Bootstrap p<0.05 (real ETH):           "
          f"{'SI' if crit_p else 'NO'} "
          f"(p={boot['p_value']:.3f})" if boot else "  (no bootstrap)")
    print(f"  Mediana sintetico > 0:                 "
          f"{'SI' if crit_med else 'NO'} ({np.median(annuals)*100:+.1f}%)")
    print(f"  >=14/20 sinteticas positivas:          "
          f"{'SI' if crit_pos else 'NO'} ({n_pos}/20)")
    print(f"  Edge vs null > 5%:                     "
          f"{'SI' if crit_edge else 'NO'} ({edge*100:+.1f}%)")
    veredict = 'KEEP' if (crit_p and crit_med and crit_pos and crit_edge) else 'REJECT'
    print(f"\n  -> VEREDICTO: {veredict}")
    if veredict == 'REJECT':
        n_pass = sum([crit_p, crit_med, crit_pos, crit_edge])
        print(f"     ({n_pass}/4 criterios pasados — necesita 4/4 para KEEP)")

    # =====================================================================
    # GUARDAR RESULTADOS
    # =====================================================================
    summary = {
        'agent': 'J',
        'strategy_name': 'ETH 1D Donchian + EMA daily + ATR trailing (rescaled from 4h)',
        'timeframe': '1D',
        'params': PARAMS,
        'rescaling_decisions': {
            'donchian_n': '4h:55 (~9d) -> 1D:10 (~10d) [proporcional al horizonte]',
            'max_bars': '4h:60 (10d) -> 1D:10 (10d) [mismo techo temporal]',
            'atr_n': '14 igual (Wilder, conceptual)',
            'vol_ma_n': '20 igual (concepto MA mensual)',
            'ema_1d': '50/200 igual (ya son daily)',
            'trail_atr_mult': '2.5 igual (constante universal)',
            'trail_floor/ceiling': '2.5%/6% igual (sin tunear)',
            'funding': 'DESACTIVADO (no hay ETH funding parquet)',
        },
        'wf': {'folds_ok': wf['folds_ok'],
               'folds_evaluated': wf['folds_evaluated'],
               'folds_total': wf['folds_total'],
               'folds': wf['folds']},
        'overall': M,
        'bootstrap': boot,
        'synth': {
            'n': N_SYNTH,
            'annuals': annuals,
            'median_annual': float(np.median(annuals)),
            'mean_annual': float(np.mean(annuals)),
            'n_positive': n_pos,
            'n_above_10pct': n_pos_10,
        },
        'null': {
            'n': len(null_annuals),
            'annuals': null_annuals,
            'median_annual': float(np.median(null_annuals)),
            'mean_annual': float(np.mean(null_annuals)),
        },
        'edge_vs_null': float(edge),
        'comparison_eth_4h': {
            'eth_a_4h': eth_4h_ref,
            'eth_j_1d': {
                'n_trades': M['n'], 'wr': M['wr'], 'pf': M['pf'],
                'annual': M['annual_return'], 'max_dd': M['max_dd'],
                'bootstrap_p': boot['p_value'] if boot else None,
            },
        },
        'criteria': {
            'bootstrap_p_lt_005': bool(crit_p),
            'synth_median_positive': bool(crit_med),
            'synth_14_of_20_positive': bool(crit_pos),
            'edge_vs_null_gt_5pct': bool(crit_edge),
        },
        'veredict': veredict,
        'cutoff': PARAMS['cutoff_date'],
    }
    out_json = HERE / 'results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, default=str, indent=2)
    print(f'\nResultados guardados en {out_json}')
    return summary


if __name__ == '__main__':
    main()
