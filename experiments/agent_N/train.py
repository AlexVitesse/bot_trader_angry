"""
train.py — Validacion Agent N (SOL vol-compression breakout WIDE trail).

Capas obligatorias:
  1. Real SOL: WF + bootstrap
  2. 20 sintéticas (block bootstrap)
  3. Null hypothesis (shuffle returns)
  4. Cross-check: trail tight 0.8% vs wide 2.5%

Uso:
  C:/Python/python.exe experiments/agent_N/train.py
"""
from __future__ import annotations
import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from strategy import (
    PARAMS, PARAMS_TIGHT, prepare_data, signal, simulate, run_backtest,
    metrics,
)

ROOT = HERE.parent.parent
DATA = ROOT / 'data'


# =============================================================================
# CARGA
# =============================================================================
def _load_one(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df[df.index <= cutoff].sort_index()
    return df


def load_data(cutoff: str = '2025-12-31') -> dict:
    print(f'Cargando datos (cutoff <= {cutoff})...')
    cut = pd.Timestamp(cutoff, tz='UTC')

    sol_4h = _load_one(DATA / 'SOL_USDT_4h_full.parquet', cut)
    btc_1d = _load_one(DATA / 'btcusdt_1d_v15.parquet', cut)

    print(f'  SOL 4h: {len(sol_4h)} bars, '
          f'{sol_4h.index.min().date()} -> {sol_4h.index.max().date()}')
    print(f'  BTC 1d: {len(btc_1d)} bars')

    assert sol_4h.index.max() <= cut, "VIOLACION cutoff SOL 4h"
    assert btc_1d.index.max() <= cut, "VIOLACION cutoff BTC 1d"
    return {'sol_4h': sol_4h, 'btc_1d': btc_1d}


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


def walk_forward(df_sol: pd.DataFrame, params: dict) -> dict:
    fold_results = []
    for start_s, end_s in WF_SEMESTERS:
        start = pd.Timestamp(start_s, tz='UTC')
        end = pd.Timestamp(end_s, tz='UTC')
        purge_until = start + pd.Timedelta(days=PURGE_DAYS)

        all_trades = run_backtest(df_sol, params, start, end)
        trades = [t for t in all_trades
                  if pd.Timestamp(t['entry_ts']) >= purge_until]
        m = metrics(trades)

        if m['n'] == 0:
            status, ok = 'no_signal', False
        elif m['n'] < 5:
            status, ok = 'small_sample', False
        else:
            ok = (np.isfinite(m['pf']) and m['pf'] >= 1.2
                  and m['total_return'] > 0)
            status = 'evaluated'

        fold_results.append({
            'period': start_s[:7], 'n': m['n'], 'pf': m['pf'],
            'wr': m['wr'], 'total': m['total_return'],
            'monthly': m['monthly_return'], 'max_dd': m['max_dd'],
            'status': status, 'ok': ok,
        })
    # Folds con datos = folds donde hay velas SOL en el periodo
    folds_with_data = sum(1 for f in fold_results if f['status'] != 'no_signal')
    folds_ok = sum(1 for f in fold_results if f['ok'])
    return {
        'folds': fold_results,
        'folds_ok': folds_ok,
        'folds_total': len(fold_results),
        'folds_with_data': folds_with_data,
    }


# =============================================================================
# BOOTSTRAP
# =============================================================================
def bootstrap_pvalue(trades: list, n_iter: int = 3000, seed: int = 42):
    if len(trades) < 5:
        return None
    rng = np.random.default_rng(seed)
    pnls = np.array([t['pnl_pct'] for t in trades])
    k = len(pnls)
    totals = np.empty(n_iter)
    for j in range(n_iter):
        sample = rng.choice(pnls, size=k, replace=True)
        totals[j] = float(np.prod(1.0 + sample) - 1.0)
    return {'p_value': float(np.mean(totals <= 0)),
            'pctl_5': float(np.percentile(totals, 5)),
            'pctl_50': float(np.percentile(totals, 50)),
            'pctl_95': float(np.percentile(totals, 95)),
            'n_iter': n_iter, 'n_trades': k}


# =============================================================================
# BLOCK BOOTSTRAP de OHLCV (sintéticas)
# =============================================================================
def block_bootstrap_ohlcv(df, block_size=24, n_bars=None, seed=None):
    if n_bars is None:
        n_bars = len(df)
    rng = np.random.default_rng(seed)
    max_start = len(df) - block_size
    n_blocks = (n_bars // block_size) + 2
    starts = rng.integers(0, max_start, size=n_blocks)
    parts = []
    last_close = float(df['close'].iloc[0])
    for s in starts:
        block = df.iloc[s:s + block_size].copy()
        scale = last_close / float(block['open'].iloc[0])
        block[['open', 'high', 'low', 'close']] = block[['open', 'high', 'low', 'close']] * scale
        parts.append(block)
        last_close = float(block['close'].iloc[-1])
    out = pd.concat(parts).iloc[:n_bars].copy()
    out.index = pd.date_range(start=df.index[0], periods=len(out),
                              freq='4h', tz='UTC')
    return out


def shuffle_returns(df, seed=None):
    rng = np.random.default_rng(seed)
    df_c = df.copy()
    close = df_c['close'].values
    log_ret = np.log(close[1:] / close[:-1])
    perm = rng.permutation(len(log_ret))
    log_ret_shuf = log_ret[perm]
    new_close = np.empty(len(df_c))
    new_close[0] = close[0]
    new_close[1:] = close[0] * np.exp(np.cumsum(log_ret_shuf))
    ratio_h_c = (df_c['high'] / df_c['close']).values[1:][perm]
    ratio_l_c = (df_c['low'] / df_c['close']).values[1:][perm]
    ratio_o_c = (df_c['open'] / df_c['close']).values[1:][perm]
    vol_shuf = df_c['volume'].values[1:][perm]
    out = pd.DataFrame({
        'open':  np.r_[df_c['open'].iloc[0],  new_close[1:] * ratio_o_c],
        'high':  np.r_[df_c['high'].iloc[0],  new_close[1:] * ratio_h_c],
        'low':   np.r_[df_c['low'].iloc[0],   new_close[1:] * ratio_l_c],
        'close': new_close,
        'volume': np.r_[df_c['volume'].iloc[0], vol_shuf],
    }, index=df_c.index)
    return out


def run_full_pipeline(df_sol_4h, btc_1d, params):
    """Prepara y corre backtest completo. Devuelve trades + metricas."""
    sol_feat = prepare_data(df_sol_4h, btc_1d, params)
    start_full = pd.Timestamp('2020-01-01', tz='UTC')
    end_full = pd.Timestamp(params['cutoff_date'], tz='UTC')
    trades = run_backtest(sol_feat, params, start_full, end_full)
    m = metrics(trades)
    return trades, m, sol_feat


# =============================================================================
# MAIN
# =============================================================================
def fmt_pf(pf):
    return 'inf' if not np.isfinite(pf) else f'{pf:.2f}'


def split_by_side(trades):
    out = {}
    out['LONG'] = metrics([t for t in trades if t['side'] == 'LONG'])
    out['SHORT'] = metrics([t for t in trades if t['side'] == 'SHORT'])
    return out


def main():
    print('=' * 78)
    print('AGENT N — SOL VOL-COMPRESSION BREAKOUT, WIDE TRAIL (scaled to SOL vol)')
    print('=' * 78)

    data = load_data(cutoff=PARAMS['cutoff_date'])

    print('\n--- WIDE TRAIL (PARAMS principal) ---')
    print(f'  trail_atr_factor={PARAMS["trail_atr_factor"]}, '
          f'floor={PARAMS["trail_floor_pct"]}, '
          f'ceil={PARAMS["trail_ceiling_pct"]}')

    print('\nPreparando features SOL...')
    sol_feat = prepare_data(data['sol_4h'], data['btc_1d'], PARAMS)
    print(f'  SOL features: {len(sol_feat)} bars, '
          f'{sol_feat.index.min().date()} -> {sol_feat.index.max().date()}')

    # Vol diagnostics
    atr_mean = sol_feat['atr_pct'].mean() * 100
    bb_mean = sol_feat['bb_width'].mean() * 100
    n_comp = int(sol_feat['compression_sustained'].sum())
    print(f'\nDiagnostico SOL:')
    print(f'  ATR% mean: {atr_mean:.2f}%  (BTC ~2%, ETH ~2.4%)')
    print(f'  BB width mean: {bb_mean:.2f}%  (BTC ~7%, ETH ~7%)')
    print(f'  Bars en compresion sostenida: {n_comp} ({n_comp/len(sol_feat)*100:.1f}%)')

    # WF
    print(f'\nWalk-forward {len(WF_SEMESTERS)} semestres (purga {PURGE_DAYS}d)...')
    wf = walk_forward(sol_feat, PARAMS)
    fwd = wf['folds_with_data']
    print(f'  Folds con datos: {fwd}/{wf["folds_total"]}')
    print(f'  Folds OK: {wf["folds_ok"]}/{fwd} (folds con datos)')
    print(f'\n  {"period":<10} {"n":>4} {"wr":>6} {"pf":>7} {"total":>9} {"dd":>7} ok')
    for f in wf['folds']:
        flag = '[+]' if f['ok'] else ('[no-sig]' if f['status'] == 'no_signal'
                                       else ('[small]' if f['status'] == 'small_sample' else '[-]'))
        print(f"  {f['period']:<10} {f['n']:>4} {f['wr']*100:>5.1f}% "
              f"{fmt_pf(f['pf']):>7} {f['total']*100:>+8.1f}% "
              f"{f['max_dd']*100:>6.1f}% {flag}")

    # Backtest global
    print('\nBacktest global 2020-01-01 -> cutoff...')
    start_full = pd.Timestamp('2020-01-01', tz='UTC')
    end_full = pd.Timestamp(PARAMS['cutoff_date'], tz='UTC')
    all_trades = run_backtest(sol_feat, PARAMS, start_full, end_full)
    M = metrics(all_trades)
    print(f"  N={M['n']}  WR={M['wr']*100:.1f}%  PF={fmt_pf(M['pf'])}  "
          f"total={M['total_return']*100:+.1f}%  "
          f"monthly={M['monthly_return']*100:+.2f}%  "
          f"annual={M['annual_return']*100:+.1f}%  "
          f"DD={M['max_dd']*100:.1f}%  "
          f"sharpe-like={M['sharpe_like']:.2f}  "
          f"months={M['months']:.1f}")

    # Por side
    print('\nMetricas por dir:')
    side_brk = split_by_side(all_trades)
    for k, v in side_brk.items():
        if v['n'] == 0:
            print(f"  {k:<8} N=0")
            continue
        print(f"  {k:<8} N={v['n']:>3}  WR={v['wr']*100:>5.1f}%  PF={fmt_pf(v['pf']):>6}  "
              f"total={v['total_return']*100:>+7.1f}%  avg_pnl={v['avg_pnl']*100:>+6.2f}%")

    # Bootstrap
    print('\nBootstrap (3000 iter)...')
    boot = bootstrap_pvalue(all_trades, n_iter=3000)
    if boot is None:
        print('  insuficientes trades')
    else:
        sig = 'SIGNIFICATIVO' if boot['p_value'] < 0.05 else 'NO significativo'
        print(f"  p-value(retorno<=0): {boot['p_value']:.4f} -> {sig}")
        print(f"  retorno mediano resampled: {boot['pctl_50']*100:+.1f}%")
        print(f"  retorno p5: {boot['pctl_5']*100:+.1f}%   p95: {boot['pctl_95']*100:+.1f}%")

    # =================================================================
    # CAPA 2 — 20 sinteticas con block bootstrap
    # =================================================================
    print('\n--- CAPA 2: 20 sinteticas (block bootstrap) ---')
    N_SYNTH = 20
    synth_results = []
    sol_raw = data['sol_4h'].copy()
    if sol_raw.index.tz is None:
        sol_raw.index = sol_raw.index.tz_localize('UTC')

    for seed in range(N_SYNTH):
        df_s = block_bootstrap_ohlcv(sol_raw, block_size=24, seed=seed)
        # No tenemos un BTC 1d pareado para sintéticas; usamos derivacion 4h->1d
        # del propio sintético (consistente con la metodología de F sin df_1d).
        trades, m, _ = run_full_pipeline(df_s, None, PARAMS)
        synth_results.append({'seed': seed, **m})
        print(f"  serie {seed:2d}: N={m['n']:3d}  WR={m['wr']*100:>5.1f}%  "
              f"PF={fmt_pf(m['pf']):>6}  annual={m['annual_return']*100:+6.1f}%  "
              f"DD={m['max_dd']*100:>5.1f}%")

    annuals = [r['annual_return'] for r in synth_results]
    print(f"\n  Distribucion annual sinteticas:")
    print(f"    mediana = {np.median(annuals)*100:+.1f}%")
    print(f"    media   = {np.mean(annuals)*100:+.1f}%")
    print(f"    p25-p75 = [{np.percentile(annuals,25)*100:+.1f}%, "
          f"{np.percentile(annuals,75)*100:+.1f}%]")
    print(f"    p5-p95  = [{np.percentile(annuals,5)*100:+.1f}%, "
          f"{np.percentile(annuals,95)*100:+.1f}%]")
    n_pos = sum(1 for a in annuals if a > 0)
    n_pos_10 = sum(1 for a in annuals if a > 0.10)
    print(f"    # series annual > 0:    {n_pos}/{N_SYNTH}")
    print(f"    # series annual > 10%:  {n_pos_10}/{N_SYNTH}")

    # =================================================================
    # CAPA 3 — Null hypothesis (shuffle returns)
    # =================================================================
    print('\n--- CAPA 3: Null hypothesis (shuffle returns) ---')
    null_results = []
    for seed in range(10):
        df_null = shuffle_returns(sol_raw, seed=seed)
        trades, m, _ = run_full_pipeline(df_null, None, PARAMS)
        null_results.append(m)
        print(f"  seed {seed}: N={m['n']:3d}  annual={m['annual_return']*100:+.1f}%  "
              f"WR={m['wr']*100:.1f}%")
    null_annuals = [r['annual_return'] for r in null_results]
    print(f"\n  Mediana null: {np.median(null_annuals)*100:+.1f}%")
    edge_vs_null = np.median(annuals) - np.median(null_annuals)
    print(f"  Edge sintetico vs null: {edge_vs_null*100:+.1f}%")

    # =================================================================
    # CAPA 4 — Cross-check tight vs wide
    # =================================================================
    print('\n--- CAPA 4: Tight (0.8%) vs Wide (2.5%) — cross-check ---')

    sol_feat_tight = prepare_data(data['sol_4h'], data['btc_1d'], PARAMS_TIGHT)
    trades_tight = run_backtest(sol_feat_tight, PARAMS_TIGHT, start_full, end_full)
    M_tight = metrics(trades_tight)
    boot_tight = bootstrap_pvalue(trades_tight, n_iter=3000)
    p_t = boot_tight['p_value'] if boot_tight else None

    print(f"\n  Tight: trail_atr_factor={PARAMS_TIGHT['trail_atr_factor']}, "
          f"floor={PARAMS_TIGHT['trail_floor_pct']}")
    print(f"    N={M_tight['n']}  WR={M_tight['wr']*100:.1f}%  "
          f"PF={fmt_pf(M_tight['pf'])}  annual={M_tight['annual_return']*100:+.1f}%  "
          f"DD={M_tight['max_dd']*100:.1f}%  p={p_t:.3f}" if p_t else
          f"    N={M_tight['n']}  WR={M_tight['wr']*100:.1f}%  "
          f"PF={fmt_pf(M_tight['pf'])}  annual={M_tight['annual_return']*100:+.1f}%  "
          f"DD={M_tight['max_dd']*100:.1f}%")

    p_w = boot['p_value'] if boot else None
    print(f"\n  Wide:  trail_atr_factor={PARAMS['trail_atr_factor']}, "
          f"floor={PARAMS['trail_floor_pct']}")
    print(f"    N={M['n']}  WR={M['wr']*100:.1f}%  "
          f"PF={fmt_pf(M['pf'])}  annual={M['annual_return']*100:+.1f}%  "
          f"DD={M['max_dd']*100:.1f}%  p={p_w:.3f}" if p_w else
          f"    N={M['n']}  WR={M['wr']*100:.1f}%  "
          f"PF={fmt_pf(M['pf'])}  annual={M['annual_return']*100:+.1f}%  "
          f"DD={M['max_dd']*100:.1f}%")

    # SELF-AUDIT
    print('\nSELF-AUDIT:')
    # 1) no solape
    overlap_violations = 0
    for i in range(1, len(all_trades)):
        if pd.Timestamp(all_trades[i]['entry_ts']) < pd.Timestamp(all_trades[i - 1]['exit_ts']):
            overlap_violations += 1
    print(f'  Trades solapados (deberia ser 0): {overlap_violations}')
    print(f'  Cutoff respetado: {sol_feat.index.max()} <= {PARAMS["cutoff_date"]}')

    # Sanity
    print('\nSANITY CHECKS:')
    flags = []
    if M['n'] > 0 and M['pf'] > 4:
        flags.append(f"PF {M['pf']:.2f} > 4 -> sospechoso")
    if M['n'] > 0 and M['wr'] > 0.65:
        flags.append(f"WR {M['wr']*100:.1f}% > 65% -> sospechoso")
    if M['n'] > 0 and M['max_dd'] < 0.05:
        flags.append(f"DD {M['max_dd']*100:.1f}% < 5% -> sospechoso")
    if flags:
        for f in flags:
            print(f'  [!] {f}')
    else:
        print('  [+] PF/WR/DD razonables')

    # VEREDICTO
    print('\n' + '=' * 70)
    print('VEREDICTO Agent N — SOL vol breakout WIDE trail')
    print('=' * 70)
    sig_real = (boot is not None) and (boot['p_value'] < 0.05)
    med_pos = np.median(annuals) > 0
    n_pos_ok = n_pos >= 14
    edge_real = edge_vs_null > 0.05
    fwd = wf['folds_with_data']
    wf_ok = wf['folds_ok'] >= max(6, int(0.6 * fwd))
    print(f"  WF: {wf['folds_ok']}/{fwd} folds con datos {'[OK]' if wf_ok else '[NO]'}")
    print(f"  Bootstrap p<0.05 real:                {'SI' if sig_real else 'NO'} "
          f"(p={boot['p_value']:.3f})" if boot else "  Boot: insufficient trades")
    print(f"  Mediana sintetica > 0:                {'SI' if med_pos else 'NO'} "
          f"({np.median(annuals)*100:+.1f}%)")
    print(f"  >=14/20 sintéticas positivas:         {'SI' if n_pos_ok else 'NO'} "
          f"({n_pos}/{N_SYNTH})")
    print(f"  Edge vs null > 5%:                    {'SI' if edge_real else 'NO'} "
          f"({edge_vs_null*100:+.1f}%)")

    all_pass = sig_real and med_pos and n_pos_ok and edge_real and wf_ok
    if all_pass:
        print('\n  -> KEEP: SOL vol breakout WIDE trail tiene edge demostrable.')
    else:
        print('\n  -> REJECT: alguna capa falla, no es edge real.')

    # GUARDAR
    summary = {
        'agent': 'N',
        'strategy_name': 'SOL vol breakout WIDE trail (scaled to SOL volatility)',
        'params': PARAMS,
        'params_tight': PARAMS_TIGHT,
        'wf': {
            'folds_ok': wf['folds_ok'],
            'folds_total': wf['folds_total'],
            'folds_with_data': fwd,
            'folds': wf['folds'],
        },
        'overall': M,
        'side_breakdown': side_brk,
        'bootstrap': boot,
        'synthetic': {
            'n_series': N_SYNTH,
            'median_annual': float(np.median(annuals)),
            'mean_annual': float(np.mean(annuals)),
            'n_positive': n_pos,
            'n_positive_10pct': n_pos_10,
            'p5': float(np.percentile(annuals, 5)),
            'p95': float(np.percentile(annuals, 95)),
            'series': synth_results,
        },
        'null': {
            'median_annual': float(np.median(null_annuals)),
            'series': null_results,
        },
        'edge_vs_null': float(edge_vs_null),
        'cross_check_tight_vs_wide': {
            'tight': {
                'params': {'factor': PARAMS_TIGHT['trail_atr_factor'],
                           'floor': PARAMS_TIGHT['trail_floor_pct'],
                           'ceil': PARAMS_TIGHT['trail_ceiling_pct']},
                'metrics': M_tight,
                'bootstrap_p': p_t,
            },
            'wide': {
                'params': {'factor': PARAMS['trail_atr_factor'],
                           'floor': PARAMS['trail_floor_pct'],
                           'ceil': PARAMS['trail_ceiling_pct']},
                'metrics': M,
                'bootstrap_p': p_w,
            },
        },
        'self_audit': {
            'overlap_violations': overlap_violations,
            'cutoff_respected': True,
            'sanity_flags': flags,
        },
        'verdict': 'KEEP' if all_pass else 'REJECT',
    }

    out_json = HERE / 'results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, default=str, indent=2)
    print(f'\nResultados guardados en {out_json}')

    return summary


if __name__ == '__main__':
    main()
