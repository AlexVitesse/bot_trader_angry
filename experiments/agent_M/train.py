"""
train.py -- Validacion 4-capas de Agent M (SOL beta-leveraged a A's BTC signal).

Capas:
  1. In-sample real SOL 2020-09 a 2025-12-31: WF + bootstrap p (>=2000 iter)
  2. 20 series sintéticas SOL (block bootstrap), usando BTC REAL para senales
  3. Null hypothesis: shuffle SOL retornos (estructura temporal destruida)
  4. Cross-check: reemplazar A's BTC signal con SENAL RANDOM (mismo trade rate)

Veredicto KEEP si:
  - Bootstrap p<0.05 real
  - Mediana sintético >0
  - >=14/20 sintéticas positivas
  - Edge vs null >5%
  - Cross-check random falla claramente (edge real > random + 5%)

Uso: C:/Python/python.exe experiments/agent_M/train.py
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
from strategy import (  # noqa: E402
    PARAMS, prepare_data, signal, simulate, run_backtest, metrics
)

ROOT = HERE.parent.parent
DATA = ROOT / 'data'


# =============================================================================
# CARGA DE DATOS
# =============================================================================
def load_all_data(cutoff: str = '2025-12-31') -> dict:
    print(f'Cargando datos (cutoff <= {cutoff})...')
    cut = pd.Timestamp(cutoff, tz='UTC')

    df_sol = pd.read_parquet(DATA / 'SOL_USDT_4h_full.parquet')
    if df_sol.index.tz is None:
        df_sol.index = df_sol.index.tz_localize('UTC')
    df_sol = df_sol[df_sol.index <= cut].sort_index()

    df_btc_4h = pd.read_parquet(DATA / 'BTC_USDT_4h_full.parquet')
    if df_btc_4h.index.tz is None:
        df_btc_4h.index = df_btc_4h.index.tz_localize('UTC')
    df_btc_4h = df_btc_4h[df_btc_4h.index <= cut].sort_index()

    df_btc_1d = pd.read_parquet(DATA / 'btcusdt_1d_v15.parquet')
    if df_btc_1d.index.tz is None:
        df_btc_1d.index = df_btc_1d.index.tz_localize('UTC')
    df_btc_1d = df_btc_1d[df_btc_1d.index <= cut].sort_index()

    fund = pd.read_parquet(DATA / 'btc_v15_funding.parquet')
    if fund.index.tz is None:
        fund.index = fund.index.tz_localize('UTC')
    fund = fund[fund.index <= cut].sort_index()

    print(f'  SOL 4h: {len(df_sol)} bars, {df_sol.index.min().date()} -> {df_sol.index.max().date()}')
    print(f'  BTC 4h: {len(df_btc_4h)} bars, {df_btc_4h.index.min().date()} -> {df_btc_4h.index.max().date()}')
    print(f'  BTC 1d: {len(df_btc_1d)} bars')
    print(f'  funding: {len(fund)} rows')

    assert df_sol.index.max() <= cut
    assert df_btc_4h.index.max() <= cut
    return {'df_sol': df_sol, 'df_btc_4h': df_btc_4h,
            'df_btc_1d': df_btc_1d, 'fund': fund}


# =============================================================================
# WALK-FORWARD
# =============================================================================
# SOL empieza 2020-08-11 -> 2020-H1 quedará vacío. 2020-H2 tendrá warmup parcial.
# El esquema sigue siendo semestral con purga 14 dias.
WF_SEMESTERS = [
    ('2020-01-01', '2020-06-30'),  # SIN DATOS SOL (no_data)
    ('2020-07-01', '2020-12-31'),  # warmup -> probablemente no_signal
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


def walk_forward(df_sol, df_btc_A, params) -> dict:
    fold_results = []
    for start_s, end_s in WF_SEMESTERS:
        start = pd.Timestamp(start_s, tz='UTC')
        end = pd.Timestamp(end_s, tz='UTC')
        purge_until = start + pd.Timedelta(days=PURGE_DAYS)

        in_window = (df_sol.index >= start) & (df_sol.index <= end)
        idxs = np.where(in_window)[0]
        if len(idxs) < 200:
            fold_results.append({'period': start_s[:7], 'n': 0, 'pf': 0.0,
                                 'wr': 0.0, 'total': 0.0, 'monthly': 0.0,
                                 'max_dd': 0.0, 'ok': False,
                                 'status': 'no_data', 'no_data': True})
            continue
        start_i, end_i = int(idxs[0]), int(idxs[-1]) + 1
        trades_all = run_backtest(df_sol, df_btc_A, params,
                                  start_i=start_i, end_i=end_i)
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
    folds_evaluated = sum(1 for r in fold_results
                          if not r['no_data'] and r.get('status') == 'evaluated')
    folds_total = sum(1 for r in fold_results if not r['no_data'])
    return {'folds': fold_results, 'folds_ok': folds_ok,
            'folds_evaluated': folds_evaluated,
            'folds_total': folds_total}


# =============================================================================
# BOOTSTRAP DE SIGNIFICANCIA
# =============================================================================
def bootstrap_pvalue(trades, n_iter=3000, seed=42) -> dict | None:
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
# CAPA 2: SINTETICAS (block bootstrap SOL, BTC REAL como senal-fuente)
# =============================================================================
def joint_block_bootstrap(df_sol: pd.DataFrame, df_btc: pd.DataFrame,
                          block_size=24, seed=0) -> tuple:
    """
    Block bootstrap CONJUNTO: sample bloques de timestamps comunes y aplica
    el MISMO sampling a SOL y a BTC. Preserva la estructura de correlacion
    SOL-BTC dentro de cada bloque (que es la unica que importa para nuestra
    hipotesis "SOL es BTC apalancado").

    El indice del output es 4h consecutivos desde la fecha de inicio real.
    Tanto SOL como BTC son reescalados para preservar continuidad de precio.
    """
    rng = np.random.default_rng(seed)
    # Indice comun entre SOL y BTC para poder samplear coherentemente
    common = df_sol.index.intersection(df_btc.index)
    df_sol_c = df_sol.loc[common]
    df_btc_c = df_btc.loc[common]
    n_bars = len(df_sol_c)
    if n_bars < block_size * 2:
        raise ValueError('Datos insuficientes para joint bootstrap')

    max_start = n_bars - block_size
    n_blocks = (n_bars // block_size) + 2
    starts = rng.integers(0, max_start, size=n_blocks)

    sol_parts, btc_parts = [], []
    last_sol = float(df_sol_c['close'].iloc[0])
    last_btc = float(df_btc_c['close'].iloc[0])
    for s in starts:
        sol_block = df_sol_c.iloc[s:s + block_size].copy()
        btc_block = df_btc_c.iloc[s:s + block_size].copy()
        # Reescalado para continuidad
        sol_scale = last_sol / float(sol_block['open'].iloc[0])
        btc_scale = last_btc / float(btc_block['open'].iloc[0])
        sol_block[['open', 'high', 'low', 'close']] *= sol_scale
        btc_block[['open', 'high', 'low', 'close']] *= btc_scale
        sol_parts.append(sol_block)
        btc_parts.append(btc_block)
        last_sol = float(sol_block['close'].iloc[-1])
        last_btc = float(btc_block['close'].iloc[-1])

    sol_out = pd.concat(sol_parts).iloc[:n_bars].copy()
    btc_out = pd.concat(btc_parts).iloc[:n_bars].copy()
    new_index = pd.date_range(start=common[0], periods=n_bars, freq='4h', tz='UTC')
    sol_out.index = new_index
    btc_out.index = new_index
    return sol_out, btc_out


def synthetic_runs(df_sol_real, df_btc_4h_real, df_btc_1d, fund,
                    n_synth=20, block_size=24):
    """
    Para cada serie sintetica:
      - SOL y BTC son sinteticos JUNTOS (block bootstrap conjunto -> preserva corr)
      - La senal de A se calcula sobre BTC sintetico (no real)
      - Esto testa si la hipotesis "SOL apalancado a BTC" + senal A tiene edge
        en universos paralelos donde la dinamica conjunta SOL-BTC es similar
        pero las fechas/secuencia son distintas (no es solo casualidad de muestra)
    NOTA: Usamos df_btc_1d=None (derivado del 4h sintetico con shift1) y
    funding=None para evitar mezclar series reales con sinteticas.
    """
    results = []
    for seed in range(n_synth):
        df_sol_s, df_btc_s = joint_block_bootstrap(
            df_sol_real, df_btc_4h_real, block_size=block_size, seed=seed)
        # BTC sintetico -> deriva su propio daily, sin funding real
        df_sol_feat, df_btc_feat, _ = prepare_data(
            df_sol_s, df_btc_s, None, None, PARAMS)
        trades = run_backtest(df_sol_feat, df_btc_feat, PARAMS)
        m = metrics(trades)
        results.append(m)
        print(f"  synth {seed:2d}: N={m['n']:3d} WR={m['wr']:.1%} "
              f"PF={m['pf']:.2f} annual={m['annual']:+.1%}")
    return results


# =============================================================================
# CAPA 3: NULL (shuffle de retornos SOL -- destruye estructura temporal)
# =============================================================================
def shuffle_returns(df: pd.DataFrame, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df_c = df.copy()
    log_ret = np.log(df_c['close'].values[1:] / df_c['close'].values[:-1])
    perm = rng.permutation(len(log_ret))
    log_ret_shuf = log_ret[perm]
    new_close = np.empty(len(df_c))
    new_close[0] = df_c['close'].iloc[0]
    new_close[1:] = df_c['close'].iloc[0] * np.exp(np.cumsum(log_ret_shuf))
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


def null_runs(df_sol_real, df_btc_4h, df_btc_1d, fund, n_null=10):
    """
    Null: shuffle SOL retornos. BTC y A's signal son REALES. La correlacion
    SOL-BTC se rompe; con corr_min=0.5, casi todas las senales seran rechazadas.
    Para que el null tenga muestras tradeables, RELAJAMOS la corr_min a -inf
    (i.e. la senal A entra siempre). Esto mide: "si SOL no tiene estructura
    temporal alineada con BTC, ¿que pasa con A's senal + simulador en SOL?"
    """
    params_null = dict(PARAMS)
    params_null['corr_min'] = -10.0  # sin filtro
    results = []
    for seed in range(n_null):
        df_sol_null = shuffle_returns(df_sol_real, seed=seed)
        df_sol_feat, df_btc_feat, _ = prepare_data(
            df_sol_null, df_btc_4h, df_btc_1d, fund, params_null)
        trades = run_backtest(df_sol_feat, df_btc_feat, params_null)
        m = metrics(trades)
        results.append(m)
        print(f"  null  {seed:2d}: N={m['n']:3d} WR={m['wr']:.1%} "
              f"PF={m['pf']:.2f} annual={m['annual']:+.1%}")
    return results


# =============================================================================
# CAPA 4: CROSS-CHECK -- A's signal reemplazada por RANDOM al mismo trade rate
# =============================================================================
def cross_check_random_signal(df_sol_feat, df_btc_feat, real_trade_rate, n_seeds=5):
    """
    Reemplaza A.signal por una funcion random que dispara 'LONG' con probabilidad p
    tal que el numero esperado de senales en el universo coincida con el real.
    Si el edge es del beta, una senal random deberia dar resultados similares
    (mala noticia). Si el edge es de A, random deberia ser claramente peor.
    """
    n_bars = len(df_sol_feat)
    # Tasa real de senales (signals/bars validos)
    # real_trade_rate = trades_reales / bars_validos
    annuals = []
    n_trades_list = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)

        def random_signal(df_btc_A, idx):
            if rng.random() < real_trade_rate:
                return 'LONG'
            return None

        trades = run_backtest(df_sol_feat, df_btc_feat, PARAMS,
                              a_signal_fn=random_signal)
        m = metrics(trades)
        annuals.append(m['annual'])
        n_trades_list.append(m['n'])
        print(f"  random seed {seed}: N={m['n']:3d} WR={m['wr']:.1%} "
              f"PF={m['pf']:.2f} annual={m['annual']:+.1%}")
    return annuals, n_trades_list


# =============================================================================
# MAIN
# =============================================================================
def fmt_pf(pf):
    return 'inf' if not np.isfinite(pf) else f'{pf:.2f}'


def main():
    print('=' * 78)
    print('AGENT M -- SOL LEVERAGED-BETA on A signal (BTC LONG + corr filter)')
    print('=' * 78)

    data = load_all_data(cutoff=PARAMS['cutoff_date'])
    df_sol = data['df_sol']
    df_btc_4h = data['df_btc_4h']
    df_btc_1d = data['df_btc_1d']
    fund = data['fund']

    print('\nPreparando features SOL + BTC(A)...')
    df_sol_feat, df_btc_feat, common_idx = prepare_data(
        df_sol, df_btc_4h, df_btc_1d, fund, PARAMS)
    print(f'  Common features ready: {len(df_sol_feat)} bars, '
          f'{df_sol_feat.index.min().date()} -> {df_sol_feat.index.max().date()}')
    print(f'  Corr median (SOL-BTC 168bar): {df_sol_feat["corr_sol_btc"].median():.3f}')
    print(f'  Pct bars con corr>={PARAMS["corr_min"]}: '
          f'{(df_sol_feat["corr_sol_btc"] >= PARAMS["corr_min"]).mean():.1%}')

    # ===================================================================
    # CAPA 1: Walk-forward + bootstrap REAL
    # ===================================================================
    print(f'\n--- CAPA 1: Walk-forward {len(WF_SEMESTERS)} semestres ---')
    wf = walk_forward(df_sol_feat, df_btc_feat, PARAMS)
    print(f'\n  Folds OK (PF>=1.2, total>0, n>=3): '
          f'{wf["folds_ok"]}/{wf["folds_evaluated"]} evaluados '
          f'({wf["folds_total"]} con datos)')
    print(f'  {"period":<10} {"n":>4} {"wr":>6} {"pf":>7} {"total":>9} '
          f'{"monthly":>9} {"dd":>7} status')
    for f in wf['folds']:
        if f['no_data']:
            print(f"  {f['period']:<10} sin datos (SOL no existia)")
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

    # Backtest global
    print('\nBacktest global 2020-08 -> 2025-12-31...')
    all_trades = run_backtest(df_sol_feat, df_btc_feat, PARAMS)
    M = metrics(all_trades)
    print(f"  N={M['n']}  WR={M['wr']*100:.1f}%  PF={fmt_pf(M['pf'])}  "
          f"total={M['total_return']*100:+.1f}%  annual={M['annual']*100:+.1f}%  "
          f"DD={M['max_dd']*100:.1f}%  sharpe-like={M['sharpe_like']:.2f}")

    # Bootstrap real
    print('\nBootstrap real (3000 iter)...')
    boot = bootstrap_pvalue(all_trades, n_iter=3000)
    if boot is None:
        print('  insuficientes trades')
    else:
        sig = 'SIGNIFICATIVO' if boot['p_value'] < 0.05 else 'NO significativo'
        print(f"  p-value(retorno<=0 por azar): {boot['p_value']:.4f} -> {sig}")
        print(f"  retorno mediano resampled: {boot['pctl_50']*100:+.1f}%")
        print(f"  retorno p5: {boot['pctl_5']*100:+.1f}%  p95: {boot['pctl_95']*100:+.1f}%")

    # ===================================================================
    # CAPA 2: Sinteticas (block bootstrap SOL + BTC real)
    # ===================================================================
    N_SYNTH = 20
    print(f'\n--- CAPA 2: {N_SYNTH} series sinteticas SOL (BTC real) ---')
    synth = synthetic_runs(df_sol, df_btc_4h, df_btc_1d, fund, n_synth=N_SYNTH)
    synth_annuals = [r['annual'] for r in synth]
    n_pos = sum(1 for a in synth_annuals if a > 0)
    print(f'\n  Distribucion annual sinteticas:')
    print(f'    mediana={np.median(synth_annuals):+.1%} '
          f'media={np.mean(synth_annuals):+.1%}')
    print(f'    p5-p95=[{np.percentile(synth_annuals,5):+.1%}, '
          f'{np.percentile(synth_annuals,95):+.1%}]')
    print(f'    >0: {n_pos}/{N_SYNTH}')
    inside = (np.percentile(synth_annuals, 5) <= M['annual']
              <= np.percentile(synth_annuals, 95))
    print(f"    Real SOL ({M['annual']:+.1%}) esta "
          f"{'dentro' if inside else 'FUERA'} de p5-p95")

    # ===================================================================
    # CAPA 3: Null (retornos shuffleados)
    # ===================================================================
    print('\n--- CAPA 3: Null (shuffle SOL retornos) ---')
    null = null_runs(df_sol, df_btc_4h, df_btc_1d, fund, n_null=10)
    null_annuals = [r['annual'] for r in null]
    print(f'\n  Mediana null: {np.median(null_annuals):+.1%}')
    edge_vs_null = np.median(synth_annuals) - np.median(null_annuals)
    print(f'  Edge synth - null: {edge_vs_null:+.1%}')

    # ===================================================================
    # CAPA 4: Cross-check random signal
    # ===================================================================
    print('\n--- CAPA 4: Cross-check (A signal -> random uniform) ---')
    # Tasa de senales reales: trades_reales / bars_efectivos
    # bars efectivos: len(df_sol_feat) - warmup
    bars_eff = len(df_sol_feat) - PARAMS['min_bars_warmup']
    real_rate = M['n'] / max(bars_eff, 1)
    print(f"  Tasa real de senales: {real_rate:.5f} (n={M['n']}, bars={bars_eff})")
    random_annuals, random_n = cross_check_random_signal(
        df_sol_feat, df_btc_feat, real_rate, n_seeds=5)
    print(f'  Mediana annual random: {np.median(random_annuals):+.1%}')
    edge_vs_random = M['annual'] - np.median(random_annuals)
    print(f'  Edge real - mediana random: {edge_vs_random:+.1%}')

    # ===================================================================
    # VEREDICTO
    # ===================================================================
    print('\n' + '=' * 78)
    print('VEREDICTO AGENT M (SOL beta-leveraged)')
    print('=' * 78)
    p_real = boot['p_value'] if boot else 1.0
    c1 = p_real < 0.05
    c2 = np.median(synth_annuals) > 0
    c3 = n_pos >= 14
    c4 = edge_vs_null > 0.05
    c5 = edge_vs_random > 0.05
    print(f"  Bootstrap p<0.05 (real):                 {'SI' if c1 else 'NO'} ({p_real:.4f})")
    print(f"  Mediana sintetico > 0:                   {'SI' if c2 else 'NO'} "
          f"({np.median(synth_annuals):+.1%})")
    print(f"  >=14/20 sinteticas positivas:            {'SI' if c3 else 'NO'} ({n_pos}/{N_SYNTH})")
    print(f"  Edge sintetico vs null > 5%:             {'SI' if c4 else 'NO'} ({edge_vs_null:+.1%})")
    print(f"  Cross-check: real > random+5%:           {'SI' if c5 else 'NO'} ({edge_vs_random:+.1%})")
    all_pass = c1 and c2 and c3 and c4 and c5
    print(f"\n  -> {'KEEP' if all_pass else 'REJECT'}")

    # Guardar
    summary = {
        'params': PARAMS,
        'wf': {'folds_ok': wf['folds_ok'], 'folds_total': wf['folds_total'],
               'folds_evaluated': wf['folds_evaluated'], 'folds': wf['folds']},
        'overall': M,
        'bootstrap': boot,
        'synth': {
            'n_synth': N_SYNTH,
            'median_annual': float(np.median(synth_annuals)),
            'mean_annual': float(np.mean(synth_annuals)),
            'n_positive': n_pos,
            'p5': float(np.percentile(synth_annuals, 5)),
            'p95': float(np.percentile(synth_annuals, 95)),
            'annuals': [float(a) for a in synth_annuals],
        },
        'null': {
            'n_null': len(null_annuals),
            'median_annual': float(np.median(null_annuals)),
            'annuals': [float(a) for a in null_annuals],
        },
        'cross_check_random': {
            'n_seeds': len(random_annuals),
            'median_annual': float(np.median(random_annuals)),
            'mean_n': float(np.mean(random_n)),
            'annuals': [float(a) for a in random_annuals],
        },
        'edge_vs_null': float(edge_vs_null),
        'edge_vs_random': float(edge_vs_random),
        'verdict': 'KEEP' if all_pass else 'REJECT',
        'criteria': {
            'p_real<0.05': bool(c1),
            'synth_median>0': bool(c2),
            'synth_14_positive': bool(c3),
            'edge_vs_null>5%': bool(c4),
            'edge_vs_random>5%': bool(c5),
        },
        'cutoff': PARAMS['cutoff_date'],
    }
    out_json = HERE / 'results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, default=str, indent=2)
    print(f'\nResultados guardados en {out_json}')
    return summary


if __name__ == '__main__':
    main()
