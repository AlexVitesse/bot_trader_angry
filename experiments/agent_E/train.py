"""
train.py — Validacion in-sample de la estrategia Agent E (funding-extremos).

Reglas inviolables aplicadas:
  - Datos cortados a <= 2025-12-31 SIEMPRE
  - Walk-forward por semestre con gap de purga (2 semanas) entre ventanas
  - Bootstrap >=3000 iter sobre pnl por trade (muestra pequena -> mas iter)
  - Una posicion a la vez (sin solape) en el motor
  - Sin look-ahead intrabar
  - Por direccion: validacion separada LONG vs SHORT

Uso:
  C:/Python/python.exe experiments/agent_E/train.py
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

    df_4h = pd.read_parquet(DATA / 'BTC_USDT_4h_full.parquet')
    if df_4h.index.tz is None:
        df_4h.index = df_4h.index.tz_localize('UTC')
    df_4h = df_4h[df_4h.index <= cut].sort_index()

    fund = pd.read_parquet(DATA / 'btc_v15_funding.parquet')
    if fund.index.tz is None:
        fund.index = fund.index.tz_localize('UTC')
    fund = fund[fund.index <= cut].sort_index()

    print(f'  4h:      {len(df_4h)} bars, {df_4h.index.min().date()} -> {df_4h.index.max().date()}')
    print(f'  funding: {len(fund)} rows, {fund.index.min().date()} -> {fund.index.max().date()}')

    assert df_4h.index.max() <= cut, "VIOLACION cutoff 4h"
    assert fund.index.max() <= cut, "VIOLACION cutoff funding"

    return {'df_4h': df_4h, 'fund': fund}


# =============================================================================
# WALK-FORWARD CON GAP DE PURGA
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
PURGE_DAYS = 14  # 2 semanas


def walk_forward(df_features: pd.DataFrame, params: dict,
                 side_filter: str | None = None) -> dict:
    """
    WF por semestre con purga de 14 dias al inicio. Si side_filter se indica
    ('LONG' o 'SHORT'), solo cuenta trades de esa direccion.
    """
    fold_results = []
    for start_s, end_s in WF_SEMESTERS:
        start = pd.Timestamp(start_s, tz='UTC')
        end = pd.Timestamp(end_s, tz='UTC')
        purge_until = start + pd.Timedelta(days=PURGE_DAYS)

        in_window = (df_features.index >= start) & (df_features.index <= end)
        idxs = np.where(in_window)[0]
        if len(idxs) < 200:
            fold_results.append({'period': start_s[:7], 'n': 0, 'pf': 0.0,
                                 'wr': 0.0, 'total': 0.0, 'ok': False,
                                 'no_data': True})
            continue
        start_i, end_i = idxs[0], idxs[-1] + 1
        trades_all = run_backtest(df_features, params,
                                  start_i=start_i, end_i=end_i)
        # Purga
        trades = [t for t in trades_all
                  if pd.Timestamp(t['entry_ts']) >= purge_until]
        # Side filter
        if side_filter:
            trades = [t for t in trades if t['side'] == side_filter]
        m = metrics(trades)

        # Politica de fold (mismo criterio que Agent A):
        #   - n>=3: minimo trades para calcular metricas fiables
        #   - PF>=1.2 y total>0 para "ok"
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
            'period': start_s[:7], 'n': m['n'],
            'n_long': m.get('n_long', 0), 'n_short': m.get('n_short', 0),
            'pf': m['pf'], 'wr': m['wr'],
            'total': m['total_return'], 'monthly': m.get('monthly_return', 0),
            'max_dd': m['max_dd'], 'ok': ok, 'status': status,
            'no_data': False,
        })
    folds_ok = sum(1 for r in fold_results if r['ok'])
    folds_evaluated = sum(1 for r in fold_results
                          if not r['no_data']
                          and r.get('status') == 'evaluated')
    folds_total = sum(1 for r in fold_results if not r['no_data'])
    return {'folds': fold_results, 'folds_ok': folds_ok,
            'folds_evaluated': folds_evaluated,
            'folds_total': folds_total}


# =============================================================================
# BOOTSTRAP
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
# MAIN
# =============================================================================
def fmt_pf(pf):
    return 'inf' if not np.isfinite(pf) else f'{pf:.2f}'


def print_folds(label, wf):
    print(f'\n  Walk-forward {label}: Folds OK (PF>=1.2, total>0, n>=3): '
          f'{wf["folds_ok"]}/{wf["folds_evaluated"]} evaluados '
          f'({wf["folds_total"]} con datos)')
    pf_evaluated = [f['pf'] for f in wf['folds']
                    if f.get('status') == 'evaluated' and np.isfinite(f['pf'])]
    if pf_evaluated:
        print(f'  PF median (folds evaluados): {np.median(pf_evaluated):.2f}')
    print(f'  {"period":<10} {"n":>4} {"L":>3} {"S":>3} {"wr":>6} {"pf":>6} '
          f'{"total":>9} {"dd":>7} ok')
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
        print(f"  {f['period']:<10} {f['n']:>4} {f['n_long']:>3} {f['n_short']:>3} "
              f"{f['wr']*100:>5.1f}% {fmt_pf(f['pf']):>6} "
              f"{f['total']*100:>+8.1f}% "
              f"{f['max_dd']*100:>6.1f}% {flag}")


def main():
    print('=' * 78)
    print('AGENT E — FUNDING-EXTREMOS MEAN-REVERSION (BTC 4h, BIDIRECCIONAL)')
    print('=' * 78)

    data = load_all_data(cutoff=PARAMS['cutoff_date'])
    df_4h, fund = data['df_4h'], data['fund']

    print('\nPreparando features (funding rate + zscore, sin look-ahead)...')
    df_feat = prepare_data(df_4h, fund, PARAMS)
    print(f'  features ready: {len(df_feat)} bars, '
          f'{df_feat.index.min().date()} -> {df_feat.index.max().date()}')
    n_long_setup = ((df_feat['funding_z'] < PARAMS['long_z_max']) &
                    (df_feat['bullish'] == 1)).sum()
    n_short_setup = ((df_feat['funding_z'] > PARAMS['short_z_min']) &
                     (df_feat['bearish'] == 1)).sum()
    print(f'  LONG setup activations (z<{PARAMS["long_z_max"]}+bullish): {n_long_setup}')
    print(f'  SHORT setup activations (z>{PARAMS["short_z_min"]}+bearish): {n_short_setup}')

    # ---- WF combinado ----
    wf_all = walk_forward(df_feat, PARAMS, side_filter=None)
    print_folds('COMBINED (LONG+SHORT)', wf_all)

    # ---- WF por direccion ----
    wf_long = walk_forward(df_feat, PARAMS, side_filter='LONG')
    print_folds('LONG-only (subset de los mismos trades)', wf_long)

    wf_short = walk_forward(df_feat, PARAMS, side_filter='SHORT')
    print_folds('SHORT-only (subset de los mismos trades)', wf_short)

    # ---- Backtest global 2020-01-01 -> cutoff ----
    print('\nBacktest global 2020-01-01 -> cutoff (UNA posicion a la vez)...')
    start_full = pd.Timestamp('2020-01-01', tz='UTC')
    idxs = np.where(df_feat.index >= start_full)[0]
    start_i = int(idxs[0]) if len(idxs) else 0
    all_trades = run_backtest(df_feat, PARAMS,
                              start_i=start_i, end_i=len(df_feat))
    M = metrics(all_trades)
    print(f"  N={M['n']} (LONG={M['n_long']}, SHORT={M['n_short']})  "
          f"WR={M['wr']*100:.1f}%  PF={fmt_pf(M['pf'])}  "
          f"total={M['total_return']*100:+.1f}%  "
          f"monthly={M['monthly_return']*100:+.2f}%  "
          f"annual={M['annual_return']*100:+.1f}%  "
          f"DD={M['max_dd']*100:.1f}%  "
          f"sharpe-like={M['sharpe_like']:.2f}  months={M['months']:.1f}")
    print(f"  avg_holding={M['avg_holding_bars']*4/24:.1f}d  "
          f"funding_contrib={M['funding_contrib_pct']:+.1f}% del PnL")

    # ---- Subset LONG y SHORT ----
    long_trades = [t for t in all_trades if t['side'] == 'LONG']
    short_trades = [t for t in all_trades if t['side'] == 'SHORT']
    ML = metrics(long_trades)
    MS = metrics(short_trades)
    print(f"\n  LONG-only: N={ML['n']} WR={ML['wr']*100:.1f}% PF={fmt_pf(ML['pf'])} "
          f"total={ML['total_return']*100:+.1f}% DD={ML['max_dd']*100:.1f}% "
          f"funding_contrib={ML['funding_contrib_pct']:+.1f}%")
    print(f"  SHORT-only: N={MS['n']} WR={MS['wr']*100:.1f}% PF={fmt_pf(MS['pf'])} "
          f"total={MS['total_return']*100:+.1f}% DD={MS['max_dd']*100:.1f}% "
          f"funding_contrib={MS['funding_contrib_pct']:+.1f}%")

    # ---- Bootstrap ----
    print('\nBootstrap (3000 iter, muestra pequena -> alto n_iter)...')
    boot_all = bootstrap_pvalue(all_trades, n_iter=3000)
    boot_long = bootstrap_pvalue(long_trades, n_iter=3000)
    boot_short = bootstrap_pvalue(short_trades, n_iter=3000)
    for label, b in [('COMBINED', boot_all), ('LONG', boot_long),
                     ('SHORT', boot_short)]:
        if b is None:
            print(f'  {label}: insuficientes trades')
            continue
        sig = 'SIGNIFICATIVO' if b['p_value'] < 0.05 else 'NO significativo'
        print(f"  {label}: p={b['p_value']:.4f} -> {sig}  "
              f"(N={b['n_trades']}, mediana resampled={b['pctl_50']*100:+.1f}%, "
              f"p5={b['pctl_5']*100:+.1f}%, p95={b['pctl_95']*100:+.1f}%)")

    # ---- Sanity checks ----
    print('\nSANITY CHECKS:')
    if M['pf'] > 4:
        print(f"  [!] PF {M['pf']:.2f} > 4 -> sospechar overfitting/look-ahead")
    if M['wr'] > 0.70 and M['n'] < 100:
        print(f"  [!] WR {M['wr']*100:.1f}% > 70% con n<100 -> casi seguro overfit")
    if M['max_dd'] < 0.05:
        print(f"  [!] DD {M['max_dd']*100:.1f}% < 5% -> sospechar (sample no adverso)")
    if M['n'] < 30:
        print(f"  [!] Solo {M['n']} trades en 6 anos -> muestra muy pequena, riesgo azar alto")
    if (M['pf'] <= 4 and M['wr'] <= 0.65 and M['max_dd'] >= 0.05
            and M['n'] >= 30):
        print('  [+] PF/WR/DD/N en rango razonable')

    # ---- Resumen ----
    summary = {
        'params': {k: v for k, v in PARAMS.items()},
        'wf_combined': {'folds_ok': wf_all['folds_ok'],
                        'folds_evaluated': wf_all['folds_evaluated'],
                        'folds_total': wf_all['folds_total'],
                        'folds': wf_all['folds']},
        'wf_long': {'folds_ok': wf_long['folds_ok'],
                    'folds_evaluated': wf_long['folds_evaluated'],
                    'folds_total': wf_long['folds_total'],
                    'folds': wf_long['folds']},
        'wf_short': {'folds_ok': wf_short['folds_ok'],
                     'folds_evaluated': wf_short['folds_evaluated'],
                     'folds_total': wf_short['folds_total'],
                     'folds': wf_short['folds']},
        'overall': M,
        'long_only': ML,
        'short_only': MS,
        'bootstrap_combined': boot_all,
        'bootstrap_long': boot_long,
        'bootstrap_short': boot_short,
        'cutoff': PARAMS['cutoff_date'],
        'n_trades_2020_2025': M['n'],
    }
    out_json = HERE / 'results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, default=str, indent=2)
    print(f'\nResultados guardados en {out_json}')

    return summary


if __name__ == '__main__':
    main()
