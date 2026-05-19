"""
train.py — Validacion in-sample de la estrategia Agent A.

Reglas inviolables aplicadas:
  - Datos cortados a <= 2025-12-31 SIEMPRE
  - Walk-forward con gap de purga (2 semanas) entre ventanas
  - Bootstrap >=2000 iter sobre pnl por trade
  - Una posicion a la vez (sin solape) en el motor
  - Sin look-ahead intrabar en el trailing

Uso:
  C:/Python/python.exe experiments/agent_A/train.py
"""

from __future__ import annotations
import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# importar strategy.py de este mismo directorio
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from strategy import (
    PARAMS, prepare_data, signal, simulate, run_backtest, metrics
)

ROOT = HERE.parent.parent
DATA = ROOT / 'data'


# =============================================================================
# CARGA DE DATOS (con cutoff inmediato)
# =============================================================================
def load_all_data(cutoff: str = '2025-12-31') -> dict:
    print(f'Cargando datos (cutoff <= {cutoff})...')
    cut = pd.Timestamp(cutoff, tz='UTC')

    df_4h = pd.read_parquet(DATA / 'BTC_USDT_4h_full.parquet')
    if df_4h.index.tz is None:
        df_4h.index = df_4h.index.tz_localize('UTC')
    df_4h = df_4h[df_4h.index <= cut].sort_index()

    df_1d = pd.read_parquet(DATA / 'btcusdt_1d_v15.parquet')
    if df_1d.index.tz is None:
        df_1d.index = df_1d.index.tz_localize('UTC')
    df_1d = df_1d[df_1d.index <= cut].sort_index()

    fund = pd.read_parquet(DATA / 'btc_v15_funding.parquet')
    if fund.index.tz is None:
        fund.index = fund.index.tz_localize('UTC')
    fund = fund[fund.index <= cut].sort_index()

    print(f'  4h: {len(df_4h)} bars, {df_4h.index.min().date()} -> {df_4h.index.max().date()}')
    print(f'  1d: {len(df_1d)} bars, {df_1d.index.min().date()} -> {df_1d.index.max().date()}')
    print(f'  funding: {len(fund)} rows, {fund.index.min().date()} -> {fund.index.max().date()}')

    # Verificacion: cutoff respetado
    assert df_4h.index.max() <= cut, "VIOLACION cutoff 4h"
    assert df_1d.index.max() <= cut, "VIOLACION cutoff 1d"
    assert fund.index.max() <= cut, "VIOLACION cutoff funding"

    return {'df_4h': df_4h, 'df_1d': df_1d, 'fund': fund}


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
PURGE_DAYS = 14  # 2 semanas de gap entre ventanas para evitar leakage


def walk_forward(df_features: pd.DataFrame, params: dict) -> dict:
    """
    WF por semestre. Cada semestre comienza tras un GAP de PURGE_DAYS dias
    sin trades. (Como no hay re-entrenamiento, el "purge" se aplica
    descartando ventanas que abren dentro del gap inicial — efectivamente
    queremos asegurar que el primer trade del semestre no este "tocando" el
    final del semestre anterior. Como cada trade dura <=60*4h = 10 dias, un
    gap de 14 dias garantiza que las ventanas no comparten trades).
    """
    fold_results = []
    for start_s, end_s in WF_SEMESTERS:
        start = pd.Timestamp(start_s, tz='UTC')
        end = pd.Timestamp(end_s, tz='UTC')
        purge_until = start + pd.Timedelta(days=PURGE_DAYS)

        # idx dentro del semestre
        in_window = (df_features.index >= start) & (df_features.index <= end)
        idxs = np.where(in_window)[0]
        if len(idxs) < 200:
            fold_results.append({'period': start_s[:7], 'n': 0, 'pf': 0.0,
                                 'wr': 0.0, 'total': 0.0, 'ok': False,
                                 'no_data': True})
            continue
        start_i, end_i = idxs[0], idxs[-1] + 1
        trades_all = run_backtest(df_features, params, start_i=start_i, end_i=end_i)
        # PURGE: descartar trades cuya entry sea antes del purge_until
        trades = [t for t in trades_all
                  if pd.Timestamp(t['entry_ts']) >= purge_until]
        m = metrics(trades)
        # Politica de fold:
        #   - "no_signal": el filtro macro stayed out -> neutral (no cuenta como
        #     pase ni como fallo). Esto sucede en mercados bear donde el filtro
        #     daily EMA50<EMA200 hace bien su trabajo. Es FEATURE, no bug.
        #   - n>=3: minimo trades para calcular metricas fiables.
        #   - n in [1,2]: indeterminado, no cuenta como fallo (muestra peq.)
        if m['n'] == 0:
            status = 'no_signal'
            ok = False  # no cuenta como pase
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


def main():
    print('=' * 78)
    print('AGENT A — TREND-FOLLOWING BREAKOUT + ATR TRAILING (BTC 4h, LONG-ONLY)')
    print('=' * 78)

    data = load_all_data(cutoff=PARAMS['cutoff_date'])
    df_4h, df_1d, fund = data['df_4h'], data['df_1d'], data['fund']

    print('\nPreparando features...')
    df_feat = prepare_data(df_4h, df_1d, fund, PARAMS)
    print(f'  features ready: {len(df_feat)} bars, '
          f'{df_feat.index.min().date()} -> {df_feat.index.max().date()}')

    # Walk-forward
    print(f'\nWalk-forward {len(WF_SEMESTERS)} semestres '
          f'(purga: {PURGE_DAYS}d entre folds)...')
    wf = walk_forward(df_feat, PARAMS)
    print(f'\n  Folds OK (PF>=1.2, total>0, n>=3): '
          f'{wf["folds_ok"]}/{wf["folds_evaluated"]} evaluados '
          f'({wf["folds_total"]} con datos)')
    pf_evaluated = [f['pf'] for f in wf['folds']
                    if f.get('status') == 'evaluated' and np.isfinite(f['pf'])]
    if pf_evaluated:
        print(f'  PF median (folds evaluados): {np.median(pf_evaluated):.2f}')
        print(f'  PF mean (folds evaluados): {np.mean(pf_evaluated):.2f}')
    print(f'  {"period":<10} {"n":>4} {"wr":>6} {"pf":>7} {"total":>9} '
          f'{"monthly":>9} {"dd":>7} ok')
    for f in wf['folds']:
        if f['no_data']:
            print(f"  {f['period']:<10} sin datos")
            continue
        st = f.get('status', '?')
        if st == 'no_signal':
            flag = '[no-signal]'  # filtro stayed out: bear market filtrado correctamente
        elif st == 'small_sample':
            flag = '[small-n]'
        else:
            flag = '[+]' if f['ok'] else '[-]'
        print(f"  {f['period']:<10} {f['n']:>4} {f['wr']*100:>5.1f}% "
              f"{fmt_pf(f['pf']):>7} {f['total']*100:>+8.1f}% "
              f"{f['monthly']*100:>+8.2f}% {f['max_dd']*100:>6.1f}% {flag}")

    # Backtest global (2020-01-01 -> cutoff) para bootstrap + metricas overall
    print('\nBacktest global 2020-01-01 -> cutoff...')
    start_full = pd.Timestamp('2020-01-01', tz='UTC')
    idxs = np.where(df_feat.index >= start_full)[0]
    start_i = int(idxs[0]) if len(idxs) else 0
    all_trades = run_backtest(df_feat, PARAMS, start_i=start_i, end_i=len(df_feat))
    M = metrics(all_trades)
    print(f"  N={M['n']}  WR={M['wr']*100:.1f}%  PF={fmt_pf(M['pf'])}  "
          f"total={M['total_return']*100:+.1f}%  "
          f"monthly={M['monthly_return']*100:+.2f}%  DD={M['max_dd']*100:.1f}%  "
          f"sharpe-like={M['sharpe_like']:.2f}  months={M['months']:.1f}")

    # Bootstrap
    print('\nBootstrap (3000 iter)...')
    boot = bootstrap_pvalue(all_trades, n_iter=3000)
    if boot is None:
        print('  insuficientes trades')
    else:
        sig = 'SIGNIFICATIVO' if boot['p_value'] < 0.05 else 'NO significativo'
        print(f"  p-value(retorno<=0 por azar): {boot['p_value']:.4f} -> {sig}")
        print(f"  retorno mediano resampled: {boot['pctl_50']*100:+.1f}%")
        print(f"  retorno percentil 5: {boot['pctl_5']*100:+.1f}%")
        print(f"  retorno percentil 95: {boot['pctl_95']*100:+.1f}%")

    # Sanity-check
    print('\nSANITY CHECKS:')
    if M['pf'] > 4:
        print(f"  [!] PF {M['pf']:.2f} > 4 -> sospechar overfitting/look-ahead")
    if M['wr'] > 0.65:
        print(f"  [!] WR {M['wr']*100:.1f}% > 65% -> sospechar overfitting")
    if M['max_dd'] < 0.05:
        print(f"  [!] DD {M['max_dd']*100:.1f}% < 5% -> sospechar (sample no adverso)")
    if M['pf'] <= 4 and M['wr'] <= 0.65 and M['max_dd'] >= 0.05:
        print('  [+] PF/WR/DD en rango razonable para crypto-4h')

    # Guardar resultados
    summary = {
        'params': PARAMS,
        'wf': {'folds_ok': wf['folds_ok'], 'folds_total': wf['folds_total'],
               'folds': wf['folds']},
        'overall': M,
        'bootstrap': boot,
        'cutoff': PARAMS['cutoff_date'],
        'n_trades_2020_2025': M['n'],
    }
    # JSON-friendly: convertir timestamps a string en folds (M no tiene timestamps)
    out_json = HERE / 'results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, default=str, indent=2)
    print(f'\nResultados guardados en {out_json}')

    return summary


if __name__ == '__main__':
    main()
