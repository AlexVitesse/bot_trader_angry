"""
sensitivity.py -- Honesto barrido de thresholds para verificar que NO hay un
sweet spot escondido. Si ningun threshold supera el cross-check vs random,
el veredicto NEGATIVO se confirma.

CRITICO: este es DIAGNOSTICO. NO se usan los resultados para tunear el modelo
de produccion. Es para confirmar que la conclusion negativa es robusta.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import strategy as S  # noqa


def evaluate_thresholds(mvrv_thr, exch_thr, short_thr, df_oc_real=None, base=None):
    p = dict(S.PARAMS)
    p['mvrv_z_block_long'] = mvrv_thr
    p['exch_netflow_z_block_long'] = exch_thr
    p['mvrv_z_unblock_short'] = short_thr
    is_start = pd.Timestamp('2020-01-01', tz='UTC')
    is_end = pd.Timestamp('2026-01-01', tz='UTC')
    r = S.backtest_v2_plus_onchain(is_start, is_end, p)
    trades = r['trades_with_filter']
    n_vet = len(r['vetoed'])
    if not trades:
        return None
    m = S.metrics(trades)
    pp = S.bootstrap_p(trades)
    return {'n': m['n'], 'vetoed': n_vet, 'wr': m['wr'], 'pf': m['pf'],
            'annual': m['annual'], 'dd': m['max_dd'], 'p': pp}


def main():
    is_start = pd.Timestamp('2020-01-01', tz='UTC')
    is_end = pd.Timestamp('2026-01-01', tz='UTC')

    # Baseline
    p_base = dict(S.PARAMS)
    p_base['enable_mvrv_filter'] = False
    p_base['enable_exchflow_filter'] = False
    p_base['enable_short_filter'] = False
    r_base = S.backtest_v2_plus_onchain(is_start, is_end, p_base)
    base_trades = r_base['trades_with_filter']
    base_m = S.metrics(base_trades)
    base_p = S.bootstrap_p(base_trades)
    print(f'V2 baseline: n={base_m["n"]}, PF={base_m["pf"]:.2f}, '
          f'annual={base_m["annual"]:+.1%}, p={base_p:.3f}')

    print()
    print('Sensibilidad MVRV LONG threshold (manteniendo exch=2.0, short=-1.5):')
    print(f"{'mvrv_thr':>10s}  {'n':>4s}  {'vet':>3s}  {'WR':>5s}  {'PF':>5s}  {'ann':>7s}  {'DD':>5s}  {'p':>5s}")
    for thr in [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0]:
        r = evaluate_thresholds(thr, 2.0, -1.5)
        if r:
            print(f"  {thr:>8.2f}    {r['n']:>3}  {r['vetoed']:>3}  "
                  f"{r['wr']:>5.1%}  {r['pf']:>5.2f}  {r['annual']:>+6.1%}  "
                  f"{r['dd']:>5.1%}  {r['p']:.3f}")

    print()
    print('Sensibilidad MVRV SHORT (bloquear if MVRV z < X):')
    print(f"{'short_thr':>10s}  {'n':>4s}  {'vet':>3s}  {'WR':>5s}  {'PF':>5s}  {'ann':>7s}  {'p':>5s}")
    for thr in [-2.5, -2.0, -1.5, -1.0, -0.5]:
        r = evaluate_thresholds(2.5, 2.0, thr)
        if r:
            print(f"  {thr:>8.2f}    {r['n']:>3}  {r['vetoed']:>3}  "
                  f"{r['wr']:>5.1%}  {r['pf']:>5.2f}  {r['annual']:>+6.1%}  {r['p']:.3f}")

    print()
    print('Sensibilidad exchange flow LONG (bloquear if exch_z > X):')
    print(f"{'exch_thr':>10s}  {'n':>4s}  {'vet':>3s}  {'WR':>5s}  {'PF':>5s}  {'ann':>7s}  {'p':>5s}")
    for thr in [1.0, 1.5, 1.8, 2.0, 2.5, 3.0]:
        r = evaluate_thresholds(2.5, thr, -1.5)
        if r:
            print(f"  {thr:>8.2f}    {r['n']:>3}  {r['vetoed']:>3}  "
                  f"{r['wr']:>5.1%}  {r['pf']:>5.2f}  {r['annual']:>+6.1%}  {r['p']:.3f}")


if __name__ == '__main__':
    main()
