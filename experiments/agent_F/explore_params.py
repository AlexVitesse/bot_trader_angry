"""
Exploracion HONESTA de parametros — el unico objetivo es entender si la
estrategia tiene edge o si es ruido. NO se optimiza para producir la
configuracion ganadora — solo para reportar la SENSIBILIDAD.

Reportar:
- LONG-only vs BIDIRECTIONAL
- Efecto compression_percentile (15, 20, 25)
- Efecto breakout_n (8, 12, 16)
- Sensibilidad al regime_filter
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from strategy import PARAMS, prepare_data, run_backtest, metrics
from train import load_all_data, walk_forward, bootstrap_pvalue


def evaluate(params, btc_4h, btc_1d, btc_fund, eth_4h, eth_1d):
    btc_feat = prepare_data(btc_4h, btc_1d, btc_fund, params)
    eth_feat = prepare_data(eth_4h, eth_1d, None, params)
    wf = walk_forward(btc_feat, eth_feat, params)
    start = pd.Timestamp('2020-01-01', tz='UTC')
    end = pd.Timestamp(params['cutoff_date'], tz='UTC')
    trades = run_backtest(btc_feat, eth_feat, params, start, end)
    M = metrics(trades)
    boot = bootstrap_pvalue(trades, n_iter=2000)
    return {
        'wf_agg': f"{wf['folds_ok_agg']}/{wf['folds_total']}",
        'wf_btc': f"{wf['folds_ok_btc']}/{wf['folds_total']}",
        'wf_eth': f"{wf['folds_ok_eth']}/{wf['folds_total']}",
        'n': M['n'], 'wr': M['wr'], 'pf': M['pf'],
        'total': M['total_return'], 'annual': M['annual_return'],
        'dd': M['max_dd'], 'sharpe': M['sharpe_annual'],
        'boot_p': boot['p_value'] if boot else None,
    }


def fmt(r):
    pf = 'inf' if not np.isfinite(r['pf']) else f"{r['pf']:.2f}"
    bp = f"{r['boot_p']:.3f}" if r['boot_p'] is not None else 'n/a'
    return (f"WF agg {r['wf_agg']} BTC {r['wf_btc']} ETH {r['wf_eth']} | "
            f"N={r['n']:>3} WR={r['wr']*100:>4.1f}% PF={pf:>5} "
            f"ann={r['annual']*100:>+5.1f}% DD={r['dd']*100:>4.1f}% "
            f"shar={r['sharpe']:>4.2f} p={bp}")


def main():
    data = load_all_data(cutoff=PARAMS['cutoff_date'])
    btc_4h, btc_1d, btc_fund = data['btc_4h'], data['btc_1d'], data['btc_fund']
    eth_4h, eth_1d = data['eth_4h'], data['eth_1d']

    print('=' * 78)
    print('EXPLORACION DE PARAMETROS (HONESTA — no es selection bias)')
    print('=' * 78)

    print('\n## Direccionalidad')
    p = {**PARAMS}
    print(f"  BASELINE (bidirectional):       {fmt(evaluate(p, btc_4h, btc_1d, btc_fund, eth_4h, eth_1d))}")
    p2 = {**PARAMS, 'enable_short': False}
    print(f"  LONG-only:                      {fmt(evaluate(p2, btc_4h, btc_1d, btc_fund, eth_4h, eth_1d))}")
    p3 = {**PARAMS, 'enable_long': False}
    print(f"  SHORT-only:                     {fmt(evaluate(p3, btc_4h, btc_1d, btc_fund, eth_4h, eth_1d))}")

    print('\n## Compression percentile (cuanto de la cola es "comprimida")')
    for cp in [0.15, 0.20, 0.25, 0.30]:
        p = {**PARAMS, 'compression_percentile': cp}
        r = evaluate(p, btc_4h, btc_1d, btc_fund, eth_4h, eth_1d)
        print(f"  compression_percentile={cp:.2f}: {fmt(r)}")

    print('\n## Breakout N (cuantas velas atras para el high/low)')
    for n in [8, 12, 16, 20]:
        p = {**PARAMS, 'breakout_n': n}
        r = evaluate(p, btc_4h, btc_1d, btc_fund, eth_4h, eth_1d)
        print(f"  breakout_n={n:>2}: {fmt(r)}")

    print('\n## Compression min bars (cuantas velas consecutivas comprimidas)')
    for mb in [2, 3, 5, 8]:
        p = {**PARAMS, 'compression_min_bars': mb}
        r = evaluate(p, btc_4h, btc_1d, btc_fund, eth_4h, eth_1d)
        print(f"  compression_min_bars={mb:>2}: {fmt(r)}")

    print('\n## Regime filter (EMA50_1d vs EMA200_1d)')
    p = {**PARAMS, 'regime_filter_enabled': False}
    print(f"  SIN filtro daily:               {fmt(evaluate(p, btc_4h, btc_1d, btc_fund, eth_4h, eth_1d))}")
    p = {**PARAMS, 'regime_filter_enabled': True}
    print(f"  CON filtro daily (baseline):    {fmt(evaluate(p, btc_4h, btc_1d, btc_fund, eth_4h, eth_1d))}")

    print('\n## Target vol (sizing)')
    for tv in [0.008, 0.012, 0.018, 0.025]:
        p = {**PARAMS, 'target_vol_pct': tv}
        r = evaluate(p, btc_4h, btc_1d, btc_fund, eth_4h, eth_1d)
        print(f"  target_vol_pct={tv:.3f}: {fmt(r)}")

    print('\n## Trailing ATR multiplier')
    for ta in [1.5, 2.0, 2.5, 3.0]:
        p = {**PARAMS, 'trail_atr_mult': ta}
        r = evaluate(p, btc_4h, btc_1d, btc_fund, eth_4h, eth_1d)
        print(f"  trail_atr_mult={ta}: {fmt(r)}")

    print('\n## Vol ratio min')
    for vr in [1.0, 1.2, 1.5, 2.0]:
        p = {**PARAMS, 'vol_ratio_min': vr}
        r = evaluate(p, btc_4h, btc_1d, btc_fund, eth_4h, eth_1d)
        print(f"  vol_ratio_min={vr}: {fmt(r)}")


if __name__ == '__main__':
    main()
