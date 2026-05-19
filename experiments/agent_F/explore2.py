"""
Segunda ronda exploracion: combinaciones que parecen mas razonables A PRIORI.
Reportar HONESTAMENTE, no elegir el mejor.
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
from strategy import PARAMS, prepare_data, run_backtest, metrics, metrics_portfolio_50_50
from train import load_all_data, walk_forward, bootstrap_pvalue


def evaluate(params, btc_4h, btc_1d, btc_fund, eth_4h, eth_1d):
    btc_feat = prepare_data(btc_4h, btc_1d, btc_fund, params)
    eth_feat = prepare_data(eth_4h, eth_1d, None, params)
    wf = walk_forward(btc_feat, eth_feat, params)
    start = pd.Timestamp('2020-01-01', tz='UTC')
    end = pd.Timestamp(params['cutoff_date'], tz='UTC')
    trades = run_backtest(btc_feat, eth_feat, params, start, end)
    M = metrics(trades)
    M5050 = metrics_portfolio_50_50(trades) if trades else {}
    boot = bootstrap_pvalue(trades, n_iter=2000)
    return {
        'wf_agg': f"{wf['folds_ok_agg']}/{wf['folds_total']}",
        'wf_btc': f"{wf['folds_ok_btc']}/{wf['folds_total']}",
        'wf_eth': f"{wf['folds_ok_eth']}/{wf['folds_total']}",
        'n': M['n'], 'wr': M['wr'], 'pf': M['pf'],
        'annual': M['annual_return'], 'dd': M['max_dd'],
        'sharpe': M['sharpe_annual'],
        'annual_5050': M5050.get('annual_return_50_50', 0),
        'dd_5050': M5050.get('max_dd_50_50', 0),
        'boot_p': boot['p_value'] if boot else None,
    }


def fmt(r):
    pf = 'inf' if not np.isfinite(r['pf']) else f"{r['pf']:.2f}"
    bp = f"{r['boot_p']:.3f}" if r['boot_p'] is not None else 'n/a'
    return (f"WF agg {r['wf_agg']} BTC {r['wf_btc']} ETH {r['wf_eth']} | "
            f"N={r['n']:>3} WR={r['wr']*100:>4.1f}% PF={pf:>5} "
            f"ann={r['annual']*100:>+5.1f}% DD={r['dd']*100:>4.1f}% "
            f"5050: ann={r['annual_5050']*100:>+5.1f}% DD={r['dd_5050']*100:>4.1f}% "
            f"shar={r['sharpe']:>4.2f} p={bp}")


def main():
    data = load_all_data(cutoff=PARAMS['cutoff_date'])
    btc_4h, btc_1d, btc_fund = data['btc_4h'], data['btc_1d'], data['btc_fund']
    eth_4h, eth_1d = data['eth_4h'], data['eth_1d']

    print('\n## Combinaciones razonables a priori')
    # 1) Baseline
    print(f"  baseline:                       {fmt(evaluate(PARAMS, btc_4h, btc_1d, btc_fund, eth_4h, eth_1d))}")
    # 2) Mas restrictivo: vol_ratio 1.5 + min_bars 5
    p = {**PARAMS, 'vol_ratio_min': 1.5, 'compression_min_bars': 5}
    print(f"  vr1.5 + min_bars5:              {fmt(evaluate(p, btc_4h, btc_1d, btc_fund, eth_4h, eth_1d))}")
    # 3) Mas conservador: trail 1.5, vol_ratio 1.5
    p = {**PARAMS, 'trail_atr_mult': 1.5, 'vol_ratio_min': 1.5}
    print(f"  trail1.5 + vr1.5:               {fmt(evaluate(p, btc_4h, btc_1d, btc_fund, eth_4h, eth_1d))}")
    # 4) LONG-only + min_bars 5 + vr 1.5
    p = {**PARAMS, 'enable_short': False, 'vol_ratio_min': 1.5, 'compression_min_bars': 5}
    print(f"  LONG + vr1.5 + min_bars5:       {fmt(evaluate(p, btc_4h, btc_1d, btc_fund, eth_4h, eth_1d))}")
    # 5) Sin filtro daily + bidireccional (mas trades)
    p = {**PARAMS, 'regime_filter_enabled': False}
    print(f"  sin daily filter:               {fmt(evaluate(p, btc_4h, btc_1d, btc_fund, eth_4h, eth_1d))}")
    # 6) Combinacion robusta: min_bars 5, vol_ratio 1.5, trail 2.0
    p = {**PARAMS, 'compression_min_bars': 5, 'vol_ratio_min': 1.5, 'trail_atr_mult': 2.0}
    print(f"  ROBUST candidate (mb5+vr1.5):   {fmt(evaluate(p, btc_4h, btc_1d, btc_fund, eth_4h, eth_1d))}")
    # 7) min_bars 8 (mas restrictivo)
    p = {**PARAMS, 'compression_min_bars': 8, 'vol_ratio_min': 1.5}
    print(f"  min_bars8 + vr1.5:              {fmt(evaluate(p, btc_4h, btc_1d, btc_fund, eth_4h, eth_1d))}")
    # 8) min_bars 8 + breakout_n 16
    p = {**PARAMS, 'compression_min_bars': 8, 'breakout_n': 16}
    print(f"  min_bars8 + bn16:               {fmt(evaluate(p, btc_4h, btc_1d, btc_fund, eth_4h, eth_1d))}")
    # 9) target_vol 0.018 con leverage cap 4
    p = {**PARAMS, 'target_vol_pct': 0.018, 'max_leverage': 4.0}
    print(f"  tv1.8 + maxlev4:                {fmt(evaluate(p, btc_4h, btc_1d, btc_fund, eth_4h, eth_1d))}")


if __name__ == '__main__':
    main()
