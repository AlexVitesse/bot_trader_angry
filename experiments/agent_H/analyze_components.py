"""
analyze_components.py — Diagnostico transparente de las dos subsenales:
  1) PRIMARY: ratio_uptrend + ratio_accel
  2) SECONDARY: ratio_oversold (mean-rev)

Tambien corre el control test (random ratio) sobre cada subsenal por separado.
Esto NO tunea params nuevos — solo aisla los dos componentes ya definidos
en strategy.py para entender que aporta cada uno.

Si secondary parece tener edge: corre bootstrap + random control para verificar
si el edge es atribuible al ratio o azar.
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
from strategy import PARAMS, prepare_data, run_backtest, metrics

ROOT = HERE.parent.parent
DATA = ROOT / 'data'

# Importar funciones del train.py
sys.path.insert(0, str(HERE))
from train import (load_all_data, walk_forward, bootstrap_pvalue,
                   synthesize_random_ratio_like, fmt_pf)


def run_with_signal_filter(eth, btc, ratio, fund, params, mode: str):
    """
    mode in {'primary', 'secondary', 'both'}. Hace prepare + run_backtest
    forzando una de las subsenales a 0 si corresponde.
    """
    p = dict(params)
    df = prepare_data(eth, btc, ratio, fund, p)
    if mode == 'primary':
        df = df.copy()
        df['ratio_oversold'] = 0
    elif mode == 'secondary':
        df = df.copy()
        # forzar uptrend o accel a 0 -> primary nunca dispara
        df['ratio_uptrend'] = 0
    # else: both -> df sin tocar
    return df


def main():
    print('=' * 78)
    print('AGENT H — analisis por COMPONENTE (PRIMARY vs SECONDARY)')
    print('=' * 78)
    data = load_all_data(cutoff=PARAMS['cutoff_date'])
    eth, btc, ratio, fund = data['eth'], data['btc'], data['ratio'], data['fund']

    start = pd.Timestamp('2020-01-01', tz='UTC')
    end = pd.Timestamp('2025-12-31', tz='UTC')

    results_per_mode = {}
    for mode in ['both', 'primary', 'secondary']:
        print(f'\n--- {mode.upper()} ---')
        df = run_with_signal_filter(eth, btc, ratio, fund, PARAMS, mode=mode)
        idxs = np.where((df.index >= start) & (df.index <= end))[0]
        si, ei = int(idxs[0]), int(idxs[-1]) + 1
        trades = run_backtest(df, PARAMS, start_i=si, end_i=ei)
        m = metrics(trades)
        boot = bootstrap_pvalue(trades, n_iter=3000)
        wf = walk_forward(df, PARAMS)
        print(f"  N={m['n']} WR={m['wr']*100:.1f}% PF={fmt_pf(m['pf'])} "
              f"total={m['total_return']*100:+.1f}% "
              f"annual={m['annual_return']*100:+.2f}% "
              f"DD={m['max_dd']*100:.1f}%")
        if boot:
            print(f"  bootstrap p={boot['p_value']:.4f}  "
                  f"median resample={boot['pctl_50']*100:+.1f}%")
        print(f"  WF folds_ok={wf['folds_ok']}/{wf['folds_evaluated']} evaluated")
        results_per_mode[mode] = {
            'metrics': m, 'bootstrap': boot,
            'wf_folds_ok': wf['folds_ok'],
            'wf_folds_evaluated': wf['folds_evaluated'],
            'wf_folds_total': wf['folds_total'],
            'folds': wf['folds'],
        }

    # Random control para SECONDARY (la subsenal que parece tener edge)
    print('\n--- CONTROL TEST (RANDOM RATIO) sobre SECONDARY-only ---')
    print('Si la SECONDARY tiene edge real del ratio, el random debe quedar peor.')
    annuals = []
    n_trades_list = []
    for seed in range(30):
        rand_close = synthesize_random_ratio_like(ratio['close'], seed=seed)
        rand_df = pd.DataFrame({'close': rand_close})
        df_feat = prepare_data(eth, btc, rand_df, fund, PARAMS)
        df_feat = df_feat.copy()
        df_feat['ratio_uptrend'] = 0  # SECONDARY-only
        idxs = np.where((df_feat.index >= start) & (df_feat.index <= end))[0]
        si, ei = int(idxs[0]), int(idxs[-1]) + 1
        trades = run_backtest(df_feat, PARAMS, start_i=si, end_i=ei)
        m = metrics(trades)
        annuals.append(m['annual_return'])
        n_trades_list.append(m['n'])
        if (seed + 1) % 10 == 0:
            print(f'  seed {seed+1}/30: n={m["n"]}, annual={m["annual_return"]*100:+.2f}%')
    annuals = np.array(annuals)
    real_annual = results_per_mode['secondary']['metrics']['annual_return']
    p_above = float((annuals >= real_annual).mean())
    pos_pct = (1 - p_above) * 100
    print(f"\n  SECONDARY-only REAL annual: {real_annual*100:+.2f}%")
    print(f"  random p25/p50/p75: {np.percentile(annuals,25)*100:+.2f}% / "
          f"{np.percentile(annuals,50)*100:+.2f}% / "
          f"{np.percentile(annuals,75)*100:+.2f}%")
    print(f"  random p5/p95: {np.percentile(annuals,5)*100:+.2f}% / "
          f"{np.percentile(annuals,95)*100:+.2f}%")
    print(f"  real esta en el {pos_pct:.0f}-percentil de la distribucion random")
    if p_above < 0.05:
        verdict = 'PASA — edge atribuible al ratio'
    elif p_above < 0.25:
        verdict = 'MARGINAL — real > p75 random pero no p95'
    else:
        verdict = 'FALLA — edge indistinguible de ratio aleatorio'
    print(f"  VERDICT (secondary control): {verdict}")

    secondary_control = {
        'real_annual': real_annual,
        'random_mean_annual': float(annuals.mean()),
        'random_median': float(np.median(annuals)),
        'random_p25': float(np.percentile(annuals, 25)),
        'random_p75': float(np.percentile(annuals, 75)),
        'random_p5': float(np.percentile(annuals, 5)),
        'random_p95': float(np.percentile(annuals, 95)),
        'real_percentile': pos_pct,
        'p_above_random_ge_real': p_above,
        'verdict': verdict,
        'random_mean_n_trades': float(np.mean(n_trades_list)),
    }

    out = {
        'modes': {k: {
            'metrics': v['metrics'],
            'bootstrap': v['bootstrap'],
            'wf_folds_ok': v['wf_folds_ok'],
            'wf_folds_evaluated': v['wf_folds_evaluated'],
            'wf_folds_total': v['wf_folds_total'],
            'folds': v['folds'],
        } for k, v in results_per_mode.items()},
        'secondary_random_control': secondary_control,
    }
    out_json = HERE / 'components.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, default=str, indent=2)
    print(f'\nResultados guardados en {out_json}')


if __name__ == '__main__':
    main()
