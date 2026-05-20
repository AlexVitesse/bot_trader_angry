"""
train.py — Validacion in-sample de la estrategia Agent H (ETH/BTC ratio).

Reglas inviolables:
  - Cutoff <= 2025-12-31
  - Walk-forward por semestre con purga de 14 dias
  - Bootstrap >= 2000 iter
  - Una posicion a la vez (sin solape)
  - Sin look-ahead intrabar en el trailing
  - Sin re-tunear params: PARAMS frozen en strategy.py

Test critico de control:
  - Reemplazar el ratio real por un random walk con misma estadistica
  - Repetir 30 veces (seeds distintas)
  - Si la annual del real esta dentro del p25-p75 del random
    -> el "edge del ratio" es indistinguible de azar -> REJECT
  - Si esta por encima del p95 random -> edge atribuible al ratio

Uso:
  C:/Python/python.exe experiments/agent_H/train.py
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

    eth = pd.read_parquet(DATA / 'ETH_USDT_4h_full.parquet')
    if eth.index.tz is None:
        eth.index = eth.index.tz_localize('UTC')
    eth = eth[eth.index <= cut].sort_index()

    btc = pd.read_parquet(DATA / 'BTC_USDT_4h_full.parquet')
    if btc.index.tz is None:
        btc.index = btc.index.tz_localize('UTC')
    btc = btc[btc.index <= cut].sort_index()

    ratio = pd.read_parquet(DATA / 'ethbtc_daily_history.parquet')
    if ratio.index.tz is None:
        ratio.index = ratio.index.tz_localize('UTC')
    ratio = ratio[ratio.index <= cut].sort_index()

    fund = pd.read_parquet(DATA / 'btc_v15_funding.parquet')
    if fund.index.tz is None:
        fund.index = fund.index.tz_localize('UTC')
    fund = fund[fund.index <= cut].sort_index()

    print(f'  ETH 4h: {len(eth)} bars, {eth.index.min().date()} -> {eth.index.max().date()}')
    print(f'  BTC 4h: {len(btc)} bars, {btc.index.min().date()} -> {btc.index.max().date()}')
    print(f'  Ratio 1d: {len(ratio)} bars, {ratio.index.min().date()} -> {ratio.index.max().date()}')
    print(f'  Funding: {len(fund)} rows, {fund.index.min().date()} -> {fund.index.max().date()}')

    assert eth.index.max() <= cut, "VIOLACION cutoff ETH"
    assert btc.index.max() <= cut, "VIOLACION cutoff BTC"
    assert ratio.index.max() <= cut, "VIOLACION cutoff ratio"
    assert fund.index.max() <= cut, "VIOLACION cutoff funding"

    return {'eth': eth, 'btc': btc, 'ratio': ratio, 'fund': fund}


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
        start = pd.Timestamp(start_s, tz='UTC')
        end = pd.Timestamp(end_s, tz='UTC')
        purge_until = start + pd.Timedelta(days=PURGE_DAYS)

        in_window = (df_features.index >= start) & (df_features.index <= end)
        idxs = np.where(in_window)[0]
        if len(idxs) < 200:
            fold_results.append({'period': start_s[:7], 'n': 0, 'pf': 0.0,
                                 'wr': 0.0, 'total': 0.0, 'monthly': 0.0,
                                 'max_dd': 0.0,
                                 'ok': False, 'status': 'no_data', 'no_data': True})
            continue
        start_i, end_i = idxs[0], idxs[-1] + 1
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
    folds_evaluated = sum(1 for r in fold_results
                          if not r['no_data'] and r.get('status') == 'evaluated')
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
# CONTROL TEST CRITICO: random ratio
# =============================================================================
def synthesize_random_ratio_like(real_ratio: pd.Series, seed: int) -> pd.Series:
    """
    Genera una serie 'ratio aleatoria' con MISMA distribucion estadistica
    (mismo dia inicial, misma volatilidad de log-returns, mismo punto final
    aproximado) pero ORDEN ALEATORIO de los movimientos.

    Mantiene: longitud, indice, escala global del ratio (alrededor de la
    misma media), volatilidad de log-returns.

    Pierde: la estructura temporal del ratio real (uptrends, downtrends,
    swings) — exactamente lo que la estrategia explota.

    Si el edge no cae al usar este ratio aleatorio -> el edge NO provenia
    del ratio.
    """
    rng = np.random.default_rng(seed)
    real = real_ratio.dropna().values.astype(float)
    log_ret = np.log(real[1:] / real[:-1])
    # shuffle in place
    perm = rng.permutation(len(log_ret))
    new_close = np.empty(len(real))
    new_close[0] = real[0]
    new_close[1:] = real[0] * np.exp(np.cumsum(log_ret[perm]))
    return pd.Series(new_close, index=real_ratio.dropna().index, name='close')


def run_random_control(eth, btc, real_ratio, fund, params, n_repeats=30) -> dict:
    """
    Repite el backtest reemplazando el ratio real por un ratio aleatorio.
    Devuelve distribucion de annual_returns.
    """
    print(f'\n--- CONTROL TEST: ratio shuffled (n={n_repeats} seeds) ---')
    annuals = []
    n_trades_list = []
    pf_list = []
    for seed in range(n_repeats):
        rand_close = synthesize_random_ratio_like(real_ratio['close'], seed=seed)
        rand_df = pd.DataFrame({'close': rand_close})
        df_feat = prepare_data(eth, btc, rand_df, fund, params)
        start_full = pd.Timestamp('2020-01-01', tz='UTC')
        idxs = np.where(df_feat.index >= start_full)[0]
        si = int(idxs[0]) if len(idxs) else 0
        trades = run_backtest(df_feat, params, start_i=si, end_i=len(df_feat))
        m = metrics(trades)
        annuals.append(m['annual_return'])
        n_trades_list.append(m['n'])
        pf_list.append(m['pf'] if np.isfinite(m['pf']) else 999)
        if (seed + 1) % 10 == 0:
            print(f'  seed {seed+1}/{n_repeats}: n={m["n"]}, annual={m["annual_return"]*100:+.2f}%')
    annuals = np.array(annuals)
    n_trades_arr = np.array(n_trades_list)
    return {
        'mean_annual': float(annuals.mean()),
        'median_annual': float(np.median(annuals)),
        'p5_annual': float(np.percentile(annuals, 5)),
        'p25_annual': float(np.percentile(annuals, 25)),
        'p75_annual': float(np.percentile(annuals, 75)),
        'p95_annual': float(np.percentile(annuals, 95)),
        'min_annual': float(annuals.min()),
        'max_annual': float(annuals.max()),
        'mean_n_trades': float(n_trades_arr.mean()),
        'frac_positive': float((annuals > 0).mean()),
        'all_annuals': [float(x) for x in annuals],
    }


# =============================================================================
# MAIN
# =============================================================================
def fmt_pf(pf):
    return 'inf' if not np.isfinite(pf) else f'{pf:.2f}'


def main():
    print('=' * 78)
    print('AGENT H — ETH/BTC RATIO ROTATION (ETH/USDT 4h, LONG-ONLY)')
    print('=' * 78)

    data = load_all_data(cutoff=PARAMS['cutoff_date'])
    eth, btc, ratio, fund = data['eth'], data['btc'], data['ratio'], data['fund']

    print('\nPreparando features (REAL ratio)...')
    df_feat = prepare_data(eth, btc, ratio, fund, PARAMS)
    print(f'  features ready: {len(df_feat)} bars, '
          f'{df_feat.index.min().date()} -> {df_feat.index.max().date()}')

    # Diagnostico de senales
    has_uptrend = (df_feat['ratio_uptrend'] >= 1).sum()
    has_accel = (df_feat['ratio_accel'] >= 1).sum()
    has_oversold = (df_feat['ratio_oversold'] >= 1).sum()
    btc_bull = (df_feat['btc_bull_1d'] >= 1).sum()
    print(f'  Senal diagn: uptrend={has_uptrend}, accel={has_accel}, '
          f'oversold={has_oversold}, btc_bull={btc_bull} / {len(df_feat)} bars')

    # WF
    print(f'\nWalk-forward {len(WF_SEMESTERS)} semestres '
          f'(purga: {PURGE_DAYS}d)...')
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
            flag = '[no-signal]'
        elif st == 'small_sample':
            flag = '[small-n]'
        else:
            flag = '[+]' if f['ok'] else '[-]'
        print(f"  {f['period']:<10} {f['n']:>4} {f['wr']*100:>5.1f}% "
              f"{fmt_pf(f['pf']):>7} {f['total']*100:>+8.1f}% "
              f"{f['monthly']*100:>+8.2f}% {f['max_dd']*100:>6.1f}% {flag}")

    # Backtest global
    print('\nBacktest global 2020-01-01 -> cutoff...')
    start_full = pd.Timestamp('2020-01-01', tz='UTC')
    idxs = np.where(df_feat.index >= start_full)[0]
    start_i = int(idxs[0]) if len(idxs) else 0
    all_trades = run_backtest(df_feat, PARAMS, start_i=start_i, end_i=len(df_feat))
    M = metrics(all_trades)
    print(f"  N={M['n']}  WR={M['wr']*100:.1f}%  PF={fmt_pf(M['pf'])}  "
          f"total={M['total_return']*100:+.1f}%  "
          f"monthly={M['monthly_return']*100:+.2f}%  "
          f"annual={M['annual_return']*100:+.2f}%  "
          f"DD={M['max_dd']*100:.1f}%  "
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
        print(f"  retorno p5: {boot['pctl_5']*100:+.1f}%")
        print(f"  retorno p95: {boot['pctl_95']*100:+.1f}%")

    # Sanity
    print('\nSANITY CHECKS:')
    if M['pf'] > 4:
        print(f"  [!] PF {M['pf']:.2f} > 4 -> sospechar overfitting/look-ahead")
    if M['wr'] > 0.65:
        print(f"  [!] WR {M['wr']*100:.1f}% > 65% -> sospechar")
    if M['max_dd'] < 0.05:
        print(f"  [!] DD {M['max_dd']*100:.1f}% < 5% -> sospechar (sample no adverso)")
    if M['pf'] <= 4 and M['wr'] <= 0.65 and M['max_dd'] >= 0.05:
        print('  [+] PF/WR/DD en rango razonable')

    # CONTROL TEST: random ratio
    control = run_random_control(eth, btc, ratio, fund, PARAMS, n_repeats=30)
    print(f"\n  random ratio annual: mean={control['mean_annual']*100:+.2f}%, "
          f"median={control['median_annual']*100:+.2f}%")
    print(f"  random p25-p75: {control['p25_annual']*100:+.2f}% .. "
          f"{control['p75_annual']*100:+.2f}%")
    print(f"  random p5-p95: {control['p5_annual']*100:+.2f}% .. "
          f"{control['p95_annual']*100:+.2f}%")
    print(f"  random mean n_trades: {control['mean_n_trades']:.1f}, "
          f"frac positive: {control['frac_positive']*100:.0f}%")

    # Veredicto del control: donde cae el real
    real_annual = M['annual_return']
    p_above_random = float((np.array(control['all_annuals']) >= real_annual).mean())
    diff_vs_median = (real_annual - control['median_annual']) * 100
    print(f"\n  REAL annual: {real_annual*100:+.2f}%")
    print(f"  posicion del REAL en la distribucion del CONTROL: "
          f"{(1-p_above_random)*100:.0f}-percentil "
          f"(p_above_random_ge_real={p_above_random:.3f})")
    print(f"  diff vs median random: {diff_vs_median:+.2f}%")
    if p_above_random < 0.05:
        verdict_control = 'PASA — edge atribuible al ratio (real > p95 random)'
    elif p_above_random < 0.25:
        verdict_control = 'MARGINAL — real > p75 random'
    else:
        verdict_control = 'FALLA — edge indistinguible de ratio aleatorio'
    print(f"  CONTROL VERDICT: {verdict_control}")

    # Guardar
    summary = {
        'params': PARAMS,
        'wf': {'folds_ok': wf['folds_ok'], 'folds_total': wf['folds_total'],
               'folds_evaluated': wf['folds_evaluated'],
               'folds': wf['folds']},
        'overall': M,
        'bootstrap': boot,
        'control_random_ratio': control,
        'control_verdict': verdict_control,
        'control_p_above_random': p_above_random,
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
