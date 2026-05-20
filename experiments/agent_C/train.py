"""
agent_C/train.py
================
Investigation + validation of the regime-adaptive strategy.

REGLA INVIOLABLE: only data with timestamp <= 2025-12-31.

Validation layers:
  B. Walk-forward purged 12 semesters with >=2 week gap on each side
  C. Bootstrap (n=3000) over the in-sample trades
  R. Per-regime walk-forward: each sub-strategy must earn its own keep

Output: prints metrics and writes a JSON summary.

Run with: C:/Python/python.exe experiments/agent_C/train.py
"""

import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import strategy as S


CUTOFF = pd.Timestamp('2025-12-31 23:59:59', tz='UTC')
COMMISSION = S.PARAMS['commission_one_way']

# 12 semesters 2020-2025 (the V15 baseline windows)
WF_FOLDS = [
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
PURGE_DAYS = 14  # 2-week gap each side


def load_btc_4h():
    df = pd.read_parquet(ROOT / 'data' / 'BTC_USDT_4h_full.parquet')
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df.sort_index()
    # CUTOFF — invariolable
    df = df[df.index <= CUTOFF]
    return df


def fold_indices_purged(df, start_s, end_s, purge_days=PURGE_DAYS):
    """
    Return the boolean mask of bars belonging to this fold AFTER applying
    purge: drop the first and last `purge_days` of the fold so train/test
    windows can't leak through overlapping ATR/EMA tails.
    """
    s = pd.Timestamp(start_s, tz='UTC') + pd.Timedelta(days=purge_days)
    e = pd.Timestamp(end_s, tz='UTC') - pd.Timedelta(days=purge_days)
    if e <= s:
        return np.array([], dtype=int)
    mask = (df.index >= s) & (df.index <= e)
    return np.where(mask)[0]


def calc_metrics(trades):
    if not trades:
        return {'n': 0, 'wr': 0.0, 'pf': 0.0, 'avg': 0.0, 'total': 0.0,
                'equity': 1.0, 'max_dd': 0.0}
    n = len(trades)
    pnls = np.array([t['pnl_pct'] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    wr = len(wins) / n
    gw = wins.sum() if len(wins) else 0.0
    gl = abs(losses.sum()) if len(losses) else 0.0
    pf = (gw / gl) if gl > 1e-9 else float('inf')
    cum, peak, max_dd = 1.0, 1.0, 0.0
    for t in sorted(trades, key=lambda x: x['ts']):
        cum *= (1 + t['pnl_pct'])
        peak = max(peak, cum)
        max_dd = max(max_dd, (peak - cum) / peak)
    return {'n': n, 'wr': wr, 'pf': pf, 'avg': float(pnls.mean()),
            'total': cum - 1.0, 'equity': cum, 'max_dd': max_dd}


def fold_ok(m, pf_min=1.2, n_min=5):
    if m['n'] < n_min:
        return False
    if not np.isfinite(m['pf']):
        return False
    return m['pf'] >= pf_min and m['total'] > 0


def bootstrap_pvalue(trades, n_iter=3000, seed=42):
    if len(trades) < 5:
        return None
    rng = np.random.default_rng(seed)
    pnls = np.array([t['pnl_pct'] for t in trades])
    k = len(pnls)
    totals = np.empty(n_iter)
    for j in range(n_iter):
        sample = rng.choice(pnls, size=k, replace=True)
        totals[j] = np.prod(1 + sample) - 1.0
    return {
        'p_value': float(np.mean(totals <= 0)),
        'pctl_5': float(np.percentile(totals, 5)),
        'pctl_50': float(np.percentile(totals, 50)),
    }


def monthly_returns(trades):
    """Return a pd.Series of monthly compounded returns (in fraction)."""
    if not trades:
        return pd.Series(dtype=float)
    df = pd.DataFrame([{'ts': t['ts'], 'pnl': t['pnl_pct']} for t in trades])
    df['ts'] = pd.to_datetime(df['ts'])
    df['month'] = df['ts'].dt.to_period('M')
    out = df.groupby('month')['pnl'].apply(lambda s: np.prod(1 + s) - 1)
    return out


def annualized_return_from_trades(trades):
    """Geometric-mean monthly return -> annualize."""
    mr = monthly_returns(trades)
    if len(mr) < 3:
        return 0.0
    # geometric mean per month
    gmean_m = np.exp(np.log(1 + mr).mean()) - 1
    return (1 + gmean_m)**12 - 1


def run_wf(df, params, label='full'):
    """Walk-forward by semester, with purge on each fold."""
    folds_results = []
    all_trades = []
    for start_s, end_s in WF_FOLDS:
        idxs = fold_indices_purged(df, start_s, end_s)
        if len(idxs) < 100:
            folds_results.append({'period': start_s[:7], 'n': 0, 'ok': False,
                                  'nodata': True})
            continue
        trades = S.run_engine(df, params, int(idxs[0]), int(idxs[-1]) + 1)
        m = calc_metrics(trades)
        folds_results.append({
            'period': start_s[:7],
            'n': m['n'], 'wr': m['wr'], 'pf': m['pf'],
            'total': m['total'], 'max_dd': m['max_dd'],
            'ok': fold_ok(m), 'nodata': False,
        })
        all_trades.extend(trades)
    folds_ok = sum(1 for f in folds_results if f['ok'])
    folds_total = sum(1 for f in folds_results if not f['nodata'])
    return folds_results, folds_ok, folds_total, all_trades


def run_full_period(df, params, start='2020-01-01', end='2025-12-31'):
    """Continuous run, no fold split, for global metrics + bootstrap."""
    s = pd.Timestamp(start, tz='UTC')
    e = pd.Timestamp(end, tz='UTC')
    idxs = np.where((df.index >= s) & (df.index <= e))[0]
    if len(idxs) < 100:
        return []
    return S.run_engine(df, params, int(idxs[0]), int(idxs[-1]) + 1)


# -----------------------------------------------------------------------------
# Per-regime audit: build a strategy variant where only ONE regime fires
# -----------------------------------------------------------------------------
def variant_only(regime_keep):
    """Return a PARAMS copy that only allows trades in the chosen regime."""
    p = dict(S.PARAMS)
    p['_only_regime'] = regime_keep
    return p


class RegimeFilteredStrategy:
    """Wrap the strategy so we can call run_engine with a regime filter."""
    def __init__(self, only_regime):
        self.only_regime = only_regime

    def run(self, df, params, start_i, end_i):
        trades = []
        i = max(start_i, params.get('min_history_bars', 220))
        end_i = min(end_i, len(df) - 2)
        while i < end_i:
            s, regime = S.signal(df, i, params)
            if s is None or regime != self.only_regime:
                i += 1
                continue
            t = S.simulate(df, i, params, regime)
            if t['outcome'] == 'NONE':
                i += 1
                continue
            trades.append(t)
            i += int(t['bars']) + 1
        return trades


def run_wf_regime(df, params, only_regime):
    """WF where only signals from `only_regime` fire. Used for per-regime audit."""
    strat = RegimeFilteredStrategy(only_regime)
    folds_results = []
    all_trades = []
    for start_s, end_s in WF_FOLDS:
        idxs = fold_indices_purged(df, start_s, end_s)
        if len(idxs) < 100:
            folds_results.append({'period': start_s[:7], 'n': 0, 'ok': False,
                                  'nodata': True})
            continue
        trades = strat.run(df, params, int(idxs[0]), int(idxs[-1]) + 1)
        m = calc_metrics(trades)
        folds_results.append({
            'period': start_s[:7], 'n': m['n'], 'wr': m['wr'], 'pf': m['pf'],
            'total': m['total'], 'ok': fold_ok(m), 'nodata': False,
        })
        all_trades.extend(trades)
    folds_ok = sum(1 for f in folds_results if f['ok'])
    folds_total = sum(1 for f in folds_results if not f['nodata'])
    return folds_results, folds_ok, folds_total, all_trades


# -----------------------------------------------------------------------------
def main():
    print('=' * 70)
    print('AGENT C — Regime-Adaptive BTC/USDT 4h')
    print('=' * 70)
    print(f'Loading BTC 4h, capping at {CUTOFF}...')
    df_raw = load_btc_4h()
    print(f'  Raw bars: {len(df_raw)}, range: {df_raw.index.min()} -> {df_raw.index.max()}')

    df = S.prepare(df_raw)
    # SANITY: confirm cutoff is respected after prepare()
    assert df.index.max() <= CUTOFF, f"CUTOFF VIOLATION: {df.index.max()}"
    print(f'  Prepared: {len(df)} bars (drop NaN warmup)')

    # ---- regime distribution sanity ----
    regimes_series = pd.Series([S.detect_regime(df, i) for i in range(len(df))],
                               index=df.index)
    print('\nRegime distribution (in-sample 2020-2025):')
    print(regimes_series[regimes_series.index >= pd.Timestamp('2020-01-01', tz='UTC')]
          .value_counts(normalize=True).round(3).to_string())

    # ---- WF combined ----
    print('\n' + '-' * 70)
    print('WALK-FORWARD COMBINED (BULL pullback + RANGE meanrev)')
    print('-' * 70)
    folds, ok, total, wf_trades = run_wf(df, S.PARAMS, 'combined')
    print(f'  Folds passed (n>=5, PF>=1.2, total>0): {ok}/{total}')
    print(f'  {"Period":<10} {"N":>4} {"WR":>7} {"PF":>7} {"Tot":>9} {"DD":>7} {"OK":>3}')
    for f in folds:
        if f.get('nodata'):
            print(f'  {f["period"]:<10}   --   (no data)')
        else:
            pfs = 'inf' if not np.isfinite(f['pf']) else f'{f["pf"]:.2f}'
            print(f'  {f["period"]:<10} {f["n"]:>4} {f["wr"]:>6.0%} '
                  f'{pfs:>7} {f["total"]:>+8.1%} {f["max_dd"]:>6.1%} '
                  f'{"OK" if f["ok"] else "no":>3}')

    m_wf = calc_metrics(wf_trades)
    print(f'\n  Combined WF metrics: N={m_wf["n"]} WR={m_wf["wr"]:.1%} '
          f'PF={"inf" if not np.isfinite(m_wf["pf"]) else f"{m_wf['pf']:.2f}"} '
          f'Total={m_wf["total"]:+.1%} DD={m_wf["max_dd"]:.1%}')

    # ---- Full continuous run 2020-2025 (for bootstrap + monthly) ----
    print('\n' + '-' * 70)
    print('CONTINUOUS RUN 2020-01-01 -> 2025-12-31 (no fold split)')
    print('-' * 70)
    cont_trades = run_full_period(df, S.PARAMS)
    m_cont = calc_metrics(cont_trades)
    print(f'  N={m_cont["n"]} WR={m_cont["wr"]:.1%} '
          f'PF={"inf" if not np.isfinite(m_cont["pf"]) else f"{m_cont['pf']:.2f}"} '
          f'Total={m_cont["total"]:+.1%} DD={m_cont["max_dd"]:.1%}')
    monthly = monthly_returns(cont_trades)
    if len(monthly):
        print(f'  Months observed: {len(monthly)}, mean monthly return '
              f'(arith): {monthly.mean()*100:.2f}%, median: {monthly.median()*100:.2f}%')
        # months win-rate
        mwr = (monthly > 0).mean()
        print(f'  Months positive: {mwr:.1%}')
    ann = annualized_return_from_trades(cont_trades)
    print(f'  Annualized return (geometric monthly): {ann*100:.1f}%')

    # ---- bootstrap ----
    boot = bootstrap_pvalue(cont_trades)
    if boot:
        print(f"  Bootstrap p-value (return<=0 by chance): {boot['p_value']:.3f}")
        print(f"  Median resampled total: {boot['pctl_50']:+.1%}, "
              f"5th pctile: {boot['pctl_5']:+.1%}")

    # ---- per-regime audit ----
    print('\n' + '-' * 70)
    print('PER-REGIME WALK-FORWARD AUDIT (each sub-strategy alone)')
    print('-' * 70)
    per_regime = {}
    for reg in ('BULL', 'RANGE'):
        folds_r, ok_r, total_r, tr_r = run_wf_regime(df, S.PARAMS, reg)
        m_r = calc_metrics(tr_r)
        boot_r = bootstrap_pvalue(tr_r)
        print(f'\n  [{reg}] folds {ok_r}/{total_r} | N={m_r["n"]} '
              f'WR={m_r["wr"]:.1%} PF={"inf" if not np.isfinite(m_r["pf"]) else f"{m_r['pf']:.2f}"} '
              f'Total={m_r["total"]:+.1%} DD={m_r["max_dd"]:.1%}')
        if boot_r:
            print(f'       bootstrap p={boot_r["p_value"]:.3f}')
        print(f'       {"Period":<10} {"N":>4} {"WR":>6} {"PF":>7} {"Tot":>9} {"OK":>3}')
        for f in folds_r:
            if f.get('nodata'):
                continue
            pfs = 'inf' if not np.isfinite(f['pf']) else f'{f["pf"]:.2f}'
            print(f'       {f["period"]:<10} {f["n"]:>4} {f["wr"]:>5.0%} '
                  f'{pfs:>7} {f["total"]:>+8.1%} {"OK" if f["ok"] else "no":>3}')
        per_regime[reg] = {
            'wf': f'{ok_r}/{total_r}',
            'n': m_r['n'], 'wr': m_r['wr'], 'pf': m_r['pf'],
            'total': m_r['total'], 'max_dd': m_r['max_dd'],
            'bootstrap_p': boot_r['p_value'] if boot_r else None,
        }

    # ---- monthly return summary ----
    monthly_avg = float(monthly.mean()) if len(monthly) else 0.0

    # ---- write summary ----
    pf_med = float(np.nanmedian(
        [f['pf'] for f in folds if not f.get('nodata') and np.isfinite(f['pf'])]
    )) if folds else 0.0

    summary = {
        'agent': 'C',
        'strategy_name': 'Regime-Adaptive BULL-pullback + RANGE-meanrev (BEAR flat)',
        'cutoff': str(CUTOFF.date()),
        'regimes_active': ['BULL', 'RANGE'],
        'params': {k: v for k, v in S.PARAMS.items() if not k.startswith('_')},
        'in_sample_wf': f'{ok}/{total}',
        'in_sample_pf_median': pf_med,
        'in_sample_wr': float(m_wf['wr']),
        'in_sample_n_trades': m_wf['n'],
        'in_sample_total_return': float(m_wf['total']),
        'in_sample_max_dd': float(m_wf['max_dd']),
        'continuous_pf': float(m_cont['pf']) if np.isfinite(m_cont['pf']) else None,
        'continuous_total': float(m_cont['total']),
        'continuous_max_dd': float(m_cont['max_dd']),
        'monthly_return_arith_mean': monthly_avg,
        'annualized_return_geom': float(ann),
        'months_observed': int(len(monthly)),
        'months_positive_pct': float((monthly > 0).mean()) if len(monthly) else 0.0,
        'bootstrap_pvalue': boot['p_value'] if boot else None,
        'bootstrap_median': boot['pctl_50'] if boot else None,
        'per_regime': per_regime,
        'regime_distribution': regimes_series.value_counts(normalize=True).round(3).to_dict(),
    }
    out = HERE / 'results.json'
    out.write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    print(f'\nSaved summary -> {out}')

    # ---- self-audit alerts ----
    print('\n' + '=' * 70)
    print('SELF-AUDIT (sanity flags)')
    print('=' * 70)
    flags = []
    if np.isfinite(m_wf['pf']) and m_wf['pf'] > 4.0:
        flags.append(f'PF combined {m_wf["pf"]:.2f} > 4 -> investigate')
    if m_wf['wr'] > 0.65:
        flags.append(f'WR {m_wf["wr"]:.1%} > 65% -> investigate')
    if m_wf['max_dd'] < 0.05:
        flags.append(f'DD {m_wf["max_dd"]:.1%} < 5% -> investigate (sample too clean?)')
    if m_wf['n'] < 20:
        flags.append(f'n_trades {m_wf["n"]} < 20 -> too few for inference')
    if flags:
        for f in flags:
            print(f'  ! {f}')
    else:
        print('  OK — no automatic sanity flag tripped')


if __name__ == '__main__':
    main()
