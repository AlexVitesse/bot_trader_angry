"""
agent_I/train.py
================
Investigation + validation of ETH RANGE mean-reversion strategy.

REGLA INVIOLABLE: only data with timestamp <= 2025-12-31.

Validation layers:
  B. Walk-forward purged 12 semesters with >=2 week gap on each side
  C. Bootstrap (n=3000) over the in-sample trades
  D. Per-direction WF: LONG-only and SHORT-only
  E. Regime frequency: how much of the sample is RANGE? are RANGE trades the
     positive ones?

Output: prints metrics and writes a JSON summary.

Run with:  C:/Python/python.exe experiments/agent_I/train.py
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
PURGE_DAYS = 14


def load_eth_4h():
    df = pd.read_parquet(ROOT / 'data' / 'ETH_USDT_4h_full.parquet')
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df.sort_index()
    df = df[df.index <= CUTOFF]
    return df


def fold_indices_purged(df, start_s, end_s, purge_days=PURGE_DAYS):
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
        'pctl_95': float(np.percentile(totals, 95)),
    }


def monthly_returns(trades):
    if not trades:
        return pd.Series(dtype=float)
    df = pd.DataFrame([{'ts': t['ts'], 'pnl': t['pnl_pct']} for t in trades])
    df['ts'] = pd.to_datetime(df['ts'])
    df['month'] = df['ts'].dt.to_period('M')
    return df.groupby('month')['pnl'].apply(lambda s: np.prod(1 + s) - 1)


def annualized_return_from_trades(trades):
    mr = monthly_returns(trades)
    if len(mr) < 3:
        return 0.0
    gmean_m = np.exp(np.log(1 + mr).mean()) - 1
    return (1 + gmean_m)**12 - 1


def run_wf(df, params, side_filter=None):
    """Walk-forward. side_filter in {None,'LONG','SHORT'} restricts the side."""
    folds_results = []
    all_trades = []
    for start_s, end_s in WF_FOLDS:
        idxs = fold_indices_purged(df, start_s, end_s)
        if len(idxs) < 100:
            folds_results.append({'period': start_s[:7], 'n': 0, 'ok': False,
                                  'nodata': True})
            continue
        trades = S.run_engine(df, params, int(idxs[0]), int(idxs[-1]) + 1)
        if side_filter is not None:
            trades = [t for t in trades if t['side'] == side_filter]
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
    s = pd.Timestamp(start, tz='UTC')
    e = pd.Timestamp(end, tz='UTC')
    idxs = np.where((df.index >= s) & (df.index <= e))[0]
    if len(idxs) < 100:
        return []
    return S.run_engine(df, params, int(idxs[0]), int(idxs[-1]) + 1)


def print_folds(label, folds):
    print(f'  [{label}]   {"Period":<10} {"N":>4} {"WR":>6} {"PF":>7} {"Tot":>9} {"DD":>7} OK')
    for f in folds:
        if f.get('nodata'):
            print(f'           {f["period"]:<10}   --   (no data)')
            continue
        pfs = 'inf' if not np.isfinite(f['pf']) else f'{f["pf"]:.2f}'
        print(f'           {f["period"]:<10} {f["n"]:>4} {f["wr"]:>5.0%} '
              f'{pfs:>7} {f["total"]:>+8.1%} {f["max_dd"]:>6.1%} '
              f'{"OK" if f["ok"] else "no"}')


def main():
    print('=' * 70)
    print('AGENT I  -  ETH RANGE Mean-Reversion (4h)')
    print('=' * 70)
    print(f'Loading ETH 4h, capping at {CUTOFF}...')
    df_raw = load_eth_4h()
    print(f'  Raw bars: {len(df_raw)}, range: {df_raw.index.min()} -> {df_raw.index.max()}')

    df = S.prepare(df_raw)
    assert df.index.max() <= CUTOFF, f'CUTOFF VIOLATION: {df.index.max()}'
    print(f'  Prepared: {len(df)} bars (after warmup drop)')

    # ---- regime distribution ----
    regimes_series = pd.Series([S.detect_regime(df, i) for i in range(len(df))],
                               index=df.index)
    full_mask = regimes_series.index >= pd.Timestamp('2020-01-01', tz='UTC')
    reg_dist = regimes_series[full_mask].value_counts(normalize=True).round(3)
    print('\nRegime distribution (in-sample 2020-2025):')
    print(reg_dist.to_string())

    # ---- WF combined ----
    print('\n' + '-' * 70)
    print('WALK-FORWARD COMBINED (LONG + SHORT in RANGE)')
    print('-' * 70)
    folds, ok, total, wf_trades = run_wf(df, S.PARAMS)
    print(f'  Folds passed (n>=5, PF>=1.2, total>0): {ok}/{total}')
    print_folds('COMB', folds)

    m_wf = calc_metrics(wf_trades)
    pf_str = 'inf' if not np.isfinite(m_wf['pf']) else f'{m_wf["pf"]:.2f}'
    print(f'\n  Combined WF metrics: N={m_wf["n"]} WR={m_wf["wr"]:.1%} '
          f'PF={pf_str} Total={m_wf["total"]:+.1%} DD={m_wf["max_dd"]:.1%}')

    # ---- Full continuous run ----
    print('\n' + '-' * 70)
    print('CONTINUOUS 2020-01-01 -> 2025-12-31  (no fold split)')
    print('-' * 70)
    cont_trades = run_full_period(df, S.PARAMS)
    m_cont = calc_metrics(cont_trades)
    pf_str = 'inf' if not np.isfinite(m_cont['pf']) else f'{m_cont["pf"]:.2f}'
    print(f'  N={m_cont["n"]} WR={m_cont["wr"]:.1%} PF={pf_str} '
          f'Total={m_cont["total"]:+.1%} DD={m_cont["max_dd"]:.1%}')
    monthly = monthly_returns(cont_trades)
    if len(monthly):
        print(f'  Months observed: {len(monthly)}, mean monthly return: '
              f'{monthly.mean()*100:.2f}%, median: {monthly.median()*100:.2f}%')
        mwr = (monthly > 0).mean()
        print(f'  Months positive: {mwr:.1%}')
    ann = annualized_return_from_trades(cont_trades)
    print(f'  Annualized return (geom monthly): {ann*100:.1f}%')

    # frequency
    years = 6.0
    n_per_year = m_cont['n'] / years
    print(f'  Trade frequency: {m_cont["n"]} trades / 6y  ->  {n_per_year:.1f}/yr')

    # ---- bootstrap ----
    boot = bootstrap_pvalue(cont_trades)
    if boot:
        print(f"  Bootstrap p-value (return<=0 by chance): {boot['p_value']:.3f}")
        print(f"  Median resampled: {boot['pctl_50']:+.1%}, "
              f"5th: {boot['pctl_5']:+.1%}, 95th: {boot['pctl_95']:+.1%}")

    # ---- per-direction audit ----
    print('\n' + '-' * 70)
    print('PER-DIRECTION WF (LONG-only / SHORT-only)')
    print('-' * 70)
    per_dir = {}
    for sd in ('LONG', 'SHORT'):
        folds_d, ok_d, total_d, tr_d = run_wf(df, S.PARAMS, side_filter=sd)
        m_d = calc_metrics(tr_d)
        boot_d = bootstrap_pvalue(tr_d)
        pf_str = 'inf' if not np.isfinite(m_d['pf']) else f'{m_d["pf"]:.2f}'
        print(f'\n  [{sd}] folds {ok_d}/{total_d} | N={m_d["n"]} '
              f'WR={m_d["wr"]:.1%} PF={pf_str} '
              f'Total={m_d["total"]:+.1%} DD={m_d["max_dd"]:.1%}')
        if boot_d:
            print(f'       bootstrap p={boot_d["p_value"]:.3f}')
        print_folds(sd, folds_d)
        per_dir[sd] = {
            'wf': f'{ok_d}/{total_d}', 'n': m_d['n'], 'wr': m_d['wr'],
            'pf': m_d['pf'] if np.isfinite(m_d['pf']) else None,
            'total': m_d['total'], 'max_dd': m_d['max_dd'],
            'bootstrap_p': boot_d['p_value'] if boot_d else None,
        }

    # ---- exit reason breakdown ----
    if cont_trades:
        from collections import Counter
        reasons = Counter(t.get('exit_reason', '?') for t in cont_trades)
        print('\n  Exit reasons (continuous):')
        for r, c in reasons.most_common():
            print(f'    {r:<8}: {c} ({c/m_cont["n"]:.0%})')

    # ---- write summary ----
    pf_med = float(np.nanmedian(
        [f['pf'] for f in folds if not f.get('nodata') and np.isfinite(f['pf'])]
    )) if folds else 0.0

    summary = {
        'agent': 'I',
        'strategy_name': 'ETH RANGE Mean-Reversion (RSI/BB extremes, fade with vol confirmation)',
        'cutoff': str(CUTOFF.date()),
        'pair': 'ETH/USDT',
        'tf': '4h',
        'params': dict(S.PARAMS),
        'in_sample_wf': f'{ok}/{total}',
        'in_sample_pf_median_fold': pf_med,
        'in_sample_wr': float(m_wf['wr']),
        'in_sample_n_trades': m_wf['n'],
        'in_sample_total_return': float(m_wf['total']),
        'in_sample_max_dd': float(m_wf['max_dd']),
        'continuous_pf': float(m_cont['pf']) if np.isfinite(m_cont['pf']) else None,
        'continuous_total': float(m_cont['total']),
        'continuous_max_dd': float(m_cont['max_dd']),
        'continuous_n': m_cont['n'],
        'continuous_wr': float(m_cont['wr']),
        'trades_per_year': n_per_year,
        'monthly_arith_mean': float(monthly.mean()) if len(monthly) else 0.0,
        'months_positive_pct': float((monthly > 0).mean()) if len(monthly) else 0.0,
        'annualized_return_geom': float(ann),
        'bootstrap_pvalue': boot['p_value'] if boot else None,
        'bootstrap_median': boot['pctl_50'] if boot else None,
        'per_direction': per_dir,
        'regime_distribution': reg_dist.to_dict(),
        'pct_time_in_range': float(reg_dist.get('RANGE', 0.0)),
    }
    out = HERE / 'results.json'
    out.write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    print(f'\nSaved summary -> {out}')

    # ---- self-audit ----
    print('\n' + '=' * 70)
    print('SELF-AUDIT (sanity flags)')
    print('=' * 70)
    flags = []
    if np.isfinite(m_wf['pf']) and m_wf['pf'] > 4.0:
        flags.append(f'PF combined {m_wf["pf"]:.2f} > 4 -> investigate')
    if m_wf['wr'] > 0.70:
        flags.append(f'WR {m_wf["wr"]:.1%} > 70% -> investigate')
    if m_wf['max_dd'] < 0.03 and m_wf['n'] >= 20:
        flags.append(f'DD {m_wf["max_dd"]:.1%} < 3% -> investigate (too clean)')
    if m_cont['n'] < 30:
        flags.append(f'Total trades {m_cont["n"]} < 30 -> setup too rare to operate')
    elif m_cont['n'] < 60:
        flags.append(f'Total trades {m_cont["n"]} < 60 -> below brief\'s 10/year target')
    if flags:
        for f in flags:
            print(f'  ! {f}')
    else:
        print('  OK - no automatic sanity flag tripped')


if __name__ == '__main__':
    main()
