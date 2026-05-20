"""
evaluate_adaptive_eth.py -- Test adaptive improvements for ETH V15 committee
===========================================================================
Same concept as BTC FG combo: quality filters + EMA crossover for SHORT.

Configs tested:
  A) Baseline (current ETH committee)
  B) + Quality filter on ETH breakout (skip quality < 50)
  C) + EMA20 < EMA50 filter on ETH SHORT
  D) + Quality on ETH breakout + RR tiers
  E) + All: quality breakout + RR tiers + SHORT EMA filter
  F) + Quality filter on BTC follower (skip quality < 50)
  G) + All + follower quality filter
"""
import sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pandas_ta as pta
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from v15_framework import (
    load_pair_4h, load_btc_4h,
    compute_features_4h, compute_macro_daily, merge_daily_to_4h,
    sim_trade_fixed, metrics, WF_FOLDS, COMMISSION,
)
from evaluate_eth_bear import (
    detect_regime, add_extra_features, add_eth_specific_features,
    sim_short, equity_stats,
)

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("ETH Adaptive Evaluation")
print("=" * 70)

print("\nLoading data...")
df_eth_raw = load_pair_4h('ETH')
df_btc_raw = load_btc_4h()

df_eth = compute_features_4h(df_eth_raw.copy())
df_btc = compute_features_4h(df_btc_raw.copy())

try:
    from v15_framework import load_pair_1d, load_btc_1d
    eth_1d = load_pair_1d('ETH')
    btc_1d = load_btc_1d()
except Exception:
    eth_1d = df_eth_raw.resample('1D').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'}).dropna()
    btc_1d = df_btc_raw.resample('1D').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'}).dropna()

eth_macro = compute_macro_daily(eth_1d)
btc_macro = compute_macro_daily(btc_1d)
df_eth = merge_daily_to_4h(df_eth, eth_macro)
df_btc = merge_daily_to_4h(df_btc, btc_macro)
df_eth = add_extra_features(df_eth)
df_eth = add_eth_specific_features(df_eth, df_btc)

regimes_eth = df_eth.apply(lambda r: detect_regime(r), axis=1)
regimes_btc = df_btc.apply(lambda r: detect_regime(r), axis=1)

eth_ret = df_eth['close'].pct_change()
btc_close_a = df_btc['close'].reindex(df_eth.index, method='ffill')
btc_ret = btc_close_a.pct_change()
corr_20 = eth_ret.rolling(20).corr(btc_ret)

print(f"  ETH: {len(df_eth)} bars ({df_eth.index[0].date()} - {df_eth.index[-1].date()})")
print(f"  BTC: {len(df_btc)} bars")


# ============================================================
# QUALITY SCORE (adapted for ETH)
# ============================================================
def compute_quality_eth(df, i, regime):
    """Quality score 0-100 for ETH breakouts."""
    row = df.iloc[i]
    score = 0

    # 1. BB compression (25 pts) - relative to 50-bar median
    start = max(0, i - 50)
    bb_series = df['bb_width'].iloc[start:i]
    if len(bb_series) >= 10:
        median_bb = bb_series.median()
        current_bb = float(df['bb_width'].iloc[max(0, i - 1)])
        if median_bb > 0:
            ratio = current_bb / median_bb
            if ratio < 0.5:
                score += 25
            elif ratio < 0.7:
                score += 18
            elif ratio < 1.0:
                score += 10

    # 2. Vol spike strength (20 pts) - ETH thresholds (lower than BTC)
    vol_ratio = float(row.get('vol_ratio', 1.0))
    if vol_ratio >= 2.5:
        score += 20
    elif vol_ratio >= 2.0:
        score += 16
    elif vol_ratio >= 1.5:
        score += 12
    elif vol_ratio >= 1.2:
        score += 6

    # 3. DI+ crossover (15 pts)
    di_diff = float(row.get('di_diff', 0))
    if i >= 1:
        prev_di_diff = float(df.iloc[i - 1].get('di_diff', 0))
        if di_diff > 0 and prev_di_diff <= 0:
            score += 15
        elif di_diff > 5:
            score += 8
        elif di_diff > 0:
            score += 4

    # 4. Regime alignment (20 pts)
    if regime == 'BULL':
        score += 20
    elif regime == 'RANGE':
        score += 10

    # 5. EMA stack (10 pts)
    ema20 = float(row.get('ema20', 0))
    ema50 = float(row.get('ema50', 0))
    close = float(row['close'])
    if ema20 > 0 and ema50 > 0:
        if close > ema20 > ema50:
            score += 10
        elif close > ema50:
            score += 5

    # 6. RSI zone (10 pts) - slightly wider for ETH
    rsi = float(row.get('rsi14', 50))
    if 42 <= rsi <= 68:
        score += 10
    elif 35 <= rsi <= 75:
        score += 5

    return min(score, 100)


def compute_quality_btc(df_btc, btc_i, regime_btc):
    """Quality score for BTC breakout (for follower filter)."""
    row = df_btc.iloc[btc_i]
    score = 0

    start = max(0, btc_i - 50)
    bb_series = df_btc['bb_width'].iloc[start:btc_i]
    if len(bb_series) >= 10:
        median_bb = bb_series.median()
        current_bb = float(df_btc['bb_width'].iloc[max(0, btc_i - 1)])
        if median_bb > 0:
            ratio = current_bb / median_bb
            if ratio < 0.5:
                score += 25
            elif ratio < 0.7:
                score += 18
            elif ratio < 1.0:
                score += 10

    vol_ratio = float(row.get('vol_ratio', 1.0))
    if vol_ratio >= 3.0:
        score += 20
    elif vol_ratio >= 2.5:
        score += 16
    elif vol_ratio >= 2.0:
        score += 12
    elif vol_ratio >= 1.5:
        score += 6

    di_diff = float(row.get('di_diff', 0))
    if btc_i >= 1:
        prev_di_diff = float(df_btc.iloc[btc_i - 1].get('di_diff', 0))
        if di_diff > 0 and prev_di_diff <= 0:
            score += 15
        elif di_diff > 5:
            score += 8
        elif di_diff > 0:
            score += 4

    if regime_btc == 'BULL':
        score += 20
    elif regime_btc == 'RANGE':
        score += 10

    ema20 = float(row.get('ema20', 0))
    ema50 = float(row.get('ema50', 0))
    close = float(row['close'])
    if ema20 > 0 and ema50 > 0:
        if close > ema20 > ema50:
            score += 10
        elif close > ema50:
            score += 5

    rsi = float(row.get('rsi14', 50))
    if 45 <= rsi <= 65:
        score += 10
    elif 35 <= rsi <= 75:
        score += 5

    return min(score, 100)


# ============================================================
# DETECTORS
# ============================================================
def detect_breakout_eth_std(df, i):
    """Standard ETH breakout (current production logic)."""
    if i < 25:
        return None
    row = df.iloc[i]
    high_N = float(df['high'].iloc[i - 20:i].max())
    if float(row['close']) <= high_N:
        return None
    if float(row.get('vol_ratio', 1)) < 1.3:
        return None
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > 3.5:
        return None
    recent_bb = df['bb_width'].iloc[i - 5:i]
    if (recent_bb < 5.5).sum() < 2:
        return None
    if df['adx14'].iloc[i - 3:i].mean() > 32:
        return None
    entry = float(row['close'])
    sl_raw = float(df['low'].iloc[max(0, i - 5):i].min()) * 0.997
    sl_pct = (entry - sl_raw) / entry
    if sl_pct < 0.005 or sl_pct > 0.06:
        return None
    atr_pct = float(row.get('atr_pct', 2.5))
    tp_pct = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    sl_pct_atr = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    return {'direction': 'LONG', 'setup': 'BREAKOUT_ETH',
            'entry': entry, 'tp_pct': tp_pct, 'sl_pct': sl_pct_atr}


def detect_breakout_b_btc(df_btc, btc_i):
    """BTC breakout B (for follower detection)."""
    if btc_i < 25:
        return None
    row = df_btc.iloc[btc_i]
    high_N = float(df_btc['high'].iloc[btc_i - 20:btc_i].max())
    if float(row['close']) <= high_N:
        return None
    if float(row.get('vol_ratio', 1)) < 1.8:
        return None
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > 2.5:
        return None
    recent_bb = df_btc['bb_width'].iloc[btc_i - 5:btc_i]
    if (recent_bb < 4.0).sum() < 3:
        return None
    if df_btc['adx14'].iloc[btc_i - 3:btc_i].mean() > 28:
        return None
    entry = float(row['close'])
    sl_raw = float(df_btc['low'].iloc[max(0, btc_i - 5):btc_i].min()) * 0.997
    sl_pct = (entry - sl_raw) / entry
    if sl_pct < 0.005 or sl_pct > 0.04:
        return None
    return {'direction': 'LONG', 'setup': 'BREAKOUT_B',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': sl_pct * 1.5}


def detect_pullback_btc(df_btc, btc_i):
    """BTC pullback EMA20 (for follower detection)."""
    if btc_i < 25:
        return None
    row = df_btc.iloc[btc_i]
    prev = df_btc.iloc[btc_i - 1]
    c = float(row['close'])
    o = float(row['open'])
    ema20 = float(row.get('ema20', 0))
    ema50 = float(row.get('ema50', 0))
    if ema20 <= 0 or ema50 <= 0:
        return None
    if c < ema50:
        return None
    dist = (c - ema20) / ema20
    if dist < -0.005 or dist > 0.015:
        return None
    adx = float(row.get('adx14', 0))
    if adx < 15:
        return None
    rsi = float(row.get('rsi14', 50))
    if rsi < 33 or rsi > 58:
        return None
    if float(row.get('vol_ratio', 1)) > 2.0:
        return None
    if c <= o or float(prev['close']) >= float(prev['open']):
        return None
    entry = c
    atr_pct = float(row.get('atr_pct', 2.0))
    sl_pct = max(min(atr_pct / 100 * 1.0, 0.03), 0.01)
    return {'direction': 'LONG', 'setup': 'PULLBACK_EMA20',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': sl_pct * 1.67}


def detect_multi_conf(df, i):
    """ETH SHORT multi-confluence (baseline)."""
    if i < 25:
        return None
    row = df.iloc[i]
    c_val, o_val = float(row['close']), float(row['open'])
    if c_val >= o_val:
        return None
    if float(row.get('rsi14', 50)) < 60:
        return None
    if float(row.get('bb_pct', 0.5)) < 0.75:
        return None
    if float(row.get('vol_ratio', 1)) < 1.0:
        return None
    entry = c_val
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    sl = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    return {'direction': 'SHORT', 'setup': 'MULTI_CONF',
            'entry': entry, 'tp_pct': tp, 'sl_pct': sl}


def detect_bb_upper(df, i):
    """ETH SHORT BB upper (baseline)."""
    if i < 25:
        return None
    row = df.iloc[i]
    c_val, o_val = float(row['close']), float(row['open'])
    if c_val >= o_val:
        return None
    if float(row.get('bb_pct', 0.5)) < 0.90:
        return None
    entry = c_val
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    sl = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    return {'direction': 'SHORT', 'setup': 'BB_UPPER',
            'entry': entry, 'tp_pct': tp, 'sl_pct': sl}


# ============================================================
# COMMITTEE RUNNER (configurable)
# ============================================================
def run_committee(df_eth, df_btc, regimes_eth, regimes_btc, corr_20,
                  start_s, end_s, cfg):
    """Run ETH committee with configurable filters.

    cfg keys:
      quality_min_breakout: int or None  (skip ETH breakout < quality)
      quality_rr_breakout: bool          (RR by quality tier for ETH breakout)
      short_ema_filter: bool             (EMA20 < EMA50 filter for SHORT)
      quality_min_follower: int or None  (skip BTC follower < quality)
    """
    trades = []
    qm_b = cfg.get('quality_min_breakout', None)
    qm_f = cfg.get('quality_min_follower', None)
    q_rr = cfg.get('quality_rr_breakout', False)
    s_ema = cfg.get('short_ema_filter', False)

    for i in range(30, len(df_eth)):
        ts = df_eth.index[i]
        if ts < pd.Timestamp(start_s, tz='UTC') or ts > pd.Timestamp(end_s, tz='UTC'):
            continue
        if i + 18 >= len(df_eth):
            continue

        regime = regimes_eth.iloc[i]
        trade = None

        if regime in ('BULL', 'RANGE'):
            # BTC follower
            if ts in df_btc.index:
                btc_i = df_btc.index.get_loc(ts)
                if btc_i >= 30:
                    regime_btc = regimes_btc.iloc[btc_i]
                    btc_signal = None
                    if regime_btc in ('BULL', 'RANGE'):
                        btc_signal = detect_breakout_b_btc(df_btc, btc_i)
                        if btc_signal is None and regime_btc == 'BULL':
                            btc_signal = detect_pullback_btc(df_btc, btc_i)

                    if btc_signal is not None:
                        cv = corr_20.get(ts, 0)
                        if not pd.isna(cv) and cv >= 0.5:
                            # Follower quality filter
                            if qm_f is not None:
                                q = compute_quality_btc(df_btc, btc_i, regime_btc)
                                if q < qm_f:
                                    btc_signal = None

                    if btc_signal is not None:
                        cv = corr_20.get(ts, 0)
                        if not pd.isna(cv) and cv >= 0.5:
                            row = df_eth.iloc[i]
                            entry = float(row['close'])
                            atr_pct = float(row.get('atr_pct', 2.5))
                            sl_pct = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
                            tp_pct = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
                            trade = {'direction': 'LONG',
                                     'setup': f"FOLLOW_{btc_signal['setup']}",
                                     'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}

            # ETH standalone breakout
            if trade is None:
                bo = detect_breakout_eth_std(df_eth, i)
                if bo is not None:
                    # Quality filter
                    if qm_b is not None:
                        quality = compute_quality_eth(df_eth, i, regime)
                        if quality < qm_b:
                            bo = None
                        elif q_rr:
                            # RR by quality tier
                            row = df_eth.iloc[i]
                            atr_pct = float(row.get('atr_pct', 2.5))
                            sl_pct = bo['sl_pct']
                            if quality >= 70:
                                tp_pct = max(min(atr_pct / 100 * 3.0, 0.10), 0.03)
                            else:
                                tp_pct = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
                            bo['tp_pct'] = tp_pct
                    trade = bo

        elif regime == 'BEAR':
            # SHORT: multi-conf or BB upper
            t = detect_multi_conf(df_eth, i)
            if t is None:
                t = detect_bb_upper(df_eth, i)
            if t is not None:
                # EMA filter
                if s_ema:
                    ema20 = float(df_eth.iloc[i].get('ema20', 0))
                    ema50 = float(df_eth.iloc[i].get('ema50', 0))
                    if ema20 > 0 and ema50 > 0 and ema20 >= ema50:
                        t = None
                trade = t

        if trade is None:
            continue

        d = trade.get('direction', 'LONG')
        if d == 'LONG':
            out = sim_trade_fixed(df_eth, i, trade['entry'],
                                  trade['tp_pct'], trade['sl_pct'], max_bars=18)
        else:
            out = sim_short(df_eth, i, trade['entry'],
                           trade['tp_pct'], trade['sl_pct'], max_bars=16)

        trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                      'setup': trade['setup'], 'direction': d})
    return trades


# ============================================================
# CONFIGS TO TEST
# ============================================================
CONFIGS = {
    'A) Baseline': {},
    'B) Quality>=50 breakout': {'quality_min_breakout': 50},
    'C) SHORT EMA filter': {'short_ema_filter': True},
    'D) Quality+RR breakout': {'quality_min_breakout': 50, 'quality_rr_breakout': True},
    'E) Quality+RR+SHORT EMA': {'quality_min_breakout': 50, 'quality_rr_breakout': True,
                                 'short_ema_filter': True},
    'F) Follower quality>=50': {'quality_min_follower': 50},
    'G) All filters': {'quality_min_breakout': 50, 'quality_rr_breakout': True,
                       'short_ema_filter': True, 'quality_min_follower': 50},
}


# ============================================================
# WALK-FORWARD EVALUATION
# ============================================================
print("\n" + "=" * 70)
print("WALK-FORWARD: 12 SEMESTERS (2020-2025)")
print("=" * 70)

all_results = {}
for name, cfg in CONFIGS.items():
    folds = []
    all_trades = []
    for start_s, end_s in WF_FOLDS:
        period = f"{start_s[:7]}/{end_s[5:7]}"
        trades = run_committee(df_eth, df_btc, regimes_eth, regimes_btc,
                              corr_20, start_s, end_s, cfg)
        m = metrics(trades, period)
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        folds.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                      'pf': m['pf'], 'ok': ok})
        all_trades.extend(trades)

    folds_ok = sum(1 for f in folds if f['ok'])
    m_all = metrics(all_trades, 'OOS')
    eq, dd = equity_stats(all_trades)

    longs = [t for t in all_trades if t['direction'] == 'LONG']
    shorts = [t for t in all_trades if t['direction'] == 'SHORT']
    m_long = metrics(longs, 'L')
    m_short = metrics(shorts, 'S')

    all_results[name] = {
        'folds': folds, 'folds_ok': folds_ok,
        'trades': all_trades, 'metrics': m_all,
        'eq': eq, 'dd': dd,
        'm_long': m_long, 'm_short': m_short,
        'n_long': len(longs), 'n_short': len(shorts),
    }

    print(f"\n  {name}:")
    for f in folds:
        wr_s = f"{f['wr']:.0%}" if f['n'] > 0 else "n/a"
        pf_s = f"{f['pf']:.2f}" if f['n'] > 0 else "n/a"
        ok_s = "+" if f['ok'] else "-"
        print(f"    {f['period']}: N={f['n']:>3} WR={wr_s:>4} PF={pf_s:>5} {ok_s}")
    print(f"    WF: {folds_ok}/12 | N={m_all['n']} (L:{len(longs)} S:{len(shorts)}) "
          f"WR={m_all['wr']:.1%} PF={m_all['pf']:.2f} "
          f"${1000*eq:.0f} DD={dd:.1%}")


# ============================================================
# COMPARISON TABLE
# ============================================================
print("\n" + "=" * 70)
print("COMPARISON TABLE")
print("=" * 70)

hdr = (f"  {'Config':<26} | {'WF':>5} | {'N':>4} | {'L':>3} | {'S':>3} | "
       f"{'WR':>5} | {'PF':>5} | {'$1K->':>7} | {'DD':>5} | "
       f"{'S_WR':>5} | {'S_PF':>5}")
print(hdr)
print(f"  {'-'*100}")

baseline_eq = all_results['A) Baseline']['eq']
for name, r in all_results.items():
    m = r['metrics']
    delta = r['eq'] * 1000 - baseline_eq * 1000
    delta_s = f" ({delta:+.0f})" if name != 'A) Baseline' else ""
    s_wr = f"{r['m_short']['wr']:.0%}" if r['n_short'] > 0 else "n/a"
    s_pf = f"{r['m_short']['pf']:.2f}" if r['n_short'] > 0 else "n/a"
    mk = "**" if r['folds_ok'] >= 8 and m['pf'] >= 1.25 else "  "
    print(f"  {mk}{name:<24} | {r['folds_ok']:>2}/12 | {m['n']:>4} | "
          f"{r['n_long']:>3} | {r['n_short']:>3} | "
          f"{m['wr']:.0%} | {m['pf']:.2f} | "
          f"${1000*r['eq']:>6.0f}{delta_s} | {r['dd']:.0%} | "
          f"{s_wr:>5} | {s_pf:>5}")


# ============================================================
# OOS 2026 (Jan-Mar)
# ============================================================
print("\n" + "=" * 70)
print("OOS 2026 (Jan-Mar) -- unseen data")
print("=" * 70)

OOS_START = '2026-01-01'
OOS_END = '2026-03-01'

eth_2026 = df_eth[(df_eth.index >= OOS_START) & (df_eth.index <= OOS_END)]
if len(eth_2026) > 1:
    eth_p0 = float(eth_2026['close'].iloc[0])
    eth_p1 = float(eth_2026['close'].iloc[-1])
    print(f"\n  ETH: ${eth_p0:.0f} -> ${eth_p1:.0f} ({(eth_p1/eth_p0-1)*100:+.1f}%)")
else:
    print("  No 2026 data available")

for name, cfg in CONFIGS.items():
    trades = run_committee(df_eth, df_btc, regimes_eth, regimes_btc,
                          corr_20, OOS_START, OOS_END, cfg)
    m = metrics(trades, '2026')
    eq, dd = equity_stats(trades)
    nl = sum(1 for t in trades if t['direction'] == 'LONG')
    ns = sum(1 for t in trades if t['direction'] == 'SHORT')
    print(f"  {name:<26}: N={m['n']:>2} (L:{nl} S:{ns}) WR={m['wr']:.0%} "
          f"PF={m['pf']:.2f} ${1000*eq:.0f} DD={dd:.1%}")

    if trades:
        for t in sorted(trades, key=lambda x: x['ts']):
            print(f"    {str(t['ts'])[:10]} {t['setup']:<22} {t['direction']:<5} "
                  f"{t['outcome']:<3} {t['pnl_pct']:>+6.2%}")


# ============================================================
# FOLD DETAIL: Failed folds improvement check
# ============================================================
print("\n" + "=" * 70)
print("FOLD-BY-FOLD: A (baseline) vs best variant")
print("=" * 70)

# Find best non-baseline
best_name = max(
    [n for n in all_results if n != 'A) Baseline'],
    key=lambda n: all_results[n]['eq'])

print(f"  Best variant: {best_name}\n")
print(f"  {'Fold':<12} | {'A_n':>3} {'A_WR':>5} {'A_PF':>5} {'A_ok':>3} | "
      f"{'B_n':>3} {'B_WR':>5} {'B_PF':>5} {'B_ok':>3} | Delta")
print(f"  {'-'*72}")

a_folds = all_results['A) Baseline']['folds']
b_folds = all_results[best_name]['folds']
for af, bf in zip(a_folds, b_folds):
    a_wr = f"{af['wr']:.0%}" if af['n'] > 0 else "n/a"
    a_pf = f"{af['pf']:.2f}" if af['n'] > 0 else "n/a"
    b_wr = f"{bf['wr']:.0%}" if bf['n'] > 0 else "n/a"
    b_pf = f"{bf['pf']:.2f}" if bf['n'] > 0 else "n/a"
    improved = "BETTER" if bf['ok'] and not af['ok'] else ""
    worsened = "WORSE" if af['ok'] and not bf['ok'] else ""
    delta = improved or worsened or ("same" if af['ok'] == bf['ok'] else "")
    print(f"  {af['period']:<12} | {af['n']:>3} {a_wr:>5} {a_pf:>5} "
          f"{'+'if af['ok'] else '-':>3} | "
          f"{bf['n']:>3} {b_wr:>5} {b_pf:>5} {'+'if bf['ok'] else '-':>3} | {delta}")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
