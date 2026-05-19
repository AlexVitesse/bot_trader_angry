"""
evaluate_sol_v15.py -- SOL/USDT V15 Strategy Evaluation
=========================================================
Cross-asset validation showed ETH logic fails for SOL (WF 4/12, PF 0.97).
Main problem: BTC pullback follower (189 trades, 37% WR) -- SOL diverges.
What worked: BTC breakout follower (53% WR), BB_UPPER SHORT (59% WR).

SOL specifics: ~1.5x more volatile than ETH (ATR% 3.6% vs 2.4%), BB width 6-10%.

4 parts:
  1. LONG Standalone (Breakout SOL grid) + BTC breakout follower only
  2. SHORT (BEAR regime) -- BB_UPPER + Multi-conf grids
  3. Full committees (LONG + SHORT, WF 12 folds)
  4. OOS 2026 (Jan-Feb)

Anti-overfitting: report top 5 configs, check param clusters.

Usage:
  python evaluate_sol_v15.py
"""

import sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from v15_framework import (
    load_pair_4h, load_btc_4h,
    compute_features_4h, compute_macro_daily, merge_daily_to_4h,
    sim_trade_fixed, metrics, WF_FOLDS, COMMISSION,
)

# ==============================================================
# CONSTANTS
# ==============================================================
REGIME_DEAD_ZONE = 0.02
MIN_BEAR_BARS = 30
OOS_START = '2026-01-01'
OOS_END = '2026-03-01'


# ==============================================================
# HELPERS (same patterns as cross-asset validation)
# ==============================================================
def detect_regime(row):
    ema20 = row.get('ema20_1d', None)
    ema50 = row.get('ema50_1d', None)
    ema200 = row.get('ema200_1d', None)
    close = row.get('close', None)
    if ema20 is None or ema50 is None or pd.isna(ema20) or pd.isna(ema50):
        return 'RANGE'
    dist = (ema20 - ema50) / ema50
    if dist > REGIME_DEAD_ZONE:
        return 'BULL'
    elif dist < -REGIME_DEAD_ZONE:
        if ema200 is not None and close is not None and not pd.isna(ema200):
            if close > ema200:
                return 'RANGE'
        if close is not None and not pd.isna(close):
            if close > ema50:
                return 'RANGE'
        return 'BEAR'
    return 'RANGE'


def add_extra_features(df):
    df = df.copy()
    c, v = df['close'], df['volume']
    df['rsi_slope'] = df['rsi14'].diff(3)
    vol_ma5 = v.rolling(5).mean()
    vol_ma20 = v.rolling(20).mean()
    df['vol_slope'] = (vol_ma5 / vol_ma20.replace(0, np.nan) - 1) * 100
    df['ret_10'] = c.pct_change(10) * 100
    up = (c > c.shift(1)).astype(int)
    df['consec_up'] = up.rolling(8).sum()
    return df


def sim_short(df, entry_bar, entry_price, tp_pct, sl_pct, max_bars=16):
    tp = entry_price * (1 - tp_pct)
    sl = entry_price * (1 + sl_pct)
    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df):
            ep = float(df['close'].iloc[-1])
            pnl = (entry_price - ep) / entry_price - 2 * COMMISSION
            return ('TP' if ep < entry_price else 'SL'), ep, pnl, i
        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        if hi >= sl:
            pnl = -sl_pct - 2 * COMMISSION
            if lo <= tp and float(df['close'].iloc[b]) < (sl + tp) / 2:
                pnl = tp_pct - 2 * COMMISSION
                return 'TP', tp, pnl, i
            return 'SL', sl, pnl, i
        if lo <= tp:
            pnl = tp_pct - 2 * COMMISSION
            return 'TP', tp, pnl, i
    ep = float(df['close'].iloc[min(entry_bar + max_bars, len(df) - 1)])
    pnl = (entry_price - ep) / entry_price - 2 * COMMISSION
    return ('TP' if ep < entry_price else 'SL'), ep, pnl, max_bars


def equity_stats(trades):
    if not trades:
        return 1.0, 0
    cum = 1.0; peak = 1.0; max_dd = 0
    for t in sorted(trades, key=lambda x: x['ts']):
        cum *= (1 + t['pnl_pct'])
        peak = max(peak, cum)
        dd = (peak - cum) / peak
        max_dd = max(max_dd, dd)
    return cum, max_dd


# ==============================================================
# BTC SIGNAL DETECTORS (breakout only -- no pullback for SOL)
# ==============================================================
def detect_breakout_b_btc(df_btc, i):
    if i < 25: return None
    row = df_btc.iloc[i]
    high20 = float(df_btc['high'].iloc[i-20:i].max())
    if row['close'] <= high20: return None
    if row.get('vol_ratio', 1) < 1.8: return None
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > 2.5: return None
    recent_bb = df_btc['bb_width'].iloc[i-5:i]
    if (recent_bb < 4.0).sum() < 3: return None
    if df_btc['adx14'].iloc[i-3:i].mean() > 28: return None
    entry = float(row['close'])
    sl_raw = float(df_btc['low'].iloc[i-5:i].min()) * 0.997
    sl_pct = (entry - sl_raw) / entry
    if sl_pct < 0.005 or sl_pct > 0.04: return None
    return {'setup': 'BRK_BTC'}


# ==============================================================
# SOL BREAKOUT DETECTOR (parametric)
# ==============================================================
def detect_breakout_sol(df, i, vol_min=1.2, bb_max=8.0, adx_max=32, bar_max=7.0):
    """Breakout B adapted for SOL: wider thresholds for higher volatility."""
    if i < 25: return None
    row = df.iloc[i]
    high20 = float(df['high'].iloc[i-20:i].max())
    if row['close'] <= high20: return None
    if row.get('vol_ratio', 1) < vol_min: return None
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > bar_max: return None
    recent_bb = df['bb_width'].iloc[i-5:i]
    if (recent_bb < bb_max).sum() < 2: return None
    if df['adx14'].iloc[i-3:i].mean() > adx_max: return None
    entry = float(row['close'])
    sl_raw = float(df['low'].iloc[i-5:i].min()) * 0.995
    sl_pct_raw = (entry - sl_raw) / entry
    if sl_pct_raw < 0.005 or sl_pct_raw > 0.08: return None
    return entry


def make_sol_breakout_trade(df, i, entry, tp_mult, sl_mult, tp_cap, sl_cap):
    """Build trade dict with ATR-based TP/SL using given multipliers and caps."""
    row = df.iloc[i]
    atr_pct = float(row.get('atr_pct', 3.5))
    sl_pct = max(min(atr_pct / 100 * sl_mult, sl_cap), 0.015)
    tp_pct = max(min(atr_pct / 100 * tp_mult, tp_cap), 0.025)
    return {'direction': 'LONG', 'setup': 'BRK_SOL',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}


# ==============================================================
# SHORT DETECTORS (parametric for grid)
# ==============================================================
def detect_bb_upper_sol(df, i, bb_min=0.90):
    if i < 25: return None
    row = df.iloc[i]
    c_val, o_val = float(row['close']), float(row['open'])
    if c_val >= o_val: return None  # bearish candle required
    if float(row.get('bb_pct', 0.5)) < bb_min: return None
    entry = c_val
    atr_pct = float(row.get('atr_pct', 3.5))
    tp = max(min(atr_pct / 100 * 2.5, 0.10), 0.025)
    sl = max(min(atr_pct / 100 * 1.5, 0.06), 0.015)
    return {'direction': 'SHORT', 'setup': 'BB_UPPER',
            'entry': entry, 'tp_pct': tp, 'sl_pct': sl}


def detect_multi_conf_sol(df, i, rsi_min=62, bb_min=0.78, vol_min=1.0):
    if i < 25: return None
    row = df.iloc[i]
    c_val, o_val = float(row['close']), float(row['open'])
    if c_val >= o_val: return None  # bearish candle required
    if float(row.get('rsi14', 50)) < rsi_min: return None
    if float(row.get('bb_pct', 0.5)) < bb_min: return None
    if float(row.get('vol_ratio', 1)) < vol_min: return None
    entry = c_val
    atr_pct = float(row.get('atr_pct', 3.5))
    tp = max(min(atr_pct / 100 * 2.5, 0.10), 0.025)
    sl = max(min(atr_pct / 100 * 1.5, 0.06), 0.015)
    return {'direction': 'SHORT', 'setup': 'MULTI_CONF',
            'entry': entry, 'tp_pct': tp, 'sl_pct': sl}


# ==============================================================
# DATA LOADING
# ==============================================================
def load_data():
    print("Loading SOL 4h data...")
    df_sol_raw = load_pair_4h('SOL')
    df_sol = compute_features_4h(df_sol_raw.copy())
    df_sol = add_extra_features(df_sol)

    # Daily macro: resample 4h -> 1d (no daily file for SOL)
    try:
        from v15_framework import load_pair_1d
        sol_1d = load_pair_1d('SOL')
        print("  Loaded SOL daily data")
    except (FileNotFoundError, Exception):
        print("  No daily data for SOL, resampling 4h -> 1d")
        sol_1d = df_sol_raw.resample('1D').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()

    sol_macro = compute_macro_daily(sol_1d)
    df_sol = merge_daily_to_4h(df_sol, sol_macro)

    regimes_sol = df_sol.apply(lambda r: detect_regime(r), axis=1)

    # BTC
    print("Loading BTC 4h data...")
    df_btc_raw = load_btc_4h()
    df_btc = compute_features_4h(df_btc_raw.copy())
    df_btc = add_extra_features(df_btc)
    try:
        from v15_framework import load_btc_1d
        btc_1d = load_btc_1d()
    except Exception:
        btc_1d = df_btc_raw.resample('1D').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()
    btc_macro = compute_macro_daily(btc_1d)
    df_btc = merge_daily_to_4h(df_btc, btc_macro)
    regimes_btc = df_btc.apply(lambda r: detect_regime(r), axis=1)

    # Correlation SOL-BTC
    sol_ret = df_sol['close'].pct_change()
    btc_close_a = df_btc['close'].reindex(df_sol.index, method='ffill')
    btc_ret = btc_close_a.pct_change()
    corr_20 = sol_ret.rolling(20).corr(btc_ret)

    print(f"  SOL: {len(df_sol)} bars ({df_sol.index[0].date()} to {df_sol.index[-1].date()})")
    reg_counts = regimes_sol.value_counts().to_dict()
    print(f"  SOL regimes: {reg_counts}")
    print(f"  SOL ATR%: mean={df_sol['atr_pct'].mean():.2f}%, "
          f"median={df_sol['atr_pct'].median():.2f}%")
    print(f"  SOL BB width: mean={df_sol['bb_width'].mean():.2f}%, "
          f"median={df_sol['bb_width'].median():.2f}%")
    corr_mean = corr_20.dropna().mean()
    corr_above_05 = (corr_20.dropna() >= 0.5).mean()
    print(f"  SOL-BTC corr: mean={corr_mean:.3f}, >=0.5 = {corr_above_05:.1%}")

    # BEAR bars per fold
    bear_folds = []
    for start_s, end_s in WF_FOLDS:
        mask = (df_sol.index >= pd.Timestamp(start_s, tz='UTC')) & \
               (df_sol.index <= pd.Timestamp(end_s, tz='UTC'))
        fold_reg = regimes_sol[mask]
        n_bear = (fold_reg == 'BEAR').sum()
        bear_folds.append(n_bear)
    valid_bear = sum(1 for b in bear_folds if b >= MIN_BEAR_BARS)
    print(f"  BEAR bars per fold: {bear_folds}")
    print(f"  Valid BEAR folds (>={MIN_BEAR_BARS} bars): {valid_bear}/12")

    return (df_sol, df_sol_raw, regimes_sol,
            df_btc, regimes_btc, corr_20, bear_folds)


# ==============================================================
# PART 1: LONG Standalone (Breakout SOL grid)
# ==============================================================
def part1_long_standalone(df_sol, df_btc, regimes_sol, regimes_btc, corr_20):
    print("\n" + "=" * 70)
    print("PART 1: LONG Standalone -- Breakout SOL Grid")
    print("=" * 70)

    # --- Grid breakout SOL ---
    vol_mins = [1.0, 1.2, 1.5]
    bb_maxs = [7.0, 8.0, 10.0]
    bar_maxs = [5.0, 7.0]

    # TP/SL profiles
    profiles = {
        'A': (2.5, 1.5, 0.10, 0.06),  # tp_mult, sl_mult, tp_cap, sl_cap
        'B': (3.0, 2.0, 0.12, 0.07),
    }

    grid_results = []

    for vol_min in vol_mins:
        for bb_max in bb_maxs:
            for bar_max in bar_maxs:
                for prof_name, (tp_m, sl_m, tp_c, sl_c) in profiles.items():
                    label = f"V{vol_min}_BB{bb_max}_BAR{bar_max}_{prof_name}"
                    all_trades = []
                    fold_ok_count = 0

                    for start_s, end_s in WF_FOLDS:
                        fold_trades = []
                        for i in range(30, len(df_sol)):
                            ts = df_sol.index[i]
                            if ts < pd.Timestamp(start_s, tz='UTC'):
                                continue
                            if ts > pd.Timestamp(end_s, tz='UTC'):
                                continue
                            if i + 18 >= len(df_sol):
                                continue
                            regime = regimes_sol.iloc[i]
                            if regime not in ('BULL', 'RANGE'):
                                continue

                            entry = detect_breakout_sol(df_sol, i, vol_min, bb_max, 32, bar_max)
                            if entry is None:
                                continue
                            trade = make_sol_breakout_trade(df_sol, i, entry, tp_m, sl_m, tp_c, sl_c)
                            out = sim_trade_fixed(df_sol, i, trade['entry'],
                                                  trade['tp_pct'], trade['sl_pct'], max_bars=18)
                            fold_trades.append({
                                'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                                'setup': 'BRK_SOL', 'direction': 'LONG', 'entry': trade['entry']
                            })
                        m = metrics(fold_trades, label)
                        if m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0:
                            fold_ok_count += 1
                        all_trades.extend(fold_trades)

                    m_all = metrics(all_trades, label)
                    eq, dd = equity_stats(all_trades)
                    grid_results.append({
                        'label': label, 'vol': vol_min, 'bb': bb_max, 'bar': bar_max,
                        'prof': prof_name, 'folds_ok': fold_ok_count,
                        'n': m_all['n'], 'wr': m_all['wr'], 'pf': m_all['pf'],
                        'eq': eq, 'dd': dd,
                    })

    # Sort by PF descending, show top 10
    grid_results.sort(key=lambda x: x['pf'], reverse=True)
    print(f"\n  Breakout SOL standalone -- Top 10 by PF (of {len(grid_results)} combos):")
    print(f"  {'Config':<28} | {'WF':>5} | {'N':>4} | {'WR':>6} | {'PF':>6} | {'$1K->':>7} | {'DD':>6}")
    print(f"  {'-'*80}")
    for r in grid_results[:10]:
        print(f"  {r['label']:<28} | {r['folds_ok']:>2}/12 | {r['n']:>4} | "
              f"{r['wr']:.1%} | {r['pf']:.2f} | ${1000*r['eq']:>6.0f} | {r['dd']:.1%}")

    # --- BTC Breakout Follower Only (no pullback) ---
    print(f"\n  BTC Breakout Follower Only (no pullback):")
    follower_results = []
    for corr_th in [0.4, 0.5]:
        label = f"FOLLOW_BRK_corr{corr_th}"
        all_trades = []
        fold_ok_count = 0

        for start_s, end_s in WF_FOLDS:
            fold_trades = []
            for i in range(30, len(df_sol)):
                ts = df_sol.index[i]
                if ts < pd.Timestamp(start_s, tz='UTC'):
                    continue
                if ts > pd.Timestamp(end_s, tz='UTC'):
                    continue
                if i + 18 >= len(df_sol):
                    continue
                regime = regimes_sol.iloc[i]
                if regime not in ('BULL', 'RANGE'):
                    continue
                if ts not in df_btc.index:
                    continue
                btc_i = df_btc.index.get_loc(ts)
                if btc_i < 30:
                    continue
                regime_btc = regimes_btc.iloc[btc_i]
                if regime_btc not in ('BULL', 'RANGE'):
                    continue
                btc_signal = detect_breakout_b_btc(df_btc, btc_i)
                if btc_signal is None:
                    continue
                cv = corr_20.get(ts, 0)
                if pd.isna(cv) or cv < corr_th:
                    continue
                row = df_sol.iloc[i]
                entry = float(row['close'])
                atr_pct = float(row.get('atr_pct', 3.5))
                sl_pct = max(min(atr_pct / 100 * 1.5, 0.06), 0.015)
                tp_pct = max(min(atr_pct / 100 * 2.5, 0.10), 0.025)
                out = sim_trade_fixed(df_sol, i, entry, tp_pct, sl_pct, max_bars=18)
                fold_trades.append({
                    'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                    'setup': 'FOLLOW_BRK_BTC', 'direction': 'LONG', 'entry': entry
                })
            m = metrics(fold_trades, label)
            if m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0:
                fold_ok_count += 1
            all_trades.extend(fold_trades)

        m_all = metrics(all_trades, label)
        eq, dd = equity_stats(all_trades)
        follower_results.append({
            'label': label, 'corr': corr_th,
            'folds_ok': fold_ok_count, 'n': m_all['n'], 'wr': m_all['wr'],
            'pf': m_all['pf'], 'eq': eq, 'dd': dd,
        })
        print(f"    {label}: WF {fold_ok_count}/12 N={m_all['n']} "
              f"WR={m_all['wr']:.1%} PF={m_all['pf']:.2f} "
              f"$1K->${1000*eq:.0f} DD={dd:.1%}")

    # --- Combined: best standalone + BTC breakout follower ---
    # Pick best standalone by PF (with folds_ok >= 5)
    viable = [r for r in grid_results if r['folds_ok'] >= 5]
    if not viable:
        viable = grid_results[:3]
    best_standalone = viable[0] if viable else grid_results[0]
    best_follower = max(follower_results, key=lambda x: x['pf'])

    print(f"\n  Combined: best standalone ({best_standalone['label']}) + BTC breakout follower (corr>={best_follower['corr']}):")

    bs = best_standalone
    bs_vol, bs_bb, bs_bar, bs_prof = bs['vol'], bs['bb'], bs['bar'], bs['prof']
    tp_m, sl_m, tp_c, sl_c = profiles[bs_prof]
    corr_th = best_follower['corr']

    combo_results = []
    combo_trades_all = []
    for start_s, end_s in WF_FOLDS:
        fold_trades = []
        for i in range(30, len(df_sol)):
            ts = df_sol.index[i]
            if ts < pd.Timestamp(start_s, tz='UTC'):
                continue
            if ts > pd.Timestamp(end_s, tz='UTC'):
                continue
            if i + 18 >= len(df_sol):
                continue
            regime = regimes_sol.iloc[i]
            if regime not in ('BULL', 'RANGE'):
                continue

            trade = None
            # Try BTC breakout follower first
            if ts in df_btc.index:
                btc_i = df_btc.index.get_loc(ts)
                if btc_i >= 30:
                    regime_btc = regimes_btc.iloc[btc_i]
                    if regime_btc in ('BULL', 'RANGE'):
                        btc_signal = detect_breakout_b_btc(df_btc, btc_i)
                        if btc_signal is not None:
                            cv = corr_20.get(ts, 0)
                            if not pd.isna(cv) and cv >= corr_th:
                                row = df_sol.iloc[i]
                                entry = float(row['close'])
                                atr_pct = float(row.get('atr_pct', 3.5))
                                sl_pct = max(min(atr_pct / 100 * 1.5, 0.06), 0.015)
                                tp_pct = max(min(atr_pct / 100 * 2.5, 0.10), 0.025)
                                trade = {'direction': 'LONG', 'setup': 'FOLLOW_BRK_BTC',
                                         'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}
            # Then standalone breakout
            if trade is None:
                entry = detect_breakout_sol(df_sol, i, bs_vol, bs_bb, 32, bs_bar)
                if entry is not None:
                    trade = make_sol_breakout_trade(df_sol, i, entry, tp_m, sl_m, tp_c, sl_c)

            if trade is None:
                continue
            out = sim_trade_fixed(df_sol, i, trade['entry'],
                                  trade['tp_pct'], trade['sl_pct'], max_bars=18)
            fold_trades.append({
                'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                'setup': trade['setup'], 'direction': 'LONG', 'entry': trade['entry']
            })
        m = metrics(fold_trades, '')
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        period = f"{start_s[:7]}/{end_s[5:7]}"
        combo_results.append({'period': period, 'n': m['n'], 'wr': m['wr'], 'pf': m['pf'], 'ok': ok})
        combo_trades_all.extend(fold_trades)

    m_combo = metrics(combo_trades_all, 'COMBINED')
    eq_c, dd_c = equity_stats(combo_trades_all)
    folds_ok_c = sum(1 for r in combo_results if r['ok'])

    for r in combo_results:
        wr_s = f"{r['wr']:.1%}" if r['n'] > 0 else "n/a"
        pf_s = f"{r['pf']:.2f}" if r['n'] > 0 else "n/a"
        ok_s = "+" if r['ok'] else "-"
        print(f"    {r['period']}: N={r['n']:>3} WR={wr_s:>6} PF={pf_s:>6} {ok_s}")
    print(f"    Combined LONG: WF {folds_ok_c}/12 N={m_combo['n']} "
          f"WR={m_combo['wr']:.1%} PF={m_combo['pf']:.2f} "
          f"$1K->${1000*eq_c:.0f} DD={dd_c:.1%}")

    # Setup breakdown
    setups = {}
    for t in combo_trades_all:
        s = t['setup']
        if s not in setups: setups[s] = []
        setups[s].append(t)
    for s_name, ts in sorted(setups.items()):
        wins = sum(1 for t in ts if t['pnl_pct'] > 0)
        wr = wins / len(ts) if ts else 0
        print(f"      {s_name}: N={len(ts)} WR={wr:.1%}")

    return {
        'grid': grid_results,
        'follower': follower_results,
        'best_standalone': best_standalone,
        'best_follower': best_follower,
        'combo_folds_ok': folds_ok_c,
        'combo_trades': combo_trades_all,
        'combo_pf': m_combo['pf'],
    }


# ==============================================================
# PART 2: SHORT (BEAR regime)
# ==============================================================
def part2_short_bear(df_sol, regimes_sol, bear_folds_counts):
    print("\n" + "=" * 70)
    print("PART 2: SHORT in BEAR Regime")
    print("=" * 70)

    valid_folds = sum(1 for b in bear_folds_counts if b >= MIN_BEAR_BARS)
    threshold = max(int(valid_folds * 0.6), 1)
    print(f"  Valid BEAR folds: {valid_folds}/12, need >= {threshold} positive (60%)")

    # --- BB_UPPER grid ---
    print(f"\n  BB_UPPER Grid:")
    bb_upper_results = []
    for bb_min in [0.88, 0.90, 0.92]:
        label = f"BB_UPPER_bb{bb_min}"
        all_trades = []; folds_pos = 0; folds_tested = 0

        for fi, (start_s, end_s) in enumerate(WF_FOLDS):
            if bear_folds_counts[fi] < MIN_BEAR_BARS:
                continue
            folds_tested += 1
            fold_trades = []
            for i in range(30, len(df_sol)):
                ts = df_sol.index[i]
                if ts < pd.Timestamp(start_s, tz='UTC'): continue
                if ts > pd.Timestamp(end_s, tz='UTC'): continue
                if i + 16 >= len(df_sol): continue
                if regimes_sol.iloc[i] != 'BEAR': continue
                trade = detect_bb_upper_sol(df_sol, i, bb_min)
                if trade is None: continue
                out = sim_short(df_sol, i, trade['entry'],
                                trade['tp_pct'], trade['sl_pct'], max_bars=16)
                fold_trades.append({
                    'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                    'setup': 'BB_UPPER', 'direction': 'SHORT', 'entry': trade['entry']
                })
            m = metrics(fold_trades, label)
            if m['n'] >= 2 and m['wr'] > 0.40 and m['pf'] > 0.9:
                folds_pos += 1
            all_trades.extend(fold_trades)

        m_all = metrics(all_trades, label)
        eq, dd = equity_stats(all_trades)
        bb_upper_results.append({
            'label': label, 'bb': bb_min,
            'folds_pos': folds_pos, 'folds_tested': folds_tested,
            'n': m_all['n'], 'wr': m_all['wr'], 'pf': m_all['pf'],
            'eq': eq, 'dd': dd,
        })
        tag = "OK" if folds_pos >= threshold else "FAIL"
        print(f"    {label}: {folds_pos}/{folds_tested} BEAR folds OK, "
              f"N={m_all['n']} WR={m_all['wr']:.1%} PF={m_all['pf']:.2f} "
              f"$1K->${1000*eq:.0f} DD={dd:.1%} -> {tag}")

    # --- Multi-conf grid ---
    print(f"\n  Multi-conf Grid:")
    multi_results = []
    for rsi_min in [62, 65, 68]:
        for bb_min in [0.78, 0.80, 0.85]:
            for vol_min in [1.0, 1.2, 1.3]:
                label = f"MC_rsi{rsi_min}_bb{bb_min}_v{vol_min}"
                all_trades = []; folds_pos = 0; folds_tested = 0

                for fi, (start_s, end_s) in enumerate(WF_FOLDS):
                    if bear_folds_counts[fi] < MIN_BEAR_BARS:
                        continue
                    folds_tested += 1
                    fold_trades = []
                    for i in range(30, len(df_sol)):
                        ts = df_sol.index[i]
                        if ts < pd.Timestamp(start_s, tz='UTC'): continue
                        if ts > pd.Timestamp(end_s, tz='UTC'): continue
                        if i + 16 >= len(df_sol): continue
                        if regimes_sol.iloc[i] != 'BEAR': continue
                        trade = detect_multi_conf_sol(df_sol, i, rsi_min, bb_min, vol_min)
                        if trade is None: continue
                        out = sim_short(df_sol, i, trade['entry'],
                                        trade['tp_pct'], trade['sl_pct'], max_bars=16)
                        fold_trades.append({
                            'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                            'setup': 'MULTI_CONF', 'direction': 'SHORT', 'entry': trade['entry']
                        })
                    m = metrics(fold_trades, label)
                    if m['n'] >= 2 and m['wr'] > 0.40 and m['pf'] > 0.9:
                        folds_pos += 1
                    all_trades.extend(fold_trades)

                m_all = metrics(all_trades, label)
                eq, dd = equity_stats(all_trades)
                multi_results.append({
                    'label': label, 'rsi': rsi_min, 'bb': bb_min, 'vol': vol_min,
                    'folds_pos': folds_pos, 'folds_tested': folds_tested,
                    'n': m_all['n'], 'wr': m_all['wr'], 'pf': m_all['pf'],
                    'eq': eq, 'dd': dd,
                })

    # Sort multi by PF, show top 10
    multi_results.sort(key=lambda x: x['pf'], reverse=True)
    print(f"\n  Multi-conf Top 10 by PF (of {len(multi_results)} combos):")
    print(f"  {'Config':<28} | {'BEAR OK':>8} | {'N':>4} | {'WR':>6} | {'PF':>6} | {'$1K->':>7} | {'DD':>6}")
    print(f"  {'-'*80}")
    for r in multi_results[:10]:
        tag = "OK" if r['folds_pos'] >= threshold else "FAIL"
        print(f"  {r['label']:<28} | {r['folds_pos']:>2}/{r['folds_tested']:>2}   | {r['n']:>4} | "
              f"{r['wr']:.1%} | {r['pf']:.2f} | ${1000*r['eq']:>6.0f} | {r['dd']:.1%} {tag}")

    return {
        'bb_upper': bb_upper_results,
        'multi_conf': multi_results,
        'threshold': threshold,
        'valid_folds': valid_folds,
    }


# ==============================================================
# PART 3: Full Committees (LONG + SHORT, WF 12 folds)
# ==============================================================
def part3_committees(df_sol, df_btc, regimes_sol, regimes_btc, corr_20,
                     part1_results, part2_results, bear_folds_counts):
    print("\n" + "=" * 70)
    print("PART 3: Full Committees (LONG + SHORT)")
    print("=" * 70)

    # Best LONG params
    bs = part1_results['best_standalone']
    bf = part1_results['best_follower']
    bs_vol, bs_bb, bs_bar, bs_prof = bs['vol'], bs['bb'], bs['bar'], bs['prof']
    profiles = {
        'A': (2.5, 1.5, 0.10, 0.06),
        'B': (3.0, 2.0, 0.12, 0.07),
    }
    tp_m, sl_m, tp_c, sl_c = profiles[bs_prof]
    corr_th = bf['corr']

    # Best SHORT params
    threshold = part2_results['threshold']
    best_bb = max(part2_results['bb_upper'], key=lambda x: x['pf'])
    best_mc = part2_results['multi_conf'][0] if part2_results['multi_conf'] else None

    print(f"\n  LONG config: standalone={bs['label']}, follower=corr>={corr_th}")
    print(f"  SHORT BB_UPPER: bb>={best_bb['bb']} ({best_bb['folds_pos']}/{best_bb['folds_tested']} BEAR folds)")
    if best_mc:
        print(f"  SHORT Multi-conf: rsi>={best_mc['rsi']} bb>={best_mc['bb']} vol>={best_mc['vol']} "
              f"({best_mc['folds_pos']}/{best_mc['folds_tested']} BEAR folds)")

    # Define committee configurations
    def make_short_detector(use_bb=False, use_mc=False):
        bb_th = best_bb['bb']
        mc_rsi = best_mc['rsi'] if best_mc else 65
        mc_bb = best_mc['bb'] if best_mc else 0.80
        mc_vol = best_mc['vol'] if best_mc else 1.0

        def detect(df, i):
            if use_mc:
                t = detect_multi_conf_sol(df, i, mc_rsi, mc_bb, mc_vol)
                if t is not None: return t
            if use_bb:
                t = detect_bb_upper_sol(df, i, bb_th)
                if t is not None: return t
            return None
        return detect

    committees = {
        'Solo LONG': None,
        'LONG + BB_UPPER': make_short_detector(use_bb=True),
        'LONG + Multi-conf': make_short_detector(use_mc=True),
        'LONG + BB + Multi': make_short_detector(use_bb=True, use_mc=True),
    }

    committee_results = {}
    for name, short_fn in committees.items():
        all_trades = []
        fold_results = []

        for start_s, end_s in WF_FOLDS:
            fold_trades = []
            for i in range(30, len(df_sol)):
                ts = df_sol.index[i]
                if ts < pd.Timestamp(start_s, tz='UTC'): continue
                if ts > pd.Timestamp(end_s, tz='UTC'): continue
                if i + 18 >= len(df_sol): continue

                regime = regimes_sol.iloc[i]
                trade = None

                if regime in ('BULL', 'RANGE'):
                    # BTC breakout follower
                    if ts in df_btc.index:
                        btc_i = df_btc.index.get_loc(ts)
                        if btc_i >= 30:
                            regime_btc = regimes_btc.iloc[btc_i]
                            if regime_btc in ('BULL', 'RANGE'):
                                btc_signal = detect_breakout_b_btc(df_btc, btc_i)
                                if btc_signal is not None:
                                    cv = corr_20.get(ts, 0)
                                    if not pd.isna(cv) and cv >= corr_th:
                                        row = df_sol.iloc[i]
                                        entry = float(row['close'])
                                        atr_pct = float(row.get('atr_pct', 3.5))
                                        s = max(min(atr_pct / 100 * 1.5, 0.06), 0.015)
                                        t = max(min(atr_pct / 100 * 2.5, 0.10), 0.025)
                                        trade = {'direction': 'LONG', 'setup': 'FOLLOW_BRK_BTC',
                                                 'entry': entry, 'sl_pct': s, 'tp_pct': t}
                    # Standalone breakout
                    if trade is None:
                        entry = detect_breakout_sol(df_sol, i, bs_vol, bs_bb, 32, bs_bar)
                        if entry is not None:
                            trade = make_sol_breakout_trade(df_sol, i, entry, tp_m, sl_m, tp_c, sl_c)

                elif regime == 'BEAR' and short_fn is not None:
                    trade = short_fn(df_sol, i)

                if trade is None:
                    continue

                d = trade.get('direction', 'LONG')
                if d == 'LONG':
                    out = sim_trade_fixed(df_sol, i, trade['entry'],
                                          trade['tp_pct'], trade['sl_pct'], max_bars=18)
                else:
                    out = sim_short(df_sol, i, trade['entry'],
                                    trade['tp_pct'], trade['sl_pct'], max_bars=16)

                fold_trades.append({
                    'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                    'setup': trade['setup'], 'direction': d, 'entry': trade['entry']
                })

            m = metrics(fold_trades, '')
            ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
            period = f"{start_s[:7]}/{end_s[5:7]}"
            fold_results.append({'period': period, 'n': m['n'], 'wr': m['wr'], 'pf': m['pf'], 'ok': ok})
            all_trades.extend(fold_trades)

        m_all = metrics(all_trades, name)
        eq, dd = equity_stats(all_trades)
        folds_ok = sum(1 for r in fold_results if r['ok'])

        # Fold 1 (2020-H1) may have no data -- count non-empty folds
        folds_with_data = sum(1 for r in fold_results if r['n'] > 0)
        folds_total = max(folds_with_data, 11)  # at least 11

        passed = folds_ok >= 7 and m_all['pf'] >= 1.0
        marginal = folds_ok >= 6 and m_all['pf'] >= 1.0
        tag = "APROBADO" if passed else ("MARGINAL" if marginal else "RECHAZADO")

        print(f"\n  --- {name} ---")
        for r in fold_results:
            wr_s = f"{r['wr']:.1%}" if r['n'] > 0 else "n/a"
            pf_s = f"{r['pf']:.2f}" if r['n'] > 0 else "n/a"
            ok_s = "+" if r['ok'] else ("-" if r['n'] > 0 else ".")
            print(f"    {r['period']}: N={r['n']:>3} WR={wr_s:>6} PF={pf_s:>6} {ok_s}")

        print(f"    WF: {folds_ok}/{folds_total} | N={m_all['n']} WR={m_all['wr']:.1%} "
              f"PF={m_all['pf']:.2f} $1K->${1000*eq:.0f} DD={dd:.1%} -> {tag}")

        # Setup breakdown
        setups = {}
        for t in all_trades:
            s = t['setup']
            if s not in setups: setups[s] = []
            setups[s].append(t)
        for s_name, ts in sorted(setups.items()):
            wins = sum(1 for t in ts if t['pnl_pct'] > 0)
            wr = wins / len(ts) if ts else 0
            d = ts[0]['direction'] if ts else '?'
            print(f"      {s_name} ({d}): N={len(ts)} WR={wr:.1%}")

        committee_results[name] = {
            'folds_ok': folds_ok, 'folds_total': folds_total,
            'n': m_all['n'], 'wr': m_all['wr'], 'pf': m_all['pf'],
            'eq': eq, 'dd': dd, 'tag': tag, 'trades': all_trades,
            'fold_results': fold_results,
        }

    # Summary table
    print(f"\n  {'Committee':<22} | {'WF':>8} | {'N':>4} | {'WR':>6} | {'PF':>6} | {'$1K->':>7} | {'DD':>6} | Veredicto")
    print(f"  {'-'*85}")
    for name, r in committee_results.items():
        mark = "**" if r['tag'] == "APROBADO" else "  "
        print(f"  {mark}{name:<20} | {r['folds_ok']:>2}/{r['folds_total']:>2}   | {r['n']:>4} | "
              f"{r['wr']:.1%} | {r['pf']:.2f} | ${1000*r['eq']:>6.0f} | {r['dd']:.1%} | {r['tag']}")

    return committee_results


# ==============================================================
# PART 4: OOS 2026 (Jan-Feb)
# ==============================================================
def part4_oos_2026(df_sol, df_btc, regimes_sol, regimes_btc, corr_20,
                   part1_results, part2_results, committee_results):
    print("\n" + "=" * 70)
    print("PART 4: OOS 2026 (Jan-Feb)")
    print("=" * 70)

    # Market context
    mask_oos = (df_sol.index >= pd.Timestamp(OOS_START, tz='UTC')) & \
               (df_sol.index < pd.Timestamp(OOS_END, tz='UTC'))
    if mask_oos.sum() == 0:
        print("  No SOL data for 2026 OOS period!")
        return

    sol_start = float(df_sol.loc[mask_oos, 'close'].iloc[0])
    sol_end = float(df_sol.loc[mask_oos, 'close'].iloc[-1])
    sol_ret = (sol_end / sol_start - 1) * 100

    mask_btc = (df_btc.index >= pd.Timestamp(OOS_START, tz='UTC')) & \
               (df_btc.index < pd.Timestamp(OOS_END, tz='UTC'))
    if mask_btc.sum() > 0:
        btc_start = float(df_btc.loc[mask_btc, 'close'].iloc[0])
        btc_end = float(df_btc.loc[mask_btc, 'close'].iloc[-1])
        btc_ret = (btc_end / btc_start - 1) * 100
    else:
        btc_start = btc_end = btc_ret = 0

    print(f"  SOL: ${sol_start:.2f} -> ${sol_end:.2f} ({sol_ret:+.1f}%)")
    print(f"  BTC: ${btc_start:.0f} -> ${btc_end:.0f} ({btc_ret:+.1f}%)")

    # Regime distribution in OOS
    oos_regimes = regimes_sol[mask_oos]
    print(f"  SOL Regimes 2026: {oos_regimes.value_counts().to_dict()}")

    # Best LONG params
    bs = part1_results['best_standalone']
    bf = part1_results['best_follower']
    bs_vol, bs_bb, bs_bar, bs_prof = bs['vol'], bs['bb'], bs['bar'], bs['prof']
    profiles = {'A': (2.5, 1.5, 0.10, 0.06), 'B': (3.0, 2.0, 0.12, 0.07)}
    tp_m, sl_m, tp_c, sl_c = profiles[bs_prof]
    corr_th = bf['corr']

    best_bb = max(part2_results['bb_upper'], key=lambda x: x['pf'])
    best_mc = part2_results['multi_conf'][0] if part2_results['multi_conf'] else None

    def make_short_fn(use_bb=False, use_mc=False):
        bb_th = best_bb['bb']
        mc_rsi = best_mc['rsi'] if best_mc else 65
        mc_bb = best_mc['bb'] if best_mc else 0.80
        mc_vol = best_mc['vol'] if best_mc else 1.0
        def detect(df, i):
            if use_mc:
                t = detect_multi_conf_sol(df, i, mc_rsi, mc_bb, mc_vol)
                if t is not None: return t
            if use_bb:
                t = detect_bb_upper_sol(df, i, bb_th)
                if t is not None: return t
            return None
        return detect

    committee_configs = {
        'Solo LONG': None,
        'LONG + BB_UPPER': make_short_fn(use_bb=True),
        'LONG + Multi-conf': make_short_fn(use_mc=True),
        'LONG + BB + Multi': make_short_fn(use_bb=True, use_mc=True),
    }

    for cname, short_fn in committee_configs.items():
        trades = []
        for i in range(30, len(df_sol)):
            ts = df_sol.index[i]
            if ts < pd.Timestamp(OOS_START, tz='UTC'): continue
            if ts >= pd.Timestamp(OOS_END, tz='UTC'): continue
            if i + 18 >= len(df_sol): continue

            regime = regimes_sol.iloc[i]
            trade = None

            if regime in ('BULL', 'RANGE'):
                if ts in df_btc.index:
                    btc_i = df_btc.index.get_loc(ts)
                    if btc_i >= 30:
                        regime_btc = regimes_btc.iloc[btc_i]
                        if regime_btc in ('BULL', 'RANGE'):
                            btc_signal = detect_breakout_b_btc(df_btc, btc_i)
                            if btc_signal is not None:
                                cv = corr_20.get(ts, 0)
                                if not pd.isna(cv) and cv >= corr_th:
                                    row = df_sol.iloc[i]
                                    entry = float(row['close'])
                                    atr_pct = float(row.get('atr_pct', 3.5))
                                    s = max(min(atr_pct / 100 * 1.5, 0.06), 0.015)
                                    t = max(min(atr_pct / 100 * 2.5, 0.10), 0.025)
                                    trade = {'direction': 'LONG', 'setup': 'FOLLOW_BRK_BTC',
                                             'entry': entry, 'sl_pct': s, 'tp_pct': t}
                if trade is None:
                    entry = detect_breakout_sol(df_sol, i, bs_vol, bs_bb, 32, bs_bar)
                    if entry is not None:
                        trade = make_sol_breakout_trade(df_sol, i, entry, tp_m, sl_m, tp_c, sl_c)

            elif regime == 'BEAR' and short_fn is not None:
                trade = short_fn(df_sol, i)

            if trade is None:
                continue

            d = trade.get('direction', 'LONG')
            if d == 'LONG':
                out = sim_trade_fixed(df_sol, i, trade['entry'],
                                      trade['tp_pct'], trade['sl_pct'], max_bars=18)
            else:
                out = sim_short(df_sol, i, trade['entry'],
                                trade['tp_pct'], trade['sl_pct'], max_bars=16)
            trades.append({
                'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                'setup': trade['setup'], 'direction': d, 'entry': trade['entry']
            })

        print(f"\n  --- {cname} (OOS 2026) ---")
        if not trades:
            print("    No trades in OOS period")
            continue

        # Trade-by-trade detail
        for t in sorted(trades, key=lambda x: x['ts']):
            print(f"    {str(t['ts'])[:19]:<22} {t['setup']:<18} {t['direction']:<6} "
                  f"${t['entry']:>8.2f} {t['outcome']:<4} {t['pnl_pct']:>+7.2%}")

        m = metrics(trades, cname)
        eq, dd = equity_stats(trades)
        n_long = sum(1 for t in trades if t['direction'] == 'LONG')
        n_short = sum(1 for t in trades if t['direction'] == 'SHORT')
        pnl_long = sum(t['pnl_pct'] for t in trades if t['direction'] == 'LONG')
        pnl_short = sum(t['pnl_pct'] for t in trades if t['direction'] == 'SHORT')

        print(f"    N={m['n']} (L={n_long} S={n_short}) WR={m['wr']:.1%} PF={m['pf']:.2f} "
              f"$1K->${1000*eq:.0f} DD={dd:.1%}")
        print(f"    LONG PnL: {pnl_long:+.2%} | SHORT PnL: {pnl_short:+.2%} | "
              f"Total: {pnl_long + pnl_short:+.2%}")
        print(f"    vs SOL buy-and-hold: {sol_ret:+.1f}%")


# ==============================================================
# MAIN
# ==============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("SOL/USDT V15 Strategy Evaluation")
    print("=" * 70)

    (df_sol, df_sol_raw, regimes_sol,
     df_btc, regimes_btc, corr_20, bear_folds_counts) = load_data()

    # Part 1: LONG
    p1 = part1_long_standalone(df_sol, df_btc, regimes_sol, regimes_btc, corr_20)

    # Part 2: SHORT
    p2 = part2_short_bear(df_sol, regimes_sol, bear_folds_counts)

    # Part 3: Committees
    p3 = part3_committees(df_sol, df_btc, regimes_sol, regimes_btc, corr_20,
                          p1, p2, bear_folds_counts)

    # Part 4: OOS 2026
    part4_oos_2026(df_sol, df_btc, regimes_sol, regimes_btc, corr_20, p1, p2, p3)

    # ==============================================================
    # FINAL VERDICT
    # ==============================================================
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    any_approved = any(r['tag'] == 'APROBADO' for r in p3.values())
    any_marginal = any(r['tag'] == 'MARGINAL' for r in p3.values())

    if any_approved:
        best = max(p3.items(), key=lambda x: x[1]['pf'] if x[1]['tag'] == 'APROBADO' else 0)
        print(f"  APROBADO: {best[0]} -> candidato para bot")
        print(f"    WF {best[1]['folds_ok']}/{best[1]['folds_total']} PF={best[1]['pf']:.2f} "
              f"WR={best[1]['wr']:.1%} DD={best[1]['dd']:.1%}")
    elif any_marginal:
        best = max(p3.items(), key=lambda x: x[1]['pf'] if x[1]['tag'] == 'MARGINAL' else 0)
        print(f"  MARGINAL: {best[0]} -> necesita mas validacion")
        print(f"    WF {best[1]['folds_ok']}/{best[1]['folds_total']} PF={best[1]['pf']:.2f} "
              f"WR={best[1]['wr']:.1%} DD={best[1]['dd']:.1%}")
    else:
        print("  RECHAZADO: Ningun comite pasa WF >= 7 + PF >= 1.0")
        print("  SOL queda fuera del V15 por ahora")

    # Anti-overfitting check: are top configs clustered?
    print(f"\n  Anti-overfitting: param cluster analysis")
    top5 = p1['grid'][:5]
    if len(top5) >= 3:
        vols = [r['vol'] for r in top5]
        bbs = [r['bb'] for r in top5]
        bars = [r['bar'] for r in top5]
        vol_range = max(vols) - min(vols)
        bb_range = max(bbs) - min(bbs)
        bar_range = max(bars) - min(bars)
        clustered = vol_range <= 0.5 and bb_range <= 3.0 and bar_range <= 2.0
        print(f"    Top 5 LONG: vol=[{min(vols)}-{max(vols)}] bb=[{min(bbs)}-{max(bbs)}] "
              f"bar=[{min(bars)}-{max(bars)}]")
        print(f"    Param cluster: {'YES (robust)' if clustered else 'NO (suspicious - may be overfitting)'}")
    else:
        print("    Not enough configs to analyze")

    print("=" * 70)
