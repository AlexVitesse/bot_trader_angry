"""
evaluate_adaptive_trailing.py -- ATR-adaptive trailing vs fixed 0.8%
====================================================================
Tests whether scaling the trailing stop with ATR improves results
across ADA, SOL, and DOGE.

Fixed: trail_dist = 0.008 (always 0.8%)
Adaptive: trail_dist = max(floor, atr_pct * factor)

Example with factor=0.25, floor=0.008:
  - ATR 2% -> trail = max(0.008, 0.02*0.25) = 0.008 (floor)
  - ATR 3% -> trail = max(0.008, 0.03*0.25) = 0.008 (floor)
  - ATR 4% -> trail = max(0.008, 0.04*0.25) = 0.010
  - ATR 6% -> trail = max(0.008, 0.06*0.25) = 0.015

This adapts to volatility: wider stops in high-vol, tight in low-vol.
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
    metrics, WF_FOLDS, COMMISSION,
)

REGIME_DEAD_ZONE = 0.02
OOS_START = '2026-01-01'
OOS_END   = '2026-03-15'
MIN_BEAR_BARS = 30
PAIRS = ['ADA', 'SOL', 'DOGE']


# ==============================================================
# HELPERS
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
# ADAPTIVE TRAILING SIMULATIONS
# ==============================================================
def sim_long_adaptive(df, entry_bar, entry_price, atr_pct_entry,
                      factor, floor, max_bars=30):
    """LONG trailing with ATR-adaptive distance."""
    trail_dist = max(floor, atr_pct_entry * factor)
    sl_price = entry_price * (1 - trail_dist)
    peak = entry_price

    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df):
            exit_p = float(df['close'].iloc[-1])
            pnl = (exit_p - entry_price) / entry_price - 2 * COMMISSION
            return ('TP' if exit_p > entry_price else 'SL'), exit_p, pnl, i, trail_dist
        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        if hi > peak:
            peak = hi
        sl_price = max(sl_price, peak * (1 - trail_dist))
        if lo <= sl_price:
            pnl = (sl_price - entry_price) / entry_price - 2 * COMMISSION
            return ('TP' if sl_price > entry_price else 'SL'), sl_price, pnl, i, trail_dist
    exit_p = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (exit_p - entry_price) / entry_price - 2 * COMMISSION
    return ('TP' if exit_p > entry_price else 'SL'), exit_p, pnl, max_bars, trail_dist


def sim_short_adaptive(df, entry_bar, entry_price, atr_pct_entry,
                       factor, floor, max_bars=30):
    """SHORT trailing with ATR-adaptive distance."""
    trail_dist = max(floor, atr_pct_entry * factor)
    sl_price = entry_price * (1 + trail_dist)
    trough = entry_price

    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df):
            exit_p = float(df['close'].iloc[-1])
            pnl = (entry_price - exit_p) / entry_price - 2 * COMMISSION
            return ('TP' if exit_p < entry_price else 'SL'), exit_p, pnl, i, trail_dist
        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        if lo < trough:
            trough = lo
        new_sl = trough * (1 + trail_dist)
        sl_price = min(sl_price, new_sl)
        if hi >= sl_price:
            pnl = (entry_price - sl_price) / entry_price - 2 * COMMISSION
            return ('TP' if sl_price < entry_price else 'SL'), sl_price, pnl, i, trail_dist
    exit_p = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (entry_price - exit_p) / entry_price - 2 * COMMISSION
    return ('TP' if exit_p < entry_price else 'SL'), exit_p, pnl, max_bars, trail_dist


def sim_long_fixed(df, entry_bar, entry_price, sl_pct, max_bars=30):
    """LONG trailing with fixed distance (baseline)."""
    sl_price = entry_price * (1 - sl_pct)
    peak = entry_price
    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df):
            exit_p = float(df['close'].iloc[-1])
            pnl = (exit_p - entry_price) / entry_price - 2 * COMMISSION
            return ('TP' if exit_p > entry_price else 'SL'), exit_p, pnl, i
        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        if hi > peak:
            peak = hi
        sl_price = max(sl_price, peak * (1 - sl_pct))
        if lo <= sl_price:
            pnl = (sl_price - entry_price) / entry_price - 2 * COMMISSION
            return ('TP' if sl_price > entry_price else 'SL'), sl_price, pnl, i
    exit_p = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (exit_p - entry_price) / entry_price - 2 * COMMISSION
    return ('TP' if exit_p > entry_price else 'SL'), exit_p, pnl, max_bars


def sim_short_fixed(df, entry_bar, entry_price, sl_pct, max_bars=30):
    """SHORT trailing with fixed distance (baseline)."""
    sl_price = entry_price * (1 + sl_pct)
    trough = entry_price
    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df):
            exit_p = float(df['close'].iloc[-1])
            pnl = (entry_price - exit_p) / entry_price - 2 * COMMISSION
            return ('TP' if exit_p < entry_price else 'SL'), exit_p, pnl, i
        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        if lo < trough:
            trough = lo
        new_sl = trough * (1 + sl_pct)
        sl_price = min(sl_price, new_sl)
        if hi >= sl_price:
            pnl = (entry_price - sl_price) / entry_price - 2 * COMMISSION
            return ('TP' if sl_price < entry_price else 'SL'), sl_price, pnl, i
    exit_p = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (entry_price - exit_p) / entry_price - 2 * COMMISSION
    return ('TP' if exit_p < entry_price else 'SL'), exit_p, pnl, max_bars


# ==============================================================
# DETECTORS
# ==============================================================
def detect_breakout(df, idx, vol_min=1.2, bb_max=7.0, bar_max=5.0):
    row = df.iloc[idx]
    close = row['close']
    high20 = row.get('high20', None)
    vol_ratio = row.get('vol_ratio', 0)
    bb_width = row.get('bb_width', 99)
    ret_1 = row.get('ret_1', 0)
    if high20 is None or pd.isna(high20):
        return False
    if close <= high20 or vol_ratio < vol_min or bb_width > bb_max or abs(ret_1) > bar_max:
        return False
    return True


def detect_btc_breakout(df_btc, idx):
    if idx < 25:
        return False
    row = df_btc.iloc[idx]
    close = float(row['close'])
    high20 = df_btc['high'].iloc[max(0, idx-20):idx].max()
    vol_ratio = float(row.get('vol_ratio', 0))
    return close > high20 and vol_ratio >= 1.0


def detect_btc_breakdown(df_btc, idx):
    if idx < 25:
        return False
    row = df_btc.iloc[idx]
    close = float(row['close'])
    low20 = df_btc['low'].iloc[max(0, idx-20):idx].min()
    vol_ratio = float(row.get('vol_ratio', 0))
    return close < low20 and vol_ratio >= 1.0


# ==============================================================
# DATA LOADING
# ==============================================================
def load_pair_data(pair):
    df_raw = load_pair_4h(pair)
    df = compute_features_4h(df_raw.copy())
    try:
        from v15_framework import load_pair_1d
        pair_1d = load_pair_1d(pair)
    except Exception:
        pair_1d = df_raw.resample('1D').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()
    macro = compute_macro_daily(pair_1d)
    df = merge_daily_to_4h(df, macro)
    regimes = df.apply(lambda r: detect_regime(r), axis=1)

    df_btc = compute_features_4h(load_btc_4h().copy())
    common_idx = df.index.intersection(df_btc.index)
    if len(common_idx) > 100:
        pair_ret = df.loc[common_idx, 'close'].pct_change()
        btc_ret = df_btc.loc[common_idx, 'close'].pct_change()
        roll_corr = pair_ret.rolling(168).corr(btc_ret)
        df['pair_btc_corr'] = roll_corr.reindex(df.index).ffill()
    else:
        df['pair_btc_corr'] = 0.7

    return df, regimes, df_btc


# ==============================================================
# WALK-FORWARD: generates entry list (shared between fixed/adaptive)
# ==============================================================
def generate_long_entries(df, df_btc, regimes, vol_min=1.2, bb_max=7.0,
                          btc_corr_min=0.4, max_bars=30):
    """Generate all LONG entry points (shared across exit methods)."""
    entries = []
    for fi, (start_s, end_s) in enumerate(WF_FOLDS):
        fold_start = pd.Timestamp(start_s, tz='UTC')
        fold_end = pd.Timestamp(end_s, tz='UTC')
        fold_mask = (df.index >= fold_start) & (df.index <= fold_end)
        fold_indices = np.where(fold_mask)[0]

        for idx in fold_indices:
            if idx + max_bars + 1 >= len(df):
                continue
            reg = regimes.iloc[idx]
            if reg == 'BEAR':
                continue
            ts = df.index[idx]
            entry_price = float(df['close'].iloc[idx])
            atr_pct = float(df['atr_pct'].iloc[idx]) / 100.0  # convert from % to decimal

            setup = None
            if detect_breakout(df, idx, vol_min, bb_max):
                setup = 'BRK'
            else:
                btc_idx = df_btc.index.get_indexer([ts], method='nearest')[0]
                if 0 <= btc_idx < len(df_btc) and detect_btc_breakout(df_btc, btc_idx):
                    corr_val = float(df['pair_btc_corr'].iloc[idx])
                    if corr_val >= btc_corr_min:
                        setup = 'FBTC'

            if setup:
                entries.append({
                    'idx': idx, 'ts': ts, 'entry_price': entry_price,
                    'atr_pct': atr_pct, 'setup': setup, 'fold': fi,
                    'direction': 'LONG'
                })
    return entries


def generate_short_entries(df, df_btc, regimes, btc_corr_min=0.4, max_bars=30):
    """Generate all SHORT entry points (shared across exit methods)."""
    entries = []
    for fi, (start_s, end_s) in enumerate(WF_FOLDS):
        fold_start = pd.Timestamp(start_s, tz='UTC')
        fold_end = pd.Timestamp(end_s, tz='UTC')
        fold_mask = (df.index >= fold_start) & (df.index <= fold_end)
        fold_indices = np.where(fold_mask)[0]

        bear_bars = sum(1 for idx in fold_indices if regimes.iloc[idx] == 'BEAR')
        if bear_bars < MIN_BEAR_BARS:
            continue

        for idx in fold_indices:
            if idx + max_bars + 1 >= len(df):
                continue
            reg = regimes.iloc[idx]
            if reg != 'BEAR':
                continue
            ts = df.index[idx]
            entry_price = float(df['close'].iloc[idx])
            atr_pct = float(df['atr_pct'].iloc[idx]) / 100.0

            btc_idx = df_btc.index.get_indexer([ts], method='nearest')[0]
            if 0 <= btc_idx < len(df_btc) and detect_btc_breakdown(df_btc, btc_idx):
                corr_val = float(df['pair_btc_corr'].iloc[idx])
                if corr_val >= btc_corr_min:
                    entries.append({
                        'idx': idx, 'ts': ts, 'entry_price': entry_price,
                        'atr_pct': atr_pct, 'setup': 'BTC_BD', 'fold': fi,
                        'direction': 'SHORT'
                    })
    return entries


# ==============================================================
# EVALUATE a config (fixed or adaptive) against entry list
# ==============================================================
def evaluate_config(entries, df, config_name, direction, exit_fn):
    """Run exit_fn on all entries, compute per-fold WF and aggregate."""
    trades = []
    for e in entries:
        result = exit_fn(e)
        trades.append({
            'ts': e['ts'], 'pnl_pct': result['pnl'],
            'outcome': result['outcome'], 'setup': e['setup'],
            'direction': direction, 'entry': e['entry_price'],
            'trail_dist': result.get('trail_dist', None),
            'fold': e['fold']
        })

    # Per-fold WF
    fold_results = []
    for fi in range(len(WF_FOLDS)):
        ft = [t for t in trades if t['fold'] == fi]
        m = metrics(ft, '')
        if direction == 'LONG':
            ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        else:
            ok = (m['n'] >= 1 and m['pf'] > 0.8)
        fold_results.append({'n': m['n'], 'wr': m['wr'], 'pf': m['pf'], 'ok': ok})

    m_all = metrics(trades, config_name)
    eq, dd = equity_stats(trades)

    if direction == 'LONG':
        valid = [r for r in fold_results if r['n'] > 0]
        folds_ok = sum(1 for r in valid if r['ok'])
        folds_total = len(valid)
    else:
        valid = [r for r in fold_results if r['n'] > 0]
        folds_ok = sum(1 for r in valid if r['ok'])
        folds_total = len(valid)

    return {
        'name': config_name, 'folds_ok': folds_ok, 'folds_total': folds_total,
        'n': m_all['n'], 'wr': m_all['wr'], 'pf': m_all['pf'],
        'eq': eq, 'dd': dd, 'trades': trades
    }


def print_result(r, direction):
    if direction == 'LONG':
        passed = r['folds_ok'] >= 7 and r['pf'] >= 1.0
    else:
        threshold = max(1, int(r['folds_total'] * 0.6))
        passed = r['folds_ok'] >= threshold and r['n'] >= 5 and r['pf'] >= 1.0
    tag = "OK" if passed else "NO"
    print(f"    {r['name']:<40} WF {r['folds_ok']:>2}/{r['folds_total']:>2} | "
          f"N={r['n']:>4} WR={r['wr']:.1%} PF={r['pf']:.2f} "
          f"$1K->${1000*r['eq']:.0f} DD={r['dd']:.1%}  [{tag}]")
    return passed


# ==============================================================
# OOS EVALUATION
# ==============================================================
def oos_eval(entries_all, df, exit_fn, direction):
    """Run OOS 2026."""
    oos_start = pd.Timestamp(OOS_START, tz='UTC')
    oos_end = pd.Timestamp(OOS_END, tz='UTC')

    oos_entries = [e for e in entries_all if oos_start <= e['ts'] <= oos_end]
    trades = []
    for e in oos_entries:
        result = exit_fn(e)
        trades.append({
            'ts': e['ts'], 'pnl_pct': result['pnl'],
            'outcome': result['outcome'], 'setup': e['setup'],
            'trail_dist': result.get('trail_dist', None),
            'entry': e['entry_price'],
        })
    return trades


def generate_oos_entries(df, df_btc, regimes, direction,
                         vol_min=1.2, bb_max=7.0, btc_corr_min=0.4, max_bars=30):
    """Generate entries for OOS period (not tied to WF folds)."""
    oos_start = pd.Timestamp(OOS_START, tz='UTC')
    oos_end = pd.Timestamp(OOS_END, tz='UTC')
    oos_mask = (df.index >= oos_start) & (df.index <= oos_end)
    oos_indices = np.where(oos_mask)[0]

    entries = []
    for idx in oos_indices:
        if idx + max_bars + 1 >= len(df):
            continue
        reg = regimes.iloc[idx]
        ts = df.index[idx]
        entry_price = float(df['close'].iloc[idx])
        atr_pct = float(df['atr_pct'].iloc[idx]) / 100.0

        if direction == 'LONG':
            if reg == 'BEAR':
                continue
            setup = None
            if detect_breakout(df, idx, vol_min, bb_max):
                setup = 'BRK'
            else:
                btc_idx = df_btc.index.get_indexer([ts], method='nearest')[0]
                if 0 <= btc_idx < len(df_btc) and detect_btc_breakout(df_btc, btc_idx):
                    corr_val = float(df['pair_btc_corr'].iloc[idx])
                    if corr_val >= btc_corr_min:
                        setup = 'FBTC'
            if setup:
                entries.append({
                    'idx': idx, 'ts': ts, 'entry_price': entry_price,
                    'atr_pct': atr_pct, 'setup': setup, 'direction': 'LONG'
                })
        else:  # SHORT
            if reg != 'BEAR':
                continue
            btc_idx = df_btc.index.get_indexer([ts], method='nearest')[0]
            if 0 <= btc_idx < len(df_btc) and detect_btc_breakdown(df_btc, btc_idx):
                corr_val = float(df['pair_btc_corr'].iloc[idx])
                if corr_val >= btc_corr_min:
                    entries.append({
                        'idx': idx, 'ts': ts, 'entry_price': entry_price,
                        'atr_pct': atr_pct, 'setup': 'BTC_BD', 'direction': 'SHORT'
                    })
    return entries


# ==============================================================
# MAIN
# ==============================================================
if __name__ == '__main__':
    print("=" * 75)
    print("ATR-Adaptive Trailing Stop Evaluation")
    print("Fixed 0.8% vs max(floor, ATR * factor)")
    print("=" * 75)

    # Configs to test
    # (name, factor, floor)
    adaptive_configs = [
        ('ATR*0.20_floor0.6%',  0.20, 0.006),
        ('ATR*0.25_floor0.6%',  0.25, 0.006),
        ('ATR*0.25_floor0.8%',  0.25, 0.008),
        ('ATR*0.30_floor0.6%',  0.30, 0.006),
        ('ATR*0.30_floor0.8%',  0.30, 0.008),
        ('ATR*0.35_floor0.8%',  0.35, 0.008),
        ('ATR*0.40_floor0.8%',  0.40, 0.008),
        ('ATR*0.50_floor0.8%',  0.50, 0.008),
    ]

    # Load BTC once
    df_btc = compute_features_4h(load_btc_4h().copy())

    summary_rows = []

    for pair in PAIRS:
        print(f"\n{'='*75}")
        print(f"  {pair}/USDT")
        print(f"{'='*75}")

        df, regimes, _ = load_pair_data(pair)
        atr_mean = df['atr_pct'].mean()
        print(f"  ATR mean: {atr_mean:.2f}%")

        # Generate entries ONCE (same for all configs)
        long_entries = generate_long_entries(df, df_btc, regimes)
        short_entries = generate_short_entries(df, df_btc, regimes)
        oos_long_entries = generate_oos_entries(df, df_btc, regimes, 'LONG')
        oos_short_entries = generate_oos_entries(df, df_btc, regimes, 'SHORT')

        print(f"  Entries: {len(long_entries)} LONG, {len(short_entries)} SHORT")
        print(f"  OOS entries: {len(oos_long_entries)} LONG, {len(oos_short_entries)} SHORT")

        # --- BASELINE: Fixed 0.8% ---
        print(f"\n  --- BASELINE: Fixed 0.8% ---")

        def fixed_long_exit(df, e):
            out = sim_long_fixed(df, e['idx'], e['entry_price'], 0.008)
            return {'outcome': out[0], 'pnl': out[2], 'trail_dist': 0.008}

        def fixed_short_exit(df, e):
            out = sim_short_fixed(df, e['idx'], e['entry_price'], 0.008)
            return {'outcome': out[0], 'pnl': out[2], 'trail_dist': 0.008}

        rl = evaluate_config(long_entries, df, 'Fixed_0.8%', 'LONG',
                             lambda e, _df=df: fixed_long_exit(_df, e))
        print_result(rl, 'LONG')
        rs = evaluate_config(short_entries, df, 'Fixed_0.8%', 'SHORT',
                             lambda e, _df=df: fixed_short_exit(_df, e))
        print_result(rs, 'SHORT')

        # OOS baseline
        oos_l = [fixed_long_exit(df, e) for e in oos_long_entries]
        oos_s = [fixed_short_exit(df, e) for e in oos_short_entries]
        oos_l_ret = sum(t['pnl'] for t in oos_l)
        oos_s_ret = sum(t['pnl'] for t in oos_s)
        print(f"    OOS 2026: LONG {oos_l_ret:+.2%} ({len(oos_l)}t) "
              f"SHORT {oos_s_ret:+.2%} ({len(oos_s)}t) "
              f"TOTAL {oos_l_ret+oos_s_ret:+.2%}")

        summary_rows.append({
            'pair': pair, 'config': 'Fixed_0.8%',
            'long_wf': f"{rl['folds_ok']}/{rl['folds_total']}",
            'long_pf': rl['pf'], 'long_dd': rl['dd'],
            'short_wf': f"{rs['folds_ok']}/{rs['folds_total']}",
            'short_pf': rs['pf'], 'short_dd': rs['dd'],
            'oos_long': oos_l_ret, 'oos_short': oos_s_ret,
            'oos_total': oos_l_ret + oos_s_ret,
        })

        # --- ADAPTIVE CONFIGS ---
        print(f"\n  --- ADAPTIVE: max(floor, ATR * factor) ---")

        for cfg_name, factor, floor in adaptive_configs:
            def make_long_exit(f, fl):
                def exit_fn(e, _df=df):
                    out = sim_long_adaptive(_df, e['idx'], e['entry_price'],
                                           e['atr_pct'], f, fl)
                    return {'outcome': out[0], 'pnl': out[2], 'trail_dist': out[4]}
                return exit_fn

            def make_short_exit(f, fl):
                def exit_fn(e, _df=df):
                    out = sim_short_adaptive(_df, e['idx'], e['entry_price'],
                                            e['atr_pct'], f, fl)
                    return {'outcome': out[0], 'pnl': out[2], 'trail_dist': out[4]}
                return exit_fn

            long_exit = make_long_exit(factor, floor)
            short_exit = make_short_exit(factor, floor)

            rl = evaluate_config(long_entries, df, cfg_name, 'LONG', long_exit)
            print_result(rl, 'LONG')
            rs = evaluate_config(short_entries, df, cfg_name, 'SHORT', short_exit)
            print_result(rs, 'SHORT')

            # OOS
            oos_l = [long_exit(e) for e in oos_long_entries]
            oos_s = [short_exit(e) for e in oos_short_entries]
            oos_l_ret = sum(t['pnl'] for t in oos_l)
            oos_s_ret = sum(t['pnl'] for t in oos_s)
            print(f"    OOS 2026: LONG {oos_l_ret:+.2%} ({len(oos_l)}t) "
                  f"SHORT {oos_s_ret:+.2%} ({len(oos_s)}t) "
                  f"TOTAL {oos_l_ret+oos_s_ret:+.2%}")

            # ATR distribution for this config
            if long_entries:
                dists = [max(floor, e['atr_pct'] * factor) for e in long_entries]
                print(f"    Trail dist: min={min(dists):.3%} median={np.median(dists):.3%} "
                      f"max={max(dists):.3%} (at_floor={sum(1 for d in dists if d <= floor+0.0001)/len(dists):.0%})")

            summary_rows.append({
                'pair': pair, 'config': cfg_name,
                'long_wf': f"{rl['folds_ok']}/{rl['folds_total']}",
                'long_pf': rl['pf'], 'long_dd': rl['dd'],
                'short_wf': f"{rs['folds_ok']}/{rs['folds_total']}",
                'short_pf': rs['pf'], 'short_dd': rs['dd'],
                'oos_long': oos_l_ret, 'oos_short': oos_s_ret,
                'oos_total': oos_l_ret + oos_s_ret,
            })

    # ==============================================================
    # CROSS-PAIR SUMMARY
    # ==============================================================
    print(f"\n{'='*75}")
    print("CROSS-PAIR SUMMARY")
    print(f"{'='*75}")
    print(f"  {'Config':<25} | {'ADA OOS':>10} | {'SOL OOS':>10} | {'DOGE OOS':>10} | {'AVG OOS':>10}")
    print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    configs_list = ['Fixed_0.8%'] + [c[0] for c in adaptive_configs]
    for cfg in configs_list:
        vals = {}
        for pair in PAIRS:
            row = [r for r in summary_rows if r['pair'] == pair and r['config'] == cfg]
            if row:
                vals[pair] = row[0]['oos_total']
            else:
                vals[pair] = 0
        avg = np.mean(list(vals.values()))
        print(f"  {cfg:<25} | {vals.get('ADA',0):>+9.2%} | {vals.get('SOL',0):>+9.2%} | "
              f"{vals.get('DOGE',0):>+9.2%} | {avg:>+9.2%}")

    # Best config per pair
    print(f"\n  Best config per pair (by OOS total):")
    for pair in PAIRS:
        pair_rows = [r for r in summary_rows if r['pair'] == pair]
        if pair_rows:
            best = max(pair_rows, key=lambda x: x['oos_total'])
            print(f"    {pair}: {best['config']} -> OOS {best['oos_total']:+.2%}")

    # Best universal config (avg OOS across all pairs)
    print(f"\n  Best universal config (avg OOS across pairs):")
    best_avg = -999
    best_cfg = None
    for cfg in configs_list:
        vals = []
        for pair in PAIRS:
            row = [r for r in summary_rows if r['pair'] == pair and r['config'] == cfg]
            if row:
                vals.append(row[0]['oos_total'])
        if vals:
            avg = np.mean(vals)
            if avg > best_avg:
                best_avg = avg
                best_cfg = cfg
    print(f"    {best_cfg} -> avg OOS {best_avg:+.2%}")

    print(f"\n{'='*75}")
    print("DONE")
    print(f"{'='*75}")
