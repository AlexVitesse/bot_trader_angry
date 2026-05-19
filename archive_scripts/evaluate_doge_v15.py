"""
evaluate_doge_v15.py -- DOGE V15 evaluation with tight trailing
===============================================================
Tests the BTC-follower + tight trailing approach that APROBADO for ADA/SOL:
  - LONG (BULL/RANGE): BTC-follower + standalone breakout, tight trailing 0.8%
  - SHORT (BEAR): BTC-breakdown follower, tight trailing 0.8%

DOGE specifics vs ADA/SOL:
  - Higher beta to BTC (amplifies moves) -> should benefit more from trailing
  - Higher volatility -> may need adjusted bb_max
  - 92% overfitting with ML -> rule-based only

Walk-forward: 12 folds, need >=7/12 LONG, >=60% valid BEAR folds SHORT.
OOS: 2026-01-01 to 2026-03-15.

Usage:
  python evaluate_doge_v15.py
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

# ==============================================================
# CONSTANTS
# ==============================================================
REGIME_DEAD_ZONE = 0.02
OOS_START = '2026-01-01'
OOS_END = '2026-03-15'
TRAIN_CUTOFF = '2025-12-31'
MIN_BEAR_BARS = 30
PAIR = 'DOGE'


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


def compute_sized_equity(trades, sizing_mult):
    if not trades:
        return 1.0, 0
    cum = 1.0; peak = 1.0; max_dd = 0
    for t in sorted(trades, key=lambda x: x['ts']):
        sized_pnl = t['pnl_pct'] * sizing_mult
        cum *= (1 + sized_pnl)
        peak = max(peak, cum)
        dd = (peak - cum) / peak
        max_dd = max(max_dd, dd)
    return cum, max_dd


# ==============================================================
# TRADE SIMULATIONS
# ==============================================================
def sim_long_trailing(df, entry_bar, entry_price, sl_pct,
                      trail_trigger_pct=None, max_bars=30):
    """LONG trailing stop. Tracks highs (peak)."""
    sl_price = entry_price * (1 - sl_pct)
    peak = entry_price
    trailing_active = (trail_trigger_pct is None)

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
        if not trailing_active and trail_trigger_pct is not None:
            if hi >= entry_price * (1 + trail_trigger_pct):
                trailing_active = True
        if trailing_active:
            sl_price = max(sl_price, peak * (1 - sl_pct))
        if lo <= sl_price:
            pnl = (sl_price - entry_price) / entry_price - 2 * COMMISSION
            return ('TP' if sl_price > entry_price else 'SL'), sl_price, pnl, i
    exit_p = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (exit_p - entry_price) / entry_price - 2 * COMMISSION
    return ('TP' if exit_p > entry_price else 'SL'), exit_p, pnl, max_bars


def sim_short_trailing(df, entry_bar, entry_price, sl_pct,
                       trail_trigger_pct=None, max_bars=30):
    """SHORT trailing stop. Tracks lows (trough)."""
    sl_price = entry_price * (1 + sl_pct)
    trough = entry_price
    trailing_active = (trail_trigger_pct is None)

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
        if not trailing_active and trail_trigger_pct is not None:
            if lo <= entry_price * (1 - trail_trigger_pct):
                trailing_active = True
        if trailing_active:
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
    """Breakout above 20-bar high."""
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
    """BTC breaks above 20-bar high -> follower LONG."""
    if idx < 25:
        return False
    row = df_btc.iloc[idx]
    close = float(row['close'])
    high20 = df_btc['high'].iloc[max(0, idx-20):idx].max()
    vol_ratio = float(row.get('vol_ratio', 0))
    if close <= high20 or vol_ratio < 1.0:
        return False
    return True


def detect_btc_breakdown(df_btc, idx):
    """BTC breaks below 20-bar low -> follower SHORT."""
    if idx < 25:
        return False
    row = df_btc.iloc[idx]
    close = float(row['close'])
    low20 = df_btc['low'].iloc[max(0, idx-20):idx].min()
    vol_ratio = float(row.get('vol_ratio', 0))
    if close >= low20 or vol_ratio < 1.0:
        return False
    return True


# ==============================================================
# DATA LOADING
# ==============================================================
def load_pair_data(pair):
    print(f"  Loading {pair} 4h data...")
    df_raw = load_pair_4h(pair)
    df = compute_features_4h(df_raw.copy())

    try:
        from v15_framework import load_pair_1d
        pair_1d = load_pair_1d(pair)
    except (FileNotFoundError, Exception):
        print(f"    No daily data for {pair}, resampling 4h -> 1d")
        pair_1d = df_raw.resample('1D').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()
    macro = compute_macro_daily(pair_1d)
    df = merge_daily_to_4h(df, macro)

    regimes = df.apply(lambda r: detect_regime(r), axis=1)

    # BTC cross-reference
    df_btc = compute_features_4h(load_btc_4h().copy())
    common_idx = df.index.intersection(df_btc.index)
    if len(common_idx) > 100:
        pair_ret = df.loc[common_idx, 'close'].pct_change()
        btc_ret = df_btc.loc[common_idx, 'close'].pct_change()
        roll_corr = pair_ret.rolling(168).corr(btc_ret)
        df['pair_btc_corr'] = roll_corr.reindex(df.index).ffill()
    else:
        df['pair_btc_corr'] = 0.7

    print(f"    {pair}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
    print(f"    ATR%: {df['atr_pct'].mean():.2f}%, BB width: {df['bb_width'].mean():.1f}%")
    reg_counts = regimes.value_counts().to_dict()
    print(f"    Regimes: {reg_counts}")
    corr_mean = df['pair_btc_corr'].mean()
    print(f"    DOGE-BTC corr: mean={corr_mean:.3f}")

    return df, regimes, df_btc


# ==============================================================
# LONG WALK-FORWARD (BTC-follower + breakout, trailing)
# ==============================================================
def long_wf(df, df_btc, regimes, trail_sl, trail_trigger, max_bars, name,
            vol_min=1.2, bb_max=7.0, btc_corr_min=0.4):
    """Rule-based LONG walk-forward with trailing stop exits."""
    results = []
    all_trades = []

    for fi, (start_s, end_s) in enumerate(WF_FOLDS):
        fold_start = pd.Timestamp(start_s, tz='UTC')
        fold_end = pd.Timestamp(end_s, tz='UTC')
        fold_mask = (df.index >= fold_start) & (df.index <= fold_end)
        fold_indices = np.where(fold_mask)[0]

        fold_trades = []
        for idx in fold_indices:
            if idx + max_bars + 1 >= len(df):
                continue
            reg = regimes.iloc[idx]
            if reg == 'BEAR':
                continue
            ts = df.index[idx]
            entry = float(df['close'].iloc[idx])
            triggered = False

            # 1. Standalone breakout
            if detect_breakout(df, idx, vol_min, bb_max):
                out = sim_long_trailing(df, idx, entry, trail_sl, trail_trigger, max_bars)
                fold_trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                                    'setup': 'BRK', 'direction': 'LONG', 'entry': entry})
                triggered = True

            # 2. BTC-follower
            if not triggered:
                btc_idx = df_btc.index.get_indexer([ts], method='nearest')[0]
                if 0 <= btc_idx < len(df_btc) and detect_btc_breakout(df_btc, btc_idx):
                    corr_val = float(df['pair_btc_corr'].iloc[idx])
                    if corr_val >= btc_corr_min:
                        out = sim_long_trailing(df, idx, entry, trail_sl, trail_trigger, max_bars)
                        fold_trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                                            'setup': 'FBTC', 'direction': 'LONG', 'entry': entry})

        m = metrics(fold_trades, '')
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        results.append({'period': f"{start_s[:7]}/{end_s[5:7]}", 'n': m['n'],
                        'wr': m['wr'], 'pf': m['pf'], 'ok': ok})
        all_trades.extend(fold_trades)

    m_all = metrics(all_trades, name)
    eq, dd = equity_stats(all_trades)
    folds_ok = sum(1 for r in results if r['ok'])
    folds_data = sum(1 for r in results if r['n'] > 0)
    return {'name': name, 'folds_ok': folds_ok, 'folds_data': folds_data,
            'results': results, 'trades': all_trades,
            'n': m_all['n'], 'wr': m_all['wr'], 'pf': m_all['pf'],
            'eq': eq, 'dd': dd}


# ==============================================================
# SHORT WALK-FORWARD (BTC-breakdown follower, trailing)
# ==============================================================
def short_wf(df, df_btc, regimes, trail_sl, trail_trigger, max_bars, name,
             btc_corr_min=0.4):
    """SHORT walk-forward: BTC-breakdown follower, BEAR only, trailing exits."""
    results = []
    all_trades = []

    for fi, (start_s, end_s) in enumerate(WF_FOLDS):
        fold_start = pd.Timestamp(start_s, tz='UTC')
        fold_end = pd.Timestamp(end_s, tz='UTC')
        fold_mask = (df.index >= fold_start) & (df.index <= fold_end)
        fold_indices = np.where(fold_mask)[0]

        # Count BEAR bars in this fold
        bear_bars = sum(1 for idx in fold_indices if regimes.iloc[idx] == 'BEAR')

        fold_trades = []
        if bear_bars >= MIN_BEAR_BARS:
            for idx in fold_indices:
                if idx + max_bars + 1 >= len(df):
                    continue
                reg = regimes.iloc[idx]
                if reg != 'BEAR':
                    continue
                ts = df.index[idx]
                entry = float(df['close'].iloc[idx])

                # BTC breakdown -> follower SHORT
                btc_idx = df_btc.index.get_indexer([ts], method='nearest')[0]
                if 0 <= btc_idx < len(df_btc) and detect_btc_breakdown(df_btc, btc_idx):
                    corr_val = float(df['pair_btc_corr'].iloc[idx])
                    if corr_val >= btc_corr_min:
                        out = sim_short_trailing(df, idx, entry, trail_sl, trail_trigger, max_bars)
                        fold_trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                                            'setup': 'BTC_BREAKDOWN', 'direction': 'SHORT',
                                            'entry': entry})

        if bear_bars >= MIN_BEAR_BARS:
            m = metrics(fold_trades, '')
            ok = (m['n'] >= 1 and m['pf'] > 0.8)  # looser for SHORT (fewer trades)
            results.append({'period': f"{start_s[:7]}/{end_s[5:7]}", 'n': m['n'],
                            'wr': m['wr'], 'pf': m['pf'], 'ok': ok, 'bear_bars': bear_bars})
        else:
            results.append({'period': f"{start_s[:7]}/{end_s[5:7]}", 'n': 0,
                            'wr': 0, 'pf': 0, 'ok': False, 'bear_bars': bear_bars,
                            'skip': True})
        all_trades.extend(fold_trades)

    m_all = metrics(all_trades, name)
    eq, dd = equity_stats(all_trades)
    valid_folds = [r for r in results if not r.get('skip', False)]
    folds_ok = sum(1 for r in valid_folds if r['ok'])
    folds_total = len(valid_folds)
    return {'name': name, 'folds_ok': folds_ok, 'folds_total': folds_total,
            'results': results, 'trades': all_trades,
            'n': m_all['n'], 'wr': m_all['wr'], 'pf': m_all['pf'],
            'eq': eq, 'dd': dd}


# ==============================================================
# PRINT HELPERS
# ==============================================================
def print_long_result(r, prefix="  "):
    passed = r['folds_ok'] >= 7 and r['pf'] >= 1.0
    marginal = r['folds_ok'] >= 6 and r['pf'] >= 1.0
    tag = "APROBADO" if passed else ("MARGINAL" if marginal else "RECHAZADO")
    mark = "**" if passed else ("* " if marginal else "  ")
    print(f"{prefix}{mark}{r['name']:<40} WF {r['folds_ok']:>2}/{r['folds_data']:>2} | "
          f"N={r['n']:>4} WR={r['wr']:.1%} PF={r['pf']:.2f} "
          f"$1K->${1000*r['eq']:.0f} DD={r['dd']:.1%} -> {tag}")
    return tag


def print_short_result(r, prefix="  "):
    valid = r['folds_total']
    ok = r['folds_ok']
    threshold = max(1, int(valid * 0.6))
    passed = ok >= threshold and r['n'] >= 5 and r['pf'] >= 1.0
    marginal = ok >= max(1, int(valid * 0.5)) and r['n'] >= 3
    tag = "APROBADO" if passed else ("MARGINAL" if marginal else "RECHAZADO")
    mark = "**" if passed else ("* " if marginal else "  ")
    print(f"{prefix}{mark}{r['name']:<40} WF {ok:>2}/{valid:>2} | "
          f"N={r['n']:>4} WR={r['wr']:.1%} PF={r['pf']:.2f} "
          f"$1K->${1000*r['eq']:.0f} DD={r['dd']:.1%} -> {tag}")
    return tag


# ==============================================================
# MAIN
# ==============================================================
if __name__ == '__main__':
    print("=" * 70)
    print(f"DOGE V15 Evaluation -- Tight Trailing (ADA/SOL winning approach)")
    print("=" * 70)

    df, regimes, df_btc = load_pair_data(PAIR)

    # ===========================================================
    # PART 1: LONG (BULL/RANGE) -- Multiple trailing configs
    # ===========================================================
    print(f"\n{'='*70}")
    print("PART 1: LONG (BULL/RANGE) -- BTC-follower + Breakout + Trailing")
    print(f"{'='*70}")

    long_configs = [
        # (name, trail_sl, trail_trigger, max_bars, vol_min, bb_max, corr_min)
        # ADA/SOL winner: tight 0.8%, immediate, corr>=0.4
        ('tight_0.8pct_imm',     0.008, None,  30, 1.2, 7.0, 0.4),
        ('tight_1.0pct_imm',     0.010, None,  30, 1.2, 7.0, 0.4),
        ('tight_1.2pct_imm',     0.012, None,  30, 1.2, 7.0, 0.4),
        ('tight_1.5pct_imm',     0.015, None,  30, 1.2, 7.0, 0.4),
        # Higher volatility DOGE: wider trailing
        ('tight_2.0pct_imm',     0.020, None,  30, 1.2, 7.0, 0.4),
        # Wider bb_max for DOGE (more volatile)
        ('tight_0.8pct_bb8',     0.008, None,  30, 1.2, 8.0, 0.4),
        ('tight_1.0pct_bb8',     0.010, None,  30, 1.2, 8.0, 0.4),
        # Lower corr (DOGE can decouple from BTC)
        ('tight_0.8pct_corr03',  0.008, None,  30, 1.2, 7.0, 0.3),
        # Delayed activation (wait for 1% profit first)
        ('trail_0.8pct_trig1pct', 0.008, 0.01, 30, 1.2, 7.0, 0.4),
        ('trail_1.0pct_trig1pct', 0.010, 0.01, 30, 1.2, 7.0, 0.4),
        # Fixed TP/SL baselines for comparison
    ]

    # Also test fixed TP/SL baselines
    fixed_configs = [
        ('FIXED_TP3_SL1.5', 0.03, 0.015, 18),
        ('FIXED_TP4_SL2',   0.04, 0.02,  15),
        ('FIXED_TP6_SL4',   0.06, 0.04,  15),
    ]

    long_results = []

    # Trailing configs
    for name, t_sl, t_trig, mbars, vmin, bbmax, cmin in long_configs:
        full_name = f"LONG_{name}"
        r = long_wf(df, df_btc, regimes, t_sl, t_trig, mbars, full_name,
                    vol_min=vmin, bb_max=bbmax, btc_corr_min=cmin)
        tag = print_long_result(r)
        long_results.append((full_name, r, tag))

    # Fixed TP/SL baselines
    for name, tp, sl, timeout in fixed_configs:
        full_name = f"LONG_{name}"
        # Reuse long_wf but with fixed sim instead of trailing
        results = []
        all_trades = []
        for fi, (start_s, end_s) in enumerate(WF_FOLDS):
            fold_start = pd.Timestamp(start_s, tz='UTC')
            fold_end = pd.Timestamp(end_s, tz='UTC')
            fold_mask = (df.index >= fold_start) & (df.index <= fold_end)
            fold_indices = np.where(fold_mask)[0]
            fold_trades = []
            for idx in fold_indices:
                if idx + timeout + 1 >= len(df):
                    continue
                reg = regimes.iloc[idx]
                if reg == 'BEAR':
                    continue
                ts = df.index[idx]
                entry = float(df['close'].iloc[idx])
                triggered = False
                if detect_breakout(df, idx, 1.2, 7.0):
                    out = sim_trade_fixed(df, idx, entry, tp, sl, max_bars=timeout)
                    fold_trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                                        'setup': 'BRK', 'direction': 'LONG', 'entry': entry})
                    triggered = True
                if not triggered:
                    btc_idx = df_btc.index.get_indexer([ts], method='nearest')[0]
                    if 0 <= btc_idx < len(df_btc) and detect_btc_breakout(df_btc, btc_idx):
                        corr_val = float(df['pair_btc_corr'].iloc[idx])
                        if corr_val >= 0.4:
                            out = sim_trade_fixed(df, idx, entry, tp, sl, max_bars=timeout)
                            fold_trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                                                'setup': 'FBTC', 'direction': 'LONG', 'entry': entry})
            m = metrics(fold_trades, '')
            ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
            results.append({'period': f"{start_s[:7]}/{end_s[5:7]}", 'n': m['n'],
                            'wr': m['wr'], 'pf': m['pf'], 'ok': ok})
            all_trades.extend(fold_trades)
        m_all = metrics(all_trades, full_name)
        eq, dd = equity_stats(all_trades)
        folds_ok = sum(1 for r in results if r['ok'])
        folds_data = sum(1 for r in results if r['n'] > 0)
        r = {'name': full_name, 'folds_ok': folds_ok, 'folds_data': folds_data,
             'results': results, 'trades': all_trades,
             'n': m_all['n'], 'wr': m_all['wr'], 'pf': m_all['pf'],
             'eq': eq, 'dd': dd}
        tag = print_long_result(r)
        long_results.append((full_name, r, tag))

    # ===========================================================
    # PART 2: SHORT (BEAR) -- BTC-breakdown + trailing
    # ===========================================================
    print(f"\n{'='*70}")
    print("PART 2: SHORT (BEAR) -- BTC-breakdown follower + Trailing")
    print(f"{'='*70}")

    short_configs = [
        # (name, trail_sl, trail_trigger, max_bars, corr_min)
        ('tight_0.8pct_imm',     0.008, None,  30, 0.4),
        ('tight_1.0pct_imm',     0.010, None,  30, 0.4),
        ('tight_1.2pct_imm',     0.012, None,  30, 0.4),
        ('tight_1.5pct_imm',     0.015, None,  30, 0.4),
        ('tight_2.0pct_imm',     0.020, None,  30, 0.4),
        # Lower corr
        ('tight_0.8pct_corr03',  0.008, None,  30, 0.3),
        ('tight_1.0pct_corr03',  0.010, None,  30, 0.3),
        # Delayed trigger
        ('trail_0.8pct_trig1pct', 0.008, 0.01, 30, 0.4),
        ('trail_1.0pct_trig1pct', 0.010, 0.01, 30, 0.4),
    ]

    short_results = []

    for name, t_sl, t_trig, mbars, cmin in short_configs:
        full_name = f"SHORT_{name}"
        r = short_wf(df, df_btc, regimes, t_sl, t_trig, mbars, full_name,
                     btc_corr_min=cmin)
        tag = print_short_result(r)
        short_results.append((full_name, r, tag))

    # ===========================================================
    # PART 3: Sizing-adjusted DD
    # ===========================================================
    print(f"\n{'='*70}")
    print("PART 3: Sizing-Adjusted DD")
    print(f"{'='*70}")

    all_results = long_results + short_results
    candidates = [(n, r, t) for n, r, t in all_results
                  if r.get('folds_ok', 0) >= 5 and r['n'] >= 5]

    if candidates:
        print(f"  {'Config':<40} | {'1.0x DD':>7} | {'0.5x DD':>7} | {'0.3x DD':>7} | 0.3x Ann%")
        print(f"  {'-'*85}")
        years = (df.index[-1] - df.index[0]).days / 365.25
        for name, r, tag in candidates:
            eq_03, dd_03 = compute_sized_equity(r['trades'], 0.3)
            eq_05, dd_05 = compute_sized_equity(r['trades'], 0.5)
            ann_03 = (eq_03 ** (1.0 / years) - 1) * 100 if eq_03 > 0 and years > 0 else 0
            print(f"  {name:<40} | {r['dd']:>6.1%} | {dd_05:>6.1%} | {dd_03:>6.1%} | {ann_03:>5.1f}%")

    # ===========================================================
    # PART 4: Per-fold detail (best LONG + best SHORT)
    # ===========================================================
    print(f"\n{'='*70}")
    print("PART 4: Per-Fold Detail")
    print(f"{'='*70}")

    # Best LONG
    approved_long = [(n, r, t) for n, r, t in long_results if t == 'APROBADO']
    if not approved_long:
        approved_long = [(n, r, t) for n, r, t in long_results if t == 'MARGINAL']
    if approved_long:
        best_long_name, best_long, _ = max(approved_long, key=lambda x: x[1]['pf'])
        print(f"\n  Best LONG: {best_long_name}")
        print(f"  {'Fold':<18} | {'N':>3} | {'WR':>6} | {'PF':>6} | OK?")
        print(f"  {'-'*50}")
        for r in best_long['results']:
            print(f"  {r['period']:<18} | {r['n']:>3} | {r['wr']:>5.1%} | {r['pf']:>5.2f} | {'YES' if r['ok'] else 'no'}")

    # Best SHORT
    approved_short = [(n, r, t) for n, r, t in short_results if t == 'APROBADO']
    if not approved_short:
        approved_short = [(n, r, t) for n, r, t in short_results if t == 'MARGINAL']
    if approved_short:
        best_short_name, best_short, _ = max(approved_short, key=lambda x: x[1]['pf'])
        print(f"\n  Best SHORT: {best_short_name}")
        print(f"  {'Fold':<18} | {'N':>3} | {'WR':>6} | {'PF':>6} | BEAR | OK?")
        print(f"  {'-'*60}")
        for r in best_short['results']:
            skip = r.get('skip', False)
            bear = r.get('bear_bars', 0)
            if skip:
                print(f"  {r['period']:<18} | {'':>3} | {'':>6} | {'':>6} | {bear:>4} | SKIP (<{MIN_BEAR_BARS})")
            else:
                print(f"  {r['period']:<18} | {r['n']:>3} | {r['wr']:>5.1%} | {r['pf']:>5.2f} | {bear:>4} | {'YES' if r['ok'] else 'no'}")

    # ===========================================================
    # PART 5: OOS 2026
    # ===========================================================
    print(f"\n{'='*70}")
    print("PART 5: OOS 2026 (Jan-Mar)")
    print(f"{'='*70}")

    mask_oos = (df.index >= pd.Timestamp(OOS_START, tz='UTC')) & \
               (df.index < pd.Timestamp(OOS_END, tz='UTC'))
    if mask_oos.sum() > 0:
        p_start = float(df.loc[mask_oos, 'close'].iloc[0])
        p_end = float(df.loc[mask_oos, 'close'].iloc[-1])
        p_ret = (p_end / p_start - 1) * 100
        print(f"  DOGE B&H: ${p_start:.5f} -> ${p_end:.5f} ({p_ret:+.1f}%)")
        oos_reg = regimes[mask_oos].value_counts().to_dict()
        print(f"  Regimes 2026: {oos_reg}")

    oos_indices = np.where(mask_oos)[0]

    # OOS for best LONG
    if approved_long:
        name, best, _ = max(approved_long, key=lambda x: x[1]['pf'])
        # Extract config from name
        trail_sl = 0.008  # default
        trail_trig = None
        if '1.0pct' in name: trail_sl = 0.010
        elif '1.2pct' in name: trail_sl = 0.012
        elif '1.5pct' in name: trail_sl = 0.015
        elif '2.0pct' in name: trail_sl = 0.020
        if 'trig1pct' in name: trail_trig = 0.01
        corr_min = 0.3 if 'corr03' in name else 0.4
        bb_max = 8.0 if 'bb8' in name else 7.0

        long_oos_trades = []
        for idx in oos_indices:
            if idx + 30 + 1 >= len(df):
                continue
            reg = regimes.iloc[idx]
            if reg == 'BEAR':
                continue
            ts = df.index[idx]
            entry = float(df['close'].iloc[idx])
            triggered = False
            if detect_breakout(df, idx, 1.2, bb_max):
                out = sim_long_trailing(df, idx, entry, trail_sl, trail_trig, 30)
                long_oos_trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                                        'setup': 'BRK', 'entry': entry, 'bars': out[3]})
                triggered = True
            if not triggered:
                btc_idx = df_btc.index.get_indexer([ts], method='nearest')[0]
                if 0 <= btc_idx < len(df_btc) and detect_btc_breakout(df_btc, btc_idx):
                    corr_val = float(df['pair_btc_corr'].iloc[idx])
                    if corr_val >= corr_min:
                        out = sim_long_trailing(df, idx, entry, trail_sl, trail_trig, 30)
                        long_oos_trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                                                'setup': 'FBTC', 'entry': entry, 'bars': out[3]})

        print(f"\n  --- LONG OOS ({name}) ---")
        if long_oos_trades:
            for t in sorted(long_oos_trades, key=lambda x: x['ts']):
                print(f"    {str(t['ts'])[:19]} ${t['entry']:>8.5f} {t['setup']:<5} "
                      f"{t['outcome']:<4} {t['pnl_pct']:>+7.2%} ({t['bars']}bars)")
            m = metrics(long_oos_trades, name)
            total = sum(t['pnl_pct'] for t in long_oos_trades)
            print(f"    N={m['n']} WR={m['wr']:.1%} PF={m['pf']:.2f} Total={total:+.2%}")
        else:
            print("    No LONG trades in OOS")

    # OOS for best SHORT
    if approved_short:
        name, best, _ = max(approved_short, key=lambda x: x[1]['pf'])
        trail_sl = 0.008
        trail_trig = None
        if '1.0pct' in name: trail_sl = 0.010
        elif '1.2pct' in name: trail_sl = 0.012
        elif '1.5pct' in name: trail_sl = 0.015
        elif '2.0pct' in name: trail_sl = 0.020
        if 'trig1pct' in name: trail_trig = 0.01
        corr_min = 0.3 if 'corr03' in name else 0.4

        short_oos_trades = []
        for idx in oos_indices:
            if idx + 30 + 1 >= len(df):
                continue
            reg = regimes.iloc[idx]
            if reg != 'BEAR':
                continue
            ts = df.index[idx]
            entry = float(df['close'].iloc[idx])
            btc_idx = df_btc.index.get_indexer([ts], method='nearest')[0]
            if 0 <= btc_idx < len(df_btc) and detect_btc_breakdown(df_btc, btc_idx):
                corr_val = float(df['pair_btc_corr'].iloc[idx])
                if corr_val >= corr_min:
                    out = sim_short_trailing(df, idx, entry, trail_sl, trail_trig, 30)
                    short_oos_trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                                            'setup': 'BTC_BD', 'entry': entry, 'bars': out[3]})

        print(f"\n  --- SHORT OOS ({name}) ---")
        if short_oos_trades:
            for t in sorted(short_oos_trades, key=lambda x: x['ts']):
                print(f"    {str(t['ts'])[:19]} ${t['entry']:>8.5f} {t['setup']:<6} "
                      f"{t['outcome']:<4} {t['pnl_pct']:>+7.2%} ({t['bars']}bars)")
            m = metrics(short_oos_trades, name)
            total = sum(t['pnl_pct'] for t in short_oos_trades)
            print(f"    N={m['n']} WR={m['wr']:.1%} PF={m['pf']:.2f} Total={total:+.2%}")
        else:
            print("    No SHORT trades in OOS")

    # Combined OOS
    if approved_long and approved_short and 'long_oos_trades' in dir() and 'short_oos_trades' in dir():
        combined = long_oos_trades + short_oos_trades
        if combined:
            m = metrics(combined, 'COMBINED')
            total = sum(t['pnl_pct'] for t in combined)
            eq, dd = equity_stats(combined)
            print(f"\n  --- COMBINED OOS ---")
            print(f"    LONG: {sum(t['pnl_pct'] for t in long_oos_trades):+.2%} ({len(long_oos_trades)}t)")
            print(f"    SHORT: {sum(t['pnl_pct'] for t in short_oos_trades):+.2%} ({len(short_oos_trades)}t)")
            print(f"    TOTAL: {total:+.2%} (vs DOGE B&H: {p_ret:+.1f}%)")
            print(f"    N={m['n']} WR={m['wr']:.1%} PF={m['pf']:.2f} DD={dd:.1%}")

    # ===========================================================
    # SUMMARY
    # ===========================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    n_long_ok = sum(1 for _, _, t in long_results if t == 'APROBADO')
    n_long_mg = sum(1 for _, _, t in long_results if t == 'MARGINAL')
    n_short_ok = sum(1 for _, _, t in short_results if t == 'APROBADO')
    n_short_mg = sum(1 for _, _, t in short_results if t == 'MARGINAL')

    print(f"  LONG:  {n_long_ok} APROBADO, {n_long_mg} MARGINAL de {len(long_results)} configs")
    print(f"  SHORT: {n_short_ok} APROBADO, {n_short_mg} MARGINAL de {len(short_results)} configs")

    if n_long_ok > 0 or n_short_ok > 0:
        print(f"\n  DOGE V15 tiene potencial con tight trailing!")
    else:
        print(f"\n  DOGE V15 tight trailing no paso criterios minimos.")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")
