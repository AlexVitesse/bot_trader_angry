"""
evaluate_short_alt_v15.py -- SHORT strategies for ADA & SOL in BEAR regime
==========================================================================
Context: LONG with tight trailing passes all criteria for both pairs.
Now we test SHORT in BEAR to complement, similar to ETH's Multi-conf + BB_UPPER.

Strategies tested:
  1. Multi-conf SHORT (RSI>60 + BB>0.75 + bearish + vol) — ETH's winner
  2. BB_UPPER SHORT (BB>0.90 + bearish)
  3. BTC breakdown follower (SHORT when BTC breaks down)
  4. Combinations of above

Exit methods:
  A. Fixed TP/SL (ATR-based, like ETH)
  B. Fixed small TP/SL (TP3/SL1.5)
  C. Trailing stop (tight 0.8%, the LONG winner approach)

Walk-forward: BEAR folds only (>=30 BEAR bars per fold). Threshold: >=60% of valid folds.

Usage:
  python evaluate_short_alt_v15.py
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
    metrics, WF_FOLDS, COMMISSION,
)

# ==============================================================
# CONSTANTS
# ==============================================================
REGIME_DEAD_ZONE = 0.02
OOS_START = '2026-01-01'
OOS_END = '2026-03-15'
TRAIN_CUTOFF = '2025-12-31'
MIN_BEAR_BARS = 30

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
# SHORT TRADE SIMULATIONS
# ==============================================================
def sim_short(df, entry_bar, entry_price, tp_pct, sl_pct, max_bars=16):
    """Fixed TP/SL SHORT trade. TP=price drops, SL=price rises."""
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
    ep = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (entry_price - ep) / entry_price - 2 * COMMISSION
    return ('TP' if ep < entry_price else 'SL'), ep, pnl, max_bars


def sim_short_trailing(df, entry_bar, entry_price, sl_pct,
                       trail_trigger_pct=None, max_bars=30):
    """
    Trailing stop for SHORT direction.
    Tracks price LOWS (trough). SL moves down with new lows.
    """
    sl_price = entry_price * (1 + sl_pct)  # SL above entry
    trough = entry_price
    trailing_active = (trail_trigger_pct is None)

    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df):
            exit_p = float(df['close'].iloc[-1])
            pnl = (entry_price - exit_p) / entry_price - 2 * COMMISSION
            outcome = 'TP' if exit_p < entry_price else 'SL'
            return outcome, exit_p, pnl, i

        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])

        # Track new lows
        if lo < trough:
            trough = lo

        # Activate trailing when profit threshold reached
        if not trailing_active and trail_trigger_pct is not None:
            if lo <= entry_price * (1 - trail_trigger_pct):
                trailing_active = True

        # Move SL down (tighter) as price drops
        if trailing_active:
            new_sl = trough * (1 + sl_pct)
            sl_price = min(sl_price, new_sl)

        # Check SL hit (price rises above trailing SL)
        if hi >= sl_price:
            pnl = (entry_price - sl_price) / entry_price - 2 * COMMISSION
            outcome = 'TP' if sl_price < entry_price else 'SL'
            return outcome, sl_price, pnl, i

    # Timeout
    exit_p = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (entry_price - exit_p) / entry_price - 2 * COMMISSION
    outcome = 'TP' if exit_p < entry_price else 'SL'
    return outcome, exit_p, pnl, max_bars


# ==============================================================
# SHORT SIGNAL DETECTORS
# ==============================================================
def detect_multi_conf(df, i):
    """Multi-confluence SHORT: RSI>60 + BB>0.75 + bearish candle + vol."""
    if i < 25:
        return None
    row = df.iloc[i]
    c_val, o_val = float(row['close']), float(row['open'])
    if c_val >= o_val:  # Must be bearish candle
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
    return {'setup': 'MULTI_CONF', 'entry': entry, 'tp_pct': tp, 'sl_pct': sl}


def detect_bb_upper(df, i):
    """BB upper SHORT: BB_pct > 0.90 + bearish candle."""
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
    return {'setup': 'BB_UPPER', 'entry': entry, 'tp_pct': tp, 'sl_pct': sl}


def detect_multi_conf_relaxed(df, i):
    """Relaxed multi-conf: RSI>55 + BB>0.70 + bearish."""
    if i < 25:
        return None
    row = df.iloc[i]
    c_val, o_val = float(row['close']), float(row['open'])
    if c_val >= o_val:
        return None
    if float(row.get('rsi14', 50)) < 55:
        return None
    if float(row.get('bb_pct', 0.5)) < 0.70:
        return None
    entry = c_val
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    sl = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    return {'setup': 'MULTI_RLX', 'entry': entry, 'tp_pct': tp, 'sl_pct': sl}


def detect_btc_breakdown(df_btc, i):
    """BTC breaks below 20-bar low — follower SHORT signal."""
    if i < 25:
        return False
    row = df_btc.iloc[i]
    close = float(row['close'])
    low20 = df_btc['low'].iloc[max(0, i-20):i].min()
    vol_ratio = float(row.get('vol_ratio', 0))
    if close >= low20 or vol_ratio < 1.0:
        return False
    return True


def detect_rsi_overbought(df, i):
    """RSI overbought SHORT: RSI>70 + bearish in BEAR."""
    if i < 25:
        return None
    row = df.iloc[i]
    c_val, o_val = float(row['close']), float(row['open'])
    if c_val >= o_val:
        return None
    if float(row.get('rsi14', 50)) < 70:
        return None
    entry = c_val
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    sl = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    return {'setup': 'RSI_OB', 'entry': entry, 'tp_pct': tp, 'sl_pct': sl}


# ==============================================================
# DATA LOADING (reuse from evaluate_alt_v15_solutions.py)
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
    bear_count = reg_counts.get('BEAR', 0)
    print(f"    BEAR bars: {bear_count} ({bear_count/len(df)*100:.1f}%)")

    return df, regimes, df_btc


# ==============================================================
# WALK-FORWARD: SHORT in BEAR only
# ==============================================================
def get_valid_bear_folds(df, regimes):
    """Determine which folds have enough BEAR bars (>=30)."""
    valid = []
    for start_s, end_s in WF_FOLDS:
        mask = (df.index >= pd.Timestamp(start_s, tz='UTC')) & \
               (df.index <= pd.Timestamp(end_s, tz='UTC'))
        bear_bars = (regimes[mask] == 'BEAR').sum()
        valid.append(bear_bars >= MIN_BEAR_BARS)
    return valid


def wf_short_fixed(df, regimes, valid_folds, detect_fn, name,
                   max_bars=16, df_btc=None, btc_corr_min=0.4):
    """Walk-forward SHORT with fixed TP/SL (ATR-based from detector)."""
    results = []
    all_trades = []

    for fi, (start_s, end_s) in enumerate(WF_FOLDS):
        period = f"{start_s[:7]}/{end_s[5:7]}"
        if not valid_folds[fi]:
            results.append({'period': period, 'n': 0, 'ok': False, 'skip': True})
            continue

        trades = []
        for i in range(30, len(df)):
            ts = df.index[i]
            if ts < pd.Timestamp(start_s, tz='UTC') or ts > pd.Timestamp(end_s, tz='UTC'):
                continue
            if regimes.iloc[i] != 'BEAR':
                continue
            if i + max_bars + 2 >= len(df):
                continue

            # BTC breakdown follower
            if detect_fn == 'BTC_BREAKDOWN':
                btc_idx = df_btc.index.get_indexer([ts], method='nearest')[0]
                if btc_idx < 0 or btc_idx >= len(df_btc):
                    continue
                if not detect_btc_breakdown(df_btc, btc_idx):
                    continue
                corr_val = float(df['pair_btc_corr'].iloc[i])
                if corr_val < btc_corr_min:
                    continue
                entry = float(df['close'].iloc[i])
                atr_pct = float(df.iloc[i].get('atr_pct', 2.5))
                tp_pct = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
                sl_pct = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
                out = sim_short(df, i, entry, tp_pct, sl_pct, max_bars=max_bars)
                trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                               'setup': 'BTC_BRK', 'direction': 'SHORT', 'entry': entry})
            else:
                trade = detect_fn(df, i)
                if trade is None:
                    continue
                mb = trade.get('max_bars', max_bars)
                out = sim_short(df, i, trade['entry'], trade['tp_pct'], trade['sl_pct'], max_bars=mb)
                trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                               'setup': trade.get('setup', '?'), 'direction': 'SHORT',
                               'entry': trade['entry']})

        m = metrics(trades, period)
        ok = (m['n'] >= 2 and m['wr'] > 0.40 and m['pf'] > 0.9)
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                        'pf': m['pf'], 'ok': ok, 'skip': False})
        all_trades.extend(trades)

    m_all = metrics(all_trades, name)
    eq, dd = equity_stats(all_trades)
    folds_ok = sum(1 for r in results if r['ok'])
    folds_valid = sum(1 for r in results if not r.get('skip', False))
    return {'name': name, 'folds_ok': folds_ok, 'folds_valid': folds_valid,
            'trades': all_trades, 'n': m_all['n'], 'wr': m_all['wr'],
            'pf': m_all['pf'], 'eq': eq, 'dd': dd, 'fold_results': results}


def wf_short_small_fixed(df, regimes, valid_folds, detect_fn, name,
                         tp_pct, sl_pct, max_bars=18, df_btc=None, btc_corr_min=0.4):
    """Walk-forward SHORT with small fixed TP/SL (overriding detector's ATR-based)."""
    results = []
    all_trades = []

    for fi, (start_s, end_s) in enumerate(WF_FOLDS):
        period = f"{start_s[:7]}/{end_s[5:7]}"
        if not valid_folds[fi]:
            results.append({'period': period, 'n': 0, 'ok': False, 'skip': True})
            continue

        trades = []
        for i in range(30, len(df)):
            ts = df.index[i]
            if ts < pd.Timestamp(start_s, tz='UTC') or ts > pd.Timestamp(end_s, tz='UTC'):
                continue
            if regimes.iloc[i] != 'BEAR':
                continue
            if i + max_bars + 2 >= len(df):
                continue

            if detect_fn == 'BTC_BREAKDOWN':
                btc_idx = df_btc.index.get_indexer([ts], method='nearest')[0]
                if btc_idx < 0 or btc_idx >= len(df_btc):
                    continue
                if not detect_btc_breakdown(df_btc, btc_idx):
                    continue
                corr_val = float(df['pair_btc_corr'].iloc[i])
                if corr_val < btc_corr_min:
                    continue
                entry = float(df['close'].iloc[i])
                out = sim_short(df, i, entry, tp_pct, sl_pct, max_bars=max_bars)
                trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                               'setup': 'BTC_BRK', 'direction': 'SHORT', 'entry': entry})
            else:
                trade = detect_fn(df, i)
                if trade is None:
                    continue
                entry = trade['entry']
                out = sim_short(df, i, entry, tp_pct, sl_pct, max_bars=max_bars)
                trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                               'setup': trade.get('setup', '?'), 'direction': 'SHORT',
                               'entry': entry})

        m = metrics(trades, period)
        ok = (m['n'] >= 2 and m['wr'] > 0.40 and m['pf'] > 0.9)
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                        'pf': m['pf'], 'ok': ok, 'skip': False})
        all_trades.extend(trades)

    m_all = metrics(all_trades, name)
    eq, dd = equity_stats(all_trades)
    folds_ok = sum(1 for r in results if r['ok'])
    folds_valid = sum(1 for r in results if not r.get('skip', False))
    return {'name': name, 'folds_ok': folds_ok, 'folds_valid': folds_valid,
            'trades': all_trades, 'n': m_all['n'], 'wr': m_all['wr'],
            'pf': m_all['pf'], 'eq': eq, 'dd': dd}


def wf_short_trailing(df, regimes, valid_folds, detect_fn, name,
                      sl_pct=0.008, trail_trigger_pct=None, max_bars=30,
                      df_btc=None, btc_corr_min=0.4):
    """Walk-forward SHORT with trailing stop."""
    results = []
    all_trades = []

    for fi, (start_s, end_s) in enumerate(WF_FOLDS):
        period = f"{start_s[:7]}/{end_s[5:7]}"
        if not valid_folds[fi]:
            results.append({'period': period, 'n': 0, 'ok': False, 'skip': True})
            continue

        trades = []
        for i in range(30, len(df)):
            ts = df.index[i]
            if ts < pd.Timestamp(start_s, tz='UTC') or ts > pd.Timestamp(end_s, tz='UTC'):
                continue
            if regimes.iloc[i] != 'BEAR':
                continue
            if i + max_bars + 2 >= len(df):
                continue

            if detect_fn == 'BTC_BREAKDOWN':
                btc_idx = df_btc.index.get_indexer([ts], method='nearest')[0]
                if btc_idx < 0 or btc_idx >= len(df_btc):
                    continue
                if not detect_btc_breakdown(df_btc, btc_idx):
                    continue
                corr_val = float(df['pair_btc_corr'].iloc[i])
                if corr_val < btc_corr_min:
                    continue
                entry = float(df['close'].iloc[i])
            else:
                trade = detect_fn(df, i)
                if trade is None:
                    continue
                entry = trade['entry']

            out = sim_short_trailing(df, i, entry, sl_pct,
                                    trail_trigger_pct=trail_trigger_pct,
                                    max_bars=max_bars)
            trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                           'setup': detect_fn if isinstance(detect_fn, str) else 'RULES',
                           'direction': 'SHORT', 'entry': entry})

        m = metrics(trades, period)
        ok = (m['n'] >= 2 and m['wr'] > 0.40 and m['pf'] > 0.9)
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                        'pf': m['pf'], 'ok': ok, 'skip': False})
        all_trades.extend(trades)

    m_all = metrics(all_trades, name)
    eq, dd = equity_stats(all_trades)
    folds_ok = sum(1 for r in results if r['ok'])
    folds_valid = sum(1 for r in results if not r.get('skip', False))
    return {'name': name, 'folds_ok': folds_ok, 'folds_valid': folds_valid,
            'trades': all_trades, 'n': m_all['n'], 'wr': m_all['wr'],
            'pf': m_all['pf'], 'eq': eq, 'dd': dd}


# ==============================================================
# DISPLAY HELPERS
# ==============================================================
def fmt_result(r):
    """Format a single result line."""
    eq_usd = r['eq'] * 1000
    status = 'RECHAZADO'
    prefix = '  '
    threshold = 0.60
    if r['folds_valid'] > 0:
        ratio = r['folds_ok'] / r['folds_valid']
        if ratio >= threshold:
            status = 'APROBADO'
            prefix = '**'
        elif ratio >= 0.50:
            status = 'MARGINAL'
            prefix = '* '
    line = (f"{prefix}{r['name']:<42s} WF {r['folds_ok']:2d}/{r['folds_valid']:2d} | "
            f"N={r['n']:4d} WR={r['wr']*100:.1f}% PF={r['pf']:.2f} "
            f"$1K->${eq_usd:.0f} DD={r['dd']*100:.1f}% -> {status}")
    return line


# ==============================================================
# COMBO HELPERS
# ==============================================================
def make_combo_detect(detectors):
    """Combine multiple detectors; first match wins."""
    def detect(df, i):
        for fn in detectors:
            t = fn(df, i)
            if t is not None:
                return t
        return None
    return detect


# ==============================================================
# MAIN EVALUATION
# ==============================================================
def evaluate_pair_short(pair):
    print(f"\n{'='*70}")
    print(f"  {pair}/USDT -- SHORT in BEAR Evaluation")
    print(f"{'='*70}")

    df, regimes, df_btc = load_pair_data(pair)

    # Valid BEAR folds
    valid_folds = get_valid_bear_folds(df, regimes)
    n_valid = sum(valid_folds)
    print(f"    Valid BEAR folds (>={MIN_BEAR_BARS} bars): {n_valid}/12")
    for fi, (s, e) in enumerate(WF_FOLDS):
        mask = (df.index >= pd.Timestamp(s, tz='UTC')) & (df.index <= pd.Timestamp(e, tz='UTC'))
        bear_n = (regimes[mask] == 'BEAR').sum()
        v = 'OK' if valid_folds[fi] else 'SKIP'
        print(f"      {s[:7]}: BEAR={bear_n:4d} [{v}]")

    threshold_folds = max(1, int(n_valid * 0.60))
    print(f"    Threshold: >= {threshold_folds}/{n_valid} folds (60%)")

    all_results = []

    # ----------------------------------------------------------
    # PART 1: Fixed ATR-based TP/SL (like ETH)
    # ----------------------------------------------------------
    print(f"\n--- PART 1: Fixed ATR-based TP/SL (SHORT in BEAR) ---")

    detectors = {
        'MULTI_CONF':     detect_multi_conf,
        'BB_UPPER':       detect_bb_upper,
        'MULTI_RLX':      detect_multi_conf_relaxed,
        'RSI_OB':         detect_rsi_overbought,
        'MULTI+BB':       make_combo_detect([detect_multi_conf, detect_bb_upper]),
        'MULTI_RLX+BB':   make_combo_detect([detect_multi_conf_relaxed, detect_bb_upper]),
        'BTC_BREAKDOWN':  'BTC_BREAKDOWN',
    }

    for det_name, det_fn in detectors.items():
        r = wf_short_fixed(df, regimes, valid_folds, det_fn,
                           f"SHORT_{det_name}_ATR",
                           max_bars=16, df_btc=df_btc)
        all_results.append(r)
        print(f"  {fmt_result(r)}")

    # ----------------------------------------------------------
    # PART 2: Small fixed TP/SL (TP3/SL1.5, TP2.5/SL1.5)
    # ----------------------------------------------------------
    print(f"\n--- PART 2: Small Fixed TP/SL (SHORT in BEAR) ---")

    small_tpsl = [
        ('TP3_SL1.5',   0.03,  0.015, 18),
        ('TP2.5_SL1.5', 0.025, 0.015, 18),
        ('TP2_SL1.5',   0.02,  0.015, 18),
    ]

    best_detectors = ['MULTI_CONF', 'BB_UPPER', 'MULTI+BB', 'BTC_BREAKDOWN']

    for det_name in best_detectors:
        det_fn = detectors[det_name]
        for tpsl_name, tp, sl, mb in small_tpsl:
            r = wf_short_small_fixed(df, regimes, valid_folds, det_fn,
                                     f"SHORT_{det_name}_{tpsl_name}",
                                     tp_pct=tp, sl_pct=sl, max_bars=mb,
                                     df_btc=df_btc)
            all_results.append(r)
            print(f"  {fmt_result(r)}")

    # ----------------------------------------------------------
    # PART 3: Trailing stop (tight 0.8% + others)
    # ----------------------------------------------------------
    print(f"\n--- PART 3: Trailing Stop SHORT ---")

    trail_configs = [
        ('TRAIL_tight',  0.008, None),
        ('TRAIL_1pct',   0.010, None),
        ('TRAIL_1.5pct', 0.015, None),
        ('TRAIL_50pct',  0.015, 0.015),  # trigger at 1.5% profit
    ]

    for det_name in best_detectors:
        det_fn = detectors[det_name]
        for trail_name, t_sl, t_trigger in trail_configs:
            r = wf_short_trailing(df, regimes, valid_folds, det_fn,
                                  f"SHORT_{det_name}_{trail_name}",
                                  sl_pct=t_sl, trail_trigger_pct=t_trigger,
                                  max_bars=30, df_btc=df_btc)
            all_results.append(r)
            print(f"  {fmt_result(r)}")

    # ----------------------------------------------------------
    # PART 4: Sizing-adjusted DD
    # ----------------------------------------------------------
    print(f"\n--- PART 4: Sizing-Adjusted DD ---")

    # Filter configs with reasonable results
    sized_candidates = [r for r in all_results if r['n'] >= 10]
    if sized_candidates:
        print(f"  {'Config':<44s} |    WF | 1.0x DD | 0.5x DD | 0.4x DD | 0.3x DD | 0.4x $1K->")
        print(f"  {'-'*105}")
        for r in sorted(sized_candidates, key=lambda x: -x['folds_ok']):
            dd_05 = compute_sized_equity(r['trades'], 0.5)[1]
            dd_04 = compute_sized_equity(r['trades'], 0.4)[1]
            dd_03 = compute_sized_equity(r['trades'], 0.3)[1]
            eq_04 = compute_sized_equity(r['trades'], 0.4)[0] * 1000
            flag_05 = ' <25%' if dd_05 < 0.25 else ''
            flag_04 = ' <25%' if dd_04 < 0.25 else ''
            flag_03 = ' <25%' if dd_03 < 0.25 else ''
            print(f"  {r['name']:<44s} | {r['folds_ok']:2d}/{r['folds_valid']:2d} | "
                  f"{r['dd']*100:5.1f}% | {dd_05*100:5.1f}%{flag_05} | "
                  f"{dd_04*100:5.1f}%{flag_04} | {dd_03*100:5.1f}%{flag_03} | "
                  f"${eq_04:.0f}")

    # ----------------------------------------------------------
    # PART 5: Summary ranked by PF
    # ----------------------------------------------------------
    print(f"\n--- PART 5: Summary ({pair}) ---")
    ranked = sorted(all_results, key=lambda x: -x['pf'])
    print(f"\n  {'Config':<44s} |    WF |    N |     WR |     PF |   $1K-> |     DD | Status")
    print(f"  {'-'*105}")
    for r in ranked[:20]:
        eq_usd = r['eq'] * 1000
        if r['folds_valid'] == 0:
            status = 'NO_DATA'
        elif r['folds_ok'] / r['folds_valid'] >= 0.60:
            status = 'APROBADO'
        elif r['folds_ok'] / r['folds_valid'] >= 0.50:
            status = 'MARGINAL'
        else:
            status = 'RECHAZADO'
        prefix = '**' if status == 'APROBADO' else ('* ' if status == 'MARGINAL' else '  ')
        print(f"  {prefix}{r['name']:<42s} | {r['folds_ok']:2d}/{r['folds_valid']:2d} | "
              f"{r['n']:4d} | {r['wr']*100:5.1f}% | {r['pf']:5.2f} | "
              f"${eq_usd:7.0f} | {r['dd']*100:5.1f}% | {status}")

    aprobado = [r for r in all_results if r['folds_valid'] > 0 and
                r['folds_ok'] / r['folds_valid'] >= 0.60]
    marginal = [r for r in all_results if r['folds_valid'] > 0 and
                0.50 <= r['folds_ok'] / r['folds_valid'] < 0.60]
    print(f"\n  APROBADO: {len(aprobado)} | MARGINAL: {len(marginal)}")

    # DD < 25% list
    dd25_configs = []
    for r in all_results:
        if r['n'] < 5:
            continue
        for sz in [1.0, 0.8, 0.5, 0.4, 0.3]:
            eq_s, dd_s = compute_sized_equity(r['trades'], sz)
            if dd_s < 0.25:
                years = max(1, (df.index[-1] - df.index[0]).days / 365.25)
                ann = ((eq_s ** (1/years)) - 1) * 100 * sz
                dd25_configs.append({
                    'name': r['name'], 'wf': f"{r['folds_ok']}/{r['folds_valid']}",
                    'sizing': sz, 'dd': dd_s, 'ann': ann, 'eq': eq_s * 1000
                })
                break

    if dd25_configs:
        print(f"\n  Configs with DD < 25% (sized):")
        print(f"  {'Config':<44s} |    WF | Sizing |     DD |   Ann% |   $1K->")
        print(f"  {'-'*90}")
        for c in sorted(dd25_configs, key=lambda x: -x['ann'])[:10]:
            print(f"  {c['name']:<44s} | {c['wf']:>5s} | {c['sizing']:5.1f}x | "
                  f"{c['dd']*100:5.1f}% | {c['ann']:5.1f}% | ${c['eq']:7.0f}")

    # ----------------------------------------------------------
    # PART 6: OOS 2026
    # ----------------------------------------------------------
    print(f"\n--- PART 6: OOS 2026 ({pair}) ---")

    oos_mask = (df.index >= pd.Timestamp(OOS_START, tz='UTC')) & \
               (df.index <= pd.Timestamp(OOS_END, tz='UTC'))
    if oos_mask.sum() == 0:
        print("  No OOS data available")
        return all_results

    oos_start_price = float(df.loc[oos_mask, 'close'].iloc[0])
    oos_end_price = float(df.loc[oos_mask, 'close'].iloc[-1])
    oos_bh = (oos_end_price - oos_start_price) / oos_start_price * 100
    print(f"  {pair}: ${oos_start_price:.4f} -> ${oos_end_price:.4f} ({oos_bh:+.1f}%)")
    oos_regimes = regimes[oos_mask].value_counts().to_dict()
    print(f"  Regimes 2026: {oos_regimes}")

    # Test top configs in OOS
    top_configs = sorted(aprobado + marginal, key=lambda x: -x['pf'])[:5]
    if not top_configs:
        top_configs = sorted(all_results, key=lambda x: -x['pf'])[:3]

    for r in top_configs:
        name = r['name']
        print(f"\n  --- {name} (OOS) ---")

        oos_trades = []
        oos_indices = np.where(oos_mask)[0]

        for i in oos_indices:
            if regimes.iloc[i] != 'BEAR':
                continue
            if i + 30 + 2 >= len(df):
                continue

            # Determine which detector to use from the name
            entry = None
            if 'BTC_BREAKDOWN' in name:
                ts = df.index[i]
                btc_idx = df_btc.index.get_indexer([ts], method='nearest')[0]
                if btc_idx < 0 or btc_idx >= len(df_btc):
                    continue
                if not detect_btc_breakdown(df_btc, btc_idx):
                    continue
                corr_val = float(df['pair_btc_corr'].iloc[i])
                if corr_val < 0.4:
                    continue
                entry = float(df['close'].iloc[i])
                setup = 'BTC_BRK'
            else:
                # Parse detector from name
                if 'MULTI_RLX+BB' in name:
                    det = make_combo_detect([detect_multi_conf_relaxed, detect_bb_upper])
                elif 'MULTI+BB' in name:
                    det = make_combo_detect([detect_multi_conf, detect_bb_upper])
                elif 'MULTI_RLX' in name:
                    det = detect_multi_conf_relaxed
                elif 'MULTI_CONF' in name:
                    det = detect_multi_conf
                elif 'BB_UPPER' in name:
                    det = detect_bb_upper
                elif 'RSI_OB' in name:
                    det = detect_rsi_overbought
                else:
                    continue

                trade = det(df, i)
                if trade is None:
                    continue
                entry = trade['entry']
                setup = trade['setup']

            # Determine exit method from name
            if 'TRAIL_tight' in name:
                out = sim_short_trailing(df, i, entry, 0.008, None, 30)
            elif 'TRAIL_1pct' in name:
                out = sim_short_trailing(df, i, entry, 0.010, None, 30)
            elif 'TRAIL_1.5pct' in name:
                out = sim_short_trailing(df, i, entry, 0.015, None, 30)
            elif 'TRAIL_50pct' in name:
                out = sim_short_trailing(df, i, entry, 0.015, 0.015, 30)
            elif 'TP3_SL1.5' in name:
                out = sim_short(df, i, entry, 0.03, 0.015, 18)
            elif 'TP2.5_SL1.5' in name:
                out = sim_short(df, i, entry, 0.025, 0.015, 18)
            elif 'TP2_SL1.5' in name:
                out = sim_short(df, i, entry, 0.02, 0.015, 18)
            else:
                # ATR-based
                atr_pct = float(df.iloc[i].get('atr_pct', 2.5))
                tp_pct = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
                sl_pct = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
                out = sim_short(df, i, entry, tp_pct, sl_pct, 16)

            oos_trades.append({
                'outcome': out[0], 'pnl_pct': out[2], 'ts': df.index[i],
                'setup': setup, 'direction': 'SHORT', 'entry': entry
            })

        if not oos_trades:
            print("    No trades in OOS")
        else:
            for t in sorted(oos_trades, key=lambda x: x['ts']):
                print(f"    {t['ts']} ${t['entry']:10.4f} {t['outcome']:<5s} {t['pnl_pct']*100:+.2f}%")
            m = metrics(oos_trades, '')
            eq_oos, dd_oos = equity_stats(oos_trades)
            total_pnl = sum(t['pnl_pct'] for t in oos_trades) * 100
            print(f"    N={m['n']} WR={m['wr']*100:.1f}% PF={m['pf']:.2f} "
                  f"Total={total_pnl:+.2f}% vs {pair} B&H: {oos_bh:+.1f}%")

    return all_results


# ==============================================================
# MAIN
# ==============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("ADA & SOL -- SHORT in BEAR Evaluation")
    print("Strategies: Multi-conf, BB_UPPER, BTC-breakdown, combos")
    print("Exits: ATR-based, Small TP/SL, Trailing stop")
    print("=" * 70)

    for pair in ['ADA', 'SOL']:
        evaluate_pair_short(pair)

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")
