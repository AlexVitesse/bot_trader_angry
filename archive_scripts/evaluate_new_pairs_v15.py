"""
evaluate_new_pairs_v15.py -- Screen multiple pairs for V15 trailing
===================================================================
Tests BTC-follower + tight trailing (ATR*0.20 floor 0.6%) on new pairs.
Same approach that APROBADO for ADA (+36%), SOL (+26%), DOGE (+42%).

Pairs: LINK, AVAX, DOT, NEAR, XRP, BNB, ATOM
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

# ATR-adaptive config (winner from cross-pair evaluation)
TRAIL_FACTOR = 0.20
TRAIL_FLOOR  = 0.006

PAIRS = ['LTC', 'ETC', 'BCH', 'TRX', 'UNI', 'AAVE', 'ARB', 'OP', 'FET', 'SUI']


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


def sim_long_trailing(df, entry_bar, entry_price, trail_dist, max_bars=30):
    sl_price = entry_price * (1 - trail_dist)
    peak = entry_price
    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df):
            exit_p = float(df['close'].iloc[-1])
            pnl = (exit_p - entry_price) / entry_price - 2 * COMMISSION
            return ('TP' if exit_p > entry_price else 'SL'), pnl, i
        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        if hi > peak:
            peak = hi
        sl_price = max(sl_price, peak * (1 - trail_dist))
        if lo <= sl_price:
            pnl = (sl_price - entry_price) / entry_price - 2 * COMMISSION
            return ('TP' if sl_price > entry_price else 'SL'), pnl, i
    exit_p = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (exit_p - entry_price) / entry_price - 2 * COMMISSION
    return ('TP' if exit_p > entry_price else 'SL'), pnl, max_bars


def sim_short_trailing(df, entry_bar, entry_price, trail_dist, max_bars=30):
    sl_price = entry_price * (1 + trail_dist)
    trough = entry_price
    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df):
            exit_p = float(df['close'].iloc[-1])
            pnl = (entry_price - exit_p) / entry_price - 2 * COMMISSION
            return ('TP' if exit_p < entry_price else 'SL'), pnl, i
        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        if lo < trough:
            trough = lo
        new_sl = trough * (1 + trail_dist)
        sl_price = min(sl_price, new_sl)
        if hi >= sl_price:
            pnl = (entry_price - sl_price) / entry_price - 2 * COMMISSION
            return ('TP' if sl_price < entry_price else 'SL'), pnl, i
    exit_p = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (entry_price - exit_p) / entry_price - 2 * COMMISSION
    return ('TP' if exit_p < entry_price else 'SL'), pnl, max_bars


def detect_breakout(df, idx, vol_min=1.2, bb_max=7.0, bar_max=5.0):
    if idx < 25:
        return False
    row = df.iloc[idx]
    close = row['close']
    high20 = df['high'].iloc[idx-20:idx].max()
    vol_ratio = row.get('vol_ratio', 0)
    bb_width = row.get('bb_width', 99)
    ret_1 = row.get('ret_1', 0)
    if close <= high20 or vol_ratio < vol_min or bb_width > bb_max or abs(ret_1) > bar_max:
        return False
    return True


def detect_btc_breakout(df_btc, idx):
    if idx < 25:
        return False
    close = float(df_btc.iloc[idx]['close'])
    high20 = float(df_btc['high'].iloc[max(0, idx-20):idx].max())
    vol_ratio = float(df_btc.iloc[idx].get('vol_ratio', 0))
    return close > high20 and vol_ratio >= 1.0


def detect_btc_breakdown(df_btc, idx):
    if idx < 25:
        return False
    close = float(df_btc.iloc[idx]['close'])
    low20 = float(df_btc['low'].iloc[max(0, idx-20):idx].min())
    vol_ratio = float(df_btc.iloc[idx].get('vol_ratio', 0))
    return close < low20 and vol_ratio >= 1.0


def load_pair_data(pair, df_btc):
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

    common_idx = df.index.intersection(df_btc.index)
    if len(common_idx) > 100:
        pair_ret = df.loc[common_idx, 'close'].pct_change()
        btc_ret = df_btc.loc[common_idx, 'close'].pct_change()
        roll_corr = pair_ret.rolling(168).corr(btc_ret)
        df['pair_btc_corr'] = roll_corr.reindex(df.index).ffill()
    else:
        df['pair_btc_corr'] = 0.7

    return df, regimes


def evaluate_pair(pair, df, regimes, df_btc, bb_max=7.0, corr_min=0.4):
    """Full evaluation: LONG WF + SHORT WF + OOS."""
    max_bars = 30

    # ---- LONG WF ----
    long_results = []
    long_trades = []
    for fi, (start_s, end_s) in enumerate(WF_FOLDS):
        fold_start = pd.Timestamp(start_s, tz='UTC')
        fold_end = pd.Timestamp(end_s, tz='UTC')
        fold_mask = (df.index >= fold_start) & (df.index <= fold_end)
        fold_indices = np.where(fold_mask)[0]
        fold_trades = []
        for idx in fold_indices:
            if idx + max_bars + 1 >= len(df):
                continue
            if regimes.iloc[idx] == 'BEAR':
                continue
            ts = df.index[idx]
            entry = float(df['close'].iloc[idx])
            atr_pct = float(df['atr_pct'].iloc[idx]) / 100.0
            trail_dist = max(TRAIL_FLOOR, atr_pct * TRAIL_FACTOR)
            triggered = False

            if detect_breakout(df, idx, 1.2, bb_max):
                out = sim_long_trailing(df, idx, entry, trail_dist, max_bars)
                fold_trades.append({'outcome': out[0], 'pnl_pct': out[1], 'ts': ts,
                                    'setup': 'BRK', 'fold': fi})
                triggered = True

            if not triggered:
                btc_idx = df_btc.index.get_indexer([ts], method='nearest')[0]
                if 0 <= btc_idx < len(df_btc) and detect_btc_breakout(df_btc, btc_idx):
                    corr_val = float(df['pair_btc_corr'].iloc[idx])
                    if corr_val >= corr_min:
                        out = sim_long_trailing(df, idx, entry, trail_dist, max_bars)
                        fold_trades.append({'outcome': out[0], 'pnl_pct': out[1], 'ts': ts,
                                            'setup': 'FBTC', 'fold': fi})

        m = metrics(fold_trades, '')
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        long_results.append({'n': m['n'], 'wr': m['wr'], 'pf': m['pf'], 'ok': ok})
        long_trades.extend(fold_trades)

    long_m = metrics(long_trades, '')
    long_eq, long_dd = equity_stats(long_trades)
    long_wf_ok = sum(1 for r in long_results if r['ok'])
    long_wf_data = sum(1 for r in long_results if r['n'] > 0)

    # ---- SHORT WF ----
    short_results = []
    short_trades = []
    for fi, (start_s, end_s) in enumerate(WF_FOLDS):
        fold_start = pd.Timestamp(start_s, tz='UTC')
        fold_end = pd.Timestamp(end_s, tz='UTC')
        fold_mask = (df.index >= fold_start) & (df.index <= fold_end)
        fold_indices = np.where(fold_mask)[0]
        bear_bars = sum(1 for idx in fold_indices if regimes.iloc[idx] == 'BEAR')
        fold_trades = []
        if bear_bars >= MIN_BEAR_BARS:
            for idx in fold_indices:
                if idx + max_bars + 1 >= len(df):
                    continue
                if regimes.iloc[idx] != 'BEAR':
                    continue
                ts = df.index[idx]
                entry = float(df['close'].iloc[idx])
                atr_pct = float(df['atr_pct'].iloc[idx]) / 100.0
                trail_dist = max(TRAIL_FLOOR, atr_pct * TRAIL_FACTOR)
                btc_idx = df_btc.index.get_indexer([ts], method='nearest')[0]
                if 0 <= btc_idx < len(df_btc) and detect_btc_breakdown(df_btc, btc_idx):
                    corr_val = float(df['pair_btc_corr'].iloc[idx])
                    if corr_val >= corr_min:
                        out = sim_short_trailing(df, idx, entry, trail_dist, max_bars)
                        fold_trades.append({'outcome': out[0], 'pnl_pct': out[1], 'ts': ts,
                                            'setup': 'BTC_BD', 'fold': fi})

        if bear_bars >= MIN_BEAR_BARS:
            m = metrics(fold_trades, '')
            ok = (m['n'] >= 1 and m['pf'] > 0.8)
            short_results.append({'n': m['n'], 'wr': m['wr'], 'pf': m['pf'],
                                  'ok': ok, 'bear': bear_bars})
        else:
            short_results.append({'n': 0, 'skip': True, 'bear': bear_bars})
        short_trades.extend(fold_trades)

    short_m = metrics(short_trades, '')
    short_eq, short_dd = equity_stats(short_trades)
    valid_short = [r for r in short_results if not r.get('skip', False)]
    short_wf_ok = sum(1 for r in valid_short if r['ok'])
    short_wf_total = len(valid_short)

    # ---- OOS 2026 ----
    oos_start = pd.Timestamp(OOS_START, tz='UTC')
    oos_end = pd.Timestamp(OOS_END, tz='UTC')
    oos_mask = (df.index >= oos_start) & (df.index <= oos_end)
    oos_indices = np.where(oos_mask)[0]

    oos_long = []
    oos_short = []
    for idx in oos_indices:
        if idx + max_bars + 1 >= len(df):
            continue
        ts = df.index[idx]
        entry = float(df['close'].iloc[idx])
        atr_pct = float(df['atr_pct'].iloc[idx]) / 100.0
        trail_dist = max(TRAIL_FLOOR, atr_pct * TRAIL_FACTOR)
        reg = regimes.iloc[idx]

        if reg != 'BEAR':
            triggered = False
            if detect_breakout(df, idx, 1.2, bb_max):
                out = sim_long_trailing(df, idx, entry, trail_dist, max_bars)
                oos_long.append({'pnl_pct': out[1], 'ts': ts, 'setup': 'BRK'})
                triggered = True
            if not triggered:
                btc_idx = df_btc.index.get_indexer([ts], method='nearest')[0]
                if 0 <= btc_idx < len(df_btc) and detect_btc_breakout(df_btc, btc_idx):
                    corr_val = float(df['pair_btc_corr'].iloc[idx])
                    if corr_val >= corr_min:
                        out = sim_long_trailing(df, idx, entry, trail_dist, max_bars)
                        oos_long.append({'pnl_pct': out[1], 'ts': ts, 'setup': 'FBTC'})
        else:
            btc_idx = df_btc.index.get_indexer([ts], method='nearest')[0]
            if 0 <= btc_idx < len(df_btc) and detect_btc_breakdown(df_btc, btc_idx):
                corr_val = float(df['pair_btc_corr'].iloc[idx])
                if corr_val >= corr_min:
                    out = sim_short_trailing(df, idx, entry, trail_dist, max_bars)
                    oos_short.append({'pnl_pct': out[1], 'ts': ts, 'setup': 'BTC_BD'})

    oos_l_ret = sum(t['pnl_pct'] for t in oos_long)
    oos_s_ret = sum(t['pnl_pct'] for t in oos_short)

    # B&H
    oos_df = df.loc[oos_start:oos_end]
    if len(oos_df) > 1:
        bh = (oos_df['close'].iloc[-1] / oos_df['close'].iloc[0]) - 1
    else:
        bh = 0

    return {
        'pair': pair,
        'bars': len(df),
        'atr_mean': df['atr_pct'].mean(),
        'corr_mean': df['pair_btc_corr'].mean(),
        'long_wf': f"{long_wf_ok}/{long_wf_data}",
        'long_wf_ok': long_wf_ok,
        'long_n': long_m['n'], 'long_wr': long_m['wr'], 'long_pf': long_m['pf'],
        'long_eq': long_eq, 'long_dd': long_dd,
        'short_wf': f"{short_wf_ok}/{short_wf_total}",
        'short_wf_ok': short_wf_ok, 'short_wf_total': short_wf_total,
        'short_n': short_m['n'], 'short_wr': short_m['wr'], 'short_pf': short_m['pf'],
        'short_eq': short_eq, 'short_dd': short_dd,
        'oos_long': oos_l_ret, 'oos_long_n': len(oos_long),
        'oos_short': oos_s_ret, 'oos_short_n': len(oos_short),
        'oos_total': oos_l_ret + oos_s_ret,
        'bh': bh,
    }


if __name__ == '__main__':
    print("=" * 80)
    print("V15 New Pairs Screening -- ATR*0.20 floor 0.6% trailing")
    print("=" * 80)

    df_btc = compute_features_4h(load_btc_4h().copy())
    print(f"  BTC: {len(df_btc)} bars loaded")

    results = []
    for pair in PAIRS:
        print(f"\n  --- {pair}/USDT ---")
        try:
            df, regimes = load_pair_data(pair, df_btc)
            print(f"    {len(df)} bars | ATR={df['atr_pct'].mean():.2f}% | "
                  f"BTC corr={df['pair_btc_corr'].mean():.3f}")
            reg_counts = regimes.value_counts().to_dict()
            print(f"    Regimes: {reg_counts}")

            r = evaluate_pair(pair, df, regimes, df_btc)
            results.append(r)

            long_pass = r['long_wf_ok'] >= 7
            short_threshold = max(1, int(r['short_wf_total'] * 0.6)) if r['short_wf_total'] > 0 else 1
            short_pass = r['short_wf_ok'] >= short_threshold and r['short_n'] >= 5

            print(f"    LONG:  WF {r['long_wf']} | N={r['long_n']} WR={r['long_wr']:.1%} "
                  f"PF={r['long_pf']:.2f} DD={r['long_dd']:.1%}  "
                  f"[{'OK' if long_pass else 'FAIL'}]")
            print(f"    SHORT: WF {r['short_wf']} | N={r['short_n']} WR={r['short_wr']:.1%} "
                  f"PF={r['short_pf']:.2f} DD={r['short_dd']:.1%}  "
                  f"[{'OK' if short_pass else 'FAIL'}]")
            print(f"    OOS 2026: LONG {r['oos_long']:+.2%} ({r['oos_long_n']}t) "
                  f"SHORT {r['oos_short']:+.2%} ({r['oos_short_n']}t) "
                  f"TOTAL {r['oos_total']:+.2%} (B&H: {r['bh']:+.1%})")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Pair':<6} | {'LONG WF':>8} | {'L_PF':>5} | {'L_DD':>5} | "
          f"{'SHORT WF':>9} | {'S_PF':>5} | {'S_DD':>5} | "
          f"{'OOS Tot':>8} | {'B&H':>7} | {'Verdict':>10}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*5}-+-{'-'*5}-+-"
          f"{'-'*9}-+-{'-'*5}-+-{'-'*5}-+-"
          f"{'-'*8}-+-{'-'*7}-+-{'-'*10}")

    for r in sorted(results, key=lambda x: x['oos_total'], reverse=True):
        long_pass = r['long_wf_ok'] >= 7
        st = r['short_wf_total']
        short_threshold = max(1, int(st * 0.6)) if st > 0 else 1
        short_pass = r['short_wf_ok'] >= short_threshold and r['short_n'] >= 5

        if long_pass and short_pass:
            verdict = 'APROBADO'
        elif long_pass:
            verdict = 'LONG ONLY'
        elif short_pass:
            verdict = 'SHORT ONLY'
        else:
            verdict = 'RECHAZADO'

        print(f"  {r['pair']:<6} | {r['long_wf']:>8} | {r['long_pf']:>5.2f} | "
              f"{r['long_dd']:>4.1%} | {r['short_wf']:>9} | {r['short_pf']:>5.2f} | "
              f"{r['short_dd']:>4.1%} | {r['oos_total']:>+7.2%} | "
              f"{r['bh']:>+6.1%} | {verdict:>10}")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")
