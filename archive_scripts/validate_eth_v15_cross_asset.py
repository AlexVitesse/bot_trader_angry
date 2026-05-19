"""
validate_eth_v15_cross_asset.py -- Cross-asset validation for ETH V15 Committee
================================================================================
Test the same ETH committee logic (BTC-follower + Breakout + SHORT multi-conf/BB)
on SOL and ADA (correlated L1s).

Criteria: PF > 1.0 in aggregate per pair.
If it fails, ETH committee is still valid for ETH only.

Usage:
  python validate_eth_v15_cross_asset.py
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

REGIME_DEAD_ZONE = 0.02


# ============================================================
# HELPERS
# ============================================================
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
    ep = float(df['close'].iloc[entry_bar + max_bars])
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


# ============================================================
# DETECTORS (same logic as ETH committee)
# ============================================================
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


def detect_pullback_btc(df_btc, i):
    if i < 25: return None
    row = df_btc.iloc[i]
    prev = df_btc.iloc[i-1]
    c, o = float(row['close']), float(row['open'])
    ema20 = float(row.get('ema20', 0))
    ema50 = float(row.get('ema50', 0))
    if ema20 <= 0 or ema50 <= 0 or c < ema50: return None
    dist = (c - ema20) / ema20
    if dist < -0.005 or dist > 0.015: return None
    if float(row.get('adx14', 0)) < 15: return None
    rsi = float(row.get('rsi14', 50))
    if rsi < 33 or rsi > 58: return None
    if c <= o or float(prev['close']) >= float(prev['open']): return None
    if float(row.get('vol_ratio', 1)) > 2.0: return None
    return {'setup': 'PB_BTC'}


def detect_breakout_pair(df, i, vol_min=1.3, bb_max=5.5, adx_max=32, bar_max=3.5):
    """Breakout B adapted for altcoins (ETH-like params)."""
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
    sl_pct = (entry - sl_raw) / entry
    if sl_pct < 0.005 or sl_pct > 0.06: return None
    atr_pct = float(row.get('atr_pct', 2.5))
    sl_pct = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    tp_pct = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    return {'direction': 'LONG', 'setup': 'BRK_PAIR',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}


def detect_multi_conf(df, i):
    if i < 25: return None
    row = df.iloc[i]
    c_val, o_val = float(row['close']), float(row['open'])
    if c_val >= o_val: return None
    if float(row.get('rsi14', 50)) < 60: return None
    if float(row.get('bb_pct', 0.5)) < 0.75: return None
    if float(row.get('vol_ratio', 1)) < 1.0: return None
    entry = c_val
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    sl = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    return {'direction': 'SHORT', 'setup': 'MULTI_CONF',
            'entry': entry, 'tp_pct': tp, 'sl_pct': sl}


def detect_bb_upper(df, i):
    if i < 25: return None
    row = df.iloc[i]
    c_val, o_val = float(row['close']), float(row['open'])
    if c_val >= o_val: return None
    if float(row.get('bb_pct', 0.5)) < 0.90: return None
    entry = c_val
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    sl = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    return {'direction': 'SHORT', 'setup': 'BB_UPPER',
            'entry': entry, 'tp_pct': tp, 'sl_pct': sl}


def detect_short_combined(df, i):
    t = detect_multi_conf(df, i)
    if t is not None:
        return t
    return detect_bb_upper(df, i)


# ============================================================
# COMMITTEE (LONG + SHORT)
# ============================================================
def run_committee(df_pair, df_btc, regimes_pair, regimes_btc, corr_20,
                  start_s, end_s):
    trades = []
    for i in range(30, len(df_pair)):
        ts = df_pair.index[i]
        if ts < pd.Timestamp(start_s, tz='UTC') or ts > pd.Timestamp(end_s, tz='UTC'):
            continue
        if i + 18 >= len(df_pair):
            continue

        regime = regimes_pair.iloc[i]
        trade = None

        if regime in ('BULL', 'RANGE'):
            # LONG: BTC follower + standalone breakout
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
                            row = df_pair.iloc[i]
                            entry = float(row['close'])
                            atr_pct = float(row.get('atr_pct', 2.5))
                            sl_pct = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
                            tp_pct = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
                            trade = {'direction': 'LONG',
                                     'setup': f"FOLLOW_{btc_signal['setup']}",
                                     'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}
            if trade is None:
                trade = detect_breakout_pair(df_pair, i)

        elif regime == 'BEAR':
            trade = detect_short_combined(df_pair, i)

        if trade is None:
            continue

        d = trade.get('direction', 'LONG')
        if d == 'LONG':
            out = sim_trade_fixed(df_pair, i, trade['entry'],
                                  trade['tp_pct'], trade['sl_pct'], max_bars=18)
        else:
            out = sim_short(df_pair, i, trade['entry'],
                           trade['tp_pct'], trade['sl_pct'], max_bars=16)

        trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                      'setup': trade['setup'], 'direction': d, 'entry': trade['entry']})
    return trades


def wf_committee(df_pair, df_btc, regimes_pair, regimes_btc, corr_20, pair_name):
    results = []
    all_trades = []
    for start_s, end_s in WF_FOLDS:
        period = f"{start_s[:7]}/{end_s[5:7]}"
        trades = run_committee(df_pair, df_btc, regimes_pair, regimes_btc,
                              corr_20, start_s, end_s)
        m = metrics(trades, period)
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                        'pf': m['pf'], 'ok': ok})
        all_trades.extend(trades)
    folds_ok = sum(1 for r in results if r['ok'])
    return {'folds': results, 'folds_ok': folds_ok, 'all_trades': all_trades}


# ============================================================
# PREPARE DATA FOR A PAIR
# ============================================================
def prepare_pair(pair_name, df_btc, btc_macro):
    print(f"\n  Loading {pair_name}...")
    df_raw = load_pair_4h(pair_name)
    df = compute_features_4h(df_raw.copy())
    df = add_extra_features(df)

    # Daily macro: try loading 1d, fallback to resampling 4h
    try:
        from v15_framework import load_pair_1d
        pair_1d = load_pair_1d(pair_name)
    except (FileNotFoundError, Exception):
        print(f"    No daily data for {pair_name}, resampling 4h -> 1d")
        pair_1d = df_raw.resample('1D').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()

    pair_macro = compute_macro_daily(pair_1d)
    df = merge_daily_to_4h(df, pair_macro)

    regimes = df.apply(lambda r: detect_regime(r), axis=1)

    # Correlation with BTC
    pair_ret = df['close'].pct_change()
    btc_close_a = df_btc['close'].reindex(df.index, method='ffill')
    btc_ret = btc_close_a.pct_change()
    corr_20 = pair_ret.rolling(20).corr(btc_ret)

    print(f"    {pair_name}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
    reg_counts = regimes.value_counts().to_dict()
    print(f"    Regimes: {reg_counts}")

    return df, regimes, corr_20


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("ETH V15 Committee -- Cross-Asset Validation (SOL, ADA)")
    print("=" * 70)

    # Load BTC (shared)
    print("\nLoading BTC...")
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

    # Also run ETH as reference
    PAIRS = ['ETH', 'SOL', 'ADA']
    results_all = {}

    for pair_name in PAIRS:
        print(f"\n{'='*70}")
        print(f"  {pair_name}/USDT Committee")
        print(f"{'='*70}")

        try:
            df_pair, regimes_pair, corr_20 = prepare_pair(pair_name, df_btc, btc_macro)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        wf = wf_committee(df_pair, df_btc, regimes_pair, regimes_btc, corr_20, pair_name)
        m = metrics(wf['all_trades'], f'{pair_name} OOS')
        eq, dd = equity_stats(wf['all_trades'])

        # Print folds
        for r in wf['folds']:
            wr_s = f"{r['wr']:.1%}" if r['n'] > 0 else "n/a"
            pf_s = f"{r['pf']:.2f}" if r['n'] > 0 else "n/a"
            ok_s = "+" if r['ok'] else "-"
            print(f"    {r['period']}: N={r['n']:>3} WR={wr_s:>6} PF={pf_s:>6} {ok_s}")

        passed = wf['folds_ok'] >= 7 and m['pf'] >= 1.2
        marginal = wf['folds_ok'] >= 6 and m['pf'] >= 1.0
        tag = "APROBADO" if passed else ("MARGINAL" if marginal else "RECHAZADO")

        print(f"\n    WF: {wf['folds_ok']}/12 | N={m['n']} WR={m['wr']:.1%} PF={m['pf']:.2f} "
              f"$1K -> ${1000*eq:.0f} DD={dd:.1%} -> {tag}")

        # Breakdown by setup
        setups = {}
        for t in wf['all_trades']:
            s = t.get('setup', '?')
            if s not in setups: setups[s] = []
            setups[s].append(t)
        for s_name, ts in sorted(setups.items()):
            wins = sum(1 for t in ts if t['pnl_pct'] > 0)
            wr = wins / len(ts) if ts else 0
            print(f"      {s_name}: N={len(ts)} WR={wr:.1%}")

        results_all[pair_name] = {
            'folds_ok': wf['folds_ok'], 'n': m['n'],
            'wr': m['wr'], 'pf': m['pf'],
            'equity': eq, 'dd': dd, 'tag': tag,
        }

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("CROSS-ASSET SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Pair':<8} | {'WF':>5} | {'N':>4} | {'WR':>6} | {'PF':>6} | {'$1K->':>7} | {'DD':>6} | Veredicto")
    print(f"  {'-'*75}")
    for pair_name, r in results_all.items():
        mark = "**" if r['tag'] == "APROBADO" else "  "
        print(f"  {mark}{pair_name:<6} | {r['folds_ok']:>2}/12 | {r['n']:>4} | "
              f"{r['wr']:.1%} | {r['pf']:.2f} | ${1000*r['equity']:>6.0f} | "
              f"{r['dd']:.1%} | {r['tag']}")

    # Final verdict
    n_positive = sum(1 for r in results_all.values() if r['pf'] > 1.0)
    print(f"\n  Pairs with PF > 1.0: {n_positive}/{len(results_all)}")
    if n_positive == len(results_all):
        print("  -> Cross-asset PASSED: committee logic generalizes")
    else:
        failed = [p for p, r in results_all.items() if r['pf'] <= 1.0]
        print(f"  -> Cross-asset PARTIAL: failed for {', '.join(failed)}")
        print("  -> ETH committee remains valid for ETH only")
    print("=" * 70)
