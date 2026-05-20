"""
evaluate_ada_v15.py -- ADA/USDT V15 Strategy Evaluation
=========================================================
Context:
  - V14 ADA ensemble: 11/12 WF folds, +458% PnL, 50.6% WR, 4.3% overfitting drop
  - V15 ETH committee cross-applied to ADA: RECHAZADO (PF 0.89)
  - ADA SHORT: RECHAZADO in all versions (35.3% WR, -222%)
  - ADA-BTC corr ~0.70, ADA-ETH ~0.75
  - Vol filter vol_ratio > 2.0 improved V14 WR 58.2% -> 65.6%

ADA vs SOL: ADA has full data from 2020-01 (all 12 folds covered), lower
volatility than SOL, and historically the 2nd-best ML pair after BTC.

Parts:
  1. V14-style ML ensemble (10 features, expanding WF, th=0.50/0.55/0.60)
  2. ML + Regime filter (BULL/RANGE only)
  3. ML + Vol filter (vol_ratio > 2.0, proven V14 enhancement)
  4. ML + Combined filters (regime + vol + momentum)
  5. Rule-based strategies (breakout ADA, BTC-follower) for comparison
  6. ATR-based TP/SL variants
  7. OOS 2026 (Jan-Mar, reserved)

Usage:
  python evaluate_ada_v15.py
"""

import sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pandas_ta as pta
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

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

# V14 proven TP/SL for ADA
TP_FIXED = 0.06
SL_FIXED = 0.04
TIMEOUT = 15

# Feature sets (use _n suffix to avoid overwriting raw columns)
FEATURES_V14 = ['rsi_n', 'macd_norm', 'adx_n', 'bb_pct', 'atr_pct_n',
                'ret_3', 'ret_5_n', 'ret_10_n', 'vol_ratio', 'trend']


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


def compute_ml_features(df):
    """Compute V14-style ML features with _n suffix to avoid column conflicts."""
    out = df.copy()
    c = out['close']

    # V14 features normalized
    out['rsi_n'] = out['rsi14'] / 100.0
    out['adx_n'] = out['adx14'] / 100.0
    out['atr_pct_n'] = out['atr_pct'] / 100.0  # ratio (raw atr_pct stays as %)

    # MACD normalized
    macd_obj = pta.macd(c, fast=12, slow=26, signal=9)
    if macd_obj is not None:
        macd_line = macd_obj.iloc[:, 0]
        out['macd_norm'] = macd_line / c
    else:
        out['macd_norm'] = 0.0

    # Returns (as ratio)
    out['ret_3'] = c.pct_change(3)
    out['ret_5_n'] = c.pct_change(5)
    out['ret_10_n'] = c.pct_change(10)

    # Trend: close > SMA50
    sma50 = c.rolling(50).mean()
    out['trend'] = (c > sma50).astype(float)

    return out.dropna(subset=FEATURES_V14)


def create_labels(df, tp_pct=TP_FIXED, sl_pct=SL_FIXED, max_bars=TIMEOUT):
    """Create binary labels: 1=TP hit before SL within max_bars."""
    labels = np.zeros(len(df), dtype=int)
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    for i in range(len(df) - max_bars - 1):
        entry = closes[i]
        tp = entry * (1 + tp_pct)
        sl = entry * (1 - sl_pct)
        for j in range(1, max_bars + 1):
            b = i + j
            if lows[b] <= sl:
                if highs[b] >= tp and closes[b] > (sl + tp) / 2:
                    labels[i] = 1
                break
            if highs[b] >= tp:
                labels[i] = 1
                break
    return labels


# ==============================================================
# RULE-BASED DETECTORS (V15-style)
# ==============================================================
def detect_breakout_ada(df, idx, vol_min=1.2, bb_max=6.0, bar_max=5.0):
    """Breakout above 20-bar high with vol/bb/bar confirmations."""
    row = df.iloc[idx]
    close = row['close']
    high20 = row.get('high20', None)
    vol_ratio = row.get('vol_ratio', 0)
    bb_width = row.get('bb_width', 99)
    ret_1 = row.get('ret_1', 0)

    if high20 is None or pd.isna(high20):
        return False
    if close <= high20:
        return False
    if vol_ratio < vol_min:
        return False
    if bb_width > bb_max:
        return False
    if abs(ret_1) > bar_max:
        return False
    return True


def detect_btc_breakout(df_btc, idx):
    """Detect BTC breakout for follower signals."""
    if idx < 1:
        return False
    row = df_btc.iloc[idx]
    close = row['close']
    high20 = row.get('high20', None)
    vol_ratio = row.get('vol_ratio', 0)

    if high20 is None or pd.isna(high20):
        return False
    if close <= high20:
        return False
    if vol_ratio < 1.0:
        return False
    return True


# ==============================================================
# DATA LOADING
# ==============================================================
def load_data():
    print("Loading ADA 4h data...")
    df_ada_raw = load_pair_4h('ADA')
    df_ada = compute_features_4h(df_ada_raw.copy())

    # Daily macro for ADA
    try:
        from v15_framework import load_pair_1d
        ada_1d = load_pair_1d('ADA')
    except (FileNotFoundError, Exception):
        print("  No daily data for ADA, resampling 4h -> 1d")
        ada_1d = df_ada_raw.resample('1D').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()
    ada_macro = compute_macro_daily(ada_1d)
    df_ada = merge_daily_to_4h(df_ada, ada_macro)

    # Regimes
    regimes = df_ada.apply(lambda r: detect_regime(r), axis=1)

    # ML features (may drop rows due to NaN)
    df_ada = compute_ml_features(df_ada)

    # Re-align regimes after feature computation dropna
    regimes = regimes.reindex(df_ada.index)

    # Labels
    print("  Creating labels (TP=6%, SL=4%, timeout=15)...")
    labels = create_labels(df_ada, TP_FIXED, SL_FIXED, TIMEOUT)
    df_ada['label'] = labels

    # Stats
    win_rate_raw = labels[:len(labels)-TIMEOUT-1].mean()
    atr_mean = float(df_ada['atr_pct'].mean())
    bb_mean = float(df_ada['bb_width'].mean())
    print(f"  ADA: {len(df_ada)} bars ({df_ada.index[0].date()} to {df_ada.index[-1].date()})")
    print(f"  Raw label WR: {win_rate_raw:.1%} (TP=6%/SL=4%)")
    reg_counts = regimes.value_counts().to_dict()
    print(f"  Regimes: {reg_counts}")
    print(f"  ATR%: mean={atr_mean:.2f}%, BB width: mean={bb_mean:.1f}%")

    # Load BTC for cross-asset
    print("  Loading BTC 4h for cross-asset...")
    df_btc_raw = load_btc_4h()
    df_btc = compute_features_4h(df_btc_raw.copy())

    # ADA-BTC correlation
    common_idx = df_ada.index.intersection(df_btc.index)
    if len(common_idx) > 100:
        ada_ret = df_ada.loc[common_idx, 'close'].pct_change()
        btc_ret = df_btc.loc[common_idx, 'close'].pct_change()
        roll_corr = ada_ret.rolling(168).corr(btc_ret)  # 168 = ~4 weeks of 4h
        df_ada['ada_btc_corr'] = roll_corr.reindex(df_ada.index).ffill()
        pct_high = (df_ada['ada_btc_corr'] >= 0.5).mean()
        print(f"  ADA-BTC corr: mean={df_ada['ada_btc_corr'].mean():.3f}, "
              f">= 0.5: {pct_high:.1%}")
    else:
        df_ada['ada_btc_corr'] = 0.7

    return df_ada, regimes, df_btc


# ==============================================================
# ML WALK-FORWARD ENGINE
# ==============================================================
def ml_walk_forward(df, regimes, feature_cols, name,
                    threshold=0.50, regime_filter=None,
                    vol_filter=None, momentum_filter=False,
                    tp_pct=TP_FIXED, sl_pct=SL_FIXED):
    """Expanding-window walk-forward for ML ensemble."""
    results = []
    all_trades = []

    X_all = df[feature_cols].values
    y_all = df['label'].values

    for fi, (start_s, end_s) in enumerate(WF_FOLDS):
        fold_start = pd.Timestamp(start_s, tz='UTC')
        fold_end = pd.Timestamp(end_s, tz='UTC')

        train_mask = df.index < fold_start
        test_mask = (df.index >= fold_start) & (df.index <= fold_end)

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]

        if len(X_train) < 200:
            period = f"{start_s[:7]}/{end_s[5:7]}"
            results.append({'period': period, 'n': 0, 'wr': 0, 'pf': 0, 'ok': False})
            continue

        # Train ensemble
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        rf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                     min_samples_leaf=10, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                         min_samples_leaf=10, random_state=42)
        rf.fit(X_train_s, y_train)
        gb.fit(X_train_s, y_train)

        # Predict on test fold
        X_test = X_all[test_mask]
        X_test_s = scaler.transform(X_test)
        prob_rf = rf.predict_proba(X_test_s)[:, 1]
        prob_gb = gb.predict_proba(X_test_s)[:, 1]

        # Simulate trades
        fold_trades = []
        test_indices = np.where(test_mask)[0]

        for k, idx in enumerate(test_indices):
            if idx + TIMEOUT + 1 >= len(df):
                continue

            # ML signal: both models agree above threshold
            if prob_rf[k] < threshold or prob_gb[k] < threshold:
                continue

            ts = df.index[idx]

            # Regime filter
            if regime_filter is not None:
                reg = regimes.iloc[idx]
                if reg not in regime_filter:
                    continue

            # Vol filter (V14 proven: vol_ratio > 2.0)
            if vol_filter is not None:
                vr = float(df['vol_ratio'].iloc[idx])
                if vr < vol_filter:
                    continue

            # Momentum filter
            if momentum_filter:
                ret3 = float(df['ret_3'].iloc[idx])
                if ret3 < -0.03:
                    continue

            entry = float(df['close'].iloc[idx])
            out = sim_trade_fixed(df, idx, entry, tp_pct, sl_pct, max_bars=TIMEOUT)
            fold_trades.append({
                'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                'setup': 'ML_ENS', 'direction': 'LONG', 'entry': entry,
                'prob': float((prob_rf[k] + prob_gb[k]) / 2),
            })

        m = metrics(fold_trades, '')
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        period = f"{start_s[:7]}/{end_s[5:7]}"
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'], 'pf': m['pf'], 'ok': ok})
        all_trades.extend(fold_trades)

    # Summary
    m_all = metrics(all_trades, name)
    eq, dd = equity_stats(all_trades)
    folds_ok = sum(1 for r in results if r['ok'])
    folds_with_data = sum(1 for r in results if r['n'] > 0)

    return {
        'name': name, 'folds': results,
        'folds_ok': folds_ok, 'folds_with_data': folds_with_data,
        'trades': all_trades,
        'n': m_all['n'], 'wr': m_all['wr'], 'pf': m_all['pf'],
        'eq': eq, 'dd': dd,
    }


def ml_walk_forward_atr(df, regimes, feature_cols, name,
                        threshold=0.50, regime_filter=None,
                        vol_filter=None, momentum_filter=False,
                        tp_mult=2.5, sl_mult=1.5, tp_cap=0.10, sl_cap=0.06):
    """ML walk-forward with ATR-based TP/SL per trade."""
    results = []
    all_trades = []

    X_all = df[feature_cols].values
    y_all = df['label'].values

    for fi, (start_s, end_s) in enumerate(WF_FOLDS):
        fold_start = pd.Timestamp(start_s, tz='UTC')
        fold_end = pd.Timestamp(end_s, tz='UTC')

        train_mask = df.index < fold_start
        test_mask = (df.index >= fold_start) & (df.index <= fold_end)

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]

        if len(X_train) < 200:
            period = f"{start_s[:7]}/{end_s[5:7]}"
            results.append({'period': period, 'n': 0, 'wr': 0, 'pf': 0, 'ok': False})
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        rf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                     min_samples_leaf=10, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                         min_samples_leaf=10, random_state=42)
        rf.fit(X_train_s, y_train)
        gb.fit(X_train_s, y_train)

        X_test = X_all[test_mask]
        X_test_s = scaler.transform(X_test)
        prob_rf = rf.predict_proba(X_test_s)[:, 1]
        prob_gb = gb.predict_proba(X_test_s)[:, 1]

        fold_trades = []
        test_indices = np.where(test_mask)[0]

        for k, idx in enumerate(test_indices):
            if idx + 18 >= len(df):
                continue
            if prob_rf[k] < threshold or prob_gb[k] < threshold:
                continue

            ts = df.index[idx]
            if regime_filter is not None:
                if regimes.iloc[idx] not in regime_filter:
                    continue
            if vol_filter is not None:
                if float(df['vol_ratio'].iloc[idx]) < vol_filter:
                    continue
            if momentum_filter:
                if float(df['ret_3'].iloc[idx]) < -0.03:
                    continue

            entry = float(df['close'].iloc[idx])
            # ATR-based TP/SL
            atr_pct_raw = float(df['atr_pct'].iloc[idx])  # in %
            sl_pct = max(min(atr_pct_raw / 100 * sl_mult, sl_cap), 0.015)
            tp_pct = max(min(atr_pct_raw / 100 * tp_mult, tp_cap), 0.025)

            out = sim_trade_fixed(df, idx, entry, tp_pct, sl_pct, max_bars=18)
            fold_trades.append({
                'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                'setup': 'ML_ATR', 'direction': 'LONG', 'entry': entry,
                'prob': float((prob_rf[k] + prob_gb[k]) / 2),
            })

        m = metrics(fold_trades, '')
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        period = f"{start_s[:7]}/{end_s[5:7]}"
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'], 'pf': m['pf'], 'ok': ok})
        all_trades.extend(fold_trades)

    m_all = metrics(all_trades, name)
    eq, dd = equity_stats(all_trades)
    folds_ok = sum(1 for r in results if r['ok'])
    folds_with_data = sum(1 for r in results if r['n'] > 0)

    return {
        'name': name, 'folds': results,
        'folds_ok': folds_ok, 'folds_with_data': folds_with_data,
        'trades': all_trades,
        'n': m_all['n'], 'wr': m_all['wr'], 'pf': m_all['pf'],
        'eq': eq, 'dd': dd,
    }


# ==============================================================
# RULE-BASED WALK-FORWARD
# ==============================================================
def rules_walk_forward(df, df_btc, regimes, name,
                       use_breakout=True, use_btc_follow=True,
                       vol_min=1.2, bb_max=6.0, bar_max=5.0,
                       btc_corr_min=0.5,
                       tp_pct=TP_FIXED, sl_pct=SL_FIXED,
                       use_atr=False, tp_mult=2.5, sl_mult=1.5,
                       tp_cap=0.10, sl_cap=0.06):
    """Rule-based walk-forward (breakout ADA + BTC-follower)."""
    results = []
    all_trades = []

    for fi, (start_s, end_s) in enumerate(WF_FOLDS):
        fold_start = pd.Timestamp(start_s, tz='UTC')
        fold_end = pd.Timestamp(end_s, tz='UTC')
        fold_mask = (df.index >= fold_start) & (df.index <= fold_end)
        fold_indices = np.where(fold_mask)[0]

        fold_trades = []

        for idx in fold_indices:
            if idx + TIMEOUT + 1 >= len(df):
                continue

            # Only trade in BULL/RANGE
            reg = regimes.iloc[idx]
            if reg == 'BEAR':
                continue

            ts = df.index[idx]
            entry = float(df['close'].iloc[idx])

            # TP/SL
            if use_atr:
                atr_raw = float(df['atr_pct'].iloc[idx])
                sl_p = max(min(atr_raw / 100 * sl_mult, sl_cap), 0.015)
                tp_p = max(min(atr_raw / 100 * tp_mult, tp_cap), 0.025)
            else:
                tp_p, sl_p = tp_pct, sl_pct

            triggered = False

            # ADA standalone breakout
            if use_breakout and detect_breakout_ada(df, idx, vol_min, bb_max, bar_max):
                out = sim_trade_fixed(df, idx, entry, tp_p, sl_p, max_bars=TIMEOUT)
                fold_trades.append({
                    'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                    'setup': 'BRK_ADA', 'direction': 'LONG', 'entry': entry,
                })
                triggered = True

            # BTC breakout follower
            if use_btc_follow and not triggered:
                # Find matching BTC index
                btc_idx_loc = df_btc.index.get_indexer([ts], method='nearest')[0]
                if btc_idx_loc >= 0 and btc_idx_loc < len(df_btc):
                    if detect_btc_breakout(df_btc, btc_idx_loc):
                        # Check ADA-BTC correlation
                        corr_val = float(df['ada_btc_corr'].iloc[idx]) if 'ada_btc_corr' in df.columns else 0.7
                        if corr_val >= btc_corr_min:
                            out = sim_trade_fixed(df, idx, entry, tp_p, sl_p, max_bars=TIMEOUT)
                            fold_trades.append({
                                'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                                'setup': 'FOLLOW_BTC', 'direction': 'LONG', 'entry': entry,
                            })

        m = metrics(fold_trades, '')
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        period = f"{start_s[:7]}/{end_s[5:7]}"
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'], 'pf': m['pf'], 'ok': ok})
        all_trades.extend(fold_trades)

    m_all = metrics(all_trades, name)
    eq, dd = equity_stats(all_trades)
    folds_ok = sum(1 for r in results if r['ok'])
    folds_with_data = sum(1 for r in results if r['n'] > 0)

    return {
        'name': name, 'folds': results,
        'folds_ok': folds_ok, 'folds_with_data': folds_with_data,
        'trades': all_trades,
        'n': m_all['n'], 'wr': m_all['wr'], 'pf': m_all['pf'],
        'eq': eq, 'dd': dd,
    }


# ==============================================================
# PRINT HELPERS
# ==============================================================
def print_wf_result(r, show_folds=True):
    if show_folds:
        for f in r['folds']:
            wr_s = f"{f['wr']:.1%}" if f['n'] > 0 else "n/a"
            pf_s = f"{f['pf']:.2f}" if f['n'] > 0 else "n/a"
            ok_s = "+" if f['ok'] else ("-" if f['n'] > 0 else ".")
            print(f"    {f['period']}: N={f['n']:>3} WR={wr_s:>6} PF={pf_s:>6} {ok_s}")

    passed = r['folds_ok'] >= 7 and r['pf'] >= 1.0
    marginal = r['folds_ok'] >= 6 and r['pf'] >= 1.0
    tag = "APROBADO" if passed else ("MARGINAL" if marginal else "RECHAZADO")

    print(f"    WF: {r['folds_ok']}/{r['folds_with_data']} | N={r['n']} WR={r['wr']:.1%} "
          f"PF={r['pf']:.2f} $1K->${1000*r['eq']:.0f} DD={r['dd']:.1%} -> {tag}")
    return tag


# ==============================================================
# OOS 2026
# ==============================================================
def run_oos_ml(df, regimes, feature_cols, name,
               threshold=0.50, regime_filter=None,
               vol_filter=None, momentum_filter=False,
               tp_pct=TP_FIXED, sl_pct=SL_FIXED,
               use_atr=False, tp_mult=2.5, sl_mult=1.5,
               tp_cap=0.10, sl_cap=0.06):
    """Train on ALL pre-2026, test on 2026 OOS."""
    train_mask = df.index <= pd.Timestamp(TRAIN_CUTOFF, tz='UTC')
    test_mask = (df.index >= pd.Timestamp(OOS_START, tz='UTC')) & \
                (df.index < pd.Timestamp(OOS_END, tz='UTC'))

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, 'label'].values
    X_test = df.loc[test_mask, feature_cols].values

    if len(X_train) < 200 or test_mask.sum() == 0:
        print(f"    {name}: insufficient data")
        return []

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                 min_samples_leaf=10, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                     min_samples_leaf=10, random_state=42)
    rf.fit(X_train_s, y_train)
    gb.fit(X_train_s, y_train)

    prob_rf = rf.predict_proba(X_test_s)[:, 1]
    prob_gb = gb.predict_proba(X_test_s)[:, 1]

    trades = []
    test_indices = np.where(test_mask)[0]

    for k, idx in enumerate(test_indices):
        if idx + TIMEOUT + 1 >= len(df):
            continue
        if prob_rf[k] < threshold or prob_gb[k] < threshold:
            continue

        ts = df.index[idx]
        if regime_filter is not None:
            if regimes.iloc[idx] not in regime_filter:
                continue
        if vol_filter is not None:
            if float(df['vol_ratio'].iloc[idx]) < vol_filter:
                continue
        if momentum_filter:
            if float(df['ret_3'].iloc[idx]) < -0.03:
                continue

        entry = float(df['close'].iloc[idx])

        if use_atr:
            atr_raw = float(df['atr_pct'].iloc[idx])
            sl_p = max(min(atr_raw / 100 * sl_mult, sl_cap), 0.015)
            tp_p = max(min(atr_raw / 100 * tp_mult, tp_cap), 0.025)
        else:
            tp_p, sl_p = tp_pct, sl_pct

        out = sim_trade_fixed(df, idx, entry, tp_p, sl_p, max_bars=TIMEOUT)
        trades.append({
            'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
            'setup': 'ML_ENS', 'direction': 'LONG', 'entry': entry,
            'prob': float((prob_rf[k] + prob_gb[k]) / 2),
        })

    return trades


def run_oos_rules(df, df_btc, regimes, name,
                  use_breakout=True, use_btc_follow=True,
                  vol_min=1.2, bb_max=6.0, bar_max=5.0,
                  btc_corr_min=0.5,
                  tp_pct=TP_FIXED, sl_pct=SL_FIXED,
                  use_atr=False, tp_mult=2.5, sl_mult=1.5,
                  tp_cap=0.10, sl_cap=0.06):
    """Rule-based OOS 2026."""
    test_mask = (df.index >= pd.Timestamp(OOS_START, tz='UTC')) & \
                (df.index < pd.Timestamp(OOS_END, tz='UTC'))
    test_indices = np.where(test_mask)[0]

    trades = []
    for idx in test_indices:
        if idx + TIMEOUT + 1 >= len(df):
            continue

        reg = regimes.iloc[idx]
        if reg == 'BEAR':
            continue

        ts = df.index[idx]
        entry = float(df['close'].iloc[idx])

        if use_atr:
            atr_raw = float(df['atr_pct'].iloc[idx])
            sl_p = max(min(atr_raw / 100 * sl_mult, sl_cap), 0.015)
            tp_p = max(min(atr_raw / 100 * tp_mult, tp_cap), 0.025)
        else:
            tp_p, sl_p = tp_pct, sl_pct

        triggered = False
        if use_breakout and detect_breakout_ada(df, idx, vol_min, bb_max, bar_max):
            out = sim_trade_fixed(df, idx, entry, tp_p, sl_p, max_bars=TIMEOUT)
            trades.append({
                'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                'setup': 'BRK_ADA', 'direction': 'LONG', 'entry': entry,
            })
            triggered = True

        if use_btc_follow and not triggered:
            btc_idx_loc = df_btc.index.get_indexer([ts], method='nearest')[0]
            if 0 <= btc_idx_loc < len(df_btc):
                if detect_btc_breakout(df_btc, btc_idx_loc):
                    corr_val = float(df['ada_btc_corr'].iloc[idx]) if 'ada_btc_corr' in df.columns else 0.7
                    if corr_val >= btc_corr_min:
                        out = sim_trade_fixed(df, idx, entry, tp_p, sl_p, max_bars=TIMEOUT)
                        trades.append({
                            'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                            'setup': 'FOLLOW_BTC', 'direction': 'LONG', 'entry': entry,
                        })

    return trades


# ==============================================================
# MAIN
# ==============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("ADA/USDT -- V15 Strategy Evaluation")
    print("=" * 70)

    df_ada, regimes, df_btc = load_data()

    # ============================================================
    # PART 1: ML Ensemble V14-style (10 features)
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 1: ML Ensemble V14-style (10 features, TP=6%/SL=4%)")
    print("=" * 70)

    configs_p1 = []
    for th in [0.50, 0.55, 0.60]:
        name = f"ML_V14_th{th}"
        print(f"\n  --- {name} ---")
        r = ml_walk_forward(df_ada, regimes, FEATURES_V14, name, threshold=th)
        tag = print_wf_result(r)
        configs_p1.append((name, r, tag))

    # ============================================================
    # PART 2: ML + Regime filter (BULL/RANGE only)
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 2: ML + Regime filter (BULL/RANGE only)")
    print("=" * 70)

    configs_p2 = []
    for th in [0.50, 0.55, 0.60]:
        name = f"ML_REGIME_th{th}"
        print(f"\n  --- {name} ---")
        r = ml_walk_forward(df_ada, regimes, FEATURES_V14, name,
                            threshold=th, regime_filter=('BULL', 'RANGE'))
        tag = print_wf_result(r)
        configs_p2.append((name, r, tag))

    # ============================================================
    # PART 3: ML + Vol filter (vol_ratio > 2.0, proven V14)
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 3: ML + Vol filter (vol_ratio > 2.0)")
    print("=" * 70)

    configs_p3 = []
    for th in [0.50, 0.55, 0.60]:
        name = f"ML_VOL2_th{th}"
        print(f"\n  --- {name} ---")
        r = ml_walk_forward(df_ada, regimes, FEATURES_V14, name,
                            threshold=th, vol_filter=2.0)
        tag = print_wf_result(r)
        configs_p3.append((name, r, tag))

    # ============================================================
    # PART 4: ML + Combined filters (regime + vol + momentum)
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 4: ML + Combined filters")
    print("=" * 70)

    configs_p4 = []
    filter_combos = [
        ('REGIME+VOL', ('BULL', 'RANGE'), 2.0, False),
        ('REGIME+MOM', ('BULL', 'RANGE'), None, True),
        ('ALL_FILTERS', ('BULL', 'RANGE'), 2.0, True),
        ('VOL+MOM', None, 2.0, True),
        ('VOL1.5', None, 1.5, False),  # lower vol filter
    ]

    for filter_name, reg_f, vol_f, mom_f in filter_combos:
        for th in [0.50, 0.55]:
            name = f"ML_{filter_name}_th{th}"
            print(f"\n  --- {name} ---")
            r = ml_walk_forward(df_ada, regimes, FEATURES_V14, name,
                                threshold=th, regime_filter=reg_f,
                                vol_filter=vol_f, momentum_filter=mom_f)
            tag = print_wf_result(r, show_folds=False)
            configs_p4.append((name, r, tag))

    # ============================================================
    # PART 5: Rule-based strategies (comparison)
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 5: Rule-based strategies (breakout ADA + BTC-follower)")
    print("=" * 70)

    configs_p5 = []
    rule_combos = [
        # (name, breakout, btc_follow, vol_min, bb_max, bar_max, corr_min)
        ('BRK_ADA_v1.2_bb6', True, False, 1.2, 6.0, 5.0, 0.5),
        ('BRK_ADA_v1.0_bb7', True, False, 1.0, 7.0, 5.0, 0.5),
        ('BRK_ADA_v1.5_bb5', True, False, 1.5, 5.0, 5.0, 0.5),
        ('BTC_FOLLOW_c0.5', False, True, 1.2, 6.0, 5.0, 0.5),
        ('BTC_FOLLOW_c0.4', False, True, 1.2, 6.0, 5.0, 0.4),
        ('COMBINED_v1.2', True, True, 1.2, 6.0, 5.0, 0.5),
        ('COMBINED_v1.0', True, True, 1.0, 7.0, 5.0, 0.5),
    ]

    for rname, brk, btc_f, vm, bbm, barm, corr_m in rule_combos:
        # Fixed TP/SL
        name = f"RULES_{rname}_FIX"
        print(f"\n  --- {name} ---")
        r = rules_walk_forward(df_ada, df_btc, regimes, name,
                               use_breakout=brk, use_btc_follow=btc_f,
                               vol_min=vm, bb_max=bbm, bar_max=barm,
                               btc_corr_min=corr_m)
        tag = print_wf_result(r, show_folds=False)
        configs_p5.append((name, r, tag))

        # ATR-based TP/SL
        name2 = f"RULES_{rname}_ATR"
        print(f"\n  --- {name2} ---")
        r2 = rules_walk_forward(df_ada, df_btc, regimes, name2,
                                use_breakout=brk, use_btc_follow=btc_f,
                                vol_min=vm, bb_max=bbm, bar_max=barm,
                                btc_corr_min=corr_m,
                                use_atr=True, tp_mult=2.5, sl_mult=1.5,
                                tp_cap=0.10, sl_cap=0.06)
        tag2 = print_wf_result(r2, show_folds=False)
        configs_p5.append((name2, r2, tag2))

    # ============================================================
    # PART 6: ATR-based TP/SL for ML (best ML configs)
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 6: ATR-based TP/SL for ML")
    print("=" * 70)

    configs_p6 = []
    atr_profiles = [
        ('ATR_2.5x1.5', 2.5, 1.5, 0.10, 0.06),
        ('ATR_3.0x2.0', 3.0, 2.0, 0.12, 0.07),
        ('ATR_2.0x1.5', 2.0, 1.5, 0.08, 0.05),
    ]

    for atr_name, tp_m, sl_m, tp_c, sl_c in atr_profiles:
        for th in [0.50, 0.55]:
            # Plain ML + ATR
            name = f"ML_{atr_name}_th{th}"
            print(f"\n  --- {name} ---")
            r = ml_walk_forward_atr(df_ada, regimes, FEATURES_V14, name,
                                     threshold=th,
                                     tp_mult=tp_m, sl_mult=sl_m,
                                     tp_cap=tp_c, sl_cap=sl_c)
            tag = print_wf_result(r, show_folds=False)
            configs_p6.append((name, r, tag))

        # Best combo: regime + vol + ATR, th=0.50
        name3 = f"ML_{atr_name}_RV_th0.50"
        print(f"\n  --- {name3} ---")
        r3 = ml_walk_forward_atr(df_ada, regimes, FEATURES_V14, name3,
                                  threshold=0.50,
                                  regime_filter=('BULL', 'RANGE'),
                                  vol_filter=2.0,
                                  tp_mult=tp_m, sl_mult=sl_m,
                                  tp_cap=tp_c, sl_cap=sl_c)
        tag3 = print_wf_result(r3, show_folds=False)
        configs_p6.append((name3, r3, tag3))

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY: All Configurations")
    print("=" * 70)

    all_results = configs_p1 + configs_p2 + configs_p3 + configs_p4 + configs_p5 + configs_p6
    all_results.sort(key=lambda x: x[1]['pf'] if x[1]['n'] >= 10 else 0, reverse=True)

    print(f"\n  {'Config':<35} | {'WF':>5} | {'N':>4} | {'WR':>6} | {'PF':>6} | {'$1K->':>7} | {'DD':>6} | Veredicto")
    print(f"  {'-'*100}")

    for name, r, tag in all_results:
        if r['n'] < 3:
            continue
        mark = "**" if tag == "APROBADO" else ("* " if tag == "MARGINAL" else "  ")
        print(f"  {mark}{name:<33} | {r['folds_ok']:>2}/{r['folds_with_data']:>2} | {r['n']:>4} | "
              f"{r['wr']:.1%} | {r['pf']:.2f} | ${1000*r['eq']:>6.0f} | {r['dd']:.1%} | {tag}")

    # How many pass WF?
    approved = [(n, r, t) for n, r, t in all_results if t == 'APROBADO' and r['n'] >= 10]
    marginal = [(n, r, t) for n, r, t in all_results if t == 'MARGINAL' and r['n'] >= 10]
    print(f"\n  APROBADO: {len(approved)} configs | MARGINAL: {len(marginal)} configs")

    # ============================================================
    # PART 7: OOS 2026
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 7: OOS 2026 (Jan-Mar, trained on pre-2026)")
    print("=" * 70)

    # Market context
    mask_oos = (df_ada.index >= pd.Timestamp(OOS_START, tz='UTC')) & \
               (df_ada.index < pd.Timestamp(OOS_END, tz='UTC'))
    if mask_oos.sum() > 0:
        ada_start = float(df_ada.loc[mask_oos, 'close'].iloc[0])
        ada_end = float(df_ada.loc[mask_oos, 'close'].iloc[-1])
        ada_ret = (ada_end / ada_start - 1) * 100
        print(f"  ADA: ${ada_start:.4f} -> ${ada_end:.4f} ({ada_ret:+.1f}%)")
        oos_reg = regimes[mask_oos].value_counts().to_dict()
        print(f"  Regimes 2026: {oos_reg}")
    else:
        print("  No ADA data in 2026 period!")
        ada_ret = 0

    # Test top configs by PF (with N >= 10) on OOS
    candidates = [(n, r, t) for n, r, t in all_results
                  if r['n'] >= 10 and r['pf'] >= 1.0][:8]

    oos_results = []
    for cname, cr, ctag in candidates:
        # Parse config parameters from name
        is_atr = 'ATR' in cname and 'ML' in cname
        is_rules = 'RULES' in cname
        is_regime = 'REGIME' in cname or 'RV' in cname or 'ALL_FILTERS' in cname
        is_vol = 'VOL' in cname or 'RV' in cname or 'ALL_FILTERS' in cname
        is_mom = 'MOM' in cname or 'ALL_FILTERS' in cname

        th = 0.50
        if 'th0.55' in cname: th = 0.55
        elif 'th0.60' in cname: th = 0.60

        # Parse vol filter value
        vol_f = None
        if is_vol:
            vol_f = 2.0
            if 'VOL1.5' in cname:
                vol_f = 1.5

        regime_f = ('BULL', 'RANGE') if is_regime else None
        mom_f = is_mom

        print(f"\n  --- {cname} (OOS 2026) ---")

        if is_rules:
            # Parse rule params from name
            brk = 'BRK' in cname or 'COMBINED' in cname
            btc_f = 'FOLLOW' in cname or 'COMBINED' in cname
            use_atr_r = 'ATR' in cname
            trades = run_oos_rules(df_ada, df_btc, regimes, cname,
                                   use_breakout=brk, use_btc_follow=btc_f,
                                   use_atr=use_atr_r)
        elif is_atr:
            # Parse ATR params
            if '3.0x2.0' in cname:
                tp_m, sl_m, tp_c, sl_c = 3.0, 2.0, 0.12, 0.07
            elif '2.0x1.5' in cname:
                tp_m, sl_m, tp_c, sl_c = 2.0, 1.5, 0.08, 0.05
            else:
                tp_m, sl_m, tp_c, sl_c = 2.5, 1.5, 0.10, 0.06
            trades = run_oos_ml(df_ada, regimes, FEATURES_V14, cname,
                                threshold=th, regime_filter=regime_f,
                                vol_filter=vol_f, momentum_filter=mom_f,
                                use_atr=True, tp_mult=tp_m, sl_mult=sl_m,
                                tp_cap=tp_c, sl_cap=sl_c)
        else:
            trades = run_oos_ml(df_ada, regimes, FEATURES_V14, cname,
                                threshold=th, regime_filter=regime_f,
                                vol_filter=vol_f, momentum_filter=mom_f)

        if not trades:
            print("    No trades in OOS period")
            oos_results.append((cname, 0, 0, 0, 1.0, 0))
            continue

        for t in sorted(trades, key=lambda x: x['ts']):
            prob_s = f"prob={t['prob']:.2f} " if 'prob' in t else ""
            print(f"    {str(t['ts'])[:19]:<22} ${t['entry']:>8.4f} "
                  f"{prob_s}{t['outcome']:<4} {t['pnl_pct']:>+7.2%}")

        m = metrics(trades, cname)
        eq, dd = equity_stats(trades)
        total_pnl = sum(t['pnl_pct'] for t in trades)
        print(f"    N={m['n']} WR={m['wr']:.1%} PF={m['pf']:.2f} "
              f"$1K->${1000*eq:.0f} DD={dd:.1%} Total={total_pnl:+.2%}")
        print(f"    vs ADA buy-and-hold: {ada_ret:+.1f}%")
        oos_results.append((cname, m['n'], m['wr'], m['pf'], eq, total_pnl))

    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if approved:
        print(f"\n  APROBADO ({len(approved)} configs):")
        for name, r, _ in approved:
            print(f"    {name}: WF {r['folds_ok']}/{r['folds_with_data']} "
                  f"PF={r['pf']:.2f} WR={r['wr']:.1%} DD={r['dd']:.1%} N={r['n']}")

        # Check DD constraint (<= 25%)
        low_dd = [(n, r) for n, r, _ in approved if r['dd'] <= 0.25]
        if low_dd:
            print(f"\n  With DD <= 25%: {len(low_dd)} configs")
            for name, r in low_dd:
                print(f"    {name}: DD={r['dd']:.1%}")
        else:
            print("\n  WARNING: No APROBADO config with DD <= 25%")

    elif marginal:
        print(f"\n  MARGINAL ({len(marginal)} configs):")
        for name, r, _ in marginal:
            print(f"    {name}: WF {r['folds_ok']}/{r['folds_with_data']} "
                  f"PF={r['pf']:.2f} WR={r['wr']:.1%} DD={r['dd']:.1%} N={r['n']}")
    else:
        print("\n  RECHAZADO: Ningun config pasa WF >= 7 + PF >= 1.0")

    # Period analysis: where do trades concentrate?
    print("\n  --- Trade Distribution by Period ---")
    all_ml = [(n, r, t) for n, r, t in all_results if 'ML_V14_th0.50' == n]
    if all_ml:
        base = all_ml[0][1]
        for f in base['folds']:
            print(f"    {f['period']}: N={f['n']:>3}")

    # OOS summary
    print("\n  --- OOS 2026 Summary ---")
    for cname, n, wr, pf, eq, total in oos_results:
        if n > 0:
            print(f"    {cname}: N={n} WR={wr:.1%} PF={pf:.2f} Total={total:+.2%}")
        else:
            print(f"    {cname}: No trades")

    print("=" * 70)
