"""
evaluate_alt_v15_solutions.py -- New approaches for SOL & ADA
=============================================================
Both pairs RECHAZADO with V15 standard approach (TP=6%/SL=4%, RF+GB ML).
This script tests 4 genuinely untested approaches:

  1. Smaller TP/SL targets (TP=2-4% / SL=1.5-2.5%)
  2. Trailing stop (proven in V7, never in V15)
  3. Position sizing as DD solution (mathematical)
  4. LightGBM regression (V7 algorithm, 34 features)

Usage:
  python evaluate_alt_v15_solutions.py
"""

import sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pandas_ta as pta
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

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

TPSL_CONFIGS = [
    {'name': 'TP6_SL4',    'tp': 0.06,  'sl': 0.04,  'timeout': 15},
    {'name': 'TP4_SL2.5',  'tp': 0.04,  'sl': 0.025, 'timeout': 15},
    {'name': 'TP4_SL2',    'tp': 0.04,  'sl': 0.02,  'timeout': 15},
    {'name': 'TP3_SL2',    'tp': 0.03,  'sl': 0.02,  'timeout': 15},
    {'name': 'TP3_SL1.5',  'tp': 0.03,  'sl': 0.015, 'timeout': 18},
    {'name': 'TP2.5_SL1.5','tp': 0.025, 'sl': 0.015, 'timeout': 18},
    {'name': 'TP2_SL1.5',  'tp': 0.02,  'sl': 0.015, 'timeout': 18},
]

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


def compute_sized_equity(trades, sizing_mult):
    """Equity/DD scaled by position sizing multiplier."""
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


def compute_ml_features(df):
    """V14-style ML features (10) with _n suffix."""
    out = df.copy()
    c = out['close']
    out['rsi_n'] = out['rsi14'] / 100.0
    out['adx_n'] = out['adx14'] / 100.0
    out['atr_pct_n'] = out['atr_pct'] / 100.0
    macd_obj = pta.macd(c, fast=12, slow=26, signal=9)
    if macd_obj is not None:
        out['macd_norm'] = macd_obj.iloc[:, 0] / c
    else:
        out['macd_norm'] = 0.0
    out['ret_3'] = c.pct_change(3)
    out['ret_5_n'] = c.pct_change(5)
    out['ret_10_n'] = c.pct_change(10)
    sma50 = c.rolling(50).mean()
    out['trend'] = (c > sma50).astype(float)
    return out.dropna(subset=FEATURES_V14)


def compute_v7_features(df):
    """V7-style 34 features (from ml_export_models.py)."""
    feat = pd.DataFrame(index=df.index)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    for p in [1, 3, 5, 10, 20]:
        feat[f'ret_{p}'] = c.pct_change(p)
    feat['atr14'] = pta.atr(h, l, c, length=14)
    feat['atr_r'] = feat['atr14'] / feat['atr14'].rolling(50).mean()
    feat['vol5'] = c.pct_change().rolling(5).std()
    feat['vol20'] = c.pct_change().rolling(20).std()
    feat['rsi14'] = pta.rsi(c, length=14)
    feat['rsi7'] = pta.rsi(c, length=7)
    sr = pta.stochrsi(c, length=14, rsi_length=14, k=3, d=3)
    if sr is not None:
        feat['srsi_k'] = sr.iloc[:, 0]
    macd = pta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        feat['macd_h'] = macd.iloc[:, 1]
    feat['roc5'] = pta.roc(c, length=5)
    feat['roc20'] = pta.roc(c, length=20)
    for el in [8, 21, 55, 100, 200]:
        e = pta.ema(c, length=el)
        feat[f'ema{el}_d'] = (c - e) / e * 100
    feat['ema8_sl'] = pta.ema(c, length=8).pct_change(3) * 100
    feat['ema55_sl'] = pta.ema(c, length=55).pct_change(5) * 100
    bb = pta.bbands(c, length=20, std=2.0)
    if bb is not None:
        bw = bb.iloc[:, 2] - bb.iloc[:, 0]
        feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw
        feat['bb_w'] = bw / bb.iloc[:, 1] * 100
    feat['vr'] = v / v.rolling(20).mean()
    feat['spr'] = (h - l) / c * 100
    feat['body'] = abs(c - df['open']) / (h - l + 1e-10)
    ax = pta.adx(h, l, c, length=14)
    if ax is not None:
        feat['adx'] = ax.iloc[:, 0]
        feat['dip'] = ax.iloc[:, 1]
        feat['dim'] = ax.iloc[:, 2]
    hr = df.index.hour; dw = df.index.dayofweek
    feat['h_s'] = np.sin(2 * np.pi * hr / 24)
    feat['h_c'] = np.cos(2 * np.pi * hr / 24)
    feat['d_s'] = np.sin(2 * np.pi * dw / 7)
    feat['d_c'] = np.cos(2 * np.pi * dw / 7)
    return feat


def create_labels(df, tp_pct, sl_pct, max_bars):
    """Binary labels: 1=TP hit before SL within max_bars."""
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
# TRAILING STOP SIMULATION
# ==============================================================
def sim_trade_trailing(df, entry_bar, entry_price, sl_pct,
                       trail_trigger_pct=None, max_bars=30):
    """
    Trailing stop without fixed TP.
    - sl_pct: distance from peak as fraction (e.g., 0.02)
    - trail_trigger_pct: None=immediate trailing; float=activate when profit reaches this %
    - Returns: (outcome, exit_price, pnl_pct, bars_held)
    """
    sl_price = entry_price * (1 - sl_pct)
    peak = entry_price
    trailing_active = (trail_trigger_pct is None)

    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df):
            exit_p = float(df['close'].iloc[-1])
            pnl = (exit_p - entry_price) / entry_price - 2 * COMMISSION
            outcome = 'TP' if exit_p > entry_price else 'SL'
            return outcome, exit_p, pnl, i

        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        cl = float(df['close'].iloc[b])

        if hi > peak:
            peak = hi

        if not trailing_active and trail_trigger_pct is not None:
            if hi >= entry_price * (1 + trail_trigger_pct):
                trailing_active = True

        if trailing_active:
            sl_price = max(sl_price, peak * (1 - sl_pct))

        if lo <= sl_price:
            pnl = (sl_price - entry_price) / entry_price - 2 * COMMISSION
            outcome = 'TP' if sl_price > entry_price else 'SL'
            return outcome, sl_price, pnl, i

    # Timeout
    exit_p = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (exit_p - entry_price) / entry_price - 2 * COMMISSION
    outcome = 'TP' if exit_p > entry_price else 'SL'
    return outcome, exit_p, pnl, max_bars


# ==============================================================
# RULE-BASED DETECTORS
# ==============================================================
def detect_breakout(df, idx, vol_min=1.2, bb_max=6.0, bar_max=5.0):
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
    """BTC breakout for follower signals."""
    if idx < 1:
        return False
    row = df_btc.iloc[idx]
    close = row['close']
    high20 = row.get('high20', None)
    vol_ratio = row.get('vol_ratio', 0)
    if high20 is None or pd.isna(high20) or close <= high20 or vol_ratio < 1.0:
        return False
    return True


# ==============================================================
# DATA LOADING
# ==============================================================
def load_pair_data(pair):
    """Load pair data with features, regimes, and BTC cross-reference."""
    print(f"  Loading {pair} 4h data...")
    df_raw = load_pair_4h(pair)
    df = compute_features_4h(df_raw.copy())

    # Daily macro
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

    # Regimes
    regimes = df.apply(lambda r: detect_regime(r), axis=1)

    # ML features
    df = compute_ml_features(df)
    regimes = regimes.reindex(df.index)

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
    print(f"    {pair}-BTC corr: mean={corr_mean:.3f}")

    return df, regimes, df_btc


# ==============================================================
# ML WALK-FORWARD (generic TP/SL)
# ==============================================================
def ml_walk_forward(df, regimes, labels, tp_pct, sl_pct, timeout, name,
                    threshold=0.50, regime_filter=True):
    """Expanding-window ML walk-forward with custom TP/SL."""
    results = []
    all_trades = []
    X_all = df[FEATURES_V14].values
    y_all = labels

    for fi, (start_s, end_s) in enumerate(WF_FOLDS):
        fold_start = pd.Timestamp(start_s, tz='UTC')
        fold_end = pd.Timestamp(end_s, tz='UTC')
        train_mask = df.index < fold_start
        test_mask = (df.index >= fold_start) & (df.index <= fold_end)

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        if len(X_train) < 200:
            results.append({'period': f"{start_s[:7]}/{end_s[5:7]}", 'n': 0, 'wr': 0, 'pf': 0, 'ok': False})
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, min_samples_leaf=10, random_state=42)
        rf.fit(X_train_s, y_train)
        gb.fit(X_train_s, y_train)

        X_test = X_all[test_mask]
        X_test_s = scaler.transform(X_test)
        prob_rf = rf.predict_proba(X_test_s)[:, 1]
        prob_gb = gb.predict_proba(X_test_s)[:, 1]

        fold_trades = []
        test_indices = np.where(test_mask)[0]
        for k, idx in enumerate(test_indices):
            if idx + timeout + 1 >= len(df):
                continue
            if prob_rf[k] < threshold or prob_gb[k] < threshold:
                continue
            if regime_filter:
                reg = regimes.iloc[idx]
                if reg == 'BEAR':
                    continue
            entry = float(df['close'].iloc[idx])
            out = sim_trade_fixed(df, idx, entry, tp_pct, sl_pct, max_bars=timeout)
            fold_trades.append({
                'outcome': out[0], 'pnl_pct': out[2], 'ts': df.index[idx],
                'setup': 'ML', 'direction': 'LONG', 'entry': entry,
            })

        m = metrics(fold_trades, '')
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        results.append({'period': f"{start_s[:7]}/{end_s[5:7]}", 'n': m['n'], 'wr': m['wr'], 'pf': m['pf'], 'ok': ok})
        all_trades.extend(fold_trades)

    m_all = metrics(all_trades, name)
    eq, dd = equity_stats(all_trades)
    folds_ok = sum(1 for r in results if r['ok'])
    folds_data = sum(1 for r in results if r['n'] > 0)
    return {'name': name, 'folds_ok': folds_ok, 'folds_data': folds_data,
            'trades': all_trades, 'n': m_all['n'], 'wr': m_all['wr'],
            'pf': m_all['pf'], 'eq': eq, 'dd': dd}


# ==============================================================
# RULES WALK-FORWARD (generic TP/SL)
# ==============================================================
def rules_walk_forward(df, df_btc, regimes, tp_pct, sl_pct, timeout, name,
                       vol_min=1.2, bb_max=7.0, btc_corr_min=0.5):
    """Rule-based walk-forward: breakout + BTC-follower."""
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

            if detect_breakout(df, idx, vol_min, bb_max):
                out = sim_trade_fixed(df, idx, entry, tp_pct, sl_pct, max_bars=timeout)
                fold_trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                                    'setup': 'BRK', 'direction': 'LONG', 'entry': entry})
                triggered = True

            if not triggered:
                btc_idx = df_btc.index.get_indexer([ts], method='nearest')[0]
                if 0 <= btc_idx < len(df_btc) and detect_btc_breakout(df_btc, btc_idx):
                    corr_val = float(df['pair_btc_corr'].iloc[idx])
                    if corr_val >= btc_corr_min:
                        out = sim_trade_fixed(df, idx, entry, tp_pct, sl_pct, max_bars=timeout)
                        fold_trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                                            'setup': 'FBTC', 'direction': 'LONG', 'entry': entry})

        m = metrics(fold_trades, '')
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        results.append({'period': f"{start_s[:7]}/{end_s[5:7]}", 'n': m['n'], 'wr': m['wr'], 'pf': m['pf'], 'ok': ok})
        all_trades.extend(fold_trades)

    m_all = metrics(all_trades, name)
    eq, dd = equity_stats(all_trades)
    folds_ok = sum(1 for r in results if r['ok'])
    folds_data = sum(1 for r in results if r['n'] > 0)
    return {'name': name, 'folds_ok': folds_ok, 'folds_data': folds_data,
            'trades': all_trades, 'n': m_all['n'], 'wr': m_all['wr'],
            'pf': m_all['pf'], 'eq': eq, 'dd': dd}


# ==============================================================
# TRAILING WALK-FORWARD (reuses ML or rules entry signals)
# ==============================================================
def trailing_walk_forward(df, regimes, labels, sl_pct, name,
                          trail_trigger_pct=None, max_bars=30,
                          use_ml=True, threshold=0.50, df_btc=None):
    """Walk-forward with trailing stop exits."""
    results = []
    all_trades = []
    X_all = df[FEATURES_V14].values if use_ml else None
    y_all = labels if use_ml else None

    for fi, (start_s, end_s) in enumerate(WF_FOLDS):
        fold_start = pd.Timestamp(start_s, tz='UTC')
        fold_end = pd.Timestamp(end_s, tz='UTC')
        train_mask = df.index < fold_start
        test_mask = (df.index >= fold_start) & (df.index <= fold_end)
        test_indices = np.where(test_mask)[0]

        # Train ML if needed
        prob_rf = prob_gb = None
        if use_ml:
            X_train, y_train = X_all[train_mask], y_all[train_mask]
            if len(X_train) < 200:
                results.append({'period': f"{start_s[:7]}/{end_s[5:7]}", 'n': 0, 'wr': 0, 'pf': 0, 'ok': False})
                continue
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, min_samples_leaf=10, random_state=42)
            rf.fit(X_train_s, y_train)
            gb.fit(X_train_s, y_train)
            X_test_s = scaler.transform(X_all[test_mask])
            prob_rf = rf.predict_proba(X_test_s)[:, 1]
            prob_gb = gb.predict_proba(X_test_s)[:, 1]

        fold_trades = []
        for k, idx in enumerate(test_indices):
            if idx + max_bars + 1 >= len(df):
                continue
            reg = regimes.iloc[idx]
            if reg == 'BEAR':
                continue

            # Entry signal
            if use_ml:
                if prob_rf is None or prob_rf[k] < threshold or prob_gb[k] < threshold:
                    continue
            else:
                # Rule-based: breakout or BTC-follower
                ts = df.index[idx]
                is_brk = detect_breakout(df, idx, 1.2, 7.0)
                is_btc = False
                if not is_brk and df_btc is not None:
                    btc_idx = df_btc.index.get_indexer([ts], method='nearest')[0]
                    if 0 <= btc_idx < len(df_btc) and detect_btc_breakout(df_btc, btc_idx):
                        corr_val = float(df['pair_btc_corr'].iloc[idx])
                        is_btc = corr_val >= 0.5
                if not is_brk and not is_btc:
                    continue

            entry = float(df['close'].iloc[idx])
            out = sim_trade_trailing(df, idx, entry, sl_pct, trail_trigger_pct, max_bars)
            fold_trades.append({
                'outcome': out[0], 'pnl_pct': out[2], 'ts': df.index[idx],
                'setup': 'ML_TRAIL' if use_ml else 'RULE_TRAIL',
                'direction': 'LONG', 'entry': entry,
            })

        m = metrics(fold_trades, '')
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        results.append({'period': f"{start_s[:7]}/{end_s[5:7]}", 'n': m['n'], 'wr': m['wr'], 'pf': m['pf'], 'ok': ok})
        all_trades.extend(fold_trades)

    m_all = metrics(all_trades, name)
    eq, dd = equity_stats(all_trades)
    folds_ok = sum(1 for r in results if r['ok'])
    folds_data = sum(1 for r in results if r['n'] > 0)
    return {'name': name, 'folds_ok': folds_ok, 'folds_data': folds_data,
            'trades': all_trades, 'n': m_all['n'], 'wr': m_all['wr'],
            'pf': m_all['pf'], 'eq': eq, 'dd': dd}


# ==============================================================
# LIGHTGBM WALK-FORWARD
# ==============================================================
def lgbm_walk_forward(df, df_v7feat, regimes, tp_pct, sl_pct, timeout, name,
                      threshold=0.008, use_trailing=False, trail_sl=None, trail_trigger=None):
    """LightGBM regression walk-forward (V7 approach)."""
    if not HAS_LGBM:
        return None

    results = []
    all_trades = []
    fcols = [c for c in df_v7feat.columns if c in df_v7feat.columns]
    target = df['close'].shift(-5) / df['close'] - 1
    valid_mask = df_v7feat.notna().all(axis=1) & target.notna()
    X_all = df_v7feat[valid_mask].values
    y_all = target[valid_mask].values
    idx_map = np.where(valid_mask)[0]
    ts_all = df.index[valid_mask]

    for fi, (start_s, end_s) in enumerate(WF_FOLDS):
        fold_start = pd.Timestamp(start_s, tz='UTC')
        fold_end = pd.Timestamp(end_s, tz='UTC')
        train_sel = ts_all < fold_start
        test_sel = (ts_all >= fold_start) & (ts_all <= fold_end)

        if train_sel.sum() < 200:
            results.append({'period': f"{start_s[:7]}/{end_s[5:7]}", 'n': 0, 'wr': 0, 'pf': 0, 'ok': False})
            continue

        model = lgb.LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=1.0, num_leaves=31,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        model.fit(X_all[train_sel], y_all[train_sel])
        preds = model.predict(X_all[test_sel])
        test_idx = idx_map[test_sel]

        fold_trades = []
        for k, orig_idx in enumerate(test_idx):
            if orig_idx + timeout + 1 >= len(df):
                continue
            if preds[k] < threshold:
                continue
            reg = regimes.iloc[orig_idx]
            if reg == 'BEAR':
                continue

            entry = float(df['close'].iloc[orig_idx])

            if use_trailing and trail_sl is not None:
                out = sim_trade_trailing(df, orig_idx, entry, trail_sl, trail_trigger, max_bars=30)
            else:
                out = sim_trade_fixed(df, orig_idx, entry, tp_pct, sl_pct, max_bars=timeout)

            fold_trades.append({
                'outcome': out[0], 'pnl_pct': out[2], 'ts': df.index[orig_idx],
                'setup': 'LGBM', 'direction': 'LONG', 'entry': entry,
                'pred': float(preds[k]),
            })

        m = metrics(fold_trades, '')
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        results.append({'period': f"{start_s[:7]}/{end_s[5:7]}", 'n': m['n'], 'wr': m['wr'], 'pf': m['pf'], 'ok': ok})
        all_trades.extend(fold_trades)

    m_all = metrics(all_trades, name)
    eq, dd = equity_stats(all_trades)
    folds_ok = sum(1 for r in results if r['ok'])
    folds_data = sum(1 for r in results if r['n'] > 0)
    return {'name': name, 'folds_ok': folds_ok, 'folds_data': folds_data,
            'trades': all_trades, 'n': m_all['n'], 'wr': m_all['wr'],
            'pf': m_all['pf'], 'eq': eq, 'dd': dd}


# ==============================================================
# PRINT HELPERS
# ==============================================================
def print_result(r, prefix="  "):
    if r is None:
        print(f"{prefix}SKIPPED (lightgbm not available)")
        return "SKIP"
    passed = r['folds_ok'] >= 7 and r['pf'] >= 1.0
    marginal = r['folds_ok'] >= 6 and r['pf'] >= 1.0
    tag = "APROBADO" if passed else ("MARGINAL" if marginal else "RECHAZADO")
    mark = "**" if passed else ("* " if marginal else "  ")
    print(f"{prefix}{mark}{r['name']:<35} WF {r['folds_ok']:>2}/{r['folds_data']:>2} | "
          f"N={r['n']:>4} WR={r['wr']:.1%} PF={r['pf']:.2f} "
          f"$1K->${1000*r['eq']:.0f} DD={r['dd']:.1%} -> {tag}")
    return tag


# ==============================================================
# EVALUATE ONE PAIR
# ==============================================================
def evaluate_pair(pair):
    print(f"\n{'='*70}")
    print(f"  {pair}/USDT -- New Solutions Evaluation")
    print(f"{'='*70}")

    df, regimes, df_btc = load_pair_data(pair)
    all_results = []

    # ============================================================
    # PART 1: TP/SL Sweep
    # ============================================================
    print(f"\n--- PART 1: TP/SL Sweep ({pair}) ---")

    # 1A: Raw label WR
    print(f"\n  1A: Raw Label Win Rates")
    print(f"  {'Config':<15} | {'Raw WR':>7} | {'BE WR':>6} | {'Edge':>6} | Viable?")
    print(f"  {'-'*60}")

    viable_configs = []
    for cfg in TPSL_CONFIGS:
        labs = create_labels(df, cfg['tp'], cfg['sl'], cfg['timeout'])
        raw_wr = labs[:len(labs) - cfg['timeout'] - 1].mean()
        be_wr = cfg['sl'] / (cfg['tp'] + cfg['sl'])
        edge = raw_wr - be_wr
        viable = edge > 0.02  # need at least 2% edge
        mark = "YES" if viable else "no"
        print(f"  {cfg['name']:<15} | {raw_wr:>6.1%} | {be_wr:>5.1%} | {edge:>+5.1%} | {mark}")
        if viable:
            viable_configs.append((cfg, labs))

    if not viable_configs:
        print(f"  WARNING: No viable TP/SL config for {pair}!")
        # Still test top 3 by edge
        all_edges = []
        for cfg in TPSL_CONFIGS:
            labs = create_labels(df, cfg['tp'], cfg['sl'], cfg['timeout'])
            raw_wr = labs[:len(labs) - cfg['timeout'] - 1].mean()
            be_wr = cfg['sl'] / (cfg['tp'] + cfg['sl'])
            all_edges.append((raw_wr - be_wr, cfg, labs))
        all_edges.sort(reverse=True)
        viable_configs = [(cfg, labs) for _, cfg, labs in all_edges[:3]]

    # 1B: ML Walk-Forward for viable configs
    print(f"\n  1B: ML Walk-Forward (RF+GB, 10 features)")
    for cfg, labs in viable_configs:
        for th in [0.50, 0.55]:
            name = f"ML_{cfg['name']}_th{th}"
            r = ml_walk_forward(df, regimes, labs, cfg['tp'], cfg['sl'],
                                cfg['timeout'], name, threshold=th)
            tag = print_result(r)
            all_results.append((name, r, tag, cfg))

    # 1C: Rules Walk-Forward for viable configs
    print(f"\n  1C: Rule-Based Walk-Forward")
    for cfg, labs in viable_configs:
        name = f"RULES_{cfg['name']}"
        r = rules_walk_forward(df, df_btc, regimes, cfg['tp'], cfg['sl'],
                               cfg['timeout'], name)
        tag = print_result(r)
        all_results.append((name, r, tag, cfg))

    # ============================================================
    # PART 2: Trailing Stop
    # ============================================================
    print(f"\n--- PART 2: Trailing Stop ({pair}) ---")

    # Use the best TP/SL from Part 1 (or baseline) for entry labels
    best_cfg = viable_configs[0][0] if viable_configs else TPSL_CONFIGS[0]
    best_labs = viable_configs[0][1] if viable_configs else create_labels(df, 0.06, 0.04, 15)

    trail_configs = [
        ('TRAIL_imm',      best_cfg['sl'], None,              30),
        ('TRAIL_50pct',    best_cfg['sl'], best_cfg['tp']*0.5, 30),
        ('TRAIL_1.5pct',   best_cfg['sl'], 0.015,             30),
        ('TRAIL_tight_imm', best_cfg['sl']*0.7, None,         30),
    ]

    for tname, t_sl, t_trigger, t_bars in trail_configs:
        # ML + trailing
        name = f"ML_{best_cfg['name']}_{tname}"
        r = trailing_walk_forward(df, regimes, best_labs, t_sl, name,
                                  trail_trigger_pct=t_trigger, max_bars=t_bars,
                                  use_ml=True, threshold=0.50)
        tag = print_result(r)
        all_results.append((name, r, tag, best_cfg))

        # Rules + trailing
        name2 = f"RULES_{best_cfg['name']}_{tname}"
        r2 = trailing_walk_forward(df, regimes, best_labs, t_sl, name2,
                                   trail_trigger_pct=t_trigger, max_bars=t_bars,
                                   use_ml=False, df_btc=df_btc)
        tag2 = print_result(r2)
        all_results.append((name2, r2, tag2, best_cfg))

    # ============================================================
    # PART 3: Sizing-Adjusted DD
    # ============================================================
    print(f"\n--- PART 3: Sizing-Adjusted DD ({pair}) ---")

    # Include configs that pass WF >= 6
    candidates = [(n, r, t, c) for n, r, t, c in all_results if r['folds_ok'] >= 6 and r['n'] >= 10]

    if candidates:
        print(f"  {'Config':<35} | {'WF':>5} | {'1.0x DD':>7} | {'0.5x DD':>7} | {'0.4x DD':>7} | {'0.3x DD':>7} | 0.4x $1K->")
        print(f"  {'-'*105}")
        for name, r, tag, cfg in candidates:
            eq_03, dd_03 = compute_sized_equity(r['trades'], 0.3)
            eq_04, dd_04 = compute_sized_equity(r['trades'], 0.4)
            eq_05, dd_05 = compute_sized_equity(r['trades'], 0.5)
            ok_04 = " <25%" if dd_04 <= 0.25 else ""
            ok_03 = " <25%" if dd_03 <= 0.25 else ""
            print(f"  {name:<35} | {r['folds_ok']:>2}/{r['folds_data']:>2} | {r['dd']:>6.1%} | "
                  f"{dd_05:>6.1%} | {dd_04:>6.1%}{ok_04} | {dd_03:>6.1%}{ok_03} | "
                  f"${1000*eq_04:.0f}")
    else:
        print(f"  No configs with WF >= 6 for {pair}")

    # ============================================================
    # PART 4: LightGBM Regression
    # ============================================================
    print(f"\n--- PART 4: LightGBM Regression ({pair}) ---")

    if HAS_LGBM:
        print("  Computing V7 features (34)...")
        df_v7 = compute_v7_features(df)
        df_v7 = df_v7.replace([np.inf, -np.inf], np.nan)
        # Align with df index
        common = df.index.intersection(df_v7.dropna().index)
        df_v7 = df_v7.loc[common]

        lgbm_tp = best_cfg['tp']
        lgbm_sl = best_cfg['sl']
        lgbm_to = best_cfg['timeout']

        for th in [0.005, 0.008, 0.01]:
            # Fixed TP/SL
            name = f"LGBM_{best_cfg['name']}_th{th}"
            r = lgbm_walk_forward(df, df_v7, regimes, lgbm_tp, lgbm_sl, lgbm_to,
                                  name, threshold=th)
            if r:
                tag = print_result(r)
                all_results.append((name, r, tag, best_cfg))

            # Trailing
            name2 = f"LGBM_{best_cfg['name']}_TRAIL_th{th}"
            r2 = lgbm_walk_forward(df, df_v7, regimes, lgbm_tp, lgbm_sl, lgbm_to,
                                   name2, threshold=th, use_trailing=True,
                                   trail_sl=best_cfg['sl'], trail_trigger=best_cfg['tp']*0.5)
            if r2:
                tag2 = print_result(r2)
                all_results.append((name2, r2, tag2, best_cfg))
    else:
        print("  SKIPPED: lightgbm not installed")

    # ============================================================
    # PART 5: Summary + Best Combos
    # ============================================================
    print(f"\n--- PART 5: Summary ({pair}) ---")

    valid_results = [(n, r, t, c) for n, r, t, c in all_results if r['n'] >= 5]
    valid_results.sort(key=lambda x: x[1]['pf'] if x[1]['n'] >= 10 else 0, reverse=True)

    print(f"\n  {'Config':<40} | {'WF':>5} | {'N':>4} | {'WR':>6} | {'PF':>6} | {'$1K->':>7} | {'DD':>6} | Status")
    print(f"  {'-'*105}")
    for name, r, tag, cfg in valid_results[:20]:
        mark = "**" if tag == "APROBADO" else ("* " if tag == "MARGINAL" else "  ")
        print(f"  {mark}{name:<38} | {r['folds_ok']:>2}/{r['folds_data']:>2} | {r['n']:>4} | "
              f"{r['wr']:.1%} | {r['pf']:.2f} | ${1000*r['eq']:>6.0f} | {r['dd']:.1%} | {tag}")

    approved = [(n, r, t, c) for n, r, t, c in all_results if t == 'APROBADO' and r['n'] >= 10]
    marginal = [(n, r, t, c) for n, r, t, c in all_results if t == 'MARGINAL' and r['n'] >= 10]
    print(f"\n  APROBADO: {len(approved)} | MARGINAL: {len(marginal)}")

    # Find best with DD < 25% (using sizing)
    best_sized = []
    for name, r, tag, cfg in all_results:
        if r['folds_ok'] >= 6 and r['n'] >= 10:
            for sz in [1.0, 0.8, 0.6, 0.5, 0.4, 0.3]:
                eq_s, dd_s = compute_sized_equity(r['trades'], sz)
                if dd_s <= 0.25:
                    years = (df.index[-1] - df.index[0]).days / 365.25
                    ann_ret = (eq_s ** (1.0 / years) - 1) * 100 if eq_s > 0 and years > 0 else 0
                    best_sized.append((name, r, sz, dd_s, eq_s, ann_ret, cfg))
                    break

    if best_sized:
        print(f"\n  Configs with DD < 25% (sized):")
        print(f"  {'Config':<40} | {'WF':>5} | {'Sizing':>6} | {'DD':>6} | {'Ann%':>6} | {'$1K->':>7}")
        print(f"  {'-'*90}")
        best_sized.sort(key=lambda x: x[5], reverse=True)
        for name, r, sz, dd_s, eq_s, ann, cfg in best_sized[:10]:
            print(f"  {name:<40} | {r['folds_ok']:>2}/{r['folds_data']:>2} | {sz:>5.1f}x | "
                  f"{dd_s:>5.1%} | {ann:>5.1f}% | ${1000*eq_s:>6.0f}")

    # ============================================================
    # PART 6: OOS 2026
    # ============================================================
    print(f"\n--- PART 6: OOS 2026 ({pair}) ---")

    mask_oos = (df.index >= pd.Timestamp(OOS_START, tz='UTC')) & \
               (df.index < pd.Timestamp(OOS_END, tz='UTC'))
    if mask_oos.sum() > 0:
        p_start = float(df.loc[mask_oos, 'close'].iloc[0])
        p_end = float(df.loc[mask_oos, 'close'].iloc[-1])
        p_ret = (p_end / p_start - 1) * 100
        print(f"  {pair}: ${p_start:.4f} -> ${p_end:.4f} ({p_ret:+.1f}%)")
        oos_reg = regimes[mask_oos].value_counts().to_dict()
        print(f"  Regimes 2026: {oos_reg}")

    # Test top 5 configs on OOS
    oos_candidates = [(n, r, t, c) for n, r, t, c in all_results
                      if r['n'] >= 10 and r['pf'] >= 1.0]
    oos_candidates.sort(key=lambda x: x[1]['pf'], reverse=True)

    for name, r, tag, cfg in oos_candidates[:5]:
        # Determine approach from name
        is_trail = 'TRAIL' in name
        is_lgbm = 'LGBM' in name
        is_ml = name.startswith('ML_')
        is_rules = name.startswith('RULES_')

        tp_pct = cfg['tp']
        sl_pct = cfg['sl']
        timeout = cfg['timeout']

        train_mask = df.index <= pd.Timestamp(TRAIN_CUTOFF, tz='UTC')
        test_mask = mask_oos

        print(f"\n  --- {name} (OOS) ---")

        if is_lgbm and HAS_LGBM:
            # LightGBM OOS
            target = df['close'].shift(-5) / df['close'] - 1
            valid = df_v7.notna().all(axis=1) & target.notna()
            X_train = df_v7[valid & train_mask].values
            y_train = target[valid & train_mask].values
            if len(X_train) < 200:
                print("    Insufficient training data")
                continue
            model = lgb.LGBMRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.6, min_child_samples=20,
                reg_alpha=0.1, reg_lambda=1.0, num_leaves=31,
                random_state=42, n_jobs=-1, verbose=-1,
            )
            model.fit(X_train, y_train)
            test_valid = valid & test_mask
            if test_valid.sum() == 0:
                print("    No valid test bars")
                continue
            preds = model.predict(df_v7[test_valid].values)
            test_idx = np.where(test_valid)[0]
            th = 0.008
            if 'th0.005' in name: th = 0.005
            elif 'th0.01' in name: th = 0.01

            trades = []
            for k, idx in enumerate(test_idx):
                if idx + timeout + 1 >= len(df) or preds[k] < th:
                    continue
                if regimes.iloc[idx] == 'BEAR':
                    continue
                entry = float(df['close'].iloc[idx])
                if is_trail:
                    out = sim_trade_trailing(df, idx, entry, sl_pct, tp_pct * 0.5, 30)
                else:
                    out = sim_trade_fixed(df, idx, entry, tp_pct, sl_pct, max_bars=timeout)
                trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': df.index[idx],
                               'entry': entry, 'pred': float(preds[k])})

        elif is_ml or (is_trail and 'ML' in name):
            # ML OOS
            labs = create_labels(df, tp_pct, sl_pct, timeout)
            X_train_d = df.loc[train_mask, FEATURES_V14].values
            y_train_d = labs[np.where(train_mask)[0]]
            if len(X_train_d) < 200:
                print("    Insufficient training data")
                continue
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train_d)
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, min_samples_leaf=10, random_state=42)
            rf.fit(X_tr, y_train_d)
            gb.fit(X_tr, y_train_d)
            test_idx = np.where(test_mask)[0]
            X_te = scaler.transform(df.iloc[test_idx][FEATURES_V14].values)
            p_rf = rf.predict_proba(X_te)[:, 1]
            p_gb = gb.predict_proba(X_te)[:, 1]

            th = 0.50
            if 'th0.55' in name: th = 0.55

            trades = []
            for k, idx in enumerate(test_idx):
                if idx + timeout + 1 >= len(df):
                    continue
                if p_rf[k] < th or p_gb[k] < th:
                    continue
                if regimes.iloc[idx] == 'BEAR':
                    continue
                entry = float(df['close'].iloc[idx])
                if is_trail:
                    t_sl = sl_pct
                    t_trig = tp_pct * 0.5 if '50pct' in name else (0.015 if '1.5pct' in name else None)
                    out = sim_trade_trailing(df, idx, entry, t_sl, t_trig, 30)
                else:
                    out = sim_trade_fixed(df, idx, entry, tp_pct, sl_pct, max_bars=timeout)
                trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': df.index[idx],
                               'entry': entry})

        else:
            # Rules OOS
            test_idx = np.where(test_mask)[0]
            trades = []
            for idx in test_idx:
                if idx + timeout + 1 >= len(df):
                    continue
                if regimes.iloc[idx] == 'BEAR':
                    continue
                ts = df.index[idx]
                entry = float(df['close'].iloc[idx])
                is_brk = detect_breakout(df, idx, 1.2, 7.0)
                is_btc_f = False
                if not is_brk:
                    btc_idx = df_btc.index.get_indexer([ts], method='nearest')[0]
                    if 0 <= btc_idx < len(df_btc) and detect_btc_breakout(df_btc, btc_idx):
                        corr_val = float(df['pair_btc_corr'].iloc[idx])
                        is_btc_f = corr_val >= 0.5
                if not is_brk and not is_btc_f:
                    continue
                if is_trail:
                    t_sl = sl_pct
                    t_trig = tp_pct * 0.5 if '50pct' in name else (0.015 if '1.5pct' in name else None)
                    out = sim_trade_trailing(df, idx, entry, t_sl, t_trig, 30)
                else:
                    out = sim_trade_fixed(df, idx, entry, tp_pct, sl_pct, max_bars=timeout)
                trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': ts, 'entry': entry})

        if not trades:
            print("    No trades in OOS")
            continue
        for t in sorted(trades, key=lambda x: x['ts']):
            pred_s = f" pred={t['pred']:.4f}" if 'pred' in t else ""
            print(f"    {str(t['ts'])[:19]} ${t['entry']:>8.4f}{pred_s} "
                  f"{t['outcome']:<4} {t['pnl_pct']:>+7.2%}")
        m = metrics(trades, name)
        eq, dd = equity_stats(trades)
        total = sum(t['pnl_pct'] for t in trades)
        print(f"    N={m['n']} WR={m['wr']:.1%} PF={m['pf']:.2f} Total={total:+.2%} "
              f"vs {pair} B&H: {p_ret:+.1f}%")

    return all_results


# ==============================================================
# MAIN
# ==============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("ADA & SOL -- New Solutions Evaluation")
    print("4 approaches: Smaller TP/SL, Trailing Stop, Sizing, LightGBM")
    print("=" * 70)

    for pair in ['ADA', 'SOL']:
        results = evaluate_pair(pair)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
