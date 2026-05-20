"""
evaluate_sol_v15_dedicated.py -- SOL/USDT Dedicated ML Model
==============================================================
Historia: V14 ADA ensemble cross-applied a SOL = 9/10 WF folds, +250% PnL.
V2 GBR con 54 features = overfit (63% hist -> 12% OOS).
V15 rules = PF 1.80 pero WF 5/12.

Enfoque dedicado SOL:
  - ML ensemble (RF+GB) con 10-13 features (NO 54)
  - Entrenado EN datos SOL (expanding window WF)
  - LONG-only (SHORT fallo en TODAS las versiones)
  - TP/SL: 6%/4% (probado V14) + ATR-adaptive
  - Filtros: regimen, momentum (ret_3 > -0.03)
  - OOS 2026 (ene-mar) reservado

Parts:
  1. ML Ensemble V14-style (10 features, expanding window WF)
  2. ML + Regime filter (BULL/RANGE only)
  3. ML + Momentum filters
  4. SOL-enhanced features (13 features)
  5. Fixed vs ATR-based TP/SL comparison
  6. Best configs: full simulation con sim_trade_fixed
  7. OOS 2026

Usage:
  python evaluate_sol_v15_dedicated.py
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

# V14 proven TP/SL for SOL
TP_FIXED = 0.06
SL_FIXED = 0.04
TIMEOUT = 15

# Feature sets
FEATURES_V14 = ['rsi_n', 'macd_norm', 'adx_n', 'bb_pct', 'atr_pct_n',
                'ret_3', 'ret_5_n', 'ret_10_n', 'vol_ratio', 'trend']

FEATURES_SOL = FEATURES_V14 + ['ema20_slope_n', 'bb_width_n', 'range_pos']


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
    """Compute V14-style features + SOL-enhanced features."""
    out = df.copy()
    c = out['close']

    # V14 features (normalized 0-1 or ratio) -- use _n suffix to avoid overwriting raw columns
    out['rsi_n'] = out['rsi14'] / 100.0
    out['adx_n'] = out['adx14'] / 100.0
    # bb_pct already 0-1
    out['atr_pct_n'] = out['atr_pct'] / 100.0  # ratio (raw atr_pct stays as %)

    # MACD normalized
    macd_obj = pta.macd(c, fast=12, slow=26, signal=9)
    if macd_obj is not None:
        macd_line = macd_obj.iloc[:, 0]  # MACD line
        out['macd_norm'] = macd_line / c
    else:
        out['macd_norm'] = 0.0

    # Returns (as ratio, not %)
    out['ret_3'] = c.pct_change(3)
    out['ret_5_n'] = c.pct_change(5)
    out['ret_10_n'] = c.pct_change(10)

    # Trend: close > SMA50
    sma50 = c.rolling(50).mean()
    out['trend'] = (c > sma50).astype(float)

    # SOL-enhanced features
    out['ema20_slope_n'] = out['ema20_slope'] / 10.0  # normalize
    out['bb_width_n'] = out['bb_width'] / 100.0  # normalize
    # range_pos already 0-1

    return out.dropna(subset=FEATURES_SOL)


def create_labels(df, tp_pct=TP_FIXED, sl_pct=SL_FIXED, max_bars=TIMEOUT):
    """Create binary labels: 1=TP hit before SL within max_bars (matches sim_trade_fixed)."""
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
            # Check SL first (consistent with sim_trade_fixed)
            if lows[b] <= sl:
                # Same bar TP+SL: use close heuristic
                if highs[b] >= tp and closes[b] > (sl + tp) / 2:
                    labels[i] = 1
                break
            if highs[b] >= tp:
                labels[i] = 1
                break
    return labels


# ==============================================================
# DATA LOADING
# ==============================================================
def load_data():
    print("Loading SOL 4h data...")
    df_sol_raw = load_pair_4h('SOL')
    df_sol = compute_features_4h(df_sol_raw.copy())

    # Daily macro
    try:
        from v15_framework import load_pair_1d
        sol_1d = load_pair_1d('SOL')
    except (FileNotFoundError, Exception):
        print("  No daily data for SOL, resampling 4h -> 1d")
        sol_1d = df_sol_raw.resample('1D').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()
    sol_macro = compute_macro_daily(sol_1d)
    df_sol = merge_daily_to_4h(df_sol, sol_macro)

    # Extra V15 features
    c, v = df_sol['close'], df_sol['volume']
    df_sol['rsi_slope'] = df_sol['rsi14'].diff(3)
    vol_ma5 = v.rolling(5).mean()
    vol_ma20 = v.rolling(20).mean()
    df_sol['vol_slope'] = (vol_ma5 / vol_ma20.replace(0, np.nan) - 1) * 100
    up = (c > c.shift(1)).astype(int)
    df_sol['consec_up'] = up.rolling(8).sum()

    # Regimes
    regimes = df_sol.apply(lambda r: detect_regime(r), axis=1)

    # ML features (may drop some rows due to NaN)
    df_sol = compute_ml_features(df_sol)

    # Re-align regimes to df_sol's index after dropna
    regimes = regimes.reindex(df_sol.index)

    # Labels
    print("  Creating labels (TP=6%, SL=4%, timeout=15)...")
    labels = create_labels(df_sol, TP_FIXED, SL_FIXED, TIMEOUT)
    df_sol['label'] = labels

    # Stats
    win_rate_raw = labels[:len(labels)-TIMEOUT-1].mean()
    atr_mean = float(df_sol['atr_pct'].mean())
    print(f"  SOL: {len(df_sol)} bars ({df_sol.index[0].date()} to {df_sol.index[-1].date()})")
    print(f"  Raw label WR: {win_rate_raw:.1%} (TP=6%/SL=4%)")
    reg_counts = regimes.value_counts().to_dict()
    print(f"  Regimes: {reg_counts}")
    print(f"  ATR%: mean={atr_mean:.2f}%")

    return df_sol, regimes


# ==============================================================
# ML WALK-FORWARD ENGINE
# ==============================================================
def ml_walk_forward(df, regimes, feature_cols, name,
                    threshold=0.50, regime_filter=None,
                    momentum_filter=False, tp_pct=TP_FIXED, sl_pct=SL_FIXED):
    """
    Expanding-window walk-forward for ML ensemble.
    Train on ALL data before each fold, test on fold.
    """
    results = []
    all_trades = []
    all_probs = []

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
        avg_prob = (prob_rf + prob_gb) / 2

        # Generate signals and simulate
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

            # Momentum filter
            if momentum_filter:
                ret3 = float(df['ret_3'].iloc[idx])
                if ret3 < -0.03:
                    continue

            entry = float(df['close'].iloc[idx])

            # Simulate trade
            out = sim_trade_fixed(df, idx, entry, tp_pct, sl_pct, max_bars=TIMEOUT)
            fold_trades.append({
                'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                'setup': 'ML_ENS', 'direction': 'LONG', 'entry': entry,
                'prob': float(avg_prob[k]),
            })
            all_probs.append(float(avg_prob[k]))

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
        'name': name,
        'folds': results,
        'folds_ok': folds_ok,
        'folds_with_data': folds_with_data,
        'trades': all_trades,
        'n': m_all['n'], 'wr': m_all['wr'], 'pf': m_all['pf'],
        'eq': eq, 'dd': dd,
        'avg_prob': np.mean(all_probs) if all_probs else 0,
    }


def print_wf_result(r, show_folds=True):
    """Print walk-forward result."""
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
          f"PF={r['pf']:.2f} $1K->${1000*r['eq']:.0f} DD={r['dd']:.1%} "
          f"avg_prob={r['avg_prob']:.3f} -> {tag}")
    return tag


# ==============================================================
# ML WALK-FORWARD WITH ATR-BASED TP/SL
# ==============================================================
def ml_walk_forward_atr(df, regimes, feature_cols, name,
                        threshold=0.50, regime_filter=None,
                        momentum_filter=False,
                        tp_mult=3.0, sl_mult=2.0, tp_cap=0.12, sl_cap=0.07):
    """Same as ml_walk_forward but with ATR-based TP/SL per trade."""
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
            if momentum_filter:
                if float(df['ret_3'].iloc[idx]) < -0.03:
                    continue

            entry = float(df['close'].iloc[idx])
            # ATR-based TP/SL
            atr_pct_raw = float(df['atr_pct'].iloc[idx])  # already in %
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
        'eq': eq, 'dd': dd, 'avg_prob': 0,
    }


# ==============================================================
# OOS 2026
# ==============================================================
def run_oos_2026(df, regimes, feature_cols, name,
                 threshold=0.50, regime_filter=None, momentum_filter=False,
                 tp_pct=TP_FIXED, sl_pct=SL_FIXED, use_atr=False,
                 tp_mult=3.0, sl_mult=2.0, tp_cap=0.12, sl_cap=0.07):
    """Train on ALL pre-2026 data, test on 2026 OOS."""
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
        if momentum_filter:
            if float(df['ret_3'].iloc[idx]) < -0.03:
                continue

        entry = float(df['close'].iloc[idx])

        if use_atr:
            atr_pct_raw = float(df.iloc[idx].get('atr_pct', 0.035)) * 100
            sl_p = max(min(atr_pct_raw / 100 * sl_mult, sl_cap), 0.015)
            tp_p = max(min(atr_pct_raw / 100 * tp_mult, tp_cap), 0.025)
        else:
            tp_p, sl_p = tp_pct, sl_pct

        out = sim_trade_fixed(df, idx, entry, tp_p, sl_p, max_bars=TIMEOUT)
        trades.append({
            'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
            'setup': 'ML_ENS', 'direction': 'LONG', 'entry': entry,
            'prob': float((prob_rf[k] + prob_gb[k]) / 2),
        })

    return trades


# ==============================================================
# MAIN
# ==============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("SOL/USDT -- Dedicated ML Model Evaluation")
    print("=" * 70)

    df_sol, regimes = load_data()

    # Feature availability check
    for f in FEATURES_V14:
        if f not in df_sol.columns:
            print(f"  WARNING: feature '{f}' not found!")
    for f in FEATURES_SOL:
        if f not in df_sol.columns:
            print(f"  WARNING: SOL feature '{f}' not found!")

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
        r = ml_walk_forward(df_sol, regimes, FEATURES_V14, name, threshold=th)
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
        r = ml_walk_forward(df_sol, regimes, FEATURES_V14, name,
                            threshold=th, regime_filter=('BULL', 'RANGE'))
        tag = print_wf_result(r)
        configs_p2.append((name, r, tag))

    # ============================================================
    # PART 3: ML + Momentum filter (ret_3 > -0.03)
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 3: ML + Regime + Momentum filter (ret_3 > -0.03)")
    print("=" * 70)

    configs_p3 = []
    for th in [0.50, 0.55, 0.60]:
        name = f"ML_MOM_th{th}"
        print(f"\n  --- {name} ---")
        r = ml_walk_forward(df_sol, regimes, FEATURES_V14, name,
                            threshold=th, regime_filter=('BULL', 'RANGE'),
                            momentum_filter=True)
        tag = print_wf_result(r)
        configs_p3.append((name, r, tag))

    # ============================================================
    # PART 4: SOL-enhanced features (13 features)
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 4: SOL-enhanced features (13 features)")
    print("=" * 70)

    configs_p4 = []
    for th in [0.50, 0.55, 0.60]:
        # Without filters
        name = f"ML_SOL13_th{th}"
        print(f"\n  --- {name} ---")
        r = ml_walk_forward(df_sol, regimes, FEATURES_SOL, name, threshold=th)
        tag = print_wf_result(r, show_folds=False)
        configs_p4.append((name, r, tag))

        # With regime + momentum filters
        name2 = f"ML_SOL13_FILT_th{th}"
        print(f"\n  --- {name2} ---")
        r2 = ml_walk_forward(df_sol, regimes, FEATURES_SOL, name2,
                             threshold=th, regime_filter=('BULL', 'RANGE'),
                             momentum_filter=True)
        tag2 = print_wf_result(r2, show_folds=False)
        configs_p4.append((name2, r2, tag2))

    # ============================================================
    # PART 5: ATR-based TP/SL (best configs from above)
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 5: ATR-based TP/SL comparison")
    print("=" * 70)

    # Find best fixed TP/SL config
    all_configs = configs_p1 + configs_p2 + configs_p3 + configs_p4
    best_fixed = max(all_configs, key=lambda x: x[1]['pf'] if x[1]['n'] > 10 else 0)
    print(f"\n  Best fixed TP/SL: {best_fixed[0]} (PF={best_fixed[1]['pf']:.2f})")

    # ATR variants
    configs_p5 = []
    atr_profiles = [
        ('ATR_2.5x1.5', 2.5, 1.5, 0.10, 0.06),  # ETH-style
        ('ATR_3.0x2.0', 3.0, 2.0, 0.12, 0.07),   # SOL-wider
        ('ATR_2.0x1.5', 2.0, 1.5, 0.08, 0.06),   # Conservative
    ]

    for atr_name, tp_m, sl_m, tp_c, sl_c in atr_profiles:
        for th in [0.50, 0.55]:
            # V14 features + regime + momentum
            name = f"ML_{atr_name}_th{th}"
            print(f"\n  --- {name} ---")
            r = ml_walk_forward_atr(df_sol, regimes, FEATURES_V14, name,
                                     threshold=th, regime_filter=('BULL', 'RANGE'),
                                     momentum_filter=True,
                                     tp_mult=tp_m, sl_mult=sl_m,
                                     tp_cap=tp_c, sl_cap=sl_c)
            tag = print_wf_result(r, show_folds=False)
            configs_p5.append((name, r, tag))

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY: All Configurations")
    print("=" * 70)

    all_results = all_configs + configs_p5
    # Sort by PF (with minimum N filter)
    all_results.sort(key=lambda x: x[1]['pf'] if x[1]['n'] >= 10 else 0, reverse=True)

    print(f"\n  {'Config':<28} | {'WF':>5} | {'N':>4} | {'WR':>6} | {'PF':>6} | {'$1K->':>7} | {'DD':>6} | Veredicto")
    print(f"  {'-'*90}")

    for name, r, tag in all_results:
        if r['n'] < 3:
            continue
        mark = "**" if tag == "APROBADO" else ("* " if tag == "MARGINAL" else "  ")
        print(f"  {mark}{name:<26} | {r['folds_ok']:>2}/{r['folds_with_data']:>2} | {r['n']:>4} | "
              f"{r['wr']:.1%} | {r['pf']:.2f} | ${1000*r['eq']:>6.0f} | {r['dd']:.1%} | {tag}")

    # ============================================================
    # PART 6: OOS 2026
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 6: OOS 2026 (Jan-Mar, trained on pre-2026)")
    print("=" * 70)

    # Market context
    mask_oos = (df_sol.index >= pd.Timestamp(OOS_START, tz='UTC')) & \
               (df_sol.index < pd.Timestamp(OOS_END, tz='UTC'))
    if mask_oos.sum() > 0:
        sol_start = float(df_sol.loc[mask_oos, 'close'].iloc[0])
        sol_end = float(df_sol.loc[mask_oos, 'close'].iloc[-1])
        sol_ret = (sol_end / sol_start - 1) * 100
        print(f"  SOL: ${sol_start:.2f} -> ${sol_end:.2f} ({sol_ret:+.1f}%)")
        oos_reg = regimes[mask_oos].value_counts().to_dict()
        print(f"  Regimes 2026: {oos_reg}")
    else:
        print("  No SOL data in 2026 period!")
        sol_ret = 0

    # Test top configs on OOS
    # Pick top 5 by PF with N >= 10
    top_configs = [(n, r, t) for n, r, t in all_results if r['n'] >= 10][:5]

    oos_results = []
    for name, r, tag in top_configs:
        # Parse config from name
        is_atr = 'ATR' in name
        is_regime = 'REGIME' in name or 'MOM' in name or 'FILT' in name or is_atr
        is_momentum = 'MOM' in name or 'FILT' in name or is_atr
        is_sol13 = 'SOL13' in name
        feat_cols = FEATURES_SOL if is_sol13 else FEATURES_V14

        # Parse threshold
        th = 0.50
        if 'th0.55' in name: th = 0.55
        elif 'th0.60' in name: th = 0.60

        regime_f = ('BULL', 'RANGE') if is_regime else None
        mom_f = is_momentum

        if is_atr:
            # Parse ATR params from name
            if '3.0x2.0' in name:
                tp_m, sl_m, tp_c, sl_c = 3.0, 2.0, 0.12, 0.07
            elif '2.0x1.5' in name:
                tp_m, sl_m, tp_c, sl_c = 2.0, 1.5, 0.08, 0.06
            else:
                tp_m, sl_m, tp_c, sl_c = 2.5, 1.5, 0.10, 0.06
            trades = run_oos_2026(df_sol, regimes, feat_cols, name,
                                  threshold=th, regime_filter=regime_f,
                                  momentum_filter=mom_f, use_atr=True,
                                  tp_mult=tp_m, sl_mult=sl_m,
                                  tp_cap=tp_c, sl_cap=sl_c)
        else:
            trades = run_oos_2026(df_sol, regimes, feat_cols, name,
                                  threshold=th, regime_filter=regime_f,
                                  momentum_filter=mom_f)

        print(f"\n  --- {name} (OOS 2026) ---")
        if not trades:
            print("    No trades in OOS period")
            oos_results.append((name, 0, 0, 0, 0, 0))
            continue

        for t in sorted(trades, key=lambda x: x['ts']):
            print(f"    {str(t['ts'])[:19]:<22} ${t['entry']:>8.2f} "
                  f"prob={t['prob']:.2f} {t['outcome']:<4} {t['pnl_pct']:>+7.2%}")

        m = metrics(trades, name)
        eq, dd = equity_stats(trades)
        total_pnl = sum(t['pnl_pct'] for t in trades)
        print(f"    N={m['n']} WR={m['wr']:.1%} PF={m['pf']:.2f} "
              f"$1K->${1000*eq:.0f} DD={dd:.1%} Total={total_pnl:+.2%}")
        print(f"    vs SOL buy-and-hold: {sol_ret:+.1f}%")
        oos_results.append((name, m['n'], m['wr'], m['pf'], eq, total_pnl))

    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    approved = [(n, r, t) for n, r, t in all_results if t == 'APROBADO' and r['n'] >= 10]
    marginal = [(n, r, t) for n, r, t in all_results if t == 'MARGINAL' and r['n'] >= 10]

    if approved:
        print(f"\n  APROBADO ({len(approved)} configs):")
        for name, r, _ in approved:
            print(f"    {name}: WF {r['folds_ok']}/{r['folds_with_data']} "
                  f"PF={r['pf']:.2f} WR={r['wr']:.1%} DD={r['dd']:.1%} N={r['n']}")
    elif marginal:
        print(f"\n  MARGINAL ({len(marginal)} configs):")
        for name, r, _ in marginal:
            print(f"    {name}: WF {r['folds_ok']}/{r['folds_with_data']} "
                  f"PF={r['pf']:.2f} WR={r['wr']:.1%} DD={r['dd']:.1%} N={r['n']}")
    else:
        print("\n  RECHAZADO: Ningun config pasa WF >= 7 + PF >= 1.0")

    # Anti-overfitting: compare V14 features vs SOL-enhanced
    v14_results = [r for _, r, _ in configs_p1 + configs_p2 + configs_p3 if r['n'] >= 10]
    sol_results = [r for _, r, _ in configs_p4 if r['n'] >= 10]
    if v14_results and sol_results:
        best_v14_pf = max(r['pf'] for r in v14_results)
        best_sol_pf = max(r['pf'] for r in sol_results)
        print(f"\n  Feature comparison: V14(10feat) best PF={best_v14_pf:.2f} vs SOL(13feat) best PF={best_sol_pf:.2f}")
        if best_sol_pf > best_v14_pf * 1.2:
            print("    WARNING: SOL features significantly better -> may indicate overfitting to extra features")
        else:
            print("    OK: similar performance -> extra features not adding overfit risk")

    print("=" * 70)
