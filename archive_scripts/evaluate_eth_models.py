"""
evaluate_eth_models.py — ETH Multi-Model Evaluation with ETH-Specific Features
================================================================================
ETH failed with BTC features (AUC ~0.5). This script tests a different approach:

NEW features (not used before):
  1. ETH/BTC ratio (momentum relativo, slope, z-score)
  2. BTC as predictor (ret, RSI, vol_ratio)
  3. Volatility regime (ATR percentile, vol clustering)
  4. Mean-reversion (VWAP distance, price z-score)
  5. Microstructure (range ratio, close position, consecutive candles)

Models tested (walk-forward 12 folds):
  1. GradientBoosting (baseline, same as BTC)
  2. RandomForest
  3. LightGBM
  4. XGBoost
  5. Stacking (meta-model combining 1-4)

Success criteria:
  - WF >= 7/12 folds positive
  - WR > 40%
  - PF > 1.2
  - If NONE passes -> ETH rejected definitively

Usage:
  python evaluate_eth_models.py
  python evaluate_eth_models.py --direction short
  python evaluate_eth_models.py --tp 0.04 --sl 0.02
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pandas_ta as pta
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from v15_framework import (
    load_pair_4h, load_btc_4h, compute_features_4h,
    compute_macro_daily, merge_daily_to_4h,
    sim_trade_fixed, metrics, print_metrics,
    WF_FOLDS, OOS_START, OOS_END, COMMISSION,
)

# ============================================================
# CONFIG
# ============================================================
DIRECTION = 'long'  # overridden by --direction
TP_PCT = 0.03
SL_PCT = 0.015
MAX_BARS = 16

# ============================================================
# ETH-SPECIFIC FEATURES
# ============================================================

def compute_eth_base_features(df_eth: pd.DataFrame) -> pd.DataFrame:
    """Standard TA features (same as BTC V15 baseline)."""
    return compute_features_4h(df_eth)


def compute_ethbtc_features(df_eth: pd.DataFrame, df_btc: pd.DataFrame) -> pd.DataFrame:
    """
    ETH/BTC ratio features — momentum relativo ETH vs BTC.
    These capture ETH-specific alpha vs just following BTC.
    """
    feat = pd.DataFrame(index=df_eth.index)

    # Align BTC to ETH index
    btc_close = df_btc['close'].reindex(df_eth.index, method='ffill')
    eth_close = df_eth['close']

    # ETH/BTC ratio
    ratio = eth_close / btc_close.replace(0, np.nan)
    feat['ethbtc_ratio'] = ratio
    feat['ethbtc_ratio_ma20'] = ratio.rolling(20).mean()

    # Ratio momentum: is ETH outperforming BTC?
    feat['ethbtc_slope_5'] = ratio.pct_change(5) * 100
    feat['ethbtc_slope_20'] = ratio.pct_change(20) * 100

    # Ratio z-score (mean-reversion signal)
    ratio_mean = ratio.rolling(60).mean()
    ratio_std = ratio.rolling(60).std()
    feat['ethbtc_zscore'] = ((ratio - ratio_mean) / ratio_std.clip(lower=1e-8)).clip(-4, 4)

    return feat


def compute_btc_cross_features(df_btc: pd.DataFrame, df_eth: pd.DataFrame) -> pd.DataFrame:
    """
    BTC as a leading indicator for ETH.
    BTC often moves first; ETH follows with a lag.
    """
    feat = pd.DataFrame(index=df_eth.index)

    btc_close = df_btc['close'].reindex(df_eth.index, method='ffill')
    btc_high = df_btc['high'].reindex(df_eth.index, method='ffill')
    btc_low = df_btc['low'].reindex(df_eth.index, method='ffill')
    btc_vol = df_btc['volume'].reindex(df_eth.index, method='ffill')

    # BTC returns (leading signal)
    feat['btc_ret_1'] = btc_close.pct_change(1) * 100
    feat['btc_ret_5'] = btc_close.pct_change(5) * 100

    # BTC RSI
    feat['btc_rsi14'] = pta.rsi(btc_close, length=14)

    # BTC volume ratio
    btc_vol_ma = btc_vol.rolling(20).mean()
    feat['btc_vol_ratio'] = btc_vol / btc_vol_ma.replace(0, np.nan)

    # Correlation ETH-BTC rolling 20 bars
    eth_ret = df_eth['close'].pct_change()
    btc_ret = btc_close.pct_change()
    feat['eth_btc_corr_20'] = eth_ret.rolling(20).corr(btc_ret)

    return feat


def compute_volatility_features(df_eth: pd.DataFrame) -> pd.DataFrame:
    """
    Volatility regime features — ETH-specific vol clustering.
    ETH is more volatile than BTC; modeling this is key.
    """
    h, l, c = df_eth['high'], df_eth['low'], df_eth['close']
    feat = pd.DataFrame(index=df_eth.index)

    # ATR percentile (vs last 60 bars) — is current vol high or low?
    atr = pta.atr(h, l, c, length=14)
    atr_pct = atr / c * 100
    feat['atr_pct_rank'] = atr_pct.rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # Realized volatility (20-bar rolling std of returns)
    ret = c.pct_change()
    feat['realized_vol_20'] = ret.rolling(20).std() * 100

    # Vol of vol (is vol itself unstable?)
    feat['vol_of_vol'] = feat['realized_vol_20'].rolling(20).std()

    # High-low range as % of close
    feat['hl_range_pct'] = (h - l) / c * 100

    return feat


def compute_meanrev_features(df_eth: pd.DataFrame) -> pd.DataFrame:
    """
    Mean-reversion signals — ETH tends to revert more than BTC.
    """
    c = df_eth['close']
    v = df_eth['volume']
    feat = pd.DataFrame(index=df_eth.index)

    # VWAP proxy (20-bar volume-weighted average)
    vwap_20 = (c * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    feat['vwap20_dist'] = (c - vwap_20) / vwap_20 * 100

    # Price z-score (how far from 20-bar mean)
    c_mean = c.rolling(20).mean()
    c_std = c.rolling(20).std()
    feat['price_zscore'] = ((c - c_mean) / c_std.clip(lower=1e-8)).clip(-4, 4)

    # Distance from 50-bar midpoint
    mid_50 = (df_eth['high'].rolling(50).max() + df_eth['low'].rolling(50).min()) / 2
    feat['mid50_dist'] = (c - mid_50) / mid_50 * 100

    return feat


def compute_microstructure_features(df_eth: pd.DataFrame) -> pd.DataFrame:
    """
    Candle microstructure — captures ETH-specific intrabar patterns.
    """
    o, h, l, c = df_eth['open'], df_eth['high'], df_eth['low'], df_eth['close']
    feat = pd.DataFrame(index=df_eth.index)

    # Close position in range (0=low, 1=high)
    hl_range = (h - l).clip(lower=1e-10)
    feat['close_in_range'] = (c - l) / hl_range

    # Body to range ratio
    feat['body_range_ratio'] = (c - o).abs() / hl_range

    # Upper/lower shadow ratio
    feat['shadow_ratio'] = np.where(
        (h - c.clip(lower=o)) > 0,
        (h - pd.concat([o, c], axis=1).max(axis=1)) /
        (pd.concat([o, c], axis=1).min(axis=1) - l).clip(lower=1e-10),
        1.0
    )
    feat['shadow_ratio'] = feat['shadow_ratio'].clip(0, 10)

    # Consecutive direction (bullish/bearish streak)
    direction = np.sign(c - o)
    feat['consec_bull'] = (direction == 1).astype(int).rolling(5).sum()
    feat['consec_bear'] = (direction == -1).astype(int).rolling(5).sum()

    # Gap from previous close
    feat['gap_pct'] = (o - c.shift(1)) / c.shift(1) * 100

    return feat


# ============================================================
# FULL FEATURE PIPELINE
# ============================================================

# All ETH-specific feature names (for reference)
ETH_SPECIFIC_FEATURES = [
    # ETH/BTC ratio
    'ethbtc_ratio', 'ethbtc_ratio_ma20', 'ethbtc_slope_5',
    'ethbtc_slope_20', 'ethbtc_zscore',
    # BTC cross
    'btc_ret_1', 'btc_ret_5', 'btc_rsi14', 'btc_vol_ratio', 'eth_btc_corr_20',
    # Volatility
    'atr_pct_rank', 'realized_vol_20', 'vol_of_vol', 'hl_range_pct',
    # Mean-reversion
    'vwap20_dist', 'price_zscore', 'mid50_dist',
    # Microstructure
    'close_in_range', 'body_range_ratio', 'shadow_ratio',
    'consec_bull', 'consec_bear', 'gap_pct',
]

# Base TA features from compute_features_4h (subset that's most useful)
BASE_TA_FEATURES = [
    'ema20_slope', 'ema50_slope', 'ema200_dist',
    'rsi14', 'atr_pct', 'bb_pct', 'bb_width',
    'adx14', 'di_diff', 'vol_ratio', 'range_pos',
    'ret_1', 'ret_5',
]

# Combined feature list for ML (~35 features)
ALL_FEATURES = BASE_TA_FEATURES + ETH_SPECIFIC_FEATURES


def build_eth_features(df_eth: pd.DataFrame, df_btc: pd.DataFrame) -> pd.DataFrame:
    """Build complete feature matrix for ETH with all feature groups."""
    # Base TA
    df_ta = compute_eth_base_features(df_eth)

    # Add extra features needed (rsi_slope, vol_slope, etc.)
    c, v = df_ta['close'], df_ta['volume']
    df_ta['rsi_slope'] = df_ta['rsi14'].diff(3)
    vol_ma5 = v.rolling(5).mean()
    vol_ma20 = v.rolling(20).mean()
    df_ta['vol_slope'] = (vol_ma5 / vol_ma20.replace(0, np.nan) - 1) * 100

    # Daily macro (from ETH itself, not BTC — ETH has its own structure)
    df_daily = compute_macro_daily(df_ta)
    df_ta = merge_daily_to_4h(df_ta, df_daily)

    # ETH-specific feature groups
    ethbtc_feat = compute_ethbtc_features(df_eth, df_btc)
    btc_cross = compute_btc_cross_features(df_btc, df_eth)
    vol_feat = compute_volatility_features(df_eth)
    mr_feat = compute_meanrev_features(df_eth)
    micro_feat = compute_microstructure_features(df_eth)

    # Merge all
    result = df_ta.copy()
    for feat_df in [ethbtc_feat, btc_cross, vol_feat, mr_feat, micro_feat]:
        for col in feat_df.columns:
            result[col] = feat_df[col]

    # Clean infinities
    result = result.replace([np.inf, -np.inf], np.nan)

    # Drop rows without minimum features
    result = result.dropna(subset=['ema20', 'ema50', 'rsi14', 'atr_pct'])

    return result


# ============================================================
# LABELS
# ============================================================

def create_labels(df, direction='long', tp_pct=0.03, sl_pct=0.015, max_bars=16):
    """Binary label: 1 if TP hit before SL within max_bars."""
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    labels = np.full(n, np.nan)

    for i in range(n - max_bars - 1):
        entry = closes[i]
        if direction == 'long':
            tp = entry * (1 + tp_pct)
            sl = entry * (1 - sl_pct)
            for j in range(i + 1, i + max_bars + 1):
                if highs[j] >= tp:
                    labels[i] = 1
                    break
                if lows[j] <= sl:
                    labels[i] = 0
                    break
            else:
                labels[i] = 1 if closes[i + max_bars] > entry else 0
        else:  # short
            tp = entry * (1 - tp_pct)
            sl = entry * (1 + sl_pct)
            for j in range(i + 1, i + max_bars + 1):
                if lows[j] <= tp:
                    labels[i] = 1
                    break
                if highs[j] >= sl:
                    labels[i] = 0
                    break
            else:
                labels[i] = 1 if closes[i + max_bars] < entry else 0

    return pd.Series(labels, index=df.index)


# ============================================================
# MODELS
# ============================================================

def get_models():
    """Return dict of model name -> (constructor, needs_scaling)."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression

    models = {}

    models['GBM'] = (
        lambda: GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8, random_state=42
        ),
        True  # needs scaling
    )

    models['RF'] = (
        lambda: RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=15,
            max_features='sqrt', random_state=42, n_jobs=-1
        ),
        False
    )

    try:
        import lightgbm as lgb
        models['LGBM'] = (
            lambda: lgb.LGBMClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1, n_jobs=-1
            ),
            False
        )
    except ImportError:
        print("  WARNING: lightgbm not installed, skipping LGBM")

    try:
        import xgboost as xgb
        models['XGB'] = (
            lambda: xgb.XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0, n_jobs=-1,
                eval_metric='logloss'
            ),
            False
        )
    except ImportError:
        print("  WARNING: xgboost not installed, skipping XGB")

    # Stacking: combine GBM + RF + (LGBM or XGB if available)
    base_estimators = [
        ('gbm', GradientBoostingClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8, random_state=42
        )),
        ('rf', RandomForestClassifier(
            n_estimators=150, max_depth=5, min_samples_leaf=15,
            max_features='sqrt', random_state=42, n_jobs=-1
        )),
    ]
    try:
        import xgboost as xgb
        base_estimators.append(('xgb', xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            min_child_weight=20, subsample=0.8, random_state=42,
            verbosity=0, eval_metric='logloss'
        )))
    except ImportError:
        pass

    models['STACK'] = (
        lambda: StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(C=1.0, max_iter=500),
            cv=3, n_jobs=-1, passthrough=False
        ),
        True  # meta-learner benefits from scaling
    )

    return models


# ============================================================
# WALK-FORWARD EVALUATION (ML, expanding window)
# ============================================================

def walk_forward_ml(df, labels, features, model_constructor, needs_scaling,
                    threshold=0.50, direction='long',
                    tp_pct=0.03, sl_pct=0.015, max_bars=16):
    """
    Walk-forward with expanding training window.
    Returns fold results + all OOS trades.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    results = []
    all_trades = []

    for fold_idx, (start_s, end_s) in enumerate(WF_FOLDS):
        test_mask = (df.index >= start_s) & (df.index <= end_s)
        train_mask = df.index < start_s

        period_label = f"{start_s[:7]}/{end_s[5:7]}"

        # Training data
        y_train = labels[train_mask]
        valid_train = y_train.notna()
        X_train = df.loc[train_mask, features][valid_train].fillna(0)
        y_train = y_train[valid_train]

        # Need minimum training data
        min_samples = 500
        min_positive = 20
        if len(X_train) < min_samples or y_train.sum() < min_positive or (len(y_train) - y_train.sum()) < min_positive:
            results.append({
                'period': period_label, 'n': 0, 'wr': 0, 'pf': 0,
                'ok': False, 'auc': 0, 'signals': 0, 'note': 'insufficient_data'
            })
            continue

        # Scale if needed
        scaler = None
        X_tr = X_train.values
        if needs_scaling:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)

        # Train
        model = model_constructor()
        try:
            model.fit(X_tr, y_train)
        except Exception as e:
            results.append({
                'period': period_label, 'n': 0, 'wr': 0, 'pf': 0,
                'ok': False, 'auc': 0, 'signals': 0, 'note': f'train_error: {e}'
            })
            continue

        # Test data
        X_test = df.loc[test_mask, features].fillna(0)
        y_test = labels[test_mask]
        valid_test = y_test.notna()
        X_test_valid = X_test[valid_test]
        y_test_valid = y_test[valid_test]

        if len(X_test_valid) == 0:
            results.append({
                'period': period_label, 'n': 0, 'wr': 0, 'pf': 0,
                'ok': False, 'auc': 0, 'signals': 0, 'note': 'no_test_data'
            })
            continue

        X_te = X_test_valid.values
        if scaler is not None:
            X_te = scaler.transform(X_te)

        # Predict
        probs = model.predict_proba(X_te)[:, 1]

        # AUC
        try:
            auc = roc_auc_score(y_test_valid, probs)
        except Exception:
            auc = 0.5

        # Simulate trades on signals above threshold
        signal_mask = probs >= threshold
        n_signals = signal_mask.sum()

        trades = []
        if n_signals > 0:
            signal_indices = X_test_valid.index[signal_mask]
            for ts in signal_indices:
                global_i = df.index.get_loc(ts)
                if global_i + max_bars >= len(df):
                    continue
                entry = float(df['close'].iloc[global_i])

                if direction == 'long':
                    out = sim_trade_fixed(df, global_i, entry, tp_pct, sl_pct, max_bars)
                else:
                    out = sim_short(df, global_i, entry, tp_pct, sl_pct, max_bars)

                trades.append({
                    'outcome': out[0], 'pnl_pct': out[2],
                    'ts': ts, 'prob': float(probs[list(X_test_valid.index).index(ts)]),
                })

        m = metrics(trades, period_label)
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)

        results.append({
            'period': period_label, 'n': m['n'], 'wr': m['wr'], 'pf': m['pf'],
            'ok': ok, 'auc': auc, 'signals': n_signals,
        })
        all_trades.extend(trades)

    folds_ok = sum(1 for r in results if r['ok'])
    return {
        'folds': results, 'folds_ok': folds_ok,
        'folds_total': len(results), 'approved': folds_ok >= 7,
        'all_trades': all_trades,
    }


# ============================================================
# SHORT SIMULATION
# ============================================================

def sim_short(df, entry_bar, entry_price, tp_pct, sl_pct, max_bars=16):
    """Simulate a SHORT trade."""
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


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(df, labels, features, model_constructor, needs_scaling,
                        direction, tp_pct, sl_pct, max_bars):
    """Test thresholds 0.45-0.70 and return best by PF."""
    best_t, best_pf, best_wr, best_n = 0.50, 0, 0, 0
    print("\n  Threshold search:")
    for t in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        wf = walk_forward_ml(
            df, labels, features, model_constructor, needs_scaling,
            threshold=t, direction=direction,
            tp_pct=tp_pct, sl_pct=sl_pct, max_bars=max_bars
        )
        oos_trades = [tr for tr in wf['all_trades']
                      if OOS_START <= str(tr['ts'])[:10] <= OOS_END]
        m = metrics(oos_trades, f't={t}')
        mark = '*' if m['pf'] > best_pf and m['n'] >= 10 else ' '
        print(f"    {mark} t={t:.2f} | N={m['n']:>4} | WR={m['wr']:.1%} | "
              f"PF={m['pf']:.2f} | folds={wf['folds_ok']}/12")
        if m['pf'] > best_pf and m['n'] >= 10:
            best_t, best_pf, best_wr, best_n = t, m['pf'], m['wr'], m['n']

    return best_t


# ============================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================

def analyze_features(df, labels, features, model_constructor, needs_scaling):
    """Train on all data before 2024 and show feature importance."""
    from sklearn.preprocessing import StandardScaler

    train_mask = df.index < '2024-01-01'
    y = labels[train_mask]
    valid = y.notna()
    X = df.loc[train_mask, features][valid].fillna(0)
    y = y[valid]

    if needs_scaling:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values

    model = model_constructor()
    model.fit(X_scaled, y)

    # Feature importance (works for tree-based models)
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=features)
        print("\n  Top 15 features:")
        for feat, val in imp.nlargest(15).items():
            tag = '[ETH]' if feat in ETH_SPECIFIC_FEATURES else '[TA]'
            print(f"    {tag} {feat:<25} {val:.4f}")
        return imp
    return None


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='ETH Multi-Model Evaluation')
    parser.add_argument('--direction', default='long', choices=['long', 'short'])
    parser.add_argument('--tp', type=float, default=0.03, help='Take profit %')
    parser.add_argument('--sl', type=float, default=0.015, help='Stop loss %')
    parser.add_argument('--max-bars', type=int, default=16)
    parser.add_argument('--threshold', type=float, default=0.50)
    parser.add_argument('--search-threshold', action='store_true',
                        help='Search best threshold per model')
    args = parser.parse_args()

    direction = args.direction
    tp_pct = args.tp
    sl_pct = args.sl
    max_bars = args.max_bars
    threshold = args.threshold

    print("=" * 70)
    print(f"ETH MULTI-MODEL EVALUATION — {direction.upper()}")
    print(f"TP={tp_pct:.1%} SL={sl_pct:.1%} MaxBars={max_bars} Threshold={threshold}")
    print("=" * 70)

    # ---- Load data ----
    print("\nLoading ETH 4H...")
    df_eth_raw = load_pair_4h('ETH')
    print(f"  ETH: {len(df_eth_raw)} bars ({df_eth_raw.index[0].date()} - {df_eth_raw.index[-1].date()})")

    print("Loading BTC 4H (for cross-features)...")
    df_btc_raw = load_btc_4h()
    print(f"  BTC: {len(df_btc_raw)} bars ({df_btc_raw.index[0].date()} - {df_btc_raw.index[-1].date()})")

    # ---- Build features ----
    print("\nBuilding ETH features (~35 features)...")
    df = build_eth_features(df_eth_raw, df_btc_raw)
    print(f"  Result: {len(df)} bars, {len(ALL_FEATURES)} ML features")

    # Verify all features exist
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        print(f"  WARNING: missing features: {missing}")
        available = [f for f in ALL_FEATURES if f in df.columns]
    else:
        available = ALL_FEATURES
    print(f"  Available features: {len(available)}")

    # ---- Feature stats ----
    print("\n  Feature coverage (non-NaN %):")
    for feat in available[:5]:
        pct = df[feat].notna().mean()
        print(f"    {feat:<25} {pct:.1%}")
    print(f"    ... ({len(available) - 5} more)")

    # ---- Labels ----
    print(f"\nCreating {direction.upper()} labels (TP={tp_pct:.1%}, SL={sl_pct:.1%})...")
    labels = create_labels(df, direction=direction, tp_pct=tp_pct,
                          sl_pct=sl_pct, max_bars=max_bars)
    valid_labels = labels[labels.notna()]
    base_rate = valid_labels.mean()
    print(f"  Base rate: {base_rate:.1%} ({int(valid_labels.sum())}/{len(valid_labels)})")
    break_even = sl_pct / (tp_pct + sl_pct)
    print(f"  Break-even WR: {break_even:.1%}")

    # ---- Models ----
    print("\nLoading models...")
    all_models = get_models()
    print(f"  Models: {list(all_models.keys())}")

    # ---- Walk-forward per model ----
    summary = []

    for model_name, (constructor, needs_scaling) in all_models.items():
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"{'='*70}")

        # Feature importance
        if model_name in ('GBM', 'XGB', 'RF', 'LGBM'):
            analyze_features(df, labels, available, constructor, needs_scaling)

        # Threshold search if requested
        if args.search_threshold:
            best_t = find_best_threshold(
                df, labels, available, constructor, needs_scaling,
                direction, tp_pct, sl_pct, max_bars
            )
            print(f"  -> Best threshold: {best_t}")
            use_threshold = best_t
        else:
            use_threshold = threshold

        # Walk-forward
        print(f"\n  Walk-forward (threshold={use_threshold:.2f})...")
        wf = walk_forward_ml(
            df, labels, available, constructor, needs_scaling,
            threshold=use_threshold, direction=direction,
            tp_pct=tp_pct, sl_pct=sl_pct, max_bars=max_bars
        )

        # Print fold results
        print(f"\n  {'Period':<14} | {'N':>4} | {'Sig':>4} | {'WR':>7} | {'PF':>6} | {'AUC':>5} | OK")
        print("  " + "-" * 60)
        for r in wf['folds']:
            ok_s = '+' if r['ok'] else '-'
            wr_s = f"{r['wr']:.1%}" if r['n'] > 0 else 'n/a'
            pf_s = f"{r['pf']:.2f}" if r['n'] > 0 else 'n/a'
            auc_s = f"{r['auc']:.3f}" if r.get('auc', 0) > 0 else 'n/a'
            print(f"  {r['period']:<14} | {r['n']:>4} | {r.get('signals',0):>4} | "
                  f"{wr_s:>7} | {pf_s:>6} | {auc_s:>5} | {ok_s}")

        print(f"\n  Folds OK: {wf['folds_ok']}/12 | "
              f"{'APPROVED' if wf['approved'] else 'REJECTED'}")

        # OOS metrics
        oos_trades = [t for t in wf['all_trades']
                      if OOS_START <= str(t['ts'])[:10] <= OOS_END]
        m_oos = metrics(oos_trades, f'{model_name} OOS')

        if m_oos['n'] > 0:
            print(f"\n  OOS ({OOS_START}-{OOS_END}):")
            print(f"    N={m_oos['n']} | WR={m_oos['wr']:.1%} | PF={m_oos['pf']:.2f} | "
                  f"{m_oos['trades_pm']:.1f}t/m")

            # Equity
            cum = 1.0
            peak = 1.0
            max_dd = 0
            for t in sorted(oos_trades, key=lambda x: x['ts']):
                cum *= (1 + t['pnl_pct'])
                peak = max(peak, cum)
                dd = (peak - cum) / peak
                max_dd = max(max_dd, dd)
            print(f"    Equity: $1000 -> ${1000*cum:.0f} | MaxDD: {max_dd:.1%}")

        # Average AUC
        aucs = [r['auc'] for r in wf['folds'] if r.get('auc', 0) > 0]
        avg_auc = np.mean(aucs) if aucs else 0.5

        summary.append({
            'model': model_name,
            'folds_ok': wf['folds_ok'],
            'approved': wf['approved'],
            'n_oos': m_oos['n'],
            'wr': m_oos['wr'],
            'pf': m_oos['pf'],
            'avg_auc': avg_auc,
            'threshold': use_threshold,
            'tpm': m_oos['trades_pm'],
        })

    # ---- COMPARISON TABLE ----
    print(f"\n\n{'='*70}")
    print("COMPARISON TABLE — ETH MULTI-MODEL EVALUATION")
    print(f"Direction: {direction.upper()} | TP={tp_pct:.1%} SL={sl_pct:.1%}")
    print(f"{'='*70}")
    print(f"\n  {'Model':<8} | {'WF':>5} | {'N':>5} | {'WR':>7} | {'PF':>6} | "
          f"{'AUC':>5} | {'t/m':>4} | {'Thresh':>6} | Result")
    print("  " + "-" * 70)

    any_approved = False
    best_model = None
    best_pf = 0

    for s in summary:
        verdict = 'APPROVED' if s['approved'] else 'REJECTED'
        if s['approved']:
            any_approved = True
        if s['pf'] > best_pf and s['n_oos'] >= 10:
            best_pf = s['pf']
            best_model = s['model']

        print(f"  {s['model']:<8} | {s['folds_ok']:>2}/12 | {s['n_oos']:>5} | "
              f"{s['wr']:.1%} | {s['pf']:.2f} | {s['avg_auc']:.3f} | "
              f"{s['tpm']:.1f} | {s['threshold']:.2f} | {verdict}")

    # ---- FINAL VERDICT ----
    print(f"\n{'='*70}")
    if any_approved:
        print(f"RESULT: ETH HAS VIABLE MODEL(S)")
        print(f"  Best: {best_model} (PF={best_pf:.2f})")
        print(f"  Next: calibrate thresholds + cross-asset validation")
    else:
        print(f"RESULT: ETH REJECTED — No model passes WF >= 7/12")
        print(f"  Best attempt: {best_model} (PF={best_pf:.2f})" if best_model else "  No viable model")
        print(f"  ETH features did not provide sufficient edge")
    print(f"{'='*70}")

    # ---- Save results ----
    _save_results(summary, direction, tp_pct, sl_pct, any_approved, best_model)

    return any_approved, best_model


def _save_results(summary, direction, tp_pct, sl_pct, any_approved, best_model):
    """Save evaluation results to docs/."""
    doc_dir = ROOT / 'docs'
    doc_dir.mkdir(exist_ok=True)
    fpath = doc_dir / 'V15_ETH_evaluation.md'

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write("# V15 ETH Model Evaluation\n\n")
        f.write(f"**Date**: {datetime.now().date()}\n")
        f.write(f"**Direction**: {direction.upper()}\n")
        f.write(f"**TP/SL**: {tp_pct:.1%} / {sl_pct:.1%}\n")
        result = "VIABLE" if any_approved else "REJECTED"
        f.write(f"**Result**: {result}\n\n")

        f.write("## Approach\n\n")
        f.write("ETH failed with standard BTC features (AUC ~0.5). "
                "This evaluation uses ETH-specific features:\n\n")
        f.write("1. **ETH/BTC ratio** - momentum relativo, slope, z-score\n")
        f.write("2. **BTC cross-features** - BTC ret, RSI, vol as predictors\n")
        f.write("3. **Volatility regime** - ATR percentile, realized vol, vol-of-vol\n")
        f.write("4. **Mean-reversion** - VWAP distance, price z-score\n")
        f.write("5. **Microstructure** - close-in-range, shadow ratio, consecutive candles\n\n")
        f.write(f"Total features: ~{len(ALL_FEATURES)}\n\n")

        f.write("## Results\n\n")
        f.write("| Model | WF Folds | N OOS | WR | PF | Avg AUC | t/m | Result |\n")
        f.write("|-------|----------|-------|----|----|---------|-----|--------|\n")
        for s in summary:
            verdict = "APPROVED" if s['approved'] else "REJECTED"
            f.write(f"| {s['model']} | {s['folds_ok']}/12 | {s['n_oos']} | "
                    f"{s['wr']:.1%} | {s['pf']:.2f} | {s['avg_auc']:.3f} | "
                    f"{s['tpm']:.1f} | {verdict} |\n")

        f.write(f"\n## Verdict\n\n")
        if any_approved:
            f.write(f"**ETH has viable model(s)**. Best: {best_model}.\n\n")
            f.write("### Next steps\n")
            f.write("1. Calibrate threshold for production\n")
            f.write("2. Cross-asset validation (ETH features on similar L1s)\n")
            f.write("3. Train production model with train_v15_prod.py --pair ETH\n")
        else:
            f.write("**ETH REJECTED definitively.** No model passes WF >= 7/12 "
                    "with ETH-specific features.\n\n")
            f.write("ETH-specific features (ETH/BTC ratio, volatility regime, "
                    "mean-reversion) did not provide sufficient edge over random.\n")

    print(f"\n  Results saved to {fpath}")


if __name__ == '__main__':
    main()
