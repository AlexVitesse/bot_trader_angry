"""
train_v15_prod.py — Train V15 SHORT model for production
=========================================================
Trains GradientBoosting SHORT model on ALL BEAR bars before cutoff date.
Saves model + scaler + meta to strategies/btc_v15/models/

Usage:
  python train_v15_prod.py              # Use existing CSV data
  python train_v15_prod.py --refresh    # Download fresh data from Binance first

Run with PRODUCTION Python to avoid sklearn version mismatch:
  C:\\Users\\pcdec\\AppData\\Local\\pypoetry\\Cache\\virtualenvs\\
  binance-scalper-bot-ofXWUGOe-py3.12\\Scripts\\python.exe train_v15_prod.py
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import json
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timezone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
sys.path.insert(0, str(ROOT))

from v15_framework import (
    load_btc_4h, compute_features_4h, compute_macro_daily,
    merge_daily_to_4h, load_funding, COMMISSION
)


def refresh_btc_data():
    """Download fresh BTC 4H data from Binance and update CSV."""
    import ccxt
    print("\nDownloading fresh BTC/USDT 4H data from Binance...")

    exchange = ccxt.binance({'enableRateLimit': True})
    csv_path = DATA / 'BTCUSDT_4h.csv'

    # Read existing data to find last timestamp
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing['timestamp'] = pd.to_datetime(existing['timestamp'])
        last_ts = int(existing['timestamp'].iloc[-1].timestamp() * 1000)
        print(f"  Existing data until: {existing['timestamp'].iloc[-1]}")
    else:
        # Start from 2018
        last_ts = int(datetime(2018, 1, 1).timestamp() * 1000)
        existing = pd.DataFrame()
        print("  No existing data, downloading from 2018...")

    # Fetch new candles after last timestamp
    all_new = []
    since = last_ts + 1  # Next ms after last candle
    while True:
        candles = exchange.fetch_ohlcv('BTC/USDT', '4h', since=since, limit=1000)
        if not candles:
            break
        all_new.extend(candles)
        since = candles[-1][0] + 1
        print(f"  Fetched {len(all_new)} new candles...", end='\r')
        time.sleep(exchange.rateLimit / 1000)

    if not all_new:
        print("  Already up to date.")
        return

    # Convert to DataFrame
    new_df = pd.DataFrame(all_new, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')

    # Merge with existing
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset='timestamp').sort_values('timestamp')
    else:
        combined = new_df.sort_values('timestamp')

    combined.to_csv(csv_path, index=False)
    print(f"  Updated: {len(combined)} total bars "
          f"({combined['timestamp'].iloc[0].date()} - {combined['timestamp'].iloc[-1].date()})")
    print(f"  Added {len(all_new)} new candles")

# ============================================================
# CONFIG (identical to validated backtest)
# ============================================================
CUTOFF = '2026-03-01'
SHORT_TP = 0.025
SHORT_SL = 0.015
SHORT_MAX_BARS = 16
SHORT_THRESHOLD = 0.60
REGIME_DEAD_ZONE = 0.02
FUNDING_VETO_LONG = 2.0
FUNDING_VETO_SHORT = -1.5

SHORT_FEATURES = [
    'ema200_dist', 'ema20_slope', 'ema50_slope',
    'rsi14', 'rsi_slope', 'di_diff', 'adx14',
    'bb_pct', 'bb_width', 'atr_pct',
    'range_pos', 'vol_ratio', 'vol_slope',
    'ret_1', 'ret_5', 'ret_10',
    'consec_up', 'bull_1d',
]

MODEL_DIR = ROOT / 'strategies' / 'btc_v15' / 'models'


def add_extra_features(df):
    """Add derived features needed by SHORT model."""
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


def create_short_labels(df, tp_pct=SHORT_TP, sl_pct=SHORT_SL, max_bars=SHORT_MAX_BARS):
    """Label: price drops tp_pct before rising sl_pct in max_bars."""
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    labels = np.full(len(df), np.nan)
    for i in range(len(df) - max_bars - 1):
        entry = closes[i]
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


def main():
    print("=" * 60)
    print("V15 PRODUCTION TRAINING — SHORT GBM MODEL")
    print("=" * 60)

    # Refresh data if requested
    if '--refresh' in sys.argv:
        refresh_btc_data()

    # Load data
    print("\nLoading data...")
    df_raw = load_btc_4h()
    df = compute_features_4h(df_raw)
    df = add_extra_features(df)
    df_daily = compute_macro_daily(df)
    df = merge_daily_to_4h(df, df_daily)
    print(f"  Total: {len(df)} bars ({df.index[0].date()} - {df.index[-1].date()})")

    # Filter training data (before cutoff)
    train_mask = df.index < CUTOFF
    df_train = df[train_mask]
    print(f"  Training: {len(df_train)} bars (before {CUTOFF})")

    # Create SHORT labels
    print("\nCreating SHORT labels...")
    labels = create_short_labels(df_train)
    valid = labels.notna()

    # Filter BEAR bars only
    bear_mask = df_train.get('bull_1d', pd.Series(1, index=df_train.index)) == 0
    train_valid = valid & bear_mask
    df_fit = df_train[train_valid]
    y_fit = labels[train_valid]

    print(f"  BEAR bars with labels: {len(df_fit)}")
    print(f"  Positive rate (SHORT wins): {y_fit.mean():.1%}")
    print(f"  Positive samples: {int(y_fit.sum())}")
    print(f"  Negative samples: {int(len(y_fit) - y_fit.sum())}")

    # Extract features
    X = df_fit[SHORT_FEATURES].fillna(0)

    # Train scaler
    print("\nTraining StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    print("Training GradientBoosting...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_scaled, y_fit)

    # Quick in-sample check
    probs = model.predict_proba(X_scaled)[:, 1]
    above_thresh = probs >= SHORT_THRESHOLD
    if above_thresh.sum() > 0:
        triggered = y_fit[above_thresh]
        print(f"\n  In-sample (threshold={SHORT_THRESHOLD}):")
        print(f"    Signals: {above_thresh.sum()}")
        print(f"    WR: {triggered.mean():.1%}")
    else:
        print(f"  WARNING: No signals above threshold {SHORT_THRESHOLD}")

    # Feature importance
    print("\n  Top 5 features:")
    imp = pd.Series(model.feature_importances_, index=SHORT_FEATURES)
    for feat, val in imp.nlargest(5).items():
        print(f"    {feat}: {val:.3f}")

    # Save models
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {MODEL_DIR}/...")
    joblib.dump(model, MODEL_DIR / 'short_gbm.pkl')
    joblib.dump(scaler, MODEL_DIR / 'short_scaler.pkl')
    print("  short_gbm.pkl saved")
    print("  short_scaler.pkl saved")

    # Save meta
    import sklearn
    meta = {
        'version': 'V15',
        'asset': 'BTC',
        'model_type': 'GradientBoostingClassifier',
        'sklearn_version': sklearn.__version__,
        'training_date': datetime.now(timezone.utc).isoformat(),
        'cutoff_date': CUTOFF,
        'training_samples': len(df_fit),
        'positive_rate': float(y_fit.mean()),

        # Features
        'short_features': SHORT_FEATURES,

        # Thresholds (from validated backtest)
        'short_threshold': SHORT_THRESHOLD,
        'regime_dead_zone': REGIME_DEAD_ZONE,
        'funding_veto_long': FUNDING_VETO_LONG,
        'funding_veto_short': FUNDING_VETO_SHORT,

        # SHORT params
        'short_tp_pct': SHORT_TP,
        'short_sl_pct': SHORT_SL,
        'short_max_bars': SHORT_MAX_BARS,

        # Breakout B params
        'breakout_vol_min': 1.8,
        'breakout_bb_max': 4.0,
        'breakout_adx_max': 28,
        'breakout_bar_move_max': 2.5,
        'breakout_sl_min': 0.005,
        'breakout_sl_max': 0.04,
        'breakout_rr': 1.5,

        # Pullback EMA20 params
        'pullback_dist_min': -0.005,
        'pullback_dist_max': 0.015,
        'pullback_adx_min': 15,
        'pullback_rsi_min': 33,
        'pullback_rsi_max': 58,
        'pullback_vol_max': 2.0,
        'pullback_atr_sl_mult': 1.0,
        'pullback_atr_sl_min': 0.01,
        'pullback_atr_sl_max': 0.03,
        'pullback_rr': 1.67,

        # Max bars
        'long_max_bars': 16,
        'short_max_bars': SHORT_MAX_BARS,

        # Backtest metrics (from WF 8/12 approved run)
        'backtest': {
            'wf_folds_ok': 8,
            'wf_folds_total': 12,
            'oos_wr': 0.48,
            'oos_pf': 1.35,
            'oos_trades_pm': 7.3,
            'equity_1k': 7116,
            'max_dd': 0.35,
            'cagr_pct': 37.5,
        },
    }

    with open(MODEL_DIR / 'meta_v15.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    print("  meta_v15.json saved")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Model: {MODEL_DIR / 'short_gbm.pkl'}")
    print(f"  Scaler: {MODEL_DIR / 'short_scaler.pkl'}")
    print(f"  Meta: {MODEL_DIR / 'meta_v15.json'}")


if __name__ == '__main__':
    main()
