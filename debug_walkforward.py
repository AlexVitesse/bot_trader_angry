"""Debug walk-forward validation issue."""
import pandas as pd
import numpy as np
import pandas_ta as ta
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
TRAIN_END = '2024-12-31'
VAL_START = '2025-01-01'
VAL_END = '2025-09-30'

def compute_features_v2(df):
    feat = pd.DataFrame(index=df.index)
    c, h, l, o, v = df['close'], df['high'], df['low'], df['open'], df['volume']

    for p in [1, 2, 3, 5, 10, 20, 50]:
        feat[f'ret_{p}'] = c.pct_change(p)

    feat['atr14'] = ta.atr(h, l, c, length=14)
    feat['vol5'] = c.pct_change().rolling(5).std()
    feat['vol20'] = c.pct_change().rolling(20).std()
    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        feat['macd_h'] = macd.iloc[:, 1]

    for el in [8, 21, 55, 100]:
        e = ta.ema(c, length=el)
        feat[f'ema{el}_d'] = (c - e) / e * 100

    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        bw = bb.iloc[:, 2] - bb.iloc[:, 0]
        feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw

    feat['vr'] = v / v.rolling(20).mean()

    return feat


# Load BTC data
print("Loading BTC data...")
df = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_full.parquet')
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
df = df.sort_index()
print(f"Data range: {df.index.min()} to {df.index.max()}")

# Split data
train_df = df[:TRAIN_END]
val_df = df[VAL_START:VAL_END]
print(f"Train: {len(train_df)} rows")
print(f"Validation: {len(val_df)} rows")

# Compute features on train
print("\nComputing train features...")
train_feat = compute_features_v2(train_df)
train_feat = train_feat.replace([np.inf, -np.inf], np.nan)

# Target
target = train_df['close'].pct_change(5).shift(-5)
valid_idx = train_feat.dropna().index.intersection(target.dropna().index)
X_train = train_feat.loc[valid_idx].iloc[:-5]
y_train = target.loc[valid_idx].iloc[:-5]

print(f"Training samples: {len(X_train)}")
feature_cols = list(X_train.columns)

# Scale and train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
model.fit(X_scaled, y_train)

train_preds = model.predict(X_scaled)
train_corr = np.corrcoef(train_preds, y_train)[0, 1]
pred_std = np.std(train_preds)
print(f"Train correlation: {train_corr:.4f}")
print(f"Train pred_std: {pred_std:.6f}")

# Now compute features on validation with lookback
print("\nComputing validation features...")
# Need history for features
val_start_idx = df.index.get_loc(val_df.index[0])
lookback_start = max(0, val_start_idx - 200)
work_df = df.iloc[lookback_start:].copy()
work_df = work_df[:VAL_END]

val_feat = compute_features_v2(work_df)
val_feat = val_feat.replace([np.inf, -np.inf], np.nan)

# Get only validation period features
val_feat_only = val_feat.loc[val_df.index]
print(f"Validation feature rows: {len(val_feat_only)}")

# Make predictions on validation
cols = [c for c in feature_cols if c in val_feat_only.columns]
X_val = val_feat_only[cols].fillna(0)
X_val_scaled = scaler.transform(X_val)
val_preds = model.predict(X_val_scaled)

print(f"\nValidation predictions:")
print(f"  Range: [{val_preds.min():.6f}, {val_preds.max():.6f}]")
print(f"  Mean: {val_preds.mean():.6f}")
print(f"  Std: {val_preds.std():.6f}")

# Compute conviction
val_conf = np.abs(val_preds) / pred_std
print(f"\nValidation conviction:")
print(f"  Range: [{val_conf.min():.3f}, {val_conf.max():.3f}]")
print(f"  Mean: {val_conf.mean():.3f}")
print(f"  >0.5: {(val_conf >= 0.5).sum()}")
print(f"  >1.0: {(val_conf >= 1.0).sum()}")
print(f"  >2.0: {(val_conf >= 2.0).sum()}")

# Correlation on validation
val_target = val_df['close'].pct_change(5).shift(-5)
valid_val_idx = X_val.index.intersection(val_target.dropna().index)[:-5]
if len(valid_val_idx) > 10:
    val_corr = np.corrcoef(
        model.predict(scaler.transform(X_val.loc[valid_val_idx])),
        val_target.loc[valid_val_idx]
    )[0, 1]
    print(f"\nValidation correlation: {val_corr:.4f}")
    print(f"Correlation DROP: {train_corr - val_corr:.4f} ({(train_corr - val_corr)/train_corr*100:.1f}%)")
