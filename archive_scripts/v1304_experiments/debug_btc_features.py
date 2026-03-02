import joblib
import pandas as pd
import pandas_ta as ta
import numpy as np

print("Loading model and data...", flush=True)
model = joblib.load('models/v95_v7_BTCUSDT.pkl')
df = pd.read_parquet('data/BTC_USDT_4h_history.parquet')
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')

print(f"Data: {len(df)} rows", flush=True)
print(f"Model features: {model.feature_name_}", flush=True)

# Compute features
print("\nComputing features...", flush=True)
feat = pd.DataFrame(index=df.index)
c, h, l, v = df['close'], df['high'], df['low'], df['volume']

for p in [1, 3, 5, 10, 20]:
    feat[f'ret_{p}'] = c.pct_change(p)

feat['atr14'] = ta.atr(h, l, c, length=14)
feat['atr_r'] = feat['atr14'] / feat['atr14'].rolling(50).mean()
feat['vol5'] = c.pct_change().rolling(5).std()
feat['vol20'] = c.pct_change().rolling(20).std()
feat['rsi14'] = ta.rsi(c, length=14)
feat['rsi7'] = ta.rsi(c, length=7)

sr = ta.stochrsi(c, length=14, rsi_length=14, k=3, d=3)
if sr is not None:
    feat['srsi_k'] = sr.iloc[:, 0]
macd = ta.macd(c, fast=12, slow=26, signal=9)
if macd is not None:
    feat['macd_h'] = macd.iloc[:, 1]

feat['roc5'] = ta.roc(c, length=5)
feat['roc20'] = ta.roc(c, length=20)

for el in [8, 21, 55, 100, 200]:
    e = ta.ema(c, length=el)
    feat[f'ema{el}_d'] = (c - e) / e * 100
feat['ema8_sl'] = ta.ema(c, length=8).pct_change(3) * 100
feat['ema55_sl'] = ta.ema(c, length=55).pct_change(5) * 100

bb = ta.bbands(c, length=20, std=2.0)
if bb is not None:
    bw = bb.iloc[:, 2] - bb.iloc[:, 0]
    feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw
    feat['bb_w'] = bw / bb.iloc[:, 1] * 100

feat['vr'] = v / v.rolling(20).mean()
feat['spr'] = (h - l) / c * 100
feat['body'] = abs(c - df['open']) / (h - l + 1e-10)

ax = ta.adx(h, l, c, length=14)
if ax is not None:
    feat['adx'] = ax.iloc[:, 0]
    feat['dip'] = ax.iloc[:, 1]
    feat['dim'] = ax.iloc[:, 2]

hr = df.index.hour
dw = df.index.dayofweek
feat['h_s'] = np.sin(2 * np.pi * hr / 24)
feat['h_c'] = np.cos(2 * np.pi * hr / 24)
feat['d_s'] = np.sin(2 * np.pi * dw / 7)
feat['d_c'] = np.cos(2 * np.pi * dw / 7)

print(f"\nComputed features: {list(feat.columns)}", flush=True)

# Check which model features are missing
model_feats = set(model.feature_name_)
computed_feats = set(feat.columns)
missing = model_feats - computed_feats
extra = computed_feats - model_feats

print(f"\nMissing features (in model but not computed): {missing}", flush=True)
print(f"Extra features (computed but not in model): {extra}", flush=True)

# Check if fcols intersection works
fcols = [c for c in model.feature_name_ if c in feat.columns]
print(f"\nUsable features: {len(fcols)}/{len(model.feature_name_)}", flush=True)

# Check NaN counts
print(f"\nNaN check at row 200:", flush=True)
row = feat.iloc[200]
for c in fcols:
    if pd.isna(row[c]):
        print(f"  {c}: NaN", flush=True)

print(f"\nNaN check at row 500:", flush=True)
row = feat.iloc[500]
nan_count = 0
for c in fcols:
    if pd.isna(row[c]):
        print(f"  {c}: NaN", flush=True)
        nan_count += 1
print(f"Total NaN: {nan_count}/{len(fcols)}", flush=True)
