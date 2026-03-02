"""
Analisis de diferentes umbrales de conviction para BTC
"""
import joblib
import pandas as pd
import pandas_ta as ta
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ANALISIS DE CONVICTION THRESHOLDS - BTC")
print("="*60)

# Config
TP_PCT = 0.03
SL_PCT = 0.015
MAX_HOLD = 20

# Load
model = joblib.load('models/v95_v7_BTCUSDT.pkl')
df = pd.read_parquet('data/BTC_USDT_4h_history.parquet')
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')

# Compute features (same as before)
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

fcols = list(model.feature_name_)

# Generate all predictions
predictions = []
for i in range(200, len(df)):
    ts = df.index[i]
    row = feat.iloc[i]
    if row[fcols].isna().any():
        continue
    pred = model.predict(row[fcols].values.reshape(1, -1))[0]
    direction = 1 if pred > 0 else -1
    conviction = abs(pred) / 0.005
    predictions.append({
        'idx': i,
        'ts': ts,
        'pred': pred,
        'direction': direction,
        'conviction': conviction,
    })

print(f"\nTotal predicciones: {len(predictions)}")

# Distribution of predictions
preds_arr = np.array([p['pred'] for p in predictions])
convs_arr = np.array([p['conviction'] for p in predictions])

print(f"\nDistribucion de predicciones (pred raw):")
print(f"  Mean: {preds_arr.mean():.6f}")
print(f"  Std:  {preds_arr.std():.6f}")
print(f"  Min:  {preds_arr.min():.6f}")
print(f"  Max:  {preds_arr.max():.6f}")

print(f"\nDistribucion de conviction:")
print(f"  Mean: {convs_arr.mean():.2f}")
print(f"  Std:  {convs_arr.std():.2f}")

for thresh in [0.5, 0.75, 1.0, 1.5, 2.0]:
    above = (convs_arr >= thresh).sum()
    print(f"  Conv >= {thresh}: {above} ({above/len(convs_arr)*100:.1f}%)")

# Test different thresholds
print(f"\n{'='*60}")
print("BACKTEST POR THRESHOLD")
print(f"{'='*60}")

def backtest(signals):
    trades = []
    for sig in signals:
        idx = sig['idx']
        if idx + 1 >= len(df):
            continue
        entry = df.iloc[idx + 1]['open']
        direction = sig['direction']
        tp_price = entry * (1 + TP_PCT) if direction == 1 else entry * (1 - TP_PCT)
        sl_price = entry * (1 - SL_PCT) if direction == 1 else entry * (1 + SL_PCT)
        pnl_pct = None
        for j in range(idx + 1, min(idx + MAX_HOLD + 1, len(df))):
            candle = df.iloc[j]
            if direction == 1:
                if candle['low'] <= sl_price:
                    pnl_pct = -SL_PCT
                    break
                if candle['high'] >= tp_price:
                    pnl_pct = TP_PCT
                    break
            else:
                if candle['high'] >= sl_price:
                    pnl_pct = -SL_PCT
                    break
                if candle['low'] <= tp_price:
                    pnl_pct = TP_PCT
                    break
        if pnl_pct is None:
            exit_idx = min(idx + MAX_HOLD, len(df) - 1)
            exit_price = df.iloc[exit_idx]['close']
            pnl_pct = (exit_price - entry) / entry * direction
        trades.append({'pnl_pct': pnl_pct, 'win': pnl_pct > 0, 'direction': direction})
    return trades

print(f"\n{'Threshold':<10} {'Trades':>8} {'WR':>8} {'PnL%':>10} {'LONG':>8} {'SHORT':>8}")
print("-"*60)

for thresh in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    signals = [p for p in predictions if p['conviction'] >= thresh]
    if not signals:
        continue
    trades = backtest(signals)
    n = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n * 100 if n > 0 else 0
    pnl = sum(t['pnl_pct'] for t in trades) * 100
    longs = sum(1 for t in trades if t['direction'] == 1)
    shorts = n - longs
    print(f"{thresh:<10.2f} {n:>8} {wr:>7.1f}% {pnl:>9.2f}% {longs:>8} {shorts:>8}")

# Best threshold analysis
print(f"\n{'='*60}")
print("ANALISIS: Mejor configuracion")
print(f"{'='*60}")

# Find optimal threshold (highest PnL with reasonable WR)
best_thresh = 0
best_score = -999
for thresh in np.arange(0.0, 2.5, 0.1):
    signals = [p for p in predictions if p['conviction'] >= thresh]
    if len(signals) < 5:  # need at least 5 trades
        continue
    trades = backtest(signals)
    n = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n * 100
    pnl = sum(t['pnl_pct'] for t in trades) * 100
    # Score = PnL weighted by WR (prefer higher WR)
    score = pnl * (wr / 100)
    if score > best_score:
        best_score = score
        best_thresh = thresh

print(f"\nMejor threshold: {best_thresh:.1f}")
signals = [p for p in predictions if p['conviction'] >= best_thresh]
trades = backtest(signals)
n = len(trades)
wins = sum(1 for t in trades if t['win'])
wr = wins / n * 100
pnl = sum(t['pnl_pct'] for t in trades) * 100
print(f"  Trades: {n}")
print(f"  WR: {wr:.1f}%")
print(f"  PnL: {pnl:.2f}%")
