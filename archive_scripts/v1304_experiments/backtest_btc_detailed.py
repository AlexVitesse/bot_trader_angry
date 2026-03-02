"""
Backtest detallado de BTC
=========================
"""
import joblib
import pandas as pd
import pandas_ta as ta
import numpy as np
from pathlib import Path

print("="*60, flush=True)
print("BACKTEST BTC DETALLADO", flush=True)
print("="*60, flush=True)

# Config
TP_PCT = 0.03
SL_PCT = 0.015
MAX_HOLD = 20
CONVICTION_MIN = 1.0

# Load
model = joblib.load('models/v95_v7_BTCUSDT.pkl')
df = pd.read_parquet('data/BTC_USDT_4h_history.parquet')
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')

print(f"Data: {len(df)} candles", flush=True)
print(f"Range: {df.index[0]} to {df.index[-1]}", flush=True)

# Compute features
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
print("\nGenerating predictions...", flush=True)
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
        'rsi14': row['rsi14'],
        'bb_pos': row['bb_pos'],
        'adx': row['adx'],
    })

print(f"Total predictions: {len(predictions)}", flush=True)

# Filter by conviction
signals = [p for p in predictions if p['conviction'] >= CONVICTION_MIN]
print(f"Signals (conv >= {CONVICTION_MIN}): {len(signals)}", flush=True)

# Simulate trades
print("\nSimulating trades...", flush=True)
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
    exit_reason = None
    bars = 0

    for j in range(idx + 1, min(idx + MAX_HOLD + 1, len(df))):
        bars += 1
        candle = df.iloc[j]

        if direction == 1:
            if candle['low'] <= sl_price:
                pnl_pct = -SL_PCT
                exit_reason = 'SL'
                break
            if candle['high'] >= tp_price:
                pnl_pct = TP_PCT
                exit_reason = 'TP'
                break
        else:
            if candle['high'] >= sl_price:
                pnl_pct = -SL_PCT
                exit_reason = 'SL'
                break
            if candle['low'] <= tp_price:
                pnl_pct = TP_PCT
                exit_reason = 'TP'
                break

    if pnl_pct is None:
        exit_idx = min(idx + MAX_HOLD, len(df) - 1)
        exit_price = df.iloc[exit_idx]['close']
        pnl_pct = (exit_price - entry) / entry * direction
        exit_reason = 'TIME'

    trades.append({
        'ts': sig['ts'],
        'direction': 'LONG' if direction == 1 else 'SHORT',
        'conviction': sig['conviction'],
        'entry': entry,
        'pnl_pct': pnl_pct,
        'win': pnl_pct > 0,
        'exit_reason': exit_reason,
        'bars': bars,
        'rsi14': sig['rsi14'],
        'bb_pos': sig['bb_pos'],
        'adx': sig['adx'],
    })

# Results
print(f"\n{'='*60}", flush=True)
print("RESULTADOS", flush=True)
print(f"{'='*60}", flush=True)

if trades:
    tdf = pd.DataFrame(trades)
    n = len(tdf)
    wins = tdf['win'].sum()
    wr = wins / n * 100
    pnl = tdf['pnl_pct'].sum() * 100

    print(f"\nTotal trades: {n}", flush=True)
    print(f"Wins: {wins}", flush=True)
    print(f"Win Rate: {wr:.1f}%", flush=True)
    print(f"PnL total: {pnl:.2f}%", flush=True)

    # By direction
    print(f"\nPor direccion:", flush=True)
    for d in ['LONG', 'SHORT']:
        sub = tdf[tdf['direction'] == d]
        if len(sub) > 0:
            sub_wr = sub['win'].mean() * 100
            sub_pnl = sub['pnl_pct'].sum() * 100
            print(f"  {d}: {len(sub)} trades, WR={sub_wr:.1f}%, PnL={sub_pnl:.2f}%", flush=True)

    # By exit reason
    print(f"\nPor salida:", flush=True)
    for r in ['TP', 'SL', 'TIME']:
        sub = tdf[tdf['exit_reason'] == r]
        print(f"  {r}: {len(sub)} ({len(sub)/n*100:.0f}%)", flush=True)

    # Monthly breakdown
    print(f"\nPor mes:", flush=True)
    tdf['month'] = pd.to_datetime(tdf['ts']).dt.to_period('M')
    for month, grp in tdf.groupby('month'):
        m_wr = grp['win'].mean() * 100
        m_pnl = grp['pnl_pct'].sum() * 100
        print(f"  {month}: {len(grp)} trades, WR={m_wr:.1f}%, PnL={m_pnl:.2f}%", flush=True)

    # Sample trades
    print(f"\nUltimos 10 trades:", flush=True)
    for _, t in tdf.tail(10).iterrows():
        win_str = 'WIN' if t['win'] else 'LOSS'
        print(f"  {t['ts'].strftime('%m-%d %H:%M')} | {t['direction']:5} | conv={t['conviction']:.1f} | {t['exit_reason']:4} | {win_str}", flush=True)
else:
    print("No trades!", flush=True)
