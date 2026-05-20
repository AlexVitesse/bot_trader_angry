"""
diagnose_no_trades.py -- Why is the bot not generating trades?
Check recent bars for signals under OLD (baseline) vs NEW (adaptive) config.
"""
import sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from v15_framework import (
    load_btc_4h, compute_features_4h, compute_macro_daily,
    merge_daily_to_4h, COMMISSION,
)
from backtest_v15_committee import add_extra_features, detect_regime

print("=" * 70)
print("DIAGNOSTIC: Why no trades?")
print("=" * 70)

print("\nLoading data...")
df_raw = load_btc_4h()
df = compute_features_4h(df_raw)
df = add_extra_features(df)
df_daily = compute_macro_daily(df)
df = merge_daily_to_4h(df, df_daily)

print(f"  Data: {len(df)} bars, last: {df.index[-1]}")

# Check last 60 bars (~10 days)
LOOKBACK = 60
df_recent = df.iloc[-LOOKBACK:]

print(f"\n--- LAST {LOOKBACK} BARS ({df_recent.index[0].date()} to {df_recent.index[-1].date()}) ---\n")

# 1. Regime
print("1. REGIME:")
for i in range(len(df_recent)):
    row = df_recent.iloc[i]
    regime = detect_regime(row)
    if i % 6 == 0 or i == len(df_recent) - 1:
        ema20_1d = row.get('ema20_1d', 0)
        ema50_1d = row.get('ema50_1d', 0)
        dist = (ema20_1d - ema50_1d) / ema50_1d * 100 if ema50_1d else 0
        print(f"  {str(df_recent.index[i])[:16]} -> {regime:>5} "
              f"(EMA20/50 dist={dist:+.2f}%)")

# 2. Breakout check (OLD static)
print("\n2. BREAKOUT SIGNALS (OLD static rules):")
breakouts_old = 0
for idx in range(20, len(df_recent)):
    i = len(df) - LOOKBACK + idx
    row = df.iloc[i]
    regime = detect_regime(row)
    if regime not in ('BULL', 'RANGE'):
        continue

    high_N = float(df['high'].iloc[i-20:i].max())
    close = float(row['close'])
    vol = float(row.get('vol_ratio', 1))
    bb_w = df['bb_width'].iloc[i-5:i]
    bb_count = (bb_w < 4.0).sum()
    adx_avg = df['adx14'].iloc[i-3:i].mean()
    bar_move = abs(close - float(row['open'])) / float(row['open']) * 100

    breakout = close > high_N
    vol_ok = vol >= 1.8
    bb_ok = bb_count >= 3
    adx_ok = adx_avg <= 28
    move_ok = bar_move <= 2.5

    if breakout:
        sl_raw = float(df['low'].iloc[max(0,i-5):i].min()) * 0.997
        sl_pct = (close - sl_raw) / close
        sl_ok = 0.005 <= sl_pct <= 0.04
        status = ("PASS" if all([vol_ok, bb_ok, adx_ok, move_ok, sl_ok])
                  else "BLOCKED")
        reasons = []
        if not vol_ok: reasons.append(f"vol={vol:.1f}<1.8")
        if not bb_ok: reasons.append(f"bb_count={bb_count}<3")
        if not adx_ok: reasons.append(f"adx={adx_avg:.0f}>28")
        if not move_ok: reasons.append(f"move={bar_move:.1f}%>2.5%")
        if not sl_ok: reasons.append(f"sl={sl_pct:.1%} out of range")
        print(f"  {str(df.index[i])[:16]} BREAKOUT {regime:>5}: {status} "
              f"{'| ' + ', '.join(reasons) if reasons else ''}")
        breakouts_old += 1 if status == "PASS" else 0

if breakouts_old == 0:
    print("  -> NO breakouts passed old rules either!")

# 3. Breakout with NEW quality filter
print("\n3. BREAKOUT SIGNALS (NEW quality>=50 filter):")
def compute_quality(df, i, regime):
    row = df.iloc[i]
    score = 0
    start = max(0, i - 50)
    bb_series = df['bb_width'].iloc[start:i]
    if len(bb_series) >= 10:
        median_bb = bb_series.median()
        current_bb = float(df['bb_width'].iloc[max(0, i-1)])
        if median_bb > 0:
            ratio = current_bb / median_bb
            if ratio < 0.5: score += 25
            elif ratio < 0.7: score += 18
            elif ratio < 1.0: score += 10
    vol_ratio = float(row.get('vol_ratio', 1.0))
    if vol_ratio >= 3.0: score += 20
    elif vol_ratio >= 2.5: score += 16
    elif vol_ratio >= 2.0: score += 12
    elif vol_ratio >= 1.5: score += 6
    di_diff = float(row.get('di_diff', 0))
    if i >= 1:
        prev_di = float(df.iloc[i-1].get('di_diff', 0))
        if di_diff > 0 and prev_di <= 0: score += 15
        elif di_diff > 5: score += 8
        elif di_diff > 0: score += 4
    if regime == 'BULL': score += 20
    elif regime == 'RANGE': score += 10
    ema20 = float(row.get('ema20', 0))
    ema50 = float(row.get('ema50', 0))
    close = float(row['close'])
    if ema20 > 0 and ema50 > 0:
        if close > ema20 > ema50: score += 10
        elif close > ema50: score += 5
    rsi = float(row.get('rsi14', 50))
    if 45 <= rsi <= 65: score += 10
    elif 35 <= rsi <= 75: score += 5
    return min(score, 100)

for idx in range(20, len(df_recent)):
    i = len(df) - LOOKBACK + idx
    row = df.iloc[i]
    regime = detect_regime(row)
    if regime not in ('BULL', 'RANGE'):
        continue

    high_N = float(df['high'].iloc[i-20:i].max())
    close = float(row['close'])
    vol = float(row.get('vol_ratio', 1))
    bb_w = df['bb_width'].iloc[i-5:i]
    bb_count = (bb_w < 4.0).sum()
    adx_avg = df['adx14'].iloc[i-3:i].mean()
    bar_move = abs(close - float(row['open'])) / float(row['open']) * 100

    breakout = close > high_N
    if not breakout:
        continue

    vol_ok = vol >= 1.8
    bb_ok = bb_count >= 3
    adx_ok = adx_avg <= 28
    move_ok = bar_move <= 2.5
    sl_raw = float(df['low'].iloc[max(0,i-5):i].min()) * 0.997
    sl_pct = (close - sl_raw) / close
    sl_ok = 0.005 <= sl_pct <= 0.04

    passes_static = all([vol_ok, bb_ok, adx_ok, move_ok, sl_ok])
    quality = compute_quality(df, i, regime)

    status = "PASS" if (passes_static and quality >= 50) else "BLOCKED"
    block_reason = ""
    if passes_static and quality < 50:
        block_reason = f"QUALITY={quality}<50 (was PASS in old rules!)"
    elif not passes_static:
        block_reason = "failed static rules"

    print(f"  {str(df.index[i])[:16]} Q={quality:>3} {regime:>5} {status} {block_reason}")

# 4. SHORT check
print("\n4. SHORT ML SIGNALS:")
print("  (checking BEAR bars for SHORT candidates)")
import pickle
model_dir = ROOT / 'strategies' / 'btc_v15' / 'models'
import json
meta = json.load(open(model_dir / 'meta_v15.json'))
features = meta.get('short_features', [])

try:
    short_model = pickle.load(open(model_dir / 'short_gbm.pkl', 'rb'))
    short_scaler = pickle.load(open(model_dir / 'short_scaler.pkl', 'rb'))
    print(f"  Model loaded OK, features: {len(features)}")
except Exception as e:
    short_model = None
    print(f"  Model load FAILED: {e}")

short_signals = 0
for idx in range(20, len(df_recent)):
    i = len(df) - LOOKBACK + idx
    row = df.iloc[i]
    regime = detect_regime(row)
    if regime != 'BEAR':
        continue

    if short_model is None:
        print(f"  {str(df.index[i])[:16]} BEAR - no model")
        continue

    x_vals = [float(row.get(f, 0)) for f in features]
    x = np.array(x_vals).reshape(1, -1)
    x = np.nan_to_num(x, nan=0.0)
    from sklearn.preprocessing import StandardScaler
    x_scaled = short_scaler.transform(x)
    prob = float(short_model.predict_proba(x_scaled)[0][1])

    ema20 = float(row.get('ema20', 0))
    ema50 = float(row.get('ema50', 0))
    ema_cross = ema20 < ema50 if (ema20 > 0 and ema50 > 0) else False

    prob_ok = prob >= 0.60
    ema_ok = ema_cross

    status = "PASS" if (prob_ok and ema_ok) else "BLOCKED"
    reasons = []
    if not prob_ok: reasons.append(f"prob={prob:.2f}<0.60")
    if not ema_ok: reasons.append(f"EMA20={ema20:.0f}>=EMA50={ema50:.0f}")
    blocked_old = "old=PASS" if prob_ok else "old=BLOCKED"

    print(f"  {str(df.index[i])[:16]} BEAR prob={prob:.2f} EMA20{'<' if ema_cross else '>='}"
          f"EMA50 -> {status} {blocked_old} {'| '+', '.join(reasons) if reasons else ''}")
    if prob_ok:
        short_signals += 1

# 5. Summary
print(f"\n--- SUMMARY ---")
regimes = df_recent.apply(lambda r: detect_regime(r), axis=1)
regime_counts = regimes.value_counts()
print(f"  Regimes in last {LOOKBACK} bars: {dict(regime_counts)}")
print(f"  Current regime: {detect_regime(df.iloc[-1])}")
print(f"  Current price: ${float(df['close'].iloc[-1]):,.0f}")
print(f"  EMA20_1d: {float(df.iloc[-1].get('ema20_1d', 0)):,.0f}")
print(f"  EMA50_1d: {float(df.iloc[-1].get('ema50_1d', 0)):,.0f}")
last = df.iloc[-1]
print(f"  vol_ratio: {float(last.get('vol_ratio', 0)):.2f}")
print(f"  bb_width: {float(last.get('bb_width', 0)):.2f}")
print(f"  rsi14: {float(last.get('rsi14', 0)):.1f}")
print(f"  EMA20_4h: {float(last.get('ema20', 0)):,.0f}")
print(f"  EMA50_4h: {float(last.get('ema50', 0)):,.0f}")

print("\n" + "=" * 70)
