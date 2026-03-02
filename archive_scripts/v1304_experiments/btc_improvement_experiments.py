"""
BTC Model Improvement Experiments
=================================
1. Higher conviction threshold (>= 1.5)
2. Macro features (BTC.D correlation proxy, volume patterns)
3. Ensemble V2 + V3
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
import ccxt

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

TP_PCT = 0.03
SL_PCT = 0.015
MAX_HOLD = 20

print('='*70)
print('BTC IMPROVEMENT EXPERIMENTS')
print('='*70)

# =============================================================================
# 1. LOAD DATA AND MODELS
# =============================================================================
print('\n[1] Loading data and models...')

df = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_full.parquet')
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')

v2_data = joblib.load(MODELS_DIR / 'btc_v2_gradientboosting.pkl')
v2_model = v2_data['model']
v2_cols = v2_data['feature_cols']

v3_data = joblib.load(MODELS_DIR / 'btc_v3_recent.pkl')
v3_model = v3_data['model']
v3_cols = v3_data['feature_cols']

print(f'    Data: {len(df)} candles')
print(f'    V2 features: {len(v2_cols)}')
print(f'    V3 features: {len(v3_cols)}')

# =============================================================================
# 2. CALCULATE BASE FEATURES
# =============================================================================
print('\n[2] Calculating features...')

feat = pd.DataFrame(index=df.index)
c, h, l, v = df['close'], df['high'], df['low'], df['volume']

# Returns
for p in [1, 2, 3, 5, 10, 20, 50]:
    feat[f'ret_{p}'] = c.pct_change(p)

# Volatility
feat['atr14'] = ta.atr(h, l, c, length=14)
feat['atr_r'] = feat['atr14'] / feat['atr14'].rolling(50).mean()
feat['vol5'] = c.pct_change().rolling(5).std()
feat['vol20'] = c.pct_change().rolling(20).std()
feat['vol_ratio'] = feat['vol5'] / (feat['vol20'] + 1e-10)

# RSI
feat['rsi14'] = ta.rsi(c, length=14)
feat['rsi7'] = ta.rsi(c, length=7)
feat['rsi21'] = ta.rsi(c, length=21)

# Stochastic RSI
sr = ta.stochrsi(c, length=14, rsi_length=14, k=3, d=3)
if sr is not None:
    feat['srsi_k'] = sr.iloc[:, 0]
    feat['srsi_d'] = sr.iloc[:, 1]

# MACD
macd = ta.macd(c, fast=12, slow=26, signal=9)
if macd is not None:
    feat['macd'] = macd.iloc[:, 0]
    feat['macd_h'] = macd.iloc[:, 1]
    feat['macd_s'] = macd.iloc[:, 2]

# ROC
feat['roc5'] = ta.roc(c, length=5)
feat['roc10'] = ta.roc(c, length=10)
feat['roc20'] = ta.roc(c, length=20)

# EMAs
for el in [8, 21, 55, 100, 200]:
    e = ta.ema(c, length=el)
    feat[f'ema{el}_d'] = (c - e) / e * 100
feat['ema8_sl'] = ta.ema(c, length=8).pct_change(3) * 100
feat['ema21_sl'] = ta.ema(c, length=21).pct_change(5) * 100
feat['ema55_sl'] = ta.ema(c, length=55).pct_change(5) * 100

# Bollinger Bands
bb = ta.bbands(c, length=20, std=2.0)
if bb is not None:
    bw = bb.iloc[:, 2] - bb.iloc[:, 0]
    feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw
    feat['bb_w'] = bw / bb.iloc[:, 1] * 100

# Volume
feat['vr'] = v / v.rolling(20).mean()
feat['vr5'] = v / v.rolling(5).mean()

# Price action
feat['spr'] = (h - l) / c * 100
feat['body'] = abs(c - df['open']) / (h - l + 1e-10)
feat['upper_wick'] = (h - np.maximum(c, df['open'])) / (h - l + 1e-10)
feat['lower_wick'] = (np.minimum(c, df['open']) - l) / (h - l + 1e-10)

# ADX
ax = ta.adx(h, l, c, length=14)
if ax is not None:
    feat['adx'] = ax.iloc[:, 0]
    feat['dip'] = ax.iloc[:, 1]
    feat['dim'] = ax.iloc[:, 2]
    feat['di_diff'] = feat['dip'] - feat['dim']

# Choppiness
atr_1 = ta.atr(h, l, c, length=1)
atr_sum = atr_1.rolling(14).sum()
high_max = h.rolling(14).max()
low_min = l.rolling(14).min()
feat['chop'] = 100 * np.log10(atr_sum / (high_max - low_min + 1e-10)) / np.log10(14)

# Time features
hr = df.index.hour
dw = df.index.dayofweek
feat['h_s'] = np.sin(2 * np.pi * hr / 24)
feat['h_c'] = np.cos(2 * np.pi * hr / 24)
feat['d_s'] = np.sin(2 * np.pi * dw / 7)
feat['d_c'] = np.cos(2 * np.pi * dw / 7)

# Lag features
for lag in [1, 2, 3]:
    feat[f'ret1_lag{lag}'] = feat['ret_1'].shift(lag)
    feat[f'rsi14_lag{lag}'] = feat['rsi14'].shift(lag)

# =============================================================================
# 3. MACRO FEATURES (Option 2)
# =============================================================================
print('\n[3] Adding macro features...')

# Volume momentum (proxy for institutional activity)
feat['vol_mom_5'] = v.pct_change(5)
feat['vol_mom_20'] = v.pct_change(20)
feat['vol_trend'] = v.rolling(5).mean() / v.rolling(20).mean()

# Price-volume divergence
feat['pv_div'] = feat['ret_5'] * feat['vol_mom_5'].apply(np.sign)

# Volatility regime
feat['vol_regime'] = feat['vol20'] / feat['vol20'].rolling(50).mean()
feat['vol_expansion'] = (feat['vol5'] > feat['vol20']).astype(int)

# Range analysis (proxy for consolidation/breakout)
feat['range_20'] = (h.rolling(20).max() - l.rolling(20).min()) / c * 100
feat['range_ratio'] = feat['spr'] / feat['range_20']

# Momentum divergence (RSI vs price)
price_higher = (c > c.shift(10)).astype(int)
rsi_higher = (feat['rsi14'] > feat['rsi14'].shift(10)).astype(int)
feat['mom_div'] = price_higher - rsi_higher

# Weekly/monthly returns (longer cycles)
feat['ret_30'] = c.pct_change(30)  # ~5 days
feat['ret_42'] = c.pct_change(42)  # ~1 week

# Higher timeframe trend proxy
feat['ema_stack'] = ((feat['ema8_d'] > 0).astype(int) +
                     (feat['ema21_d'] > 0).astype(int) +
                     (feat['ema55_d'] > 0).astype(int) +
                     (feat['ema100_d'] > 0).astype(int) +
                     (feat['ema200_d'] > 0).astype(int))

print(f'    Added 12 macro features')

# =============================================================================
# 4. PREPARE TEST DATA
# =============================================================================
feat = feat.dropna()

# Test period: Feb 2026
test_period = feat[feat.index >= '2026-02-01'].copy()
train_period = feat[feat.index < '2026-02-01'].copy()

print(f'\n[4] Test period: {len(test_period)} samples (Feb 2026)')

# =============================================================================
# 5. BACKTEST FUNCTION
# =============================================================================
def backtest(preds_series, df_ref, conv_min=1.0):
    trades = []
    for idx, pred in preds_series.items():
        if abs(pred) / 0.005 < conv_min:
            continue
        try:
            loc = df_ref.index.get_loc(idx)
        except:
            continue
        if loc + 1 >= len(df_ref):
            continue

        direction = 1 if pred > 0 else -1
        entry = df_ref.iloc[loc + 1]['open']
        tp = entry * (1 + TP_PCT) if direction == 1 else entry * (1 - TP_PCT)
        sl = entry * (1 - SL_PCT) if direction == 1 else entry * (1 + SL_PCT)

        pnl = None
        for j in range(loc + 1, min(loc + MAX_HOLD + 1, len(df_ref))):
            candle = df_ref.iloc[j]
            if direction == 1:
                if candle['low'] <= sl:
                    pnl = -SL_PCT
                    break
                if candle['high'] >= tp:
                    pnl = TP_PCT
                    break
            else:
                if candle['high'] >= sl:
                    pnl = -SL_PCT
                    break
                if candle['low'] <= tp:
                    pnl = TP_PCT
                    break
        if pnl is None:
            exit_idx = min(loc + MAX_HOLD, len(df_ref) - 1)
            exit_p = df_ref.iloc[exit_idx]['close']
            pnl = (exit_p - entry) / entry * direction

        trades.append({'pnl': pnl, 'win': pnl > 0})

    if not trades:
        return 0, 0, 0
    n = len(trades)
    wins = sum(t['win'] for t in trades)
    wr = wins / n * 100
    pnl_total = sum(t['pnl'] for t in trades) * 100
    return n, wr, pnl_total

# =============================================================================
# 6. EXPERIMENT 1: HIGHER CONVICTION THRESHOLD
# =============================================================================
print('\n' + '='*70)
print('EXPERIMENT 1: CONVICTION THRESHOLD')
print('='*70)

v2_preds = pd.Series(v2_model.predict(test_period[v2_cols]), index=test_period.index)

print(f'\n{"Conv":>6} {"Trades":>8} {"WR":>8} {"PnL":>10}')
print('-'*35)

for conv in [0.5, 1.0, 1.5, 2.0, 2.5]:
    n, wr, pnl = backtest(v2_preds, df, conv)
    if n > 0:
        print(f'{conv:>6.1f} {n:>8} {wr:>7.1f}% {pnl:>+9.2f}%')
    else:
        print(f'{conv:>6.1f} {n:>8}       -          -')

# =============================================================================
# 7. EXPERIMENT 2: MACRO FEATURES MODEL
# =============================================================================
print('\n' + '='*70)
print('EXPERIMENT 2: MACRO FEATURES MODEL')
print('='*70)

# Extended feature columns
macro_cols = v2_cols + [
    'vol_mom_5', 'vol_mom_20', 'vol_trend', 'pv_div',
    'vol_regime', 'vol_expansion', 'range_20', 'range_ratio',
    'mom_div', 'ret_30', 'ret_42', 'ema_stack'
]

# Check all columns exist
missing = [c for c in macro_cols if c not in feat.columns]
if missing:
    print(f'    Missing columns: {missing}')
    macro_cols = [c for c in macro_cols if c in feat.columns]

print(f'    Training with {len(macro_cols)} features...')

# Train new model with macro features
train_macro = train_period[macro_cols].dropna()
train_target = train_period.loc[train_macro.index, 'ret_1'].shift(-1).dropna()
common_idx = train_macro.index.intersection(train_target.index)
train_macro = train_macro.loc[common_idx]
train_target = train_target.loc[common_idx]

# Use same hyperparams as V2
macro_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
)

macro_model.fit(train_macro, train_target)

# Test
test_macro = test_period[macro_cols].dropna()
macro_preds = pd.Series(macro_model.predict(test_macro), index=test_macro.index)

print(f'\n{"Conv":>6} {"Trades":>8} {"WR":>8} {"PnL":>10}')
print('-'*35)

for conv in [0.5, 1.0, 1.5, 2.0]:
    n, wr, pnl = backtest(macro_preds, df, conv)
    if n > 0:
        print(f'{conv:>6.1f} {n:>8} {wr:>7.1f}% {pnl:>+9.2f}%')
    else:
        print(f'{conv:>6.1f} {n:>8}       -          -')

# Feature importance for new features
print('\n    New feature importance (macro only):')
macro_importance = pd.DataFrame({
    'feature': macro_cols,
    'importance': macro_model.feature_importances_
}).sort_values('importance', ascending=False)

new_feats = [f for f in macro_cols if f not in v2_cols]
new_importance = macro_importance[macro_importance['feature'].isin(new_feats)]
for _, row in new_importance.iterrows():
    print(f'      {row["feature"]:<15} {row["importance"]:.4f}')

# =============================================================================
# 8. EXPERIMENT 3: ENSEMBLE V2 + V3
# =============================================================================
print('\n' + '='*70)
print('EXPERIMENT 3: ENSEMBLE V2 + V3')
print('='*70)

# Simple average ensemble
v3_preds = pd.Series(v3_model.predict(test_period[v3_cols]), index=test_period.index)

# Align predictions
common_idx = v2_preds.index.intersection(v3_preds.index)
v2_aligned = v2_preds.loc[common_idx]
v3_aligned = v3_preds.loc[common_idx]

# Different ensemble strategies
print('\n    Strategy: Simple Average')
ensemble_avg = (v2_aligned + v3_aligned) / 2

print(f'\n{"Conv":>6} {"Trades":>8} {"WR":>8} {"PnL":>10}')
print('-'*35)

for conv in [0.5, 1.0, 1.5, 2.0]:
    n, wr, pnl = backtest(ensemble_avg, df, conv)
    if n > 0:
        print(f'{conv:>6.1f} {n:>8} {wr:>7.1f}% {pnl:>+9.2f}%')
    else:
        print(f'{conv:>6.1f} {n:>8}       -          -')

# Weighted ensemble (V2 has better track record)
print('\n    Strategy: Weighted (70% V2, 30% V3)')
ensemble_weighted = v2_aligned * 0.7 + v3_aligned * 0.3

print(f'\n{"Conv":>6} {"Trades":>8} {"WR":>8} {"PnL":>10}')
print('-'*35)

for conv in [0.5, 1.0, 1.5, 2.0]:
    n, wr, pnl = backtest(ensemble_weighted, df, conv)
    if n > 0:
        print(f'{conv:>6.1f} {n:>8} {wr:>7.1f}% {pnl:>+9.2f}%')
    else:
        print(f'{conv:>6.1f} {n:>8}       -          -')

# Agreement filter (only trade when both agree on direction)
print('\n    Strategy: Agreement Filter (both must agree)')
agreement_mask = (v2_aligned > 0) == (v3_aligned > 0)
ensemble_agreement = v2_aligned.copy()
ensemble_agreement[~agreement_mask] = 0  # Zero out disagreements

print(f'\n{"Conv":>6} {"Trades":>8} {"WR":>8} {"PnL":>10}')
print('-'*35)

for conv in [0.5, 1.0, 1.5, 2.0]:
    n, wr, pnl = backtest(ensemble_agreement, df, conv)
    if n > 0:
        print(f'{conv:>6.1f} {n:>8} {wr:>7.1f}% {pnl:>+9.2f}%')
    else:
        print(f'{conv:>6.1f} {n:>8}       -          -')

# =============================================================================
# 9. FINAL COMPARISON
# =============================================================================
print('\n' + '='*70)
print('FINAL COMPARISON (Conv >= 1.0)')
print('='*70)

results = []

# Baseline V2
n, wr, pnl = backtest(v2_preds, df, 1.0)
results.append(('V2 Baseline (conv>=1.0)', n, wr, pnl))

# V2 high conviction
n, wr, pnl = backtest(v2_preds, df, 1.5)
results.append(('V2 High Conv (>=1.5)', n, wr, pnl))

n, wr, pnl = backtest(v2_preds, df, 2.0)
results.append(('V2 Very High Conv (>=2.0)', n, wr, pnl))

# Macro model
n, wr, pnl = backtest(macro_preds, df, 1.0)
results.append(('Macro Features Model', n, wr, pnl))

# Ensembles
n, wr, pnl = backtest(ensemble_avg, df, 1.0)
results.append(('Ensemble Avg V2+V3', n, wr, pnl))

n, wr, pnl = backtest(ensemble_weighted, df, 1.0)
results.append(('Ensemble Weighted 70/30', n, wr, pnl))

n, wr, pnl = backtest(ensemble_agreement, df, 1.0)
results.append(('Ensemble Agreement', n, wr, pnl))

print(f'\n{"Strategy":<30} {"Trades":>8} {"WR":>8} {"PnL":>10}')
print('-'*60)

for name, n, wr, pnl in results:
    if n > 0:
        print(f'{name:<30} {n:>8} {wr:>7.1f}% {pnl:>+9.2f}%')
    else:
        print(f'{name:<30} {n:>8}       -          -')

# Find best
best = max(results, key=lambda x: x[3] if x[1] > 0 else -999)
print(f'\nBest strategy: {best[0]}')
print(f'  {best[1]} trades, {best[2]:.1f}% WR, {best[3]:+.2f}% PnL')

# =============================================================================
# 10. SAVE BEST MODEL IF IMPROVED
# =============================================================================
print('\n' + '='*70)
print('RECOMMENDATIONS')
print('='*70)

baseline_pnl = results[0][3]  # V2 baseline
improvements = [(name, n, wr, pnl) for name, n, wr, pnl in results if pnl > baseline_pnl and n >= 5]

if improvements:
    print('\nStrategies that IMPROVED over baseline:')
    for name, n, wr, pnl in improvements:
        improvement = pnl - baseline_pnl
        print(f'  - {name}: +{improvement:.2f}% improvement')
else:
    print('\nNo strategy significantly improved over V2 baseline.')
    print('Consider: V2 with conv >= 1.5 for fewer but higher quality trades.')

print('\n[COMPLETED]')
