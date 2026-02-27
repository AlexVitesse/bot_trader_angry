"""
BTC Experiments V2 - More Advanced Approaches
==============================================
1. CatBoost algorithm
2. Meta-filter (predict when V2 is correct)
3. Multi-timeframe features
4. Optimized TP/SL ratios
5. Different target: TP hit before SL
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

print('='*70)
print('BTC EXPERIMENTS V2 - ADVANCED APPROACHES')
print('='*70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print('\n[1] Loading data...')

df = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_full.parquet')
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')

print(f'    {len(df)} candles')

# Load V2 model for meta-filter experiment
v2_data = joblib.load(MODELS_DIR / 'btc_v2_gradientboosting.pkl')
v2_model = v2_data['model']
v2_cols = v2_data['feature_cols']

# =============================================================================
# 2. BASE FEATURES (same as V2)
# =============================================================================
print('\n[2] Calculating base features...')

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

base_feature_cols = list(feat.columns)

# =============================================================================
# 3. MULTI-TIMEFRAME FEATURES
# =============================================================================
print('\n[3] Adding multi-timeframe features...')

# Daily context (6 candles of 4h = 1 day)
feat['daily_ret'] = c.pct_change(6)
feat['daily_high'] = h.rolling(6).max()
feat['daily_low'] = l.rolling(6).min()
feat['daily_range'] = (feat['daily_high'] - feat['daily_low']) / c * 100
feat['pos_in_daily'] = (c - feat['daily_low']) / (feat['daily_high'] - feat['daily_low'] + 1e-10)

# Weekly context (42 candles of 4h = 1 week)
feat['weekly_ret'] = c.pct_change(42)
feat['weekly_high'] = h.rolling(42).max()
feat['weekly_low'] = l.rolling(42).min()
feat['weekly_range'] = (feat['weekly_high'] - feat['weekly_low']) / c * 100
feat['pos_in_weekly'] = (c - feat['weekly_low']) / (feat['weekly_high'] - feat['weekly_low'] + 1e-10)

# Higher TF RSI
feat['rsi_daily'] = ta.rsi(c.rolling(6).mean(), length=14)
feat['rsi_weekly'] = ta.rsi(c.rolling(42).mean(), length=14)

# Trend alignment
ema_d = ta.ema(c, length=6)
ema_w = ta.ema(c, length=42)
feat['trend_align'] = ((c > ema_d).astype(int) + (c > ema_w).astype(int) +
                       (ema_d > ema_w).astype(int)) / 3

mtf_feature_cols = base_feature_cols + [
    'daily_ret', 'daily_range', 'pos_in_daily',
    'weekly_ret', 'weekly_range', 'pos_in_weekly',
    'rsi_daily', 'rsi_weekly', 'trend_align'
]

print(f'    Added 9 multi-TF features')

# =============================================================================
# 4. TRADE SIMULATION FUNCTION
# =============================================================================
def simulate_trade(idx, direction, df_ref, tp_pct, sl_pct, max_hold=20):
    """Simulate trade with custom TP/SL."""
    try:
        loc = df_ref.index.get_loc(idx)
    except:
        return None, None

    if loc + 1 >= len(df_ref):
        return None, None

    entry = df_ref.iloc[loc + 1]['open']
    tp = entry * (1 + tp_pct) if direction == 1 else entry * (1 - tp_pct)
    sl = entry * (1 - sl_pct) if direction == 1 else entry * (1 + sl_pct)

    for j in range(loc + 1, min(loc + max_hold + 1, len(df_ref))):
        candle = df_ref.iloc[j]
        if direction == 1:
            if candle['low'] <= sl:
                return -sl_pct, 'SL'
            if candle['high'] >= tp:
                return tp_pct, 'TP'
        else:
            if candle['high'] >= sl:
                return -sl_pct, 'SL'
            if candle['low'] <= tp:
                return tp_pct, 'TP'

    exit_idx = min(loc + max_hold, len(df_ref) - 1)
    exit_p = df_ref.iloc[exit_idx]['close']
    pnl = (exit_p - entry) / entry * direction
    return pnl, 'TIME'

def backtest(preds_series, df_ref, conv_min, tp_pct, sl_pct):
    """Backtest with configurable TP/SL."""
    trades = []
    for idx, pred in preds_series.items():
        if abs(pred) / 0.005 < conv_min:
            continue
        direction = 1 if pred > 0 else -1
        pnl, exit_type = simulate_trade(idx, direction, df_ref, tp_pct, sl_pct)
        if pnl is not None:
            trades.append({'pnl': pnl, 'win': pnl > 0, 'exit': exit_type})

    if not trades:
        return 0, 0, 0, {}
    n = len(trades)
    wr = sum(t['win'] for t in trades) / n * 100
    pnl_total = sum(t['pnl'] for t in trades) * 100
    exits = {e: sum(1 for t in trades if t['exit'] == e) for e in ['TP', 'SL', 'TIME']}
    return n, wr, pnl_total, exits

# =============================================================================
# 5. CREATE LABELS
# =============================================================================
print('\n[4] Creating trade labels...')

# Label: did TP hit before SL? (for LONG)
tp_pct_default = 0.03
sl_pct_default = 0.015

labels_pnl = pd.Series(index=feat.index, dtype=float)
labels_tp_first = pd.Series(index=feat.index, dtype=int)

for idx in feat.index:
    pnl, exit_type = simulate_trade(idx, 1, df, tp_pct_default, sl_pct_default)
    if pnl is not None:
        labels_pnl.loc[idx] = pnl
        labels_tp_first.loc[idx] = 1 if exit_type == 'TP' else 0

feat['trade_pnl'] = labels_pnl
feat['tp_first'] = labels_tp_first
feat['target'] = c.pct_change().shift(-1)

feat = feat.dropna()
print(f'    {len(feat)} samples')

# Split
test_data = feat[feat.index >= '2026-02-01'].copy()
val_data = feat[(feat.index >= '2025-10-01') & (feat.index < '2026-02-01')].copy()
train_data = feat[feat.index < '2025-10-01'].copy()

print(f'    Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}')

# =============================================================================
# EXPERIMENT 1: CATBOOST
# =============================================================================
print('\n' + '='*70)
print('EXPERIMENT 1: CATBOOST')
print('='*70)

try:
    from catboost import CatBoostRegressor
    HAVE_CATBOOST = True
except ImportError:
    HAVE_CATBOOST = False
    print('    CatBoost not installed. Installing...')
    import subprocess
    subprocess.run(['pip', 'install', 'catboost', '-q'], capture_output=True)
    try:
        from catboost import CatBoostRegressor
        HAVE_CATBOOST = True
    except:
        print('    Could not install CatBoost. Skipping.')

if HAVE_CATBOOST:
    print('    Training CatBoost...')

    X_train = train_data[base_feature_cols]
    y_train = train_data['target']

    cat_model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        subsample=0.8,
        random_state=42,
        verbose=False
    )

    cat_model.fit(X_train, y_train)

    # Test predictions
    cat_preds = pd.Series(cat_model.predict(test_data[base_feature_cols]), index=test_data.index)

    print(f'\n    Results:')
    print(f'    {"Conv>=":>8} {"Trades":>8} {"WR":>8} {"PnL":>10}')
    print('    ' + '-'*38)

    for conv in [0.5, 1.0, 1.5, 2.0]:
        n, wr, pnl, _ = backtest(cat_preds, df, conv, tp_pct_default, sl_pct_default)
        if n > 0:
            print(f'    {conv:>8.1f} {n:>8} {wr:>7.1f}% {pnl:>+9.2f}%')
        else:
            print(f'    {conv:>8.1f} {n:>8}       -          -')

# =============================================================================
# EXPERIMENT 2: META-FILTER (Predict when V2 is correct)
# =============================================================================
print('\n' + '='*70)
print('EXPERIMENT 2: META-FILTER (Predict when V2 is correct)')
print('='*70)

print('    Generating V2 predictions on train data...')

# Get V2 predictions on all data
v2_preds_all = pd.Series(v2_model.predict(feat[v2_cols]), index=feat.index)

# For each prediction, did V2 get it right?
v2_correct = pd.Series(index=feat.index, dtype=int)
for idx in feat.index:
    pred = v2_preds_all.loc[idx]
    direction = 1 if pred > 0 else -1
    pnl, _ = simulate_trade(idx, direction, df, tp_pct_default, sl_pct_default)
    if pnl is not None:
        v2_correct.loc[idx] = 1 if pnl > 0 else 0

feat['v2_correct'] = v2_correct
feat['v2_pred'] = v2_preds_all
feat['v2_conviction'] = abs(v2_preds_all) / 0.005

# Meta features: V2 prediction characteristics + market state
meta_features = base_feature_cols + ['v2_pred', 'v2_conviction']

print('    Training meta-filter classifier...')

# Train meta-filter on V2's performance - get fresh data with v2_correct
train_meta = feat[feat.index < '2025-10-01'].copy()
train_meta = train_meta.dropna(subset=['v2_correct'])
X_meta = train_meta[meta_features]
y_meta = train_meta['v2_correct']

meta_clf = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42
)
meta_clf.fit(X_meta, y_meta)

# Get meta predictions on test
test_meta = feat[feat.index >= '2026-02-01'].copy()

meta_proba = pd.Series(
    meta_clf.predict_proba(test_meta[meta_features])[:, 1],
    index=test_meta.index
)

print(f'\n    Using meta-filter to filter V2 predictions:')
print(f'    {"P(correct)>":>12} {"Trades":>8} {"WR":>8} {"PnL":>10}')
print('    ' + '-'*42)

v2_test_preds = v2_preds_all.loc[test_meta.index]

for thresh in [0.5, 0.55, 0.6, 0.65, 0.7]:
    # Only trade when meta-filter says V2 will be correct
    filtered_mask = meta_proba >= thresh
    filtered_preds = v2_test_preds[filtered_mask]

    if len(filtered_preds) == 0:
        print(f'    {thresh:>12.0%} {0:>8}       -          -')
        continue

    # Also apply conviction filter
    trades = []
    for idx, pred in filtered_preds.items():
        if abs(pred) / 0.005 < 1.0:  # Conv >= 1.0
            continue
        direction = 1 if pred > 0 else -1
        pnl, _ = simulate_trade(idx, direction, df, tp_pct_default, sl_pct_default)
        if pnl is not None:
            trades.append({'pnl': pnl, 'win': pnl > 0})

    if trades:
        n = len(trades)
        wr = sum(t['win'] for t in trades) / n * 100
        pnl_total = sum(t['pnl'] for t in trades) * 100
        print(f'    {thresh:>12.0%} {n:>8} {wr:>7.1f}% {pnl_total:>+9.2f}%')
    else:
        print(f'    {thresh:>12.0%} {0:>8}       -          -')

# =============================================================================
# EXPERIMENT 3: MULTI-TIMEFRAME MODEL
# =============================================================================
print('\n' + '='*70)
print('EXPERIMENT 3: MULTI-TIMEFRAME MODEL')
print('='*70)

# Check available columns
available_mtf = [c for c in mtf_feature_cols if c in feat.columns]
print(f'    Using {len(available_mtf)} features (including MTF)...')

X_train_mtf = train_data[available_mtf]
y_train_mtf = train_data['target']

mtf_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

mtf_model.fit(X_train_mtf, y_train_mtf)

mtf_preds = pd.Series(mtf_model.predict(test_data[available_mtf]), index=test_data.index)

print(f'\n    Results:')
print(f'    {"Conv>=":>8} {"Trades":>8} {"WR":>8} {"PnL":>10}')
print('    ' + '-'*38)

for conv in [0.5, 1.0, 1.5, 2.0]:
    n, wr, pnl, _ = backtest(mtf_preds, df, conv, tp_pct_default, sl_pct_default)
    if n > 0:
        print(f'    {conv:>8.1f} {n:>8} {wr:>7.1f}% {pnl:>+9.2f}%')
    else:
        print(f'    {conv:>8.1f} {n:>8}       -          -')

# Feature importance for MTF features
mtf_importance = pd.DataFrame({
    'feature': available_mtf,
    'importance': mtf_model.feature_importances_
}).sort_values('importance', ascending=False)

new_mtf = [f for f in available_mtf if f not in base_feature_cols]
print(f'\n    MTF feature importance:')
for f in new_mtf:
    imp = mtf_importance[mtf_importance['feature'] == f]['importance'].values
    if len(imp) > 0:
        print(f'      {f:<20} {imp[0]:.4f}')

# =============================================================================
# EXPERIMENT 4: OPTIMIZED TP/SL
# =============================================================================
print('\n' + '='*70)
print('EXPERIMENT 4: OPTIMIZED TP/SL RATIOS')
print('='*70)

# Use V2 model predictions, test different TP/SL combinations
v2_test_preds = pd.Series(v2_model.predict(test_data[v2_cols]), index=test_data.index)

tp_sl_combos = [
    (0.02, 0.01),   # 2% TP, 1% SL (2:1)
    (0.025, 0.0125), # 2.5% TP, 1.25% SL (2:1)
    (0.03, 0.015),   # 3% TP, 1.5% SL (2:1) - current
    (0.03, 0.01),    # 3% TP, 1% SL (3:1)
    (0.04, 0.02),    # 4% TP, 2% SL (2:1)
    (0.04, 0.01),    # 4% TP, 1% SL (4:1)
    (0.02, 0.02),    # 2% TP, 2% SL (1:1)
    (0.015, 0.01),   # 1.5% TP, 1% SL (1.5:1)
]

print(f'\n    Testing TP/SL combinations (conv >= 1.0):')
print(f'    {"TP":>6} {"SL":>6} {"Ratio":>6} {"Trades":>8} {"WR":>8} {"PnL":>10} {"TP%":>6} {"SL%":>6}')
print('    ' + '-'*65)

best_combo = None
best_pnl = -999

for tp, sl in tp_sl_combos:
    ratio = tp / sl
    n, wr, pnl, exits = backtest(v2_test_preds, df, 1.0, tp, sl)

    if n > 0:
        tp_pct_exits = exits.get('TP', 0) / n * 100
        sl_pct_exits = exits.get('SL', 0) / n * 100
        print(f'    {tp*100:>5.1f}% {sl*100:>5.1f}% {ratio:>5.1f}x {n:>8} {wr:>7.1f}% {pnl:>+9.2f}% {tp_pct_exits:>5.1f}% {sl_pct_exits:>5.1f}%')

        if pnl > best_pnl:
            best_pnl = pnl
            best_combo = (tp, sl, ratio, n, wr, pnl)

if best_combo:
    print(f'\n    Best: TP={best_combo[0]*100:.1f}%, SL={best_combo[1]*100:.1f}% ({best_combo[2]:.1f}x)')
    print(f'          {best_combo[3]} trades, {best_combo[4]:.1f}% WR, {best_combo[5]:+.2f}% PnL')

# =============================================================================
# EXPERIMENT 5: PREDICT TP BEFORE SL (Binary target)
# =============================================================================
print('\n' + '='*70)
print('EXPERIMENT 5: PREDICT TP BEFORE SL (Binary Classification)')
print('='*70)

X_train_tp = train_data[base_feature_cols]
y_train_tp = train_data['tp_first']

tp_clf = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    random_state=42
)

print('    Training TP-first classifier...')
tp_clf.fit(X_train_tp, y_train_tp)

tp_proba = pd.Series(
    tp_clf.predict_proba(test_data[base_feature_cols])[:, 1],
    index=test_data.index
)

print(f'\n    Results (LONG only when P(TP first) > threshold):')
print(f'    {"P(TP)>":>10} {"Trades":>8} {"WR":>8} {"PnL":>10}')
print('    ' + '-'*40)

for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
    trades = []
    for idx in tp_proba.index:
        if tp_proba.loc[idx] < thresh:
            continue
        pnl, _ = simulate_trade(idx, 1, df, tp_pct_default, sl_pct_default)
        if pnl is not None:
            trades.append({'pnl': pnl, 'win': pnl > 0})

    if trades:
        n = len(trades)
        wr = sum(t['win'] for t in trades) / n * 100
        pnl_total = sum(t['pnl'] for t in trades) * 100
        print(f'    {thresh:>10.0%} {n:>8} {wr:>7.1f}% {pnl_total:>+9.2f}%')
    else:
        print(f'    {thresh:>10.0%} {0:>8}       -          -')

# =============================================================================
# FINAL COMPARISON
# =============================================================================
print('\n' + '='*70)
print('FINAL COMPARISON')
print('='*70)

results = []

# V2 Baseline
n, wr, pnl, _ = backtest(v2_test_preds, df, 1.0, 0.03, 0.015)
results.append(('V2 Baseline (3%/1.5%)', n, wr, pnl))

# CatBoost
if HAVE_CATBOOST:
    n, wr, pnl, _ = backtest(cat_preds, df, 1.0, 0.03, 0.015)
    results.append(('CatBoost', n, wr, pnl))

# Meta-filter
filtered_mask = meta_proba >= 0.55
filtered_preds = v2_test_preds[filtered_mask]
trades = []
for idx, pred in filtered_preds.items():
    if abs(pred) / 0.005 < 1.0:
        continue
    direction = 1 if pred > 0 else -1
    pnl_t, _ = simulate_trade(idx, direction, df, 0.03, 0.015)
    if pnl_t is not None:
        trades.append({'pnl': pnl_t, 'win': pnl_t > 0})
if trades:
    n = len(trades)
    wr = sum(t['win'] for t in trades) / n * 100
    pnl = sum(t['pnl'] for t in trades) * 100
    results.append(('Meta-filter P>55%', n, wr, pnl))

# MTF Model
n, wr, pnl, _ = backtest(mtf_preds, df, 1.0, 0.03, 0.015)
results.append(('Multi-TF Model', n, wr, pnl))

# Best TP/SL
if best_combo:
    tp_opt, sl_opt = best_combo[0], best_combo[1]
    n, wr, pnl, _ = backtest(v2_test_preds, df, 1.0, tp_opt, sl_opt)
    results.append((f'V2 + TP/SL opt ({tp_opt*100:.0f}%/{sl_opt*100:.0f}%)', n, wr, pnl))

# TP-first classifier
trades = []
for idx in tp_proba.index:
    if tp_proba.loc[idx] < 0.6:
        continue
    pnl_t, _ = simulate_trade(idx, 1, df, 0.03, 0.015)
    if pnl_t is not None:
        trades.append({'pnl': pnl_t, 'win': pnl_t > 0})
if trades:
    n = len(trades)
    wr = sum(t['win'] for t in trades) / n * 100
    pnl = sum(t['pnl'] for t in trades) * 100
    results.append(('TP-first clf P>60%', n, wr, pnl))

print(f'\n{"Strategy":<35} {"Trades":>8} {"WR":>8} {"PnL":>10}')
print('-'*65)

for name, n, wr, pnl in results:
    if n > 0:
        print(f'{name:<35} {n:>8} {wr:>7.1f}% {pnl:>+9.2f}%')
    else:
        print(f'{name:<35} {n:>8}       -          -')

# Find best
valid_results = [(name, n, wr, pnl) for name, n, wr, pnl in results if n >= 5]
if valid_results:
    best = max(valid_results, key=lambda x: x[3])
    baseline_pnl = results[0][3]

    print(f'\nBest strategy: {best[0]}')
    print(f'  {best[1]} trades, {best[2]:.1f}% WR, {best[3]:+.2f}% PnL')

    if best[3] > baseline_pnl:
        print(f'\n  [IMPROVEMENT] +{best[3] - baseline_pnl:.2f}% vs baseline!')
    else:
        print(f'\n  No improvement over baseline.')

print('\n[COMPLETED]')
