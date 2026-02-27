"""
BTC Advanced Improvement Experiments
=====================================
1. Classification model (predict WIN/LOSS)
2. Rolling window training (last 6 months only)
3. RANGE regime filter (only trade in RANGE)
4. Deep Learning LSTM
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

TP_PCT = 0.03
SL_PCT = 0.015
MAX_HOLD = 20

print('='*70)
print('BTC ADVANCED EXPERIMENTS')
print('='*70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print('\n[1] Loading data...')

df = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_full.parquet')
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')

print(f'    {len(df)} candles: {df.index[0].date()} to {df.index[-1].date()}')

# =============================================================================
# 2. CALCULATE FEATURES
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

feature_cols = [c for c in feat.columns]

# =============================================================================
# 3. CALCULATE REGIME
# =============================================================================
print('\n[3] Detecting market regime...')

ema20 = ta.ema(c, length=20)
ema50 = ta.ema(c, length=50)
ret_20d = c.pct_change(20 * 6)

regime = pd.Series(index=df.index, dtype=str)
for i in range(len(df)):
    if i < 200:
        regime.iloc[i] = 'UNKNOWN'
        continue

    price = c.iloc[i]
    e20 = ema20.iloc[i]
    e50 = ema50.iloc[i]
    r20 = ret_20d.iloc[i]

    if pd.isna(e20) or pd.isna(e50) or pd.isna(r20):
        regime.iloc[i] = 'UNKNOWN'
    elif price > e20 and e20 > e50 and r20 > 0.05:
        regime.iloc[i] = 'BULL'
    elif price < e20 and e20 < e50 and r20 < -0.05:
        regime.iloc[i] = 'BEAR'
    else:
        regime.iloc[i] = 'RANGE'

df['regime'] = regime
feat['regime'] = regime

# =============================================================================
# 4. SIMULATE TRADES TO GET WIN/LOSS LABELS
# =============================================================================
print('\n[4] Simulating trades for labels...')

def simulate_trade(idx, direction, df_ref):
    """Simulate a single trade and return PnL."""
    try:
        loc = df_ref.index.get_loc(idx)
    except:
        return None

    if loc + 1 >= len(df_ref):
        return None

    entry = df_ref.iloc[loc + 1]['open']
    tp = entry * (1 + TP_PCT) if direction == 1 else entry * (1 - TP_PCT)
    sl = entry * (1 - SL_PCT) if direction == 1 else entry * (1 + SL_PCT)

    pnl = None
    for j in range(loc + 1, min(loc + MAX_HOLD + 1, len(df_ref))):
        candle = df_ref.iloc[j]
        if direction == 1:
            if candle['low'] <= sl:
                return -SL_PCT
            if candle['high'] >= tp:
                return TP_PCT
        else:
            if candle['high'] >= sl:
                return -SL_PCT
            if candle['low'] <= tp:
                return TP_PCT

    # Timeout
    exit_idx = min(loc + MAX_HOLD, len(df_ref) - 1)
    exit_p = df_ref.iloc[exit_idx]['close']
    return (exit_p - entry) / entry * direction

# Create labels: for each point, what would happen if we went LONG?
print('    Creating trade outcome labels...')
labels = pd.Series(index=feat.index, dtype=float)
for idx in feat.index:
    pnl = simulate_trade(idx, 1, df)  # Simulate LONG
    if pnl is not None:
        labels.loc[idx] = pnl

feat['trade_pnl'] = labels
feat['is_win'] = (feat['trade_pnl'] > 0).astype(int)

# Drop rows without labels
feat = feat.dropna()
print(f'    {len(feat)} samples with labels')

# =============================================================================
# 5. BACKTEST FUNCTION
# =============================================================================
def backtest_predictions(preds_series, directions_series, df_ref, filter_mask=None):
    """Backtest with predictions and directions."""
    trades = []

    for idx in preds_series.index:
        if filter_mask is not None and idx in filter_mask.index:
            if not filter_mask.loc[idx]:
                continue

        pred = preds_series.loc[idx]
        direction = directions_series.loc[idx] if idx in directions_series.index else 1

        try:
            loc = df_ref.index.get_loc(idx)
        except:
            continue

        if loc + 1 >= len(df_ref):
            continue

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

def backtest_classifier(proba_series, threshold, df_ref, direction=1):
    """Backtest classifier: only trade when P(win) > threshold."""
    trades = []

    for idx, prob in proba_series.items():
        if prob < threshold:
            continue

        try:
            loc = df_ref.index.get_loc(idx)
        except:
            continue

        if loc + 1 >= len(df_ref):
            continue

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
# 6. SPLIT DATA
# =============================================================================
# Test: Feb 2026
# Validation: Oct 2025 - Jan 2026
# Train: Everything before Oct 2025

test_data = feat[feat.index >= '2026-02-01'].copy()
val_data = feat[(feat.index >= '2025-10-01') & (feat.index < '2026-02-01')].copy()
train_data = feat[feat.index < '2025-10-01'].copy()

print(f'\n    Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}')

# =============================================================================
# EXPERIMENT 1: CLASSIFICATION MODEL
# =============================================================================
print('\n' + '='*70)
print('EXPERIMENT 1: CLASSIFICATION MODEL (Predict WIN/LOSS)')
print('='*70)

X_train = train_data[feature_cols]
y_train = train_data['is_win']

X_val = val_data[feature_cols]
y_val = val_data['is_win']

X_test = test_data[feature_cols]
y_test = test_data['is_win']

clf = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
)

print('    Training classifier...')
clf.fit(X_train, y_train)

# Get probabilities
val_proba = pd.Series(clf.predict_proba(X_val)[:, 1], index=X_val.index)
test_proba = pd.Series(clf.predict_proba(X_test)[:, 1], index=X_test.index)

print(f'\n    Validation accuracy: {accuracy_score(y_val, clf.predict(X_val)):.1%}')
print(f'    Test accuracy: {accuracy_score(y_test, clf.predict(X_test)):.1%}')

print(f'\n    Backtest with probability thresholds (LONG only):')
print(f'    {"P(win)>":>10} {"Trades":>8} {"WR":>8} {"PnL":>10}')
print('    ' + '-'*40)

for thresh in [0.5, 0.55, 0.6, 0.65, 0.7]:
    n, wr, pnl = backtest_classifier(test_proba, thresh, df, direction=1)
    if n > 0:
        print(f'    {thresh:>10.0%} {n:>8} {wr:>7.1f}% {pnl:>+9.2f}%')
    else:
        print(f'    {thresh:>10.0%} {n:>8}       -          -')

# Save best classifier
joblib.dump({
    'model': clf,
    'feature_cols': feature_cols,
    'type': 'classifier'
}, MODELS_DIR / 'btc_classifier.pkl')

# =============================================================================
# EXPERIMENT 2: ROLLING WINDOW (Last 6 months only)
# =============================================================================
print('\n' + '='*70)
print('EXPERIMENT 2: ROLLING WINDOW (Train on last 6 months)')
print('='*70)

# Train only on recent data (6 months before test)
rolling_train = feat[(feat.index >= '2025-08-01') & (feat.index < '2026-02-01')].copy()

X_roll = rolling_train[feature_cols]
y_roll = rolling_train['trade_pnl'].shift(-1).dropna()
common_idx = X_roll.index.intersection(y_roll.index)
X_roll = X_roll.loc[common_idx]
y_roll = y_roll.loc[common_idx]

print(f'    Training on {len(X_roll)} recent samples (Aug 2025 - Jan 2026)...')

roll_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    min_samples_split=15,
    min_samples_leaf=8,
    random_state=42,
)

roll_model.fit(X_roll, y_roll)

# Predictions
roll_preds = pd.Series(roll_model.predict(X_test), index=X_test.index)

print(f'\n    Backtest with conviction thresholds:')
print(f'    {"Conv>=":>8} {"Trades":>8} {"WR":>8} {"PnL":>10}')
print('    ' + '-'*38)

def backtest_regressor(preds, df_ref, conv_min):
    trades = []
    for idx, pred in preds.items():
        if abs(pred) / 0.005 < conv_min:
            continue
        direction = 1 if pred > 0 else -1
        pnl = simulate_trade(idx, direction, df_ref)
        if pnl is not None:
            trades.append({'pnl': pnl, 'win': pnl > 0})

    if not trades:
        return 0, 0, 0
    n = len(trades)
    wr = sum(t['win'] for t in trades) / n * 100
    pnl_total = sum(t['pnl'] for t in trades) * 100
    return n, wr, pnl_total

for conv in [0.5, 1.0, 1.5, 2.0]:
    n, wr, pnl = backtest_regressor(roll_preds, df, conv)
    if n > 0:
        print(f'    {conv:>8.1f} {n:>8} {wr:>7.1f}% {pnl:>+9.2f}%')
    else:
        print(f'    {conv:>8.1f} {n:>8}       -          -')

# =============================================================================
# EXPERIMENT 3: RANGE REGIME FILTER
# =============================================================================
print('\n' + '='*70)
print('EXPERIMENT 3: RANGE REGIME FILTER (Only trade in RANGE)')
print('='*70)

# Load V2 model
v2_data = joblib.load(MODELS_DIR / 'btc_v2_gradientboosting.pkl')
v2_model = v2_data['model']
v2_cols = v2_data['feature_cols']

# Get V2 predictions on test
v2_preds = pd.Series(v2_model.predict(test_data[v2_cols]), index=test_data.index)

# Create regime filter
test_regime = df.loc[test_data.index, 'regime']

print(f'\n    Test period regime distribution:')
print(test_regime.value_counts())

print(f'\n    Backtest V2 by regime (conv >= 1.0):')
print(f'    {"Regime":>10} {"Trades":>8} {"WR":>8} {"PnL":>10}')
print('    ' + '-'*40)

for reg in ['BULL', 'BEAR', 'RANGE', 'ALL']:
    if reg == 'ALL':
        mask = pd.Series(True, index=test_data.index)
    else:
        mask = test_regime == reg

    filtered_preds = v2_preds[mask]
    n, wr, pnl = backtest_regressor(filtered_preds, df, 1.0)
    if n > 0:
        print(f'    {reg:>10} {n:>8} {wr:>7.1f}% {pnl:>+9.2f}%')
    else:
        print(f'    {reg:>10} {n:>8}       -          -')

# =============================================================================
# EXPERIMENT 4: DEEP LEARNING (LSTM)
# =============================================================================
print('\n' + '='*70)
print('EXPERIMENT 4: DEEP LEARNING (LSTM)')
print('='*70)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    print('    PyTorch not available. Skipping LSTM experiment.')

if HAVE_TORCH:
    print('    Preparing sequence data...')

    # Create sequences
    SEQ_LEN = 10

    def create_sequences(data, feature_cols, seq_len):
        X, y, indices = [], [], []
        values = data[feature_cols].values
        labels = data['is_win'].values

        for i in range(seq_len, len(data)):
            X.append(values[i-seq_len:i])
            y.append(labels[i])
            indices.append(data.index[i])

        return np.array(X), np.array(y), indices

    # Scale features
    scaler = StandardScaler()
    train_scaled = train_data.copy()
    train_scaled[feature_cols] = scaler.fit_transform(train_data[feature_cols])

    test_scaled = test_data.copy()
    test_scaled[feature_cols] = scaler.transform(test_data[feature_cols])

    X_train_seq, y_train_seq, _ = create_sequences(train_scaled, feature_cols, SEQ_LEN)
    X_test_seq, y_test_seq, test_indices = create_sequences(test_scaled, feature_cols, SEQ_LEN)

    print(f'    Train sequences: {len(X_train_seq)}')
    print(f'    Test sequences: {len(X_test_seq)}')

    # Define LSTM
    class LSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=0.2)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            out = self.fc(lstm_out[:, -1, :])
            return out.squeeze()

    # Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'    Device: {device}')

    model = LSTMClassifier(len(feature_cols)).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_seq),
        torch.FloatTensor(y_train_seq)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Train
    print('    Training LSTM...')
    model.train()
    for epoch in range(20):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f'      Epoch {epoch+1}/20, Loss: {total_loss/len(train_loader):.4f}')

    # Predict
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
        test_proba_lstm = model(X_test_tensor).cpu().numpy()

    lstm_proba = pd.Series(test_proba_lstm, index=test_indices)

    print(f'\n    Backtest with probability thresholds:')
    print(f'    {"P(win)>":>10} {"Trades":>8} {"WR":>8} {"PnL":>10}')
    print('    ' + '-'*40)

    for thresh in [0.5, 0.55, 0.6, 0.65, 0.7]:
        n, wr, pnl = backtest_classifier(lstm_proba, thresh, df, direction=1)
        if n > 0:
            print(f'    {thresh:>10.0%} {n:>8} {wr:>7.1f}% {pnl:>+9.2f}%')
        else:
            print(f'    {thresh:>10.0%} {n:>8}       -          -')

    # Save LSTM
    torch.save({
        'model_state': model.state_dict(),
        'scaler': scaler,
        'feature_cols': feature_cols,
        'seq_len': SEQ_LEN
    }, MODELS_DIR / 'btc_lstm.pt')

# =============================================================================
# FINAL COMPARISON
# =============================================================================
print('\n' + '='*70)
print('FINAL COMPARISON')
print('='*70)

results = []

# Baseline V2
n, wr, pnl = backtest_regressor(v2_preds, df, 1.0)
results.append(('V2 Baseline (conv>=1.0)', n, wr, pnl))

# Classifier
n, wr, pnl = backtest_classifier(test_proba, 0.55, df, direction=1)
results.append(('Classifier P(win)>55%', n, wr, pnl))

n, wr, pnl = backtest_classifier(test_proba, 0.6, df, direction=1)
results.append(('Classifier P(win)>60%', n, wr, pnl))

# Rolling window
n, wr, pnl = backtest_regressor(roll_preds, df, 1.0)
results.append(('Rolling 6mo (conv>=1.0)', n, wr, pnl))

# RANGE only
range_mask = test_regime == 'RANGE'
range_preds = v2_preds[range_mask]
n, wr, pnl = backtest_regressor(range_preds, df, 1.0)
results.append(('V2 RANGE only', n, wr, pnl))

# LSTM
if HAVE_TORCH:
    n, wr, pnl = backtest_classifier(lstm_proba, 0.55, df, direction=1)
    results.append(('LSTM P(win)>55%', n, wr, pnl))

    n, wr, pnl = backtest_classifier(lstm_proba, 0.6, df, direction=1)
    results.append(('LSTM P(win)>60%', n, wr, pnl))

print(f'\n{"Strategy":<30} {"Trades":>8} {"WR":>8} {"PnL":>10}')
print('-'*60)

for name, n, wr, pnl in results:
    if n > 0:
        print(f'{name:<30} {n:>8} {wr:>7.1f}% {pnl:>+9.2f}%')
    else:
        print(f'{name:<30} {n:>8}       -          -')

# Find best
valid_results = [(name, n, wr, pnl) for name, n, wr, pnl in results if n >= 5]
if valid_results:
    best = max(valid_results, key=lambda x: x[3])
    print(f'\nBest strategy: {best[0]}')
    print(f'  {best[1]} trades, {best[2]:.1f}% WR, {best[3]:+.2f}% PnL')

    baseline_pnl = results[0][3]
    if best[3] > baseline_pnl:
        print(f'\n  [IMPROVEMENT] +{best[3] - baseline_pnl:.2f}% vs baseline!')
    else:
        print(f'\n  No improvement over baseline.')

print('\n[COMPLETED]')
