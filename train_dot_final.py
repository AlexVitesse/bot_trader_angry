"""Train and save DOT V14 models with optimized parameters"""
import warnings
import numpy as np
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')

TP_PCT = 0.05
SL_PCT = 0.03
TIMEOUT = 15

FEATURE_COLS = ['rsi', 'macd_norm', 'adx', 'bb_pct', 'atr_pct',
                'ret_3', 'ret_5', 'ret_10', 'vol_ratio', 'trend']

df = pd.read_csv('data/DOTUSDT_4h.csv', parse_dates=['timestamp'])

# Features
df['rsi'] = ta.momentum.rsi(df['close'], window=14) / 100
macd = ta.trend.MACD(df['close'])
df['macd_norm'] = macd.macd() / df['close']
df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14) / 100
bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
df['bb_pct'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-10)
df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
df['atr_pct'] = df['atr'] / df['close']
df['ret_3'] = df['close'].pct_change(3)
df['ret_5'] = df['close'].pct_change(5)
df['ret_10'] = df['close'].pct_change(10)
df['vol_sma'] = df['volume'].rolling(20).mean()
df['vol_ratio'] = df['volume'] / df['vol_sma']
df['sma_50'] = df['close'].rolling(50).mean()
df['trend'] = (df['close'] > df['sma_50']).astype(float)
df = df.dropna()

# Create labels
labels = []
for i in range(len(df) - TIMEOUT - 1):
    entry = df['close'].iloc[i]
    future = df.iloc[i+1:i+TIMEOUT+1]
    tp_price = entry * (1 + TP_PCT)
    sl_price = entry * (1 - SL_PCT)
    won = False
    for _, row in future.iterrows():
        if row['high'] >= tp_price:
            won = True
            break
        if row['low'] <= sl_price:
            break
    labels.append(1 if won else 0)

df_labeled = df.iloc[:len(labels)].copy()
df_labeled['label'] = labels

X = df_labeled[FEATURE_COLS].values
y = df_labeled['label'].values

# Scale and train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)

rf.fit(X_scaled, y)
gb.fit(X_scaled, y)

# Save models
output_dir = Path('strategies/dot_v14/models')
output_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(rf, output_dir / 'random_forest.pkl')
joblib.dump(gb, output_dir / 'gradient_boosting.pkl')
joblib.dump(scaler, output_dir / 'scaler.pkl')

metadata = {
    'feature_cols': FEATURE_COLS,
    'voting_threshold': 2,
    'tp_pct': TP_PCT,
    'sl_pct': SL_PCT,
    'timeout_candles': TIMEOUT,
    'training_samples': len(X),
    'models': ['RandomForest', 'GradientBoosting'],
}
joblib.dump(metadata, output_dir / 'metadata.pkl')

print("DOT V14 Models Saved Successfully")
print(f"  Output: {output_dir}")
print(f"  Samples: {len(X)}")
print(f"  TP/SL: {TP_PCT*100:.0f}%/{SL_PCT*100:.0f}%")
