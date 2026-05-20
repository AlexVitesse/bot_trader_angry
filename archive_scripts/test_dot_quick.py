"""Quick test for DOT with fewer combinations"""
import warnings
import numpy as np
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

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

print(f"DOT data: {len(df)} candles")

# Test 3 configurations
configs = [
    (0.05, 0.03, 15),  # Conservative
    (0.08, 0.04, 20),  # Wider
    (0.04, 0.02, 10),  # Tight
]

for tp, sl, timeout in configs:
    print(f"\nTP {tp*100:.0f}% / SL {sl*100:.0f}% / Timeout {timeout}:")

    # Create labels
    labels = []
    for i in range(len(df) - timeout - 1):
        entry = df['close'].iloc[i]
        future = df.iloc[i+1:i+timeout+1]
        tp_price = entry * (1 + tp)
        sl_price = entry * (1 - sl)
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

    # Walk-forward 8 folds
    n_folds = 8
    fold_size = len(df_labeled) // (n_folds + 1)
    scaler = StandardScaler()

    results = []
    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, len(df_labeled))

        X_train = scaler.fit_transform(X[:train_end])
        X_test = scaler.transform(X[test_start:test_end])
        y_train, y_test = y[:train_end], y[test_start:test_end]

        rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)

        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)

        prob_rf = rf.predict_proba(X_test)[:, 1]
        prob_gb = gb.predict_proba(X_test)[:, 1]

        # Vote: both agree
        trade_mask = (prob_rf > 0.5) & (prob_gb > 0.5)
        n_trades = trade_mask.sum()

        if n_trades > 0:
            wins = y_test[trade_mask].sum()
            pnl = wins * tp - (n_trades - wins) * sl
            results.append(pnl * 100)
        else:
            results.append(0)

    positive = len([r for r in results if r > 0])
    total_pnl = sum(results)
    print(f"  {positive}/{n_folds} folds+, PnL {total_pnl:+.1f}%")
