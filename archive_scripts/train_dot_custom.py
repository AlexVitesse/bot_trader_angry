"""
Train DOT V14 with custom parameters
Testing different TP/SL combinations
"""

import warnings
import numpy as np
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

TIMEFRAME = '4h'

FEATURE_COLS = [
    'rsi', 'macd_norm', 'adx', 'bb_pct', 'atr_pct',
    'ret_3', 'ret_5', 'ret_10', 'vol_ratio', 'trend'
]

def load_data():
    df = pd.read_csv('data/DOTUSDT_4h.csv', parse_dates=['timestamp'])
    return df

def compute_features(df):
    df = df.copy()
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
    return df.dropna()

def create_labels(df, tp_pct, sl_pct, timeout):
    labels = []
    for i in range(len(df) - timeout - 1):
        entry = df['close'].iloc[i]
        future = df.iloc[i+1:i+timeout+1]
        tp_price = entry * (1 + tp_pct)
        sl_price = entry * (1 - sl_pct)
        won = False
        for _, row in future.iterrows():
            if row['high'] >= tp_price:
                won = True
                break
            if row['low'] <= sl_price:
                break
        labels.append(1 if won else 0)
    return pd.Series(labels, index=df.index[:len(labels)])

def walk_forward_test(df, tp_pct, sl_pct, timeout, n_folds=10):
    labels = create_labels(df, tp_pct, sl_pct, timeout)
    df = df.iloc[:len(labels)].copy()
    df['label'] = labels.values

    X = df[FEATURE_COLS].values
    y = df['label'].values

    scaler = StandardScaler()
    fold_size = len(df) // (n_folds + 1)

    results = []
    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = test_start + fold_size

        if test_end > len(df):
            break

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        lr = LogisticRegression(max_iter=1000, random_state=42)

        rf.fit(X_train_scaled, y_train)
        gb.fit(X_train_scaled, y_train)
        lr.fit(X_train_scaled, y_train)

        prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
        prob_gb = gb.predict_proba(X_test_scaled)[:, 1]
        prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

        votes = ((prob_rf > 0.5).astype(int) + (prob_gb > 0.5).astype(int) + (prob_lr > 0.5).astype(int))

        trade_mask = votes >= 2
        n_trades = trade_mask.sum()

        if n_trades > 0:
            wins = y_test[trade_mask].sum()
            wr = wins / n_trades
            pnl = wins * tp_pct - (n_trades - wins) * sl_pct
            results.append({'pnl': pnl * 100, 'trades': n_trades, 'wr': wr})
        else:
            results.append({'pnl': 0, 'trades': 0, 'wr': 0})

    positive_folds = len([r for r in results if r['pnl'] > 0])
    total_pnl = sum(r['pnl'] for r in results)
    total_trades = sum(r['trades'] for r in results)

    return {
        'positive_folds': positive_folds,
        'total_folds': len(results),
        'total_pnl': total_pnl,
        'total_trades': total_trades
    }

# Test different parameter combinations
print("=" * 70)
print("DOT V14 - Testing Different Parameters")
print("=" * 70)

df = load_data()
df = compute_features(df)
print(f"Data: {len(df)} candles\n")

# Parameter grid
tp_options = [0.04, 0.05, 0.06, 0.08]
sl_options = [0.02, 0.03, 0.04]
timeout_options = [10, 15, 20]

best_result = None
best_params = None

for tp in tp_options:
    for sl in sl_options:
        for timeout in timeout_options:
            result = walk_forward_test(df, tp, sl, timeout)

            if result['total_trades'] > 50:  # Minimum trades
                pct_positive = result['positive_folds'] / result['total_folds']

                if pct_positive >= 0.6:  # At least 60% positive
                    print(f"TP {tp*100:.0f}% / SL {sl*100:.0f}% / TO {timeout}: "
                          f"{result['positive_folds']}/{result['total_folds']} folds+, "
                          f"PnL {result['total_pnl']:+.1f}%, "
                          f"Trades {result['total_trades']}")

                    if best_result is None or result['total_pnl'] > best_result['total_pnl']:
                        best_result = result
                        best_params = {'tp': tp, 'sl': sl, 'timeout': timeout}

print("\n" + "=" * 70)
if best_result:
    print(f"MEJOR CONFIGURACION:")
    print(f"  TP: {best_params['tp']*100:.0f}%")
    print(f"  SL: {best_params['sl']*100:.0f}%")
    print(f"  Timeout: {best_params['timeout']} candles")
    print(f"  Folds: {best_result['positive_folds']}/{best_result['total_folds']} positivos")
    print(f"  PnL: {best_result['total_pnl']:+.1f}%")
    print(f"  Trades: {best_result['total_trades']}")
else:
    print("No se encontro configuracion valida para DOT")
    print("Considerar:")
    print("  - DOT tiene menos historia que otros assets")
    print("  - El modelo puede no ser adecuado para DOT")
    print("  - Probar enfoque diferente (trend following puro)")
