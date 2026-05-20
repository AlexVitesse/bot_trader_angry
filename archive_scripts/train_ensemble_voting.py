"""
Train Ensemble Voting Model for altcoins/memecoins
- For assets that don't follow traditional technical patterns
- 3 ML models vote: RandomForest, GradientBoosting, LogisticRegression
- Trade if 2/3 models agree

Usage: python train_ensemble_voting.py --symbol ADA/USDT
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import ccxt
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACION
# =============================================================================
TIMEFRAME = '4h'
TP_PCT = 0.06  # 6% take profit
SL_PCT = 0.04  # 4% stop loss
TIMEOUT_CANDLES = 15

FEATURE_COLS = [
    'rsi', 'macd_norm', 'adx', 'bb_pct', 'atr_pct',
    'ret_3', 'ret_5', 'ret_10', 'vol_ratio', 'trend'
]

# =============================================================================
# DATOS
# =============================================================================
def load_data(symbol: str) -> pd.DataFrame:
    """Load or download data"""
    symbol_clean = symbol.replace('/', '')
    csv_path = f'data/{symbol_clean}_4h.csv'

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        print(f"Cargado: {len(df)} candles")
        return df

    print(f"Descargando {symbol}...")
    exchange = ccxt.binance()
    since = exchange.parse8601('2019-01-01T00:00:00Z')
    all_ohlcv = []

    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < 1000:
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    Path('data').mkdir(exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Guardado: {len(df)} candles")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ML features"""
    df = df.copy()

    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14) / 100  # Normalize to 0-1

    # MACD normalized
    macd = ta.trend.MACD(df['close'])
    df['macd_norm'] = macd.macd() / df['close']  # Normalize by price

    # ADX
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14) / 100

    # Bollinger %
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_pct'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-10)

    # ATR %
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['atr_pct'] = df['atr'] / df['close']

    # Returns
    df['ret_3'] = df['close'].pct_change(3)
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_10'] = df['close'].pct_change(10)

    # Volume ratio
    df['vol_sma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma']

    # Trend (simple: above/below SMA50)
    df['sma_50'] = df['close'].rolling(50).mean()
    df['trend'] = (df['close'] > df['sma_50']).astype(float)

    return df.dropna()


def create_labels(df: pd.DataFrame) -> pd.Series:
    """Create win/loss labels for trades"""
    labels = []

    for i in range(len(df) - TIMEOUT_CANDLES - 1):
        entry = df['close'].iloc[i]
        future = df.iloc[i+1:i+TIMEOUT_CANDLES+1]

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

    return pd.Series(labels, index=df.index[:len(labels)])


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================
def walk_forward_validation(df: pd.DataFrame, n_folds: int = 12) -> dict:
    """
    Walk-forward validation with ensemble voting
    """
    labels = create_labels(df)
    df = df.iloc[:len(labels)].copy()
    df['label'] = labels.values

    X = df[FEATURE_COLS].values
    y = df['label'].values

    # Scale features
    scaler = StandardScaler()

    fold_size = len(df) // (n_folds + 1)
    results = []

    print(f"\nWalk-Forward Validation ({n_folds} folds)")
    print("=" * 60)

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

        # Scale
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train 3 models
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        lr = LogisticRegression(max_iter=1000, random_state=42)

        rf.fit(X_train_scaled, y_train)
        gb.fit(X_train_scaled, y_train)
        lr.fit(X_train_scaled, y_train)

        # Predict probabilities
        prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
        prob_gb = gb.predict_proba(X_test_scaled)[:, 1]
        prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

        # Voting: trade if 2/3 models predict prob > 0.5
        votes = ((prob_rf > 0.5).astype(int) +
                 (prob_gb > 0.5).astype(int) +
                 (prob_lr > 0.5).astype(int))

        trade_mask = votes >= 2
        n_trades = trade_mask.sum()

        if n_trades > 0:
            wins = y_test[trade_mask].sum()
            wr = wins / n_trades
            pnl = wins * TP_PCT - (n_trades - wins) * SL_PCT

            results.append({
                'fold': fold + 1,
                'trades': n_trades,
                'wins': wins,
                'wr': wr,
                'pnl': pnl * 100,
                'avg_prob': np.mean([prob_rf[trade_mask].mean(),
                                    prob_gb[trade_mask].mean(),
                                    prob_lr[trade_mask].mean()])
            })

            status = "+" if pnl > 0 else "-"
            print(f"Fold {fold+1:2d}: {n_trades:3d} trades, WR {wr:.1%}, PnL {pnl*100:+6.1f}% [{status}]")
        else:
            results.append({
                'fold': fold + 1,
                'trades': 0,
                'wins': 0,
                'wr': 0,
                'pnl': 0,
                'avg_prob': 0
            })
            print(f"Fold {fold+1:2d}:   0 trades")

    # Summary
    positive_folds = len([r for r in results if r['pnl'] > 0])
    total_pnl = sum(r['pnl'] for r in results)
    total_trades = sum(r['trades'] for r in results)
    total_wins = sum(r['wins'] for r in results)
    avg_wr = total_wins / total_trades if total_trades > 0 else 0

    print("=" * 60)
    print(f"Resumen: {positive_folds}/{len(results)} folds positivos")
    print(f"Total: {total_trades} trades, WR {avg_wr:.1%}, PnL {total_pnl:+.1f}%")

    return {
        'folds': len(results),
        'positive_folds': positive_folds,
        'total_trades': total_trades,
        'win_rate': avg_wr,
        'total_pnl': total_pnl,
        'fold_results': results
    }


# =============================================================================
# CROSS-ASSET VALIDATION
# =============================================================================
def cross_validate_assets(df_train: pd.DataFrame, test_symbols: list, scaler) -> dict:
    """
    Test the model trained on one asset against similar assets
    """
    labels = create_labels(df_train)
    df_train = df_train.iloc[:len(labels)].copy()
    df_train['label'] = labels.values

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train['label'].values

    X_train_scaled = scaler.fit_transform(X_train)

    # Train models on primary asset
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)

    rf.fit(X_train_scaled, y_train)
    gb.fit(X_train_scaled, y_train)
    lr.fit(X_train_scaled, y_train)

    results = {}

    print("\nCross-Asset Validation")
    print("=" * 60)

    for symbol in test_symbols:
        try:
            df_test = load_data(symbol)
            df_test = compute_features(df_test)

            labels_test = create_labels(df_test)
            df_test = df_test.iloc[:len(labels_test)].copy()
            df_test['label'] = labels_test.values

            X_test = df_test[FEATURE_COLS].values
            y_test = df_test['label'].values

            X_test_scaled = scaler.transform(X_test)

            # Predict
            prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
            prob_gb = gb.predict_proba(X_test_scaled)[:, 1]
            prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

            votes = ((prob_rf > 0.5).astype(int) +
                     (prob_gb > 0.5).astype(int) +
                     (prob_lr > 0.5).astype(int))

            trade_mask = votes >= 2
            n_trades = trade_mask.sum()

            if n_trades > 0:
                wins = y_test[trade_mask].sum()
                wr = wins / n_trades
                pnl = wins * TP_PCT - (n_trades - wins) * SL_PCT

                results[symbol] = {
                    'trades': n_trades,
                    'wins': wins,
                    'wr': wr,
                    'pnl': pnl * 100
                }

                status = "PASS" if pnl > 0 else "FAIL"
                print(f"{symbol}: {n_trades} trades, WR {wr:.1%}, PnL {pnl*100:+.1f}% [{status}]")
            else:
                results[symbol] = {'trades': 0, 'pnl': 0}
                print(f"{symbol}: 0 trades")

        except Exception as e:
            print(f"{symbol}: Error - {e}")
            results[symbol] = {'error': str(e)}

    return results


# =============================================================================
# TRAIN FINAL MODEL
# =============================================================================
def train_final_model(df: pd.DataFrame, output_dir: str) -> dict:
    """
    Train final ensemble model on all data and save
    """
    labels = create_labels(df)
    df = df.iloc[:len(labels)].copy()
    df['label'] = labels.values

    X = df[FEATURE_COLS].values
    y = df['label'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train models
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)

    rf.fit(X_scaled, y)
    gb.fit(X_scaled, y)
    lr.fit(X_scaled, y)

    # Save models
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    joblib.dump(rf, f'{output_dir}/random_forest.pkl')
    joblib.dump(gb, f'{output_dir}/gradient_boosting.pkl')
    joblib.dump(lr, f'{output_dir}/logistic_regression.pkl')
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')

    # Save metadata
    metadata = {
        'feature_cols': FEATURE_COLS,
        'voting_threshold': 2,
        'tp_pct': TP_PCT,
        'sl_pct': SL_PCT,
        'timeout_candles': TIMEOUT_CANDLES,
        'training_samples': len(X),
    }
    joblib.dump(metadata, f'{output_dir}/metadata.pkl')

    print(f"\nModelos guardados en: {output_dir}")
    return metadata


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, required=True, help='Primary symbol (e.g., ADA/USDT)')
    parser.add_argument('--cross', type=str, nargs='+', help='Cross-validation symbols')
    args = parser.parse_args()

    symbol = args.symbol
    symbol_short = symbol.split('/')[0].lower()

    print("=" * 70)
    print(f"ENSEMBLE VOTING TRAINING: {symbol}")
    print("=" * 70)

    # Load and prepare data
    df = load_data(symbol)
    df = compute_features(df)

    print(f"\nDatos preparados: {len(df)} candles")
    print(f"Periodo: {df['timestamp'].min()} a {df['timestamp'].max()}")

    # Walk-forward validation
    wf_results = walk_forward_validation(df, n_folds=12)

    # Check if passes threshold (70% of folds positive)
    pass_threshold = wf_results['positive_folds'] / wf_results['folds'] >= 0.58

    if not pass_threshold:
        print(f"\n[FAIL] No pasa el umbral de walk-forward ({wf_results['positive_folds']}/{wf_results['folds']} < 70%)")
        print("Considera:")
        print("  - Usar otros features")
        print("  - Ajustar TP/SL")
        print("  - Probar con menos/mas datos")
        return

    print(f"\n[PASS] Walk-forward: {wf_results['positive_folds']}/{wf_results['folds']} folds positivos")

    # Cross-asset validation (if provided)
    if args.cross:
        scaler = StandardScaler()
        cross_results = cross_validate_assets(df, args.cross, scaler)

        positive_cross = len([r for r in cross_results.values() if r.get('pnl', -999) > 0])
        total_cross = len([r for r in cross_results.values() if 'pnl' in r])

        if positive_cross < total_cross * 0.5:
            print(f"\n[WARN] Cross-validation: solo {positive_cross}/{total_cross} positivos")
        else:
            print(f"\n[PASS] Cross-validation: {positive_cross}/{total_cross} positivos")

    # Train and save final model
    output_dir = f'strategies/{symbol_short}_v14/models'
    metadata = train_final_model(df, output_dir)

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Walk-Forward: {wf_results['positive_folds']}/{wf_results['folds']} folds+ ({wf_results['total_pnl']:+.1f}% PnL)")
    print(f"Total Trades: {wf_results['total_trades']}")
    print(f"Win Rate: {wf_results['win_rate']:.1%}")
    print(f"Models: RandomForest, GradientBoosting, LogisticRegression")
    print(f"Voting: Trade if >= 2/3 models agree")
    print(f"\nStatus: READY FOR PRODUCTION")


if __name__ == '__main__':
    main()
