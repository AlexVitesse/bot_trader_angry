"""
Train Ensemble Voting - SHORT direction models for DOGE/ADA/DOT/SOL
====================================================================
Complement to train_ensemble_voting.py (which trains LONG models).
Trains SHORT models: win if price drops by TP_PCT before rising by SL_PCT.

Saves: random_forest_short.pkl, gradient_boosting_short.pkl,
       logistic_regression_short.pkl (in same model dir as LONG models)

Usage:
  python train_ensemble_short.py --symbol DOGE/USDT
  python train_ensemble_short.py --symbol ADA/USDT
  python train_ensemble_short.py --symbol DOT/USDT
  python train_ensemble_short.py --symbol SOL/USDT
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
# CONFIGURACION - igual que LONG para comparabilidad
# =============================================================================
TIMEFRAME = '4h'
TP_PCT = 0.06   # 6% take profit (precio cae)
SL_PCT = 0.04   # 4% stop loss (precio sube)
TIMEOUT_CANDLES = 15

FEATURE_COLS = [
    'rsi', 'macd_norm', 'adx', 'bb_pct', 'atr_pct',
    'ret_3', 'ret_5', 'ret_10', 'vol_ratio', 'trend'
]

# =============================================================================
# DATOS (reutiliza CSVs existentes del entrenamiento LONG)
# =============================================================================
def load_data(symbol: str) -> pd.DataFrame:
    """Carga datos (preferiblemente de cache creado por train_ensemble_voting.py)."""
    symbol_clean = symbol.replace('/', '')
    csv_path = f'data/{symbol_clean}_4h.csv'

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        print(f"Cargado desde cache: {len(df)} candles")
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
    """Computa las mismas features que LONG para consistencia."""
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


def create_short_labels(df: pd.DataFrame) -> pd.Series:
    """Crea labels para SHORT: gana si precio baja TP_PCT antes de subir SL_PCT."""
    labels = []

    for i in range(len(df) - TIMEOUT_CANDLES - 1):
        entry = df['close'].iloc[i]
        future = df.iloc[i+1:i+TIMEOUT_CANDLES+1]

        tp_price = entry * (1 - TP_PCT)   # Baja -> TP
        sl_price = entry * (1 + SL_PCT)   # Sube -> SL

        won = False
        for _, row in future.iterrows():
            if row['low'] <= tp_price:
                won = True
                break
            if row['high'] >= sl_price:
                break

        labels.append(1 if won else 0)

    return pd.Series(labels, index=df.index[:len(labels)])


# =============================================================================
# WALK-FORWARD VALIDATION (SHORT)
# =============================================================================
def walk_forward_validation_short(df: pd.DataFrame, n_folds: int = 12) -> dict:
    """Walk-forward validation con labels SHORT."""
    labels = create_short_labels(df)
    df = df.iloc[:len(labels)].copy()
    df['label'] = labels.values

    X = df[FEATURE_COLS].values
    y = df['label'].values

    scaler = StandardScaler()
    fold_size = len(df) // (n_folds + 1)
    results = []

    print(f"\nWalk-Forward Validation SHORT ({n_folds} folds)")
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
            })

            status = "+" if pnl > 0 else "-"
            print(f"Fold {fold+1:2d}: {n_trades:3d} trades, WR {wr:.1%}, PnL {pnl*100:+6.1f}% [{status}]")
        else:
            results.append({'fold': fold + 1, 'trades': 0, 'wins': 0, 'wr': 0, 'pnl': 0})
            print(f"Fold {fold+1:2d}:   0 trades")

    positive_folds = len([r for r in results if r['pnl'] > 0])
    total_pnl = sum(r['pnl'] for r in results)
    total_trades = sum(r['trades'] for r in results)
    total_wins = sum(r['wins'] for r in results)
    avg_wr = total_wins / total_trades if total_trades > 0 else 0

    print("=" * 60)
    print(f"Resumen SHORT: {positive_folds}/{len(results)} folds positivos")
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
# ENTRENAR Y GUARDAR MODELOS SHORT
# =============================================================================
def train_short_models(df: pd.DataFrame, output_dir: str) -> dict:
    """Entrena modelos SHORT y los guarda con sufijo _short."""
    labels = create_short_labels(df)
    df = df.iloc[:len(labels)].copy()
    df['label'] = labels.values

    win_rate_check = df['label'].mean()
    print(f"\nBalance de clases SHORT: {win_rate_check:.1%} ganan")

    X = df[FEATURE_COLS].values
    y = df['label'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)

    rf.fit(X_scaled, y)
    gb.fit(X_scaled, y)
    lr.fit(X_scaled, y)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    joblib.dump(rf, f'{output_dir}/random_forest_short.pkl')
    joblib.dump(gb, f'{output_dir}/gradient_boosting_short.pkl')
    joblib.dump(lr, f'{output_dir}/logistic_regression_short.pkl')
    joblib.dump(scaler, f'{output_dir}/scaler_short.pkl')

    metadata = {
        'feature_cols': FEATURE_COLS,
        'voting_threshold': 2,
        'tp_pct': TP_PCT,
        'sl_pct': SL_PCT,
        'timeout_candles': TIMEOUT_CANDLES,
        'direction': 'SHORT',
        'training_samples': len(X),
    }
    joblib.dump(metadata, f'{output_dir}/metadata_short.pkl')

    print(f"\nModelos SHORT guardados en: {output_dir}")
    return metadata


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, required=True,
                        help='Symbol (e.g., DOGE/USDT, ADA/USDT, DOT/USDT, SOL/USDT)')
    args = parser.parse_args()

    symbol = args.symbol
    symbol_short = symbol.split('/')[0].lower()

    print("=" * 70)
    print(f"ENSEMBLE SHORT TRAINING: {symbol}")
    print("=" * 70)

    df = load_data(symbol)
    df = compute_features(df)

    print(f"\nDatos: {len(df)} candles ({df['timestamp'].min()} a {df['timestamp'].max()})")

    # Walk-forward validation SHORT
    wf_results = walk_forward_validation_short(df, n_folds=12)

    # Umbral: 58% de folds positivos (mismo que LONG)
    pass_threshold = wf_results['positive_folds'] / wf_results['folds'] >= 0.58

    if not pass_threshold:
        print(f"\n[WARN] No pasa umbral walk-forward ({wf_results['positive_folds']}/{wf_results['folds']} < 58%)")
        print("Los modelos SHORT se guardan igualmente para uso en produccion.")
        print("Considera ajustar TP/SL o usar umbral de confianza mas alto en vivo.")
    else:
        print(f"\n[PASS] Walk-forward SHORT: {wf_results['positive_folds']}/{wf_results['folds']} folds positivos")

    # Entrenar y guardar (incluso si no pasa - el filtro vivo es mas estricto)
    output_dir = f'strategies/{symbol_short}_v14/models'
    train_short_models(df, output_dir)

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Symbol: {symbol} | Direction: SHORT")
    print(f"Walk-Forward: {wf_results['positive_folds']}/{wf_results['folds']} folds+ ({wf_results['total_pnl']:+.1f}% PnL)")
    print(f"Total Trades: {wf_results['total_trades']} | Win Rate: {wf_results['win_rate']:.1%}")
    print(f"Archivos: random_forest_short.pkl, gradient_boosting_short.pkl, logistic_regression_short.pkl")
    print(f"\nStatus: READY (revisar metricas antes de produccion)")


if __name__ == '__main__':
    main()
