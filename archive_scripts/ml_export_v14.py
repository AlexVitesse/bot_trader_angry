"""
ML Export V14 - Entrenamiento unificado de todos los modelos V14
================================================================
Entrena y exporta los modelos ensemble para todos los pares V14:
- DOGE, ADA, DOT, SOL: Ensemble voting (RF + GB + LR)
- BTC: Solo setups (sin ML filter - mejor rendimiento validado)

Uso: python ml_export_v14.py [--force] [--pair SYMBOL]

Opciones:
  --force    Reentrenar aunque existan modelos
  --pair     Entrenar solo un par especifico (ej: --pair DOGE)
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import ccxt
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACION V14
# =============================================================================
from config.settings import BOT_VERSION

TIMEFRAME = '4h'
MIN_CANDLES = 2000  # Minimo de velas para entrenar

# Configuracion por modelo (TP/SL y si usa LR)
MODEL_CONFIGS = {
    'DOGE': {
        'symbol': 'DOGE/USDT',
        'tp_pct': 0.06,
        'sl_pct': 0.04,
        'timeout': 15,
        'use_lr': True,
        'min_folds_positive': 0.58,  # 58% de folds positivos
    },
    'ADA': {
        'symbol': 'ADA/USDT',
        'tp_pct': 0.06,
        'sl_pct': 0.04,
        'timeout': 15,
        'use_lr': True,
        'min_folds_positive': 0.58,
    },
    'DOT': {
        'symbol': 'DOT/USDT',
        'tp_pct': 0.05,
        'sl_pct': 0.03,
        'timeout': 15,
        'use_lr': False,  # DOT usa solo RF+GB (2 modelos)
        'min_folds_positive': 0.58,
    },
    'SOL': {
        'symbol': 'SOL/USDT',
        'tp_pct': 0.06,
        'sl_pct': 0.04,
        'timeout': 15,
        'use_lr': True,
        'min_folds_positive': 0.58,
    },
}

FEATURE_COLS = [
    'rsi', 'macd_norm', 'adx', 'bb_pct', 'atr_pct',
    'ret_3', 'ret_5', 'ret_10', 'vol_ratio', 'trend'
]


# =============================================================================
# FUNCIONES DE DATOS
# =============================================================================
def download_data(symbol: str, exchange: ccxt.Exchange) -> pd.DataFrame:
    """Descarga datos historicos desde Binance."""
    print(f"    Descargando {symbol}...")
    since = exchange.parse8601('2020-01-01T00:00:00Z')
    all_ohlcv = []

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break
        except Exception as e:
            print(f"    Error descargando: {e}")
            break

    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def load_or_download_data(symbol: str, exchange: ccxt.Exchange) -> pd.DataFrame:
    """Carga datos del CSV o los descarga si no existen."""
    symbol_clean = symbol.replace('/', '')
    csv_path = Path(f'data/{symbol_clean}_4h.csv')

    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        # Verificar si necesita actualizacion (mas de 1 dia de antiguedad)
        last_date = pd.to_datetime(df['timestamp'].iloc[-1])
        if (datetime.now() - last_date).days > 1:
            print(f"    Datos antiguos, actualizando...")
            df_new = download_data(symbol, exchange)
            if not df_new.empty:
                df = df_new
                csv_path.parent.mkdir(exist_ok=True)
                df.to_csv(csv_path, index=False)
        print(f"    Cargado: {len(df)} candles")
        return df

    # Descargar si no existe
    df = download_data(symbol, exchange)
    if not df.empty:
        csv_path.parent.mkdir(exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"    Guardado: {len(df)} candles")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features para el modelo ensemble."""
    df = df.copy()

    # RSI normalizado
    df['rsi'] = ta.momentum.rsi(df['close'], window=14) / 100

    # MACD normalizado por precio
    macd = ta.trend.MACD(df['close'])
    df['macd_norm'] = macd.macd() / df['close']

    # ADX normalizado
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14) / 100

    # Bollinger %
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    bb_range = bb.bollinger_hband() - bb.bollinger_lband()
    df['bb_pct'] = (df['close'] - bb.bollinger_lband()) / (bb_range + 1e-10)

    # ATR %
    df['atr_pct'] = ta.volatility.average_true_range(
        df['high'], df['low'], df['close'], window=14
    ) / df['close']

    # Retornos
    df['ret_3'] = df['close'].pct_change(3)
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_10'] = df['close'].pct_change(10)

    # Volume ratio
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    # Trend (sobre SMA50)
    df['trend'] = (df['close'] > df['close'].rolling(50).mean()).astype(float)

    return df.dropna()


def create_labels(df: pd.DataFrame, tp_pct: float, sl_pct: float, timeout: int) -> pd.Series:
    """Crea labels win/loss para trades LONG."""
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


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================
def walk_forward_validation(df: pd.DataFrame, config: dict, n_folds: int = 12) -> dict:
    """Walk-forward validation con ensemble voting."""
    labels = create_labels(df, config['tp_pct'], config['sl_pct'], config['timeout'])
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

        # Entrenar modelos
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        rf.fit(X_train_scaled, y_train)
        gb.fit(X_train_scaled, y_train)

        prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
        prob_gb = gb.predict_proba(X_test_scaled)[:, 1]

        if config['use_lr']:
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_train_scaled, y_train)
            prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
            votes = ((prob_rf > 0.5).astype(int) +
                     (prob_gb > 0.5).astype(int) +
                     (prob_lr > 0.5).astype(int))
            trade_mask = votes >= 2
        else:
            votes = ((prob_rf > 0.5).astype(int) +
                     (prob_gb > 0.5).astype(int))
            trade_mask = votes >= 2

        n_trades = trade_mask.sum()

        if n_trades > 0:
            wins = y_test[trade_mask].sum()
            wr = wins / n_trades
            pnl = wins * config['tp_pct'] - (n_trades - wins) * config['sl_pct']
            results.append({'fold': fold + 1, 'trades': n_trades, 'wins': wins, 'wr': wr, 'pnl': pnl * 100})
        else:
            results.append({'fold': fold + 1, 'trades': 0, 'wins': 0, 'wr': 0, 'pnl': 0})

    positive_folds = len([r for r in results if r['pnl'] > 0])
    total_trades = sum(r['trades'] for r in results)
    total_wins = sum(r['wins'] for r in results)
    avg_wr = total_wins / total_trades if total_trades > 0 else 0
    total_pnl = sum(r['pnl'] for r in results)

    return {
        'folds': len(results),
        'positive_folds': positive_folds,
        'total_trades': total_trades,
        'win_rate': avg_wr,
        'total_pnl': total_pnl,
    }


# =============================================================================
# TRAIN & SAVE
# =============================================================================
def train_and_save_model(df: pd.DataFrame, config: dict, output_dir: Path) -> dict:
    """Entrena modelo final y lo guarda."""
    labels = create_labels(df, config['tp_pct'], config['sl_pct'], config['timeout'])
    df = df.iloc[:len(labels)].copy()
    df['label'] = labels.values

    X = df[FEATURE_COLS].values
    y = df['label'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entrenar modelos
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    rf.fit(X_scaled, y)
    gb.fit(X_scaled, y)

    # Guardar
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, output_dir / 'random_forest.pkl')
    joblib.dump(gb, output_dir / 'gradient_boosting.pkl')
    joblib.dump(scaler, output_dir / 'scaler.pkl')

    if config['use_lr']:
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_scaled, y)
        joblib.dump(lr, output_dir / 'logistic_regression.pkl')

    # Metadata
    metadata = {
        'feature_cols': FEATURE_COLS,
        'voting_threshold': 2,
        'tp_pct': config['tp_pct'],
        'sl_pct': config['sl_pct'],
        'timeout_candles': config['timeout'],
        'training_samples': len(X),
        'use_lr': config['use_lr'],
        'trained_at': datetime.now().isoformat(),
        'version': BOT_VERSION,
    }
    joblib.dump(metadata, output_dir / 'metadata.pkl')

    return metadata


# =============================================================================
# MAIN
# =============================================================================
def train_single_model(name: str, config: dict, exchange: ccxt.Exchange, force: bool = False) -> bool:
    """Entrena un modelo individual."""
    symbol = config['symbol']
    output_dir = Path(f'strategies/{name.lower()}_v14/models')

    # Verificar si ya existe
    if not force and (output_dir / 'random_forest.pkl').exists():
        print(f"    Ya existe (use --force para reentrenar)")
        return True

    # Cargar datos
    df = load_or_download_data(symbol, exchange)
    if df.empty or len(df) < MIN_CANDLES:
        print(f"    ERROR: Datos insuficientes ({len(df)} < {MIN_CANDLES})")
        return False

    # Compute features
    df = compute_features(df)
    print(f"    Features: {len(df)} candles")

    # Walk-forward validation
    wf = walk_forward_validation(df, config)
    pass_rate = wf['positive_folds'] / wf['folds']

    print(f"    Walk-Forward: {wf['positive_folds']}/{wf['folds']} folds+ "
          f"({pass_rate:.0%}), WR {wf['win_rate']:.1%}, PnL {wf['total_pnl']:+.1f}%")

    if pass_rate < config['min_folds_positive']:
        print(f"    WARN: No pasa umbral ({pass_rate:.0%} < {config['min_folds_positive']:.0%})")
        # Continuar de todos modos pero advertir

    # Train and save
    train_and_save_model(df, config, output_dir)
    print(f"    Guardado: {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description=f'ML Export {BOT_VERSION}')
    parser.add_argument('--force', action='store_true', help='Reentrenar aunque existan modelos')
    parser.add_argument('--pair', type=str, help='Entrenar solo un par (ej: DOGE)')
    args = parser.parse_args()

    print("=" * 70)
    print(f"ML EXPORT {BOT_VERSION} - Entrenamiento de Modelos Ensemble")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Modelos: {', '.join(MODEL_CONFIGS.keys())}")
    print()

    # Conectar exchange
    exchange = ccxt.binance({'enableRateLimit': True})

    # Filtrar por par si se especifica
    if args.pair:
        pair_upper = args.pair.upper()
        if pair_upper not in MODEL_CONFIGS:
            print(f"ERROR: Par {pair_upper} no encontrado. Disponibles: {list(MODEL_CONFIGS.keys())}")
            sys.exit(1)
        models_to_train = {pair_upper: MODEL_CONFIGS[pair_upper]}
    else:
        models_to_train = MODEL_CONFIGS

    # Entrenar cada modelo
    results = {}
    total = len(models_to_train)

    for i, (name, config) in enumerate(models_to_train.items(), 1):
        print(f"[{i}/{total}] {name} ({config['symbol']})...")
        success = train_single_model(name, config, exchange, args.force)
        results[name] = 'OK' if success else 'FAIL'
        print()

    # Resumen
    print("=" * 70)
    print("RESUMEN")
    print("=" * 70)
    for name, status in results.items():
        emoji = "✓" if status == 'OK' else "✗"
        print(f"  {emoji} {name}: {status}")

    ok_count = sum(1 for s in results.values() if s == 'OK')
    print()
    print(f"Completado: {ok_count}/{len(results)} modelos")
    print(f"Modelos guardados en: strategies/*/models/")

    # Exit code
    sys.exit(0 if ok_count == len(results) else 1)


if __name__ == '__main__':
    main()
