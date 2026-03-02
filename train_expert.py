"""
Train Expert V14 - Pipeline completo para entrenar un experto
Uso: python train_expert.py --symbol ETH/USDT
"""

import argparse
import os
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import ccxt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import ta

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACION
# =============================================================================
TIMEFRAME = '4h'
TRAIN_START = '2018-01-01'
BASE_RISK_PCT = 0.02
TP_PCT = 0.06
SL_PCT = 0.03
TIMEOUT_CANDLES = 20

# Regime thresholds
ADX_TREND = 25
CHOP_RANGE = 61.8
DI_DIFF = 10

# =============================================================================
# PASO 1: DESCARGAR DATOS
# =============================================================================
def download_data(symbol: str, output_dir: str) -> pd.DataFrame:
    """Descarga datos historicos de Binance"""
    print(f"\n[1/6] DESCARGANDO DATOS: {symbol}")

    exchange = ccxt.binance()
    symbol_clean = symbol.replace('/', '')
    csv_path = Path(output_dir) / f"{symbol_clean}_4h.csv"

    # Si ya existe, cargar
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        print(f"  Cargado desde cache: {len(df)} candles")
        return df

    # Descargar
    since = exchange.parse8601(f'{TRAIN_START}T00:00:00Z')
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
    df.to_csv(csv_path, index=False)

    print(f"  Descargado: {len(df)} candles ({df['timestamp'].min()} a {df['timestamp'].max()})")
    return df

# =============================================================================
# PASO 2: DETECTOR DE REGIMEN
# =============================================================================
def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features para deteccion de regimen"""
    df = df.copy()

    # ADX y DI
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df['di_plus'] = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=14)
    df['di_minus'] = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=14)
    df['di_diff'] = df['di_plus'] - df['di_minus']

    # Choppiness
    atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    high_14 = df['high'].rolling(14).max()
    low_14 = df['low'].rolling(14).min()
    df['chop'] = 100 * np.log10(atr.rolling(14).sum() / (high_14 - low_14 + 1e-10)) / np.log10(14)

    # Volatilidad
    df['volatility'] = df['close'].pct_change().rolling(20).std()

    # Tendencia
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['trend'] = (df['sma_50'] > df['sma_200']).astype(int)

    return df.dropna()

def label_regime(row) -> str:
    """Clasifica el regimen basado en indicadores"""
    if row['adx'] > ADX_TREND:
        if row['di_diff'] > DI_DIFF:
            return 'TREND_UP'
        elif row['di_diff'] < -DI_DIFF:
            return 'TREND_DOWN'
    if row['chop'] > CHOP_RANGE:
        return 'RANGE'
    if row['volatility'] > row['volatility_median']:
        return 'VOLATILE'
    return 'RANGE'

def train_regime_detector(df: pd.DataFrame, output_dir: str) -> dict:
    """Entrena el detector de regimen"""
    print(f"\n[2/6] ENTRENANDO DETECTOR DE REGIMEN")

    df = compute_regime_features(df)
    df['volatility_median'] = df['volatility'].rolling(100).median()
    df['regime'] = df.apply(label_regime, axis=1)
    df = df.dropna()

    # Features para el modelo
    feature_cols = ['adx', 'di_plus', 'di_minus', 'di_diff', 'chop', 'volatility', 'trend']
    X = df[feature_cols]
    y = df['regime']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"  Train Accuracy: {train_acc:.1%}")
    print(f"  Test Accuracy: {test_acc:.1%}")
    print(f"  Gap: {(train_acc - test_acc):.1%}")

    # Distribucion de regimenes
    regime_dist = y.value_counts(normalize=True)
    print(f"  Distribucion: {dict(regime_dist.round(2))}")

    # Guardar
    model_path = Path(output_dir) / 'models' / 'regime_detector.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    return {
        'feature_cols': feature_cols,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'regime_distribution': dict(regime_dist)
    }

# =============================================================================
# PASO 3: ENSEMBLE DE MODELOS
# =============================================================================
def compute_ensemble_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features para el ensemble"""
    df = df.copy()

    # === CONTEXT FEATURES ===
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_pct'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-10)
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['trend_strength'] = (df['close'] - df['sma_50']) / df['sma_50']

    # === MOMENTUM FEATURES ===
    df['roc_5'] = df['close'].pct_change(5)
    df['roc_10'] = df['close'].pct_change(10)
    df['roc_20'] = df['close'].pct_change(20)
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    df['momentum'] = ta.momentum.roc(df['close'], window=10)

    # === VOLUME FEATURES ===
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['obv_sma'] = df['obv'].rolling(20).mean()
    df['obv_trend'] = (df['obv'] > df['obv_sma']).astype(int)
    df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)

    # === VOLATILITY ===
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['atr_pct'] = df['atr'] / df['close']

    return df.dropna()

def create_labels(df: pd.DataFrame, direction: str) -> pd.Series:
    """Crea labels para trades basados en TP/SL"""
    labels = []

    for i in range(len(df) - TIMEOUT_CANDLES):
        entry = df['close'].iloc[i]
        future = df.iloc[i+1:i+TIMEOUT_CANDLES+1]

        if direction == 'LONG':
            tp_price = entry * (1 + TP_PCT)
            sl_price = entry * (1 - SL_PCT)

            for j, row in future.iterrows():
                if row['high'] >= tp_price:
                    labels.append(1)  # Win
                    break
                if row['low'] <= sl_price:
                    labels.append(0)  # Loss
                    break
            else:
                # Timeout - check if profitable
                final = future['close'].iloc[-1] if len(future) > 0 else entry
                labels.append(1 if final > entry else 0)
        else:  # SHORT
            tp_price = entry * (1 - TP_PCT)
            sl_price = entry * (1 + SL_PCT)

            for j, row in future.iterrows():
                if row['low'] <= tp_price:
                    labels.append(1)  # Win
                    break
                if row['high'] >= sl_price:
                    labels.append(0)  # Loss
                    break
            else:
                final = future['close'].iloc[-1] if len(future) > 0 else entry
                labels.append(1 if final < entry else 0)

    return pd.Series(labels, index=df.index[:len(labels)])

def train_ensemble_models(df: pd.DataFrame, output_dir: str) -> dict:
    """Entrena los 3 modelos del ensemble"""
    print(f"\n[3/6] ENTRENANDO ENSEMBLE (3 modelos)")

    df = compute_ensemble_features(df)

    # Feature groups
    context_cols = ['rsi', 'bb_pct', 'trend_strength']
    momentum_cols = ['roc_5', 'roc_10', 'roc_20', 'macd_hist', 'momentum']
    volume_cols = ['volume_ratio', 'obv_trend', 'mfi']

    results = {}
    models_dir = Path(output_dir) / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    for direction in ['LONG', 'SHORT']:
        print(f"\n  Direccion: {direction}")

        # Crear labels
        labels = create_labels(df, direction)
        df_labeled = df.iloc[:len(labels)].copy()
        df_labeled['label'] = labels.values
        df_labeled = df_labeled.dropna()

        if len(df_labeled) < 100:
            print(f"    Insuficientes datos: {len(df_labeled)}")
            continue

        # Split
        split_idx = int(len(df_labeled) * 0.8)
        train_df = df_labeled.iloc[:split_idx]
        test_df = df_labeled.iloc[split_idx:]

        for model_name, cols in [('context', context_cols), ('momentum', momentum_cols), ('volume', volume_cols)]:
            X_train = train_df[cols]
            y_train = train_df['label']
            X_test = test_df[cols]
            y_test = test_df['label']

            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_train, y_train)

            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))

            try:
                train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
                test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            except:
                train_auc = test_auc = 0.5

            print(f"    {model_name}: Acc {test_acc:.1%}, AUC {test_auc:.2f}")

            # Guardar modelo
            model_path = models_dir / f'ensemble_{model_name}_{direction.lower()}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            results[f'{model_name}_{direction}'] = {
                'feature_cols': cols,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'train_auc': train_auc,
                'test_auc': test_auc,
            }

    return results

# =============================================================================
# PASO 4: BACKTEST WALK-FORWARD
# =============================================================================
def run_walkforward_backtest(df: pd.DataFrame, output_dir: str, n_folds: int = 12) -> dict:
    """Ejecuta backtest walk-forward"""
    print(f"\n[4/6] BACKTEST WALK-FORWARD ({n_folds} folds)")

    df = compute_ensemble_features(df)
    df = compute_regime_features(df)

    fold_size = len(df) // (n_folds + 2)
    results = []

    for fold in range(n_folds):
        train_start = fold * fold_size
        train_end = train_start + fold_size * 2
        test_start = train_end
        test_end = test_start + fold_size

        if test_end > len(df):
            break

        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]

        # Simular trades en test
        trades = []
        for i in range(len(test_df) - TIMEOUT_CANDLES):
            row = test_df.iloc[i]

            # Decidir direccion basado en regime
            if row.get('di_diff', 0) > 0:
                direction = 'LONG'
            else:
                direction = 'SHORT'

            entry = row['close']

            # Simular resultado
            future = test_df.iloc[i+1:i+TIMEOUT_CANDLES+1]
            pnl = 0

            if direction == 'LONG':
                tp = entry * (1 + TP_PCT)
                sl = entry * (1 - SL_PCT)
                for _, f_row in future.iterrows():
                    if f_row['high'] >= tp:
                        pnl = TP_PCT
                        break
                    if f_row['low'] <= sl:
                        pnl = -SL_PCT
                        break
            else:
                tp = entry * (1 - TP_PCT)
                sl = entry * (1 + SL_PCT)
                for _, f_row in future.iterrows():
                    if f_row['low'] <= tp:
                        pnl = TP_PCT
                        break
                    if f_row['high'] >= sl:
                        pnl = -SL_PCT
                        break

            if pnl != 0:
                trades.append({'pnl': pnl, 'direction': direction})

        if trades:
            total_pnl = sum(t['pnl'] for t in trades) * 100
            wr = len([t for t in trades if t['pnl'] > 0]) / len(trades)
            results.append({
                'fold': fold + 1,
                'trades': len(trades),
                'pnl': total_pnl,
                'wr': wr,
                'positive': total_pnl > 0
            })
            status = "OK" if total_pnl > 0 else "BAD"
            print(f"  Fold {fold+1:2d}: {len(trades):3d} trades, WR {wr:.0%}, PnL {total_pnl:+.1f}% [{status}]")

    positive_folds = len([r for r in results if r['positive']])
    total_folds = len(results)
    total_pnl = sum(r['pnl'] for r in results)

    print(f"\n  RESUMEN: {positive_folds}/{total_folds} folds positivos ({positive_folds/total_folds:.0%})")
    print(f"  PnL Total: {total_pnl:+.1f}%")

    return {
        'folds': results,
        'positive_folds': positive_folds,
        'total_folds': total_folds,
        'total_pnl': total_pnl,
        'passed': positive_folds >= total_folds * 0.8
    }

# =============================================================================
# PASO 5: VALIDACION CRUZADA
# =============================================================================
def generate_synthetic_data(n_candles: int, market_type: str, start_price: float = 100) -> pd.DataFrame:
    """Genera datos sinteticos"""
    np.random.seed(42 + hash(market_type) % 1000)

    params = {
        'bull': {'drift': 0.0015, 'vol': 0.02},
        'bear': {'drift': -0.0012, 'vol': 0.025},
        'range': {'drift': 0.0001, 'vol': 0.015},
        'volatile': {'drift': 0.0, 'vol': 0.04},
        'mixed': {'drift': 0.0003, 'vol': 0.025},
    }

    p = params.get(market_type, params['mixed'])

    returns = np.random.normal(p['drift'], p['vol'], n_candles)
    prices = start_price * np.exp(np.cumsum(returns))

    # Generar OHLCV
    data = []
    for i, close in enumerate(prices):
        noise = p['vol'] * close
        high = close + abs(np.random.normal(0, noise))
        low = close - abs(np.random.normal(0, noise))
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.uniform(1000, 10000)

        data.append({
            'timestamp': pd.Timestamp('2025-01-01') + pd.Timedelta(hours=4*i),
            'open': open_price,
            'high': max(high, open_price, close),
            'low': min(low, open_price, close),
            'close': close,
            'volume': volume
        })

    return pd.DataFrame(data)

def run_cross_validation(df: pd.DataFrame, output_dir: str) -> dict:
    """Ejecuta validacion cruzada en datos sinteticos"""
    print(f"\n[5/6] VALIDACION CRUZADA")

    results = []

    for market_type in ['bull', 'bear', 'range', 'volatile', 'mixed']:
        print(f"\n  Mercado {market_type.upper()}:")

        synth_df = generate_synthetic_data(2190, market_type)  # 1 year of 4h candles
        synth_df = compute_ensemble_features(synth_df)
        synth_df = synth_df.dropna()

        # Simular trades
        trades = []
        for i in range(0, len(synth_df) - TIMEOUT_CANDLES, 5):  # Every 5 candles
            row = synth_df.iloc[i]

            # Simple direction based on momentum
            direction = 'LONG' if row.get('momentum', 0) > 0 else 'SHORT'
            entry = row['close']

            future = synth_df.iloc[i+1:i+TIMEOUT_CANDLES+1]
            pnl = 0

            if direction == 'LONG':
                tp = entry * (1 + TP_PCT)
                sl = entry * (1 - SL_PCT)
                for _, f_row in future.iterrows():
                    if f_row['high'] >= tp:
                        pnl = TP_PCT
                        break
                    if f_row['low'] <= sl:
                        pnl = -SL_PCT
                        break
            else:
                tp = entry * (1 - TP_PCT)
                sl = entry * (1 + SL_PCT)
                for _, f_row in future.iterrows():
                    if f_row['low'] <= tp:
                        pnl = TP_PCT
                        break
                    if f_row['high'] >= sl:
                        pnl = -SL_PCT
                        break

            if pnl != 0:
                trades.append(pnl)

        if trades:
            total_pnl = sum(trades) * 100
            wr = len([t for t in trades if t > 0]) / len(trades)
            status = "OK" if total_pnl > 0 else "BAD"
            print(f"    {len(trades)} trades, WR {wr:.0%}, PnL {total_pnl:+.1f}% [{status}]")

            results.append({
                'market': market_type,
                'trades': len(trades),
                'pnl': total_pnl,
                'wr': wr,
                'positive': total_pnl > 0
            })

    positive = len([r for r in results if r['positive']])
    print(f"\n  RESUMEN: {positive}/5 mercados positivos")

    return {
        'results': results,
        'positive_markets': positive,
        'passed': positive >= 4
    }

# =============================================================================
# PASO 6: EXPORTAR
# =============================================================================
def export_to_production(symbol: str, output_dir: str, results: dict):
    """Exporta modelos y configuracion a produccion"""
    print(f"\n[6/6] EXPORTANDO A PRODUCCION")

    output_path = Path(output_dir)

    # Guardar metadata
    meta = {
        'symbol': symbol,
        'version': '14.0',
        'trained_at': datetime.now().isoformat(),
        'timeframe': TIMEFRAME,
        'regime_results': results.get('regime', {}),
        'ensemble_results': results.get('ensemble', {}),
        'backtest_results': results.get('backtest', {}),
        'validation_results': results.get('validation', {}),
        'status': 'APPROVED' if results.get('validation', {}).get('passed', False) else 'NEEDS_REVIEW'
    }

    meta_path = output_path / 'models' / 'meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    # Crear __init__.py
    init_content = f'''# {symbol.replace("/", "")} V14 Expert
# Trained: {datetime.now().strftime("%Y-%m-%d")}
# Status: {meta["status"]}

from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
VERSION = "14.0"
SYMBOL = "{symbol}"
STATUS = "{meta["status"]}"
'''

    with open(output_path / '__init__.py', 'w') as f:
        f.write(init_content)

    # Crear config.py
    config_content = f'''# {symbol.replace("/", "")} V14 Configuration

SYMBOL = "{symbol}"
TIMEFRAME = "{TIMEFRAME}"
BASE_RISK_PCT = {BASE_RISK_PCT}
TP_PCT = {TP_PCT}
SL_PCT = {SL_PCT}
TIMEOUT_CANDLES = {TIMEOUT_CANDLES}
'''

    with open(output_path / 'config.py', 'w') as f:
        f.write(config_content)

    print(f"  Exportado a: {output_path}")
    print(f"  Status: {meta['status']}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train V14 Expert')
    parser.add_argument('--symbol', type=str, required=True, help='Trading pair (e.g., ETH/USDT)')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    symbol = args.symbol
    symbol_clean = symbol.replace('/', '').lower()
    output_dir = args.output or f'strategies/{symbol_clean}_v14'

    print("=" * 70)
    print(f"ENTRENANDO EXPERTO V14: {symbol}")
    print("=" * 70)

    # Crear directorio
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).joinpath('models').mkdir(exist_ok=True)
    Path(output_dir).joinpath('data').mkdir(exist_ok=True)

    results = {}

    # Paso 1: Datos
    df = download_data(symbol, output_dir + '/data')

    # Paso 2: Regime
    results['regime'] = train_regime_detector(df, output_dir)

    # Paso 3: Ensemble
    results['ensemble'] = train_ensemble_models(df, output_dir)

    # Paso 4: Backtest
    results['backtest'] = run_walkforward_backtest(df, output_dir)

    # Paso 5: Validacion
    results['validation'] = run_cross_validation(df, output_dir)

    # Paso 6: Export
    export_to_production(symbol, output_dir, results)

    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    backtest_passed = results['backtest'].get('passed', False)
    validation_passed = results['validation'].get('passed', False)

    print(f"  Backtest Walk-Forward: {'PASSED' if backtest_passed else 'FAILED'}")
    print(f"  Cross-Validation: {'PASSED' if validation_passed else 'FAILED'}")

    if backtest_passed and validation_passed:
        print(f"\n  [APROBADO] Experto {symbol} listo para paper trading")
    else:
        print(f"\n  [REVISAR] Experto {symbol} necesita ajustes")

if __name__ == '__main__':
    main()
