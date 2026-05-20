"""
VALIDACION DE MODELOS CON DATOS SINTETICOS
=============================================
Objetivo: Probar cada modelo con datos que NUNCA ha visto.

Metodologia:
1. Generar datos sinteticos (bull, bear, range, mixed, volatile)
2. Aplicar cada modelo ensemble a los datos sinteticos
3. Colectar TODOS los trades (buenos y malos)
4. Analizar features de trades perdedores vs ganadores
5. Identificar patrones de fallo

Modelos a probar:
- DOGE ensemble
- ADA ensemble
- DOT ensemble
- SOL ensemble
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from datetime import datetime
import ta

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACION
# =============================================================================

MODELS = {
    'DOGE': {
        'path': 'strategies/doge_v14/models',
        'tp': 0.06,
        'sl': 0.04,
    },
    'ADA': {
        'path': 'strategies/ada_v14/models',
        'tp': 0.06,
        'sl': 0.04,
    },
    'DOT': {
        'path': 'strategies/dot_v14/models',
        'tp': 0.05,
        'sl': 0.03,
    },
    'SOL': {
        'path': 'strategies/sol_v14/models',
        'tp': 0.06,
        'sl': 0.04,
    },
}

FEATURE_COLS = ['rsi', 'macd_norm', 'adx', 'bb_pct', 'atr_pct',
                'ret_3', 'ret_5', 'ret_10', 'vol_ratio', 'trend']

TIMEOUT_CANDLES = 15

# =============================================================================
# GENERADOR DE DATOS SINTETICOS (de validation.py)
# =============================================================================

def generate_synthetic_ohlcv(n_candles, market_type='mixed', start_price=100, seed=None):
    """
    Genera datos OHLCV sinteticos con Geometric Brownian Motion.
    """
    if seed:
        np.random.seed(seed)

    params = {
        'bull': {'drift': 0.0015, 'vol': 0.02},
        'bear': {'drift': -0.0012, 'vol': 0.025},
        'range': {'drift': 0.0001, 'vol': 0.015},
        'volatile': {'drift': 0.0, 'vol': 0.04},
    }

    dates = pd.date_range(start='2025-01-01', periods=n_candles, freq='4h')
    closes = [start_price]

    if market_type == 'mixed':
        segment_size = n_candles // 4
        segments = ['bull', 'bear', 'range', 'bull']

        for i in range(1, n_candles):
            segment_idx = min(i // segment_size, len(segments) - 1)
            current_type = segments[segment_idx]
            p = params[current_type]
            returns = np.random.normal(p['drift'], p['vol'])
            new_price = closes[-1] * (1 + returns)
            closes.append(max(new_price, 0.01))
    else:
        p = params.get(market_type, params['range'])
        for i in range(1, n_candles):
            returns = np.random.normal(p['drift'], p['vol'])
            new_price = closes[-1] * (1 + returns)
            closes.append(max(new_price, 0.01))

    closes = np.array(closes)
    highs = closes * (1 + np.abs(np.random.normal(0.005, 0.003, n_candles)))
    lows = closes * (1 - np.abs(np.random.normal(0.005, 0.003, n_candles)))
    opens = np.roll(closes, 1)
    opens[0] = start_price

    base_volume = 1000000
    vol_noise = np.random.lognormal(0, 0.5, n_candles)
    price_changes = np.abs(np.diff(closes, prepend=closes[0])) / closes
    volume = base_volume * vol_noise * (1 + price_changes * 10)

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volume
    }, index=dates)

    return df


# =============================================================================
# FEATURES
# =============================================================================

def compute_features(df):
    """Compute ML features para ensemble"""
    feat = pd.DataFrame(index=df.index)

    feat['rsi'] = ta.momentum.rsi(df['close'], window=14) / 100
    macd = ta.trend.MACD(df['close'])
    feat['macd_norm'] = macd.macd() / df['close']
    feat['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14) / 100

    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    feat['bb_pct'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-10)
    feat['atr_pct'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14) / df['close']

    feat['ret_3'] = df['close'].pct_change(3)
    feat['ret_5'] = df['close'].pct_change(5)
    feat['ret_10'] = df['close'].pct_change(10)

    feat['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    feat['trend'] = (df['close'] > df['close'].rolling(50).mean()).astype(float)

    # Features adicionales para analisis
    feat['rsi_raw'] = ta.momentum.rsi(df['close'], window=14)
    feat['adx_raw'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    feat['ret_1'] = df['close'].pct_change(1)
    feat['ret_20'] = df['close'].pct_change(20)
    feat['high_20_dist'] = (df['close'] - df['high'].rolling(20).max()) / df['close']
    feat['low_20_dist'] = (df['close'] - df['low'].rolling(20).min()) / df['close']

    return feat.dropna()


# =============================================================================
# MODELO ENSEMBLE
# =============================================================================

def load_models(model_path):
    """Carga modelos ensemble"""
    try:
        models = {
            'scaler': joblib.load(f'{model_path}/scaler.pkl'),
            'rf': joblib.load(f'{model_path}/random_forest.pkl'),
            'gb': joblib.load(f'{model_path}/gradient_boosting.pkl')
        }
        lr_path = Path(model_path) / 'logistic_regression.pkl'
        if lr_path.exists():
            models['lr'] = joblib.load(lr_path)
        return models
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None


def predict_ensemble(models, X):
    """Prediccion ensemble con probabilidades"""
    X_scaled = models['scaler'].transform(X)

    prob_rf = models['rf'].predict_proba(X_scaled)[:, 1]
    prob_gb = models['gb'].predict_proba(X_scaled)[:, 1]

    if 'lr' in models:
        prob_lr = models['lr'].predict_proba(X_scaled)[:, 1]
        votes = (prob_rf > 0.5).astype(int) + (prob_gb > 0.5).astype(int) + (prob_lr > 0.5).astype(int)
        avg_prob = (prob_rf + prob_gb + prob_lr) / 3
    else:
        votes = (prob_rf > 0.5).astype(int) + (prob_gb > 0.5).astype(int)
        avg_prob = (prob_rf + prob_gb) / 2

    signals = votes >= 2

    return signals, avg_prob, prob_rf, prob_gb


# =============================================================================
# SIMULACION DE TRADES
# =============================================================================

def simulate_trade(df, idx, tp, sl):
    """Simula un trade y retorna resultado con detalles"""
    if idx >= len(df) - TIMEOUT_CANDLES - 1:
        return None

    entry_price = df['close'].iloc[idx]
    entry_time = df.index[idx]

    for i in range(1, min(TIMEOUT_CANDLES + 1, len(df) - idx)):
        future_price = df['close'].iloc[idx + i]
        pnl = (future_price - entry_price) / entry_price

        if pnl >= tp:
            return {
                'outcome': 1,  # Win
                'pnl': pnl,
                'bars': i,
                'exit_reason': 'TP',
                'entry_price': entry_price,
                'exit_price': future_price,
                'entry_time': entry_time
            }
        elif pnl <= -sl:
            return {
                'outcome': 0,  # Loss
                'pnl': -sl,
                'bars': i,
                'exit_reason': 'SL',
                'entry_price': entry_price,
                'exit_price': future_price,
                'entry_time': entry_time
            }

    # Timeout
    final_price = df['close'].iloc[min(idx + TIMEOUT_CANDLES, len(df) - 1)]
    final_pnl = (final_price - entry_price) / entry_price
    return {
        'outcome': 1 if final_pnl > 0 else 0,
        'pnl': final_pnl,
        'bars': TIMEOUT_CANDLES,
        'exit_reason': 'TIMEOUT',
        'entry_price': entry_price,
        'exit_price': final_price,
        'entry_time': entry_time
    }


# =============================================================================
# BACKTEST CON COLECCION DE TRADES
# =============================================================================

def backtest_model(model_name, model_config, df, market_type):
    """Backtest de un modelo y colecta todos los trades"""

    models = load_models(model_config['path'])
    if models is None:
        return None, []

    feat = compute_features(df)
    common_idx = feat.index.intersection(df.index)
    df_aligned = df.loc[common_idx]
    feat = feat.loc[common_idx]

    X = feat[FEATURE_COLS].values
    signals, avg_prob, prob_rf, prob_gb = predict_ensemble(models, X)

    all_trades = []
    skip_until = -1

    for i, signal in enumerate(signals):
        if i <= skip_until:
            continue
        if not signal:
            continue

        result = simulate_trade(df_aligned, i, model_config['tp'], model_config['sl'])
        if result:
            # Agregar features del momento de entrada
            trade_features = feat.iloc[i].to_dict()
            result['model'] = model_name
            result['market_type'] = market_type
            result['prob_avg'] = avg_prob[i]
            result['prob_rf'] = prob_rf[i]
            result['prob_gb'] = prob_gb[i]
            result['features'] = trade_features
            all_trades.append(result)
            skip_until = i + result['bars']

    # Calcular metricas
    if not all_trades:
        return None, []

    trades_df = pd.DataFrame(all_trades)
    n = len(trades_df)
    wins = trades_df['outcome'].sum()
    wr = wins / n * 100
    pnl = trades_df['pnl'].sum() * 100

    return {
        'model': model_name,
        'market_type': market_type,
        'trades': n,
        'wins': int(wins),
        'losses': int(n - wins),
        'wr': round(wr, 1),
        'pnl': round(pnl, 1),
    }, all_trades


# =============================================================================
# ANALISIS DE TRADES MALOS
# =============================================================================

def analyze_bad_trades(all_trades):
    """Analiza diferencias entre trades buenos y malos"""

    if not all_trades:
        return None

    # Separar buenos y malos
    good_trades = [t for t in all_trades if t['outcome'] == 1]
    bad_trades = [t for t in all_trades if t['outcome'] == 0]

    if not bad_trades or not good_trades:
        return None

    print(f"\n{'='*70}")
    print("ANALISIS DE TRADES MALOS vs BUENOS")
    print(f"{'='*70}")
    print(f"Total: {len(all_trades)} trades")
    print(f"Buenos: {len(good_trades)} ({len(good_trades)/len(all_trades)*100:.1f}%)")
    print(f"Malos: {len(bad_trades)} ({len(bad_trades)/len(all_trades)*100:.1f}%)")

    # Extraer features
    feature_names = list(all_trades[0]['features'].keys())

    print(f"\n{'Feature':<20} {'Good Avg':>12} {'Bad Avg':>12} {'Diff':>12} {'Potential':<10}")
    print("-" * 70)

    potential_filters = []

    for feat_name in feature_names:
        good_vals = [t['features'].get(feat_name, 0) for t in good_trades if not pd.isna(t['features'].get(feat_name, 0))]
        bad_vals = [t['features'].get(feat_name, 0) for t in bad_trades if not pd.isna(t['features'].get(feat_name, 0))]

        if not good_vals or not bad_vals:
            continue

        good_avg = np.mean(good_vals)
        bad_avg = np.mean(bad_vals)
        diff = good_avg - bad_avg
        diff_pct = abs(diff / (good_avg + 1e-10)) * 100

        # Identificar features con diferencia significativa
        potential = ""
        if diff_pct > 15:
            if diff > 0:
                potential = f"> {bad_avg:.3f}"
            else:
                potential = f"< {bad_avg:.3f}"
            potential_filters.append({
                'feature': feat_name,
                'good_avg': good_avg,
                'bad_avg': bad_avg,
                'diff': diff,
                'diff_pct': diff_pct,
                'filter_suggestion': potential
            })

        print(f"{feat_name:<20} {good_avg:>12.4f} {bad_avg:>12.4f} {diff:>+12.4f} {potential:<10}")

    # Analisis de probabilidades
    print(f"\n{'='*70}")
    print("ANALISIS DE PROBABILIDADES")
    print(f"{'='*70}")

    good_probs = [t['prob_avg'] for t in good_trades]
    bad_probs = [t['prob_avg'] for t in bad_trades]

    print(f"Prob promedio en trades BUENOS: {np.mean(good_probs):.3f}")
    print(f"Prob promedio en trades MALOS: {np.mean(bad_probs):.3f}")
    print(f"Diferencia: {np.mean(good_probs) - np.mean(bad_probs):+.3f}")

    # Distribucion por rangos de probabilidad
    print(f"\nDistribucion por rango de probabilidad:")
    for low, high in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]:
        good_in_range = len([p for p in good_probs if low <= p < high])
        bad_in_range = len([p for p in bad_probs if low <= p < high])
        total_in_range = good_in_range + bad_in_range
        if total_in_range > 0:
            wr_range = good_in_range / total_in_range * 100
            print(f"  Prob {low:.1f}-{high:.1f}: {total_in_range:>4} trades, WR {wr_range:.1f}%")

    return potential_filters


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("VALIDACION DE MODELOS ENSEMBLE CON DATOS SINTETICOS")
    print("=" * 80)
    print("\nObjetivo: Probar modelos con datos que NUNCA han visto")
    print("y analizar por que fallan los trades perdedores.\n")

    # Generar datos sinteticos
    candles_per_year = 2190  # 1 año de velas 4h
    market_types = ['bull', 'bear', 'range', 'mixed', 'volatile']

    all_results = []
    all_trades = []

    for model_name, model_config in MODELS.items():
        print(f"\n{'='*60}")
        print(f"MODELO: {model_name}")
        print(f"{'='*60}")

        model_trades = []

        for market_type in market_types:
            print(f"\n  Generando datos {market_type.upper()}...", end=" ")

            # Usar seed diferente por modelo y tipo para variedad
            seed = hash(f"{model_name}_{market_type}") % 10000
            df_synthetic = generate_synthetic_ohlcv(candles_per_year, market_type,
                                                     start_price=100, seed=seed)

            result, trades = backtest_model(model_name, model_config, df_synthetic, market_type)

            if result:
                status = "OK" if result['pnl'] > 0 else "FAIL"
                print(f"{result['trades']} trades, WR {result['wr']}%, PnL {result['pnl']:+.1f}% [{status}]")
                all_results.append(result)
                model_trades.extend(trades)
            else:
                print("Sin trades")

        # Analisis de trades malos del modelo
        if model_trades:
            all_trades.extend(model_trades)
            print(f"\n  --- Analisis de trades malos para {model_name} ---")
            analyze_bad_trades(model_trades)

    # Resumen final
    print("\n" + "=" * 80)
    print("RESUMEN FINAL - TODOS LOS MODELOS")
    print("=" * 80)

    print(f"\n{'Modelo':<8} {'Market':<10} {'Trades':<8} {'WR%':<8} {'PnL%':<10} {'Status':<8}")
    print("-" * 60)

    for r in all_results:
        status = "OK" if r['pnl'] > 0 else "FAIL"
        print(f"{r['model']:<8} {r['market_type']:<10} {r['trades']:<8} {r['wr']:<8} {r['pnl']:+<10.1f} {status:<8}")

    # Estadisticas globales
    positive = sum(1 for r in all_results if r['pnl'] > 0)
    total = len(all_results)

    print(f"\nResultados positivos: {positive}/{total} ({positive/total*100:.0f}%)")

    # Analisis global de todos los trades
    print("\n" + "=" * 80)
    print("ANALISIS GLOBAL DE TODOS LOS TRADES")
    print("=" * 80)

    potential_filters = analyze_bad_trades(all_trades)

    # Guardar trades para analisis posterior
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': all_results,
        'trades_summary': {
            'total': len(all_trades),
            'wins': len([t for t in all_trades if t['outcome'] == 1]),
            'losses': len([t for t in all_trades if t['outcome'] == 0]),
        }
    }

    # Guardar trades detallados (sin features completos para no hacer el archivo muy grande)
    trades_export = []
    for t in all_trades:
        trades_export.append({
            'model': t['model'],
            'market_type': t['market_type'],
            'outcome': t['outcome'],
            'pnl': t['pnl'],
            'prob_avg': t['prob_avg'],
            'prob_rf': t['prob_rf'],
            'prob_gb': t['prob_gb'],
            'exit_reason': t['exit_reason'],
            'bars': t['bars'],
            # Features clave
            'rsi': t['features'].get('rsi', 0),
            'adx': t['features'].get('adx', 0),
            'bb_pct': t['features'].get('bb_pct', 0),
            'ret_3': t['features'].get('ret_3', 0),
            'vol_ratio': t['features'].get('vol_ratio', 0),
            'trend': t['features'].get('trend', 0),
        })

    output['trades'] = trades_export

    output_path = Path('analysis/synthetic_validation_results.json')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResultados guardados en: {output_path}")

    if potential_filters:
        print(f"\n{'='*80}")
        print("FILTROS POTENCIALES IDENTIFICADOS")
        print("=" * 80)
        for pf in sorted(potential_filters, key=lambda x: -x['diff_pct'])[:5]:
            print(f"  {pf['feature']}: {pf['filter_suggestion']} (diff {pf['diff_pct']:.1f}%)")


if __name__ == '__main__':
    main()
