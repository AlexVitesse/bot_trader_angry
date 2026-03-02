"""
VALIDACION COMPLETA DE MODELOS V14 CON DATOS SINTETICOS ESPECIFICOS
====================================================================

Objetivo: Probar cada modelo con datos sintéticos que imitan el comportamiento
específico de cada tipo de moneda.

Modelos a validar:
1. DOGE - Memecoins (alta volatilidad, movimientos explosivos)
2. ADA - Smart Contracts (volatilidad media, ciclos de desarrollo)
3. DOT - Infraestructura (volatilidad moderada, más estable)
4. SOL - Comportamiento propio (alta volatilidad, recuperaciones rápidas)

Para cada modelo:
- Generar 5 escenarios sintéticos específicos
- Colectar todos los trades (buenos y malos)
- Analizar features de trades perdedores
- Proponer filtros específicos por modelo
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from datetime import datetime
import ta
from collections import defaultdict

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACION DE MODELOS
# =============================================================================

MODELS_CONFIG = {
    'DOGE': {
        'path': 'strategies/doge_v14/models',
        'tp': 0.06,
        'sl': 0.04,
        'type': 'memecoin',
        'cross_pairs': ['PEPE', 'FLOKI'],  # SHIB excluido
    },
    'ADA': {
        'path': 'strategies/ada_v14/models',
        'tp': 0.06,
        'sl': 0.04,
        'type': 'smart_contract',
        'cross_pairs': ['ATOM', 'AVAX', 'POL'],
    },
    'DOT': {
        'path': 'strategies/dot_v14/models',
        'tp': 0.05,
        'sl': 0.03,
        'type': 'infrastructure',
        'cross_pairs': ['LINK', 'ALGO', 'FIL', 'NEAR'],
    },
    'SOL': {
        'path': 'strategies/sol_v14/models',
        'tp': 0.06,
        'sl': 0.04,
        'type': 'solana',
        'cross_pairs': [],
    },
}

FEATURE_COLS = ['rsi', 'macd_norm', 'adx', 'bb_pct', 'atr_pct',
                'ret_3', 'ret_5', 'ret_10', 'vol_ratio', 'trend']

TIMEOUT_CANDLES = 15
CANDLES_PER_YEAR = 2190  # 365 * 6

# =============================================================================
# GENERADORES DE DATOS SINTETICOS POR TIPO DE MONEDA
# =============================================================================

def generate_memecoin_synthetic(n_candles, market_type, start_price=0.10, seed=None):
    """
    Genera datos sintéticos para MEMECOINS (DOGE, SHIB, PEPE, FLOKI)
    Características:
    - Volatilidad muy alta (3-8% por vela)
    - Pumps y dumps explosivos
    - Periodos de consolidación largos seguidos de movimientos violentos
    - Correlación con sentiment de mercado
    """
    if seed:
        np.random.seed(seed)

    # Parámetros específicos de memecoins
    params = {
        'bull': {'drift': 0.003, 'vol': 0.05, 'pump_prob': 0.05, 'pump_size': 0.15},
        'bear': {'drift': -0.002, 'vol': 0.06, 'pump_prob': 0.02, 'pump_size': -0.12},
        'range': {'drift': 0.0, 'vol': 0.04, 'pump_prob': 0.03, 'pump_size': 0.08},
        'volatile': {'drift': 0.0, 'vol': 0.08, 'pump_prob': 0.08, 'pump_size': 0.20},
        'mixed': None,  # Handled separately
    }

    dates = pd.date_range(start='2025-01-01', periods=n_candles, freq='4h')
    closes = [start_price]
    volumes = [1000000]

    if market_type == 'mixed':
        segment_size = n_candles // 5
        segments = ['range', 'bull', 'bear', 'volatile', 'range']

        for i in range(1, n_candles):
            segment_idx = min(i // segment_size, len(segments) - 1)
            current_type = segments[segment_idx]
            p = params[current_type]

            # Movimiento base
            returns = np.random.normal(p['drift'], p['vol'])

            # Pump/dump aleatorio (característica de memecoins)
            if np.random.random() < p['pump_prob']:
                returns += p['pump_size'] * (1 if np.random.random() > 0.5 else -1)

            new_price = closes[-1] * (1 + returns)
            closes.append(max(new_price, start_price * 0.01))

            # Volumen correlacionado con movimiento
            vol_mult = 1 + abs(returns) * 20 + (5 if abs(returns) > 0.05 else 0)
            volumes.append(volumes[-1] * np.random.lognormal(0, 0.3) * vol_mult / vol_mult)
    else:
        p = params[market_type]
        for i in range(1, n_candles):
            returns = np.random.normal(p['drift'], p['vol'])
            if np.random.random() < p['pump_prob']:
                returns += p['pump_size'] * (1 if np.random.random() > 0.5 else -1)
            new_price = closes[-1] * (1 + returns)
            closes.append(max(new_price, start_price * 0.01))
            vol_mult = 1 + abs(returns) * 20
            volumes.append(volumes[-1] * np.random.lognormal(0, 0.3) * vol_mult / vol_mult)

    closes = np.array(closes)
    volumes = np.array(volumes)

    # OHLV con wicks más grandes (típico de memecoins)
    wick_size = np.abs(np.random.normal(0.015, 0.01, n_candles))
    highs = closes * (1 + wick_size)
    lows = closes * (1 - wick_size)
    opens = np.roll(closes, 1)
    opens[0] = start_price

    return pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': volumes
    }, index=dates)


def generate_smart_contract_synthetic(n_candles, market_type, start_price=1.0, seed=None):
    """
    Genera datos sintéticos para SMART CONTRACTS (ADA, ATOM, AVAX, POL)
    Características:
    - Volatilidad media (2-4% por vela)
    - Ciclos relacionados con desarrollos/upgrades
    - Correlación moderada con ETH
    - Movimientos más suaves que memecoins
    """
    if seed:
        np.random.seed(seed)

    params = {
        'bull': {'drift': 0.0018, 'vol': 0.025, 'cycle_amp': 0.02},
        'bear': {'drift': -0.0015, 'vol': 0.03, 'cycle_amp': 0.015},
        'range': {'drift': 0.0002, 'vol': 0.02, 'cycle_amp': 0.025},
        'volatile': {'drift': 0.0, 'vol': 0.045, 'cycle_amp': 0.03},
        'mixed': None,
    }

    dates = pd.date_range(start='2025-01-01', periods=n_candles, freq='4h')
    closes = [start_price]

    # Ciclo de desarrollo (simula upgrades, releases)
    cycle_period = 180  # ~30 días

    if market_type == 'mixed':
        segment_size = n_candles // 4
        segments = ['bull', 'range', 'bear', 'bull']

        for i in range(1, n_candles):
            segment_idx = min(i // segment_size, len(segments) - 1)
            current_type = segments[segment_idx]
            p = params[current_type]

            # Componente cíclico (desarrollo)
            cycle = p['cycle_amp'] * np.sin(2 * np.pi * i / cycle_period)

            returns = np.random.normal(p['drift'] + cycle * 0.1, p['vol'])
            new_price = closes[-1] * (1 + returns)
            closes.append(max(new_price, start_price * 0.01))
    else:
        p = params[market_type]
        for i in range(1, n_candles):
            cycle = p['cycle_amp'] * np.sin(2 * np.pi * i / cycle_period)
            returns = np.random.normal(p['drift'] + cycle * 0.1, p['vol'])
            new_price = closes[-1] * (1 + returns)
            closes.append(max(new_price, start_price * 0.01))

    closes = np.array(closes)

    highs = closes * (1 + np.abs(np.random.normal(0.008, 0.004, n_candles)))
    lows = closes * (1 - np.abs(np.random.normal(0.008, 0.004, n_candles)))
    opens = np.roll(closes, 1)
    opens[0] = start_price

    base_volume = 500000
    vol_noise = np.random.lognormal(0, 0.4, n_candles)
    price_changes = np.abs(np.diff(closes, prepend=closes[0])) / closes
    volume = base_volume * vol_noise * (1 + price_changes * 8)

    return pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': volume
    }, index=dates)


def generate_infrastructure_synthetic(n_candles, market_type, start_price=10.0, seed=None):
    """
    Genera datos sintéticos para INFRAESTRUCTURA (DOT, LINK, ALGO, FIL, NEAR)
    Características:
    - Volatilidad moderada (1.5-3% por vela)
    - Más estable, menos pumps
    - Correlación con adopción institucional
    - Tendencias más claras y sostenidas
    """
    if seed:
        np.random.seed(seed)

    params = {
        'bull': {'drift': 0.0012, 'vol': 0.018, 'trend_strength': 0.3},
        'bear': {'drift': -0.001, 'vol': 0.022, 'trend_strength': 0.25},
        'range': {'drift': 0.0001, 'vol': 0.015, 'trend_strength': 0.1},
        'volatile': {'drift': 0.0, 'vol': 0.035, 'trend_strength': 0.15},
        'mixed': None,
    }

    dates = pd.date_range(start='2025-01-01', periods=n_candles, freq='4h')
    closes = [start_price]

    # Tendencia subyacente (adopción gradual)
    trend = 0

    if market_type == 'mixed':
        segment_size = n_candles // 4
        segments = ['range', 'bull', 'bear', 'range']

        for i in range(1, n_candles):
            segment_idx = min(i // segment_size, len(segments) - 1)
            current_type = segments[segment_idx]
            p = params[current_type]

            # Actualizar tendencia gradualmente
            trend = trend * 0.99 + p['drift'] * p['trend_strength']

            returns = np.random.normal(p['drift'] + trend, p['vol'])
            new_price = closes[-1] * (1 + returns)
            closes.append(max(new_price, start_price * 0.01))
    else:
        p = params[market_type]
        for i in range(1, n_candles):
            trend = trend * 0.99 + p['drift'] * p['trend_strength']
            returns = np.random.normal(p['drift'] + trend, p['vol'])
            new_price = closes[-1] * (1 + returns)
            closes.append(max(new_price, start_price * 0.01))

    closes = np.array(closes)

    # Wicks más pequeños (menos manipulación)
    highs = closes * (1 + np.abs(np.random.normal(0.005, 0.003, n_candles)))
    lows = closes * (1 - np.abs(np.random.normal(0.005, 0.003, n_candles)))
    opens = np.roll(closes, 1)
    opens[0] = start_price

    base_volume = 2000000
    vol_noise = np.random.lognormal(0, 0.35, n_candles)
    price_changes = np.abs(np.diff(closes, prepend=closes[0])) / closes
    volume = base_volume * vol_noise * (1 + price_changes * 5)

    return pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': volume
    }, index=dates)


def generate_solana_synthetic(n_candles, market_type, start_price=100.0, seed=None):
    """
    Genera datos sintéticos para SOLANA
    Características:
    - Alta volatilidad pero con recuperaciones rápidas
    - Movimientos correlacionados con actividad DeFi
    - Caídas bruscas seguidas de rebotes (patrón histórico de SOL)
    """
    if seed:
        np.random.seed(seed)

    params = {
        'bull': {'drift': 0.002, 'vol': 0.035, 'crash_prob': 0.01, 'recovery_speed': 0.3},
        'bear': {'drift': -0.0015, 'vol': 0.04, 'crash_prob': 0.02, 'recovery_speed': 0.2},
        'range': {'drift': 0.0003, 'vol': 0.028, 'crash_prob': 0.005, 'recovery_speed': 0.25},
        'volatile': {'drift': 0.0, 'vol': 0.055, 'crash_prob': 0.03, 'recovery_speed': 0.35},
        'mixed': None,
    }

    dates = pd.date_range(start='2025-01-01', periods=n_candles, freq='4h')
    closes = [start_price]

    recovering = False
    recovery_target = start_price

    if market_type == 'mixed':
        segment_size = n_candles // 4
        segments = ['bull', 'volatile', 'bear', 'bull']

        for i in range(1, n_candles):
            segment_idx = min(i // segment_size, len(segments) - 1)
            current_type = segments[segment_idx]
            p = params[current_type]

            if recovering:
                # Recuperación rápida después de crash
                returns = p['recovery_speed'] * (recovery_target - closes[-1]) / closes[-1]
                returns += np.random.normal(0, p['vol'] * 0.5)
                if closes[-1] >= recovery_target * 0.95:
                    recovering = False
            else:
                returns = np.random.normal(p['drift'], p['vol'])

                # Crash aleatorio
                if np.random.random() < p['crash_prob']:
                    crash_size = np.random.uniform(0.08, 0.15)
                    returns = -crash_size
                    recovering = True
                    recovery_target = closes[-1] * 0.95

            new_price = closes[-1] * (1 + returns)
            closes.append(max(new_price, start_price * 0.01))
    else:
        p = params[market_type]
        for i in range(1, n_candles):
            if recovering:
                returns = p['recovery_speed'] * (recovery_target - closes[-1]) / closes[-1]
                returns += np.random.normal(0, p['vol'] * 0.5)
                if closes[-1] >= recovery_target * 0.95:
                    recovering = False
            else:
                returns = np.random.normal(p['drift'], p['vol'])
                if np.random.random() < p['crash_prob']:
                    crash_size = np.random.uniform(0.08, 0.15)
                    returns = -crash_size
                    recovering = True
                    recovery_target = closes[-1] * 0.95

            new_price = closes[-1] * (1 + returns)
            closes.append(max(new_price, start_price * 0.01))

    closes = np.array(closes)

    highs = closes * (1 + np.abs(np.random.normal(0.01, 0.005, n_candles)))
    lows = closes * (1 - np.abs(np.random.normal(0.01, 0.005, n_candles)))
    opens = np.roll(closes, 1)
    opens[0] = start_price

    base_volume = 3000000
    vol_noise = np.random.lognormal(0, 0.5, n_candles)
    price_changes = np.abs(np.diff(closes, prepend=closes[0])) / closes
    volume = base_volume * vol_noise * (1 + price_changes * 12)

    return pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': volume
    }, index=dates)


# Mapeo de generadores
GENERATORS = {
    'memecoin': generate_memecoin_synthetic,
    'smart_contract': generate_smart_contract_synthetic,
    'infrastructure': generate_infrastructure_synthetic,
    'solana': generate_solana_synthetic,
}


# =============================================================================
# FEATURES Y MODELO
# =============================================================================

def compute_features(df):
    """Compute ML features"""
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

    # Features adicionales para análisis
    feat['rsi_raw'] = ta.momentum.rsi(df['close'], window=14)
    feat['ret_1'] = df['close'].pct_change(1)
    feat['ret_20'] = df['close'].pct_change(20)

    return feat.dropna()


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
    """Predicción ensemble"""
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


def simulate_trade(df, idx, tp, sl):
    """Simula un trade"""
    if idx >= len(df) - TIMEOUT_CANDLES - 1:
        return None

    entry_price = df['close'].iloc[idx]

    for i in range(1, min(TIMEOUT_CANDLES + 1, len(df) - idx)):
        future_price = df['close'].iloc[idx + i]
        pnl = (future_price - entry_price) / entry_price

        if pnl >= tp:
            return {'outcome': 1, 'pnl': pnl, 'bars': i, 'exit_reason': 'TP'}
        elif pnl <= -sl:
            return {'outcome': 0, 'pnl': -sl, 'bars': i, 'exit_reason': 'SL'}

    final_price = df['close'].iloc[min(idx + TIMEOUT_CANDLES, len(df) - 1)]
    final_pnl = (final_price - entry_price) / entry_price
    return {
        'outcome': 1 if final_pnl > 0 else 0,
        'pnl': final_pnl,
        'bars': TIMEOUT_CANDLES,
        'exit_reason': 'TIMEOUT'
    }


# =============================================================================
# BACKTEST Y ANÁLISIS
# =============================================================================

def backtest_model(model_name, config, df, market_type):
    """Backtest de un modelo"""
    models = load_models(config['path'])
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

        result = simulate_trade(df_aligned, i, config['tp'], config['sl'])
        if result:
            trade_features = feat.iloc[i].to_dict()
            result['model'] = model_name
            result['market_type'] = market_type
            result['prob_avg'] = avg_prob[i]
            result['prob_rf'] = prob_rf[i]
            result['prob_gb'] = prob_gb[i]
            result['features'] = trade_features
            all_trades.append(result)
            skip_until = i + result['bars']

    if not all_trades:
        return None, []

    trades_df = pd.DataFrame(all_trades)
    n = len(trades_df)
    wins = trades_df['outcome'].sum()

    return {
        'model': model_name,
        'market_type': market_type,
        'trades': n,
        'wins': int(wins),
        'losses': int(n - wins),
        'wr': round(wins / n * 100, 1),
        'pnl': round(trades_df['pnl'].sum() * 100, 1),
    }, all_trades


def analyze_model_trades(model_name, trades):
    """Analiza trades de un modelo específico"""
    if not trades:
        return None

    good = [t for t in trades if t['outcome'] == 1]
    bad = [t for t in trades if t['outcome'] == 0]

    if not bad or not good:
        return None

    print(f"\n{'='*70}")
    print(f"ANÁLISIS DE TRADES - {model_name}")
    print(f"{'='*70}")
    print(f"Total: {len(trades)} | Buenos: {len(good)} ({len(good)/len(trades)*100:.1f}%) | Malos: {len(bad)} ({len(bad)/len(trades)*100:.1f}%)")

    # Análisis de features
    feature_analysis = []
    feature_names = [f for f in trades[0]['features'].keys() if not f.startswith('_')]

    print(f"\n{'Feature':<15} {'Good':<10} {'Bad':<10} {'Diff':<10} {'Filter':<20}")
    print("-" * 65)

    for feat_name in sorted(feature_names):
        good_vals = [t['features'].get(feat_name, 0) for t in good if pd.notna(t['features'].get(feat_name))]
        bad_vals = [t['features'].get(feat_name, 0) for t in bad if pd.notna(t['features'].get(feat_name))]

        if not good_vals or not bad_vals:
            continue

        good_avg = np.mean(good_vals)
        bad_avg = np.mean(bad_vals)
        diff = good_avg - bad_avg
        diff_pct = abs(diff / (abs(good_avg) + 1e-10)) * 100

        filter_suggestion = ""
        if diff_pct > 20:
            if diff > 0:
                filter_suggestion = f"> {bad_avg:.4f}"
            else:
                filter_suggestion = f"< {bad_avg:.4f}"

            feature_analysis.append({
                'feature': feat_name,
                'good_avg': good_avg,
                'bad_avg': bad_avg,
                'diff': diff,
                'diff_pct': diff_pct,
                'filter': filter_suggestion
            })

        print(f"{feat_name:<15} {good_avg:<10.4f} {bad_avg:<10.4f} {diff:<+10.4f} {filter_suggestion:<20}")

    # Análisis de probabilidad
    print(f"\nProbabilidades:")
    print(f"  Buenos: {np.mean([t['prob_avg'] for t in good]):.3f}")
    print(f"  Malos:  {np.mean([t['prob_avg'] for t in bad]):.3f}")

    # Top 3 filtros potenciales
    if feature_analysis:
        print(f"\nTOP FILTROS POTENCIALES para {model_name}:")
        for fa in sorted(feature_analysis, key=lambda x: -x['diff_pct'])[:3]:
            print(f"  {fa['feature']}: {fa['filter']} (diff {fa['diff_pct']:.1f}%)")

    return feature_analysis


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("VALIDACIÓN COMPLETA DE MODELOS V14")
    print("Datos sintéticos específicos por tipo de moneda")
    print("=" * 80)

    market_types = ['bull', 'bear', 'range', 'mixed', 'volatile']
    all_results = []
    model_trades = defaultdict(list)

    for model_name, config in MODELS_CONFIG.items():
        print(f"\n{'#'*80}")
        print(f"# MODELO: {model_name} (tipo: {config['type']})")
        print(f"# Cross-pairs: {config['cross_pairs'] if config['cross_pairs'] else 'Ninguno'}")
        print(f"{'#'*80}")

        generator = GENERATORS[config['type']]

        for market_type in market_types:
            seed = hash(f"{model_name}_{market_type}_v2") % 10000

            # Generar datos específicos del tipo de moneda
            df = generator(CANDLES_PER_YEAR, market_type, seed=seed)

            result, trades = backtest_model(model_name, config, df, market_type)

            if result:
                status = "OK" if result['pnl'] > 0 else "FAIL"
                print(f"  {market_type.upper():<10}: {result['trades']:>3} trades, WR {result['wr']:>5.1f}%, PnL {result['pnl']:>+7.1f}% [{status}]")
                all_results.append(result)
                model_trades[model_name].extend(trades)
            else:
                print(f"  {market_type.upper():<10}: Sin trades")

        # Análisis de trades malos para este modelo
        if model_trades[model_name]:
            analyze_model_trades(model_name, model_trades[model_name])

    # Resumen final
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)

    print(f"\n{'Modelo':<8} {'Mercado':<10} {'Trades':<8} {'WR%':<8} {'PnL%':<10} {'Status':<8}")
    print("-" * 60)

    for r in all_results:
        status = "OK" if r['pnl'] > 0 else "FAIL"
        print(f"{r['model']:<8} {r['market_type']:<10} {r['trades']:<8} {r['wr']:<8.1f} {r['pnl']:<+10.1f} {status:<8}")

    # Estadísticas por modelo
    print(f"\n{'='*80}")
    print("RESUMEN POR MODELO")
    print("=" * 80)

    for model_name in MODELS_CONFIG.keys():
        model_results = [r for r in all_results if r['model'] == model_name]
        if model_results:
            total_trades = sum(r['trades'] for r in model_results)
            total_wins = sum(r['wins'] for r in model_results)
            total_pnl = sum(r['pnl'] for r in model_results)
            positive_scenarios = sum(1 for r in model_results if r['pnl'] > 0)

            avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0

            status = "APROBADO" if positive_scenarios >= 3 else "REVISAR"
            print(f"\n{model_name}:")
            print(f"  Escenarios positivos: {positive_scenarios}/5")
            print(f"  Total trades: {total_trades}")
            print(f"  WR promedio: {avg_wr:.1f}%")
            print(f"  PnL total: {total_pnl:+.1f}%")
            print(f"  Status: [{status}]")

    # Guardar resultados
    output = {
        'timestamp': datetime.now().isoformat(),
        'methodology': 'Synthetic data specific to coin type',
        'results': all_results,
        'model_analysis': {}
    }

    for model_name, trades in model_trades.items():
        if trades:
            output['model_analysis'][model_name] = {
                'total_trades': len(trades),
                'wins': len([t for t in trades if t['outcome'] == 1]),
                'losses': len([t for t in trades if t['outcome'] == 0]),
            }

    output_path = Path('analysis/validation_synthetic_v2.json')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResultados guardados en: {output_path}")


if __name__ == '__main__':
    main()
