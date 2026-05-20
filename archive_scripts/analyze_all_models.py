"""
Analisis completo de TODOS los modelos V14
- BTC V14 (regimen + setups + ensemble)
- SOL V14 (modelo dedicado)
- ADA, DOGE, DOT (modelos base)

Genera trades con todas las features para analisis de filtros
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from datetime import datetime
import ta
import pandas_ta as pta
from enum import Enum

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACION
# =============================================================================
TIMEOUT = 15

FEATURE_COLS = ['rsi', 'macd_norm', 'adx', 'bb_pct', 'atr_pct',
                'ret_3', 'ret_5', 'ret_10', 'vol_ratio', 'trend']

# Todos los modelos a analizar
MODELS_CONFIG = {
    'BTC': {
        'type': 'btc_v14',
        'path': 'strategies/btc_v14/models',
        'tp': 0.04, 'sl': 0.015,
        'pairs': ['BTCUSDT']
    },
    'SOL': {
        'type': 'ensemble',
        'path': 'strategies/sol_v14/models',
        'tp': 0.06, 'sl': 0.04,
        'pairs': ['SOLUSDT']
    },
    'ADA': {
        'type': 'ensemble',
        'path': 'strategies/ada_v14/models',
        'tp': 0.06, 'sl': 0.04,
        'pairs': ['ADAUSDT', 'ATOMUSDT', 'AVAXUSDT']
    },
    'DOGE': {
        'type': 'ensemble',
        'path': 'strategies/doge_v14/models',
        'tp': 0.06, 'sl': 0.04,
        'pairs': ['DOGEUSDT']
    },
    'DOT': {
        'type': 'ensemble',
        'path': 'strategies/dot_v14/models',
        'tp': 0.05, 'sl': 0.03,
        'pairs': ['DOTUSDT', 'LINKUSDT']
    },
}

# =============================================================================
# BTC V14 - Clases
# =============================================================================

class Regime(Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    VOLATILE = "VOLATILE"

class Strategy(Enum):
    TREND_FOLLOW_LONG = "TREND_FOLLOW_LONG"
    TREND_FOLLOW_SHORT = "TREND_FOLLOW_SHORT"
    MEAN_REVERSION_LONG = "MEAN_REVERSION_LONG"
    MEAN_REVERSION_SHORT = "MEAN_REVERSION_SHORT"
    BREAKOUT_LONG = "BREAKOUT_LONG"
    BREAKOUT_SHORT = "BREAKOUT_SHORT"

BTC_STRATEGY_PARAMS = {
    Strategy.TREND_FOLLOW_LONG: {'tp': 0.04, 'sl': 0.015},
    Strategy.TREND_FOLLOW_SHORT: {'tp': 0.04, 'sl': 0.015},
    Strategy.MEAN_REVERSION_LONG: {'tp': 0.025, 'sl': 0.012},
    Strategy.MEAN_REVERSION_SHORT: {'tp': 0.025, 'sl': 0.012},
    Strategy.BREAKOUT_LONG: {'tp': 0.05, 'sl': 0.02},
    Strategy.BREAKOUT_SHORT: {'tp': 0.05, 'sl': 0.02},
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(pair):
    csv_path = Path(f'data/{pair}_4h.csv')
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    return df


def load_ensemble_models(model_path):
    model_dir = Path(model_path)
    try:
        models = {
            'scaler': joblib.load(model_dir / 'scaler.pkl'),
            'rf': joblib.load(model_dir / 'random_forest.pkl'),
            'gb': joblib.load(model_dir / 'gradient_boosting.pkl')
        }
        lr_path = model_dir / 'logistic_regression.pkl'
        if lr_path.exists():
            models['lr'] = joblib.load(lr_path)
        return models
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

# =============================================================================
# FEATURES - ENSEMBLE
# =============================================================================

def compute_ensemble_features(df):
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
    return feat.dropna()

# =============================================================================
# FEATURES - BTC V14
# =============================================================================

def compute_btc_features(df):
    """Features completas para BTC V14"""
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    feat = pd.DataFrame(index=df.index)

    # ADX/DI
    adx_df = pta.adx(h, l, c, length=14)
    if adx_df is not None:
        feat['adx'] = adx_df.iloc[:, 0]
        feat['di_plus'] = adx_df.iloc[:, 1]
        feat['di_minus'] = adx_df.iloc[:, 2]
        feat['di_diff'] = feat['di_plus'] - feat['di_minus']

    chop = pta.chop(h, l, c, length=14)
    feat['chop'] = chop if chop is not None else 50

    feat['ema20'] = pta.ema(c, length=20)
    feat['ema50'] = pta.ema(c, length=50)
    feat['ema200'] = pta.ema(c, length=200)
    feat['ema20_dist'] = (c - feat['ema20']) / feat['ema20'] * 100
    feat['ema200_dist'] = (c - feat['ema200']) / feat['ema200'] * 100
    feat['ema20_slope'] = feat['ema20'].pct_change(5) * 100
    feat['ema50_slope'] = feat['ema50'].pct_change(10) * 100

    feat['atr'] = pta.atr(h, l, c, length=14)
    feat['atr_pct'] = feat['atr'] / c * 100

    bb = pta.bbands(c, length=20)
    if bb is not None:
        feat['bb_width'] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / bb.iloc[:, 1] * 100
        feat['bb_pct'] = (c - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])

    feat['rsi14'] = pta.rsi(c, length=14)
    feat['rsi7'] = pta.rsi(c, length=7)

    stoch = pta.stoch(h, l, c, k=14, d=3)
    if stoch is not None:
        feat['stoch_k'] = stoch.iloc[:, 0]

    feat['ret_5'] = c.pct_change(5) * 100
    feat['ret_20'] = c.pct_change(20) * 100
    feat['ret_3'] = c.pct_change(3)  # Para filtros

    feat['vol_ratio'] = v / v.rolling(20).mean()
    feat['vol_trend'] = v.rolling(5).mean() / v.rolling(20).mean()
    obv = (np.sign(c.diff()) * v).cumsum()
    feat['obv_slope'] = obv.pct_change(10) * 100

    feat['high_20'] = h.rolling(20).max()
    feat['low_20'] = l.rolling(20).min()
    feat['range_pos'] = (c - feat['low_20']) / (feat['high_20'] - feat['low_20'])
    feat['consec_up'] = (c > c.shift(1)).rolling(10).sum()
    feat['consec_down'] = (c < c.shift(1)).rolling(10).sum()

    # Trend para filtros
    feat['trend'] = (c > feat['ema50']).astype(float)

    return feat.dropna()


def detect_btc_regime(row):
    adx = row.get('adx', 20)
    di_diff = row.get('di_diff', 0)
    chop = row.get('chop', 50)
    atr_pct = row.get('atr_pct', 2)
    bb_width = row.get('bb_width', 5)
    ema20_slope = row.get('ema20_slope', 0)
    ema50_slope = row.get('ema50_slope', 0)

    if pd.isna(adx) or pd.isna(chop):
        return Regime.RANGE

    if atr_pct > 4 and bb_width > 8:
        return Regime.VOLATILE

    if adx > 25 and chop < 50:
        if di_diff > 5 and ema20_slope > 0:
            return Regime.TREND_UP
        elif di_diff < -5 and ema20_slope < 0:
            return Regime.TREND_DOWN

    if chop > 55 or adx < 20:
        return Regime.RANGE

    if ema50_slope > 0.5:
        return Regime.TREND_UP
    elif ema50_slope < -0.5:
        return Regime.TREND_DOWN

    return Regime.RANGE


def detect_btc_setup(row, regime):
    rsi14 = row.get('rsi14', 50)
    bb_pct = row.get('bb_pct', 0.5)
    range_pos = row.get('range_pos', 0.5)
    ema20_dist = row.get('ema20_dist', 0)
    ema200_dist = row.get('ema200_dist', 0)
    vol_ratio = row.get('vol_ratio', 1)
    consec_up = row.get('consec_up', 0)
    consec_down = row.get('consec_down', 0)

    if pd.isna(rsi14):
        return None, None

    if regime == Regime.TREND_UP:
        if rsi14 < 40 and bb_pct < 0.3 and ema200_dist > 0:
            return Strategy.TREND_FOLLOW_LONG, 'PULLBACK_UPTREND'
        elif rsi14 < 30 and ema20_dist < -2:
            return Strategy.TREND_FOLLOW_LONG, 'OVERSOLD_UPTREND'

    elif regime == Regime.TREND_DOWN:
        if rsi14 > 60 and bb_pct > 0.7 and ema200_dist < 0:
            return Strategy.TREND_FOLLOW_SHORT, 'RALLY_DOWNTREND'
        elif rsi14 > 70 and ema20_dist > 2:
            return Strategy.TREND_FOLLOW_SHORT, 'OVERBOUGHT_DOWNTREND'

    elif regime == Regime.RANGE:
        if range_pos < 0.2 and rsi14 < 35:
            return Strategy.MEAN_REVERSION_LONG, 'SUPPORT_BOUNCE'
        elif range_pos > 0.8 and rsi14 > 65:
            return Strategy.MEAN_REVERSION_SHORT, 'RESISTANCE_REJECT'

    elif regime == Regime.VOLATILE:
        if bb_pct > 1.0 and vol_ratio > 1.5 and consec_up >= 3:
            return Strategy.BREAKOUT_LONG, 'BREAKOUT_UP'
        elif bb_pct < 0 and vol_ratio > 1.5 and consec_down >= 3:
            return Strategy.BREAKOUT_SHORT, 'BREAKOUT_DOWN'

    return None, None


def get_btc_ensemble_confidence(row, direction):
    """Obtiene confianza del ensemble BTC"""
    probs = []

    for name in ['context', 'momentum', 'volume']:
        model_path = Path(f'strategies/btc_v14/models/{name}_{direction}.pkl')
        if not model_path.exists():
            continue

        try:
            model_data = joblib.load(model_path)
            model = model_data['model']
            features = model_data['features']

            X = np.array([row.get(f, 0) for f in features]).reshape(1, -1)
            if np.isnan(X).any():
                continue

            if 'scaler' in model_data:
                X = model_data['scaler'].transform(X)

            prob = model.predict_proba(X)[0, 1]
            probs.append(prob)
        except:
            continue

    return np.mean(probs) if probs else 0.5

# =============================================================================
# BACKTESTING
# =============================================================================

def simulate_trade(df, idx, tp_pct, sl_pct, direction='LONG'):
    """Simula un trade y retorna resultado"""
    if idx >= len(df) - TIMEOUT - 1:
        return None

    entry_price = df['close'].iloc[idx]

    if direction == 'LONG':
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
    else:
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)

    exit_reason = 'TIMEOUT'
    exit_price = df['close'].iloc[min(idx + TIMEOUT, len(df) - 1)]
    bars_held = TIMEOUT
    max_profit = 0
    max_dd = 0

    for i in range(1, min(TIMEOUT + 1, len(df) - idx)):
        high = df['high'].iloc[idx + i]
        low = df['low'].iloc[idx + i]

        if direction == 'LONG':
            high_pnl = (high - entry_price) / entry_price
            low_pnl = (low - entry_price) / entry_price
        else:
            high_pnl = (entry_price - low) / entry_price
            low_pnl = (entry_price - high) / entry_price

        max_profit = max(max_profit, high_pnl)
        max_dd = min(max_dd, low_pnl)

        if direction == 'LONG':
            if high >= tp_price:
                exit_reason = 'TP'
                exit_price = tp_price
                bars_held = i
                break
            if low <= sl_price:
                exit_reason = 'SL'
                exit_price = sl_price
                bars_held = i
                break
        else:
            if low <= tp_price:
                exit_reason = 'TP'
                exit_price = tp_price
                bars_held = i
                break
            if high >= sl_price:
                exit_reason = 'SL'
                exit_price = sl_price
                bars_held = i
                break

    if direction == 'LONG':
        pnl_pct = (exit_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - exit_price) / entry_price

    return {
        'entry_price': entry_price,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'bars_held': bars_held,
        'pnl_pct': round(pnl_pct * 100, 2),
        'max_profit_pct': round(max_profit * 100, 2),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'direction': direction
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("ANALISIS COMPLETO - TODOS LOS MODELOS V14")
    print("=" * 70)

    all_trades = []

    for model_name, config in MODELS_CONFIG.items():
        print(f"\n{'='*50}")
        print(f"MODELO: {model_name}")
        print(f"{'='*50}")

        # BTC V14
        if config['type'] == 'btc_v14':
            btc_models_exist = Path('strategies/btc_v14/models/context_long.pkl').exists()
            if not btc_models_exist:
                print("  Modelos BTC no encontrados, saltando...")
                continue

            for pair in config['pairs']:
                print(f"  {pair}...", end=" ", flush=True)
                df = load_data(pair)
                if df is None:
                    print("NO DATA")
                    continue

                feat = compute_btc_features(df)
                trades = []
                skip_until = -1

                for idx in range(len(feat)):
                    if idx <= skip_until:
                        continue

                    row = feat.iloc[idx]
                    regime = detect_btc_regime(row)
                    strategy, setup_name = detect_btc_setup(row, regime)

                    if strategy:
                        direction = 'LONG' if 'LONG' in strategy.value else 'SHORT'
                        confidence = get_btc_ensemble_confidence(row, direction.lower())

                        if confidence >= 0.35:
                            params = BTC_STRATEGY_PARAMS.get(strategy, {'tp': 0.04, 'sl': 0.015})
                            result = simulate_trade(df, idx, params['tp'], params['sl'], direction)

                            if result:
                                trade_info = {
                                    'model': model_name,
                                    'pair': pair,
                                    'timestamp': str(feat.index[idx]),
                                    'probability': round(float(confidence), 4),
                                    'setup': f"{regime.value}:{setup_name}",
                                    'features': {
                                        'rsi14': round(float(row.get('rsi14', 0)), 4),
                                        'bb_pct': round(float(row.get('bb_pct', 0)), 4),
                                        'atr_pct': round(float(row.get('atr_pct', 0)), 4),
                                        'ret_3': round(float(row.get('ret_3', 0)), 4),
                                        'ret_5': round(float(row.get('ret_5', 0)), 4),
                                        'vol_ratio': round(float(row.get('vol_ratio', 0)), 4),
                                        'trend': round(float(row.get('trend', 0)), 4),
                                        'adx': round(float(row.get('adx', 0)), 4),
                                        'regime': regime.value,
                                    },
                                    'trade': result
                                }
                                trades.append(trade_info)
                                skip_until = idx + result['bars_held']

                wins = len([t for t in trades if t['trade']['exit_reason'] == 'TP'])
                print(f"{len(trades)} trades, WR {wins/len(trades)*100:.1f}%" if trades else "0 trades")
                all_trades.extend(trades)

        # Ensemble models (SOL, ADA, DOGE, DOT)
        elif config['type'] == 'ensemble':
            models = load_ensemble_models(config['path'])
            if not models:
                print(f"  No se pudo cargar modelo {config['path']}")
                continue

            for pair in config['pairs']:
                print(f"  {pair}...", end=" ", flush=True)
                df = load_data(pair)
                if df is None:
                    print("NO DATA")
                    continue

                feat = compute_ensemble_features(df)
                common_idx = feat.index.intersection(df.index)
                df = df.loc[common_idx]
                feat = feat.loc[common_idx]

                # Vectorized predictions
                X = feat[FEATURE_COLS].values
                X_scaled = models['scaler'].transform(X)
                prob_rf = models['rf'].predict_proba(X_scaled)[:, 1]
                prob_gb = models['gb'].predict_proba(X_scaled)[:, 1]

                if 'lr' in models:
                    prob_lr = models['lr'].predict_proba(X_scaled)[:, 1]
                    votes = (prob_rf > 0.5).astype(int) + (prob_gb > 0.5).astype(int) + (prob_lr > 0.5).astype(int)
                    avg_prob = (prob_rf + prob_gb + prob_lr) / 3
                    signals = votes >= 2
                else:
                    votes = (prob_rf > 0.5).astype(int) + (prob_gb > 0.5).astype(int)
                    avg_prob = (prob_rf + prob_gb) / 2
                    signals = votes >= 2

                trades = []
                skip_until = -1

                for idx in np.where(signals)[0]:
                    if idx <= skip_until:
                        continue

                    result = simulate_trade(df, idx, config['tp'], config['sl'], 'LONG')
                    if result:
                        trade_info = {
                            'model': model_name,
                            'pair': pair,
                            'timestamp': str(feat.index[idx]),
                            'probability': round(float(avg_prob[idx]), 4),
                            'features': {col: round(float(feat[col].iloc[idx]), 4) for col in FEATURE_COLS},
                            'trade': result
                        }
                        trades.append(trade_info)
                        skip_until = idx + result['bars_held']

                wins = len([t for t in trades if t['trade']['exit_reason'] == 'TP'])
                print(f"{len(trades)} trades, WR {wins/len(trades)*100:.1f}%" if trades else "0 trades")
                all_trades.extend(trades)

    # Guardar resultados
    output_dir = Path('analysis')
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'all_models_trades.json', 'w') as f:
        json.dump(all_trades, f, indent=2)

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN POR MODELO")
    print("=" * 70)

    models_summary = {}
    for t in all_trades:
        m = t['model']
        if m not in models_summary:
            models_summary[m] = {'trades': 0, 'wins': 0}
        models_summary[m]['trades'] += 1
        if t['trade']['exit_reason'] == 'TP':
            models_summary[m]['wins'] += 1

    print(f"\n{'Modelo':<10} {'Trades':<10} {'Wins':<10} {'WR%':<10}")
    print("-" * 40)
    for m, s in sorted(models_summary.items()):
        wr = s['wins'] / s['trades'] * 100 if s['trades'] > 0 else 0
        print(f"{m:<10} {s['trades']:<10} {s['wins']:<10} {wr:.1f}%")

    print(f"\nTotal: {len(all_trades)} trades")
    print(f"Guardado en: analysis/all_models_trades.json")


if __name__ == '__main__':
    main()
