"""
Analisis de trades fallidos por modelo - VERSION OPTIMIZADA
- Prueba todos los modelos con datos reales
- Guarda trades perdedores con todas sus features
- Identifica patrones de fallo
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

# Configuracion
TIMEOUT = 15

FEATURE_COLS = ['rsi', 'macd_norm', 'adx', 'bb_pct', 'atr_pct',
                'ret_3', 'ret_5', 'ret_10', 'vol_ratio', 'trend']

# Modelos con pares representativos
MODELS_TO_TEST = {
    'ADA': {
        'path': 'strategies/ada_v14/models',
        'tp': 0.06, 'sl': 0.04,
        'test_pairs': ['ADAUSDT', 'SOLUSDT']
    },
    'DOGE': {
        'path': 'strategies/doge_v14/models',
        'tp': 0.06, 'sl': 0.04,
        'test_pairs': ['DOGEUSDT']
    },
    'DOT': {
        'path': 'strategies/dot_v14/models',
        'tp': 0.05, 'sl': 0.03,
        'test_pairs': ['DOTUSDT', 'LINKUSDT']
    },
}


def load_models(model_path):
    """Carga modelos ensemble"""
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
        print(f"Error: {e}")
        return None


def load_data(pair):
    """Carga datos reales"""
    csv_path = Path(f'data/{pair}_4h.csv')
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    return df


def compute_features(df):
    """Calcula features"""
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


def get_all_predictions(models, feat):
    """Obtiene predicciones para todos los candles de una vez"""
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

    return signals, avg_prob, prob_rf, prob_gb


def simulate_trades_fast(df, feat, signals, probs, prob_rf, prob_gb, tp_pct, sl_pct, model_name, pair):
    """Simula trades solo donde hay senales - version rapida"""
    all_trades = []
    failed_trades = []

    signal_indices = np.where(signals)[0]

    skip_until = -1
    for idx in signal_indices:
        if idx <= skip_until:
            continue
        if idx >= len(df) - TIMEOUT - 1:
            continue

        entry_price = df['close'].iloc[idx]
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)

        exit_reason = 'TIMEOUT'
        exit_price = df['close'].iloc[min(idx + TIMEOUT, len(df) - 1)]
        bars_held = TIMEOUT
        max_profit = 0
        max_dd = 0

        for i in range(1, min(TIMEOUT + 1, len(df) - idx)):
            high = df['high'].iloc[idx + i]
            low = df['low'].iloc[idx + i]

            high_pnl = (high - entry_price) / entry_price
            low_pnl = (low - entry_price) / entry_price
            max_profit = max(max_profit, high_pnl)
            max_dd = min(max_dd, low_pnl)

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

        pnl_pct = (exit_price - entry_price) / entry_price

        trade_info = {
            'model': model_name,
            'pair': pair,
            'timestamp': str(feat.index[idx]),
            'probability': round(float(probs[idx]), 4),
            'model_probs': {
                'rf': round(float(prob_rf[idx]), 4),
                'gb': round(float(prob_gb[idx]), 4)
            },
            'features': {col: round(float(feat[col].iloc[idx]), 4) for col in FEATURE_COLS},
            'trade': {
                'entry_price': round(entry_price, 6),
                'exit_price': round(exit_price, 6),
                'exit_reason': exit_reason,
                'bars_held': bars_held,
                'pnl_pct': round(pnl_pct * 100, 2),
                'max_profit_pct': round(max_profit * 100, 2),
                'max_drawdown_pct': round(max_dd * 100, 2),
            }
        }

        all_trades.append(trade_info)

        if exit_reason in ['SL', 'TIMEOUT'] and pnl_pct < 0:
            failed_trades.append(trade_info)

        skip_until = idx + bars_held

    return all_trades, failed_trades


def main():
    print("=" * 70)
    print("ANALISIS DE TRADES FALLIDOS - DATOS REALES")
    print("=" * 70)

    all_failed = []
    all_trades = []
    model_stats = {}

    for model_name, config in MODELS_TO_TEST.items():
        print(f"\n[{model_name}]")
        models = load_models(config['path'])
        if not models:
            continue

        model_trades = []
        model_failed = []

        for pair in config['test_pairs']:
            print(f"  {pair}...", end=" ", flush=True)

            df = load_data(pair)
            if df is None:
                print("NO DATA")
                continue

            feat = compute_features(df)
            common_idx = feat.index.intersection(df.index)
            df = df.loc[common_idx]
            feat = feat.loc[common_idx]

            signals, probs, prob_rf, prob_gb = get_all_predictions(models, feat)

            trades, failed = simulate_trades_fast(
                df, feat, signals, probs, prob_rf, prob_gb,
                config['tp'], config['sl'], model_name, pair
            )

            model_trades.extend(trades)
            model_failed.extend(failed)

            print(f"{len(trades)} trades, {len(failed)} failed")

        all_trades.extend(model_trades)
        all_failed.extend(model_failed)

        wins = len([t for t in model_trades if t['trade']['exit_reason'] == 'TP'])
        losses = len([t for t in model_trades if t['trade']['exit_reason'] == 'SL'])
        timeouts = len([t for t in model_trades if t['trade']['exit_reason'] == 'TIMEOUT'])

        model_stats[model_name] = {
            'total': len(model_trades),
            'wins': wins,
            'losses': losses,
            'timeouts': timeouts,
            'win_rate': round(wins / len(model_trades) * 100, 1) if model_trades else 0,
            'failed': len(model_failed)
        }

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN POR MODELO")
    print("=" * 70)
    print(f"\n{'Modelo':<10} {'Total':<8} {'Wins':<8} {'Losses':<8} {'TO':<8} {'WR%':<8} {'Failed':<8}")
    print("-" * 58)

    for model, stats in model_stats.items():
        print(f"{model:<10} {stats['total']:<8} {stats['wins']:<8} {stats['losses']:<8} "
              f"{stats['timeouts']:<8} {stats['win_rate']:<8} {stats['failed']:<8}")

    # Analisis de fallos
    if all_failed:
        print("\n" + "=" * 70)
        print("ANALISIS DE TRADES FALLIDOS")
        print("=" * 70)

        df_failed = pd.DataFrame([{
            'model': t['model'],
            'pair': t['pair'],
            'prob': t['probability'],
            'exit_reason': t['trade']['exit_reason'],
            'pnl': t['trade']['pnl_pct'],
            'max_profit': t['trade']['max_profit_pct'],
            'bars': t['trade']['bars_held'],
            **t['features']
        } for t in all_failed])

        print(f"\nTotal failed: {len(df_failed)}")

        print(f"\nPor modelo:")
        for model, count in df_failed.groupby('model').size().items():
            print(f"  {model}: {count}")

        print(f"\nPor par (top 5):")
        for pair, count in df_failed.groupby('pair').size().sort_values(ascending=False).head(5).items():
            print(f"  {pair}: {count}")

        print(f"\nPor razon:")
        for reason, count in df_failed.groupby('exit_reason').size().items():
            print(f"  {reason}: {count}")

        print(f"\nFeatures promedio en fallos:")
        for col in FEATURE_COLS:
            print(f"  {col}: {df_failed[col].mean():.4f}")

        print(f"\nProbabilidad promedio: {df_failed['prob'].mean():.4f}")
        print(f"Max profit antes de SL: {df_failed['max_profit'].mean():.2f}%")
        print(f"Barras promedio: {df_failed['bars'].mean():.1f}")

    # Guardar
    output_dir = Path('analysis')
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'failed_trades_detail.json', 'w') as f:
        json.dump(all_failed, f, indent=2)

    with open(output_dir / 'all_trades.json', 'w') as f:
        json.dump(all_trades, f, indent=2)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'model_stats': model_stats,
        'total_trades': len(all_trades),
        'total_failed': len(all_failed)
    }
    with open(output_dir / 'failure_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n" + "=" * 70)
    print(f"GUARDADO EN analysis/:")
    print(f"  failed_trades_detail.json - {len(all_failed)} trades fallidos")
    print(f"  all_trades.json - {len(all_trades)} trades totales")
    print("=" * 70)


if __name__ == '__main__':
    main()
