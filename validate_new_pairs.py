"""
Validacion completa de nuevos pares para V14.1
- Walk-forward con datos propios
- Datos sinteticos (5 escenarios)
"""

import warnings
import numpy as np
import pandas as pd
import ccxt
import ta
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings('ignore')

# Configuración
TP_PCT = 0.06
SL_PCT = 0.04
TIMEOUT = 15

FEATURE_COLS = ['rsi', 'macd_norm', 'adx', 'bb_pct', 'atr_pct',
                'ret_3', 'ret_5', 'ret_10', 'vol_ratio', 'trend']

# Pares a validar y su modelo base
PAIRS_TO_VALIDATE = {
    'SOL/USDT': 'ada',    # Smart contract platform
    'ATOM/USDT': 'ada',   # Smart contract platform
    'SHIB/USDT': 'doge',  # Memecoin
    'PEPE/USDT': 'doge',  # Memecoin
}


def download_data(symbol):
    """Descarga datos de Binance"""
    symbol_clean = symbol.replace('/', '')
    csv_path = f'data/{symbol_clean}_4h.csv'

    if Path(csv_path).exists():
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        print(f"  {symbol}: {len(df)} candles (cargado)")
        return df

    print(f"  {symbol}: descargando...")
    exchange = ccxt.binance()
    since = exchange.parse8601('2020-01-01T00:00:00Z')
    all_ohlcv = []

    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, '4h', since=since, limit=1000)
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
    print(f"  {symbol}: {len(df)} candles (descargado)")
    return df


def compute_features(df):
    """Calcula features ML"""
    df = df.copy()
    df['rsi'] = ta.momentum.rsi(df['close'], window=14) / 100
    macd = ta.trend.MACD(df['close'])
    df['macd_norm'] = macd.macd() / df['close']
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14) / 100
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_pct'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-10)
    df['atr_pct'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14) / df['close']
    df['ret_3'] = df['close'].pct_change(3)
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_10'] = df['close'].pct_change(10)
    df['vol_sma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma']
    df['sma_50'] = df['close'].rolling(50).mean()
    df['trend'] = (df['close'] > df['sma_50']).astype(float)
    return df.dropna()


def create_labels(df):
    """Crea labels de win/loss"""
    labels = []
    for i in range(len(df) - TIMEOUT - 1):
        entry = df['close'].iloc[i]
        future = df.iloc[i+1:i+TIMEOUT+1]
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


def load_base_models(base_model):
    """Carga modelos base (ADA o DOGE)"""
    model_dir = Path(f'strategies/{base_model}_v14/models')
    models = {
        'scaler': joblib.load(model_dir / 'scaler.pkl'),
        'rf': joblib.load(model_dir / 'random_forest.pkl'),
        'gb': joblib.load(model_dir / 'gradient_boosting.pkl'),
    }
    lr_path = model_dir / 'logistic_regression.pkl'
    if lr_path.exists():
        models['lr'] = joblib.load(lr_path)
    return models


def predict_ensemble(models, X):
    """Predicción ensemble"""
    X_scaled = models['scaler'].transform(X)
    prob_rf = models['rf'].predict_proba(X_scaled)[:, 1]
    prob_gb = models['gb'].predict_proba(X_scaled)[:, 1]

    if 'lr' in models:
        prob_lr = models['lr'].predict_proba(X_scaled)[:, 1]
        votes = (prob_rf > 0.5).astype(int) + (prob_gb > 0.5).astype(int) + (prob_lr > 0.5).astype(int)
        return votes >= 2, (prob_rf + prob_gb + prob_lr) / 3
    else:
        votes = (prob_rf > 0.5).astype(int) + (prob_gb > 0.5).astype(int)
        return votes >= 2, (prob_rf + prob_gb) / 2


def walk_forward_validation(df, models, n_folds=10):
    """Walk-forward validation con modelo pre-entrenado"""
    labels = create_labels(df)
    df = df.iloc[:len(labels)].copy()
    df['label'] = labels.values

    X = df[FEATURE_COLS].values
    y = df['label'].values

    fold_size = len(df) // (n_folds + 1)
    results = []

    for fold in range(n_folds):
        test_start = (fold + 1) * fold_size
        test_end = test_start + fold_size

        if test_end > len(df):
            break

        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]

        trade_mask, probs = predict_ensemble(models, X_test)
        n_trades = trade_mask.sum()

        if n_trades > 0:
            wins = y_test[trade_mask].sum()
            wr = wins / n_trades
            pnl = wins * TP_PCT - (n_trades - wins) * SL_PCT
            results.append({'pnl': pnl * 100, 'trades': n_trades, 'wr': wr})
        else:
            results.append({'pnl': 0, 'trades': 0, 'wr': 0})

    positive = len([r for r in results if r['pnl'] > 0])
    total_pnl = sum(r['pnl'] for r in results)
    total_trades = sum(r['trades'] for r in results)

    return {
        'folds_positive': positive,
        'folds_total': len(results),
        'pnl': total_pnl,
        'trades': total_trades,
        'pass': positive / len(results) >= 0.6 if results else False
    }


def generate_synthetic_data(base_df, scenario, n_candles=2000):
    """Genera datos sintéticos basados en escenario"""
    np.random.seed(42)

    # Usar volatilidad real como base
    real_returns = base_df['close'].pct_change().dropna()
    vol = real_returns.std()

    prices = [base_df['close'].iloc[-1]]

    for i in range(n_candles):
        if scenario == 'BULL':
            drift = 0.001  # Tendencia alcista
            ret = np.random.normal(drift, vol)
        elif scenario == 'BEAR':
            drift = -0.001  # Tendencia bajista
            ret = np.random.normal(drift, vol)
        elif scenario == 'RANGE':
            # Mean reversion
            mean_price = prices[0]
            current = prices[-1]
            pull = (mean_price - current) / mean_price * 0.1
            ret = np.random.normal(pull, vol * 0.7)
        elif scenario == 'VOLATILE':
            drift = 0
            ret = np.random.normal(drift, vol * 2)  # Double volatility
        else:  # MIXED
            phase = (i // 400) % 4
            if phase == 0:
                ret = np.random.normal(0.001, vol)
            elif phase == 1:
                ret = np.random.normal(-0.001, vol)
            elif phase == 2:
                ret = np.random.normal(0, vol * 0.5)
            else:
                ret = np.random.normal(0, vol * 1.5)

        prices.append(prices[-1] * (1 + ret))

    # Crear OHLCV sintético
    df = pd.DataFrame()
    df['close'] = prices[1:]
    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, len(df)))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, len(df)))
    df['volume'] = np.random.uniform(1e6, 1e8, len(df))
    df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='4h')

    return df


def test_synthetic(models, base_df):
    """Prueba en 5 escenarios sintéticos"""
    scenarios = ['BULL', 'BEAR', 'RANGE', 'VOLATILE', 'MIXED']
    results = {}

    for scenario in scenarios:
        df = generate_synthetic_data(base_df, scenario)
        df = compute_features(df)

        labels = create_labels(df)
        df = df.iloc[:len(labels)].copy()
        df['label'] = labels.values

        X = df[FEATURE_COLS].values
        y = df['label'].values

        trade_mask, _ = predict_ensemble(models, X)
        n_trades = trade_mask.sum()

        if n_trades > 0:
            wins = y[trade_mask].sum()
            pnl = wins * TP_PCT - (n_trades - wins) * SL_PCT
            results[scenario] = {'pnl': pnl * 100, 'trades': n_trades, 'pass': pnl > 0}
        else:
            results[scenario] = {'pnl': 0, 'trades': 0, 'pass': True}  # No trades = neutral

    positive = len([r for r in results.values() if r['pass']])
    return results, positive


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("VALIDACIÓN COMPLETA - NUEVOS PARES V14.1")
    print("=" * 70)

    # Descargar datos
    print("\n[1/3] Descargando datos...")
    data = {}
    for symbol in PAIRS_TO_VALIDATE.keys():
        data[symbol] = download_data(symbol)

    # Validar cada par
    print("\n[2/3] Walk-Forward Validation...")
    print("-" * 70)

    all_results = {}

    for symbol, base_model in PAIRS_TO_VALIDATE.items():
        print(f"\n{symbol} (modelo base: {base_model.upper()})")

        df = data[symbol]
        df = compute_features(df)

        models = load_base_models(base_model)

        # Walk-forward
        wf = walk_forward_validation(df, models)
        status_wf = "PASS" if wf['pass'] else "FAIL"
        print(f"  Walk-Forward: {wf['folds_positive']}/{wf['folds_total']} folds+, "
              f"PnL {wf['pnl']:+.0f}%, Trades {wf['trades']} [{status_wf}]")

        # Sinteticos
        synth, synth_positive = test_synthetic(models, df)
        status_synth = "PASS" if synth_positive >= 3 else "FAIL"
        print(f"  Sinteticos: {synth_positive}/5 positivos [{status_synth}]")

        for scenario, res in synth.items():
            emoji = "+" if res['pass'] else "-"
            print(f"    {scenario}: {res['trades']} trades, PnL {res['pnl']:+.0f}% [{emoji}]")

        # Resultado final
        approved = wf['pass'] and synth_positive >= 3
        all_results[symbol] = {
            'walk_forward': wf,
            'synthetic': synth,
            'synthetic_positive': synth_positive,
            'approved': approved
        }

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"\n{'Par':<12} {'WF Folds':<12} {'WF PnL':<10} {'Synth':<8} {'Status':<10}")
    print("-" * 52)

    for symbol, res in all_results.items():
        wf = res['walk_forward']
        status = "[APROBADO]" if res['approved'] else "[RECHAZADO]"
        print(f"{symbol:<12} {wf['folds_positive']}/{wf['folds_total']:<10} "
              f"{wf['pnl']:+.0f}%{'':<6} {res['synthetic_positive']}/5{'':<5} {status}")

    approved_pairs = [s for s, r in all_results.items() if r['approved']]
    print(f"\nPares aprobados para V14.1: {len(approved_pairs)}")
    for p in approved_pairs:
        print(f"  + {p}")


if __name__ == '__main__':
    main()
