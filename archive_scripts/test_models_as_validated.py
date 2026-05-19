"""
TEST DE MODELOS TAL CUAL FUERON VALIDADOS
==========================================
Objetivo: Probar cada modelo EXACTAMENTE como fue entrenado y validado,
sin filtros adicionales ni modificaciones.

Modelos a probar:
1. BTC V14 - Setups tecnicos (validation.py)
2. ETH V14 - Setups tecnicos
3. ADA V14 - Ensemble voting (modelo base)
4. Cross-pairs ADA: ATOM, AVAX, POL (usando modelo ADA)
5. DOGE V14 - Ensemble voting
6. DOT V14 - Ensemble voting

Metodologia:
- Usar los mismos TP/SL de la validacion original
- Usar la misma logica de senales
- NO aplicar filtros adicionales
- Reportar WR y PnL por par
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

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACION - TAL CUAL VALIDACION ORIGINAL
# =============================================================================

DATA_DIR = Path('data')
TIMEOUT = 50  # Mismo que validation.py

# Configuracion por modelo (de los archivos de validacion originales)
# TODOS los 16 pares del paper trading
MODELS_CONFIG = {
    # === BTC - validation.py usa TP 3%, SL 1.5% ===
    'BTC': {
        'type': 'btc_setups',
        'tp': 0.03,
        'sl': 0.015,
        'pairs': ['BTCUSDT'],
    },
    # === ETH - mismas reglas que BTC (cross-validation) ===
    'ETH': {
        'type': 'btc_setups',
        'tp': 0.03,
        'sl': 0.015,
        'pairs': ['ETHUSDT'],
    },
    # === DOGE nativo + memecoins ===
    'DOGE': {
        'type': 'ensemble',
        'path': 'strategies/doge_v14/models',
        'tp': 0.06,
        'sl': 0.04,
        'pairs': ['DOGEUSDT'],
    },
    'DOGE_MEME': {
        'type': 'ensemble',
        'path': 'strategies/doge_v14/models',
        'tp': 0.06,
        'sl': 0.04,
        'pairs': ['1000SHIBUSDT', '1000PEPEUSDT', '1000FLOKIUSDT'],
    },
    # === ADA nativo + smart contracts ===
    'ADA': {
        'type': 'ensemble',
        'path': 'strategies/ada_v14/models',
        'tp': 0.06,
        'sl': 0.04,
        'pairs': ['ADAUSDT'],
    },
    'ADA_CROSS': {
        'type': 'ensemble',
        'path': 'strategies/ada_v14/models',
        'tp': 0.06,
        'sl': 0.04,
        'pairs': ['ATOMUSDT', 'AVAXUSDT', 'POLUSDT'],
    },
    # === DOT nativo + infraestructura ===
    'DOT': {
        'type': 'ensemble',
        'path': 'strategies/dot_v14/models',
        'tp': 0.05,
        'sl': 0.03,
        'pairs': ['DOTUSDT'],
    },
    'DOT_INFRA': {
        'type': 'ensemble',
        'path': 'strategies/dot_v14/models',
        'tp': 0.05,
        'sl': 0.03,
        'pairs': ['LINKUSDT', 'ALGOUSDT', 'FILUSDT', 'NEARUSDT'],
    },
    # === SOL dedicado ===
    'SOL': {
        'type': 'ensemble',
        'path': 'strategies/sol_v14/models',
        'tp': 0.06,
        'sl': 0.04,
        'pairs': ['SOLUSDT'],
    },
}

# Features para ensemble
ENSEMBLE_FEATURES = ['rsi', 'macd_norm', 'adx', 'bb_pct', 'atr_pct',
                     'ret_3', 'ret_5', 'ret_10', 'vol_ratio', 'trend']

# =============================================================================
# BTC/ETH SETUPS (de validation.py)
# =============================================================================

SETUP_DEFINITIONS = {
    'PULLBACK_UPTREND': {
        'direction': 'long',
        'conditions': lambda r: r.get('rsi14', 50) < 40 and r.get('bb_pct', 0.5) < 0.3 and r.get('ema200_dist', 0) > 0,
    },
    'OVERSOLD_EXTREME': {
        'direction': 'long',
        'conditions': lambda r: r.get('rsi14', 50) < 25 and r.get('bb_pct', 0.5) < 0.2,
    },
    'SUPPORT_BOUNCE': {
        'direction': 'long',
        'conditions': lambda r: r.get('range_pos', 0.5) < 0.15 and r.get('rsi14', 50) < 35,
    },
    'CAPITULATION': {
        'direction': 'long',
        'conditions': lambda r: r.get('consec_down', 0) >= 4 and r.get('rsi14', 50) < 30 and r.get('vol_ratio', 1) > 1.5,
    },
    'RALLY_DOWNTREND': {
        'direction': 'short',
        'conditions': lambda r: r.get('rsi14', 50) > 60 and r.get('bb_pct', 0.5) > 0.7 and r.get('ema200_dist', 0) < 0,
    },
    'OVERBOUGHT_EXTREME': {
        'direction': 'short',
        'conditions': lambda r: r.get('rsi14', 50) > 75 and r.get('bb_pct', 0.5) > 0.8,
    },
    'RESISTANCE_REJECTION': {
        'direction': 'short',
        'conditions': lambda r: r.get('range_pos', 0.5) > 0.85 and r.get('rsi14', 50) > 65,
    },
    'EXHAUSTION': {
        'direction': 'short',
        'conditions': lambda r: r.get('consec_up', 0) >= 4 and r.get('rsi14', 50) > 70 and r.get('vol_ratio', 1) > 1.5,
    },
}

def compute_btc_features(df):
    """Features para BTC/ETH setups (de validation.py)"""
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    feat = pd.DataFrame(index=df.index)

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

    feat['atr_pct'] = pta.atr(h, l, c, length=14) / c * 100
    bb = pta.bbands(c, length=20)
    if bb is not None:
        feat['bb_upper'] = bb.iloc[:, 2]
        feat['bb_lower'] = bb.iloc[:, 0]
        feat['bb_mid'] = bb.iloc[:, 1]
        feat['bb_width'] = (feat['bb_upper'] - feat['bb_lower']) / feat['bb_mid'] * 100
        feat['bb_pct'] = (c - feat['bb_lower']) / (feat['bb_upper'] - feat['bb_lower'])

    feat['rsi14'] = pta.rsi(c, length=14)
    feat['rsi7'] = pta.rsi(c, length=7)
    stoch = pta.stoch(h, l, c, k=14, d=3)
    if stoch is not None:
        feat['stoch_k'] = stoch.iloc[:, 0]

    feat['ret_5'] = c.pct_change(5) * 100
    feat['ret_20'] = c.pct_change(20) * 100

    feat['vol_ratio'] = v / v.rolling(20).mean()
    feat['high_20'] = h.rolling(20).max()
    feat['low_20'] = l.rolling(20).min()
    feat['range_pos'] = (c - feat['low_20']) / (feat['high_20'] - feat['low_20'])
    feat['consec_down'] = (c < c.shift(1)).rolling(10).sum()
    feat['consec_up'] = (c > c.shift(1)).rolling(10).sum()

    return feat

def detect_btc_setups(feat):
    """Detecta setups BTC/ETH"""
    setups = []
    for idx in feat.index:
        row = feat.loc[idx].to_dict()
        if pd.isna(row.get('rsi14')) or pd.isna(row.get('bb_pct')):
            continue
        for setup_name, setup_def in SETUP_DEFINITIONS.items():
            try:
                if setup_def['conditions'](row):
                    setups.append({
                        'idx': idx,
                        'setup': setup_name,
                        'direction': setup_def['direction']
                    })
            except:
                continue
    return pd.DataFrame(setups) if setups else None

# =============================================================================
# ENSEMBLE (DOGE/ADA/DOT)
# =============================================================================

def compute_ensemble_features(df):
    """Features para ensemble"""
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

def load_ensemble_models(model_path):
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
        print(f"Error loading {model_path}: {e}")
        return None

def predict_ensemble(models, X):
    """Prediccion ensemble - 2/3 majority voting"""
    X_scaled = models['scaler'].transform(X)
    prob_rf = models['rf'].predict_proba(X_scaled)[:, 1]
    prob_gb = models['gb'].predict_proba(X_scaled)[:, 1]

    if 'lr' in models:
        prob_lr = models['lr'].predict_proba(X_scaled)[:, 1]
        votes = (prob_rf > 0.5).astype(int) + (prob_gb > 0.5).astype(int) + (prob_lr > 0.5).astype(int)
        signals = votes >= 2
    else:
        votes = (prob_rf > 0.5).astype(int) + (prob_gb > 0.5).astype(int)
        signals = votes >= 2

    return signals

# =============================================================================
# BACKTESTING
# =============================================================================

def load_data(pair):
    """Carga datos de un par"""
    csv_path = DATA_DIR / f'{pair}_4h.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    return df

def simulate_trade(df, idx, tp, sl, direction='long'):
    """Simula un trade"""
    if idx >= len(df) - TIMEOUT - 1:
        return None

    entry_price = df['close'].iloc[idx]

    for i in range(1, min(TIMEOUT + 1, len(df) - idx)):
        future_price = df['close'].iloc[idx + i]

        if direction == 'long':
            pnl = (future_price - entry_price) / entry_price
        else:
            pnl = (entry_price - future_price) / entry_price

        if pnl >= tp:
            return {'outcome': 1, 'pnl': pnl, 'bars': i}
        elif pnl <= -sl:
            return {'outcome': 0, 'pnl': -sl, 'bars': i}

    return None  # Timeout - no contar

def backtest_btc_setups(pair, tp, sl):
    """Backtest de setups BTC/ETH"""
    df = load_data(pair)
    if df is None:
        return None

    feat = compute_btc_features(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    setups_df = detect_btc_setups(feat)
    if setups_df is None:
        return None

    results = []
    for _, row in setups_df.iterrows():
        idx = df.index.get_loc(row['idx'])
        result = simulate_trade(df, idx, tp, sl, row['direction'])
        if result:
            result['setup'] = row['setup']
            result['direction'] = row['direction']
            results.append(result)

    return pd.DataFrame(results) if results else None

def backtest_ensemble(pair, model_path, tp, sl):
    """Backtest de ensemble"""
    df = load_data(pair)
    if df is None:
        return None

    models = load_ensemble_models(model_path)
    if models is None:
        return None

    feat = compute_ensemble_features(df)
    common_idx = feat.index.intersection(df.index)
    df_aligned = df.loc[common_idx]
    feat = feat.loc[common_idx]

    X = feat[ENSEMBLE_FEATURES].values
    signals = predict_ensemble(models, X)

    results = []
    skip_until = -1

    for i, signal in enumerate(signals):
        if i <= skip_until:
            continue
        if not signal:
            continue

        result = simulate_trade(df_aligned, i, tp, sl, 'long')
        if result:
            results.append(result)
            skip_until = i + result['bars']

    return pd.DataFrame(results) if results else None

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("TEST DE MODELOS TAL CUAL FUERON VALIDADOS")
    print("=" * 80)
    print("\nMetodologia:")
    print("- Usar EXACTAMENTE los mismos parametros de validacion")
    print("- NO aplicar filtros adicionales")
    print("- Reportar resultados por par")
    print()

    all_results = []

    for model_name, config in MODELS_CONFIG.items():
        print(f"\n{'='*60}")
        print(f"MODELO: {model_name}")
        print(f"Tipo: {config['type']}")
        print(f"TP/SL: {config['tp']*100:.0f}% / {config['sl']*100:.1f}%")
        print(f"{'='*60}")

        for pair in config['pairs']:
            print(f"\n  {pair}...", end=" ", flush=True)

            if config['type'] == 'btc_setups':
                results_df = backtest_btc_setups(pair, config['tp'], config['sl'])
            else:
                results_df = backtest_ensemble(pair, config['path'], config['tp'], config['sl'])

            if results_df is None or len(results_df) == 0:
                print("NO DATA o sin trades")
                continue

            n = len(results_df)
            wins = results_df['outcome'].sum()
            wr = wins / n * 100
            pnl = results_df['pnl'].sum() * 100

            status = "OK" if pnl > 0 else "PROBLEMA"
            print(f"{n} trades, WR {wr:.1f}%, PnL {pnl:+.1f}% [{status}]")

            all_results.append({
                'model': model_name,
                'pair': pair,
                'trades': n,
                'wins': int(wins),
                'wr': round(wr, 1),
                'pnl': round(pnl, 1),
                'tp': config['tp'],
                'sl': config['sl'],
                'status': status
            })

            # Detalle por setup (solo para BTC/ETH)
            if config['type'] == 'btc_setups' and 'setup' in results_df.columns:
                print(f"\n    Por Setup:")
                for setup in results_df['setup'].unique():
                    subset = results_df[results_df['setup'] == setup]
                    sub_wr = subset['outcome'].mean() * 100
                    sub_pnl = subset['pnl'].sum() * 100
                    sub_status = "OK" if sub_pnl > 0 else "BAD"
                    print(f"      {setup:<25}: {len(subset):>4} trades, WR {sub_wr:>5.1f}%, PnL {sub_pnl:>+8.1f}% [{sub_status}]")

    # Resumen final
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)

    print(f"\n{'Modelo':<12} {'Par':<12} {'Trades':<8} {'WR%':<8} {'PnL%':<10} {'Status':<10}")
    print("-" * 60)

    for r in all_results:
        print(f"{r['model']:<12} {r['pair']:<12} {r['trades']:<8} {r['wr']:<8} {r['pnl']:+<10.1f} {r['status']:<10}")

    # Identificar problemas
    problems = [r for r in all_results if r['status'] == 'PROBLEMA']
    if problems:
        print(f"\n{'='*80}")
        print("PARES CON PROBLEMAS (PnL negativo)")
        print("=" * 80)
        for p in problems:
            print(f"  {p['model']} - {p['pair']}: WR {p['wr']}%, PnL {p['pnl']}%")

    # Guardar resultados
    output = {
        'timestamp': datetime.now().isoformat(),
        'methodology': 'Test as originally validated - no additional filters',
        'results': all_results
    }

    output_path = Path('analysis/validation_test_results.json')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResultados guardados en: {output_path}")


if __name__ == '__main__':
    main()
