"""
ML Export V13.04 - Low-Overfitting Ridge Models
================================================
Exporta modelos Ridge(alpha=100) con 7 features minimas.
Validado con walk-forward y testeado en bear market (Ene-Feb 2026).

Caracteristicas:
- Ridge regression con alta regularizacion (alpha=100)
- Solo 7 features: ret_1, ret_5, ret_20, vol20, rsi14, ema21_d, vr
- LONG_ONLY (shorts no funcionan con este modelo)
- Sin overfitting verificado con walk-forward 5 ventanas

Pares incluidos (por performance en walk-forward + bear market):
- DOGE: 90/100 WF, 81% WR en bear
- ADA:  90/100 WF, 70% WR en bear
- DOT:  N/A WF, 77% WR en bear
- XRP:  85/100 WF, 54% WR en bear
- BTC:  100/100 WF, 50% WR en bear (peor en bear, mejor en general)

Uso: python ml_export_v1304.py
"""

import json
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)

# Pares para V13.04 (ordenados por confianza)
V1304_PAIRS = {
    'DOGE': {'direction': 'LONG_ONLY', 'tp_pct': 0.02, 'sl_pct': 0.02, 'conv_min': 1.0, 'tier': 1},
    'ADA':  {'direction': 'LONG_ONLY', 'tp_pct': 0.02, 'sl_pct': 0.02, 'conv_min': 1.0, 'tier': 1},
    'DOT':  {'direction': 'LONG_ONLY', 'tp_pct': 0.02, 'sl_pct': 0.02, 'conv_min': 1.0, 'tier': 1},
    'XRP':  {'direction': 'LONG_ONLY', 'tp_pct': 0.02, 'sl_pct': 0.02, 'conv_min': 1.0, 'tier': 2},
    'BTC':  {'direction': 'LONG_ONLY', 'tp_pct': 0.02, 'sl_pct': 0.02, 'conv_min': 1.0, 'tier': 2},
}

# Pares excluidos (para futuro re-entrenamiento)
EXCLUDED_PAIRS = {
    'ETH':  {'reason': 'Negative PnL in walk-forward', 'potential': 'LONG_ONLY'},
    'BNB':  {'reason': 'Only 1/5 folds profitable', 'potential': 'LONG_ONLY'},
    'LINK': {'reason': 'Negative PnL despite high consistency', 'potential': 'LONG_ONLY'},
    'NEAR': {'reason': 'Inconsistent correlation (65/100)', 'potential': 'LONG_ONLY'},
    'AVAX': {'reason': 'Not enough data for full validation', 'potential': 'LONG_ONLY'},
}

FEATURE_COLS = ['ret_1', 'ret_5', 'ret_20', 'vol20', 'rsi14', 'ema21_d', 'vr']


def compute_features_minimal(df):
    """7 features - lowest overfitting."""
    feat = pd.DataFrame(index=df.index)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    feat['ret_1'] = c.pct_change(1)
    feat['ret_5'] = c.pct_change(5)
    feat['ret_20'] = c.pct_change(20)
    feat['vol20'] = c.pct_change().rolling(20).std()
    feat['rsi14'] = ta.rsi(c, length=14)
    ema21 = ta.ema(c, length=21)
    feat['ema21_d'] = (c - ema21) / ema21 * 100
    feat['vr'] = v / v.rolling(20).mean()
    return feat


def load_data(pair):
    """Load pair data from parquet."""
    patterns = [
        f'{pair}_USDT_4h_full.parquet',
        f'{pair}_USDT_4h_backtest.parquet',
        f'{pair}_USDT_4h_history.parquet',
    ]
    for pattern in patterns:
        file_path = DATA_DIR / pattern
        if file_path.exists():
            df = pd.read_parquet(file_path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            return df.sort_index()
    return None


def train_and_export_pair(pair, config):
    """Train Ridge model for a single pair and export."""
    df = load_data(pair)
    if df is None:
        print(f'  {pair}: ERROR - No data found')
        return None

    # Compute features
    feat = compute_features_minimal(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # Target: 5-period forward return
    target = df['close'].pct_change(5).shift(-5)

    # Get valid indices
    valid_idx = feat.dropna().index.intersection(target.dropna().index)
    X = feat.loc[valid_idx].iloc[:-5]
    y = target.loc[valid_idx].iloc[:-5]

    if len(X) < 500:
        print(f'  {pair}: ERROR - Insufficient data ({len(X)} samples)')
        return None

    # Train on ALL data (no split - we already validated with walk-forward)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=100.0)
    model.fit(X_scaled, y)

    # Compute pred_std for conviction calculation
    preds = model.predict(X_scaled)
    pred_std = float(np.std(preds))

    # Correlation (train)
    corr = float(np.corrcoef(preds, y)[0, 1])

    # Export model
    model_path = MODELS_DIR / f'v1304_{pair}.pkl'
    joblib.dump(model, model_path)

    # Export scaler
    scaler_path = MODELS_DIR / f'v1304_{pair}_scaler.pkl'
    joblib.dump(scaler, scaler_path)

    print(f'  {pair}: {len(X):,} samples | corr={corr:.4f} | pred_std={pred_std:.6f}')

    return {
        'pair': pair,
        'n_samples': len(X),
        'date_start': str(X.index.min().date()),
        'date_end': str(X.index.max().date()),
        'corr': corr,
        'pred_std': pred_std,
        'config': config,
    }


def main():
    print('=' * 70)
    print('EXPORT V13.04 - LOW-OVERFITTING RIDGE MODELS')
    print('=' * 70)
    print(f'\nModel: Ridge(alpha=100) with 7 minimal features')
    print(f'Direction: LONG_ONLY (shorts disabled)')
    print(f'Validation: Walk-forward 5 windows + Bear market test (Jan-Feb 2026)')

    print(f'\n[1/3] Training models for {len(V1304_PAIRS)} pairs...')

    results = {}
    for pair, config in V1304_PAIRS.items():
        result = train_and_export_pair(pair, config)
        if result:
            results[pair] = result

    print(f'\n[2/3] Saving V13.04 metadata...')

    # Build metadata
    meta = {
        'version': 'v13.04',
        'trained_at': datetime.utcnow().isoformat(),
        'model_type': 'Ridge',
        'model_params': {'alpha': 100.0},
        'feature_cols': FEATURE_COLS,
        'direction': 'LONG_ONLY',
        'pairs': {},
        'excluded_pairs': EXCLUDED_PAIRS,
        'notes': [
            'Low-overfitting model validated with walk-forward',
            'Tested in bear market (Jan-Feb 2026) with positive results',
            'LONG_ONLY - model does not predict shorts accurately',
            'Re-train monthly for best results',
        ],
    }

    for pair, result in results.items():
        meta['pairs'][pair] = {
            'n_samples': result['n_samples'],
            'date_range': f"{result['date_start']} to {result['date_end']}",
            'corr': result['corr'],
            'pred_std': result['pred_std'],
            'direction': result['config']['direction'],
            'tp_pct': result['config']['tp_pct'],
            'sl_pct': result['config']['sl_pct'],
            'conv_min': result['config']['conv_min'],
            'tier': result['config']['tier'],
        }

    meta_path = MODELS_DIR / 'v1304_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  Metadata saved: {meta_path}')

    print(f'\n[3/3] Summary')
    print('=' * 70)
    print(f"\n{'Pair':<6} {'Samples':<10} {'Corr':<8} {'Tier':<6} {'Direction':<12}")
    print('-' * 50)

    for pair, result in sorted(results.items(), key=lambda x: x[1]['config']['tier']):
        tier = result['config']['tier']
        direction = result['config']['direction']
        print(f"{pair:<6} {result['n_samples']:<10,} {result['corr']:<8.4f} {tier:<6} {direction:<12}")

    print('-' * 50)
    print(f"\nModels exported to: {MODELS_DIR.absolute()}")
    print(f"Files created:")
    for pair in results:
        print(f"  - v1304_{pair}.pkl (model)")
        print(f"  - v1304_{pair}_scaler.pkl (scaler)")
    print(f"  - v1304_meta.json (metadata)")

    print(f'\n' + '=' * 70)
    print('EXCLUDED PAIRS (for future re-training):')
    print('=' * 70)
    for pair, info in EXCLUDED_PAIRS.items():
        print(f"  {pair}: {info['reason']}")

    print(f'\n' + '=' * 70)
    print('NEXT STEPS:')
    print('=' * 70)
    print("""
1. Integrar V13.04 en src/ml_strategy.py
2. Agregar flag ML_V1304_ENABLED en config/settings.py
3. Implementar logica LONG_ONLY en el bot
4. Correr en paper trading antes de produccion

Para re-entrenar pares excluidos:
  - Modificar EXCLUDED_PAIRS -> V1304_PAIRS en este script
  - Ejecutar: python ml_export_v1304.py
""")


if __name__ == '__main__':
    main()
