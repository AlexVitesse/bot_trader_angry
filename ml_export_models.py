"""
ML Export Models - Entrena y exporta modelos para produccion
============================================================
Entrena LightGBM en TODOS los datos disponibles y guarda modelos + metadata.
Ejecutar antes del bot live, y re-ejecutar mensualmente para reentrenamiento.

Uso: poetry run python ml_export_models.py
"""

import json
import time
import numpy as np
import pandas as pd
import pandas_ta as ta
import lightgbm as lgb
import optuna
import joblib
import ccxt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)

PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT',
    'NEAR/USDT',
]

MIN_CANDLES = 3000  # Min ~2 anos de 4h
N_TRIALS = 40       # Optuna trials
HORIZON = 5         # Predecir 5 velas adelante (20h)


def download_pair(symbol, timeframe='4h', since='2020-01-01'):
    """Descarga o cachea datos de un par."""
    safe = symbol.replace('/', '_')
    cache = DATA_DIR / f'{safe}_{timeframe}_history.parquet'
    if cache.exists() and (time.time() - cache.stat().st_mtime) / 3600 < 24:
        return pd.read_parquet(cache)
    exchange = ccxt.binance({'enableRateLimit': True})
    since_ts = int(datetime.strptime(since, '%Y-%m-%d').timestamp() * 1000)
    rows = []
    while True:
        try:
            c = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=1000)
        except Exception as e:
            print(f'  Error {symbol}: {e}'); time.sleep(5); continue
        if not c: break
        rows.extend(c)
        since_ts = c[-1][0] + 1
        if len(c) < 1000: break
        time.sleep(0.15)
    if not rows: return None
    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    df.to_parquet(cache)
    return df


def compute_features(df):
    """Mismas features que V7 - EXACTAS."""
    feat = pd.DataFrame(index=df.index)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    for p in [1, 3, 5, 10, 20]:
        feat[f'ret_{p}'] = c.pct_change(p)

    feat['atr14'] = ta.atr(h, l, c, length=14)
    feat['atr_r'] = feat['atr14'] / feat['atr14'].rolling(50).mean()
    feat['vol5'] = c.pct_change().rolling(5).std()
    feat['vol20'] = c.pct_change().rolling(20).std()

    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)
    sr = ta.stochrsi(c, length=14, rsi_length=14, k=3, d=3)
    if sr is not None: feat['srsi_k'] = sr.iloc[:, 0]
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None: feat['macd_h'] = macd.iloc[:, 1]
    feat['roc5'] = ta.roc(c, length=5)
    feat['roc20'] = ta.roc(c, length=20)

    for el in [8, 21, 55, 100, 200]:
        e = ta.ema(c, length=el)
        feat[f'ema{el}_d'] = (c - e) / e * 100
    feat['ema8_sl'] = ta.ema(c, length=8).pct_change(3) * 100
    feat['ema55_sl'] = ta.ema(c, length=55).pct_change(5) * 100

    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        bw = bb.iloc[:, 2] - bb.iloc[:, 0]
        feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw
        feat['bb_w'] = bw / bb.iloc[:, 1] * 100

    feat['vr'] = v / v.rolling(20).mean()
    feat['spr'] = (h - l) / c * 100
    feat['body'] = abs(c - df['open']) / (h - l + 1e-10)

    ax = ta.adx(h, l, c, length=14)
    if ax is not None:
        feat['adx'] = ax.iloc[:, 0]
        feat['dip'] = ax.iloc[:, 1]
        feat['dim'] = ax.iloc[:, 2]

    hr = df.index.hour; dw = df.index.dayofweek
    feat['h_s'] = np.sin(2*np.pi*hr/24)
    feat['h_c'] = np.cos(2*np.pi*hr/24)
    feat['d_s'] = np.sin(2*np.pi*dw/7)
    feat['d_c'] = np.cos(2*np.pi*dw/7)

    return feat


def train_and_export(pair, df, n_trials=N_TRIALS, horizon=HORIZON):
    """Entrena modelo en todos los datos y exporta."""
    feat = compute_features(df)
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
    fcols = list(feat.columns)

    tgt = f'fwd_{horizon}'
    feat[tgt] = (df['close'].shift(-horizon) - df['close']) / df['close']
    ds = feat.dropna()

    if len(ds) < MIN_CANDLES:
        print(f'  {pair:<12} OMITIDO ({len(ds)} velas, min {MIN_CANDLES})')
        return None

    X = ds[fcols].fillna(0)
    y = ds[tgt].clip(ds[tgt].quantile(0.01), ds[tgt].quantile(0.99))

    # Optuna: usar ultimo 15% como validacion
    sp = int(len(X) * 0.85)
    Xt, Xv = X.iloc[:sp], X.iloc[sp:]
    yt, yv = y.iloc[:sp], y.iloc[sp:]

    def obj(trial):
        p = {
            'n_estimators': trial.suggest_int('ne', 100, 500),
            'max_depth': trial.suggest_int('md', 3, 8),
            'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('ss', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('cs', 0.4, 1.0),
            'min_child_samples': trial.suggest_int('mc', 10, 80),
            'reg_alpha': trial.suggest_float('ra', 1e-6, 5.0, log=True),
            'reg_lambda': trial.suggest_float('rl', 1e-6, 5.0, log=True),
            'num_leaves': trial.suggest_int('nl', 15, 80),
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        }
        m = lgb.LGBMRegressor(**p)
        m.fit(Xt, yt, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        pr = m.predict(Xv)
        c = np.corrcoef(pr, yv.values)[0, 1]
        return c if not np.isnan(c) else 0.0

    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

    # Re-mapear keys
    bp = study.best_params.copy()
    bp.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})
    rename = {'ne': 'n_estimators', 'md': 'max_depth', 'lr': 'learning_rate',
              'ss': 'subsample', 'cs': 'colsample_bytree', 'mc': 'min_child_samples',
              'ra': 'reg_alpha', 'rl': 'reg_lambda', 'nl': 'num_leaves'}
    bp2 = {rename.get(k, k): v for k, v in bp.items()}

    # Entrenar modelo FINAL en todos los datos
    model = lgb.LGBMRegressor(**bp2)
    model.fit(X, y, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(50, verbose=False)])

    # pred_std para confidence (desde predicciones de validacion)
    val_preds = model.predict(Xv)
    pred_std = float(np.std(val_preds))
    corr = float(np.corrcoef(val_preds, yv.values)[0, 1])

    # Guardar modelo
    safe = pair.replace('/', '_')
    model_path = MODELS_DIR / f'v7_{safe}.pkl'
    joblib.dump(model, model_path)

    print(f'  {pair:<12} {len(ds):>6,} velas | corr={corr:.4f} | '
          f'pred_std={pred_std:.4f} | optuna={study.best_value:.4f}')

    return {
        'pair': pair,
        'feature_cols': fcols,
        'pred_std': pred_std,
        'corr': corr,
        'n_samples': len(ds),
        'best_params': bp2,
    }


def main():
    t0 = time.time()
    print('=' * 60)
    print('EXPORT MODELOS ML V7 - PRODUCCION')
    print('=' * 60)

    # Descargar datos
    print('\n[1/3] Descargando datos...')
    all_data = {}
    for pair in PAIRS:
        df = download_pair(pair, '4h')
        if df is not None and len(df) > 500:
            all_data[pair] = df
            print(f'  {pair:<12} {len(df):>6,} velas | '
                  f'{df.index[0].date()} a {df.index[-1].date()}')

    # Entrenar modelos
    print(f'\n[2/3] Entrenando modelos ({N_TRIALS} Optuna trials)...')
    results = {}
    pred_stds = {}
    feature_cols = None

    for pair, df in all_data.items():
        result = train_and_export(pair, df)
        if result:
            results[pair] = result
            pred_stds[pair] = result['pred_std']
            if feature_cols is None:
                feature_cols = result['feature_cols']

    # Guardar metadata
    print(f'\n[3/3] Guardando metadata...')
    meta = {
        'trained_at': datetime.utcnow().isoformat(),
        'pairs': list(results.keys()),
        'feature_cols': feature_cols or [],
        'pred_stds': pred_stds,
        'horizon': HORIZON,
        'n_trials': N_TRIALS,
        'stats': {
            pair: {'corr': r['corr'], 'n_samples': r['n_samples']}
            for pair, r in results.items()
        }
    }
    meta_path = MODELS_DIR / 'v7_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    elapsed = (time.time() - t0) / 60
    print(f'\n{"=" * 60}')
    print(f'COMPLETADO: {len(results)} modelos exportados en {elapsed:.1f} min')
    print(f'Modelos guardados en: {MODELS_DIR.absolute()}')
    print(f'Metadata: {meta_path}')
    print(f'\nPara iniciar el bot:')
    print(f'  poetry run python -m src.ml_bot')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
