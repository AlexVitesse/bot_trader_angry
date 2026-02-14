"""
ML Export Models - Entrena y exporta modelos para produccion
============================================================
Entrena LightGBM en TODOS los datos disponibles y guarda modelos + metadata.
V7: Per-pair regression models (signal generation)
V8.4: MacroScorer classifier (macro-aware threshold + sizing)
V8.5: ConvictionScorer regressor (per-trade PnL prediction -> sizing + filtering)

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
from collections import defaultdict
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


def train_macro_scorer(all_data, macro_feat, n_trials_v7=15, n_trials_macro=30):
    """Train V8.4 MacroScorer using nested walk-forward.

    Steps:
    1. Split all 4h data at midpoint
    2. Train V7 models on early half
    3. Run V7 backtest on late half -> OOS trades
    4. Create daily labels (1=V7 profitable, 0=V7 lost)
    5. Train MacroScorer classifier on macro features + labels
    6. Re-train FINAL model on ALL labels

    Returns: (model, feature_cols, auc) or (None, None, None) on failure.
    """
    from ml_train_v7 import (
        train_all, backtest, detect_regime, download_pair as dl_v7,
    )
    from sklearn.metrics import roc_auc_score

    # Download BTC daily for regime detection
    print('  Descargando BTC daily para regime...')
    btc_d = dl_v7('BTC/USDT', '1d')
    if btc_d is None:
        print('  ERROR: No se pudo descargar BTC daily')
        return None, None, None

    regime = detect_regime(btc_d)

    # Find midpoint of available data
    all_dates = set()
    for pair, df in all_data.items():
        all_dates.update(df.index.tolist())
    all_dates = sorted(all_dates)
    if len(all_dates) < 3000:
        print(f'  ERROR: Datos insuficientes ({len(all_dates)} velas, necesita 3000+)')
        return None, None, None

    mid = all_dates[len(all_dates) // 2].strftime('%Y-%m')
    print(f'  Nested WF: train [start..{mid}], eval [{mid}..end]')

    # Train V7 on early half
    print(f'  Entrenando V7 en early half ({n_trials_v7} trials)...')
    early_models = train_all(all_data, mid, horizon=5, n_trials=n_trials_v7)

    if not early_models:
        print('  ERROR: No se pudieron entrenar modelos V7 en early half')
        return None, None, None

    # Run V7 backtest on late half -> OOS trades
    print('  Corriendo V7 backtest en late half...')
    trades, _, _, _ = backtest(
        all_data, regime, early_models,
        use_trailing=True, use_atr=True, use_compound=False, max_notional=300,
    )
    print(f'  {len(trades)} OOS trades generados')

    if len(trades) < 50:
        print('  ERROR: Insuficientes trades para entrenar MacroScorer')
        return None, None, None

    # Create daily labels: 1=V7 profitable day, 0=V7 lost
    daily_pnl = defaultdict(float)
    for t in trades:
        daily_pnl[t['time'].strftime('%Y-%m-%d')] += t['pnl']

    labels = pd.Series(daily_pnl)
    labels.index = pd.to_datetime(labels.index)
    y = (labels > 0).astype(int)

    macro_daily = macro_feat.reindex(y.index, method='ffill')
    valid = macro_daily.dropna().index.intersection(y.index)

    if len(valid) < 50:
        print(f'  ERROR: Solo {len(valid)} dias con labels y macro features')
        return None, None, None

    X = macro_daily.loc[valid]
    y = y.loc[valid]
    fcols = list(X.columns)
    X_clean = X[fcols].fillna(0)

    print(f'  {len(y)} dias de entrenamiento | '
          f'positivos: {y.sum()} ({y.mean():.0%}) | negativos: {(1 - y).sum()}')

    # Optuna optimization
    sp = int(len(X_clean) * 0.8)
    Xt, Xv = X_clean.iloc[:sp], X_clean.iloc[sp:]
    yt, yv = y.iloc[:sp], y.iloc[sp:]

    def obj(trial):
        p = {
            'n_estimators': trial.suggest_int('ne', 50, 300),
            'max_depth': trial.suggest_int('md', 2, 6),
            'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('ss', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('cs', 0.4, 1.0),
            'min_child_samples': trial.suggest_int('mc', 20, 100),
            'reg_alpha': trial.suggest_float('ra', 1e-5, 5.0, log=True),
            'reg_lambda': trial.suggest_float('rl', 1e-5, 5.0, log=True),
            'num_leaves': trial.suggest_int('nl', 10, 50),
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
            'objective': 'binary',
        }
        m = lgb.LGBMClassifier(**p)
        m.fit(Xt, yt, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(20, verbose=False)])
        probs = m.predict_proba(Xv)[:, 1]
        try:
            return roc_auc_score(yv, probs)
        except Exception:
            return 0.5

    print(f'  Optimizando MacroScorer ({n_trials_macro} Optuna trials)...')
    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=n_trials_macro, show_progress_bar=False)

    rename = {'ne': 'n_estimators', 'md': 'max_depth', 'lr': 'learning_rate',
              'ss': 'subsample', 'cs': 'colsample_bytree', 'mc': 'min_child_samples',
              'ra': 'reg_alpha', 'rl': 'reg_lambda', 'nl': 'num_leaves'}
    bp = {rename.get(k, k): v for k, v in study.best_params.items()}
    bp.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1, 'objective': 'binary'})

    # Train FINAL model on ALL data
    model = lgb.LGBMClassifier(**bp)
    model.fit(X_clean, y, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(30, verbose=False)])

    # Evaluate on validation set
    val_probs = model.predict_proba(Xv)[:, 1]
    try:
        auc = roc_auc_score(yv, val_probs)
    except Exception:
        auc = 0.5

    print(f'  MacroScorer AUC: {auc:.3f} (optuna best: {study.best_value:.3f})')

    return model, fcols, auc


def train_conviction_scorer(all_data, macro_feat, macro_scorer, macro_fcols,
                            n_trials_v7=15, n_trials_conv=30):
    """Train V8.5 ConvictionScorer using nested walk-forward OOS trades.

    Steps:
    1. Train V7 on early half + run backtest on late half -> OOS trades
    2. Score each trade with MacroScorer features at entry time
    3. Train ConvictionScorer regressor on (entry features -> trade PnL)

    Returns: (model, feature_cols, pred_std, corr) or (None, None, None, None).
    """
    from ml_train_v7 import (
        train_all, detect_regime, download_pair as dl_v7,
    )
    from ml_train_v85 import (
        backtest_collect_trade_features, prepare_conviction_data,
        CONVICTION_FEATURES,
    )
    from ml_train_v84 import compute_risk_off_multipliers

    # Download BTC daily for regime detection
    print('  Descargando BTC daily para regime...')
    btc_d = dl_v7('BTC/USDT', '1d')
    if btc_d is None:
        print('  ERROR: No se pudo descargar BTC daily')
        return None, None, None, None

    regime = detect_regime(btc_d)

    # Find midpoint of available data
    all_dates = set()
    for pair, df in all_data.items():
        all_dates.update(df.index.tolist())
    all_dates = sorted(all_dates)
    if len(all_dates) < 3000:
        print(f'  ERROR: Datos insuficientes ({len(all_dates)} velas)')
        return None, None, None, None

    mid = all_dates[len(all_dates) // 2].strftime('%Y-%m')
    print(f'  Nested WF: train [start..{mid}], eval [{mid}..end]')

    # Train V7 on early half
    print(f'  Entrenando V7 en early half ({n_trials_v7} trials)...')
    early_models = train_all(all_data, mid, horizon=5, n_trials=n_trials_v7)

    if not early_models:
        print('  ERROR: No V7 models trained')
        return None, None, None, None

    # Compute risk-off multipliers
    ro_mults = compute_risk_off_multipliers(macro_feat)

    # Run backtest collecting per-trade features
    print('  Corriendo backtest con trade features...')
    enriched_trades = backtest_collect_trade_features(
        all_data, regime, early_models,
        macro_scorer=macro_scorer, macro_feat=macro_feat,
        macro_fcols=macro_fcols, risk_off_mults=ro_mults,
    )
    print(f'  {len(enriched_trades)} OOS trades con features')

    if len(enriched_trades) < 80:
        print(f'  ERROR: Solo {len(enriched_trades)} trades (min 80)')
        return None, None, None, None

    # Prepare training data
    X, y = prepare_conviction_data(enriched_trades)
    if X is None:
        print('  ERROR: No se pudo preparar datos de entrenamiento')
        return None, None, None, None

    fcols = list(X.columns)
    X_clean = X.fillna(0)

    sp = int(len(X_clean) * 0.8)
    Xt, Xv = X_clean.iloc[:sp], X_clean.iloc[sp:]
    yt, yv = y.iloc[:sp], y.iloc[sp:]

    print(f'  {len(y)} trades de entrenamiento | '
          f'PnL medio: ${y.mean():.2f} | std: ${y.std():.2f}')

    # Optuna optimization
    def obj(trial):
        p = {
            'n_estimators': trial.suggest_int('ne', 50, 300),
            'max_depth': trial.suggest_int('md', 2, 6),
            'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('ss', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('cs', 0.4, 1.0),
            'min_child_samples': trial.suggest_int('mc', 10, 80),
            'reg_alpha': trial.suggest_float('ra', 1e-5, 5.0, log=True),
            'reg_lambda': trial.suggest_float('rl', 1e-5, 5.0, log=True),
            'num_leaves': trial.suggest_int('nl', 10, 50),
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        }
        m = lgb.LGBMRegressor(**p)
        m.fit(Xt, yt, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(20, verbose=False)])
        pr = m.predict(Xv)
        corr = np.corrcoef(pr, yv.values)[0, 1]
        return corr if not np.isnan(corr) else 0.0

    print(f'  Optimizando ConvictionScorer ({n_trials_conv} Optuna trials)...')
    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=n_trials_conv, show_progress_bar=False)

    rename = {'ne': 'n_estimators', 'md': 'max_depth', 'lr': 'learning_rate',
              'ss': 'subsample', 'cs': 'colsample_bytree', 'mc': 'min_child_samples',
              'ra': 'reg_alpha', 'rl': 'reg_lambda', 'nl': 'num_leaves'}
    bp = {rename.get(k, k): v for k, v in study.best_params.items()}
    bp.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})

    # Train FINAL model on ALL data
    model = lgb.LGBMRegressor(**bp)
    model.fit(X_clean, y, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(30, verbose=False)])

    # Compute pred_std for sizing normalization
    preds_all = model.predict(X_clean)
    pred_std = float(np.std(preds_all))

    # Evaluate on validation set
    val_preds = model.predict(Xv)
    corr = float(np.corrcoef(val_preds, yv.values)[0, 1])

    print(f'  ConvictionScorer corr: {corr:.3f} (optuna best: {study.best_value:.3f})')
    print(f'  pred_std: {pred_std:.4f}')

    return model, fcols, pred_std, corr


def main():
    t0 = time.time()
    print('=' * 60)
    print('EXPORT MODELOS ML V7 + V8.4 + V8.5 - PRODUCCION')
    print('=' * 60)

    # =========================================================
    # [1/5] Download data
    # =========================================================
    print('\n[1/5] Descargando datos...')
    all_data = {}
    for pair in PAIRS:
        df = download_pair(pair, '4h')
        if df is not None and len(df) > 500:
            all_data[pair] = df
            print(f'  {pair:<12} {len(df):>6,} velas | '
                  f'{df.index[0].date()} a {df.index[-1].date()}')

    # =========================================================
    # [2/5] Train V7 per-pair models
    # =========================================================
    print(f'\n[2/5] Entrenando modelos V7 ({N_TRIALS} Optuna trials)...')
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

    # Save V7 metadata
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
    print(f'  V7 metadata guardada: {meta_path}')

    # =========================================================
    # [3/5] Train V8.4 MacroScorer
    # =========================================================
    print(f'\n[3/5] Entrenando V8.4 MacroScorer...')
    scorer = None
    scorer_fcols = []
    macro_feat = None
    try:
        from macro_data import download_all_macro, compute_macro_features

        macro = download_all_macro()
        macro_feat = compute_macro_features(
            macro.get('dxy'), macro.get('gold'), macro.get('spy'),
            macro.get('tnx'), macro.get('ethbtc'),
        )

        if macro_feat is not None and len(macro_feat) > 100:
            scorer, scorer_fcols, auc = train_macro_scorer(
                all_data, macro_feat,
                n_trials_v7=15,   # Less trials for nested V7 (speed)
                n_trials_macro=30,
            )

            if scorer is not None:
                scorer_path = MODELS_DIR / 'v84_macro_scorer.pkl'
                joblib.dump(scorer, scorer_path)
                print(f'  MacroScorer guardado: {scorer_path}')

                v84_meta = {
                    'trained_at': datetime.utcnow().isoformat(),
                    'macro_feature_cols': scorer_fcols,
                    'auc': auc,
                    'n_macro_features': len(scorer_fcols),
                }
                v84_meta_path = MODELS_DIR / 'v84_meta.json'
                with open(v84_meta_path, 'w') as f:
                    json.dump(v84_meta, f, indent=2)
                print(f'  V8.4 metadata guardada: {v84_meta_path}')
            else:
                print('  AVISO: MacroScorer no pudo entrenarse - bot usara V7 puro')
        else:
            print('  AVISO: Datos macro insuficientes - bot usara V7 puro')
    except ImportError as e:
        print(f'  AVISO: Dependencias macro no disponibles ({e}) - bot usara V7 puro')
    except Exception as e:
        print(f'  ERROR en MacroScorer: {e} - bot usara V7 puro')

    # =========================================================
    # [4/5] Train V8.5 ConvictionScorer
    # =========================================================
    print(f'\n[4/5] Entrenando V8.5 ConvictionScorer...')
    try:
        if scorer is not None and macro_feat is not None:
            conv_model, conv_fcols, conv_pred_std, conv_corr = train_conviction_scorer(
                all_data, macro_feat, scorer, scorer_fcols,
                n_trials_v7=15,
                n_trials_conv=30,
            )

            if conv_model is not None:
                conv_path = MODELS_DIR / 'v85_conviction_scorer.pkl'
                joblib.dump(conv_model, conv_path)
                print(f'  ConvictionScorer guardado: {conv_path}')

                v85_meta = {
                    'trained_at': datetime.utcnow().isoformat(),
                    'conviction_feature_cols': conv_fcols,
                    'conviction_pred_std': conv_pred_std,
                    'corr': conv_corr,
                    'n_features': len(conv_fcols),
                }
                v85_meta_path = MODELS_DIR / 'v85_meta.json'
                with open(v85_meta_path, 'w') as f:
                    json.dump(v85_meta, f, indent=2)
                print(f'  V8.5 metadata guardada: {v85_meta_path}')
            else:
                print('  AVISO: ConvictionScorer no pudo entrenarse - bot usara V8.4')
        else:
            print('  AVISO: MacroScorer requerido para ConvictionScorer - bot usara V7/V8.4')
    except ImportError as e:
        print(f'  AVISO: Dependencias V8.5 no disponibles ({e}) - bot usara V8.4')
    except Exception as e:
        print(f'  ERROR en ConvictionScorer: {e} - bot usara V8.4')

    # =========================================================
    # [5/5] Summary
    # =========================================================
    elapsed = (time.time() - t0) / 60
    print(f'\n{"=" * 60}')
    print(f'COMPLETADO: {len(results)} modelos V7 exportados en {elapsed:.1f} min')
    print(f'Modelos guardados en: {MODELS_DIR.absolute()}')
    print(f'\nPara iniciar el bot:')
    print(f'  poetry run python -m src.ml_bot')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
