"""
ML Train V9.5 - LossDetector por Par + V7 Mejorado
===================================================
Mejoras sobre V9:
1. LossDetector individual por par (11 modelos vs 1 generico)
2. V7 con mas trials para pares debiles (SOL, AVAX, BTC, BNB)
3. Usa data COMPLETA 2020-2026 (incluye bear 2022)
4. Threshold optimizado por par

Uso: poetry run python ml_train_v95.py
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
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# CONFIGURACION V9.5
# ============================================================================
DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)

PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT',
    'NEAR/USDT',
]

# Pares debiles que necesitan mas trials
WEAK_PAIRS = ['SOL/USDT', 'AVAX/USDT', 'BTC/USDT', 'BNB/USDT']
N_TRIALS_WEAK = 100    # Mas trials para pares debiles
N_TRIALS_NORMAL = 40   # Trials normales
N_TRIALS_LD = 30       # Trials para LossDetector por par

MIN_CANDLES = 3000
HORIZON = 5
MIN_TRADES_PER_PAIR = 50  # Minimo trades para entrenar LD

# Features del LossDetector (mismas que V9)
LOSS_FEATURES = [
    'cs_conf', 'cs_pred_mag', 'cs_macro_score', 'cs_risk_off',
    'cs_regime_bull', 'cs_regime_bear', 'cs_regime_range',
    'cs_atr_pct', 'cs_n_open', 'cs_pred_sign',
    'ld_conviction_pred',
    'ld_pair_rsi14', 'ld_pair_bb_pct', 'ld_pair_vol_ratio',
    'ld_pair_ret_5', 'ld_pair_ret_20',
    'ld_btc_ret_5', 'ld_btc_rsi14', 'ld_btc_vol20',
    'ld_hour', 'ld_tp_sl_ratio',
]


def download_pair(symbol, timeframe='4h', since='2020-01-01'):
    """Descarga datos historicos completos."""
    safe = symbol.replace('/', '_')
    cache = DATA_DIR / f'{safe}_{timeframe}_v95.parquet'

    # Forzar re-descarga si cache tiene mas de 12h
    if cache.exists() and (time.time() - cache.stat().st_mtime) / 3600 < 12:
        return pd.read_parquet(cache)

    print(f'  Descargando {symbol}...')
    exchange = ccxt.binance({'enableRateLimit': True})
    since_ts = int(datetime.strptime(since, '%Y-%m-%d').timestamp() * 1000)
    rows = []

    while True:
        try:
            c = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=1000)
        except Exception as e:
            print(f'    Error {symbol}: {e}')
            time.sleep(5)
            continue
        if not c:
            break
        rows.extend(c)
        since_ts = c[-1][0] + 1
        if len(c) < 1000:
            break
        time.sleep(0.15)

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    df.to_parquet(cache)
    return df


def compute_features(df):
    """Features V7 - identicas al modelo original."""
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
    if sr is not None:
        feat['srsi_k'] = sr.iloc[:, 0]
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        feat['macd_h'] = macd.iloc[:, 1]
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

    hr = df.index.hour
    dw = df.index.dayofweek
    feat['h_s'] = np.sin(2 * np.pi * hr / 24)
    feat['h_c'] = np.cos(2 * np.pi * hr / 24)
    feat['d_s'] = np.sin(2 * np.pi * dw / 7)
    feat['d_c'] = np.cos(2 * np.pi * dw / 7)

    return feat


def train_v7_model(pair, df, n_trials):
    """Entrena modelo V7 para un par."""
    feat = compute_features(df)
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
    fcols = list(feat.columns)

    tgt = f'fwd_{HORIZON}'
    feat[tgt] = (df['close'].shift(-HORIZON) - df['close']) / df['close']
    ds = feat.dropna()

    if len(ds) < MIN_CANDLES:
        return None

    X = ds[fcols].fillna(0)
    y = ds[tgt].clip(ds[tgt].quantile(0.01), ds[tgt].quantile(0.99))

    # Split 85/15
    sp = int(len(X) * 0.85)
    Xt, Xv = X.iloc[:sp], X.iloc[sp:]
    yt, yv = y.iloc[:sp], y.iloc[sp:]

    def obj(trial):
        p = {
            'n_estimators': trial.suggest_int('ne', 50, 400),
            'max_depth': trial.suggest_int('md', 3, 8),
            'learning_rate': trial.suggest_float('lr', 0.005, 0.15, log=True),
            'subsample': trial.suggest_float('ss', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('cs', 0.3, 1.0),
            'min_child_samples': trial.suggest_int('mc', 5, 100),
            'reg_alpha': trial.suggest_float('ra', 1e-6, 10.0, log=True),
            'reg_lambda': trial.suggest_float('rl', 1e-6, 10.0, log=True),
            'num_leaves': trial.suggest_int('nl', 8, 80),
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        }
        m = lgb.LGBMRegressor(**p)
        m.fit(Xt, yt, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(25, verbose=False)])
        pr = m.predict(Xv)
        corr = np.corrcoef(pr, yv.values)[0, 1]
        return corr if not np.isnan(corr) else 0.0

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

    # Renombrar parametros
    rename = {'ne': 'n_estimators', 'md': 'max_depth', 'lr': 'learning_rate',
              'ss': 'subsample', 'cs': 'colsample_bytree', 'mc': 'min_child_samples',
              'ra': 'reg_alpha', 'rl': 'reg_lambda', 'nl': 'num_leaves'}
    bp = {rename.get(k, k): v for k, v in study.best_params.items()}
    bp.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})

    # Entrenar modelo final en TODOS los datos
    model = lgb.LGBMRegressor(**bp)
    model.fit(X, y, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(30, verbose=False)])

    # Calcular pred_std
    preds_all = model.predict(X)
    pred_std = float(np.std(preds_all))

    # Evaluar en validacion
    val_preds = model.predict(Xv)
    corr = float(np.corrcoef(val_preds, yv.values)[0, 1])

    return {
        'model': model,
        'feature_cols': fcols,
        'pred_std': pred_std,
        'corr': corr,
        'n_samples': len(X),
        'best_trial': study.best_value,
    }


def detect_regime(btc_df):
    """Detecta regimen de mercado usando BTC."""
    c = btc_df['close']
    ema20 = ta.ema(c, length=20)
    ema50 = ta.ema(c, length=50)
    ret20 = c.pct_change(20)

    regime = pd.Series('RANGE', index=btc_df.index)
    regime[(c > ema20) & (ema20 > ema50) & (ret20 > 0.05)] = 'BULL'
    regime[(c < ema20) & (ema20 < ema50) & (ret20 < -0.05)] = 'BEAR'
    return regime


def run_backtest_collect_trades(all_data, v7_models, regime_series):
    """Ejecuta backtest y recolecta trades con features para LossDetector.

    Usa TODOS los datos para generar trades (no split).
    """
    from collections import defaultdict

    trades = []

    # Parametros de trading
    TP_PCT = 0.03
    SL_PCT = 0.015
    MAX_HOLD = 30

    btc_df = all_data['BTC/USDT']

    for pair, df in all_data.items():
        if pair not in v7_models:
            continue

        model_info = v7_models[pair]
        model = model_info['model']
        fcols = model_info['feature_cols']
        pred_std = model_info['pred_std']

        feat = compute_features(df)
        feat = feat.replace([np.inf, -np.inf], np.nan)

        # Generar predicciones
        valid_idx = feat.dropna().index
        if len(valid_idx) < 100:
            continue

        X = feat.loc[valid_idx, fcols].fillna(0)
        preds = model.predict(X)

        # Simular trades
        position = None

        for i, (ts, pred) in enumerate(zip(valid_idx, preds)):
            if i < 50:  # Warmup
                continue

            # Obtener regime
            regime = 'RANGE'
            if ts in regime_series.index:
                regime = regime_series.loc[ts]

            # Calcular confianza
            conf = abs(pred) / pred_std if pred_std > 1e-8 else 0
            direction = 1 if pred > 0 else -1

            # Filtrar por regime
            if regime == 'BULL' and direction == -1:
                continue
            if regime == 'BEAR' and direction == 1:
                continue

            price = df.loc[ts, 'close']

            # Si no hay posicion y hay senal fuerte
            if position is None and conf > 0.7:
                # Calcular features para LossDetector
                idx_pos = df.index.get_loc(ts)

                # ATR %
                atr = ta.atr(df['high'], df['low'], df['close'], length=14)
                atr_pct = atr.iloc[idx_pos] / price if idx_pos < len(atr) else 0.02

                # RSI
                rsi = ta.rsi(df['close'], length=14)
                rsi_val = rsi.iloc[idx_pos] if idx_pos < len(rsi) else 50

                # BB position
                bb = ta.bbands(df['close'], length=20, std=2.0)
                if bb is not None and idx_pos < len(bb):
                    bb_lower = bb.iloc[idx_pos, 0]
                    bb_upper = bb.iloc[idx_pos, 2]
                    bb_pct = (price - bb_lower) / (bb_upper - bb_lower + 1e-10)
                else:
                    bb_pct = 0.5

                # Volume ratio
                vol = df['volume']
                vol_ma = vol.rolling(20).mean()
                vol_ratio = vol.iloc[idx_pos] / vol_ma.iloc[idx_pos] if idx_pos < len(vol_ma) else 1.0

                # Returns
                ret_5 = df['close'].pct_change(5).iloc[idx_pos] if idx_pos >= 5 else 0
                ret_20 = df['close'].pct_change(20).iloc[idx_pos] if idx_pos >= 20 else 0

                # BTC context
                btc_idx = btc_df.index.get_indexer([ts], method='nearest')[0]
                btc_ret_5 = btc_df['close'].pct_change(5).iloc[btc_idx] if btc_idx >= 5 else 0
                btc_rsi = ta.rsi(btc_df['close'], length=14)
                btc_rsi_val = btc_rsi.iloc[btc_idx] if btc_idx < len(btc_rsi) else 50
                btc_vol = btc_df['close'].pct_change().rolling(20).std()
                btc_vol_val = btc_vol.iloc[btc_idx] if btc_idx < len(btc_vol) else 0.02

                position = {
                    'pair': pair,
                    'entry_time': ts,
                    'entry_price': price,
                    'direction': direction,
                    'confidence': conf,
                    'regime': regime,
                    'atr_pct': atr_pct,
                    'hold_bars': 0,
                    # Features para LossDetector
                    'cs_conf': conf,
                    'cs_pred_mag': abs(pred),
                    'cs_macro_score': 0.5,  # Placeholder
                    'cs_risk_off': 1.0,
                    'cs_regime_bull': 1.0 if regime == 'BULL' else 0.0,
                    'cs_regime_bear': 1.0 if regime == 'BEAR' else 0.0,
                    'cs_regime_range': 1.0 if regime == 'RANGE' else 0.0,
                    'cs_atr_pct': atr_pct,
                    'cs_n_open': 0,
                    'cs_pred_sign': float(direction),
                    'ld_conviction_pred': pred,
                    'ld_pair_rsi14': rsi_val / 100.0,
                    'ld_pair_bb_pct': bb_pct,
                    'ld_pair_vol_ratio': vol_ratio,
                    'ld_pair_ret_5': ret_5,
                    'ld_pair_ret_20': ret_20,
                    'ld_btc_ret_5': btc_ret_5,
                    'ld_btc_rsi14': btc_rsi_val / 100.0,
                    'ld_btc_vol20': btc_vol_val,
                    'ld_hour': ts.hour / 24.0,
                    'ld_tp_sl_ratio': TP_PCT / SL_PCT,
                }
                continue

            # Gestionar posicion abierta
            if position is not None and position['pair'] == pair:
                position['hold_bars'] += 1
                entry = position['entry_price']
                d = position['direction']

                # Calcular PnL
                if d == 1:  # LONG
                    pnl_pct = (price - entry) / entry
                else:  # SHORT
                    pnl_pct = (entry - price) / entry

                # Check TP/SL/MaxHold
                exit_reason = None
                if pnl_pct >= TP_PCT:
                    exit_reason = 'TP'
                elif pnl_pct <= -SL_PCT:
                    exit_reason = 'SL'
                elif position['hold_bars'] >= MAX_HOLD:
                    exit_reason = 'TIME'

                if exit_reason:
                    # Calcular PnL en dolares (simulado con $100 notional)
                    pnl_usd = 100 * pnl_pct

                    trade = {
                        **position,
                        'exit_time': ts,
                        'exit_price': price,
                        'exit_reason': exit_reason,
                        'pnl_pct': pnl_pct,
                        'pnl': pnl_usd,
                    }
                    trades.append(trade)
                    position = None

    return trades


def train_loss_detector_for_pair(pair_trades, pair, n_trials=N_TRIALS_LD):
    """Entrena LossDetector para un par especifico."""
    if len(pair_trades) < MIN_TRADES_PER_PAIR:
        print(f'    {pair}: solo {len(pair_trades)} trades (min {MIN_TRADES_PER_PAIR}). SKIP.')
        return None, None, 0.0, 0.55

    df = pd.DataFrame(pair_trades)

    # Features disponibles
    fcols = [c for c in LOSS_FEATURES if c in df.columns]
    X = df[fcols].fillna(0).copy()
    X = X.replace([np.inf, -np.inf], 0)

    # Target: 1 = perdio dinero
    y = (df['pnl'] < 0).astype(int)

    n_loss = y.sum()
    n_win = len(y) - n_loss
    loss_rate = n_loss / len(y) * 100

    if n_loss < 10 or n_win < 10:
        print(f'    {pair}: clases desbalanceadas ({n_win}W/{n_loss}L). SKIP.')
        return None, None, 0.0, 0.55

    # Split 80/20
    sp = int(len(X) * 0.8)
    Xt, Xv = X.iloc[:sp], X.iloc[sp:]
    yt, yv = y.iloc[:sp], y.iloc[sp:]

    scale = n_win / n_loss if n_loss > 0 else 1.0

    def obj(trial):
        p = {
            'n_estimators': trial.suggest_int('ne', 30, 200),
            'max_depth': trial.suggest_int('md', 2, 5),
            'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('ss', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('cs', 0.4, 1.0),
            'min_child_samples': trial.suggest_int('mc', 5, 50),
            'reg_alpha': trial.suggest_float('ra', 1e-5, 5.0, log=True),
            'reg_lambda': trial.suggest_float('rl', 1e-5, 5.0, log=True),
            'num_leaves': trial.suggest_int('nl', 5, 30),
            'scale_pos_weight': scale,
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        }
        m = lgb.LGBMClassifier(**p)
        m.fit(Xt, yt, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(15, verbose=False)])
        proba = m.predict_proba(Xv)[:, 1]
        try:
            auc = roc_auc_score(yv, proba)
        except:
            auc = 0.5
        return auc

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

    # Renombrar parametros
    rename = {'ne': 'n_estimators', 'md': 'max_depth', 'lr': 'learning_rate',
              'ss': 'subsample', 'cs': 'colsample_bytree', 'mc': 'min_child_samples',
              'ra': 'reg_alpha', 'rl': 'reg_lambda', 'nl': 'num_leaves'}
    bp = {rename.get(k, k): v for k, v in study.best_params.items()}
    bp.update({'scale_pos_weight': scale, 'random_state': 42, 'n_jobs': -1, 'verbose': -1})

    # Entrenar modelo final
    model = lgb.LGBMClassifier(**bp)
    model.fit(X, y, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(20, verbose=False)])

    # Evaluar
    proba_val = model.predict_proba(Xv)[:, 1]
    try:
        auc = roc_auc_score(yv, proba_val)
    except:
        auc = 0.5

    # Optimizar threshold para este par
    best_threshold = 0.55
    best_score = 0
    for thresh in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        preds = (proba_val >= thresh).astype(int)
        # Score: queremos rechazar trades perdedores pero no rechazar ganadores
        # True Positive Rate para losses (recall de losses)
        tp_loss = ((preds == 1) & (yv == 1)).sum()
        fn_loss = ((preds == 0) & (yv == 1)).sum()
        recall_loss = tp_loss / (tp_loss + fn_loss + 1e-10)

        # False Positive Rate (rechazar ganadores por error)
        fp = ((preds == 1) & (yv == 0)).sum()
        tn = ((preds == 0) & (yv == 0)).sum()
        fpr = fp / (fp + tn + 1e-10)

        # Score: maximizar recall de losses, minimizar FPR
        score = recall_loss - 0.5 * fpr
        if score > best_score:
            best_score = score
            best_threshold = thresh

    return model, fcols, auc, best_threshold


def main():
    t0 = time.time()
    print('=' * 60)
    print('ENTRENAMIENTO V9.5 - LossDetector por Par')
    print('=' * 60)

    # =========================================================================
    # [1] Descargar datos
    # =========================================================================
    print('\n[1/4] Descargando datos completos 2020-2026...')
    all_data = {}
    for pair in PAIRS:
        df = download_pair(pair, '4h', since='2020-01-01')
        if df is not None and len(df) > MIN_CANDLES:
            all_data[pair] = df
            print(f'  {pair:<12} {len(df):>6,} velas | '
                  f'{df.index[0].date()} a {df.index[-1].date()}')

    if len(all_data) < 5:
        print('ERROR: No hay suficientes datos')
        return

    # =========================================================================
    # [2] Entrenar modelos V7 mejorados
    # =========================================================================
    print(f'\n[2/4] Entrenando modelos V7...')
    v7_models = {}

    for pair in all_data.keys():
        n_trials = N_TRIALS_WEAK if pair in WEAK_PAIRS else N_TRIALS_NORMAL
        print(f'  {pair:<12} ({n_trials} trials)...', end=' ', flush=True)

        result = train_v7_model(pair, all_data[pair], n_trials)
        if result:
            v7_models[pair] = result
            print(f'corr={result["corr"]:.3f} (best={result["best_trial"]:.3f})')
        else:
            print('SKIP (datos insuficientes)')

    # Guardar modelos V7 mejorados
    print('\n  Guardando modelos V7...')
    for pair, info in v7_models.items():
        safe = pair.replace('/', '')
        joblib.dump(info['model'], MODELS_DIR / f'v95_v7_{safe}.pkl')

    # =========================================================================
    # [3] Generar trades para entrenamiento de LossDetector
    # =========================================================================
    print(f'\n[3/4] Generando trades con backtest completo...')

    # Detectar regime
    btc_daily = download_pair('BTC/USDT', '1d', since='2020-01-01')
    regime_series = detect_regime(btc_daily)

    # Ejecutar backtest
    all_trades = run_backtest_collect_trades(all_data, v7_models, regime_series)
    print(f'  Total trades generados: {len(all_trades)}')

    # Agrupar por par
    trades_by_pair = {}
    for t in all_trades:
        pair = t['pair']
        if pair not in trades_by_pair:
            trades_by_pair[pair] = []
        trades_by_pair[pair].append(t)

    print('  Trades por par:')
    for pair, trades in sorted(trades_by_pair.items()):
        wins = sum(1 for t in trades if t['pnl'] > 0)
        print(f'    {pair:<12} {len(trades):>4} trades ({wins}W/{len(trades)-wins}L)')

    # =========================================================================
    # [4] Entrenar LossDetector por par
    # =========================================================================
    print(f'\n[4/4] Entrenando LossDetector por par ({N_TRIALS_LD} trials cada uno)...')

    ld_models = {}
    ld_meta = {
        'version': '9.5',
        'trained_at': datetime.now().isoformat(),
        'pairs': {},
    }

    for pair in PAIRS:
        if pair not in trades_by_pair:
            print(f'  {pair:<12} sin trades. SKIP.')
            continue

        pair_trades = trades_by_pair[pair]
        print(f'  {pair:<12}...', end=' ', flush=True)

        model, fcols, auc, threshold = train_loss_detector_for_pair(
            pair_trades, pair, N_TRIALS_LD
        )

        if model is not None:
            safe = pair.replace('/', '')
            model_path = MODELS_DIR / f'v95_ld_{safe}.pkl'
            joblib.dump(model, model_path)

            ld_models[pair] = {
                'model': model,
                'fcols': fcols,
                'auc': auc,
                'threshold': threshold,
            }
            ld_meta['pairs'][pair] = {
                'auc': auc,
                'threshold': threshold,
                'n_trades': len(pair_trades),
                'n_features': len(fcols),
            }
            print(f'AUC={auc:.3f} | thresh={threshold:.2f}')
        else:
            print('SKIP')

    # Guardar metadata
    ld_meta['feature_cols'] = LOSS_FEATURES
    with open(MODELS_DIR / 'v95_meta.json', 'w') as f:
        json.dump(ld_meta, f, indent=2)

    # =========================================================================
    # Resumen
    # =========================================================================
    elapsed = (time.time() - t0) / 60
    print('\n' + '=' * 60)
    print(f'COMPLETADO en {elapsed:.1f} minutos')
    print('=' * 60)

    print(f'\nModelos V7 entrenados: {len(v7_models)}')
    for pair, info in v7_models.items():
        marker = '*' if pair in WEAK_PAIRS else ' '
        print(f'  {marker} {pair:<12} corr={info["corr"]:.3f}')

    print(f'\nLossDetectors V9.5 entrenados: {len(ld_models)}')
    for pair, info in ld_models.items():
        print(f'  {pair:<12} AUC={info["auc"]:.3f} | thresh={info["threshold"]:.2f}')

    print(f'\nArchivos guardados en: {MODELS_DIR}/')
    print('  v95_v7_*.pkl      - Modelos V7 mejorados')
    print('  v95_ld_*.pkl      - LossDetector por par')
    print('  v95_meta.json     - Metadata')

    # Comparar con V7 original
    print('\n[Comparacion V7 original vs V9.5]')
    try:
        with open(MODELS_DIR / 'v7_meta.json') as f:
            v7_orig = json.load(f)
        for pair in WEAK_PAIRS:
            if pair in v7_models and pair in v7_orig.get('stats', {}):
                orig_corr = v7_orig['stats'][pair]['corr']
                new_corr = v7_models[pair]['corr']
                diff = new_corr - orig_corr
                emoji = '+' if diff > 0 else ''
                print(f'  {pair:<12} {orig_corr:.3f} -> {new_corr:.3f} ({emoji}{diff:.3f})')
    except:
        pass


if __name__ == '__main__':
    main()
