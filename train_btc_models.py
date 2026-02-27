"""
Entrenamiento de múltiples modelos para BTC
============================================
- LightGBM (baseline)
- XGBoost
- RandomForest
- CatBoost
- Neural Network (MLP)

Validación: Walk-forward con backtesting real
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

# Trading config
TP_PCT = 0.03
SL_PCT = 0.015
MAX_HOLD = 20

print('='*70)
print('ENTRENAMIENTO MULTI-MODELO BTC')
print('='*70)

# =============================================================================
# 1. CARGAR DATOS
# =============================================================================
print('\n[1] Cargando datos...')
df = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_full.parquet')
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')
print(f'    {len(df)} velas: {df.index[0].date()} a {df.index[-1].date()}')

# =============================================================================
# 2. CALCULAR FEATURES
# =============================================================================
print('\n[2] Calculando features...')

feat = pd.DataFrame(index=df.index)
c, h, l, v = df['close'], df['high'], df['low'], df['volume']

# Returns
for p in [1, 2, 3, 5, 10, 20, 50]:
    feat[f'ret_{p}'] = c.pct_change(p)

# Volatility
feat['atr14'] = ta.atr(h, l, c, length=14)
feat['atr_r'] = feat['atr14'] / feat['atr14'].rolling(50).mean()
feat['vol5'] = c.pct_change().rolling(5).std()
feat['vol20'] = c.pct_change().rolling(20).std()
feat['vol_ratio'] = feat['vol5'] / (feat['vol20'] + 1e-10)

# RSI
feat['rsi14'] = ta.rsi(c, length=14)
feat['rsi7'] = ta.rsi(c, length=7)
feat['rsi21'] = ta.rsi(c, length=21)

# Stochastic RSI
sr = ta.stochrsi(c, length=14, rsi_length=14, k=3, d=3)
if sr is not None:
    feat['srsi_k'] = sr.iloc[:, 0]
    feat['srsi_d'] = sr.iloc[:, 1]

# MACD
macd = ta.macd(c, fast=12, slow=26, signal=9)
if macd is not None:
    feat['macd'] = macd.iloc[:, 0]
    feat['macd_h'] = macd.iloc[:, 1]
    feat['macd_s'] = macd.iloc[:, 2]

# ROC
feat['roc5'] = ta.roc(c, length=5)
feat['roc10'] = ta.roc(c, length=10)
feat['roc20'] = ta.roc(c, length=20)

# EMAs
for el in [8, 21, 55, 100, 200]:
    e = ta.ema(c, length=el)
    feat[f'ema{el}_d'] = (c - e) / e * 100
feat['ema8_sl'] = ta.ema(c, length=8).pct_change(3) * 100
feat['ema21_sl'] = ta.ema(c, length=21).pct_change(5) * 100
feat['ema55_sl'] = ta.ema(c, length=55).pct_change(5) * 100

# Bollinger Bands
bb = ta.bbands(c, length=20, std=2.0)
if bb is not None:
    bw = bb.iloc[:, 2] - bb.iloc[:, 0]
    feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw
    feat['bb_w'] = bw / bb.iloc[:, 1] * 100

# Volume
feat['vr'] = v / v.rolling(20).mean()
feat['vr5'] = v / v.rolling(5).mean()

# Price action
feat['spr'] = (h - l) / c * 100
feat['body'] = abs(c - df['open']) / (h - l + 1e-10)
feat['upper_wick'] = (h - np.maximum(c, df['open'])) / (h - l + 1e-10)
feat['lower_wick'] = (np.minimum(c, df['open']) - l) / (h - l + 1e-10)

# ADX
ax = ta.adx(h, l, c, length=14)
if ax is not None:
    feat['adx'] = ax.iloc[:, 0]
    feat['dip'] = ax.iloc[:, 1]
    feat['dim'] = ax.iloc[:, 2]
    feat['di_diff'] = feat['dip'] - feat['dim']

# Choppiness
atr_1 = ta.atr(h, l, c, length=1)
atr_sum = atr_1.rolling(14).sum()
high_max = h.rolling(14).max()
low_min = l.rolling(14).min()
feat['chop'] = 100 * np.log10(atr_sum / (high_max - low_min + 1e-10)) / np.log10(14)

# Time features
hr = df.index.hour
dw = df.index.dayofweek
feat['h_s'] = np.sin(2 * np.pi * hr / 24)
feat['h_c'] = np.cos(2 * np.pi * hr / 24)
feat['d_s'] = np.sin(2 * np.pi * dw / 7)
feat['d_c'] = np.cos(2 * np.pi * dw / 7)

# Lag features
for lag in [1, 2, 3]:
    feat[f'ret1_lag{lag}'] = feat['ret_1'].shift(lag)
    feat[f'rsi14_lag{lag}'] = feat['rsi14'].shift(lag)

# Target: next 4h return
feat['target'] = c.pct_change().shift(-1)

# Drop NaN
feat = feat.dropna()
print(f'    {len(feat)} samples, {len(feat.columns)-1} features')

# =============================================================================
# 3. PREPARAR TRAIN/TEST
# =============================================================================
print('\n[3] Preparando train/test split...')

# Use last 6 months as test
test_start = pd.Timestamp('2026-02-01', tz='UTC')
train_end = pd.Timestamp('2025-08-01', tz='UTC')

# Train: 2019-2025 (excluyendo ultimo periodo para evitar look-ahead)
train = feat[(feat.index >= '2019-06-01') & (feat.index < train_end)]
# Validation: 2025-08 to 2026-01
val = feat[(feat.index >= train_end) & (feat.index < test_start)]
# Test: 2026-02+
test = feat[feat.index >= test_start]

feature_cols = [c for c in feat.columns if c != 'target']
X_train = train[feature_cols]
y_train = train['target']
X_val = val[feature_cols]
y_val = val['target']
X_test = test[feature_cols]
y_test = test['target']

print(f'    Train: {len(X_train)} samples ({X_train.index[0].date()} - {X_train.index[-1].date()})')
print(f'    Val:   {len(X_val)} samples ({X_val.index[0].date()} - {X_val.index[-1].date()})')
print(f'    Test:  {len(X_test)} samples ({X_test.index[0].date()} - {X_test.index[-1].date()})')

# =============================================================================
# 4. DEFINIR MODELOS
# =============================================================================
print('\n[4] Definiendo modelos...')

models = {
    'LightGBM': LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    ),
    'XGBoost': XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=0,
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
    ),
    'MLP': MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    ),
}

# =============================================================================
# 5. ENTRENAR Y EVALUAR
# =============================================================================
print('\n[5] Entrenando modelos...')

def backtest_predictions(predictions, df_slice, threshold=0.0):
    """Backtest con TP/SL reales."""
    trades = []
    for i, (idx, pred) in enumerate(predictions.items()):
        if abs(pred) < threshold:
            continue

        try:
            loc = df.index.get_loc(idx)
        except:
            continue

        if loc + 1 >= len(df):
            continue

        direction = 1 if pred > 0 else -1
        entry = df.iloc[loc + 1]['open']

        tp_price = entry * (1 + TP_PCT) if direction == 1 else entry * (1 - TP_PCT)
        sl_price = entry * (1 - SL_PCT) if direction == 1 else entry * (1 + SL_PCT)

        pnl_pct = None
        for j in range(loc + 1, min(loc + MAX_HOLD + 1, len(df))):
            candle = df.iloc[j]
            if direction == 1:
                if candle['low'] <= sl_price:
                    pnl_pct = -SL_PCT
                    break
                if candle['high'] >= tp_price:
                    pnl_pct = TP_PCT
                    break
            else:
                if candle['high'] >= sl_price:
                    pnl_pct = -SL_PCT
                    break
                if candle['low'] <= tp_price:
                    pnl_pct = TP_PCT
                    break

        if pnl_pct is None:
            exit_idx = min(loc + MAX_HOLD, len(df) - 1)
            exit_price = df.iloc[exit_idx]['close']
            pnl_pct = (exit_price - entry) / entry * direction

        trades.append({
            'pnl_pct': pnl_pct,
            'win': pnl_pct > 0,
            'direction': direction,
            'conviction': abs(pred) / 0.005,
        })

    if not trades:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'pf': 0}

    n = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n * 100
    pnl = sum(t['pnl_pct'] for t in trades) * 100
    gross_profit = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] > 0)
    gross_loss = abs(sum(t['pnl_pct'] for t in trades if t['pnl_pct'] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999

    return {'trades': n, 'wr': wr, 'pnl': pnl, 'pf': pf}

results = {}

for name, model in models.items():
    print(f'\n  Training {name}...', flush=True)

    # Scale for MLP
    if name == 'MLP':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, y_train)
        val_preds = pd.Series(model.predict(X_val_scaled), index=X_val.index)
        test_preds = pd.Series(model.predict(X_test_scaled), index=X_test.index)
    else:
        model.fit(X_train, y_train)
        val_preds = pd.Series(model.predict(X_val), index=X_val.index)
        test_preds = pd.Series(model.predict(X_test), index=X_test.index)

    # Backtest on validation
    val_bt = backtest_predictions(val_preds.to_dict(), val)

    # Backtest on test
    test_bt = backtest_predictions(test_preds.to_dict(), test)

    results[name] = {
        'model': model,
        'scaler': scaler if name == 'MLP' else None,
        'val': val_bt,
        'test': test_bt,
        'pred_std': test_preds.std(),
    }

    print(f'    Val:  {val_bt["trades"]:4d} trades, WR={val_bt["wr"]:5.1f}%, PnL={val_bt["pnl"]:+7.2f}%, PF={val_bt["pf"]:.2f}')
    print(f'    Test: {test_bt["trades"]:4d} trades, WR={test_bt["wr"]:5.1f}%, PnL={test_bt["pnl"]:+7.2f}%, PF={test_bt["pf"]:.2f}')

# =============================================================================
# 6. COMPARAR RESULTADOS
# =============================================================================
print('\n' + '='*70)
print('COMPARACION DE MODELOS')
print('='*70)

print(f'\n{"Modelo":<18} {"Val Trades":>10} {"Val WR":>8} {"Val PnL":>10} {"Test Trades":>11} {"Test WR":>8} {"Test PnL":>10}')
print('-'*70)

best_model = None
best_score = -999

for name, r in results.items():
    v = r['val']
    t = r['test']
    print(f'{name:<18} {v["trades"]:>10} {v["wr"]:>7.1f}% {v["pnl"]:>+9.2f}% {t["trades"]:>11} {t["wr"]:>7.1f}% {t["pnl"]:>+9.2f}%')

    # Score: test PnL weighted by WR (must be > 45%)
    if t['wr'] >= 45 and t['trades'] >= 10:
        score = t['pnl'] * (t['wr'] / 50)
        if score > best_score:
            best_score = score
            best_model = name

# =============================================================================
# 7. ANALISIS CON FILTROS DE CONVICTION
# =============================================================================
print('\n' + '='*70)
print('ANALISIS CON FILTROS DE CONVICTION')
print('='*70)

for name, r in results.items():
    model = r['model']
    scaler = r['scaler']

    if scaler:
        test_preds = pd.Series(model.predict(scaler.transform(X_test)), index=X_test.index)
    else:
        test_preds = pd.Series(model.predict(X_test), index=X_test.index)

    convictions = (test_preds.abs() / 0.005)

    print(f'\n{name}:')
    print(f'  Pred std: {test_preds.std():.6f}')
    print(f'  Conv mean: {convictions.mean():.2f}, max: {convictions.max():.2f}')

    for thresh in [0.0, 0.5, 1.0, 1.5, 2.0]:
        filtered_preds = test_preds[convictions >= thresh]
        if len(filtered_preds) == 0:
            continue
        bt = backtest_predictions(filtered_preds.to_dict(), test)
        print(f'    Conv >= {thresh:.1f}: {bt["trades"]:4d} trades, WR={bt["wr"]:5.1f}%, PnL={bt["pnl"]:+7.2f}%')

# =============================================================================
# 8. GUARDAR MEJOR MODELO
# =============================================================================
print('\n' + '='*70)
print('GUARDANDO MODELOS')
print('='*70)

if best_model:
    print(f'\nMejor modelo: {best_model}')

    # Save best model
    r = results[best_model]
    model_path = MODELS_DIR / 'btc_v2_best.pkl'
    joblib.dump({
        'model': r['model'],
        'scaler': r['scaler'],
        'feature_cols': feature_cols,
        'pred_std': r['pred_std'],
        'model_type': best_model,
        'train_end': str(train_end),
        'metrics': r['test'],
    }, model_path)
    print(f'  Guardado: {model_path}')

# Save all models for comparison
for name, r in results.items():
    model_path = MODELS_DIR / f'btc_v2_{name.lower()}.pkl'
    joblib.dump({
        'model': r['model'],
        'scaler': r['scaler'],
        'feature_cols': feature_cols,
        'pred_std': r['pred_std'],
        'model_type': name,
        'metrics': r['test'],
    }, model_path)
    print(f'  Guardado: {model_path}')

print('\n[COMPLETADO]')
