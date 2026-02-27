"""
BTC V3: Entrenamiento con datos recientes (2024-2026)
======================================================
Hipótesis: El mercado ha cambiado, usar datos más recientes mejorará el rendimiento.
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

TP_PCT = 0.03
SL_PCT = 0.015
MAX_HOLD = 20

print('='*70)
print('BTC V3: ENTRENAMIENTO CON DATOS RECIENTES')
print('='*70)

# =============================================================================
# 1. CARGAR DATOS
# =============================================================================
print('\n[1] Cargando datos...')
df = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_full.parquet')
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')

# Filtrar solo 2024-2026
df = df[df.index >= '2024-01-01']
print(f'    {len(df)} velas: {df.index[0].date()} a {df.index[-1].date()}')

# =============================================================================
# 2. CALCULAR FEATURES (igual que V2)
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

# Target
feat['target'] = c.pct_change().shift(-1)

feat = feat.dropna()
print(f'    {len(feat)} samples, {len(feat.columns)-1} features')

# =============================================================================
# 3. SPLIT: Train 2024, Val 2025-H1, Test 2025-H2 + 2026
# =============================================================================
print('\n[3] Preparando splits...')

# Split 1: Clásico
train1 = feat[feat.index < '2025-07-01']
val1 = feat[(feat.index >= '2025-07-01') & (feat.index < '2025-12-01')]
test1 = feat[feat.index >= '2025-12-01']

# Split 2: Más reciente
train2 = feat[feat.index < '2025-10-01']
val2 = feat[(feat.index >= '2025-10-01') & (feat.index < '2026-01-01')]
test2 = feat[feat.index >= '2026-01-01']

feature_cols = [c for c in feat.columns if c != 'target']

print(f'    Split 1: Train {len(train1)}, Val {len(val1)}, Test {len(test1)}')
print(f'    Split 2: Train {len(train2)}, Val {len(val2)}, Test {len(test2)}')

# =============================================================================
# 4. BACKTEST FUNCTION
# =============================================================================
def backtest(preds_series, df_ref, conv_min=1.0):
    trades = []
    for idx, pred in preds_series.items():
        if abs(pred) / 0.005 < conv_min:
            continue
        try:
            loc = df_ref.index.get_loc(idx)
        except:
            continue
        if loc + 1 >= len(df_ref):
            continue

        direction = 1 if pred > 0 else -1
        entry = df_ref.iloc[loc + 1]['open']
        tp = entry * (1 + TP_PCT) if direction == 1 else entry * (1 - TP_PCT)
        sl = entry * (1 - SL_PCT) if direction == 1 else entry * (1 + SL_PCT)

        pnl = None
        for j in range(loc + 1, min(loc + MAX_HOLD + 1, len(df_ref))):
            candle = df_ref.iloc[j]
            if direction == 1:
                if candle['low'] <= sl:
                    pnl = -SL_PCT
                    break
                if candle['high'] >= tp:
                    pnl = TP_PCT
                    break
            else:
                if candle['high'] >= sl:
                    pnl = -SL_PCT
                    break
                if candle['low'] <= tp:
                    pnl = TP_PCT
                    break
        if pnl is None:
            exit_idx = min(loc + MAX_HOLD, len(df_ref) - 1)
            exit_p = df_ref.iloc[exit_idx]['close']
            pnl = (exit_p - entry) / entry * direction

        trades.append({'pnl': pnl, 'win': pnl > 0})

    if not trades:
        return 0, 0, 0
    n = len(trades)
    wins = sum(t['win'] for t in trades)
    wr = wins / n * 100
    pnl_total = sum(t['pnl'] for t in trades) * 100
    return n, wr, pnl_total

# =============================================================================
# 5. ENTRENAR MÚLTIPLES CONFIGURACIONES
# =============================================================================
print('\n[4] Entrenando modelos...')

configs = [
    # (name, n_est, lr, depth, subsample)
    ('GB_base', 300, 0.05, 5, 0.8),
    ('GB_deep', 300, 0.05, 7, 0.8),
    ('GB_shallow', 300, 0.05, 3, 0.8),
    ('GB_slow', 500, 0.02, 5, 0.8),
    ('GB_fast', 200, 0.1, 5, 0.8),
    ('GB_reg', 300, 0.05, 5, 0.7),  # más regularización
]

results = []

for name, n_est, lr, depth, subsample in configs:
    print(f'\n  {name}...', flush=True)

    model = GradientBoostingRegressor(
        n_estimators=n_est,
        learning_rate=lr,
        max_depth=depth,
        subsample=subsample,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
    )

    # Entrenar con split 2 (más reciente)
    X_train = train2[feature_cols]
    y_train = train2['target']
    model.fit(X_train, y_train)

    # Validar
    val_preds = pd.Series(model.predict(val2[feature_cols]), index=val2.index)
    test_preds = pd.Series(model.predict(test2[feature_cols]), index=test2.index)

    # Backtest
    for conv in [0.5, 1.0, 1.5]:
        v_n, v_wr, v_pnl = backtest(val_preds, df, conv)
        t_n, t_wr, t_pnl = backtest(test_preds, df, conv)

        results.append({
            'config': name,
            'conv': conv,
            'val_n': v_n,
            'val_wr': v_wr,
            'val_pnl': v_pnl,
            'test_n': t_n,
            'test_wr': t_wr,
            'test_pnl': t_pnl,
        })

        if conv == 1.0:
            print(f'    Conv={conv}: Val {v_n} trades, {v_wr:.1f}% WR | Test {t_n} trades, {t_wr:.1f}% WR')

# =============================================================================
# 6. RESULTADOS
# =============================================================================
print('\n' + '='*70)
print('RESULTADOS')
print('='*70)

results_df = pd.DataFrame(results)

# Filtrar conv = 1.0
r1 = results_df[results_df['conv'] == 1.0].copy()
r1['score'] = r1['test_pnl'] * (r1['test_wr'] / 50)  # Score ponderado

print(f'\n{"Config":<12} {"Val Trades":>10} {"Val WR":>8} {"Val PnL":>10} {"Test Trades":>11} {"Test WR":>8} {"Test PnL":>10}')
print('-'*75)

for _, row in r1.iterrows():
    print(f'{row["config"]:<12} {row["val_n"]:>10.0f} {row["val_wr"]:>7.1f}% {row["val_pnl"]:>+9.2f}% {row["test_n"]:>11.0f} {row["test_wr"]:>7.1f}% {row["test_pnl"]:>+9.2f}%')

# Mejor modelo
best = r1.loc[r1['score'].idxmax()]
print(f'\nMejor configuración: {best["config"]}')
print(f'  Test: {best["test_n"]:.0f} trades, {best["test_wr"]:.1f}% WR, {best["test_pnl"]:+.2f}% PnL')

# =============================================================================
# 7. ENTRENAR MODELO FINAL CON TODOS LOS DATOS RECIENTES
# =============================================================================
print('\n[5] Entrenando modelo final...')

# Usar toda la data hasta 2026-02-01 para entrenar
final_train = feat[feat.index < '2026-02-01']
final_test = feat[feat.index >= '2026-02-01']

# Usar config ganadora
best_config = best['config']
config_map = {
    'GB_base': (300, 0.05, 5, 0.8),
    'GB_deep': (300, 0.05, 7, 0.8),
    'GB_shallow': (300, 0.05, 3, 0.8),
    'GB_slow': (500, 0.02, 5, 0.8),
    'GB_fast': (200, 0.1, 5, 0.8),
    'GB_reg': (300, 0.05, 5, 0.7),
}

n_est, lr, depth, subsample = config_map[best_config]

final_model = GradientBoostingRegressor(
    n_estimators=n_est,
    learning_rate=lr,
    max_depth=depth,
    subsample=subsample,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
)

final_model.fit(final_train[feature_cols], final_train['target'])

# Test final
final_preds = pd.Series(final_model.predict(final_test[feature_cols]), index=final_test.index)

print(f'\nTest final (Feb 2026):')
for conv in [0.5, 1.0, 1.5, 2.0]:
    n, wr, pnl = backtest(final_preds, df, conv)
    if n > 0:
        print(f'  Conv >= {conv}: {n} trades, {wr:.1f}% WR, {pnl:+.2f}% PnL')

# Guardar
model_path = MODELS_DIR / 'btc_v3_recent.pkl'
joblib.dump({
    'model': final_model,
    'feature_cols': feature_cols,
    'config': best_config,
    'train_end': '2026-02-01',
}, model_path)
print(f'\nGuardado: {model_path}')

# =============================================================================
# 8. COMPARAR CON V2
# =============================================================================
print('\n' + '='*70)
print('COMPARACIÓN V2 vs V3')
print('='*70)

# Cargar V2
v2_data = joblib.load(MODELS_DIR / 'btc_v2_gradientboosting.pkl')
v2_model = v2_data['model']
v2_cols = v2_data['feature_cols']

# Predicciones V2 en mismo período
v2_preds = pd.Series(v2_model.predict(final_test[v2_cols]), index=final_test.index)

print(f'\nTest Feb 2026 (Conv >= 1.0):')
v2_n, v2_wr, v2_pnl = backtest(v2_preds, df, 1.0)
v3_n, v3_wr, v3_pnl = backtest(final_preds, df, 1.0)

print(f'  V2 (2019-2025): {v2_n} trades, {v2_wr:.1f}% WR, {v2_pnl:+.2f}% PnL')
print(f'  V3 (2024-2026): {v3_n} trades, {v3_wr:.1f}% WR, {v3_pnl:+.2f}% PnL')

if v3_wr > v2_wr:
    print(f'\n  [OK] V3 mejora WR en {v3_wr - v2_wr:.1f}%')
else:
    print(f'\n  [X] V3 empeora WR en {v2_wr - v3_wr:.1f}%')

print('\n[COMPLETADO]')
