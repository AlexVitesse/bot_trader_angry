"""
Análisis profundo del modelo BTC V2 GradientBoosting
=====================================================
1. Feature importance
2. Rendimiento por régimen (bull/bear/range)
3. Rendimiento por año
4. Análisis de errores
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

# Trading config
TP_PCT = 0.03
SL_PCT = 0.015
MAX_HOLD = 20

print('='*70)
print('ANÁLISIS PROFUNDO BTC V2 - GradientBoosting')
print('='*70)

# =============================================================================
# 1. CARGAR MODELO Y DATOS
# =============================================================================
print('\n[1] Cargando modelo y datos...')

model_data = joblib.load(MODELS_DIR / 'btc_v2_gradientboosting.pkl')
model = model_data['model']
feature_cols = model_data['feature_cols']

df = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_full.parquet')
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')

print(f'    Modelo: GradientBoosting')
print(f'    Features: {len(feature_cols)}')
print(f'    Data: {len(df)} velas')

# =============================================================================
# 2. FEATURE IMPORTANCE
# =============================================================================
print('\n[2] Feature Importance (Top 20)...')

importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f'\n{"Rank":<5} {"Feature":<20} {"Importance":>12}')
print('-'*40)
for i, (_, row) in enumerate(importance.head(20).iterrows()):
    print(f'{i+1:<5} {row["feature"]:<20} {row["importance"]:>12.4f}')

# Guardar importance completa
importance.to_csv(MODELS_DIR / 'btc_v2_feature_importance.csv', index=False)

# =============================================================================
# 3. RECALCULAR FEATURES
# =============================================================================
print('\n[3] Calculando features...')

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

feat = feat.dropna()

# =============================================================================
# 4. DETECTAR RÉGIMEN
# =============================================================================
print('\n[4] Detectando régimen de mercado...')

# Calcular régimen basado en EMAs y retornos
ema20 = ta.ema(c, length=20)
ema50 = ta.ema(c, length=50)
ret_20d = c.pct_change(20 * 6)  # 20 días en velas de 4h

regime = pd.Series(index=df.index, dtype=str)
for i in range(len(df)):
    if i < 200:
        regime.iloc[i] = 'UNKNOWN'
        continue

    price = c.iloc[i]
    e20 = ema20.iloc[i]
    e50 = ema50.iloc[i]
    r20 = ret_20d.iloc[i]

    if pd.isna(e20) or pd.isna(e50) or pd.isna(r20):
        regime.iloc[i] = 'UNKNOWN'
    elif price > e20 and e20 > e50 and r20 > 0.05:
        regime.iloc[i] = 'BULL'
    elif price < e20 and e20 < e50 and r20 < -0.05:
        regime.iloc[i] = 'BEAR'
    else:
        regime.iloc[i] = 'RANGE'

df['regime'] = regime

print('    Distribución de régimen:')
print(regime.value_counts())

# =============================================================================
# 5. BACKTEST FUNCTION
# =============================================================================
def backtest(predictions_df, conviction_min=1.0):
    """Backtest con TP/SL."""
    trades = []

    for idx, row in predictions_df.iterrows():
        pred = row['pred']
        conviction = abs(pred) / 0.005

        if conviction < conviction_min:
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
        exit_reason = None

        for j in range(loc + 1, min(loc + MAX_HOLD + 1, len(df))):
            candle = df.iloc[j]
            if direction == 1:
                if candle['low'] <= sl_price:
                    pnl_pct = -SL_PCT
                    exit_reason = 'SL'
                    break
                if candle['high'] >= tp_price:
                    pnl_pct = TP_PCT
                    exit_reason = 'TP'
                    break
            else:
                if candle['high'] >= sl_price:
                    pnl_pct = -SL_PCT
                    exit_reason = 'SL'
                    break
                if candle['low'] <= tp_price:
                    pnl_pct = TP_PCT
                    exit_reason = 'TP'
                    break

        if pnl_pct is None:
            exit_idx = min(loc + MAX_HOLD, len(df) - 1)
            exit_price = df.iloc[exit_idx]['close']
            pnl_pct = (exit_price - entry) / entry * direction
            exit_reason = 'TIME'

        trades.append({
            'ts': idx,
            'direction': direction,
            'pnl_pct': pnl_pct,
            'win': pnl_pct > 0,
            'regime': df.loc[idx, 'regime'] if idx in df.index else 'UNKNOWN',
        })

    return pd.DataFrame(trades) if trades else pd.DataFrame()

# =============================================================================
# 6. GENERAR PREDICCIONES COMPLETAS
# =============================================================================
print('\n[5] Generando predicciones...')

# Filtrar por features disponibles
feat_aligned = feat[feature_cols].dropna()
predictions = pd.DataFrame({
    'pred': model.predict(feat_aligned),
}, index=feat_aligned.index)

print(f'    {len(predictions)} predicciones generadas')

# =============================================================================
# 7. BACKTEST POR RÉGIMEN
# =============================================================================
print('\n[6] Backtest por régimen de mercado...')

# Agregar régimen a predictions
predictions['regime'] = df.loc[predictions.index, 'regime']

print(f'\n{"Régimen":<10} {"Trades":>8} {"WR":>8} {"PnL":>10} {"PF":>8}')
print('-'*50)

for reg in ['BULL', 'BEAR', 'RANGE']:
    reg_preds = predictions[predictions['regime'] == reg].copy()
    if len(reg_preds) == 0:
        continue

    trades_df = backtest(reg_preds, conviction_min=1.0)

    if len(trades_df) > 0:
        n = len(trades_df)
        wins = trades_df['win'].sum()
        wr = wins / n * 100
        pnl = trades_df['pnl_pct'].sum() * 100
        gp = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
        gl = abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())
        pf = gp / gl if gl > 0 else 999
        print(f'{reg:<10} {n:>8} {wr:>7.1f}% {pnl:>+9.2f}% {pf:>7.2f}')
    else:
        print(f'{reg:<10}        0       -          -        -')

# =============================================================================
# 8. BACKTEST POR AÑO
# =============================================================================
print('\n[7] Backtest por año...')

predictions['year'] = predictions.index.year

print(f'\n{"Año":<6} {"Trades":>8} {"WR":>8} {"PnL":>10} {"PF":>8}')
print('-'*45)

for year in sorted(predictions['year'].unique()):
    year_preds = predictions[predictions['year'] == year].copy()
    trades_df = backtest(year_preds, conviction_min=1.0)

    if len(trades_df) > 0:
        n = len(trades_df)
        wins = trades_df['win'].sum()
        wr = wins / n * 100
        pnl = trades_df['pnl_pct'].sum() * 100
        gp = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
        gl = abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())
        pf = gp / gl if gl > 0 else 999
        print(f'{year:<6} {n:>8} {wr:>7.1f}% {pnl:>+9.2f}% {pf:>7.2f}')

# =============================================================================
# 9. ANÁLISIS DE DIRECCIÓN
# =============================================================================
print('\n[8] Análisis por dirección...')

all_trades = backtest(predictions, conviction_min=1.0)

if len(all_trades) > 0:
    print(f'\n{"Dir":<8} {"Trades":>8} {"WR":>8} {"PnL":>10}')
    print('-'*40)

    for dir_val, dir_name in [(1, 'LONG'), (-1, 'SHORT')]:
        dir_trades = all_trades[all_trades['direction'] == dir_val]
        if len(dir_trades) > 0:
            n = len(dir_trades)
            wr = dir_trades['win'].mean() * 100
            pnl = dir_trades['pnl_pct'].sum() * 100
            print(f'{dir_name:<8} {n:>8} {wr:>7.1f}% {pnl:>+9.2f}%')

# =============================================================================
# 10. RESUMEN
# =============================================================================
print('\n' + '='*70)
print('RESUMEN')
print('='*70)

total_trades = backtest(predictions, conviction_min=1.0)
if len(total_trades) > 0:
    n = len(total_trades)
    wins = total_trades['win'].sum()
    wr = wins / n * 100
    pnl = total_trades['pnl_pct'].sum() * 100

    print(f'\nTotal trades (conv >= 1.0): {n}')
    print(f'Win Rate: {wr:.1f}%')
    print(f'PnL total: {pnl:+.2f}%')

    # Por régimen
    print('\nMejor régimen:')
    for reg in ['BULL', 'BEAR', 'RANGE']:
        reg_trades = total_trades[total_trades['regime'] == reg]
        if len(reg_trades) >= 10:
            reg_wr = reg_trades['win'].mean() * 100
            print(f'  {reg}: {len(reg_trades)} trades, WR={reg_wr:.1f}%')

print('\n[COMPLETADO]')
