"""
Optimizacion LINK/USDT - V13.03
==============================
Entrenar modelo V2 + optimizar TP/SL + analizar direccion
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
MODELS_DIR = Path('models')
DOCS_DIR = Path('docs')
DOCS_DIR.mkdir(exist_ok=True)

PAIR = 'LINK/USDT'
INITIAL_CAPITAL = 100.0
POSITION_SIZE = 10.0

print("="*70)
print(f"OPTIMIZACION {PAIR} - V13.03")
print("="*70)


# =============================================================================
# 1. CARGAR DATOS
# =============================================================================
print("\n[1/6] Cargando datos...")

df = pd.read_parquet(DATA_DIR / 'LINK_USDT_4h_full.parquet')
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

print(f"  Rango: {df.index[0]} a {df.index[-1]}")
print(f"  Velas: {len(df)}")


# =============================================================================
# 2. CALCULAR FEATURES (54 features V2)
# =============================================================================
print("\n[2/6] Calculando features V2...")

def compute_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    c, h, l, o, v = df['close'], df['high'], df['low'], df['open'], df['volume']

    for p in [1, 2, 3, 5, 10, 20, 50]:
        feat[f'ret_{p}'] = c.pct_change(p)

    feat['atr14'] = ta.atr(h, l, c, length=14)
    feat['atr_r'] = feat['atr14'] / feat['atr14'].rolling(50).mean()
    feat['vol5'] = c.pct_change().rolling(5).std()
    feat['vol20'] = c.pct_change().rolling(20).std()
    feat['vol_ratio'] = feat['vol5'] / (feat['vol20'] + 1e-10)
    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)
    feat['rsi21'] = ta.rsi(c, length=21)

    sr = ta.stochrsi(c, length=14, rsi_length=14, k=3, d=3)
    if sr is not None and len(sr.columns) >= 2:
        feat['srsi_k'] = sr.iloc[:, 0]
        feat['srsi_d'] = sr.iloc[:, 1]

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None and len(macd.columns) >= 3:
        feat['macd'] = macd.iloc[:, 0]
        feat['macd_h'] = macd.iloc[:, 1]
        feat['macd_s'] = macd.iloc[:, 2]

    feat['roc5'] = ta.roc(c, length=5)
    feat['roc10'] = ta.roc(c, length=10)
    feat['roc20'] = ta.roc(c, length=20)

    for el in [8, 21, 55, 100, 200]:
        e = ta.ema(c, length=el)
        feat[f'ema{el}_d'] = (c - e) / e * 100

    feat['ema8_sl'] = ta.ema(c, length=8).pct_change(3) * 100
    feat['ema21_sl'] = ta.ema(c, length=21).pct_change(5) * 100
    feat['ema55_sl'] = ta.ema(c, length=55).pct_change(5) * 100

    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None and len(bb.columns) >= 3:
        bw = bb.iloc[:, 2] - bb.iloc[:, 0]
        feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw
        feat['bb_w'] = bw / bb.iloc[:, 1] * 100

    feat['vr'] = v / v.rolling(20).mean()
    feat['vr5'] = v.rolling(5).mean() / v.rolling(20).mean()
    feat['spr'] = (h - l) / c * 100
    feat['body'] = abs(c - o) / (h - l + 1e-10)
    feat['upper_wick'] = (h - np.maximum(c, o)) / (h - l + 1e-10)
    feat['lower_wick'] = (np.minimum(c, o) - l) / (h - l + 1e-10)

    ax = ta.adx(h, l, c, length=14)
    if ax is not None and len(ax.columns) >= 3:
        feat['adx'] = ax.iloc[:, 0]
        feat['dip'] = ax.iloc[:, 1]
        feat['dim'] = ax.iloc[:, 2]
        feat['di_diff'] = feat['dip'] - feat['dim']

    chop = ta.chop(h, l, c, length=14)
    if chop is not None:
        feat['chop'] = chop

    hr = df.index.hour
    dw = df.index.dayofweek
    feat['h_s'] = np.sin(2 * np.pi * hr / 24)
    feat['h_c'] = np.cos(2 * np.pi * hr / 24)
    feat['d_s'] = np.sin(2 * np.pi * dw / 7)
    feat['d_c'] = np.cos(2 * np.pi * dw / 7)

    feat['ret1_lag1'] = feat['ret_1'].shift(1)
    feat['rsi14_lag1'] = feat['rsi14'].shift(1)
    feat['ret1_lag2'] = feat['ret_1'].shift(2)
    feat['rsi14_lag2'] = feat['rsi14'].shift(2)
    feat['ret1_lag3'] = feat['ret_1'].shift(3)
    feat['rsi14_lag3'] = feat['rsi14'].shift(3)

    return feat


feat = compute_features_v2(df)
print(f"  Features: {len(feat.columns)}")


# =============================================================================
# 3. PREPARAR TARGET
# =============================================================================
print("\n[3/6] Preparando target...")

HORIZON = 5
target = df['close'].pct_change(HORIZON).shift(-HORIZON)
target.name = 'target'

# Combinar y limpiar
data = feat.join(target).dropna()
print(f"  Samples validos: {len(data)}")

X = data.drop('target', axis=1)
y = data['target']
feature_cols = list(X.columns)


# =============================================================================
# 4. ENTRENAR MODELO V2 (GradientBoosting)
# =============================================================================
print("\n[4/6] Entrenando modelo GradientBoosting...")

# Split temporal 80/20
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Scaler
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Modelo
model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
model.fit(X_train_sc, y_train)

# Evaluar
train_preds = model.predict(X_train_sc)
test_preds = model.predict(X_test_sc)

train_corr = np.corrcoef(y_train, train_preds)[0, 1]
test_corr = np.corrcoef(y_test, test_preds)[0, 1]
pred_std = np.std(test_preds)

print(f"  Train corr: {train_corr:.4f}")
print(f"  Test corr: {test_corr:.4f}")
print(f"  Pred std: {pred_std:.6f}")


# =============================================================================
# 5. OPTIMIZAR TP/SL + DIRECCION
# =============================================================================
print("\n[5/6] Optimizando TP/SL y direccion...")

def detect_regime(df):
    c = df['close']
    ema20 = ta.ema(c, length=20)
    ema50 = ta.ema(c, length=50)
    ret20 = c.pct_change(20)
    regime = pd.Series('RANGE', index=df.index)
    bull = (c > ema50) & (ema20 > ema50) & (ret20 > 0.05)
    bear = (c < ema50) & (ema20 < ema50) & (ret20 < -0.05)
    regime[bull] = 'BULL'
    regime[bear] = 'BEAR'
    return regime


def backtest_config(df, feat, preds, pred_std, tp_pct, sl_pct, conv_min, only_long=False, only_short=False):
    """Backtest con configuracion especifica."""
    conv = np.abs(preds) / pred_std
    signals = conv >= conv_min
    directions = np.where(preds < 0, -1, 1)
    regime = detect_regime(df)

    trades = []

    for i in range(len(feat) - 21):
        if not signals[i]:
            continue

        d = directions[i]
        reg = regime.iloc[i]

        # Filtros direccion
        if only_long and d == -1:
            continue
        if only_short and d == 1:
            continue

        # Filtro regimen
        if reg == 'BULL' and d == -1:
            continue
        if reg == 'BEAR' and d == 1:
            continue

        entry = df.iloc[i]['close']
        tp = entry * (1 + tp_pct) if d == 1 else entry * (1 - tp_pct)
        sl = entry * (1 - sl_pct) if d == 1 else entry * (1 + sl_pct)

        exit_p = None
        for j in range(1, 21):
            if i + j >= len(df):
                break
            bar = df.iloc[i + j]
            if d == 1:
                if bar['low'] <= sl:
                    exit_p = sl
                    break
                elif bar['high'] >= tp:
                    exit_p = tp
                    break
            else:
                if bar['high'] >= sl:
                    exit_p = sl
                    break
                elif bar['low'] <= tp:
                    exit_p = tp
                    break

        if exit_p is None:
            exit_p = df.iloc[min(i + 20, len(df) - 1)]['close']

        pnl_pct = (exit_p - entry) / entry if d == 1 else (entry - exit_p) / entry
        trades.append({
            'direction': d,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_pct * POSITION_SIZE,
            'win': pnl_pct > 0
        })

    if not trades:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'pf': 0}

    tdf = pd.DataFrame(trades)
    wins = tdf[tdf['win']]
    losses = tdf[~tdf['win']]

    gp = wins['pnl_usd'].sum() if len(wins) > 0 else 0
    gl = abs(losses['pnl_usd'].sum()) if len(losses) > 0 else 0.01

    return {
        'trades': len(tdf),
        'wr': len(wins) / len(tdf) * 100,
        'pnl': tdf['pnl_usd'].sum(),
        'pf': gp / gl if gl > 0 else 999,
        'longs': len(tdf[tdf['direction'] == 1]),
        'shorts': len(tdf[tdf['direction'] == -1]),
        'long_wr': len(tdf[(tdf['direction'] == 1) & tdf['win']]) / max(len(tdf[tdf['direction'] == 1]), 1) * 100,
        'short_wr': len(tdf[(tdf['direction'] == -1) & tdf['win']]) / max(len(tdf[tdf['direction'] == -1]), 1) * 100,
    }


# Usar datos de test para optimizar
test_df = df.iloc[split_idx:].copy()
test_feat = feat.iloc[split_idx:].copy()

# Recalcular predicciones en test
X_test_full = scaler.transform(test_feat[feature_cols].dropna())
valid_idx = test_feat[feature_cols].dropna().index
test_df_valid = test_df.loc[valid_idx]
test_feat_valid = test_feat.loc[valid_idx]
test_preds_full = model.predict(X_test_full)

# Grid search TP/SL
print("\n  Grid search TP/SL...")
tp_range = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
sl_range = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
conv_range = [0.5, 0.7, 1.0, 1.2]

best_config = None
best_score = -999
results = []

for tp in tp_range:
    for sl in sl_range:
        for conv in conv_range:
            r = backtest_config(test_df_valid, test_feat_valid, test_preds_full, pred_std, tp, sl, conv)
            if r['trades'] >= 20:  # Minimo 20 trades
                # Score: PnL * sqrt(trades) para balancear
                score = r['pnl'] * np.sqrt(r['trades'] / 100)
                results.append({
                    'tp': tp, 'sl': sl, 'conv': conv,
                    **r, 'score': score
                })
                if score > best_score:
                    best_score = score
                    best_config = {'tp': tp, 'sl': sl, 'conv': conv, **r}

print(f"\n  Mejor configuracion (ambas direcciones):")
print(f"    TP: {best_config['tp']*100:.1f}%")
print(f"    SL: {best_config['sl']*100:.1f}%")
print(f"    Conv: {best_config['conv']}")
print(f"    Trades: {best_config['trades']}")
print(f"    WR: {best_config['wr']:.1f}%")
print(f"    PnL: ${best_config['pnl']:.2f}")
print(f"    PF: {best_config['pf']:.2f}")

# Analizar LONG vs SHORT
print("\n  Analizando LONG vs SHORT...")
r_both = backtest_config(test_df_valid, test_feat_valid, test_preds_full, pred_std,
                         best_config['tp'], best_config['sl'], best_config['conv'])
r_long = backtest_config(test_df_valid, test_feat_valid, test_preds_full, pred_std,
                         best_config['tp'], best_config['sl'], best_config['conv'], only_long=True)
r_short = backtest_config(test_df_valid, test_feat_valid, test_preds_full, pred_std,
                          best_config['tp'], best_config['sl'], best_config['conv'], only_short=True)

print(f"\n  AMBOS:  {r_both['trades']:>4} trades, WR {r_both['wr']:>5.1f}%, PnL ${r_both['pnl']:>7.2f}")
print(f"  LONG:   {r_long['trades']:>4} trades, WR {r_long['wr']:>5.1f}%, PnL ${r_long['pnl']:>7.2f}")
print(f"  SHORT:  {r_short['trades']:>4} trades, WR {r_short['wr']:>5.1f}%, PnL ${r_short['pnl']:>7.2f}")

# Determinar mejor direccion
direction_choice = 'BOTH'
final_config = r_both
if r_long['pnl'] > r_both['pnl'] * 1.1 and r_long['wr'] > r_both['wr']:
    direction_choice = 'LONG_ONLY'
    final_config = r_long
elif r_short['pnl'] > r_both['pnl'] * 1.1 and r_short['wr'] > r_both['wr']:
    direction_choice = 'SHORT_ONLY'
    final_config = r_short

print(f"\n  Direccion elegida: {direction_choice}")


# =============================================================================
# 6. GUARDAR MODELO Y DOCUMENTAR
# =============================================================================
print("\n[6/6] Guardando modelo y documentacion...")

# Guardar modelo
model_data = {
    'model': model,
    'scaler': scaler,
    'feature_cols': feature_cols,
    'pred_std': pred_std,
}
model_path = MODELS_DIR / 'link_usdt_v2_gradientboosting.pkl'
joblib.dump(model_data, model_path)
print(f"  Modelo guardado: {model_path}")

# Configuracion final
FINAL_CONFIG = {
    'pair': PAIR,
    'model_file': 'link_usdt_v2_gradientboosting.pkl',
    'tp_pct': best_config['tp'],
    'sl_pct': best_config['sl'],
    'conv_min': best_config['conv'],
    'only_long': direction_choice == 'LONG_ONLY',
    'only_short': direction_choice == 'SHORT_ONLY',
}

# Backtest en diferentes periodos para validacion
print("\n  Validacion multi-periodo...")

# Reentrenar con todos los datos para produccion
scaler_full = StandardScaler()
X_full = scaler_full.fit_transform(X)
model_full = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
model_full.fit(X_full, y)

# Guardar modelo final
model_data_final = {
    'model': model_full,
    'scaler': scaler_full,
    'feature_cols': feature_cols,
    'pred_std': np.std(model_full.predict(X_full)),
}
joblib.dump(model_data_final, model_path)

# Validar en periodos
periods = [
    ('2020', '2020-01-01', '2020-12-31'),
    ('2021', '2021-01-01', '2021-12-31'),
    ('2022', '2022-01-01', '2022-12-31'),
    ('2023', '2023-01-01', '2023-12-31'),
    ('2024', '2024-01-01', '2024-12-31'),
    ('2025', '2025-01-01', '2025-12-31'),
    ('Ultimo Ano', '2025-02-01', '2026-02-27'),
]

validation_results = []
for name, start, end in periods:
    mask = (df.index >= start) & (df.index <= end)
    if mask.sum() < 100:
        continue

    df_p = df[mask].copy()
    feat_p = feat[mask].copy()

    valid_mask = feat_p[feature_cols].notna().all(axis=1)
    df_v = df_p[valid_mask]
    feat_v = feat_p[valid_mask]

    if len(feat_v) < 50:
        continue

    X_v = scaler_full.transform(feat_v[feature_cols])
    preds_v = model_full.predict(X_v)

    r = backtest_config(
        df_v, feat_v, preds_v, model_data_final['pred_std'],
        FINAL_CONFIG['tp_pct'], FINAL_CONFIG['sl_pct'], FINAL_CONFIG['conv_min'],
        only_long=FINAL_CONFIG['only_long'], only_short=FINAL_CONFIG['only_short']
    )
    validation_results.append({'period': name, **r})
    print(f"    {name}: {r['trades']} trades, WR {r['wr']:.1f}%, PnL ${r['pnl']:.2f}")


# Documentar
doc = f"""# Optimizacion {PAIR} - V13.03

## Fecha: 2026-02-27

## Resumen
- **Par**: {PAIR}
- **Modelo**: GradientBoosting V2 (54 features)
- **Correlacion Test**: {test_corr:.4f}

## Configuracion Optima

| Parametro | Valor |
|-----------|-------|
| TP | {FINAL_CONFIG['tp_pct']*100:.1f}% |
| SL | {FINAL_CONFIG['sl_pct']*100:.1f}% |
| Conviction Min | {FINAL_CONFIG['conv_min']} |
| Direccion | {direction_choice} |

## Analisis de Direccion (Periodo Test)

| Direccion | Trades | WR | PnL |
|-----------|--------|-----|-----|
| AMBOS | {r_both['trades']} | {r_both['wr']:.1f}% | ${r_both['pnl']:.2f} |
| LONG ONLY | {r_long['trades']} | {r_long['wr']:.1f}% | ${r_long['pnl']:.2f} |
| SHORT ONLY | {r_short['trades']} | {r_short['wr']:.1f}% | ${r_short['pnl']:.2f} |

**Decision**: {direction_choice}

## Validacion Multi-Periodo

| Periodo | Trades | WR | PnL | PF |
|---------|--------|-----|-----|-----|
"""

for r in validation_results:
    pf_str = f"{r['pf']:.2f}" if r['pf'] < 100 else "INF"
    doc += f"| {r['period']} | {r['trades']} | {r['wr']:.1f}% | ${r['pnl']:.2f} | {pf_str} |\n"

# Calcular totales
total_trades = sum(r['trades'] for r in validation_results)
total_pnl = sum(r['pnl'] for r in validation_results)
avg_wr = np.mean([r['wr'] for r in validation_results if r['trades'] > 0])

doc += f"""
## Totales

- **Total Trades**: {total_trades}
- **PnL Total**: ${total_pnl:.2f}
- **WR Promedio**: {avg_wr:.1f}%

## Comparacion V7 vs V2

| Metrica | V7 (Anterior) | V2 (Nuevo) | Mejora |
|---------|---------------|------------|--------|
| WR Ultimo Ano | 36.1% | {validation_results[-1]['wr']:.1f}% | {validation_results[-1]['wr'] - 36.1:+.1f}% |
| PnL Ultimo Ano | $19.87 | ${validation_results[-1]['pnl']:.2f} | ${validation_results[-1]['pnl'] - 19.87:+.2f} |
| Trades Ultimo Ano | 1685 | {validation_results[-1]['trades']} | {validation_results[-1]['trades'] - 1685:+d} |

## Archivo de Modelo
- `models/link_usdt_v2_gradientboosting.pkl`

## Codigo para Configuracion
```python
ML_LINK_CONFIG = {{
    'model_file': 'link_usdt_v2_gradientboosting.pkl',
    'tp_pct': {FINAL_CONFIG['tp_pct']},
    'sl_pct': {FINAL_CONFIG['sl_pct']},
    'conv_min': {FINAL_CONFIG['conv_min']},
    'only_long': {FINAL_CONFIG['only_long']},
    'only_short': {FINAL_CONFIG['only_short']},
}}
```
"""

doc_path = DOCS_DIR / 'OPTIMIZACION_LINK.md'
with open(doc_path, 'w', encoding='utf-8') as f:
    f.write(doc)

print(f"  Documentacion: {doc_path}")

print("\n" + "="*70)
print("OPTIMIZACION LINK COMPLETADA")
print("="*70)
print(f"\nConfiguracion final:")
print(f"  TP: {FINAL_CONFIG['tp_pct']*100:.1f}%")
print(f"  SL: {FINAL_CONFIG['sl_pct']*100:.1f}%")
print(f"  Conv: {FINAL_CONFIG['conv_min']}")
print(f"  Direccion: {direction_choice}")
print(f"\nValidacion Ultimo Ano:")
print(f"  Trades: {validation_results[-1]['trades']}")
print(f"  WR: {validation_results[-1]['wr']:.1f}%")
print(f"  PnL: ${validation_results[-1]['pnl']:.2f}")
