"""
Analiza optimizacion de TP/SL para mejorar WR.
El problema: WR "crudo" es 50.8% pero ejecucion real es 40%.
Causa probable: TP demasiado lejos, SL se alcanza primero.
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

PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']  # Solo 3 pares para velocidad

def load_data(pair):
    safe = pair.replace('/', '_')
    for suffix in ['_4h_v95.parquet', '_4h_history.parquet']:
        cache = DATA_DIR / f'{safe}{suffix}'
        if cache.exists():
            df = pd.read_parquet(cache)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            return df
    return None

def compute_features(df):
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
    atr_1 = ta.atr(h, l, c, length=1)
    atr_sum = atr_1.rolling(14).sum()
    high_max = h.rolling(14).max()
    low_min = l.rolling(14).min()
    feat['chop'] = 100 * np.log10(atr_sum / (high_max - low_min + 1e-10)) / np.log10(14)
    return feat

print("="*70)
print("OPTIMIZACION DE TP/SL PARA MEJORAR WR")
print("="*70)

# Cargar modelos
models = {}
for pair in PAIRS:
    safe = pair.replace('/', '')
    try:
        models[safe] = joblib.load(MODELS_DIR / f'v95_v7_{safe}.pkl')
    except:
        pass

# Cargar datos
pair_data = {}
for pair in PAIRS:
    df = load_data(pair)
    if df is not None:
        pair_data[pair] = df

print(f"\nPares cargados: {len(pair_data)}")

# Probar diferentes combinaciones de TP/SL ATR multipliers
print("\n[1] PROBANDO COMBINACIONES TP/SL (ATR multipliers)")
print("-"*70)

results = []
MAX_HOLD = 20

for tp_mult in [1.5, 2.0, 2.5, 3.0]:
    for sl_mult in [0.75, 1.0, 1.25]:
        wins = 0
        losses = 0
        total_pnl = 0

        for pair, df in pair_data.items():
            df = df[(df.index >= '2025-02-01') & (df.index < '2026-02-24')]
            if len(df) < 250:
                continue

            safe = pair.replace('/', '')
            model = models.get(safe)
            if model is None:
                continue

            feat = compute_features(df)
            fcols = [c for c in model.feature_name_ if c in feat.columns]

            for i in range(250, len(df) - MAX_HOLD - 1):
                ts = df.index[i]
                row = df.iloc[i]
                price = row['close']

                X = feat.loc[ts:ts][fcols]
                if X.isna().any().any():
                    continue

                pred = model.predict(X)[0]
                sig = 1 if pred > 0.5 else -1
                conviction = abs(pred - 0.5) * 10

                # Filtros
                chop_val = feat.loc[ts, 'chop'] if 'chop' in feat.columns else 50
                if conviction < 1.8 or chop_val > 50:
                    continue

                # Calcular TP/SL con ATR
                atr = feat.loc[ts, 'atr14']
                tp_price = price + sig * atr * tp_mult
                sl_price = price - sig * atr * sl_mult

                # Simular trade
                hit_tp = False
                hit_sl = False

                for j in range(1, MAX_HOLD + 1):
                    if i + j >= len(df):
                        break
                    future_row = df.iloc[i + j]
                    future_high = future_row['high']
                    future_low = future_row['low']

                    if sig == 1:  # LONG
                        if future_high >= tp_price:
                            hit_tp = True
                            break
                        if future_low <= sl_price:
                            hit_sl = True
                            break
                    else:  # SHORT
                        if future_low <= tp_price:
                            hit_tp = True
                            break
                        if future_high >= sl_price:
                            hit_sl = True
                            break

                if hit_tp:
                    wins += 1
                    total_pnl += atr * tp_mult / price * 100  # % gain
                elif hit_sl:
                    losses += 1
                    total_pnl -= atr * sl_mult / price * 100  # % loss
                else:
                    # Timeout - close at last price
                    exit_price = df.iloc[min(i + MAX_HOLD, len(df) - 1)]['close']
                    pnl_pct = (exit_price - price) / price * sig * 100
                    if pnl_pct > 0:
                        wins += 1
                    else:
                        losses += 1
                    total_pnl += pnl_pct

        total = wins + losses
        wr = wins / total * 100 if total > 0 else 0
        rr = tp_mult / sl_mult

        results.append({
            'tp': tp_mult,
            'sl': sl_mult,
            'rr': rr,
            'trades': total,
            'wins': wins,
            'losses': losses,
            'wr': wr,
            'pnl': total_pnl,
            'ev': (wr/100 * tp_mult) - ((100-wr)/100 * sl_mult),  # Expected value
        })

# Ordenar por WR
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('wr', ascending=False)

print(f"\n{'TP':>5} {'SL':>5} {'R:R':>5} {'Trades':>8} {'Wins':>8} {'WR':>8} {'PnL%':>10} {'EV':>8}")
print("-"*70)
for _, r in results_df.head(15).iterrows():
    print(f"{r['tp']:>5.1f} {r['sl']:>5.2f} {r['rr']:>5.1f} {r['trades']:>8} {r['wins']:>8} {r['wr']:>7.1f}% {r['pnl']:>9.1f}% {r['ev']:>7.3f}")

# Mejor por WR
best_wr = results_df.iloc[0]
# Mejor por PnL
best_pnl = results_df.sort_values('pnl', ascending=False).iloc[0]
# Mejor por EV
best_ev = results_df.sort_values('ev', ascending=False).iloc[0]

print("\n" + "="*70)
print("MEJORES CONFIGURACIONES")
print("="*70)

print(f"\nMEJOR WR ({best_wr['wr']:.1f}%):")
print(f"  TP: {best_wr['tp']:.1f}x ATR, SL: {best_wr['sl']:.2f}x ATR")
print(f"  R:R = {best_wr['rr']:.1f}, Trades: {best_wr['trades']}")

print(f"\nMEJOR PnL ({best_pnl['pnl']:.1f}%):")
print(f"  TP: {best_pnl['tp']:.1f}x ATR, SL: {best_pnl['sl']:.2f}x ATR")
print(f"  WR: {best_pnl['wr']:.1f}%, R:R = {best_pnl['rr']:.1f}")

print(f"\nMEJOR EV ({best_ev['ev']:.3f}):")
print(f"  TP: {best_ev['tp']:.1f}x ATR, SL: {best_ev['sl']:.2f}x ATR")
print(f"  WR: {best_ev['wr']:.1f}%, R:R = {best_ev['rr']:.1f}")

# Comparar con configuracion actual (2.5x TP, 1.0x SL)
current = results_df[(results_df['tp'] == 2.5) & (results_df['sl'] == 1.0)]
if len(current) > 0:
    curr = current.iloc[0]
    print(f"\nCONFIGURACION ACTUAL (2.5x/1.0x):")
    print(f"  WR: {curr['wr']:.1f}%, PnL: {curr['pnl']:.1f}%, EV: {curr['ev']:.3f}")

print("\n" + "="*70)
print("RECOMENDACION")
print("="*70)

# La mejor combinacion de WR + PnL razonable
best_balanced = results_df[(results_df['wr'] >= 50) & (results_df['pnl'] > 0)]
if len(best_balanced) > 0:
    best_balanced = best_balanced.sort_values('pnl', ascending=False).iloc[0]
    print(f"\nMEJOR BALANCE (WR >= 50% y PnL positivo):")
    print(f"  TP: {best_balanced['tp']:.1f}x ATR, SL: {best_balanced['sl']:.2f}x ATR")
    print(f"  WR: {best_balanced['wr']:.1f}%, PnL: {best_balanced['pnl']:.1f}%")
    print(f"  R:R = {best_balanced['rr']:.1f}")
else:
    # Si no hay ninguno con WR >= 50%, mostrar el mejor WR
    print(f"\nNo hay configuracion con WR >= 50%")
    print(f"Mejor WR disponible: {best_wr['wr']:.1f}%")
