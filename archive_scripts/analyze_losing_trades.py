"""
Analiza trades perdedores para entender patrones y mejorar WR.
Objetivo: Subir WR de 40% a 50-55% (nivel profesional).
"""
import json
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from regime_detector_v2 import RegimeDetector, MarketRegime, get_strategy_for_regime

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT',
]

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
    """Calcula todas las 34 features para V7 model."""
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

    # Choppiness (extra)
    atr_1 = ta.atr(h, l, c, length=1)
    atr_sum = atr_1.rolling(14).sum()
    high_max = h.rolling(14).max()
    low_min = l.rolling(14).min()
    feat['chop'] = 100 * np.log10(atr_sum / (high_max - low_min + 1e-10)) / np.log10(14)

    # Volume ratio (extra)
    feat['vol_ratio'] = v / v.rolling(20).mean()

    return feat

print("="*70)
print("ANALISIS DE TRADES PERDEDORES")
print("="*70)
print("\nObjetivo: Identificar patrones para subir WR de 40% a 50-55%\n")

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

btc_df = load_data('BTC/USDT')
detector = RegimeDetector()

# Simular trades y analizar perdedores
print("[1] Simulando trades del ultimo ano...")

all_trades = []
losing_patterns = defaultdict(list)

for pair, df in pair_data.items():
    df = df[(df.index >= '2025-02-01') & (df.index < '2026-02-24')]
    if len(df) < 250:
        continue

    safe = pair.replace('/', '')
    model = models.get(safe)
    if model is None:
        continue

    feat = compute_features(df)
    regimes = detector.detect_regime_series(df)
    fcols = [c for c in model.feature_name_ if c in feat.columns]

    for i in range(250, len(df) - 10):  # -10 para ver resultado
        ts = df.index[i]
        row = df.iloc[i]
        price = row['close']

        X = feat.loc[ts:ts][fcols]
        if X.isna().any().any():
            continue

        pred = model.predict(X)[0]
        sig = 1 if pred > 0.5 else -1
        conviction = abs(pred - 0.5) * 10

        # Filtros basicos
        chop_val = feat.loc[ts, 'chop'] if 'chop' in feat.columns else 50
        if conviction < 1.8 or chop_val > 50:
            continue

        # Calcular resultado (simplificado: 5 velas adelante)
        future_price = df.iloc[i+5]['close']
        pnl_pct = (future_price - price) / price * sig * 100

        # Recopilar datos del trade
        trade = {
            'pair': pair,
            'ts': ts,
            'dir': sig,
            'conviction': conviction,
            'result': 'WIN' if pnl_pct > 0 else 'LOSS',
            'pnl_pct': pnl_pct,
            'rsi14': feat.loc[ts, 'rsi14'],
            'rsi7': feat.loc[ts, 'rsi7'],
            'adx': feat.loc[ts, 'adx'] if 'adx' in feat.columns else 0,
            'chop': chop_val,
            'bb_pos': feat.loc[ts, 'bb_pos'] if 'bb_pos' in feat.columns else 0.5,
            'vol_ratio': feat.loc[ts, 'vol_ratio'] if 'vol_ratio' in feat.columns else 1,
            'ret_5': feat.loc[ts, 'ret_5'] if 'ret_5' in feat.columns else 0,
            'regime': regimes.loc[ts, 'regime'],
            'hour': ts.hour,
            'dayofweek': ts.dayofweek,
        }
        all_trades.append(trade)

trades_df = pd.DataFrame(all_trades)
print(f"  Total trades analizados: {len(trades_df):,}")

wins = trades_df[trades_df['result'] == 'WIN']
losses = trades_df[trades_df['result'] == 'LOSS']

print(f"  Wins: {len(wins):,} ({len(wins)/len(trades_df)*100:.1f}%)")
print(f"  Losses: {len(losses):,} ({len(losses)/len(trades_df)*100:.1f}%)")

# Analizar patrones de perdedores
print("\n" + "="*70)
print("[2] PATRONES EN TRADES PERDEDORES")
print("="*70)

print("\n--- RSI en momento de entrada ---")
print(f"{'RSI Range':<20} {'Wins':>8} {'Losses':>8} {'WR':>8}")
print("-"*50)
for low, high in [(0, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 100)]:
    mask = (trades_df['rsi14'] >= low) & (trades_df['rsi14'] < high)
    subset = trades_df[mask]
    if len(subset) > 10:
        w = len(subset[subset['result'] == 'WIN'])
        l = len(subset[subset['result'] == 'LOSS'])
        wr = w / len(subset) * 100
        print(f"RSI {low}-{high:<13} {w:>8} {l:>8} {wr:>7.1f}%")

print("\n--- ADX (fuerza de tendencia) ---")
print(f"{'ADX Range':<20} {'Wins':>8} {'Losses':>8} {'WR':>8}")
print("-"*50)
for low, high in [(0, 15), (15, 20), (20, 25), (25, 35), (35, 100)]:
    mask = (trades_df['adx'] >= low) & (trades_df['adx'] < high)
    subset = trades_df[mask]
    if len(subset) > 10:
        w = len(subset[subset['result'] == 'WIN'])
        l = len(subset[subset['result'] == 'LOSS'])
        wr = w / len(subset) * 100
        print(f"ADX {low}-{high:<13} {w:>8} {l:>8} {wr:>7.1f}%")

print("\n--- Choppiness Index ---")
print(f"{'Chop Range':<20} {'Wins':>8} {'Losses':>8} {'WR':>8}")
print("-"*50)
for low, high in [(0, 38), (38, 45), (45, 50), (50, 55), (55, 100)]:
    mask = (trades_df['chop'] >= low) & (trades_df['chop'] < high)
    subset = trades_df[mask]
    if len(subset) > 10:
        w = len(subset[subset['result'] == 'WIN'])
        l = len(subset[subset['result'] == 'LOSS'])
        wr = w / len(subset) * 100
        print(f"Chop {low}-{high:<12} {w:>8} {l:>8} {wr:>7.1f}%")

print("\n--- Conviction del modelo ---")
print(f"{'Conv Range':<20} {'Wins':>8} {'Losses':>8} {'WR':>8}")
print("-"*50)
for low, high in [(1.8, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 10)]:
    mask = (trades_df['conviction'] >= low) & (trades_df['conviction'] < high)
    subset = trades_df[mask]
    if len(subset) > 10:
        w = len(subset[subset['result'] == 'WIN'])
        l = len(subset[subset['result'] == 'LOSS'])
        wr = w / len(subset) * 100
        print(f"Conv {low:.1f}-{high:.1f}{'':>8} {w:>8} {l:>8} {wr:>7.1f}%")

print("\n--- Bollinger Band Position ---")
print(f"{'BB Pos Range':<20} {'Wins':>8} {'Losses':>8} {'WR':>8}")
print("-"*50)
for low, high in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 2.0)]:
    mask = (trades_df['bb_pos'] >= low) & (trades_df['bb_pos'] < high)
    subset = trades_df[mask]
    if len(subset) > 10:
        w = len(subset[subset['result'] == 'WIN'])
        l = len(subset[subset['result'] == 'LOSS'])
        wr = w / len(subset) * 100
        print(f"BB {low:.1f}-{high:.1f}{'':>10} {w:>8} {l:>8} {wr:>7.1f}%")

print("\n--- Volumen relativo ---")
print(f"{'Vol Ratio':<20} {'Wins':>8} {'Losses':>8} {'WR':>8}")
print("-"*50)
for low, high in [(0, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 2.0), (2.0, 10)]:
    mask = (trades_df['vol_ratio'] >= low) & (trades_df['vol_ratio'] < high)
    subset = trades_df[mask]
    if len(subset) > 10:
        w = len(subset[subset['result'] == 'WIN'])
        l = len(subset[subset['result'] == 'LOSS'])
        wr = w / len(subset) * 100
        print(f"Vol {low:.1f}-{high:.1f}{'':>10} {w:>8} {l:>8} {wr:>7.1f}%")

print("\n--- Direccion del trade ---")
print(f"{'Direction':<20} {'Wins':>8} {'Losses':>8} {'WR':>8}")
print("-"*50)
for d, name in [(1, 'LONG'), (-1, 'SHORT')]:
    subset = trades_df[trades_df['dir'] == d]
    if len(subset) > 10:
        w = len(subset[subset['result'] == 'WIN'])
        l = len(subset[subset['result'] == 'LOSS'])
        wr = w / len(subset) * 100
        print(f"{name:<20} {w:>8} {l:>8} {wr:>7.1f}%")

print("\n--- Hora del dia (UTC) ---")
print(f"{'Hour Range':<20} {'Wins':>8} {'Losses':>8} {'WR':>8}")
print("-"*50)
for low, high in [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]:
    mask = (trades_df['hour'] >= low) & (trades_df['hour'] < high)
    subset = trades_df[mask]
    if len(subset) > 10:
        w = len(subset[subset['result'] == 'WIN'])
        l = len(subset[subset['result'] == 'LOSS'])
        wr = w / len(subset) * 100
        print(f"Hour {low:02d}-{high:02d}{'':>10} {w:>8} {l:>8} {wr:>7.1f}%")

print("\n--- Por par ---")
print(f"{'Pair':<20} {'Wins':>8} {'Losses':>8} {'WR':>8}")
print("-"*50)
for pair in sorted(trades_df['pair'].unique()):
    subset = trades_df[trades_df['pair'] == pair]
    if len(subset) > 10:
        w = len(subset[subset['result'] == 'WIN'])
        l = len(subset[subset['result'] == 'LOSS'])
        wr = w / len(subset) * 100
        marker = " **" if wr < 35 else (" !!" if wr > 50 else "")
        print(f"{pair:<20} {w:>8} {l:>8} {wr:>7.1f}%{marker}")

# Identificar mejores filtros
print("\n" + "="*70)
print("[3] RECOMENDACIONES PARA MEJORAR WR")
print("="*70)

# Calcular WR con filtros combinados
best_filters = []

# Probar combinaciones
for min_conv in [2.0, 2.5, 3.0]:
    for max_chop in [38, 42, 45]:
        for min_adx in [20, 25, 30]:
            mask = (
                (trades_df['conviction'] >= min_conv) &
                (trades_df['chop'] < max_chop) &
                (trades_df['adx'] >= min_adx)
            )
            subset = trades_df[mask]
            if len(subset) >= 50:
                wr = len(subset[subset['result'] == 'WIN']) / len(subset) * 100
                best_filters.append({
                    'min_conv': min_conv,
                    'max_chop': max_chop,
                    'min_adx': min_adx,
                    'trades': len(subset),
                    'wr': wr,
                })

best_filters = sorted(best_filters, key=lambda x: -x['wr'])[:10]

print("\nMEJORES COMBINACIONES DE FILTROS:")
print(f"{'Conv>=':<8} {'Chop<':<8} {'ADX>=':<8} {'Trades':>8} {'WR':>8}")
print("-"*50)
for f in best_filters:
    print(f"{f['min_conv']:<8.1f} {f['max_chop']:<8} {f['min_adx']:<8} {f['trades']:>8} {f['wr']:>7.1f}%")

# WR actual vs mejor filtro
if best_filters:
    best = best_filters[0]
    current_wr = len(wins) / len(trades_df) * 100
    print(f"\nWR ACTUAL: {current_wr:.1f}%")
    print(f"WR CON MEJOR FILTRO: {best['wr']:.1f}% (+{best['wr']-current_wr:.1f}%)")
    print(f"  Conv >= {best['min_conv']}")
    print(f"  Chop < {best['max_chop']}")
    print(f"  ADX >= {best['min_adx']}")
    print(f"  Trades reducidos: {len(trades_df)} -> {best['trades']} (-{(1-best['trades']/len(trades_df))*100:.0f}%)")
