"""
Analiza rendimiento por par (moneda)
====================================
Identifica:
- Pares con buen WR y ganancias (mantener)
- Pares con mal WR y perdidas (excluir)
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

ALL_PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT',
    'NEAR/USDT',
]

INITIAL_CAPITAL = 500.0
MAX_HOLD = 20

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


def backtest_pair(pair, df, model, start_date, end_date):
    """Backtest un par individual."""
    df = df[(df.index >= start_date) & (df.index < end_date)]
    if len(df) < 250:
        return None

    feat = compute_features(df)
    fcols = [c for c in model.feature_name_ if c in feat.columns]

    trades = []
    balance = INITIAL_CAPITAL
    peak = balance
    max_dd = 0
    pos = None

    for i in range(250, len(df)):
        row = df.iloc[i]
        ts = df.index[i]
        price = row['close']

        if pos is not None:
            pnl_pct = (price - pos['entry']) / pos['entry'] * pos['dir']
            hit_tp = pnl_pct >= pos['tp_pct']
            hit_sl = pnl_pct <= -pos['sl_pct']
            timeout = (i - pos['bar']) >= MAX_HOLD

            if hit_tp or hit_sl or timeout:
                pnl = pos['size'] * pos['dir'] * (price - pos['entry'])
                balance += pnl
                peak = max(peak, balance)
                dd = (peak - balance) / peak * 100
                max_dd = max(max_dd, dd)
                trades.append({
                    'pnl': pnl,
                    'exit': 'tp' if hit_tp else ('sl' if hit_sl else 'timeout'),
                })
                pos = None

        if pos is None:
            X = feat.loc[ts:ts][fcols]
            if X.isna().any().any():
                continue

            pred = model.predict(X)[0]
            sig = 1 if pred > 0.5 else -1
            conviction = abs(pred - 0.5) * 10

            rsi = feat.loc[ts, 'rsi14']
            chop = feat.loc[ts, 'chop'] if 'chop' in feat.columns else 50
            atr = feat.loc[ts, 'atr14']

            # Filtros basicos
            if conviction < 1.8:
                continue
            if chop > 50:
                continue
            if not (35 <= rsi <= 65):
                continue

            tp_pct = atr / price * 2.0
            sl_pct = atr / price * 1.0

            risk_amt = balance * 0.02
            size = risk_amt / (sl_pct * price) if sl_pct > 0 else 0

            pos = {
                'entry': price,
                'dir': sig,
                'size': size,
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
                'bar': i,
            }

    if pos is not None:
        pnl = pos['size'] * pos['dir'] * (df.iloc[-1]['close'] - pos['entry'])
        balance += pnl
        trades.append({'pnl': pnl, 'exit': 'eod'})

    if not trades:
        return None

    wins = len([t for t in trades if t['pnl'] > 0])
    total_pnl = sum(t['pnl'] for t in trades)

    return {
        'trades': len(trades),
        'wins': wins,
        'wr': wins / len(trades) * 100,
        'pnl': total_pnl,
        'max_dd': max_dd,
    }


print("="*70)
print("ANALISIS DE RENDIMIENTO POR PAR")
print("="*70)

# Cargar modelos
models = {}
for pair in ALL_PAIRS:
    safe = pair.replace('/', '')
    try:
        models[safe] = joblib.load(MODELS_DIR / f'v95_v7_{safe}.pkl')
    except:
        pass

# Cargar datos
pair_data = {}
for pair in ALL_PAIRS:
    df = load_data(pair)
    if df is not None:
        pair_data[pair] = df

print(f"\nPares disponibles: {len(pair_data)}")

# Periodos
periods = [
    ('Ultimo Ano', '2025-02-01', '2026-02-24'),
    ('Bear Market 2022', '2022-01-01', '2023-01-01'),
]

all_results = {}

for period_name, start, end in periods:
    print(f"\n{'='*70}")
    print(f"PERIODO: {period_name}")
    print(f"{'='*70}")

    results = []

    for pair in ALL_PAIRS:
        if pair not in pair_data:
            continue

        safe = pair.replace('/', '')
        model = models.get(safe)
        if model is None:
            continue

        metrics = backtest_pair(pair, pair_data[pair], model, start, end)
        if metrics:
            results.append({
                'pair': pair,
                **metrics,
            })

    # Ordenar por WR
    results = sorted(results, key=lambda x: -x['wr'])

    print(f"\n{'Par':<12} {'Trades':>8} {'Wins':>6} {'WR':>8} {'PnL':>12} {'Max DD':>8}")
    print("-"*60)

    for r in results:
        wr_mark = " **" if r['wr'] >= 50 else (" !!" if r['wr'] < 40 else "")
        pnl_mark = " $" if r['pnl'] > 0 else " X"
        print(f"{r['pair']:<12} {r['trades']:>8} {r['wins']:>6} {r['wr']:>7.1f}%{wr_mark} ${r['pnl']:>10,.0f}{pnl_mark} {r['max_dd']:>7.1f}%")

    all_results[period_name] = results

# Resumen combinado
print("\n" + "="*70)
print("RESUMEN: PARES RECOMENDADOS")
print("="*70)

# Calcular score combinado
combined_scores = {}
for pair in ALL_PAIRS:
    scores = []
    for period_name in periods:
        period_results = {r['pair']: r for r in all_results.get(period_name[0], [])}
        if pair in period_results:
            r = period_results[pair]
            # Score = WR + (PnL > 0 ? 10 : -10)
            score = r['wr'] + (10 if r['pnl'] > 0 else -10)
            scores.append(score)
    if scores:
        combined_scores[pair] = np.mean(scores)

# Ordenar por score
sorted_pairs = sorted(combined_scores.items(), key=lambda x: -x[1])

print(f"\n{'Par':<12} {'Score':>10} {'Recomendacion':<20}")
print("-"*50)

good_pairs = []
bad_pairs = []

for pair, score in sorted_pairs:
    if score >= 50:
        rec = "MANTENER **"
        good_pairs.append(pair)
    elif score >= 40:
        rec = "MANTENER"
        good_pairs.append(pair)
    else:
        rec = "EXCLUIR !!"
        bad_pairs.append(pair)
    print(f"{pair:<12} {score:>10.1f} {rec:<20}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print(f"\nPARES A MANTENER ({len(good_pairs)}):")
for p in good_pairs:
    print(f"  - {p}")

print(f"\nPARES A EXCLUIR ({len(bad_pairs)}):")
for p in bad_pairs:
    print(f"  - {p}")

# Guardar lista de pares buenos
import json
with open(MODELS_DIR / 'v12_good_pairs.json', 'w') as f:
    json.dump({'good_pairs': good_pairs, 'excluded_pairs': bad_pairs}, f, indent=2)
print(f"\nGuardado en models/v12_good_pairs.json")
