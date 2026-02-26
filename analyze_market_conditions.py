"""
Análisis de Modelos por Condición de Mercado
=============================================
Clasificar cada modelo según donde mejor funciona:
- BEAR: Mercado bajista (precio cayendo)
- LATERAL: Mercado lateral (sin tendencia clara)
- BULL: Mercado alcista (precio subiendo)

Objetivo: Saber qué modelo usar en cada condición.
"""

import json
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent / 'data'
MODELS_DIR = Path(__file__).parent / 'models'

INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.02

# Definir periodos por tipo de mercado
MARKET_PERIODS = {
    'BEAR': [
        ('2022-01-01', '2022-12-31', 'Bear Market 2022'),
        ('2021-11-01', '2022-06-30', 'Crash post-ATH'),
        ('2025-01-01', '2026-02-24', 'Correccion 2025'),
    ],
    'BULL': [
        ('2020-10-01', '2021-04-30', 'Bull Run 2020-21'),
        ('2021-07-01', '2021-11-10', 'Bull Run Q3-Q4 2021'),
        ('2023-10-01', '2024-03-31', 'Rally 2023-24'),
        ('2024-10-01', '2024-12-31', 'Rally Q4 2024'),
    ],
    'LATERAL': [
        ('2022-06-01', '2022-10-31', 'Consolidacion 2022'),
        ('2023-03-01', '2023-09-30', 'Lateral 2023'),
        ('2024-04-01', '2024-09-30', 'Consolidacion 2024'),
    ],
}


def load_data_4h(pair):
    safe = pair.replace('/', '_')
    for suffix in ['_4h_v95.parquet', '_4h_history.parquet']:
        cache = DATA_DIR / f'{safe}{suffix}'
        if cache.exists():
            df = pd.read_parquet(cache)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            return df
    return None


def load_model(pair):
    safe = pair.replace('/', '').replace('_', '')
    try:
        return joblib.load(MODELS_DIR / f'v95_v7_{safe}.pkl')
    except:
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


def run_backtest(df, model, start_date, end_date, min_conv=1.0, max_chop=60):
    """Backtest simple con filtros. conv=1.0 significa pred >= 0.5% retorno."""
    df_period = df[(df.index >= start_date) & (df.index < end_date)]
    if len(df_period) < 50:
        return None

    feat = compute_features(df)
    feat = feat[(feat.index >= start_date) & (feat.index < end_date)]
    fcols = [c for c in model.feature_name_ if c in feat.columns]
    if not fcols:
        return None

    trades = []
    balance = INITIAL_CAPITAL
    pos = None

    for i in range(50, len(df_period)):
        ts = df_period.index[i]
        if ts not in feat.index:
            continue
        price = df_period.iloc[i]['close']

        if pos is not None:
            pnl_pct = (price - pos['entry']) / pos['entry'] * pos['dir']
            if pnl_pct >= pos['tp_pct'] or pnl_pct <= -pos['sl_pct'] or (i - pos['bar']) >= 20:
                pnl = pos['size'] * pos['dir'] * (price - pos['entry'])
                balance += pnl
                trades.append({'pnl': pnl, 'dir': pos['dir']})
                pos = None

        if pos is None:
            rsi = feat.loc[ts, 'rsi14'] if 'rsi14' in feat.columns else 50
            chop = feat.loc[ts, 'chop'] if 'chop' in feat.columns else 50
            atr = feat.loc[ts, 'atr14'] if 'atr14' in feat.columns else price * 0.02

            X = feat.loc[ts:ts][fcols]
            if X.isna().any().any():
                continue

            pred = model.predict(X)[0]
            # FIX: El modelo predice RETORNOS, no probabilidades
            # pred > 0 = LONG, pred < 0 = SHORT
            sig = 1 if pred > 0 else -1
            # Calcular conviction basado en magnitud de predicción
            # Usar 0.01 como referencia (1% de retorno es fuerte señal)
            conviction = abs(pred) / 0.005  # pred=0.5%->conv=1, pred=1%->conv=2

            if conviction < min_conv or chop > max_chop:
                continue

            tp_pct = atr / price * 2.0
            sl_pct = atr / price * 1.0
            risk_amt = balance * RISK_PER_TRADE
            size = risk_amt / (sl_pct * price) if sl_pct > 0 else 0
            pos = {'entry': price, 'dir': sig, 'size': size,
                   'tp_pct': tp_pct, 'sl_pct': sl_pct, 'bar': i}

    if pos:
        pnl = pos['size'] * pos['dir'] * (df_period.iloc[-1]['close'] - pos['entry'])
        trades.append({'pnl': pnl, 'dir': pos['dir']})

    if not trades:
        return None

    wins = sum(1 for t in trades if t['pnl'] > 0)
    longs = [t for t in trades if t['dir'] == 1]
    shorts = [t for t in trades if t['dir'] == -1]

    long_wr = sum(1 for t in longs if t['pnl'] > 0) / len(longs) * 100 if longs else 0
    short_wr = sum(1 for t in shorts if t['pnl'] > 0) / len(shorts) * 100 if shorts else 0

    return {
        'trades': len(trades),
        'wins': wins,
        'wr': wins / len(trades) * 100,
        'pnl': sum(t['pnl'] for t in trades),
        'longs': len(longs),
        'long_wr': long_wr,
        'shorts': len(shorts),
        'short_wr': short_wr,
    }


def main():
    print("="*80, flush=True)
    print("ANÁLISIS DE MODELOS POR CONDICIÓN DE MERCADO", flush=True)
    print("="*80, flush=True)
    print("\nObjetivo: Identificar qué modelo usar en BEAR, LATERAL, BULL", flush=True)

    pairs = ['ADA/USDT', 'XRP/USDT', 'DOT/USDT', 'DOGE/USDT',
             'ETH/USDT', 'AVAX/USDT', 'NEAR/USDT', 'LINK/USDT']

    # Resultados por condición de mercado
    results_by_condition = {'BEAR': [], 'BULL': [], 'LATERAL': []}

    for pair in pairs:
        print(f"\n{'='*60}", flush=True)
        print(f"PAR: {pair}", flush=True)
        print("="*60, flush=True)

        df = load_data_4h(pair)
        model = load_model(pair)

        if df is None or model is None:
            print(f"  [!] Sin datos o modelo", flush=True)
            continue

        pair_results = {'BEAR': [], 'BULL': [], 'LATERAL': []}

        for condition, periods in MARKET_PERIODS.items():
            print(f"\n  {condition}:", flush=True)

            for start, end, name in periods:
                result = run_backtest(df, model, start, end)
                if result and result['trades'] >= 5:
                    pair_results[condition].append(result)
                    wr_mark = "**" if result['wr'] >= 50 else ""
                    print(f"    {name}: {result['trades']} tr, WR {result['wr']:.1f}%{wr_mark}, "
                          f"L:{result['long_wr']:.0f}% S:{result['short_wr']:.0f}%, "
                          f"PnL ${result['pnl']:.0f}", flush=True)

        # Calcular promedios por condición
        print(f"\n  RESUMEN {pair}:", flush=True)
        for condition in ['BEAR', 'BULL', 'LATERAL']:
            if pair_results[condition]:
                avg_wr = np.mean([r['wr'] for r in pair_results[condition]])
                avg_long_wr = np.mean([r['long_wr'] for r in pair_results[condition]])
                avg_short_wr = np.mean([r['short_wr'] for r in pair_results[condition]])
                total_pnl = sum(r['pnl'] for r in pair_results[condition])
                total_trades = sum(r['trades'] for r in pair_results[condition])

                results_by_condition[condition].append({
                    'pair': pair,
                    'avg_wr': avg_wr,
                    'avg_long_wr': avg_long_wr,
                    'avg_short_wr': avg_short_wr,
                    'total_pnl': total_pnl,
                    'total_trades': total_trades,
                })

                wr_mark = "**" if avg_wr >= 50 else ""
                print(f"    {condition}: WR {avg_wr:.1f}%{wr_mark} (L:{avg_long_wr:.0f}% S:{avg_short_wr:.0f}%), "
                      f"{total_trades} tr, ${total_pnl:.0f}", flush=True)

    # Rankings por condición
    print("\n" + "="*80, flush=True)
    print("RANKINGS POR CONDICIÓN DE MERCADO", flush=True)
    print("="*80, flush=True)

    classification = {'BEAR': [], 'BULL': [], 'LATERAL': []}

    for condition in ['BEAR', 'BULL', 'LATERAL']:
        print(f"\n{'='*40}", flush=True)
        print(f"MEJORES MODELOS PARA {condition}", flush=True)
        print("="*40, flush=True)

        # Ordenar por WR
        sorted_results = sorted(results_by_condition[condition],
                               key=lambda x: x['avg_wr'], reverse=True)

        print(f"\n{'#':<3} {'Par':<12} {'WR':<8} {'Long WR':<10} {'Short WR':<10} {'PnL':<10}", flush=True)
        print("-"*55, flush=True)

        for i, r in enumerate(sorted_results, 1):
            wr_mark = "**" if r['avg_wr'] >= 50 else ""
            viable = "VIABLE" if r['avg_wr'] >= 48 and r['total_pnl'] > 0 else ""
            print(f"{i:<3} {r['pair']:<12} {r['avg_wr']:<7.1f}%{wr_mark} "
                  f"{r['avg_long_wr']:<9.0f}% {r['avg_short_wr']:<9.0f}% "
                  f"${r['total_pnl']:<9,.0f} {viable}", flush=True)

            if r['avg_wr'] >= 48 and r['total_pnl'] > 0:
                classification[condition].append(r['pair'])

    # Resumen final - Clasificación
    print("\n" + "="*80, flush=True)
    print("CLASIFICACIÓN FINAL - QUÉ MODELO USAR EN CADA CONDICIÓN", flush=True)
    print("="*80, flush=True)

    for condition in ['BEAR', 'BULL', 'LATERAL']:
        viable_pairs = classification[condition]
        if viable_pairs:
            print(f"\n{condition} MARKET -> Usar: {', '.join(viable_pairs)}", flush=True)
        else:
            print(f"\n{condition} MARKET -> Ningún modelo viable (WR<48%)", flush=True)

    # Análisis especial: qué dirección funciona mejor en cada condición
    print("\n" + "="*80, flush=True)
    print("ANÁLISIS DIRECCIONAL", flush=True)
    print("="*80, flush=True)

    for condition in ['BEAR', 'BULL', 'LATERAL']:
        if results_by_condition[condition]:
            avg_long = np.mean([r['avg_long_wr'] for r in results_by_condition[condition]])
            avg_short = np.mean([r['avg_short_wr'] for r in results_by_condition[condition]])

            if avg_long > avg_short + 5:
                direction = "LONGS funcionan mejor"
            elif avg_short > avg_long + 5:
                direction = "SHORTS funcionan mejor"
            else:
                direction = "Ambas direcciones similares"

            print(f"\n{condition}: Long WR {avg_long:.1f}% vs Short WR {avg_short:.1f}% -> {direction}", flush=True)

    # Guardar resultados
    output = {
        'fecha': datetime.now().isoformat(),
        'descripcion': 'Clasificacion de modelos por condicion de mercado',
        'clasificacion': {k: v for k, v in classification.items()},
        'detalles': {
            condition: [
                {
                    'pair': r['pair'],
                    'avg_wr': r['avg_wr'],
                    'avg_long_wr': r['avg_long_wr'],
                    'avg_short_wr': r['avg_short_wr'],
                    'total_pnl': r['total_pnl'],
                }
                for r in results_by_condition[condition]
            ]
            for condition in ['BEAR', 'BULL', 'LATERAL']
        }
    }

    with open(MODELS_DIR / 'market_condition_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n\nGuardado en market_condition_analysis.json", flush=True)


if __name__ == '__main__':
    main()
