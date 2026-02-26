"""
Backtest V11: Filtros de Trader Profesional
============================================
Objetivo: Subir WR de 40% a 50%+ como un trader profesional.

Filtros basados en analisis de trades perdedores:
1. RSI 50-60: Mejor zona (58% WR vs 40% en RSI<30)
2. BB Position 0.4-0.8: Mejor zona (54-55% WR vs 47% en extremos)
3. ADX 20-35: Tendencia optima (53% WR vs 45% en ADX>35)
4. Volume ratio 0.5-2.0: Evitar volumen extremo
5. TP/SL optimizado: 1.5x/1.25x para mejor WR
"""

import json
import time
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT',
]

INITIAL_CAPITAL = 500.0
RISK_PER_TRADE = 0.02
MAX_HOLD = 15  # Reducido para TP mas rapido

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

    # Choppiness
    atr_1 = ta.atr(h, l, c, length=1)
    atr_sum = atr_1.rolling(14).sum()
    high_max = h.rolling(14).max()
    low_min = l.rolling(14).min()
    feat['chop'] = 100 * np.log10(atr_sum / (high_max - low_min + 1e-10)) / np.log10(14)

    return feat


def run_backtest_v11_pro(
    pair_data, models, start_date, end_date,
    # Filtros PRO basados en analisis
    min_rsi=40, max_rsi=70,      # Evitar extremos RSI
    min_bb=0.25, max_bb=0.75,    # Evitar extremos BB (modificado de 0.4-0.8)
    min_adx=15, max_adx=40,      # Tendencia optima
    min_vol=0.5, max_vol=2.5,    # Volumen normal
    min_conviction=2.0,          # Conviction alto
    max_chop=50,                 # Chop bajo
    # TP/SL optimizado para WR
    tp_mult=1.5,
    sl_mult=1.25,
):
    """V11 PRO: Filtros de trader profesional para WR 50%+."""
    trades = []

    for pair, df in pair_data.items():
        df = df[(df.index >= start_date) & (df.index < end_date)]
        if len(df) < 250:
            continue

        safe = pair.replace('/', '')
        model = models.get(safe)
        if model is None:
            continue

        feat = compute_features(df)
        fcols = [c for c in model.feature_name_ if c in feat.columns]

        balance = INITIAL_CAPITAL
        peak = balance
        max_dd = 0
        pos = None
        filtered_reasons = {'rsi': 0, 'bb': 0, 'adx': 0, 'vol': 0, 'conv': 0, 'chop': 0}

        for i in range(250, len(df)):
            row = df.iloc[i]
            ts = df.index[i]
            price = row['close']

            # Gestionar posicion abierta
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
                        'pair': pair,
                        'dir': pos['dir'],
                        'entry': pos['entry'],
                        'exit': price,
                        'pnl': pnl,
                        'exit_reason': 'tp' if hit_tp else ('sl' if hit_sl else 'timeout'),
                    })
                    pos = None

            # Buscar nueva entrada
            if pos is None:
                X = feat.loc[ts:ts][fcols]
                if X.isna().any().any():
                    continue

                pred = model.predict(X)[0]
                sig = 1 if pred > 0.5 else -1
                conviction = abs(pred - 0.5) * 10

                # FILTROS PRO
                rsi = feat.loc[ts, 'rsi14']
                bb_pos = feat.loc[ts, 'bb_pos'] if 'bb_pos' in feat.columns else 0.5
                adx = feat.loc[ts, 'adx'] if 'adx' in feat.columns else 25
                vol_ratio = feat.loc[ts, 'vr'] if 'vr' in feat.columns else 1
                chop = feat.loc[ts, 'chop'] if 'chop' in feat.columns else 50

                # Aplicar filtros
                if not (min_rsi <= rsi <= max_rsi):
                    filtered_reasons['rsi'] += 1
                    continue
                if not (min_bb <= bb_pos <= max_bb):
                    filtered_reasons['bb'] += 1
                    continue
                if not (min_adx <= adx <= max_adx):
                    filtered_reasons['adx'] += 1
                    continue
                if not (min_vol <= vol_ratio <= max_vol):
                    filtered_reasons['vol'] += 1
                    continue
                if conviction < min_conviction:
                    filtered_reasons['conv'] += 1
                    continue
                if chop > max_chop:
                    filtered_reasons['chop'] += 1
                    continue

                # Confirmar direccion con indicadores
                # LONG: RSI < 60, precio subiendo (ret_5 > 0)
                # SHORT: RSI > 40, precio bajando (ret_5 < 0)
                ret_5 = feat.loc[ts, 'ret_5'] if 'ret_5' in feat.columns else 0
                if sig == 1 and ret_5 < -0.02:  # Long pero precio cayendo mucho
                    continue
                if sig == -1 and ret_5 > 0.02:  # Short pero precio subiendo mucho
                    continue

                # Calcular TP/SL
                atr = feat.loc[ts, 'atr14']
                tp_pct = atr / price * tp_mult
                sl_pct = atr / price * sl_mult

                risk_amt = balance * RISK_PER_TRADE
                size = risk_amt / (sl_pct * price) if sl_pct > 0 else 0

                pos = {
                    'entry': price,
                    'dir': sig,
                    'size': size,
                    'tp_pct': tp_pct,
                    'sl_pct': sl_pct,
                    'bar': i,
                }

        # Cerrar posicion al final
        if pos is not None:
            pnl = pos['size'] * pos['dir'] * (df.iloc[-1]['close'] - pos['entry'])
            balance += pnl
            trades.append({
                'pair': pair,
                'dir': pos['dir'],
                'entry': pos['entry'],
                'exit': df.iloc[-1]['close'],
                'pnl': pnl,
                'exit_reason': 'eod',
            })

    return trades, max_dd, filtered_reasons


def compute_metrics(trades, max_dd):
    if not trades:
        return {'trades': 0, 'wins': 0, 'wr': 0, 'pnl': 0, 'pf': 0, 'return_pct': 0, 'max_dd': 0}

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]

    gross_profit = sum(t['pnl'] for t in wins)
    gross_loss = abs(sum(t['pnl'] for t in losses))

    return {
        'trades': len(trades),
        'wins': len(wins),
        'wr': len(wins) / len(trades) * 100 if trades else 0,
        'pnl': sum(t['pnl'] for t in trades),
        'pf': gross_profit / gross_loss if gross_loss > 0 else 999,
        'return_pct': sum(t['pnl'] for t in trades) / INITIAL_CAPITAL * 100,
        'max_dd': max_dd,
    }


def main():
    print("="*70)
    print("BACKTEST V11: FILTROS DE TRADER PROFESIONAL")
    print("="*70)
    print("\nObjetivo: WR >= 50% (nivel profesional)")

    # Cargar modelos
    print("\n[1] Cargando modelos...")
    models = {}
    for pair in PAIRS:
        safe = pair.replace('/', '')
        try:
            models[safe] = joblib.load(MODELS_DIR / f'v95_v7_{safe}.pkl')
        except:
            pass
    print(f"  Modelos cargados: {len(models)}")

    # Cargar datos
    print("\n[2] Cargando datos...")
    pair_data = {}
    for pair in PAIRS:
        df = load_data(pair)
        if df is not None:
            pair_data[pair] = df
    print(f"  Pares cargados: {len(pair_data)}")

    # Configuraciones a probar
    configs = [
        # Config actual para referencia
        {
            'name': 'V9.5 Actual',
            'min_rsi': 0, 'max_rsi': 100,
            'min_bb': 0, 'max_bb': 2,
            'min_adx': 0, 'max_adx': 100,
            'min_vol': 0, 'max_vol': 100,
            'min_conviction': 1.8,
            'max_chop': 50,
            'tp_mult': 2.5, 'sl_mult': 1.0,
        },
        # V11 PRO - Filtros conservadores
        {
            'name': 'V11 PRO',
            'min_rsi': 40, 'max_rsi': 70,
            'min_bb': 0.25, 'max_bb': 0.75,
            'min_adx': 15, 'max_adx': 40,
            'min_vol': 0.5, 'max_vol': 2.5,
            'min_conviction': 2.0,
            'max_chop': 48,
            'tp_mult': 1.5, 'sl_mult': 1.0,
        },
        # V11 ULTRA - Mas selectivo
        {
            'name': 'V11 ULTRA',
            'min_rsi': 45, 'max_rsi': 65,
            'min_bb': 0.3, 'max_bb': 0.7,
            'min_adx': 20, 'max_adx': 35,
            'min_vol': 0.6, 'max_vol': 2.0,
            'min_conviction': 2.5,
            'max_chop': 45,
            'tp_mult': 1.5, 'sl_mult': 0.75,
        },
    ]

    # Periodos
    periods = [
        ('Ultimo Ano', '2025-02-01', '2026-02-24'),
        ('Bear Market 2022', '2022-01-01', '2023-01-01'),
    ]

    for period_name, start, end in periods:
        print(f"\n{'='*70}")
        print(f"PERIODO: {period_name}")
        print(f"{'='*70}")

        print(f"\n{'Config':<15} {'Trades':>8} {'Wins':>8} {'WR':>8} {'PnL':>12} {'DD':>8}")
        print("-"*70)

        for cfg in configs:
            trades, max_dd, filtered = run_backtest_v11_pro(
                pair_data, models, start, end,
                min_rsi=cfg['min_rsi'], max_rsi=cfg['max_rsi'],
                min_bb=cfg['min_bb'], max_bb=cfg['max_bb'],
                min_adx=cfg['min_adx'], max_adx=cfg['max_adx'],
                min_vol=cfg['min_vol'], max_vol=cfg['max_vol'],
                min_conviction=cfg['min_conviction'],
                max_chop=cfg['max_chop'],
                tp_mult=cfg['tp_mult'], sl_mult=cfg['sl_mult'],
            )
            metrics = compute_metrics(trades, max_dd)

            marker = " **" if metrics['wr'] >= 50 else ""
            print(f"{cfg['name']:<15} {metrics['trades']:>8} {metrics['wins']:>8} "
                  f"{metrics['wr']:>7.1f}% ${metrics['pnl']:>10,.0f} {metrics['max_dd']:>7.1f}%{marker}")

            if cfg['name'] != 'V9.5 Actual':
                total_filtered = sum(filtered.values())
                print(f"  Filtrados: RSI={filtered['rsi']}, BB={filtered['bb']}, "
                      f"ADX={filtered['adx']}, Vol={filtered['vol']}, "
                      f"Conv={filtered['conv']}, Chop={filtered['chop']}")


if __name__ == '__main__':
    main()
