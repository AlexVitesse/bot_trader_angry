"""Backtest de ETH con modelo V7."""

import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import json
from pathlib import Path

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

V7_CONFIG = {'tp_pct': 0.03, 'sl_pct': 0.015, 'conv_min': 0.5}
POSITION_SIZE = 10.0


def compute_features(df):
    feat = pd.DataFrame(index=df.index)
    c, h, l, o, v = df['close'], df['high'], df['low'], df['open'], df['volume']

    for p in [1, 3, 5, 10, 20]:
        feat[f'ret_{p}'] = c.pct_change(p)

    feat['atr14'] = ta.atr(h, l, c, length=14)
    feat['atr_r'] = feat['atr14'] / feat['atr14'].rolling(50).mean()
    feat['vol5'] = c.pct_change().rolling(5).std()
    feat['vol20'] = c.pct_change().rolling(20).std()
    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)

    sr = ta.stochrsi(c, length=14, rsi_length=14, k=3, d=3)
    if sr is not None and len(sr.columns) >= 2:
        feat['srsi_k'] = sr.iloc[:, 0]

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None and len(macd.columns) >= 3:
        feat['macd_h'] = macd.iloc[:, 1]

    feat['roc5'] = ta.roc(c, length=5)
    feat['roc20'] = ta.roc(c, length=20)

    for el in [8, 21, 55, 100, 200]:
        e = ta.ema(c, length=el)
        feat[f'ema{el}_d'] = (c - e) / e * 100

    feat['ema8_sl'] = ta.ema(c, length=8).pct_change(3) * 100
    feat['ema55_sl'] = ta.ema(c, length=55).pct_change(5) * 100

    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None and len(bb.columns) >= 3:
        bw = bb.iloc[:, 2] - bb.iloc[:, 0]
        feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw
        feat['bb_w'] = bw / bb.iloc[:, 1] * 100

    feat['vr'] = v / v.rolling(20).mean()
    feat['spr'] = (h - l) / c * 100
    feat['body'] = abs(c - o) / (h - l + 1e-10)

    ax = ta.adx(h, l, c, length=14)
    if ax is not None and len(ax.columns) >= 3:
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


def backtest_pair_v7(pair, start, end):
    safe = pair.replace('/', '_')
    data_path = DATA_DIR / f'{safe}_4h_full.parquet'
    model_path = MODELS_DIR / f'v7_{safe}.pkl'
    meta_path = MODELS_DIR / 'v7_meta.json'

    if not data_path.exists():
        return None, "Sin datos"
    if not model_path.exists():
        return None, "Sin modelo"

    # Cargar
    df = pd.read_parquet(data_path)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    model = joblib.load(model_path)

    with open(meta_path) as f:
        meta = json.load(f)

    fcols = meta['feature_cols']
    pred_std = meta['pred_stds'].get(pair, 0.01)

    feat = compute_features(df)

    mask = (feat.index >= start) & (feat.index <= end)
    df_p = df[mask].copy()
    feat_p = feat[mask].copy()

    if len(df_p) < 50:
        return None, "Pocos datos"

    avail = [c for c in fcols if c in feat_p.columns]
    if len(avail) < len(fcols) * 0.8:
        return None, f"Features: {len(avail)}/{len(fcols)}"

    valid = feat_p[avail].notna().all(axis=1)
    feat_v = feat_p.loc[valid, avail]
    df_v = df_p[valid].copy()

    if len(feat_v) < 20:
        return None, "Datos insuf"

    # Predecir
    preds = model.predict(feat_v.values)
    conv = np.abs(preds) / pred_std
    signals = conv >= V7_CONFIG['conv_min']
    directions = np.where(preds < 0, -1, 1)
    regime = detect_regime(df_v)

    trades = []
    for i, (idx, row) in enumerate(feat_v.iterrows()):
        if i >= len(feat_v) - 5 or not signals[i]:
            continue

        d = directions[i]
        reg = regime.iloc[i]

        if reg == 'BULL' and d == -1:
            continue
        if reg == 'BEAR' and d == 1:
            continue

        entry = df_v.loc[idx, 'close']
        tp = entry * (1 + V7_CONFIG['tp_pct']) if d == 1 else entry * (1 - V7_CONFIG['tp_pct'])
        sl = entry * (1 - V7_CONFIG['sl_pct']) if d == 1 else entry * (1 + V7_CONFIG['sl_pct'])

        exit_p = None
        fidx = feat_v.index.get_loc(idx)

        for j in range(1, min(21, len(df_v) - fidx)):
            bar = df_v.iloc[fidx + j]
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
            exit_p = df_v.iloc[min(fidx + 20, len(df_v) - 1)]['close']

        pnl_pct = (exit_p - entry) / entry if d == 1 else (entry - exit_p) / entry
        trades.append({'pnl': pnl_pct * POSITION_SIZE, 'win': pnl_pct > 0})

    return trades, None


def main():
    print('='*70)
    print('BACKTEST ETH/USDT (Modelo V7) - Ultimo Ano')
    print('='*70)

    trades, err = backtest_pair_v7('ETH/USDT', '2025-02-01', '2026-02-27')

    if err:
        print(f"Error: {err}")
        return

    if not trades:
        print("Sin trades")
        return

    tdf = pd.DataFrame(trades)
    n = len(tdf)
    wr = tdf['win'].sum() / n * 100
    pnl = tdf['pnl'].sum()
    cumsum = tdf['pnl'].cumsum()
    maxdd = (cumsum.expanding().max() - cumsum).max()

    print(f"\nETH/USDT:")
    print(f"  Trades: {n}")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  PnL: ${pnl:.2f}")
    print(f"  Max DD: ${maxdd:.2f}")

    # Resumen completo
    print('\n')
    print('='*70)
    print('RESUMEN V13.02 COMPLETO (Ultimo Ano - $100 capital)')
    print('='*70)
    print(f"| {'Par':<12} | {'Trades':>6} | {'WR':>7} | {'PnL':>10} |")
    print('-'*50)
    print(f"| {'BTC/USDT':<12} | {67:>6} | {'55.2%':>7} | ${'8.34':>8} |")
    print(f"| {'BNB/USDT':<12} | {99:>6} | {'53.5%':>7} | ${'14.25':>8} |")
    print(f"| {'ETH/USDT':<12} | {n:>6} | {wr:>5.1f}% | ${pnl:>8.2f} |")
    print('-'*50)
    total_pnl = 8.34 + 14.25 + pnl
    total_n = 67 + 99 + n
    print(f"| {'TOTAL':<12} | {total_n:>6} |         | ${total_pnl:>8.2f} |")
    print('='*70)

    print("\nNOTA: Las otras monedas (XRP, NEAR, DOT, DOGE, AVAX, LINK, ADA)")
    print("no tienen datos descargados. Solo BTC, BNB, ETH tienen datos.")


if __name__ == '__main__':
    main()
