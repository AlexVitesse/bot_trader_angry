"""
Analisis profundo del modelo BTC actual
=======================================
Por que falla? Que podemos mejorar?
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

# Config igual que V13
TP_PCT = 0.03
SL_PCT = 0.015
MAX_HOLD = 20
CONVICTION_MIN = 1.0

def load_data():
    df = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_history.parquet')
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df

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

    return feat

def simulate_trades(df, feat, model, start_date, end_date):
    """Simula trades y retorna lista detallada."""
    trades = []
    fcols = [c for c in model.feature_name_ if c in feat.columns]

    mask = (df.index >= start_date) & (df.index < end_date)
    test_idx = df[mask].index

    for ts in test_idx:
        if ts not in feat.index:
            continue
        row = feat.loc[ts]

        if row[fcols].isna().any():
            continue

        try:
            pred = model.predict(row[fcols].values.reshape(1, -1))[0]
        except:
            continue

        direction = 1 if pred > 0 else -1
        conviction = abs(pred) / 0.005

        if conviction < CONVICTION_MIN:
            continue

        # Simular trade
        idx = df.index.get_loc(ts)
        if idx + 1 >= len(df):
            continue

        entry = df.iloc[idx + 1]['open']
        tp_price = entry * (1 + TP_PCT) if direction == 1 else entry * (1 - TP_PCT)
        sl_price = entry * (1 - SL_PCT) if direction == 1 else entry * (1 + SL_PCT)

        pnl_pct = None
        exit_reason = None
        exit_ts = None
        bars_held = 0

        for j in range(idx + 1, min(idx + MAX_HOLD + 1, len(df))):
            bars_held += 1
            candle = df.iloc[j]

            if direction == 1:
                if candle['low'] <= sl_price:
                    pnl_pct = -SL_PCT
                    exit_reason = 'SL'
                    exit_ts = candle.name
                    break
                if candle['high'] >= tp_price:
                    pnl_pct = TP_PCT
                    exit_reason = 'TP'
                    exit_ts = candle.name
                    break
            else:
                if candle['high'] >= sl_price:
                    pnl_pct = -SL_PCT
                    exit_reason = 'SL'
                    exit_ts = candle.name
                    break
                if candle['low'] <= tp_price:
                    pnl_pct = TP_PCT
                    exit_reason = 'TP'
                    exit_ts = candle.name
                    break

        if pnl_pct is None:
            exit_idx = min(idx + MAX_HOLD, len(df) - 1)
            exit_price = df.iloc[exit_idx]['close']
            pnl_pct = (exit_price - entry) / entry * direction
            exit_reason = 'TIMEOUT'
            exit_ts = df.index[exit_idx]

        trades.append({
            'entry_ts': ts,
            'exit_ts': exit_ts,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'entry': entry,
            'conviction': conviction,
            'pred': pred,
            'pnl_pct': pnl_pct,
            'win': pnl_pct > 0,
            'exit_reason': exit_reason,
            'bars_held': bars_held,
            'rsi14': row.get('rsi14', np.nan),
            'bb_pos': row.get('bb_pos', np.nan),
            'adx': row.get('adx', np.nan),
            'vol_ratio': row.get('vr', np.nan),
        })

    return trades

def analyze_trades(trades):
    """Analiza patrones en trades ganadores vs perdedores."""
    if not trades:
        print("No hay trades para analizar")
        return

    df = pd.DataFrame(trades)

    print(f"\n{'='*60}")
    print("ANALISIS DE TRADES BTC")
    print(f"{'='*60}")

    # Metricas generales
    n = len(df)
    wins = df['win'].sum()
    wr = wins / n * 100
    pnl = df['pnl_pct'].sum() * 100

    print(f"\nMETRICAS GENERALES:")
    print(f"  Total trades: {n}")
    print(f"  Wins: {wins}")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  PnL total: {pnl:.2f}%")

    # Por direccion
    print(f"\nPOR DIRECCION:")
    for dir in ['LONG', 'SHORT']:
        subset = df[df['direction'] == dir]
        if len(subset) > 0:
            dir_wr = subset['win'].mean() * 100
            dir_pnl = subset['pnl_pct'].sum() * 100
            print(f"  {dir}: {len(subset)} trades, WR={dir_wr:.1f}%, PnL={dir_pnl:.2f}%")

    # Por exit reason
    print(f"\nPOR RAZON DE SALIDA:")
    for reason in ['TP', 'SL', 'TIMEOUT']:
        subset = df[df['exit_reason'] == reason]
        if len(subset) > 0:
            print(f"  {reason}: {len(subset)} trades ({len(subset)/n*100:.1f}%)")

    # Comparacion winners vs losers
    winners = df[df['win'] == True]
    losers = df[df['win'] == False]

    print(f"\nCOMPARACION WINNERS vs LOSERS:")
    metrics = ['conviction', 'rsi14', 'bb_pos', 'adx', 'vol_ratio', 'bars_held']

    for m in metrics:
        if m in df.columns:
            w_mean = winners[m].mean()
            l_mean = losers[m].mean()
            diff = w_mean - l_mean
            print(f"  {m:12}: Winners={w_mean:.2f}, Losers={l_mean:.2f}, Diff={diff:+.2f}")

    # Patrones de perdida
    print(f"\nPATRONES EN PERDEDORES:")
    if len(losers) > 0:
        # RSI extremo
        rsi_extreme = losers[(losers['rsi14'] > 70) | (losers['rsi14'] < 30)]
        print(f"  RSI extremo (>70 o <30): {len(rsi_extreme)} ({len(rsi_extreme)/len(losers)*100:.0f}%)")

        # BB extremo
        bb_extreme = losers[(losers['bb_pos'] > 0.9) | (losers['bb_pos'] < 0.1)]
        print(f"  BB extremo (>0.9 o <0.1): {len(bb_extreme)} ({len(bb_extreme)/len(losers)*100:.0f}%)")

        # Baja conviction
        low_conv = losers[losers['conviction'] < 2]
        print(f"  Baja conviction (<2): {len(low_conv)} ({len(low_conv)/len(losers)*100:.0f}%)")

        # ADX bajo
        low_adx = losers[losers['adx'] < 20]
        print(f"  ADX bajo (<20 sin tendencia): {len(low_adx)} ({len(low_adx)/len(losers)*100:.0f}%)")

    # Por mes
    print(f"\nPOR MES:")
    df['month'] = pd.to_datetime(df['entry_ts']).dt.to_period('M')
    monthly = df.groupby('month').agg({
        'win': ['count', 'sum', 'mean'],
        'pnl_pct': 'sum'
    })
    monthly.columns = ['trades', 'wins', 'wr', 'pnl']
    monthly['wr'] = monthly['wr'] * 100
    monthly['pnl'] = monthly['pnl'] * 100

    for idx, row in monthly.iterrows():
        print(f"  {idx}: {int(row['trades'])} trades, WR={row['wr']:.0f}%, PnL={row['pnl']:.1f}%")

    return df

if __name__ == '__main__':
    print("Cargando datos BTC...")
    df = load_data()
    print(f"  {len(df)} velas: {df.index[0]} a {df.index[-1]}")

    print("\nCalculando features...")
    feat = compute_features(df)

    print("\nCargando modelo BTC...")
    model = joblib.load(MODELS_DIR / 'v95_v7_BTCUSDT.pkl')
    print(f"  {len(model.feature_name_)} features")

    # Test ultimos 30 dias
    end_date = df.index[-1]
    start_date = end_date - pd.Timedelta(days=30)

    print(f"\nSimulando trades: {start_date.date()} a {end_date.date()}...")
    trades = simulate_trades(df, feat, model, start_date, end_date)

    # Analizar
    trades_df = analyze_trades(trades)

    # Test ultimos 3 meses para mas datos
    print("\n" + "="*60)
    print("TEST EXTENDIDO: Ultimos 3 meses")
    print("="*60)

    start_date_3m = end_date - pd.Timedelta(days=90)
    trades_3m = simulate_trades(df, feat, model, start_date_3m, end_date)
    analyze_trades(trades_3m)
