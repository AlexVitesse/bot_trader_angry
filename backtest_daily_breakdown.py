"""
Analisis Dia por Dia - Ultimos 14 dias
======================================
Ver exactamente que paso cada dia
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

DATA_DIR = Path(__file__).parent / 'data'
MODELS_DIR = Path(__file__).parent / 'models'

END_DATE = '2026-02-25'
START_DATE = '2026-02-11'

INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.02

PAIRS = ['XRP/USDT', 'ETH/USDT', 'DOGE/USDT', 'ADA/USDT',
         'AVAX/USDT', 'LINK/USDT', 'DOT/USDT', 'NEAR/USDT']


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


def collect_all_trades(pairs, start_date, end_date):
    """Recolectar todos los trades con timestamp exacto."""
    all_trades = []

    for pair in pairs:
        df = load_data_4h(pair)
        model = load_model(pair)

        if df is None or model is None:
            continue

        df_period = df[(df.index >= start_date) & (df.index < end_date)]
        if len(df_period) < 10:
            continue

        feat = compute_features(df)
        feat = feat[(feat.index >= start_date) & (feat.index < end_date)]
        fcols = [c for c in model.feature_name_ if c in feat.columns]

        if not fcols:
            continue

        pos = None
        balance = INITIAL_CAPITAL

        for i in range(20, len(df_period)):
            ts = df_period.index[i]
            if ts not in feat.index:
                continue
            price = df_period.iloc[i]['close']
            atr = feat.loc[ts, 'atr14'] if 'atr14' in feat.columns else price * 0.02
            chop = feat.loc[ts, 'chop'] if 'chop' in feat.columns else 50

            if pos is not None:
                pnl_pct = (price - pos['entry']) / pos['entry'] * pos['dir']
                exit_reason = None
                if pnl_pct >= pos['tp_pct']:
                    exit_reason = 'TP'
                elif pnl_pct <= -pos['sl_pct']:
                    exit_reason = 'SL'
                elif (i - pos['bar']) >= 20:
                    exit_reason = 'TIME'

                if exit_reason:
                    pnl = pos['size'] * pos['dir'] * (price - pos['entry'])
                    balance += pnl
                    all_trades.append({
                        'pair': pair,
                        'entry_time': pos['entry_time'],
                        'exit_time': ts,
                        'exit_date': ts.date(),
                        'direction': 'LONG' if pos['dir'] == 1 else 'SHORT',
                        'entry_price': pos['entry'],
                        'exit_price': price,
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'win': pnl > 0
                    })
                    pos = None

            if pos is None:
                X = feat.loc[ts:ts][fcols]
                if X.isna().any().any():
                    continue

                pred = model.predict(X)[0]
                # Usando threshold CORRECTO
                sig = 1 if pred > 0 else -1
                conviction = abs(pred) / 0.005

                # Filtro basico
                if conviction < 1.0:
                    continue
                if chop > 60:
                    continue

                tp_pct = atr / price * 2.0
                sl_pct = atr / price * 1.0
                risk_amt = balance * RISK_PER_TRADE
                size = risk_amt / (sl_pct * price) if sl_pct > 0 else 0
                pos = {
                    'entry': price,
                    'entry_time': ts,
                    'dir': sig,
                    'size': size,
                    'tp_pct': tp_pct,
                    'sl_pct': sl_pct,
                    'bar': i
                }

        # Cerrar posicion abierta al final
        if pos:
            price = df_period.iloc[-1]['close']
            pnl = pos['size'] * pos['dir'] * (price - pos['entry'])
            all_trades.append({
                'pair': pair,
                'entry_time': pos['entry_time'],
                'exit_time': df_period.index[-1],
                'exit_date': df_period.index[-1].date(),
                'direction': 'LONG' if pos['dir'] == 1 else 'SHORT',
                'entry_price': pos['entry'],
                'exit_price': price,
                'pnl': pnl,
                'exit_reason': 'OPEN',
                'win': pnl > 0
            })

    return all_trades


def main():
    print("=" * 80, flush=True)
    print("ANALISIS DIA POR DIA - ULTIMOS 14 DIAS", flush=True)
    print("=" * 80, flush=True)
    print(f"\nPeriodo: {START_DATE} a {END_DATE}", flush=True)

    # Recolectar todos los trades
    trades = collect_all_trades(PAIRS, START_DATE, END_DATE)

    if not trades:
        print("\nNo se encontraron trades en este periodo", flush=True)
        return

    df_trades = pd.DataFrame(trades)

    # Mostrar primero el mercado (BTC como referencia)
    btc_df = load_data_4h('BTC/USDT')
    if btc_df is not None:
        btc_period = btc_df[(btc_df.index >= START_DATE) & (btc_df.index < END_DATE)]
        if len(btc_period) > 0:
            btc_start = btc_period.iloc[0]['close']
            btc_end = btc_period.iloc[-1]['close']
            btc_change = (btc_end - btc_start) / btc_start * 100
            print(f"\nBTC en este periodo: ${btc_start:,.0f} -> ${btc_end:,.0f} ({btc_change:+.1f}%)", flush=True)

    print(f"\nTotal trades: {len(trades)}", flush=True)
    print(f"Wins: {df_trades['win'].sum()} ({df_trades['win'].mean()*100:.1f}%)", flush=True)
    print(f"PnL total: ${df_trades['pnl'].sum():.0f}", flush=True)

    # Agrupar por dia
    print("\n" + "=" * 80, flush=True)
    print("DESGLOSE POR DIA", flush=True)
    print("=" * 80, flush=True)

    daily_pnl = df_trades.groupby('exit_date').agg({
        'pnl': ['sum', 'count'],
        'win': 'sum'
    })
    daily_pnl.columns = ['pnl', 'trades', 'wins']
    daily_pnl['wr'] = daily_pnl['wins'] / daily_pnl['trades'] * 100

    print(f"\n{'Fecha':<15} {'Trades':<10} {'Wins':<8} {'WR':<10} {'PnL':<15} {'Status'}", flush=True)
    print("-" * 65, flush=True)

    cumulative_pnl = 0
    worst_day = None
    worst_pnl = 0

    for date, row in daily_pnl.iterrows():
        cumulative_pnl += row['pnl']
        status = "[+]" if row['pnl'] > 0 else "[-]"
        wr_mark = "**" if row['wr'] >= 50 else ""
        print(f"{str(date):<15} {int(row['trades']):<10} {int(row['wins']):<8} {row['wr']:.0f}%{wr_mark:<5} ${row['pnl']:>10,.0f}   {status} (acum: ${cumulative_pnl:,.0f})", flush=True)

        if row['pnl'] < worst_pnl:
            worst_pnl = row['pnl']
            worst_day = date

    # Estadisticas
    print("\n" + "=" * 80, flush=True)
    print("ESTADISTICAS", flush=True)
    print("=" * 80, flush=True)

    print(f"\n  Dias con trades: {len(daily_pnl)}", flush=True)
    print(f"  Dias positivos: {(daily_pnl['pnl'] > 0).sum()}", flush=True)
    print(f"  Dias negativos: {(daily_pnl['pnl'] < 0).sum()}", flush=True)
    print(f"\n  Mejor dia: ${daily_pnl['pnl'].max():,.0f}", flush=True)
    print(f"  Peor dia: ${daily_pnl['pnl'].min():,.0f} ({worst_day})", flush=True)
    print(f"  PnL promedio/dia: ${daily_pnl['pnl'].mean():,.0f}", flush=True)

    # Max drawdown
    cumulative = daily_pnl['pnl'].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_dd = drawdown.min()

    print(f"\n  Max Drawdown: ${max_dd:,.0f}", flush=True)

    # Desglose por direccion
    print("\n" + "=" * 80, flush=True)
    print("DESGLOSE POR DIRECCION", flush=True)
    print("=" * 80, flush=True)

    for direction in ['LONG', 'SHORT']:
        dir_trades = df_trades[df_trades['direction'] == direction]
        if len(dir_trades) > 0:
            wr = dir_trades['win'].mean() * 100
            pnl = dir_trades['pnl'].sum()
            wr_mark = "**" if wr >= 50 else ""
            print(f"\n  {direction}: {len(dir_trades)} trades, WR {wr:.1f}%{wr_mark}, PnL ${pnl:,.0f}", flush=True)

    # Desglose por par
    print("\n" + "=" * 80, flush=True)
    print("DESGLOSE POR PAR", flush=True)
    print("=" * 80, flush=True)

    pair_stats = df_trades.groupby('pair').agg({
        'pnl': 'sum',
        'win': ['sum', 'count']
    })
    pair_stats.columns = ['pnl', 'wins', 'trades']
    pair_stats['wr'] = pair_stats['wins'] / pair_stats['trades'] * 100
    pair_stats = pair_stats.sort_values('pnl', ascending=False)

    print(f"\n{'Par':<12} {'Trades':<10} {'WR':<10} {'PnL':<15}", flush=True)
    print("-" * 50, flush=True)

    for pair, row in pair_stats.iterrows():
        wr_mark = "**" if row['wr'] >= 50 else ""
        status = "[+]" if row['pnl'] > 0 else "[-]"
        print(f"{pair:<12} {int(row['trades']):<10} {row['wr']:.0f}%{wr_mark:<5} ${row['pnl']:>10,.0f} {status}", flush=True)

    # Ultimos 3 dias detallados
    print("\n" + "=" * 80, flush=True)
    print("ULTIMOS 3 DIAS - DETALLE DE TRADES", flush=True)
    print("=" * 80, flush=True)

    recent_dates = sorted(df_trades['exit_date'].unique())[-3:]

    for date in recent_dates:
        day_trades = df_trades[df_trades['exit_date'] == date].sort_values('exit_time')
        day_pnl = day_trades['pnl'].sum()
        print(f"\n{date} - PnL: ${day_pnl:,.0f}", flush=True)
        print("-" * 60, flush=True)

        for _, trade in day_trades.iterrows():
            status = "WIN" if trade['win'] else "LOSS"
            print(f"  {trade['pair']:<10} {trade['direction']:<6} {trade['exit_reason']:<5} ${trade['pnl']:>8,.0f} [{status}]", flush=True)


if __name__ == '__main__':
    main()
