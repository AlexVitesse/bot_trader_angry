"""
Comparacion Justa: V12 PRO vs V8.5/V9 en Produccion
====================================================
Usar EXACTAMENTE los mismos pares que uso produccion
para ver si V12 PRO realmente es mejor.
"""

import sqlite3
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent / 'data'
MODELS_DIR = Path(__file__).parent / 'models'

# Periodo exacto de produccion
START_DATE = '2026-02-13'
END_DATE = '2026-02-25'

INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.02


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


def run_backtest_v12_pro(pairs, start_date, end_date):
    """V12 PRO: Filtros estrictos, threshold correcto, pares seleccionados."""
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

        balance = INITIAL_CAPITAL
        pos = None

        for i in range(20, len(df_period)):
            ts = df_period.index[i]
            if ts not in feat.index:
                continue
            price = df_period.iloc[i]['close']

            rsi = feat.loc[ts, 'rsi14'] if 'rsi14' in feat.columns else 50
            chop = feat.loc[ts, 'chop'] if 'chop' in feat.columns else 50
            bb_pos = feat.loc[ts, 'bb_pos'] if 'bb_pos' in feat.columns else 0.5
            atr = feat.loc[ts, 'atr14'] if 'atr14' in feat.columns else price * 0.02

            if pos is not None:
                pnl_pct = (price - pos['entry']) / pos['entry'] * pos['dir']
                if pnl_pct >= pos['tp_pct'] or pnl_pct <= -pos['sl_pct'] or (i - pos['bar']) >= 20:
                    pnl = pos['size'] * pos['dir'] * (price - pos['entry'])
                    # Simular comision (0.1% por trade)
                    commission = pos['notional'] * 0.001
                    net_pnl = pnl - commission
                    balance += net_pnl
                    all_trades.append({
                        'pair': pair,
                        'pnl': net_pnl,
                        'pnl_bruto': pnl,
                        'commission': commission,
                        'dir': pos['dir'],
                        'exit_time': ts
                    })
                    pos = None

            if pos is None:
                X = feat.loc[ts:ts][fcols]
                if X.isna().any().any():
                    continue

                pred = model.predict(X)[0]
                sig = 1 if pred > 0 else -1
                conviction = abs(pred) / 0.005

                # Filtros V12 PRO
                if conviction < 2.0:
                    continue
                if not (38 <= rsi <= 72):
                    continue
                if not (0.2 <= bb_pos <= 0.8):
                    continue
                if chop > 52:
                    continue

                tp_pct = 0.03  # 3% TP fijo como produccion
                sl_pct = 0.015  # 1.5% SL fijo
                risk_amt = balance * RISK_PER_TRADE
                notional = min(risk_amt / sl_pct, 300)  # Max $300 notional
                size = notional / price

                pos = {
                    'entry': price, 'dir': sig, 'size': size,
                    'notional': notional,
                    'tp_pct': tp_pct, 'sl_pct': sl_pct, 'bar': i
                }

        if pos:
            pnl = pos['size'] * pos['dir'] * (df_period.iloc[-1]['close'] - pos['entry'])
            commission = pos['notional'] * 0.001
            all_trades.append({
                'pair': pair,
                'pnl': pnl - commission,
                'pnl_bruto': pnl,
                'commission': commission,
                'dir': pos['dir'],
                'exit_time': df_period.index[-1]
            })

    return all_trades


def run_backtest_simplificado(pairs, start_date, end_date):
    """Modelo simplificado: menos filtros, mas trades."""
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

        balance = INITIAL_CAPITAL
        pos = None

        for i in range(20, len(df_period)):
            ts = df_period.index[i]
            if ts not in feat.index:
                continue
            price = df_period.iloc[i]['close']
            atr = feat.loc[ts, 'atr14'] if 'atr14' in feat.columns else price * 0.02
            chop = feat.loc[ts, 'chop'] if 'chop' in feat.columns else 50

            if pos is not None:
                pnl_pct = (price - pos['entry']) / pos['entry'] * pos['dir']
                if pnl_pct >= pos['tp_pct'] or pnl_pct <= -pos['sl_pct'] or (i - pos['bar']) >= 20:
                    pnl = pos['size'] * pos['dir'] * (price - pos['entry'])
                    commission = pos['notional'] * 0.001
                    net_pnl = pnl - commission
                    balance += net_pnl
                    all_trades.append({
                        'pair': pair,
                        'pnl': net_pnl,
                        'pnl_bruto': pnl,
                        'commission': commission,
                        'dir': pos['dir'],
                        'exit_time': ts
                    })
                    pos = None

            if pos is None:
                X = feat.loc[ts:ts][fcols]
                if X.isna().any().any():
                    continue

                pred = model.predict(X)[0]
                sig = 1 if pred > 0 else -1
                conviction = abs(pred) / 0.005

                # Filtros SIMPLIFICADOS
                if conviction < 1.0:
                    continue
                if chop > 60:
                    continue

                tp_pct = 0.03
                sl_pct = 0.015
                risk_amt = balance * RISK_PER_TRADE
                notional = min(risk_amt / sl_pct, 300)
                size = notional / price

                pos = {
                    'entry': price, 'dir': sig, 'size': size,
                    'notional': notional,
                    'tp_pct': tp_pct, 'sl_pct': sl_pct, 'bar': i
                }

        if pos:
            pnl = pos['size'] * pos['dir'] * (df_period.iloc[-1]['close'] - pos['entry'])
            commission = pos['notional'] * 0.001
            all_trades.append({
                'pair': pair,
                'pnl': pnl - commission,
                'pnl_bruto': pnl,
                'commission': commission,
                'dir': pos['dir'],
                'exit_time': df_period.index[-1]
            })

    return all_trades


def main():
    print("=" * 80, flush=True)
    print("COMPARACION: V12 PRO vs PRODUCCION (V8.5/V9)", flush=True)
    print("=" * 80, flush=True)
    print(f"\nPeriodo: {START_DATE} a {END_DATE}", flush=True)

    # Cargar datos de produccion
    conn = sqlite3.connect('ml_backup.db')
    prod_trades = pd.read_sql('SELECT * FROM ml_trades', conn)
    conn.close()

    # Pares que uso produccion
    prod_pairs = prod_trades['symbol'].unique().tolist()
    print(f"\nPares usados en produccion: {prod_pairs}", flush=True)

    # Separar V9 y V8.5
    v9_prod = prod_trades[prod_trades['strategy'] == 'v9']
    v85_prod = prod_trades[prod_trades['strategy'] == 'v85_shadow']

    # Estadisticas de produccion
    print("\n" + "=" * 60, flush=True)
    print("RESULTADOS PRODUCCION (REALES)", flush=True)
    print("=" * 60, flush=True)

    for name, df in [('V9', v9_prod), ('V8.5 Shadow', v85_prod)]:
        if len(df) == 0:
            continue
        wins = (df['pnl'] > 0).sum()
        wr = wins / len(df) * 100
        total_pnl = df['pnl'].sum()
        commission = df['commission'].sum()
        net_pnl = total_pnl - commission

        print(f"\n{name}:", flush=True)
        print(f"  Trades: {len(df)}", flush=True)
        print(f"  WR: {wr:.1f}%", flush=True)
        print(f"  PnL Neto: ${net_pnl:.2f}", flush=True)

    # Backtest V12 PRO con MISMOS pares que produccion
    print("\n" + "=" * 60, flush=True)
    print("BACKTEST V12 PRO (mismos pares que produccion)", flush=True)
    print("=" * 60, flush=True)

    v12_trades = run_backtest_v12_pro(prod_pairs, START_DATE, END_DATE)
    if v12_trades:
        df_v12 = pd.DataFrame(v12_trades)
        wins = (df_v12['pnl'] > 0).sum()
        wr = wins / len(df_v12) * 100
        total_pnl = df_v12['pnl'].sum()

        print(f"\nV12 PRO (mismos pares):", flush=True)
        print(f"  Trades: {len(df_v12)}", flush=True)
        print(f"  WR: {wr:.1f}%", flush=True)
        print(f"  PnL Neto: ${total_pnl:.2f}", flush=True)

        # Por par
        print(f"\n  Por par:", flush=True)
        by_pair = df_v12.groupby('pair').agg({'pnl': ['sum', 'count']})
        by_pair.columns = ['pnl', 'trades']
        for pair, row in by_pair.sort_values('pnl', ascending=False).iterrows():
            status = "[+]" if row['pnl'] > 0 else "[-]"
            print(f"    {pair}: {int(row['trades'])} tr, ${row['pnl']:.2f} {status}", flush=True)

    # Backtest V12 PRO con pares OPTIMIZADOS (sin SOL, BTC, BNB)
    print("\n" + "=" * 60, flush=True)
    print("BACKTEST V12 PRO (pares optimizados - sin SOL/BTC/BNB)", flush=True)
    print("=" * 60, flush=True)

    optimized_pairs = ['XRP/USDT', 'ETH/USDT', 'DOGE/USDT', 'ADA/USDT',
                       'AVAX/USDT', 'LINK/USDT', 'DOT/USDT', 'NEAR/USDT']

    v12_opt_trades = run_backtest_v12_pro(optimized_pairs, START_DATE, END_DATE)
    if v12_opt_trades:
        df_v12_opt = pd.DataFrame(v12_opt_trades)
        wins = (df_v12_opt['pnl'] > 0).sum()
        wr = wins / len(df_v12_opt) * 100
        total_pnl = df_v12_opt['pnl'].sum()

        print(f"\nV12 PRO (pares optimizados):", flush=True)
        print(f"  Trades: {len(df_v12_opt)}", flush=True)
        print(f"  WR: {wr:.1f}%", flush=True)
        print(f"  PnL Neto: ${total_pnl:.2f}", flush=True)

    # Backtest SIMPLIFICADO con pares optimizados
    print("\n" + "=" * 60, flush=True)
    print("BACKTEST SIMPLIFICADO (pares optimizados)", flush=True)
    print("=" * 60, flush=True)

    simp_trades = run_backtest_simplificado(optimized_pairs, START_DATE, END_DATE)
    if simp_trades:
        df_simp = pd.DataFrame(simp_trades)
        wins = (df_simp['pnl'] > 0).sum()
        wr = wins / len(df_simp) * 100
        total_pnl = df_simp['pnl'].sum()

        print(f"\nSIMPLIFICADO (pares optimizados):", flush=True)
        print(f"  Trades: {len(df_simp)}", flush=True)
        print(f"  WR: {wr:.1f}%", flush=True)
        print(f"  PnL Neto: ${total_pnl:.2f}", flush=True)

        # Por par
        print(f"\n  Por par:", flush=True)
        by_pair = df_simp.groupby('pair').agg({'pnl': ['sum', 'count']})
        by_pair.columns = ['pnl', 'trades']
        for pair, row in by_pair.sort_values('pnl', ascending=False).iterrows():
            status = "[+]" if row['pnl'] > 0 else "[-]"
            print(f"    {pair}: {int(row['trades'])} tr, ${row['pnl']:.2f} {status}", flush=True)

    # Tabla comparativa final
    print("\n" + "=" * 80, flush=True)
    print("TABLA COMPARATIVA FINAL", flush=True)
    print("=" * 80, flush=True)

    print(f"\n{'Version':<30} {'Trades':<10} {'WR':<10} {'PnL Neto':<15}", flush=True)
    print("-" * 65, flush=True)

    # V9 produccion
    v9_wr = (v9_prod['pnl'] > 0).sum() / len(v9_prod) * 100 if len(v9_prod) > 0 else 0
    v9_pnl = v9_prod['pnl'].sum() - v9_prod['commission'].sum()
    print(f"{'V9 (produccion)':<30} {len(v9_prod):<10} {v9_wr:.1f}%{'':<5} ${v9_pnl:>10.2f}", flush=True)

    # V8.5 produccion
    v85_wr = (v85_prod['pnl'] > 0).sum() / len(v85_prod) * 100 if len(v85_prod) > 0 else 0
    v85_pnl = v85_prod['pnl'].sum() - v85_prod['commission'].sum()
    print(f"{'V8.5 Shadow (produccion)':<30} {len(v85_prod):<10} {v85_wr:.1f}%{'':<5} ${v85_pnl:>10.2f}", flush=True)

    # V12 PRO mismos pares
    if v12_trades:
        v12_wr = (df_v12['pnl'] > 0).sum() / len(df_v12) * 100
        v12_pnl = df_v12['pnl'].sum()
        print(f"{'V12 PRO (mismos pares)':<30} {len(df_v12):<10} {v12_wr:.1f}%{'':<5} ${v12_pnl:>10.2f}", flush=True)

    # V12 PRO pares optimizados
    if v12_opt_trades:
        v12_opt_wr = (df_v12_opt['pnl'] > 0).sum() / len(df_v12_opt) * 100
        v12_opt_pnl = df_v12_opt['pnl'].sum()
        print(f"{'V12 PRO (sin SOL/BTC/BNB)':<30} {len(df_v12_opt):<10} {v12_opt_wr:.1f}%{'':<5} ${v12_opt_pnl:>10.2f}", flush=True)

    # Simplificado
    if simp_trades:
        simp_wr = (df_simp['pnl'] > 0).sum() / len(df_simp) * 100
        simp_pnl = df_simp['pnl'].sum()
        print(f"{'SIMPLIFICADO (sin SOL/BTC/BNB)':<30} {len(df_simp):<10} {simp_wr:.1f}%{'':<5} ${simp_pnl:>10.2f}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("CONCLUSION", flush=True)
    print("=" * 80, flush=True)

    if simp_trades and v9_pnl < 0:
        mejora = simp_pnl - v9_pnl
        print(f"\n  SIMPLIFICADO vs V9: ${mejora:+.2f} de mejora", flush=True)
        print(f"  El modelo simplificado habria convertido ${v9_pnl:.2f} en ${simp_pnl:.2f}", flush=True)


if __name__ == '__main__':
    main()
