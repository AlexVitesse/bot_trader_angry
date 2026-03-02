"""
Backtest Comparativo: Últimos 14 días
=====================================
Comparar V8.5, V9, V12 PRO y Modelo Simplificado
en el período reciente donde V8.5 y V9 perdieron.

Período: 11 Feb 2026 - 25 Feb 2026
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
import json
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).parent / 'data'
MODELS_DIR = Path(__file__).parent / 'models'

# Período de análisis
END_DATE = '2026-02-25'
START_DATE = '2026-02-11'

INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.02

# Pares a analizar
PAIRS_V12_PRO = ['XRP/USDT', 'ETH/USDT', 'DOGE/USDT', 'ADA/USDT',
                  'AVAX/USDT', 'LINK/USDT', 'DOT/USDT', 'NEAR/USDT']

# Pares simplificados (solo los mejores)
PAIRS_SIMPLIFIED = ['NEAR/USDT', 'XRP/USDT', 'DOT/USDT', 'ETH/USDT', 'AVAX/USDT', 'LINK/USDT']


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

    # Choppiness Index
    atr_1 = ta.atr(h, l, c, length=1)
    atr_sum = atr_1.rolling(14).sum()
    high_max = h.rolling(14).max()
    low_min = l.rolling(14).min()
    feat['chop'] = 100 * np.log10(atr_sum / (high_max - low_min + 1e-10)) / np.log10(14)

    return feat


def run_backtest_v85(df, model, start_date, end_date):
    """V8.5: Modelo base con filtros originales (INCORRECTOS - pred > 0.5)."""
    df_period = df[(df.index >= start_date) & (df.index < end_date)]
    if len(df_period) < 10:
        return None

    feat = compute_features(df)
    feat = feat[(feat.index >= start_date) & (feat.index < end_date)]
    fcols = [c for c in model.feature_name_ if c in feat.columns]
    if not fcols:
        return None

    trades = []
    balance = INITIAL_CAPITAL
    pos = None

    for i in range(20, len(df_period)):
        ts = df_period.index[i]
        if ts not in feat.index:
            continue
        price = df_period.iloc[i]['close']
        atr = feat.loc[ts, 'atr14'] if 'atr14' in feat.columns else price * 0.02

        if pos is not None:
            pnl_pct = (price - pos['entry']) / pos['entry'] * pos['dir']
            if pnl_pct >= pos['tp_pct'] or pnl_pct <= -pos['sl_pct'] or (i - pos['bar']) >= 20:
                pnl = pos['size'] * pos['dir'] * (price - pos['entry'])
                balance += pnl
                trades.append({'pnl': pnl, 'dir': pos['dir'], 'ts': ts})
                pos = None

        if pos is None:
            X = feat.loc[ts:ts][fcols]
            if X.isna().any().any():
                continue

            pred = model.predict(X)[0]
            # V8.5 BUG: usa threshold 0.5 (INCORRECTO)
            sig = 1 if pred > 0.5 else -1
            conviction = abs(pred - 0.5) * 10

            if conviction < 1.5:
                continue

            tp_pct = atr / price * 2.0
            sl_pct = atr / price * 1.0
            risk_amt = balance * RISK_PER_TRADE
            size = risk_amt / (sl_pct * price) if sl_pct > 0 else 0
            pos = {'entry': price, 'dir': sig, 'size': size,
                   'tp_pct': tp_pct, 'sl_pct': sl_pct, 'bar': i}

    if pos:
        pnl = pos['size'] * pos['dir'] * (df_period.iloc[-1]['close'] - pos['entry'])
        trades.append({'pnl': pnl, 'dir': pos['dir'], 'ts': df_period.index[-1]})

    if not trades:
        return None

    wins = sum(1 for t in trades if t['pnl'] > 0)
    total_pnl = sum(t['pnl'] for t in trades)

    return {
        'trades': len(trades),
        'wins': wins,
        'wr': wins / len(trades) * 100 if trades else 0,
        'pnl': total_pnl,
        'final_balance': INITIAL_CAPITAL + total_pnl,
    }


def run_backtest_v12_pro(df, model, start_date, end_date):
    """V12 PRO: Filtros estrictos + threshold CORRECTO."""
    df_period = df[(df.index >= start_date) & (df.index < end_date)]
    if len(df_period) < 10:
        return None

    feat = compute_features(df)
    feat = feat[(feat.index >= start_date) & (feat.index < end_date)]
    fcols = [c for c in model.feature_name_ if c in feat.columns]
    if not fcols:
        return None

    trades = []
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
                balance += pnl
                trades.append({'pnl': pnl, 'dir': pos['dir'], 'ts': ts})
                pos = None

        if pos is None:
            X = feat.loc[ts:ts][fcols]
            if X.isna().any().any():
                continue

            pred = model.predict(X)[0]
            # V12 PRO: threshold CORRECTO
            sig = 1 if pred > 0 else -1
            conviction = abs(pred) / 0.005  # pred=0.5%->conv=1

            # Filtros V12 PRO
            if conviction < 2.0:
                continue
            if not (38 <= rsi <= 72):
                continue
            if not (0.2 <= bb_pos <= 0.8):
                continue
            if chop > 52:
                continue

            tp_pct = atr / price * 2.0
            sl_pct = atr / price * 1.0
            risk_amt = balance * RISK_PER_TRADE
            size = risk_amt / (sl_pct * price) if sl_pct > 0 else 0
            pos = {'entry': price, 'dir': sig, 'size': size,
                   'tp_pct': tp_pct, 'sl_pct': sl_pct, 'bar': i}

    if pos:
        pnl = pos['size'] * pos['dir'] * (df_period.iloc[-1]['close'] - pos['entry'])
        trades.append({'pnl': pnl, 'dir': pos['dir'], 'ts': df_period.index[-1]})

    if not trades:
        return None

    wins = sum(1 for t in trades if t['pnl'] > 0)
    total_pnl = sum(t['pnl'] for t in trades)

    return {
        'trades': len(trades),
        'wins': wins,
        'wr': wins / len(trades) * 100 if trades else 0,
        'pnl': total_pnl,
        'final_balance': INITIAL_CAPITAL + total_pnl,
    }


def run_backtest_simplified(df, model, start_date, end_date):
    """Modelo Simplificado: Sin filtros complejos, solo threshold correcto + conviction."""
    df_period = df[(df.index >= start_date) & (df.index < end_date)]
    if len(df_period) < 10:
        return None

    feat = compute_features(df)
    feat = feat[(feat.index >= start_date) & (feat.index < end_date)]
    fcols = [c for c in model.feature_name_ if c in feat.columns]
    if not fcols:
        return None

    trades = []
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
                balance += pnl
                trades.append({'pnl': pnl, 'dir': pos['dir'], 'ts': ts})
                pos = None

        if pos is None:
            X = feat.loc[ts:ts][fcols]
            if X.isna().any().any():
                continue

            pred = model.predict(X)[0]
            # SIMPLIFICADO: threshold correcto
            sig = 1 if pred > 0 else -1
            conviction = abs(pred) / 0.005

            # Solo filtro básico: conviction + chop
            if conviction < 1.0:
                continue
            if chop > 60:
                continue

            tp_pct = atr / price * 2.0
            sl_pct = atr / price * 1.0
            risk_amt = balance * RISK_PER_TRADE
            size = risk_amt / (sl_pct * price) if sl_pct > 0 else 0
            pos = {'entry': price, 'dir': sig, 'size': size,
                   'tp_pct': tp_pct, 'sl_pct': sl_pct, 'bar': i}

    if pos:
        pnl = pos['size'] * pos['dir'] * (df_period.iloc[-1]['close'] - pos['entry'])
        trades.append({'pnl': pnl, 'dir': pos['dir'], 'ts': df_period.index[-1]})

    if not trades:
        return None

    wins = sum(1 for t in trades if t['pnl'] > 0)
    total_pnl = sum(t['pnl'] for t in trades)

    return {
        'trades': len(trades),
        'wins': wins,
        'wr': wins / len(trades) * 100 if trades else 0,
        'pnl': total_pnl,
        'final_balance': INITIAL_CAPITAL + total_pnl,
    }


def main():
    print("=" * 80, flush=True)
    print("BACKTEST COMPARATIVO: ULTIMOS 14 DIAS", flush=True)
    print("=" * 80, flush=True)
    print(f"\nPeriodo: {START_DATE} a {END_DATE}", flush=True)
    print("Objetivo: Comparar V8.5, V12 PRO y Modelo Simplificado\n", flush=True)

    results = {
        'V8.5 (bug)': {'trades': 0, 'wins': 0, 'pnl': 0},
        'V12 PRO': {'trades': 0, 'wins': 0, 'pnl': 0},
        'SIMPLIFICADO': {'trades': 0, 'wins': 0, 'pnl': 0},
    }

    results_by_pair = {}

    print("=" * 60, flush=True)
    print("RESULTADOS POR PAR", flush=True)
    print("=" * 60, flush=True)

    for pair in PAIRS_V12_PRO:
        print(f"\n{pair}:", flush=True)

        df = load_data_4h(pair)
        model = load_model(pair)

        if df is None or model is None:
            print(f"  [!] Sin datos o modelo", flush=True)
            continue

        results_by_pair[pair] = {}

        # V8.5 (con bug)
        r85 = run_backtest_v85(df, model, START_DATE, END_DATE)
        if r85:
            results_by_pair[pair]['V8.5'] = r85
            results['V8.5 (bug)']['trades'] += r85['trades']
            results['V8.5 (bug)']['wins'] += r85['wins']
            results['V8.5 (bug)']['pnl'] += r85['pnl']
            wr_mark = "**" if r85['wr'] >= 50 else ""
            print(f"  V8.5 (bug): {r85['trades']} tr, WR {r85['wr']:.1f}%{wr_mark}, PnL ${r85['pnl']:.0f}", flush=True)
        else:
            print(f"  V8.5 (bug): 0 trades", flush=True)

        # V12 PRO
        r12 = run_backtest_v12_pro(df, model, START_DATE, END_DATE)
        if r12:
            results_by_pair[pair]['V12 PRO'] = r12
            results['V12 PRO']['trades'] += r12['trades']
            results['V12 PRO']['wins'] += r12['wins']
            results['V12 PRO']['pnl'] += r12['pnl']
            wr_mark = "**" if r12['wr'] >= 50 else ""
            print(f"  V12 PRO:   {r12['trades']} tr, WR {r12['wr']:.1f}%{wr_mark}, PnL ${r12['pnl']:.0f}", flush=True)
        else:
            print(f"  V12 PRO:   0 trades", flush=True)

        # Simplificado (solo para pares seleccionados)
        if pair in PAIRS_SIMPLIFIED:
            rs = run_backtest_simplified(df, model, START_DATE, END_DATE)
            if rs:
                results_by_pair[pair]['SIMPLIFICADO'] = rs
                results['SIMPLIFICADO']['trades'] += rs['trades']
                results['SIMPLIFICADO']['wins'] += rs['wins']
                results['SIMPLIFICADO']['pnl'] += rs['pnl']
                wr_mark = "**" if rs['wr'] >= 50 else ""
                print(f"  SIMPLIF:   {rs['trades']} tr, WR {rs['wr']:.1f}%{wr_mark}, PnL ${rs['pnl']:.0f}", flush=True)
            else:
                print(f"  SIMPLIF:   0 trades", flush=True)

    # Resumen
    print("\n" + "=" * 80, flush=True)
    print("RESUMEN COMPARATIVO - ULTIMOS 14 DIAS", flush=True)
    print("=" * 80, flush=True)

    print(f"\n{'Version':<20} {'Trades':<10} {'Wins':<10} {'WR':<12} {'PnL':<15} {'Status'}", flush=True)
    print("-" * 75, flush=True)

    for version, data in results.items():
        if data['trades'] > 0:
            wr = data['wins'] / data['trades'] * 100
            wr_mark = "**" if wr >= 50 else ""
            status = "GANO" if data['pnl'] > 0 else "PERDIO"
            status_mark = "[+]" if data['pnl'] > 0 else "[-]"
            print(f"{version:<20} {data['trades']:<10} {data['wins']:<10} {wr:.1f}%{wr_mark:<6} ${data['pnl']:<14,.0f} {status_mark} {status}", flush=True)
        else:
            print(f"{version:<20} {'0':<10} {'0':<10} {'N/A':<12} {'$0':<15} N/A", flush=True)

    # Analisis
    print("\n" + "=" * 80, flush=True)
    print("ANALISIS", flush=True)
    print("=" * 80, flush=True)

    # Mejor version
    best_version = max(results.keys(), key=lambda x: results[x]['pnl'])
    print(f"\n  MEJOR VERSION: {best_version} (PnL ${results[best_version]['pnl']:,.0f})", flush=True)

    # Diferencia V12 PRO vs V8.5
    diff = results['V12 PRO']['pnl'] - results['V8.5 (bug)']['pnl']
    print(f"\n  V12 PRO vs V8.5: ${diff:+,.0f}", flush=True)

    if results['SIMPLIFICADO']['trades'] > 0:
        diff_simp = results['SIMPLIFICADO']['pnl'] - results['V12 PRO']['pnl']
        print(f"  SIMPLIFICADO vs V12 PRO: ${diff_simp:+,.0f}", flush=True)

    # Guardar resultados
    output = {
        'periodo': f'{START_DATE} a {END_DATE}',
        'fecha_analisis': datetime.now().isoformat(),
        'resumen': {k: v for k, v in results.items()},
        'por_par': {
            pair: {
                version: {
                    'trades': data['trades'],
                    'wr': data['wr'],
                    'pnl': data['pnl']
                }
                for version, data in pair_data.items()
            }
            for pair, pair_data in results_by_pair.items()
        }
    }

    with open(MODELS_DIR / 'backtest_last_14_days.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n\nGuardado en backtest_last_14_days.json", flush=True)


if __name__ == '__main__':
    main()
