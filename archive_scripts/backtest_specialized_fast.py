"""
Bot Especializado - Version Rapida
==================================
Grid reducido para prueba rapida.
"""

import json
import random
import sys
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

random.seed(42)
np.random.seed(42)


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


def load_existing_model(pair):
    # pair = 'ADA/USDT' -> queremos 'v95_v7_ADAUSDT.pkl'
    safe = pair.replace('/', '').replace('_', '')  # ADAUSDT
    try:
        return joblib.load(MODELS_DIR / f'v95_v7_{safe}.pkl')
    except Exception as e:
        print(f"    Error cargando modelo: {e}", flush=True)
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


def find_best_worst_years(df):
    df = df.copy()
    df['year'] = df.index.year
    yearly_returns = {}
    for year in df['year'].unique():
        year_data = df[df['year'] == year]
        if len(year_data) < 100:
            continue
        ret = (year_data['close'].iloc[-1] - year_data['close'].iloc[0]) / year_data['close'].iloc[0]
        yearly_returns[year] = ret
    if not yearly_returns:
        return None, None, {}
    best_year = max(yearly_returns, key=yearly_returns.get)
    worst_year = min(yearly_returns, key=yearly_returns.get)
    return best_year, worst_year, yearly_returns


def run_backtest(df, model, start_date, end_date, params):
    df_period = df[(df.index >= start_date) & (df.index < end_date)]
    if len(df_period) < 100:
        return {'trades': 0, 'wins': 0, 'wr': 0, 'pnl': 0, 'pf': 0}

    feat = compute_features(df)
    feat = feat[(feat.index >= start_date) & (feat.index < end_date)]
    fcols = [c for c in model.feature_name_ if c in feat.columns]
    if not fcols:
        return {'trades': 0, 'wins': 0, 'wr': 0, 'pnl': 0, 'pf': 0}

    trades = []
    balance = INITIAL_CAPITAL
    pos = None

    min_conv = params['min_conviction']
    rsi_min, rsi_max = params['rsi_range']
    max_chop = params['max_chop']
    tp_mult = params['tp_mult']

    for i in range(100, len(df_period)):
        ts = df_period.index[i]
        if ts not in feat.index:
            continue
        price = df_period.iloc[i]['close']

        if pos is not None:
            pnl_pct = (price - pos['entry']) / pos['entry'] * pos['dir']
            if pnl_pct >= pos['tp_pct'] or pnl_pct <= -pos['sl_pct'] or (i - pos['bar']) >= 20:
                pnl = pos['size'] * pos['dir'] * (price - pos['entry'])
                balance += pnl
                trades.append({'pnl': pnl})
                pos = None

        if pos is None:
            rsi = feat.loc[ts, 'rsi14'] if 'rsi14' in feat.columns else 50
            chop = feat.loc[ts, 'chop'] if 'chop' in feat.columns else 50
            atr = feat.loc[ts, 'atr14'] if 'atr14' in feat.columns else price * 0.02

            X = feat.loc[ts:ts][fcols]
            if X.isna().any().any():
                continue

            pred = model.predict(X)[0]
            sig = 1 if pred > 0.5 else -1
            conviction = abs(pred - 0.5) * 10

            if conviction < min_conv:
                continue
            if not (rsi_min <= rsi <= rsi_max):
                continue
            if chop > max_chop:
                continue

            tp_pct = atr / price * tp_mult
            sl_pct = atr / price * 1.0
            risk_amt = balance * RISK_PER_TRADE
            size = risk_amt / (sl_pct * price) if sl_pct > 0 else 0
            pos = {'entry': price, 'dir': sig, 'size': size,
                   'tp_pct': tp_pct, 'sl_pct': sl_pct, 'bar': i}

    if pos:
        pnl = pos['size'] * pos['dir'] * (df_period.iloc[-1]['close'] - pos['entry'])
        trades.append({'pnl': pnl})

    if not trades:
        return {'trades': 0, 'wins': 0, 'wr': 0, 'pnl': 0, 'pf': 0}

    wins = sum(1 for t in trades if t['pnl'] > 0)
    gp = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gl = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    return {
        'trades': len(trades),
        'wins': wins,
        'wr': wins / len(trades) * 100 if trades else 0,
        'pnl': sum(t['pnl'] for t in trades),
        'pf': gp / gl if gl > 0 else 99,
    }


def optimize_fast(df, model, pair):
    """Optimizacion rapida con menos combinaciones."""
    print(f"    Optimizando {pair}...", flush=True)

    # Grid REDUCIDO
    configs = [
        {'min_conviction': 1.5, 'rsi_range': (30, 70), 'max_chop': 60, 'tp_mult': 2.0},
        {'min_conviction': 1.8, 'rsi_range': (35, 65), 'max_chop': 55, 'tp_mult': 2.0},
        {'min_conviction': 2.0, 'rsi_range': (38, 72), 'max_chop': 52, 'tp_mult': 2.0},
        {'min_conviction': 1.5, 'rsi_range': (30, 70), 'max_chop': 60, 'tp_mult': 2.5},
        {'min_conviction': 1.8, 'rsi_range': (35, 65), 'max_chop': 55, 'tp_mult': 2.5},
        {'min_conviction': 1.2, 'rsi_range': (25, 75), 'max_chop': 65, 'tp_mult': 2.0},
    ]

    best_score = -999
    best_params = None

    for cfg in configs:
        result = run_backtest(df, model, '2020-01-01', '2024-01-01', cfg)
        if result['trades'] >= 20:
            score = result['wr'] * np.sqrt(result['trades']) * (1 if result['pnl'] > 0 else 0.5)
            if score > best_score:
                best_score = score
                best_params = cfg.copy()
                best_params['opt_wr'] = result['wr']
                best_params['opt_trades'] = result['trades']
                best_params['opt_pnl'] = result['pnl']

    if not best_params:
        best_params = configs[2]  # Default
        best_params['opt_wr'] = 0
        best_params['opt_trades'] = 0
        best_params['opt_pnl'] = 0

    print(f"      -> conv={best_params['min_conviction']}, chop={best_params['max_chop']}, "
          f"opt: {best_params['opt_trades']} tr, WR {best_params['opt_wr']:.1f}%", flush=True)

    return best_params


def main():
    print("="*70, flush=True)
    print("BOTS ESPECIALIZADOS - PRUEBAS EXHAUSTIVAS", flush=True)
    print("="*70, flush=True)

    pairs = ['ADA/USDT', 'XRP/USDT', 'DOT/USDT', 'DOGE/USDT',
             'ETH/USDT', 'AVAX/USDT', 'NEAR/USDT', 'LINK/USDT']

    all_results = []

    for pair in pairs:
        print(f"\n{'='*70}", flush=True)
        print(f"PAR: {pair}", flush=True)
        print("="*70, flush=True)

        df = load_data_4h(pair)
        model = load_existing_model(pair)

        if df is None or model is None:
            print(f"  [!] Sin datos o modelo", flush=True)
            continue

        print(f"  Datos: {df.index.min().date()} a {df.index.max().date()}", flush=True)

        best_year, worst_year, yearly_rets = find_best_worst_years(df)
        print(f"  Mejor ano: {best_year} ({yearly_rets.get(best_year, 0)*100:+.0f}%)", flush=True)
        print(f"  Peor ano: {worst_year} ({yearly_rets.get(worst_year, 0)*100:+.0f}%)", flush=True)

        params = optimize_fast(df, model, pair)

        # Test periodos
        periods = {
            'PEOR_ANO': (f'{worst_year}-01-01', f'{worst_year+1}-01-01') if worst_year else None,
            'MEJOR_ANO': (f'{best_year}-01-01', f'{best_year+1}-01-01') if best_year else None,
            'ULTIMO_ANO': ('2024-02-01', '2025-02-24'),
            '2025': ('2025-01-01', '2026-02-24'),
        }

        print(f"\n  {'Periodo':<15} {'Trades':<8} {'WR':<10} {'PnL':<12} {'PF':<8}", flush=True)
        print("  " + "-"*55, flush=True)

        results = {}
        for name, period in periods.items():
            if period is None:
                continue
            m = run_backtest(df, model, period[0], period[1], params)
            results[name] = m
            wr_mark = "**" if m['wr'] >= 50 else ""
            print(f"  {name:<15} {m['trades']:<8} {m['wr']:<9.1f}%{wr_mark} "
                  f"${m['pnl']:<10,.0f} {m['pf']:<.2f}", flush=True)

        # Crear ano sintetico
        random.seed(42)
        months_pool = list(pd.date_range('2020-01-01', '2025-12-01', freq='MS'))
        if len(months_pool) >= 12:
            selected = random.sample(months_pool, 12)
            synth_trades = 0
            synth_wins = 0
            synth_pnl = 0
            for m_start in selected:
                m_end = m_start + pd.DateOffset(months=1)
                r = run_backtest(df, model, m_start.strftime('%Y-%m-%d'), m_end.strftime('%Y-%m-%d'), params)
                synth_trades += r['trades']
                synth_wins += r['wins']
                synth_pnl += r['pnl']

            synth_wr = synth_wins / synth_trades * 100 if synth_trades > 0 else 0
            results['SINTETICO'] = {'trades': synth_trades, 'wins': synth_wins, 'wr': synth_wr, 'pnl': synth_pnl}
            wr_mark = "**" if synth_wr >= 50 else ""
            print(f"  {'SINTETICO':<15} {synth_trades:<8} {synth_wr:<9.1f}%{wr_mark} "
                  f"${synth_pnl:<10,.0f} {'N/A':<8}", flush=True)

        # Resumen
        valid = [r for r in results.values() if r['trades'] > 0]
        if valid:
            avg_wr = np.mean([r['wr'] for r in valid])
            total_pnl = sum(r['pnl'] for r in valid)
            total_trades = sum(r['trades'] for r in valid)
        else:
            avg_wr = 0
            total_pnl = 0
            total_trades = 0

        status = "VIABLE" if avg_wr >= 50 and total_pnl > 0 else "REVISAR"

        print(f"\n  RESUMEN: {total_trades} trades, WR {avg_wr:.1f}%, PnL ${total_pnl:,.0f} -> {status}", flush=True)

        all_results.append({
            'pair': pair,
            'params': params,
            'avg_wr': avg_wr,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'status': status,
            'results': results,
        })

    # Ranking
    print("\n" + "="*70, flush=True)
    print("RANKING FINAL", flush=True)
    print("="*70, flush=True)

    all_results.sort(key=lambda x: (x['status'] == 'VIABLE', x['avg_wr']), reverse=True)

    print(f"\n{'#':<3} {'Par':<12} {'Trades':<8} {'WR':<10} {'PnL':<12} {'Status'}", flush=True)
    print("-"*55, flush=True)
    for i, r in enumerate(all_results, 1):
        wr_mark = "**" if r['avg_wr'] >= 50 else ""
        print(f"{i:<3} {r['pair']:<12} {r['total_trades']:<8} {r['avg_wr']:<9.1f}%{wr_mark} "
              f"${r['total_pnl']:<10,.0f} {r['status']}", flush=True)

    # Guardar
    with open(MODELS_DIR / 'specialized_fast_results.json', 'w') as f:
        json.dump({'fecha': datetime.now().isoformat(), 'resultados': all_results}, f, indent=2, default=str)

    print(f"\nGuardado en specialized_fast_results.json", flush=True)

    # Viables
    viable = [r for r in all_results if r['status'] == 'VIABLE']
    if viable:
        print(f"\n*** PARES VIABLES: {', '.join(r['pair'] for r in viable)} ***", flush=True)


if __name__ == '__main__':
    main()
