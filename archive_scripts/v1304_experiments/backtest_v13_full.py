"""
Backtest V13.02 COMPLETO - Todos los pares
==========================================
- BTC/USDT: V2 modelo (TP=4%/SL=2%)
- BNB/USDT: V2 modelo (TP=7%/SL=3.5%, solo SHORT)
- Otros: V7 modelo generico (TP=3%/SL=1.5%)
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

INITIAL_CAPITAL = 100.0
POSITION_SIZE = 10.0

# =============================================================================
# CONFIGURACION V13.02 COMPLETA
# =============================================================================
# V2 models (optimizados)
V2_PAIRS = {
    'BTC/USDT': {
        'model_file': 'btc_v2_gradientboosting.pkl',
        'tp_pct': 0.04,
        'sl_pct': 0.02,
        'conv_min': 1.0,
        'only_short': False,
        'model_type': 'v2',
    },
    'BNB/USDT': {
        'model_file': 'bnb_usdt_v2_gradientboosting.pkl',
        'tp_pct': 0.07,
        'sl_pct': 0.035,
        'conv_min': 1.0,
        'only_short': True,
        'model_type': 'v2',
    },
}

# V7 models (genericos)
V7_PAIRS = ['XRP/USDT', 'NEAR/USDT', 'DOT/USDT', 'ETH/USDT', 'DOGE/USDT', 'AVAX/USDT', 'LINK/USDT', 'ADA/USDT']

V7_CONFIG = {
    'tp_pct': 0.03,
    'sl_pct': 0.015,
    'conv_min': 0.5,
    'only_short': False,
    'model_type': 'v7',
}

# Cargar metadata V7
with open(MODELS_DIR / 'v7_meta.json') as f:
    V7_META = json.load(f)


def compute_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """54 features para modelos V2."""
    feat = pd.DataFrame(index=df.index)
    c, h, l, o, v = df['close'], df['high'], df['low'], df['open'], df['volume']

    for p in [1, 2, 3, 5, 10, 20, 50]:
        feat[f'ret_{p}'] = c.pct_change(p)

    feat['atr14'] = ta.atr(h, l, c, length=14)
    feat['atr_r'] = feat['atr14'] / feat['atr14'].rolling(50).mean()
    feat['vol5'] = c.pct_change().rolling(5).std()
    feat['vol20'] = c.pct_change().rolling(20).std()
    feat['vol_ratio'] = feat['vol5'] / (feat['vol20'] + 1e-10)
    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)
    feat['rsi21'] = ta.rsi(c, length=21)

    sr = ta.stochrsi(c, length=14, rsi_length=14, k=3, d=3)
    if sr is not None and len(sr.columns) >= 2:
        feat['srsi_k'] = sr.iloc[:, 0]
        feat['srsi_d'] = sr.iloc[:, 1]

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None and len(macd.columns) >= 3:
        feat['macd'] = macd.iloc[:, 0]
        feat['macd_h'] = macd.iloc[:, 1]
        feat['macd_s'] = macd.iloc[:, 2]

    feat['roc5'] = ta.roc(c, length=5)
    feat['roc10'] = ta.roc(c, length=10)
    feat['roc20'] = ta.roc(c, length=20)

    for el in [8, 21, 55, 100, 200]:
        e = ta.ema(c, length=el)
        feat[f'ema{el}_d'] = (c - e) / e * 100

    feat['ema8_sl'] = ta.ema(c, length=8).pct_change(3) * 100
    feat['ema21_sl'] = ta.ema(c, length=21).pct_change(5) * 100
    feat['ema55_sl'] = ta.ema(c, length=55).pct_change(5) * 100

    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None and len(bb.columns) >= 3:
        bw = bb.iloc[:, 2] - bb.iloc[:, 0]
        feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw
        feat['bb_w'] = bw / bb.iloc[:, 1] * 100

    feat['vr'] = v / v.rolling(20).mean()
    feat['vr5'] = v.rolling(5).mean() / v.rolling(20).mean()
    feat['spr'] = (h - l) / c * 100
    feat['body'] = abs(c - o) / (h - l + 1e-10)
    feat['upper_wick'] = (h - np.maximum(c, o)) / (h - l + 1e-10)
    feat['lower_wick'] = (np.minimum(c, o) - l) / (h - l + 1e-10)

    ax = ta.adx(h, l, c, length=14)
    if ax is not None and len(ax.columns) >= 3:
        feat['adx'] = ax.iloc[:, 0]
        feat['dip'] = ax.iloc[:, 1]
        feat['dim'] = ax.iloc[:, 2]
        feat['di_diff'] = feat['dip'] - feat['dim']

    chop = ta.chop(h, l, c, length=14)
    if chop is not None:
        feat['chop'] = chop

    hr = df.index.hour
    dw = df.index.dayofweek
    feat['h_s'] = np.sin(2 * np.pi * hr / 24)
    feat['h_c'] = np.cos(2 * np.pi * hr / 24)
    feat['d_s'] = np.sin(2 * np.pi * dw / 7)
    feat['d_c'] = np.cos(2 * np.pi * dw / 7)

    feat['ret1_lag1'] = feat['ret_1'].shift(1)
    feat['rsi14_lag1'] = feat['rsi14'].shift(1)
    feat['ret1_lag2'] = feat['ret_1'].shift(2)
    feat['rsi14_lag2'] = feat['rsi14'].shift(2)
    feat['ret1_lag3'] = feat['ret_1'].shift(3)
    feat['rsi14_lag3'] = feat['rsi14'].shift(3)

    return feat


def compute_features_v7(df: pd.DataFrame) -> pd.DataFrame:
    """34 features para modelos V7."""
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


def detect_regime(df: pd.DataFrame) -> pd.Series:
    """Detecta regimen de mercado."""
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


def load_data(pair: str) -> pd.DataFrame:
    """Carga datos de un par."""
    safe = pair.replace('/', '_')
    path = DATA_DIR / f'{safe}_4h_full.parquet'
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def load_model_v2(pair: str) -> Dict:
    """Carga modelo V2."""
    config = V2_PAIRS[pair]
    path = MODELS_DIR / config['model_file']
    return joblib.load(path)


def load_model_v7(pair: str):
    """Carga modelo V7."""
    safe = pair.replace('/', '_')
    path = MODELS_DIR / f'v7_{safe}.pkl'
    if not path.exists():
        return None
    return joblib.load(path)


def backtest_pair_v2(pair: str, start: str, end: str) -> List[Dict]:
    """Backtest par con modelo V2."""
    config = V2_PAIRS[pair]
    df = load_data(pair)
    if df is None:
        return []

    model_data = load_model_v2(pair)
    feat = compute_features_v2(df)

    mask = (feat.index >= start) & (feat.index <= end)
    df_p = df[mask].copy()
    feat_p = feat[mask].copy()

    if len(df_p) < 50:
        return []

    feature_cols = model_data['feature_cols']
    avail = [c for c in feature_cols if c in feat_p.columns]
    if len(avail) < len(feature_cols) * 0.8:
        return []

    valid = feat_p[avail].notna().all(axis=1)
    feat_v = feat_p.loc[valid, avail]
    df_v = df_p[valid].copy()

    if len(feat_v) < 20:
        return []

    if model_data['scaler'] is not None:
        X = model_data['scaler'].transform(feat_v)
    else:
        X = feat_v.values

    preds = model_data['model'].predict(X)
    conv = np.abs(preds) / model_data['pred_std']
    signals = conv >= config['conv_min']
    directions = np.where(preds < 0, -1, 1)
    regime = detect_regime(df_v)

    return simulate_trades(pair, df_v, feat_v, signals, directions, regime, config)


def backtest_pair_v7(pair: str, start: str, end: str) -> List[Dict]:
    """Backtest par con modelo V7."""
    df = load_data(pair)
    if df is None:
        return []

    model = load_model_v7(pair)
    if model is None:
        return []

    feat = compute_features_v7(df)
    fcols = V7_META['feature_cols']
    pred_std = V7_META['pred_stds'].get(pair, 0.01)

    mask = (feat.index >= start) & (feat.index <= end)
    df_p = df[mask].copy()
    feat_p = feat[mask].copy()

    if len(df_p) < 50:
        return []

    avail = [c for c in fcols if c in feat_p.columns]
    if len(avail) < len(fcols) * 0.8:
        return []

    valid = feat_p[avail].notna().all(axis=1)
    feat_v = feat_p.loc[valid, avail]
    df_v = df_p[valid].copy()

    if len(feat_v) < 20:
        return []

    preds = model.predict(feat_v.values)
    conv = np.abs(preds) / pred_std
    signals = conv >= V7_CONFIG['conv_min']
    directions = np.where(preds < 0, -1, 1)
    regime = detect_regime(df_v)

    return simulate_trades(pair, df_v, feat_v, signals, directions, regime, V7_CONFIG)


def simulate_trades(pair, df_v, feat_v, signals, directions, regime, config) -> List[Dict]:
    """Simula trades con TP/SL."""
    trades = []
    tp_pct = config['tp_pct']
    sl_pct = config['sl_pct']
    only_short = config.get('only_short', False)

    for i, (idx, row) in enumerate(feat_v.iterrows()):
        if i >= len(feat_v) - 5 or not signals[i]:
            continue

        d = directions[i]
        reg = regime.iloc[i]

        if only_short and d == 1:
            continue
        if reg == 'BULL' and d == -1:
            continue
        if reg == 'BEAR' and d == 1:
            continue

        entry = df_v.loc[idx, 'close']
        tp = entry * (1 + tp_pct) if d == 1 else entry * (1 - tp_pct)
        sl = entry * (1 - sl_pct) if d == 1 else entry * (1 + sl_pct)

        exit_p = None
        exit_reason = 'timeout'
        fidx = feat_v.index.get_loc(idx)

        for j in range(1, min(21, len(df_v) - fidx)):
            bar = df_v.iloc[fidx + j]
            if d == 1:
                if bar['low'] <= sl:
                    exit_p = sl
                    exit_reason = 'sl'
                    break
                elif bar['high'] >= tp:
                    exit_p = tp
                    exit_reason = 'tp'
                    break
            else:
                if bar['high'] >= sl:
                    exit_p = sl
                    exit_reason = 'sl'
                    break
                elif bar['low'] <= tp:
                    exit_p = tp
                    exit_reason = 'tp'
                    break

        if exit_p is None:
            exit_p = df_v.iloc[min(fidx + 20, len(df_v) - 1)]['close']

        pnl_pct = (exit_p - entry) / entry if d == 1 else (entry - exit_p) / entry
        trades.append({
            'pair': pair,
            'entry_time': idx,
            'direction': 'LONG' if d == 1 else 'SHORT',
            'regime': reg,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_pct * POSITION_SIZE,
            'result': 'win' if pnl_pct > 0 else 'loss',
        })

    return trades


def backtest_pair(pair: str, start: str, end: str) -> List[Dict]:
    """Backtest un par (auto-detecta V2 o V7)."""
    if pair in V2_PAIRS:
        return backtest_pair_v2(pair, start, end)
    else:
        return backtest_pair_v7(pair, start, end)


def calculate_metrics(trades: List[Dict]) -> Dict:
    """Calcula metricas."""
    if not trades:
        return {'trades': 0, 'win_rate': 0, 'pnl': 0, 'max_dd': 0, 'profit_factor': 0}

    df = pd.DataFrame(trades)
    wins = df[df['result'] == 'win']
    losses = df[df['result'] == 'loss']

    pnl = df['pnl_usd'].sum()
    wr = len(wins) / len(df) * 100

    gp = wins['pnl_usd'].sum() if len(wins) > 0 else 0
    gl = abs(losses['pnl_usd'].sum()) if len(losses) > 0 else 0.01
    pf = gp / gl if gl > 0 else 999

    cumsum = df['pnl_usd'].cumsum()
    peak = cumsum.expanding().max()
    dd = peak - cumsum
    max_dd = dd.max()

    return {
        'trades': len(df),
        'win_rate': wr,
        'pnl': pnl,
        'max_dd': max_dd,
        'profit_factor': min(pf, 999),
    }


def run_backtest(name: str, start: str, end: str, all_pairs: List[str]) -> Dict:
    """Corre backtest para un periodo."""
    print(f"\n{'='*70}")
    print(f"PERIODO: {name}")
    print(f"Rango: {start} a {end}")
    print('='*70)

    all_trades = []
    pair_results = {}

    for pair in all_pairs:
        try:
            trades = backtest_pair(pair, start, end)
            all_trades.extend(trades)
            metrics = calculate_metrics(trades)
            pair_results[pair] = metrics

            model_type = "V2" if pair in V2_PAIRS else "V7"
            print(f"\n{pair} [{model_type}]:")
            print(f"  Trades: {metrics['trades']}")
            print(f"  Win Rate: {metrics['win_rate']:.1f}%")
            print(f"  PnL: ${metrics['pnl']:.2f}")

        except Exception as e:
            print(f"\n{pair}: ERROR - {e}")
            pair_results[pair] = {'trades': 0, 'win_rate': 0, 'pnl': 0, 'max_dd': 0, 'profit_factor': 0}

    total = calculate_metrics(all_trades)

    print(f"\n{'-'*50}")
    print(f"TOTAL V13.02:")
    print(f"  Trades: {total['trades']}")
    print(f"  Win Rate: {total['win_rate']:.1f}%")
    print(f"  PnL: ${total['pnl']:.2f}")
    print(f"  Max DD: ${total['max_dd']:.2f} ({total['max_dd']/INITIAL_CAPITAL*100:.1f}%)")
    pf = total['profit_factor']
    print(f"  PF: {pf:.2f}" if pf < 100 else "  PF: INF")

    return {
        'period': name,
        'start': start,
        'end': end,
        'total': total,
        'by_pair': pair_results,
    }


def find_best_worst_year(pairs: List[str]) -> Tuple[int, int]:
    """Encuentra mejor y peor ano."""
    print("\nAnalizando rendimiento por ano...")

    year_pnl = {}

    for year in range(2020, 2026):
        try:
            all_trades = []
            for pair in pairs:
                trades = backtest_pair(pair, f"{year}-01-01", f"{year}-12-31")
                all_trades.extend(trades)
            if all_trades:
                m = calculate_metrics(all_trades)
                year_pnl[year] = m['pnl']
                print(f"  {year}: {m['trades']} trades, ${m['pnl']:.2f}")
        except:
            pass

    if year_pnl:
        best = max(year_pnl, key=year_pnl.get)
        worst = min(year_pnl, key=year_pnl.get)
        return best, worst

    return 2021, 2022


def main():
    print("="*70)
    print("BACKTEST V13.02 COMPLETO - TODOS LOS PARES")
    print("="*70)
    print("\nPares habilitados:")
    print("  V2 (optimizados): BTC/USDT, BNB/USDT")
    print("  V7 (genericos): " + ", ".join(V7_PAIRS))

    ALL_PAIRS = list(V2_PAIRS.keys()) + V7_PAIRS
    results = []

    # 1. Ultimo ano
    print("\n\n" + "#"*70)
    print("1. ULTIMO ANO (Feb 2025 - Feb 2026)")
    print("#"*70)
    r = run_backtest("Ultimo Ano", "2025-02-01", "2026-02-27", ALL_PAIRS)
    results.append(r)

    # Mejor/peor ano
    best_year, worst_year = find_best_worst_year(ALL_PAIRS)
    print(f"\nMejor: {best_year}, Peor: {worst_year}")

    # 2. Mejor ano
    print("\n\n" + "#"*70)
    print(f"2. MEJOR ANO ({best_year})")
    print("#"*70)
    r = run_backtest(f"Mejor ({best_year})", f"{best_year}-01-01", f"{best_year}-12-31", ALL_PAIRS)
    results.append(r)

    # 3. Peor ano
    print("\n\n" + "#"*70)
    print(f"3. PEOR ANO ({worst_year})")
    print("#"*70)
    r = run_backtest(f"Peor ({worst_year})", f"{worst_year}-01-01", f"{worst_year}-12-31", ALL_PAIRS)
    results.append(r)

    # 4. Sintetico (H1 2022 crash + H2 2023 recovery)
    print("\n\n" + "#"*70)
    print("4. ANO SINTETICO (Crash + Recovery)")
    print("#"*70)

    all_synth = []
    for pair in ALL_PAIRS:
        trades_crash = backtest_pair(pair, "2022-01-01", "2022-06-30")
        trades_recov = backtest_pair(pair, "2023-07-01", "2023-12-31")
        all_synth.extend(trades_crash + trades_recov)

    synth_m = calculate_metrics(all_synth)
    synth_by_pair = {}
    for pair in ALL_PAIRS:
        pair_trades = [t for t in all_synth if t['pair'] == pair]
        synth_by_pair[pair] = calculate_metrics(pair_trades)

    print(f"\nSintetico (Crash H1-2022 + Recovery H2-2023):")
    print(f"  Trades: {synth_m['trades']}")
    print(f"  Win Rate: {synth_m['win_rate']:.1f}%")
    print(f"  PnL: ${synth_m['pnl']:.2f}")
    print(f"  Max DD: ${synth_m['max_dd']:.2f}")

    results.append({
        'period': 'Sintetico',
        'total': synth_m,
        'by_pair': synth_by_pair
    })

    # ==========================================================================
    # RESUMEN FINAL
    # ==========================================================================
    print("\n\n" + "#"*70)
    print("RESUMEN FINAL V13.02 (Capital $100)")
    print("#"*70)

    print("\n" + "="*85)
    print(f"| {'Periodo':<22} | {'Trades':>6} | {'WR':>7} | {'PnL':>10} | {'MaxDD%':>8} | {'PF':>6} |")
    print("="*85)

    for r in results:
        t = r['total']
        pf_str = f"{t['profit_factor']:.2f}" if t['profit_factor'] < 100 else "INF"
        dd_pct = t['max_dd'] / INITIAL_CAPITAL * 100
        print(f"| {r['period']:<22} | {t['trades']:>6} | {t['win_rate']:>5.1f}% | ${t['pnl']:>8.2f} | {dd_pct:>6.1f}% | {pf_str:>6} |")

    print("="*85)

    # Por moneda
    print("\n\nDETALLE POR MONEDA:")
    print("="*90)
    print(f"| {'Par':<12} | {'Tipo':<4} | {'Ultimo Ano':>15} | {'Mejor Ano':>15} | {'Peor Ano':>15} | {'Sintetico':>15} |")
    print("="*90)

    for pair in ALL_PAIRS:
        mtype = "V2" if pair in V2_PAIRS else "V7"
        cols = []
        for r in results:
            m = r['by_pair'].get(pair, {})
            if m.get('trades', 0) > 0:
                cols.append(f"{m['trades']:>3}t {m['win_rate']:>4.0f}% ${m['pnl']:>5.0f}")
            else:
                cols.append(f"{'N/A':>15}")
        print(f"| {pair:<12} | {mtype:<4} | {cols[0]:>15} | {cols[1]:>15} | {cols[2]:>15} | {cols[3]:>15} |")

    print("="*90)

    # Guardar CSV
    summary = []
    for r in results:
        row = {
            'periodo': r['period'],
            'total_trades': r['total']['trades'],
            'total_wr': r['total']['win_rate'],
            'total_pnl': r['total']['pnl'],
            'total_maxdd': r['total']['max_dd'],
            'total_maxdd_pct': r['total']['max_dd'] / INITIAL_CAPITAL * 100,
            'total_pf': r['total']['profit_factor'],
        }
        for pair, m in r['by_pair'].items():
            pk = pair.replace('/', '_')
            row[f'{pk}_trades'] = m.get('trades', 0)
            row[f'{pk}_wr'] = m.get('win_rate', 0)
            row[f'{pk}_pnl'] = m.get('pnl', 0)
        summary.append(row)

    df_out = pd.DataFrame(summary)
    df_out.to_csv(DATA_DIR / 'backtest_v13_full.csv', index=False)
    print(f"\n\nGuardado: data/backtest_v13_full.csv")


if __name__ == '__main__':
    main()
