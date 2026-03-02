"""
Backtest V13.03 - Todos los pares optimizados
==============================================
Todos los pares con modelos V2 GradientBoosting optimizados
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

INITIAL_CAPITAL = 100.0
POSITION_SIZE = 10.0

# =============================================================================
# V13.03 CONFIGURACION - TODOS OPTIMIZADOS
# =============================================================================
PAIRS_CONFIG = {
    # V2 Originales (BTC + BNB)
    'BTC/USDT': {
        'model_file': 'btc_v2_gradientboosting.pkl',
        'tp_pct': 0.04, 'sl_pct': 0.02, 'conv_min': 1.0,
        'only_long': False, 'only_short': False,
    },
    'BNB/USDT': {
        'model_file': 'bnb_usdt_v2_gradientboosting.pkl',
        'tp_pct': 0.07, 'sl_pct': 0.035, 'conv_min': 1.0,
        'only_long': False, 'only_short': True,
    },
    # V2 Nuevos optimizados
    'XRP/USDT': {
        'model_file': 'xrp_usdt_v2_gradientboosting.pkl',
        'tp_pct': 0.08, 'sl_pct': 0.04, 'conv_min': 0.5,
        'only_long': False, 'only_short': False,
    },
    'ETH/USDT': {
        'model_file': 'eth_usdt_v2_gradientboosting.pkl',
        'tp_pct': 0.08, 'sl_pct': 0.04, 'conv_min': 0.5,
        'only_long': False, 'only_short': False,
    },
    'AVAX/USDT': {
        'model_file': 'avax_usdt_v2_gradientboosting.pkl',
        'tp_pct': 0.07, 'sl_pct': 0.02, 'conv_min': 0.5,
        'only_long': False, 'only_short': False,
    },
    'ADA/USDT': {
        'model_file': 'ada_usdt_v2_gradientboosting.pkl',
        'tp_pct': 0.05, 'sl_pct': 0.04, 'conv_min': 0.5,
        'only_long': False, 'only_short': False,
    },
    'LINK/USDT': {
        'model_file': 'link_usdt_v2_gradientboosting.pkl',
        'tp_pct': 0.07, 'sl_pct': 0.04, 'conv_min': 0.5,
        'only_long': False, 'only_short': False,
    },
    'DOGE/USDT': {
        'model_file': 'doge_usdt_v2_gradientboosting.pkl',
        'tp_pct': 0.05, 'sl_pct': 0.025, 'conv_min': 0.5,
        'only_long': False, 'only_short': False,
    },
    'NEAR/USDT': {
        'model_file': 'near_usdt_v2_gradientboosting.pkl',
        'tp_pct': 0.05, 'sl_pct': 0.015, 'conv_min': 0.5,
        'only_long': False, 'only_short': True,  # SOLO SHORT
    },
    'DOT/USDT': {
        'model_file': 'dot_usdt_v2_gradientboosting.pkl',
        'tp_pct': 0.06, 'sl_pct': 0.025, 'conv_min': 0.5,
        'only_long': False, 'only_short': True,  # SOLO SHORT
    },
}


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


def detect_regime(df: pd.DataFrame) -> pd.Series:
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


def backtest_pair(pair: str, start: str, end: str) -> List[Dict]:
    """Backtest un par con modelo V2."""
    config = PAIRS_CONFIG[pair]
    safe = pair.replace('/', '_')

    # Cargar datos
    data_path = DATA_DIR / f'{safe}_4h_full.parquet'
    if not data_path.exists():
        return []

    df = pd.read_parquet(data_path)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Cargar modelo
    model_path = MODELS_DIR / config['model_file']
    if not model_path.exists():
        return []

    model_data = joblib.load(model_path)

    # Features
    feat = compute_features_v2(df)

    # Filtrar periodo
    mask = (feat.index >= start) & (feat.index <= end)
    df_p = df[mask].copy()
    feat_p = feat[mask].copy()

    if len(df_p) < 50:
        return []

    # Alinear features
    feature_cols = model_data['feature_cols']
    avail = [c for c in feature_cols if c in feat_p.columns]
    if len(avail) < len(feature_cols) * 0.8:
        return []

    valid = feat_p[avail].notna().all(axis=1)
    feat_v = feat_p.loc[valid, avail]
    df_v = df_p[valid].copy()

    if len(feat_v) < 20:
        return []

    # Predicciones
    if model_data['scaler'] is not None:
        X = model_data['scaler'].transform(feat_v)
    else:
        X = feat_v.values

    preds = model_data['model'].predict(X)
    conv = np.abs(preds) / model_data['pred_std']
    signals = conv >= config['conv_min']
    directions = np.where(preds < 0, -1, 1)
    regime = detect_regime(df_v)

    # Simular trades
    trades = []
    tp_pct = config['tp_pct']
    sl_pct = config['sl_pct']

    for i, (idx, row) in enumerate(feat_v.iterrows()):
        if i >= len(feat_v) - 5 or not signals[i]:
            continue

        d = directions[i]
        reg = regime.iloc[i]

        # Filtros direccion
        if config['only_short'] and d == 1:
            continue
        if config['only_long'] and d == -1:
            continue

        # Filtro regimen
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


def calculate_metrics(trades: List[Dict]) -> Dict:
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


def run_backtest(name: str, start: str, end: str) -> Dict:
    print(f"\n{'='*70}")
    print(f"PERIODO: {name}")
    print(f"Rango: {start} a {end}")
    print('='*70)

    all_trades = []
    pair_results = {}

    for pair in PAIRS_CONFIG.keys():
        try:
            trades = backtest_pair(pair, start, end)
            all_trades.extend(trades)
            metrics = calculate_metrics(trades)
            pair_results[pair] = metrics

            dir_flag = ""
            if PAIRS_CONFIG[pair]['only_short']:
                dir_flag = " [SHORT]"
            elif PAIRS_CONFIG[pair]['only_long']:
                dir_flag = " [LONG]"

            print(f"\n{pair}{dir_flag}:")
            print(f"  Trades: {metrics['trades']}")
            print(f"  Win Rate: {metrics['win_rate']:.1f}%")
            print(f"  PnL: ${metrics['pnl']:.2f}")

        except Exception as e:
            print(f"\n{pair}: ERROR - {e}")
            pair_results[pair] = {'trades': 0, 'win_rate': 0, 'pnl': 0, 'max_dd': 0, 'profit_factor': 0}

    total = calculate_metrics(all_trades)

    print(f"\n{'-'*50}")
    print(f"TOTAL V13.03:")
    print(f"  Trades: {total['trades']}")
    print(f"  Win Rate: {total['win_rate']:.1f}%")
    print(f"  PnL: ${total['pnl']:.2f}")
    print(f"  Max DD: ${total['max_dd']:.2f} ({total['max_dd']/INITIAL_CAPITAL*100:.1f}%)")
    pf = total['profit_factor']
    print(f"  PF: {pf:.2f}" if pf < 100 else "  PF: INF")

    return {
        'period': name,
        'total': total,
        'by_pair': pair_results,
    }


def main():
    print("="*70)
    print("BACKTEST V13.03 - TODOS LOS PARES OPTIMIZADOS")
    print("="*70)
    print("\n10 pares con modelos V2 GradientBoosting")
    print("SHORT ONLY: BNB, NEAR, DOT")

    results = []

    # 1. Ultimo ano
    r = run_backtest("Ultimo Ano", "2025-02-01", "2026-02-27")
    results.append(r)

    # 2. Mejor ano (2022)
    r = run_backtest("Mejor (2022)", "2022-01-01", "2022-12-31")
    results.append(r)

    # 3. Peor ano (2025)
    r = run_backtest("Peor (2025)", "2025-01-01", "2025-12-31")
    results.append(r)

    # 4. Sintetico
    all_synth = []
    for pair in PAIRS_CONFIG.keys():
        all_synth.extend(backtest_pair(pair, "2022-01-01", "2022-06-30"))
        all_synth.extend(backtest_pair(pair, "2023-07-01", "2023-12-31"))

    synth_m = calculate_metrics(all_synth)
    synth_by_pair = {}
    for pair in PAIRS_CONFIG.keys():
        pair_trades = [t for t in all_synth if t['pair'] == pair]
        synth_by_pair[pair] = calculate_metrics(pair_trades)

    print(f"\n{'='*70}")
    print("PERIODO: Sintetico (Crash + Recovery)")
    print('='*70)
    print(f"\nTOTAL V13.03:")
    print(f"  Trades: {synth_m['trades']}")
    print(f"  Win Rate: {synth_m['win_rate']:.1f}%")
    print(f"  PnL: ${synth_m['pnl']:.2f}")
    print(f"  Max DD: ${synth_m['max_dd']:.2f}")

    results.append({'period': 'Sintetico', 'total': synth_m, 'by_pair': synth_by_pair})

    # RESUMEN FINAL
    print("\n\n" + "#"*70)
    print("RESUMEN FINAL V13.03 (Capital $100)")
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

    # Comparacion V13.02 vs V13.03
    print("\n\nCOMPARACION V13.02 vs V13.03 (Ultimo Ano):")
    print("="*60)
    print(f"| {'Metrica':<20} | {'V13.02':>15} | {'V13.03':>15} |")
    print("-"*60)

    v1302 = {'trades': 8142, 'wr': 35.4, 'pnl': 79.96, 'maxdd': 16.1}
    v1303 = results[0]['total']

    print(f"| {'Trades':<20} | {v1302['trades']:>15} | {v1303['trades']:>15} |")
    print(f"| {'Win Rate':<20} | {v1302['wr']:>14.1f}% | {v1303['win_rate']:>14.1f}% |")
    print(f"| {'PnL':<20} | ${v1302['pnl']:>13.2f} | ${v1303['pnl']:>13.2f} |")
    print(f"| {'Max DD':<20} | {v1302['maxdd']:>14.1f}% | {v1303['max_dd']/INITIAL_CAPITAL*100:>14.1f}% |")
    print("="*60)

    # Guardar CSV
    summary = []
    for r in results:
        row = {
            'periodo': r['period'],
            'total_trades': r['total']['trades'],
            'total_wr': r['total']['win_rate'],
            'total_pnl': r['total']['pnl'],
            'total_maxdd': r['total']['max_dd'],
            'total_pf': r['total']['profit_factor'],
        }
        for pair, m in r['by_pair'].items():
            pk = pair.replace('/', '_')
            row[f'{pk}_trades'] = m.get('trades', 0)
            row[f'{pk}_wr'] = m.get('win_rate', 0)
            row[f'{pk}_pnl'] = m.get('pnl', 0)
        summary.append(row)

    df_out = pd.DataFrame(summary)
    df_out.to_csv(DATA_DIR / 'backtest_v13_03.csv', index=False)
    print(f"\n\nGuardado: data/backtest_v13_03.csv")


if __name__ == '__main__':
    main()
