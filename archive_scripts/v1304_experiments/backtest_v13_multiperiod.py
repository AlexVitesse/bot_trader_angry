"""
Backtest V13.02 Multi-Periodo
==============================
Evalua BTC + BNB en diferentes escenarios temporales.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

# =============================================================================
# CONFIGURACION V13.02
# =============================================================================
PAIRS_CONFIG = {
    'BTC/USDT': {
        'model_file': 'btc_v2_gradientboosting.pkl',
        'tp_pct': 0.04,
        'sl_pct': 0.02,
        'conv_min': 1.0,
        'only_short': False,
        'data_file': 'BTC_USDT_4h_full.parquet',
    },
    'BNB/USDT': {
        'model_file': 'bnb_usdt_v2_gradientboosting.pkl',
        'tp_pct': 0.07,
        'sl_pct': 0.035,
        'conv_min': 1.0,
        'only_short': True,
        'data_file': 'BNB_USDT_4h_full.parquet',
    },
}

INITIAL_CAPITAL = 100.0
POSITION_SIZE = 10.0  # $10 por trade


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """54 features para modelos V2 (usando pandas_ta)."""
    feat = pd.DataFrame(index=df.index)
    c, h, l, o, v = df['close'], df['high'], df['low'], df['open'], df['volume']

    # Returns
    for p in [1, 2, 3, 5, 10, 20, 50]:
        feat[f'ret_{p}'] = c.pct_change(p)

    # ATR
    feat['atr14'] = ta.atr(h, l, c, length=14)
    feat['atr_r'] = feat['atr14'] / feat['atr14'].rolling(50).mean()

    # Volatility
    feat['vol5'] = c.pct_change().rolling(5).std()
    feat['vol20'] = c.pct_change().rolling(20).std()
    feat['vol_ratio'] = feat['vol5'] / (feat['vol20'] + 1e-10)

    # RSI
    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)
    feat['rsi21'] = ta.rsi(c, length=21)

    # Stoch RSI
    sr = ta.stochrsi(c, length=14, rsi_length=14, k=3, d=3)
    if sr is not None and len(sr.columns) >= 2:
        feat['srsi_k'] = sr.iloc[:, 0]
        feat['srsi_d'] = sr.iloc[:, 1]

    # MACD
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None and len(macd.columns) >= 3:
        feat['macd'] = macd.iloc[:, 0]
        feat['macd_h'] = macd.iloc[:, 1]
        feat['macd_s'] = macd.iloc[:, 2]

    # ROC
    feat['roc5'] = ta.roc(c, length=5)
    feat['roc10'] = ta.roc(c, length=10)
    feat['roc20'] = ta.roc(c, length=20)

    # EMAs
    for el in [8, 21, 55, 100, 200]:
        e = ta.ema(c, length=el)
        feat[f'ema{el}_d'] = (c - e) / e * 100

    feat['ema8_sl'] = ta.ema(c, length=8).pct_change(3) * 100
    feat['ema21_sl'] = ta.ema(c, length=21).pct_change(5) * 100
    feat['ema55_sl'] = ta.ema(c, length=55).pct_change(5) * 100

    # Bollinger
    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None and len(bb.columns) >= 3:
        bw = bb.iloc[:, 2] - bb.iloc[:, 0]
        feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw
        feat['bb_w'] = bw / bb.iloc[:, 1] * 100

    # Volume
    feat['vr'] = v / v.rolling(20).mean()
    feat['vr5'] = v.rolling(5).mean() / v.rolling(20).mean()

    # Candles
    feat['spr'] = (h - l) / c * 100
    feat['body'] = abs(c - o) / (h - l + 1e-10)
    feat['upper_wick'] = (h - np.maximum(c, o)) / (h - l + 1e-10)
    feat['lower_wick'] = (np.minimum(c, o) - l) / (h - l + 1e-10)

    # ADX
    ax = ta.adx(h, l, c, length=14)
    if ax is not None and len(ax.columns) >= 3:
        feat['adx'] = ax.iloc[:, 0]
        feat['dip'] = ax.iloc[:, 1]
        feat['dim'] = ax.iloc[:, 2]
        feat['di_diff'] = feat['dip'] - feat['dim']

    # Choppiness
    chop = ta.chop(h, l, c, length=14)
    if chop is not None:
        feat['chop'] = chop

    # Time encoding
    hr = df.index.hour
    dw = df.index.dayofweek
    feat['h_s'] = np.sin(2 * np.pi * hr / 24)
    feat['h_c'] = np.cos(2 * np.pi * hr / 24)
    feat['d_s'] = np.sin(2 * np.pi * dw / 7)
    feat['d_c'] = np.cos(2 * np.pi * dw / 7)

    # Lags
    feat['ret1_lag1'] = feat['ret_1'].shift(1)
    feat['rsi14_lag1'] = feat['rsi14'].shift(1)
    feat['ret1_lag2'] = feat['ret_1'].shift(2)
    feat['rsi14_lag2'] = feat['rsi14'].shift(2)
    feat['ret1_lag3'] = feat['ret_1'].shift(3)
    feat['rsi14_lag3'] = feat['rsi14'].shift(3)

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
    config = PAIRS_CONFIG[pair]
    path = DATA_DIR / config['data_file']
    df = pd.read_parquet(path)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def load_model(pair: str) -> Dict:
    """Carga modelo para un par."""
    config = PAIRS_CONFIG[pair]
    path = MODELS_DIR / config['model_file']
    return joblib.load(path)


def backtest_pair(pair: str, start_date: str, end_date: str) -> List[Dict]:
    """Backtest un par en un rango de fechas."""

    config = PAIRS_CONFIG[pair]
    df = load_data(pair)
    model_data = load_model(pair)

    # Features
    feat = compute_features(df)

    # Filtrar periodo
    mask = (feat.index >= start_date) & (feat.index <= end_date)
    df_period = df[mask].copy()
    feat_period = feat[mask].copy()

    if len(df_period) < 50:
        return []

    # Alinear features
    feature_cols = model_data['feature_cols']
    available = [c for c in feature_cols if c in feat_period.columns]

    if len(available) < len(feature_cols) * 0.8:
        return []

    # Usar solo features disponibles
    valid = feat_period[available].notna().all(axis=1)
    feat_valid = feat_period.loc[valid, available]
    df_valid = df_period[valid].copy()

    if len(feat_valid) < 20:
        return []

    # Predicciones
    if model_data['scaler'] is not None:
        X = model_data['scaler'].transform(feat_valid)
    else:
        X = feat_valid.values

    preds = model_data['model'].predict(X)
    conv = np.abs(preds) / model_data['pred_std']
    signals = conv >= config['conv_min']
    directions = np.where(preds < 0, -1, 1)

    regime = detect_regime(df_valid)

    # Simular trades
    trades = []
    tp_pct = config['tp_pct']
    sl_pct = config['sl_pct']
    only_short = config.get('only_short', False)

    for i, (idx, row) in enumerate(feat_valid.iterrows()):
        if i >= len(feat_valid) - 5:
            break

        if not signals[i]:
            continue

        direction = directions[i]
        reg = regime.iloc[i]

        # Filtros
        if only_short and direction == 1:
            continue
        if reg == 'BULL' and direction == -1:
            continue
        if reg == 'BEAR' and direction == 1:
            continue

        entry_price = df_valid.loc[idx, 'close']

        if direction == 1:
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
        else:
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)

        # Buscar salida
        exit_price = None
        exit_reason = 'timeout'
        future_idx = feat_valid.index.get_loc(idx)

        for j in range(1, min(21, len(df_valid) - future_idx)):
            bar = df_valid.iloc[future_idx + j]
            high, low = bar['high'], bar['low']

            if direction == 1:
                if low <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'sl'
                    break
                elif high >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'
                    break
            else:
                if high >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'sl'
                    break
                elif low <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'
                    break

        if exit_price is None:
            exit_price = df_valid.iloc[min(future_idx + 20, len(df_valid) - 1)]['close']

        # PnL
        if direction == 1:
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        pnl_usd = pnl_pct * POSITION_SIZE

        trades.append({
            'pair': pair,
            'entry_time': idx,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'regime': reg,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd,
            'result': 'win' if pnl_pct > 0 else 'loss',
        })

    return trades


def calculate_metrics(trades: List[Dict]) -> Dict:
    """Calcula metricas de trades."""
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
    """Corre backtest para un periodo."""

    print(f"\n{'='*60}")
    print(f"PERIODO: {name}")
    print(f"Rango: {start} a {end}")
    print('='*60)

    all_trades = []
    pair_results = {}

    for pair in PAIRS_CONFIG.keys():
        try:
            trades = backtest_pair(pair, start, end)
            all_trades.extend(trades)
            metrics = calculate_metrics(trades)
            pair_results[pair] = metrics

            print(f"\n{pair}:")
            print(f"  Trades: {metrics['trades']}")
            print(f"  Win Rate: {metrics['win_rate']:.1f}%")
            print(f"  PnL: ${metrics['pnl']:.2f}")
            print(f"  Max DD: ${metrics['max_dd']:.2f}")
            pf = metrics['profit_factor']
            print(f"  PF: {pf:.2f}" if pf < 100 else "  PF: INF")

        except Exception as e:
            print(f"\n{pair}: ERROR - {e}")
            pair_results[pair] = {'trades': 0, 'win_rate': 0, 'pnl': 0, 'max_dd': 0, 'profit_factor': 0}

    total = calculate_metrics(all_trades)

    print(f"\n{'-'*40}")
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


def find_best_worst_year() -> Tuple[int, int]:
    """Encuentra mejor y peor ano."""
    print("\nAnalizando rendimiento por ano (BTC)...")

    year_pnl = {}

    for year in range(2018, 2026):
        try:
            trades = backtest_pair('BTC/USDT', f"{year}-01-01", f"{year}-12-31")
            if trades:
                m = calculate_metrics(trades)
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
    print("="*60)
    print("BACKTEST V13.02 MULTI-PERIODO")
    print("BTC + BNB (Solo SHORT para BNB)")
    print("="*60)

    results = []

    # 1. Ultimo ano
    print("\n\n" + "#"*60)
    print("1. ULTIMO ANO (Feb 2025 - Feb 2026)")
    print("#"*60)
    r = run_backtest("Ultimo Ano", "2025-02-01", "2026-02-27")
    results.append(r)

    # Mejor/peor ano
    best_year, worst_year = find_best_worst_year()
    print(f"\nMejor: {best_year}, Peor: {worst_year}")

    # 2. Mejor ano
    print("\n\n" + "#"*60)
    print(f"2. MEJOR ANO ({best_year})")
    print("#"*60)
    r = run_backtest(f"Mejor ({best_year})", f"{best_year}-01-01", f"{best_year}-12-31")
    results.append(r)

    # 3. Peor ano
    print("\n\n" + "#"*60)
    print(f"3. PEOR ANO ({worst_year})")
    print("#"*60)
    r = run_backtest(f"Peor ({worst_year})", f"{worst_year}-01-01", f"{worst_year}-12-31")
    results.append(r)

    # 4. Sintetico (H1 2022 crash + H2 2023 recovery)
    print("\n\n" + "#"*60)
    print("4. ANO SINTETICO (Crash + Recovery)")
    print("#"*60)

    trades_crash = backtest_pair('BTC/USDT', "2022-01-01", "2022-06-30")
    trades_crash += backtest_pair('BNB/USDT', "2022-01-01", "2022-06-30")
    trades_recov = backtest_pair('BTC/USDT', "2023-07-01", "2023-12-31")
    trades_recov += backtest_pair('BNB/USDT', "2023-07-01", "2023-12-31")

    all_synth = trades_crash + trades_recov
    synth_m = calculate_metrics(all_synth)

    btc_synth = [t for t in all_synth if t['pair'] == 'BTC/USDT']
    bnb_synth = [t for t in all_synth if t['pair'] == 'BNB/USDT']

    print(f"\nSintetico (Crash H1-2022 + Recovery H2-2023):")
    print(f"  Trades: {synth_m['trades']}")
    print(f"  Win Rate: {synth_m['win_rate']:.1f}%")
    print(f"  PnL: ${synth_m['pnl']:.2f}")
    print(f"  Max DD: ${synth_m['max_dd']:.2f}")

    results.append({
        'period': 'Sintetico',
        'start': 'Mixed',
        'end': 'Mixed',
        'total': synth_m,
        'by_pair': {
            'BTC/USDT': calculate_metrics(btc_synth),
            'BNB/USDT': calculate_metrics(bnb_synth),
        }
    })

    # ==========================================================================
    # RESUMEN FINAL
    # ==========================================================================
    print("\n\n" + "#"*60)
    print("RESUMEN FINAL V13.02")
    print("#"*60)

    print("\n" + "="*80)
    print(f"| {'Periodo':<22} | {'Trades':>6} | {'WR':>7} | {'PnL':>10} | {'MaxDD%':>8} | {'PF':>6} |")
    print("="*80)

    for r in results:
        t = r['total']
        pf_str = f"{t['profit_factor']:.2f}" if t['profit_factor'] < 100 else "INF"
        dd_pct = t['max_dd'] / INITIAL_CAPITAL * 100
        print(f"| {r['period']:<22} | {t['trades']:>6} | {t['win_rate']:>5.1f}% | ${t['pnl']:>8.2f} | {dd_pct:>6.1f}% | {pf_str:>6} |")

    print("="*80)

    # Por moneda
    print("\n\nDETALLE POR MONEDA:")
    print("="*60)

    for pair in PAIRS_CONFIG.keys():
        print(f"\n{pair}:")
        print("-"*60)
        print(f"| {'Periodo':<22} | {'Trades':>6} | {'WR':>7} | {'PnL':>10} |")
        print("-"*60)

        for r in results:
            m = r['by_pair'].get(pair, {})
            if m.get('trades', 0) > 0:
                print(f"| {r['period']:<22} | {m['trades']:>6} | {m['win_rate']:>5.1f}% | ${m['pnl']:>8.2f} |")
            else:
                print(f"| {r['period']:<22} | {0:>6} | {'N/A':>7} | ${'0.00':>8} |")

        print("-"*60)

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
    df_out.to_csv(DATA_DIR / 'backtest_v13_multiperiod.csv', index=False)
    print(f"\n\nGuardado: data/backtest_v13_multiperiod.csv")


if __name__ == '__main__':
    main()
