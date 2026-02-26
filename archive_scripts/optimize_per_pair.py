"""
Optimizador de Parametros por Par - V12.1
==========================================
Encuentra los mejores parametros para cada par individualmente.
"""

import json
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from regime_detector_v2 import RegimeDetector, MarketRegime

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

# Todos los pares (incluyendo los excluidos para ver si funcionan con otros params)
ALL_PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
    'DOGE/USDT', 'SOL/USDT', 'DOT/USDT', 'AVAX/USDT', 'LINK/USDT', 'NEAR/USDT'
]

INITIAL_CAPITAL = 500.0
RISK_PER_TRADE = 0.02


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

    atr_1 = ta.atr(h, l, c, length=1)
    atr_sum = atr_1.rolling(14).sum()
    high_max = h.rolling(14).max()
    low_min = l.rolling(14).min()
    feat['chop'] = 100 * np.log10(atr_sum / (high_max - low_min + 1e-10)) / np.log10(14)

    return feat


def backtest_single_pair(df, model, feat, regimes, params, start_date, end_date):
    """Backtest un solo par con parametros especificos."""
    df = df[(df.index >= start_date) & (df.index < end_date)]
    if len(df) < 250:
        return None

    fcols = [c for c in model.feature_name_ if c in feat.columns]
    feat = feat[(feat.index >= start_date) & (feat.index < end_date)]
    regimes = regimes[(regimes.index >= start_date) & (regimes.index < end_date)]

    # Extraer parametros
    min_conv = params['min_conviction']
    rsi_min, rsi_max = params['rsi_range']
    bb_min, bb_max = params['bb_range']
    max_chop = params['max_chop']
    tp_mult = params['tp_mult']
    sl_mult = params['sl_mult']

    trades = []
    balance = INITIAL_CAPITAL
    pos = None

    for i in range(250, len(df)):
        ts = df.index[i]
        if ts not in feat.index:
            continue
        price = df.iloc[i]['close']

        if pos is not None:
            pnl_pct = (price - pos['entry']) / pos['entry'] * pos['dir']
            hit_tp = pnl_pct >= pos['tp_pct']
            hit_sl = pnl_pct <= -pos['sl_pct']
            timeout = (i - pos['bar']) >= 20

            if hit_tp or hit_sl or timeout:
                pnl = pos['size'] * pos['dir'] * (price - pos['entry'])
                balance += pnl
                trades.append({'pnl': pnl})
                pos = None

        if pos is None:
            if ts not in regimes.index:
                continue
            regime_str = regimes.loc[ts, 'regime']
            try:
                regime = MarketRegime(regime_str)
            except:
                continue

            if regime not in [MarketRegime.BULL_TREND, MarketRegime.WEAK_TREND, MarketRegime.HIGH_VOL]:
                continue

            rsi = feat.loc[ts, 'rsi14']
            bb_pos = feat.loc[ts, 'bb_pos'] if 'bb_pos' in feat.columns else 0.5
            chop = feat.loc[ts, 'chop'] if 'chop' in feat.columns else 50
            atr = feat.loc[ts, 'atr14']

            X = feat.loc[ts:ts][fcols]
            if X.isna().any().any():
                continue

            pred = model.predict(X)[0]
            sig = 1 if pred > 0.5 else -1
            conviction = abs(pred - 0.5) * 10

            # Filtros con parametros especificos del par
            if conviction < min_conv:
                continue
            if not (rsi_min <= rsi <= rsi_max):
                continue
            if not (bb_min <= bb_pos <= bb_max):
                continue
            if chop > max_chop:
                continue

            # Solo LONG en BULL
            if regime == MarketRegime.BULL_TREND and sig != 1:
                continue

            tp_pct = atr / price * tp_mult
            sl_pct = atr / price * sl_mult
            risk_amt = balance * RISK_PER_TRADE
            size = risk_amt / (sl_pct * price) if sl_pct > 0 else 0

            pos = {'entry': price, 'dir': sig, 'size': size,
                   'tp_pct': tp_pct, 'sl_pct': sl_pct, 'bar': i}

    if pos:
        pnl = pos['size'] * pos['dir'] * (df.iloc[-1]['close'] - pos['entry'])
        trades.append({'pnl': pnl})

    if not trades:
        return None

    wins = len([t for t in trades if t['pnl'] > 0])
    return {
        'trades': len(trades),
        'wins': wins,
        'wr': wins / len(trades) * 100,
        'pnl': sum(t['pnl'] for t in trades),
    }


def optimize_pair(pair, df, model, detector, periods):
    """Encuentra los mejores parametros para un par."""
    print(f"\n{'='*50}")
    print(f"OPTIMIZANDO: {pair}")
    print(f"{'='*50}")

    feat = compute_features(df)
    regimes = detector.detect_regime_series(df)

    # Grid de parametros a probar
    param_grid = {
        'min_conviction': [1.5, 2.0, 2.5],
        'rsi_range': [(30, 70), (35, 75), (38, 72), (40, 65)],
        'bb_range': [(0.15, 0.85), (0.2, 0.8), (0.25, 0.75)],
        'max_chop': [48, 50, 52, 55],
        'tp_mult': [1.5, 2.0, 2.5],
        'sl_mult': [0.8, 1.0, 1.25],
    }

    # Parametros base V12
    base_params = {
        'min_conviction': 2.0,
        'rsi_range': (38, 72),
        'bb_range': (0.2, 0.8),
        'max_chop': 52,
        'tp_mult': 2.0,
        'sl_mult': 1.0,
    }

    # Resultado con parametros base
    base_results = []
    for period_name, start, end in periods:
        r = backtest_single_pair(df, model, feat, regimes, base_params, start, end)
        if r:
            base_results.append(r)

    if base_results:
        base_wr = np.mean([r['wr'] for r in base_results])
        base_pnl = sum(r['pnl'] for r in base_results)
        print(f"Base V12: WR {base_wr:.1f}%, PnL ${base_pnl:.0f}")
    else:
        base_wr, base_pnl = 0, 0
        print("Base V12: Sin trades")

    # Buscar mejores parametros
    best_score = base_wr + (base_pnl / 100)  # Score compuesto
    best_params = base_params.copy()
    best_wr = base_wr
    best_pnl = base_pnl

    # Optimizar cada parametro individualmente (mas rapido que grid completo)
    for param_name, values in param_grid.items():
        for value in values:
            test_params = best_params.copy()
            test_params[param_name] = value

            results = []
            for period_name, start, end in periods:
                r = backtest_single_pair(df, model, feat, regimes, test_params, start, end)
                if r:
                    results.append(r)

            if results:
                avg_wr = np.mean([r['wr'] for r in results])
                total_pnl = sum(r['pnl'] for r in results)
                score = avg_wr + (total_pnl / 100)

                if score > best_score and avg_wr >= 45:  # Minimo WR 45%
                    best_score = score
                    best_params[param_name] = value
                    best_wr = avg_wr
                    best_pnl = total_pnl

    # Resultado final
    improvement_wr = best_wr - base_wr
    improvement_pnl = best_pnl - base_pnl

    print(f"\nMejor config encontrada:")
    print(f"  Conviction: >= {best_params['min_conviction']}")
    print(f"  RSI: {best_params['rsi_range']}")
    print(f"  BB: {best_params['bb_range']}")
    print(f"  Chop: < {best_params['max_chop']}")
    print(f"  TP: {best_params['tp_mult']}x ATR")
    print(f"  SL: {best_params['sl_mult']}x ATR")
    print(f"\nResultado: WR {best_wr:.1f}% ({improvement_wr:+.1f}%), PnL ${best_pnl:.0f} (${improvement_pnl:+.0f})")

    return {
        'pair': pair,
        'base_wr': base_wr,
        'base_pnl': base_pnl,
        'best_params': best_params,
        'best_wr': best_wr,
        'best_pnl': best_pnl,
        'improvement_wr': improvement_wr,
        'improvement_pnl': improvement_pnl,
    }


def main():
    print("="*70)
    print("OPTIMIZACION DE PARAMETROS POR PAR - V12.1")
    print("="*70)

    detector = RegimeDetector()

    periods = [
        ('Ultimo Ano', '2025-02-01', '2026-02-24'),
        ('Bear Market 2022', '2022-01-01', '2023-01-01'),
    ]

    results = {}

    for pair in ALL_PAIRS:
        df = load_data(pair)
        if df is None:
            print(f"\n{pair}: Sin datos")
            continue

        safe = pair.replace('/', '')
        try:
            model = joblib.load(MODELS_DIR / f'v95_v7_{safe}.pkl')
        except:
            print(f"\n{pair}: Sin modelo")
            continue

        result = optimize_pair(pair, df, model, detector, periods)
        results[pair] = result

    # Resumen
    print("\n" + "="*70)
    print("RESUMEN - PARAMETROS OPTIMIZADOS POR PAR")
    print("="*70)

    print(f"\n{'Par':<12} {'Base WR':>10} {'Opt WR':>10} {'Mejora':>10} {'PnL Opt':>12}")
    print("-"*60)

    total_improvement_wr = 0
    total_improvement_pnl = 0
    count = 0

    for pair, r in sorted(results.items(), key=lambda x: -x[1]['best_wr']):
        mark = "**" if r['best_wr'] >= 50 else ""
        print(f"{pair:<12} {r['base_wr']:>9.1f}% {r['best_wr']:>9.1f}% {r['improvement_wr']:>+9.1f}% ${r['best_pnl']:>10,.0f} {mark}")
        total_improvement_wr += r['improvement_wr']
        total_improvement_pnl += r['improvement_pnl']
        count += 1

    print("-"*60)
    print(f"{'PROMEDIO':<12} {'':<10} {'':<10} {total_improvement_wr/count:>+9.1f}% ${total_improvement_pnl:>10,.0f}")

    # Guardar configuracion
    config = {
        'version': 'V12.1',
        'description': 'Parametros optimizados por par',
        'pairs': {}
    }

    for pair, r in results.items():
        config['pairs'][pair] = {
            'params': r['best_params'],
            'expected_wr': r['best_wr'],
            'expected_pnl': r['best_pnl'],
        }

    with open(MODELS_DIR / 'v12_1_pair_params.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nConfiguracion guardada en models/v12_1_pair_params.json")


if __name__ == '__main__':
    main()
