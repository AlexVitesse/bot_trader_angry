"""
Backtest V12: Sistema Hibrido Adaptativo
=========================================
Combina:
- V11 PRO filters cuando mercado es dificil (ultimo ano)
- V9.5 agresivo cuando mercado tiene tendencia (bear market)

Detecta automaticamente que tipo de mercado es y adapta filtros.
"""

import json
import time
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from regime_detector_v2 import RegimeDetector, MarketRegime

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT',
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


def run_backtest_v12_hybrid(pair_data, models, start_date, end_date, regime_detector):
    """
    V12 Hibrido: Adapta filtros segun regimen.

    En TREND (bull/bear): Filtros relajados, TP agresivo
    En WEAK/LATERAL: Filtros estrictos para WR
    """
    trades = []

    for pair, df in pair_data.items():
        df = df[(df.index >= start_date) & (df.index < end_date)]
        if len(df) < 250:
            continue

        safe = pair.replace('/', '')
        model = models.get(safe)
        if model is None:
            continue

        feat = compute_features(df)
        regimes = regime_detector.detect_regime_series(df)
        fcols = [c for c in model.feature_name_ if c in feat.columns]

        balance = INITIAL_CAPITAL
        peak = balance
        max_dd = 0
        pos = None

        for i in range(250, len(df)):
            row = df.iloc[i]
            ts = df.index[i]
            price = row['close']

            # Gestionar posicion
            if pos is not None:
                pnl_pct = (price - pos['entry']) / pos['entry'] * pos['dir']
                hit_tp = pnl_pct >= pos['tp_pct']
                hit_sl = pnl_pct <= -pos['sl_pct']
                timeout = (i - pos['bar']) >= pos.get('max_hold', 20)

                if hit_tp or hit_sl or timeout:
                    pnl = pos['size'] * pos['dir'] * (price - pos['entry'])
                    balance += pnl
                    peak = max(peak, balance)
                    dd = (peak - balance) / peak * 100
                    max_dd = max(max_dd, dd)
                    trades.append({
                        'pair': pair,
                        'dir': pos['dir'],
                        'entry': pos['entry'],
                        'exit': price,
                        'pnl': pnl,
                        'exit_reason': 'tp' if hit_tp else ('sl' if hit_sl else 'timeout'),
                        'regime': pos.get('regime', 'unknown'),
                    })
                    pos = None

            if pos is None:
                # Detectar regimen
                regime_str = regimes.loc[ts, 'regime']
                try:
                    regime = MarketRegime(regime_str)
                except:
                    regime = MarketRegime.UNKNOWN

                # Obtener indicadores
                rsi = feat.loc[ts, 'rsi14']
                bb_pos = feat.loc[ts, 'bb_pos'] if 'bb_pos' in feat.columns else 0.5
                adx = feat.loc[ts, 'adx'] if 'adx' in feat.columns else 25
                vol_ratio = feat.loc[ts, 'vr'] if 'vr' in feat.columns else 1
                chop = feat.loc[ts, 'chop'] if 'chop' in feat.columns else 50
                atr = feat.loc[ts, 'atr14']

                # Generar senal
                X = feat.loc[ts:ts][fcols]
                if X.isna().any().any():
                    continue

                pred = model.predict(X)[0]
                sig = 1 if pred > 0.5 else -1
                conviction = abs(pred - 0.5) * 10

                # LOGICA HIBRIDA POR REGIMEN
                if regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
                    # TENDENCIA FUERTE: Filtros relajados, TP agresivo
                    # Como en Bear Market 2022 que funciona bien
                    if conviction < 1.5:
                        continue
                    if chop > 55:
                        continue
                    # Verificar direccion con tendencia
                    if regime == MarketRegime.BULL_TREND and sig != 1:
                        continue
                    if regime == MarketRegime.BEAR_TREND and sig != -1:
                        continue

                    tp_mult = 2.5
                    sl_mult = 1.0
                    max_hold = 25

                elif regime == MarketRegime.HIGH_VOL:
                    # ALTA VOLATILIDAD: Filtros moderados, posicion reducida
                    if conviction < 2.0:
                        continue
                    if not (35 <= rsi <= 65):
                        continue

                    tp_mult = 2.0
                    sl_mult = 1.25
                    max_hold = 15

                elif regime == MarketRegime.WEAK_TREND:
                    # TENDENCIA DEBIL: Filtros ESTRICTOS para WR
                    # Como en ultimo ano que necesita WR alto
                    if conviction < 2.0:
                        continue
                    if not (40 <= rsi <= 70):
                        continue
                    if not (0.25 <= bb_pos <= 0.75):
                        continue
                    if not (15 <= adx <= 40):
                        continue
                    if chop > 48:
                        continue

                    tp_mult = 1.5
                    sl_mult = 1.0
                    max_hold = 15

                elif regime == MarketRegime.LOW_VOL:
                    # BAJA VOLATILIDAD: TP/SL fijos pequenos
                    if conviction < 1.8:
                        continue
                    if not (35 <= rsi <= 65):
                        continue

                    # TP/SL fijos
                    tp_pct = 0.015  # 1.5%
                    sl_pct = 0.01   # 1%
                    max_hold = 20

                    risk_amt = balance * RISK_PER_TRADE
                    size = risk_amt / (sl_pct * price) if sl_pct > 0 else 0

                    pos = {
                        'entry': price,
                        'dir': sig,
                        'size': size,
                        'tp_pct': tp_pct,
                        'sl_pct': sl_pct,
                        'bar': i,
                        'max_hold': max_hold,
                        'regime': regime.value,
                    }
                    continue

                else:
                    # LATERAL o UNKNOWN: NO OPERAR
                    continue

                # Calcular TP/SL con ATR
                tp_pct = atr / price * tp_mult
                sl_pct = atr / price * sl_mult

                risk_amt = balance * RISK_PER_TRADE
                size = risk_amt / (sl_pct * price) if sl_pct > 0 else 0

                pos = {
                    'entry': price,
                    'dir': sig,
                    'size': size,
                    'tp_pct': tp_pct,
                    'sl_pct': sl_pct,
                    'bar': i,
                    'max_hold': max_hold,
                    'regime': regime.value,
                }

        if pos is not None:
            pnl = pos['size'] * pos['dir'] * (df.iloc[-1]['close'] - pos['entry'])
            balance += pnl
            trades.append({
                'pair': pair,
                'dir': pos['dir'],
                'entry': pos['entry'],
                'exit': df.iloc[-1]['close'],
                'pnl': pnl,
                'exit_reason': 'eod',
                'regime': pos.get('regime', 'unknown'),
            })

    return trades, max_dd


def compute_metrics(trades, max_dd):
    if not trades:
        return {'trades': 0, 'wins': 0, 'wr': 0, 'pnl': 0, 'pf': 0, 'return_pct': 0, 'max_dd': 0}

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]

    gross_profit = sum(t['pnl'] for t in wins)
    gross_loss = abs(sum(t['pnl'] for t in losses))

    return {
        'trades': len(trades),
        'wins': len(wins),
        'wr': len(wins) / len(trades) * 100 if trades else 0,
        'pnl': sum(t['pnl'] for t in trades),
        'pf': gross_profit / gross_loss if gross_loss > 0 else 999,
        'return_pct': sum(t['pnl'] for t in trades) / INITIAL_CAPITAL * 100,
        'max_dd': max_dd,
    }


def main():
    print("="*70)
    print("BACKTEST V12: SISTEMA HIBRIDO ADAPTATIVO")
    print("="*70)
    print("\nLogica:")
    print("- TREND fuerte: Filtros relajados, TP 2.5x (como Bear Market)")
    print("- WEAK TREND: Filtros estrictos para WR (como ultimo ano)")
    print("- HIGH VOL: Filtros moderados, posicion reducida")
    print("- LATERAL: NO OPERAR")

    # Cargar modelos
    models = {}
    for pair in PAIRS:
        safe = pair.replace('/', '')
        try:
            models[safe] = joblib.load(MODELS_DIR / f'v95_v7_{safe}.pkl')
        except:
            pass

    # Cargar datos
    pair_data = {}
    for pair in PAIRS:
        df = load_data(pair)
        if df is not None:
            pair_data[pair] = df

    detector = RegimeDetector()

    periods = [
        ('Ultimo Ano', '2025-02-01', '2026-02-24'),
        ('Bear Market 2022', '2022-01-01', '2023-01-01'),
    ]

    results = {}

    for period_name, start, end in periods:
        print(f"\n{'='*70}")
        print(f"PERIODO: {period_name}")
        print(f"{'='*70}")

        trades, max_dd = run_backtest_v12_hybrid(pair_data, models, start, end, detector)
        metrics = compute_metrics(trades, max_dd)

        print(f"\nResultados V12 Hibrido:")
        print(f"  Trades: {metrics['trades']}")
        print(f"  Win Rate: {metrics['wr']:.1f}%")
        print(f"  PnL: ${metrics['pnl']:,.0f}")
        print(f"  Profit Factor: {metrics['pf']:.2f}")
        print(f"  Max DD: {metrics['max_dd']:.1f}%")

        # Distribucion por regimen
        if trades:
            print(f"\n  Por regimen:")
            regime_stats = {}
            for t in trades:
                r = t.get('regime', 'unknown')
                if r not in regime_stats:
                    regime_stats[r] = {'wins': 0, 'losses': 0, 'pnl': 0}
                if t['pnl'] > 0:
                    regime_stats[r]['wins'] += 1
                else:
                    regime_stats[r]['losses'] += 1
                regime_stats[r]['pnl'] += t['pnl']

            for r, s in sorted(regime_stats.items(), key=lambda x: -x[1]['wins']-x[1]['losses']):
                total = s['wins'] + s['losses']
                wr = s['wins'] / total * 100 if total > 0 else 0
                print(f"    {r:<15}: {total:>4} trades, WR {wr:>5.1f}%, PnL ${s['pnl']:>8,.0f}")

        results[period_name] = metrics

    # Resumen
    print("\n" + "="*70)
    print("RESUMEN FINAL V12 HIBRIDO")
    print("="*70)

    total_pnl = sum(r['pnl'] for r in results.values())
    avg_wr = np.mean([r['wr'] for r in results.values()])
    avg_dd = np.mean([r['max_dd'] for r in results.values()])

    print(f"\nPnL Total: ${total_pnl:,.0f}")
    print(f"WR Promedio: {avg_wr:.1f}%")
    print(f"DD Promedio: {avg_dd:.1f}%")


if __name__ == '__main__':
    main()
