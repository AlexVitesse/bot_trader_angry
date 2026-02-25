"""
Backtest V12 + Support/Resistance Filter
=========================================
Agrega filtro de S/R para mejorar timing de entrada.
Compara resultados con y sin S/R.
"""

import json
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from regime_detector_v2 import RegimeDetector, MarketRegime
from support_resistance import SupportResistance, should_take_long, should_take_short

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

# Cargar pares filtrados
with open(MODELS_DIR / 'v12_good_pairs.json') as f:
    pair_config = json.load(f)
    GOOD_PAIRS = pair_config['good_pairs']

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


def run_backtest(pair_data, models, start_date, end_date, regime_detector, use_sr=False, sr_detector=None):
    """
    Backtest con opción de usar S/R.

    Args:
        use_sr: Si True, aplica filtro de S/R
        sr_detector: Instancia de SupportResistance
    """
    trades = []
    sr_rejections = {'at_resistance': 0, 'low_rr': 0, 'at_support_short': 0}

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
                        'pnl': pnl,
                        'exit_reason': 'tp' if hit_tp else ('sl' if hit_sl else 'timeout'),
                        'regime': pos.get('regime', 'unknown'),
                        'sr_filtered': pos.get('sr_filtered', False),
                    })
                    pos = None

            if pos is None:
                regime_str = regimes.loc[ts, 'regime']
                try:
                    regime = MarketRegime(regime_str)
                except:
                    continue

                if regime not in [MarketRegime.BULL_TREND, MarketRegime.WEAK_TREND, MarketRegime.HIGH_VOL]:
                    continue

                rsi = feat.loc[ts, 'rsi14']
                bb_pos = feat.loc[ts, 'bb_pos'] if 'bb_pos' in feat.columns else 0.5
                adx = feat.loc[ts, 'adx'] if 'adx' in feat.columns else 25
                chop = feat.loc[ts, 'chop'] if 'chop' in feat.columns else 50
                atr = feat.loc[ts, 'atr14']

                X = feat.loc[ts:ts][fcols]
                if X.isna().any().any():
                    continue

                pred = model.predict(X)[0]
                sig = 1 if pred > 0.5 else -1
                conviction = abs(pred - 0.5) * 10

                # FILTROS UNIVERSALES PARA WR 50%+
                if conviction < 2.0:
                    continue
                if not (38 <= rsi <= 72):
                    continue
                if not (0.2 <= bb_pos <= 0.8):
                    continue
                if chop > 52:
                    continue

                # FILTRO S/R (si está activado)
                sr_filtered = False
                if use_sr and sr_detector is not None:
                    sr_analysis = sr_detector.analyze(df, i)

                    if sig == 1:  # LONG
                        should_trade, reason = should_take_long(sr_analysis, min_rr=1.2)
                        if not should_trade:
                            if sr_analysis.at_resistance:
                                sr_rejections['at_resistance'] += 1
                            else:
                                sr_rejections['low_rr'] += 1
                            continue
                        sr_filtered = True
                    else:  # SHORT
                        should_trade, reason = should_take_short(sr_analysis, min_rr=1.2)
                        if not should_trade:
                            sr_rejections['at_support_short'] += 1
                            continue
                        sr_filtered = True

                # Ajustes por regimen
                if regime == MarketRegime.BULL_TREND:
                    if sig != 1:
                        continue
                    tp_mult = 2.0
                    sl_mult = 1.0
                    max_hold = 25
                    pos_mult = 1.0

                elif regime == MarketRegime.WEAK_TREND:
                    if not (15 <= adx <= 40):
                        continue
                    tp_mult = 1.5
                    sl_mult = 1.0
                    max_hold = 15
                    pos_mult = 0.85

                elif regime == MarketRegime.HIGH_VOL:
                    if not (40 <= rsi <= 60):
                        continue
                    tp_mult = 2.0
                    sl_mult = 1.25
                    max_hold = 12
                    pos_mult = 0.5

                else:
                    continue

                tp_pct = atr / price * tp_mult
                sl_pct = atr / price * sl_mult

                risk_amt = balance * RISK_PER_TRADE * pos_mult
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
                    'sr_filtered': sr_filtered,
                }

        if pos is not None:
            pnl = pos['size'] * pos['dir'] * (df.iloc[-1]['close'] - pos['entry'])
            balance += pnl
            trades.append({
                'pair': pair,
                'dir': pos['dir'],
                'pnl': pnl,
                'exit_reason': 'eod',
                'regime': pos.get('regime', 'unknown'),
                'sr_filtered': pos.get('sr_filtered', False),
            })

    return trades, max_dd, sr_rejections


def compute_metrics(trades, max_dd):
    if not trades:
        return {'trades': 0, 'wins': 0, 'wr': 0, 'pnl': 0, 'pf': 0, 'max_dd': 0}

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
        'max_dd': max_dd,
    }


def main():
    print("="*70)
    print("BACKTEST V12 + SUPPORT/RESISTANCE FILTER")
    print("="*70)

    models = {}
    for pair in GOOD_PAIRS:
        safe = pair.replace('/', '')
        try:
            models[safe] = joblib.load(MODELS_DIR / f'v95_v7_{safe}.pkl')
        except:
            pass

    pair_data = {}
    for pair in GOOD_PAIRS:
        df = load_data(pair)
        if df is not None:
            pair_data[pair] = df

    detector = RegimeDetector()
    sr_detector = SupportResistance(lookback=20, threshold_pct=0.02, min_touches=2)

    periods = [
        ('Ultimo Ano', '2025-02-01', '2026-02-24'),
        ('Bear Market 2022', '2022-01-01', '2023-01-01'),
    ]

    print("\n" + "="*70)
    print("COMPARACION: V12 SIN S/R vs V12 CON S/R")
    print("="*70)

    for period_name, start, end in periods:
        print(f"\n{'='*70}")
        print(f"PERIODO: {period_name}")
        print(f"{'='*70}")

        # Sin S/R
        trades_no_sr, max_dd_no_sr, _ = run_backtest(
            pair_data, models, start, end, detector,
            use_sr=False
        )
        metrics_no_sr = compute_metrics(trades_no_sr, max_dd_no_sr)

        # Con S/R
        trades_sr, max_dd_sr, sr_rejections = run_backtest(
            pair_data, models, start, end, detector,
            use_sr=True, sr_detector=sr_detector
        )
        metrics_sr = compute_metrics(trades_sr, max_dd_sr)

        print(f"\n{'SIN S/R':<20} {'CON S/R':<20} {'DIFERENCIA':<15}")
        print("-"*55)

        wr_diff = metrics_sr['wr'] - metrics_no_sr['wr']
        wr_mark = "MEJOR" if wr_diff > 0 else "PEOR" if wr_diff < 0 else "IGUAL"
        print(f"Trades: {metrics_no_sr['trades']:<12} {metrics_sr['trades']:<12} {metrics_sr['trades'] - metrics_no_sr['trades']:+d}")
        print(f"WR:     {metrics_no_sr['wr']:<11.1f}% {metrics_sr['wr']:<11.1f}% {wr_diff:+.1f}% ({wr_mark})")
        print(f"PnL:    ${metrics_no_sr['pnl']:<10,.0f} ${metrics_sr['pnl']:<10,.0f} ${metrics_sr['pnl'] - metrics_no_sr['pnl']:+,.0f}")
        print(f"PF:     {metrics_no_sr['pf']:<12.2f} {metrics_sr['pf']:<12.2f} {metrics_sr['pf'] - metrics_no_sr['pf']:+.2f}")
        print(f"Max DD: {metrics_no_sr['max_dd']:<11.1f}% {metrics_sr['max_dd']:<11.1f}% {metrics_sr['max_dd'] - metrics_no_sr['max_dd']:+.1f}%")

        print(f"\nTrades rechazados por S/R:")
        print(f"  - Cerca de resistencia (LONG): {sr_rejections['at_resistance']}")
        print(f"  - R:R bajo: {sr_rejections['low_rr']}")
        print(f"  - Cerca de soporte (SHORT): {sr_rejections['at_support_short']}")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)


if __name__ == '__main__':
    main()
