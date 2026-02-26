"""
Backtest V12 + Support/Resistance Filter (OPTIMIZADO)
=====================================================
Versión rápida con S/R pre-calculado.
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

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

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


def compute_sr_fast(df, lookback=20):
    """
    Calcula S/R de forma vectorizada (RÁPIDO).
    Usa rolling max/min como resistencia/soporte simple.
    """
    sr = pd.DataFrame(index=df.index)

    # Soporte = mínimo de últimas N velas
    sr['support'] = df['low'].rolling(lookback).min().shift(1)

    # Resistencia = máximo de últimas N velas
    sr['resistance'] = df['high'].rolling(lookback).max().shift(1)

    # Distancia al soporte/resistencia (%)
    sr['dist_support'] = (df['close'] - sr['support']) / df['close'] * 100
    sr['dist_resistance'] = (sr['resistance'] - df['close']) / df['close'] * 100

    # Flags: cerca de S/R (dentro del 2%)
    sr['at_support'] = sr['dist_support'] < 2.0
    sr['at_resistance'] = sr['dist_resistance'] < 2.0

    # R:R ratio (potencial ganancia vs pérdida)
    potential_profit = sr['resistance'] - df['close']
    potential_loss = df['close'] - sr['support']
    sr['rr_ratio'] = potential_profit / (potential_loss + 1e-10)

    return sr


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

    sr_ta = ta.stochrsi(c, length=14, rsi_length=14, k=3, d=3)
    if sr_ta is not None:
        feat['srsi_k'] = sr_ta.iloc[:, 0]
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

    # Agregar S/R features (no usados por modelo, solo para filtro)
    sr = compute_sr_fast(df)
    feat['sr_dist_support'] = sr['dist_support']
    feat['sr_dist_resistance'] = sr['dist_resistance']
    feat['sr_at_support'] = sr['at_support']
    feat['sr_at_resistance'] = sr['at_resistance']
    feat['sr_rr_ratio'] = sr['rr_ratio']

    return feat


def run_backtest(pair_data, models, start_date, end_date, regime_detector, use_sr=False, min_rr=1.2):
    """Backtest con opción de usar S/R."""
    trades = []
    sr_stats = {'rejected_resistance': 0, 'rejected_low_rr': 0, 'rejected_support_short': 0}

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
        fcols = [c for c in model.feature_name_ if c in feat.columns and not c.startswith('sr_')]

        balance = INITIAL_CAPITAL
        peak = balance
        max_dd = 0
        pos = None

        for i in range(250, len(df)):
            ts = df.index[i]
            price = df.iloc[i]['close']

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
                    trades.append({'pair': pair, 'pnl': pnl})
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

                # FILTROS UNIVERSALES
                if conviction < 2.0:
                    continue
                if not (38 <= rsi <= 72):
                    continue
                if not (0.2 <= bb_pos <= 0.8):
                    continue
                if chop > 52:
                    continue

                # FILTRO S/R
                if use_sr:
                    at_support = feat.loc[ts, 'sr_at_support']
                    at_resistance = feat.loc[ts, 'sr_at_resistance']
                    rr_ratio = feat.loc[ts, 'sr_rr_ratio']

                    if sig == 1:  # LONG
                        if at_resistance:
                            sr_stats['rejected_resistance'] += 1
                            continue
                        if rr_ratio < min_rr and not at_support:
                            sr_stats['rejected_low_rr'] += 1
                            continue
                    else:  # SHORT
                        if at_support:
                            sr_stats['rejected_support_short'] += 1
                            continue

                # Ajustes por regimen
                if regime == MarketRegime.BULL_TREND:
                    if sig != 1:
                        continue
                    tp_mult, sl_mult, max_hold, pos_mult = 2.0, 1.0, 25, 1.0
                elif regime == MarketRegime.WEAK_TREND:
                    if not (15 <= adx <= 40):
                        continue
                    tp_mult, sl_mult, max_hold, pos_mult = 1.5, 1.0, 15, 0.85
                elif regime == MarketRegime.HIGH_VOL:
                    if not (40 <= rsi <= 60):
                        continue
                    tp_mult, sl_mult, max_hold, pos_mult = 2.0, 1.25, 12, 0.5
                else:
                    continue

                tp_pct = atr / price * tp_mult
                sl_pct = atr / price * sl_mult
                risk_amt = balance * RISK_PER_TRADE * pos_mult
                size = risk_amt / (sl_pct * price) if sl_pct > 0 else 0

                pos = {'entry': price, 'dir': sig, 'size': size, 'tp_pct': tp_pct,
                       'sl_pct': sl_pct, 'bar': i, 'max_hold': max_hold}

        if pos:
            pnl = pos['size'] * pos['dir'] * (df.iloc[-1]['close'] - pos['entry'])
            trades.append({'pair': pair, 'pnl': pnl})

    return trades, max_dd, sr_stats


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
    print("BACKTEST V12 + SUPPORT/RESISTANCE (OPTIMIZADO)")
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

    periods = [
        ('Ultimo Ano', '2025-02-01', '2026-02-24'),
        ('Bear Market 2022', '2022-01-01', '2023-01-01'),
    ]

    all_results = {'sin_sr': [], 'con_sr': []}

    for period_name, start, end in periods:
        print(f"\n{'='*70}")
        print(f"PERIODO: {period_name}")
        print(f"{'='*70}")

        # Sin S/R
        trades_no_sr, max_dd_no_sr, _ = run_backtest(
            pair_data, models, start, end, detector, use_sr=False
        )
        m_no_sr = compute_metrics(trades_no_sr, max_dd_no_sr)

        # Con S/R
        trades_sr, max_dd_sr, sr_stats = run_backtest(
            pair_data, models, start, end, detector, use_sr=True, min_rr=1.2
        )
        m_sr = compute_metrics(trades_sr, max_dd_sr)

        all_results['sin_sr'].append(m_no_sr)
        all_results['con_sr'].append(m_sr)

        print(f"\n{'Metrica':<12} {'SIN S/R':<15} {'CON S/R':<15} {'Diferencia':<15}")
        print("-"*57)

        wr_diff = m_sr['wr'] - m_no_sr['wr']
        wr_mark = ">> MEJOR" if wr_diff > 0.5 else ("<< PEOR" if wr_diff < -0.5 else "~igual")

        print(f"{'Trades':<12} {m_no_sr['trades']:<15} {m_sr['trades']:<15} {m_sr['trades'] - m_no_sr['trades']:+d}")
        print(f"{'Win Rate':<12} {m_no_sr['wr']:<14.1f}% {m_sr['wr']:<14.1f}% {wr_diff:+.1f}% {wr_mark}")
        print(f"{'PnL':<12} ${m_no_sr['pnl']:<13,.0f} ${m_sr['pnl']:<13,.0f} ${m_sr['pnl'] - m_no_sr['pnl']:+,.0f}")
        print(f"{'P. Factor':<12} {m_no_sr['pf']:<15.2f} {m_sr['pf']:<15.2f} {m_sr['pf'] - m_no_sr['pf']:+.2f}")
        print(f"{'Max DD':<12} {m_no_sr['max_dd']:<14.1f}% {m_sr['max_dd']:<14.1f}% {m_sr['max_dd'] - m_no_sr['max_dd']:+.1f}%")

        total_rejected = sr_stats['rejected_resistance'] + sr_stats['rejected_low_rr'] + sr_stats['rejected_support_short']
        print(f"\nTrades filtrados por S/R: {total_rejected}")
        print(f"  - LONG cerca de resistencia: {sr_stats['rejected_resistance']}")
        print(f"  - R:R bajo (<1.2): {sr_stats['rejected_low_rr']}")
        print(f"  - SHORT cerca de soporte: {sr_stats['rejected_support_short']}")

    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)

    avg_wr_no_sr = np.mean([r['wr'] for r in all_results['sin_sr']])
    avg_wr_sr = np.mean([r['wr'] for r in all_results['con_sr']])
    total_pnl_no_sr = sum(r['pnl'] for r in all_results['sin_sr'])
    total_pnl_sr = sum(r['pnl'] for r in all_results['con_sr'])

    print(f"\nSIN S/R:  WR {avg_wr_no_sr:.1f}%, PnL ${total_pnl_no_sr:,.0f}")
    print(f"CON S/R:  WR {avg_wr_sr:.1f}%, PnL ${total_pnl_sr:,.0f}")
    print(f"\nDiferencia: WR {avg_wr_sr - avg_wr_no_sr:+.1f}%, PnL ${total_pnl_sr - total_pnl_no_sr:+,.0f}")

    if avg_wr_sr > avg_wr_no_sr and total_pnl_sr > total_pnl_no_sr:
        print("\n*** S/R MEJORA AMBAS METRICAS - IMPLEMENTAR ***")
    elif avg_wr_sr > avg_wr_no_sr:
        print("\n** S/R mejora WR pero reduce PnL - evaluar trade-off **")
    elif total_pnl_sr > total_pnl_no_sr:
        print("\n** S/R mejora PnL pero reduce WR - evaluar trade-off **")
    else:
        print("\n!! S/R no mejora - mantener sin S/R !!")


if __name__ == '__main__':
    main()
