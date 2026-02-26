"""
Backtest V12.4 - Detector de Regimen CALIBRADO
==============================================
Diferencia vs V12.3:
- V12.3: Cambio FILTROS por regimen (FALLO - detector muy conservador)
- V12.4: Calibra el DETECTOR para detectar mas BULL_TREND
         Y mantiene filtros V12 base para WEAK_TREND (no endurecer)

Cambios en detector:
- adx_trend_threshold: 20 -> 15 (detectar BULL_TREND mas facil)
- adx_weak_threshold: 15 -> 12
- chop_lateral: 62 -> 65 (menos LATERAL)
"""

import json
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
from enum import Enum

DATA_DIR = Path(__file__).parent / 'data'
MODELS_DIR = Path(__file__).parent / 'models'

INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.02


class MarketRegime(Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    WEAK_TREND = "weak_trend"
    LATERAL = "lateral"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    UNKNOWN = "unknown"


# Filtros por regimen - V12.4
# CLAVE: WEAK_TREND = V12 base (no endurecer)
# BULL_TREND = relajados para aprovechar tendencia
REGIME_FILTERS = {
    MarketRegime.BULL_TREND: {
        'min_conviction': 1.8,      # RELAJADO
        'rsi_range': (35, 75),
        'bb_range': (0.15, 0.85),
        'max_chop': 55,
        'tp_mult': 2.5,
        'sl_mult': 1.0,
        'max_hold': 35,
        'pos_mult': 1.0,
        'description': 'Tendencia alcista - relajar filtros',
    },
    MarketRegime.BEAR_TREND: {
        'min_conviction': 1.8,
        'rsi_range': (25, 65),
        'bb_range': (0.15, 0.85),
        'max_chop': 55,
        'tp_mult': 2.0,
        'sl_mult': 1.0,
        'max_hold': 25,
        'pos_mult': 0.8,
        'description': 'Tendencia bajista',
    },
    MarketRegime.WEAK_TREND: {
        # IGUAL QUE V12 BASE - no endurecer!
        'min_conviction': 2.0,
        'rsi_range': (38, 72),
        'bb_range': (0.2, 0.8),
        'max_chop': 52,
        'tp_mult': 2.0,
        'sl_mult': 1.0,
        'max_hold': 25,
        'pos_mult': 0.85,
        'description': 'Tendencia debil - V12 base',
    },
    MarketRegime.HIGH_VOL: {
        'min_conviction': 2.2,
        'rsi_range': (40, 60),
        'bb_range': (0.25, 0.75),
        'max_chop': 50,
        'tp_mult': 2.5,
        'sl_mult': 1.25,
        'max_hold': 15,
        'pos_mult': 0.5,
        'description': 'Alta volatilidad - conservador',
    },
    MarketRegime.LOW_VOL: {
        'min_conviction': 2.0,
        'rsi_range': (38, 72),
        'bb_range': (0.2, 0.8),
        'max_chop': 52,
        'tp_mult': 1.5,
        'sl_mult': 0.8,
        'max_hold': 30,
        'pos_mult': 1.0,
        'description': 'Baja volatilidad',
    },
}

ACTIVE_PAIRS = ['ADA/USDT', 'XRP/USDT', 'DOT/USDT', 'DOGE/USDT',
                'ETH/USDT', 'AVAX/USDT', 'NEAR/USDT', 'LINK/USDT']


class CalibratedRegimeDetector:
    """
    Detector de regimen CALIBRADO para detectar mas BULL_TREND.

    Cambios vs V2:
    - adx_trend_threshold: 20 -> 15
    - adx_weak_threshold: 15 -> 12
    - chop_lateral: 62 -> 65
    """

    def __init__(
        self,
        adx_trend_threshold: float = 15,    # CALIBRADO: antes 20
        adx_weak_threshold: float = 12,     # CALIBRADO: antes 15
        chop_lateral_threshold: float = 65, # CALIBRADO: antes 62
        atr_high_mult: float = 1.5,
        atr_low_mult: float = 0.7,
    ):
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_weak_threshold = adx_weak_threshold
        self.chop_lateral_threshold = chop_lateral_threshold
        self.atr_high_mult = atr_high_mult
        self.atr_low_mult = atr_low_mult

    def detect_regime_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta regimen para todo el DataFrame."""
        c, h, l = df['close'], df['high'], df['low']

        # EMAs
        ema8 = ta.ema(c, length=8)
        ema21 = ta.ema(c, length=21)
        ema55 = ta.ema(c, length=55)

        # ADX
        adx_data = ta.adx(h, l, c, length=14)
        adx = adx_data.iloc[:, 0] if adx_data is not None else pd.Series(20, index=df.index)
        dmi_plus = adx_data.iloc[:, 1] if adx_data is not None else pd.Series(25, index=df.index)
        dmi_minus = adx_data.iloc[:, 2] if adx_data is not None else pd.Series(25, index=df.index)

        # Chop
        atr_1 = ta.atr(h, l, c, length=1)
        atr_sum = atr_1.rolling(14).sum()
        high_max = h.rolling(14).max()
        low_min = l.rolling(14).min()
        chop = 100 * np.log10(atr_sum / (high_max - low_min + 1e-10)) / np.log10(14)

        # ATR ratio
        atr14 = ta.atr(h, l, c, length=14)
        atr_avg = atr14.rolling(50).mean()
        atr_ratio = atr14 / atr_avg

        regimes = []
        reasons = []

        for i in range(len(df)):
            a = adx.iloc[i] if not pd.isna(adx.iloc[i]) else 20
            ch = chop.iloc[i] if not pd.isna(chop.iloc[i]) else 50
            ar = atr_ratio.iloc[i] if not pd.isna(atr_ratio.iloc[i]) else 1.0
            dp = dmi_plus.iloc[i] if not pd.isna(dmi_plus.iloc[i]) else 25
            dm = dmi_minus.iloc[i] if not pd.isna(dmi_minus.iloc[i]) else 25
            e8 = ema8.iloc[i] if not pd.isna(ema8.iloc[i]) else c.iloc[i]
            e21 = ema21.iloc[i] if not pd.isna(ema21.iloc[i]) else c.iloc[i]
            e55 = ema55.iloc[i] if not pd.isna(ema55.iloc[i]) else c.iloc[i]

            # LATERAL: muy choppy
            if ch > self.chop_lateral_threshold and a < self.adx_weak_threshold:
                regimes.append('lateral')
                reasons.append('extreme_chop')
                continue

            # HIGH_VOL
            if ar > self.atr_high_mult:
                regimes.append('high_vol')
                reasons.append('high_volatility')
                continue

            # LOW_VOL
            if ar < self.atr_low_mult:
                regimes.append('low_vol')
                reasons.append('low_volatility')
                continue

            # CALIBRADO: BULL_TREND mas facil (ADX > 15 en vez de 20)
            emas_bull = (e8 > e21) or (e21 > e55)  # Solo 2 de 3
            if a > self.adx_trend_threshold and dp > dm and emas_bull:
                regimes.append('bull_trend')
                reasons.append('uptrend')
                continue

            # BEAR_TREND
            emas_bear = (e8 < e21) or (e21 < e55)
            if a > self.adx_trend_threshold and dm > dp and emas_bear:
                regimes.append('bear_trend')
                reasons.append('downtrend')
                continue

            # Default: WEAK_TREND
            regimes.append('weak_trend')
            reasons.append('unclear')

        return pd.DataFrame({
            'regime': regimes,
            'regime_reason': reasons,
            'adx': adx.values,
            'chop': chop.values,
            'atr14': atr14.values,
        }, index=df.index)


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
    """Calcular features igual que V12.3 para compatibilidad con modelo."""
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


def run_backtest_v12_4(pair_data, models, start_date, end_date, detector):
    """Backtest V12.4 con detector calibrado."""
    trades = []
    regime_stats = {r.value: {'trades': 0, 'wins': 0, 'pnl': 0} for r in REGIME_FILTERS.keys()}

    for pair in ACTIVE_PAIRS:
        if pair not in pair_data:
            continue

        df = pair_data[pair]
        df_period = df[(df.index >= start_date) & (df.index < end_date)]
        if len(df_period) < 250:
            continue

        safe = pair.replace('/', '')
        model = models.get(safe)
        if model is None:
            continue

        feat = compute_features(df)
        feat = feat[(feat.index >= start_date) & (feat.index < end_date)]
        regimes = detector.detect_regime_series(df)
        regimes = regimes[(regimes.index >= start_date) & (regimes.index < end_date)]
        fcols = [c for c in model.feature_name_ if c in feat.columns]

        balance = INITIAL_CAPITAL
        pos = None

        for i in range(250, len(df_period)):
            ts = df_period.index[i]
            if ts not in feat.index:
                continue
            price = df_period.iloc[i]['close']

            if pos is not None:
                pnl_pct = (price - pos['entry']) / pos['entry'] * pos['dir']
                hit_tp = pnl_pct >= pos['tp_pct']
                hit_sl = pnl_pct <= -pos['sl_pct']
                timeout = (i - pos['bar']) >= pos.get('max_hold', 20)

                if hit_tp or hit_sl or timeout:
                    pnl = pos['size'] * pos['dir'] * (price - pos['entry'])
                    balance += pnl

                    trade = {'pair': pair, 'pnl': pnl, 'regime': pos['regime']}
                    trades.append(trade)

                    r = pos['regime']
                    if r in regime_stats:
                        regime_stats[r]['trades'] += 1
                        regime_stats[r]['pnl'] += pnl
                        if pnl > 0:
                            regime_stats[r]['wins'] += 1

                    pos = None

            if pos is None:
                if ts not in regimes.index:
                    continue
                regime_str = regimes.loc[ts, 'regime']
                try:
                    regime = MarketRegime(regime_str)
                except:
                    continue

                if regime not in REGIME_FILTERS:
                    continue
                if regime == MarketRegime.LATERAL:
                    continue

                rf = REGIME_FILTERS[regime]
                min_conv = rf['min_conviction']
                rsi_min, rsi_max = rf['rsi_range']
                bb_min, bb_max = rf['bb_range']
                max_chop = rf['max_chop']
                tp_mult = rf['tp_mult']
                sl_mult = rf['sl_mult']
                max_hold = rf['max_hold']
                pos_mult = rf['pos_mult']

                rsi = feat.loc[ts, 'rsi14'] if 'rsi14' in feat.columns else 50
                bb_pos_val = feat.loc[ts, 'bb_pos'] if 'bb_pos' in feat.columns else 0.5
                chop = feat.loc[ts, 'chop'] if 'chop' in feat.columns else 50
                atr = feat.loc[ts, 'atr14'] if 'atr14' in feat.columns else price * 0.02

                X = feat.loc[ts:ts][fcols]
                if X.isna().any().any():
                    continue

                pred = model.predict(X)[0]
                sig = 1 if pred > 0.5 else -1
                conviction = abs(pred - 0.5) * 10

                # Filtros por regimen
                if conviction < min_conv:
                    continue
                if not (rsi_min <= rsi <= rsi_max):
                    continue
                if not (bb_min <= bb_pos_val <= bb_max):
                    continue
                if chop > max_chop:
                    continue

                # Solo LONG en BULL, solo SHORT en BEAR
                if regime == MarketRegime.BULL_TREND and sig != 1:
                    continue
                if regime == MarketRegime.BEAR_TREND and sig != -1:
                    continue

                tp_pct = atr / price * tp_mult
                sl_pct = atr / price * sl_mult
                risk_amt = balance * RISK_PER_TRADE * pos_mult
                size = risk_amt / (sl_pct * price) if sl_pct > 0 else 0

                pos = {
                    'entry': price, 'dir': sig, 'size': size,
                    'tp_pct': tp_pct, 'sl_pct': sl_pct, 'bar': i,
                    'max_hold': max_hold, 'regime': regime.value
                }

        if pos:
            pnl = pos['size'] * pos['dir'] * (df_period.iloc[-1]['close'] - pos['entry'])
            trade = {'pair': pair, 'pnl': pnl, 'regime': pos['regime']}
            trades.append(trade)
            r = pos['regime']
            if r in regime_stats:
                regime_stats[r]['trades'] += 1
                regime_stats[r]['pnl'] += pnl
                if pnl > 0:
                    regime_stats[r]['wins'] += 1

    return trades, regime_stats


def run_backtest_v12_base(pair_data, models, start_date, end_date, detector):
    """Backtest V12 base con filtros universales."""
    trades = []

    min_conv = 2.0
    rsi_min, rsi_max = 38, 72
    bb_min, bb_max = 0.2, 0.8
    max_chop = 52

    for pair in ACTIVE_PAIRS:
        if pair not in pair_data:
            continue

        df = pair_data[pair]
        df_period = df[(df.index >= start_date) & (df.index < end_date)]
        if len(df_period) < 250:
            continue

        safe = pair.replace('/', '')
        model = models.get(safe)
        if model is None:
            continue

        feat = compute_features(df)
        feat = feat[(feat.index >= start_date) & (feat.index < end_date)]
        regimes = detector.detect_regime_series(df)
        regimes = regimes[(regimes.index >= start_date) & (regimes.index < end_date)]
        fcols = [c for c in model.feature_name_ if c in feat.columns]

        balance = INITIAL_CAPITAL
        pos = None

        for i in range(250, len(df_period)):
            ts = df_period.index[i]
            if ts not in feat.index:
                continue
            price = df_period.iloc[i]['close']

            if pos is not None:
                pnl_pct = (price - pos['entry']) / pos['entry'] * pos['dir']
                hit_tp = pnl_pct >= pos['tp_pct']
                hit_sl = pnl_pct <= -pos['sl_pct']
                timeout = (i - pos['bar']) >= 20

                if hit_tp or hit_sl or timeout:
                    pnl = pos['size'] * pos['dir'] * (price - pos['entry'])
                    balance += pnl
                    trades.append({'pair': pair, 'pnl': pnl})
                    pos = None

            if pos is None:
                if ts not in regimes.index:
                    continue
                regime_str = regimes.loc[ts, 'regime']
                if regime_str == 'lateral':
                    continue

                rsi = feat.loc[ts, 'rsi14'] if 'rsi14' in feat.columns else 50
                bb_pos_val = feat.loc[ts, 'bb_pos'] if 'bb_pos' in feat.columns else 0.5
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
                if not (bb_min <= bb_pos_val <= bb_max):
                    continue
                if chop > max_chop:
                    continue

                tp_pct = atr / price * 2.0
                sl_pct = atr / price * 1.0
                risk_amt = balance * RISK_PER_TRADE
                size = risk_amt / (sl_pct * price) if sl_pct > 0 else 0

                pos = {'entry': price, 'dir': sig, 'size': size,
                       'tp_pct': tp_pct, 'sl_pct': sl_pct, 'bar': i}

        if pos:
            pnl = pos['size'] * pos['dir'] * (df_period.iloc[-1]['close'] - pos['entry'])
            trades.append({'pair': pair, 'pnl': pnl})

    return trades


def compute_metrics(trades):
    if not trades:
        return {'trades': 0, 'wins': 0, 'wr': 0, 'pnl': 0, 'pf': 0}
    wins = sum(1 for t in trades if t['pnl'] > 0)
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99
    return {
        'trades': len(trades),
        'wins': wins,
        'wr': wins / len(trades) * 100 if trades else 0,
        'pnl': sum(t['pnl'] for t in trades),
        'pf': pf,
    }


def main():
    print("="*70)
    print("BACKTEST V12.4 - DETECTOR DE REGIMEN CALIBRADO")
    print("="*70)

    print("\nDiferencia vs V12.3:")
    print("- V12.3: Cambio filtros (FALLO - detector detectaba WEAK_TREND 97%)")
    print("- V12.4: Calibra DETECTOR para detectar mas BULL_TREND")
    print("         WEAK_TREND usa filtros V12 base (no endurecer)")

    print("\nCalibracion del detector:")
    print("- adx_trend_threshold: 20 -> 15")
    print("- adx_weak_threshold: 15 -> 12")
    print("- chop_lateral: 62 -> 65")
    print("- EMAs: solo 2 de 3 alineadas")

    print("\nFiltros por regimen:")
    for regime, rf in REGIME_FILTERS.items():
        print(f"\n  {regime.value}:")
        print(f"    Conviction >= {rf['min_conviction']}")
        print(f"    RSI: {rf['rsi_range']}")
        print(f"    Chop < {rf['max_chop']}")
        print(f"    -> {rf['description']}")

    # Cargar modelos
    models = {}
    pair_data = {}
    for pair in ACTIVE_PAIRS:
        safe = pair.replace('/', '')
        try:
            models[safe] = joblib.load(MODELS_DIR / f'v95_v7_{safe}USDT.pkl')
        except:
            try:
                models[safe] = joblib.load(MODELS_DIR / f'v7_{pair.replace("/", "_")}.pkl')
            except:
                continue
        df = load_data(pair)
        if df is not None:
            pair_data[pair] = df

    print(f"\nModelos cargados: {len(models)}")
    print(f"Pares con datos: {len(pair_data)}")

    # Detectores
    detector_calibrated = CalibratedRegimeDetector()  # V12.4
    from regime_detector_v2 import RegimeDetector
    detector_original = RegimeDetector()  # V12 base

    periods = [
        ('Ultimo Ano', '2024-02-01', '2025-02-24'),
        ('Bear Market 2022', '2022-01-01', '2023-01-01'),
    ]

    all_v12 = []
    all_v12_4 = []

    for period_name, start, end in periods:
        print(f"\n{'='*70}")
        print(f"PERIODO: {period_name}")
        print("="*70)

        # V12 Base
        trades_v12 = run_backtest_v12_base(pair_data, models, start, end, detector_original)
        m_v12 = compute_metrics(trades_v12)
        all_v12.append(m_v12)

        # V12.4 Calibrado
        trades_v12_4, regime_stats = run_backtest_v12_4(
            pair_data, models, start, end, detector_calibrated
        )
        m_v12_4 = compute_metrics(trades_v12_4)
        all_v12_4.append(m_v12_4)

        wr_diff = m_v12_4['wr'] - m_v12['wr']
        wr_mark = ">> MEJOR" if wr_diff > 1 else ("<< PEOR" if wr_diff < -1 else "~igual")

        print(f"\n{'Metrica':<12} {'V12 Base':<15} {'V12.4 Calib':<15} {'Diferencia':<12}")
        print("-"*55)
        print(f"{'Trades':<12} {m_v12['trades']:<15} {m_v12_4['trades']:<15} {m_v12_4['trades'] - m_v12['trades']:+d}")
        print(f"{'Win Rate':<12} {m_v12['wr']:<14.1f}% {m_v12_4['wr']:<14.1f}% {wr_diff:+.1f}% {wr_mark}")
        print(f"{'PnL':<12} ${m_v12['pnl']:<13,.0f} ${m_v12_4['pnl']:<13,.0f} ${m_v12_4['pnl'] - m_v12['pnl']:+,.0f}")
        print(f"{'P. Factor':<12} {m_v12['pf']:<15.2f} {m_v12_4['pf']:<15.2f} {m_v12_4['pf'] - m_v12['pf']:+.2f}")

        # Desglose por regimen
        print(f"\nDesglose V12.4 por regimen:")
        for regime_val, stats in regime_stats.items():
            if stats['trades'] > 0:
                wr = stats['wins'] / stats['trades'] * 100
                wr_flag = "**" if wr >= 50 else ""
                print(f"  {regime_val:<15} {stats['trades']:>4} trades, WR {wr:>5.1f}%{wr_flag}, PnL ${stats['pnl']:>8,.0f}")

    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)

    avg_wr_v12 = np.mean([r['wr'] for r in all_v12])
    avg_wr_v12_4 = np.mean([r['wr'] for r in all_v12_4])
    total_pnl_v12 = sum(r['pnl'] for r in all_v12)
    total_pnl_v12_4 = sum(r['pnl'] for r in all_v12_4)
    total_trades_v12 = sum(r['trades'] for r in all_v12)
    total_trades_v12_4 = sum(r['trades'] for r in all_v12_4)

    print(f"\n{'Version':<20} {'Trades':<10} {'WR':<12} {'PnL':<12}")
    print("-"*55)
    print(f"{'V12 Base':<20} {total_trades_v12:<10} {avg_wr_v12:<11.1f}% ${total_pnl_v12:<10,.0f}")
    print(f"{'V12.4 Calibrado':<20} {total_trades_v12_4:<10} {avg_wr_v12_4:<11.1f}% ${total_pnl_v12_4:<10,.0f}")
    print("-"*55)
    print(f"{'Diferencia':<20} {total_trades_v12_4 - total_trades_v12:+d}        {avg_wr_v12_4 - avg_wr_v12:+.1f}%       ${total_pnl_v12_4 - total_pnl_v12:+,.0f}")

    # Veredicto
    if avg_wr_v12_4 > avg_wr_v12 and total_pnl_v12_4 > total_pnl_v12:
        print("\n*** V12.4 MEJORA AMBAS METRICAS ***")
        result = "EXITO"
    elif avg_wr_v12_4 > avg_wr_v12:
        print("\n** V12.4 mejora WR **")
        result = "MEJORA_WR"
    elif total_pnl_v12_4 > total_pnl_v12:
        print("\n** V12.4 mejora PnL **")
        result = "MEJORA_PNL"
    else:
        print("\n!! V12.4 no mejora !!")
        result = "SIN_MEJORA"

    # Guardar config
    config = {
        'version': 'V12.4',
        'description': 'Detector de regimen calibrado',
        'result': result,
        'detector_params': {
            'adx_trend_threshold': 15,
            'adx_weak_threshold': 12,
            'chop_lateral_threshold': 65,
        },
        'regime_filters': {k.value: v for k, v in REGIME_FILTERS.items()},
        'metrics': {
            'v12_wr': avg_wr_v12,
            'v12_pnl': total_pnl_v12,
            'v12_4_wr': avg_wr_v12_4,
            'v12_4_pnl': total_pnl_v12_4,
        }
    }

    config_path = MODELS_DIR / 'v12_4_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)

    print(f"\nConfiguracion guardada en {config_path.name}")


if __name__ == '__main__':
    main()
