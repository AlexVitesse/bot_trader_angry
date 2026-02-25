"""
Backtest V12.2b - Filtros Universales + TP/SL Especializado
============================================================
Mantiene los filtros estrictos de V12 que funcionan.
Solo varia TP/SL segun volatilidad historica del par.
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

INITIAL_CAPITAL = 500.0
RISK_PER_TRADE = 0.02

# Filtros UNIVERSALES (los mismos de V12 que funcionan)
UNIVERSAL_FILTERS = {
    'min_conviction': 2.0,
    'rsi_range': (38, 72),
    'bb_range': (0.2, 0.8),
    'max_chop': 52,
}

# Solo TP/SL especializado por par (basado en volatilidad historica)
PAIR_TPSL = {
    # Alta volatilidad historica -> TP mas grande
    'DOGE/USDT': {'tp_mult': 2.8, 'sl_mult': 1.0},  # Muy volatil
    'SOL/USDT': {'tp_mult': 2.5, 'sl_mult': 1.0},   # Alta vol (excluido por WR)
    'AVAX/USDT': {'tp_mult': 2.3, 'sl_mult': 1.0},

    # Volatilidad media -> TP estandar
    'ADA/USDT': {'tp_mult': 2.0, 'sl_mult': 1.0},
    'XRP/USDT': {'tp_mult': 2.0, 'sl_mult': 1.0},
    'DOT/USDT': {'tp_mult': 2.0, 'sl_mult': 1.0},
    'NEAR/USDT': {'tp_mult': 2.0, 'sl_mult': 1.0},
    'LINK/USDT': {'tp_mult': 2.0, 'sl_mult': 1.0},

    # Baja volatilidad -> TP mas pequeño (tomar profits rapido)
    'ETH/USDT': {'tp_mult': 1.8, 'sl_mult': 1.0},
    'BTC/USDT': {'tp_mult': 1.5, 'sl_mult': 1.0},   # Excluido por WR
    'BNB/USDT': {'tp_mult': 1.5, 'sl_mult': 1.0},   # Excluido por WR
}

# Pares activos (excluidos BTC, SOL, BNB por WR bajo)
ACTIVE_PAIRS = ['ADA/USDT', 'XRP/USDT', 'DOT/USDT', 'DOGE/USDT',
                'ETH/USDT', 'AVAX/USDT', 'NEAR/USDT', 'LINK/USDT']


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


def run_backtest(pair_data, models, start_date, end_date, regime_detector, use_specialized_tpsl=True):
    """Backtest con filtros universales y TP/SL opcional especializado."""
    trades = []
    pair_metrics = {}

    min_conv = UNIVERSAL_FILTERS['min_conviction']
    rsi_min, rsi_max = UNIVERSAL_FILTERS['rsi_range']
    bb_min, bb_max = UNIVERSAL_FILTERS['bb_range']
    max_chop = UNIVERSAL_FILTERS['max_chop']

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

        # TP/SL del par
        if use_specialized_tpsl and pair in PAIR_TPSL:
            tp_mult = PAIR_TPSL[pair]['tp_mult']
            sl_mult = PAIR_TPSL[pair]['sl_mult']
        else:
            tp_mult = 2.0  # Default V12
            sl_mult = 1.0

        feat = compute_features(df)
        feat = feat[(feat.index >= start_date) & (feat.index < end_date)]
        regimes = regime_detector.detect_regime_series(df)
        regimes = regimes[(regimes.index >= start_date) & (regimes.index < end_date)]
        fcols = [c for c in model.feature_name_ if c in feat.columns]

        balance = INITIAL_CAPITAL
        peak = balance
        max_dd = 0
        pos = None
        pair_trades = []

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
                    peak = max(peak, balance)
                    dd = (peak - balance) / peak * 100
                    max_dd = max(max_dd, dd)
                    exit_reason = 'tp' if hit_tp else ('sl' if hit_sl else 'timeout')
                    trade = {'pair': pair, 'pnl': pnl, 'exit': exit_reason}
                    trades.append(trade)
                    pair_trades.append(trade)
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
                bb_pos_val = feat.loc[ts, 'bb_pos'] if 'bb_pos' in feat.columns else 0.5
                adx = feat.loc[ts, 'adx'] if 'adx' in feat.columns else 25
                chop = feat.loc[ts, 'chop'] if 'chop' in feat.columns else 50
                atr = feat.loc[ts, 'atr14']

                X = feat.loc[ts:ts][fcols]
                if X.isna().any().any():
                    continue

                pred = model.predict(X)[0]
                sig = 1 if pred > 0.5 else -1
                conviction = abs(pred - 0.5) * 10

                # FILTROS UNIVERSALES (igual que V12)
                if conviction < min_conv:
                    continue
                if not (rsi_min <= rsi <= rsi_max):
                    continue
                if not (bb_min <= bb_pos_val <= bb_max):
                    continue
                if chop > max_chop:
                    continue

                # Ajustes por regimen
                if regime == MarketRegime.BULL_TREND:
                    if sig != 1:
                        continue
                    max_hold = 25
                    pos_mult = 1.0
                elif regime == MarketRegime.WEAK_TREND:
                    if not (15 <= adx <= 40):
                        continue
                    max_hold = 15
                    pos_mult = 0.85
                elif regime == MarketRegime.HIGH_VOL:
                    if not (40 <= rsi <= 60):
                        continue
                    max_hold = 12
                    pos_mult = 0.5
                else:
                    continue

                tp_pct = atr / price * tp_mult
                sl_pct = atr / price * sl_mult
                risk_amt = balance * RISK_PER_TRADE * pos_mult
                size = risk_amt / (sl_pct * price) if sl_pct > 0 else 0

                pos = {'entry': price, 'dir': sig, 'size': size,
                       'tp_pct': tp_pct, 'sl_pct': sl_pct, 'bar': i, 'max_hold': max_hold}

        if pos:
            pnl = pos['size'] * pos['dir'] * (df_period.iloc[-1]['close'] - pos['entry'])
            trade = {'pair': pair, 'pnl': pnl, 'exit': 'eod'}
            trades.append(trade)
            pair_trades.append(trade)

        if pair_trades:
            wins = len([t for t in pair_trades if t['pnl'] > 0])
            pair_metrics[pair] = {
                'trades': len(pair_trades),
                'wins': wins,
                'wr': wins / len(pair_trades) * 100,
                'pnl': sum(t['pnl'] for t in pair_trades),
                'tp_mult': tp_mult,
            }

    return trades, max_dd, pair_metrics


def compute_metrics(trades):
    if not trades:
        return {'trades': 0, 'wins': 0, 'wr': 0, 'pnl': 0, 'pf': 0}
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    gross_profit = sum(t['pnl'] for t in wins)
    gross_loss = abs(sum(t['pnl'] for t in losses))
    return {
        'trades': len(trades),
        'wins': len(wins),
        'wr': len(wins) / len(trades) * 100,
        'pnl': sum(t['pnl'] for t in trades),
        'pf': gross_profit / gross_loss if gross_loss > 0 else 999,
    }


def main():
    print("="*70)
    print("BACKTEST V12.2b - FILTROS UNIVERSALES + TP/SL ESPECIALIZADO")
    print("="*70)

    print("\nEstrategia:")
    print("  - Filtros: UNIVERSALES (igual que V12 - funcionan bien)")
    print("  - TP/SL: ESPECIALIZADO por volatilidad del par")
    print("\nTP por par:")
    for pair in ACTIVE_PAIRS:
        tp = PAIR_TPSL.get(pair, {}).get('tp_mult', 2.0)
        print(f"    {pair}: TP {tp}x ATR")

    models = {}
    pair_data = {}
    for pair in ACTIVE_PAIRS:
        safe = pair.replace('/', '')
        try:
            models[safe] = joblib.load(MODELS_DIR / f'v95_v7_{safe}.pkl')
        except:
            continue
        df = load_data(pair)
        if df is not None:
            pair_data[pair] = df

    detector = RegimeDetector()

    periods = [
        ('Ultimo Ano', '2025-02-01', '2026-02-24'),
        ('Bear Market 2022', '2022-01-01', '2023-01-01'),
    ]

    print("\n" + "="*70)
    print("COMPARACION: V12 (TP fijo) vs V12.2b (TP especializado)")
    print("="*70)

    all_v12 = []
    all_v12_2b = []

    for period_name, start, end in periods:
        print(f"\n{'='*70}")
        print(f"PERIODO: {period_name}")
        print(f"{'='*70}")

        # V12 (TP fijo 2.0x para todos)
        trades_v12, dd_v12, pm_v12 = run_backtest(
            pair_data, models, start, end, detector, use_specialized_tpsl=False
        )
        m_v12 = compute_metrics(trades_v12)
        all_v12.append(m_v12)

        # V12.2b (TP especializado)
        trades_v12_2b, dd_v12_2b, pm_v12_2b = run_backtest(
            pair_data, models, start, end, detector, use_specialized_tpsl=True
        )
        m_v12_2b = compute_metrics(trades_v12_2b)
        all_v12_2b.append(m_v12_2b)

        wr_diff = m_v12_2b['wr'] - m_v12['wr']
        pnl_diff = m_v12_2b['pnl'] - m_v12['pnl']

        print(f"\n{'Metrica':<12} {'V12 (TP fijo)':<18} {'V12.2b (TP esp)':<18} {'Diferencia':<12}")
        print("-"*60)
        print(f"{'Trades':<12} {m_v12['trades']:<18} {m_v12_2b['trades']:<18} {m_v12_2b['trades'] - m_v12['trades']:+d}")
        print(f"{'Win Rate':<12} {m_v12['wr']:<17.1f}% {m_v12_2b['wr']:<17.1f}% {wr_diff:+.1f}%")
        print(f"{'PnL':<12} ${m_v12['pnl']:<16,.0f} ${m_v12_2b['pnl']:<16,.0f} ${pnl_diff:+,.0f}")
        print(f"{'P. Factor':<12} {m_v12['pf']:<18.2f} {m_v12_2b['pf']:<18.2f} {m_v12_2b['pf'] - m_v12['pf']:+.2f}")

        print(f"\nDesglose V12.2b por par:")
        for p, pm in sorted(pm_v12_2b.items(), key=lambda x: -x[1]['pnl']):
            wr_flag = "**" if pm['wr'] >= 50 else ""
            print(f"  {p:<12} TP{pm['tp_mult']:.1f}x {pm['trades']:>3} trades, WR {pm['wr']:>5.1f}%{wr_flag}, PnL ${pm['pnl']:>8,.0f}")

    # Resumen
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)

    avg_wr_v12 = np.mean([r['wr'] for r in all_v12])
    avg_wr_v12_2b = np.mean([r['wr'] for r in all_v12_2b])
    total_pnl_v12 = sum(r['pnl'] for r in all_v12)
    total_pnl_v12_2b = sum(r['pnl'] for r in all_v12_2b)

    print(f"\nV12 (TP fijo 2.0x):      WR {avg_wr_v12:.1f}%, PnL ${total_pnl_v12:,.0f}")
    print(f"V12.2b (TP especializado): WR {avg_wr_v12_2b:.1f}%, PnL ${total_pnl_v12_2b:,.0f}")
    print(f"\nDiferencia: WR {avg_wr_v12_2b - avg_wr_v12:+.1f}%, PnL ${total_pnl_v12_2b - total_pnl_v12:+,.0f}")

    if avg_wr_v12_2b >= 50 and total_pnl_v12_2b > total_pnl_v12:
        print("\n*** V12.2b GANA: Mantiene WR 50%+ y mejora PnL ***")
        result = "EXITO"
    elif total_pnl_v12_2b > total_pnl_v12:
        print("\n** V12.2b mejora PnL **")
        result = "MEJORA_PNL"
    else:
        print("\n!! V12.2b no mejora significativamente !!")
        result = "SIN_MEJORA"

    # Guardar
    config = {
        'version': 'V12.2b',
        'description': 'Filtros universales V12 + TP/SL especializado por volatilidad',
        'result': result,
        'filters': UNIVERSAL_FILTERS,
        'pair_tpsl': PAIR_TPSL,
        'active_pairs': ACTIVE_PAIRS,
        'metrics': {
            'v12_wr': avg_wr_v12,
            'v12_pnl': total_pnl_v12,
            'v12_2b_wr': avg_wr_v12_2b,
            'v12_2b_pnl': total_pnl_v12_2b,
        }
    }

    with open(MODELS_DIR / 'v12_2b_config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    print(f"\nConfiguracion guardada en models/v12_2b_config.json")


if __name__ == '__main__':
    main()
