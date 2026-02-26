"""
Backtest V12.1 - Parametros Especializados por Par
===================================================
Cada par tiene sus propios parametros basados en su comportamiento.
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

# Parametros especializados por par basados en analisis previo
# Pares con alto WR historico -> parametros mas agresivos
# Pares con bajo WR -> parametros mas conservadores
PAIR_PARAMS = {
    # TIER 1: Alto WR historico (>50%) - Parametros agresivos
    'ADA/USDT': {
        'min_conviction': 1.8,   # Menor (mas trades, WR alto)
        'rsi_range': (35, 75),   # Mas amplio
        'bb_range': (0.15, 0.85),
        'max_chop': 55,          # Mas tolerante
        'tp_mult': 2.0,
        'sl_mult': 1.0,
        'tier': 1,
    },
    'XRP/USDT': {
        'min_conviction': 1.8,
        'rsi_range': (35, 75),
        'bb_range': (0.15, 0.85),
        'max_chop': 55,
        'tp_mult': 2.0,
        'sl_mult': 1.0,
        'tier': 1,
    },
    'DOT/USDT': {
        'min_conviction': 1.8,
        'rsi_range': (35, 75),
        'bb_range': (0.15, 0.85),
        'max_chop': 55,
        'tp_mult': 2.0,
        'sl_mult': 1.0,
        'tier': 1,
    },
    'DOGE/USDT': {
        'min_conviction': 1.8,
        'rsi_range': (35, 75),
        'bb_range': (0.15, 0.85),
        'max_chop': 55,
        'tp_mult': 2.5,          # Mayor TP (alta vol)
        'sl_mult': 1.0,
        'tier': 1,
    },

    # TIER 2: WR medio (45-50%) - Parametros balanceados
    'ETH/USDT': {
        'min_conviction': 2.0,
        'rsi_range': (38, 72),
        'bb_range': (0.2, 0.8),
        'max_chop': 52,
        'tp_mult': 1.8,
        'sl_mult': 1.0,
        'tier': 2,
    },
    'AVAX/USDT': {
        'min_conviction': 2.0,
        'rsi_range': (38, 72),
        'bb_range': (0.2, 0.8),
        'max_chop': 52,
        'tp_mult': 2.0,
        'sl_mult': 1.0,
        'tier': 2,
    },
    'NEAR/USDT': {
        'min_conviction': 2.0,
        'rsi_range': (38, 72),
        'bb_range': (0.2, 0.8),
        'max_chop': 52,
        'tp_mult': 2.0,
        'sl_mult': 1.0,
        'tier': 2,
    },
    'LINK/USDT': {
        'min_conviction': 2.0,
        'rsi_range': (38, 72),
        'bb_range': (0.2, 0.8),
        'max_chop': 52,
        'tp_mult': 1.8,
        'sl_mult': 1.0,
        'tier': 2,
    },

    # TIER 3: WR bajo (<45%) - Parametros muy conservadores
    'BTC/USDT': {
        'min_conviction': 2.5,   # Mayor (menos trades, mas selectivo)
        'rsi_range': (42, 68),   # Mas estrecho
        'bb_range': (0.25, 0.75),
        'max_chop': 48,          # Menos tolerante
        'tp_mult': 1.5,          # TP mas cercano
        'sl_mult': 1.0,
        'tier': 3,
    },
    'SOL/USDT': {
        'min_conviction': 2.5,
        'rsi_range': (42, 68),
        'bb_range': (0.25, 0.75),
        'max_chop': 48,
        'tp_mult': 1.5,
        'sl_mult': 1.0,
        'tier': 3,
    },
    'BNB/USDT': {
        'min_conviction': 2.5,
        'rsi_range': (42, 68),
        'bb_range': (0.25, 0.75),
        'max_chop': 48,
        'tp_mult': 1.5,
        'sl_mult': 1.0,
        'tier': 3,
    },
}

# Parametros universales V12 (para comparacion)
BASE_PARAMS = {
    'min_conviction': 2.0,
    'rsi_range': (38, 72),
    'bb_range': (0.2, 0.8),
    'max_chop': 52,
    'tp_mult': 2.0,
    'sl_mult': 1.0,
}


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


def run_backtest(pair_data, models, start_date, end_date, regime_detector, use_specialized=True):
    """Backtest con o sin parametros especializados por par."""
    trades = []

    for pair, df in pair_data.items():
        df_period = df[(df.index >= start_date) & (df.index < end_date)]
        if len(df_period) < 250:
            continue

        safe = pair.replace('/', '')
        model = models.get(safe)
        if model is None:
            continue

        # Seleccionar parametros
        if use_specialized and pair in PAIR_PARAMS:
            params = PAIR_PARAMS[pair]
        else:
            params = BASE_PARAMS

        feat = compute_features(df)
        feat = feat[(feat.index >= start_date) & (feat.index < end_date)]
        regimes = regime_detector.detect_regime_series(df)
        regimes = regimes[(regimes.index >= start_date) & (regimes.index < end_date)]
        fcols = [c for c in model.feature_name_ if c in feat.columns]

        min_conv = params['min_conviction']
        rsi_min, rsi_max = params['rsi_range']
        bb_min, bb_max = params['bb_range']
        max_chop = params['max_chop']
        tp_mult = params['tp_mult']
        sl_mult = params['sl_mult']

        balance = INITIAL_CAPITAL
        peak = balance
        max_dd = 0
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
                    peak = max(peak, balance)
                    dd = (peak - balance) / peak * 100
                    max_dd = max(max_dd, dd)
                    trades.append({'pair': pair, 'pnl': pnl, 'tier': params.get('tier', 0)})
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

                # Filtros con parametros del par
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
            trades.append({'pair': pair, 'pnl': pnl, 'tier': params.get('tier', 0)})

    return trades, max_dd


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
    print("BACKTEST V12.1 - PARAMETROS ESPECIALIZADOS POR PAR")
    print("="*70)

    print("\nTIERS de pares:")
    print("  TIER 1 (agresivo): ADA, XRP, DOT, DOGE - WR historico >50%")
    print("  TIER 2 (balanceado): ETH, AVAX, NEAR, LINK - WR 45-50%")
    print("  TIER 3 (conservador): BTC, SOL, BNB - WR <45%")

    models = {}
    pair_data = {}
    for pair in PAIR_PARAMS.keys():
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

    all_base = []
    all_specialized = []

    for period_name, start, end in periods:
        print(f"\n{'='*70}")
        print(f"PERIODO: {period_name}")
        print(f"{'='*70}")

        # V12 Base (sin especializacion)
        trades_base, _ = run_backtest(pair_data, models, start, end, detector, use_specialized=False)
        m_base = compute_metrics(trades_base)

        # V12.1 Especializado
        trades_spec, _ = run_backtest(pair_data, models, start, end, detector, use_specialized=True)
        m_spec = compute_metrics(trades_spec)

        all_base.append(m_base)
        all_specialized.append(m_spec)

        wr_diff = m_spec['wr'] - m_base['wr']
        wr_mark = ">> MEJOR" if wr_diff > 0.5 else ("<< PEOR" if wr_diff < -0.5 else "~igual")

        print(f"\n{'Metrica':<12} {'V12 Base':<15} {'V12.1 Spec':<15} {'Diferencia':<15}")
        print("-"*57)
        print(f"{'Trades':<12} {m_base['trades']:<15} {m_spec['trades']:<15} {m_spec['trades'] - m_base['trades']:+d}")
        print(f"{'Win Rate':<12} {m_base['wr']:<14.1f}% {m_spec['wr']:<14.1f}% {wr_diff:+.1f}% {wr_mark}")
        print(f"{'PnL':<12} ${m_base['pnl']:<13,.0f} ${m_spec['pnl']:<13,.0f} ${m_spec['pnl'] - m_base['pnl']:+,.0f}")
        print(f"{'P. Factor':<12} {m_base['pf']:<15.2f} {m_spec['pf']:<15.2f} {m_spec['pf'] - m_base['pf']:+.2f}")

        # Desglose por par
        print(f"\nDesglose por par (V12.1):")
        pair_stats = {}
        for t in trades_spec:
            p = t['pair']
            if p not in pair_stats:
                pair_stats[p] = {'wins': 0, 'losses': 0, 'pnl': 0, 'tier': t.get('tier', 0)}
            if t['pnl'] > 0:
                pair_stats[p]['wins'] += 1
            else:
                pair_stats[p]['losses'] += 1
            pair_stats[p]['pnl'] += t['pnl']

        for p, s in sorted(pair_stats.items(), key=lambda x: -x[1]['pnl']):
            total = s['wins'] + s['losses']
            wr = s['wins'] / total * 100 if total > 0 else 0
            tier_mark = f"T{s['tier']}" if s['tier'] else ""
            wr_flag = "**" if wr >= 50 else ""
            print(f"  {p:<12} {tier_mark:<3} {total:>3} trades, WR {wr:>5.1f}%{wr_flag}, PnL ${s['pnl']:>8,.0f}")

    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)

    avg_wr_base = np.mean([r['wr'] for r in all_base])
    avg_wr_spec = np.mean([r['wr'] for r in all_specialized])
    total_pnl_base = sum(r['pnl'] for r in all_base)
    total_pnl_spec = sum(r['pnl'] for r in all_specialized)
    total_trades_base = sum(r['trades'] for r in all_base)
    total_trades_spec = sum(r['trades'] for r in all_specialized)

    print(f"\nV12 Base:        {total_trades_base} trades, WR {avg_wr_base:.1f}%, PnL ${total_pnl_base:,.0f}")
    print(f"V12.1 Especial:  {total_trades_spec} trades, WR {avg_wr_spec:.1f}%, PnL ${total_pnl_spec:,.0f}")
    print(f"\nDiferencia: WR {avg_wr_spec - avg_wr_base:+.1f}%, PnL ${total_pnl_spec - total_pnl_base:+,.0f}")

    if avg_wr_spec > avg_wr_base and total_pnl_spec > total_pnl_base:
        print("\n*** V12.1 MEJORA AMBAS METRICAS ***")
        result = "EXITO"
    elif avg_wr_spec > avg_wr_base:
        print("\n** V12.1 mejora WR pero reduce PnL **")
        result = "PARCIAL"
    elif total_pnl_spec > total_pnl_base:
        print("\n** V12.1 mejora PnL pero reduce WR **")
        result = "PARCIAL"
    else:
        print("\n!! V12.1 no mejora - mantener V12 base !!")
        result = "FALLO"

    # Guardar configuracion
    config = {
        'version': 'V12.1',
        'description': 'Parametros especializados por par (3 tiers)',
        'result': result,
        'metrics': {
            'base_wr': avg_wr_base,
            'specialized_wr': avg_wr_spec,
            'base_pnl': total_pnl_base,
            'specialized_pnl': total_pnl_spec,
        },
        'pair_params': PAIR_PARAMS,
    }

    with open(MODELS_DIR / 'v12_1_config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    print(f"\nConfiguracion guardada en models/v12_1_config.json")


if __name__ == '__main__':
    main()
