"""
Backtest V12.2 - Exclusion + Especializacion Combinadas
========================================================
Combina lo mejor de V12 (exclusion de pares malos) y V12.1 (params por TIER).

V12.2 = 8 pares buenos + parametros TIER especializados
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

# Solo pares buenos (excluimos BTC, SOL, BNB como en V12)
# Con parametros especializados por TIER (como V12.1)
PAIR_PARAMS_V12_2 = {
    # TIER 1: Alto WR historico - Parametros agresivos
    'ADA/USDT': {
        'min_conviction': 1.8,
        'rsi_range': (35, 75),
        'bb_range': (0.15, 0.85),
        'max_chop': 55,
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
        'tp_mult': 2.5,
        'sl_mult': 1.0,
        'tier': 1,
    },

    # TIER 2: WR medio - Parametros balanceados
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
}

# Pares excluidos (TIER 3 - bajo rendimiento)
EXCLUDED_PAIRS = ['BTC/USDT', 'SOL/USDT', 'BNB/USDT']


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


def run_backtest_v12_2(pair_data, models, start_date, end_date, regime_detector):
    """Backtest V12.2 con exclusion + especializacion."""
    trades = []
    pair_metrics = {}

    for pair, df in pair_data.items():
        if pair not in PAIR_PARAMS_V12_2:
            continue  # Skip excluded pairs

        params = PAIR_PARAMS_V12_2[pair]

        df_period = df[(df.index >= start_date) & (df.index < end_date)]
        if len(df_period) < 250:
            continue

        safe = pair.replace('/', '')
        model = models.get(safe)
        if model is None:
            continue

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
                    trade = {'pair': pair, 'pnl': pnl, 'tier': params['tier']}
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

                # Filtros especializados por par
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
            trade = {'pair': pair, 'pnl': pnl, 'tier': params['tier']}
            trades.append(trade)
            pair_trades.append(trade)

        # Guardar metricas por par
        if pair_trades:
            wins = len([t for t in pair_trades if t['pnl'] > 0])
            pair_metrics[pair] = {
                'trades': len(pair_trades),
                'wins': wins,
                'wr': wins / len(pair_trades) * 100,
                'pnl': sum(t['pnl'] for t in pair_trades),
                'tier': params['tier'],
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
    print("BACKTEST V12.2 - EXCLUSION + ESPECIALIZACION COMBINADAS")
    print("="*70)

    print("\nConfiguracion:")
    print(f"  Pares activos: {len(PAIR_PARAMS_V12_2)} (TIER 1 + TIER 2)")
    print(f"  Pares excluidos: {EXCLUDED_PAIRS}")
    print("\n  TIER 1 (agresivo): ADA, XRP, DOT, DOGE")
    print("  TIER 2 (balanceado): ETH, AVAX, NEAR, LINK")

    models = {}
    pair_data = {}
    for pair in PAIR_PARAMS_V12_2.keys():
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

    all_results = []

    for period_name, start, end in periods:
        print(f"\n{'='*70}")
        print(f"PERIODO: {period_name}")
        print(f"{'='*70}")

        trades, max_dd, pair_metrics = run_backtest_v12_2(
            pair_data, models, start, end, detector
        )
        m = compute_metrics(trades)
        all_results.append(m)

        wr_mark = " >> OBJETIVO 50%+" if m['wr'] >= 50 else ""
        print(f"\nResultados V12.2:")
        print(f"  Trades: {m['trades']}")
        print(f"  Win Rate: {m['wr']:.1f}%{wr_mark}")
        print(f"  PnL: ${m['pnl']:,.0f}")
        print(f"  Profit Factor: {m['pf']:.2f}")
        print(f"  Max DD: {max_dd:.1f}%")

        print(f"\nDesglose por par:")
        for p, pm in sorted(pair_metrics.items(), key=lambda x: -x[1]['pnl']):
            tier_mark = f"T{pm['tier']}"
            wr_flag = "**" if pm['wr'] >= 50 else ""
            print(f"  {p:<12} {tier_mark:<3} {pm['trades']:>3} trades, WR {pm['wr']:>5.1f}%{wr_flag}, PnL ${pm['pnl']:>8,.0f}")

        # Por TIER
        tier1_trades = [t for t in trades if t['tier'] == 1]
        tier2_trades = [t for t in trades if t['tier'] == 2]

        if tier1_trades:
            t1_wins = len([t for t in tier1_trades if t['pnl'] > 0])
            t1_wr = t1_wins / len(tier1_trades) * 100
            t1_pnl = sum(t['pnl'] for t in tier1_trades)
            print(f"\n  TIER 1 total: {len(tier1_trades)} trades, WR {t1_wr:.1f}%, PnL ${t1_pnl:,.0f}")

        if tier2_trades:
            t2_wins = len([t for t in tier2_trades if t['pnl'] > 0])
            t2_wr = t2_wins / len(tier2_trades) * 100
            t2_pnl = sum(t['pnl'] for t in tier2_trades)
            print(f"  TIER 2 total: {len(tier2_trades)} trades, WR {t2_wr:.1f}%, PnL ${t2_pnl:,.0f}")

    # Resumen final
    print("\n" + "="*70)
    print("COMPARACION FINAL: V12 vs V12.1 vs V12.2")
    print("="*70)

    avg_wr = np.mean([r['wr'] for r in all_results])
    total_pnl = sum(r['pnl'] for r in all_results)
    total_trades = sum(r['trades'] for r in all_results)

    print(f"\n{'Version':<15} {'Pares':<8} {'Trades':<10} {'WR':<10} {'PnL':<12}")
    print("-"*55)
    print(f"{'V12 Base':<15} {'8':<8} {'~1,400':<10} {'50.4%':<10} {'$2,426':<12}")
    print(f"{'V12.1 Spec':<15} {'11':<8} {'1,753':<10} {'46.7%':<10} {'$2,904':<12}")
    print(f"{'V12.2 Combined':<15} {'8':<8} {total_trades:<10} {f'{avg_wr:.1f}%':<10} {f'${total_pnl:,.0f}':<12}")

    # Determinar ganador
    print("\n" + "-"*55)
    if avg_wr >= 50 and total_pnl > 2426:
        print("*** V12.2 GANA: Mejor WR y PnL que V12 ***")
        result = "EXITO_TOTAL"
    elif avg_wr >= 50:
        print("** V12.2 mantiene WR 50%+ **")
        result = "EXITO_WR"
    elif total_pnl > 2426:
        print("** V12.2 mejora PnL pero pierde WR **")
        result = "EXITO_PNL"
    else:
        print("!! V12.2 no supera a V12 !!")
        result = "FALLO"

    # Guardar configuracion
    config = {
        'version': 'V12.2',
        'description': 'Exclusion de pares malos + Especializacion por TIER',
        'result': result,
        'excluded_pairs': EXCLUDED_PAIRS,
        'metrics': {
            'avg_wr': avg_wr,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
        },
        'comparison': {
            'v12_wr': 50.4,
            'v12_pnl': 2426,
            'v12_1_wr': 46.7,
            'v12_1_pnl': 2904,
        },
        'pair_params': {k: v for k, v in PAIR_PARAMS_V12_2.items()},
    }

    with open(MODELS_DIR / 'v12_2_config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    print(f"\nConfiguracion guardada en models/v12_2_config.json")


if __name__ == '__main__':
    main()
