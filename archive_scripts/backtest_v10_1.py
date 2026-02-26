"""
Backtest V10.1: Sistema Adaptativo con Detector de Regimen V2
=============================================================
Compara:
- V9.5+ATR (estrategia fija)
- V10.1 Adaptativo (detector V2 menos restrictivo)

V10.1 usa el nuevo WEAK_TREND que opera con posicion reducida
en lugar de no operar en mercados ambiguos.

Uso: poetry run python backtest_v10_1.py
"""

import json
import time
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from regime_detector_v2 import (
    RegimeDetector, MarketRegime, RegimeStrategy,
    get_strategy_for_regime, should_take_trade, REGIME_STRATEGIES
)

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT',
    'NEAR/USDT',
]

INITIAL_CAPITAL = 500.0
BASE_RISK_PER_TRADE = 0.02
MAX_POSITIONS = 3
MAX_HOLD = 30


def load_data(pair):
    """Carga datos OHLCV."""
    safe = pair.replace('/', '_')
    for suffix in ['_4h_v95.parquet', '_4h_history.parquet']:
        cache = DATA_DIR / f'{safe}{suffix}'
        if cache.exists():
            df = pd.read_parquet(cache)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            return df
    return None


def precompute_loss_features(df, btc_df):
    """Precomputa features para LossDetector."""
    c = df['close']
    rsi14 = ta.rsi(c, length=14) / 100.0
    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        bb_lower, bb_upper = bb.iloc[:, 0], bb.iloc[:, 2]
        bb_pct = (c - bb_lower) / (bb_upper - bb_lower + 1e-10)
    else:
        bb_pct = pd.Series(0.5, index=df.index)

    vol_ma = df['volume'].rolling(20).mean()
    vol_ratio = df['volume'] / vol_ma
    ret_5 = c.pct_change(5)
    ret_20 = c.pct_change(20)
    atr = ta.atr(df['high'], df['low'], c, length=14)
    atr_pct = atr / c

    btc_c = btc_df['close']
    btc_rsi = ta.rsi(btc_c, length=14) / 100.0
    btc_vol = btc_c.pct_change().rolling(20).std()
    btc_ret_5 = btc_c.pct_change(5)

    btc_rsi_aligned = btc_rsi.reindex(df.index, method='ffill')
    btc_vol_aligned = btc_vol.reindex(df.index, method='ffill')
    btc_ret_5_aligned = btc_ret_5.reindex(df.index, method='ffill')

    return pd.DataFrame({
        'ld_pair_rsi14': rsi14,
        'ld_pair_bb_pct': bb_pct,
        'ld_pair_vol_ratio': vol_ratio,
        'ld_pair_ret_5': ret_5,
        'ld_pair_ret_20': ret_20,
        'ld_btc_ret_5': btc_ret_5_aligned,
        'ld_btc_rsi14': btc_rsi_aligned,
        'ld_btc_vol20': btc_vol_aligned,
        'atr_pct': atr_pct,
    }, index=df.index).fillna(0)


def compute_features(df):
    """Calcula features para V7 model."""
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

    # Choppiness
    atr_1 = ta.atr(h, l, c, length=1)
    atr_sum = atr_1.rolling(14).sum()
    high_max = h.rolling(14).max()
    low_min = l.rolling(14).min()
    feat['chop'] = 100 * np.log10(atr_sum / (high_max - low_min + 1e-10)) / np.log10(14)

    return feat


def run_backtest_v95_atr(
    pair_data, btc_df, models, ld_models, meta,
    start_date, end_date,
    conf_threshold=1.8, chop_threshold=50,
    atr_tp_mult=2.5, atr_sl_mult=1.0,
):
    """V9.5+ATR backtest (estrategia fija de referencia)."""
    trades = []
    for pair, df in pair_data.items():
        df = df[(df.index >= start_date) & (df.index < end_date)]
        if len(df) < 250:
            continue

        safe = pair.replace('/', '')
        v7_model = models.get(safe)
        ld_model = ld_models.get(safe)
        threshold = meta.get('pairs', {}).get(safe, {}).get('threshold', 0.5)

        if v7_model is None or ld_model is None:
            continue

        feat = compute_features(df)
        loss_feats = precompute_loss_features(df, btc_df)
        fcols = [c for c in v7_model.feature_name_ if c in feat.columns]

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
                timeout = (i - pos['bar']) >= MAX_HOLD

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
                    })
                    pos = None

            if pos is None:
                X = feat.loc[ts:ts][fcols]
                if X.isna().any().any():
                    continue

                pred = v7_model.predict(X)[0]
                sig = 1 if pred > 0.5 else -1

                chop_val = feat.loc[ts, 'chop'] if 'chop' in feat.columns else 50
                conviction = abs(pred - 0.5) * 10
                if conviction < conf_threshold or chop_val > chop_threshold:
                    continue

                # LossDetector (21 features)
                if ts not in loss_feats.index:
                    continue
                feat_row = loss_feats.loc[ts]
                atr_val = feat.loc[ts, 'atr14']
                atr_pct = atr_val / price if price > 0 else 0
                tp_sl_ratio = 2.5  # ATR TP/SL ratio

                # Orden exacto de features
                ld_vec = np.array([
                    conviction,                         # cs_conf
                    abs(pred - 0.5),                   # cs_pred_mag
                    0.5,                               # cs_macro_score (neutral)
                    0,                                 # cs_risk_off
                    1 if sig == 1 else 0,              # cs_regime_bull
                    1 if sig == -1 else 0,             # cs_regime_bear
                    0,                                 # cs_regime_range
                    atr_pct,                           # cs_atr_pct
                    0,                                 # cs_n_open
                    sig,                               # cs_pred_sign
                    conviction,                        # ld_conviction_pred
                    feat_row.get('ld_pair_rsi14', 0),
                    feat_row.get('ld_pair_bb_pct', 0),
                    feat_row.get('ld_pair_vol_ratio', 0),
                    feat_row.get('ld_pair_ret_5', 0),
                    feat_row.get('ld_pair_ret_20', 0),
                    feat_row.get('ld_btc_ret_5', 0),
                    feat_row.get('ld_btc_rsi14', 0),
                    feat_row.get('ld_btc_vol20', 0),
                    ts.hour / 24.0,                    # ld_hour (normalized)
                    tp_sl_ratio,                       # ld_tp_sl_ratio
                ]).reshape(1, -1)
                prob_loss = ld_model.predict_proba(ld_vec)[0, 1]
                if prob_loss > threshold:
                    continue
                tp_pct = atr_val / price * atr_tp_mult
                sl_pct = atr_val / price * atr_sl_mult
                risk_amt = balance * BASE_RISK_PER_TRADE
                size = risk_amt / (sl_pct * price) if sl_pct > 0 else 0

                pos = {
                    'entry': price,
                    'dir': sig,
                    'size': size,
                    'tp_pct': tp_pct,
                    'sl_pct': sl_pct,
                    'bar': i,
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
            })

    return trades, max_dd


def run_backtest_v10_1_adaptive(
    pair_data, btc_df, models, ld_models, meta,
    start_date, end_date,
    regime_detector,
):
    """V10.1 Adaptativo con detector V2."""
    trades = []

    for pair, df in pair_data.items():
        df = df[(df.index >= start_date) & (df.index < end_date)]
        if len(df) < 250:
            continue

        safe = pair.replace('/', '')
        v7_model = models.get(safe)
        ld_model = ld_models.get(safe)
        threshold = meta.get('pairs', {}).get(safe, {}).get('threshold', 0.5)

        if v7_model is None or ld_model is None:
            continue

        feat = compute_features(df)
        loss_feats = precompute_loss_features(df, btc_df)
        regimes = regime_detector.detect_regime_series(df)
        fcols = [c for c in v7_model.feature_name_ if c in feat.columns]

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
                timeout = (i - pos['bar']) >= pos.get('max_hold', MAX_HOLD)

                # Trailing stop
                if pos.get('use_trailing', False) and pnl_pct > 0:
                    new_sl = pnl_pct * 0.5
                    pos['sl_pct'] = max(pos['sl_pct'], new_sl)

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
                except ValueError:
                    regime = MarketRegime.UNKNOWN

                strategy = get_strategy_for_regime(regime)

                # No operar si el regimen no lo permite
                if not strategy.should_trade:
                    continue

                # Generar senal
                X = feat.loc[ts:ts][fcols]
                if X.isna().any().any():
                    continue

                pred = v7_model.predict(X)[0]
                sig = 1 if pred > 0.5 else -1

                # Verificar si direccion permitida
                if sig not in strategy.allowed_directions:
                    continue

                # Calcular conviction
                conviction = abs(pred - 0.5) * 10

                # Verificar conviction minimo del regimen
                if conviction < strategy.min_conviction:
                    continue

                # LossDetector (21 features)
                if ts not in loss_feats.index:
                    continue
                feat_row = loss_feats.loc[ts]
                atr_val = regimes.loc[ts, 'atr']
                atr_pct = atr_val / price if price > 0 else 0
                tp_sl_ratio = strategy.tp_multiplier / strategy.sl_multiplier if strategy.sl_multiplier > 0 else 2.5

                # Orden exacto de features
                ld_vec = np.array([
                    conviction,                         # cs_conf
                    abs(pred - 0.5),                   # cs_pred_mag
                    0.5,                               # cs_macro_score (neutral)
                    0,                                 # cs_risk_off
                    1 if regime == MarketRegime.BULL_TREND else 0,  # cs_regime_bull
                    1 if regime == MarketRegime.BEAR_TREND else 0,  # cs_regime_bear
                    1 if regime == MarketRegime.LATERAL else 0,     # cs_regime_range
                    atr_pct,                           # cs_atr_pct
                    0,                                 # cs_n_open
                    sig,                               # cs_pred_sign
                    conviction,                        # ld_conviction_pred
                    feat_row.get('ld_pair_rsi14', 0),
                    feat_row.get('ld_pair_bb_pct', 0),
                    feat_row.get('ld_pair_vol_ratio', 0),
                    feat_row.get('ld_pair_ret_5', 0),
                    feat_row.get('ld_pair_ret_20', 0),
                    feat_row.get('ld_btc_ret_5', 0),
                    feat_row.get('ld_btc_rsi14', 0),
                    feat_row.get('ld_btc_vol20', 0),
                    ts.hour / 24.0,                    # ld_hour (normalized)
                    tp_sl_ratio,                       # ld_tp_sl_ratio
                ]).reshape(1, -1)
                prob_loss = ld_model.predict_proba(ld_vec)[0, 1]
                if prob_loss > threshold:
                    continue

                # Calcular TP/SL segun estrategia del regimen
                if strategy.use_atr:
                    tp_pct = atr_val / price * strategy.tp_multiplier
                    sl_pct = atr_val / price * strategy.sl_multiplier
                else:
                    tp_pct = strategy.tp_multiplier
                    sl_pct = strategy.sl_multiplier

                # Position sizing ajustado por regimen
                risk_amt = balance * BASE_RISK_PER_TRADE * strategy.position_size_mult
                size = risk_amt / (sl_pct * price) if sl_pct > 0 else 0

                pos = {
                    'entry': price,
                    'dir': sig,
                    'size': size,
                    'tp_pct': tp_pct,
                    'sl_pct': sl_pct,
                    'bar': i,
                    'use_trailing': strategy.use_trailing,
                    'max_hold': strategy.max_hold_candles,
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
    """Calcula metricas de backtest."""
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
    print("="*60)
    print("BACKTEST V10.1: Sistema Adaptativo con Detector V2")
    print("="*60)

    # Cargar modelos
    print("\n[1] Cargando modelos...")
    models = {}
    ld_models = {}
    for pair in PAIRS:
        safe = pair.replace('/', '')
        try:
            models[safe] = joblib.load(MODELS_DIR / f'v95_v7_{safe}.pkl')
            ld_models[safe] = joblib.load(MODELS_DIR / f'v95_ld_{safe}.pkl')
        except FileNotFoundError:
            pass

    with open(MODELS_DIR / 'v95_meta.json') as f:
        meta = json.load(f)

    print(f"  Modelos cargados: {len(models)}")

    # Cargar datos
    print("\n[2] Cargando datos...")
    pair_data = {}
    for pair in PAIRS:
        df = load_data(pair)
        if df is not None:
            pair_data[pair] = df
    btc_df = load_data('BTC/USDT')
    print(f"  Pares cargados: {len(pair_data)}")

    # Inicializar detector V2
    detector = RegimeDetector()

    # Periodos de test
    periods = [
        ('Ultimo Ano (Feb 2025 - Feb 2026)', '2025-02-01', '2026-02-24'),
        ('Bear Market 2022', '2022-01-01', '2023-01-01'),
    ]

    results = {}

    for period_name, start, end in periods:
        print(f"\n{'='*60}")
        print(f"PERIODO: {period_name}")
        print(f"{'='*60}")

        # V9.5+ATR
        print("\n[V9.5+ATR] Ejecutando...")
        t0 = time.time()
        trades_v95, dd_v95 = run_backtest_v95_atr(
            pair_data, btc_df, models, ld_models, meta,
            start, end,
        )
        metrics_v95 = compute_metrics(trades_v95, dd_v95)
        print(f"  Tiempo: {time.time()-t0:.1f}s")

        # V10.1 Adaptativo
        print("\n[V10.1 Adaptive] Ejecutando...")
        t0 = time.time()
        trades_v10, dd_v10 = run_backtest_v10_1_adaptive(
            pair_data, btc_df, models, ld_models, meta,
            start, end, detector,
        )
        metrics_v10 = compute_metrics(trades_v10, dd_v10)
        print(f"  Tiempo: {time.time()-t0:.1f}s")

        # Mostrar resultados
        print(f"\nRESULTADOS: {period_name}")
        print("-"*60)
        print(f"{'Metrica':<15} {'V9.5+ATR':>15} {'V10.1':>15} {'Diff':>12}")
        print("-"*60)

        for k in ['trades', 'wins', 'wr', 'pnl', 'pf', 'return_pct', 'max_dd']:
            v1 = metrics_v95[k]
            v2 = metrics_v10[k]
            diff = v2 - v1
            if k in ['wr', 'max_dd', 'return_pct']:
                print(f"{k:<15} {v1:>14.1f}% {v2:>14.1f}% {diff:>+11.1f}%")
            elif k == 'pnl':
                print(f"{k:<15} ${v1:>13,.0f} ${v2:>13,.0f} ${diff:>+10,.0f}")
            elif k == 'pf':
                print(f"{k:<15} {v1:>15.2f} {v2:>15.2f} {diff:>+12.2f}")
            else:
                print(f"{k:<15} {v1:>15,} {v2:>15,} {diff:>+12,}")

        results[period_name] = {
            'v95': {**metrics_v95, 'name': 'V9.5+ATR'},
            'v10': {**metrics_v10, 'name': 'V10.1 Adaptativo'},
        }

        # Distribucion por regimen (solo V10.1)
        if trades_v10:
            print(f"\nDistribucion por regimen (V10.1):")
            regime_counts = {}
            for t in trades_v10:
                r = t.get('regime', 'unknown')
                regime_counts[r] = regime_counts.get(r, 0) + 1
            for r, c in sorted(regime_counts.items(), key=lambda x: -x[1]):
                pct = c / len(trades_v10) * 100
                print(f"  {r:<15}: {c:>5} ({pct:>5.1f}%)")

    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)

    total_v95 = sum(results[p]['v95']['pnl'] for p in results)
    total_v10 = sum(results[p]['v10']['pnl'] for p in results)
    avg_dd_v95 = np.mean([results[p]['v95']['max_dd'] for p in results])
    avg_dd_v10 = np.mean([results[p]['v10']['max_dd'] for p in results])

    print(f"\nPnL Total V9.5+ATR:   ${total_v95:,.0f}")
    print(f"PnL Total V10.1:      ${total_v10:,.0f}")
    diff_pct = (total_v10 - total_v95) / total_v95 * 100 if total_v95 != 0 else 0
    print(f"Diferencia:           {diff_pct:+.1f}%")

    print(f"\nDD Promedio V9.5+ATR: {avg_dd_v95:.1f}%")
    print(f"DD Promedio V10.1:    {avg_dd_v10:.1f}%")

    winner = "V9.5+ATR" if total_v95 > total_v10 else "V10.1 Adaptativo"
    print(f"\nGANADOR: {winner}")

    # Guardar resultados
    output = {
        **{f'period{i+1}': results[p] for i, p in enumerate(results)},
        'summary': {
            'total_pnl_v95': total_v95,
            'total_pnl_v10': total_v10,
            'avg_dd_v95': avg_dd_v95,
            'avg_dd_v10': avg_dd_v10,
            'winner': winner,
        }
    }
    with open(MODELS_DIR / 'backtest_v10_1_adaptive.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResultados guardados en models/backtest_v10_1_adaptive.json")


if __name__ == '__main__':
    main()
