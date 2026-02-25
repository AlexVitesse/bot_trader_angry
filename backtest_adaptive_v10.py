"""
Backtest V10: Sistema Adaptativo con Detector de Regimen
========================================================
Compara:
- V9.5+ATR (estrategia fija)
- V10 Adaptativo (estrategia segun regimen)

El sistema V10 detecta el regimen de mercado y adapta:
- Direccion permitida (solo longs en bull, solo shorts en bear)
- TP/SL dinamicos segun volatilidad
- NO opera en mercados laterales
- Tamano de posicion segun riesgo del regimen

Uso: poetry run python backtest_adaptive_v10.py
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

from regime_detector import (
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
    """Precomputa features para LossDetector (igual que backtest_experiments_v3)."""
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

    return feat


def simulate_trade_with_strategy(
    df: pd.DataFrame,
    entry_idx: int,
    direction: int,
    strategy: RegimeStrategy,
    atr_value: float,
    entry_price: float,
) -> dict:
    """
    Simula un trade usando la estrategia del regimen.
    """
    if not strategy.should_trade:
        return None

    # Calcular TP/SL segun estrategia
    if strategy.use_atr:
        atr_pct = atr_value / entry_price
        tp_pct = atr_pct * strategy.tp_multiplier
        sl_pct = atr_pct * strategy.sl_multiplier
    else:
        tp_pct = strategy.tp_multiplier
        sl_pct = strategy.sl_multiplier

    tp_price = entry_price * (1 + tp_pct) if direction == 1 else entry_price * (1 - tp_pct)
    sl_price = entry_price * (1 - sl_pct) if direction == 1 else entry_price * (1 + sl_pct)

    # Trailing stop
    if strategy.use_trailing:
        highest = entry_price if direction == 1 else entry_price
        lowest = entry_price if direction == -1 else entry_price
        trail_pct = sl_pct * 1.5  # Trailing a 1.5x SL

    max_hold = min(strategy.max_hold_candles, len(df) - entry_idx - 1)

    pnl_pct = None
    exit_reason = None
    exit_idx = None

    for j in range(entry_idx + 1, entry_idx + max_hold + 1):
        if j >= len(df):
            break

        candle = df.iloc[j]

        # Actualizar trailing
        if strategy.use_trailing:
            if direction == 1:
                if candle['high'] > highest:
                    highest = candle['high']
                    # Mover SL hacia arriba
                    new_trail_sl = highest * (1 - trail_pct)
                    if new_trail_sl > sl_price:
                        sl_price = new_trail_sl
            else:
                if candle['low'] < lowest:
                    lowest = candle['low']
                    new_trail_sl = lowest * (1 + trail_pct)
                    if new_trail_sl < sl_price:
                        sl_price = new_trail_sl

        # Check SL
        if direction == 1:
            if candle['low'] <= sl_price:
                pnl_pct = (sl_price - entry_price) / entry_price
                exit_reason = 'sl'
                exit_idx = j
                break
            if candle['high'] >= tp_price:
                pnl_pct = tp_pct
                exit_reason = 'tp'
                exit_idx = j
                break
        else:
            if candle['high'] >= sl_price:
                pnl_pct = (entry_price - sl_price) / entry_price
                exit_reason = 'sl'
                exit_idx = j
                break
            if candle['low'] <= tp_price:
                pnl_pct = tp_pct
                exit_reason = 'tp'
                exit_idx = j
                break

    # Timeout
    if pnl_pct is None:
        exit_idx = min(entry_idx + max_hold, len(df) - 1)
        exit_price = df.iloc[exit_idx]['close']
        pnl_pct = (exit_price - entry_price) / entry_price * direction
        exit_reason = 'timeout'

    return {
        'pnl_pct': pnl_pct,
        'exit_reason': exit_reason,
        'exit_idx': exit_idx,
        'tp_pct': tp_pct,
        'sl_pct': sl_pct,
        'position_mult': strategy.position_size_mult,
    }


def run_backtest_v95_atr(all_data, all_preds, all_chop, all_atr, all_loss_features,
                          v95_detectors, v95_thresholds, test_start, test_end):
    """V9.5+ATR: Estrategia FIJA (referencia)."""
    trades = []

    V95_CONF_MIN = 1.8
    V95_CHOP_MAX = 50
    V95_THRESH_BOOST = 0.15
    V95_ATR_TP_MULT = 2.5
    V95_ATR_SL_MULT = 1.0

    for pair in PAIRS:
        if pair not in all_preds or pair not in all_data:
            continue
        if pair not in v95_detectors:
            continue

        preds = all_preds[pair]
        df = all_data[pair]
        loss_feats = all_loss_features.get(pair, pd.DataFrame())
        model = v95_detectors[pair]
        thresh = v95_thresholds[pair] + V95_THRESH_BOOST

        for ts, pred in preds.items():
            if ts < test_start or ts >= test_end:
                continue

            direction = pred.get('direction', 0)
            conviction = pred.get('conviction', 0)

            if direction == 0 or abs(conviction) < V95_CONF_MIN:
                continue

            chop_val = all_chop[pair].get(ts)
            if chop_val is None or chop_val >= V95_CHOP_MAX:
                continue

            # Get loss features from DataFrame
            if ts not in loss_feats.index:
                continue
            feat_row = loss_feats.loc[ts]

            try:
                # Build full feature vector for LossDetector
                pred_val = pred.get('pred', 0)
                full_features = pd.Series({
                    'cs_conf': conviction,
                    'cs_pred_mag': abs(pred_val),
                    'cs_macro_score': 0.5,
                    'cs_risk_off': 0,
                    'cs_regime_bull': 1 if direction == 1 else 0,
                    'cs_regime_bear': 1 if direction == -1 else 0,
                    'cs_regime_range': 0,
                    'cs_atr_pct': feat_row.get('atr_pct', 0.03) if hasattr(feat_row, 'get') else feat_row['atr_pct'] if 'atr_pct' in feat_row.index else 0.03,
                    'cs_n_open': 1,
                    'cs_pred_sign': direction,
                    'ld_conviction_pred': conviction,
                    'ld_pair_rsi14': feat_row['ld_pair_rsi14'] if 'ld_pair_rsi14' in feat_row.index else 0.5,
                    'ld_pair_bb_pct': feat_row['ld_pair_bb_pct'] if 'ld_pair_bb_pct' in feat_row.index else 0.5,
                    'ld_pair_vol_ratio': feat_row['ld_pair_vol_ratio'] if 'ld_pair_vol_ratio' in feat_row.index else 1.0,
                    'ld_pair_ret_5': feat_row['ld_pair_ret_5'] if 'ld_pair_ret_5' in feat_row.index else 0,
                    'ld_pair_ret_20': feat_row['ld_pair_ret_20'] if 'ld_pair_ret_20' in feat_row.index else 0,
                    'ld_btc_ret_5': feat_row['ld_btc_ret_5'] if 'ld_btc_ret_5' in feat_row.index else 0,
                    'ld_btc_rsi14': feat_row['ld_btc_rsi14'] if 'ld_btc_rsi14' in feat_row.index else 0.5,
                    'ld_btc_vol20': feat_row['ld_btc_vol20'] if 'ld_btc_vol20' in feat_row.index else 0.02,
                    'ld_hour': ts.hour,
                    'ld_tp_sl_ratio': 2.5,
                })
                p_loss = float(model.predict_proba(full_features.values.reshape(1, -1))[0][1])
                if p_loss > thresh:
                    continue
            except:
                continue

            try:
                idx = df.index.get_loc(ts)
            except:
                continue

            if idx + 1 >= len(df):
                continue

            entry = df.iloc[idx + 1]['open']
            atr_val = all_atr[pair].get(ts)
            if atr_val is None:
                continue

            atr_pct = atr_val / entry
            tp_pct = atr_pct * V95_ATR_TP_MULT
            sl_pct = atr_pct * V95_ATR_SL_MULT

            tp_price = entry * (1 + tp_pct) if direction == 1 else entry * (1 - tp_pct)
            sl_price = entry * (1 - sl_pct) if direction == 1 else entry * (1 + sl_pct)

            pnl_pct = None
            for j in range(idx + 1, min(idx + MAX_HOLD + 1, len(df))):
                candle = df.iloc[j]
                if direction == 1:
                    if candle['low'] <= sl_price:
                        pnl_pct = -sl_pct
                        break
                    if candle['high'] >= tp_price:
                        pnl_pct = tp_pct
                        break
                else:
                    if candle['high'] >= sl_price:
                        pnl_pct = -sl_pct
                        break
                    if candle['low'] <= tp_price:
                        pnl_pct = tp_pct
                        break

            if pnl_pct is None:
                exit_idx = min(idx + MAX_HOLD, len(df) - 1)
                exit_price = df.iloc[exit_idx]['close']
                pnl_pct = (exit_price - entry) / entry * direction

            trades.append({
                'pair': pair,
                'entry_ts': ts,
                'direction': direction,
                'pnl_pct': pnl_pct,
                'win': pnl_pct > 0,
                'regime': 'fixed',
            })

    return trades


def run_backtest_v10_adaptive(all_data, all_preds, all_regimes, all_atr, all_loss_features,
                               v95_detectors, v95_thresholds, test_start, test_end):
    """V10 Adaptativo: Estrategia segun REGIMEN."""
    trades = []
    skipped_by_regime = {r.value: 0 for r in MarketRegime}

    V95_THRESH_BOOST = 0.15

    for pair in PAIRS:
        if pair not in all_preds or pair not in all_data:
            continue
        if pair not in v95_detectors:
            continue

        preds = all_preds[pair]
        df = all_data[pair]
        regimes = all_regimes[pair]
        loss_feats = all_loss_features.get(pair, pd.DataFrame())
        model = v95_detectors[pair]
        thresh = v95_thresholds[pair] + V95_THRESH_BOOST

        for ts, pred in preds.items():
            if ts < test_start or ts >= test_end:
                continue

            direction = pred.get('direction', 0)
            conviction = pred.get('conviction', 0)

            if direction == 0:
                continue

            # Obtener regimen actual
            regime_row = regimes.get(ts)
            if regime_row is None:
                continue

            regime_str = regime_row.get('regime', 'unknown')
            try:
                regime = MarketRegime(regime_str)
            except:
                regime = MarketRegime.UNKNOWN

            strategy = get_strategy_for_regime(regime)

            # Verificar si debemos operar
            should_trade, reason = should_take_trade(regime, direction, conviction)

            if not should_trade:
                skipped_by_regime[regime.value] += 1
                continue

            # LossDetector (igual que V9.5)
            if ts not in loss_feats.index:
                continue
            feat_row = loss_feats.loc[ts]

            try:
                # Build full feature vector for LossDetector
                pred_val = pred.get('pred', 0)
                full_features = pd.Series({
                    'cs_conf': conviction,
                    'cs_pred_mag': abs(pred_val),
                    'cs_macro_score': 0.5,
                    'cs_risk_off': 0,
                    'cs_regime_bull': 1 if direction == 1 else 0,
                    'cs_regime_bear': 1 if direction == -1 else 0,
                    'cs_regime_range': 0,
                    'cs_atr_pct': feat_row['atr_pct'] if 'atr_pct' in feat_row.index else 0.03,
                    'cs_n_open': 1,
                    'cs_pred_sign': direction,
                    'ld_conviction_pred': conviction,
                    'ld_pair_rsi14': feat_row['ld_pair_rsi14'] if 'ld_pair_rsi14' in feat_row.index else 0.5,
                    'ld_pair_bb_pct': feat_row['ld_pair_bb_pct'] if 'ld_pair_bb_pct' in feat_row.index else 0.5,
                    'ld_pair_vol_ratio': feat_row['ld_pair_vol_ratio'] if 'ld_pair_vol_ratio' in feat_row.index else 1.0,
                    'ld_pair_ret_5': feat_row['ld_pair_ret_5'] if 'ld_pair_ret_5' in feat_row.index else 0,
                    'ld_pair_ret_20': feat_row['ld_pair_ret_20'] if 'ld_pair_ret_20' in feat_row.index else 0,
                    'ld_btc_ret_5': feat_row['ld_btc_ret_5'] if 'ld_btc_ret_5' in feat_row.index else 0,
                    'ld_btc_rsi14': feat_row['ld_btc_rsi14'] if 'ld_btc_rsi14' in feat_row.index else 0.5,
                    'ld_btc_vol20': feat_row['ld_btc_vol20'] if 'ld_btc_vol20' in feat_row.index else 0.02,
                    'ld_hour': ts.hour,
                    'ld_tp_sl_ratio': 2.5,
                })
                p_loss = float(model.predict_proba(full_features.values.reshape(1, -1))[0][1])
                if p_loss > thresh:
                    continue
            except:
                continue

            try:
                idx = df.index.get_loc(ts)
            except:
                continue

            if idx + 1 >= len(df):
                continue

            entry = df.iloc[idx + 1]['open']
            atr_val = all_atr[pair].get(ts)
            if atr_val is None:
                continue

            # Simular trade con estrategia del regimen
            result = simulate_trade_with_strategy(
                df, idx + 1, direction, strategy, atr_val, entry
            )

            if result is None:
                continue

            # Ajustar PnL por tamano de posicion
            adjusted_pnl = result['pnl_pct'] * result['position_mult']

            trades.append({
                'pair': pair,
                'entry_ts': ts,
                'direction': direction,
                'pnl_pct': adjusted_pnl,
                'raw_pnl_pct': result['pnl_pct'],
                'win': result['pnl_pct'] > 0,
                'regime': regime.value,
                'exit_reason': result['exit_reason'],
                'position_mult': result['position_mult'],
            })

    return trades, skipped_by_regime


def calculate_metrics(trades, name):
    """Calcula metricas de un conjunto de trades."""
    if not trades:
        return {
            'name': name, 'trades': 0, 'wins': 0, 'wr': 0,
            'pnl': 0, 'pf': 0, 'return_pct': 0, 'max_dd': 0,
        }

    n_trades = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n_trades * 100

    pnl = sum(t['pnl_pct'] * INITIAL_CAPITAL for t in trades)
    gross_profit = sum(t['pnl_pct'] * INITIAL_CAPITAL for t in trades if t['pnl_pct'] > 0)
    gross_loss = abs(sum(t['pnl_pct'] * INITIAL_CAPITAL for t in trades if t['pnl_pct'] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999

    return_pct = pnl / INITIAL_CAPITAL * 100

    # Max drawdown
    equity = [INITIAL_CAPITAL]
    for t in sorted(trades, key=lambda x: x['entry_ts']):
        equity.append(equity[-1] + t['pnl_pct'] * INITIAL_CAPITAL)

    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        'name': name,
        'trades': n_trades,
        'wins': wins,
        'wr': wr,
        'pnl': pnl,
        'pf': pf,
        'return_pct': return_pct,
        'max_dd': max_dd,
    }


def main():
    print('=' * 70)
    print('BACKTEST V10: SISTEMA ADAPTATIVO CON DETECTOR DE REGIMEN')
    print('=' * 70)

    # Periodos
    period1_start = pd.Timestamp('2025-02-01', tz='UTC')
    period1_end = pd.Timestamp('2026-02-24', tz='UTC')
    period1_name = 'Ultimo Ano (Feb 2025 - Feb 2026)'

    period2_start = pd.Timestamp('2022-01-01', tz='UTC')
    period2_end = pd.Timestamp('2023-01-01', tz='UTC')
    period2_name = 'Bear Market 2022'

    t0 = time.time()

    # =========================================================================
    # CARGAR DATOS
    # =========================================================================
    print('\n[1] Cargando datos...')
    all_data = {}
    for pair in PAIRS:
        df = load_data(pair)
        if df is not None:
            all_data[pair] = df
            print(f'  {pair}: {len(df)} rows')

    # =========================================================================
    # GENERAR PREDICCIONES V7
    # =========================================================================
    print('\n[2] Generando predicciones V7...')
    all_preds = {}
    v7_models = {}

    for pair in PAIRS:
        safe = pair.replace('/', '_')
        model_path = MODELS_DIR / f'v7_{safe}.pkl'
        if model_path.exists():
            v7_models[pair] = joblib.load(model_path)

    for pair in PAIRS:
        if pair not in all_data or pair not in v7_models:
            continue

        df = all_data[pair]
        feat = compute_features(df)
        model = v7_models[pair]

        if hasattr(model, 'feature_names_in_'):
            fcols = list(model.feature_names_in_)
        else:
            fcols = [c for c in feat.columns]

        preds = {}
        for i in range(200, len(df)):
            ts = df.index[i]
            row = feat.iloc[i]

            if row[fcols].isna().any():
                continue

            try:
                pred = model.predict(row[fcols].values.reshape(1, -1))[0]
                direction = 1 if pred > 0.01 else -1 if pred < -0.01 else 0
                conviction = abs(pred) * 100
                preds[ts] = {'direction': direction, 'conviction': conviction, 'pred': pred}
            except:
                continue

        all_preds[pair] = preds
        print(f'  {pair}: {len(preds)} predicciones')

    # =========================================================================
    # DETECTAR REGIMENES
    # =========================================================================
    print('\n[3] Detectando regimenes de mercado...')
    detector = RegimeDetector()
    all_regimes = {}
    all_atr = {}
    all_chop = {}

    for pair, df in all_data.items():
        regime_df = detector.detect_regime_series(df)

        # Convertir a dict para acceso rapido
        regimes = {}
        for ts in regime_df.index:
            regimes[ts] = {
                'regime': regime_df.loc[ts, 'regime'],
                'atr': regime_df.loc[ts, 'atr'],
                'chop': regime_df.loc[ts, 'chop'],
            }
        all_regimes[pair] = regimes

        # ATR y Chop para V9.5
        all_atr[pair] = {ts: regime_df.loc[ts, 'atr'] for ts in regime_df.index if pd.notna(regime_df.loc[ts, 'atr'])}
        all_chop[pair] = {ts: regime_df.loc[ts, 'chop'] for ts in regime_df.index if pd.notna(regime_df.loc[ts, 'chop'])}

        # Contar regimenes
        regime_counts = regime_df['regime'].value_counts()
        print(f'  {pair}: {dict(regime_counts)}')

    # =========================================================================
    # PRECOMPUTAR LOSS FEATURES
    # =========================================================================
    print('\n[4] Precomputando loss features...')
    all_loss_features = {}
    btc_df = all_data.get('BTC/USDT')

    for pair in PAIRS:
        if pair not in all_data:
            continue
        df = all_data[pair]
        all_loss_features[pair] = precompute_loss_features(df, btc_df)
        print(f'  {pair}: {len(all_loss_features[pair])} rows')

    # =========================================================================
    # CARGAR LOSSDETECTORS
    # =========================================================================
    print('\n[5] Cargando LossDetectors...')
    v95_detectors = {}
    v95_thresholds = {}

    meta_path = MODELS_DIR / 'v95_meta.json'
    if meta_path.exists():
        meta = json.load(open(meta_path))
        for pair in PAIRS:
            safe = pair.replace('/', '')
            model_path = MODELS_DIR / f'v95_ld_{safe}.pkl'
            if model_path.exists():
                v95_detectors[pair] = joblib.load(model_path)
                v95_thresholds[pair] = meta['pairs'].get(pair, {}).get('threshold', 0.5)

    print(f'  Cargados: {len(v95_detectors)} modelos')

    # =========================================================================
    # BACKTEST PERIODO 1: ULTIMO ANO
    # =========================================================================
    print(f'\n[5] Backtest {period1_name}...')

    print('  V9.5+ATR (fijo)...')
    trades_v95_p1 = run_backtest_v95_atr(
        all_data, all_preds, all_chop, all_atr, all_loss_features,
        v95_detectors, v95_thresholds, period1_start, period1_end
    )
    m_v95_p1 = calculate_metrics(trades_v95_p1, 'V9.5+ATR')

    print('  V10 Adaptativo...')
    trades_v10_p1, skipped_p1 = run_backtest_v10_adaptive(
        all_data, all_preds, all_regimes, all_atr, all_loss_features,
        v95_detectors, v95_thresholds, period1_start, period1_end
    )
    m_v10_p1 = calculate_metrics(trades_v10_p1, 'V10 Adaptativo')

    # =========================================================================
    # BACKTEST PERIODO 2: BEAR MARKET
    # =========================================================================
    print(f'\n[6] Backtest {period2_name}...')

    print('  V9.5+ATR (fijo)...')
    trades_v95_p2 = run_backtest_v95_atr(
        all_data, all_preds, all_chop, all_atr, all_loss_features,
        v95_detectors, v95_thresholds, period2_start, period2_end
    )
    m_v95_p2 = calculate_metrics(trades_v95_p2, 'V9.5+ATR')

    print('  V10 Adaptativo...')
    trades_v10_p2, skipped_p2 = run_backtest_v10_adaptive(
        all_data, all_preds, all_regimes, all_atr, all_loss_features,
        v95_detectors, v95_thresholds, period2_start, period2_end
    )
    m_v10_p2 = calculate_metrics(trades_v10_p2, 'V10 Adaptativo')

    # =========================================================================
    # RESULTADOS
    # =========================================================================
    elapsed = time.time() - t0

    print('\n' + '=' * 70)
    print(f'RESULTADOS: {period1_name}')
    print('=' * 70)
    print(f'{"Metrica":<20} {"V9.5+ATR":>15} {"V10 Adapt":>15} {"Diff":>12}')
    print('-' * 70)
    print(f'{"Trades":<20} {m_v95_p1["trades"]:>15} {m_v10_p1["trades"]:>15} {m_v10_p1["trades"]-m_v95_p1["trades"]:>+12}')
    print(f'{"Win Rate":<20} {m_v95_p1["wr"]:>14.1f}% {m_v10_p1["wr"]:>14.1f}% {m_v10_p1["wr"]-m_v95_p1["wr"]:>+11.1f}%')
    print(f'{"PnL":<20} ${m_v95_p1["pnl"]:>13,.0f} ${m_v10_p1["pnl"]:>13,.0f} ${m_v10_p1["pnl"]-m_v95_p1["pnl"]:>+11,.0f}')
    print(f'{"Profit Factor":<20} {m_v95_p1["pf"]:>15.2f} {m_v10_p1["pf"]:>15.2f} {m_v10_p1["pf"]-m_v95_p1["pf"]:>+12.2f}')
    print(f'{"Max Drawdown":<20} {m_v95_p1["max_dd"]:>14.1f}% {m_v10_p1["max_dd"]:>14.1f}% {m_v10_p1["max_dd"]-m_v95_p1["max_dd"]:>+11.1f}%')

    print('\n' + '=' * 70)
    print(f'RESULTADOS: {period2_name}')
    print('=' * 70)
    print(f'{"Metrica":<20} {"V9.5+ATR":>15} {"V10 Adapt":>15} {"Diff":>12}')
    print('-' * 70)
    print(f'{"Trades":<20} {m_v95_p2["trades"]:>15} {m_v10_p2["trades"]:>15} {m_v10_p2["trades"]-m_v95_p2["trades"]:>+12}')
    print(f'{"Win Rate":<20} {m_v95_p2["wr"]:>14.1f}% {m_v10_p2["wr"]:>14.1f}% {m_v10_p2["wr"]-m_v95_p2["wr"]:>+11.1f}%')
    print(f'{"PnL":<20} ${m_v95_p2["pnl"]:>13,.0f} ${m_v10_p2["pnl"]:>13,.0f} ${m_v10_p2["pnl"]-m_v95_p2["pnl"]:>+11,.0f}')
    print(f'{"Profit Factor":<20} {m_v95_p2["pf"]:>15.2f} {m_v10_p2["pf"]:>15.2f} {m_v10_p2["pf"]-m_v95_p2["pf"]:>+12.2f}')
    print(f'{"Max Drawdown":<20} {m_v95_p2["max_dd"]:>14.1f}% {m_v10_p2["max_dd"]:>14.1f}% {m_v10_p2["max_dd"]-m_v95_p2["max_dd"]:>+11.1f}%')

    # Trades por regimen
    print('\n' + '=' * 70)
    print('DISTRIBUCION DE TRADES V10 POR REGIMEN')
    print('=' * 70)

    for period_name, trades in [(period1_name, trades_v10_p1), (period2_name, trades_v10_p2)]:
        if not trades:
            continue
        print(f'\n{period_name}:')
        regime_stats = {}
        for t in trades:
            r = t['regime']
            if r not in regime_stats:
                regime_stats[r] = {'count': 0, 'wins': 0, 'pnl': 0}
            regime_stats[r]['count'] += 1
            regime_stats[r]['wins'] += 1 if t['win'] else 0
            regime_stats[r]['pnl'] += t['pnl_pct'] * INITIAL_CAPITAL

        for regime, stats in sorted(regime_stats.items()):
            wr = stats['wins'] / stats['count'] * 100 if stats['count'] > 0 else 0
            print(f'  {regime:<15}: {stats["count"]:>5} trades, WR={wr:>5.1f}%, PnL=${stats["pnl"]:>8,.0f}')

    # Trades evitados por regimen lateral
    print('\n' + '=' * 70)
    print('TRADES EVITADOS POR REGIMEN')
    print('=' * 70)
    print(f'\nPeriodo 1: {skipped_p1}')
    print(f'Periodo 2: {skipped_p2}')

    # Resumen final
    print('\n' + '=' * 70)
    print('RESUMEN FINAL')
    print('=' * 70)

    total_v95 = m_v95_p1['pnl'] + m_v95_p2['pnl']
    total_v10 = m_v10_p1['pnl'] + m_v10_p2['pnl']
    avg_dd_v95 = (m_v95_p1['max_dd'] + m_v95_p2['max_dd']) / 2
    avg_dd_v10 = (m_v10_p1['max_dd'] + m_v10_p2['max_dd']) / 2

    print(f'\nV9.5+ATR (Fijo):')
    print(f'  PnL Total: ${total_v95:,.0f}')
    print(f'  Max DD promedio: {avg_dd_v95:.1f}%')

    print(f'\nV10 Adaptativo:')
    print(f'  PnL Total: ${total_v10:,.0f}')
    print(f'  Max DD promedio: {avg_dd_v10:.1f}%')

    diff_pnl = total_v10 - total_v95
    diff_dd = avg_dd_v10 - avg_dd_v95

    print(f'\nDiferencia:')
    pct_diff = (diff_pnl/total_v95*100) if total_v95 != 0 else 0
    print(f'  PnL: ${diff_pnl:+,.0f} ({pct_diff:+.1f}%)')
    print(f'  Max DD: {diff_dd:+.1f}%')

    winner = 'V10 Adaptativo' if total_v10 > total_v95 else 'V9.5+ATR'
    print(f'\nGANADOR: {winner}')

    print(f'\n[Completado en {elapsed:.1f}s]')

    # Guardar resultados
    results = {
        'period1': {'v95': m_v95_p1, 'v10': m_v10_p1},
        'period2': {'v95': m_v95_p2, 'v10': m_v10_p2},
        'summary': {
            'total_pnl_v95': total_v95,
            'total_pnl_v10': total_v10,
            'avg_dd_v95': avg_dd_v95,
            'avg_dd_v10': avg_dd_v10,
            'winner': winner,
        }
    }

    with open(MODELS_DIR / 'backtest_v10_adaptive.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f'\nResultados guardados en: {MODELS_DIR}/backtest_v10_adaptive.json')


if __name__ == '__main__':
    main()
