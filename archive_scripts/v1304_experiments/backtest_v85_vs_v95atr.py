"""
Backtest Comparativo: V8.5 vs V9.5+ATR
======================================
Compara ambas estrategias en:
1. Ultimo ano (Feb 2025 - Feb 2026)
2. Ano dificil (2022 - bear market)

V8.5: Macro + Conviction (sin LossDetector)
V9.5+ATR: Macro + Conviction + LossDetector + ATR dinamico

Uso: poetry run python backtest_v85_vs_v95atr.py
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

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT',
    'NEAR/USDT',
]

# Configuracion comun
INITIAL_CAPITAL = 500.0
RISK_PER_TRADE = 0.02
MAX_POSITIONS = 3
MAX_HOLD = 30

# V8.5 config
V85_TP = 0.03
V85_SL = 0.015
V85_CONF_MIN = 1.5

# V9.5+ATR config
V95_CONF_MIN = 1.8
V95_CHOP_MAX = 50
V95_THRESH_BOOST = 0.15
V95_ATR_TP_MULT = 2.5
V95_ATR_SL_MULT = 1.0


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


def compute_choppiness(df, length=14):
    """Calcula Choppiness Index."""
    atr = ta.atr(df['high'], df['low'], df['close'], length=1)
    atr_sum = atr.rolling(length).sum()
    high_max = df['high'].rolling(length).max()
    low_min = df['low'].rolling(length).min()
    chop = 100 * np.log10(atr_sum / (high_max - low_min + 1e-10)) / np.log10(length)
    return chop


def compute_atr(df, length=14):
    """Calcula ATR."""
    return ta.atr(df['high'], df['low'], df['close'], length=length)


def compute_loss_features(df, direction, conviction, btc_df=None):
    """Calcula features para LossDetector."""
    idx = len(df) - 1
    c = df['close'].iloc[idx]

    # Pair features
    rsi14 = ta.rsi(df['close'], length=14).iloc[idx]
    bb = ta.bbands(df['close'], length=20, std=2.0)
    bb_pct = 0.5
    if bb is not None:
        bw = bb.iloc[:, 2] - bb.iloc[:, 0]
        bb_pct = ((df['close'] - bb.iloc[:, 0]) / bw).iloc[idx]

    vol_ratio = (df['volume'] / df['volume'].rolling(20).mean()).iloc[idx]
    ret_5 = df['close'].pct_change(5).iloc[idx]
    ret_20 = df['close'].pct_change(20).iloc[idx]

    # BTC context
    btc_ret_5 = 0
    btc_rsi14 = 50
    btc_vol20 = 0.02
    if btc_df is not None and len(btc_df) > 20:
        btc_ret_5 = btc_df['close'].pct_change(5).iloc[-1]
        btc_rsi14 = ta.rsi(btc_df['close'], length=14).iloc[-1]
        btc_vol20 = btc_df['close'].pct_change().rolling(20).std().iloc[-1]

    # Hour
    hour = df.index[idx].hour

    # TP/SL ratio (fixed for this)
    tp_sl_ratio = 2.0

    features = pd.Series({
        'cs_conf': conviction,
        'cs_pred_mag': abs(conviction),
        'cs_macro_score': 0.5,
        'cs_risk_off': 0,
        'cs_regime_bull': 1 if direction == 1 else 0,
        'cs_regime_bear': 1 if direction == -1 else 0,
        'cs_regime_range': 0,
        'cs_atr_pct': 0.03,
        'cs_n_open': 1,
        'cs_pred_sign': direction,
        'ld_conviction_pred': conviction,
        'ld_pair_rsi14': rsi14 if pd.notna(rsi14) else 50,
        'ld_pair_bb_pct': bb_pct if pd.notna(bb_pct) else 0.5,
        'ld_pair_vol_ratio': vol_ratio if pd.notna(vol_ratio) else 1.0,
        'ld_pair_ret_5': ret_5 if pd.notna(ret_5) else 0,
        'ld_pair_ret_20': ret_20 if pd.notna(ret_20) else 0,
        'ld_btc_ret_5': btc_ret_5 if pd.notna(btc_ret_5) else 0,
        'ld_btc_rsi14': btc_rsi14 if pd.notna(btc_rsi14) else 50,
        'ld_btc_vol20': btc_vol20 if pd.notna(btc_vol20) else 0.02,
        'ld_hour': hour,
        'ld_tp_sl_ratio': tp_sl_ratio,
    })

    return features


def run_backtest_v85(all_data, all_preds, test_start, test_end):
    """
    Backtest V8.5: Macro + Conviction (sin LossDetector)
    TP/SL fijos: 3% / 1.5%
    """
    trades = []

    for pair in PAIRS:
        if pair not in all_preds or pair not in all_data:
            continue

        preds = all_preds[pair]
        df = all_data[pair]

        for ts, pred in preds.items():
            if ts < test_start or ts >= test_end:
                continue

            direction = pred.get('direction', 0)
            conviction = pred.get('conviction', 0)

            # V8.5: solo filtro por conviction minimo
            if direction == 0 or abs(conviction) < V85_CONF_MIN:
                continue

            # Simular trade
            try:
                idx = df.index.get_loc(ts)
            except:
                continue

            if idx + 1 >= len(df):
                continue

            entry = df.iloc[idx + 1]['open']
            tp_price = entry * (1 + V85_TP) if direction == 1 else entry * (1 - V85_TP)
            sl_price = entry * (1 - V85_SL) if direction == 1 else entry * (1 + V85_SL)

            # Check next candles
            pnl_pct = None
            exit_ts = None
            for j in range(idx + 1, min(idx + MAX_HOLD + 1, len(df))):
                candle = df.iloc[j]
                if direction == 1:
                    if candle['low'] <= sl_price:
                        pnl_pct = -V85_SL
                        exit_ts = candle.name
                        break
                    if candle['high'] >= tp_price:
                        pnl_pct = V85_TP
                        exit_ts = candle.name
                        break
                else:
                    if candle['high'] >= sl_price:
                        pnl_pct = -V85_SL
                        exit_ts = candle.name
                        break
                    if candle['low'] <= tp_price:
                        pnl_pct = V85_TP
                        exit_ts = candle.name
                        break

            if pnl_pct is None:
                exit_idx = min(idx + MAX_HOLD, len(df) - 1)
                exit_price = df.iloc[exit_idx]['close']
                pnl_pct = (exit_price - entry) / entry * direction
                exit_ts = df.index[exit_idx]

            trades.append({
                'pair': pair,
                'entry_ts': ts,
                'exit_ts': exit_ts,
                'direction': direction,
                'entry': entry,
                'pnl_pct': pnl_pct,
                'win': pnl_pct > 0
            })

    return trades


def run_backtest_v95_atr(all_data, all_preds, all_chop, all_atr, all_loss_features,
                          v95_detectors, v95_thresholds, test_start, test_end):
    """
    Backtest V9.5+ATR: Macro + Conviction + LossDetector + ATR dinamico
    """
    trades = []

    for pair in PAIRS:
        if pair not in all_preds or pair not in all_data:
            continue
        if pair not in v95_detectors:
            continue

        preds = all_preds[pair]
        df = all_data[pair]
        loss_feats = all_loss_features.get(pair, {})
        model = v95_detectors[pair]
        thresh = v95_thresholds[pair] + V95_THRESH_BOOST

        for ts, pred in preds.items():
            if ts < test_start or ts >= test_end:
                continue

            direction = pred.get('direction', 0)
            conviction = pred.get('conviction', 0)

            # Filtro 1: Conviction minimo
            if direction == 0 or abs(conviction) < V95_CONF_MIN:
                continue

            # Filtro 2: Choppiness
            chop_val = all_chop[pair].get(ts)
            if chop_val is None or chop_val >= V95_CHOP_MAX:
                continue

            # Filtro 3: LossDetector
            feat_row = loss_feats.get(ts)
            if feat_row is None:
                continue

            try:
                fcols = [c for c in feat_row.index if c != 'is_loss']
                p_loss = float(model.predict_proba(feat_row[fcols].values.reshape(1, -1))[0][1])
                if p_loss > thresh:
                    continue
            except:
                continue

            # Simular trade con ATR dinamico
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

            # Check next candles
            pnl_pct = None
            exit_ts = None
            for j in range(idx + 1, min(idx + MAX_HOLD + 1, len(df))):
                candle = df.iloc[j]
                if direction == 1:
                    if candle['low'] <= sl_price:
                        pnl_pct = -sl_pct
                        exit_ts = candle.name
                        break
                    if candle['high'] >= tp_price:
                        pnl_pct = tp_pct
                        exit_ts = candle.name
                        break
                else:
                    if candle['high'] >= sl_price:
                        pnl_pct = -sl_pct
                        exit_ts = candle.name
                        break
                    if candle['low'] <= tp_price:
                        pnl_pct = tp_pct
                        exit_ts = candle.name
                        break

            if pnl_pct is None:
                exit_idx = min(idx + MAX_HOLD, len(df) - 1)
                exit_price = df.iloc[exit_idx]['close']
                pnl_pct = (exit_price - entry) / entry * direction
                exit_ts = df.index[exit_idx]

            trades.append({
                'pair': pair,
                'entry_ts': ts,
                'exit_ts': exit_ts,
                'direction': direction,
                'entry': entry,
                'pnl_pct': pnl_pct,
                'win': pnl_pct > 0
            })

    return trades


def calculate_metrics(trades, period_name):
    """Calcula metricas de un conjunto de trades."""
    if not trades:
        return {
            'period': period_name,
            'trades': 0,
            'wins': 0,
            'wr': 0,
            'pnl': 0,
            'pf': 0,
            'return_pct': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_dd': 0,
        }

    n_trades = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n_trades * 100

    pnl = sum(t['pnl_pct'] * INITIAL_CAPITAL for t in trades)
    gross_profit = sum(t['pnl_pct'] * INITIAL_CAPITAL for t in trades if t['pnl_pct'] > 0)
    gross_loss = abs(sum(t['pnl_pct'] * INITIAL_CAPITAL for t in trades if t['pnl_pct'] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999

    return_pct = pnl / INITIAL_CAPITAL * 100

    winning_trades = [t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]
    losing_trades = [t['pnl_pct'] for t in trades if t['pnl_pct'] < 0]
    avg_win = np.mean(winning_trades) * 100 if winning_trades else 0
    avg_loss = np.mean(losing_trades) * 100 if losing_trades else 0

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
        'period': period_name,
        'trades': n_trades,
        'wins': wins,
        'wr': wr,
        'pnl': pnl,
        'pf': pf,
        'return_pct': return_pct,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_dd': max_dd,
    }


def print_comparison(v85_metrics, v95_metrics, period_name):
    """Imprime comparacion lado a lado."""
    print(f'\n{"="*70}')
    print(f'PERIODO: {period_name}')
    print(f'{"="*70}')
    print(f'{"Metrica":<20} {"V8.5":>15} {"V9.5+ATR":>15} {"Diferencia":>15}')
    print('-' * 70)

    metrics = [
        ('Trades', 'trades', '', 0),
        ('Wins', 'wins', '', 0),
        ('Win Rate', 'wr', '%', 1),
        ('PnL', 'pnl', '$', 0),
        ('Profit Factor', 'pf', '', 2),
        ('Return', 'return_pct', '%', 1),
        ('Avg Win', 'avg_win', '%', 2),
        ('Avg Loss', 'avg_loss', '%', 2),
        ('Max Drawdown', 'max_dd', '%', 1),
    ]

    for name, key, suffix, decimals in metrics:
        v85_val = v85_metrics[key]
        v95_val = v95_metrics[key]
        diff = v95_val - v85_val

        if suffix == '$':
            v85_str = f'${v85_val:,.0f}'
            v95_str = f'${v95_val:,.0f}'
            diff_str = f'{diff:+,.0f}'
        elif suffix == '%':
            v85_str = f'{v85_val:.{decimals}f}%'
            v95_str = f'{v95_val:.{decimals}f}%'
            diff_str = f'{diff:+.{decimals}f}%'
        else:
            v85_str = f'{v85_val:.{decimals}f}'
            v95_str = f'{v95_val:.{decimals}f}'
            diff_str = f'{diff:+.{decimals}f}'

        # Color indicator
        if key in ['wr', 'pnl', 'pf', 'return_pct', 'avg_win']:
            indicator = '++' if diff > 0 else '--' if diff < 0 else '=='
        elif key == 'max_dd':
            indicator = '++' if diff < 0 else '--' if diff > 0 else '=='
        else:
            indicator = ''

        print(f'{name:<20} {v85_str:>15} {v95_str:>15} {diff_str:>12} {indicator}')


def main():
    print('=' * 70)
    print('BACKTEST COMPARATIVO: V8.5 vs V9.5+ATR')
    print('=' * 70)

    # Periodos de prueba
    # 1. Ultimo ano: Feb 2025 - Feb 2026
    period1_start = pd.Timestamp('2025-02-01', tz='UTC')
    period1_end = pd.Timestamp('2026-02-24', tz='UTC')
    period1_name = 'Ultimo Ano (Feb 2025 - Feb 2026)'

    # 2. Ano dificil: 2022 (bear market crypto)
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
            print(f'  {pair}: {len(df)} rows ({df.index[0].date()} - {df.index[-1].date()})')

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

        # Get feature columns from model
        if hasattr(model, 'feature_names_in_'):
            fcols = list(model.feature_names_in_)
        else:
            fcols = [c for c in feat.columns if c in feat.columns]

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
    # PRECOMPUTAR INDICADORES PARA V9.5
    # =========================================================================
    print('\n[3] Precomputando indicadores V9.5...')
    all_chop = {}
    all_atr = {}
    all_loss_features = {}

    for pair, df in all_data.items():
        # Choppiness
        chop = compute_choppiness(df)
        all_chop[pair] = {ts: v for ts, v in zip(df.index, chop) if pd.notna(v)}

        # ATR
        atr = compute_atr(df)
        all_atr[pair] = {ts: v for ts, v in zip(df.index, atr) if pd.notna(v)}

    # Loss features (pre-computed if available)
    for pair in PAIRS:
        safe = pair.replace('/', '_')
        feat_path = MODELS_DIR / f'v95_{safe}_loss_features.pkl'
        if feat_path.exists():
            import pickle
            all_loss_features[pair] = pickle.load(open(feat_path, 'rb'))
        else:
            # Generate on the fly
            if pair not in all_data or pair not in all_preds:
                continue
            df = all_data[pair]
            preds = all_preds[pair]
            btc_df = all_data.get('BTC/USDT')

            loss_feats = {}
            for ts, pred in preds.items():
                try:
                    idx = df.index.get_loc(ts)
                    if idx < 50:
                        continue
                    df_slice = df.iloc[:idx+1]
                    btc_slice = btc_df.iloc[:idx+1] if btc_df is not None else None
                    feat = compute_loss_features(df_slice, pred['direction'], pred['conviction'], btc_slice)
                    loss_feats[ts] = feat
                except:
                    continue
            all_loss_features[pair] = loss_feats

    # =========================================================================
    # CARGAR LOSSDETECTORS V9.5
    # =========================================================================
    print('\n[4] Cargando LossDetectors V9.5...')
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
                # Threshold stored in pairs dict
                v95_thresholds[pair] = meta['pairs'].get(pair, {}).get('threshold', 0.5)
                print(f'  {pair}: threshold={v95_thresholds[pair]:.2f}')

    # =========================================================================
    # BACKTEST PERIODO 1: ULTIMO ANO
    # =========================================================================
    print(f'\n[5] Backtest {period1_name}...')

    print('  Running V8.5...')
    trades_v85_p1 = run_backtest_v85(all_data, all_preds, period1_start, period1_end)
    metrics_v85_p1 = calculate_metrics(trades_v85_p1, period1_name)

    print('  Running V9.5+ATR...')
    trades_v95_p1 = run_backtest_v95_atr(all_data, all_preds, all_chop, all_atr,
                                          all_loss_features, v95_detectors, v95_thresholds,
                                          period1_start, period1_end)
    metrics_v95_p1 = calculate_metrics(trades_v95_p1, period1_name)

    print_comparison(metrics_v85_p1, metrics_v95_p1, period1_name)

    # =========================================================================
    # BACKTEST PERIODO 2: BEAR MARKET 2022
    # =========================================================================
    print(f'\n[6] Backtest {period2_name}...')

    print('  Running V8.5...')
    trades_v85_p2 = run_backtest_v85(all_data, all_preds, period2_start, period2_end)
    metrics_v85_p2 = calculate_metrics(trades_v85_p2, period2_name)

    print('  Running V9.5+ATR...')
    trades_v95_p2 = run_backtest_v95_atr(all_data, all_preds, all_chop, all_atr,
                                          all_loss_features, v95_detectors, v95_thresholds,
                                          period2_start, period2_end)
    metrics_v95_p2 = calculate_metrics(trades_v95_p2, period2_name)

    print_comparison(metrics_v85_p2, metrics_v95_p2, period2_name)

    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    elapsed = time.time() - t0

    print('\n' + '=' * 70)
    print('RESUMEN FINAL')
    print('=' * 70)

    print(f'\n{"Periodo":<25} {"Estrategia":<15} {"Trades":>8} {"WR":>8} {"PnL":>10} {"PF":>6} {"DD":>8}')
    print('-' * 70)

    for m in [metrics_v85_p1, metrics_v95_p1, metrics_v85_p2, metrics_v95_p2]:
        strat = 'V8.5' if 'v85' in str(m) else 'V9.5+ATR'
        # Determine strategy from context
        if m == metrics_v85_p1 or m == metrics_v85_p2:
            strat = 'V8.5'
        else:
            strat = 'V9.5+ATR'

        period_short = 'Ultimo Ano' if '2025' in m['period'] else 'Bear 2022'
        print(f'{period_short:<25} {strat:<15} {m["trades"]:>8} {m["wr"]:>7.1f}% {m["pnl"]:>9.0f}$ {m["pf"]:>6.2f} {m["max_dd"]:>7.1f}%')

    # Ganador
    print('\n' + '-' * 70)
    total_v85 = metrics_v85_p1['pnl'] + metrics_v85_p2['pnl']
    total_v95 = metrics_v95_p1['pnl'] + metrics_v95_p2['pnl']

    avg_wr_v85 = (metrics_v85_p1['wr'] + metrics_v85_p2['wr']) / 2
    avg_wr_v95 = (metrics_v95_p1['wr'] + metrics_v95_p2['wr']) / 2

    print(f'\nTOTAL COMBINADO:')
    print(f'  V8.5:     PnL=${total_v85:,.0f} | WR avg={avg_wr_v85:.1f}%')
    print(f'  V9.5+ATR: PnL=${total_v95:,.0f} | WR avg={avg_wr_v95:.1f}%')

    winner = 'V9.5+ATR' if total_v95 > total_v85 else 'V8.5'
    diff_pct = abs(total_v95 - total_v85) / max(abs(total_v85), 1) * 100
    print(f'\n  GANADOR: {winner} (+{diff_pct:.1f}% PnL)')

    print(f'\n[Completado en {elapsed:.1f}s]')

    # Save results
    results = {
        'period1': {
            'name': period1_name,
            'v85': metrics_v85_p1,
            'v95_atr': metrics_v95_p1,
        },
        'period2': {
            'name': period2_name,
            'v85': metrics_v85_p2,
            'v95_atr': metrics_v95_p2,
        },
        'summary': {
            'total_pnl_v85': total_v85,
            'total_pnl_v95': total_v95,
            'winner': winner,
        }
    }

    with open(MODELS_DIR / 'backtest_comparison_v85_v95atr.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f'\nResultados guardados en: {MODELS_DIR}/backtest_comparison_v85_v95atr.json')


if __name__ == '__main__':
    main()
