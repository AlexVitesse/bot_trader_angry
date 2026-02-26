"""
Backtest Experiments V2 - Maximizar WR sin perder PnL
=====================================================
Partiendo de la mejor config: thresh+0.15 & conf>=1.5 & chop<60
Objetivo: WR >= 45% manteniendo PnL >= $1000

Uso: poetry run python backtest_experiments_v2.py
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

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT',
    'NEAR/USDT',
]

# Pares con peor performance historico
WEAK_PAIRS = ['SOL/USDT', 'BTC/USDT', 'ADA/USDT']

TP_PCT = 0.03
SL_PCT = 0.015
MAX_HOLD = 30
INITIAL_CAPITAL = 500.0
RISK_PER_TRADE = 0.02
MAX_POSITIONS = 3

LOSS_FEATURES = [
    'cs_conf', 'cs_pred_mag', 'cs_macro_score', 'cs_risk_off',
    'cs_regime_bull', 'cs_regime_bear', 'cs_regime_range',
    'cs_atr_pct', 'cs_n_open', 'cs_pred_sign',
    'ld_conviction_pred',
    'ld_pair_rsi14', 'ld_pair_bb_pct', 'ld_pair_vol_ratio',
    'ld_pair_ret_5', 'ld_pair_ret_20',
    'ld_btc_ret_5', 'ld_btc_rsi14', 'ld_btc_vol20',
    'ld_hour', 'ld_tp_sl_ratio',
]


def load_data(pair):
    safe = pair.replace('/', '_')
    for suffix in ['_4h_v95.parquet', '_v95.parquet', '_4h.parquet', '.parquet']:
        cache = DATA_DIR / f'{safe}{suffix}'
        if cache.exists():
            return pd.read_parquet(cache)
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

    return feat


def compute_choppiness(df, length=14):
    atr = ta.atr(df['high'], df['low'], df['close'], length=1)
    atr_sum = atr.rolling(length).sum()
    high_max = df['high'].rolling(length).max()
    low_min = df['low'].rolling(length).min()
    chop = 100 * np.log10(atr_sum / (high_max - low_min + 1e-10)) / np.log10(length)
    return chop


def compute_adx(df):
    ax = ta.adx(df['high'], df['low'], df['close'], length=14)
    if ax is not None:
        return ax.iloc[:, 0]
    return pd.Series(25, index=df.index)


def precompute_loss_features(df, btc_df):
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


def detect_regime(btc_df):
    c = btc_df['close']
    ema20 = ta.ema(c, length=20)
    ema50 = ta.ema(c, length=50)
    ret20 = c.pct_change(20)

    regime = pd.Series('RANGE', index=btc_df.index)
    regime[(c > ema20) & (ema20 > ema50) & (ret20 > 0.05)] = 'BULL'
    regime[(c < ema20) & (ema20 < ema50) & (ret20 < -0.05)] = 'BEAR'
    return regime


def run_backtest(all_data, all_preds, all_loss_features, all_chop, all_adx,
                 regime_series, test_start,
                 loss_detectors_per_pair, thresholds_per_pair,
                 conf_threshold=1.5,
                 threshold_boost=0.15,
                 chop_filter=60,
                 adx_filter=None,
                 exclude_pairs=None,
                 only_pairs=None,
                 hour_filter=None,
                 rsi_filter=None):
    """Run backtest with advanced filters."""

    exclude_pairs = exclude_pairs or []

    capital = INITIAL_CAPITAL
    positions = {}
    trades = []

    all_times = set()
    for pair, df in all_data.items():
        if pair in exclude_pairs:
            continue
        if only_pairs and pair not in only_pairs:
            continue
        valid_times = df[df.index >= test_start].index.tolist()
        all_times.update(valid_times)
    timeline = sorted(all_times)

    for ts in timeline:
        # Update positions
        for pair in list(positions.keys()):
            if pair not in all_data:
                continue
            df = all_data[pair]
            if ts not in df.index:
                continue

            pos = positions[pair]
            pos['hold_bars'] += 1

            price = df.loc[ts, 'close']
            entry = pos['entry_price']
            d = pos['direction']

            pnl_pct = (price - entry) / entry if d == 1 else (entry - price) / entry

            exit_reason = None
            if pnl_pct >= TP_PCT:
                exit_reason = 'TP'
            elif pnl_pct <= -SL_PCT:
                exit_reason = 'SL'
            elif pos['hold_bars'] >= MAX_HOLD:
                exit_reason = 'TIME'

            if exit_reason:
                pnl_usd = pos['size'] * pnl_pct
                capital += pnl_usd
                trades.append({'pair': pair, 'pnl': pnl_usd, 'reason': exit_reason})
                del positions[pair]

        if len(positions) >= MAX_POSITIONS:
            continue

        for pair in all_data.keys():
            if pair in exclude_pairs:
                continue
            if only_pairs and pair not in only_pairs:
                continue
            if pair in positions:
                continue
            if pair not in all_preds:
                continue

            preds_df = all_preds[pair]
            if ts not in preds_df.index:
                continue

            pred = preds_df.loc[ts, 'pred']
            conf = preds_df.loc[ts, 'conf']

            if conf < conf_threshold:
                continue

            direction = 1 if pred > 0 else -1

            regime = regime_series.get(ts, 'RANGE') if ts in regime_series.index else 'RANGE'
            if regime == 'BULL' and direction == -1:
                continue
            if regime == 'BEAR' and direction == 1:
                continue

            # Choppiness filter
            if chop_filter is not None and pair in all_chop:
                chop_val = all_chop[pair].get(ts, 50)
                if not np.isnan(chop_val) and chop_val > chop_filter:
                    continue

            # ADX filter (tendencia fuerte)
            if adx_filter is not None and pair in all_adx:
                adx_val = all_adx[pair].get(ts, 25)
                if not np.isnan(adx_val) and adx_val < adx_filter:
                    continue

            # Hour filter
            if hour_filter is not None:
                hour = ts.hour
                if hour not in hour_filter:
                    continue

            # RSI filter (evitar extremos)
            if rsi_filter is not None and pair in all_loss_features:
                rsi_val = all_loss_features[pair].loc[ts, 'ld_pair_rsi14'] * 100
                rsi_low, rsi_high = rsi_filter
                if rsi_val < rsi_low or rsi_val > rsi_high:
                    continue

            price = all_data[pair].loc[ts, 'close']

            # LossDetector filter
            skip = False
            if pair in loss_detectors_per_pair:
                model = loss_detectors_per_pair[pair]
                threshold = thresholds_per_pair.get(pair, 0.55) + threshold_boost
                threshold = min(threshold, 0.90)

                lf = all_loss_features[pair].loc[ts]
                features = {
                    'cs_conf': conf, 'cs_pred_mag': abs(pred),
                    'cs_macro_score': 0.5, 'cs_risk_off': 1.0,
                    'cs_regime_bull': 1.0 if regime == 'BULL' else 0.0,
                    'cs_regime_bear': 1.0 if regime == 'BEAR' else 0.0,
                    'cs_regime_range': 1.0 if regime == 'RANGE' else 0.0,
                    'cs_atr_pct': lf['atr_pct'], 'cs_n_open': len(positions),
                    'cs_pred_sign': float(direction),
                    'ld_conviction_pred': pred,
                    'ld_pair_rsi14': lf['ld_pair_rsi14'],
                    'ld_pair_bb_pct': lf['ld_pair_bb_pct'],
                    'ld_pair_vol_ratio': lf['ld_pair_vol_ratio'],
                    'ld_pair_ret_5': lf['ld_pair_ret_5'],
                    'ld_pair_ret_20': lf['ld_pair_ret_20'],
                    'ld_btc_ret_5': lf['ld_btc_ret_5'],
                    'ld_btc_rsi14': lf['ld_btc_rsi14'],
                    'ld_btc_vol20': lf['ld_btc_vol20'],
                    'ld_hour': ts.hour / 24.0,
                    'ld_tp_sl_ratio': TP_PCT / SL_PCT,
                }
                df_feat = pd.DataFrame([features])
                cols = [c for c in LOSS_FEATURES if c in df_feat.columns]
                if cols:
                    p_loss = float(model.predict_proba(df_feat[cols])[0][1])
                    skip = p_loss > threshold

            if skip:
                continue

            position_size = capital * RISK_PER_TRADE / SL_PCT
            position_size = min(position_size, capital * 0.3)

            positions[pair] = {
                'entry_time': ts, 'entry_price': price,
                'direction': direction, 'size': position_size,
                'hold_bars': 0,
            }

    # Close remaining
    for pair in list(positions.keys()):
        if pair in all_data:
            df = all_data[pair]
            last_price = df['close'].iloc[-1]
            pos = positions[pair]
            d = pos['direction']
            pnl_pct = (last_price - pos['entry_price']) / pos['entry_price'] if d == 1 else (pos['entry_price'] - last_price) / pos['entry_price']
            pnl_usd = pos['size'] * pnl_pct
            capital += pnl_usd
            trades.append({'pair': pair, 'pnl': pnl_usd, 'reason': 'END'})

    if not trades:
        return {'trades': 0, 'wins': 0, 'wr': 0, 'pnl': 0, 'pf': 0, 'return_pct': 0}

    df_trades = pd.DataFrame(trades)
    wins = (df_trades['pnl'] > 0).sum()
    total_pnl = df_trades['pnl'].sum()
    gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df_trades[df_trades['pnl'] <= 0]['pnl'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    return_pct = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    return {
        'trades': len(df_trades),
        'wins': wins,
        'wr': wins / len(df_trades) * 100,
        'pnl': total_pnl,
        'pf': pf,
        'return_pct': return_pct,
    }


def main():
    t0 = time.time()
    print('=' * 80)
    print('EXPERIMENTOS V2: Maximizar WR manteniendo PnL >= $1000')
    print('=' * 80)
    print('Baseline: thresh+0.15 & conf>=1.5 & chop<60 (WR=42.5%, PnL=$1358)')

    # Load data
    print('\n[1] Cargando datos...')
    all_data = {}
    all_chop = {}
    all_adx = {}
    for pair in PAIRS:
        df = load_data(pair)
        if df is not None:
            all_data[pair] = df
            all_chop[pair] = compute_choppiness(df)
            all_adx[pair] = compute_adx(df)

    btc_df = all_data.get('BTC/USDT')
    regime_series = detect_regime(btc_df)

    print('\n[2] Precomputando...')
    with open(MODELS_DIR / 'v7_meta.json') as f:
        v7_meta = json.load(f)
    fcols = v7_meta.get('feature_cols', [])
    pred_stds = v7_meta.get('pred_stds', {})

    all_preds = {}
    all_loss_features = {}

    for pair in PAIRS:
        if pair not in all_data:
            continue

        safe = pair.replace('/', '_')
        model = None
        for prefix in ['v95_v7_', 'v7_']:
            model_path = MODELS_DIR / f'{prefix}{safe}.pkl'
            if model_path.exists():
                model = joblib.load(model_path)
                break

        if model is None:
            continue

        df = all_data[pair]
        feat = compute_features(df).replace([np.inf, -np.inf], np.nan)

        X = feat[fcols].fillna(0)
        preds = model.predict(X)
        ps = pred_stds.get(pair, np.std(preds))
        if ps < 1e-8:
            ps = 0.01

        all_preds[pair] = pd.DataFrame({
            'pred': preds,
            'conf': np.abs(preds) / ps,
        }, index=df.index)

        all_loss_features[pair] = precompute_loss_features(df, btc_df)

    print('\n[3] Cargando LossDetectors...')
    v95_detectors = {}
    v95_thresholds = {}

    with open(MODELS_DIR / 'v95_meta.json') as f:
        v95_meta = json.load(f)

    for pair, info in v95_meta.get('pairs', {}).items():
        safe = pair.replace('/', '')
        model_path = MODELS_DIR / f'v95_ld_{safe}.pkl'
        if model_path.exists():
            v95_detectors[pair] = joblib.load(model_path)
            v95_thresholds[pair] = info.get('threshold', 0.55)

    min_date = min(df.index.min() for df in all_data.values())
    max_date = max(df.index.max() for df in all_data.values())
    test_start = min_date + (max_date - min_date) * 0.8

    print('\n[4] Ejecutando experimentos...')
    print('-' * 80)
    results = []

    # BASELINE (mejor anterior)
    print('  [BASELINE] thresh+0.15 & conf>=1.5 & chop<60...', end=' ', flush=True)
    r = run_backtest(all_data, all_preds, all_loss_features, all_chop, all_adx,
                     regime_series, test_start, v95_detectors, v95_thresholds,
                     conf_threshold=1.5, threshold_boost=0.15, chop_filter=60)
    r['name'] = 'BASELINE'
    results.append(r)
    print(f'WR={r["wr"]:.1f}%, PnL=${r["pnl"]:.0f}')

    # ========================================================================
    # EXPERIMENTOS: Subir confidence aun mas
    # ========================================================================
    print('\n  [EXP A] Confidence mas alta...')
    for conf in [1.8, 2.0, 2.5, 3.0]:
        print(f'    conf>={conf}...', end=' ', flush=True)
        r = run_backtest(all_data, all_preds, all_loss_features, all_chop, all_adx,
                         regime_series, test_start, v95_detectors, v95_thresholds,
                         conf_threshold=conf, threshold_boost=0.15, chop_filter=60)
        r['name'] = f'conf>={conf}'
        results.append(r)
        print(f'WR={r["wr"]:.1f}%, PnL=${r["pnl"]:.0f}')

    # ========================================================================
    # EXPERIMENTOS: Chop mas restrictivo
    # ========================================================================
    print('\n  [EXP B] Chop mas restrictivo...')
    for chop in [55, 50, 45]:
        print(f'    chop<{chop}...', end=' ', flush=True)
        r = run_backtest(all_data, all_preds, all_loss_features, all_chop, all_adx,
                         regime_series, test_start, v95_detectors, v95_thresholds,
                         conf_threshold=1.5, threshold_boost=0.15, chop_filter=chop)
        r['name'] = f'chop<{chop}'
        results.append(r)
        print(f'WR={r["wr"]:.1f}%, PnL=${r["pnl"]:.0f}')

    # ========================================================================
    # EXPERIMENTOS: ADX filter (solo tendencias fuertes)
    # ========================================================================
    print('\n  [EXP C] ADX filter (tendencia fuerte)...')
    for adx in [20, 25, 30]:
        print(f'    adx>{adx}...', end=' ', flush=True)
        r = run_backtest(all_data, all_preds, all_loss_features, all_chop, all_adx,
                         regime_series, test_start, v95_detectors, v95_thresholds,
                         conf_threshold=1.5, threshold_boost=0.15, chop_filter=60,
                         adx_filter=adx)
        r['name'] = f'adx>{adx}'
        results.append(r)
        print(f'WR={r["wr"]:.1f}%, PnL=${r["pnl"]:.0f}')

    # ========================================================================
    # EXPERIMENTOS: Excluir pares debiles
    # ========================================================================
    print('\n  [EXP D] Excluir pares debiles...')
    exclusions = [
        (['SOL/USDT'], 'no_SOL'),
        (['BTC/USDT'], 'no_BTC'),
        (['SOL/USDT', 'BTC/USDT'], 'no_SOL_BTC'),
        (['SOL/USDT', 'BTC/USDT', 'ADA/USDT'], 'no_weak'),
    ]
    for exclude, name in exclusions:
        print(f'    {name}...', end=' ', flush=True)
        r = run_backtest(all_data, all_preds, all_loss_features, all_chop, all_adx,
                         regime_series, test_start, v95_detectors, v95_thresholds,
                         conf_threshold=1.5, threshold_boost=0.15, chop_filter=60,
                         exclude_pairs=exclude)
        r['name'] = name
        results.append(r)
        print(f'WR={r["wr"]:.1f}%, PnL=${r["pnl"]:.0f}')

    # ========================================================================
    # EXPERIMENTOS: Solo mejores pares
    # ========================================================================
    print('\n  [EXP E] Solo mejores pares (ETH, XRP, DOGE, LINK, DOT, NEAR)...')
    best_pairs = ['ETH/USDT', 'XRP/USDT', 'DOGE/USDT', 'LINK/USDT', 'DOT/USDT', 'NEAR/USDT']
    print(f'    only_best...', end=' ', flush=True)
    r = run_backtest(all_data, all_preds, all_loss_features, all_chop, all_adx,
                     regime_series, test_start, v95_detectors, v95_thresholds,
                     conf_threshold=1.5, threshold_boost=0.15, chop_filter=60,
                     only_pairs=best_pairs)
    r['name'] = 'only_best_6'
    results.append(r)
    print(f'WR={r["wr"]:.1f}%, PnL=${r["pnl"]:.0f}')

    # ========================================================================
    # EXPERIMENTOS: RSI filter (evitar extremos)
    # ========================================================================
    print('\n  [EXP F] RSI filter (evitar extremos)...')
    rsi_ranges = [(30, 70), (25, 75), (20, 80)]
    for rsi_low, rsi_high in rsi_ranges:
        print(f'    rsi {rsi_low}-{rsi_high}...', end=' ', flush=True)
        r = run_backtest(all_data, all_preds, all_loss_features, all_chop, all_adx,
                         regime_series, test_start, v95_detectors, v95_thresholds,
                         conf_threshold=1.5, threshold_boost=0.15, chop_filter=60,
                         rsi_filter=(rsi_low, rsi_high))
        r['name'] = f'rsi_{rsi_low}_{rsi_high}'
        results.append(r)
        print(f'WR={r["wr"]:.1f}%, PnL=${r["pnl"]:.0f}')

    # ========================================================================
    # EXPERIMENTOS: Combinaciones prometedoras
    # ========================================================================
    print('\n  [EXP G] Combinaciones prometedoras...')
    combos = [
        {'conf_threshold': 2.0, 'chop_filter': 55, 'name': 'conf2.0+chop55'},
        {'conf_threshold': 2.0, 'adx_filter': 25, 'name': 'conf2.0+adx25'},
        {'conf_threshold': 1.8, 'chop_filter': 50, 'name': 'conf1.8+chop50'},
        {'conf_threshold': 1.8, 'exclude_pairs': ['SOL/USDT', 'BTC/USDT'], 'name': 'conf1.8+no_weak2'},
        {'conf_threshold': 2.0, 'chop_filter': 55, 'adx_filter': 20, 'name': 'conf2.0+chop55+adx20'},
        {'conf_threshold': 2.0, 'only_pairs': best_pairs, 'name': 'conf2.0+best6'},
        {'conf_threshold': 2.5, 'chop_filter': 50, 'name': 'conf2.5+chop50'},
        {'conf_threshold': 1.5, 'chop_filter': 50, 'adx_filter': 25, 'name': 'chop50+adx25'},
    ]

    for combo in combos:
        name = combo.pop('name')
        print(f'    {name}...', end=' ', flush=True)
        r = run_backtest(all_data, all_preds, all_loss_features, all_chop, all_adx,
                         regime_series, test_start, v95_detectors, v95_thresholds,
                         threshold_boost=0.15, **combo)
        r['name'] = name
        combo['name'] = name
        results.append(r)
        print(f'WR={r["wr"]:.1f}%, PnL=${r["pnl"]:.0f}')

    # ========================================================================
    # RESULTADOS
    # ========================================================================
    elapsed = time.time() - t0
    print(f'\n[Completado en {elapsed:.1f}s]')

    # Filtrar resultados con PnL >= $800 (mantener beneficio)
    good_results = [r for r in results if r['pnl'] >= 800]
    good_results_sorted = sorted(good_results, key=lambda x: x['wr'], reverse=True)

    print('\n' + '=' * 80)
    print('RESULTADOS CON PnL >= $800 (ordenados por WR)')
    print('=' * 80)
    print(f'{"#":<3} {"Config":<30} {"Trades":>7} {"WR%":>7} {"PnL":>9} {"PF":>6}')
    print('-' * 80)

    for i, r in enumerate(good_results_sorted[:15], 1):
        marker = '*' if r['name'] == 'BASELINE' else ' '
        print(f'{marker}{i:<2} {r["name"]:<30} {r["trades"]:>7} {r["wr"]:>6.1f}% '
              f'{r["pnl"]:>9.0f} {r["pf"]:>6.2f}')

    # Mejor WR con PnL >= $1000
    best_wr_1000 = [r for r in results if r['pnl'] >= 1000]
    if best_wr_1000:
        best = max(best_wr_1000, key=lambda x: x['wr'])
        print('\n' + '=' * 80)
        print(f'MEJOR WR CON PnL >= $1000:')
        print(f'  Config: {best["name"]}')
        print(f'  WR: {best["wr"]:.1f}% | Trades: {best["trades"]} | PnL: ${best["pnl"]:.0f} | PF: {best["pf"]:.2f}')

    # Save
    with open(MODELS_DIR / 'experiment_results_v2.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == '__main__':
    main()
