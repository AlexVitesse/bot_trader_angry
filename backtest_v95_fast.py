"""
Backtest Comparativo V9.5 vs V9 vs V8.5 (OPTIMIZADO)
=====================================================
Version rapida que precomputa features.

Uso: poetry run python backtest_v95_fast.py
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

# ============================================================================
# CONFIGURACION
# ============================================================================
DATA_DIR = Path('data')
MODELS_DIR = Path('models')

PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT',
    'NEAR/USDT',
]

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


def load_data(pair, timeframe='4h'):
    """Carga datos desde cache."""
    safe = pair.replace('/', '_')
    for suffix in ['_4h_v95.parquet', '_v95.parquet', '_4h.parquet', '.parquet']:
        cache = DATA_DIR / f'{safe}{suffix}'
        if cache.exists():
            return pd.read_parquet(cache)
    return None


def compute_features(df):
    """Features V7."""
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


def precompute_loss_features(df, btc_df):
    """Precompute LossDetector features for all bars."""
    c = df['close']

    # Pair TA
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

    # BTC context (aligned to pair index)
    btc_c = btc_df['close']
    btc_rsi = ta.rsi(btc_c, length=14) / 100.0
    btc_vol = btc_c.pct_change().rolling(20).std()
    btc_ret_5 = btc_c.pct_change(5)

    # Align BTC to pair index
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
    """Detecta regimen de mercado."""
    c = btc_df['close']
    ema20 = ta.ema(c, length=20)
    ema50 = ta.ema(c, length=50)
    ret20 = c.pct_change(20)

    regime = pd.Series('RANGE', index=btc_df.index)
    regime[(c > ema20) & (ema20 > ema50) & (ret20 > 0.05)] = 'BULL'
    regime[(c < ema20) & (ema20 < ema50) & (ret20 < -0.05)] = 'BEAR'
    return regime


def run_backtest(strategy_name, all_data, all_features, all_preds, all_loss_features,
                 regime_series, test_start,
                 loss_detector=None, loss_threshold=0.55,
                 loss_detectors_per_pair=None, thresholds_per_pair=None):
    """Run backtest - optimized version."""

    loss_detectors_per_pair = loss_detectors_per_pair or {}
    thresholds_per_pair = thresholds_per_pair or {}

    capital = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    positions = {}
    trades = []

    # Build unified timeline from test period
    all_times = set()
    for pair, df in all_data.items():
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
                trades.append({
                    'pair': pair, 'entry_time': pos['entry_time'],
                    'exit_time': ts, 'direction': d,
                    'pnl_pct': pnl_pct, 'pnl': pnl_usd, 'reason': exit_reason,
                })
                del positions[pair]

        peak = max(peak, capital)

        # Generate signals
        if len(positions) >= MAX_POSITIONS:
            continue

        for pair in all_data.keys():
            if pair in positions:
                continue
            if pair not in all_preds:
                continue

            preds_df = all_preds[pair]
            if ts not in preds_df.index:
                continue

            pred = preds_df.loc[ts, 'pred']
            conf = preds_df.loc[ts, 'conf']

            if conf < 0.7:
                continue

            direction = 1 if pred > 0 else -1

            # Regime filter
            regime = regime_series.get(ts, 'RANGE') if ts in regime_series.index else 'RANGE'
            if regime == 'BULL' and direction == -1:
                continue
            if regime == 'BEAR' and direction == 1:
                continue

            price = all_data[pair].loc[ts, 'close']

            # LossDetector filter
            skip = False
            if pair in loss_detectors_per_pair:
                # V9.5
                model = loss_detectors_per_pair[pair]
                threshold = thresholds_per_pair.get(pair, 0.55)
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

            elif loss_detector is not None:
                # V9 generic
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
                    p_loss = float(loss_detector.predict_proba(df_feat[cols])[0][1])
                    skip = p_loss > loss_threshold

            if skip:
                continue

            # Open position
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
            last_ts = df.index[-1]
            last_price = df['close'].iloc[-1]
            pos = positions[pair]
            d = pos['direction']
            pnl_pct = (last_price - pos['entry_price']) / pos['entry_price'] if d == 1 else (pos['entry_price'] - last_price) / pos['entry_price']
            pnl_usd = pos['size'] * pnl_pct
            capital += pnl_usd
            trades.append({
                'pair': pair, 'entry_time': pos['entry_time'],
                'exit_time': last_ts, 'direction': d,
                'pnl_pct': pnl_pct, 'pnl': pnl_usd, 'reason': 'END',
            })

    # Metrics
    if not trades:
        return {'strategy': strategy_name, 'trades': 0, 'wins': 0, 'wr': 0, 'pnl': 0, 'pf': 0, 'dd': 0, 'return_pct': 0}

    df_trades = pd.DataFrame(trades)
    wins = (df_trades['pnl'] > 0).sum()
    total_pnl = df_trades['pnl'].sum()
    gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df_trades[df_trades['pnl'] <= 0]['pnl'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    return_pct = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    return {
        'strategy': strategy_name,
        'trades': len(df_trades),
        'wins': wins,
        'wr': wins / len(df_trades) * 100,
        'pnl': total_pnl,
        'pf': pf,
        'return_pct': return_pct,
        'final_capital': capital,
    }


def main():
    t0 = time.time()
    print('=' * 70)
    print('BACKTEST COMPARATIVO OPTIMIZADO: V9.5 vs V9 vs V8.5')
    print('=' * 70)

    # Load data
    print('\n[1] Cargando datos...')
    all_data = {}
    for pair in PAIRS:
        df = load_data(pair)
        if df is not None:
            all_data[pair] = df
            print(f'  {pair}: {len(df)} velas')

    if not all_data:
        print('ERROR: No hay datos.')
        return

    btc_df = all_data.get('BTC/USDT')
    if btc_df is None:
        print('ERROR: BTC data required')
        return

    regime_series = detect_regime(btc_df)

    # Load V7 models and precompute predictions
    print('\n[2] Cargando modelos y precomputando predicciones...')

    v7_meta_path = MODELS_DIR / 'v7_meta.json'
    if not v7_meta_path.exists():
        print('ERROR: v7_meta.json not found')
        return

    with open(v7_meta_path) as f:
        v7_meta = json.load(f)

    fcols = v7_meta.get('feature_cols', [])
    pred_stds = v7_meta.get('pred_stds', {})

    all_features = {}
    all_preds = {}
    all_loss_features = {}

    for pair in PAIRS:
        if pair not in all_data:
            continue

        # Load model (try V9.5 first, then standard)
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

        # Compute features once
        feat = compute_features(df)
        feat = feat.replace([np.inf, -np.inf], np.nan)
        all_features[pair] = feat

        # Compute predictions for all bars
        X = feat[fcols].fillna(0)
        preds = model.predict(X)
        ps = pred_stds.get(pair, np.std(preds))
        if ps < 1e-8:
            ps = 0.01

        all_preds[pair] = pd.DataFrame({
            'pred': preds,
            'conf': np.abs(preds) / ps,
        }, index=df.index)

        # Precompute loss features
        all_loss_features[pair] = precompute_loss_features(df, btc_df)

        print(f'  {pair}: OK')

    # Load LossDetectors
    print('\n[3] Cargando LossDetectors...')

    v9_loss_detector = None
    v9_threshold = 0.55
    v9_model_path = MODELS_DIR / 'v9_loss_detector.pkl'
    v9_meta_path = MODELS_DIR / 'v9_meta.json'

    if v9_model_path.exists():
        v9_loss_detector = joblib.load(v9_model_path)
        if v9_meta_path.exists():
            with open(v9_meta_path) as f:
                meta = json.load(f)
            v9_threshold = meta.get('loss_threshold', 0.55)
        print(f'  V9 generic: OK (threshold={v9_threshold})')

    v95_detectors = {}
    v95_thresholds = {}
    v95_meta_path = MODELS_DIR / 'v95_meta.json'

    if v95_meta_path.exists():
        with open(v95_meta_path) as f:
            v95_meta = json.load(f)
        for pair, info in v95_meta.get('pairs', {}).items():
            safe = pair.replace('/', '')
            model_path = MODELS_DIR / f'v95_ld_{safe}.pkl'
            if model_path.exists():
                v95_detectors[pair] = joblib.load(model_path)
                v95_thresholds[pair] = info.get('threshold', 0.55)
        print(f'  V9.5 per-pair: {len(v95_detectors)} modelos')

    # Test period
    min_date = min(df.index.min() for df in all_data.values())
    max_date = max(df.index.max() for df in all_data.values())
    test_start = min_date + (max_date - min_date) * 0.8
    print(f'\n  Test period: {test_start.date()} to {max_date.date()}')

    # Run backtests
    print('\n[4] Ejecutando backtests...')
    results = []

    print('  Running V8.5...', end=' ', flush=True)
    r = run_backtest('V8.5', all_data, all_features, all_preds, all_loss_features,
                     regime_series, test_start)
    results.append(r)
    print(f'{r["trades"]} trades, WR={r["wr"]:.1f}%, PnL=${r["pnl"]:.2f}')

    if v9_loss_detector is not None:
        print('  Running V9...', end=' ', flush=True)
        r = run_backtest('V9', all_data, all_features, all_preds, all_loss_features,
                         regime_series, test_start,
                         loss_detector=v9_loss_detector, loss_threshold=v9_threshold)
        results.append(r)
        print(f'{r["trades"]} trades, WR={r["wr"]:.1f}%, PnL=${r["pnl"]:.2f}')

    if v95_detectors:
        print('  Running V9.5...', end=' ', flush=True)
        r = run_backtest('V9.5', all_data, all_features, all_preds, all_loss_features,
                         regime_series, test_start,
                         loss_detectors_per_pair=v95_detectors,
                         thresholds_per_pair=v95_thresholds)
        results.append(r)
        print(f'{r["trades"]} trades, WR={r["wr"]:.1f}%, PnL=${r["pnl"]:.2f}')

    # Results
    elapsed = time.time() - t0
    print(f'\n[Completado en {elapsed:.1f}s]')

    print('\n' + '=' * 70)
    print('RESULTADOS COMPARATIVOS')
    print('=' * 70)
    print(f'{"Strategy":<10} {"Trades":>8} {"Wins":>6} {"WR%":>8} '
          f'{"PnL":>10} {"PF":>6} {"Ret%":>10}')
    print('-' * 70)

    for r in results:
        print(f'{r["strategy"]:<10} {r["trades"]:>8} {r["wins"]:>6} '
              f'{r["wr"]:>7.1f}% {r["pnl"]:>10.2f} {r["pf"]:>6.2f} '
              f'{r["return_pct"]:>9.1f}%')

    if results:
        best = max(results, key=lambda x: x['pnl'])
        print('-' * 70)
        print(f'GANADOR: {best["strategy"]} con ${best["pnl"]:.2f} '
              f'({best["return_pct"]:.1f}% return)')

    # Save
    with open(MODELS_DIR / 'backtest_comparison.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == '__main__':
    main()
