"""
evaluate_eth_options.py — ETH: 3 estrategias alternativas
==========================================================
Opcion 1: 1D ML (menos ruido, features ETH-especificos)
Opcion 2: Rule-based 4H (Breakout B + Pullback EMA20, sin ML)
Opcion 3: BTC-follower (entrar ETH cuando BTC da senal + correlacion alta)

Usage:
  python evaluate_eth_options.py
"""

import sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pandas_ta as pta
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from v15_framework import (
    load_pair_4h, load_btc_4h, load_pair_1d, load_btc_1d,
    compute_features_4h, compute_macro_daily, merge_daily_to_4h,
    sim_trade_fixed, metrics, print_metrics,
    WF_FOLDS, OOS_START, OOS_END, COMMISSION,
)

# ============================================================
# SHARED: regime detection + short sim
# ============================================================
REGIME_DEAD_ZONE = 0.02

def detect_regime(row):
    ema20 = row.get('ema20_1d', None)
    ema50 = row.get('ema50_1d', None)
    ema200 = row.get('ema200_1d', None)
    close = row.get('close', None)
    if ema20 is None or ema50 is None or pd.isna(ema20) or pd.isna(ema50):
        return 'RANGE'
    dist = (ema20 - ema50) / ema50
    if dist > REGIME_DEAD_ZONE:
        return 'BULL'
    elif dist < -REGIME_DEAD_ZONE:
        if ema200 is not None and close is not None and not pd.isna(ema200):
            if close > ema200:
                return 'RANGE'
        if close is not None and not pd.isna(close):
            if close > ema50:
                return 'RANGE'
        return 'BEAR'
    return 'RANGE'


def sim_short(df, entry_bar, entry_price, tp_pct, sl_pct, max_bars=16):
    tp = entry_price * (1 - tp_pct)
    sl = entry_price * (1 + sl_pct)
    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df):
            ep = float(df['close'].iloc[-1])
            pnl = (entry_price - ep) / entry_price - 2 * COMMISSION
            return ('TP' if ep < entry_price else 'SL'), ep, pnl, i
        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        if hi >= sl:
            pnl = -sl_pct - 2 * COMMISSION
            if lo <= tp and float(df['close'].iloc[b]) < (sl + tp) / 2:
                pnl = tp_pct - 2 * COMMISSION
                return 'TP', tp, pnl, i
            return 'SL', sl, pnl, i
        if lo <= tp:
            pnl = tp_pct - 2 * COMMISSION
            return 'TP', tp, pnl, i
    ep = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (entry_price - ep) / entry_price - 2 * COMMISSION
    return ('TP' if ep < entry_price else 'SL'), ep, pnl, max_bars


def add_extra_features(df):
    df = df.copy()
    c, v = df['close'], df['volume']
    df['rsi_slope'] = df['rsi14'].diff(3)
    vol_ma5 = v.rolling(5).mean()
    vol_ma20 = v.rolling(20).mean()
    df['vol_slope'] = (vol_ma5 / vol_ma20.replace(0, np.nan) - 1) * 100
    df['ret_10'] = c.pct_change(10) * 100
    up = (c > c.shift(1)).astype(int)
    df['consec_up'] = up.rolling(8).sum()
    return df


# ============================================================
# OPCION 2: RULE-BASED 4H
# ============================================================

def detect_breakout_b(df, i):
    """Breakout B: close > high20, vol >= 1.8x, BB estrecho."""
    if i < 25:
        return None
    row = df.iloc[i]
    high20 = float(df['high'].iloc[i-20:i].max())
    if row['close'] <= high20:
        return None
    if row.get('vol_ratio', 1) < 1.8:
        return None
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > 2.5:
        return None
    recent_bb = df['bb_width'].iloc[i-5:i]
    narrow_bars = (recent_bb < 4.0).sum()
    if narrow_bars < 3:
        return None
    prev_adx = df['adx14'].iloc[i-3:i].mean()
    if prev_adx > 28:
        return None
    entry = float(row['close'])
    sl_raw = float(df['low'].iloc[i-5:i].min()) * 0.997
    sl_pct = (entry - sl_raw) / entry
    if sl_pct < 0.005 or sl_pct > 0.04:
        return None
    tp_pct = sl_pct * 1.5
    return {'direction': 'LONG', 'setup': 'BREAKOUT_B',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}


def detect_pullback_ema20(df, i):
    """Pullback a EMA20 en tendencia alcista."""
    if i < 25:
        return None
    row = df.iloc[i]
    prev = df.iloc[i-1]
    c = float(row['close'])
    o = float(row['open'])
    ema20 = float(row.get('ema20', 0))
    ema50 = float(row.get('ema50', 0))
    if ema20 <= 0 or ema50 <= 0 or c < ema50:
        return None
    dist_ema20 = (c - ema20) / ema20
    if dist_ema20 < -0.005 or dist_ema20 > 0.015:
        return None
    adx = float(row.get('adx14', 0))
    if adx < 15:
        return None
    rsi = float(row.get('rsi14', 50))
    if rsi < 33 or rsi > 58:
        return None
    if c <= o:
        return None
    if float(prev['close']) >= float(prev['open']):
        return None
    vol_ratio = float(row.get('vol_ratio', 1))
    if vol_ratio > 2.0:
        return None
    atr_pct = float(row.get('atr_pct', 2.0))
    entry = c
    sl_pct = max(min(atr_pct / 100 * 1.0, 0.03), 0.01)
    tp_pct = sl_pct * 1.67
    return {'direction': 'LONG', 'setup': 'PULLBACK_EMA20',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}


def run_rule_based_wf(df):
    """Walk-forward rule-based: Breakout B + Pullback EMA20 en BULL, Breakout solo en RANGE."""
    results = []
    all_trades = []

    for start_s, end_s in WF_FOLDS:
        test_mask = (df.index >= start_s) & (df.index <= end_s)
        df_test = df[test_mask]
        period = f"{start_s[:7]}/{end_s[5:7]}"

        if len(df_test) < 50:
            results.append({'period': period, 'n': 0, 'wr': 0, 'pf': 0,
                           'ok': False, 'n_brk': 0, 'n_pb': 0})
            continue

        trades = []
        for i in range(len(df)):
            ts = df.index[i]
            if ts < pd.Timestamp(start_s, tz='UTC') or ts > pd.Timestamp(end_s, tz='UTC'):
                continue
            if i < 30:
                continue

            regime = detect_regime(df.iloc[i])
            trade = None

            if regime == 'BULL':
                trade = detect_breakout_b(df, i)
                if trade is None:
                    trade = detect_pullback_ema20(df, i)
            elif regime == 'RANGE':
                trade = detect_breakout_b(df, i)

            if trade is None:
                continue

            out = sim_trade_fixed(df, i, trade['entry'],
                                  trade['tp_pct'], trade['sl_pct'], max_bars=16)
            trades.append({
                'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                'setup': trade['setup'], 'regime': regime,
            })

        m = metrics(trades, period)
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        n_brk = sum(1 for t in trades if t['setup'] == 'BREAKOUT_B')
        n_pb = sum(1 for t in trades if t['setup'] == 'PULLBACK_EMA20')

        results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                        'pf': m['pf'], 'ok': ok, 'n_brk': n_brk, 'n_pb': n_pb})
        all_trades.extend(trades)

    folds_ok = sum(1 for r in results if r['ok'])
    return {'folds': results, 'folds_ok': folds_ok, 'all_trades': all_trades}


# ============================================================
# OPCION 1: 1D ML
# ============================================================

def compute_features_1d(df):
    """Features TA para velas diarias."""
    df = df.copy()
    h, l, c, v, o = df['high'], df['low'], df['close'], df['volume'], df['open']

    for n in [20, 50, 200]:
        df[f'ema{n}'] = pta.ema(c, length=n)
    df['ema20_slope'] = df['ema20'].pct_change(5) * 100
    df['ema50_slope'] = df['ema50'].pct_change(10) * 100
    df['ema200_dist'] = (c - df['ema200']) / df['ema200'] * 100

    df['rsi14'] = pta.rsi(c, length=14)
    df['rsi7'] = pta.rsi(c, length=7)
    df['rsi_slope'] = df['rsi14'].diff(3)

    atr = pta.atr(h, l, c, length=14)
    df['atr14'] = atr
    df['atr_pct'] = atr / c * 100

    bb = pta.bbands(c, length=20)
    if bb is not None:
        bb_low, bb_mid, bb_up = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
        df['bb_pct'] = (c - bb_low) / (bb_up - bb_low).replace(0, np.nan)
        df['bb_width'] = (bb_up - bb_low) / bb_mid * 100
    else:
        df['bb_pct'] = 0.5; df['bb_width'] = 5.0

    adx_df = pta.adx(h, l, c, length=14)
    if adx_df is not None:
        df['adx14'] = adx_df.iloc[:, 0]
        df['di_plus'] = adx_df.iloc[:, 1]
        df['di_minus'] = adx_df.iloc[:, 2]
        df['di_diff'] = df['di_plus'] - df['di_minus']
    else:
        df['adx14'] = 20.0; df['di_diff'] = 0.0

    vol_ma = v.rolling(20).mean()
    df['vol_ratio'] = v / vol_ma.replace(0, np.nan)

    df['high20'] = h.rolling(20).max().shift(1)
    df['low20'] = l.rolling(20).min().shift(1)
    df['range_pos'] = (c - df['low20']) / (df['high20'] - df['low20']).replace(0, np.nan)

    df['ret_1'] = c.pct_change(1) * 100
    df['ret_5'] = c.pct_change(5) * 100
    df['ret_10'] = c.pct_change(10) * 100

    # Candle micro
    candle_range = (h - l).clip(lower=1e-10)
    df['body_ratio'] = (c - o).abs() / candle_range
    df['close_in_range'] = (c - l) / candle_range

    # Consecutive
    direction = np.sign(c - o)
    df['consec_bull'] = (direction == 1).astype(int).rolling(5).sum()

    # Vol
    ret = c.pct_change()
    df['realized_vol_10'] = ret.rolling(10).std() * 100

    return df.dropna(subset=['ema20', 'ema50', 'ema200', 'rsi14'])


def compute_ethbtc_features_1d(df_eth, df_btc):
    """ETH/BTC features en 1D."""
    feat = pd.DataFrame(index=df_eth.index)
    btc_close = df_btc['close'].reindex(df_eth.index, method='ffill')
    eth_close = df_eth['close']

    ratio = eth_close / btc_close.replace(0, np.nan)
    feat['ethbtc_ratio'] = ratio
    feat['ethbtc_slope_5'] = ratio.pct_change(5) * 100
    feat['ethbtc_slope_20'] = ratio.pct_change(20) * 100
    ratio_mean = ratio.rolling(30).mean()
    ratio_std = ratio.rolling(30).std()
    feat['ethbtc_zscore'] = ((ratio - ratio_mean) / ratio_std.clip(lower=1e-8)).clip(-4, 4)

    feat['btc_ret_1'] = btc_close.pct_change(1) * 100
    feat['btc_ret_5'] = btc_close.pct_change(5) * 100
    feat['btc_rsi14'] = pta.rsi(btc_close, length=14)

    btc_vol = df_btc['volume'].reindex(df_eth.index, method='ffill')
    btc_vol_ma = btc_vol.rolling(20).mean()
    feat['btc_vol_ratio'] = btc_vol / btc_vol_ma.replace(0, np.nan)

    # Mean-rev
    vwap_10 = (eth_close * df_eth['volume']).rolling(10).sum() / df_eth['volume'].rolling(10).sum().replace(0, np.nan)
    feat['vwap10_dist'] = (eth_close - vwap_10) / vwap_10 * 100

    c_mean = eth_close.rolling(20).mean()
    c_std = eth_close.rolling(20).std()
    feat['price_zscore'] = ((eth_close - c_mean) / c_std.clip(lower=1e-8)).clip(-4, 4)

    return feat


ML_FEATURES_1D = [
    'ema20_slope', 'ema50_slope', 'ema200_dist',
    'rsi14', 'rsi7', 'rsi_slope', 'atr_pct',
    'bb_pct', 'bb_width', 'adx14', 'di_diff',
    'vol_ratio', 'range_pos',
    'ret_1', 'ret_5', 'ret_10',
    'body_ratio', 'close_in_range', 'consec_bull', 'realized_vol_10',
    # ETH-especificos
    'ethbtc_ratio', 'ethbtc_slope_5', 'ethbtc_slope_20', 'ethbtc_zscore',
    'btc_ret_1', 'btc_ret_5', 'btc_rsi14', 'btc_vol_ratio',
    'vwap10_dist', 'price_zscore',
]


def create_labels_1d(df, direction='long', tp_pct=0.05, sl_pct=0.025, max_bars=10):
    """Labels para 1D (TP/SL mas amplios que 4H)."""
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    labels = np.full(n, np.nan)

    for i in range(n - max_bars - 1):
        entry = closes[i]
        if direction == 'long':
            tp = entry * (1 + tp_pct)
            sl = entry * (1 - sl_pct)
            for j in range(i + 1, i + max_bars + 1):
                if highs[j] >= tp:
                    labels[i] = 1; break
                if lows[j] <= sl:
                    labels[i] = 0; break
            else:
                labels[i] = 1 if closes[i + max_bars] > entry else 0
        else:
            tp = entry * (1 - tp_pct)
            sl = entry * (1 + sl_pct)
            for j in range(i + 1, i + max_bars + 1):
                if lows[j] <= tp:
                    labels[i] = 1; break
                if highs[j] >= sl:
                    labels[i] = 0; break
            else:
                labels[i] = 1 if closes[i + max_bars] < entry else 0

    return pd.Series(labels, index=df.index)


def sim_trade_1d(df, entry_bar, entry_price, tp_pct, sl_pct, max_bars=10):
    """Simular trade en 1D."""
    return sim_trade_fixed(df, entry_bar, entry_price, tp_pct, sl_pct, max_bars)


# WF para velas diarias: folds de 1 anio (no semestres, hay pocas barras)
WF_FOLDS_1D = [
    ('2020-01-01', '2020-12-31'),
    ('2021-01-01', '2021-12-31'),
    ('2022-01-01', '2022-12-31'),
    ('2023-01-01', '2023-12-31'),
    ('2024-01-01', '2024-12-31'),
    ('2025-01-01', '2025-12-31'),
]


def run_1d_ml_wf(df, labels, features, tp_pct=0.05, sl_pct=0.025, max_bars=10, threshold=0.50):
    """Walk-forward ML en 1D con expanding window."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    models_to_test = {
        'GBM': lambda: GradientBoostingClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.05,
            min_samples_leaf=15, subsample=0.8, random_state=42),
        'RF': lambda: RandomForestClassifier(
            n_estimators=150, max_depth=4, min_samples_leaf=10,
            max_features='sqrt', random_state=42, n_jobs=-1),
    }
    try:
        import lightgbm as lgb
        models_to_test['LGBM'] = lambda: lgb.LGBMClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            min_child_samples=15, subsample=0.8, random_state=42, verbose=-1)
    except ImportError:
        pass
    try:
        import xgboost as xgb
        models_to_test['XGB'] = lambda: xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            min_child_weight=15, subsample=0.8, random_state=42,
            verbosity=0, eval_metric='logloss')
    except ImportError:
        pass

    all_results = {}
    for model_name, constructor in models_to_test.items():
        results = []
        all_trades = []

        for start_s, end_s in WF_FOLDS_1D:
            test_mask = (df.index >= start_s) & (df.index <= end_s)
            train_mask = df.index < start_s
            period = f"{start_s[:4]}"

            y_train = labels[train_mask]
            valid = y_train.notna()
            X_train = df.loc[train_mask, features][valid].fillna(0)
            y_train = y_train[valid]

            if len(X_train) < 200 or y_train.sum() < 10:
                results.append({'period': period, 'n': 0, 'wr': 0, 'pf': 0,
                               'ok': False, 'auc': 0})
                continue

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train.values)
            model = constructor()
            model.fit(X_tr, y_train)

            X_test = df.loc[test_mask, features].fillna(0)
            y_test = labels[test_mask]
            valid_t = y_test.notna()
            X_test_v = X_test[valid_t]
            y_test_v = y_test[valid_t]

            if len(X_test_v) == 0:
                results.append({'period': period, 'n': 0, 'wr': 0, 'pf': 0,
                               'ok': False, 'auc': 0})
                continue

            X_te = scaler.transform(X_test_v.values)
            probs = model.predict_proba(X_te)[:, 1]

            try:
                auc = roc_auc_score(y_test_v, probs)
            except:
                auc = 0.5

            # Simular
            signal_mask = probs >= threshold
            trades = []
            if signal_mask.sum() > 0:
                for ts in X_test_v.index[signal_mask]:
                    gi = df.index.get_loc(ts)
                    if gi + max_bars >= len(df):
                        continue
                    entry = float(df['close'].iloc[gi])
                    out = sim_trade_1d(df, gi, entry, tp_pct, sl_pct, max_bars)
                    trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': ts})

            m = metrics(trades, period)
            ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
            results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                           'pf': m['pf'], 'ok': ok, 'auc': auc})
            all_trades.extend(trades)

        folds_ok = sum(1 for r in results if r['ok'])
        oos_trades = [t for t in all_trades if '2022' <= str(t['ts'])[:4] <= '2025']
        m_oos = metrics(oos_trades, f'{model_name} OOS')

        all_results[model_name] = {
            'folds': results, 'folds_ok': folds_ok,
            'oos': m_oos, 'all_trades': all_trades,
        }

    return all_results


# ============================================================
# OPCION 3: BTC-FOLLOWER
# ============================================================

def run_btc_follower_wf(df_eth, df_btc):
    """
    ETH entra LONG cuando:
    1. BTC genera senal LONG (Breakout B o Pullback EMA20 en BTC)
    2. Correlacion ETH-BTC rolling 20 > 0.7
    3. ETH no esta divergiendo (ethbtc_slope_5 > -2%)
    4. ETH en tendencia similar (ema20 > ema50)
    """
    # Calcular correlacion ETH-BTC
    eth_ret = df_eth['close'].pct_change()
    btc_close = df_btc['close'].reindex(df_eth.index, method='ffill')
    btc_ret = btc_close.pct_change()
    corr_20 = eth_ret.rolling(20).corr(btc_ret)

    # ETH/BTC ratio slope
    ratio = df_eth['close'] / btc_close.replace(0, np.nan)
    ratio_slope_5 = ratio.pct_change(5) * 100

    results = []
    all_trades = []

    for start_s, end_s in WF_FOLDS:
        period = f"{start_s[:7]}/{end_s[5:7]}"
        trades = []

        for i in range(30, len(df_btc)):
            ts_btc = df_btc.index[i]
            if ts_btc < pd.Timestamp(start_s, tz='UTC') or ts_btc > pd.Timestamp(end_s, tz='UTC'):
                continue

            # 1. BTC genera senal LONG?
            regime_btc = detect_regime(df_btc.iloc[i])
            btc_signal = None
            if regime_btc == 'BULL':
                btc_signal = detect_breakout_b(df_btc, i)
                if btc_signal is None:
                    btc_signal = detect_pullback_ema20(df_btc, i)
            elif regime_btc == 'RANGE':
                btc_signal = detect_breakout_b(df_btc, i)

            if btc_signal is None:
                continue

            # Encontrar barra correspondiente en ETH
            if ts_btc not in df_eth.index:
                # Buscar la barra mas cercana
                idx_pos = df_eth.index.searchsorted(ts_btc)
                if idx_pos >= len(df_eth):
                    continue
                ts_eth = df_eth.index[idx_pos]
                if abs((ts_eth - ts_btc).total_seconds()) > 4 * 3600:
                    continue
            else:
                ts_eth = ts_btc

            eth_i = df_eth.index.get_loc(ts_eth)
            if eth_i < 25 or eth_i + 16 >= len(df_eth):
                continue

            # 2. Correlacion alta?
            c = corr_20.get(ts_eth, 0)
            if pd.isna(c) or c < 0.7:
                continue

            # 3. ETH no diverge? (ratio no cayendo fuerte)
            rs = ratio_slope_5.get(ts_eth, 0)
            if pd.isna(rs) or rs < -2.0:
                continue

            # 4. ETH en tendencia similar
            eth_row = df_eth.iloc[eth_i]
            ema20_eth = eth_row.get('ema20', 0)
            ema50_eth = eth_row.get('ema50', 0)
            if ema20_eth <= 0 or ema50_eth <= 0:
                continue
            if ema20_eth < ema50_eth:
                continue

            # Entrar ETH con ATR-based TP/SL
            entry = float(eth_row['close'])
            atr_pct = float(eth_row.get('atr_pct', 2.0))
            sl_pct = max(min(atr_pct / 100 * 1.2, 0.04), 0.01)
            tp_pct = sl_pct * 1.5

            out = sim_trade_fixed(df_eth, eth_i, entry, tp_pct, sl_pct, max_bars=16)
            trades.append({
                'outcome': out[0], 'pnl_pct': out[2], 'ts': ts_eth,
                'setup': 'BTC_FOLLOWER', 'btc_setup': btc_signal['setup'],
                'corr': float(c),
            })

        m = metrics(trades, period)
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                        'pf': m['pf'], 'ok': ok})
        all_trades.extend(trades)

    folds_ok = sum(1 for r in results if r['ok'])
    return {'folds': results, 'folds_ok': folds_ok, 'all_trades': all_trades}


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("ETH EVALUATION — 3 OPCIONES")
    print("=" * 70)

    # ---- Cargar datos comunes ----
    print("\nCargando datos...")
    df_eth_raw = load_pair_4h('ETH')
    df_btc_raw = load_btc_4h()
    print(f"  ETH 4H: {len(df_eth_raw)} bars ({df_eth_raw.index[0].date()} - {df_eth_raw.index[-1].date()})")
    print(f"  BTC 4H: {len(df_btc_raw)} bars")

    # Features 4H para ETH y BTC
    df_eth = compute_features_4h(df_eth_raw)
    df_eth = add_extra_features(df_eth)
    df_daily_eth = compute_macro_daily(df_eth)
    df_eth = merge_daily_to_4h(df_eth, df_daily_eth)

    df_btc = compute_features_4h(df_btc_raw)
    df_btc = add_extra_features(df_btc)
    df_daily_btc = compute_macro_daily(df_btc)
    df_btc = merge_daily_to_4h(df_btc, df_daily_btc)

    print(f"  ETH features: {len(df_eth)} bars")
    print(f"  BTC features: {len(df_btc)} bars")

    # Regimenes ETH
    regimes = df_eth.apply(lambda r: detect_regime(r), axis=1)
    for r in ['BULL', 'BEAR', 'RANGE']:
        print(f"  ETH {r}: {(regimes == r).mean():.1%}")

    # ============================================================
    # OPCION 2: RULE-BASED 4H
    # ============================================================
    print(f"\n{'='*70}")
    print("OPCION 2: RULE-BASED 4H (Breakout B + Pullback EMA20)")
    print(f"{'='*70}")

    wf_rules = run_rule_based_wf(df_eth)

    print(f"\n  {'Periodo':<14} | {'N':>4} | {'BRK':>3} | {'PB':>3} | {'WR':>7} | {'PF':>6} | OK")
    print("  " + "-" * 55)
    for r in wf_rules['folds']:
        ok_s = '+' if r['ok'] else '-'
        wr_s = f"{r['wr']:.1%}" if r['n'] > 0 else 'n/a'
        pf_s = f"{r['pf']:.2f}" if r['n'] > 0 else 'n/a'
        print(f"  {r['period']:<14} | {r['n']:>4} | {r.get('n_brk',0):>3} | "
              f"{r.get('n_pb',0):>3} | {wr_s:>7} | {pf_s:>6} | {ok_s}")

    folds_ok_rules = wf_rules['folds_ok']
    print(f"\n  Folds OK: {folds_ok_rules}/12")

    oos_rules = [t for t in wf_rules['all_trades']
                 if OOS_START <= str(t['ts'])[:10] <= OOS_END]
    m_rules = metrics(oos_rules, 'Rule-based OOS')
    if m_rules['n'] > 0:
        print(f"  OOS: N={m_rules['n']} | WR={m_rules['wr']:.1%} | PF={m_rules['pf']:.2f} | {m_rules['trades_pm']:.1f}t/m")
        # Equity
        cum = 1.0; peak = 1.0; max_dd = 0
        for t in sorted(oos_rules, key=lambda x: x['ts']):
            cum *= (1 + t['pnl_pct'])
            peak = max(peak, cum)
            max_dd = max(max_dd, (peak - cum) / peak)
        print(f"  Equity: $1000 -> ${1000*cum:.0f} | MaxDD: {max_dd:.1%}")
        # Por setup
        for setup in ['BREAKOUT_B', 'PULLBACK_EMA20']:
            st = [t for t in oos_rules if t['setup'] == setup]
            ms = metrics(st, setup)
            if ms['n'] > 0:
                print(f"    {setup}: N={ms['n']} WR={ms['wr']:.1%} PF={ms['pf']:.2f}")
    else:
        print("  OOS: Sin trades")

    passed_rules = (folds_ok_rules >= 7 and m_rules.get('pf', 0) >= 1.2)
    print(f"  Veredicto: {'APROBADO' if passed_rules else 'RECHAZADO'}")

    # ============================================================
    # OPCION 1: 1D ML
    # ============================================================
    print(f"\n{'='*70}")
    print("OPCION 1: ML 1D (velas diarias, features ETH-especificos)")
    print(f"{'='*70}")

    # Cargar 1D
    try:
        df_eth_1d_raw = load_pair_1d('ETH')
    except FileNotFoundError:
        # Resamplear desde 4H
        print("  No hay datos 1D, resampling desde 4H...")
        df_eth_1d_raw = df_eth_raw.resample('1D').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()

    try:
        df_btc_1d_raw = load_btc_1d()
    except:
        df_btc_1d_raw = df_btc_raw.resample('1D').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()

    print(f"  ETH 1D: {len(df_eth_1d_raw)} bars ({df_eth_1d_raw.index[0].date()} - {df_eth_1d_raw.index[-1].date()})")

    # Features 1D
    df_1d = compute_features_1d(df_eth_1d_raw)
    ethbtc_1d = compute_ethbtc_features_1d(df_eth_1d_raw, df_btc_1d_raw)
    for col in ethbtc_1d.columns:
        df_1d[col] = ethbtc_1d[col]
    df_1d = df_1d.replace([np.inf, -np.inf], np.nan)

    available_1d = [f for f in ML_FEATURES_1D if f in df_1d.columns]
    print(f"  Features: {len(available_1d)}/{len(ML_FEATURES_1D)}")

    # Labels 1D (TP/SL mas amplios para diario)
    TP_1D = 0.05
    SL_1D = 0.025
    MAX_BARS_1D = 10
    labels_1d = create_labels_1d(df_1d, 'long', TP_1D, SL_1D, MAX_BARS_1D)
    valid_1d = labels_1d[labels_1d.notna()]
    print(f"  Labels: {len(valid_1d)} valid, base rate={valid_1d.mean():.1%}")
    print(f"  TP={TP_1D:.1%} SL={SL_1D:.1%} MaxBars={MAX_BARS_1D}")
    print(f"  Break-even WR: {SL_1D/(TP_1D+SL_1D):.1%}")

    # Walk-forward 1D (folds anuales, menos datos)
    print("\n  Walk-forward 1D (folds anuales)...")
    results_1d = run_1d_ml_wf(df_1d, labels_1d, available_1d,
                               tp_pct=TP_1D, sl_pct=SL_1D,
                               max_bars=MAX_BARS_1D, threshold=0.50)

    best_1d_model = None
    best_1d_pf = 0

    for model_name, data in results_1d.items():
        print(f"\n  {model_name}:")
        print(f"    {'Anio':<6} | {'N':>4} | {'WR':>7} | {'PF':>6} | {'AUC':>5} | OK")
        for r in data['folds']:
            ok_s = '+' if r['ok'] else '-'
            wr_s = f"{r['wr']:.1%}" if r['n'] > 0 else 'n/a'
            pf_s = f"{r['pf']:.2f}" if r['n'] > 0 else 'n/a'
            auc_s = f"{r['auc']:.3f}" if r.get('auc', 0) > 0 else 'n/a'
            print(f"    {r['period']:<6} | {r['n']:>4} | {wr_s:>7} | {pf_s:>6} | {auc_s:>5} | {ok_s}")

        print(f"    Folds OK: {data['folds_ok']}/6")
        m = data['oos']
        if m['n'] > 0:
            print(f"    OOS: N={m['n']} WR={m['wr']:.1%} PF={m['pf']:.2f} {m['trades_pm']:.1f}t/m")
            if m['pf'] > best_1d_pf and m['n'] >= 5:
                best_1d_pf = m['pf']
                best_1d_model = model_name

    # Necesita 4/6 folds (equivalente a 7/12)
    passed_1d = any(d['folds_ok'] >= 4 and d['oos']['pf'] >= 1.2
                    for d in results_1d.values())
    print(f"\n  Veredicto 1D: {'APROBADO' if passed_1d else 'RECHAZADO'}")
    if best_1d_model:
        print(f"  Mejor: {best_1d_model} PF={best_1d_pf:.2f}")

    # ============================================================
    # OPCION 3: BTC-FOLLOWER
    # ============================================================
    print(f"\n{'='*70}")
    print("OPCION 3: BTC-FOLLOWER (ETH entra cuando BTC da senal)")
    print(f"{'='*70}")
    print("  Condiciones: BTC senal LONG + corr>0.7 + ETH no diverge + ETH ema20>ema50")

    wf_follower = run_btc_follower_wf(df_eth, df_btc)

    print(f"\n  {'Periodo':<14} | {'N':>4} | {'WR':>7} | {'PF':>6} | OK")
    print("  " + "-" * 45)
    for r in wf_follower['folds']:
        ok_s = '+' if r['ok'] else '-'
        wr_s = f"{r['wr']:.1%}" if r['n'] > 0 else 'n/a'
        pf_s = f"{r['pf']:.2f}" if r['n'] > 0 else 'n/a'
        print(f"  {r['period']:<14} | {r['n']:>4} | {wr_s:>7} | {pf_s:>6} | {ok_s}")

    folds_ok_follower = wf_follower['folds_ok']
    print(f"\n  Folds OK: {folds_ok_follower}/12")

    oos_follower = [t for t in wf_follower['all_trades']
                    if OOS_START <= str(t['ts'])[:10] <= OOS_END]
    m_follower = metrics(oos_follower, 'Follower OOS')
    if m_follower['n'] > 0:
        print(f"  OOS: N={m_follower['n']} | WR={m_follower['wr']:.1%} | "
              f"PF={m_follower['pf']:.2f} | {m_follower['trades_pm']:.1f}t/m")
        cum = 1.0; peak = 1.0; max_dd = 0
        for t in sorted(oos_follower, key=lambda x: x['ts']):
            cum *= (1 + t['pnl_pct'])
            peak = max(peak, cum)
            max_dd = max(max_dd, (peak - cum) / peak)
        print(f"  Equity: $1000 -> ${1000*cum:.0f} | MaxDD: {max_dd:.1%}")
    else:
        print("  OOS: Sin trades")

    passed_follower = (folds_ok_follower >= 7 and m_follower.get('pf', 0) >= 1.2)
    print(f"  Veredicto: {'APROBADO' if passed_follower else 'RECHAZADO'}")

    # ============================================================
    # RESUMEN FINAL
    # ============================================================
    print(f"\n\n{'='*70}")
    print("RESUMEN FINAL — ETH 3 OPCIONES")
    print(f"{'='*70}")

    options = [
        ('ML 4H (prev)', '6/12', 'WR=42% PF=1.09 AUC=0.53', 'RECHAZADO'),
        ('Rule-based 4H', f'{folds_ok_rules}/12',
         f"WR={m_rules['wr']:.1%} PF={m_rules['pf']:.2f}" if m_rules['n'] > 0 else 'Sin trades',
         'APROBADO' if passed_rules else 'RECHAZADO'),
        ('ML 1D', f'{max((d["folds_ok"] for d in results_1d.values()), default=0)}/6',
         f"Mejor: {best_1d_model} PF={best_1d_pf:.2f}" if best_1d_model else 'Sin modelo viable',
         'APROBADO' if passed_1d else 'RECHAZADO'),
        ('BTC-follower', f'{folds_ok_follower}/12',
         f"WR={m_follower['wr']:.1%} PF={m_follower['pf']:.2f}" if m_follower['n'] > 0 else 'Sin trades',
         'APROBADO' if passed_follower else 'RECHAZADO'),
    ]

    print(f"\n  {'Opcion':<18} | {'WF':>6} | {'Metricas':<30} | Veredicto")
    print("  " + "-" * 75)
    for name, wf, met, verd in options:
        print(f"  {name:<18} | {wf:>6} | {met:<30} | {verd}")

    any_passed = passed_rules or passed_1d or passed_follower
    print(f"\n  {'='*70}")
    if any_passed:
        print("  ETH TIENE OPCION VIABLE")
        if passed_follower:
            print("  -> Recomendacion: BTC-follower (aprovecha modelo BTC validado)")
        elif passed_rules:
            print("  -> Recomendacion: Rule-based (simple, sin ML)")
        elif passed_1d:
            print(f"  -> Recomendacion: ML 1D ({best_1d_model})")
    else:
        print("  ETH RECHAZADO DEFINITIVAMENTE")
        print("  Ninguna de las 4 opciones (ML 4H, Rule-based, ML 1D, BTC-follower)")
        print("  produce resultados que justifiquen operar ETH.")
    print(f"  {'='*70}")

    return passed_rules, passed_1d, passed_follower


if __name__ == '__main__':
    main()
