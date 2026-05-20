"""
evaluate_eth_bear.py — ETH BEAR: ML SHORT + Ensemble + Comite Completo
========================================================================
ETH LONG en BULL/RANGE ya esta aprobado (Opcion 5, PF=1.35).
Ahora probar si podemos operar ETH SHORT en BEAR.

Enfoque: igual que BTC V15 SHORT (entrenar ML solo en barras BEAR).

Modelos individuales:
  1. GBM (baseline, como BTC)
  2. RandomForest
  3. LightGBM
  4. XGBoost

Ensembles:
  5. Voting (mayoria de 4 modelos)
  6. Stacking (meta-learner sobre los 4)
  7. Probabilidad promedio (avg prob >= threshold)

Reglas SHORT (sin ML):
  8. Mean reversion: RSI overbought + bearish candle + BEAR regime
  9. BTC SHORT follower

Al final: COMITE COMPLETO = LONG rules (BULL/RANGE) + mejor SHORT (BEAR)

Usage:
  python evaluate_eth_bear.py
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
    load_pair_4h, load_btc_4h,
    compute_features_4h, compute_macro_daily, merge_daily_to_4h,
    sim_trade_fixed, metrics, print_metrics,
    WF_FOLDS, OOS_START, OOS_END, COMMISSION,
)


# ============================================================
# SHARED
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


def add_eth_specific_features(df_eth, df_btc):
    """Features ETH-especificos que mostraron importancia en ronda 1."""
    df = df_eth.copy()
    btc_close = df_btc['close'].reindex(df.index, method='ffill')
    eth_close = df['close']

    # ETH/BTC ratio
    ratio = eth_close / btc_close.replace(0, np.nan)
    df['ethbtc_ratio'] = ratio
    df['ethbtc_slope_5'] = ratio.pct_change(5) * 100
    df['ethbtc_slope_20'] = ratio.pct_change(20) * 100
    ratio_mean = ratio.rolling(60).mean()
    ratio_std = ratio.rolling(60).std()
    df['ethbtc_zscore'] = ((ratio - ratio_mean) / ratio_std.clip(lower=1e-8)).clip(-4, 4)

    # BTC cross
    df['btc_ret_1'] = btc_close.pct_change(1) * 100
    df['btc_ret_5'] = btc_close.pct_change(5) * 100
    df['btc_rsi14'] = pta.rsi(btc_close, length=14)
    btc_vol = df_btc['volume'].reindex(df.index, method='ffill')
    btc_vol_ma = btc_vol.rolling(20).mean()
    df['btc_vol_ratio'] = btc_vol / btc_vol_ma.replace(0, np.nan)

    # Volatility
    atr = pta.atr(df['high'], df['low'], df['close'], length=14)
    atr_pct = atr / df['close'] * 100
    df['atr_pct_rank'] = atr_pct.rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    ret = df['close'].pct_change()
    df['realized_vol_20'] = ret.rolling(20).std() * 100

    # Mean-reversion
    c_mean = df['close'].rolling(20).mean()
    c_std = df['close'].rolling(20).std()
    df['price_zscore'] = ((df['close'] - c_mean) / c_std.clip(lower=1e-8)).clip(-4, 4)

    # Correlacion ETH-BTC
    eth_ret = df['close'].pct_change()
    btc_ret = btc_close.pct_change()
    df['eth_btc_corr_20'] = eth_ret.rolling(20).corr(btc_ret)

    return df.replace([np.inf, -np.inf], np.nan)


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


def equity_stats(trades):
    if not trades:
        return 1.0, 0
    cum = 1.0; peak = 1.0; max_dd = 0
    for t in sorted(trades, key=lambda x: x['ts']):
        cum *= (1 + t['pnl_pct'])
        peak = max(peak, cum)
        dd = (peak - cum) / peak
        max_dd = max(max_dd, dd)
    return cum, max_dd


# ============================================================
# SHORT LABELS
# ============================================================

def create_short_labels(df, tp_pct=0.03, sl_pct=0.02, max_bars=16):
    """Label: precio baja tp_pct antes de subir sl_pct."""
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    labels = np.full(n, np.nan)
    for i in range(n - max_bars - 1):
        entry = closes[i]
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


# ============================================================
# ML FEATURES para SHORT
# ============================================================

# Features base (como BTC V15 SHORT)
SHORT_FEATURES_BASE = [
    'ema200_dist', 'ema20_slope', 'ema50_slope',
    'rsi14', 'rsi_slope', 'di_diff', 'adx14',
    'bb_pct', 'bb_width', 'atr_pct',
    'range_pos', 'vol_ratio', 'vol_slope',
    'ret_1', 'ret_5', 'ret_10',
    'consec_up', 'bull_1d',
]

# Features ETH-especificos adicionales
SHORT_FEATURES_ETH = SHORT_FEATURES_BASE + [
    'ethbtc_slope_5', 'ethbtc_slope_20', 'ethbtc_zscore',
    'btc_ret_1', 'btc_ret_5', 'btc_rsi14', 'btc_vol_ratio',
    'atr_pct_rank', 'realized_vol_20', 'price_zscore',
    'eth_btc_corr_20',
]


# ============================================================
# MODEL CONSTRUCTORS
# ============================================================

def get_short_models():
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    models = {}

    models['GBM'] = lambda: GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        min_samples_leaf=20, subsample=0.8, random_state=42)

    models['RF'] = lambda: RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=15,
        max_features='sqrt', random_state=42, n_jobs=-1)

    try:
        import lightgbm as lgb
        models['LGBM'] = lambda: lgb.LGBMClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1, n_jobs=-1)
    except ImportError:
        pass

    try:
        import xgboost as xgb
        models['XGB'] = lambda: xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, eval_metric='logloss', n_jobs=-1)
    except ImportError:
        pass

    return models


# ============================================================
# WALK-FORWARD ML SHORT (solo BEAR bars)
# ============================================================

def wf_short_ml(df, labels, features, model_constructors,
                tp_pct=0.03, sl_pct=0.02, max_bars=16,
                thresholds=[0.50, 0.55, 0.60]):
    """
    Walk-forward para SHORT ML, entrenando SOLO en barras BEAR.
    Prueba multiples thresholds y retorna el mejor por modelo.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    # Pre-calcular regimenes
    regimes = df.apply(lambda r: detect_regime(r), axis=1)

    all_model_results = {}

    for model_name, constructor in model_constructors.items():
        best_threshold = 0.50
        best_folds_ok = 0
        best_result = None

        for threshold in thresholds:
            results = []
            all_trades = []

            for fold_idx, (start_s, end_s) in enumerate(WF_FOLDS):
                test_mask = (df.index >= start_s) & (df.index <= end_s)
                train_mask = df.index < start_s
                period = f"{start_s[:7]}/{end_s[5:7]}"

                # Entrenar SOLO en barras BEAR
                y_train = labels[train_mask]
                bear_train = regimes[train_mask] == 'BEAR'
                valid = y_train.notna() & bear_train
                X_train = df.loc[train_mask, features][valid].fillna(0)
                y_train = y_train[valid]

                if len(X_train) < 200 or y_train.sum() < 10 or (len(y_train) - y_train.sum()) < 10:
                    results.append({'period': period, 'n': 0, 'wr': 0,
                                   'pf': 0, 'ok': False, 'auc': 0})
                    continue

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train.values)
                model = constructor()
                try:
                    model.fit(X_tr, y_train)
                except Exception:
                    results.append({'period': period, 'n': 0, 'wr': 0,
                                   'pf': 0, 'ok': False, 'auc': 0})
                    continue

                # Testear SOLO en barras BEAR del periodo test
                bear_test = regimes[test_mask] == 'BEAR'
                y_test = labels[test_mask]
                valid_test = y_test.notna() & bear_test
                X_test = df.loc[test_mask, features][valid_test].fillna(0)
                y_test_v = y_test[valid_test]

                if len(X_test) == 0:
                    results.append({'period': period, 'n': 0, 'wr': 0,
                                   'pf': 0, 'ok': False, 'auc': 0})
                    continue

                X_te = scaler.transform(X_test.values)
                probs = model.predict_proba(X_te)[:, 1]

                try:
                    auc = roc_auc_score(y_test_v, probs)
                except:
                    auc = 0.5

                # Simular SHORT trades
                signal_mask = probs >= threshold
                trades = []
                if signal_mask.sum() > 0:
                    for ts in X_test.index[signal_mask]:
                        gi = df.index.get_loc(ts)
                        if gi + max_bars >= len(df):
                            continue
                        entry = float(df['close'].iloc[gi])
                        # SL/TP adaptados a vol ETH
                        atr_pct_val = float(df['atr_pct'].iloc[gi]) if 'atr_pct' in df.columns else 2.5
                        sl = max(min(atr_pct_val / 100 * 1.5, 0.05), 0.015)
                        tp = max(min(atr_pct_val / 100 * 2.5, 0.08), 0.025)

                        out = sim_short(df, gi, entry, tp, sl, max_bars)
                        trades.append({
                            'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                            'setup': f'SHORT_ML_{model_name}',
                            'prob': float(probs[list(X_test.index).index(ts)]),
                        })

                m = metrics(trades, period)
                ok = (m['n'] >= 2 and m['wr'] > 0.35 and m['pf'] > 0.9)
                results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                               'pf': m['pf'], 'ok': ok, 'auc': auc})
                all_trades.extend(trades)

            folds_ok = sum(1 for r in results if r['ok'])
            if folds_ok > best_folds_ok:
                best_folds_ok = folds_ok
                best_threshold = threshold
                best_result = {'folds': results, 'folds_ok': folds_ok,
                              'all_trades': all_trades, 'threshold': threshold}

        all_model_results[model_name] = best_result or {
            'folds': results, 'folds_ok': folds_ok,
            'all_trades': all_trades, 'threshold': threshold}

    return all_model_results


# ============================================================
# ENSEMBLE APPROACHES
# ============================================================

def wf_ensemble(df, labels, features, model_constructors,
                tp_pct=0.03, sl_pct=0.02, max_bars=16, threshold=0.55):
    """
    Walk-forward con 3 ensembles:
    - Voting: mayoria de modelos predicen SHORT
    - Avg Prob: promedio de probabilidades >= threshold
    - Stacking: meta-learner LogReg sobre probabilidades
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    regimes = df.apply(lambda r: detect_regime(r), axis=1)
    ensemble_results = {'Voting': [], 'AvgProb': [], 'Stacking': []}
    ensemble_trades = {'Voting': [], 'AvgProb': [], 'Stacking': []}

    for fold_idx, (start_s, end_s) in enumerate(WF_FOLDS):
        test_mask = (df.index >= start_s) & (df.index <= end_s)
        train_mask = df.index < start_s
        period = f"{start_s[:7]}/{end_s[5:7]}"

        # Train data: solo BEAR
        y_train = labels[train_mask]
        bear_train = regimes[train_mask] == 'BEAR'
        valid = y_train.notna() & bear_train
        X_train = df.loc[train_mask, features][valid].fillna(0)
        y_train_v = y_train[valid]

        if len(X_train) < 200 or y_train_v.sum() < 10:
            for ens in ensemble_results:
                ensemble_results[ens].append({
                    'period': period, 'n': 0, 'wr': 0, 'pf': 0, 'ok': False})
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train.values)

        # Entrenar todos los modelos
        trained = {}
        for name, constructor in model_constructors.items():
            try:
                m = constructor()
                m.fit(X_tr, y_train_v)
                trained[name] = m
            except:
                pass

        if len(trained) < 2:
            for ens in ensemble_results:
                ensemble_results[ens].append({
                    'period': period, 'n': 0, 'wr': 0, 'pf': 0, 'ok': False})
            continue

        # Test data: solo BEAR
        bear_test = regimes[test_mask] == 'BEAR'
        y_test = labels[test_mask]
        valid_test = y_test.notna() & bear_test
        X_test = df.loc[test_mask, features][valid_test].fillna(0)

        if len(X_test) == 0:
            for ens in ensemble_results:
                ensemble_results[ens].append({
                    'period': period, 'n': 0, 'wr': 0, 'pf': 0, 'ok': False})
            continue

        X_te = scaler.transform(X_test.values)

        # Obtener probabilidades de cada modelo
        all_probs = {}
        for name, model in trained.items():
            all_probs[name] = model.predict_proba(X_te)[:, 1]

        prob_matrix = np.column_stack(list(all_probs.values()))
        avg_prob = prob_matrix.mean(axis=1)
        votes = (prob_matrix >= threshold).sum(axis=1)
        majority = len(trained) / 2

        # Stacking: entrenar meta-learner en train
        train_probs = {}
        for name, model in trained.items():
            train_probs[name] = model.predict_proba(X_tr)[:, 1]
        train_prob_matrix = np.column_stack(list(train_probs.values()))
        meta = LogisticRegression(C=1.0, max_iter=300)
        meta.fit(train_prob_matrix, y_train_v)
        stack_prob = meta.predict_proba(prob_matrix)[:, 1]

        # Simular para cada ensemble
        for ens_name, signal_mask in [
            ('Voting', votes > majority),
            ('AvgProb', avg_prob >= threshold),
            ('Stacking', stack_prob >= threshold),
        ]:
            trades = []
            if signal_mask.sum() > 0:
                for idx_pos in np.where(signal_mask)[0]:
                    ts = X_test.index[idx_pos]
                    gi = df.index.get_loc(ts)
                    if gi + max_bars >= len(df):
                        continue
                    entry = float(df['close'].iloc[gi])
                    atr_pct_val = float(df['atr_pct'].iloc[gi]) if 'atr_pct' in df.columns else 2.5
                    sl = max(min(atr_pct_val / 100 * 1.5, 0.05), 0.015)
                    tp = max(min(atr_pct_val / 100 * 2.5, 0.08), 0.025)

                    out = sim_short(df, gi, entry, tp, sl, max_bars)
                    trades.append({
                        'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                        'setup': f'SHORT_{ens_name}',
                    })

            m = metrics(trades, period)
            ok = (m['n'] >= 2 and m['wr'] > 0.35 and m['pf'] > 0.9)
            ensemble_results[ens_name].append({
                'period': period, 'n': m['n'], 'wr': m['wr'],
                'pf': m['pf'], 'ok': ok})
            ensemble_trades[ens_name].extend(trades)

    final = {}
    for ens_name in ensemble_results:
        folds_ok = sum(1 for r in ensemble_results[ens_name] if r['ok'])
        final[ens_name] = {
            'folds': ensemble_results[ens_name],
            'folds_ok': folds_ok,
            'all_trades': ensemble_trades[ens_name],
        }
    return final


# ============================================================
# RULE-BASED SHORT
# ============================================================

def detect_short_meanrev(df, i):
    """SHORT mean-reversion: RSI overbought en BEAR + vela bearish."""
    if i < 25:
        return None
    row = df.iloc[i]
    prev = df.iloc[i-1]
    c, o = float(row['close']), float(row['open'])

    # RSI overbought (bounce en bear)
    rsi = float(row.get('rsi14', 50))
    if rsi < 58 or rsi > 78:
        return None

    # BB alto (cerca de banda superior)
    bb_pct = float(row.get('bb_pct', 0.5))
    if bb_pct < 0.75:
        return None

    # Vela bearish (rechazo)
    if c >= o:
        return None

    # Vela anterior fue alcista (confirma bounce)
    if float(prev['close']) <= float(prev['open']):
        return None

    # ATR-based TP/SL adaptados
    atr_pct = float(row.get('atr_pct', 2.5))
    entry = c
    sl_pct = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    tp_pct = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)

    return {'direction': 'SHORT', 'setup': 'SHORT_MEANREV',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}


def detect_short_breakdown(df, i):
    """SHORT breakdown: close < low20, volumen confirma, en BEAR."""
    if i < 25:
        return None
    row = df.iloc[i]
    c = float(row['close'])

    # Ruptura bajista: close < min 20 barras
    low20 = float(df['low'].iloc[i-20:i].min())
    if c >= low20:
        return None

    # Volumen confirma
    if row.get('vol_ratio', 1) < 1.3:
        return None

    # ADX muestra tendencia
    if float(row.get('adx14', 0)) < 15:
        return None

    # RSI no oversold extremo (no comprar el panico)
    rsi = float(row.get('rsi14', 50))
    if rsi < 25:
        return None

    entry = c
    atr_pct = float(row.get('atr_pct', 2.5))
    sl_pct = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    tp_pct = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)

    return {'direction': 'SHORT', 'setup': 'SHORT_BREAKDOWN',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}


def run_rules_short(df_eth):
    """Walk-forward SHORT rules: meanrev + breakdown, solo en BEAR."""
    regimes = df_eth.apply(lambda r: detect_regime(r), axis=1)
    results = []
    all_trades = []

    for start_s, end_s in WF_FOLDS:
        period = f"{start_s[:7]}/{end_s[5:7]}"
        trades = []

        for i in range(30, len(df_eth)):
            ts = df_eth.index[i]
            if ts < pd.Timestamp(start_s, tz='UTC') or ts > pd.Timestamp(end_s, tz='UTC'):
                continue
            if regimes.iloc[i] != 'BEAR':
                continue
            if i + 16 >= len(df_eth):
                continue

            trade = detect_short_meanrev(df_eth, i)
            if trade is None:
                trade = detect_short_breakdown(df_eth, i)
            if trade is None:
                continue

            out = sim_short(df_eth, i, trade['entry'],
                           trade['tp_pct'], trade['sl_pct'], max_bars=16)
            trades.append({
                'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                'setup': trade['setup'],
            })

        m = metrics(trades, period)
        ok = (m['n'] >= 2 and m['wr'] > 0.35 and m['pf'] > 0.9)
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                        'pf': m['pf'], 'ok': ok})
        all_trades.extend(trades)

    folds_ok = sum(1 for r in results if r['ok'])
    return {'folds': results, 'folds_ok': folds_ok, 'all_trades': all_trades}


# ============================================================
# COMITE COMPLETO: LONG rules (BULL/RANGE) + SHORT (BEAR)
# ============================================================

def detect_breakout_eth(df, i):
    """Breakout B adaptado ETH."""
    if i < 25:
        return None
    row = df.iloc[i]
    high20 = float(df['high'].iloc[i-20:i].max())
    if row['close'] <= high20:
        return None
    if row.get('vol_ratio', 1) < 1.3:
        return None
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > 3.5:
        return None
    recent_bb = df['bb_width'].iloc[i-5:i]
    if (recent_bb < 5.5).sum() < 2:
        return None
    if df['adx14'].iloc[i-3:i].mean() > 32:
        return None
    entry = float(row['close'])
    sl_raw = float(df['low'].iloc[i-5:i].min()) * 0.995
    sl_pct = (entry - sl_raw) / entry
    if sl_pct < 0.005 or sl_pct > 0.06:
        return None
    atr_pct = float(row.get('atr_pct', 2.5))
    sl_pct = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    tp_pct = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    return {'direction': 'LONG', 'setup': 'BRK_ETH',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}


def run_full_committee(df_eth, df_btc, short_method, short_data=None):
    """
    Comite completo ETH:
    BULL/RANGE -> LONG (follower + breakout ETH) con TP/SL adaptados
    BEAR -> SHORT (segun short_method)
    """
    regimes_btc = df_btc.apply(lambda r: detect_regime(r), axis=1)
    regimes_eth = df_eth.apply(lambda r: detect_regime(r), axis=1)

    # Cross data
    eth_ret = df_eth['close'].pct_change()
    btc_close = df_btc['close'].reindex(df_eth.index, method='ffill')
    btc_ret = btc_close.pct_change()
    corr_20 = eth_ret.rolling(20).corr(btc_ret)

    results = []
    all_trades = []

    for start_s, end_s in WF_FOLDS:
        period = f"{start_s[:7]}/{end_s[5:7]}"
        trades = []

        # Para ML SHORT, entrenar modelo en expanding window
        short_model_data = None
        if short_method == 'ml' and short_data is not None:
            from sklearn.preprocessing import StandardScaler
            train_mask = df_eth.index < start_s
            y_tr = short_data['labels'][train_mask]
            bear_tr = regimes_eth[train_mask] == 'BEAR'
            valid = y_tr.notna() & bear_tr
            X_tr = df_eth.loc[train_mask, short_data['features']][valid].fillna(0)
            y_tr_v = y_tr[valid]
            if len(X_tr) >= 200 and y_tr_v.sum() >= 10 and (len(y_tr_v) - y_tr_v.sum()) >= 10:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr.values)
                model = short_data['constructor']()
                try:
                    model.fit(X_tr_s, y_tr_v)
                    short_model_data = {'model': model, 'scaler': scaler,
                                       'features': short_data['features'],
                                       'threshold': short_data['threshold']}
                except:
                    pass

        for i in range(30, len(df_eth)):
            ts = df_eth.index[i]
            if ts < pd.Timestamp(start_s, tz='UTC') or ts > pd.Timestamp(end_s, tz='UTC'):
                continue
            if i + 18 >= len(df_eth):
                continue

            regime_eth = regimes_eth.iloc[i]
            trade = None

            if regime_eth in ('BULL', 'RANGE'):
                # LONG: follower + breakout ETH (con TP/SL adaptados)
                # Follower
                if ts in df_btc.index:
                    btc_i = df_btc.index.get_loc(ts)
                    if btc_i >= 30:
                        regime_btc = regimes_btc.iloc[btc_i]
                        from evaluate_eth_v2 import detect_breakout_b_btc, detect_pullback_btc
                        btc_signal = None
                        if regime_btc in ('BULL', 'RANGE'):
                            btc_signal = detect_breakout_b_btc(df_btc, btc_i)
                            if btc_signal is None and regime_btc == 'BULL':
                                btc_signal = detect_pullback_btc(df_btc, btc_i)
                        if btc_signal is not None:
                            c = corr_20.get(ts, 0)
                            if not pd.isna(c) and c >= 0.5:
                                row = df_eth.iloc[i]
                                entry = float(row['close'])
                                atr_pct = float(row.get('atr_pct', 2.5))
                                sl_pct = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
                                tp_pct = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
                                trade = {'direction': 'LONG',
                                         'setup': f"FOLLOW_{btc_signal['setup']}",
                                         'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}

                # Breakout ETH
                if trade is None:
                    trade = detect_breakout_eth(df_eth, i)

            elif regime_eth == 'BEAR':
                # SHORT
                if short_method == 'rules':
                    trade = detect_short_meanrev(df_eth, i)
                    if trade is None:
                        trade = detect_short_breakdown(df_eth, i)
                elif short_method == 'ml' and short_model_data is not None:
                    row = df_eth.iloc[i]
                    feats = short_model_data['features']
                    x = pd.DataFrame([row[feats].fillna(0).values], columns=feats)
                    x_s = short_model_data['scaler'].transform(x)
                    prob = short_model_data['model'].predict_proba(x_s)[0][1]
                    if prob >= short_model_data['threshold']:
                        entry = float(row['close'])
                        atr_pct = float(row.get('atr_pct', 2.5))
                        sl_pct = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
                        tp_pct = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
                        trade = {'direction': 'SHORT', 'setup': 'SHORT_ML',
                                 'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}

            if trade is None:
                continue

            if trade['direction'] == 'LONG':
                out = sim_trade_fixed(df_eth, i, trade['entry'],
                                      trade['tp_pct'], trade['sl_pct'], max_bars=18)
            else:
                out = sim_short(df_eth, i, trade['entry'],
                               trade['tp_pct'], trade['sl_pct'], max_bars=16)

            trades.append({
                'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                'setup': trade['setup'], 'direction': trade['direction'],
            })

        m = metrics(trades, period)
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                        'pf': m['pf'], 'ok': ok})
        all_trades.extend(trades)

    folds_ok = sum(1 for r in results if r['ok'])
    return {'folds': results, 'folds_ok': folds_ok, 'all_trades': all_trades}


# ============================================================
# PRINT HELPERS
# ============================================================

def print_wf(wf, label=''):
    print(f"\n  {'Periodo':<14} | {'N':>4} | {'WR':>7} | {'PF':>6} | OK")
    print("  " + "-" * 45)
    for r in wf['folds']:
        ok_s = '+' if r['ok'] else '-'
        wr_s = f"{r['wr']:.1%}" if r['n'] > 0 else 'n/a'
        pf_s = f"{r['pf']:.2f}" if r['n'] > 0 else 'n/a'
        print(f"  {r['period']:<14} | {r['n']:>4} | {wr_s:>7} | {pf_s:>6} | {ok_s}")
    print(f"\n  Folds OK: {wf['folds_ok']}/12")


def print_oos(trades, label=''):
    oos = [t for t in trades if OOS_START <= str(t['ts'])[:10] <= OOS_END]
    m = metrics(oos, label)
    if m['n'] > 0:
        cum, max_dd = equity_stats(oos)
        print(f"  OOS: N={m['n']} | WR={m['wr']:.1%} | PF={m['pf']:.2f} | "
              f"{m['trades_pm']:.1f}t/m | ${1000*cum:.0f} | DD={max_dd:.1%}")
        # Breakdown
        setups = set(t.get('setup', '?') for t in oos)
        if len(setups) > 1:
            for s in sorted(setups):
                st = [t for t in oos if t.get('setup') == s]
                ms = metrics(st, s)
                if ms['n'] > 0:
                    print(f"    {s}: N={ms['n']} WR={ms['wr']:.1%} PF={ms['pf']:.2f}")
    else:
        print("  OOS: Sin trades")
    return m


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("ETH BEAR — ML SHORT + ENSEMBLE + COMITE COMPLETO")
    print("=" * 70)

    # Cargar datos
    print("\nCargando datos...")
    df_eth_raw = load_pair_4h('ETH')
    df_btc_raw = load_btc_4h()

    df_eth = compute_features_4h(df_eth_raw)
    df_eth = add_extra_features(df_eth)
    df_daily_eth = compute_macro_daily(df_eth)
    df_eth = merge_daily_to_4h(df_eth, df_daily_eth)
    df_eth = add_eth_specific_features(df_eth, df_btc_raw)

    df_btc = compute_features_4h(df_btc_raw)
    df_btc = add_extra_features(df_btc)
    df_daily_btc = compute_macro_daily(df_btc)
    df_btc = merge_daily_to_4h(df_btc, df_daily_btc)

    print(f"  ETH: {len(df_eth)} bars | BTC: {len(df_btc)} bars")

    # Regimenes
    regimes = df_eth.apply(lambda r: detect_regime(r), axis=1)
    bear_pct = (regimes == 'BEAR').mean()
    bear_bars = (regimes == 'BEAR').sum()
    print(f"  ETH BEAR: {bear_pct:.1%} ({bear_bars} bars)")

    # SHORT labels (TP/SL adaptados a ETH vol)
    print("\nCreando SHORT labels (TP=3%, SL=2%, max_bars=16)...")
    labels_short = create_short_labels(df_eth, tp_pct=0.03, sl_pct=0.02, max_bars=16)
    bear_labels = labels_short[regimes == 'BEAR']
    valid_bear = bear_labels[bear_labels.notna()]
    print(f"  BEAR bars con label: {len(valid_bear)}, positive rate: {valid_bear.mean():.1%}")

    # Features disponibles
    available_base = [f for f in SHORT_FEATURES_BASE if f in df_eth.columns]
    available_eth = [f for f in SHORT_FEATURES_ETH if f in df_eth.columns]
    print(f"  Features base: {len(available_base)} | ETH: {len(available_eth)}")

    models = get_short_models()
    print(f"  Modelos: {list(models.keys())}")

    # ========================================
    # PARTE 1: ML INDIVIDUAL (base features)
    # ========================================
    print(f"\n{'='*70}")
    print("PARTE 1: ML SHORT INDIVIDUAL (features base, solo BEAR)")
    print(f"{'='*70}")

    ml_base = wf_short_ml(df_eth, labels_short, available_base, models,
                           thresholds=[0.45, 0.50, 0.55, 0.60])

    for name, data in ml_base.items():
        print(f"\n  {name} (threshold={data['threshold']}):")
        fok = data['folds_ok']
        oos = [t for t in data['all_trades'] if OOS_START <= str(t['ts'])[:10] <= OOS_END]
        m = metrics(oos, name)
        print(f"    Folds OK: {fok}/12 | OOS: N={m['n']} WR={m['wr']:.1%} PF={m['pf']:.2f}")

    # ========================================
    # PARTE 2: ML INDIVIDUAL (ETH features)
    # ========================================
    print(f"\n{'='*70}")
    print("PARTE 2: ML SHORT INDIVIDUAL (features ETH-especificos, solo BEAR)")
    print(f"{'='*70}")

    ml_eth = wf_short_ml(df_eth, labels_short, available_eth, models,
                          thresholds=[0.45, 0.50, 0.55, 0.60])

    best_short_model = None
    best_short_pf = 0
    best_short_features = None

    for name, data in ml_eth.items():
        print(f"\n  {name} (threshold={data['threshold']}):")
        fok = data['folds_ok']
        oos = [t for t in data['all_trades'] if OOS_START <= str(t['ts'])[:10] <= OOS_END]
        m = metrics(oos, name)
        cum, dd = equity_stats(oos) if oos else (1.0, 0)
        print(f"    Folds OK: {fok}/12 | OOS: N={m['n']} WR={m['wr']:.1%} PF={m['pf']:.2f} DD={dd:.1%}")
        if m['pf'] > best_short_pf and m['n'] >= 5:
            best_short_pf = m['pf']
            best_short_model = name
            best_short_features = available_eth

    # ========================================
    # PARTE 3: ENSEMBLES
    # ========================================
    print(f"\n{'='*70}")
    print("PARTE 3: ENSEMBLES (Voting, AvgProb, Stacking)")
    print(f"{'='*70}")

    ens_results = wf_ensemble(df_eth, labels_short, available_eth, models, threshold=0.55)

    for ens_name, data in ens_results.items():
        fok = data['folds_ok']
        oos = [t for t in data['all_trades'] if OOS_START <= str(t['ts'])[:10] <= OOS_END]
        m = metrics(oos, ens_name)
        cum, dd = equity_stats(oos) if oos else (1.0, 0)
        print(f"  {ens_name}: Folds={fok}/12 | N={m['n']} WR={m['wr']:.1%} PF={m['pf']:.2f} DD={dd:.1%}")
        if m['pf'] > best_short_pf and m['n'] >= 5:
            best_short_pf = m['pf']
            best_short_model = f'ENS_{ens_name}'

    # ========================================
    # PARTE 4: REGLAS SHORT
    # ========================================
    print(f"\n{'='*70}")
    print("PARTE 4: REGLAS SHORT (MeanRev + Breakdown, solo BEAR)")
    print(f"{'='*70}")

    wf_rules = run_rules_short(df_eth)
    print_wf(wf_rules, 'Rules SHORT')
    m_rules = print_oos(wf_rules['all_trades'], 'Rules SHORT')

    # ========================================
    # PARTE 5: COMITES COMPLETOS
    # ========================================
    print(f"\n{'='*70}")
    print("PARTE 5: COMITES COMPLETOS (LONG rules + SHORT)")
    print(f"{'='*70}")

    # A) Solo LONG (sin SHORT) — baseline
    print("\n--- A) Solo LONG (BULL/RANGE), sin SHORT ---")
    wf_long_only = run_full_committee(df_eth, df_btc, 'none')
    print_wf(wf_long_only, 'LONG only')
    m_long_only = print_oos(wf_long_only['all_trades'], 'LONG only')

    # B) LONG + SHORT rules
    print("\n--- B) LONG + SHORT rules (MeanRev + Breakdown) ---")
    wf_rules_combo = run_full_committee(df_eth, df_btc, 'rules')
    print_wf(wf_rules_combo, 'LONG + SHORT rules')
    m_rules_combo = print_oos(wf_rules_combo['all_trades'], 'LONG + SHORT rules')

    # C) LONG + SHORT ML (mejor modelo)
    if best_short_model and not best_short_model.startswith('ENS_'):
        print(f"\n--- C) LONG + SHORT ML ({best_short_model}) ---")
        # Encontrar threshold
        best_data = ml_eth.get(best_short_model, ml_base.get(best_short_model))
        if best_data:
            short_data = {
                'labels': labels_short,
                'features': available_eth,
                'constructor': models[best_short_model],
                'threshold': best_data['threshold'],
            }
            wf_ml_combo = run_full_committee(df_eth, df_btc, 'ml', short_data)
            print_wf(wf_ml_combo, f'LONG + SHORT ML ({best_short_model})')
            m_ml_combo = print_oos(wf_ml_combo['all_trades'], f'LONG + SHORT ML')
        else:
            wf_ml_combo = None
            m_ml_combo = metrics([], 'none')
    else:
        wf_ml_combo = None
        m_ml_combo = metrics([], 'none')

    # ========================================
    # RESUMEN FINAL
    # ========================================
    print(f"\n\n{'='*70}")
    print("RESUMEN FINAL — ETH BEAR SHORT + COMITE")
    print(f"{'='*70}")

    print("\n  SHORT standalone (solo BEAR):")
    print(f"  {'Metodo':<20} | {'WF':>5} | {'N':>4} | {'WR':>7} | {'PF':>6}")
    print("  " + "-" * 55)

    # ML individual
    for name in ml_eth:
        data = ml_eth[name]
        oos = [t for t in data['all_trades'] if OOS_START <= str(t['ts'])[:10] <= OOS_END]
        m = metrics(oos, name)
        print(f"  ML {name:<16} | {data['folds_ok']:>2}/12 | {m['n']:>4} | {m['wr']:.1%} | {m['pf']:.2f}")

    # Ensembles
    for ens_name in ens_results:
        data = ens_results[ens_name]
        oos = [t for t in data['all_trades'] if OOS_START <= str(t['ts'])[:10] <= OOS_END]
        m = metrics(oos, ens_name)
        print(f"  ENS {ens_name:<15} | {data['folds_ok']:>2}/12 | {m['n']:>4} | {m['wr']:.1%} | {m['pf']:.2f}")

    # Rules
    oos_r = [t for t in wf_rules['all_trades'] if OOS_START <= str(t['ts'])[:10] <= OOS_END]
    mr = metrics(oos_r, 'Rules')
    print(f"  Rules SHORT        | {wf_rules['folds_ok']:>2}/12 | {mr['n']:>4} | {mr['wr']:.1%} | {mr['pf']:.2f}")

    print(f"\n  Comites completos (LONG + SHORT):")
    combos = [
        ('Solo LONG', wf_long_only, m_long_only),
        ('LONG+SHORT rules', wf_rules_combo, m_rules_combo),
    ]
    if wf_ml_combo:
        combos.append((f'LONG+SHORT ML({best_short_model})', wf_ml_combo, m_ml_combo))

    print(f"  {'Comite':<28} | {'WF':>5} | {'N':>5} | {'WR':>7} | {'PF':>6} | Veredicto")
    print("  " + "-" * 70)
    for name, wf, m in combos:
        passed = wf['folds_ok'] >= 7 and m['pf'] >= 1.2
        verd = 'APROBADO' if passed else 'RECHAZADO'
        print(f"  {name:<28} | {wf['folds_ok']:>2}/12 | {m['n']:>5} | "
              f"{m['wr']:.1%} | {m['pf']:.2f} | {verd}")

    print(f"\n  {'='*70}")
    best_combo = max(combos, key=lambda x: x[2]['pf'] if x[2]['n'] >= 10 else 0)
    if best_combo[1]['folds_ok'] >= 7 and best_combo[2]['pf'] >= 1.2:
        print(f"  MEJOR COMITE: {best_combo[0]} (PF={best_combo[2]['pf']:.2f})")
    else:
        print(f"  Mejor intento: {best_combo[0]} (PF={best_combo[2]['pf']:.2f})")
        print(f"  SHORT en BEAR puede no agregar valor neto al comite ETH")
    print(f"  {'='*70}")


if __name__ == '__main__':
    main()
