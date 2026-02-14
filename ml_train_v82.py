"""
ML V8.2: Pipeline Multi-Modelo - Trading Profesional
=====================================================
V7: un modelo hace todo (34 TA features).
V8.2: pipeline de modelos especializados como un equipo de trading:

  1. MacroGate (Modelo 1): "Es seguro operar hoy?"
     - DXY, Oro, S&P500, Bonos, ETH/BTC
     - Bloquea trades en dias de risk-off

  2. RegimeV2 (Modelo 2): "Que tipo de mercado es?"
     - BTC + macro + sentimiento
     - BULL/BEAR/RANGE con probabilidades

  3. V7 Signal Models (sin cambios): "Que par tiene senal?"
     - 34 TA features por par
     - LightGBM regresion

  4. ConvictionScorer (Modelo 3): "Cuanto arriesgar?"
     - Combina macro + regime + V7 signal
     - Ajusta sizing

V7 NO se toca. Los modelos nuevos FILTRAN cuando usar V7.

Ejecutar: python -u ml_train_v82.py
"""

import ccxt
import pandas as pd
import numpy as np
import pandas_ta as pta
import lightgbm as lgb
import optuna
import joblib
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Import V7 functions (reuse, don't rewrite)
from ml_train_v7 import (
    download_pair, download_all, detect_regime, map_regime_4h,
    compute_features, train_all, backtest, print_results,
    PAIRS, COMMISSION, SLIPPAGE, START_CAPITAL, MIN_TRAIN_CANDLES,
    MAX_CONCURRENT, MAX_DD_PCT, MAX_DAILY_LOSS_PCT, RISK_PER_TRADE,
    MAX_NOTIONAL, TRAILING_ACTIVATION, TRAILING_LOCK,
    LEV, ATR_TP, ATR_SL,
)
from macro_data import download_all_macro, compute_macro_features
from alt_data_fetcher import download_all_alt_data, download_fear_greed

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

# MacroGate threshold: block if P(V7_profitable) < this
MACRO_GATE_THRESHOLD = 0.45


# ============================================================
# 1. DOWNLOAD ALL DATA (V7 + macro + alt)
# ============================================================
def download_everything():
    """Descarga todos los datos necesarios para V8.2."""
    # V7 data (4h candles + BTC daily)
    all_data, btc_d = download_all()

    # Macro data (DXY, Gold, SPY, TNX, ETH/BTC)
    macro = download_all_macro()

    # Fear & Greed (for RegimeV2)
    fng = download_fear_greed()
    if fng is not None:
        print(f'  Fear&Greed  {len(fng):>6,} dias | {fng.index[0].date()} a {fng.index[-1].date()}')

    return all_data, btc_d, macro, fng


# ============================================================
# 2. MACRO GATE (Model 1)
# ============================================================
def generate_v7_oos_trades(all_data, regime, train_end, n_trials=20):
    """Generate V7 OOS trades for the training period using nested walk-forward.

    Problem: V7's train_all() only returns TEST period predictions.
    To train MacroGate, we need to know when V7 was profitable DURING TRAINING.

    Solution: split the training period in half:
      - Train V7 on early half
      - Get OOS predictions for late half
      - Run backtest on those OOS predictions
      - Return trades (with unbiased daily PnL labels)
    """
    # Find midpoint of training period
    all_dates = set()
    for pair, df in all_data.items():
        all_dates.update(df.loc[:train_end].index.tolist())
    all_dates = sorted(all_dates)
    if len(all_dates) < 2000:
        return []

    mid_idx = len(all_dates) // 2
    mid_point = all_dates[mid_idx].strftime('%Y-%m')

    print(f'    MacroGate nested WF: train V7 on [start, {mid_point}], '
          f'evaluate on [{mid_point}, {train_end}]')

    # Train V7 on early half (fewer Optuna trials for speed)
    early_models = train_all(all_data, mid_point, horizon=5, n_trials=n_trials)

    # Filter predictions to [mid_point, train_end] (the late training period)
    late_models = {}
    for p, info in early_models.items():
        mask = info['index'] < train_end
        if mask.sum() > 0:
            late_models[p] = {
                **info,
                'preds': info['preds'][mask],
                'index': info['index'][mask],
            }

    if not late_models:
        print(f'    MacroGate: no late-period models available')
        return []

    # Run V7 backtest on late training period
    trades, _, _, _ = backtest(
        all_data, regime, late_models,
        use_trailing=True, use_atr=True, use_compound=False, max_notional=300,
    )
    print(f'    MacroGate nested WF: {len(trades)} OOS trades for gate training')
    return trades


def create_macro_gate_labels(trades, macro_feat):
    """Crea labels para MacroGate: fue V7 rentable cada dia?

    Args:
        trades: lista de trades de V7 backtest
        macro_feat: DataFrame con macro features (index=fecha)

    Returns:
        X (macro features), y (1=rentable, 0=perdida) alineados por fecha
    """
    if not trades:
        return None, None

    # Aggregate V7 PnL per day
    daily_pnl = defaultdict(float)
    for t in trades:
        ds = t['time'].strftime('%Y-%m-%d')
        daily_pnl[ds] += t['pnl']

    # Create daily labels: 1 if net positive, 0 if net negative
    labels = pd.Series(daily_pnl)
    labels.index = pd.to_datetime(labels.index)
    y = (labels > 0).astype(int)

    # Align macro features to these dates
    # Macro features are already shifted(1) internally
    macro_daily = macro_feat.reindex(y.index, method='ffill')
    valid = macro_daily.dropna().index.intersection(y.index)

    if len(valid) < 30:
        return None, None

    return macro_daily.loc[valid], y.loc[valid]


def train_macro_gate(X, y, n_trials=30):
    """Entrena MacroGate: LightGBM clasificador binario.

    Returns: modelo entrenado, lista de feature columns
    """
    fcols = list(X.columns)
    X_clean = X[fcols].fillna(0)

    sp = int(len(X_clean) * 0.8)
    Xt, Xv = X_clean.iloc[:sp], X_clean.iloc[sp:]
    yt, yv = y.iloc[:sp], y.iloc[sp:]

    def obj(trial):
        p = {
            'n_estimators': trial.suggest_int('ne', 50, 300),
            'max_depth': trial.suggest_int('md', 2, 6),
            'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('ss', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('cs', 0.4, 1.0),
            'min_child_samples': trial.suggest_int('mc', 20, 100),
            'reg_alpha': trial.suggest_float('ra', 1e-5, 5.0, log=True),
            'reg_lambda': trial.suggest_float('rl', 1e-5, 5.0, log=True),
            'num_leaves': trial.suggest_int('nl', 10, 50),
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
            'objective': 'binary',
        }
        m = lgb.LGBMClassifier(**p)
        m.fit(Xt, yt, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(20, verbose=False)])
        probs = m.predict_proba(Xv)[:, 1]
        # Metric: how well does it predict V7 profitable days?
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(yv, probs)
        except:
            auc = 0.5
        return auc

    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params.copy()
    bp.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1, 'objective': 'binary'})
    rename = {'ne': 'n_estimators', 'md': 'max_depth', 'lr': 'learning_rate',
              'ss': 'subsample', 'cs': 'colsample_bytree', 'mc': 'min_child_samples',
              'ra': 'reg_alpha', 'rl': 'reg_lambda', 'nl': 'num_leaves'}
    bp2 = {rename.get(k, k): v for k, v in bp.items()}

    mdl = lgb.LGBMClassifier(**bp2)
    mdl.fit(X_clean, y, eval_set=[(Xv, yv)],
            callbacks=[lgb.early_stopping(30, verbose=False)])

    auc = study.best_value
    return mdl, fcols, auc


# ============================================================
# 3. REGIME V2 (Model 2)
# ============================================================
def compute_regime_features(btc_daily, macro_feat, fng_df=None):
    """Computa features para el detector de regimen mejorado."""
    c = btc_daily['close']
    h, l = btc_daily['high'], btc_daily['low']

    feat = pd.DataFrame(index=btc_daily.index)

    # BTC structure
    ema20 = pta.ema(c, length=20)
    ema50 = pta.ema(c, length=50)
    ema200 = pta.ema(c, length=200)
    feat['btc_ema20_vs_ema50'] = (ema20 - ema50) / (ema50 + 1e-10)
    feat['btc_vs_ema200'] = (c - ema200) / (ema200 + 1e-10)
    feat['btc_ret_20d'] = c.pct_change(20)
    feat['btc_ret_5d'] = c.pct_change(5)

    adx_df = pta.adx(h, l, c, length=14)
    if adx_df is not None:
        feat['btc_adx'] = adx_df.iloc[:, 0]

    feat['btc_vol_20d'] = c.pct_change().rolling(20).std()

    # Macro context (already shifted inside compute_macro_features)
    if macro_feat is not None:
        for col in ['dxy_ret_20d', 'spy_ret_20d', 'ethbtc_ret_20d',
                     'dxy_vs_ema50', 'gold_vs_ema50', 'dxy_spy_diverge']:
            if col in macro_feat.columns:
                feat[col] = macro_feat[col].reindex(feat.index, method='ffill')

    # Fear & Greed
    if fng_df is not None:
        fng_shifted = fng_df['fng_value'].shift(1)
        fng_aligned = fng_shifted.reindex(feat.index, method='ffill')
        feat['fng_value'] = fng_aligned
        feat['fng_ma7'] = fng_aligned.rolling(7, min_periods=1).mean()

    return feat


def create_regime_labels(btc_daily, fwd_days=20, bull_thresh=0.05, bear_thresh=-0.05):
    """Crea labels de regimen basados en retorno futuro de BTC."""
    c = btc_daily['close']
    fwd = c.shift(-fwd_days) / c - 1

    labels = pd.Series('RANGE', index=btc_daily.index)
    labels[fwd > bull_thresh] = 'BULL'
    labels[fwd < bear_thresh] = 'BEAR'
    # Drop last fwd_days rows (no future data)
    labels = labels.iloc[:-fwd_days]
    return labels


def train_regime_model(feat, labels, n_trials=30):
    """Entrena RegimeV2: LightGBM clasificador 3 clases."""
    valid_idx = feat.dropna().index.intersection(labels.index)
    if len(valid_idx) < 100:
        return None, None, 0

    X = feat.loc[valid_idx].fillna(0)
    y = labels.loc[valid_idx]
    fcols = list(X.columns)

    # Encode labels
    label_map = {'BULL': 0, 'BEAR': 1, 'RANGE': 2}
    y_enc = y.map(label_map)

    sp = int(len(X) * 0.8)
    Xt, Xv = X.iloc[:sp], X.iloc[sp:]
    yt, yv = y_enc.iloc[:sp], y_enc.iloc[sp:]

    def obj(trial):
        p = {
            'n_estimators': trial.suggest_int('ne', 50, 300),
            'max_depth': trial.suggest_int('md', 2, 6),
            'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('ss', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('cs', 0.4, 1.0),
            'min_child_samples': trial.suggest_int('mc', 20, 100),
            'reg_alpha': trial.suggest_float('ra', 1e-5, 5.0, log=True),
            'reg_lambda': trial.suggest_float('rl', 1e-5, 5.0, log=True),
            'num_leaves': trial.suggest_int('nl', 10, 50),
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
            'objective': 'multiclass', 'num_class': 3,
        }
        m = lgb.LGBMClassifier(**p)
        m.fit(Xt, yt, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(20, verbose=False)])
        acc = (m.predict(Xv) == yv.values).mean()
        return acc

    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params.copy()
    bp.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1,
               'objective': 'multiclass', 'num_class': 3})
    rename = {'ne': 'n_estimators', 'md': 'max_depth', 'lr': 'learning_rate',
              'ss': 'subsample', 'cs': 'colsample_bytree', 'mc': 'min_child_samples',
              'ra': 'reg_alpha', 'rl': 'reg_lambda', 'nl': 'num_leaves'}
    bp2 = {rename.get(k, k): v for k, v in bp.items()}

    mdl = lgb.LGBMClassifier(**bp2)
    mdl.fit(X, y_enc, eval_set=[(Xv, yv)],
            callbacks=[lgb.early_stopping(30, verbose=False)])

    acc = study.best_value
    return mdl, fcols, acc


# ============================================================
# 4. BACKTEST WITH MACRO GATE
# ============================================================
def backtest_with_gate(all_data, regime_daily, models, macro_gate_model,
                       macro_feat, macro_fcols, gate_threshold=MACRO_GATE_THRESHOLD,
                       regime_model=None, regime_feat=None,
                       thresh=0.7, use_trailing=True, use_atr=True,
                       max_notional=MAX_NOTIONAL):
    """Backtest V7 + MacroGate filter (+ optional RegimeV2).

    Same as V7 backtest but:
    - MacroGate can BLOCK all trades for a day
    - RegimeV2 can replace rule-based regime
    """
    # Timeline
    all_t = set()
    for p in models:
        if p in all_data:
            all_t.update(all_data[p].index.tolist())
    timeline = sorted(all_t)
    if not timeline:
        return [], [], 0, False

    # Regime
    if regime_model is not None and regime_feat is not None:
        # Use trained regime model
        inv_map = {0: 'BULL', 1: 'BEAR', 2: 'RANGE'}
        regime_preds = regime_model.predict(regime_feat.fillna(0))
        regime_series = pd.Series(
            [inv_map.get(p, 'RANGE') for p in regime_preds],
            index=regime_feat.index,
        )
        regime_4h = regime_series.reindex(pd.DatetimeIndex(timeline), method='ffill').fillna('RANGE')
    else:
        regime_4h = map_regime_4h(regime_daily, pd.DatetimeIndex(timeline))

    # Pre-compute MacroGate scores per day
    gate_scores = {}
    if macro_gate_model is not None:
        macro_aligned = macro_feat[macro_fcols].fillna(0)
        probs = macro_gate_model.predict_proba(macro_aligned)[:, 1]
        for dt, p in zip(macro_aligned.index, probs):
            gate_scores[dt.strftime('%Y-%m-%d')] = p

    # Pre-compute preds dict and ATRs
    pred_dict = {}
    atr_dict = {}
    for pair, info in models.items():
        pred_dict[pair] = dict(zip(info['index'], info['preds']))
        df = all_data[pair]
        atr_dict[pair] = pta.atr(df['high'], df['low'], df['close'], length=14)

    balance = START_CAPITAL
    peak = START_CAPITAL
    max_dd = 0.0
    positions = {}
    trades = []
    daily_pnl = defaultdict(float)
    paused_until = None
    killed = False
    blocked_days = 0
    total_days = 0

    for t in timeline:
        if killed:
            break
        ds = t.strftime('%Y-%m-%d')

        if paused_until and t.date() <= paused_until:
            continue

        reg = regime_4h.loc[t] if t in regime_4h.index else 'RANGE'
        lev = LEV[reg]

        # === UPDATE POSITIONS (same as V7 - always manage open positions) ===
        to_close = []
        for pair, pos in positions.items():
            df = all_data[pair]
            if t not in df.index:
                continue

            bh = df.loc[t, 'high']
            bl = df.loc[t, 'low']
            bc = df.loc[t, 'close']
            pos['bars'] += 1

            # Trailing update
            if use_trailing and pos.get('trail_on'):
                if pos['dir'] == 1:
                    new = bh * (1 - TRAILING_LOCK * pos['atr_pct'])
                    pos['trail_sl'] = max(pos.get('trail_sl', 0), new)
                else:
                    new = bl * (1 + TRAILING_LOCK * pos['atr_pct'])
                    pos['trail_sl'] = min(pos.get('trail_sl', 1e15), new)

            # Check exits
            ex_price = ex_reason = None
            if pos['dir'] == 1:
                if bh >= pos['tp']: ex_price, ex_reason = pos['tp'], 'TP'
                elif use_trailing and pos.get('trail_sl') and bl <= pos['trail_sl']:
                    ex_price, ex_reason = pos['trail_sl'], 'TRAIL'
                elif bl <= pos['sl']: ex_price, ex_reason = pos['sl'], 'SL'
                elif pos['bars'] >= pos['mh']: ex_price, ex_reason = bc, 'TIMEOUT'
                elif use_trailing and not pos.get('trail_on'):
                    g = (bh - pos['entry']) / pos['entry']
                    if g >= pos['tp_pct'] * TRAILING_ACTIVATION:
                        pos['trail_on'] = True
                        pos['trail_sl'] = pos['entry'] * (1 + g * 0.3)
            else:
                if bl <= pos['tp']: ex_price, ex_reason = pos['tp'], 'TP'
                elif use_trailing and pos.get('trail_sl') and bh >= pos['trail_sl']:
                    ex_price, ex_reason = pos['trail_sl'], 'TRAIL'
                elif bh >= pos['sl']: ex_price, ex_reason = pos['sl'], 'SL'
                elif pos['bars'] >= pos['mh']: ex_price, ex_reason = bc, 'TIMEOUT'
                elif use_trailing and not pos.get('trail_on'):
                    g = (pos['entry'] - bl) / pos['entry']
                    if g >= pos['tp_pct'] * TRAILING_ACTIVATION:
                        pos['trail_on'] = True
                        pos['trail_sl'] = pos['entry'] * (1 - g * 0.3)

            if ex_price:
                if pos['dir'] == 1:
                    gpnl = (ex_price - pos['entry']) / pos['entry']
                else:
                    gpnl = (pos['entry'] - ex_price) / pos['entry']
                cost = pos['not'] * (COMMISSION + SLIPPAGE) * 2
                pnl = pos['not'] * gpnl - cost
                balance += pnl
                daily_pnl[ds] += pnl

                trades.append({
                    'pair': pair, 'dir': pos['dir'], 'pnl': pnl,
                    'pnl_pct': gpnl * 100, 'reason': ex_reason,
                    'bars': pos['bars'], 'time': t, 'regime': pos['reg'],
                    'lev': pos['lev'], 'conf': pos['conf'], 'bal': balance,
                })
                to_close.append(pair)

        for p in to_close:
            del positions[p]

        # DD check
        if balance > peak: peak = balance
        dd = (peak - balance) / peak if peak > 0 else 0
        if dd > max_dd: max_dd = dd
        if dd >= MAX_DD_PCT:
            killed = True
            break

        # Daily loss check
        if daily_pnl[ds] < -(START_CAPITAL * MAX_DAILY_LOSS_PCT):
            paused_until = t.date()
            continue

        # === MACRO GATE CHECK ===
        # Track unique days for gate statistics
        if t.hour == 0:
            total_days += 1

        gate_score = gate_scores.get(ds, 0.5)
        if gate_score < gate_threshold:
            if t.hour == 0:
                blocked_days += 1
            continue  # BLOCKED by MacroGate - don't open new positions

        # === OPEN NEW POSITIONS (same as V7) ===
        if len(positions) >= MAX_CONCURRENT:
            continue

        cands = []
        for pair, info in models.items():
            if pair in positions: continue
            if pair not in all_data: continue
            df = all_data[pair]
            if t not in df.index: continue

            pred = pred_dict[pair].get(t)
            if pred is None: continue
            ps = info['pred_std']
            if ps < 1e-8: continue
            conf = abs(pred) / ps
            if conf < thresh: continue

            d = 1 if pred > 0 else -1

            # Regime filter
            if reg == 'BULL' and d == -1: continue
            if reg == 'BEAR' and d == 1: continue

            # Correlation filter
            same = sum(1 for p in positions.values() if p['dir'] == d)
            if same >= 2: continue

            cands.append({'pair': pair, 'dir': d, 'conf': conf})

        cands.sort(key=lambda x: x['conf'], reverse=True)

        for cand in cands:
            if len(positions) >= MAX_CONCURRENT: break
            pair = cand['pair']
            d = cand['dir']
            conf = cand['conf']
            df = all_data[pair]
            entry = df.loc[t, 'close']

            # ATR TP/SL
            atr_val = atr_dict[pair].get(t) if hasattr(atr_dict[pair], 'get') else (
                atr_dict[pair].loc[t] if t in atr_dict[pair].index else None)

            if use_atr and atr_val and not np.isnan(atr_val):
                atr_pct = atr_val / entry
                tp_pct = max(0.005, min(0.08, atr_pct * ATR_TP[reg]))
                sl_pct = max(0.003, min(0.04, atr_pct * ATR_SL[reg]))
            else:
                tp_pct = 0.03; sl_pct = 0.015

            mh = 30 if reg != 'RANGE' else 15

            # Sizing (can be adjusted by gate_score later for ConvictionScorer)
            risk_pct = RISK_PER_TRADE
            if conf > 2.0: risk_pct = 0.03
            elif conf > 1.5: risk_pct = 0.025

            base = START_CAPITAL
            risk_amt = base * risk_pct
            margin = risk_amt / (sl_pct * lev) if sl_pct > 0 else risk_amt
            notional = margin * lev
            notional = min(notional, max_notional)
            margin = notional / lev

            if d == 1:
                tp_pr = entry * (1 + tp_pct)
                sl_pr = entry * (1 - sl_pct)
            else:
                tp_pr = entry * (1 - tp_pct)
                sl_pr = entry * (1 + sl_pct)

            positions[pair] = {
                'entry': entry, 'dir': d, 'tp': tp_pr, 'sl': sl_pr,
                'tp_pct': tp_pct, 'sl_pct': sl_pct,
                'atr_pct': atr_val / entry if atr_val else 0.02,
                'not': notional, 'lev': lev, 'mh': mh,
                'bars': 0, 'conf': conf, 'reg': reg,
                'trail_on': False, 'trail_sl': None,
            }

    block_pct = blocked_days / max(total_days, 1) * 100
    return trades, block_pct, max_dd, killed


# ============================================================
# 5. METRICS EXTRACTION
# ============================================================
def extract_metrics(trades, max_dd, killed):
    """Extract metrics dict from trades for comparison."""
    if not trades:
        return None

    final = trades[-1]['bal']
    ret = ((final / START_CAPITAL) - 1) * 100
    total = len(trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    wr = wins / total * 100 if total else 0
    gp = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gl = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
    pf = gp / gl if gl > 0 else 0

    days = (trades[-1]['time'] - trades[0]['time']).days
    years = max(days / 365, 0.1)
    if final > 0 and years > 0.1:
        annual = ((final / START_CAPITAL) ** (1 / years) - 1) * 100
    else:
        annual = ret / years if years > 0 else 0

    dr = defaultdict(float)
    for t in trades: dr[t['time'].strftime('%Y-%m-%d')] += t['pnl']
    rv = list(dr.values())
    sharpe = (np.mean(rv) / np.std(rv) * np.sqrt(252)) if np.std(rv) > 0 else 0

    return {
        'ret': ret, 'annual': annual, 'trades': total, 'wr': wr,
        'pf': pf, 'dd': max_dd * 100, 'sharpe': sharpe, 'killed': killed,
        'final': final, 'days': days,
    }


# ============================================================
# 6. MAIN
# ============================================================
def main():
    t0 = time.time()

    # --- Download everything ---
    all_data, btc_d, macro, fng = download_everything()
    if len(all_data) < 3 or btc_d is None:
        print('ERROR: datos insuficientes'); return

    # --- Regime (V7 rule-based, used for baseline) ---
    regime = detect_regime(btc_d)
    bull = (regime == 'BULL').sum()
    bear = (regime == 'BEAR').sum()
    rng = (regime == 'RANGE').sum()
    print(f'\n  Regimen V7: BULL {bull}d ({bull / len(regime) * 100:.0f}%) | '
          f'BEAR {bear}d ({bear / len(regime) * 100:.0f}%) | '
          f'RANGE {rng}d ({rng / len(regime) * 100:.0f}%)')

    # --- Compute macro features ---
    macro_feat = compute_macro_features(
        macro.get('dxy'), macro.get('gold'), macro.get('spy'),
        macro.get('tnx'), macro.get('ethbtc'),
    )
    if macro_feat is None:
        print('ERROR: no macro features'); return
    print(f'  Macro features: {len(macro_feat.columns)} columnas, {len(macro_feat)} filas')

    # --- Regime features (for RegimeV2) ---
    regime_feat = compute_regime_features(btc_d, macro_feat, fng)
    regime_labels = create_regime_labels(btc_d)
    print(f'  Regime features: {len(regime_feat.columns)} columnas')

    # ================================================================
    # WALK-FORWARD: V7 vs V7+MacroGate vs V8.2 Full
    # ================================================================
    print('\n' + '=' * 70)
    print('WALK-FORWARD: V7 vs V7+MacroGate vs V8.2')
    print('=' * 70)

    folds = [
        ('2023-06', '2023-06', '2024-01', 'H2-2023'),
        ('2024-01', '2024-01', '2024-07', 'H1-2024'),
        ('2024-07', '2024-07', '2025-01', 'H2-2024'),
        ('2025-01', '2025-01', '2026-03', '2025+'),
    ]

    all_results = {'V7': [], 'V7+Gate': [], 'V8.2': []}

    for tr_e, te_s, te_e, fn in folds:
        print(f'\n  --- Fold {fn} (train hasta {tr_e}) ---')

        # Step 1: Train V7 models
        mdls = train_all(all_data, tr_e, horizon=5, n_trials=30)
        mdls_f = {}
        for p, i in mdls.items():
            m = (i['index'] >= te_s) & (i['index'] < te_e)
            if m.sum() > 0:
                mdls_f[p] = {**i, 'preds': i['preds'][m], 'index': i['index'][m]}
        if not mdls_f:
            print(f'    Sin modelos para test period')
            continue

        # Step 2: V7 baseline backtest
        tr_v7, _, mdd_v7, kil_v7 = backtest(
            all_data, regime, mdls_f,
            use_trailing=True, use_atr=True, use_compound=False, max_notional=300,
        )

        # Step 3: Train MacroGate on V7's TRAINING period results
        # Use nested walk-forward to get unbiased V7 results for training period
        oos_trades = generate_v7_oos_trades(all_data, regime, tr_e, n_trials=15)
        macro_X, macro_y = create_macro_gate_labels(oos_trades, macro_feat)

        macro_gate = None
        macro_fcols = []
        gate_auc = 0

        if macro_X is not None and len(macro_X) > 50:
            macro_gate, macro_fcols, gate_auc = train_macro_gate(macro_X, macro_y, n_trials=20)
            print(f'    MacroGate: AUC={gate_auc:.3f} | {len(macro_X)} training days')

        # Step 4: V7 + MacroGate backtest
        if macro_gate is not None:
            tr_gate, block_pct, mdd_gate, kil_gate = backtest_with_gate(
                all_data, regime, mdls_f, macro_gate, macro_feat, macro_fcols,
                gate_threshold=MACRO_GATE_THRESHOLD,
            )
        else:
            tr_gate, block_pct, mdd_gate, kil_gate = tr_v7, 0, mdd_v7, kil_v7

        # Step 5: Train RegimeV2
        regime_feat_train = regime_feat.loc[:tr_e]
        regime_labels_train = regime_labels.loc[:tr_e]
        regime_mdl, regime_fcols, regime_acc = train_regime_model(
            regime_feat_train, regime_labels_train, n_trials=20,
        )
        if regime_mdl:
            print(f'    RegimeV2: Acc={regime_acc:.3f}')

        # Step 6: V8.2 full (MacroGate + RegimeV2)
        if macro_gate is not None:
            tr_v82, block_v82, mdd_v82, kil_v82 = backtest_with_gate(
                all_data, regime, mdls_f, macro_gate, macro_feat, macro_fcols,
                gate_threshold=MACRO_GATE_THRESHOLD,
                regime_model=regime_mdl,
                regime_feat=regime_feat.loc[te_s:te_e] if regime_mdl else None,
            )
        else:
            tr_v82, block_v82, mdd_v82, kil_v82 = tr_v7, 0, mdd_v7, kil_v7

        # --- Print fold results ---
        def _fold_line(label, tr, mdd, kil, extra=''):
            if not tr:
                print(f'    {label:<12} Sin trades')
                return None
            fin = tr[-1]['bal']
            ret = ((fin / START_CAPITAL) - 1) * 100
            w = sum(1 for t in tr if t['pnl'] > 0)
            wr = w / len(tr) * 100
            gp = sum(t['pnl'] for t in tr if t['pnl'] > 0)
            gl = abs(sum(t['pnl'] for t in tr if t['pnl'] <= 0))
            pf = gp / gl if gl > 0 else 0
            mk = ' ***' if ret > 20 else (' **' if ret > 5 else (' *' if ret > 0 else ' --'))
            print(f'    {label:<12} {len(tr):>4}t | WR {wr:.0f}% | PF {pf:.2f} | '
                  f'Ret {ret:+.1f}% | DD {mdd * 100:.1f}%{mk}'
                  f'{" KILLED" if kil else ""}{extra}')
            return {'ret': ret, 'pf': pf, 'dd': mdd * 100, 'n': len(tr), 'wr': wr}

        r1 = _fold_line('V7', tr_v7, mdd_v7, kil_v7)
        r2 = _fold_line('V7+Gate', tr_gate, mdd_gate, kil_gate,
                         f' | Blocked {block_pct:.0f}%' if macro_gate else '')
        r3 = _fold_line('V8.2', tr_v82, mdd_v82, kil_v82,
                         f' | Blocked {block_v82:.0f}%' if macro_gate else '')

        if r1: all_results['V7'].append(r1)
        if r2: all_results['V7+Gate'].append(r2)
        if r3: all_results['V8.2'].append(r3)

    # --- Walk-forward summary ---
    print(f'\n  {"="*65}')
    print(f'  RESUMEN WALK-FORWARD')
    print(f'  {"="*65}')
    for config in ['V7', 'V7+Gate', 'V8.2']:
        res = all_results[config]
        if not res:
            print(f'  {config:<12} Sin resultados')
            continue
        prof = sum(1 for r in res if r['ret'] > 0)
        avg_pf = np.mean([r['pf'] for r in res])
        avg_ret = np.mean([r['ret'] for r in res])
        avg_dd = np.mean([r['dd'] for r in res])
        print(f'  {config:<12} {prof}/{len(res)} folds rentables | '
              f'PF avg {avg_pf:.2f} | Ret avg {avg_ret:+.1f}% | DD avg {avg_dd:.1f}%')

    # ================================================================
    # FULL BACKTEST (train on data until 2024-01, test on 2024+)
    # ================================================================
    print('\n' + '=' * 70)
    print('BACKTEST COMPLETO: V7 vs V7+MacroGate vs V8.2')
    print('=' * 70)

    models = train_all(all_data, '2024-01', horizon=5, n_trials=40)

    # V7 baseline
    tr_v7, _, mdd_v7, kil_v7 = backtest(
        all_data, regime, models,
        use_trailing=True, use_atr=True, use_compound=False, max_notional=300,
    )
    print_results(tr_v7, mdd_v7, kil_v7, 'V7 BASELINE')
    m_v7 = extract_metrics(tr_v7, mdd_v7, kil_v7)

    # Train MacroGate on training period V7 results (nested walk-forward)
    oos_trades = generate_v7_oos_trades(all_data, regime, '2024-01', n_trials=20)
    macro_X, macro_y = create_macro_gate_labels(oos_trades, macro_feat)

    macro_gate = None
    macro_fcols = []

    if macro_X is not None and len(macro_X) > 50:
        macro_gate, macro_fcols, gate_auc = train_macro_gate(macro_X, macro_y, n_trials=40)
        print(f'\n  MacroGate entrenado: AUC={gate_auc:.3f} | {len(macro_X)} dias de training')

        # Feature importance
        if hasattr(macro_gate, 'feature_importances_'):
            imp = sorted(zip(macro_fcols, macro_gate.feature_importances_),
                         key=lambda x: x[1], reverse=True)
            print(f'\n  MacroGate Feature Importance (top 10):')
            for i, (f, v) in enumerate(imp[:10]):
                bar = '#' * min(int(v / max(1, imp[0][1]) * 30), 30)
                print(f'    {i + 1:>2}. {f:<25} {v:>6.1f} {bar}')

    # V7 + MacroGate
    if macro_gate is not None:
        tr_gate, block_pct, mdd_gate, kil_gate = backtest_with_gate(
            all_data, regime, models, macro_gate, macro_feat, macro_fcols,
        )
        print_results(tr_gate, mdd_gate, kil_gate,
                       f'V7 + MACROGATE (blocked {block_pct:.0f}% of days)')
        m_gate = extract_metrics(tr_gate, mdd_gate, kil_gate)
    else:
        m_gate = m_v7

    # V8.2: MacroGate + RegimeV2
    regime_feat_train = regime_feat.loc[:'2024-01']
    regime_labels_train = regime_labels.loc[:'2024-01']
    regime_mdl, _, regime_acc = train_regime_model(regime_feat_train, regime_labels_train, n_trials=40)
    if regime_mdl:
        print(f'\n  RegimeV2 entrenado: Acc={regime_acc:.3f}')

    if macro_gate is not None:
        tr_v82, block_v82, mdd_v82, kil_v82 = backtest_with_gate(
            all_data, regime, models, macro_gate, macro_feat, macro_fcols,
            regime_model=regime_mdl,
            regime_feat=regime_feat.loc['2024-01':] if regime_mdl else None,
        )
        print_results(tr_v82, mdd_v82, kil_v82, f'V8.2 FULL PIPELINE (blocked {block_v82:.0f}%)')
        m_v82 = extract_metrics(tr_v82, mdd_v82, kil_v82)
    else:
        m_v82 = m_v7

    # ================================================================
    # COMPARISON TABLE
    # ================================================================
    print(f'\n{"="*70}')
    print('COMPARACION FINAL')
    print(f'{"="*70}')

    configs = [('V7', m_v7), ('V7+Gate', m_gate), ('V8.2', m_v82)]
    valid_configs = [(n, m) for n, m in configs if m is not None]

    if len(valid_configs) >= 2:
        header = f'  {"Metrica":<18}'
        for name, _ in valid_configs:
            header += f' {name:>10}'
        print(header)
        print(f'  {"-" * (18 + 11 * len(valid_configs))}')

        for key, label, fmt in [
            ('ret', 'Retorno %', '.1f'), ('trades', 'Trades', '.0f'),
            ('wr', 'Win Rate %', '.0f'), ('pf', 'Profit Factor', '.2f'),
            ('dd', 'Max DD %', '.1f'), ('sharpe', 'Sharpe', '.2f'),
        ]:
            line = f'  {label:<18}'
            vals = []
            for name, m in valid_configs:
                v = m.get(key, 0)
                vals.append(v)
                line += f' {v:>10{fmt}}'

            # Mark best
            if key == 'dd':
                best_idx = vals.index(min(vals))
            else:
                best_idx = vals.index(max(vals))
            if vals[best_idx] != vals[0]:  # If best is not V7
                line += ' <<<'
            print(line)

    # Verdict
    print(f'\n  VEREDICTO:')
    if m_v7 and m_gate:
        gate_better = (m_gate['pf'] > m_v7['pf']) or (m_gate['dd'] < m_v7['dd'])
        print(f'    MacroGate {"MEJORA" if gate_better else "NO mejora"} vs V7 solo')
        if gate_better:
            print(f'      PF: {m_v7["pf"]:.2f} -> {m_gate["pf"]:.2f}')
            print(f'      DD: {m_v7["dd"]:.1f}% -> {m_gate["dd"]:.1f}%')

    if m_v7 and m_v82:
        v82_pf_ok = m_v82['pf'] > 1.5
        v82_dd_ok = m_v82['dd'] < 15.0
        v82_better = m_v82['pf'] > m_v7['pf'] and m_v82['dd'] <= m_v7['dd'] + 2

        if v82_pf_ok and v82_dd_ok:
            print(f'\n    >>> V8.2 APROBADO: PF {m_v82["pf"]:.2f} > 1.5, DD {m_v82["dd"]:.1f}% < 15%')
        elif v82_better:
            print(f'\n    >>> V8.2 MEJORA PARCIAL: PF {m_v82["pf"]:.2f}, DD {m_v82["dd"]:.1f}%')
            print(f'        No alcanza criterios estrictos pero mejora vs V7')
        else:
            print(f'\n    >>> V8.2 NO APROBADO: PF {m_v82["pf"]:.2f}, DD {m_v82["dd"]:.1f}%')
            print(f'        Necesita mas trabajo')

    print(f'\nTiempo total: {(time.time() - t0) / 60:.1f} minutos')


if __name__ == '__main__':
    main()
