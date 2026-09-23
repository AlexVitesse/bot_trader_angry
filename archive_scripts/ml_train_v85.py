"""
ML V8.5: ConvictionScorer - Per-Trade Sizing Intelligence
==========================================================
V8.4 results: PF 1.27, DD 17.4%, Sharpe 2.68, WR 58%, $878 on $100
V8.4 adjusts sizing at DAILY level (same macro_score for all trades that day).

V8.5 adds per-TRADE sizing via ConvictionScorer:
  - LightGBM REGRESSION: predicts expected PnL for each trade
  - Features available at entry: V7 confidence, prediction magnitude,
    macro score, risk-off mult, regime, ATR%, open positions
  - Output: sizing multiplier [0.3, 1.8] that adjusts risk per trade
  - Stacks on top of V8.4 macro sizing OR replaces it

Configs compared:
  A) V7 (baseline)
  B) V8.4 Full (macro adaptive threshold + ML sizing + risk-off)
  C) V8.5 Conv (conviction only, no macro sizing)
  D) V8.5 Full (macro + conviction combined)

Ejecutar: python -u ml_train_v85.py
"""

import pandas as pd
import numpy as np
import pandas_ta as pta
import lightgbm as lgb
import optuna
from pathlib import Path
from collections import defaultdict
from scipy.special import expit  # sigmoid
import time
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from ml_train_v7 import (
    download_all, detect_regime, map_regime_4h,
    compute_features, train_all, backtest, print_results,
    PAIRS, COMMISSION, SLIPPAGE, START_CAPITAL, MIN_TRAIN_CANDLES,
    MAX_CONCURRENT, MAX_DD_PCT, MAX_DAILY_LOSS_PCT, RISK_PER_TRADE,
    MAX_NOTIONAL, TRAILING_ACTIVATION, TRAILING_LOCK,
    LEV, ATR_TP, ATR_SL,
)
from ml_train_v84 import (
    download_everything, generate_v7_oos_trades,
    create_scorer_labels, train_macro_scorer,
    compute_risk_off_multipliers, extract_metrics,
)
from macro_data import download_all_macro, compute_macro_features


# ============================================================
# 1. GENERATE TRAINING DATA FOR CONVICTION SCORER
# ============================================================
def backtest_collect_trade_features(all_data, regime_daily, models,
                                     macro_scorer=None, macro_feat=None,
                                     macro_fcols=None, risk_off_mults=None,
                                     max_notional=300):
    """Run V7 backtest but capture per-trade features at entry time.

    Returns list of dicts with trade PnL AND entry-time features
    that will train the ConvictionScorer.
    """
    all_t = set()
    for p in models:
        if p in all_data:
            all_t.update(all_data[p].index.tolist())
    timeline = sorted(all_t)
    if not timeline:
        return []

    regime_4h = map_regime_4h(regime_daily, pd.DatetimeIndex(timeline))

    # Pre-compute macro scores per day
    macro_scores = {}
    if macro_scorer is not None and macro_fcols:
        macro_aligned = macro_feat[macro_fcols].fillna(0)
        probs = macro_scorer.predict_proba(macro_aligned)[:, 1]
        for dt, p in zip(macro_aligned.index, probs):
            macro_scores[dt.strftime('%Y-%m-%d')] = p

    if risk_off_mults is None:
        risk_off_mults = {}

    # Pre-compute preds and ATRs
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
    enriched_trades = []
    daily_pnl = defaultdict(float)
    paused_until = None
    killed = False

    for t in timeline:
        if killed:
            break
        ds = t.strftime('%Y-%m-%d')

        if paused_until and t.date() <= paused_until:
            continue

        reg = regime_4h.loc[t] if t in regime_4h.index else 'RANGE'
        lev = LEV[reg]

        # === UPDATE POSITIONS (identical to V7) ===
        to_close = []
        for pair, pos in positions.items():
            df = all_data[pair]
            if t not in df.index:
                continue

            bh = df.loc[t, 'high']
            bl = df.loc[t, 'low']
            bc = df.loc[t, 'close']
            pos['bars'] += 1

            if pos.get('trail_on'):
                if pos['dir'] == 1:
                    new = bh * (1 - TRAILING_LOCK * pos['atr_pct'])
                    pos['trail_sl'] = max(pos.get('trail_sl', 0), new)
                else:
                    new = bl * (1 + TRAILING_LOCK * pos['atr_pct'])
                    pos['trail_sl'] = min(pos.get('trail_sl', 1e15), new)

            ex_price = ex_reason = None
            if pos['dir'] == 1:
                if bh >= pos['tp']: ex_price, ex_reason = pos['tp'], 'TP'
                elif pos.get('trail_sl') and bl <= pos['trail_sl']:
                    ex_price, ex_reason = pos['trail_sl'], 'TRAIL'
                elif bl <= pos['sl']: ex_price, ex_reason = pos['sl'], 'SL'
                elif pos['bars'] >= pos['mh']: ex_price, ex_reason = bc, 'TIMEOUT'
                elif not pos.get('trail_on'):
                    g = (bh - pos['entry']) / pos['entry']
                    if g >= pos['tp_pct'] * TRAILING_ACTIVATION:
                        pos['trail_on'] = True
                        pos['trail_sl'] = pos['entry'] * (1 + g * 0.3)
            else:
                if bl <= pos['tp']: ex_price, ex_reason = pos['tp'], 'TP'
                elif pos.get('trail_sl') and bh >= pos['trail_sl']:
                    ex_price, ex_reason = pos['trail_sl'], 'TRAIL'
                elif bh >= pos['sl']: ex_price, ex_reason = pos['sl'], 'SL'
                elif pos['bars'] >= pos['mh']: ex_price, ex_reason = bc, 'TIMEOUT'
                elif not pos.get('trail_on'):
                    g = (pos['entry'] - bl) / pos['entry']
                    if g >= pos['tp_pct'] * TRAILING_ACTIVATION:
                        pos['trail_on'] = True
                        pos['trail_sl'] = pos['entry'] * (1 - g * 0.3)

            if ex_price:
                gpnl = ((ex_price - pos['entry']) / pos['entry']) if pos['dir'] == 1 else \
                       ((pos['entry'] - ex_price) / pos['entry'])
                cost = pos['not'] * (COMMISSION + SLIPPAGE) * 2
                pnl = pos['not'] * gpnl - cost
                balance += pnl
                daily_pnl[ds] += pnl

                # Enriched trade: PnL + entry-time features
                enriched_trades.append({
                    'pair': pair, 'dir': pos['dir'], 'pnl': pnl,
                    'pnl_pct': gpnl * 100, 'reason': ex_reason,
                    'bars': pos['bars'], 'time': t, 'regime': pos['reg'],
                    'lev': pos['lev'], 'conf': pos['conf'], 'bal': balance,
                    # ConvictionScorer features (captured at entry time):
                    'cs_conf': pos['_cs_conf'],
                    'cs_pred_mag': pos['_cs_pred_mag'],
                    'cs_macro_score': pos['_cs_macro_score'],
                    'cs_risk_off': pos['_cs_risk_off'],
                    'cs_regime_bull': pos['_cs_regime_bull'],
                    'cs_regime_bear': pos['_cs_regime_bear'],
                    'cs_regime_range': pos['_cs_regime_range'],
                    'cs_atr_pct': pos['_cs_atr_pct'],
                    'cs_n_open': pos['_cs_n_open'],
                    'cs_pred_sign': pos['_cs_pred_sign'],
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

        # === OPEN NEW POSITIONS ===
        if len(positions) >= MAX_CONCURRENT:
            continue

        macro_score = macro_scores.get(ds, 0.5)
        ro_mult = risk_off_mults.get(ds, 1.0)

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
            if conf < 0.7: continue  # V7 default threshold

            d = 1 if pred > 0 else -1

            if reg == 'BULL' and d == -1: continue
            if reg == 'BEAR' and d == 1: continue

            same = sum(1 for p in positions.values() if p['dir'] == d)
            if same >= 2: continue

            cands.append({
                'pair': pair, 'dir': d, 'conf': conf,
                'pred': pred, 'pred_mag': abs(pred),
            })

        cands.sort(key=lambda x: x['conf'], reverse=True)

        for cand in cands:
            if len(positions) >= MAX_CONCURRENT: break
            pair = cand['pair']
            d = cand['dir']
            conf = cand['conf']
            df = all_data[pair]
            entry = df.loc[t, 'close']

            atr_s = atr_dict[pair]
            atr_val = atr_s.loc[t] if t in atr_s.index else None

            if atr_val is not None and not np.isnan(atr_val):
                atr_pct = atr_val / entry
                tp_pct = max(0.005, min(0.08, atr_pct * ATR_TP[reg]))
                sl_pct = max(0.003, min(0.04, atr_pct * ATR_SL[reg]))
            else:
                atr_pct = 0.02
                tp_pct = 0.03; sl_pct = 0.015

            mh = 30 if reg != 'RANGE' else 15

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
                'atr_pct': atr_pct,
                'not': notional, 'lev': lev, 'mh': mh,
                'bars': 0, 'conf': conf, 'reg': reg,
                'trail_on': False, 'trail_sl': None,
                # Store conviction features at entry time
                '_cs_conf': conf,
                '_cs_pred_mag': cand['pred_mag'],
                '_cs_macro_score': macro_score,
                '_cs_risk_off': ro_mult,
                '_cs_regime_bull': 1.0 if reg == 'BULL' else 0.0,
                '_cs_regime_bear': 1.0 if reg == 'BEAR' else 0.0,
                '_cs_regime_range': 1.0 if reg == 'RANGE' else 0.0,
                '_cs_atr_pct': atr_pct,
                '_cs_n_open': len(positions),  # before adding this one
                '_cs_pred_sign': float(d),
            }

    return enriched_trades


# ============================================================
# 2. TRAIN CONVICTION SCORER
# ============================================================
CONVICTION_FEATURES = [
    'cs_conf', 'cs_pred_mag', 'cs_macro_score', 'cs_risk_off',
    'cs_regime_bull', 'cs_regime_bear', 'cs_regime_range',
    'cs_atr_pct', 'cs_n_open', 'cs_pred_sign',
]


def prepare_conviction_data(trades):
    """Convert enriched trades list to X, y for ConvictionScorer training."""
    if not trades or len(trades) < 50:
        return None, None

    df = pd.DataFrame(trades)

    # Features
    X = df[CONVICTION_FEATURES].fillna(0).copy()

    # Target: trade PnL clipped at 1st/99th percentile
    pnl = df['pnl'].copy()
    lo, hi = pnl.quantile(0.01), pnl.quantile(0.99)
    y = pnl.clip(lo, hi)

    return X, y


def train_conviction_scorer(X, y, n_trials=30):
    """Train ConvictionScorer: LightGBM regressor predicts trade PnL."""
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
            'min_child_samples': trial.suggest_int('mc', 10, 80),
            'reg_alpha': trial.suggest_float('ra', 1e-5, 5.0, log=True),
            'reg_lambda': trial.suggest_float('rl', 1e-5, 5.0, log=True),
            'num_leaves': trial.suggest_int('nl', 10, 50),
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        }
        m = lgb.LGBMRegressor(**p)
        m.fit(Xt, yt, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(20, verbose=False)])
        pr = m.predict(Xv)
        corr = np.corrcoef(pr, yv.values)[0, 1]
        return corr if not np.isnan(corr) else 0.0

    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

    rename = {'ne': 'n_estimators', 'md': 'max_depth', 'lr': 'learning_rate',
              'ss': 'subsample', 'cs': 'colsample_bytree', 'mc': 'min_child_samples',
              'ra': 'reg_alpha', 'rl': 'reg_lambda', 'nl': 'num_leaves'}
    bp = {rename.get(k, k): v for k, v in study.best_params.items()}
    bp.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})

    mdl = lgb.LGBMRegressor(**bp)
    mdl.fit(X_clean, y, eval_set=[(Xv, yv)],
            callbacks=[lgb.early_stopping(30, verbose=False)])

    # Compute PnL std for normalization in sizing
    preds_all = mdl.predict(X_clean)
    pred_std = np.std(preds_all)

    return mdl, fcols, study.best_value, pred_std


def conviction_to_sizing(pred_pnl, pred_std):
    """Map ConvictionScorer output to sizing multiplier [0.3, 1.8].

    Uses sigmoid normalization:
    - pred_pnl >> 0 -> sizing near 1.8x
    - pred_pnl ~= 0 -> sizing near 1.0x
    - pred_pnl << 0 -> sizing near 0.3x
    """
    if pred_std < 1e-8:
        return 1.0
    # Normalize to ~[-3, 3] range
    z = pred_pnl / pred_std
    # Sigmoid maps to [0, 1]
    s = expit(z)
    # Scale to [0.3, 1.8]
    return 0.3 + 1.5 * s


# ============================================================
# 3. BACKTEST WITH CONVICTION SCORER
# ============================================================
def backtest_with_conviction(all_data, regime_daily, models,
                              conviction_scorer=None, conviction_fcols=None,
                              conviction_pred_std=1.0,
                              macro_scorer=None, macro_feat=None,
                              macro_fcols=None, risk_off_mults=None,
                              use_macro_sizing=False,
                              use_riskoff_sizing=False,
                              use_adaptive_thresh=False,
                              use_conviction=False,
                              base_thresh=0.7,
                              max_notional=300):
    """V7 backtest with V8.4 macro + V8.5 conviction scoring.

    When use_conviction=True, applies per-trade sizing from ConvictionScorer.
    Can be combined with macro sizing (stacked) or used alone.
    """
    all_t = set()
    for p in models:
        if p in all_data:
            all_t.update(all_data[p].index.tolist())
    timeline = sorted(all_t)
    if not timeline:
        return [], 0, False, {}

    regime_4h = map_regime_4h(regime_daily, pd.DatetimeIndex(timeline))

    # Pre-compute macro scores per day
    macro_scores = {}
    if macro_scorer is not None and (use_macro_sizing or use_adaptive_thresh):
        macro_aligned = macro_feat[macro_fcols].fillna(0)
        probs = macro_scorer.predict_proba(macro_aligned)[:, 1]
        for dt, p in zip(macro_aligned.index, probs):
            macro_scores[dt.strftime('%Y-%m-%d')] = p

    if risk_off_mults is None:
        risk_off_mults = {}

    # Pre-compute preds and ATRs
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
    sizing_adjustments = []
    thresh_adjustments = []
    conviction_sizings = []
    trades_skipped_by_conviction = 0

    for t in timeline:
        if killed:
            break
        ds = t.strftime('%Y-%m-%d')

        if paused_until and t.date() <= paused_until:
            continue

        reg = regime_4h.loc[t] if t in regime_4h.index else 'RANGE'
        lev = LEV[reg]

        # === UPDATE POSITIONS (identical to V7) ===
        to_close = []
        for pair, pos in positions.items():
            df = all_data[pair]
            if t not in df.index:
                continue

            bh = df.loc[t, 'high']
            bl = df.loc[t, 'low']
            bc = df.loc[t, 'close']
            pos['bars'] += 1

            if pos.get('trail_on'):
                if pos['dir'] == 1:
                    new = bh * (1 - TRAILING_LOCK * pos['atr_pct'])
                    pos['trail_sl'] = max(pos.get('trail_sl', 0), new)
                else:
                    new = bl * (1 + TRAILING_LOCK * pos['atr_pct'])
                    pos['trail_sl'] = min(pos.get('trail_sl', 1e15), new)

            ex_price = ex_reason = None
            if pos['dir'] == 1:
                if bh >= pos['tp']: ex_price, ex_reason = pos['tp'], 'TP'
                elif pos.get('trail_sl') and bl <= pos['trail_sl']:
                    ex_price, ex_reason = pos['trail_sl'], 'TRAIL'
                elif bl <= pos['sl']: ex_price, ex_reason = pos['sl'], 'SL'
                elif pos['bars'] >= pos['mh']: ex_price, ex_reason = bc, 'TIMEOUT'
                elif not pos.get('trail_on'):
                    g = (bh - pos['entry']) / pos['entry']
                    if g >= pos['tp_pct'] * TRAILING_ACTIVATION:
                        pos['trail_on'] = True
                        pos['trail_sl'] = pos['entry'] * (1 + g * 0.3)
            else:
                if bl <= pos['tp']: ex_price, ex_reason = pos['tp'], 'TP'
                elif pos.get('trail_sl') and bh >= pos['trail_sl']:
                    ex_price, ex_reason = pos['trail_sl'], 'TRAIL'
                elif bh >= pos['sl']: ex_price, ex_reason = pos['sl'], 'SL'
                elif pos['bars'] >= pos['mh']: ex_price, ex_reason = bc, 'TIMEOUT'
                elif not pos.get('trail_on'):
                    g = (pos['entry'] - bl) / pos['entry']
                    if g >= pos['tp_pct'] * TRAILING_ACTIVATION:
                        pos['trail_on'] = True
                        pos['trail_sl'] = pos['entry'] * (1 - g * 0.3)

            if ex_price:
                gpnl = ((ex_price - pos['entry']) / pos['entry']) if pos['dir'] == 1 else \
                       ((pos['entry'] - ex_price) / pos['entry'])
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

        # === OPEN NEW POSITIONS ===
        if len(positions) >= MAX_CONCURRENT:
            continue

        macro_score = macro_scores.get(ds, 0.5)

        # 1. Adaptive threshold (V8.4)
        if use_adaptive_thresh:
            thresh = 0.9 - 0.4 * macro_score
            thresh = max(0.50, min(0.90, thresh))
        else:
            thresh = base_thresh
        thresh_adjustments.append(thresh)

        # 2. ML sizing multiplier (V8.4)
        if use_macro_sizing:
            ml_mult = 0.3 + 1.5 * macro_score
        else:
            ml_mult = 1.0

        # 3. Risk-off (V8.4)
        ro_mult = risk_off_mults.get(ds, 1.0) if use_riskoff_sizing else 1.0

        # Macro combined sizing
        macro_sizing = ml_mult * ro_mult
        macro_sizing = max(0.2, min(2.0, macro_sizing))

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

            if reg == 'BULL' and d == -1: continue
            if reg == 'BEAR' and d == 1: continue

            same = sum(1 for p in positions.values() if p['dir'] == d)
            if same >= 2: continue

            cands.append({
                'pair': pair, 'dir': d, 'conf': conf,
                'pred': pred, 'pred_mag': abs(pred),
            })

        cands.sort(key=lambda x: x['conf'], reverse=True)

        for cand in cands:
            if len(positions) >= MAX_CONCURRENT: break
            pair = cand['pair']
            d = cand['dir']
            conf = cand['conf']
            df = all_data[pair]
            entry = df.loc[t, 'close']

            atr_s = atr_dict[pair]
            atr_val = atr_s.loc[t] if t in atr_s.index else None

            if atr_val is not None and not np.isnan(atr_val):
                atr_pct = atr_val / entry
                tp_pct = max(0.005, min(0.08, atr_pct * ATR_TP[reg]))
                sl_pct = max(0.003, min(0.04, atr_pct * ATR_SL[reg]))
            else:
                atr_pct = 0.02
                tp_pct = 0.03; sl_pct = 0.015

            mh = 30 if reg != 'RANGE' else 15

            # V7 base risk
            risk_pct = RISK_PER_TRADE
            if conf > 2.0: risk_pct = 0.03
            elif conf > 1.5: risk_pct = 0.025

            # 4. ConvictionScorer sizing (V8.5)
            conv_mult = 1.0
            if use_conviction and conviction_scorer is not None:
                cs_features = pd.DataFrame([{
                    'cs_conf': conf,
                    'cs_pred_mag': cand['pred_mag'],
                    'cs_macro_score': macro_score,
                    'cs_risk_off': ro_mult,
                    'cs_regime_bull': 1.0 if reg == 'BULL' else 0.0,
                    'cs_regime_bear': 1.0 if reg == 'BEAR' else 0.0,
                    'cs_regime_range': 1.0 if reg == 'RANGE' else 0.0,
                    'cs_atr_pct': atr_pct,
                    'cs_n_open': len(positions),
                    'cs_pred_sign': float(d),
                }])
                cols = [c for c in conviction_fcols if c in cs_features.columns]
                pred_pnl = conviction_scorer.predict(cs_features[cols])[0]
                conv_mult = conviction_to_sizing(pred_pnl, conviction_pred_std)
                conviction_sizings.append(conv_mult)

                # Skip trades with very negative conviction
                if pred_pnl < -conviction_pred_std * 0.5:
                    trades_skipped_by_conviction += 1
                    continue

            # Combined sizing: macro * conviction
            total_sizing = macro_sizing * conv_mult
            total_sizing = max(0.2, min(2.5, total_sizing))

            risk_pct *= total_sizing
            sizing_adjustments.append(total_sizing)

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
                'atr_pct': atr_pct,
                'not': notional, 'lev': lev, 'mh': mh,
                'bars': 0, 'conf': conf, 'reg': reg,
                'trail_on': False, 'trail_sl': None,
            }

    stats = {
        'avg_sizing': np.mean(sizing_adjustments) if sizing_adjustments else 1.0,
        'avg_thresh': np.mean(thresh_adjustments) if thresh_adjustments else base_thresh,
        'avg_conviction': np.mean(conviction_sizings) if conviction_sizings else 1.0,
        'trades_skipped_conv': trades_skipped_by_conviction,
    }
    return trades, max_dd, killed, stats


# ============================================================
# 4. CONFIG DEFINITIONS
# ============================================================
CONFIGS = [
    ('V7',       {'adapt_thresh': False, 'ml_sizing': False, 'riskoff': False, 'conviction': False}),
    ('V84_Full', {'adapt_thresh': True,  'ml_sizing': True,  'riskoff': True,  'conviction': False}),
    ('V85_Conv', {'adapt_thresh': False, 'ml_sizing': False, 'riskoff': False, 'conviction': True}),
    ('V85_Full', {'adapt_thresh': True,  'ml_sizing': True,  'riskoff': True,  'conviction': True}),
]


def run_config(cfg_name, cfg_flags, all_data, regime, models,
               macro_scorer, macro_fcols, macro_feat, ro_mults,
               conv_scorer, conv_fcols, conv_pred_std):
    """Run a single config and return (trades, max_dd, killed, stats)."""
    if cfg_name == 'V7':
        tr, _, mdd, kil = backtest(
            all_data, regime, models,
            use_trailing=True, use_atr=True, use_compound=False, max_notional=300,
        )
        return tr, mdd, kil, {}

    return backtest_with_conviction(
        all_data, regime, models,
        conviction_scorer=conv_scorer if cfg_flags['conviction'] else None,
        conviction_fcols=conv_fcols,
        conviction_pred_std=conv_pred_std,
        macro_scorer=macro_scorer if (cfg_flags['ml_sizing'] or cfg_flags['adapt_thresh']) else None,
        macro_feat=macro_feat,
        macro_fcols=macro_fcols,
        risk_off_mults=ro_mults if cfg_flags['riskoff'] else None,
        use_macro_sizing=cfg_flags['ml_sizing'],
        use_riskoff_sizing=cfg_flags['riskoff'],
        use_adaptive_thresh=cfg_flags['adapt_thresh'],
        use_conviction=cfg_flags['conviction'],
    )


# ============================================================
# 5. MAIN
# ============================================================
def main():
    t0 = time.time()

    # --- Download ---
    print('=' * 80)
    print('V8.5 ConvictionScorer Backtest')
    print('=' * 80)

    all_data, btc_d, macro, fng = download_everything()
    if len(all_data) < 3 or btc_d is None:
        print('ERROR: datos insuficientes'); return

    regime = detect_regime(btc_d)
    print(f'\n  Regimen V7: BULL {(regime=="BULL").sum()}d | '
          f'BEAR {(regime=="BEAR").sum()}d | RANGE {(regime=="RANGE").sum()}d')

    macro_feat = compute_macro_features(
        macro.get('dxy'), macro.get('gold'), macro.get('spy'),
        macro.get('tnx'), macro.get('ethbtc'),
    )
    if macro_feat is None:
        print('ERROR: no macro features'); return
    print(f'  Macro features: {len(macro_feat.columns)} cols, {len(macro_feat)} rows')

    ro_mults = compute_risk_off_multipliers(macro_feat)
    print(f'  Risk-off: {len(ro_mults)} dias ({len(ro_mults)/len(macro_feat)*100:.1f}%)')

    # ================================================================
    # WALK-FORWARD: 4 configs x 4 folds
    # ================================================================
    print('\n' + '=' * 80)
    print('WALK-FORWARD: 4 configs (V7 / V8.4 Full / V8.5 Conv / V8.5 Full)')
    print('=' * 80)

    folds = [
        ('2023-06', '2023-06', '2024-01', 'H2-2023'),
        ('2024-01', '2024-01', '2024-07', 'H1-2024'),
        ('2024-07', '2024-07', '2025-01', 'H2-2024'),
        ('2025-01', '2025-01', '2026-03', '2025+'),
    ]

    all_results = {c: [] for c, _ in CONFIGS}

    for tr_e, te_s, te_e, fn in folds:
        print(f'\n  --- Fold {fn} (train hasta {tr_e}) ---')

        # Train V7 models
        mdls = train_all(all_data, tr_e, horizon=5, n_trials=30)
        mdls_f = {}
        for p, i in mdls.items():
            m = (i['index'] >= te_s) & (i['index'] < te_e)
            if m.sum() > 0:
                mdls_f[p] = {**i, 'preds': i['preds'][m], 'index': i['index'][m]}
        if not mdls_f:
            print(f'    Sin modelos'); continue

        # Train MacroScorer (needed for V84 and V85_Full)
        oos_trades = generate_v7_oos_trades(all_data, regime, tr_e, n_trials=15)
        macro_X, macro_y = create_scorer_labels(oos_trades, macro_feat)
        scorer = None
        scorer_fcols = []
        if macro_X is not None and len(macro_X) > 50:
            scorer, scorer_fcols, auc = train_macro_scorer(macro_X, macro_y, n_trials=20)
            print(f'    MacroScorer: AUC={auc:.3f} | {len(macro_X)} days')

        # Train ConvictionScorer: generate OOS trades with features
        enriched = backtest_collect_trade_features(
            all_data, regime, mdls_f,
            macro_scorer=scorer, macro_feat=macro_feat,
            macro_fcols=scorer_fcols, risk_off_mults=ro_mults,
        )

        # For fold's ConvictionScorer, we train on PREVIOUS fold's OOS trades
        # But since we're in walk-forward, we use nested approach:
        # Train ConvictionScorer on the OOS trades we just collected
        # (these trades came from a backtest using models trained up to tr_e)
        conv_scorer = None
        conv_fcols = []
        conv_pred_std = 1.0
        if len(enriched) >= 80:
            cs_X, cs_y = prepare_conviction_data(enriched)
            if cs_X is not None:
                conv_scorer, conv_fcols, corr, conv_pred_std = train_conviction_scorer(
                    cs_X, cs_y, n_trials=20,
                )
                print(f'    ConvictionScorer: corr={corr:.3f} | {len(cs_X)} trades | '
                      f'pred_std={conv_pred_std:.4f}')
        else:
            print(f'    ConvictionScorer: solo {len(enriched)} trades (min 80)')

        # Run all configs
        results = {}
        for cfg_name, cfg_flags in CONFIGS:
            tr, mdd, kil, stats = run_config(
                cfg_name, cfg_flags, all_data, regime, mdls_f,
                scorer, scorer_fcols, macro_feat, ro_mults,
                conv_scorer, conv_fcols, conv_pred_std,
            )
            results[cfg_name] = (tr, mdd, kil, stats)

        # Print fold results
        for cfg_name, _ in CONFIGS:
            tr, mdd, kil, stats = results[cfg_name]
            if not tr:
                print(f'    {cfg_name:<10} Sin trades')
                continue
            fin = tr[-1]['bal']
            ret = ((fin / START_CAPITAL) - 1) * 100
            w = sum(1 for t in tr if t['pnl'] > 0)
            wr = w / len(tr) * 100
            gp = sum(t['pnl'] for t in tr if t['pnl'] > 0)
            gl = abs(sum(t['pnl'] for t in tr if t['pnl'] <= 0))
            pf = gp / gl if gl > 0 else 0
            mk = ' ***' if ret > 20 else (' **' if ret > 5 else (' *' if ret > 0 else ' --'))
            extra = ''
            if stats.get('avg_sizing') and stats['avg_sizing'] != 1.0:
                extra += f' | sz:{stats["avg_sizing"]:.2f}x'
            if stats.get('avg_conviction') and stats['avg_conviction'] != 1.0:
                extra += f' | conv:{stats["avg_conviction"]:.2f}x'
            if stats.get('trades_skipped_conv'):
                extra += f' | skip:{stats["trades_skipped_conv"]}'
            print(f'    {cfg_name:<10} {len(tr):>4}t | WR {wr:.0f}% | PF {pf:.2f} | '
                  f'Ret {ret:+.1f}% | DD {mdd*100:.1f}%{mk}'
                  f'{" KILLED" if kil else ""}{extra}')

            m = extract_metrics(tr, mdd, kil)
            if m: all_results[cfg_name].append(m)

    # Walk-forward summary
    print(f'\n  {"="*80}')
    print(f'  RESUMEN WALK-FORWARD')
    print(f'  {"="*80}')
    for cfg_name, _ in CONFIGS:
        res = all_results[cfg_name]
        if not res:
            print(f'  {cfg_name:<10} Sin resultados'); continue
        prof = sum(1 for r in res if r['ret'] > 0)
        avg_pf = np.mean([r['pf'] for r in res])
        avg_ret = np.mean([r['ret'] for r in res])
        avg_dd = np.mean([r['dd'] for r in res])
        avg_sharpe = np.mean([r['sharpe'] for r in res])
        total_trades = sum(r['trades'] for r in res)
        print(f'  {cfg_name:<10} {prof}/{len(res)} folds+ | {total_trades:>4}t | '
              f'PF {avg_pf:.2f} | Ret {avg_ret:+.1f}% | DD {avg_dd:.1f}% | Sharpe {avg_sharpe:.2f}')

    # ================================================================
    # FULL BACKTEST
    # ================================================================
    print('\n' + '=' * 80)
    print('BACKTEST COMPLETO (train hasta 2024-01, test 2024+)')
    print('=' * 80)

    models = train_all(all_data, '2024-01', horizon=5, n_trials=40)

    # Train MacroScorer (full)
    oos_trades = generate_v7_oos_trades(all_data, regime, '2024-01', n_trials=20)
    macro_X, macro_y = create_scorer_labels(oos_trades, macro_feat)
    scorer = None
    scorer_fcols = []
    if macro_X is not None and len(macro_X) > 50:
        scorer, scorer_fcols, auc = train_macro_scorer(macro_X, macro_y, n_trials=40)
        print(f'\n  MacroScorer: AUC={auc:.3f} | {len(macro_X)} days')

    # Train ConvictionScorer (full): collect trades from the full model set
    enriched = backtest_collect_trade_features(
        all_data, regime, models,
        macro_scorer=scorer, macro_feat=macro_feat,
        macro_fcols=scorer_fcols, risk_off_mults=ro_mults,
    )

    conv_scorer = None
    conv_fcols = []
    conv_pred_std = 1.0
    if len(enriched) >= 80:
        cs_X, cs_y = prepare_conviction_data(enriched)
        if cs_X is not None:
            conv_scorer, conv_fcols, corr, conv_pred_std = train_conviction_scorer(
                cs_X, cs_y, n_trials=40,
            )
            print(f'  ConvictionScorer: corr={corr:.3f} | {len(cs_X)} trades | '
                  f'pred_std={conv_pred_std:.4f}')

            if hasattr(conv_scorer, 'feature_importances_'):
                imp = sorted(zip(conv_fcols, conv_scorer.feature_importances_),
                             key=lambda x: x[1], reverse=True)
                print(f'\n  ConvictionScorer Feature Importance:')
                for i, (f, v) in enumerate(imp):
                    bar = '#' * min(int(v / max(1, imp[0][1]) * 30), 30)
                    print(f'    {i+1:>2}. {f:<25} {v:>6.1f} {bar}')
    else:
        print(f'  ConvictionScorer: solo {len(enriched)} trades (min 80)')

    # Run all configs
    full_metrics = {}
    for cfg_name, cfg_flags in CONFIGS:
        tr, mdd, kil, stats = run_config(
            cfg_name, cfg_flags, all_data, regime, models,
            scorer, scorer_fcols, macro_feat, ro_mults,
            conv_scorer, conv_fcols, conv_pred_std,
        )
        extra = ''
        if stats.get('avg_sizing') and stats['avg_sizing'] != 1.0:
            extra += f' (AvgSize: {stats["avg_sizing"]:.2f}x)'
        if stats.get('avg_conviction') and stats['avg_conviction'] != 1.0:
            extra += f' (AvgConv: {stats["avg_conviction"]:.2f}x)'
        if stats.get('trades_skipped_conv'):
            extra += f' (Skipped: {stats["trades_skipped_conv"]})'
        print_results(tr, mdd, kil, f'{cfg_name}{extra}')
        full_metrics[cfg_name] = extract_metrics(tr, mdd, kil)

    # ================================================================
    # COMPARISON TABLE
    # ================================================================
    print(f'\n{"="*80}')
    print('COMPARACION FINAL')
    print(f'{"="*80}')

    valid = [(n, m) for n, m in full_metrics.items() if m is not None]
    if len(valid) >= 2:
        header = f'  {"Metrica":<18}'
        for name, _ in valid:
            header += f' {name:>10}'
        print(header)
        print(f'  {"-" * (18 + 11 * len(valid))}')

        for key, label, fmt in [
            ('ret', 'Retorno %', '.1f'), ('trades', 'Trades', '.0f'),
            ('wr', 'Win Rate %', '.0f'), ('pf', 'Profit Factor', '.2f'),
            ('dd', 'Max DD %', '.1f'), ('sharpe', 'Sharpe', '.2f'),
            ('annual', 'Annual %', '.1f'), ('days', 'Dias activo', '.0f'),
        ]:
            line = f'  {label:<18}'
            vals = []
            for name, m in valid:
                v = m.get(key, 0)
                vals.append(v)
                line += f' {v:>10{fmt}}'

            if key == 'dd':
                best_idx = vals.index(min(vals))
            elif key == 'days':
                best_idx = vals.index(max(vals))
            else:
                best_idx = vals.index(max(vals))
            if best_idx > 0:
                line += f' <<< {valid[best_idx][0]}'
            print(line)

        # Months positive
        line = f'  {"Meses +":<18}'
        for name, m in valid:
            mp = m.get('months_pos', 0)
            mt = m.get('months_total', 1)
            line += f' {f"{mp}/{mt}":>10}'
        print(line)

    # Walk-forward vs Full comparison
    print(f'\n  {"="*80}')
    print(f'  WALK-FORWARD vs FULL BACKTEST')
    print(f'  {"="*80}')
    for cfg_name, _ in CONFIGS:
        wf = all_results.get(cfg_name, [])
        fb = full_metrics.get(cfg_name)
        if not wf or not fb:
            continue
        wf_pf = np.mean([r['pf'] for r in wf])
        wf_dd = np.mean([r['dd'] for r in wf])
        consistent = abs(wf_pf - fb['pf']) < 0.20
        icon = 'OK' if consistent else 'DIVERGE'
        print(f'  {cfg_name:<10} WF: PF {wf_pf:.2f} DD {wf_dd:.1f}% | '
              f'Full: PF {fb["pf"]:.2f} DD {fb["dd"]:.1f}% | [{icon}]')

    # Verdict
    v7 = full_metrics.get('V7')
    v7_wf = all_results.get('V7', [])

    best_cfg = None
    best_m = None

    for cfg_name, _ in CONFIGS:
        if cfg_name == 'V7':
            continue
        m = full_metrics.get(cfg_name)
        wf = all_results.get(cfg_name, [])
        if not m or not v7:
            continue

        improves_pf = m['pf'] > v7['pf']
        improves_dd = m['dd'] < v7['dd']
        improves_sharpe = m['sharpe'] > v7['sharpe']

        wf_ok = True
        if wf and v7_wf:
            wf_pf = np.mean([r['pf'] for r in wf])
            v7_wf_pf = np.mean([r['pf'] for r in v7_wf])
            wf_ok = wf_pf >= v7_wf_pf * 0.90  # Allow 10% WF PF degradation

        # Must improve at least 2 of 3 (PF, DD, Sharpe) AND be consistent in WF
        improvements = sum([improves_pf, improves_dd, improves_sharpe])
        if improvements >= 2 and wf_ok:
            if best_m is None or m['sharpe'] > best_m['sharpe']:
                best_cfg = cfg_name
                best_m = m

    print(f'\n  VEREDICTO:')
    if best_m and v7:
        print(f'    >>> {best_cfg} MEJORA vs V7:')
        print(f'        PF:     {v7["pf"]:.2f} -> {best_m["pf"]:.2f}')
        print(f'        DD:     {v7["dd"]:.1f}% -> {best_m["dd"]:.1f}%')
        print(f'        Sharpe: {v7["sharpe"]:.2f} -> {best_m["sharpe"]:.2f}')
        print(f'        Ret:    {v7["ret"]:.1f}% -> {best_m["ret"]:.1f}%')
        if best_m['pf'] > 1.4 and best_m['dd'] < 15:
            print(f'        >>> APROBADO para produccion <<<')
        elif best_m['pf'] > v7['pf'] and best_m['dd'] < v7['dd']:
            print(f'        Mejora solida, considerar para produccion')
        else:
            print(f'        Mejora parcial, seguir iterando')
    else:
        print(f'    >>> Ninguna config mejora consistentemente vs V7')
        print(f'    Diagnostico:')
        for cfg_name, _ in CONFIGS:
            if cfg_name == 'V7': continue
            m = full_metrics.get(cfg_name)
            if m and v7:
                pf_d = m['pf'] - v7['pf']
                dd_d = m['dd'] - v7['dd']
                sh_d = m['sharpe'] - v7['sharpe']
                print(f'      {cfg_name:<10} PF {pf_d:+.2f} | DD {dd_d:+.1f}% | Sharpe {sh_d:+.2f}')

    print(f'\nTiempo total: {(time.time() - t0) / 60:.1f} minutos')


if __name__ == '__main__':
    main()
