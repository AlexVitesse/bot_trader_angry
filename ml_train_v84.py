"""
ML V8.4: Macro-Aware Trading - Adaptive Threshold + Soft Risk-Off
=================================================================
V8.3 problemas:
  1. Risk-off override a BEAR bloquea TODOS los longs -> fatal en bull runs
  2. ML sizing rango muy estrecho (avg 1.04x, casi igual que V7)
  3. Risk-off se activa demasiado (21.6% de dias)

V8.4 mejoras:
  1. ADAPTIVE THRESHOLD: macro bueno -> acepta mas senales V7 (thresh baja)
                         macro malo -> requiere senales mas fuertes (thresh sube)
     Formula: thresh = 0.9 - 0.4 * macro_score
       score=0   -> thresh=0.90 (muy selectivo)
       score=0.5 -> thresh=0.70 (V7 default)
       score=1.0 -> thresh=0.50 (acepta mas senales)

  2. SOFT RISK-OFF: reduce sizing a 0.3-0.5x en vez de forzar BEAR
     Ya NO bloquea longs -> sigue operando en bull runs, solo mas chico

  3. UMBRALES ESTRICTOS: risk-off solo en condiciones extremas
     V8.3: 21.6% de dias -> V8.4 target: ~5-10% de dias

  4. SIZING AMPLIO: [0.3x, 1.8x] en vez de [0.5x, 1.5x]

5 configs comparadas:
  A) V7 (baseline)
  B) AdaptTh (umbral de confianza dinamico segun macro)
  C) SoftRO (risk-off suave: reduce sizing, no bloquea longs)
  D) ThreshRO (adaptive threshold + soft risk-off, sin ML sizing)
  E) Full (todo combinado)

Ejecutar: python -u ml_train_v84.py
"""

import pandas as pd
import numpy as np
import pandas_ta as pta
import lightgbm as lgb
import optuna
from pathlib import Path
from collections import defaultdict
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
from macro_data import download_all_macro, compute_macro_features
from alt_data_fetcher import download_fear_greed


# ============================================================
# 1. DOWNLOAD
# ============================================================
def download_everything():
    all_data, btc_d = download_all()
    macro = download_all_macro()
    fng = download_fear_greed()
    if fng is not None:
        print(f'  Fear&Greed  {len(fng):>6,} dias')
    return all_data, btc_d, macro, fng


# ============================================================
# 2. MACRO SCORER (ML - continuous probability)
# ============================================================
def generate_v7_oos_trades(all_data, regime, train_end, n_trials=15):
    """Nested walk-forward: train V7 on early half, backtest on late half.

    This generates OOS trades for MacroScorer training labels.
    V7's train_all() only returns test-period predictions, so we need
    this nested approach to get training-period labels.
    """
    all_dates = set()
    for pair, df in all_data.items():
        all_dates.update(df.loc[:train_end].index.tolist())
    all_dates = sorted(all_dates)
    if len(all_dates) < 2000:
        return []

    mid = all_dates[len(all_dates) // 2].strftime('%Y-%m')
    print(f'    MacroScorer: nested WF [start..{mid}] train, [{mid}..{train_end}] eval')

    early_mdls = train_all(all_data, mid, horizon=5, n_trials=n_trials)

    late_mdls = {}
    for p, info in early_mdls.items():
        mask = info['index'] < train_end
        if mask.sum() > 0:
            late_mdls[p] = {**info, 'preds': info['preds'][mask], 'index': info['index'][mask]}

    if not late_mdls:
        return []

    trades, _, _, _ = backtest(
        all_data, regime, late_mdls,
        use_trailing=True, use_atr=True, use_compound=False, max_notional=300,
    )
    print(f'    MacroScorer: {len(trades)} OOS trades for training')
    return trades


def create_scorer_labels(trades, macro_feat):
    """Create daily labels: 1=V7 profitable, 0=V7 lost money."""
    if not trades:
        return None, None

    daily_pnl = defaultdict(float)
    for t in trades:
        daily_pnl[t['time'].strftime('%Y-%m-%d')] += t['pnl']

    labels = pd.Series(daily_pnl)
    labels.index = pd.to_datetime(labels.index)
    y = (labels > 0).astype(int)

    macro_daily = macro_feat.reindex(y.index, method='ffill')
    valid = macro_daily.dropna().index.intersection(y.index)

    if len(valid) < 30:
        return None, None
    return macro_daily.loc[valid], y.loc[valid]


def train_macro_scorer(X, y, n_trials=30):
    """Train MacroScorer: LightGBM classifier outputs probability [0,1]."""
    fcols = list(X.columns)
    X_clean = X[fcols].fillna(0)

    sp = int(len(X_clean) * 0.8)
    Xt, Xv = X_clean.iloc[:sp], X_clean.iloc[sp:]
    yt, yv = y.iloc[:sp], y.iloc[sp:]

    from sklearn.metrics import roc_auc_score

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
        try:
            return roc_auc_score(yv, probs)
        except:
            return 0.5

    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

    rename = {'ne': 'n_estimators', 'md': 'max_depth', 'lr': 'learning_rate',
              'ss': 'subsample', 'cs': 'colsample_bytree', 'mc': 'min_child_samples',
              'ra': 'reg_alpha', 'rl': 'reg_lambda', 'nl': 'num_leaves'}
    bp = {rename.get(k, k): v for k, v in study.best_params.items()}
    bp.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1, 'objective': 'binary'})

    mdl = lgb.LGBMClassifier(**bp)
    mdl.fit(X_clean, y, eval_set=[(Xv, yv)],
            callbacks=[lgb.early_stopping(30, verbose=False)])

    return mdl, fcols, study.best_value


# ============================================================
# 3. SOFT RISK-OFF (stricter thresholds, returns multipliers)
# ============================================================
def compute_risk_off_multipliers(macro_feat):
    """Compute per-day risk-off sizing multipliers.

    MUCH STRICTER than V8.3 (which used regime override):
    - V8.3: triggered 21.6% of days, forced BEAR (blocked ALL longs)
    - V8.4: only extreme conditions (~5-10%), reduces sizing (still trades)

    Returns: dict of date_str -> multiplier (0.3 to 1.0)
    Days NOT in dict = 1.0 (normal)
    """
    mults = {}
    if macro_feat is None:
        return mults

    for dt in macro_feat.index:
        row = macro_feat.loc[dt]
        ds = dt.strftime('%Y-%m-%d')
        mult = 1.0

        dxy5 = row.get('dxy_ret_5d', 0) or 0
        spy5 = row.get('spy_ret_5d', 0) or 0
        gsr = row.get('gold_spy_ratio', 0) or 0
        dxy20 = row.get('dxy_ret_20d', 0) or 0
        spy20 = row.get('spy_ret_20d', 0) or 0

        # SEVERE: DXY up >2% AND SPY down >2% in 5d
        if dxy5 > 0.02 and spy5 < -0.02:
            mult = min(mult, 0.3)
        # MODERATE: DXY up >1.5% AND SPY down >1.5% in 5d
        elif dxy5 > 0.015 and spy5 < -0.015:
            mult = min(mult, 0.5)

        # SEVERE: Massive flight to safety (gold >> SPY by >4% in 5d)
        if gsr > 0.04:
            mult = min(mult, 0.3)
        # MODERATE: Gold outperforming SPY by >3% in 5d
        elif gsr > 0.03:
            mult = min(mult, 0.5)

        # SEVERE: Dollar surging >3% in 5d
        if dxy5 > 0.03:
            mult = min(mult, 0.3)

        # STRUCTURAL: DXY up >4% in 20d AND SPY down >4% in 20d
        if dxy20 > 0.04 and spy20 < -0.04:
            mult = min(mult, 0.4)

        if mult < 1.0:
            mults[ds] = mult

    return mults


# ============================================================
# 4. BACKTEST WITH MACRO INTELLIGENCE V2
# ============================================================
def backtest_with_macro(all_data, regime_daily, models,
                        macro_scorer=None, macro_feat=None, macro_fcols=None,
                        risk_off_mults=None,
                        use_macro_sizing=False,
                        use_riskoff_sizing=False,
                        use_adaptive_thresh=False,
                        base_thresh=0.7,
                        use_trailing=True, use_atr=True,
                        max_notional=MAX_NOTIONAL):
    """V7 backtest with V8.4 macro intelligence.

    Three independent improvements:
    1. Adaptive Threshold: adjust V7 confidence threshold by macro score
       score=0   -> thresh=0.90 (very selective, only strong signals)
       score=0.5 -> thresh=0.70 (V7 default)
       score=1.0 -> thresh=0.50 (accept more signals)

    2. ML Sizing: scale position size by macro score [0.3x to 1.8x]

    3. Soft Risk-Off: multiply sizing by risk-off multiplier (0.3-1.0)
       No regime override -> longs still allowed, just smaller
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
    riskoff_days = 0
    sizing_adjustments = []
    thresh_adjustments = []
    trades_accepted_extra = 0
    trades_rejected_extra = 0

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

            if use_trailing and pos.get('trail_on'):
                if pos['dir'] == 1:
                    new = bh * (1 - TRAILING_LOCK * pos['atr_pct'])
                    pos['trail_sl'] = max(pos.get('trail_sl', 0), new)
                else:
                    new = bl * (1 + TRAILING_LOCK * pos['atr_pct'])
                    pos['trail_sl'] = min(pos.get('trail_sl', 1e15), new)

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

        # Macro score for today (0.5 = neutral default)
        macro_score = macro_scores.get(ds, 0.5)

        # 1. Adaptive threshold
        if use_adaptive_thresh:
            # score=0 -> 0.90, score=0.5 -> 0.70, score=1.0 -> 0.50
            thresh = 0.9 - 0.4 * macro_score
            thresh = max(0.50, min(0.90, thresh))
        else:
            thresh = base_thresh
        thresh_adjustments.append(thresh)

        # 2. ML sizing multiplier
        if use_macro_sizing:
            # score=0 -> 0.3x, score=0.5 -> 1.05x, score=1.0 -> 1.8x
            ml_mult = 0.3 + 1.5 * macro_score
        else:
            ml_mult = 1.0

        # 3. Risk-off sizing multiplier (soft, NOT regime override)
        ro_mult = risk_off_mults.get(ds, 1.0) if use_riskoff_sizing else 1.0
        if ro_mult < 1.0 and t.hour == 0:
            riskoff_days += 1

        # Combined sizing
        sizing_mult = ml_mult * ro_mult
        sizing_mult = max(0.2, min(2.0, sizing_mult))  # Clamp

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

            # Regime filter (NO override, V7 rules intact)
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

            atr_s = atr_dict[pair]
            atr_val = atr_s.loc[t] if t in atr_s.index else None

            if use_atr and atr_val is not None and not np.isnan(atr_val):
                atr_pct = atr_val / entry
                tp_pct = max(0.005, min(0.08, atr_pct * ATR_TP[reg]))
                sl_pct = max(0.003, min(0.04, atr_pct * ATR_SL[reg]))
            else:
                tp_pct = 0.03; sl_pct = 0.015

            mh = 30 if reg != 'RANGE' else 15

            # V7 base risk
            risk_pct = RISK_PER_TRADE
            if conf > 2.0: risk_pct = 0.03
            elif conf > 1.5: risk_pct = 0.025

            # Apply macro sizing
            risk_pct *= sizing_mult
            sizing_adjustments.append(sizing_mult)

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

    stats = {
        'riskoff_days': riskoff_days,
        'avg_sizing': np.mean(sizing_adjustments) if sizing_adjustments else 1.0,
        'avg_thresh': np.mean(thresh_adjustments) if thresh_adjustments else base_thresh,
    }
    return trades, max_dd, killed, stats


# ============================================================
# 5. METRICS
# ============================================================
def extract_metrics(trades, max_dd, killed):
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

    months = defaultdict(float)
    for t in trades: months[t['time'].strftime('%Y-%m')] += t['pnl']
    pos_months = sum(1 for v in months.values() if v > 0)

    return {
        'ret': ret, 'annual': annual, 'trades': total, 'wr': wr,
        'pf': pf, 'dd': max_dd * 100, 'sharpe': sharpe, 'killed': killed,
        'final': final, 'days': days,
        'months_pos': pos_months, 'months_total': len(months),
    }


# ============================================================
# 6. CONFIG DEFINITIONS
# ============================================================
CONFIGS = [
    ('V7',       {'adapt_thresh': False, 'ml_sizing': False, 'riskoff': False}),
    ('AdaptTh',  {'adapt_thresh': True,  'ml_sizing': False, 'riskoff': False}),
    ('SoftRO',   {'adapt_thresh': False, 'ml_sizing': False, 'riskoff': True}),
    ('ThreshRO', {'adapt_thresh': True,  'ml_sizing': False, 'riskoff': True}),
    ('Full',     {'adapt_thresh': True,  'ml_sizing': True,  'riskoff': True}),
]


def run_config(cfg_name, cfg_flags, all_data, regime, models,
               scorer, scorer_fcols, macro_feat, ro_mults):
    """Run a single config and return (trades, max_dd, killed, stats)."""
    if cfg_name == 'V7':
        tr, _, mdd, kil = backtest(
            all_data, regime, models,
            use_trailing=True, use_atr=True, use_compound=False, max_notional=300,
        )
        return tr, mdd, kil, {}

    return backtest_with_macro(
        all_data, regime, models,
        macro_scorer=scorer if (cfg_flags['ml_sizing'] or cfg_flags['adapt_thresh']) else None,
        macro_feat=macro_feat,
        macro_fcols=scorer_fcols,
        risk_off_mults=ro_mults if cfg_flags['riskoff'] else None,
        use_macro_sizing=cfg_flags['ml_sizing'],
        use_riskoff_sizing=cfg_flags['riskoff'],
        use_adaptive_thresh=cfg_flags['adapt_thresh'],
    )


# ============================================================
# 7. MAIN
# ============================================================
def main():
    t0 = time.time()

    # --- Download ---
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

    # Risk-off multipliers (soft)
    ro_mults = compute_risk_off_multipliers(macro_feat)
    ro_days = len(ro_mults)
    ro_pct = ro_days / len(macro_feat) * 100
    severe = sum(1 for v in ro_mults.values() if v <= 0.3)
    moderate = ro_days - severe
    print(f'  Risk-off: {ro_days} dias ({ro_pct:.1f}%) '
          f'[severe(0.3x): {severe}, moderate(0.5x): {moderate}]')

    # ================================================================
    # WALK-FORWARD: 5 configs x 4 folds
    # ================================================================
    print('\n' + '=' * 80)
    print('WALK-FORWARD: 5 configs')
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

        # Train MacroScorer
        oos_trades = generate_v7_oos_trades(all_data, regime, tr_e, n_trials=15)
        macro_X, macro_y = create_scorer_labels(oos_trades, macro_feat)
        scorer = None
        scorer_fcols = []
        if macro_X is not None and len(macro_X) > 50:
            scorer, scorer_fcols, auc = train_macro_scorer(macro_X, macro_y, n_trials=20)
            print(f'    MacroScorer: AUC={auc:.3f} | {len(macro_X)} days')

        # Run all configs
        results = {}
        for cfg_name, cfg_flags in CONFIGS:
            tr, mdd, kil, stats = run_config(
                cfg_name, cfg_flags, all_data, regime, mdls_f,
                scorer, scorer_fcols, macro_feat, ro_mults,
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
            if stats.get('riskoff_days'):
                extra += f' | RO:{stats["riskoff_days"]}d'
            if stats.get('avg_sizing') and stats['avg_sizing'] != 1.0:
                extra += f' | sz:{stats["avg_sizing"]:.2f}x'
            if stats.get('avg_thresh') and abs(stats['avg_thresh'] - 0.7) > 0.01:
                extra += f' | th:{stats["avg_thresh"]:.2f}'
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

    # Train MacroScorer
    oos_trades = generate_v7_oos_trades(all_data, regime, '2024-01', n_trials=20)
    macro_X, macro_y = create_scorer_labels(oos_trades, macro_feat)
    scorer = None
    scorer_fcols = []
    if macro_X is not None and len(macro_X) > 50:
        scorer, scorer_fcols, auc = train_macro_scorer(macro_X, macro_y, n_trials=40)
        print(f'\n  MacroScorer: AUC={auc:.3f} | {len(macro_X)} days')

        if hasattr(scorer, 'feature_importances_'):
            imp = sorted(zip(scorer_fcols, scorer.feature_importances_),
                         key=lambda x: x[1], reverse=True)
            print(f'\n  Feature Importance (top 10):')
            for i, (f, v) in enumerate(imp[:10]):
                bar = '#' * min(int(v / max(1, imp[0][1]) * 30), 30)
                print(f'    {i+1:>2}. {f:<25} {v:>6.1f} {bar}')

    # Run all configs
    full_metrics = {}
    for cfg_name, cfg_flags in CONFIGS:
        tr, mdd, kil, stats = run_config(
            cfg_name, cfg_flags, all_data, regime, models,
            scorer, scorer_fcols, macro_feat, ro_mults,
        )
        extra = ''
        if stats.get('riskoff_days'):
            extra = f' (RiskOff: {stats["riskoff_days"]} days)'
        if stats.get('avg_sizing') and stats['avg_sizing'] != 1.0:
            extra += f' (AvgSize: {stats["avg_sizing"]:.2f}x)'
        if stats.get('avg_thresh') and abs(stats['avg_thresh'] - 0.7) > 0.01:
            extra += f' (AvgThresh: {stats["avg_thresh"]:.2f})'
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
        consistent = abs(wf_pf - fb['pf']) < 0.15  # WF and full agree within 0.15 PF
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

        # Must improve on full backtest
        improves_pf = m['pf'] > v7['pf']
        improves_dd = m['dd'] < v7['dd']
        improves_sharpe = m['sharpe'] > v7['sharpe']

        # Must also be consistent in walk-forward
        wf_ok = True
        if wf and v7_wf:
            wf_pf = np.mean([r['pf'] for r in wf])
            v7_wf_pf = np.mean([r['pf'] for r in v7_wf])
            wf_ok = wf_pf >= v7_wf_pf * 0.95  # Allow 5% WF PF degradation

        if (improves_pf and improves_dd and wf_ok) or \
           (improves_sharpe and improves_dd and wf_ok):
            best_cfg = cfg_name
            best_m = m
            break

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
