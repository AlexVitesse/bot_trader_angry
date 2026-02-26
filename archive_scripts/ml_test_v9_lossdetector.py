"""
V9 Test: LossDetector - Clasificador binario para filtrar trades perdedores
===========================================================================
ConvictionScorer predice PnL (regresion) con 10 features -> dificil.
LossDetector predice "va a PERDER este trade?" (clasificacion) -> mas facil.

Usa features MAS RICOS que ConvictionScorer:
  - 10 features de ConvictionScorer (conf, pred_mag, macro, regime, etc.)
  - TA del par al momento de entrada (RSI, BB%, vol_ratio, returns)
  - Contexto BTC al momento de entrada (ret, RSI, vol)
  - Prediction de ConvictionScorer (como feature adicional)
  - Ratio TP/SL, hora del dia

Pipeline: V7 -> MacroScorer -> ConvictionScorer -> LossDetector -> Ejecutar

Ejecutar: python -u ml_test_v9_lossdetector.py
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import lightgbm as lgb
import optuna
from collections import defaultdict
from scipy.special import expit
import time
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from ml_train_v7 import (
    download_all, detect_regime, map_regime_4h,
    compute_features, train_all, backtest,
    PAIRS, COMMISSION, SLIPPAGE, START_CAPITAL, MIN_TRAIN_CANDLES,
    MAX_CONCURRENT, MAX_DD_PCT, MAX_DAILY_LOSS_PCT, RISK_PER_TRADE,
    MAX_NOTIONAL, TRAILING_ACTIVATION, TRAILING_LOCK,
    LEV, ATR_TP, ATR_SL,
)
from ml_train_v84 import (
    download_everything, generate_v7_oos_trades,
    create_scorer_labels, train_macro_scorer,
    compute_risk_off_multipliers,
)
from ml_train_v85 import (
    backtest_with_conviction, backtest_collect_trade_features,
    prepare_conviction_data, train_conviction_scorer,
    conviction_to_sizing, CONVICTION_FEATURES,
)
from macro_data import compute_macro_features

import pandas_ta as pta

N_TRIALS = 15
N_TRIALS_SCORER = 20
N_TRIALS_LOSS = 25

# ============================================================
# 1. PRE-COMPUTE ENRICHED FEATURES PER PAR
# ============================================================

def precompute_pair_ta(all_data):
    """Pre-compute TA indicators for each pair (used as LossDetector features)."""
    pair_ta = {}
    for pair, df in all_data.items():
        f = pd.DataFrame(index=df.index)
        c = df['close']
        f['rsi14'] = ta.rsi(c, length=14)

        bb = ta.bbands(c, length=20, std=2.0)
        if bb is not None:
            bbu = bb.iloc[:, 0]  # upper
            bbm = bb.iloc[:, 1]  # mid
            bbl = bb.iloc[:, 2]  # lower
            bb_range = bbu - bbl
            f['bb_pct'] = np.where(bb_range > 0, (c - bbl) / bb_range, 0.5)
        else:
            f['bb_pct'] = 0.5

        vol_ma = df['volume'].rolling(20).mean()
        f['vol_ratio'] = np.where(vol_ma > 0, df['volume'] / vol_ma, 1.0)
        f['ret_5'] = c.pct_change(5)
        f['ret_20'] = c.pct_change(20)

        pair_ta[pair] = f
    return pair_ta


def compute_btc_context(btc_df):
    """BTC context features on 4h candles."""
    c = btc_df['close']
    ctx = pd.DataFrame(index=btc_df.index)
    ctx['btc_ret_5'] = c.pct_change(5)
    ctx['btc_rsi14'] = ta.rsi(c, length=14)
    ctx['btc_vol20'] = c.pct_change().rolling(20).std()
    return ctx


# ============================================================
# 2. BACKTEST V8.5 ENRICHED (captures rich features per trade)
# ============================================================

LOSS_FEATURES = CONVICTION_FEATURES + [
    'ld_conviction_pred',  # ConvictionScorer prediction
    'ld_pair_rsi14',       # RSI del par
    'ld_pair_bb_pct',      # Bollinger Band position
    'ld_pair_vol_ratio',   # Volume vs promedio
    'ld_pair_ret_5',       # Return 5 barras
    'ld_pair_ret_20',      # Return 20 barras
    'ld_btc_ret_5',        # BTC return 5 barras
    'ld_btc_rsi14',        # BTC RSI
    'ld_btc_vol20',        # BTC volatilidad
    'ld_hour',             # Hora del dia
    'ld_tp_sl_ratio',      # Ratio TP/SL
]


def backtest_v85_with_lossdetector(
    all_data, regime_daily, models,
    pair_ta, btc_ctx,
    macro_scorer=None, macro_feat=None, macro_fcols=None,
    risk_off_mults=None,
    conviction_scorer=None, conviction_fcols=None, conviction_pred_std=1.0,
    loss_detector=None, loss_fcols=None, loss_threshold=0.55,
    collect_mode=False,
    max_notional=300,
):
    """V8.5 backtest + LossDetector filter.

    If collect_mode=True: returns enriched trades (for training LossDetector).
    If loss_detector is not None: applies LossDetector filter before opening.
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
    trades = []
    daily_pnl = defaultdict(float)
    paused_until = None
    killed = False
    trades_skipped_ld = 0

    for t in timeline:
        if killed:
            break
        ds = t.strftime('%Y-%m-%d')

        if paused_until and t.date() <= paused_until:
            continue

        reg = regime_4h.loc[t] if t in regime_4h.index else 'RANGE'
        lev = LEV[reg]

        # === UPDATE POSITIONS ===
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

                trade_info = {
                    'pair': pair, 'dir': pos['dir'], 'pnl': pnl,
                    'pnl_pct': gpnl * 100, 'reason': ex_reason,
                    'bars': pos['bars'], 'time': t, 'regime': pos['reg'],
                    'lev': pos['lev'], 'conf': pos['conf'], 'bal': balance,
                }

                if collect_mode:
                    # Add entry-time features for LossDetector training
                    trade_info.update(pos.get('_ld_features', {}))

                trades.append(trade_info)
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

        # V8.4: Adaptive threshold
        thresh = 0.9 - 0.4 * macro_score
        thresh = max(0.50, min(0.90, thresh))

        # V8.4: ML sizing
        ml_mult = 0.3 + 1.5 * macro_score
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

            # ConvictionScorer
            conv_mult = 1.0
            conviction_pred = 0.0
            if conviction_scorer is not None:
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
                conviction_pred = conviction_scorer.predict(cs_features[cols])[0]
                conv_mult = conviction_to_sizing(conviction_pred, conviction_pred_std)

                # V8.5: Skip very negative conviction
                if conviction_pred < -conviction_pred_std * 0.5:
                    continue

            # Build LossDetector feature vector
            ld_features = {
                'cs_conf': conf,
                'cs_pred_mag': cand['pred_mag'],
                'cs_macro_score': macro_score,
                'cs_risk_off': ro_mult,
                'cs_regime_bull': 1.0 if reg == 'BULL' else 0.0,
                'cs_regime_bear': 1.0 if reg == 'BEAR' else 0.0,
                'cs_regime_range': 1.0 if reg == 'RANGE' else 0.0,
                'cs_atr_pct': atr_pct,
                'cs_n_open': float(len(positions)),
                'cs_pred_sign': float(d),
                'ld_conviction_pred': conviction_pred,
                'ld_pair_rsi14': pair_ta.get(pair, pd.DataFrame()).get('rsi14', pd.Series()).get(t, 50.0),
                'ld_pair_bb_pct': pair_ta.get(pair, pd.DataFrame()).get('bb_pct', pd.Series()).get(t, 0.5),
                'ld_pair_vol_ratio': pair_ta.get(pair, pd.DataFrame()).get('vol_ratio', pd.Series()).get(t, 1.0),
                'ld_pair_ret_5': pair_ta.get(pair, pd.DataFrame()).get('ret_5', pd.Series()).get(t, 0.0),
                'ld_pair_ret_20': pair_ta.get(pair, pd.DataFrame()).get('ret_20', pd.Series()).get(t, 0.0),
                'ld_btc_ret_5': btc_ctx.get('btc_ret_5', pd.Series()).get(t, 0.0) if btc_ctx is not None else 0.0,
                'ld_btc_rsi14': btc_ctx.get('btc_rsi14', pd.Series()).get(t, 50.0) if btc_ctx is not None else 50.0,
                'ld_btc_vol20': btc_ctx.get('btc_vol20', pd.Series()).get(t, 0.02) if btc_ctx is not None else 0.02,
                'ld_hour': float(t.hour),
                'ld_tp_sl_ratio': tp_pct / sl_pct if sl_pct > 0 else 2.0,
            }

            # LossDetector check
            if loss_detector is not None and loss_fcols is not None:
                ld_df = pd.DataFrame([ld_features])
                cols_ld = [c for c in loss_fcols if c in ld_df.columns]
                p_loss = loss_detector.predict_proba(ld_df[cols_ld])[0][1]
                if p_loss > loss_threshold:
                    trades_skipped_ld += 1
                    continue

            # Combined sizing
            total_sizing = macro_sizing * conv_mult
            total_sizing = max(0.2, min(2.5, total_sizing))
            risk_pct *= total_sizing

            base = START_CAPITAL
            risk_amt = base * risk_pct
            margin = risk_amt / (sl_pct * lev) if sl_pct > 0 else risk_amt
            notional = margin * lev
            notional = min(notional, max_notional)

            if d == 1:
                tp_pr = entry * (1 + tp_pct)
                sl_pr = entry * (1 - sl_pct)
            else:
                tp_pr = entry * (1 - tp_pct)
                sl_pr = entry * (1 + sl_pct)

            pos_data = {
                'entry': entry, 'dir': d, 'tp': tp_pr, 'sl': sl_pr,
                'tp_pct': tp_pct, 'sl_pct': sl_pct,
                'atr_pct': atr_pct,
                'not': notional, 'lev': lev, 'mh': mh,
                'bars': 0, 'conf': conf, 'reg': reg,
                'trail_on': False, 'trail_sl': None,
            }
            if collect_mode:
                pos_data['_ld_features'] = ld_features

            positions[pair] = pos_data

    stats = {'trades_skipped_ld': trades_skipped_ld}
    return trades, max_dd, killed, stats


# ============================================================
# 3. TRAIN LOSS DETECTOR
# ============================================================

def train_loss_detector(enriched_trades, n_trials=N_TRIALS_LOSS):
    """Train binary classifier: predicts P(trade will lose money).

    Uses richer features than ConvictionScorer.
    """
    if len(enriched_trades) < 80:
        print(f'    LossDetector: solo {len(enriched_trades)} trades, minimo 80. SKIP.')
        return None, None, 0.0

    df = pd.DataFrame(enriched_trades)

    # Features
    fcols = [c for c in LOSS_FEATURES if c in df.columns]
    X = df[fcols].fillna(0).copy()

    # Replace inf
    X = X.replace([np.inf, -np.inf], 0)

    # Binary target: 1 = trade lost money, 0 = trade made money
    y = (df['pnl'] < 0).astype(int)

    n_loss = y.sum()
    n_win = len(y) - n_loss
    loss_rate = n_loss / len(y) * 100
    print(f'    LossDetector: {len(y)} trades ({n_win}W/{n_loss}L, {loss_rate:.0f}% losses)')

    if n_loss < 20 or n_win < 20:
        print(f'    LossDetector: clases desbalanceadas, SKIP.')
        return None, None, 0.0

    sp = int(len(X) * 0.8)
    Xt, Xv = X.iloc[:sp], X.iloc[sp:]
    yt, yv = y.iloc[:sp], y.iloc[sp:]

    # Scale factor for class imbalance
    scale = n_win / n_loss if n_loss > 0 else 1.0

    sampler = optuna.samplers.TPESampler(seed=42)

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
            'scale_pos_weight': scale,
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        }
        m = lgb.LGBMClassifier(**p)
        m.fit(Xt, yt, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(20, verbose=False)])
        proba = m.predict_proba(Xv)[:, 1]
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(yv, proba)
        except:
            auc = 0.5
        return auc

    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

    rename = {'ne': 'n_estimators', 'md': 'max_depth', 'lr': 'learning_rate',
              'ss': 'subsample', 'cs': 'colsample_bytree', 'mc': 'min_child_samples',
              'ra': 'reg_alpha', 'rl': 'reg_lambda', 'nl': 'num_leaves'}
    bp = {rename.get(k, k): v for k, v in study.best_params.items()}
    bp.update({'scale_pos_weight': scale, 'random_state': 42, 'n_jobs': -1, 'verbose': -1})

    mdl = lgb.LGBMClassifier(**bp)
    mdl.fit(X, y, eval_set=[(Xv, yv)],
            callbacks=[lgb.early_stopping(30, verbose=False)])

    auc = study.best_value
    print(f'    LossDetector AUC={auc:.3f}')

    return mdl, fcols, auc


# ============================================================
# 4. METRICS HELPER
# ============================================================

def extract_metrics(trades, mdd, killed):
    if not trades:
        return {'n': 0, 'ret': 0, 'pf': 0, 'dd': 0, 'wr': 0}
    fin = trades[-1]['bal']
    ret = ((fin / START_CAPITAL) - 1) * 100
    w = sum(1 for t in trades if t['pnl'] > 0)
    wr = w / len(trades) * 100
    gpp = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gll = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
    pf = gpp / gll if gll > 0 else 0
    return {'n': len(trades), 'ret': ret, 'pf': pf, 'dd': mdd * 100, 'wr': wr}


def print_comparison(fold_name, results):
    print(f'\n  --- COMPARACION {fold_name} ---')
    best_ret = max(r['ret'] for r in results.values())
    for name, m in results.items():
        tag = ''
        if name == 'V85_Full':
            tag = ' [PROD]'
        elif name == 'V9_Full':
            tag = ' [NEW]'
        best = ' <-- BEST' if m['ret'] == best_ret and m['ret'] > 0 else ''
        print(f"    {name:<14}{tag:>7} {m['n']:>4}t | WR {m['wr']:.0f}% | "
              f"PF {m['pf']:.2f} | Ret {m['ret']:>+8.1f}% | DD {m['dd']:.1f}%{best}")


# ============================================================
# 5. MAIN WALK-FORWARD
# ============================================================

FOLDS = [
    ('H2-2023',  '2023-06', '2024-01'),
    ('H1-2024',  '2024-01', '2024-07'),
    ('H2-2024',  '2024-07', '2025-01'),
    ('2025+',    '2025-01', '2026-03'),
]


def main():
    t0 = time.time()
    print('=' * 70)
    print('V9 TEST: LossDetector - Filtro binario de trades perdedores')
    print('=' * 70)

    # --- Download data ---
    print('\n[1] Descargando datos...')
    all_data, btc_daily, macro_raw, fng = download_everything()
    if len(all_data) < 3 or btc_daily is None:
        print('ERROR: datos insuficientes'); return

    # Macro features
    macro_feat = compute_macro_features(
        macro_raw.get('dxy'), macro_raw.get('gold'), macro_raw.get('spy'),
        macro_raw.get('tnx'), macro_raw.get('ethbtc'),
    )
    if macro_feat is None:
        print('ERROR: no macro features'); return
    print(f'  Macro feat   {len(macro_feat):>6} dias')

    # BTC context
    btc_ctx = compute_btc_context(all_data['BTC/USDT'])
    print(f'  BTC context: {len(btc_ctx.columns)} features, {len(btc_ctx)} rows')

    # Pre-compute pair TA
    pair_ta = precompute_pair_ta(all_data)
    print(f'  Pair TA:     {len(pair_ta)} pares, 5 features cada uno')

    # Regime
    regime_daily = detect_regime(btc_daily)

    # --- Walk-forward ---
    all_results = defaultdict(list)

    for fold_name, train_end, test_end in FOLDS:
        print(f'\n{"=" * 60}')
        print(f'Fold {fold_name}: train < {train_end}, test [{train_end}, {test_end})')
        print(f'{"=" * 60}')

        # [A] Train V7 models
        print(f'\n  [A] V7 signals...')
        models_v7 = train_all(all_data, train_end, horizon=5, n_trials=N_TRIALS)

        if not models_v7:
            print('  No models trained. SKIP fold.')
            continue

        # [B] MacroScorer (nested walk-forward)
        print(f'\n  [B] MacroScorer...')

        macro_fcols = [c for c in macro_feat.columns if c != 'target']
        macro_scorer = None
        ro_mults = {}

        oos_trades = generate_v7_oos_trades(
            all_data, regime_daily, train_end, n_trials=N_TRIALS,
        )
        if len(oos_trades) > 50:
            macro_X, macro_y = create_scorer_labels(oos_trades, macro_feat)
            if macro_X is not None and len(macro_X) > 50:
                macro_scorer, macro_fcols, macro_auc = train_macro_scorer(
                    macro_X, macro_y, n_trials=N_TRIALS_SCORER,
                )
                print(f'    MacroScorer: {len(oos_trades)} OOS trades, AUC={macro_auc:.3f}')
                ro_mults = compute_risk_off_multipliers(macro_feat)
            else:
                print(f'    MacroScorer: insuficientes labels, SKIP.')
        else:
            print(f'    MacroScorer: solo {len(oos_trades)} trades, SKIP.')

        # [C] ConvictionScorer
        print(f'\n  [C] ConvictionScorer...')
        conv_scorer = None
        conv_fcols = CONVICTION_FEATURES
        conv_pred_std = 1.0

        enriched_trades = backtest_collect_trade_features(
            all_data, regime_daily, models_v7,
            macro_scorer=macro_scorer, macro_feat=macro_feat,
            macro_fcols=macro_fcols, risk_off_mults=ro_mults,
            max_notional=300,
        )

        if len(enriched_trades) > 50:
            X_conv, y_conv = prepare_conviction_data(enriched_trades)
            if X_conv is not None:
                conv_scorer, conv_fcols, conv_corr, conv_pred_std = \
                    train_conviction_scorer(X_conv, y_conv, n_trials=N_TRIALS_SCORER)
                print(f'    ConvictionScorer: corr={conv_corr:.3f}')

        # [D] LossDetector training
        print(f'\n  [D] LossDetector training...')
        # Run V8.5 on evaluation period to collect enriched trades
        ld_trades, _, _, _ = backtest_v85_with_lossdetector(
            all_data, regime_daily, models_v7,
            pair_ta, btc_ctx,
            macro_scorer=macro_scorer, macro_feat=macro_feat,
            macro_fcols=macro_fcols, risk_off_mults=ro_mults,
            conviction_scorer=conv_scorer, conviction_fcols=conv_fcols,
            conviction_pred_std=conv_pred_std,
            loss_detector=None,  # No LossDetector yet - just collecting
            collect_mode=True,   # Capture features
            max_notional=300,
        )

        loss_detector = None
        loss_fcols = None
        ld_auc = 0.0

        if ld_trades:
            loss_detector, loss_fcols, ld_auc = train_loss_detector(ld_trades)

        # [E] Backtests on TEST period
        print(f'\n  [E] Backtests...')

        # Filter data to test period
        test_data = {}
        for pair, df in all_data.items():
            mask = (df.index >= train_end) & (df.index < test_end)
            if mask.any():
                test_data[pair] = df[mask]

        # Recompute V7 preds on test data
        test_models = {}
        for pair, info in models_v7.items():
            mask = (info['index'] >= pd.Timestamp(train_end)) & \
                   (info['index'] < pd.Timestamp(test_end))
            if mask.any():
                test_models[pair] = {
                    'model': info['model'],
                    'fcols': info['fcols'],
                    'pred_std': info['pred_std'],
                    'preds': info['preds'][mask],
                    'index': info['index'][mask],
                }

        if not test_models:
            print('  No test models. SKIP.')
            continue

        results = {}

        # A) V7 baseline
        tr_v7, _, mdd_v7, _ = backtest(
            all_data, regime_daily, test_models,
            use_trailing=True, use_atr=True, use_compound=False, max_notional=300,
        )
        results['V7'] = extract_metrics(tr_v7, mdd_v7, False)

        # B) V8.5 Full [PROD]
        tr_v85, mdd_v85, kil_v85, _ = backtest_with_conviction(
            all_data, regime_daily, test_models,
            conviction_scorer=conv_scorer, conviction_fcols=conv_fcols,
            conviction_pred_std=conv_pred_std,
            macro_scorer=macro_scorer, macro_feat=macro_feat,
            macro_fcols=macro_fcols, risk_off_mults=ro_mults,
            use_macro_sizing=True, use_riskoff_sizing=True,
            use_adaptive_thresh=True, use_conviction=True,
            max_notional=300,
        )
        results['V85_Full'] = extract_metrics(tr_v85, mdd_v85, kil_v85)

        # C) V9: V8.5 + LossDetector (threshold 0.55)
        tr_v9a, mdd_v9a, kil_v9a, st_v9a = backtest_v85_with_lossdetector(
            all_data, regime_daily, test_models,
            pair_ta, btc_ctx,
            macro_scorer=macro_scorer, macro_feat=macro_feat,
            macro_fcols=macro_fcols, risk_off_mults=ro_mults,
            conviction_scorer=conv_scorer, conviction_fcols=conv_fcols,
            conviction_pred_std=conv_pred_std,
            loss_detector=loss_detector, loss_fcols=loss_fcols,
            loss_threshold=0.55,
            collect_mode=False, max_notional=300,
        )
        results['V9_LD55'] = extract_metrics(tr_v9a, mdd_v9a, kil_v9a)
        skipped_55 = st_v9a.get('trades_skipped_ld', 0)

        # D) V9: V8.5 + LossDetector (threshold 0.50 - more aggressive)
        tr_v9b, mdd_v9b, kil_v9b, st_v9b = backtest_v85_with_lossdetector(
            all_data, regime_daily, test_models,
            pair_ta, btc_ctx,
            macro_scorer=macro_scorer, macro_feat=macro_feat,
            macro_fcols=macro_fcols, risk_off_mults=ro_mults,
            conviction_scorer=conv_scorer, conviction_fcols=conv_fcols,
            conviction_pred_std=conv_pred_std,
            loss_detector=loss_detector, loss_fcols=loss_fcols,
            loss_threshold=0.50,
            collect_mode=False, max_notional=300,
        )
        results['V9_LD50'] = extract_metrics(tr_v9b, mdd_v9b, kil_v9b)
        skipped_50 = st_v9b.get('trades_skipped_ld', 0)

        # E) V9: V8.5 + LossDetector (threshold 0.60 - conservative)
        tr_v9c, mdd_v9c, kil_v9c, st_v9c = backtest_v85_with_lossdetector(
            all_data, regime_daily, test_models,
            pair_ta, btc_ctx,
            macro_scorer=macro_scorer, macro_feat=macro_feat,
            macro_fcols=macro_fcols, risk_off_mults=ro_mults,
            conviction_scorer=conv_scorer, conviction_fcols=conv_fcols,
            conviction_pred_std=conv_pred_std,
            loss_detector=loss_detector, loss_fcols=loss_fcols,
            loss_threshold=0.60,
            collect_mode=False, max_notional=300,
        )
        results['V9_LD60'] = extract_metrics(tr_v9c, mdd_v9c, kil_v9c)
        skipped_60 = st_v9c.get('trades_skipped_ld', 0)

        if loss_detector is not None:
            print(f'    LossDetector filtro: T50={skipped_50} skip, T55={skipped_55} skip, T60={skipped_60} skip')

        print_comparison(fold_name, results)

        for name, m in results.items():
            all_results[name].append(m)

    # === FINAL SUMMARY ===
    elapsed = (time.time() - t0) / 60
    print(f'\n{"=" * 70}')
    print(f'RESUMEN: V8.5 [PROD] vs V9 [V8.5 + LossDetector]')
    print(f'{"=" * 70}')

    print(f'\n  {"Config":<16} {"Avg Ret":>8} {"Avg PF":>7} {"Avg DD":>7} {"Folds+":>7}')
    print(f'  {"-" * 16} {"-" * 8} {"-" * 7} {"-" * 7} {"-" * 7}')

    for name in ['V7', 'V85_Full', 'V9_LD50', 'V9_LD55', 'V9_LD60']:
        if name not in all_results:
            continue
        ms = all_results[name]
        avg_ret = np.mean([m['ret'] for m in ms])
        avg_pf = np.mean([m['pf'] for m in ms])
        avg_dd = np.mean([m['dd'] for m in ms])
        folds_pos = sum(1 for m in ms if m['ret'] > 0)
        tag = ' [PROD]' if name == 'V85_Full' else ''
        print(f'  {name:<16}{tag:>7} {avg_ret:>+7.1f}% {avg_pf:>7.2f} {avg_dd:>6.1f}% {folds_pos:>4}/{len(ms)}')

    # V9 vs V8.5 comparison (best threshold)
    if 'V85_Full' in all_results:
        v85_rets = [m['ret'] for m in all_results['V85_Full']]

        best_name = None
        best_diff = -999
        for name in ['V9_LD50', 'V9_LD55', 'V9_LD60']:
            if name not in all_results:
                continue
            v9_rets = [m['ret'] for m in all_results[name]]
            wins = sum(1 for a, b in zip(v9_rets, v85_rets) if a > b)
            diff = np.mean(v9_rets) - np.mean(v85_rets)
            if diff > best_diff:
                best_diff = diff
                best_name = name
                best_wins = wins

        if best_name:
            print(f'\n  Mejor config: {best_name}')
            print(f'  {best_name} gana {best_wins}/{len(v85_rets)} folds vs V8.5')
            print(f'  Diferencia: {best_diff:+.1f}% retorno promedio')

            if best_diff > 0 and best_wins >= 3:
                print(f'\n  ** APROBADO: LossDetector mejora V8.5! **')
            else:
                print(f'\n  ** NO APROBADO: LossDetector no mejora V8.5 **')

    print(f'\n  Tiempo total: {elapsed:.1f} min')
    print('=' * 70)


if __name__ == '__main__':
    main()
