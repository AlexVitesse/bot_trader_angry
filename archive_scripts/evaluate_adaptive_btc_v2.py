"""
evaluate_adaptive_btc_v2.py  --  Round 2: Targeted Adaptive Experiments
==============================================================================
Round 1 lesson: adaptive vol HURT (added bad trades). Quality score 70+ trades
had perfect WR but too few. None fixed the 4 failed folds.

New approach: test each idea ISOLATED to find what helps and what hurts.

Configs tested:
  A : Baseline (unchanged reference)
  E : Quality filter ONLY (no vol/lookback change, just skip score<40)
  F : Quality filter strict (skip score<50)
  G : Quality-based RR ONLY (no skip, but 70+->RR=2.0, <40->RR=1.0)
  H : Adaptive vol CONSERVATIVE (1.5/1.8/2.0 instead of 1.3/1.8/2.2)
  I : Dynamic lookback ONLY (no vol/quality changes)
  J : Quality skip<40 + dynamic lookback (best of E+I, no adaptive vol)
  K : Quality skip<40 + tighter BB (need 4/5 instead of 3/5 narrow bars)
  L : Stricter bar_move (2.0% instead of 2.5%) + quality skip<40

Failed folds: 2021-H2, 2023-H1, 2025-H1, 2025-H2 — all LONG-dominated choppy.
"""

import sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from v15_framework import (
    load_btc_4h, compute_features_4h, compute_macro_daily, merge_daily_to_4h,
    sim_trade_fixed, metrics,
    WF_FOLDS, OOS_START, OOS_END, COMMISSION
)
from backtest_v15_committee import (
    add_extra_features, detect_regime, detect_long_pullback,
    detect_short_ml, sim_short, add_funding_zscore,
    create_short_labels, SHORT_FEATURES,
    LONG_MAX_BARS, SHORT_MAX_BARS,
    FUNDING_VETO_LONG, FUNDING_VETO_SHORT, CONSEC_LOSS_PAUSE,
)

# Import adaptive functions from round 1
from evaluate_adaptive_btc import (
    adaptive_vol_threshold, adaptive_bb_compression,
    adaptive_lookback, compute_signal_quality,
)


# ============================================================
# BREAKOUT DETECTORS — each variant isolated
# ============================================================

def detect_breakout_A(df, i, cfg, regime='BULL'):
    """A: Baseline static."""
    if i < 25:
        return None
    row = df.iloc[i]

    high20 = float(df['high'].iloc[i-20:i].max())
    if float(row['close']) <= high20:
        return None
    if float(row.get('vol_ratio', 1)) < cfg.get('breakout_vol_min', 1.8):
        return None
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > cfg.get('breakout_bar_move_max', 2.5):
        return None
    bb_max = cfg.get('breakout_bb_max', 4.0)
    bb_count = cfg.get('bb_count_min', 3)
    if (df['bb_width'].iloc[i-5:i] < bb_max).sum() < bb_count:
        return None
    if df['adx14'].iloc[i-3:i].mean() > cfg.get('breakout_adx_max', 28):
        return None

    entry = float(row['close'])
    sl_raw = float(df['low'].iloc[i-5:i].min()) * 0.997
    sl_pct = (entry - sl_raw) / entry
    if sl_pct < 0.005 or sl_pct > 0.04:
        return None
    rr = cfg.get('breakout_rr', 1.5)
    tp_pct = sl_pct * rr
    return {'direction': 'LONG', 'setup': 'BREAKOUT_B',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}


def detect_breakout_E(df, i, cfg, regime='BULL'):
    """E: Quality filter ONLY (skip score < quality_min). No vol/lookback change."""
    if i < 25:
        return None
    row = df.iloc[i]

    high20 = float(df['high'].iloc[i-20:i].max())
    if float(row['close']) <= high20:
        return None
    if float(row.get('vol_ratio', 1)) < cfg.get('breakout_vol_min', 1.8):
        return None
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > cfg.get('breakout_bar_move_max', 2.5):
        return None
    bb_max = cfg.get('breakout_bb_max', 4.0)
    if (df['bb_width'].iloc[i-5:i] < bb_max).sum() < 3:
        return None
    if df['adx14'].iloc[i-3:i].mean() > cfg.get('breakout_adx_max', 28):
        return None

    # QUALITY GATE
    quality = compute_signal_quality(df, i, regime, cfg)
    if quality < cfg.get('adaptive_quality_min', 40):
        return None

    entry = float(row['close'])
    sl_raw = float(df['low'].iloc[i-5:i].min()) * 0.997
    sl_pct = (entry - sl_raw) / entry
    if sl_pct < 0.005 or sl_pct > 0.04:
        return None
    rr = cfg.get('breakout_rr', 1.5)
    tp_pct = sl_pct * rr
    return {'direction': 'LONG', 'setup': 'BREAKOUT_B',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct,
            'quality': quality}


def detect_breakout_G(df, i, cfg, regime='BULL'):
    """G: Quality-based RR ONLY (no skip, just adjust RR by quality)."""
    if i < 25:
        return None
    row = df.iloc[i]

    high20 = float(df['high'].iloc[i-20:i].max())
    if float(row['close']) <= high20:
        return None
    if float(row.get('vol_ratio', 1)) < cfg.get('breakout_vol_min', 1.8):
        return None
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > cfg.get('breakout_bar_move_max', 2.5):
        return None
    bb_max = cfg.get('breakout_bb_max', 4.0)
    if (df['bb_width'].iloc[i-5:i] < bb_max).sum() < 3:
        return None
    if df['adx14'].iloc[i-3:i].mean() > cfg.get('breakout_adx_max', 28):
        return None

    quality = compute_signal_quality(df, i, regime, cfg)

    entry = float(row['close'])
    sl_raw = float(df['low'].iloc[i-5:i].min()) * 0.997
    sl_pct = (entry - sl_raw) / entry
    if sl_pct < 0.005 or sl_pct > 0.04:
        return None

    # RR by quality (no skip — every trade enters, but sizing differs)
    if quality >= 70:
        rr = cfg.get('adaptive_rr_high', 2.0)
    elif quality >= 50:
        rr = cfg.get('adaptive_rr_mid', 1.5)
    elif quality >= 30:
        rr = cfg.get('adaptive_rr_low', 1.2)
    else:
        rr = cfg.get('adaptive_rr_floor', 1.0)

    tp_pct = sl_pct * rr
    return {'direction': 'LONG', 'setup': 'BREAKOUT_B',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct,
            'quality': quality}


def detect_breakout_H(df, i, cfg, regime='BULL'):
    """H: Adaptive vol CONSERVATIVE (tighter bands 1.5/1.8/2.0)."""
    if i < 25:
        return None
    row = df.iloc[i]

    high20 = float(df['high'].iloc[i-20:i].max())
    if float(row['close']) <= high20:
        return None

    vol_min = adaptive_vol_threshold(df, i, cfg)
    if float(row.get('vol_ratio', 1)) < vol_min:
        return None

    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > cfg.get('breakout_bar_move_max', 2.5):
        return None
    bb_max = cfg.get('breakout_bb_max', 4.0)
    if (df['bb_width'].iloc[i-5:i] < bb_max).sum() < 3:
        return None
    if df['adx14'].iloc[i-3:i].mean() > cfg.get('breakout_adx_max', 28):
        return None

    entry = float(row['close'])
    sl_raw = float(df['low'].iloc[i-5:i].min()) * 0.997
    sl_pct = (entry - sl_raw) / entry
    if sl_pct < 0.005 or sl_pct > 0.04:
        return None
    rr = cfg.get('breakout_rr', 1.5)
    tp_pct = sl_pct * rr
    return {'direction': 'LONG', 'setup': 'BREAKOUT_B',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}


def detect_breakout_I(df, i, cfg, regime='BULL'):
    """I: Dynamic lookback ONLY."""
    if i < 25:
        return None
    row = df.iloc[i]

    lookback = adaptive_lookback(df, i, cfg)
    if i < lookback + 5:
        return None

    high_N = float(df['high'].iloc[i-lookback:i].max())
    if float(row['close']) <= high_N:
        return None

    if float(row.get('vol_ratio', 1)) < cfg.get('breakout_vol_min', 1.8):
        return None
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > cfg.get('breakout_bar_move_max', 2.5):
        return None
    bb_max = cfg.get('breakout_bb_max', 4.0)
    if (df['bb_width'].iloc[i-5:i] < bb_max).sum() < 3:
        return None
    if df['adx14'].iloc[i-3:i].mean() > cfg.get('breakout_adx_max', 28):
        return None

    entry = float(row['close'])
    sl_raw = float(df['low'].iloc[max(0, i-5):i].min()) * 0.997
    sl_pct = (entry - sl_raw) / entry
    if sl_pct < 0.005 or sl_pct > 0.04:
        return None
    rr = cfg.get('breakout_rr', 1.5)
    tp_pct = sl_pct * rr
    return {'direction': 'LONG', 'setup': 'BREAKOUT_B',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}


def detect_breakout_J(df, i, cfg, regime='BULL'):
    """J: Quality skip<40 + dynamic lookback (no adaptive vol)."""
    if i < 25:
        return None
    row = df.iloc[i]

    lookback = adaptive_lookback(df, i, cfg)
    if i < lookback + 5:
        return None

    high_N = float(df['high'].iloc[i-lookback:i].max())
    if float(row['close']) <= high_N:
        return None

    if float(row.get('vol_ratio', 1)) < cfg.get('breakout_vol_min', 1.8):
        return None
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > cfg.get('breakout_bar_move_max', 2.5):
        return None
    bb_max = cfg.get('breakout_bb_max', 4.0)
    if (df['bb_width'].iloc[i-5:i] < bb_max).sum() < 3:
        return None
    if df['adx14'].iloc[i-3:i].mean() > cfg.get('breakout_adx_max', 28):
        return None

    quality = compute_signal_quality(df, i, regime, cfg)
    if quality < cfg.get('adaptive_quality_min', 40):
        return None

    entry = float(row['close'])
    sl_raw = float(df['low'].iloc[max(0, i-5):i].min()) * 0.997
    sl_pct = (entry - sl_raw) / entry
    if sl_pct < 0.005 or sl_pct > 0.04:
        return None

    if quality >= 70:
        rr = cfg.get('adaptive_rr_high', 2.0)
    elif quality >= 50:
        rr = cfg.get('adaptive_rr_mid', 1.5)
    else:
        rr = cfg.get('adaptive_rr_low', 1.2)

    tp_pct = sl_pct * rr
    return {'direction': 'LONG', 'setup': 'BREAKOUT_B',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct,
            'quality': quality}


def detect_breakout_K(df, i, cfg, regime='BULL'):
    """K: Quality skip<40 + tighter BB (4/5 narrow instead of 3/5)."""
    if i < 25:
        return None
    row = df.iloc[i]

    high20 = float(df['high'].iloc[i-20:i].max())
    if float(row['close']) <= high20:
        return None
    if float(row.get('vol_ratio', 1)) < cfg.get('breakout_vol_min', 1.8):
        return None
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > cfg.get('breakout_bar_move_max', 2.5):
        return None

    # TIGHTER: 4/5 narrow bars (instead of 3/5)
    bb_max = cfg.get('breakout_bb_max', 4.0)
    bb_count = cfg.get('bb_count_min', 4)
    if (df['bb_width'].iloc[i-5:i] < bb_max).sum() < bb_count:
        return None
    if df['adx14'].iloc[i-3:i].mean() > cfg.get('breakout_adx_max', 28):
        return None

    quality = compute_signal_quality(df, i, regime, cfg)
    if quality < cfg.get('adaptive_quality_min', 40):
        return None

    entry = float(row['close'])
    sl_raw = float(df['low'].iloc[i-5:i].min()) * 0.997
    sl_pct = (entry - sl_raw) / entry
    if sl_pct < 0.005 or sl_pct > 0.04:
        return None
    rr = cfg.get('breakout_rr', 1.5)
    tp_pct = sl_pct * rr
    return {'direction': 'LONG', 'setup': 'BREAKOUT_B',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct,
            'quality': quality}


def detect_breakout_L(df, i, cfg, regime='BULL'):
    """L: Stricter bar_move (2.0%) + quality skip<40."""
    if i < 25:
        return None
    row = df.iloc[i]

    high20 = float(df['high'].iloc[i-20:i].max())
    if float(row['close']) <= high20:
        return None
    if float(row.get('vol_ratio', 1)) < cfg.get('breakout_vol_min', 1.8):
        return None

    # STRICTER bar move
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > cfg.get('breakout_bar_move_max', 2.0):
        return None

    bb_max = cfg.get('breakout_bb_max', 4.0)
    if (df['bb_width'].iloc[i-5:i] < bb_max).sum() < 3:
        return None
    if df['adx14'].iloc[i-3:i].mean() > cfg.get('breakout_adx_max', 28):
        return None

    quality = compute_signal_quality(df, i, regime, cfg)
    if quality < cfg.get('adaptive_quality_min', 40):
        return None

    entry = float(row['close'])
    sl_raw = float(df['low'].iloc[i-5:i].min()) * 0.997
    sl_pct = (entry - sl_raw) / entry
    if sl_pct < 0.005 or sl_pct > 0.04:
        return None
    rr = cfg.get('breakout_rr', 1.5)
    tp_pct = sl_pct * rr
    return {'direction': 'LONG', 'setup': 'BREAKOUT_B',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct,
            'quality': quality}


# ============================================================
# COMMITTEE RUNNER (same as v1, parametric breakout)
# ============================================================

def run_committee(df, short_model_data, start_idx, end_idx, breakout_fn, cfg):
    trades = []
    consec_losses = 0
    paused = False

    for i in range(start_idx, end_idx):
        if i < 30:
            continue
        row = df.iloc[i]
        regime = detect_regime(row)
        funding_z = row.get('funding_zscore', 0)
        trade = None

        if paused:
            paused = False
            consec_losses = 0

        if regime == 'BULL':
            if funding_z > FUNDING_VETO_LONG:
                continue
            trade = breakout_fn(df, i, cfg, regime)
            if trade is None:
                trade = detect_long_pullback(df, i)
        elif regime == 'BEAR':
            if funding_z < FUNDING_VETO_SHORT:
                continue
            trade = detect_short_ml(df, i, short_model_data)
        elif regime == 'RANGE':
            if funding_z > FUNDING_VETO_LONG:
                continue
            trade = breakout_fn(df, i, cfg, regime)

        if trade is None:
            continue

        max_b = LONG_MAX_BARS if trade['direction'] == 'LONG' else SHORT_MAX_BARS
        if trade['direction'] == 'LONG':
            out = sim_trade_fixed(df, i, trade['entry'],
                                  trade['tp_pct'], trade['sl_pct'], max_bars=max_b)
        else:
            out = sim_short(df, i, trade['entry'],
                            trade['tp_pct'], trade['sl_pct'], max_bars=max_b)

        if out[0] == 'SL':
            consec_losses += 1
            if consec_losses >= CONSEC_LOSS_PAUSE:
                paused = True
        else:
            consec_losses = 0

        trades.append({
            'outcome': out[0], 'pnl_pct': out[2], 'ts': df.index[i],
            'direction': trade['direction'], 'setup': trade['setup'],
            'regime': regime, 'funding_z': funding_z,
            'quality': trade.get('quality', -1),
        })
    return trades


# ============================================================
# WALK-FORWARD
# ============================================================

def walk_forward_variant(df, labels_short, breakout_fn, cfg, label=''):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    results = []
    all_trades = []

    for fold_idx, (start_s, end_s) in enumerate(WF_FOLDS):
        test_mask = (df.index >= start_s) & (df.index <= end_s)
        train_mask = df.index < start_s
        period_label = f"{start_s[:7]}/{end_s[5:7]}"
        df_train = df[train_mask]

        y_short_tr = labels_short[train_mask]
        bear_train = df_train.get('bull_1d', pd.Series(1, index=df_train.index)) == 0
        valid_short = y_short_tr.notna() & bear_train
        df_short_tr = df_train[valid_short]
        y_short_fit = y_short_tr[valid_short]

        short_model_data = None
        if (len(df_short_tr) >= 800 and y_short_fit.sum() >= 20
                and (len(y_short_fit) - y_short_fit.sum()) >= 20):
            X_s = df_short_tr[SHORT_FEATURES].fillna(0)
            scaler_s = StandardScaler()
            X_ss = scaler_s.fit_transform(X_s)
            model_s = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                min_samples_leaf=20, subsample=0.8, random_state=42
            )
            model_s.fit(X_ss, y_short_fit)
            short_model_data = {'model': model_s, 'scaler': scaler_s}

        df_test = df[test_mask]
        if len(df_test) == 0:
            results.append({'period': period_label, 'n': 0, 'wr': 0,
                            'pf': 0, 'ok': False, 'annual_pct': 0,
                            'n_long': 0, 'n_short': 0})
            continue

        start_bar = df.index.get_loc(df_test.index[0])
        end_bar = df.index.get_loc(df_test.index[-1]) + 1
        trades = run_committee(df, short_model_data, start_bar, end_bar,
                               breakout_fn, cfg)

        m = metrics(trades, period_label)
        days = (pd.Timestamp(end_s) - pd.Timestamp(start_s)).days
        annual = (m['avg_pnl'] * m['n'] / days * 365 * 100
                  if days > 0 and m['n'] > 0 else 0)

        n_long = sum(1 for t in trades if t['direction'] == 'LONG')
        n_short = sum(1 for t in trades if t['direction'] == 'SHORT')

        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        results.append({
            'period': period_label, 'n': m['n'],
            'wr': m['wr'], 'pf': m['pf'],
            'ok': ok, 'annual_pct': annual,
            'n_long': n_long, 'n_short': n_short,
        })
        all_trades.extend(trades)

    folds_ok = sum(1 for r in results if r['ok'])
    return {
        'folds': results, 'folds_ok': folds_ok,
        'folds_total': len(results),
        'approved': folds_ok >= 7, 'all_trades': all_trades,
    }


# ============================================================
# METRICS
# ============================================================

FAILED_FOLDS = ['2021-07/12', '2023-01/06', '2025-01/06', '2025-07/12']


def compute_full_metrics(wf):
    """OOS metrics + equity + drawdown from WF results."""
    all_trades = wf['all_trades']
    oos_trades = [t for t in all_trades
                  if OOS_START <= str(t['ts'])[:10] <= OOS_END]
    m = metrics(oos_trades, 'OOS')

    cumulative = 1.0
    peak = 1.0
    max_dd = 0.0
    for t in sorted(all_trades, key=lambda x: x['ts']):
        cumulative *= (1 + t['pnl_pct'])
        peak = max(peak, cumulative)
        dd = (peak - cumulative) / peak
        max_dd = max(max_dd, dd)

    m['max_dd'] = max_dd
    m['equity_1k'] = 1000 * cumulative

    # LONG-only OOS metrics
    long_oos = [t for t in oos_trades if t['direction'] == 'LONG']
    m_l = metrics(long_oos, 'LONG')
    m['long_n'] = m_l['n']
    m['long_wr'] = m_l['wr']
    m['long_pf'] = m_l['pf']

    # Failed folds fixed
    fixed = sum(1 for f in wf['folds']
                if f['period'] in FAILED_FOLDS and f['ok'])
    m['failed_fixed'] = fixed

    return m


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("BTC ADAPTIVE STRATEGY — ROUND 2: TARGETED EXPERIMENTS")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df_raw = load_btc_4h()
    df = compute_features_4h(df_raw)
    df = add_extra_features(df)
    df_daily = compute_macro_daily(df)
    df = merge_daily_to_4h(df, df_daily)
    df = add_funding_zscore(df)
    print(f"  {len(df)} bars | {df.index[0].date()} - {df.index[-1].date()}")

    labels_short = create_short_labels(df)

    # ---- Define all configs ----
    cfg_base = {
        'breakout_vol_min': 1.8, 'breakout_bb_max': 4.0,
        'breakout_adx_max': 28, 'breakout_bar_move_max': 2.5,
        'breakout_rr': 1.5, 'adaptive_bb_window': 50,
    }

    cfg_quality_40 = {**cfg_base, 'adaptive_quality_min': 40}
    cfg_quality_50 = {**cfg_base, 'adaptive_quality_min': 50}

    cfg_rr_only = {
        **cfg_base,
        'adaptive_rr_high': 2.0, 'adaptive_rr_mid': 1.5,
        'adaptive_rr_low': 1.2, 'adaptive_rr_floor': 1.0,
    }

    cfg_vol_conservative = {
        **cfg_base,
        'adaptive_vol_min_low': 1.5, 'adaptive_vol_min_mid': 1.8,
        'adaptive_vol_min_high': 2.0,
    }

    cfg_lookback = {
        **cfg_base,
        'adaptive_lookback_min': 12, 'adaptive_lookback_max': 30,
    }

    cfg_quality_lookback = {
        **cfg_base, 'adaptive_quality_min': 40,
        'adaptive_lookback_min': 12, 'adaptive_lookback_max': 30,
        'adaptive_rr_high': 2.0, 'adaptive_rr_mid': 1.5, 'adaptive_rr_low': 1.2,
    }

    cfg_quality_tight_bb = {
        **cfg_base, 'adaptive_quality_min': 40, 'bb_count_min': 4,
    }

    cfg_strict_bar = {
        **cfg_base, 'adaptive_quality_min': 40, 'breakout_bar_move_max': 2.0,
    }

    variants = [
        ('A baseline',    detect_breakout_A, cfg_base),
        ('E qual>=40',    detect_breakout_E, cfg_quality_40),
        ('F qual>=50',    detect_breakout_E, cfg_quality_50),
        ('G RR-by-qual',  detect_breakout_G, cfg_rr_only),
        ('H vol-conserv', detect_breakout_H, cfg_vol_conservative),
        ('I dyn-lookbk',  detect_breakout_I, cfg_lookback),
        ('J qual+lookbk', detect_breakout_J, cfg_quality_lookback),
        ('K qual+BB4/5',  detect_breakout_K, cfg_quality_tight_bb),
        ('L bar2%+qual',  detect_breakout_L, cfg_strict_bar),
    ]

    all_results = []

    for label, fn, cfg in variants:
        print(f"\n--- {label} ---")
        wf = walk_forward_variant(df, labels_short, fn, cfg, label)
        m = compute_full_metrics(wf)
        all_results.append({'label': label, 'wf': wf, 'oos': m})

        ok_s = 'OK' if wf['approved'] else 'NO'
        print(f"  WF {wf['folds_ok']}/12 {ok_s} | "
              f"N={m['n']:>3} WR={m['wr']:.1%} PF={m['pf']:.2f} | "
              f"LONG: N={m['long_n']} WR={m['long_wr']:.1%} PF={m['long_pf']:.2f} | "
              f"DD={m['max_dd']:.1%} ${m['equity_1k']:,.0f} | "
              f"fix={m['failed_fixed']}/4")

    # ================================================================
    # COMPARISON TABLE
    # ================================================================
    print("\n" + "=" * 110)
    print("COMPARISON (sorted by equity)")
    print("=" * 110)
    print(f"  {'Config':<15} | {'WF':>5} | {'N':>4} | {'WR':>6} | {'PF':>5} | "
          f"{'t/m':>4} | {'L_N':>4} | {'L_WR':>6} | {'L_PF':>5} | "
          f"{'DD':>6} | {'$1K->':>8} | {'Fix':>3}")
    print("  " + "-" * 102)

    sorted_results = sorted(all_results, key=lambda r: r['oos']['equity_1k'], reverse=True)
    for r in sorted_results:
        m = r['oos']
        wf = r['wf']
        is_baseline = 'baseline' in r['label']
        marker = ' ***' if m['equity_1k'] > 7116 else (' <<<' if is_baseline else '')
        print(f"  {r['label']:<15} | {wf['folds_ok']:>2}/12 | {m['n']:>4} | "
              f"{m['wr']:>5.1%} | {m['pf']:>5.2f} | {m['trades_pm']:>4.1f} | "
              f"{m['long_n']:>4} | {m['long_wr']:>5.1%} | {m['long_pf']:>5.2f} | "
              f"{m['max_dd']:>5.1%} | ${m['equity_1k']:>7,.0f} | "
              f"{m['failed_fixed']}/4{marker}")

    # ================================================================
    # FAILED FOLDS DETAIL
    # ================================================================
    print(f"\n{'='*110}")
    print("FAILED FOLDS DETAIL")
    print(f"{'='*110}")
    for fold_period in FAILED_FOLDS:
        print(f"\n  {fold_period}:")
        for r in all_results:
            fd = next((f for f in r['wf']['folds'] if f['period'] == fold_period), None)
            if fd:
                ok_s = '+' if fd['ok'] else '-'
                wr_s = f"{fd['wr']:.0%}" if fd['n'] > 0 else '-'
                pf_s = f"{fd['pf']:.2f}" if fd['n'] > 0 else '-'
                print(f"    {r['label']:<15} {ok_s} N={fd['n']:>3} "
                      f"L={fd['n_long']:>2} S={fd['n_short']:>2} "
                      f"WR={wr_s:>4} PF={pf_s:>5}")

    # ================================================================
    # QUALITY DISTRIBUTION (for variants with quality)
    # ================================================================
    print(f"\n{'='*110}")
    print("QUALITY SCORE DISTRIBUTION (LONG OOS trades)")
    print(f"{'='*110}")
    for r in all_results:
        oos_long = [t for t in r['wf']['all_trades']
                    if t['direction'] == 'LONG' and t.get('quality', -1) >= 0
                    and OOS_START <= str(t['ts'])[:10] <= OOS_END]
        if not oos_long:
            continue
        print(f"\n  {r['label']}:")
        for lo, hi, lbl in [(0, 30, ' 0-29'), (30, 40, '30-39'),
                             (40, 50, '40-49'), (50, 70, '50-69'), (70, 101, '  70+')]:
            q_t = [t for t in oos_long if lo <= t['quality'] < hi]
            if q_t:
                m_q = metrics(q_t, lbl)
                w = sum(1 for t in q_t if t['outcome'] == 'TP')
                l = sum(1 for t in q_t if t['outcome'] == 'SL')
                print(f"    {lbl}: N={m_q['n']:>3} W={w:>2} L={l:>2} "
                      f"WR={m_q['wr']:.0%} PF={m_q['pf']:.2f}")

    # ================================================================
    # ACCEPTANCE CRITERIA
    # ================================================================
    print(f"\n{'='*110}")
    print("ACCEPTANCE CRITERIA (must beat baseline A: PF>=1.35, WR>=48%, DD<=35%, $7.1K+)")
    print(f"{'='*110}")
    for r in all_results:
        m = r['oos']
        wf = r['wf']
        beats = (m['pf'] >= 1.35 and m['wr'] >= 0.48
                 and m['max_dd'] <= 0.35 and m['equity_1k'] >= 7116)
        improves = (m['equity_1k'] > 7116 or m['failed_fixed'] > 0
                    or (m['pf'] >= 1.35 and m['max_dd'] < 0.35))
        status = 'BEATS BASELINE' if beats else ('IMPROVES' if improves else 'no improvement')
        checks = []
        checks.append(f"PF={m['pf']:.2f}{'*' if m['pf']>=1.35 else ''}")
        checks.append(f"WR={m['wr']:.1%}{'*' if m['wr']>=0.48 else ''}")
        checks.append(f"DD={m['max_dd']:.1%}{'*' if m['max_dd']<=0.35 else ''}")
        checks.append(f"${m['equity_1k']:,.0f}{'*' if m['equity_1k']>=7116 else ''}")
        checks.append(f"fix={m['failed_fixed']}/4")
        print(f"  {r['label']:<15} {status:<16} | {' | '.join(checks)}")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
