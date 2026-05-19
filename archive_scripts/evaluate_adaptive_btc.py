"""
evaluate_adaptive_btc.py  --  BTC Adaptive Strategy Evaluation
==============================================================================
Compara 4 variantes del comite BTC V15:

  A (baseline) : Estatico actual (vol_min=1.8, lookback=20, RR=1.5)
  B            : A + adaptive vol threshold (BB_width percentil)
  C            : B + signal quality score (skip < 30, RR por tier)
  D            : C + dynamic lookback (proporcional a BB_width/mediana)

Criterios de aceptacion:
  - WF folds >= 8/12 (idealmente 9+)
  - OOS PF >= 1.30
  - Al menos 2 de los 4 folds fallidos mejoran
  - Trades/mes >= 5
  - Max DD no peor que 40%

Folds fallidos en baseline: 2021-H2, 2023-H1, 2025-H1, 2025-H2
"""

import sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from v15_framework import (
    load_btc_4h, compute_features_4h, compute_macro_daily, merge_daily_to_4h,
    load_funding, sim_trade_fixed, metrics, print_metrics,
    WF_FOLDS, OOS_START, OOS_END, COMMISSION
)
from backtest_v15_committee import (
    add_extra_features, detect_regime, detect_long_pullback,
    detect_short_ml, sim_short, add_funding_zscore,
    create_short_labels, SHORT_FEATURES,
    LONG_MAX_BARS, SHORT_MAX_BARS, SHORT_THRESHOLD,
    FUNDING_VETO_LONG, FUNDING_VETO_SHORT, CONSEC_LOSS_PAUSE,
)

# ============================================================
# ADAPTIVE FUNCTIONS (mirror what goes into ml_strategy_v15.py)
# ============================================================

def adaptive_vol_threshold(df, i, cfg):
    """Vol_ratio threshold relative to recent volatility (BB_width percentile 50 bars)."""
    window = cfg.get('adaptive_bb_window', 50)
    start = max(0, i - window)
    bb_series = df['bb_width'].iloc[start:i]
    if len(bb_series) < 10:
        return cfg.get('breakout_vol_min', 1.8)

    pctile = bb_series.rank(pct=True).iloc[-1] if len(bb_series) > 0 else 0.5

    vol_low = cfg.get('adaptive_vol_min_low', 1.3)
    vol_mid = cfg.get('adaptive_vol_min_mid', 1.8)
    vol_high = cfg.get('adaptive_vol_min_high', 2.2)

    if pctile < 0.30:
        return vol_low
    elif pctile > 0.70:
        return vol_high
    else:
        # Linear interpolation between low and high
        t = (pctile - 0.30) / 0.40
        return vol_low + t * (vol_high - vol_low)


def adaptive_bb_compression(df, i, cfg):
    """BB compression relative to median of last 50 bars (not fixed threshold)."""
    window = cfg.get('adaptive_bb_window', 50)
    start = max(0, i - window)
    bb_series = df['bb_width'].iloc[start:i]
    if len(bb_series) < 10:
        return True  # not enough data, allow

    median_bb = bb_series.median()

    # "Narrow" = below median (relative, not fixed 4.0)
    recent_bb = df['bb_width'].iloc[max(0, i-5):i]
    bb_count_min = cfg.get('adaptive_bb_count_min', 3)
    return (recent_bb < median_bb).sum() >= bb_count_min


def adaptive_lookback(df, i, cfg):
    """Lookback for high_N proportional to BB_width/median."""
    window = cfg.get('adaptive_bb_window', 50)
    start = max(0, i - window)
    bb_series = df['bb_width'].iloc[start:i]
    if len(bb_series) < 10:
        return 20  # default

    median_bb = bb_series.median()
    current_bb = float(df['bb_width'].iloc[i]) if i < len(df) else median_bb

    if median_bb <= 0:
        return 20

    ratio = current_bb / median_bb
    lb_min = cfg.get('adaptive_lookback_min', 12)
    lb_max = cfg.get('adaptive_lookback_max', 30)

    if ratio < 0.6:
        return lb_min
    elif ratio > 1.4:
        return lb_max
    else:
        # Linear interpolation
        t = (ratio - 0.6) / 0.8
        return int(lb_min + t * (lb_max - lb_min))


def compute_signal_quality(df, i, regime, cfg):
    """Score each breakout 0-100 with confluences."""
    row = df.iloc[i]
    score = 0

    # 1. BB compression (25 pts) — how narrow relative to recent
    window = cfg.get('adaptive_bb_window', 50)
    start = max(0, i - window)
    bb_series = df['bb_width'].iloc[start:i]
    if len(bb_series) >= 10:
        median_bb = bb_series.median()
        current_bb = float(df['bb_width'].iloc[max(0, i-1)])
        if median_bb > 0:
            ratio = current_bb / median_bb
            if ratio < 0.5:
                score += 25
            elif ratio < 0.7:
                score += 18
            elif ratio < 1.0:
                score += 10
            # ratio >= 1.0: 0 pts (not compressed)

    # 2. Vol spike strength (20 pts)
    vol_ratio = float(row.get('vol_ratio', 1.0))
    if vol_ratio >= 3.0:
        score += 20
    elif vol_ratio >= 2.5:
        score += 16
    elif vol_ratio >= 2.0:
        score += 12
    elif vol_ratio >= 1.5:
        score += 6
    # < 1.5: 0 pts

    # 3. DI+ crossover (15 pts)
    di_diff = float(row.get('di_diff', 0))
    if i >= 1:
        prev_di_diff = float(df.iloc[i-1].get('di_diff', 0))
        if di_diff > 0 and prev_di_diff <= 0:
            score += 15  # fresh crossover
        elif di_diff > 5:
            score += 8   # already positive and strong
        elif di_diff > 0:
            score += 4   # positive but weak

    # 4. Regime alignment (20 pts)
    if regime == 'BULL':
        score += 20
    elif regime == 'RANGE':
        score += 10
    # BEAR: 0 pts for LONG breakout

    # 5. EMA stack (10 pts): close > EMA20 > EMA50
    ema20 = float(row.get('ema20', 0))
    ema50 = float(row.get('ema50', 0))
    close = float(row['close'])
    if ema20 > 0 and ema50 > 0:
        if close > ema20 > ema50:
            score += 10
        elif close > ema50:
            score += 5

    # 6. RSI zone (10 pts): not overbought, not oversold
    rsi = float(row.get('rsi14', 50))
    if 45 <= rsi <= 65:
        score += 10  # sweet spot
    elif 35 <= rsi <= 75:
        score += 5   # acceptable
    # < 35 or > 75: 0 pts

    return min(score, 100)


# ============================================================
# BREAKOUT DETECTORS (4 variants)
# ============================================================

def detect_breakout_A(df, i, cfg):
    """Baseline: static breakout (current production)."""
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
    recent_bb = df['bb_width'].iloc[i-5:i]
    if (recent_bb < bb_max).sum() < 3:
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


def detect_breakout_B(df, i, cfg):
    """Variant B: adaptive vol threshold."""
    if i < 25:
        return None
    row = df.iloc[i]

    high20 = float(df['high'].iloc[i-20:i].max())
    if float(row['close']) <= high20:
        return None

    # ADAPTIVE: vol threshold based on BB percentile
    vol_min = adaptive_vol_threshold(df, i, cfg)
    if float(row.get('vol_ratio', 1)) < vol_min:
        return None

    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > cfg.get('breakout_bar_move_max', 2.5):
        return None

    # ADAPTIVE: BB compression relative to median
    if not adaptive_bb_compression(df, i, cfg):
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


def detect_breakout_C(df, i, cfg, regime='BULL'):
    """Variant C: B + signal quality score (skip < 30, RR by tier)."""
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

    if not adaptive_bb_compression(df, i, cfg):
        return None

    if df['adx14'].iloc[i-3:i].mean() > cfg.get('breakout_adx_max', 28):
        return None

    # SIGNAL QUALITY
    quality = compute_signal_quality(df, i, regime, cfg)
    quality_min = cfg.get('adaptive_quality_min', 30)
    if quality < quality_min:
        return None

    entry = float(row['close'])
    sl_raw = float(df['low'].iloc[i-5:i].min()) * 0.997
    sl_pct = (entry - sl_raw) / entry
    if sl_pct < 0.005 or sl_pct > 0.04:
        return None

    # RR by quality tier
    rr_low = cfg.get('adaptive_rr_low', 1.2)
    rr_mid = cfg.get('adaptive_rr_mid', 1.5)
    rr_high = cfg.get('adaptive_rr_high', 2.0)
    if quality >= 70:
        rr = rr_high
    elif quality >= 50:
        rr = rr_mid
    else:
        rr = rr_low

    tp_pct = sl_pct * rr

    return {'direction': 'LONG', 'setup': 'BREAKOUT_B',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct,
            'quality': quality}


def detect_breakout_D(df, i, cfg, regime='BULL'):
    """Variant D: C + dynamic lookback."""
    if i < 25:
        return None
    row = df.iloc[i]

    # DYNAMIC LOOKBACK
    lookback = adaptive_lookback(df, i, cfg)
    if i < lookback + 5:
        return None

    high_N = float(df['high'].iloc[i-lookback:i].max())
    if float(row['close']) <= high_N:
        return None

    vol_min = adaptive_vol_threshold(df, i, cfg)
    if float(row.get('vol_ratio', 1)) < vol_min:
        return None

    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > cfg.get('breakout_bar_move_max', 2.5):
        return None

    if not adaptive_bb_compression(df, i, cfg):
        return None

    if df['adx14'].iloc[i-3:i].mean() > cfg.get('breakout_adx_max', 28):
        return None

    # SIGNAL QUALITY
    quality = compute_signal_quality(df, i, regime, cfg)
    quality_min = cfg.get('adaptive_quality_min', 30)
    if quality < quality_min:
        return None

    entry = float(row['close'])
    sl_raw = float(df['low'].iloc[max(0, i-5):i].min()) * 0.997
    sl_pct = (entry - sl_raw) / entry
    if sl_pct < 0.005 or sl_pct > 0.04:
        return None

    rr_low = cfg.get('adaptive_rr_low', 1.2)
    rr_mid = cfg.get('adaptive_rr_mid', 1.5)
    rr_high = cfg.get('adaptive_rr_high', 2.0)
    if quality >= 70:
        rr = rr_high
    elif quality >= 50:
        rr = rr_mid
    else:
        rr = rr_low

    tp_pct = sl_pct * rr

    return {'direction': 'LONG', 'setup': 'BREAKOUT_B',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct,
            'quality': quality}


# ============================================================
# COMMITTEE RUNNER (parametric breakout detector)
# ============================================================

def run_committee_variant(df, short_model_data, start_idx, end_idx,
                          breakout_fn, cfg):
    """Run committee with a specific breakout detector variant."""
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

        # BULL -> breakout (variant) first, then pullback
        if regime == 'BULL':
            if funding_z > FUNDING_VETO_LONG:
                continue
            # Use variant-specific breakout
            if breakout_fn.__code__.co_varnames[:4] == ('df', 'i', 'cfg', 'regime'):
                trade = breakout_fn(df, i, cfg, regime)
            else:
                trade = breakout_fn(df, i, cfg)
            if trade is None:
                trade = detect_long_pullback(df, i)

        # BEAR -> SHORT ML (unchanged across variants)
        elif regime == 'BEAR':
            if funding_z < FUNDING_VETO_SHORT:
                continue
            trade = detect_short_ml(df, i, short_model_data)

        # RANGE -> breakout only (variant)
        elif regime == 'RANGE':
            if funding_z > FUNDING_VETO_LONG:
                continue
            if breakout_fn.__code__.co_varnames[:4] == ('df', 'i', 'cfg', 'regime'):
                trade = breakout_fn(df, i, cfg, regime)
            else:
                trade = breakout_fn(df, i, cfg)

        if trade is None:
            continue

        max_b = LONG_MAX_BARS if trade['direction'] == 'LONG' else SHORT_MAX_BARS
        if trade['direction'] == 'LONG':
            out = sim_trade_fixed(df, i, trade['entry'],
                                  trade['tp_pct'], trade['sl_pct'],
                                  max_bars=max_b)
        else:
            out = sim_short(df, i, trade['entry'],
                            trade['tp_pct'], trade['sl_pct'],
                            max_bars=max_b)

        if out[0] == 'SL':
            consec_losses += 1
            if consec_losses >= CONSEC_LOSS_PAUSE:
                paused = True
        else:
            consec_losses = 0

        trades.append({
            'outcome': out[0],
            'pnl_pct': out[2],
            'ts': df.index[i],
            'direction': trade['direction'],
            'setup': trade['setup'],
            'regime': regime,
            'funding_z': funding_z,
            'quality': trade.get('quality', -1),
        })

    return trades


# ============================================================
# WALK-FORWARD FOR VARIANT
# ============================================================

def walk_forward_variant(df, labels_short, breakout_fn, cfg, label=''):
    """WF with expanding SHORT model + specific breakout variant."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    results = []
    all_trades = []

    for fold_idx, (start_s, end_s) in enumerate(WF_FOLDS):
        test_mask = (df.index >= start_s) & (df.index <= end_s)
        train_mask = df.index < start_s

        period_label = f"{start_s[:7]}/{end_s[5:7]}"
        df_train = df[train_mask]

        # Train SHORT ML (same as baseline)
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
        trades = run_committee_variant(df, short_model_data, start_bar, end_bar,
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
# METRICS & COMPARISON
# ============================================================

FAILED_FOLDS = ['2021-07/12', '2023-01/06', '2025-01/06', '2025-07/12']


def compute_oos_metrics(all_trades):
    """Compute OOS metrics + equity curve + drawdown."""
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
    return m


def print_fold_table(wf, label):
    """Print fold-by-fold table."""
    print(f"\n  {'Periodo':<14} | {'N':>4} | {'L':>3} | {'S':>3} | "
          f"{'WR':>7} | {'PF':>6} | {'Anual':>8} | OK")
    print("  " + "-" * 65)
    for r in wf['folds']:
        ok_s = '+' if r['ok'] else '-'
        ann_s = f"{r['annual_pct']:.0f}%" if r['n'] > 0 else 'n/a'
        wr_s = f"{r['wr']:.1%}" if r['n'] > 0 else 'n/a'
        pf_s = f"{r['pf']:.2f}" if r['n'] > 0 else 'n/a'
        is_failed = r['period'] in FAILED_FOLDS
        marker = ' <<' if is_failed else ''
        print(f"  {r['period']:<14} | {r['n']:>4} | {r['n_long']:>3} | "
              f"{r['n_short']:>3} | {wr_s:>7} | {pf_s:>6} | "
              f"{ann_s:>8} | {ok_s}{marker}")


def print_comparison_table(results):
    """Final comparison table."""
    print("\n" + "=" * 90)
    print("COMPARISON TABLE")
    print("=" * 90)
    print(f"  {'Config':<12} | {'WF OK':>6} | {'OOS N':>5} | {'WR':>7} | "
          f"{'PF':>6} | {'t/m':>5} | {'MaxDD':>7} | {'$1K->':>8} | "
          f"{'Failed Fix':>10}")
    print("  " + "-" * 82)

    baseline_failed = set()
    for r in results:
        cfg_label = r['label']
        wf = r['wf']
        m = r['oos']

        # Check which of the 4 failed folds are now OK
        fixed = 0
        for fold in wf['folds']:
            if fold['period'] in FAILED_FOLDS and fold['ok']:
                fixed += 1
        if cfg_label == 'A (baseline)':
            for fold in wf['folds']:
                if fold['period'] in FAILED_FOLDS and not fold['ok']:
                    baseline_failed.add(fold['period'])

        print(f"  {cfg_label:<12} | {wf['folds_ok']:>2}/12  | {m['n']:>5} | "
              f"{m['wr']:>6.1%} | {m['pf']:>5.2f} | {m['trades_pm']:>4.1f} | "
              f"{m['max_dd']:>6.1%} | ${m['equity_1k']:>7,.0f} | "
              f"{fixed}/4")

    # Show which failed folds improved across variants
    print(f"\n  Failed folds detail (baseline failures marked):")
    for fold_period in FAILED_FOLDS:
        parts = [f"  {fold_period}: "]
        for r in results:
            fold_data = next((f for f in r['wf']['folds']
                              if f['period'] == fold_period), None)
            if fold_data:
                ok_s = '+' if fold_data['ok'] else '-'
                parts.append(
                    f"{r['label'][:1]}={ok_s}"
                    f"(N={fold_data['n']},WR={fold_data['wr']:.0%},PF={fold_data['pf']:.1f})"
                )
        print("  ".join(parts))


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("BTC ADAPTIVE STRATEGY EVALUATION")
    print("=" * 70)
    print("\nVariants:")
    print("  A: Baseline (static vol=1.8, lookback=20, RR=1.5)")
    print("  B: A + adaptive vol threshold (BB_width percentile)")
    print("  C: B + signal quality score (skip<30, RR by tier)")
    print("  D: C + dynamic lookback (proportional to BB_width/median)")

    # Load data
    print("\nLoading data...")
    df_raw = load_btc_4h()
    df = compute_features_4h(df_raw)
    df = add_extra_features(df)
    df_daily = compute_macro_daily(df)
    df = merge_daily_to_4h(df, df_daily)
    df = add_funding_zscore(df)
    print(f"  {len(df)} bars | {df.index[0].date()} - {df.index[-1].date()}")

    # Labels SHORT
    print("Creating SHORT labels...")
    labels_short = create_short_labels(df)

    # Configs
    cfg_base = {
        'breakout_vol_min': 1.8,
        'breakout_bb_max': 4.0,
        'breakout_adx_max': 28,
        'breakout_bar_move_max': 2.5,
        'breakout_rr': 1.5,
    }

    cfg_adaptive = {
        **cfg_base,
        'adaptive_bb_window': 50,
        'adaptive_vol_min_low': 1.3,
        'adaptive_vol_min_mid': 1.8,
        'adaptive_vol_min_high': 2.2,
        'adaptive_bb_count_min': 3,
        'adaptive_quality_min': 30,
        'adaptive_rr_low': 1.2,
        'adaptive_rr_mid': 1.5,
        'adaptive_rr_high': 2.0,
        'adaptive_lookback_min': 12,
        'adaptive_lookback_max': 30,
    }

    variants = [
        ('A (baseline)', detect_breakout_A, cfg_base),
        ('B (adap vol)', detect_breakout_B, cfg_adaptive),
        ('C (B+quality)', detect_breakout_C, cfg_adaptive),
        ('D (C+lookbk)', detect_breakout_D, cfg_adaptive),
    ]

    all_results = []

    for label, breakout_fn, cfg in variants:
        print(f"\n{'='*70}")
        print(f"VARIANT: {label}")
        print(f"{'='*70}")

        wf = walk_forward_variant(df, labels_short, breakout_fn, cfg, label)
        print_fold_table(wf, label)
        print(f"\n  Folds OK: {wf['folds_ok']}/12 | "
              f"{'APROBADO' if wf['approved'] else 'RECHAZADO'}")

        # OOS metrics
        m_oos = compute_oos_metrics(wf['all_trades'])
        print(f"\n  OOS: N={m_oos['n']} | WR={m_oos['wr']:.1%} | "
              f"PF={m_oos['pf']:.2f} | {m_oos['trades_pm']:.1f}t/m | "
              f"DD={m_oos['max_dd']:.1%} | $1K->${m_oos['equity_1k']:,.0f}")

        # Breakdown by direction
        oos_trades = [t for t in wf['all_trades']
                      if OOS_START <= str(t['ts'])[:10] <= OOS_END]
        long_t = [t for t in oos_trades if t['direction'] == 'LONG']
        short_t = [t for t in oos_trades if t['direction'] == 'SHORT']
        m_l = metrics(long_t, 'LONG')
        m_s = metrics(short_t, 'SHORT')
        print(f"    LONG:  N={m_l['n']:>3} WR={m_l['wr']:.1%} PF={m_l['pf']:.2f}")
        print(f"    SHORT: N={m_s['n']:>3} WR={m_s['wr']:.1%} PF={m_s['pf']:.2f}")

        # Quality distribution (for C and D)
        if label in ('C (B+quality)', 'D (C+lookbk)'):
            quals = [t['quality'] for t in oos_trades
                     if t['quality'] >= 0 and t['direction'] == 'LONG']
            if quals:
                print(f"\n  Quality distribution (LONG trades):")
                for lo, hi, lbl in [(30, 50, '30-49'), (50, 70, '50-69'), (70, 101, '70+')]:
                    q_trades = [t for t in oos_trades
                                if lo <= t.get('quality', -1) < hi
                                and t['direction'] == 'LONG']
                    if q_trades:
                        m_q = metrics(q_trades, lbl)
                        print(f"    Score {lbl}: N={m_q['n']:>3} "
                              f"WR={m_q['wr']:.1%} PF={m_q['pf']:.2f}")

        all_results.append({'label': label, 'wf': wf, 'oos': m_oos})

    # Final comparison
    print_comparison_table(all_results)

    # Acceptance criteria check
    print(f"\n{'='*70}")
    print("ACCEPTANCE CRITERIA CHECK")
    print(f"{'='*70}")
    for r in all_results:
        label = r['label']
        wf = r['wf']
        m = r['oos']
        fixed = sum(1 for f in wf['folds']
                    if f['period'] in FAILED_FOLDS and f['ok'])

        checks = {
            'WF >= 8/12': wf['folds_ok'] >= 8,
            'OOS PF >= 1.30': m['pf'] >= 1.30,
            'Fixed >= 2/4': fixed >= 2,
            'Trades/m >= 5': m['trades_pm'] >= 5,
            'Max DD <= 40%': m['max_dd'] <= 0.40,
        }
        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        all_pass = all(checks.values())

        status = 'PASS' if all_pass else 'PARTIAL'
        print(f"\n  {label}: {status} ({passed}/{total})")
        for check, ok in checks.items():
            print(f"    {'[+]' if ok else '[-]'} {check}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
