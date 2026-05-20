"""
evaluate_short_filters.py — SHORT ML + trend confirmation filters
=================================================================
The SHORT ML model has 82% WR OOS overall but 0% in 2025-H2.
Problem: model shorts in oversold + bulls still leading (DI+>DI-).

Test filters to confirm bearish trend before shorting:
  1. DI_diff < 0 (DI- > DI+, bears leading)
  2. EMA20_4h < EMA50_4h (4H death cross)
  3. EMA20 slope < 0 (EMA20 declining)
  4. Price < EMA20 (below short-term trend)
  5. Price < EMA50 (below mid-term trend)
  6. EMA20 < EMA50 + price < EMA20 (full bear alignment)
  7. DI_diff < 0 + EMA20 < EMA50 (double confirmation)
  8. Bearish candle + DI_diff < 0
  9. EMA20_slope < -X (steeper decline)

Full walk-forward to see real impact, not just cherry-picked periods.
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
    sim_trade_fixed, metrics, WF_FOLDS, OOS_START, OOS_END, COMMISSION
)
from backtest_v15_committee import (
    add_extra_features, detect_regime,
    sim_short, add_funding_zscore,
    create_short_labels, SHORT_FEATURES,
    SHORT_MAX_BARS, FUNDING_VETO_SHORT, CONSEC_LOSS_PAUSE,
)
# Also import LONG components for full committee
from backtest_v15_committee import (
    detect_long_pullback, LONG_MAX_BARS, FUNDING_VETO_LONG,
)
from evaluate_adaptive_btc import compute_signal_quality
from evaluate_adaptive_btc_v3 import make_combo_detector

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================
# SHORT ML DETECTOR WITH FILTERS
# ============================================================

def detect_short_ml_filtered(df, i, model_data, short_filter_fn=None):
    """SHORT ML detection with optional rule-based filter on top."""
    if model_data is None or i < 30:
        return None
    row = df.iloc[i]
    model = model_data['model']
    scaler = model_data['scaler']

    x = pd.DataFrame([row[SHORT_FEATURES].fillna(0).values], columns=SHORT_FEATURES)
    x_s = scaler.transform(x)
    prob = model.predict_proba(x_s)[0][1]

    if prob < 0.60:
        return None

    # Apply filter AFTER ML says "short"
    if short_filter_fn is not None and not short_filter_fn(df, i):
        return None

    entry = float(row['close'])
    sl_raw = float(df['high'].iloc[max(0, i-3):i+1].max()) * 1.003
    sl_pct = (sl_raw - entry) / entry
    sl_pct = min(max(sl_pct, 0.015), 0.04)
    tp_pct = sl_pct * 1.67

    return {'direction': 'SHORT', 'setup': 'ML_SHORT',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct,
            'prob': prob}


# ============================================================
# FILTER FUNCTIONS
# ============================================================

def filt_di_neg(df, i):
    """DI_diff < 0: bears leading."""
    return float(df.iloc[i].get('di_diff', 0)) < 0

def filt_di_neg5(df, i):
    """DI_diff < -5: bears clearly leading."""
    return float(df.iloc[i].get('di_diff', 0)) < -5

def filt_ema_cross(df, i):
    """EMA20 < EMA50 on 4H (death cross)."""
    row = df.iloc[i]
    ema20 = float(row.get('ema20', 0))
    ema50 = float(row.get('ema50', 0))
    return ema20 > 0 and ema50 > 0 and ema20 < ema50

def filt_ema20_slope_neg(df, i):
    """EMA20 slope < 0 (declining)."""
    return float(df.iloc[i].get('ema20_slope', 0)) < 0

def filt_ema20_slope_neg05(df, i):
    """EMA20 slope < -0.5 (clearly declining)."""
    return float(df.iloc[i].get('ema20_slope', 0)) < -0.5

def filt_price_below_ema20(df, i):
    """Price < EMA20."""
    row = df.iloc[i]
    ema20 = float(row.get('ema20', 0))
    return ema20 > 0 and float(row['close']) < ema20

def filt_price_below_ema50(df, i):
    """Price < EMA50."""
    row = df.iloc[i]
    ema50 = float(row.get('ema50', 0))
    return ema50 > 0 and float(row['close']) < ema50

def filt_full_bear_align(df, i):
    """Price < EMA20 < EMA50 (full bearish alignment on 4H)."""
    row = df.iloc[i]
    ema20 = float(row.get('ema20', 0))
    ema50 = float(row.get('ema50', 0))
    c = float(row['close'])
    return ema20 > 0 and ema50 > 0 and c < ema20 < ema50

def filt_di_neg_and_ema_cross(df, i):
    """DI_diff < 0 + EMA20 < EMA50 (double confirmation)."""
    return filt_di_neg(df, i) and filt_ema_cross(df, i)

def filt_bearish_candle_di_neg(df, i):
    """Bearish candle + DI_diff < 0."""
    row = df.iloc[i]
    bearish = float(row['close']) < float(row['open'])
    return bearish and filt_di_neg(df, i)

def filt_ema_cross_slope(df, i):
    """EMA20 < EMA50 + EMA20 slope < 0."""
    return filt_ema_cross(df, i) and filt_ema20_slope_neg(df, i)

def filt_price_ema20_di(df, i):
    """Price < EMA20 + DI_diff < 0."""
    return filt_price_below_ema20(df, i) and filt_di_neg(df, i)


# ============================================================
# COMMITTEE WITH FILTERED SHORT + FG LONG
# ============================================================

def run_committee(df, short_model_data, start_idx, end_idx,
                  breakout_fn, cfg, short_filter_fn=None):
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
            trade = detect_short_ml_filtered(df, i, short_model_data, short_filter_fn)
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
            'regime': regime,
        })
    return trades


def walk_forward(df, labels_short, breakout_fn, cfg, short_filter_fn=None):
    results = []
    all_trades = []
    for fold_idx, (start_s, end_s) in enumerate(WF_FOLDS):
        test_mask = (df.index >= start_s) & (df.index <= end_s)
        train_mask = df.index < start_s
        period = f"{start_s[:7]}/{end_s[5:7]}"
        df_train = df[train_mask]

        # Train SHORT ML
        y_tr = labels_short[train_mask]
        bear_tr = df_train.get('bull_1d', pd.Series(1, index=df_train.index)) == 0
        valid = y_tr.notna() & bear_tr
        df_short_tr = df_train[valid]
        y_short_fit = y_tr[valid]
        short_model_data = None
        if (len(df_short_tr) >= 800 and y_short_fit.sum() >= 20
                and (len(y_short_fit) - y_short_fit.sum()) >= 20):
            X = df_short_tr[SHORT_FEATURES].fillna(0)
            sc = StandardScaler()
            Xs = sc.fit_transform(X)
            mdl = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                min_samples_leaf=20, subsample=0.8, random_state=42)
            mdl.fit(Xs, y_short_fit)
            short_model_data = {'model': mdl, 'scaler': sc}

        df_test = df[test_mask]
        if len(df_test) == 0:
            results.append({'period': period, 'n': 0, 'wr': 0, 'pf': 0,
                            'ok': False, 'n_long': 0, 'n_short': 0})
            continue
        start_bar = df.index.get_loc(df_test.index[0])
        end_bar = df.index.get_loc(df_test.index[-1]) + 1
        trades = run_committee(df, short_model_data, start_bar, end_bar,
                               breakout_fn, cfg, short_filter_fn)
        m = metrics(trades, period)
        nl = sum(1 for t in trades if t['direction'] == 'LONG')
        ns = sum(1 for t in trades if t['direction'] == 'SHORT')
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                        'pf': m['pf'], 'ok': ok, 'n_long': nl, 'n_short': ns})
        all_trades.extend(trades)

    folds_ok = sum(1 for r in results if r['ok'])
    return {'folds': results, 'folds_ok': folds_ok, 'all_trades': all_trades}


FAILED_FOLDS = ['2021-07/12', '2023-01/06', '2025-01/06', '2025-07/12']

def compute_metrics(wf):
    all_t = wf['all_trades']
    oos_t = [t for t in all_t if OOS_START <= str(t['ts'])[:10] <= OOS_END]
    m = metrics(oos_t, 'OOS')
    cum = 1.0; peak = 1.0; max_dd = 0.0
    for t in sorted(all_t, key=lambda x: x['ts']):
        cum *= (1 + t['pnl_pct'])
        peak = max(peak, cum)
        dd = (peak - cum) / peak
        max_dd = max(max_dd, dd)
    m['max_dd'] = max_dd
    m['equity_1k'] = 1000 * cum

    # LONG/SHORT breakdown
    long_oos = [t for t in oos_t if t['direction'] == 'LONG']
    short_oos = [t for t in oos_t if t['direction'] == 'SHORT']
    ml = metrics(long_oos, 'L')
    ms = metrics(short_oos, 'S')
    m['long_n'] = ml['n']; m['long_wr'] = ml['wr']; m['long_pf'] = ml['pf']
    m['short_n'] = ms['n']; m['short_wr'] = ms['wr']; m['short_pf'] = ms['pf']
    m['failed_fixed'] = sum(1 for f in wf['folds']
                            if f['period'] in FAILED_FOLDS and f['ok'])
    return m


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 100)
    print("SHORT ML FILTER EVALUATION — EMA crossovers + DI + combinations")
    print("=" * 100)

    print("\nLoading data...")
    df_raw = load_btc_4h()
    df = compute_features_4h(df_raw)
    df = add_extra_features(df)
    df_daily = compute_macro_daily(df)
    df = merge_daily_to_4h(df, df_daily)
    df = add_funding_zscore(df)
    labels_short = create_short_labels(df)
    print(f"  {len(df)} bars | {df.index[0].date()} - {df.index[-1].date()}")

    # Use FG combo (best LONG) as the breakout detector
    breakout_fn = make_combo_detector(50, [1.5, (70, 2.0)])
    cfg = {'adaptive_bb_window': 50}

    # Define filter variants
    filters = [
        ('no filter (FG base)',     None),
        ('DI_diff < 0',            filt_di_neg),
        ('DI_diff < -5',           filt_di_neg5),
        ('EMA20 < EMA50',          filt_ema_cross),
        ('EMA20 slope < 0',        filt_ema20_slope_neg),
        ('EMA20 slope < -0.5',     filt_ema20_slope_neg05),
        ('price < EMA20',          filt_price_below_ema20),
        ('price < EMA50',          filt_price_below_ema50),
        ('P<EMA20<EMA50 (align)',  filt_full_bear_align),
        ('DI<0 + EMA20<EMA50',    filt_di_neg_and_ema_cross),
        ('bear candle + DI<0',     filt_bearish_candle_di_neg),
        ('EMA20<50 + slope<0',    filt_ema_cross_slope),
        ('P<EMA20 + DI<0',        filt_price_ema20_di),
    ]

    all_results = []
    for label, filt_fn in filters:
        print(f"  Running: {label}...")
        wf = walk_forward(df, labels_short, breakout_fn, cfg, filt_fn)
        m = compute_metrics(wf)
        all_results.append({'label': label, 'wf': wf, 'oos': m})

    # ================================================================
    # COMPARISON TABLE
    # ================================================================
    print(f"\n{'='*120}")
    print("COMPARISON TABLE (sorted by equity)")
    print(f"{'='*120}")
    print(f"  {'Filter':<27} | {'WF':>5} | {'N':>4} | {'WR':>6} | {'PF':>5} | "
          f"{'S_N':>3} | {'S_WR':>5} | {'S_PF':>5} | "
          f"{'L_N':>3} | {'L_WR':>5} | {'L_PF':>5} | "
          f"{'DD':>6} | {'$1K->':>8} | {'Fix':>3}")
    print("  " + "-" * 112)

    sorted_r = sorted(all_results, key=lambda r: r['oos']['equity_1k'], reverse=True)
    base_eq = all_results[0]['oos']['equity_1k']  # FG base equity
    for r in sorted_r:
        m = r['oos']
        wf = r['wf']
        is_base = 'no filter' in r['label']
        delta = m['equity_1k'] - base_eq
        marker = f' ({delta:+,.0f})' if not is_base else ' (ref)'
        print(f"  {r['label']:<27} | {wf['folds_ok']:>2}/12 | {m['n']:>4} | "
              f"{m['wr']:>5.1%} | {m['pf']:>5.2f} | "
              f"{m['short_n']:>3} | {m['short_wr']:>4.0%} | {m['short_pf']:>5.2f} | "
              f"{m['long_n']:>3} | {m['long_wr']:>4.0%} | {m['long_pf']:>5.2f} | "
              f"{m['max_dd']:>5.1%} | ${m['equity_1k']:>7,.0f} | "
              f"{m['failed_fixed']}/4{marker}")

    # ================================================================
    # FAILED FOLDS DETAIL
    # ================================================================
    print(f"\n{'='*120}")
    print("FAILED FOLDS DETAIL (focus on SHORT-affected folds)")
    print(f"{'='*120}")
    for fold_period in FAILED_FOLDS:
        print(f"\n  {fold_period}:")
        for r in all_results:
            fd = next((f for f in r['wf']['folds'] if f['period'] == fold_period), None)
            if fd:
                ok_s = '+' if fd['ok'] else '-'
                wr_s = f"{fd['wr']:.0%}" if fd['n'] > 0 else '-'
                pf_s = f"{fd['pf']:.2f}" if fd['n'] > 0 else '-'
                print(f"    {r['label']:<27} {ok_s} N={fd['n']:>3} "
                      f"L={fd['n_long']:>2} S={fd['n_short']:>2} "
                      f"WR={wr_s:>4} PF={pf_s:>5}")

    # ================================================================
    # BEST CONFIGS: FOLD-BY-FOLD TABLE
    # ================================================================
    # Show top 3 + baseline
    top_labels = [sorted_r[0]['label'], sorted_r[1]['label'], sorted_r[2]['label']]
    if 'no filter' not in top_labels:
        top_labels.append('no filter (FG base)')
    top_results = [r for r in all_results if r['label'] in top_labels]

    for r in top_results:
        label = r['label']
        wf = r['wf']
        m = r['oos']
        print(f"\n{'='*80}")
        print(f"FOLD-BY-FOLD: {label}")
        print(f"  OOS: N={m['n']} WR={m['wr']:.1%} PF={m['pf']:.2f} "
              f"DD={m['max_dd']:.1%} ${m['equity_1k']:,.0f}")
        print(f"  SHORT: N={m['short_n']} WR={m['short_wr']:.1%} PF={m['short_pf']:.2f}")
        print(f"{'='*80}")
        print(f"  {'Period':<14} | {'N':>4} | {'L':>3} | {'S':>3} | "
              f"{'WR':>6} | {'PF':>6} | OK")
        print("  " + "-" * 50)
        for f in wf['folds']:
            ok_s = '+' if f['ok'] else '-'
            wr_s = f"{f['wr']:.0%}" if f['n'] > 0 else '-'
            pf_s = f"{f['pf']:.2f}" if f['n'] > 0 else '-'
            fail = ' <<' if f['period'] in FAILED_FOLDS else ''
            print(f"  {f['period']:<14} | {f['n']:>4} | {f['n_long']:>3} | "
                  f"{f['n_short']:>3} | {wr_s:>6} | {pf_s:>6} | {ok_s}{fail}")

    print(f"\n{'='*100}")
    print("DONE")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
