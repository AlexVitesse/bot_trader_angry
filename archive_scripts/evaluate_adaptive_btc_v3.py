"""
evaluate_adaptive_btc_v3.py  --  Round 3: F+G combo test
==============================================================================
Combinar las 2 mejores ideas de round 2:
  F: skip quality < 50
  G: RR por quality tier (70+ -> 2.0, 50-69 -> 1.5)

Dado que F ya filtra <50, la combo es: skip<50 + RR variable para los que quedan.
Tambien probar variaciones del RR tier para encontrar el optimo.
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
from evaluate_adaptive_btc import compute_signal_quality


# ============================================================
# DETECTORS
# ============================================================

def detect_breakout_baseline(df, i, cfg, regime='BULL'):
    """A: Baseline."""
    if i < 25:
        return None
    row = df.iloc[i]
    high20 = float(df['high'].iloc[i-20:i].max())
    if float(row['close']) <= high20:
        return None
    if float(row.get('vol_ratio', 1)) < 1.8:
        return None
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > 2.5:
        return None
    if (df['bb_width'].iloc[i-5:i] < 4.0).sum() < 3:
        return None
    if df['adx14'].iloc[i-3:i].mean() > 28:
        return None
    entry = float(row['close'])
    sl_raw = float(df['low'].iloc[i-5:i].min()) * 0.997
    sl_pct = (entry - sl_raw) / entry
    if sl_pct < 0.005 or sl_pct > 0.04:
        return None
    tp_pct = sl_pct * 1.5
    return {'direction': 'LONG', 'setup': 'BREAKOUT_B',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}


def make_combo_detector(quality_min, rr_tiers):
    """Factory: skip<quality_min + RR by quality tier."""
    def detect(df, i, cfg, regime='BULL'):
        if i < 25:
            return None
        row = df.iloc[i]
        high20 = float(df['high'].iloc[i-20:i].max())
        if float(row['close']) <= high20:
            return None
        if float(row.get('vol_ratio', 1)) < 1.8:
            return None
        bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
        if bar_move > 2.5:
            return None
        if (df['bb_width'].iloc[i-5:i] < 4.0).sum() < 3:
            return None
        if df['adx14'].iloc[i-3:i].mean() > 28:
            return None

        quality = compute_signal_quality(df, i, regime, cfg)
        if quality < quality_min:
            return None

        entry = float(row['close'])
        sl_raw = float(df['low'].iloc[i-5:i].min()) * 0.997
        sl_pct = (entry - sl_raw) / entry
        if sl_pct < 0.005 or sl_pct > 0.04:
            return None

        # RR by quality tier
        rr = rr_tiers[0]  # default (lowest tier)
        for threshold, tier_rr in rr_tiers[1:]:
            if quality >= threshold:
                rr = tier_rr
        tp_pct = sl_pct * rr
        return {'direction': 'LONG', 'setup': 'BREAKOUT_B',
                'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct,
                'quality': quality}
    return detect


# ============================================================
# COMMITTEE + WF (same as v2)
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
            'regime': regime, 'quality': trade.get('quality', -1),
        })
    return trades


def walk_forward_variant(df, labels_short, breakout_fn, cfg):
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
                min_samples_leaf=20, subsample=0.8, random_state=42)
            model_s.fit(X_ss, y_short_fit)
            short_model_data = {'model': model_s, 'scaler': scaler_s}
        df_test = df[test_mask]
        if len(df_test) == 0:
            results.append({'period': period_label, 'n': 0, 'wr': 0,
                            'pf': 0, 'ok': False, 'n_long': 0, 'n_short': 0})
            continue
        start_bar = df.index.get_loc(df_test.index[0])
        end_bar = df.index.get_loc(df_test.index[-1]) + 1
        trades = run_committee(df, short_model_data, start_bar, end_bar, breakout_fn, cfg)
        m = metrics(trades, period_label)
        n_long = sum(1 for t in trades if t['direction'] == 'LONG')
        n_short = sum(1 for t in trades if t['direction'] == 'SHORT')
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        results.append({'period': period_label, 'n': m['n'], 'wr': m['wr'],
                        'pf': m['pf'], 'ok': ok, 'n_long': n_long, 'n_short': n_short})
        all_trades.extend(trades)
    folds_ok = sum(1 for r in results if r['ok'])
    return {'folds': results, 'folds_ok': folds_ok, 'all_trades': all_trades}


# ============================================================
# METRICS
# ============================================================

FAILED_FOLDS = ['2021-07/12', '2023-01/06', '2025-01/06', '2025-07/12']

def compute_full_metrics(wf):
    all_trades = wf['all_trades']
    oos_trades = [t for t in all_trades if OOS_START <= str(t['ts'])[:10] <= OOS_END]
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
    long_oos = [t for t in oos_trades if t['direction'] == 'LONG']
    m_l = metrics(long_oos, 'LONG')
    m['long_n'] = m_l['n']
    m['long_wr'] = m_l['wr']
    m['long_pf'] = m_l['pf']
    m['failed_fixed'] = sum(1 for f in wf['folds'] if f['period'] in FAILED_FOLDS and f['ok'])
    return m


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 90)
    print("BTC ADAPTIVE — ROUND 3: F+G COMBO (skip<50 + RR by quality)")
    print("=" * 90)

    print("\nLoading data...")
    df_raw = load_btc_4h()
    df = compute_features_4h(df_raw)
    df = add_extra_features(df)
    df_daily = compute_macro_daily(df)
    df = merge_daily_to_4h(df, df_daily)
    df = add_funding_zscore(df)
    print(f"  {len(df)} bars | {df.index[0].date()} - {df.index[-1].date()}")
    labels_short = create_short_labels(df)

    cfg = {'adaptive_bb_window': 50}

    # Define RR tier formats: (default_rr, (threshold, rr), (threshold, rr), ...)
    # Applied highest-match-wins (checked in order, last match wins)
    variants = [
        ('A baseline',           detect_breakout_baseline,
         None),
        # F alone (reference)
        ('F skip<50 RR=1.5',     make_combo_detector(50, [1.5]),
         None),
        # G alone (reference)
        ('G RR-only',            make_combo_detector(0, [1.0, (30, 1.2), (50, 1.5), (70, 2.0)]),
         None),
        # F+G: the combo
        ('FG skip50+RR',         make_combo_detector(50, [1.5, (70, 2.0)]),
         'COMBO base'),
        # Variations of the combo RR tiers
        ('FG s50 50:1.3/70:1.8', make_combo_detector(50, [1.3, (70, 1.8)]),
         'tighter RR'),
        ('FG s50 50:1.5/70:2.5', make_combo_detector(50, [1.5, (70, 2.5)]),
         'wider 70+'),
        ('FG s50 50:1.2/60:1.5/70:2.0', make_combo_detector(50, [1.2, (60, 1.5), (70, 2.0)]),
         '3-tier'),
        # Also test skip<45 (slightly more permissive)
        ('FG s45+RR',            make_combo_detector(45, [1.3, (60, 1.5), (70, 2.0)]),
         'permissive'),
        # And skip<55 (slightly stricter)
        ('FG s55+RR',            make_combo_detector(55, [1.5, (70, 2.0)]),
         'stricter'),
    ]

    all_results = []
    for label, fn, note in variants:
        wf = walk_forward_variant(df, labels_short, fn, cfg)
        m = compute_full_metrics(wf)
        all_results.append({'label': label, 'wf': wf, 'oos': m, 'note': note or ''})

    # ================================================================
    # COMPARISON TABLE
    # ================================================================
    print(f"\n{'='*105}")
    print("COMPARISON (sorted by equity)")
    print(f"{'='*105}")
    print(f"  {'Config':<25} | {'WF':>5} | {'N':>4} | {'WR':>6} | {'PF':>5} | "
          f"{'t/m':>4} | {'L_WR':>5} | {'L_PF':>5} | {'DD':>6} | {'$1K->':>8} | Note")
    print("  " + "-" * 97)

    sorted_r = sorted(all_results, key=lambda r: r['oos']['equity_1k'], reverse=True)
    for r in sorted_r:
        m = r['oos']
        wf = r['wf']
        is_base = 'baseline' in r['label']
        marker = ' <<<' if is_base else (' ***' if m['equity_1k'] > 7116 else '')
        print(f"  {r['label']:<25} | {wf['folds_ok']:>2}/12 | {m['n']:>4} | "
              f"{m['wr']:>5.1%} | {m['pf']:>5.2f} | {m['trades_pm']:>4.1f} | "
              f"{m['long_wr']:>4.1%} | {m['long_pf']:>5.2f} | "
              f"{m['max_dd']:>5.1%} | ${m['equity_1k']:>7,.0f} | "
              f"{r['note']}{marker}")

    # ================================================================
    # FAILED FOLDS — focus on 2025-H2 (closest to fixing)
    # ================================================================
    print(f"\n{'='*105}")
    print("FAILED FOLDS DETAIL")
    print(f"{'='*105}")
    for fold_period in FAILED_FOLDS:
        print(f"\n  {fold_period}:")
        for r in all_results:
            fd = next((f for f in r['wf']['folds'] if f['period'] == fold_period), None)
            if fd:
                ok_s = '+' if fd['ok'] else '-'
                wr_s = f"{fd['wr']:.0%}" if fd['n'] > 0 else '-'
                pf_s = f"{fd['pf']:.2f}" if fd['n'] > 0 else '-'
                print(f"    {r['label']:<25} {ok_s} N={fd['n']:>3} "
                      f"L={fd['n_long']:>2} S={fd['n_short']:>2} "
                      f"WR={wr_s:>4} PF={pf_s:>5}")

    # ================================================================
    # QUALITY DISTRIBUTION for combo variants
    # ================================================================
    print(f"\n{'='*105}")
    print("QUALITY SCORE DISTRIBUTION (LONG OOS)")
    print(f"{'='*105}")
    for r in all_results:
        oos_long = [t for t in r['wf']['all_trades']
                    if t['direction'] == 'LONG' and t.get('quality', -1) >= 0
                    and OOS_START <= str(t['ts'])[:10] <= OOS_END]
        if not oos_long:
            continue
        print(f"\n  {r['label']}:")
        for lo, hi, lbl in [(0, 30, ' 0-29'), (30, 45, '30-44'),
                             (45, 50, '45-49'), (50, 60, '50-59'),
                             (60, 70, '60-69'), (70, 101, '  70+')]:
            q_t = [t for t in oos_long if lo <= t['quality'] < hi]
            if q_t:
                m_q = metrics(q_t, lbl)
                w = sum(1 for t in q_t if t['outcome'] == 'TP')
                l = sum(1 for t in q_t if t['outcome'] == 'SL')
                # Compute avg RR for this tier
                avg_tp = np.mean([t['pnl_pct'] for t in q_t if t['outcome'] == 'TP']) if w > 0 else 0
                avg_sl = np.mean([abs(t['pnl_pct']) for t in q_t if t['outcome'] == 'SL']) if l > 0 else 0
                print(f"    {lbl}: N={m_q['n']:>3} W={w:>2} L={l:>2} "
                      f"WR={m_q['wr']:.0%} PF={m_q['pf']:.2f} "
                      f"avgW={avg_tp*100:+.2f}% avgL={-avg_sl*100:.2f}%")

    # ================================================================
    # VERDICT
    # ================================================================
    print(f"\n{'='*105}")
    print("VERDICT vs BASELINE (A: PF=1.35, WR=48.0%, DD=35.0%, $7,116)")
    print(f"{'='*105}")
    base_eq = 7116
    for r in sorted_r:
        m = r['oos']
        wf = r['wf']
        delta_eq = m['equity_1k'] - base_eq
        delta_dd = m['max_dd'] - 0.35
        delta_pf = m['pf'] - 1.35
        status = []
        if delta_eq > 0:
            status.append(f"equity+${delta_eq:.0f}")
        if delta_dd < 0:
            status.append(f"DD{delta_dd*100:+.1f}pp")
        if delta_pf > 0:
            status.append(f"PF{delta_pf:+.2f}")
        verdict = ', '.join(status) if status else 'no improvement'
        print(f"  {r['label']:<25} | {verdict}")

    print(f"\n{'='*90}")
    print("DONE")
    print(f"{'='*90}")


if __name__ == '__main__':
    main()
