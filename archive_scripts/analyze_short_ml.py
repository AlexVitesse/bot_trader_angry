"""
analyze_short_ml.py — Deep analysis of SHORT ML performance
Goal: understand why 0% WR in 2025-H2, find filters to improve.
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
    sim_trade_fixed, metrics, WF_FOLDS, COMMISSION
)
from backtest_v15_committee import (
    add_extra_features, detect_regime, detect_short_ml,
    sim_short, add_funding_zscore,
    create_short_labels, SHORT_FEATURES,
    SHORT_MAX_BARS, FUNDING_VETO_SHORT,
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

print('Loading...')
df_raw = load_btc_4h()
df = compute_features_4h(df_raw)
df = add_extra_features(df)
df_daily = compute_macro_daily(df)
df = merge_daily_to_4h(df, df_daily)
df = add_funding_zscore(df)
labels_short = create_short_labels(df)

# Train SHORT model with all data before 2025-H2
train_mask = df.index < '2025-07-01'
df_train = df[train_mask]
y_tr = labels_short[train_mask]
bear_tr = df_train.get('bull_1d', pd.Series(1, index=df_train.index)) == 0
valid = y_tr.notna() & bear_tr
X = df_train[valid][SHORT_FEATURES].fillna(0)
y = y_tr[valid]
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
model = GradientBoostingClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    min_samples_leaf=20, subsample=0.8, random_state=42)
model.fit(Xs, y)
short_data = {'model': model, 'scaler': scaler}
print(f'SHORT model: {len(X)} samples, pos_rate={y.mean():.1%}')

# ============================================================
# SHORT TRADES BY SEMESTER
# ============================================================
print(f'\n{"="*80}')
print('SHORT ML TRADES BY SEMESTER')
print(f'{"="*80}')

for start_s, end_s in WF_FOLDS:
    mask = (df.index >= start_s) & (df.index <= end_s)
    df_f = df[mask]
    trades = []
    for idx in range(len(df_f)):
        row = df_f.iloc[idx]
        regime = detect_regime(row)
        if regime != 'BEAR':
            continue
        gi = df.index.get_loc(df_f.index[idx])
        trade = detect_short_ml(df, gi, short_data)
        if trade is None:
            continue
        out = sim_short(df, gi, trade['entry'], trade['tp_pct'], trade['sl_pct'], SHORT_MAX_BARS)
        trades.append({
            'outcome': out[0], 'pnl_pct': out[2], 'ts': df_f.index[idx],
            'prob': trade.get('prob', 0),
        })

    if not trades:
        bear_bars = sum(1 for idx in range(len(df_f))
                        if detect_regime(df_f.iloc[idx]) == 'BEAR')
        print(f'  {start_s[:7]}/{end_s[5:7]}: 0 trades ({bear_bars} BEAR bars)')
        continue

    m = metrics(trades, '')
    wins = sum(1 for t in trades if t['outcome'] == 'TP')
    avg_prob = np.mean([t['prob'] for t in trades])
    ok = '+' if m['pf'] > 1.0 and m['wr'] > 0.38 else '-'
    print(f'  {start_s[:7]}/{end_s[5:7]}: {ok} N={m["n"]:>3} W={wins} '
          f'WR={m["wr"]:.0%} PF={m["pf"]:.2f} | avgProb={avg_prob:.2f}')

# ============================================================
# 2025-H2 TRADE-BY-TRADE
# ============================================================
print(f'\n{"="*80}')
print('2025-H2 SHORT TRADES (trade-by-trade)')
print(f'{"="*80}')

mask_h2 = (df.index >= '2025-07-01') & (df.index <= '2025-12-31')
df_h2 = df[mask_h2]
h2_signals = []
for idx in range(len(df_h2)):
    row = df_h2.iloc[idx]
    regime = detect_regime(row)
    if regime != 'BEAR':
        continue
    gi = df.index.get_loc(df_h2.index[idx])
    trade = detect_short_ml(df, gi, short_data)
    if trade is None:
        continue
    out = sim_short(df, gi, trade['entry'], trade['tp_pct'], trade['sl_pct'], SHORT_MAX_BARS)

    rsi = float(row.get('rsi14', 0))
    bb_pct = float(row.get('bb_pct', 0))
    vol = float(row.get('vol_ratio', 1))
    adx = float(row.get('adx14', 0))
    di_diff = float(row.get('di_diff', 0))
    ema200_d = float(row.get('ema200_dist', 0))
    bearish = float(row['close']) < float(row['open'])

    print(f'  {str(df_h2.index[idx])[:10]} | prob={trade["prob"]:.2f} | '
          f'{out[0]} {out[2]*100:+.2f}% ({out[3]}bars) | '
          f'RSI={rsi:.0f} BB={bb_pct:.2f} vol={vol:.1f} ADX={adx:.0f} '
          f'DI={di_diff:+.0f} bear_candle={"Y" if bearish else "N"}')

    h2_signals.append({
        'row': row, 'outcome': out[0], 'pnl_pct': out[2],
        'ts': df_h2.index[idx], 'prob': trade['prob'],
    })

# ============================================================
# THRESHOLD SENSITIVITY (2025-H2)
# ============================================================
print(f'\n{"="*80}')
print('THRESHOLD SENSITIVITY (2025-H2 SHORT only)')
print(f'{"="*80}')
for thr in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    trades = []
    for idx in range(len(df_h2)):
        row = df_h2.iloc[idx]
        regime = detect_regime(row)
        if regime != 'BEAR':
            continue
        gi = df.index.get_loc(df_h2.index[idx])
        if gi < 30:
            continue
        r = df.iloc[gi]
        x = pd.DataFrame([r[SHORT_FEATURES].fillna(0).values], columns=SHORT_FEATURES)
        x_s = scaler.transform(x)
        prob = model.predict_proba(x_s)[0][1]
        if prob < thr:
            continue
        entry = float(r['close'])
        sl_raw = float(df['high'].iloc[max(0, gi-3):gi+1].max()) * 1.003
        sl_pct = (sl_raw - entry) / entry
        sl_pct = min(max(sl_pct, 0.015), 0.04)
        tp_pct = sl_pct * 1.67
        out = sim_short(df, gi, entry, tp_pct, sl_pct, SHORT_MAX_BARS)
        trades.append({'outcome': out[0], 'pnl_pct': out[2], 'prob': prob,
                       'ts': df_h2.index[idx]})

    if trades:
        m = metrics(trades, '')
        wins = sum(1 for t in trades if t['outcome'] == 'TP')
        print(f'  thr={thr:.2f}: N={len(trades):>2} W={wins} '
              f'WR={m["wr"]:.0%} PF={m["pf"]:.2f}')
    else:
        print(f'  thr={thr:.2f}: 0 trades')

# ============================================================
# RULE-BASED FILTERS ON TOP OF ML — 2025-H2
# ============================================================
print(f'\n{"="*80}')
print('RULE-BASED SHORT FILTERS ON TOP OF ML (2025-H2)')
print(f'{"="*80}')

filters = {
    'RSI>55': lambda r: float(r.get('rsi14', 50)) > 55,
    'RSI>60': lambda r: float(r.get('rsi14', 50)) > 60,
    'BB>0.60': lambda r: float(r.get('bb_pct', 0.5)) > 0.60,
    'BB>0.70': lambda r: float(r.get('bb_pct', 0.5)) > 0.70,
    'bearish candle': lambda r: float(r['close']) < float(r['open']),
    'ADX>20': lambda r: float(r.get('adx14', 0)) > 20,
    'vol>1.2': lambda r: float(r.get('vol_ratio', 1)) > 1.2,
    'DI_diff<0': lambda r: float(r.get('di_diff', 0)) < 0,
    'ema200d<-5%': lambda r: float(r.get('ema200_dist', 0)) < -5,
    'RSI>55+bearish': lambda r: (float(r.get('rsi14', 50)) > 55 and
                                  float(r['close']) < float(r['open'])),
    'BB>0.60+bearish': lambda r: (float(r.get('bb_pct', 0.5)) > 0.60 and
                                   float(r['close']) < float(r['open'])),
    'RSI>55+BB>0.60+bear': lambda r: (float(r.get('rsi14', 50)) > 55 and
                                       float(r.get('bb_pct', 0.5)) > 0.60 and
                                       float(r['close']) < float(r['open'])),
}

print(f'  Base: {len(h2_signals)} trades, '
      f'{sum(1 for t in h2_signals if t["outcome"]=="TP")} wins')
for fname, ffn in filters.items():
    filtered = [t for t in h2_signals if ffn(t['row'])]
    if filtered:
        wins = sum(1 for t in filtered if t['outcome'] == 'TP')
        m = metrics(filtered, '')
        print(f'  + {fname:<25}: N={len(filtered):>2} W={wins} '
              f'WR={m["wr"]:.0%} PF={m["pf"]:.2f}')
    else:
        print(f'  + {fname:<25}: 0 trades (all blocked)')

# ============================================================
# SAME FILTERS — FULL OOS (2022-2026)
# ============================================================
print(f'\n{"="*80}')
print('RULE-BASED SHORT FILTERS - FULL OOS (2022-2026)')
print(f'{"="*80}')
oos_mask = (df.index >= '2022-01-01') & (df.index <= '2026-01-31')
df_oos = df[oos_mask]
all_oos_shorts = []
for idx in range(len(df_oos)):
    row = df_oos.iloc[idx]
    regime = detect_regime(row)
    if regime != 'BEAR':
        continue
    gi = df.index.get_loc(df_oos.index[idx])
    trade = detect_short_ml(df, gi, short_data)
    if trade is None:
        continue
    out = sim_short(df, gi, trade['entry'], trade['tp_pct'], trade['sl_pct'], SHORT_MAX_BARS)
    all_oos_shorts.append({
        'row': row, 'outcome': out[0], 'pnl_pct': out[2], 'ts': df_oos.index[idx],
    })

m_base = metrics(all_oos_shorts, '')
print(f'  Base ML: N={len(all_oos_shorts)} '
      f'W={sum(1 for t in all_oos_shorts if t["outcome"]=="TP")} '
      f'WR={m_base["wr"]:.1%} PF={m_base["pf"]:.2f}')

for fname, ffn in filters.items():
    filtered = [t for t in all_oos_shorts if ffn(t['row'])]
    if filtered:
        wins = sum(1 for t in filtered if t['outcome'] == 'TP')
        m = metrics(filtered, '')
        delta = m['pf'] - m_base['pf']
        print(f'  + {fname:<25}: N={len(filtered):>3} W={wins:>2} '
              f'WR={m["wr"]:.1%} PF={m["pf"]:.2f} ({delta:+.2f})')
    else:
        print(f'  + {fname:<25}: 0 trades')

print(f'\n{"="*80}')
print('DONE')
print(f'{"="*80}')
