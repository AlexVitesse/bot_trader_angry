"""
evaluate_eth_2026_oos.py — Test OOS puro: Ene-Mar 2026
=======================================================
Entrena con TODO el dato antes de 2026-01-01.
Evalua SOLO en 2026-01-01 a 2026-03-01.
NO se usa 2026 para entrenar.

Comites a evaluar:
  A) Solo LONG (BULL/RANGE)
  B) LONG + SHORT rules (MeanRev + Breakdown)
  C) LONG + SHORT ML (RF, threshold=0.60)
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
    sim_trade_fixed, metrics, COMMISSION,
)
from evaluate_eth_bear import (
    detect_regime, add_extra_features, add_eth_specific_features,
    sim_short, equity_stats,
    create_short_labels, SHORT_FEATURES_ETH,
    get_short_models,
    detect_breakout_eth,
    detect_short_meanrev, detect_short_breakdown,
)
from evaluate_eth_v2 import detect_breakout_b_btc, detect_pullback_btc

# ============================================================
OOS_START = '2026-01-01'
OOS_END   = '2026-03-01'
TRAIN_END = '2025-12-31'
# ============================================================

print("=" * 70)
print("ETH 2026 OOS -- Evaluacion pura fuera de muestra")
print("=" * 70)

# --- Cargar datos ---
print("\nCargando datos...")
df_eth_raw = load_pair_4h('ETH')
df_btc_raw = load_btc_4h()

# Verificar cobertura 2026
eth_2026 = df_eth_raw[df_eth_raw.index >= OOS_START]
print(f"  ETH total: {len(df_eth_raw)} bars ({df_eth_raw.index[0].date()} a {df_eth_raw.index[-1].date()})")
print(f"  ETH 2026: {len(eth_2026)} bars ({eth_2026.index[0].date()} a {eth_2026.index[-1].date()})")

# --- Features ---
df_eth = compute_features_4h(df_eth_raw.copy())
df_btc = compute_features_4h(df_btc_raw.copy())

# Macro diario
try:
    from v15_framework import load_pair_1d
    eth_1d = load_pair_1d('ETH')
except:
    eth_1d = df_eth_raw.resample('1D').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()

try:
    from v15_framework import load_btc_1d
    btc_1d = load_btc_1d()
except:
    btc_1d = df_btc_raw.resample('1D').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()

eth_macro = compute_macro_daily(eth_1d)
btc_macro = compute_macro_daily(btc_1d)
df_eth = merge_daily_to_4h(df_eth, eth_macro)
df_btc = merge_daily_to_4h(df_btc, btc_macro)

# Extra features
df_eth = add_extra_features(df_eth)
df_eth = add_eth_specific_features(df_eth, df_btc)

# Regimenes
regimes_eth = df_eth.apply(lambda r: detect_regime(r), axis=1)
regimes_btc = df_btc.apply(lambda r: detect_regime(r), axis=1)

# Cross data
eth_ret = df_eth['close'].pct_change()
btc_close_aligned = df_btc['close'].reindex(df_eth.index, method='ffill')
btc_ret = btc_close_aligned.pct_change()
corr_20 = eth_ret.rolling(20).corr(btc_ret)

# --- Mostrar regimen en 2026 ---
reg_2026 = regimes_eth[(regimes_eth.index >= OOS_START) & (regimes_eth.index <= OOS_END)]
reg_counts = reg_2026.value_counts()
print(f"\n  Regimen ETH en 2026 (ene-mar):")
for r, c in reg_counts.items():
    print(f"    {r}: {c} bars ({c/len(reg_2026)*100:.0f}%)")

eth_price_start = float(df_eth.loc[df_eth.index >= OOS_START, 'close'].iloc[0])
eth_price_end = float(df_eth.loc[df_eth.index <= OOS_END, 'close'].iloc[-1])
print(f"\n  ETH precio: ${eth_price_start:.0f} -> ${eth_price_end:.0f} ({(eth_price_end/eth_price_start-1)*100:+.1f}%)")

btc_start = float(df_btc.loc[df_btc.index >= OOS_START, 'close'].iloc[0])
btc_end = float(df_btc.loc[df_btc.index <= OOS_END, 'close'].iloc[-1])
print(f"  BTC precio: ${btc_start:.0f} -> ${btc_end:.0f} ({(btc_end/btc_start-1)*100:+.1f}%)")


# ============================================================
# Funcion para simular comite en un periodo
# ============================================================
def run_committee_period(df_eth, df_btc, regimes_eth, regimes_btc,
                         corr_20, start_s, end_s,
                         use_short_rules=False,
                         short_model_data=None):
    """Ejecutar comite en un periodo especifico."""
    trades = []
    for i in range(30, len(df_eth)):
        ts = df_eth.index[i]
        if ts < pd.Timestamp(start_s, tz='UTC') or ts > pd.Timestamp(end_s, tz='UTC'):
            continue
        if i + 18 >= len(df_eth):
            continue

        regime = regimes_eth.iloc[i]
        trade = None

        if regime in ('BULL', 'RANGE'):
            # LONG: follower + breakout ETH
            if ts in df_btc.index:
                btc_i = df_btc.index.get_loc(ts)
                if btc_i >= 30:
                    regime_btc = regimes_btc.iloc[btc_i]
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

            if trade is None:
                trade = detect_breakout_eth(df_eth, i)

        elif regime == 'BEAR':
            if use_short_rules:
                trade = detect_short_meanrev(df_eth, i)
                if trade is None:
                    trade = detect_short_breakdown(df_eth, i)
            elif short_model_data is not None:
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
            'entry': trade['entry'],
        })

    return trades


# ============================================================
# A) Solo LONG
# ============================================================
print("\n" + "=" * 70)
print("A) COMITE SOLO LONG (BULL/RANGE) -- 2026 OOS")
print("=" * 70)

trades_long = run_committee_period(
    df_eth, df_btc, regimes_eth, regimes_btc, corr_20,
    OOS_START, OOS_END,
    use_short_rules=False, short_model_data=None)

m = metrics(trades_long, '2026')
eq, dd = equity_stats(trades_long)
print(f"\n  N={m['n']} | WR={m['wr']:.1%} | PF={m['pf']:.2f} | $1K -> ${1000*eq:.0f} | DD={dd:.1%}")

if trades_long:
    print(f"\n  Trades detalle:")
    print(f"  {'Fecha':<22} {'Setup':<18} {'Dir':<6} {'Entry':>8} {'Result':<4} {'PnL':>7}")
    print(f"  {'-'*70}")
    for t in sorted(trades_long, key=lambda x: x['ts']):
        print(f"  {str(t['ts'])[:19]:<22} {t['setup']:<18} {t['direction']:<6} "
              f"${t['entry']:>7.1f} {t['outcome']:<4} {t['pnl_pct']:>+6.2%}")

    # Breakdown por setup
    setups = {}
    for t in trades_long:
        s = t['setup']
        if s not in setups:
            setups[s] = []
        setups[s].append(t)
    print(f"\n  Por setup:")
    for s, ts in setups.items():
        wins = sum(1 for t in ts if t['pnl_pct'] > 0)
        pnl = sum(t['pnl_pct'] for t in ts)
        print(f"    {s}: N={len(ts)} WR={wins/len(ts):.1%} PnL={pnl:+.2%}")


# ============================================================
# B) LONG + SHORT rules
# ============================================================
print("\n" + "=" * 70)
print("B) COMITE LONG + SHORT RULES -- 2026 OOS")
print("=" * 70)

trades_rules = run_committee_period(
    df_eth, df_btc, regimes_eth, regimes_btc, corr_20,
    OOS_START, OOS_END,
    use_short_rules=True, short_model_data=None)

m = metrics(trades_rules, '2026')
eq, dd = equity_stats(trades_rules)
print(f"\n  N={m['n']} | WR={m['wr']:.1%} | PF={m['pf']:.2f} | $1K -> ${1000*eq:.0f} | DD={dd:.1%}")

if trades_rules:
    print(f"\n  Trades detalle:")
    print(f"  {'Fecha':<22} {'Setup':<18} {'Dir':<6} {'Entry':>8} {'Result':<4} {'PnL':>7}")
    print(f"  {'-'*70}")
    for t in sorted(trades_rules, key=lambda x: x['ts']):
        print(f"  {str(t['ts'])[:19]:<22} {t['setup']:<18} {t['direction']:<6} "
              f"${t['entry']:>7.1f} {t['outcome']:<4} {t['pnl_pct']:>+6.2%}")

    setups = {}
    for t in trades_rules:
        s = t['setup']
        if s not in setups:
            setups[s] = []
        setups[s].append(t)
    print(f"\n  Por setup:")
    for s, ts in setups.items():
        wins = sum(1 for t in ts if t['pnl_pct'] > 0)
        pnl = sum(t['pnl_pct'] for t in ts)
        print(f"    {s}: N={len(ts)} WR={wins/len(ts):.1%} PnL={pnl:+.2%}")


# ============================================================
# C) LONG + SHORT ML (RF)
# ============================================================
print("\n" + "=" * 70)
print("C) COMITE LONG + SHORT ML (RF t=0.60) -- 2026 OOS")
print("=" * 70)

# Entrenar RF con TODO dato BEAR antes de 2026
print("\n  Entrenando RF SHORT con datos hasta 2025-12-31...")
from sklearn.preprocessing import StandardScaler

labels = create_short_labels(df_eth)
train_mask = df_eth.index <= TRAIN_END
y_train = labels[train_mask]
bear_train = regimes_eth[train_mask] == 'BEAR'
valid = y_train.notna() & bear_train
X_train = df_eth.loc[train_mask, SHORT_FEATURES_ETH][valid].fillna(0)
y_train_v = y_train[valid]

print(f"  BEAR train bars: {valid.sum()} | Positive rate: {y_train_v.mean():.1%}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train.values)

models = get_short_models()
rf = models['RF']()
rf.fit(X_train_s, y_train_v)

short_model_data = {
    'model': rf,
    'scaler': scaler,
    'features': SHORT_FEATURES_ETH,
    'threshold': 0.60,
}

# Verificar predicciones en 2026 BEAR
oos_mask = (df_eth.index >= OOS_START) & (df_eth.index <= OOS_END)
bear_oos = regimes_eth[oos_mask] == 'BEAR'
if bear_oos.sum() > 0:
    X_oos = df_eth.loc[oos_mask & (regimes_eth == 'BEAR'), SHORT_FEATURES_ETH].fillna(0)
    X_oos_s = scaler.transform(X_oos.values)
    probs = rf.predict_proba(X_oos_s)[:, 1]
    print(f"  2026 BEAR bars: {len(X_oos)}")
    print(f"  Prob stats: mean={probs.mean():.3f} median={np.median(probs):.3f} "
          f"max={probs.max():.3f} min={probs.min():.3f}")
    print(f"  Bars con prob >= 0.60: {(probs >= 0.60).sum()}")
    print(f"  Bars con prob >= 0.55: {(probs >= 0.55).sum()}")
    print(f"  Bars con prob >= 0.50: {(probs >= 0.50).sum()}")

trades_ml = run_committee_period(
    df_eth, df_btc, regimes_eth, regimes_btc, corr_20,
    OOS_START, OOS_END,
    use_short_rules=False, short_model_data=short_model_data)

m = metrics(trades_ml, '2026')
eq, dd = equity_stats(trades_ml)
print(f"\n  N={m['n']} | WR={m['wr']:.1%} | PF={m['pf']:.2f} | $1K -> ${1000*eq:.0f} | DD={dd:.1%}")

if trades_ml:
    print(f"\n  Trades detalle:")
    print(f"  {'Fecha':<22} {'Setup':<18} {'Dir':<6} {'Entry':>8} {'Result':<4} {'PnL':>7}")
    print(f"  {'-'*70}")
    for t in sorted(trades_ml, key=lambda x: x['ts']):
        print(f"  {str(t['ts'])[:19]:<22} {t['setup']:<18} {t['direction']:<6} "
              f"${t['entry']:>7.1f} {t['outcome']:<4} {t['pnl_pct']:>+6.2%}")

    setups = {}
    for t in trades_ml:
        s = t['setup']
        if s not in setups:
            setups[s] = []
        setups[s].append(t)
    print(f"\n  Por setup:")
    for s, ts in setups.items():
        wins = sum(1 for t in ts if t['pnl_pct'] > 0)
        pnl = sum(t['pnl_pct'] for t in ts)
        print(f"    {s}: N={len(ts)} WR={wins/len(ts):.1%} PnL={pnl:+.2%}")


# ============================================================
# RESUMEN
# ============================================================
print("\n" + "=" * 70)
print("RESUMEN -- ETH 2026 OOS (Ene-Mar)")
print("=" * 70)

for name, trades in [("Solo LONG", trades_long),
                     ("LONG+SHORT rules", trades_rules),
                     ("LONG+SHORT ML(RF)", trades_ml)]:
    m = metrics(trades, '2026')
    eq, dd = equity_stats(trades)
    longs = [t for t in trades if t['direction'] == 'LONG']
    shorts = [t for t in trades if t.get('direction') == 'SHORT']
    print(f"\n  {name}:")
    print(f"    N={m['n']} (L:{len(longs)} S:{len(shorts)}) | "
          f"WR={m['wr']:.1%} | PF={m['pf']:.2f} | "
          f"$1K -> ${1000*eq:.0f} | DD={dd:.1%}")

print(f"\n  Contexto: ETH {(eth_price_end/eth_price_start-1)*100:+.1f}% | "
      f"BTC {(btc_end/btc_start-1)*100:+.1f}%")
print("=" * 70)
