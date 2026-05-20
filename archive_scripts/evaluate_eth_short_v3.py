"""
evaluate_eth_short_v3.py — SHORT solo en folds con mercado BEAR
================================================================
Correccion: solo evaluar folds donde ETH tiene suficientes barras BEAR.
Si un semestre no tiene BEAR, no cuenta (ni positivo ni negativo).

Criterio: fold valido si tiene >= 30 barras BEAR.
Aprobacion: >= 60% de folds validos positivos + PF >= 1.2.
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
    metrics, WF_FOLDS, COMMISSION,
)
from evaluate_eth_bear import (
    detect_regime, add_extra_features, add_eth_specific_features,
    sim_short, equity_stats,
    create_short_labels, SHORT_FEATURES_ETH,
    get_short_models,
    detect_short_meanrev, detect_short_breakdown,
)

CUTOFF = '2025-12-31'
MIN_BEAR_BARS = 30  # minimo barras BEAR para que un fold cuente

print("=" * 70)
print("ETH SHORT v3 -- Solo folds con mercado BEAR")
print("=" * 70)

# --- Cargar datos ---
print("\nCargando datos...")
df_eth_raw = load_pair_4h('ETH')
df_btc_raw = load_btc_4h()
df_eth_raw = df_eth_raw[df_eth_raw.index <= CUTOFF]
df_btc_raw = df_btc_raw[df_btc_raw.index <= CUTOFF]

df_eth = compute_features_4h(df_eth_raw.copy())
df_btc = compute_features_4h(df_btc_raw.copy())

try:
    from v15_framework import load_pair_1d, load_btc_1d
    eth_1d = load_pair_1d('ETH')
    btc_1d = load_btc_1d()
    eth_1d = eth_1d[eth_1d.index <= CUTOFF]
    btc_1d = btc_1d[btc_1d.index <= CUTOFF]
except:
    eth_1d = df_eth_raw.resample('1D').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    btc_1d = df_btc_raw.resample('1D').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()

eth_macro = compute_macro_daily(eth_1d)
btc_macro = compute_macro_daily(btc_1d)
df_eth = merge_daily_to_4h(df_eth, eth_macro)
df_btc = merge_daily_to_4h(df_btc, btc_macro)
df_eth = add_extra_features(df_eth)
df_eth = add_eth_specific_features(df_eth, df_btc)

regimes_eth = df_eth.apply(lambda r: detect_regime(r), axis=1)
labels = create_short_labels(df_eth)

# --- Identificar folds con BEAR suficiente ---
print(f"\n  Barras BEAR por fold (minimo {MIN_BEAR_BARS} para contar):")
valid_folds = []
for start_s, end_s in WF_FOLDS:
    mask = (df_eth.index >= start_s) & (df_eth.index <= end_s)
    bear_bars = (regimes_eth[mask] == 'BEAR').sum()
    total = mask.sum()
    is_valid = bear_bars >= MIN_BEAR_BARS
    valid_folds.append(is_valid)
    tag = "VALIDO" if is_valid else "skip"
    print(f"    {start_s[:7]}/{end_s[5:7]}: {bear_bars:>4} BEAR / {total} total ({bear_bars/max(total,1)*100:.0f}%) -> {tag}")

n_valid = sum(valid_folds)
print(f"\n  Folds validos: {n_valid}/12")
print(f"  Criterio aprobacion: >= {int(n_valid * 0.6)}/{n_valid} folds OK (60%)")


# ============================================================
# WF generico para SHORT (solo folds BEAR validos)
# ============================================================
def wf_short_bear_only(df, regimes, detect_fn, name):
    results = []
    all_trades = []

    for idx, (start_s, end_s) in enumerate(WF_FOLDS):
        period = f"{start_s[:7]}/{end_s[5:7]}"

        if not valid_folds[idx]:
            results.append({'period': period, 'n': 0, 'wr': 0, 'pf': 0,
                           'ok': False, 'skip': True})
            continue

        trades = []
        for i in range(30, len(df)):
            ts = df.index[i]
            if ts < pd.Timestamp(start_s, tz='UTC') or ts > pd.Timestamp(end_s, tz='UTC'):
                continue
            if regimes.iloc[i] != 'BEAR':
                continue
            if i + 16 >= len(df):
                continue

            trade = detect_fn(df, i)
            if trade is None:
                continue

            out = sim_short(df, i, trade['entry'],
                           trade['tp_pct'], trade['sl_pct'], max_bars=16)
            trades.append({
                'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                'setup': trade.get('setup', name),
            })

        m = metrics(trades, period)
        ok = (m['n'] >= 2 and m['wr'] > 0.35 and m['pf'] > 0.9)
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                        'pf': m['pf'], 'ok': ok, 'skip': False})
        all_trades.extend(trades)

    return {'folds': results, 'all_trades': all_trades}


def wf_short_ml_bear_only(df, labels, regimes, features, model_name,
                          constructor, threshold):
    from sklearn.preprocessing import StandardScaler
    results = []
    all_trades = []

    for idx, (start_s, end_s) in enumerate(WF_FOLDS):
        period = f"{start_s[:7]}/{end_s[5:7]}"

        if not valid_folds[idx]:
            results.append({'period': period, 'n': 0, 'wr': 0, 'pf': 0,
                           'ok': False, 'skip': True})
            continue

        train_mask = df.index < start_s
        y_train = labels[train_mask]
        bear_train = regimes[train_mask] == 'BEAR'
        valid = y_train.notna() & bear_train
        X_train = df.loc[train_mask, features][valid].fillna(0)
        y_train_v = y_train[valid]

        if len(X_train) < 200 or y_train_v.sum() < 10:
            results.append({'period': period, 'n': 0, 'wr': 0, 'pf': 0,
                           'ok': False, 'skip': False})
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train.values)
        model = constructor()
        try:
            model.fit(X_train_s, y_train_v)
        except:
            results.append({'period': period, 'n': 0, 'wr': 0, 'pf': 0,
                           'ok': False, 'skip': False})
            continue

        trades = []
        for i in range(30, len(df)):
            ts = df.index[i]
            if ts < pd.Timestamp(start_s, tz='UTC') or ts > pd.Timestamp(end_s, tz='UTC'):
                continue
            if regimes.iloc[i] != 'BEAR':
                continue
            if i + 16 >= len(df):
                continue

            row = df.iloc[i]
            x = pd.DataFrame([row[features].fillna(0).values], columns=features)
            x_s = scaler.transform(x)
            prob = model.predict_proba(x_s)[0][1]

            if prob >= threshold:
                entry = float(row['close'])
                atr_pct = float(row.get('atr_pct', 2.5))
                sl_pct = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
                tp_pct = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
                out = sim_short(df, i, entry, tp_pct, sl_pct, max_bars=16)
                trades.append({
                    'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                    'setup': f'SHORT_ML_{model_name}',
                })

        m = metrics(trades, period)
        ok = (m['n'] >= 2 and m['wr'] > 0.35 and m['pf'] > 0.9)
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                        'pf': m['pf'], 'ok': ok, 'skip': False})
        all_trades.extend(trades)

    return {'folds': results, 'all_trades': all_trades}


def print_wf_bear(name, wf):
    print(f"\n--- {name} ---")
    print(f"  {'Periodo':<16} | {'N':>4} | {'WR':>6} | {'PF':>6} | {'BEAR':>5}")
    print(f"  {'-'*52}")

    folds_valid = 0
    folds_ok = 0
    for r in wf['folds']:
        if r.get('skip', False):
            print(f"  {r['period']:<16} | {'---':>4} | {'---':>6} | {'---':>6} | skip")
            continue
        folds_valid += 1
        wr_s = f"{r['wr']:.1%}" if r['n'] > 0 else "n/a"
        pf_s = f"{r['pf']:.2f}" if r['n'] > 0 else "n/a"
        ok_s = "+" if r['ok'] else "-"
        if r['ok']:
            folds_ok += 1
        print(f"  {r['period']:<16} | {r['n']:>4} | {wr_s:>6} | {pf_s:>6} | {ok_s}")

    m_all = metrics(wf['all_trades'], 'OOS')
    eq, dd = equity_stats(wf['all_trades'])
    pct = folds_ok / max(folds_valid, 1)
    passed = pct >= 0.60 and m_all['pf'] >= 1.2
    tag = "APROBADO" if passed else ("MARGINAL" if pct >= 0.50 and m_all['pf'] >= 1.0 else "RECHAZADO")

    print(f"\n  Folds BEAR OK: {folds_ok}/{folds_valid} ({pct:.0%})")
    print(f"  Total: N={m_all['n']} | WR={m_all['wr']:.1%} | PF={m_all['pf']:.2f} | "
          f"$1K -> ${1000*eq:.0f} | DD={dd:.1%} -> {tag}")

    setups = {}
    for t in wf['all_trades']:
        s = t.get('setup', 'UNKNOWN')
        if s not in setups:
            setups[s] = []
        setups[s].append(t)
    if len(setups) > 1:
        for s, ts in setups.items():
            wins = sum(1 for t in ts if t['pnl_pct'] > 0)
            print(f"    {s}: N={len(ts)} WR={wins/len(ts):.1%}")

    return m_all, eq, dd, folds_ok, folds_valid, tag


# ============================================================
# OPCIONES
# ============================================================

# 1. MeanRev solo
print("\n" + "=" * 70)
print("OPCION 1: MeanRev solo")
print("=" * 70)
wf1 = wf_short_bear_only(df_eth, regimes_eth, detect_short_meanrev, 'MeanRev')
r1 = print_wf_bear("MeanRev solo", wf1)

# 2. MeanRev v2 (RSI>58, price_zscore>0.5)
print("\n" + "=" * 70)
print("OPCION 2: MeanRev v2 (RSI>58 + price zscore)")
print("=" * 70)

def detect_meanrev_v2(df, i):
    if i < 25: return None
    row = df.iloc[i]
    c, o = float(row['close']), float(row['open'])
    if c >= o: return None
    rsi = float(row.get('rsi14', 50))
    if rsi < 58: return None
    pz = float(row.get('price_zscore', 0))
    if pz < 0.5: return None
    if float(row.get('ema20_slope', 0)) > 1.5: return None
    entry = c
    atr_pct = float(row.get('atr_pct', 2.5))
    sl_pct = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    tp_pct = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    return {'direction': 'SHORT', 'setup': 'SHORT_MEANREV_V2',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}

wf2 = wf_short_bear_only(df_eth, regimes_eth, detect_meanrev_v2, 'MeanRev_v2')
r2 = print_wf_bear("MeanRev v2", wf2)

# 3. MeanRev v2 + TP/SL mas amplio (3x ATR TP, 1.2x SL)
print("\n" + "=" * 70)
print("OPCION 3: MeanRev v2 + TP/SL amplio")
print("=" * 70)

def detect_meanrev_v2_wide(df, i):
    if i < 25: return None
    row = df.iloc[i]
    c, o = float(row['close']), float(row['open'])
    if c >= o: return None
    rsi = float(row.get('rsi14', 50))
    if rsi < 58: return None
    pz = float(row.get('price_zscore', 0))
    if pz < 0.5: return None
    if float(row.get('ema20_slope', 0)) > 1.5: return None
    entry = c
    atr_pct = float(row.get('atr_pct', 2.5))
    sl_pct = max(min(atr_pct / 100 * 1.2, 0.04), 0.012)
    tp_pct = max(min(atr_pct / 100 * 3.0, 0.10), 0.030)
    return {'direction': 'SHORT', 'setup': 'SHORT_MEANREV_V2W',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}

wf3 = wf_short_bear_only(df_eth, regimes_eth, detect_meanrev_v2_wide, 'MeanRev_v2w')
r3 = print_wf_bear("MeanRev v2 wide", wf3)

# 4. MeanRev + Breakdown
print("\n" + "=" * 70)
print("OPCION 4: MeanRev + Breakdown (original)")
print("=" * 70)

def detect_meanrev_or_breakdown(df, i):
    t = detect_short_meanrev(df, i)
    if t is not None: return t
    return detect_short_breakdown(df, i)

wf4 = wf_short_bear_only(df_eth, regimes_eth, detect_meanrev_or_breakdown, 'MR+BRK')
r4 = print_wf_bear("MeanRev + Breakdown", wf4)

# 5. RF t=0.50 (features ETH)
print("\n" + "=" * 70)
print("OPCION 5: RF t=0.50")
print("=" * 70)
model_constructors = get_short_models()
wf5 = wf_short_ml_bear_only(df_eth, labels, regimes_eth, SHORT_FEATURES_ETH,
                             'RF', model_constructors['RF'], 0.50)
r5 = print_wf_bear("RF t=0.50", wf5)

# 6. RF t=0.45
print("\n" + "=" * 70)
print("OPCION 6: RF t=0.45")
print("=" * 70)
wf6 = wf_short_ml_bear_only(df_eth, labels, regimes_eth, SHORT_FEATURES_ETH,
                             'RF', model_constructors['RF'], 0.45)
r6 = print_wf_bear("RF t=0.45", wf6)

# 7. GBM t=0.50
print("\n" + "=" * 70)
print("OPCION 7: GBM t=0.50")
print("=" * 70)
wf7 = wf_short_ml_bear_only(df_eth, labels, regimes_eth, SHORT_FEATURES_ETH,
                             'GBM', model_constructors['GBM'], 0.50)
r7 = print_wf_bear("GBM t=0.50", wf7)

# 8. GBM t=0.45
print("\n" + "=" * 70)
print("OPCION 8: GBM t=0.45")
print("=" * 70)
wf8 = wf_short_ml_bear_only(df_eth, labels, regimes_eth, SHORT_FEATURES_ETH,
                             'GBM', model_constructors['GBM'], 0.45)
r8 = print_wf_bear("GBM t=0.45", wf8)

# 9. SHORT momentum (caida fuerte + vol)
print("\n" + "=" * 70)
print("OPCION 9: SHORT momentum")
print("=" * 70)

def detect_short_momentum(df, i):
    if i < 25: return None
    row = df.iloc[i]
    c, o = float(row['close']), float(row['open'])
    if c >= o: return None
    ret_3 = (c / float(df['close'].iloc[i-3]) - 1) * 100
    if ret_3 > -2: return None
    if float(row.get('ema20_slope', 0)) > -0.3: return None
    if float(row.get('vol_ratio', 1)) < 1.2: return None
    rsi = float(row.get('rsi14', 50))
    if rsi < 20 or rsi > 60: return None
    entry = c
    atr_pct = float(row.get('atr_pct', 2.5))
    sl_pct = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    tp_pct = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    return {'direction': 'SHORT', 'setup': 'SHORT_MOMENTUM',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}

wf9 = wf_short_bear_only(df_eth, regimes_eth, detect_short_momentum, 'Momentum')
r9 = print_wf_bear("SHORT momentum", wf9)

# 10. MeanRev v2 + BTC confirma (BTC ret_5 < 0)
print("\n" + "=" * 70)
print("OPCION 10: MeanRev v2 + BTC bajando")
print("=" * 70)

def detect_meanrev_btc_confirm(df, i):
    if i < 25: return None
    row = df.iloc[i]
    c, o = float(row['close']), float(row['open'])
    if c >= o: return None
    rsi = float(row.get('rsi14', 50))
    if rsi < 58: return None
    pz = float(row.get('price_zscore', 0))
    if pz < 0.3: return None
    # BTC tambien negativo
    btc_ret = float(row.get('btc_ret_5', 0))
    if btc_ret > 0: return None
    entry = c
    atr_pct = float(row.get('atr_pct', 2.5))
    sl_pct = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    tp_pct = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    return {'direction': 'SHORT', 'setup': 'SHORT_MR_BTC',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}

wf10 = wf_short_bear_only(df_eth, regimes_eth, detect_meanrev_btc_confirm, 'MR+BTC')
r10 = print_wf_bear("MeanRev + BTC confirm", wf10)

# 11. MeanRev v2 + Momentum combo
print("\n" + "=" * 70)
print("OPCION 11: MeanRev v2 + Momentum combo")
print("=" * 70)

def detect_meanrev_v2_or_momentum(df, i):
    t = detect_meanrev_v2(df, i)
    if t is not None: return t
    return detect_short_momentum(df, i)

wf11 = wf_short_bear_only(df_eth, regimes_eth, detect_meanrev_v2_or_momentum, 'MRv2+Mom')
r11 = print_wf_bear("MeanRev v2 + Momentum", wf11)

# 12. LGBM t=0.50
print("\n" + "=" * 70)
print("OPCION 12: LGBM t=0.50")
print("=" * 70)
wf12 = wf_short_ml_bear_only(df_eth, labels, regimes_eth, SHORT_FEATURES_ETH,
                              'LGBM', model_constructors['LGBM'], 0.50)
r12 = print_wf_bear("LGBM t=0.50", wf12)


# ============================================================
# RESUMEN
# ============================================================
print("\n" + "=" * 70)
print(f"RESUMEN -- ETH SHORT v3 (solo folds BEAR, {n_valid} validos)")
print("=" * 70)

all_opts = [
    ("1. MeanRev", wf1, r1),
    ("2. MeanRev v2", wf2, r2),
    ("3. MR v2 wide", wf3, r3),
    ("4. MR+Breakdown", wf4, r4),
    ("5. RF t=0.50", wf5, r5),
    ("6. RF t=0.45", wf6, r6),
    ("7. GBM t=0.50", wf7, r7),
    ("8. GBM t=0.45", wf8, r8),
    ("9. Momentum", wf9, r9),
    ("10. MR+BTC conf", wf10, r10),
    ("11. MRv2+Mom", wf11, r11),
    ("12. LGBM t=0.50", wf12, r12),
]

print(f"\n  {'Opcion':<18} | {'BEAR OK':>8} | {'N':>5} | {'WR':>6} | {'PF':>6} | {'$1K->':>7} | {'DD':>6} | Veredicto")
print(f"  {'-'*90}")
for name, wf, (m, eq, dd, fok, fval, tag) in all_opts:
    pct_s = f"{fok}/{fval}" if fval > 0 else "0/0"
    mark = "**" if tag == "APROBADO" else "  "
    print(f"  {mark}{name:<16} | {pct_s:>8} | {m['n']:>5} | "
          f"{m['wr']:.1%} | {m['pf']:.2f} | ${1000*eq:>6.0f} | {dd:.1%} | {tag}")

approved = [(n, wf, r) for n, wf, r in all_opts if r[5] == "APROBADO"]
marginal = [(n, wf, r) for n, wf, r in all_opts if r[5] == "MARGINAL"]

if approved:
    print(f"\n  APROBADAS:")
    for n, wf, r in approved:
        print(f"    {n}: {r[3]}/{r[4]} folds BEAR OK, PF={r[0]['pf']:.2f}")
    best = max(approved, key=lambda x: x[2][0]['pf'])
    print(f"\n  MEJOR: {best[0]}")
elif marginal:
    print(f"\n  NINGUNA APROBADA, pero MARGINALES:")
    for n, wf, r in marginal:
        print(f"    {n}: {r[3]}/{r[4]} folds BEAR OK, PF={r[0]['pf']:.2f}")
else:
    print(f"\n  NINGUNA APROBADA NI MARGINAL")

print("\n" + "=" * 70)
