"""
evaluate_eth_short_v5.py — Combinar mejores SHORT + OOS 2026
=============================================================
Parte 1: Combinaciones de las mejores estrategias (datos 2020-2025)
  A) Multi conf solo (baseline)
  B) BB upper solo (baseline)
  C) Multi conf OR BB upper
  D) Multi conf OR BB upper OR MeanRev v2
  E) Multi conf OR MR optimizado
  F) BB upper OR MR optimizado
  G) Multi conf OR BB upper OR MR optimizado

Parte 2: Mejores combos como COMITE COMPLETO (LONG BULL/RANGE + SHORT BEAR)
Parte 3: OOS 2026 (Ene-Mar) para los mejores
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
    sim_trade_fixed, metrics, WF_FOLDS, COMMISSION,
)
from evaluate_eth_bear import (
    detect_regime, add_extra_features, add_eth_specific_features,
    sim_short, equity_stats,
    detect_breakout_eth,
)
from evaluate_eth_v2 import detect_breakout_b_btc, detect_pullback_btc

MIN_BEAR_BARS = 30

# ============================================================
# CARGAR DATOS (TODO, sin cortar)
# ============================================================
print("=" * 70)
print("ETH SHORT v5 -- Combinaciones + OOS 2026")
print("=" * 70)

print("\nCargando datos...")
df_eth_raw = load_pair_4h('ETH')
df_btc_raw = load_btc_4h()

df_eth = compute_features_4h(df_eth_raw.copy())
df_btc = compute_features_4h(df_btc_raw.copy())

try:
    from v15_framework import load_pair_1d, load_btc_1d
    eth_1d = load_pair_1d('ETH')
    btc_1d = load_btc_1d()
except:
    eth_1d = df_eth_raw.resample('1D').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    btc_1d = df_btc_raw.resample('1D').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()

eth_macro = compute_macro_daily(eth_1d)
btc_macro = compute_macro_daily(btc_1d)
df_eth = merge_daily_to_4h(df_eth, eth_macro)
df_btc = merge_daily_to_4h(df_btc, btc_macro)
df_eth = add_extra_features(df_eth)
df_eth = add_eth_specific_features(df_eth, df_btc)

# Features extra
c = df_eth['close']
o = df_eth['open']
up = (c > c.shift(1)).astype(int)
df_eth['consec_up_3'] = up.rolling(3).sum()

regimes_eth = df_eth.apply(lambda r: detect_regime(r), axis=1)
regimes_btc = df_btc.apply(lambda r: detect_regime(r), axis=1)

eth_ret = df_eth['close'].pct_change()
btc_close_a = df_btc['close'].reindex(df_eth.index, method='ffill')
btc_ret = btc_close_a.pct_change()
corr_20 = eth_ret.rolling(20).corr(btc_ret)

# Folds BEAR validos (solo 2020-2025)
valid_folds = []
for start_s, end_s in WF_FOLDS:
    mask = (df_eth.index >= start_s) & (df_eth.index <= end_s)
    bear_bars = (regimes_eth[mask] == 'BEAR').sum()
    valid_folds.append(bear_bars >= MIN_BEAR_BARS)
n_valid = sum(valid_folds)
print(f"  Folds BEAR validos: {n_valid}/12")


# ============================================================
# DETECTORS
# ============================================================
def detect_multi_conf(df, i):
    if i < 25: return None
    row = df.iloc[i]
    c_val, o_val = float(row['close']), float(row['open'])
    if c_val >= o_val: return None
    if float(row.get('rsi14', 50)) < 60: return None
    if float(row.get('bb_pct', 0.5)) < 0.75: return None
    if float(row.get('vol_ratio', 1)) < 1.0: return None
    entry = c_val
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    sl = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    return {'direction': 'SHORT', 'setup': 'MULTI_CONF',
            'entry': entry, 'tp_pct': tp, 'sl_pct': sl}

def detect_bb_upper(df, i):
    if i < 25: return None
    row = df.iloc[i]
    c_val, o_val = float(row['close']), float(row['open'])
    if c_val >= o_val: return None
    if float(row.get('bb_pct', 0.5)) < 0.90: return None
    entry = c_val
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    sl = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    return {'direction': 'SHORT', 'setup': 'BB_UPPER',
            'entry': entry, 'tp_pct': tp, 'sl_pct': sl}

def detect_mr_v2(df, i):
    if i < 25: return None
    row = df.iloc[i]
    c_val, o_val = float(row['close']), float(row['open'])
    if c_val >= o_val: return None
    if float(row.get('rsi14', 50)) < 58: return None
    if float(row.get('price_zscore', 0)) < 0.5: return None
    if float(row.get('ema20_slope', 0)) > 1.5: return None
    entry = c_val
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    sl = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    return {'direction': 'SHORT', 'setup': 'MEANREV_V2',
            'entry': entry, 'tp_pct': tp, 'sl_pct': sl}

def detect_mr_opt(df, i):
    if i < 25: return None
    row = df.iloc[i]
    c_val, o_val = float(row['close']), float(row['open'])
    if c_val >= o_val: return None
    if float(row.get('rsi14', 50)) < 58: return None
    if float(row.get('price_zscore', 0)) < 0.5: return None
    if float(row.get('ema20_slope', 0)) > 1.5: return None
    entry = c_val
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 1.5, 0.04), 0.012)
    sl = max(min(atr_pct / 100 * 2.0, 0.06), 0.020)
    return {'direction': 'SHORT', 'setup': 'MR_OPT',
            'entry': entry, 'tp_pct': tp, 'sl_pct': sl, 'max_bars': 8}

# Combos
def make_combo(detectors):
    def detect(df, i):
        for fn in detectors:
            t = fn(df, i)
            if t is not None:
                return t
        return None
    return detect

COMBOS = {
    'A) Multi conf': [detect_multi_conf],
    'B) BB upper': [detect_bb_upper],
    'C) Multi+BB': [detect_multi_conf, detect_bb_upper],
    'D) Multi+BB+MRv2': [detect_multi_conf, detect_bb_upper, detect_mr_v2],
    'E) Multi+MR_opt': [detect_multi_conf, detect_mr_opt],
    'F) BB+MR_opt': [detect_bb_upper, detect_mr_opt],
    'G) Multi+BB+MR_opt': [detect_multi_conf, detect_bb_upper, detect_mr_opt],
}


# ============================================================
# WF SHORT solo BEAR
# ============================================================
def wf_bear_short(df, regimes, detect_fn, max_bars=16):
    results = []
    all_trades = []
    for idx, (start_s, end_s) in enumerate(WF_FOLDS):
        period = f"{start_s[:7]}/{end_s[5:7]}"
        if not valid_folds[idx]:
            results.append({'period': period, 'n': 0, 'wr': 0, 'pf': 0, 'ok': False, 'skip': True})
            continue
        trades = []
        for i in range(30, len(df)):
            ts = df.index[i]
            if ts < pd.Timestamp(start_s, tz='UTC') or ts > pd.Timestamp(end_s, tz='UTC'):
                continue
            if regimes.iloc[i] != 'BEAR': continue
            if i + max_bars + 2 >= len(df): continue
            trade = detect_fn(df, i)
            if trade is None: continue
            mb = trade.get('max_bars', max_bars)
            out = sim_short(df, i, trade['entry'], trade['tp_pct'], trade['sl_pct'], max_bars=mb)
            trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                          'setup': trade.get('setup', '?'), 'direction': 'SHORT'})
        m = metrics(trades, period)
        ok = (m['n'] >= 2 and m['wr'] > 0.40 and m['pf'] > 0.9)
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                        'pf': m['pf'], 'ok': ok, 'skip': False})
        all_trades.extend(trades)
    return {'folds': results, 'all_trades': all_trades}


def summarize(wf):
    fv = [r for r in wf['folds'] if not r.get('skip', False)]
    fok = sum(1 for r in fv if r['ok'])
    fval = len(fv)
    m = metrics(wf['all_trades'], 'OOS')
    eq, dd = equity_stats(wf['all_trades'])
    pct = fok / max(fval, 1)
    passed = pct >= 0.60 and m['pf'] >= 1.2 and m['wr'] >= 0.45
    marginal = pct >= 0.50 and m['pf'] >= 1.0 and m['wr'] >= 0.40
    tag = "APROBADO" if passed else ("MARGINAL" if marginal else "RECHAZADO")
    return m, eq, dd, fok, fval, tag


# ============================================================
# PARTE 1: SHORT standalone combos
# ============================================================
print("\n" + "=" * 70)
print("PARTE 1: SHORT combos (solo BEAR, datos 2020-2025)")
print("=" * 70)

combo_results = {}
for name, detectors in COMBOS.items():
    detect_fn = make_combo(detectors)
    wf = wf_bear_short(df_eth, regimes_eth, detect_fn)
    s = summarize(wf)
    combo_results[name] = (wf, s)

    m, eq, dd, fok, fval, tag = s
    print(f"\n  {name}:")
    for r in wf['folds']:
        if r.get('skip'):
            print(f"    {r['period']}: skip")
            continue
        wr_s = f"{r['wr']:.1%}" if r['n'] > 0 else "n/a"
        pf_s = f"{r['pf']:.2f}" if r['n'] > 0 else "n/a"
        ok_s = "+" if r['ok'] else "-"
        print(f"    {r['period']}: N={r['n']:>3} WR={wr_s:>6} PF={pf_s:>6} {ok_s}")
    print(f"    BEAR OK: {fok}/{fval} ({fok/max(fval,1):.0%}) | N={m['n']} WR={m['wr']:.1%} "
          f"PF={m['pf']:.2f} ${1000*eq:.0f} DD={dd:.1%} -> {tag}")
    # Breakdown
    setups = {}
    for t in wf['all_trades']:
        s_name = t.get('setup', '?')
        if s_name not in setups: setups[s_name] = []
        setups[s_name].append(t)
    if len(setups) > 1:
        for s_name, ts in setups.items():
            wins = sum(1 for t in ts if t['pnl_pct'] > 0)
            print(f"      {s_name}: N={len(ts)} WR={wins/len(ts):.1%}")

print(f"\n  {'Combo':<22} | {'BEAR OK':>8} | {'N':>4} | {'WR':>6} | {'PF':>6} | {'$1K->':>7} | {'DD':>6} | Veredicto")
print(f"  {'-'*90}")
for name, (wf, (m, eq, dd, fok, fval, tag)) in combo_results.items():
    mark = "**" if tag == "APROBADO" else "  "
    print(f"  {mark}{name:<20} | {fok}/{fval} {fok/max(fval,1):>3.0%} | {m['n']:>4} | "
          f"{m['wr']:.1%} | {m['pf']:.2f} | ${1000*eq:>6.0f} | {dd:.1%} | {tag}")


# ============================================================
# PARTE 2: COMITE COMPLETO (LONG BULL/RANGE + SHORT BEAR)
# ============================================================
print("\n" + "=" * 70)
print("PARTE 2: COMITES COMPLETOS (LONG + SHORT)")
print("=" * 70)

def run_committee(df_eth, df_btc, regimes_eth, regimes_btc, corr_20,
                  short_detect_fn, start_s, end_s):
    trades = []
    for i in range(30, len(df_eth)):
        ts = df_eth.index[i]
        if ts < pd.Timestamp(start_s, tz='UTC') or ts > pd.Timestamp(end_s, tz='UTC'):
            continue
        if i + 18 >= len(df_eth): continue

        regime = regimes_eth.iloc[i]
        trade = None

        if regime in ('BULL', 'RANGE'):
            # LONG: follower + breakout ETH (TP/SL adaptados)
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
                        cv = corr_20.get(ts, 0)
                        if not pd.isna(cv) and cv >= 0.5:
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

        elif regime == 'BEAR' and short_detect_fn is not None:
            trade = short_detect_fn(df_eth, i)

        if trade is None: continue

        d = trade.get('direction', 'LONG')
        if d == 'LONG':
            out = sim_trade_fixed(df_eth, i, trade['entry'],
                                  trade['tp_pct'], trade['sl_pct'], max_bars=18)
        else:
            mb = trade.get('max_bars', 16)
            out = sim_short(df_eth, i, trade['entry'],
                           trade['tp_pct'], trade['sl_pct'], max_bars=mb)

        trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                      'setup': trade['setup'], 'direction': d, 'entry': trade['entry']})
    return trades


def wf_committee(short_detect_fn, name):
    results = []
    all_trades = []
    for start_s, end_s in WF_FOLDS:
        period = f"{start_s[:7]}/{end_s[5:7]}"
        trades = run_committee(df_eth, df_btc, regimes_eth, regimes_btc,
                              corr_20, short_detect_fn, start_s, end_s)
        m = metrics(trades, period)
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                        'pf': m['pf'], 'ok': ok})
        all_trades.extend(trades)
    folds_ok = sum(1 for r in results if r['ok'])
    return {'folds': results, 'folds_ok': folds_ok, 'all_trades': all_trades}


# Comites
committee_configs = {
    'Solo LONG': None,
    'LONG + Multi+BB': make_combo([detect_multi_conf, detect_bb_upper]),
    'LONG + Multi+BB+MRv2': make_combo([detect_multi_conf, detect_bb_upper, detect_mr_v2]),
    'LONG + Multi+BB+MR_opt': make_combo([detect_multi_conf, detect_bb_upper, detect_mr_opt]),
}

committee_results = {}
for name, short_fn in committee_configs.items():
    wf = wf_committee(short_fn, name)
    m = metrics(wf['all_trades'], 'OOS')
    eq, dd = equity_stats(wf['all_trades'])
    passed = wf['folds_ok'] >= 7 and m['pf'] >= 1.2
    tag = "APROBADO" if passed else "MARGINAL" if wf['folds_ok'] >= 6 and m['pf'] >= 1.0 else "RECHAZADO"
    committee_results[name] = (wf, m, eq, dd, tag)

    print(f"\n  {name}:")
    for r in wf['folds']:
        wr_s = f"{r['wr']:.1%}" if r['n'] > 0 else "n/a"
        pf_s = f"{r['pf']:.2f}" if r['n'] > 0 else "n/a"
        ok_s = "+" if r['ok'] else "-"
        print(f"    {r['period']}: N={r['n']:>3} WR={wr_s:>6} PF={pf_s:>6} {ok_s}")
    print(f"    WF: {wf['folds_ok']}/12 | N={m['n']} WR={m['wr']:.1%} PF={m['pf']:.2f} "
          f"${1000*eq:.0f} DD={dd:.1%} -> {tag}")
    # Breakdown
    setups = {}
    for t in wf['all_trades']:
        s_name = t.get('setup', '?')
        if s_name not in setups: setups[s_name] = []
        setups[s_name].append(t)
    for s_name, ts in sorted(setups.items()):
        wins = sum(1 for t in ts if t['pnl_pct'] > 0)
        print(f"      {s_name}: N={len(ts)} WR={wins/len(ts):.1%}")

print(f"\n  {'Comite':<26} | {'WF':>5} | {'N':>4} | {'WR':>6} | {'PF':>6} | {'$1K->':>7} | {'DD':>6} | Veredicto")
print(f"  {'-'*90}")
for name, (wf, m, eq, dd, tag) in committee_results.items():
    mark = "**" if tag == "APROBADO" else "  "
    print(f"  {mark}{name:<24} | {wf['folds_ok']:>2}/12 | {m['n']:>4} | "
          f"{m['wr']:.1%} | {m['pf']:.2f} | ${1000*eq:>6.0f} | {dd:.1%} | {tag}")


# ============================================================
# PARTE 3: OOS 2026 (Ene-Mar)
# ============================================================
print("\n" + "=" * 70)
print("PARTE 3: OOS 2026 (Ene-Mar) -- datos nunca vistos")
print("=" * 70)

OOS_START = '2026-01-01'
OOS_END = '2026-03-01'

# Contexto
eth_2026 = df_eth[(df_eth.index >= OOS_START) & (df_eth.index <= OOS_END)]
reg_2026 = regimes_eth[(regimes_eth.index >= OOS_START) & (regimes_eth.index <= OOS_END)]
reg_counts = reg_2026.value_counts()
eth_p0 = float(df_eth.loc[df_eth.index >= OOS_START, 'close'].iloc[0])
eth_p1 = float(df_eth.loc[df_eth.index <= OOS_END, 'close'].iloc[-1])
btc_p0 = float(df_btc.loc[df_btc.index >= OOS_START, 'close'].iloc[0])
btc_p1 = float(df_btc.loc[df_btc.index <= OOS_END, 'close'].iloc[-1])

print(f"\n  ETH: ${eth_p0:.0f} -> ${eth_p1:.0f} ({(eth_p1/eth_p0-1)*100:+.1f}%)")
print(f"  BTC: ${btc_p0:.0f} -> ${btc_p1:.0f} ({(btc_p1/btc_p0-1)*100:+.1f}%)")
print(f"  Regimen 2026: {dict(reg_counts)}")

for name, short_fn in committee_configs.items():
    trades = run_committee(df_eth, df_btc, regimes_eth, regimes_btc,
                          corr_20, short_fn, OOS_START, OOS_END)
    m = metrics(trades, '2026')
    eq, dd = equity_stats(trades)
    longs = [t for t in trades if t['direction'] == 'LONG']
    shorts = [t for t in trades if t['direction'] == 'SHORT']

    print(f"\n  --- {name} ---")
    print(f"  N={m['n']} (L:{len(longs)} S:{len(shorts)}) | WR={m['wr']:.1%} | "
          f"PF={m['pf']:.2f} | $1K -> ${1000*eq:.0f} | DD={dd:.1%}")

    if trades:
        print(f"\n  {'Fecha':<20} {'Setup':<18} {'Dir':<6} {'Entry':>8} {'Res':<3} {'PnL':>7}")
        print(f"  {'-'*68}")
        for t in sorted(trades, key=lambda x: x['ts']):
            print(f"  {str(t['ts'])[:19]:<20} {t['setup']:<18} {t['direction']:<6} "
                  f"${t['entry']:>7.1f} {t['outcome']:<3} {t['pnl_pct']:>+6.2%}")

# Resumen OOS 2026
print(f"\n  {'Comite':<26} | {'N':>3} | {'L':>2} | {'S':>2} | {'WR':>6} | {'PF':>6} | {'$1K->':>7} | {'DD':>6}")
print(f"  {'-'*80}")
for name, short_fn in committee_configs.items():
    trades = run_committee(df_eth, df_btc, regimes_eth, regimes_btc,
                          corr_20, short_fn, OOS_START, OOS_END)
    m = metrics(trades, '2026')
    eq, dd = equity_stats(trades)
    nl = sum(1 for t in trades if t['direction'] == 'LONG')
    ns = sum(1 for t in trades if t['direction'] == 'SHORT')
    print(f"  {name:<26} | {m['n']:>3} | {nl:>2} | {ns:>2} | "
          f"{m['wr']:.1%} | {m['pf']:.2f} | ${1000*eq:>6.0f} | {dd:.1%}")

print(f"\n  Contexto: ETH {(eth_p1/eth_p0-1)*100:+.1f}% | BTC {(btc_p1/btc_p0-1)*100:+.1f}%")
print("=" * 70)
