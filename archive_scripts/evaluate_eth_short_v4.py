"""
evaluate_eth_short_v4.py — Buscar SHORT con WR > 50% en BEAR puro
===================================================================
Enfoques nuevos enfocados en WIN RATE alto:
  1. TP mas facil de alcanzar (TP asimetrico)
  2. Exit rapido (max_bars corto)
  3. Patrones de velas bajistas
  4. Multi-confirmacion estricta
  5. Exhaustion patterns (subida en BEAR = trampa)
  6. EMA rejection (rebote a EMA20 falla en BEAR)
  7. BB extremo (price > upper BB en BEAR = short)
  8. Vol spike reversal
  9. Combinaciones top
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
)

CUTOFF = '2025-12-31'
MIN_BEAR_BARS = 30

print("=" * 70)
print("ETH SHORT v4 -- WR > 50% en BEAR puro")
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
    eth_1d = load_pair_1d('ETH')[lambda x: x.index <= CUTOFF]
    btc_1d = load_btc_1d()[lambda x: x.index <= CUTOFF]
except:
    eth_1d = df_eth_raw.resample('1D').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    btc_1d = df_btc_raw.resample('1D').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()

eth_macro = compute_macro_daily(eth_1d)
btc_macro = compute_macro_daily(btc_1d)
df_eth = merge_daily_to_4h(df_eth, eth_macro)
df_btc = merge_daily_to_4h(df_btc, btc_macro)
df_eth = add_extra_features(df_eth)
df_eth = add_eth_specific_features(df_eth, df_btc)

# Features extra para patrones
c = df_eth['close']
o = df_eth['open']
h = df_eth['high']
l = df_eth['low']

# Bearish engulfing
df_eth['bearish_engulf'] = (
    (c < o) &                           # vela actual bajista
    (c.shift(1) > o.shift(1)) &         # vela anterior alcista
    (o > c.shift(1)) &                  # apertura > cierre anterior
    (c < o.shift(1))                    # cierre < apertura anterior
).astype(int)

# Evening star (3 candle pattern)
df_eth['evening_star'] = (
    (c.shift(2) > o.shift(2)) &         # 1ra vela alcista
    (abs(c.shift(1) - o.shift(1)) < abs(c.shift(2) - o.shift(2)) * 0.3) &  # 2da vela pequena
    (c < o) &                           # 3ra vela bajista
    (c < (o.shift(2) + c.shift(2)) / 2) # cierra debajo del medio de la 1ra
).astype(int)

# Consecutive up in bear (trampa alcista)
up = (c > c.shift(1)).astype(int)
df_eth['consec_up_3'] = up.rolling(3).sum()  # 3+ velas subiendo

# Distance to EMA20
df_eth['ema20_dist_pct'] = (c - df_eth['close'].ewm(span=20).mean()) / df_eth['close'].ewm(span=20).mean() * 100

regimes_eth = df_eth.apply(lambda r: detect_regime(r), axis=1)

# Folds BEAR validos
valid_folds = []
for start_s, end_s in WF_FOLDS:
    mask = (df_eth.index >= start_s) & (df_eth.index <= end_s)
    bear_bars = (regimes_eth[mask] == 'BEAR').sum()
    valid_folds.append(bear_bars >= MIN_BEAR_BARS)

n_valid = sum(valid_folds)
print(f"  Folds BEAR validos: {n_valid}/12")
print(f"  Criterio: >= {int(n_valid * 0.6)}/{n_valid} (60%) + PF >= 1.2 + WR > 50%\n")


# ============================================================
def wf_bear(df, regimes, detect_fn, name, max_bars=16):
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
            if regimes.iloc[i] != 'BEAR':
                continue
            if i + max_bars + 2 >= len(df):
                continue
            trade = detect_fn(df, i)
            if trade is None:
                continue
            mb = trade.get('max_bars', max_bars)
            out = sim_short(df, i, trade['entry'], trade['tp_pct'], trade['sl_pct'], max_bars=mb)
            trades.append({'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                          'setup': trade.get('setup', name)})
        m = metrics(trades, period)
        ok = (m['n'] >= 2 and m['wr'] > 0.40 and m['pf'] > 0.9)
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                        'pf': m['pf'], 'ok': ok, 'skip': False})
        all_trades.extend(trades)
    return {'folds': results, 'all_trades': all_trades}


def summarize(name, wf):
    fv = [r for r in wf['folds'] if not r.get('skip', False)]
    fok = sum(1 for r in fv if r['ok'])
    fval = len(fv)
    m = metrics(wf['all_trades'], 'OOS')
    eq, dd = equity_stats(wf['all_trades'])
    pct = fok / max(fval, 1)
    passed = pct >= 0.60 and m['pf'] >= 1.2 and m['wr'] >= 0.50
    marginal = pct >= 0.50 and m['pf'] >= 1.0 and m['wr'] >= 0.45
    tag = "APROBADO" if passed else ("MARGINAL" if marginal else "RECHAZADO")
    return m, eq, dd, fok, fval, tag


def print_detail(name, wf):
    m, eq, dd, fok, fval, tag = summarize(name, wf)
    print(f"\n--- {name} ---")
    for r in wf['folds']:
        if r.get('skip'):
            print(f"  {r['period']:<16} | skip")
            continue
        wr_s = f"{r['wr']:.1%}" if r['n'] > 0 else "n/a"
        pf_s = f"{r['pf']:.2f}" if r['n'] > 0 else "n/a"
        ok_s = "+" if r['ok'] else "-"
        print(f"  {r['period']:<16} | N={r['n']:>3} WR={wr_s:>6} PF={pf_s:>6} {ok_s}")
    print(f"  BEAR OK: {fok}/{fval} ({fok/max(fval,1):.0%}) | "
          f"N={m['n']} WR={m['wr']:.1%} PF={m['pf']:.2f} "
          f"${1000*eq:.0f} DD={dd:.1%} -> {tag}")
    return m, eq, dd, fok, fval, tag


# ============================================================
# Helper para TP/SL
# ============================================================
def atr_tpsl(row, tp_mult, sl_mult, tp_min, tp_max, sl_min, sl_max):
    atr_pct = float(row.get('atr_pct', 2.5))
    sl = max(min(atr_pct / 100 * sl_mult, sl_max), sl_min)
    tp = max(min(atr_pct / 100 * tp_mult, tp_max), tp_min)
    return tp, sl


# ============================================================
# OPCION 1: TP asimetrico (TP facil = 1.5%, SL amplio = 3%)
# ============================================================
print("=" * 70)
print("1. TP asimetrico (TP chico facil de alcanzar, SL amplio)")
print("=" * 70)

def detect_tp_asym(df, i):
    row = df.iloc[i]
    c, o_val = float(row['close']), float(row['open'])
    if c >= o_val: return None
    rsi = float(row.get('rsi14', 50))
    if rsi < 45: return None  # no oversold
    entry = c
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 1.0, 0.025), 0.010)  # TP chico: 1-2.5%
    sl = max(min(atr_pct / 100 * 2.0, 0.06), 0.020)   # SL amplio: 2-6%
    return {'setup': 'ASYM_TPSL', 'entry': entry, 'tp_pct': tp, 'sl_pct': sl, 'max_bars': 10}

wf1 = wf_bear(df_eth, regimes_eth, detect_tp_asym, 'ASYM', max_bars=10)
r1 = print_detail("TP asimetrico", wf1)


# ============================================================
# 2. Exit rapido (max 6 barras = 24h)
# ============================================================
print("\n" + "=" * 70)
print("2. MeanRev RSI>60 + exit rapido (6 barras)")
print("=" * 70)

def detect_fast_exit(df, i):
    if i < 25: return None
    row = df.iloc[i]
    c, o_val = float(row['close']), float(row['open'])
    if c >= o_val: return None
    if float(row.get('rsi14', 50)) < 60: return None
    entry = c
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 2.0, 0.06), 0.015)
    sl = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    return {'setup': 'FAST_EXIT', 'entry': entry, 'tp_pct': tp, 'sl_pct': sl, 'max_bars': 6}

wf2 = wf_bear(df_eth, regimes_eth, detect_fast_exit, 'FastExit', max_bars=6)
r2 = print_detail("Fast exit", wf2)


# ============================================================
# 3. Bearish engulfing en BEAR
# ============================================================
print("\n" + "=" * 70)
print("3. Bearish engulfing en BEAR")
print("=" * 70)

def detect_bearish_engulf(df, i):
    if i < 25: return None
    row = df.iloc[i]
    if int(row.get('bearish_engulf', 0)) != 1: return None
    entry = float(row['close'])
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    sl = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    return {'setup': 'BEAR_ENGULF', 'entry': entry, 'tp_pct': tp, 'sl_pct': sl}

wf3 = wf_bear(df_eth, regimes_eth, detect_bearish_engulf, 'BearEngulf')
r3 = print_detail("Bearish engulfing", wf3)


# ============================================================
# 4. Multi-confirmacion estricta (RSI>60 + BB>0.8 + vela bajista + vol)
# ============================================================
print("\n" + "=" * 70)
print("4. Multi-confirmacion (RSI>60 + BB>0.8 + bajista + vol)")
print("=" * 70)

def detect_multi_confirm(df, i):
    if i < 25: return None
    row = df.iloc[i]
    c, o_val = float(row['close']), float(row['open'])
    if c >= o_val: return None
    if float(row.get('rsi14', 50)) < 60: return None
    if float(row.get('bb_pct', 0.5)) < 0.75: return None
    if float(row.get('vol_ratio', 1)) < 1.0: return None
    entry = c
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    sl = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    return {'setup': 'MULTI_CONF', 'entry': entry, 'tp_pct': tp, 'sl_pct': sl}

wf4 = wf_bear(df_eth, regimes_eth, detect_multi_confirm, 'MultiConf')
r4 = print_detail("Multi-confirmacion", wf4)


# ============================================================
# 5. Exhaustion: 3+ velas subiendo en BEAR + vela bajista = trampa
# ============================================================
print("\n" + "=" * 70)
print("5. Exhaustion (3+ velas up en BEAR + reversal)")
print("=" * 70)

def detect_exhaustion(df, i):
    if i < 25: return None
    row = df.iloc[i]
    c, o_val = float(row['close']), float(row['open'])
    if c >= o_val: return None  # vela actual debe ser bajista (la reversal)
    if float(row.get('consec_up_3', 0)) < 2: return None  # al menos 2 de 3 velas previas up
    # Confirm: RSI no extremo
    rsi = float(row.get('rsi14', 50))
    if rsi < 40 or rsi > 80: return None
    entry = c
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    sl = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    return {'setup': 'EXHAUSTION', 'entry': entry, 'tp_pct': tp, 'sl_pct': sl}

wf5 = wf_bear(df_eth, regimes_eth, detect_exhaustion, 'Exhaustion')
r5 = print_detail("Exhaustion", wf5)


# ============================================================
# 6. EMA20 rejection (precio sube a EMA20 y falla)
# ============================================================
print("\n" + "=" * 70)
print("6. EMA20 rejection (rebote a EMA20 falla en BEAR)")
print("=" * 70)

def detect_ema_rejection(df, i):
    if i < 25: return None
    row = df.iloc[i]
    c = float(row['close'])
    o_val = float(row['open'])
    h_val = float(row['high'])
    if c >= o_val: return None  # vela bajista

    ema20 = c * (1 + float(row.get('ema20_dist_pct', 0)) / 100)
    # High toco o supero EMA20 pero cerro debajo
    if h_val < ema20 * 0.998: return None  # high no llego a EMA20
    if c > ema20 * 0.99: return None  # cerro encima de EMA20

    # EMA20 slope negativo (tendencia bajista)
    if float(row.get('ema20_slope', 0)) > 0: return None

    entry = c
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    sl = max(min(atr_pct / 100 * 1.2, 0.04), 0.012)
    return {'setup': 'EMA_REJECT', 'entry': entry, 'tp_pct': tp, 'sl_pct': sl}

wf6 = wf_bear(df_eth, regimes_eth, detect_ema_rejection, 'EMA_Reject')
r6 = print_detail("EMA20 rejection", wf6)


# ============================================================
# 7. BB upper touch en BEAR
# ============================================================
print("\n" + "=" * 70)
print("7. BB upper touch en BEAR (overbought en bear = short)")
print("=" * 70)

def detect_bb_upper(df, i):
    if i < 25: return None
    row = df.iloc[i]
    c, o_val = float(row['close']), float(row['open'])
    if c >= o_val: return None
    bb_pct = float(row.get('bb_pct', 0.5))
    if bb_pct < 0.90: return None  # muy cerca de upper BB
    entry = c
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    sl = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    return {'setup': 'BB_UPPER', 'entry': entry, 'tp_pct': tp, 'sl_pct': sl}

wf7 = wf_bear(df_eth, regimes_eth, detect_bb_upper, 'BB_Upper')
r7 = print_detail("BB upper touch", wf7)


# ============================================================
# 8. Volume spike + bearish (panico de compra en BEAR falla)
# ============================================================
print("\n" + "=" * 70)
print("8. Volume spike bajista (vol > 2x + caida)")
print("=" * 70)

def detect_vol_spike(df, i):
    if i < 25: return None
    row = df.iloc[i]
    c, o_val = float(row['close']), float(row['open'])
    if c >= o_val: return None
    if float(row.get('vol_ratio', 1)) < 2.0: return None  # volumen 2x
    # Caida significativa
    ret = (c - o_val) / o_val
    if ret > -0.01: return None  # al menos -1% en la vela
    entry = c
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 2.0, 0.06), 0.020)
    sl = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    return {'setup': 'VOL_SPIKE', 'entry': entry, 'tp_pct': tp, 'sl_pct': sl}

wf8 = wf_bear(df_eth, regimes_eth, detect_vol_spike, 'VolSpike')
r8 = print_detail("Volume spike bajista", wf8)


# ============================================================
# 9. RSI>65 puro (mas estricto = menos trades pero mejor WR)
# ============================================================
print("\n" + "=" * 70)
print("9. RSI > 65 + bajista (ultra selectivo)")
print("=" * 70)

def detect_rsi65(df, i):
    if i < 25: return None
    row = df.iloc[i]
    c, o_val = float(row['close']), float(row['open'])
    if c >= o_val: return None
    if float(row.get('rsi14', 50)) < 65: return None
    entry = c
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    sl = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    return {'setup': 'RSI65', 'entry': entry, 'tp_pct': tp, 'sl_pct': sl}

wf9 = wf_bear(df_eth, regimes_eth, detect_rsi65, 'RSI65')
r9 = print_detail("RSI > 65", wf9)


# ============================================================
# 10. Price zscore > 1.0 + bajista (muy sobre la media en BEAR)
# ============================================================
print("\n" + "=" * 70)
print("10. Price z-score > 1.0 + bajista en BEAR")
print("=" * 70)

def detect_zscore_high(df, i):
    if i < 25: return None
    row = df.iloc[i]
    c, o_val = float(row['close']), float(row['open'])
    if c >= o_val: return None
    if float(row.get('price_zscore', 0)) < 1.0: return None
    entry = c
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    sl = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    return {'setup': 'ZSCORE_HI', 'entry': entry, 'tp_pct': tp, 'sl_pct': sl}

wf10 = wf_bear(df_eth, regimes_eth, detect_zscore_high, 'ZscoreHi')
r10 = print_detail("Z-score > 1.0", wf10)


# ============================================================
# 11. MeanRev v2 + TP chico (1x ATR) + SL amplio (2x ATR) + max 8 bars
# ============================================================
print("\n" + "=" * 70)
print("11. MeanRev v2 optimizado (TP chico, SL amplio, exit rapido)")
print("=" * 70)

def detect_mr_optimized(df, i):
    if i < 25: return None
    row = df.iloc[i]
    c, o_val = float(row['close']), float(row['open'])
    if c >= o_val: return None
    if float(row.get('rsi14', 50)) < 58: return None
    if float(row.get('price_zscore', 0)) < 0.5: return None
    if float(row.get('ema20_slope', 0)) > 1.5: return None
    entry = c
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 1.5, 0.04), 0.012)  # TP mas chico
    sl = max(min(atr_pct / 100 * 2.0, 0.06), 0.020)   # SL mas amplio
    return {'setup': 'MR_OPT', 'entry': entry, 'tp_pct': tp, 'sl_pct': sl, 'max_bars': 8}

wf11 = wf_bear(df_eth, regimes_eth, detect_mr_optimized, 'MR_Opt', max_bars=8)
r11 = print_detail("MeanRev optimizado", wf11)


# ============================================================
# 12. RSI>55 + EMA slope negativo + vela bajista grande (body > 0.5%)
# ============================================================
print("\n" + "=" * 70)
print("12. Trend continuation (EMA- + RSI rebote + vela bajista grande)")
print("=" * 70)

def detect_trend_cont(df, i):
    if i < 25: return None
    row = df.iloc[i]
    c, o_val = float(row['close']), float(row['open'])
    if c >= o_val: return None
    body = abs(c - o_val) / o_val * 100
    if body < 0.5: return None  # vela con cuerpo significativo
    if float(row.get('rsi14', 50)) < 45 or float(row.get('rsi14', 50)) > 65: return None
    if float(row.get('ema20_slope', 0)) > -0.1: return None  # EMA bajando
    entry = c
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    sl = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    return {'setup': 'TREND_CONT', 'entry': entry, 'tp_pct': tp, 'sl_pct': sl}

wf12 = wf_bear(df_eth, regimes_eth, detect_trend_cont, 'TrendCont')
r12 = print_detail("Trend continuation", wf12)


# ============================================================
# 13. Multi-confirm v2: RSI>55 + zscore>0.3 + bajista + BTC cayendo
# ============================================================
print("\n" + "=" * 70)
print("13. Multi v2 (RSI>55 + zscore + bajista + BTC ret<0)")
print("=" * 70)

def detect_multi_v2(df, i):
    if i < 25: return None
    row = df.iloc[i]
    c, o_val = float(row['close']), float(row['open'])
    if c >= o_val: return None
    if float(row.get('rsi14', 50)) < 55: return None
    if float(row.get('price_zscore', 0)) < 0.3: return None
    if float(row.get('btc_ret_5', 0)) > 0: return None  # BTC cayendo
    entry = c
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    sl = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    return {'setup': 'MULTI_V2', 'entry': entry, 'tp_pct': tp, 'sl_pct': sl}

wf13 = wf_bear(df_eth, regimes_eth, detect_multi_v2, 'MultiV2')
r13 = print_detail("Multi v2", wf13)


# ============================================================
# 14. Exhaustion + RSI>55 (rebote en BEAR con RSI alto = trampa)
# ============================================================
print("\n" + "=" * 70)
print("14. Exhaustion + RSI (velas up + RSI rebotado + reversal)")
print("=" * 70)

def detect_exhaust_rsi(df, i):
    if i < 25: return None
    row = df.iloc[i]
    c, o_val = float(row['close']), float(row['open'])
    if c >= o_val: return None
    if float(row.get('consec_up_3', 0)) < 2: return None
    if float(row.get('rsi14', 50)) < 50: return None
    entry = c
    atr_pct = float(row.get('atr_pct', 2.5))
    tp = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    sl = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    return {'setup': 'EXHAUST_RSI', 'entry': entry, 'tp_pct': tp, 'sl_pct': sl}

wf14 = wf_bear(df_eth, regimes_eth, detect_exhaust_rsi, 'ExhaustRSI')
r14 = print_detail("Exhaustion + RSI", wf14)


# ============================================================
# 15. BEST COMBO: MeanRev v2 OR Exhaustion+RSI
# ============================================================
print("\n" + "=" * 70)
print("15. Combo: MeanRev v2 OR Exhaustion+RSI")
print("=" * 70)

def detect_best_combo(df, i):
    t = detect_mr_optimized(df, i)
    if t is not None: return t
    return detect_exhaust_rsi(df, i)

wf15 = wf_bear(df_eth, regimes_eth, detect_best_combo, 'BestCombo')
r15 = print_detail("MR_opt + Exhaust", wf15)


# ============================================================
# RESUMEN
# ============================================================
print("\n" + "=" * 70)
print(f"RESUMEN -- ETH SHORT v4 (folds BEAR = {n_valid})")
print("=" * 70)

all_opts = [
    ("1. TP asym", r1), ("2. Fast exit", r2), ("3. Bear engulf", r3),
    ("4. Multi conf", r4), ("5. Exhaustion", r5), ("6. EMA reject", r6),
    ("7. BB upper", r7), ("8. Vol spike", r8), ("9. RSI>65", r9),
    ("10. Zscore>1", r10), ("11. MR optimiz", r11), ("12. Trend cont", r12),
    ("13. Multi v2", r13), ("14. Exhaust+RSI", r14), ("15. MR+Exhaust", r15),
]

print(f"\n  {'Opcion':<16} | {'BEAR OK':>8} | {'N':>5} | {'WR':>6} | {'PF':>6} | {'$1K->':>7} | {'DD':>6} | Veredicto")
print(f"  {'-'*90}")
for name, (m, eq, dd, fok, fval, tag) in all_opts:
    mark = "**" if tag == "APROBADO" else "  "
    print(f"  {mark}{name:<14} | {fok:>2}/{fval:<2} {fok/max(fval,1):>3.0%} | {m['n']:>5} | "
          f"{m['wr']:.1%} | {m['pf']:.2f} | ${1000*eq:>6.0f} | {dd:.1%} | {tag}")

approved = [x for x in all_opts if x[1][5] == "APROBADO"]
marginal = [x for x in all_opts if x[1][5] == "MARGINAL"]
wr50 = [(n, r) for n, r in all_opts if r[0]['wr'] >= 0.50 and r[0]['n'] >= 20]

if approved:
    print(f"\n  APROBADAS:")
    for n, r in approved:
        print(f"    {n}: {r[3]}/{r[4]} BEAR OK, WR={r[0]['wr']:.1%} PF={r[0]['pf']:.2f}")
if marginal:
    print(f"\n  MARGINALES:")
    for n, r in marginal:
        print(f"    {n}: {r[3]}/{r[4]} BEAR OK, WR={r[0]['wr']:.1%} PF={r[0]['pf']:.2f}")
if wr50:
    print(f"\n  WR >= 50% (N>=20):")
    for n, r in wr50:
        print(f"    {n}: WR={r[0]['wr']:.1%} PF={r[0]['pf']:.2f} N={r[0]['n']} BEAR {r[3]}/{r[4]}")

print("\n" + "=" * 70)
