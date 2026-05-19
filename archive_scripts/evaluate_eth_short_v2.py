"""
evaluate_eth_short_v2.py — Ronda 2 SHORT: mas opciones
========================================================
REGLA: Solo datos hasta 2025-12-31 para train y WF.
2026 se reserva como OOS puro (no se toca aqui).

Opciones a probar:
  1. MeanRev solo (sin Breakdown que perdio en 2026)
  2. RF con threshold mas bajo (0.50, 0.55)
  3. GBM con threshold bajo (0.45-0.55)
  4. BTC SHORT follower (cuando BTC da SHORT, ETH sigue)
  5. Rules + ML filter (MeanRev pero solo si ML prob > 0.45)
  6. SHORT volatility (operar SHORT en spikes de vol extrema)
  7. SHORT momentum (ETH cae fuerte + BTC cae = SHORT ETH)

Walk-forward: 12 folds (2020-2025), expanding window.
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
    sim_trade_fixed, metrics,
    WF_FOLDS, COMMISSION,
)
from evaluate_eth_bear import (
    detect_regime, add_extra_features, add_eth_specific_features,
    sim_short, equity_stats,
    create_short_labels, SHORT_FEATURES_ETH, SHORT_FEATURES_BASE,
    get_short_models,
    detect_short_meanrev,
)

# Solo datos hasta 2025-12-31
CUTOFF = '2025-12-31'

print("=" * 70)
print("ETH SHORT v2 -- Mas opciones (datos hasta 2025)")
print("=" * 70)

# --- Cargar datos ---
print("\nCargando datos...")
df_eth_raw = load_pair_4h('ETH')
df_btc_raw = load_btc_4h()

# Cortar en 2025
df_eth_raw = df_eth_raw[df_eth_raw.index <= CUTOFF]
df_btc_raw = df_btc_raw[df_btc_raw.index <= CUTOFF]
print(f"  ETH: {len(df_eth_raw)} bars (hasta {df_eth_raw.index[-1].date()})")
print(f"  BTC: {len(df_btc_raw)} bars (hasta {df_btc_raw.index[-1].date()})")

# --- Features ---
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
regimes_btc = df_btc.apply(lambda r: detect_regime(r), axis=1)

bear_pct = (regimes_eth == 'BEAR').mean()
print(f"  ETH BEAR: {bear_pct:.1%} del tiempo")

# Labels SHORT
labels = create_short_labels(df_eth)

# Cross data
eth_ret = df_eth['close'].pct_change()
btc_close_a = df_btc['close'].reindex(df_eth.index, method='ffill')
btc_ret = btc_close_a.pct_change()
corr_20 = eth_ret.rolling(20).corr(btc_ret)


# ============================================================
# Helper: WF para SHORT strategies
# ============================================================
def wf_short_strategy(df, regimes, detect_fn, name, tp_sl_from_trade=True):
    """Walk-forward generico para SHORT rules."""
    results = []
    all_trades = []

    for start_s, end_s in WF_FOLDS:
        period = f"{start_s[:7]}/{end_s[5:7]}"
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
                        'pf': m['pf'], 'ok': ok})
        all_trades.extend(trades)

    folds_ok = sum(1 for r in results if r['ok'])
    return {'folds': results, 'folds_ok': folds_ok, 'all_trades': all_trades}


def wf_short_ml(df, labels, regimes, features, model_name, constructor,
                threshold, tp_pct=0.03, sl_pct=0.02):
    """Walk-forward ML SHORT con threshold fijo."""
    from sklearn.preprocessing import StandardScaler
    results = []
    all_trades = []

    for start_s, end_s in WF_FOLDS:
        period = f"{start_s[:7]}/{end_s[5:7]}"
        trades = []

        train_mask = df.index < start_s
        test_mask = (df.index >= start_s) & (df.index <= end_s)

        y_train = labels[train_mask]
        bear_train = regimes[train_mask] == 'BEAR'
        valid = y_train.notna() & bear_train
        X_train = df.loc[train_mask, features][valid].fillna(0)
        y_train_v = y_train[valid]

        if len(X_train) < 200 or y_train_v.sum() < 10:
            results.append({'period': period, 'n': 0, 'wr': 0, 'pf': 0, 'ok': False})
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train.values)
        model = constructor()
        try:
            model.fit(X_train_s, y_train_v)
        except:
            results.append({'period': period, 'n': 0, 'wr': 0, 'pf': 0, 'ok': False})
            continue

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
                sl_p = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
                tp_p = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
                out = sim_short(df, i, entry, tp_p, sl_p, max_bars=16)
                trades.append({
                    'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                    'setup': f'SHORT_ML_{model_name}',
                })

        m = metrics(trades, period)
        ok = (m['n'] >= 2 and m['wr'] > 0.35 and m['pf'] > 0.9)
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                        'pf': m['pf'], 'ok': ok})
        all_trades.extend(trades)

    folds_ok = sum(1 for r in results if r['ok'])
    return {'folds': results, 'folds_ok': folds_ok, 'all_trades': all_trades}


def print_wf(name, wf):
    print(f"\n--- {name} ---")
    print(f"  {'Periodo':<16} | {'N':>4} | {'WR':>6} | {'PF':>6} | OK")
    print(f"  {'-'*50}")
    for r in wf['folds']:
        wr_s = f"{r['wr']:.1%}" if r['n'] > 0 else "n/a"
        pf_s = f"{r['pf']:.2f}" if r['n'] > 0 else "n/a"
        ok_s = "+" if r['ok'] else "-"
        print(f"  {r['period']:<16} | {r['n']:>4} | {wr_s:>6} | {pf_s:>6} | {ok_s}")
    m_all = metrics(wf['all_trades'], 'OOS')
    eq, dd = equity_stats(wf['all_trades'])
    print(f"\n  Folds OK: {wf['folds_ok']}/12")
    print(f"  Total: N={m_all['n']} | WR={m_all['wr']:.1%} | PF={m_all['pf']:.2f} | "
          f"$1K -> ${1000*eq:.0f} | DD={dd:.1%}")
    # Breakdown por setup
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
    return m_all, eq, dd


# ============================================================
# OPCION 1: MeanRev solo (sin Breakdown)
# ============================================================
print("\n" + "=" * 70)
print("OPCION 1: MeanRev solo (sin Breakdown)")
print("=" * 70)

wf1 = wf_short_strategy(df_eth, regimes_eth, detect_short_meanrev, 'MeanRev')
m1, eq1, dd1 = print_wf("MeanRev solo", wf1)


# ============================================================
# OPCION 2: RF threshold=0.50
# ============================================================
print("\n" + "=" * 70)
print("OPCION 2: RF threshold=0.50 (features ETH)")
print("=" * 70)

models = get_short_models()
wf2 = wf_short_ml(df_eth, labels, regimes_eth, SHORT_FEATURES_ETH,
                   'RF', models['RF'], threshold=0.50)
m2, eq2, dd2 = print_wf("RF t=0.50", wf2)


# ============================================================
# OPCION 3: RF threshold=0.55
# ============================================================
print("\n" + "=" * 70)
print("OPCION 3: RF threshold=0.55 (features ETH)")
print("=" * 70)

wf3 = wf_short_ml(df_eth, labels, regimes_eth, SHORT_FEATURES_ETH,
                   'RF', models['RF'], threshold=0.55)
m3, eq3, dd3 = print_wf("RF t=0.55", wf3)


# ============================================================
# OPCION 4: GBM threshold=0.50
# ============================================================
print("\n" + "=" * 70)
print("OPCION 4: GBM threshold=0.50 (features ETH)")
print("=" * 70)

wf4 = wf_short_ml(df_eth, labels, regimes_eth, SHORT_FEATURES_ETH,
                   'GBM', models['GBM'], threshold=0.50)
m4, eq4, dd4 = print_wf("GBM t=0.50", wf4)


# ============================================================
# OPCION 5: GBM threshold=0.45
# ============================================================
print("\n" + "=" * 70)
print("OPCION 5: GBM threshold=0.45 (features ETH)")
print("=" * 70)

wf5 = wf_short_ml(df_eth, labels, regimes_eth, SHORT_FEATURES_ETH,
                   'GBM', models['GBM'], threshold=0.45)
m5, eq5, dd5 = print_wf("GBM t=0.45", wf5)


# ============================================================
# OPCION 6: BTC SHORT follower
# ============================================================
print("\n" + "=" * 70)
print("OPCION 6: BTC SHORT follower (BTC da SHORT -> ETH sigue)")
print("=" * 70)

def detect_btc_short_follower(df_eth, i):
    """SHORT ETH cuando BTC muestra senal bajista + correlacion alta."""
    ts = df_eth.index[i]
    if ts not in df_btc.index:
        return None

    btc_i = df_btc.index.get_loc(ts)
    if btc_i < 30:
        return None

    # BTC esta en BEAR
    if regimes_btc.iloc[btc_i] != 'BEAR':
        return None

    btc_row = df_btc.iloc[btc_i]

    # BTC muestra momentum bajista
    btc_ret5 = float(btc_row.get('ret_5', 0))
    if btc_ret5 > -2:  # BTC debe estar cayendo
        return None

    # RSI BTC bajando pero no oversold
    btc_rsi = float(btc_row.get('rsi14', 50))
    if btc_rsi < 25 or btc_rsi > 55:
        return None

    # Correlacion alta
    c = corr_20.get(ts, 0)
    if pd.isna(c) or c < 0.5:
        return None

    # ETH tambien cayendo
    eth_row = df_eth.iloc[i]
    eth_ret5 = float(eth_row.get('ret_5', 0))
    if eth_ret5 > -1:
        return None

    entry = float(eth_row['close'])
    atr_pct = float(eth_row.get('atr_pct', 2.5))
    sl_pct = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    tp_pct = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)

    return {'direction': 'SHORT', 'setup': 'SHORT_BTC_FOLLOW',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}

wf6 = wf_short_strategy(df_eth, regimes_eth, detect_btc_short_follower, 'BTC_Follow')
m6, eq6, dd6 = print_wf("BTC SHORT follower", wf6)


# ============================================================
# OPCION 7: SHORT momentum (caida fuerte + volumen)
# ============================================================
print("\n" + "=" * 70)
print("OPCION 7: SHORT momentum (caida fuerte ETH+BTC + volumen)")
print("=" * 70)

def detect_short_momentum(df_eth, i):
    """SHORT en momentum bajista fuerte: ETH y BTC cayendo con volumen."""
    if i < 25:
        return None
    row = df_eth.iloc[i]
    c = float(row['close'])
    o = float(row['open'])

    # Vela actual bajista
    if c >= o:
        return None

    # Caida de ETH en ultimas 3 barras
    ret_3 = (c / float(df_eth['close'].iloc[i-3]) - 1) * 100
    if ret_3 > -2:  # Minimo -2% en 3 barras
        return None

    # EMA20 slope negativo
    if float(row.get('ema20_slope', 0)) > -0.3:
        return None

    # Volumen confirma
    if float(row.get('vol_ratio', 1)) < 1.2:
        return None

    # RSI no en extremo oversold
    rsi = float(row.get('rsi14', 50))
    if rsi < 20 or rsi > 60:
        return None

    # BTC tambien cayendo
    ts = df_eth.index[i]
    if ts in df_btc.index:
        btc_i = df_btc.index.get_loc(ts)
        btc_ret = float(df_btc.iloc[btc_i].get('ret_5', 0))
        if btc_ret > 0:  # BTC no esta subiendo
            return None

    entry = c
    atr_pct = float(row.get('atr_pct', 2.5))
    sl_pct = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    tp_pct = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)

    return {'direction': 'SHORT', 'setup': 'SHORT_MOMENTUM',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}

wf7 = wf_short_strategy(df_eth, regimes_eth, detect_short_momentum, 'Momentum')
m7, eq7, dd7 = print_wf("SHORT momentum", wf7)


# ============================================================
# OPCION 8: MeanRev mejorado (parametros ajustados)
# ============================================================
print("\n" + "=" * 70)
print("OPCION 8: MeanRev v2 (RSI > 60, sin limite ADX)")
print("=" * 70)

def detect_short_meanrev_v2(df_eth, i):
    """MeanRev mejorado: RSI > 60 (no 65), vela bajista, BEAR."""
    if i < 25:
        return None
    row = df_eth.iloc[i]
    c = float(row['close'])
    o = float(row['open'])

    # RSI sobrecomprado (mas permisivo que v1)
    rsi = float(row.get('rsi14', 50))
    if rsi < 58:
        return None

    # Vela bajista (cierre < apertura)
    if c >= o:
        return None

    # Price z-score alto (sobre la media)
    pz = float(row.get('price_zscore', 0))
    if pz < 0.5:
        return None

    # EMA20 slope no muy positivo (no comprar contra tendencia fuerte)
    ema_slope = float(row.get('ema20_slope', 0))
    if ema_slope > 1.5:
        return None

    entry = c
    atr_pct = float(row.get('atr_pct', 2.5))
    sl_pct = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    tp_pct = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)

    return {'direction': 'SHORT', 'setup': 'SHORT_MEANREV_V2',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}

wf8 = wf_short_strategy(df_eth, regimes_eth, detect_short_meanrev_v2, 'MeanRev_v2')
m8, eq8, dd8 = print_wf("MeanRev v2", wf8)


# ============================================================
# OPCION 9: MeanRev + ML filter (MeanRev con prob ML > 0.45)
# ============================================================
print("\n" + "=" * 70)
print("OPCION 9: MeanRev + ML filter (prob RF > 0.45)")
print("=" * 70)

def wf_rules_plus_ml(df, labels, regimes, features, constructor,
                     detect_fn, ml_threshold=0.45):
    """Rules genera senal, ML filtra."""
    from sklearn.preprocessing import StandardScaler
    results = []
    all_trades = []

    for start_s, end_s in WF_FOLDS:
        period = f"{start_s[:7]}/{end_s[5:7]}"
        trades = []

        train_mask = df.index < start_s
        y_train = labels[train_mask]
        bear_train = regimes[train_mask] == 'BEAR'
        valid = y_train.notna() & bear_train
        X_train = df.loc[train_mask, features][valid].fillna(0)
        y_train_v = y_train[valid]

        model = None
        scaler = None
        if len(X_train) >= 200 and y_train_v.sum() >= 10:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train.values)
            model = constructor()
            try:
                model.fit(X_train_s, y_train_v)
            except:
                model = None

        for i in range(30, len(df)):
            ts = df.index[i]
            if ts < pd.Timestamp(start_s, tz='UTC') or ts > pd.Timestamp(end_s, tz='UTC'):
                continue
            if regimes.iloc[i] != 'BEAR':
                continue
            if i + 16 >= len(df):
                continue

            # Primero: regla genera senal
            trade = detect_fn(df, i)
            if trade is None:
                continue

            # Segundo: ML filtra
            if model is not None:
                row = df.iloc[i]
                x = pd.DataFrame([row[features].fillna(0).values], columns=features)
                x_s = scaler.transform(x)
                prob = model.predict_proba(x_s)[0][1]
                if prob < ml_threshold:
                    continue

            out = sim_short(df, i, trade['entry'],
                           trade['tp_pct'], trade['sl_pct'], max_bars=16)
            trades.append({
                'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                'setup': trade.get('setup', 'FILTERED'),
            })

        m = metrics(trades, period)
        ok = (m['n'] >= 2 and m['wr'] > 0.35 and m['pf'] > 0.9)
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                        'pf': m['pf'], 'ok': ok})
        all_trades.extend(trades)

    folds_ok = sum(1 for r in results if r['ok'])
    return {'folds': results, 'folds_ok': folds_ok, 'all_trades': all_trades}

wf9 = wf_rules_plus_ml(df_eth, labels, regimes_eth, SHORT_FEATURES_ETH,
                        models['RF'], detect_short_meanrev, ml_threshold=0.45)
m9, eq9, dd9 = print_wf("MeanRev + RF filter", wf9)


# ============================================================
# OPCION 10: Combo MeanRev + Momentum
# ============================================================
print("\n" + "=" * 70)
print("OPCION 10: MeanRev + Momentum combinados")
print("=" * 70)

def detect_short_combo(df_eth, i):
    """Intenta MeanRev primero, luego Momentum."""
    t = detect_short_meanrev(df_eth, i)
    if t is not None:
        return t
    return detect_short_momentum(df_eth, i)

wf10 = wf_short_strategy(df_eth, regimes_eth, detect_short_combo, 'Combo')
m10, eq10, dd10 = print_wf("MeanRev + Momentum", wf10)


# ============================================================
# OPCION 11: SHORT ETH/BTC divergence
# ============================================================
print("\n" + "=" * 70)
print("OPCION 11: ETH/BTC divergence SHORT")
print("=" * 70)

def detect_short_divergence(df_eth, i):
    """SHORT cuando ETH sube vs BTC (divergencia) en BEAR -> reversion."""
    if i < 25:
        return None
    row = df_eth.iloc[i]

    # ETH/BTC ratio subiendo (ETH outperforming)
    ethbtc_slope = float(row.get('ethbtc_slope_5', 0))
    if ethbtc_slope < 1.0:  # ETH debe estar subiendo vs BTC
        return None

    # Pero ETH en BEAR general -> la subida es rebote
    # Price zscore positivo (sobre media)
    pz = float(row.get('price_zscore', 0))
    if pz < 0:
        return None

    # RSI ETH subido (rebote)
    rsi = float(row.get('rsi14', 50))
    if rsi < 45 or rsi > 75:
        return None

    # BTC sigue bajando
    btc_ret = float(row.get('btc_ret_5', 0))
    if btc_ret > 0:  # BTC no esta bajando
        return None

    entry = float(row['close'])
    atr_pct = float(row.get('atr_pct', 2.5))
    sl_pct = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    tp_pct = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)

    return {'direction': 'SHORT', 'setup': 'SHORT_DIVERGENCE',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}

wf11 = wf_short_strategy(df_eth, regimes_eth, detect_short_divergence, 'Divergence')
m11, eq11, dd11 = print_wf("ETH/BTC divergence", wf11)


# ============================================================
# OPCION 12: SHORT TP/SL mas agresivo (TP=4%, SL=2.5%)
# ============================================================
print("\n" + "=" * 70)
print("OPCION 12: MeanRev con TP/SL mas amplio (TP=2x ATR, SL=1x ATR)")
print("=" * 70)

def detect_short_meanrev_wide(df_eth, i):
    """MeanRev con TP/SL mas amplios para capturar movimientos grandes en BEAR."""
    if i < 25:
        return None
    row = df_eth.iloc[i]
    c = float(row['close'])
    o = float(row['open'])

    rsi = float(row.get('rsi14', 50))
    if rsi < 60:
        return None
    if c >= o:
        return None

    adx = float(row.get('adx14', 20))
    if adx > 35:
        return None

    entry = c
    atr_pct = float(row.get('atr_pct', 2.5))
    # Mas amplio: TP = 3x ATR, SL = 1.2x ATR (mejor ratio)
    sl_pct = max(min(atr_pct / 100 * 1.2, 0.04), 0.012)
    tp_pct = max(min(atr_pct / 100 * 3.0, 0.10), 0.030)

    return {'direction': 'SHORT', 'setup': 'SHORT_MEANREV_WIDE',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}

wf12 = wf_short_strategy(df_eth, regimes_eth, detect_short_meanrev_wide, 'MeanRev_Wide')
m12, eq12, dd12 = print_wf("MeanRev TP/SL amplio", wf12)


# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "=" * 70)
print("RESUMEN -- ETH SHORT v2 (solo datos 2020-2025)")
print("=" * 70)

all_options = [
    ("1. MeanRev solo", wf1, m1, eq1, dd1),
    ("2. RF t=0.50", wf2, m2, eq2, dd2),
    ("3. RF t=0.55", wf3, m3, eq3, dd3),
    ("4. GBM t=0.50", wf4, m4, eq4, dd4),
    ("5. GBM t=0.45", wf5, m5, eq5, dd5),
    ("6. BTC follow", wf6, m6, eq6, dd6),
    ("7. Momentum", wf7, m7, eq7, dd7),
    ("8. MeanRev v2", wf8, m8, eq8, dd8),
    ("9. MeanRev+RF", wf9, m9, eq9, dd9),
    ("10. MR+Momentum", wf10, m10, eq10, dd10),
    ("11. Divergence", wf11, m11, eq11, dd11),
    ("12. MR TP/SL wide", wf12, m12, eq12, dd12),
]

print(f"\n  {'Opcion':<20} | {'WF':>5} | {'N':>5} | {'WR':>6} | {'PF':>6} | {'$1K->':>7} | {'DD':>6}")
print(f"  {'-'*75}")
for name, wf, m, eq, dd in all_options:
    passed = "**" if wf['folds_ok'] >= 7 and m['pf'] >= 1.2 else "  "
    print(f"  {passed}{name:<18} | {wf['folds_ok']:>2}/12 | {m['n']:>5} | "
          f"{m['wr']:.1%} | {m['pf']:.2f} | ${1000*eq:>6.0f} | {dd:.1%}")

# Encontrar mejores
approved = [(n, wf, m, eq, dd) for n, wf, m, eq, dd in all_options
            if wf['folds_ok'] >= 7 and m['pf'] >= 1.2]
marginal = [(n, wf, m, eq, dd) for n, wf, m, eq, dd in all_options
            if wf['folds_ok'] >= 6 and m['pf'] >= 1.0 and (n, wf, m, eq, dd) not in approved]

if approved:
    print(f"\n  APROBADAS (WF>=7/12, PF>=1.2):")
    for n, wf, m, eq, dd in approved:
        print(f"    {n}: WF {wf['folds_ok']}/12, PF={m['pf']:.2f}, WR={m['wr']:.1%}")
    best = max(approved, key=lambda x: x[2]['pf'])
    print(f"\n  MEJOR: {best[0]} (PF={best[2]['pf']:.2f})")
else:
    print(f"\n  NINGUNA APROBADA (WF>=7/12 + PF>=1.2)")

if marginal:
    print(f"\n  MARGINALES (WF>=6/12, PF>=1.0):")
    for n, wf, m, eq, dd in marginal:
        print(f"    {n}: WF {wf['folds_ok']}/12, PF={m['pf']:.2f}, WR={m['wr']:.1%}")

print("\n" + "=" * 70)
