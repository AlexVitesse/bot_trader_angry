"""
evaluate_eth_v2.py — ETH: 5 estrategias optimizadas
=====================================================
Basado en lo que SI funciono en la ronda anterior:
  - BTC-follower: WR 46.9%, PF 1.12, MaxDD 25% (casi pasa)
  - Breakout B: WR 45.5%, PF 1.29 (buen edge, pocos trades)
  - ML folds buenos en BULL: 2023-H1 WR=58%, 2024-H1 WR=50%

5 opciones:
  1. BTC-follower TUNED (relajar filtros, mas senales BTC)
  2. Breakout B adaptado ETH (filtros mas amplios por volatilidad)
  3. HIBRIDO (follower + breakout ETH combinados)
  4. REGIME-CONDITIONAL (solo ETH LONG cuando BTC = BULL)
  5. TP/SL adaptados a volatilidad ETH (4H con TP/SL mas amplios)

Usage:
  python evaluate_eth_v2.py
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
    sim_trade_fixed, metrics, print_metrics,
    WF_FOLDS, OOS_START, OOS_END, COMMISSION,
)


# ============================================================
# SHARED
# ============================================================
REGIME_DEAD_ZONE = 0.02

def detect_regime(row):
    ema20 = row.get('ema20_1d', None)
    ema50 = row.get('ema50_1d', None)
    ema200 = row.get('ema200_1d', None)
    close = row.get('close', None)
    if ema20 is None or ema50 is None or pd.isna(ema20) or pd.isna(ema50):
        return 'RANGE'
    dist = (ema20 - ema50) / ema50
    if dist > REGIME_DEAD_ZONE:
        return 'BULL'
    elif dist < -REGIME_DEAD_ZONE:
        if ema200 is not None and close is not None and not pd.isna(ema200):
            if close > ema200:
                return 'RANGE'
        if close is not None and not pd.isna(close):
            if close > ema50:
                return 'RANGE'
        return 'BEAR'
    return 'RANGE'


def add_extra_features(df):
    df = df.copy()
    c, v = df['close'], df['volume']
    df['rsi_slope'] = df['rsi14'].diff(3)
    vol_ma5 = v.rolling(5).mean()
    vol_ma20 = v.rolling(20).mean()
    df['vol_slope'] = (vol_ma5 / vol_ma20.replace(0, np.nan) - 1) * 100
    df['ret_10'] = c.pct_change(10) * 100
    up = (c > c.shift(1)).astype(int)
    df['consec_up'] = up.rolling(8).sum()
    return df


def compute_cross_data(df_eth, df_btc):
    """Pre-calcular datos cruzados ETH-BTC."""
    eth_ret = df_eth['close'].pct_change()
    btc_close = df_btc['close'].reindex(df_eth.index, method='ffill')
    btc_ret = btc_close.pct_change()

    corr_20 = eth_ret.rolling(20).corr(btc_ret)
    ratio = df_eth['close'] / btc_close.replace(0, np.nan)
    ratio_slope_5 = ratio.pct_change(5) * 100

    return corr_20, ratio_slope_5, btc_close


def equity_stats(trades):
    """Calcular equity curve, max DD."""
    if not trades:
        return 0, 0
    cum = 1.0; peak = 1.0; max_dd = 0
    for t in sorted(trades, key=lambda x: x['ts']):
        cum *= (1 + t['pnl_pct'])
        peak = max(peak, cum)
        dd = (peak - cum) / peak
        max_dd = max(max_dd, dd)
    return cum, max_dd


# ============================================================
# SETUPS COMPARTIDOS
# ============================================================

def detect_breakout_b_btc(df_btc, i):
    """Breakout B en BTC (parametros originales)."""
    if i < 25:
        return None
    row = df_btc.iloc[i]
    high20 = float(df_btc['high'].iloc[i-20:i].max())
    if row['close'] <= high20:
        return None
    if row.get('vol_ratio', 1) < 1.8:
        return None
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > 2.5:
        return None
    recent_bb = df_btc['bb_width'].iloc[i-5:i]
    if (recent_bb < 4.0).sum() < 3:
        return None
    if df_btc['adx14'].iloc[i-3:i].mean() > 28:
        return None
    entry = float(row['close'])
    sl_raw = float(df_btc['low'].iloc[i-5:i].min()) * 0.997
    sl_pct = (entry - sl_raw) / entry
    if sl_pct < 0.005 or sl_pct > 0.04:
        return None
    return {'setup': 'BRK_BTC'}


def detect_pullback_btc(df_btc, i):
    """Pullback EMA20 en BTC."""
    if i < 25:
        return None
    row = df_btc.iloc[i]
    prev = df_btc.iloc[i-1]
    c, o = float(row['close']), float(row['open'])
    ema20 = float(row.get('ema20', 0))
    ema50 = float(row.get('ema50', 0))
    if ema20 <= 0 or ema50 <= 0 or c < ema50:
        return None
    dist = (c - ema20) / ema20
    if dist < -0.005 or dist > 0.015:
        return None
    if float(row.get('adx14', 0)) < 15:
        return None
    rsi = float(row.get('rsi14', 50))
    if rsi < 33 or rsi > 58:
        return None
    if c <= o or float(prev['close']) >= float(prev['open']):
        return None
    if float(row.get('vol_ratio', 1)) > 2.0:
        return None
    return {'setup': 'PB_BTC'}


def detect_breakout_eth_adapted(df_eth, i, vol_min=1.3, bb_max=5.5, adx_max=32, bar_max=3.5):
    """Breakout B adaptado a ETH: filtros mas relajados."""
    if i < 25:
        return None
    row = df_eth.iloc[i]
    high20 = float(df_eth['high'].iloc[i-20:i].max())
    if row['close'] <= high20:
        return None
    if row.get('vol_ratio', 1) < vol_min:
        return None
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > bar_max:
        return None
    recent_bb = df_eth['bb_width'].iloc[i-5:i]
    if (recent_bb < bb_max).sum() < 2:  # 2/5 en vez de 3/5
        return None
    if df_eth['adx14'].iloc[i-3:i].mean() > adx_max:
        return None
    entry = float(row['close'])
    sl_raw = float(df_eth['low'].iloc[i-5:i].min()) * 0.995  # 0.5% buffer
    sl_pct = (entry - sl_raw) / entry
    if sl_pct < 0.005 or sl_pct > 0.06:  # mas ancho para ETH
        return None
    tp_pct = sl_pct * 1.5
    return {'direction': 'LONG', 'setup': 'BRK_ETH',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}


def detect_pullback_eth(df_eth, i):
    """Pullback EMA20 en ETH con filtros relajados."""
    if i < 25:
        return None
    row = df_eth.iloc[i]
    prev = df_eth.iloc[i-1]
    c, o = float(row['close']), float(row['open'])
    ema20 = float(row.get('ema20', 0))
    ema50 = float(row.get('ema50', 0))
    if ema20 <= 0 or ema50 <= 0 or c < ema50:
        return None
    dist = (c - ema20) / ema20
    if dist < -0.008 or dist > 0.020:  # rango mas amplio para ETH
        return None
    if float(row.get('adx14', 0)) < 12:  # menos estricto
        return None
    rsi = float(row.get('rsi14', 50))
    if rsi < 30 or rsi > 60:  # rango mas amplio
        return None
    if c <= o:
        return None
    if float(prev['close']) >= float(prev['open']):
        return None
    if float(row.get('vol_ratio', 1)) > 2.5:  # mas permisivo
        return None
    atr_pct = float(row.get('atr_pct', 2.0))
    entry = c
    sl_pct = max(min(atr_pct / 100 * 1.2, 0.04), 0.012)  # mas ancho
    tp_pct = sl_pct * 1.5
    return {'direction': 'LONG', 'setup': 'PB_ETH',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}


# ============================================================
# OPCION 1: BTC-FOLLOWER TUNED
# ============================================================

def run_follower_tuned(df_eth, df_btc, corr_20, ratio_slope_5,
                       corr_min=0.5, div_min=-3.0, need_eth_trend=False):
    """
    BTC-follower con filtros relajados:
    - corr_min: 0.5 en vez de 0.7
    - div_min: -3% en vez de -2%
    - need_eth_trend: False (no exigir ema20>ema50 en ETH)
    """
    results = []
    all_trades = []

    for start_s, end_s in WF_FOLDS:
        period = f"{start_s[:7]}/{end_s[5:7]}"
        trades = []

        for i in range(30, len(df_btc)):
            ts = df_btc.index[i]
            if ts < pd.Timestamp(start_s, tz='UTC') or ts > pd.Timestamp(end_s, tz='UTC'):
                continue

            regime_btc = detect_regime(df_btc.iloc[i])
            btc_signal = None
            if regime_btc == 'BULL':
                btc_signal = detect_breakout_b_btc(df_btc, i)
                if btc_signal is None:
                    btc_signal = detect_pullback_btc(df_btc, i)
            elif regime_btc == 'RANGE':
                btc_signal = detect_breakout_b_btc(df_btc, i)

            if btc_signal is None:
                continue

            # Encontrar barra ETH
            if ts not in df_eth.index:
                idx_pos = df_eth.index.searchsorted(ts)
                if idx_pos >= len(df_eth):
                    continue
                ts_eth = df_eth.index[idx_pos]
                if abs((ts_eth - ts).total_seconds()) > 4 * 3600:
                    continue
            else:
                ts_eth = ts

            eth_i = df_eth.index.get_loc(ts_eth)
            if eth_i < 25 or eth_i + 16 >= len(df_eth):
                continue

            # Filtro correlacion (relajado)
            c = corr_20.get(ts_eth, 0)
            if pd.isna(c) or c < corr_min:
                continue

            # Filtro divergencia (relajado)
            rs = ratio_slope_5.get(ts_eth, 0)
            if pd.isna(rs) or rs < div_min:
                continue

            # Filtro tendencia ETH (opcional)
            if need_eth_trend:
                ema20_eth = df_eth.iloc[eth_i].get('ema20', 0)
                ema50_eth = df_eth.iloc[eth_i].get('ema50', 0)
                if ema20_eth <= 0 or ema50_eth <= 0 or ema20_eth < ema50_eth:
                    continue

            # TP/SL basado en ATR de ETH
            eth_row = df_eth.iloc[eth_i]
            entry = float(eth_row['close'])
            atr_pct = float(eth_row.get('atr_pct', 2.5))
            sl_pct = max(min(atr_pct / 100 * 1.2, 0.04), 0.012)
            tp_pct = sl_pct * 1.5

            out = sim_trade_fixed(df_eth, eth_i, entry, tp_pct, sl_pct, max_bars=16)
            trades.append({
                'outcome': out[0], 'pnl_pct': out[2], 'ts': ts_eth,
                'setup': f"FOLLOW_{btc_signal['setup']}",
            })

        m = metrics(trades, period)
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                        'pf': m['pf'], 'ok': ok})
        all_trades.extend(trades)

    folds_ok = sum(1 for r in results if r['ok'])
    return {'folds': results, 'folds_ok': folds_ok, 'all_trades': all_trades}


# ============================================================
# OPCION 2: BREAKOUT B ADAPTADO ETH
# ============================================================

def run_breakout_adapted(df_eth):
    """Breakout B con parametros adaptados a la mayor volatilidad de ETH."""
    results = []
    all_trades = []

    for start_s, end_s in WF_FOLDS:
        period = f"{start_s[:7]}/{end_s[5:7]}"
        trades = []

        for i in range(30, len(df_eth)):
            ts = df_eth.index[i]
            if ts < pd.Timestamp(start_s, tz='UTC') or ts > pd.Timestamp(end_s, tz='UTC'):
                continue

            regime = detect_regime(df_eth.iloc[i])
            if regime == 'BEAR':
                continue

            trade = detect_breakout_eth_adapted(df_eth, i)
            if trade is None and regime == 'BULL':
                trade = detect_pullback_eth(df_eth, i)

            if trade is None:
                continue

            out = sim_trade_fixed(df_eth, i, trade['entry'],
                                  trade['tp_pct'], trade['sl_pct'], max_bars=16)
            trades.append({
                'outcome': out[0], 'pnl_pct': out[2], 'ts': ts,
                'setup': trade['setup'],
            })

        m = metrics(trades, period)
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                        'pf': m['pf'], 'ok': ok})
        all_trades.extend(trades)

    folds_ok = sum(1 for r in results if r['ok'])
    return {'folds': results, 'folds_ok': folds_ok, 'all_trades': all_trades}


# ============================================================
# OPCION 3: HIBRIDO (follower + breakout ETH)
# ============================================================

def run_hybrid(df_eth, df_btc, corr_20, ratio_slope_5):
    """Combinar: BTC-follower tuned + Breakout B ETH adaptado."""
    results = []
    all_trades = []

    for start_s, end_s in WF_FOLDS:
        period = f"{start_s[:7]}/{end_s[5:7]}"
        trades = []
        traded_bars = set()  # evitar trades duplicados en la misma barra

        for i in range(30, len(df_eth)):
            ts_eth = df_eth.index[i]
            if ts_eth < pd.Timestamp(start_s, tz='UTC') or ts_eth > pd.Timestamp(end_s, tz='UTC'):
                continue
            if i in traded_bars:
                continue

            eth_i = i
            trade = None
            source = None

            # Primero: buscar senal BTC-follower
            if ts_eth in df_btc.index:
                btc_i = df_btc.index.get_loc(ts_eth)
                if btc_i >= 30:
                    regime_btc = detect_regime(df_btc.iloc[btc_i])
                    btc_signal = None
                    if regime_btc == 'BULL':
                        btc_signal = detect_breakout_b_btc(df_btc, btc_i)
                        if btc_signal is None:
                            btc_signal = detect_pullback_btc(df_btc, btc_i)
                    elif regime_btc == 'RANGE':
                        btc_signal = detect_breakout_b_btc(df_btc, btc_i)

                    if btc_signal is not None:
                        c = corr_20.get(ts_eth, 0)
                        rs = ratio_slope_5.get(ts_eth, 0)
                        if not pd.isna(c) and c >= 0.5 and not pd.isna(rs) and rs >= -3.0:
                            eth_row = df_eth.iloc[eth_i]
                            entry = float(eth_row['close'])
                            atr_pct = float(eth_row.get('atr_pct', 2.5))
                            sl_pct = max(min(atr_pct / 100 * 1.2, 0.04), 0.012)
                            tp_pct = sl_pct * 1.5
                            trade = {'direction': 'LONG', 'setup': f"FOLLOW_{btc_signal['setup']}",
                                     'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}
                            source = 'follower'

            # Si no hay follower, buscar breakout ETH propio
            if trade is None:
                regime_eth = detect_regime(df_eth.iloc[eth_i])
                if regime_eth != 'BEAR':
                    trade = detect_breakout_eth_adapted(df_eth, eth_i)
                    if trade is not None:
                        source = 'breakout'

            if trade is None:
                continue

            if eth_i + 16 >= len(df_eth):
                continue

            out = sim_trade_fixed(df_eth, eth_i, trade['entry'],
                                  trade['tp_pct'], trade['sl_pct'], max_bars=16)
            trades.append({
                'outcome': out[0], 'pnl_pct': out[2], 'ts': ts_eth,
                'setup': trade['setup'], 'source': source,
            })
            traded_bars.add(i)

        m = metrics(trades, period)
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                        'pf': m['pf'], 'ok': ok})
        all_trades.extend(trades)

    folds_ok = sum(1 for r in results if r['ok'])
    return {'folds': results, 'folds_ok': folds_ok, 'all_trades': all_trades}


# ============================================================
# OPCION 4: REGIME-CONDITIONAL (solo BTC BULL)
# ============================================================

def run_regime_conditional(df_eth, df_btc, corr_20, ratio_slope_5):
    """
    Solo operar ETH cuando BTC esta en BULL.
    Combinar: follower + breakout ETH, pero SOLO en regimen BULL de BTC.
    """
    results = []
    all_trades = []

    for start_s, end_s in WF_FOLDS:
        period = f"{start_s[:7]}/{end_s[5:7]}"
        trades = []

        for i in range(30, len(df_eth)):
            ts_eth = df_eth.index[i]
            if ts_eth < pd.Timestamp(start_s, tz='UTC') or ts_eth > pd.Timestamp(end_s, tz='UTC'):
                continue

            # Verificar regimen BTC = BULL
            if ts_eth in df_btc.index:
                btc_i = df_btc.index.get_loc(ts_eth)
            else:
                idx_pos = df_btc.index.searchsorted(ts_eth)
                btc_i = max(0, idx_pos - 1)

            if btc_i < 30 or btc_i >= len(df_btc):
                continue

            regime_btc = detect_regime(df_btc.iloc[btc_i])
            if regime_btc != 'BULL':
                continue

            # En BULL: breakout ETH o pullback ETH
            trade = detect_breakout_eth_adapted(df_eth, i)
            if trade is None:
                trade = detect_pullback_eth(df_eth, i)

            # Tambien: follower si BTC da senal
            if trade is None and ts_eth in df_btc.index:
                btc_signal = detect_breakout_b_btc(df_btc, btc_i)
                if btc_signal is None:
                    btc_signal = detect_pullback_btc(df_btc, btc_i)
                if btc_signal is not None:
                    c = corr_20.get(ts_eth, 0)
                    if not pd.isna(c) and c >= 0.4:
                        eth_row = df_eth.iloc[i]
                        entry = float(eth_row['close'])
                        atr_pct = float(eth_row.get('atr_pct', 2.5))
                        sl_pct = max(min(atr_pct / 100 * 1.2, 0.04), 0.012)
                        tp_pct = sl_pct * 1.5
                        trade = {'direction': 'LONG', 'setup': f"FOLLOW_{btc_signal['setup']}",
                                 'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}

            if trade is None or i + 16 >= len(df_eth):
                continue

            out = sim_trade_fixed(df_eth, i, trade['entry'],
                                  trade['tp_pct'], trade['sl_pct'], max_bars=16)
            trades.append({
                'outcome': out[0], 'pnl_pct': out[2], 'ts': ts_eth,
                'setup': trade['setup'],
            })

        m = metrics(trades, period)
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                        'pf': m['pf'], 'ok': ok})
        all_trades.extend(trades)

    folds_ok = sum(1 for r in results if r['ok'])
    return {'folds': results, 'folds_ok': folds_ok, 'all_trades': all_trades}


# ============================================================
# OPCION 5: TP/SL ADAPTADOS (mas amplios para volatilidad ETH)
# ============================================================

def run_tpsl_adapted(df_eth, df_btc, corr_20, ratio_slope_5):
    """
    Mismo hibrido pero con TP/SL escaleados a volatilidad ETH.
    ETH ~1.5x mas volatil que BTC -> TP/SL mas amplios.
    Usar ATR multiplicadores mayores: sl=1.5*ATR, tp=2.5*ATR.
    """
    results = []
    all_trades = []

    for start_s, end_s in WF_FOLDS:
        period = f"{start_s[:7]}/{end_s[5:7]}"
        trades = []
        traded_bars = set()

        for i in range(30, len(df_eth)):
            ts_eth = df_eth.index[i]
            if ts_eth < pd.Timestamp(start_s, tz='UTC') or ts_eth > pd.Timestamp(end_s, tz='UTC'):
                continue
            if i in traded_bars:
                continue

            trade = None

            # Follower
            if ts_eth in df_btc.index:
                btc_i = df_btc.index.get_loc(ts_eth)
                if btc_i >= 30:
                    regime_btc = detect_regime(df_btc.iloc[btc_i])
                    btc_signal = None
                    if regime_btc in ('BULL', 'RANGE'):
                        btc_signal = detect_breakout_b_btc(df_btc, btc_i)
                        if btc_signal is None and regime_btc == 'BULL':
                            btc_signal = detect_pullback_btc(df_btc, btc_i)

                    if btc_signal is not None:
                        c = corr_20.get(ts_eth, 0)
                        if not pd.isna(c) and c >= 0.5:
                            trade = {'setup': f"FOLLOW_{btc_signal['setup']}"}

            # Breakout ETH
            if trade is None:
                regime_eth = detect_regime(df_eth.iloc[i])
                if regime_eth != 'BEAR':
                    brk = detect_breakout_eth_adapted(df_eth, i)
                    if brk is not None:
                        trade = {'setup': 'BRK_ETH'}

            if trade is None or i + 18 >= len(df_eth):
                continue

            # TP/SL adaptados: mas amplios
            eth_row = df_eth.iloc[i]
            entry = float(eth_row['close'])
            atr_pct = float(eth_row.get('atr_pct', 2.5))

            # SL = 1.5 * ATR (vs 1.0-1.2 en BTC), capped 1.5%-5%
            sl_pct = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
            # TP = 2.5 * ATR (vs 1.5 en BTC) -> RR ~1.67
            tp_pct = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)

            out = sim_trade_fixed(df_eth, i, entry, tp_pct, sl_pct, max_bars=18)
            trades.append({
                'outcome': out[0], 'pnl_pct': out[2], 'ts': ts_eth,
                'setup': trade['setup'],
            })
            traded_bars.add(i)

        m = metrics(trades, period)
        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        results.append({'period': period, 'n': m['n'], 'wr': m['wr'],
                        'pf': m['pf'], 'ok': ok})
        all_trades.extend(trades)

    folds_ok = sum(1 for r in results if r['ok'])
    return {'folds': results, 'folds_ok': folds_ok, 'all_trades': all_trades}


# ============================================================
# PRINT / ANALYSIS HELPERS
# ============================================================

def print_wf(wf, label):
    print(f"\n  {'Periodo':<14} | {'N':>4} | {'WR':>7} | {'PF':>6} | OK")
    print("  " + "-" * 45)
    for r in wf['folds']:
        ok_s = '+' if r['ok'] else '-'
        wr_s = f"{r['wr']:.1%}" if r['n'] > 0 else 'n/a'
        pf_s = f"{r['pf']:.2f}" if r['n'] > 0 else 'n/a'
        print(f"  {r['period']:<14} | {r['n']:>4} | {wr_s:>7} | {pf_s:>6} | {ok_s}")
    print(f"\n  Folds OK: {wf['folds_ok']}/12")


def print_oos(trades, label):
    oos = [t for t in trades if OOS_START <= str(t['ts'])[:10] <= OOS_END]
    m = metrics(oos, label)
    if m['n'] > 0:
        cum, max_dd = equity_stats(oos)
        print(f"  OOS: N={m['n']} | WR={m['wr']:.1%} | PF={m['pf']:.2f} | "
              f"{m['trades_pm']:.1f}t/m | ${1000*cum:.0f} | DD={max_dd:.1%}")

        # Breakdown por setup
        setups = set(t.get('setup', '?') for t in oos)
        if len(setups) > 1:
            for s in sorted(setups):
                st = [t for t in oos if t.get('setup') == s]
                ms = metrics(st, s)
                if ms['n'] > 0:
                    print(f"    {s}: N={ms['n']} WR={ms['wr']:.1%} PF={ms['pf']:.2f}")
    else:
        print("  OOS: Sin trades")
    return m


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("ETH V2 — 5 ESTRATEGIAS OPTIMIZADAS")
    print("=" * 70)

    # Cargar datos
    print("\nCargando datos...")
    df_eth_raw = load_pair_4h('ETH')
    df_btc_raw = load_btc_4h()

    df_eth = compute_features_4h(df_eth_raw)
    df_eth = add_extra_features(df_eth)
    df_daily_eth = compute_macro_daily(df_eth)
    df_eth = merge_daily_to_4h(df_eth, df_daily_eth)

    df_btc = compute_features_4h(df_btc_raw)
    df_btc = add_extra_features(df_btc)
    df_daily_btc = compute_macro_daily(df_btc)
    df_btc = merge_daily_to_4h(df_btc, df_daily_btc)

    corr_20, ratio_slope_5, _ = compute_cross_data(df_eth, df_btc)

    print(f"  ETH: {len(df_eth)} bars | BTC: {len(df_btc)} bars")

    # Regimenes BTC
    btc_regimes = df_btc.apply(lambda r: detect_regime(r), axis=1)
    print(f"  BTC BULL: {(btc_regimes == 'BULL').mean():.1%} | "
          f"BEAR: {(btc_regimes == 'BEAR').mean():.1%} | "
          f"RANGE: {(btc_regimes == 'RANGE').mean():.1%}")

    summary = []

    # ---- OPCION 1: BTC-FOLLOWER TUNED ----
    print(f"\n{'='*70}")
    print("OPCION 1: BTC-FOLLOWER TUNED (corr>=0.5, div>=-3%)")
    print(f"{'='*70}")

    wf1 = run_follower_tuned(df_eth, df_btc, corr_20, ratio_slope_5,
                              corr_min=0.5, div_min=-3.0, need_eth_trend=False)
    print_wf(wf1, 'Follower Tuned')
    m1 = print_oos(wf1['all_trades'], 'Follower Tuned')
    passed1 = wf1['folds_ok'] >= 7 and m1.get('pf', 0) >= 1.2
    print(f"  {'APROBADO' if passed1 else 'RECHAZADO'}")
    summary.append(('1. Follower Tuned', wf1['folds_ok'], m1))

    # ---- OPCION 2: BREAKOUT B ADAPTADO ----
    print(f"\n{'='*70}")
    print("OPCION 2: BREAKOUT B ADAPTADO ETH (vol>=1.3, BB<5.5, ADX<32)")
    print(f"{'='*70}")

    wf2 = run_breakout_adapted(df_eth)
    print_wf(wf2, 'Breakout Adapted')
    m2 = print_oos(wf2['all_trades'], 'Breakout Adapted')
    passed2 = wf2['folds_ok'] >= 7 and m2.get('pf', 0) >= 1.2
    print(f"  {'APROBADO' if passed2 else 'RECHAZADO'}")
    summary.append(('2. Breakout Adapted', wf2['folds_ok'], m2))

    # ---- OPCION 3: HIBRIDO ----
    print(f"\n{'='*70}")
    print("OPCION 3: HIBRIDO (Follower + Breakout ETH)")
    print(f"{'='*70}")

    wf3 = run_hybrid(df_eth, df_btc, corr_20, ratio_slope_5)
    print_wf(wf3, 'Hibrido')
    m3 = print_oos(wf3['all_trades'], 'Hibrido')
    passed3 = wf3['folds_ok'] >= 7 and m3.get('pf', 0) >= 1.2
    print(f"  {'APROBADO' if passed3 else 'RECHAZADO'}")
    summary.append(('3. Hibrido', wf3['folds_ok'], m3))

    # ---- OPCION 4: REGIME-CONDITIONAL ----
    print(f"\n{'='*70}")
    print("OPCION 4: REGIME-CONDITIONAL (solo BTC BULL)")
    print(f"{'='*70}")

    wf4 = run_regime_conditional(df_eth, df_btc, corr_20, ratio_slope_5)
    print_wf(wf4, 'Regime BULL')
    m4 = print_oos(wf4['all_trades'], 'Regime BULL')
    passed4 = wf4['folds_ok'] >= 7 and m4.get('pf', 0) >= 1.2
    print(f"  {'APROBADO' if passed4 else 'RECHAZADO'}")
    summary.append(('4. Regime BULL', wf4['folds_ok'], m4))

    # ---- OPCION 5: TP/SL ADAPTADOS ----
    print(f"\n{'='*70}")
    print("OPCION 5: TP/SL ADAPTADOS (SL=1.5*ATR, TP=2.5*ATR)")
    print(f"{'='*70}")

    wf5 = run_tpsl_adapted(df_eth, df_btc, corr_20, ratio_slope_5)
    print_wf(wf5, 'TP/SL Adapt')
    m5 = print_oos(wf5['all_trades'], 'TP/SL Adapt')
    passed5 = wf5['folds_ok'] >= 7 and m5.get('pf', 0) >= 1.2
    print(f"  {'APROBADO' if passed5 else 'RECHAZADO'}")
    summary.append(('5. TP/SL Adapt', wf5['folds_ok'], m5))

    # ---- RESUMEN ----
    print(f"\n\n{'='*70}")
    print("RESUMEN — ETH V2: 5 ESTRATEGIAS")
    print(f"{'='*70}")

    print(f"\n  {'Opcion':<22} | {'WF':>5} | {'N':>5} | {'WR':>7} | {'PF':>6} | {'t/m':>5} | Veredicto")
    print("  " + "-" * 75)

    any_passed = False
    best_name = None
    best_pf = 0

    for name, folds_ok, m in summary:
        n = m['n']; wr = m['wr']; pf = m['pf']; tpm = m['trades_pm']
        passed = folds_ok >= 7 and pf >= 1.2
        if passed:
            any_passed = True
        # Tambien considerar "marginal" (folds>=6 y PF>=1.1)
        if pf > best_pf and n >= 10:
            best_pf = pf
            best_name = name
        verd = 'APROBADO' if passed else ('MARGINAL' if folds_ok >= 6 and pf >= 1.1 else 'RECHAZADO')
        print(f"  {name:<22} | {folds_ok:>2}/12 | {n:>5} | {wr:.1%} | {pf:.2f} | {tpm:.1f} | {verd}")

    print(f"\n  {'='*70}")
    if any_passed:
        print(f"  ETH APROBADO — {best_name}")
    else:
        # Check marginal
        marginals = [(name, fok, m) for name, fok, m in summary
                     if fok >= 6 and m['pf'] >= 1.1 and m['n'] >= 10]
        if marginals:
            best_m = max(marginals, key=lambda x: x[2]['pf'])
            print(f"  ETH MARGINAL — {best_m[0]} (WF {best_m[1]}/12, PF {best_m[2]['pf']:.2f})")
            print(f"  Posible candidato con tuning adicional o mas datos")
        else:
            print(f"  ETH RECHAZADO — Mejor: {best_name} (PF={best_pf:.2f})")
    print(f"  {'='*70}")


if __name__ == '__main__':
    main()
