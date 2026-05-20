"""
test_committee.py
=================
Re-test del comite ETH "aprobado" originalmente (docs/V15_ETH_evaluation.md)
con motor honesto.

Reglas reproducidas EXACTAMENTE de archive_scripts/evaluate_eth_v2.py y
evaluate_eth_short_v5.py:

LONG (no BEAR):
  - detect_breakout_b_btc (BTC strict): vol>=1.8, 3/5 BB<4, ADX<=28, bar<=2.5%
  - detect_pullback_btc (BTC pullback EMA20): dist -0.5% to +1.5%, ADX>=15, RSI 33-58
  - detect_breakout_eth_adapted (ETH propio): vol>=1.3, 2/5 BB<5.5, ADX<=32, bar<=3.5%
  - ETH-BTC correlation 168-bar >= 0.5 para follower trades

SHORT (BEAR only):
  - detect_multi_conf: bearish candle + RSI>=60 + bb_pct>=0.75 + vol>=1.0
  - detect_bb_upper:   bearish candle + bb_pct>=0.90

TP/SL adaptive (ATR-based, FIJOS — no trailing):
  TP = max(min(atr_pct*2.5, 0.08), 0.025)
  SL = max(min(atr_pct*1.5, 0.05), 0.015)
  max_bars: LONG=18, SHORT=16

MOTOR HONESTO:
  - Una posicion a la vez (advance idx tras cierre)
  - TP+SL mismo bar: SL gana (conservador)
  - Sin look-ahead (todas las features con shift apropiado)

Validacion: real + 20 sinteticas (block bootstrap) + null + bootstrap p.

Declarado en V15_ETH_evaluation.md:
  WF 8/12, WR 49%, PF 1.28, $1K -> $4820 (annual ~30%), DD 42.7%
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


# -----------------------------------------------------------------------------
# Indicadores manuales (estandar, sin pandas_ta)
# -----------------------------------------------------------------------------
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's RMA (equiv ewm con alpha=1/n)."""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    ru = _rma(up, n)
    rd = _rma(dn, n)
    rs = ru / rd.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _true_range(h, l, c):
    prev_c = c.shift(1)
    return pd.concat([(h - l).abs(),
                      (h - prev_c).abs(),
                      (l - prev_c).abs()], axis=1).max(axis=1)


def _atr(h, l, c, n: int = 14) -> pd.Series:
    return _rma(_true_range(h, l, c), n)


def _bb(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    upper = mid + k * std
    lower = mid - k * std
    return lower, mid, upper


def _adx(h, l, c, n: int = 14):
    """ADX clasico de Wilder."""
    up = h - h.shift(1)
    dn = l.shift(1) - l
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    tr = _true_range(h, l, c)
    atr_n = _rma(tr, n).replace(0, np.nan)
    plus_di = 100 * _rma(plus_dm, n) / atr_n
    minus_di = 100 * _rma(minus_dm, n) / atr_n
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _rma(dx, n)
DATA = ROOT / 'data'

COMMISSION = 0.0005  # 0.05% per side

WF_FOLDS = [
    ('2020-01-01', '2020-06-30'),
    ('2020-07-01', '2020-12-31'),
    ('2021-01-01', '2021-06-30'),
    ('2021-07-01', '2021-12-31'),
    ('2022-01-01', '2022-06-30'),
    ('2022-07-01', '2022-12-31'),
    ('2023-01-01', '2023-06-30'),
    ('2023-07-01', '2023-12-31'),
    ('2024-01-01', '2024-06-30'),
    ('2024-07-01', '2024-12-31'),
    ('2025-01-01', '2025-06-30'),
    ('2025-07-01', '2025-12-31'),
]


# =============================================================================
# Carga
# =============================================================================
def load_eth_4h():
    df = pd.read_parquet(DATA / 'ETH_USDT_4h_full.parquet').sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df.loc['2020-01-01':'2025-12-31']
    return df[['open', 'high', 'low', 'close', 'volume']].copy()


def load_btc_4h():
    df = pd.read_parquet(DATA / 'BTC_USDT_4h_full.parquet').sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df.loc['2020-01-01':'2025-12-31']
    return df[['open', 'high', 'low', 'close', 'volume']].copy()


# =============================================================================
# Features
# =============================================================================
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV -> df con ema, rsi, atr, bb, adx, vol_ratio (todos rolling pasados)."""
    df = df.copy()
    h, l, c, v = df['high'], df['low'], df['close'], df['volume']
    for n in [20, 50, 200]:
        df[f'ema{n}'] = _ema(c, n)
    df['rsi14'] = _rsi(c, 14)
    atr = _atr(h, l, c, 14)
    df['atr14'] = atr
    df['atr_pct'] = atr / c * 100   # en %
    bb_lo, bb_mid, bb_up = _bb(c, 20, 2.0)
    rng = (bb_up - bb_lo).replace(0, np.nan)
    df['bb_pct']   = (c - bb_lo) / rng
    df['bb_width'] = rng / bb_mid * 100
    df['adx14'] = _adx(h, l, c, 14)
    vol_ma = v.rolling(20).mean()
    df['vol_ratio'] = v / vol_ma.replace(0, np.nan)
    return df.dropna(subset=['ema200', 'rsi14', 'atr_pct', 'bb_pct', 'adx14'])


def compute_daily_regime(df_4h: pd.DataFrame) -> pd.Series:
    """Regime BULL/BEAR/RANGE desde daily, con shift(1)."""
    daily_c = df_4h['close'].resample('1D').last().dropna()
    e20 = daily_c.ewm(span=20, adjust=False).mean()
    e50 = daily_c.ewm(span=50, adjust=False).mean()
    e200 = daily_c.ewm(span=200, adjust=False).mean()
    daily = pd.DataFrame({'close': daily_c, 'ema20': e20, 'ema50': e50, 'ema200': e200})
    daily = daily.shift(1)   # CRITICAL: hoy usa info de ayer

    def classify(row):
        if pd.isna(row['ema50']) or pd.isna(row['ema20']):
            return 'RANGE'
        dist = (row['ema20'] - row['ema50']) / row['ema50']
        if dist > 0.02:
            return 'BULL'
        elif dist < -0.02:
            if not pd.isna(row['ema200']) and row['close'] > row['ema200']:
                return 'RANGE'
            return 'BEAR'
        return 'RANGE'

    regime_daily = daily.apply(classify, axis=1)
    return regime_daily.reindex(df_4h.index, method='ffill')


def compute_eth_btc_corr(df_eth_4h: pd.DataFrame, df_btc_4h: pd.DataFrame, window=168) -> pd.Series:
    """Rolling 168-bar correlation entre ETH ret and BTC ret. shift(1) implicitly via rolling."""
    common = df_eth_4h.index.intersection(df_btc_4h.index)
    eth_r = df_eth_4h.loc[common, 'close'].pct_change()
    btc_r = df_btc_4h.loc[common, 'close'].pct_change()
    corr = eth_r.rolling(window).corr(btc_r)
    return corr.reindex(df_eth_4h.index).ffill().fillna(0.0)


# =============================================================================
# DETECTORS (copia exacta de evaluate_eth_v2.py + evaluate_eth_short_v5.py)
# =============================================================================
def detect_breakout_b_btc(df_btc, i):
    if i < 25:
        return None
    row = df_btc.iloc[i]
    high20 = float(df_btc['high'].iloc[i - 20:i].max())
    if row['close'] <= high20:
        return None
    if row.get('vol_ratio', 1) < 1.8:
        return None
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > 2.5:
        return None
    recent_bb = df_btc['bb_width'].iloc[i - 5:i]
    if (recent_bb < 4.0).sum() < 3:
        return None
    if df_btc['adx14'].iloc[i - 3:i].mean() > 28:
        return None
    entry = float(row['close'])
    sl_raw = float(df_btc['low'].iloc[i - 5:i].min()) * 0.997
    sl_pct = (entry - sl_raw) / entry
    if sl_pct < 0.005 or sl_pct > 0.04:
        return None
    return {'setup': 'BRK_BTC'}


def detect_pullback_btc(df_btc, i):
    if i < 25:
        return None
    row = df_btc.iloc[i]
    prev = df_btc.iloc[i - 1]
    c_, o_ = float(row['close']), float(row['open'])
    ema20 = float(row.get('ema20', 0))
    ema50 = float(row.get('ema50', 0))
    if ema20 <= 0 or ema50 <= 0 or c_ < ema50:
        return None
    dist = (c_ - ema20) / ema20
    if dist < -0.005 or dist > 0.015:
        return None
    if float(row.get('adx14', 0)) < 15:
        return None
    rsi = float(row.get('rsi14', 50))
    if rsi < 33 or rsi > 58:
        return None
    if c_ <= o_ or float(prev['close']) >= float(prev['open']):
        return None
    if float(row.get('vol_ratio', 1)) > 2.0:
        return None
    return {'setup': 'PB_BTC'}


def detect_breakout_eth_adapted(df_eth, i,
                                vol_min=1.3, bb_max=5.5, adx_max=32, bar_max=3.5):
    if i < 25:
        return None
    row = df_eth.iloc[i]
    high20 = float(df_eth['high'].iloc[i - 20:i].max())
    if row['close'] <= high20:
        return None
    if row.get('vol_ratio', 1) < vol_min:
        return None
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > bar_max:
        return None
    recent_bb = df_eth['bb_width'].iloc[i - 5:i]
    if (recent_bb < bb_max).sum() < 2:
        return None
    if df_eth['adx14'].iloc[i - 3:i].mean() > adx_max:
        return None
    entry = float(row['close'])
    sl_raw = float(df_eth['low'].iloc[i - 5:i].min()) * 0.995
    sl_pct_legacy = (entry - sl_raw) / entry
    if sl_pct_legacy < 0.005 or sl_pct_legacy > 0.06:
        return None
    return {'setup': 'BRK_ETH', 'entry': entry}


def detect_multi_conf_short(df, i):
    if i < 25:
        return None
    row = df.iloc[i]
    c_, o_ = float(row['close']), float(row['open'])
    if c_ >= o_:
        return None
    if float(row.get('rsi14', 50)) < 60:
        return None
    if float(row.get('bb_pct', 0.5)) < 0.75:
        return None
    if float(row.get('vol_ratio', 1)) < 1.0:
        return None
    return {'setup': 'MULTI_CONF', 'entry': c_}


def detect_bb_upper_short(df, i):
    if i < 25:
        return None
    row = df.iloc[i]
    c_, o_ = float(row['close']), float(row['open'])
    if c_ >= o_:
        return None
    if float(row.get('bb_pct', 0.5)) < 0.90:
        return None
    return {'setup': 'BB_UPPER', 'entry': c_}


def adaptive_tpsl(atr_pct: float, side: str) -> tuple:
    """TP/SL adaptive de la doc original. atr_pct en %."""
    tp = max(min(atr_pct / 100 * 2.5, 0.08), 0.025)
    sl = max(min(atr_pct / 100 * 1.5, 0.05), 0.015)
    return tp, sl


# =============================================================================
# Honest simulator: TP/SL fijo. Tiebreaker conservador (SL gana mismo bar)
# =============================================================================
def sim_long_fixed(df, entry_bar, entry, tp_pct, sl_pct, max_bars):
    tp_price = entry * (1 + tp_pct)
    sl_price = entry * (1 - sl_pct)
    for k in range(1, max_bars + 1):
        b = entry_bar + k
        if b >= len(df):
            ep = float(df['close'].iloc[-1])
            pnl = (ep - entry) / entry - 2 * COMMISSION
            return ('TP' if ep > entry else 'SL'), pnl, k
        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        # conservador: SL primero
        if lo <= sl_price:
            return 'SL', -sl_pct - 2 * COMMISSION, k
        if hi >= tp_price:
            return 'TP', tp_pct - 2 * COMMISSION, k
    ep = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (ep - entry) / entry - 2 * COMMISSION
    return ('TP' if ep > entry else 'SL'), pnl, max_bars


def sim_short_fixed(df, entry_bar, entry, tp_pct, sl_pct, max_bars):
    tp_price = entry * (1 - tp_pct)
    sl_price = entry * (1 + sl_pct)
    for k in range(1, max_bars + 1):
        b = entry_bar + k
        if b >= len(df):
            ep = float(df['close'].iloc[-1])
            pnl = (entry - ep) / entry - 2 * COMMISSION
            return ('TP' if ep < entry else 'SL'), pnl, k
        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        # conservador: SL primero
        if hi >= sl_price:
            return 'SL', -sl_pct - 2 * COMMISSION, k
        if lo <= tp_price:
            return 'TP', tp_pct - 2 * COMMISSION, k
    ep = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (entry - ep) / entry - 2 * COMMISSION
    return ('TP' if ep < entry else 'SL'), pnl, max_bars


# =============================================================================
# Motor honesto del comite (una posicion a la vez)
# =============================================================================
def run_committee(df_eth, df_btc, regime_eth, corr_eth_btc, start_i=30, end_i=None):
    """
    Comite completo:
      - LONG (BULL/RANGE): BRK_ETH > FOLLOW_BRK_BTC > FOLLOW_PB_BTC
      - SHORT (BEAR): MULTI_CONF > BB_UPPER
    Una posicion a la vez. Tras entrar, salta hasta despues del cierre.
    """
    if end_i is None:
        end_i = len(df_eth)
    btc_pos = df_btc.index.get_indexer(df_eth.index, method='nearest')

    trades = []
    i = max(start_i, 30)
    while i < end_i - 2:
        regime = regime_eth.iloc[i]
        row = df_eth.iloc[i]
        atr_pct = float(row.get('atr_pct', 2.5))
        tp_pct, sl_pct = adaptive_tpsl(atr_pct, 'long')

        # ========== LONG (no BEAR) ==========
        if regime != 'BEAR':
            entry = float(row['close'])
            # 1. BRK_ETH propio
            sig = detect_breakout_eth_adapted(df_eth, i)
            if sig is not None:
                out = sim_long_fixed(df_eth, i, entry, tp_pct, sl_pct, max_bars=18)
                trades.append({'ts': str(df_eth.index[i]), 'side': 'LONG',
                               'setup': 'BRK_ETH',
                               'outcome': out[0], 'pnl_pct': out[1], 'bars': out[2]})
                i += out[2] + 1
                continue
            # 2. FOLLOW_BRK_BTC (corr >= 0.5)
            corr = float(corr_eth_btc.iloc[i])
            if corr >= 0.5:
                btc_i = int(btc_pos[i])
                if 0 <= btc_i < len(df_btc):
                    btc_sig = detect_breakout_b_btc(df_btc, btc_i)
                    if btc_sig is not None:
                        out = sim_long_fixed(df_eth, i, entry, tp_pct, sl_pct, max_bars=18)
                        trades.append({'ts': str(df_eth.index[i]), 'side': 'LONG',
                                       'setup': 'FOLLOW_BRK_BTC',
                                       'outcome': out[0], 'pnl_pct': out[1], 'bars': out[2]})
                        i += out[2] + 1
                        continue
                    # 3. FOLLOW_PB_BTC
                    btc_sig = detect_pullback_btc(df_btc, btc_i)
                    if btc_sig is not None:
                        out = sim_long_fixed(df_eth, i, entry, tp_pct, sl_pct, max_bars=18)
                        trades.append({'ts': str(df_eth.index[i]), 'side': 'LONG',
                                       'setup': 'FOLLOW_PB_BTC',
                                       'outcome': out[0], 'pnl_pct': out[1], 'bars': out[2]})
                        i += out[2] + 1
                        continue
        # ========== SHORT (BEAR only) ==========
        else:
            entry = float(row['close'])
            tp_pct_s, sl_pct_s = adaptive_tpsl(atr_pct, 'short')
            # 1. Multi-conf
            sig = detect_multi_conf_short(df_eth, i)
            if sig is not None:
                out = sim_short_fixed(df_eth, i, entry, tp_pct_s, sl_pct_s, max_bars=16)
                trades.append({'ts': str(df_eth.index[i]), 'side': 'SHORT',
                               'setup': 'MULTI_CONF',
                               'outcome': out[0], 'pnl_pct': out[1], 'bars': out[2]})
                i += out[2] + 1
                continue
            # 2. BB upper
            sig = detect_bb_upper_short(df_eth, i)
            if sig is not None:
                out = sim_short_fixed(df_eth, i, entry, tp_pct_s, sl_pct_s, max_bars=16)
                trades.append({'ts': str(df_eth.index[i]), 'side': 'SHORT',
                               'setup': 'BB_UPPER',
                               'outcome': out[0], 'pnl_pct': out[1], 'bars': out[2]})
                i += out[2] + 1
                continue
        i += 1
    return trades


# =============================================================================
# Metrics / bootstrap
# =============================================================================
def metrics(trades):
    if not trades:
        return {'n': 0, 'wr': 0.0, 'pf': 0.0, 'total': 0.0, 'annual': 0.0,
                'max_dd': 0.0}
    n = len(trades)
    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / n
    gw = sum(wins); gl = abs(sum(losses))
    pf = (gw / gl) if gl > 1e-9 else float('inf')
    cum, peak, mdd = 1.0, 1.0, 0.0
    for t in sorted(trades, key=lambda x: pd.to_datetime(x['ts'])):
        cum *= (1.0 + t['pnl_pct'])
        peak = max(peak, cum)
        mdd = max(mdd, (peak - cum) / peak)
    return {'n': n, 'wr': wr, 'pf': pf, 'total': cum - 1.0,
            'annual': cum ** (1.0 / 6.0) - 1.0, 'max_dd': mdd, 'cum': cum}


def bootstrap_p(trades, n_iter=3000, seed=42):
    if len(trades) < 3:
        return None
    rng = np.random.default_rng(seed)
    pnls = np.array([t['pnl_pct'] for t in trades])
    totals = np.empty(n_iter)
    for j in range(n_iter):
        s = rng.choice(pnls, size=len(pnls), replace=True)
        totals[j] = np.prod(1 + s) - 1
    return float(np.mean(totals <= 0))


def block_bootstrap_ohlcv(df, block_size=24, n_bars=None, seed=None):
    if n_bars is None:
        n_bars = len(df)
    rng = np.random.default_rng(seed)
    max_start = len(df) - block_size
    n_blocks = (n_bars // block_size) + 2
    starts = rng.integers(0, max_start, size=n_blocks)
    parts = []
    last_close = float(df['close'].iloc[0])
    for s in starts:
        block = df.iloc[s:s + block_size].copy()
        scale = last_close / float(block['open'].iloc[0])
        block[['open', 'high', 'low', 'close']] *= scale
        parts.append(block)
        last_close = float(block['close'].iloc[-1])
    out = pd.concat(parts).iloc[:n_bars].copy()
    out.index = pd.date_range(start=df.index[0], periods=len(out),
                              freq='4h', tz='UTC')
    return out


def shuffle_returns(df, seed=None):
    rng = np.random.default_rng(seed)
    df_c = df.copy()
    log_ret = np.log(df_c['close'].values[1:] / df_c['close'].values[:-1])
    perm = rng.permutation(len(log_ret))
    log_ret_shuf = log_ret[perm]
    new_close = np.empty(len(df_c))
    new_close[0] = df_c['close'].iloc[0]
    new_close[1:] = df_c['close'].iloc[0] * np.exp(np.cumsum(log_ret_shuf))
    ratio_h_c = (df_c['high'] / df_c['close']).values[1:][perm]
    ratio_l_c = (df_c['low'] / df_c['close']).values[1:][perm]
    ratio_o_c = (df_c['open'] / df_c['close']).values[1:][perm]
    vol_shuf = df_c['volume'].values[1:][perm]
    out = pd.DataFrame({
        'open':  np.r_[df_c['open'].iloc[0], new_close[1:] * ratio_o_c],
        'high':  np.r_[df_c['high'].iloc[0], new_close[1:] * ratio_h_c],
        'low':   np.r_[df_c['low'].iloc[0], new_close[1:] * ratio_l_c],
        'close': new_close,
        'volume': np.r_[df_c['volume'].iloc[0], vol_shuf],
    }, index=df_c.index)
    return out


# =============================================================================
# Walk-forward
# =============================================================================
def walk_forward(df_eth_f, df_btc_f, regime_eth, corr_eth_btc):
    folds = []
    for start_s, end_s in WF_FOLDS:
        s = pd.Timestamp(start_s, tz='UTC')
        e = pd.Timestamp(end_s, tz='UTC')
        idxs = np.where((df_eth_f.index >= s) & (df_eth_f.index <= e))[0]
        if len(idxs) < 100:
            folds.append({'period': start_s[:7], 'n': 0, 'ok': False, 'nodata': True})
            continue
        trades = run_committee(df_eth_f, df_btc_f, regime_eth, corr_eth_btc,
                               start_i=idxs[0], end_i=idxs[-1] + 1)
        m = metrics(trades)
        ok = (m['n'] >= 5 and np.isfinite(m['pf']) and m['pf'] >= 1.2 and m['total'] > 0)
        folds.append({'period': start_s[:7], 'n': m['n'], 'wr': m['wr'],
                      'pf': m['pf'], 'total': m['total'], 'ok': ok, 'nodata': False})
    ok_count = sum(1 for f in folds if f['ok'])
    total_eval = sum(1 for f in folds if not f['nodata'])
    return folds, ok_count, total_eval


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("Cargando datos...")
    df_eth = load_eth_4h()
    df_btc = load_btc_4h()
    print(f"  ETH: {len(df_eth)} bars, BTC: {len(df_btc)} bars")

    print("Computando features...")
    df_eth_f = compute_features(df_eth)
    df_btc_f = compute_features(df_btc)
    regime_eth = compute_daily_regime(df_eth)
    regime_eth = regime_eth.reindex(df_eth_f.index).ffill()
    corr_eth_btc = compute_eth_btc_corr(df_eth_f, df_btc_f, window=168)

    # ===================================================================
    # Real ETH 2020-2025
    # ===================================================================
    print("\n=== Comite ETH 2020-2025 (motor honesto) ===")
    real_trades = run_committee(df_eth_f, df_btc_f, regime_eth, corr_eth_btc)
    m_real = metrics(real_trades)
    p_real = bootstrap_p(real_trades)
    p_s = f"{p_real:.3f}" if p_real is not None else "-"
    print(f"  N={m_real['n']}  WR={m_real['wr']:.1%}  PF={m_real['pf']:.2f}  "
          f"annual={m_real['annual']:+.1%}  DD={m_real['max_dd']:.1%}  "
          f"bootstrap p={p_s}")

    # Desglose por setup
    setups = {}
    for t in real_trades:
        setups.setdefault(t['setup'], []).append(t)
    print("  Desglose:")
    for setup_name, ts in sorted(setups.items()):
        ms = metrics(ts)
        ps = bootstrap_p(ts) if len(ts) >= 3 else None
        ps_s = f"{ps:.3f}" if ps is not None else "-"
        print(f"    {setup_name:<18}  N={ms['n']:>3}  WR={ms['wr']:.1%}  "
              f"PF={ms['pf']:.2f}  total={ms['total']:+.1%}  p={ps_s}")

    # WF
    folds, wf_ok, wf_total = walk_forward(df_eth_f, df_btc_f, regime_eth, corr_eth_btc)
    print(f"\nWalk-forward {wf_ok}/{wf_total} folds aprobados (criterio n>=5, PF>=1.2, total>0):")
    for f in folds:
        if f['nodata']:
            print(f"  {f['period']}: sin datos")
        else:
            print(f"  {f['period']}: N={f['n']:>3}  WR={f['wr']:.1%}  "
                  f"PF={f['pf']:.2f}  total={f['total']:+.1%}  {'OK' if f['ok'] else '--'}")

    # Comparacion vs declarado
    print(f"\nComparacion vs DECLARADO en docs/V15_ETH_evaluation.md:")
    print(f"  Declarado: WF 8/12, WR 49%, PF 1.28, $1K->$4820 (annual ~30%), DD 42.7%")
    print(f"  Honesto:   WF {wf_ok}/{wf_total}, WR {m_real['wr']:.0%}, PF {m_real['pf']:.2f}, "
          f"${1000 * m_real['cum']:.0f}, DD {m_real['max_dd']:.0%}")

    # ===================================================================
    # 20 sinteticas
    # ===================================================================
    N_SYNTH = 20
    print(f"\n=== {N_SYNTH} series sinteticas ETH (block bootstrap, BTC real) ===")
    synth_results = []
    for seed in range(N_SYNTH):
        df_eth_s = block_bootstrap_ohlcv(df_eth, block_size=24, seed=seed)
        df_eth_sf = compute_features(df_eth_s)
        regime_s = compute_daily_regime(df_eth_s).reindex(df_eth_sf.index).ffill()
        corr_s = compute_eth_btc_corr(df_eth_sf, df_btc_f, window=168)
        trades = run_committee(df_eth_sf, df_btc_f, regime_s, corr_s)
        ms = metrics(trades)
        synth_results.append(ms)
        print(f"  serie {seed:2d}: N={ms['n']:3d}  WR={ms['wr']:.1%}  "
              f"PF={ms['pf']:.2f}  annual={ms['annual']:+.1%}")
    annuals = [r['annual'] for r in synth_results]
    print(f"\n  Distribucion annual:")
    print(f"    mediana = {np.median(annuals):+.1%}")
    print(f"    media   = {np.mean(annuals):+.1%}")
    print(f"    p25-p75 = [{np.percentile(annuals, 25):+.1%}, {np.percentile(annuals, 75):+.1%}]")
    n_pos = sum(1 for a in annuals if a > 0)
    print(f"    series con annual > 0: {n_pos}/{N_SYNTH}")

    # ===================================================================
    # Null
    # ===================================================================
    print(f"\n=== Null (shuffle aleatorio, 10 seeds) ===")
    null_annuals = []
    for seed in range(10):
        df_null = shuffle_returns(df_eth, seed=seed)
        df_null_f = compute_features(df_null)
        regime_n = compute_daily_regime(df_null).reindex(df_null_f.index).ffill()
        corr_n = compute_eth_btc_corr(df_null_f, df_btc_f, window=168)
        trades = run_committee(df_null_f, df_btc_f, regime_n, corr_n)
        ms = metrics(trades)
        null_annuals.append(ms['annual'])
        print(f"  seed {seed}: N={ms['n']:3d}  annual={ms['annual']:+.1%}")
    edge_vs_null = np.median(annuals) - np.median(null_annuals)
    print(f"\n  Mediana null:       {np.median(null_annuals):+.1%}")
    print(f"  Mediana sintetico:  {np.median(annuals):+.1%}")
    print(f"  Edge vs null:       {edge_vs_null:+.1%}")

    # ===================================================================
    # Veredicto
    # ===================================================================
    print("\n" + "=" * 70)
    print("VEREDICTO Comite ETH (motor honesto)")
    print("=" * 70)
    sig = p_real is not None and p_real < 0.05
    med_pos = np.median(annuals) > 0
    n_pos_ok = n_pos >= 14
    edge_ok = edge_vs_null > 0.05
    wf_pass = wf_total > 0 and wf_ok / max(wf_total, 1) >= 0.58
    print(f"  Bootstrap p<0.05 real:        {'SI' if sig else 'NO'} (p={p_s})")
    print(f"  WF >= 7/12:                   {'SI' if wf_pass else 'NO'} ({wf_ok}/{wf_total})")
    print(f"  Mediana sintetico > 0:        {'SI' if med_pos else 'NO'} ({np.median(annuals):+.1%})")
    print(f"  >=14/20 sinteticas positivas: {'SI' if n_pos_ok else 'NO'} ({n_pos}/{N_SYNTH})")
    print(f"  Edge vs null > 5%:            {'SI' if edge_ok else 'NO'} ({edge_vs_null:+.1%})")
    all_pass = sig and med_pos and n_pos_ok and edge_ok and wf_pass
    print(f"\n  --> Comite ETH {'APROBADO' if all_pass else 'RECHAZADO'} con motor honesto")


if __name__ == '__main__':
    main()
