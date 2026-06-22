"""
v2_engine.py — Motor V2 honesto para paper trading 3 meses
============================================================
Estrategia V2 = A (Donchian trend LONG) + F (vol-compression breakout bidir).

Bugs corregidos vs versiones previas:
1. Una posición a la vez por par (sin solape de trades)
2. Trailing stop SIN look-ahead intrabar (exit check con SL previo PRIMERO,
   update peak/SL DESPUÉS para la vela siguiente)
3. MTF con shift(1) en todas las features daily
4. TP/SL conservador: SL gana si ambos tocados misma vela

API pública (para usar desde ml_strategy_v15.py o un nuevo strategy module):

  build_features(df_4h, df_1d=None) -> df con todas las features
  detect_signal(df_features, idx, params) -> 'A_LONG' | 'F_LONG' | 'F_SHORT' | None
  simulate_trade(df_features, entry_bar, params, sig_type) -> dict con outcome/pnl/bars

Los parámetros (PARAMS_V2) están FROZEN desde validación in-sample/OOS 2026 —
NO re-tunear sin re-validar con bootstrap + sintético + null.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Constantes de costes (igual que validation tests)
COMMISSION = 0.0005   # 0.05% per side -> 0.1% round trip

# =============================================================================
# PARÁMETROS V2 — FROZEN (no tocar)
# =============================================================================
PARAMS_V2 = {
    # === A (Donchian trend LONG-only) ===
    'a_donchian_n': 55,             # Donchian breakout window 4h
    'a_ema_fast_1d': 50,            # daily EMA fast (filtro régimen)
    'a_ema_slow_1d': 200,           # daily EMA slow
    'a_vol_ma_n': 20,
    'a_vol_ratio_min': 1.2,
    'a_adx_n': 14,
    'a_adx_min': 18,
    'a_atr_n': 14,
    'a_trail_atr_mult': 2.5,
    'a_trail_floor_pct': 0.025,
    'a_trail_ceiling_pct': 0.06,
    'a_max_bars': 60,
    'a_funding_z_n': 168,
    'a_funding_z_max': 2.5,

    # === F (vol-compression breakout bidir) ===
    'f_bb_n': 20,
    'f_bb_window': 100,             # ventana para percentil BB width
    'f_compression_pctile': 0.20,   # cuantil 20 inferior = "comprimido"
    'f_compression_sustain': 3,     # bars consecutivos compressed
    'f_breakout_n': 12,             # ventana hi/lo para breakout
    'f_vol_ratio_min': 1.2,
    'f_trail_atr_mult': 2.0,
    'f_trail_floor_pct': 0.02,
    'f_trail_ceiling_pct': 0.055,
    'f_max_bars': 40,
    'f_funding_z_max_long': 2.0,
    'f_funding_z_min_short': -1.5,
    'f_enable_long': True,
    'f_enable_short': True,

    # === Universales ===
    'commission': COMMISSION,
    'min_warmup_bars': 220,         # para EMA200
}


# =============================================================================
# INDICADORES (manual — sin pandas_ta para evitar dependencia rota)
# =============================================================================
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's RMA."""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    rs = _rma(up, n) / _rma(dn, n).replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _true_range(h, l, c):
    prev_c = c.shift(1)
    return pd.concat([(h - l).abs(),
                      (h - prev_c).abs(),
                      (l - prev_c).abs()], axis=1).max(axis=1)


def _atr(h, l, c, n: int = 14) -> pd.Series:
    return _rma(_true_range(h, l, c), n)


def _adx(h, l, c, n: int = 14) -> pd.Series:
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


def _bb(close, n=20, k=2.0):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    return mid - k * std, mid, mid + k * std


# =============================================================================
# FEATURES — todas con shift(1) en MTF para evitar look-ahead
# =============================================================================
def build_features(df_4h: pd.DataFrame, df_1d: pd.DataFrame | None = None,
                   df_funding: pd.DataFrame | None = None,
                   params: dict = PARAMS_V2) -> pd.DataFrame:
    """
    Construye matriz de features para V2. df_4h es obligatorio.
    df_1d opcional (si None, se deriva por resample del 4h con shift(1)).
    df_funding opcional (si None, funding_z = 0).
    """
    df = df_4h.copy()
    h, l, c, v = df['high'], df['low'], df['close'], df['volume']

    # ----- A's features ------
    df['atr'] = _atr(h, l, c, params['a_atr_n'])
    df['atr_pct'] = df['atr'] / c
    df['adx'] = _adx(h, l, c, params['a_adx_n'])
    df['donchian_high'] = h.rolling(params['a_donchian_n']).max().shift(1)
    df['vol_ma'] = v.rolling(params['a_vol_ma_n']).mean()
    df['vol_ratio'] = v / df['vol_ma'].replace(0, np.nan)

    # ----- F's features (compression + breakout) ------
    bb_lo, bb_mid, bb_up = _bb(c, params['f_bb_n'])
    bb_width = (bb_up - bb_lo) / bb_mid.replace(0, np.nan)
    df['bb_width'] = bb_width
    bb_threshold = bb_width.rolling(params['f_bb_window']).quantile(
        params['f_compression_pctile']).shift(1)
    df['bb_compressed'] = (bb_width.shift(1) < bb_threshold).astype(int)
    # sostener N velas
    df['compression_sustained'] = (
        df['bb_compressed'].rolling(params['f_compression_sustain']).sum()
        >= params['f_compression_sustain']
    ).astype(int)
    df['hi_n'] = h.rolling(params['f_breakout_n']).max().shift(1)
    df['lo_n'] = l.rolling(params['f_breakout_n']).min().shift(1)

    # ----- Régimen daily (shift(1)) ------
    if df_1d is not None:
        df_d = df_1d.copy()
        if df_d.index.tz is None:
            df_d.index = df_d.index.tz_localize('UTC')
        ema_fast = _ema(df_d['close'], params['a_ema_fast_1d'])
        ema_slow = _ema(df_d['close'], params['a_ema_slow_1d'])
        bull_d = (ema_fast > ema_slow).astype(int).shift(1)
    else:
        # fallback: derivar daily de 4h
        daily_c = df['close'].resample('1D').last().dropna()
        ema_fast = _ema(daily_c, params['a_ema_fast_1d'])
        ema_slow = _ema(daily_c, params['a_ema_slow_1d'])
        bull_d = (ema_fast > ema_slow).astype(int).shift(1)
    df['bull_1d'] = bull_d.reindex(df.index, method='ffill')

    # ----- Funding rate z-score (shift 1) ------
    if df_funding is not None and 'funding_rate' in df_funding.columns:
        fund = df_funding.copy()
        if fund.index.tz is None:
            fund.index = fund.index.tz_localize('UTC')
        fund_s = fund['funding_rate'].resample('4h').ffill()
        n_z = params['a_funding_z_n']
        fmean = fund_s.rolling(n_z).mean()
        fstd = fund_s.rolling(n_z).std()
        z = ((fund_s - fmean) / fstd.replace(0, np.nan)).shift(1)
        df['funding_z'] = z.reindex(df.index, method='ffill').fillna(0)
    else:
        df['funding_z'] = 0.0

    return df.dropna(subset=['atr', 'adx', 'donchian_high', 'bull_1d',
                              'bb_width', 'hi_n', 'vol_ratio'])


# =============================================================================
# SIGNAL DETECTION
# =============================================================================
def _signal_a(df: pd.DataFrame, idx: int, params: dict,
              live: bool = False) -> bool:
    """A's LONG signal en vela cerrada idx.

    `live=True`: omite el guard `idx >= len-2`, que existe SOLO para que el
    backtest tenga >=2 velas futuras para simulate_trade. En vivo evaluamos la
    ultima vela cerrada (idx=len-2) y abrimos una posicion real, sin simular
    hacia adelante, asi que ese guard no aplica.
    """
    if idx < params['min_warmup_bars']:
        return False
    if not live and idx >= len(df) - 2:
        return False
    row = df.iloc[idx]
    if row.get('bull_1d', 0) < 1:
        return False
    dh = row.get('donchian_high', np.nan)
    if pd.isna(dh) or row['close'] <= dh:
        return False
    if pd.isna(row.get('vol_ratio', np.nan)) or row['vol_ratio'] < params['a_vol_ratio_min']:
        return False
    if pd.isna(row.get('adx', np.nan)) or row['adx'] < params['a_adx_min']:
        return False
    fz = row.get('funding_z', 0)
    if pd.notna(fz) and fz > params['a_funding_z_max']:
        return False
    return True


def _signal_f(df: pd.DataFrame, idx: int, params: dict,
              live: bool = False) -> str | None:
    """F's LONG/SHORT signal en vela cerrada idx. Ver _signal_a sobre `live`."""
    if idx < params['min_warmup_bars']:
        return None
    if not live and idx >= len(df) - 2:
        return None
    if df['compression_sustained'].iloc[idx - 1] != 1:
        return None
    row = df.iloc[idx]
    hi_n = row.get('hi_n', np.nan)
    lo_n = row.get('lo_n', np.nan)
    if pd.isna(hi_n) or pd.isna(lo_n):
        return None
    c = float(row['close'])
    side = None
    if params['f_enable_long'] and c > hi_n:
        side = 'LONG'
    elif params['f_enable_short'] and c < lo_n:
        side = 'SHORT'
    if side is None:
        return None
    vr = row.get('vol_ratio', np.nan)
    if pd.isna(vr) or vr < params['f_vol_ratio_min']:
        return None
    bull = row.get('bull_1d', 0)
    if side == 'LONG' and bull < 1:
        return None
    if side == 'SHORT' and bull >= 1:
        return None
    fz = row.get('funding_z', 0)
    if pd.notna(fz):
        if side == 'LONG' and fz > params['f_funding_z_max_long']:
            return None
        if side == 'SHORT' and fz < params['f_funding_z_min_short']:
            return None
    return side


def detect_signal(df: pd.DataFrame, idx: int,
                  params: dict = PARAMS_V2, live: bool = False) -> str | None:
    """
    Combina A y F con prioridad A > F.
    Devuelve: 'A_LONG' | 'F_LONG' | 'F_SHORT' | None
    `live=True` evalua la ultima vela cerrada (ver _signal_a).
    """
    if _signal_a(df, idx, params, live=live):
        return 'A_LONG'
    sigF = _signal_f(df, idx, params, live=live)
    if sigF == 'LONG':
        return 'F_LONG'
    if sigF == 'SHORT':
        return 'F_SHORT'
    return None


# =============================================================================
# TRAILING STOP HONESTO (sin look-ahead intrabar)
# =============================================================================
def _sim_long_trailing(df, entry_bar, entry_price, trail_dist, max_bars,
                       commission):
    """
    LONG trailing: exit check con SL del bar previo PRIMERO, peak/SL update
    DESPUÉS para la vela siguiente. SIN look-ahead intrabar.
    """
    sl_price = entry_price * (1 - trail_dist)
    peak = entry_price
    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df):
            ep = float(df['close'].iloc[-1])
            pnl = (ep - entry_price) / entry_price - 2 * commission
            return ('TP' if ep > entry_price else 'SL'), ep, pnl, i
        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        # 1) exit check con stop PREVIO
        if lo <= sl_price:
            exit_p = sl_price
            pnl = (exit_p - entry_price) / entry_price - 2 * commission
            return ('TP' if exit_p > entry_price else 'SL'), exit_p, pnl, i
        # 2) update peak/SL para la SIGUIENTE vela
        if hi > peak:
            peak = hi
        sl_price = max(sl_price, peak * (1 - trail_dist))
    # timeout
    ep = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (ep - entry_price) / entry_price - 2 * commission
    return ('TIMEOUT', ep, pnl, max_bars)


def _sim_short_trailing(df, entry_bar, entry_price, trail_dist, max_bars,
                        commission):
    """SHORT trailing honest (espejo de _sim_long_trailing)."""
    sl_price = entry_price * (1 + trail_dist)
    trough = entry_price
    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df):
            ep = float(df['close'].iloc[-1])
            pnl = (entry_price - ep) / entry_price - 2 * commission
            return ('TP' if ep < entry_price else 'SL'), ep, pnl, i
        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        # 1) exit check
        if hi >= sl_price:
            exit_p = sl_price
            pnl = (entry_price - exit_p) / entry_price - 2 * commission
            return ('TP' if exit_p < entry_price else 'SL'), exit_p, pnl, i
        # 2) update trough/SL
        if lo < trough:
            trough = lo
        sl_price = min(sl_price, trough * (1 + trail_dist))
    ep = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (entry_price - ep) / entry_price - 2 * commission
    return ('TIMEOUT', ep, pnl, max_bars)


def simulate_trade(df: pd.DataFrame, entry_bar: int,
                   params: dict = PARAMS_V2,
                   sig_type: str = 'A_LONG') -> dict:
    """
    Simula UN trade con TP/SL honesto.
    Returns: dict con outcome, pnl_pct, bars, exit_price, entry_price.
    """
    entry = float(df['close'].iloc[entry_bar])
    atr_pct = float(df['atr_pct'].iloc[entry_bar])

    if sig_type == 'A_LONG':
        trail = min(max(atr_pct * params['a_trail_atr_mult'],
                        params['a_trail_floor_pct']),
                    params['a_trail_ceiling_pct'])
        outcome, exit_p, pnl, bars = _sim_long_trailing(
            df, entry_bar, entry, trail, params['a_max_bars'], params['commission'])
        side = 'LONG'
    elif sig_type == 'F_LONG':
        trail = min(max(atr_pct * params['f_trail_atr_mult'],
                        params['f_trail_floor_pct']),
                    params['f_trail_ceiling_pct'])
        outcome, exit_p, pnl, bars = _sim_long_trailing(
            df, entry_bar, entry, trail, params['f_max_bars'], params['commission'])
        side = 'LONG'
    elif sig_type == 'F_SHORT':
        trail = min(max(atr_pct * params['f_trail_atr_mult'],
                        params['f_trail_floor_pct']),
                    params['f_trail_ceiling_pct'])
        outcome, exit_p, pnl, bars = _sim_short_trailing(
            df, entry_bar, entry, trail, params['f_max_bars'], params['commission'])
        side = 'SHORT'
    else:
        return {'outcome': None, 'pnl_pct': 0.0, 'bars': 0}

    return {
        'outcome': outcome,
        'side': side,
        'entry_price': entry,
        'exit_price': exit_p,
        'pnl_pct': pnl,
        'bars': bars,
        'trail_dist': trail,
        'sig_type': sig_type,
    }


# =============================================================================
# ENGINE COMPLETO (para backtest o live signal generation)
# =============================================================================
def run_v2_backtest(df_4h, df_1d=None, df_funding=None,
                    params: dict = PARAMS_V2,
                    start_i: int = None, end_i: int = None) -> list:
    """
    Backtest V2 sobre un par. UNA POSICIÓN A LA VEZ.
    Devuelve lista de trades.
    """
    df = build_features(df_4h, df_1d, df_funding, params)
    if start_i is None:
        start_i = params['min_warmup_bars']
    if end_i is None:
        end_i = len(df) - 1
    trades = []
    i = max(start_i, params['min_warmup_bars'])
    while i < end_i:
        sig = detect_signal(df, i, params)
        if sig is None:
            i += 1
            continue
        out = simulate_trade(df, i, params, sig_type=sig)
        out['ts_entry'] = str(df.index[i])
        out['idx_entry'] = i
        trades.append(out)
        i += out['bars'] + 1   # ONE POSITION AT A TIME
    return trades


def get_live_signal(df_4h, df_1d=None, df_funding=None,
                    params: dict = PARAMS_V2) -> dict | None:
    """
    Para uso live: evalúa la ÚLTIMA vela cerrada del df_4h y devuelve:
    {'side': 'LONG'|'SHORT', 'sig_type': 'A_LONG'|..., 'trail_dist': float,
     'max_bars': int, 'entry_price': float, ...}
    o None si no hay señal.
    """
    df = build_features(df_4h, df_1d, df_funding, params)
    if len(df) < params['min_warmup_bars'] + 2:
        return None
    # idx de la última vela cerrada
    idx = len(df) - 2  # -1 sería la vela en curso; -2 es la cerrada
    sig = detect_signal(df, idx, params, live=True)
    if sig is None:
        return None
    row = df.iloc[idx]
    entry = float(row['close'])
    atr_pct = float(row['atr_pct'])
    if sig == 'A_LONG':
        trail = min(max(atr_pct * params['a_trail_atr_mult'],
                        params['a_trail_floor_pct']),
                    params['a_trail_ceiling_pct'])
        side, max_bars = 'LONG', params['a_max_bars']
    elif sig == 'F_LONG':
        trail = min(max(atr_pct * params['f_trail_atr_mult'],
                        params['f_trail_floor_pct']),
                    params['f_trail_ceiling_pct'])
        side, max_bars = 'LONG', params['f_max_bars']
    else:  # F_SHORT
        trail = min(max(atr_pct * params['f_trail_atr_mult'],
                        params['f_trail_floor_pct']),
                    params['f_trail_ceiling_pct'])
        side, max_bars = 'SHORT', params['f_max_bars']
    return {
        'side': side,
        'sig_type': sig,
        'entry_price': entry,
        'trail_dist': trail,
        'max_bars': max_bars,
        'atr_pct': atr_pct,
        'ts_entry': str(df.index[idx]),
        'regime': 'BULL' if row.get('bull_1d', 0) >= 1 else 'BEAR/RANGE',
    }
