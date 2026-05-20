"""
Agent N — VOL-COMPRESSION BREAKOUT (SOL 4h) WIDE TRAIL (scaled to SOL vol).
============================================================================

Filosofia
---------
1. Idea central: igual que F (Bollinger Squeeze + vol-of-vol reversion).
   Pero el trailing tight 0.8% que "funciono" en SOL en el doc original
   ESTABA INFLADO por el bug de look-ahead intrabar (PASO0_lookahead.md).
   Con el motor honesto, ese trailing genera ruido en SOL (vol 80% vs BTC 41%).
2. Hipotesis: el MECANISMO vol-compression breakout puede tener edge en SOL
   si el trailing es PROPORCIONAL a la volatilidad. Vol SOL ~2x BTC ->
   trail debe ser proporcional. F en BTC uso trail_floor=0.020 (2%); para
   SOL escalamos a 0.025 (2.5%) y ATR mult mas moderado (0.50x ATR pct)
   con piso=2.5% y techo=8.0%.
3. Bidireccional: BEAR de SOL es violento (bounces violentos), pero con
   trail wide (~2.5%) los bounces no matan stops cada wiggle. SHORT solo
   en BEAR macro daily.

Cero overfitting — auditorias aplicadas
---------------------------------------
- Una posicion a la vez (un solo activo SOL).
- Sin look-ahead intrabar: salida vs SL HEREDADO primero, luego peak/SL.
- BB width percentile, N-bar high/low, EMA daily, rolling vol: TODOS con
  .shift(1) explicito.
- ATR Wilder RMA, sin look-ahead.
- Cutoff inviolable 2025-12-31.
- Params FROZEN a priori (escalados por vol-ratio SOL/BTC, no por grid
  search en SOL).

API publica
-----------
- PARAMS: dict frozen.
- prepare_data(df_sol_4h, df_btc_1d=None, params) -> DataFrame
- signal(df, idx, params) -> 'LONG' | 'SHORT' | None
- simulate(df, entry_bar, params, side) -> dict
- run_backtest(df_sol, params, start_dt, end_dt) -> list[trade]
- metrics(trades) -> dict
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd


# =============================================================================
# PARAMETROS FROZEN — a priori, escalados desde F BTC
# =============================================================================
PARAMS = {
    # --- Costes ---
    'commission': 0.0005,

    # --- Vol compression (BB width percentile) ---
    'bb_n': 20, 'bb_k': 2.0,
    'percentile_lookback': 100,
    'compression_percentile': 0.20,
    'compression_min_bars': 3,

    # --- Breakout trigger ---
    'breakout_n': 12,
    'vol_ratio_n': 20,
    'vol_ratio_min': 1.0,           # mas laxo que BTC F (1.2): SOL tiene vol
                                    # baseline mas alta, exigir 1.2 dropea
                                    # demasiados breakouts validos.

    # --- Filtro regimen daily (shift(1)) ---
    'ema_fast_1d': 50,
    'ema_slow_1d': 200,
    'regime_filter_enabled': True,

    # --- Trailing ATR WIDE — escalado a vol SOL ---
    'atr_n': 14,
    'trail_atr_factor': 0.50,       # 0.50 * ATR_pct (vs F's 2.0)
                                    # con SOL ATR ~3.55%: 0.50*3.55=1.78%
                                    # raw, pero piso=2.5% domina.
    'trail_floor_pct': 0.025,       # 2.5% piso (escalado ~1.25x de F 2%)
    'trail_ceiling_pct': 0.080,     # 8% techo (escalado ~1.45x de F 5.5%)
    'max_bars': 30,                 # 30 velas 4h = 5 dias (mas corto que
                                    # F's 48: SOL se mueve mas rapido).

    # --- Direccionalidad ---
    'enable_long': True,
    'enable_short': True,

    # --- Operativos ---
    'cutoff_date': '2025-12-31',
    'min_bars_warmup': 250,
}


# =============================================================================
# PARAMS para variante TIGHT (cross-check honesto)
# =============================================================================
PARAMS_TIGHT = dict(PARAMS, **{
    'trail_atr_factor': 0.30,      # mas tight (~0.30*ATR)
    'trail_floor_pct': 0.008,      # 0.8% (el "tight" original)
    'trail_ceiling_pct': 0.020,    # 2% techo
})


# =============================================================================
# INDICADORES (sin look-ahead)
# =============================================================================
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def _true_range(h, l, c):
    pc = c.shift(1)
    return pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def _atr(h, l, c, n=14):
    return _rma(_true_range(h, l, c), n)


def _bollinger_width(c: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    ma = c.rolling(n).mean()
    sd = c.rolling(n).std()
    upper = ma + k * sd
    lower = ma - k * sd
    return (upper - lower) / ma.replace(0, np.nan)


def prepare_data(df_4h: pd.DataFrame,
                 df_1d: Optional[pd.DataFrame] = None,
                 params: dict = PARAMS) -> pd.DataFrame:
    """
    Devuelve DataFrame con features para signal/simulate.
    Cutoff <= params['cutoff_date'] aplicado.
    Todas las features MTF/percentile con shift(1) -> sin look-ahead.
    """
    cutoff = pd.Timestamp(params['cutoff_date'], tz='UTC')
    df = df_4h.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df[df.index <= cutoff].sort_index()

    h, l, c, v = df['high'], df['low'], df['close'], df['volume']

    df['atr'] = _atr(h, l, c, params['atr_n'])
    df['atr_pct'] = df['atr'] / c

    df['bb_width'] = _bollinger_width(c, params['bb_n'], params['bb_k'])

    df['bb_width_pct'] = (
        df['bb_width']
        .rolling(params['percentile_lookback'])
        .rank(pct=True)
        .shift(1)
    )

    df['compressed'] = (df['bb_width_pct'] <= params['compression_percentile']).astype(int)
    min_b = params['compression_min_bars']
    df['compression_sustained'] = (df['compressed'].rolling(min_b).sum() >= min_b).astype(int)

    df['hi_n'] = h.rolling(params['breakout_n']).max().shift(1)
    df['lo_n'] = l.rolling(params['breakout_n']).min().shift(1)

    df['vol_ma'] = v.rolling(params['vol_ratio_n']).mean()
    df['vol_ratio'] = v / df['vol_ma'].replace(0, np.nan)

    # Returns 4h (info hasta t inclusive — usado solo para sanity, no para sizing)
    df['ret_4h'] = np.log(c / c.shift(1))

    # MTF: EMA daily con shift(1). Si no se pasa daily, derivar desde 4h.
    if df_1d is not None:
        df_d = df_1d.copy()
        if df_d.index.tz is None:
            df_d.index = df_d.index.tz_localize('UTC')
        df_d = df_d[df_d.index <= cutoff].sort_index()
        ema_fast = _ema(df_d['close'], params['ema_fast_1d'])
        ema_slow = _ema(df_d['close'], params['ema_slow_1d'])
        bull_d = (ema_fast > ema_slow).astype(int).shift(1)
        df['bull_1d'] = bull_d.reindex(df.index, method='ffill')
    else:
        daily_close = df['close'].resample('1D').last().dropna()
        ema_fast = _ema(daily_close, params['ema_fast_1d'])
        ema_slow = _ema(daily_close, params['ema_slow_1d'])
        bull_d = (ema_fast > ema_slow).astype(int).shift(1)
        df['bull_1d'] = bull_d.reindex(df.index, method='ffill')

    df = df.dropna(subset=['atr', 'bb_width_pct', 'hi_n', 'lo_n',
                           'vol_ratio', 'bull_1d'])
    return df


# =============================================================================
# SIGNAL
# =============================================================================
def signal(df: pd.DataFrame, idx: int, params: dict = PARAMS) -> Optional[str]:
    if idx < params['min_bars_warmup']:
        return None
    if idx >= len(df) - 2:
        return None

    row = df.iloc[idx]

    # Compresion sostenida en velas previas (idx-1)
    if df['compression_sustained'].iloc[idx - 1] != 1:
        return None

    hi_n = row.get('hi_n', np.nan)
    lo_n = row.get('lo_n', np.nan)
    if pd.isna(hi_n) or pd.isna(lo_n):
        return None

    c = float(row['close'])

    side = None
    if params.get('enable_long', True) and c > hi_n:
        side = 'LONG'
    elif params.get('enable_short', True) and c < lo_n:
        side = 'SHORT'
    if side is None:
        return None

    vr = row.get('vol_ratio', np.nan)
    if pd.isna(vr) or vr < params['vol_ratio_min']:
        return None

    if params.get('regime_filter_enabled', True):
        bull = row.get('bull_1d', 0)
        if side == 'LONG' and bull < 1:
            return None
        if side == 'SHORT' and bull >= 1:
            return None

    return side


# =============================================================================
# SIMULATE — trailing ATR, SIN look-ahead intrabar
# =============================================================================
def simulate(df: pd.DataFrame, entry_bar: int, params: dict, side: str) -> dict:
    """
    Trade abierto al close de entry_bar. Trailing ATR wide.
    En cada vela b > entry_bar:
      1) PRIMERO chequear salida vs SL HEREDADO (el SL que ya existia al inicio
         de la vela).
      2) DESPUES actualizar peak/SL para la SIGUIENTE vela.

    SHORT: trailing sigue al trough (minimo), SL toca por arriba.
    """
    n = len(df)
    entry_price = float(df['close'].iloc[entry_bar])
    entry_ts = df.index[entry_bar]
    atr_pct = float(df['atr_pct'].iloc[entry_bar])
    if not np.isfinite(atr_pct) or atr_pct <= 0:
        return {'outcome': 'SKIP', 'pnl_pct': 0.0, 'bars': 0,
                'exit_price': entry_price, 'entry_ts': entry_ts,
                'exit_ts': entry_ts, 'side': side, 'reason': 'no_atr'}

    trail_dist = atr_pct * params['trail_atr_factor']
    trail_dist = max(params['trail_floor_pct'],
                     min(params['trail_ceiling_pct'], trail_dist))

    max_bars = params['max_bars']
    commission = params['commission']

    if side == 'LONG':
        peak = entry_price
        sl_price = entry_price * (1 - trail_dist)
        for i in range(1, max_bars + 1):
            b = entry_bar + i
            if b >= n:
                exit_p = float(df['close'].iloc[-1])
                ret = (exit_p - entry_price) / entry_price
                pnl = ret - 2 * commission
                return {'outcome': 'EOD', 'pnl_pct': pnl, 'bars': i,
                        'exit_price': exit_p, 'entry_ts': entry_ts,
                        'exit_ts': df.index[-1], 'side': side,
                        'raw_ret': ret, 'reason': 'eod'}
            hi = float(df['high'].iloc[b])
            lo = float(df['low'].iloc[b])
            # 1) salida vs SL heredado
            if lo <= sl_price:
                ret = (sl_price - entry_price) / entry_price
                pnl = ret - 2 * commission
                outcome = 'TP' if sl_price > entry_price else 'SL'
                return {'outcome': outcome, 'pnl_pct': pnl, 'bars': i,
                        'exit_price': sl_price, 'entry_ts': entry_ts,
                        'exit_ts': df.index[b], 'side': side,
                        'raw_ret': ret, 'reason': 'trail'}
            # 2) actualizar peak/SL para la siguiente vela
            if hi > peak:
                peak = hi
            sl_price = max(sl_price, peak * (1 - trail_dist))
        # Timeout
        exit_p = float(df['close'].iloc[entry_bar + max_bars])
        ret = (exit_p - entry_price) / entry_price
        pnl = ret - 2 * commission
        return {'outcome': 'TIMEOUT', 'pnl_pct': pnl, 'bars': max_bars,
                'exit_price': exit_p, 'entry_ts': entry_ts,
                'exit_ts': df.index[entry_bar + max_bars],
                'side': side, 'raw_ret': ret, 'reason': 'max_bars'}

    # SHORT
    trough = entry_price
    sl_price = entry_price * (1 + trail_dist)
    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= n:
            exit_p = float(df['close'].iloc[-1])
            ret = (entry_price - exit_p) / entry_price
            pnl = ret - 2 * commission
            return {'outcome': 'EOD', 'pnl_pct': pnl, 'bars': i,
                    'exit_price': exit_p, 'entry_ts': entry_ts,
                    'exit_ts': df.index[-1], 'side': side,
                    'raw_ret': ret, 'reason': 'eod'}
        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        # 1) salida vs SL heredado (SHORT: SL toca por arriba)
        if hi >= sl_price:
            ret = (entry_price - sl_price) / entry_price
            pnl = ret - 2 * commission
            outcome = 'TP' if sl_price < entry_price else 'SL'
            return {'outcome': outcome, 'pnl_pct': pnl, 'bars': i,
                    'exit_price': sl_price, 'entry_ts': entry_ts,
                    'exit_ts': df.index[b], 'side': side,
                    'raw_ret': ret, 'reason': 'trail'}
        # 2) actualizar trough/SL para la siguiente vela
        if lo < trough:
            trough = lo
        sl_price = min(sl_price, trough * (1 + trail_dist))
    # Timeout
    exit_p = float(df['close'].iloc[entry_bar + max_bars])
    ret = (entry_price - exit_p) / entry_price
    pnl = ret - 2 * commission
    return {'outcome': 'TIMEOUT', 'pnl_pct': pnl, 'bars': max_bars,
            'exit_price': exit_p, 'entry_ts': entry_ts,
            'exit_ts': df.index[entry_bar + max_bars],
            'side': side, 'raw_ret': ret, 'reason': 'max_bars'}


# =============================================================================
# RUN_BACKTEST — una posicion a la vez en SOL
# =============================================================================
def run_backtest(df_sol: pd.DataFrame,
                 params: dict = PARAMS,
                 start_dt: Optional[pd.Timestamp] = None,
                 end_dt: Optional[pd.Timestamp] = None) -> list:
    """
    Recorre velas SOL. Una posicion a la vez (guard estricto).
    """
    trades = []
    n = len(df_sol)
    next_entry_idx = 0
    # rango temporal
    if start_dt is not None or end_dt is not None:
        idx_mask = np.ones(n, dtype=bool)
        if start_dt is not None:
            idx_mask &= df_sol.index >= start_dt
        if end_dt is not None:
            idx_mask &= df_sol.index <= end_dt
        valid_idxs = np.where(idx_mask)[0]
    else:
        valid_idxs = np.arange(n)

    valid_set = set(valid_idxs.tolist())
    i = 0
    while i < n - 1:
        if i not in valid_set or i < next_entry_idx:
            i += 1
            continue
        sig = signal(df_sol, i, params)
        if sig in ('LONG', 'SHORT'):
            out = simulate(df_sol, i, params, sig)
            if out['outcome'] != 'SKIP':
                out['asset'] = 'SOL'
                trades.append(out)
                # next_entry: vela siguiente al exit
                next_entry_idx = i + int(out['bars']) + 1
                i = next_entry_idx
                continue
        i += 1

    trades.sort(key=lambda t: pd.Timestamp(t['entry_ts']))
    return trades


# =============================================================================
# METRICAS
# =============================================================================
def metrics(trades: list) -> dict:
    if not trades:
        return {'n': 0, 'wr': 0.0, 'pf': 0.0, 'avg_pnl': 0.0,
                'total_return': 0.0, 'max_dd': 0.0, 'sharpe_like': 0.0,
                'months': 0.0, 'monthly_return': 0.0, 'annual_return': 0.0}

    pnls = np.array([t['pnl_pct'] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    n = len(pnls)
    wr = len(wins) / n
    gw = float(wins.sum())
    gl = float(abs(losses.sum()))
    pf = (gw / gl) if gl > 1e-9 else float('inf')

    eq = 1.0
    peak = 1.0
    dd = 0.0
    for p in pnls:
        eq *= (1 + p)
        peak = max(peak, eq)
        dd = max(dd, (peak - eq) / peak)
    total = eq - 1.0

    t0 = pd.to_datetime(trades[0]['entry_ts'])
    tL = pd.to_datetime(trades[-1]['exit_ts'])
    months = max(1.0, (tL - t0).days / 30.0)
    years = months / 12.0
    monthly_return = (eq ** (1 / months) - 1) if eq > 0 else -1.0
    annual_return = (eq ** (1 / years) - 1) if eq > 0 and years > 0 else -1.0

    sl = float(pnls.mean() / pnls.std()) if pnls.std() > 0 else 0.0

    return {'n': n, 'wr': float(wr), 'pf': float(pf), 'avg_pnl': float(pnls.mean()),
            'total_return': float(total), 'max_dd': float(dd),
            'sharpe_like': sl,
            'months': months, 'years': years,
            'monthly_return': float(monthly_return),
            'annual_return': float(annual_return)}
