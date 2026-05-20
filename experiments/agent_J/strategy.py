"""
Agent J - ETH/USDT 1D TREND-FOLLOWING (Donchian breakout + ATR trailing)
==========================================================================
Rescaled-A on ETH 1D. Same mechanism as agent_A (BTC 4h) but with windows
proportionally rescaled from 4h -> 1D, set A PRIORI (before seeing results).

Hipotesis a falsar
------------------
ETH en 1D tiene ciclos mas limpios que 4h (menor ruido, mayor relacion
senal/ruido). El mismo mecanismo de A puede tener edge mas claro.

REGLAS DE RESCALING (a priori, justificadas por horizonte temporal)
-------------------------------------------------------------------
| Parametro       | 4h (A)             | 1D (J)            | Justificacion                |
|-----------------|--------------------|-------------------|------------------------------|
| Donchian        | 55 bars (~9 dias)  | 10 bars (~10 dias)| Mismo horizonte temporal     |
| ATR             | 14 bars (~2.3 dias)| 14 bars (~14 dias)| Mismo concepto Wilder        |
| EMAs daily      | 50/200 daily       | 50/200 daily      | Sin cambios (ya son daily)   |
| max_bars        | 60 bars (10 dias)  | 10 bars (10 dias) | Mismo techo temporal         |
| vol_ma          | 20 bars            | 20 bars           | Comparable a MA mensual      |
| trail_atr_mult  | 2.5x               | 2.5x              | Constante universal          |
| trail_floor_pct | 2.5%               | 2.5%              | Misma cobertura de comisiones|
| trail_ceiling_pct| 6%                | 6%                | Mismo cap razonable          |
| ADX min         | 18                 | 18                | Igual (filtro de tendencia)  |
| vol_ratio_min   | 1.2                | 1.2               | Igual                        |
| funding         | activado           | DESACTIVADO       | ETH no tiene funding parquet |
|                 |                    |                   | en data/ -> honesto disable  |

Nota sobre funding: data/ tiene btc_v15_funding.parquet pero NO ethusdt_funding.
Re-usar BTC funding como proxy seria un hack. Lo HONESTO es desactivarlo.
Esto puede dejar pasar entradas en momentos de euforia ETH, pero es preferible
a inventar datos.

Sin overfitting
---------------
- Una posicion a la vez (igual que A): tras abrir trade, salta hasta DESPUES
  de la vela de cierre. Mirror exacto de revalidate_v15.py::sim_long_trailing.
- Sin look-ahead intrabar en el trailing: SL se comprueba contra el peak
  HEREDADO de velas anteriores; recien despues se actualiza peak/SL.
- MTF (daily -> daily): trivial - los datos ya son 1D. EMAs se calculan
  directamente sobre el cierre 1D con shift(1) para evitar mirar la vela
  actual desde su propio regimen.
- Donchian: rolling(N).max().shift(1) - excluye la vela actual.
- Cutoff inviolable: df = df[df.index <= '2025-12-31'].

API publica - identica a A
--------------------------
- PARAMS: dict frozen
- signal(df, idx, params) -> 'LONG' | None
- simulate(df, entry_bar, params) -> dict(outcome, pnl_pct, bars, ...)
- prepare_data(df_1d, params) -> DataFrame con features
- run_backtest(df_features, params, start_i, end_i) -> list[trade dict]
- metrics(trades) -> dict
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# =============================================================================
# PARAMETROS FROZEN - rescalados A PRIORI desde A (BTC 4h)
# =============================================================================
PARAMS = {
    # --- Costes (iguales que A) ---
    'commission': 0.0005,           # 0.05% por lado

    # --- Filtro de regimen (1d, shift(1)) - igual que A ---
    'ema_fast_1d': 50,              # tendencia de medio plazo
    'ema_slow_1d': 200,             # filtro macro clasico

    # --- Entrada: Donchian breakout 1D - RESCALADO ---
    # A: 55 bars 4h = ~9 dias. J: 10 bars 1D = ~10 dias. Mismo horizonte.
    'donchian_n': 10,
    'vol_ma_n': 20,                 # igual que A
    'vol_ratio_min': 1.2,           # igual que A
    'adx_n': 14,                    # igual que A
    'adx_min': 18,                  # igual que A

    # --- Trailing ATR - mismas magnitudes que A ---
    'atr_n': 14,                    # igual que A (Wilder)
    'trail_atr_mult': 2.5,          # igual que A (constante universal)
    'trail_floor_pct': 0.025,       # igual que A
    'trail_ceiling_pct': 0.06,      # igual que A
    'max_bars': 10,                 # RESCALADO: A=60 bars 4h=10dias -> J=10 bars 1D=10dias

    # --- Funding: DESACTIVADO (no hay parquet de ETH funding) ---
    'funding_enabled': False,
    'funding_z_n': 28,              # placeholder (no se usa)
    'funding_z_max': 2.5,

    # --- Operativos ---
    'cutoff_date': '2025-12-31',
    'min_bars_warmup': 250,         # ~250 dias = warmup para EMA200
}


# =============================================================================
# INDICADORES
# =============================================================================
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's RMA (EMA con alpha=1/n)."""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def _true_range(h, l, c):
    pc = c.shift(1)
    return pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def _atr(h, l, c, n=14):
    return _rma(_true_range(h, l, c), n)


def _adx(h, l, c, n=14):
    """ADX clasico (Wilder)."""
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                        index=h.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                         index=h.index)
    tr = _true_range(h, l, c)
    atr_n = _rma(tr, n)
    plus_di = 100 * _rma(plus_dm, n) / atr_n.replace(0, np.nan)
    minus_di = 100 * _rma(minus_dm, n) / atr_n.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _rma(dx, n)


# =============================================================================
# PREPARE_DATA - 1D data, no MTF needed (ya es daily)
# =============================================================================
def prepare_data(df_1d: pd.DataFrame, params: dict = PARAMS) -> pd.DataFrame:
    """
    Adjunta features al DataFrame 1D. Aplica cutoff inmediatamente.

    Nota: como los datos ya son 1D, las EMA daily se calculan sobre el
    propio cierre y se shifta(1) para que el regimen del dia D solo use
    info <= D-1 (igual logica que A pero sin reindex).
    """
    cutoff = pd.Timestamp(params['cutoff_date'])
    df = df_1d.copy()
    # Normalizar tz (los parquets de ETH no tienen tz; otros si)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df = df[df.index <= cutoff].sort_index()

    h, l, c, v = df['high'], df['low'], df['close'], df['volume']

    # ATR
    df['atr'] = _atr(h, l, c, params['atr_n'])
    df['atr_pct'] = df['atr'] / c

    # ADX
    df['adx'] = _adx(h, l, c, params['adx_n'])

    # Donchian (alto de las N velas anteriores, EXCLUYENDO la actual)
    df['donchian_high'] = h.rolling(params['donchian_n']).max().shift(1)

    # Volume ratio
    df['vol_ma'] = v.rolling(params['vol_ma_n']).mean()
    df['vol_ratio'] = v / df['vol_ma'].replace(0, np.nan)

    # Regimen daily: EMA fast > EMA slow, con shift(1) para evitar look-ahead
    ema_fast = _ema(c, params['ema_fast_1d'])
    ema_slow = _ema(c, params['ema_slow_1d'])
    # El regimen del dia D solo usa cierres <= D-1 -> shift(1)
    df['bull_1d'] = (ema_fast > ema_slow).astype(int).shift(1)

    # Funding (desactivado por defecto en ETH 1D)
    df['funding_z'] = 0.0

    df = df.dropna(subset=['atr', 'adx', 'donchian_high', 'vol_ratio', 'bull_1d'])
    return df


# =============================================================================
# SIGNAL
# =============================================================================
def signal(df: pd.DataFrame, idx: int, params: dict = PARAMS) -> str | None:
    """
    Devuelve 'LONG' si todas las condiciones se cumplen en la vela `idx`.
    Sin look-ahead: solo info en posiciones <= idx.
    """
    if idx < params['min_bars_warmup']:
        return None
    if idx >= len(df) - 2:
        return None

    row = df.iloc[idx]
    # 1) Filtro regimen daily
    if not (row.get('bull_1d', 0) >= 1):
        return None

    # 2) Donchian breakout
    dh = row.get('donchian_high', np.nan)
    if pd.isna(dh) or row['close'] <= dh:
        return None

    # 3) Volumen
    if pd.isna(row.get('vol_ratio', np.nan)) or row['vol_ratio'] < params['vol_ratio_min']:
        return None

    # 4) ADX
    if pd.isna(row.get('adx', np.nan)) or row['adx'] < params['adx_min']:
        return None

    # 5) Funding (desactivado por defecto)
    if params.get('funding_enabled', False):
        fz = row.get('funding_z', 0.0)
        if pd.notna(fz) and fz > params['funding_z_max']:
            return None

    return 'LONG'


# =============================================================================
# SIMULATE - trailing stop, SIN look-ahead intrabar
# =============================================================================
def simulate(df: pd.DataFrame, entry_bar: int, params: dict = PARAMS) -> dict:
    """
    Trade LONG abierto en el CIERRE de `entry_bar` con trailing stop ATR.
    Mismo simulador que A (mirror exacto), sin look-ahead intrabar.
    """
    n = len(df)
    entry_price = float(df['close'].iloc[entry_bar])
    entry_ts = df.index[entry_bar]
    atr_pct = float(df['atr_pct'].iloc[entry_bar])
    if not np.isfinite(atr_pct) or atr_pct <= 0:
        return {'outcome': 'SKIP', 'pnl_pct': 0.0, 'bars': 0,
                'exit_price': entry_price, 'entry_ts': entry_ts,
                'exit_ts': entry_ts, 'reason': 'no_atr'}

    trail_dist = atr_pct * params['trail_atr_mult']
    trail_dist = max(params['trail_floor_pct'], min(params['trail_ceiling_pct'], trail_dist))

    sl_price = entry_price * (1 - trail_dist)
    peak = entry_price
    max_bars = params['max_bars']
    commission = params['commission']

    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= n:
            exit_p = float(df['close'].iloc[-1])
            pnl = (exit_p - entry_price) / entry_price - 2 * commission
            return {'outcome': 'TIMEOUT', 'pnl_pct': pnl, 'bars': i,
                    'exit_price': exit_p, 'entry_ts': entry_ts,
                    'exit_ts': df.index[-1], 'reason': 'eod'}

        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])

        # 1) Salida contra SL HEREDADO (sin look-ahead intrabar)
        if lo <= sl_price:
            pnl = (sl_price - entry_price) / entry_price - 2 * commission
            return {'outcome': ('TP' if sl_price > entry_price else 'SL'),
                    'pnl_pct': pnl, 'bars': i,
                    'exit_price': sl_price, 'entry_ts': entry_ts,
                    'exit_ts': df.index[b], 'reason': 'trail'}

        # 2) Actualizar peak/SL para la SIGUIENTE vela
        if hi > peak:
            peak = hi
        sl_price = max(sl_price, peak * (1 - trail_dist))

    # Timeout: cerrar al cierre del max_bars
    exit_p = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (exit_p - entry_price) / entry_price - 2 * commission
    return {'outcome': 'TIMEOUT', 'pnl_pct': pnl, 'bars': max_bars,
            'exit_price': exit_p, 'entry_ts': entry_ts,
            'exit_ts': df.index[entry_bar + max_bars], 'reason': 'max_bars'}


# =============================================================================
# RUN_BACKTEST - UNA POSICION A LA VEZ (no solapado)
# =============================================================================
def run_backtest(df: pd.DataFrame, params: dict = PARAMS,
                 start_i: int | None = None, end_i: int | None = None) -> list[dict]:
    """
    Recorre velas [start_i, end_i). Al abrir trade salta hasta DESPUES de la
    vela de cierre. Sin solape.
    """
    if start_i is None:
        start_i = params['min_bars_warmup']
    if end_i is None:
        end_i = len(df)

    trades = []
    i = max(start_i, params['min_bars_warmup'])
    while i < end_i:
        sig = signal(df, i, params)
        if sig != 'LONG':
            i += 1
            continue
        out = simulate(df, i, params)
        if out['outcome'] == 'SKIP':
            i += 1
            continue
        trades.append({
            'entry_ts': out['entry_ts'],
            'exit_ts': out['exit_ts'],
            'outcome': out['outcome'],
            'pnl_pct': out['pnl_pct'],
            'bars': out['bars'],
            'side': sig,
        })
        # CRITICO: saltar DESPUES de la vela de cierre
        i += max(1, out['bars']) + 1
    return trades


# =============================================================================
# METRICAS
# =============================================================================
def metrics(trades: list[dict]) -> dict:
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

    # Equity secuencial - una posicion a la vez
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
    days = max(1.0, (tL - t0).days)
    months = max(1.0, days / 30.0)
    years = max(1.0 / 12.0, days / 365.0)
    monthly_return = (eq ** (1 / months) - 1) if eq > 0 else -1.0
    annual_return = (eq ** (1 / years) - 1) if eq > 0 else -1.0

    sl = float(pnls.mean() / pnls.std()) if pnls.std() > 0 else 0.0
    return {'n': n, 'wr': float(wr), 'pf': float(pf), 'avg_pnl': float(pnls.mean()),
            'total_return': float(total), 'max_dd': float(dd),
            'sharpe_like': sl, 'months': months, 'years': years,
            'monthly_return': float(monthly_return),
            'annual_return': float(annual_return)}
