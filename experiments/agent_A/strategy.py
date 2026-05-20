"""
Agent A — BTC/USDT 4h TREND-FOLLOWING / BREAKOUT with wide ATR trailing stop.
============================================================================

Filosofia
---------
1. Inspirada en el unico exito real del proyecto (V7: 322% anual con trailing
   stop). El "edge esta en la salida", no en la entrada.
2. Trend-following clasico estilo Donchian / Turtle:
   - filtro de regimen diario (EMA daily 50>200, con shift(1))
   - entrada cuando close 4h rompe el high de las ultimas N velas
   - confirmacion de volumen + filtro ADX para evitar choppy ranges
   - trailing stop AMPLIO (2.5x ATR o 3% floor) — el bug previo era tight
3. LONG-only en BTC (en este proyecto, SHORT en BTC ha sido trampa
   historicamente y la regla "no aprobar SHORT sin WR>break-even y WF 7/12"
   es muy estricta para un solo experimento).

Cero overfitting — auditorias aplicadas
---------------------------------------
- Una posicion a la vez: tras abrir un trade, se salta hasta despues de la
  vela en que cierra (igual que `revalidate_v15.py:sim_long_trailing` y
  que el bot real con MAX_CONCURRENT=1 por par).
- Sin look-ahead intrabar: el SL trailing comprueba la salida con el peak
  HEREDADO de velas anteriores; solo despues actualiza peak/stop para la
  vela siguiente. Espejo exacto del simulador correcto en revalidate_v15.py.
- MTF (daily -> 4h): shift(1) en compute_macro_daily.
- high_donchian: rolling(N).max().shift(1).
- Funding z-score: rolling-z-score con shift(1) -> sin look-ahead.

API publica
-----------
- PARAMS: dict frozen con todos los hiperparametros.
- signal(df, idx, params) -> 'LONG' | None
- simulate(df, entry_bar, params) -> dict(outcome, pnl_pct, bars)
- prepare_data(df_4h, df_1d, df_funding) -> DataFrame con features listas
- run_backtest(df_features, params, start_i, end_i) -> list[trade dict]

El archivo es autocontenido: no importa nada de src/ ni del framework V15.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


# =============================================================================
# PARAMETROS FROZEN
# =============================================================================
# Decisiones sobre los valores: ver SELF-AUDIT en README.md. Cada parametro
# tiene una justificacion *a priori* (literatura / V7 / sentido comun).
PARAMS = {
    # --- Costes ---
    'commission': 0.0005,           # 0.05% por lado (igual que v15_framework)

    # --- Filtro de regimen (1d, shift(1)) ---
    'ema_fast_1d': 50,              # tendencia de medio plazo
    'ema_slow_1d': 200,             # filtro macro clasico (golden cross)

    # --- Entrada: Donchian breakout 4h ---
    'donchian_n': 55,               # 55 velas 4h ~= 9.2 dias (clasico Turtle)
    'vol_ma_n': 20,                 # ventana de volumen base
    'vol_ratio_min': 1.2,           # volumen > 1.2x promedio en la vela de ruptura
    'adx_n': 14,                    # ADX standard
    'adx_min': 18,                  # filtro de tendencia minima

    # --- Trailing ATR AMPLIO (anti-bug del proyecto) ---
    'atr_n': 14,
    'trail_atr_mult': 2.5,          # 2.5x ATR — amplio, no tight
    'trail_floor_pct': 0.025,       # piso del 2.5% (no menos que esto)
    'trail_ceiling_pct': 0.06,      # techo del 6% (cap razonable)
    'max_bars': 60,                 # 60 velas 4h = 10 dias maximo en trade

    # --- Filtro de funding (opcional, anti-euforia) ---
    'funding_z_n': 168,             # 4 semanas de ventana (4h * 168 = 28 dias)
    'funding_z_max': 2.5,           # bloquear LONG si z>2.5 (mercado sobrecomprado)
    'funding_enabled': True,

    # --- Operativos ---
    'cutoff_date': '2025-12-31',    # NUNCA mires datos despues de esto
    'min_bars_warmup': 250,         # warmup minimo para indicadores
}


# =============================================================================
# CARGA Y FEATURES
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
    """ADX clasico (Wilder). Devuelve Series adx."""
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


def prepare_data(df_4h: pd.DataFrame,
                 df_1d: pd.DataFrame | None = None,
                 df_funding: pd.DataFrame | None = None,
                 params: dict = PARAMS) -> pd.DataFrame:
    """
    Adjunta todas las features necesarias para signal() y simulate().
    Aplica el cutoff <= params['cutoff_date'] de forma INMEDIATA.
    Todos los features MTF llevan shift(1) -> sin look-ahead.
    """
    cutoff = pd.Timestamp(params['cutoff_date'], tz='UTC')
    df = df_4h.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
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

    # MTF: EMA daily con shift(1)
    if df_1d is not None:
        df_d = df_1d.copy()
        if df_d.index.tz is None:
            df_d.index = df_d.index.tz_localize('UTC')
        df_d = df_d[df_d.index <= cutoff].sort_index()
        ema_fast = _ema(df_d['close'], params['ema_fast_1d'])
        ema_slow = _ema(df_d['close'], params['ema_slow_1d'])
        bull_d = (ema_fast > ema_slow).astype(int)
        bull_d = bull_d.shift(1)  # CRITICAL: shift(1) = sin look-ahead
        # Reindex al 4h con ffill (cada vela 4h del dia D usa el regimen del dia D-1)
        df['bull_1d'] = bull_d.reindex(df.index, method='ffill')
    else:
        # Fallback: derivar EMA daily desde el propio 4h (con shift(1))
        daily_close = df['close'].resample('1D').last().dropna()
        ema_fast = _ema(daily_close, params['ema_fast_1d'])
        ema_slow = _ema(daily_close, params['ema_slow_1d'])
        bull_d = (ema_fast > ema_slow).astype(int).shift(1)
        df['bull_1d'] = bull_d.reindex(df.index, method='ffill')

    # Funding rate z-score (con shift(1))
    if df_funding is not None and params.get('funding_enabled', True):
        fund = df_funding.copy()
        if fund.index.tz is None:
            fund.index = fund.index.tz_localize('UTC')
        fund = fund[fund.index <= cutoff].sort_index()
        # Resample funding a 4h con ffill
        if 'funding_rate' in fund.columns:
            fund_s = fund['funding_rate'].resample('4h').ffill()
        else:
            fund_s = fund.iloc[:, 0].resample('4h').ffill()
        n_z = params['funding_z_n']
        fmean = fund_s.rolling(n_z).mean()
        fstd = fund_s.rolling(n_z).std()
        z = (fund_s - fmean) / fstd.replace(0, np.nan)
        z = z.shift(1)  # CRITICAL: la z conocida en la vela actual usa solo info <=t-1
        df['funding_z'] = z.reindex(df.index, method='ffill')
    else:
        df['funding_z'] = 0.0

    # Limpieza: nan de warmup
    df = df.dropna(subset=['atr', 'adx', 'donchian_high', 'vol_ratio', 'bull_1d'])
    return df


# =============================================================================
# SIGNAL
# =============================================================================
def signal(df: pd.DataFrame, idx: int, params: dict = PARAMS) -> str | None:
    """
    Devuelve 'LONG' si todas las condiciones se cumplen en la vela `idx`
    (vela ya CERRADA). Sin look-ahead: usa solo info en posiciones <= idx.
    """
    if idx < params['min_bars_warmup']:
        return None
    if idx >= len(df) - 2:  # necesitamos al menos 1 vela posterior para simular
        return None

    row = df.iloc[idx]
    # 1) Filtro regimen daily: EMA50_1d > EMA200_1d (bull macro)
    if not (row.get('bull_1d', 0) >= 1):
        return None

    # 2) Donchian breakout: close > max(high de las N velas anteriores)
    dh = row.get('donchian_high', np.nan)
    if pd.isna(dh) or row['close'] <= dh:
        return None

    # 3) Confirmacion de volumen
    if pd.isna(row.get('vol_ratio', np.nan)) or row['vol_ratio'] < params['vol_ratio_min']:
        return None

    # 4) Filtro ADX: tendencia minima
    if pd.isna(row.get('adx', np.nan)) or row['adx'] < params['adx_min']:
        return None

    # 5) Funding veto: bloquear LONG si el mercado esta sobrecargado
    if params.get('funding_enabled', True):
        fz = row.get('funding_z', 0.0)
        if pd.notna(fz) and fz > params['funding_z_max']:
            return None

    return 'LONG'


# =============================================================================
# SIMULATE — trailing stop ATR amplio, SIN look-ahead intrabar
# =============================================================================
def simulate(df: pd.DataFrame, entry_bar: int, params: dict = PARAMS) -> dict:
    """
    Simula un trade LONG abierto en el CIERRE de `entry_bar` con trailing stop.

    Sin look-ahead intrabar:
      Para cada vela b > entry_bar:
        1) comprobar salida contra el SL HEREDADO de velas anteriores
        2) recien despues actualizar peak/SL con high[b] para la SIGUIENTE vela

    Devuelve dict(outcome, pnl_pct, bars, exit_price, entry_ts, exit_ts).
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
            # se acaba la data -> cerrar a close del ultimo bar
            exit_p = float(df['close'].iloc[-1])
            pnl = (exit_p - entry_price) / entry_price - 2 * commission
            return {'outcome': 'TIMEOUT' if pnl > 0 else 'TIMEOUT',
                    'pnl_pct': pnl, 'bars': i,
                    'exit_price': exit_p, 'entry_ts': entry_ts,
                    'exit_ts': df.index[-1], 'reason': 'eod'}

        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])

        # 1) Salida contra SL ya conocido
        if lo <= sl_price:
            pnl = (sl_price - entry_price) / entry_price - 2 * commission
            return {'outcome': ('TP' if sl_price > entry_price else 'SL'),
                    'pnl_pct': pnl, 'bars': i,
                    'exit_price': sl_price, 'entry_ts': entry_ts,
                    'exit_ts': df.index[b], 'reason': 'trail'}

        # 2) Actualizar peak/SL para la SIGUIENTE vela (no para esta)
        if hi > peak:
            peak = hi
        sl_price = max(sl_price, peak * (1 - trail_dist))

    # Timeout: cerrar a close del max_bars
    exit_p = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (exit_p - entry_price) / entry_price - 2 * commission
    return {'outcome': 'TIMEOUT', 'pnl_pct': pnl, 'bars': max_bars,
            'exit_price': exit_p, 'entry_ts': entry_ts,
            'exit_ts': df.index[entry_bar + max_bars], 'reason': 'max_bars'}


# =============================================================================
# RUN_BACKTEST — motor de UNA POSICION A LA VEZ (no solapado)
# =============================================================================
def run_backtest(df: pd.DataFrame, params: dict = PARAMS,
                 start_i: int | None = None, end_i: int | None = None) -> list[dict]:
    """
    Recorre las velas [start_i, end_i). Al abrir un trade, salta hasta DESPUES
    de la vela en que cierra -> jamas solapa posiciones.
    Devuelve list of trade dicts.
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
        # CRITICO: avanzar hasta DESPUES de la vela de cierre (sin solapar)
        i += max(1, out['bars']) + 1
    return trades


# =============================================================================
# METRICAS
# =============================================================================
def metrics(trades: list[dict]) -> dict:
    if not trades:
        return {'n': 0, 'wr': 0.0, 'pf': 0.0, 'avg_pnl': 0.0,
                'total_return': 0.0, 'max_dd': 0.0, 'sharpe_like': 0.0,
                'months': 0.0, 'monthly_return': 0.0}
    pnls = np.array([t['pnl_pct'] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    n = len(pnls)
    wr = len(wins) / n
    gw = float(wins.sum())
    gl = float(abs(losses.sum()))
    pf = (gw / gl) if gl > 1e-9 else float('inf')

    # equity sequencial (1 posicion a la vez -> los trades NO se solapan)
    eq = 1.0
    peak = 1.0
    dd = 0.0
    for p in pnls:
        eq *= (1 + p)
        peak = max(peak, eq)
        dd = max(dd, (peak - eq) / peak)
    total = eq - 1.0

    # tiempo cubierto
    t0 = pd.to_datetime(trades[0]['entry_ts'])
    tL = pd.to_datetime(trades[-1]['exit_ts'])
    months = max(1.0, (tL - t0).days / 30.0)
    monthly_return = (eq ** (1 / months) - 1) if eq > 0 else -1.0

    # Sharpe-like sobre pnl por trade (sin anualizar — sirve para comparar)
    sl = float(pnls.mean() / pnls.std()) if pnls.std() > 0 else 0.0
    return {'n': n, 'wr': float(wr), 'pf': float(pf), 'avg_pnl': float(pnls.mean()),
            'total_return': float(total), 'max_dd': float(dd),
            'sharpe_like': sl, 'months': months,
            'monthly_return': float(monthly_return)}
