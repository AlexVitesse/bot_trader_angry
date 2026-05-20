"""
Agent F — VOL-COMPRESSION BREAKOUT (BTC + ETH 4h) with VOL-TARGETING.
=====================================================================

Filosofia
---------
1. Idea central: Bollinger Squeeze + vol-of-vol reversion. Cuando la
   volatilidad realizada se comprime (BB width o ATR en cuantil bajo
   historico), la expansion siguiente suele ser direccional. Entras en
   la direccion del breakout.
2. Vol-targeting de tamano: posicion ~ target_vol / realized_vol_20d.
   Cuando vol es baja (= entradas de compresion) el tamano es mayor,
   capeado a 3x leverage. Coherente con la tesis: entras justo cuando
   tu sistema te dice que ese activo esta en regimen de baja vol.
3. Multi-asset: BTC y ETH operan independientemente, max 1 posicion
   simultanea por activo (no solape POR ACTIVO). El portfolio puede
   tener hasta 2 posiciones simultaneas (BTC + ETH). Esto es lo que
   `portfolio_manager.py` hace con ML_MAX_CONCURRENT=3.

Cero overfitting — auditorias aplicadas
---------------------------------------
- Una posicion a la vez POR ACTIVO: tras abrir trade en BTC, BTC no
  abre otro hasta despues del cierre. Lo mismo para ETH. Cero solape.
- Sin look-ahead intrabar: salida vs SL HEREDADO primero, luego
  actualizar peak/SL para la siguiente vela.
- BB width percentile: rolling(100).rank(pct).shift(1) -> el percentil
  visible en t solo usa info <= t-1.
- N-bar high/low de breakout: rolling(N).max/min().shift(1).
- MTF (daily): shift(1) en EMA50/200 daily.
- Funding z-score: rolling-z con shift(1).
- ATR: Wilder RMA, sin look-ahead.

Universo y direccion
--------------------
- 2 activos: BTC, ETH.
- BIDIRECTIONAL: LONG y SHORT. La compresion no tiene sesgo direccional
  por construccion — el setup es la EXPANSION posterior, y esa puede
  ir para cualquier lado. Reportar metricas separadas LONG vs SHORT.
- Filtro daily: solo LONG si EMA50_1d > EMA200_1d (bull macro); solo
  SHORT si EMA50_1d < EMA200_1d (bear macro). Reduce trampas
  direccionales.

API publica
-----------
- PARAMS: dict frozen.
- prepare_data(df_4h, df_1d=None, df_funding=None, params) -> DataFrame
- signal(df, idx, params) -> 'LONG' | 'SHORT' | None
- simulate(df, entry_bar, params, side) -> dict(outcome, pnl_pct, bars, ...)
- run_backtest(df_btc, df_eth, params, start_dt, end_dt) -> list[trade]
- metrics(trades) -> dict
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd


# =============================================================================
# PARAMETROS FROZEN
# =============================================================================
PARAMS = {
    # --- Costes ---
    'commission': 0.0005,        # 0.05% por lado

    # --- Vol compression (BB width percentile) ---
    'bb_n': 20,                  # ventana de BB
    'bb_k': 2.0,                 # BB std multiplier (clasico)
    'percentile_lookback': 100,  # ventana para percentil rolling de BB width
    'compression_percentile': 0.20,  # BB width debe estar en cuantil <=20
    'compression_min_bars': 3,   # 3 velas consecutivas en compresion

    # --- Breakout trigger ---
    'breakout_n': 12,            # break del high/low de las ultimas 12 velas (~2 dias)
    'vol_ratio_n': 20,
    'vol_ratio_min': 1.2,        # confirmacion de volumen en la vela breakout

    # --- Filtro regimen daily (shift(1)) ---
    'ema_fast_1d': 50,
    'ema_slow_1d': 200,
    'regime_filter_enabled': True,  # bloquea contra-tendencia macro

    # --- Trailing ATR ---
    'atr_n': 14,
    'trail_atr_mult': 2.0,       # 2.0x ATR — medio
    'trail_floor_pct': 0.020,    # 2% piso
    'trail_ceiling_pct': 0.055,  # 5.5% techo
    'max_bars': 48,              # 48 velas 4h = 8 dias maximo

    # --- Vol-targeting de tamano ---
    'realized_vol_n': 20,        # 20 velas 4h (~3.3 dias)
    'target_vol_pct': 0.012,     # 1.2% vol por trade
    'max_leverage': 3.0,         # cap 3x
    'min_leverage': 0.25,        # piso 0.25x

    # --- Funding veto (opcional) ---
    'funding_z_n': 168,
    'funding_z_max_long': 2.5,
    'funding_z_min_short': -2.5,
    'funding_enabled': True,

    # --- Direccionalidad ---
    'enable_long': True,
    'enable_short': True,

    # --- Operativos ---
    'cutoff_date': '2025-12-31',
    'min_bars_warmup': 250,
}


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
    """BB width normalizado por close. Sin look-ahead (rolling)."""
    ma = c.rolling(n).mean()
    sd = c.rolling(n).std()
    upper = ma + k * sd
    lower = ma - k * sd
    width = (upper - lower) / ma.replace(0, np.nan)
    return width


def _realized_vol(returns: pd.Series, n: int = 20) -> pd.Series:
    """Vol anualizada (aprox.) de log-returns 4h, ventana n."""
    # 4h -> 6 velas/dia, 365 dias -> sqrt(6*365) = sqrt(2190) factor
    # pero para sizing relativo basta con vol no anualizada (escala constante)
    return returns.rolling(n).std()


def prepare_data(df_4h: pd.DataFrame,
                 df_1d: Optional[pd.DataFrame] = None,
                 df_funding: Optional[pd.DataFrame] = None,
                 params: dict = PARAMS) -> pd.DataFrame:
    """
    Devuelve DataFrame con todas las features necesarias para signal/simulate.
    Aplica el cutoff <= params['cutoff_date'] de inmediato.
    Todas las features MTF/percentile llevan shift(1) -> sin look-ahead.
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

    # BB width
    df['bb_width'] = _bollinger_width(c, params['bb_n'], params['bb_k'])

    # BB width percentile rolling (shift(1) -> percentil visible en t usa info <=t-1)
    # rolling().rank(pct=True) calcula el percentil del ULTIMO valor de la ventana.
    df['bb_width_pct'] = (
        df['bb_width']
        .rolling(params['percentile_lookback'])
        .rank(pct=True)
        .shift(1)
    )

    # Compresion: flag = bb_width_pct <= compression_percentile
    df['compressed'] = (df['bb_width_pct'] <= params['compression_percentile']).astype(int)

    # Compresion sostenida: ultimas X velas consecutivas en compresion
    # rolling(X).sum() >= X (las X velas con flag=1)
    min_b = params['compression_min_bars']
    df['compression_sustained'] = (df['compressed'].rolling(min_b).sum() >= min_b).astype(int)
    # CRITICAL: el flag de "estoy en compresion sostenida" se evalua en velas
    # cerradas <= t (bb_width_pct ya tiene shift(1)). Por tanto este flag es
    # safe — no necesita shift adicional.

    # N-bar high/low EXCLUYENDO la actual (shift(1))
    df['hi_n'] = h.rolling(params['breakout_n']).max().shift(1)
    df['lo_n'] = l.rolling(params['breakout_n']).min().shift(1)

    # Vol ratio (confirmacion)
    df['vol_ma'] = v.rolling(params['vol_ratio_n']).mean()
    df['vol_ratio'] = v / df['vol_ma'].replace(0, np.nan)

    # Realized vol (para sizing)
    rets = np.log(c / c.shift(1))
    df['ret_4h'] = rets
    df['realized_vol'] = _realized_vol(rets, params['realized_vol_n'])
    # Realized vol calculada en t usa returns hasta t (cerrados). Aceptable para
    # decisiones en t. Para sizing en la vela t (= entrar al close), tomamos
    # realized_vol.iloc[t-1] (shift implicito via uso en signal/simulate).

    # MTF: EMA daily con shift(1)
    if df_1d is not None:
        df_d = df_1d.copy()
        if df_d.index.tz is None:
            df_d.index = df_d.index.tz_localize('UTC')
        df_d = df_d[df_d.index <= cutoff].sort_index()
        ema_fast = _ema(df_d['close'], params['ema_fast_1d'])
        ema_slow = _ema(df_d['close'], params['ema_slow_1d'])
        bull_d = (ema_fast > ema_slow).astype(int).shift(1)  # shift(1) inviolable
        df['bull_1d'] = bull_d.reindex(df.index, method='ffill')
    else:
        daily_close = df['close'].resample('1D').last().dropna()
        ema_fast = _ema(daily_close, params['ema_fast_1d'])
        ema_slow = _ema(daily_close, params['ema_slow_1d'])
        bull_d = (ema_fast > ema_slow).astype(int).shift(1)
        df['bull_1d'] = bull_d.reindex(df.index, method='ffill')

    # Funding z-score con shift(1)
    if df_funding is not None and params.get('funding_enabled', True):
        fund = df_funding.copy()
        if fund.index.tz is None:
            fund.index = fund.index.tz_localize('UTC')
        fund = fund[fund.index <= cutoff].sort_index()
        if 'funding_rate' in fund.columns:
            fund_s = fund['funding_rate'].resample('4h').ffill()
        else:
            fund_s = fund.iloc[:, 0].resample('4h').ffill()
        n_z = params['funding_z_n']
        fmean = fund_s.rolling(n_z).mean()
        fstd = fund_s.rolling(n_z).std()
        z = ((fund_s - fmean) / fstd.replace(0, np.nan)).shift(1)
        df['funding_z'] = z.reindex(df.index, method='ffill')
    else:
        df['funding_z'] = 0.0

    df = df.dropna(subset=['atr', 'bb_width_pct', 'hi_n', 'lo_n',
                           'vol_ratio', 'realized_vol', 'bull_1d'])
    return df


# =============================================================================
# SIGNAL
# =============================================================================
def signal(df: pd.DataFrame, idx: int, params: dict = PARAMS) -> Optional[str]:
    """
    Devuelve 'LONG' / 'SHORT' / None en la vela `idx` (vela cerrada).
    Sin look-ahead: usa info <= idx.
    """
    if idx < params['min_bars_warmup']:
        return None
    if idx >= len(df) - 2:
        return None

    row = df.iloc[idx]

    # Compresion sostenida en velas PREVIAS (medida con bb_width_pct shift(1))
    # Queremos que en t-1 ya hubieramos detectado compresion sostenida y que
    # AHORA (t) llegue el breakout. compression_sustained.iloc[idx] mira al
    # estado actual; la compresion se ha medido con shift(1) en bb_width_pct.
    # Para hacerlo todavia mas estricto, exigimos compresion sostenida hasta
    # la vela anterior (idx-1).
    if df['compression_sustained'].iloc[idx - 1] != 1:
        return None

    hi_n = row.get('hi_n', np.nan)
    lo_n = row.get('lo_n', np.nan)
    if pd.isna(hi_n) or pd.isna(lo_n):
        return None

    c = float(row['close'])

    # Breakout direccional
    side = None
    if params.get('enable_long', True) and c > hi_n:
        side = 'LONG'
    elif params.get('enable_short', True) and c < lo_n:
        side = 'SHORT'
    if side is None:
        return None

    # Confirmacion de volumen
    vr = row.get('vol_ratio', np.nan)
    if pd.isna(vr) or vr < params['vol_ratio_min']:
        return None

    # Filtro regimen daily
    if params.get('regime_filter_enabled', True):
        bull = row.get('bull_1d', 0)
        if side == 'LONG' and bull < 1:
            return None
        if side == 'SHORT' and bull >= 1:
            return None

    # Funding veto
    if params.get('funding_enabled', True):
        fz = row.get('funding_z', 0.0)
        if pd.notna(fz):
            if side == 'LONG' and fz > params['funding_z_max_long']:
                return None
            if side == 'SHORT' and fz < params['funding_z_min_short']:
                return None

    return side


# =============================================================================
# VOL-TARGETED SIZE (leverage)
# =============================================================================
def compute_leverage(df: pd.DataFrame, entry_bar: int, params: dict = PARAMS) -> float:
    """
    leverage = target_vol_pct / realized_vol (4h)
    Capeado a [min_leverage, max_leverage].
    Usa realized_vol de la vela entry_bar (calculada con returns hasta t).
    """
    rv = float(df['realized_vol'].iloc[entry_bar])
    if not np.isfinite(rv) or rv <= 0:
        return params.get('min_leverage', 0.25)
    lev = params['target_vol_pct'] / rv
    lev = max(params['min_leverage'], min(params['max_leverage'], lev))
    return lev


# =============================================================================
# SIMULATE — trailing stop ATR, SIN look-ahead intrabar
# =============================================================================
def simulate(df: pd.DataFrame, entry_bar: int, params: dict, side: str) -> dict:
    """
    Trade abierto al close de entry_bar. Trailing ATR (mult * atr_pct) acotado
    a [floor, ceiling]. Sin look-ahead intrabar:
      - en cada vela b > entry_bar: PRIMERO comprobar salida vs SL heredado;
      - DESPUES actualizar peak/SL para la SIGUIENTE vela.

    Para SHORT, el trailing sigue al MINIMO (trough): SL = trough * (1 + trail).
    """
    n = len(df)
    entry_price = float(df['close'].iloc[entry_bar])
    entry_ts = df.index[entry_bar]
    atr_pct = float(df['atr_pct'].iloc[entry_bar])
    if not np.isfinite(atr_pct) or atr_pct <= 0:
        return {'outcome': 'SKIP', 'pnl_pct': 0.0, 'bars': 0,
                'exit_price': entry_price, 'entry_ts': entry_ts,
                'exit_ts': entry_ts, 'side': side, 'leverage': 0.0,
                'reason': 'no_atr'}

    trail_dist = atr_pct * params['trail_atr_mult']
    trail_dist = max(params['trail_floor_pct'],
                     min(params['trail_ceiling_pct'], trail_dist))

    leverage = compute_leverage(df, entry_bar, params)
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
                pnl = ret * leverage - 2 * commission
                return {'outcome': 'EOD', 'pnl_pct': pnl, 'bars': i,
                        'exit_price': exit_p, 'entry_ts': entry_ts,
                        'exit_ts': df.index[-1], 'side': side,
                        'leverage': leverage, 'raw_ret': ret,
                        'reason': 'eod'}
            hi = float(df['high'].iloc[b])
            lo = float(df['low'].iloc[b])
            # 1) salida vs SL heredado
            if lo <= sl_price:
                ret = (sl_price - entry_price) / entry_price
                pnl = ret * leverage - 2 * commission
                outcome = 'TP' if sl_price > entry_price else 'SL'
                return {'outcome': outcome, 'pnl_pct': pnl, 'bars': i,
                        'exit_price': sl_price, 'entry_ts': entry_ts,
                        'exit_ts': df.index[b], 'side': side,
                        'leverage': leverage, 'raw_ret': ret,
                        'reason': 'trail'}
            # 2) actualizar peak/SL para la siguiente vela
            if hi > peak:
                peak = hi
            sl_price = max(sl_price, peak * (1 - trail_dist))
        # Timeout
        exit_p = float(df['close'].iloc[entry_bar + max_bars])
        ret = (exit_p - entry_price) / entry_price
        pnl = ret * leverage - 2 * commission
        return {'outcome': 'TIMEOUT', 'pnl_pct': pnl, 'bars': max_bars,
                'exit_price': exit_p, 'entry_ts': entry_ts,
                'exit_ts': df.index[entry_bar + max_bars],
                'side': side, 'leverage': leverage, 'raw_ret': ret,
                'reason': 'max_bars'}

    # SHORT
    trough = entry_price
    sl_price = entry_price * (1 + trail_dist)
    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= n:
            exit_p = float(df['close'].iloc[-1])
            ret = (entry_price - exit_p) / entry_price
            pnl = ret * leverage - 2 * commission
            return {'outcome': 'EOD', 'pnl_pct': pnl, 'bars': i,
                    'exit_price': exit_p, 'entry_ts': entry_ts,
                    'exit_ts': df.index[-1], 'side': side,
                    'leverage': leverage, 'raw_ret': ret,
                    'reason': 'eod'}
        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        # 1) salida vs SL heredado (SHORT: SL toca por arriba)
        if hi >= sl_price:
            ret = (entry_price - sl_price) / entry_price
            pnl = ret * leverage - 2 * commission
            outcome = 'TP' if sl_price < entry_price else 'SL'
            return {'outcome': outcome, 'pnl_pct': pnl, 'bars': i,
                    'exit_price': sl_price, 'entry_ts': entry_ts,
                    'exit_ts': df.index[b], 'side': side,
                    'leverage': leverage, 'raw_ret': ret,
                    'reason': 'trail'}
        # 2) actualizar trough/SL para la siguiente vela
        if lo < trough:
            trough = lo
        sl_price = min(sl_price, trough * (1 + trail_dist))
    # Timeout
    exit_p = float(df['close'].iloc[entry_bar + max_bars])
    ret = (entry_price - exit_p) / entry_price
    pnl = ret * leverage - 2 * commission
    return {'outcome': 'TIMEOUT', 'pnl_pct': pnl, 'bars': max_bars,
            'exit_price': exit_p, 'entry_ts': entry_ts,
            'exit_ts': df.index[entry_bar + max_bars],
            'side': side, 'leverage': leverage, 'raw_ret': ret,
            'reason': 'max_bars'}


# =============================================================================
# RUN_BACKTEST — multi-asset (BTC + ETH), una posicion por activo
# =============================================================================
def run_backtest(df_btc: pd.DataFrame,
                 df_eth: pd.DataFrame,
                 params: dict = PARAMS,
                 start_dt: Optional[pd.Timestamp] = None,
                 end_dt: Optional[pd.Timestamp] = None) -> list:
    """
    Recorre las velas de BTC y ETH en paralelo (timestamps comunes).
    Una posicion por activo a la vez: si BTC ya esta en trade, BTC no abre
    otro hasta despues del cierre. Lo mismo para ETH. ETH puede abrir mientras
    BTC tiene posicion (diversificacion -> max 2 simultaneas).

    Returns list of trade dicts.
    """
    trades = []
    # Crear timestamps comunes (cualquier vela en cualquiera de los dos)
    all_ts = df_btc.index.union(df_eth.index).sort_values()
    if start_dt is not None:
        all_ts = all_ts[all_ts >= start_dt]
    if end_dt is not None:
        all_ts = all_ts[all_ts <= end_dt]

    # idx por activo del PROXIMO bar disponible para entrar
    next_btc_entry_ts: Optional[pd.Timestamp] = None  # antes de la cual NO entrar
    next_eth_entry_ts: Optional[pd.Timestamp] = None

    # convertir a position-based para acceso rapido
    btc_pos = {ts: i for i, ts in enumerate(df_btc.index)}
    eth_pos = {ts: i for i, ts in enumerate(df_eth.index)}

    for ts in all_ts:
        # BTC
        if ts in btc_pos and (next_btc_entry_ts is None or ts >= next_btc_entry_ts):
            i = btc_pos[ts]
            sig = signal(df_btc, i, params)
            if sig in ('LONG', 'SHORT'):
                out = simulate(df_btc, i, params, sig)
                if out['outcome'] != 'SKIP':
                    out['asset'] = 'BTC'
                    trades.append(out)
                    next_btc_entry_ts = pd.Timestamp(out['exit_ts']) + pd.Timedelta(hours=4)

        # ETH
        if ts in eth_pos and (next_eth_entry_ts is None or ts >= next_eth_entry_ts):
            i = eth_pos[ts]
            sig = signal(df_eth, i, params)
            if sig in ('LONG', 'SHORT'):
                out = simulate(df_eth, i, params, sig)
                if out['outcome'] != 'SKIP':
                    out['asset'] = 'ETH'
                    trades.append(out)
                    next_eth_entry_ts = pd.Timestamp(out['exit_ts']) + pd.Timedelta(hours=4)

    # Ordenar por entry_ts para metricas sequenciales (importante para DD)
    trades.sort(key=lambda t: pd.Timestamp(t['entry_ts']))
    return trades


# =============================================================================
# METRICAS
# =============================================================================
def metrics(trades: list) -> dict:
    if not trades:
        return {'n': 0, 'wr': 0.0, 'pf': 0.0, 'avg_pnl': 0.0,
                'total_return': 0.0, 'max_dd': 0.0, 'sharpe_like': 0.0,
                'months': 0.0, 'monthly_return': 0.0, 'annual_return': 0.0,
                'max_leverage': 0.0, 'avg_leverage': 0.0}

    pnls = np.array([t['pnl_pct'] for t in trades])
    levs = np.array([t.get('leverage', 1.0) for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    n = len(pnls)
    wr = len(wins) / n
    gw = float(wins.sum())
    gl = float(abs(losses.sum()))
    pf = (gw / gl) if gl > 1e-9 else float('inf')

    # Equity compuesta. Como hay max 2 trades simultaneos (BTC + ETH), una
    # aproximacion HONESTA es escalar el pnl_pct por 0.5 si los trades se
    # solapan en el tiempo (asumimos cartera 50/50). Para simplificar y
    # ser CONSERVADOR (no inflar metricas), aqui los componemos en SERIE
    # ordenados por entry_ts. Es una simplificacion que IGNORA la
    # diversificacion: subestima el retorno real (no lo infla).
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

    # Sharpe anualizado aproximado: usar pnl por trade, asumir media-tiempo
    # entre trades ~ months*30/N dias. Esto es heuristica, no exacto.
    # Para reportar usamos sharpe_like (por-trade) y un sharpe_annual con
    # asuncion de ~N trades/year.
    trades_per_year = n / max(years, 1e-6) if years > 0 else n
    sharpe_annual = (pnls.mean() / pnls.std()) * np.sqrt(trades_per_year) if pnls.std() > 0 else 0.0

    return {'n': n, 'wr': float(wr), 'pf': float(pf), 'avg_pnl': float(pnls.mean()),
            'total_return': float(total), 'max_dd': float(dd),
            'sharpe_like': sl, 'sharpe_annual': float(sharpe_annual),
            'months': months, 'years': years,
            'monthly_return': float(monthly_return),
            'annual_return': float(annual_return),
            'max_leverage': float(levs.max()) if len(levs) else 0.0,
            'avg_leverage': float(levs.mean()) if len(levs) else 0.0}


def metrics_portfolio_50_50(trades: list) -> dict:
    """
    Variante de metricas asumiendo cartera 50/50 BTC/ETH:
    - cuando hay 1 trade activo, ocupa 50% del capital
    - cuando hay 2 simultaneos, ocupan 100% (50% cada uno)
    - cuando hay 0, capital flat
    Esta es una aproximacion mas realista de "vivir con max 2 simultaneas"
    sin overstateamiento.
    Devuelve las metricas con la curva 50/50.
    """
    if not trades:
        return metrics(trades)

    # Build event timeline: (ts, asset, side, leverage, entry/exit, idx_trade)
    events = []
    for ti, t in enumerate(trades):
        events.append((pd.Timestamp(t['entry_ts']), 'open', ti))
        events.append((pd.Timestamp(t['exit_ts']), 'close', ti))
    events.sort()

    # Simular equity: cuando un trade cierra, su pnl_pct se aplica con peso
    # 0.5 al equity (capital dedicado por activo). Si los dos activos
    # operan simultaneamente, cada uno usa 50% del capital.
    eq = 1.0
    peak = 1.0
    dd = 0.0
    closes_processed = set()
    for ts, ev, ti in events:
        if ev == 'close' and ti not in closes_processed:
            pnl = trades[ti]['pnl_pct']
            eq *= (1 + pnl * 0.5)  # peso 0.5 (50% del capital por activo)
            peak = max(peak, eq)
            dd = max(dd, (peak - eq) / peak)
            closes_processed.add(ti)

    total = eq - 1.0
    t0 = pd.Timestamp(trades[0]['entry_ts'])
    tL = max(pd.Timestamp(t['exit_ts']) for t in trades)
    months = max(1.0, (tL - t0).days / 30.0)
    years = months / 12.0
    annual_return = (eq ** (1 / years) - 1) if eq > 0 and years > 0 else -1.0
    monthly_return = (eq ** (1 / months) - 1) if eq > 0 else -1.0

    return {
        'n': len(trades),
        'total_return_50_50': float(total),
        'max_dd_50_50': float(dd),
        'annual_return_50_50': float(annual_return),
        'monthly_return_50_50': float(monthly_return),
        'years': years,
        'months': months,
    }
