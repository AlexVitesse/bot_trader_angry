"""
Agent H — ETH/USDT 4h ETH/BTC RATIO ROTATION (long-only).
=========================================================

Filosofia / hipotesis
---------------------
En crypto, el ratio ETH/BTC es un indicador de *rotacion de capital*:
- Cuando el ratio sube -> capital prefiere ETH (LONG ETH).
- Cuando el ratio cae -> capital sale de ETH a BTC (no LONG, evitar el lado
  caro de la rotacion).

La literatura senala que ETH/BTC suele anticipar rallies de altcoins; los
breakouts del ratio son senales adelantadas que tanto trend-following clasico
como modelos ML "puros" sobre el precio de ETH no capturan facilmente — porque
el ratio integra dos historias (la de ETH y la de BTC) en una sola serie.

Mecanismo principal
-------------------
1. Calcula ratio = ETH_close_daily / BTC_close_daily.
2. Features del ratio (con shift(1) -> sin look-ahead):
   - EMA20, EMA50 del ratio
   - slope rolling N del ratio
   - z-score 90d del ratio
   - breakout high 20d del ratio
3. Senal LONG ETH si:
   - ratio_close > ratio_ema50 (uptrend del ratio)
   - ratio_slope > 0 (acelerando)
   - BTC daily NO en bear fuerte (EMA50_1d_BTC > EMA200_1d_BTC, shift(1))
   - Funding BTC no extremo (proxy del estado de leverage del sistema)
4. Senal LONG secundaria (mean-reversion del ratio):
   - z_score < -1.5 (ratio extremadamente barato vs BTC)
   - BTC daily en bull general
5. Exit: trailing stop ATR-based AMPLIO (sin look-ahead intrabar),
   mismo patron que Agent A.
6. LONG-only. SHORT no aprobado en el proyecto.

Anti-bugs / protecciones (mismas que Agent A)
---------------------------------------------
- Una posicion a la vez (motor avanza tras el cierre del trade).
- Sin look-ahead intrabar: SL trailing comprueba salida con el peak HEREDADO
  antes de actualizar peak/SL para la siguiente vela.
- MTF (1d -> 4h): shift(1) en todos los features daily (ratio y EMA BTC).
- High20 del ratio: rolling(20).max().shift(1).
- Funding z-score: rolling-z-score con shift(1).
- Cutoff inviolable 2025-12-31 aplicado en prepare_data().

API publica (igual que los otros agentes)
-----------------------------------------
- PARAMS: dict frozen.
- prepare_data(df_eth_4h, df_btc_4h, df_ratio_daily, df_funding=None, params=PARAMS)
- signal(df, idx, params) -> 'LONG' | None
- simulate(df, entry_bar, params) -> dict(outcome, pnl_pct, bars, ...)
- run_backtest(df_features, params, start_i, end_i) -> list[trade dict]
- metrics(trades) -> dict
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# =============================================================================
# PARAMETROS FROZEN (decididos a priori, sin re-tunear)
# =============================================================================
PARAMS = {
    # --- Costes ---
    'commission': 0.0005,           # 0.05% por lado (consistente con resto del proyecto)

    # --- Ratio ETH/BTC: features daily ---
    'ratio_ema_fast': 20,           # ratio EMA fast (estandar)
    'ratio_ema_slow': 50,           # ratio EMA slow
    'ratio_slope_n': 10,            # ventana para el slope (10d = 2 semanas)
    'ratio_z_n': 90,                # z-score rolling 90d (~trimestre)
    'ratio_high_n': 20,             # breakout: high de 20 dias (~mes)

    # --- Filtro BTC daily ---
    'btc_ema_fast_1d': 50,
    'btc_ema_slow_1d': 200,

    # --- Senal primaria: ratio uptrend + acelerando ---
    'min_slope': 0.0,               # slope debe ser > 0 (subiendo)
    # Senal secundaria: mean-rev del ratio (oversold vs BTC en bull general)
    'mean_rev_z_max': -1.5,         # z < -1.5 = ratio muy bajo vs media
    'mean_rev_enabled': True,       # se evalua en backtest, se puede aislar

    # --- Filtro de volumen ETH (entry confirmation) ---
    'vol_ma_n': 20,
    'vol_ratio_min': 1.1,           # volumen ETH > 1.1x promedio (suave)

    # --- Trailing ATR AMPLIO (anti-bug) ---
    'atr_n': 14,
    'trail_atr_mult': 2.5,          # 2.5x ATR
    'trail_floor_pct': 0.025,       # piso 2.5%
    'trail_ceiling_pct': 0.07,      # techo 7% (ETH mas volatil que BTC)
    'max_bars': 60,                 # 60 velas 4h = 10 dias maximo

    # --- Filtro de funding (anti-euforia) ---
    'funding_z_n': 168,             # 4 semanas
    'funding_z_max': 2.5,           # bloquear LONG si z > 2.5
    'funding_enabled': True,        # si no hay funding ETH, se usa BTC como proxy

    # --- Operativos ---
    'cutoff_date': '2025-12-31',
    'min_bars_warmup': 250,
}


# =============================================================================
# HELPERS
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


def _rolling_slope(s: pd.Series, n: int) -> pd.Series:
    """Slope = (s - s.shift(n)) / n -> mean change per period. Positive = upward."""
    return (s - s.shift(n)) / float(n)


# =============================================================================
# prepare_data
# =============================================================================
def prepare_data(df_eth_4h: pd.DataFrame,
                 df_btc_4h: pd.DataFrame | None,
                 df_ratio_daily: pd.DataFrame | None = None,
                 df_funding: pd.DataFrame | None = None,
                 params: dict = PARAMS) -> pd.DataFrame:
    """
    Construye el DF de features para ETH 4h con todas las senales del ratio
    ETH/BTC daily (con shift(1)), el filtro daily de BTC, el ATR/volumen ETH
    y el funding z-score (proxy BTC).

    Si df_ratio_daily es None, deriva el ratio de df_eth_4h y df_btc_4h
    (resample 1D, last).
    """
    cutoff = pd.Timestamp(params['cutoff_date'], tz='UTC')
    df = df_eth_4h.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df[df.index <= cutoff].sort_index()

    h, l, c, v = df['high'], df['low'], df['close'], df['volume']

    # ATR / vol ETH
    df['atr'] = _atr(h, l, c, params['atr_n'])
    df['atr_pct'] = df['atr'] / c
    df['vol_ma'] = v.rolling(params['vol_ma_n']).mean()
    df['vol_ratio'] = v / df['vol_ma'].replace(0, np.nan)

    # ---- Ratio ETH/BTC daily ----
    if df_ratio_daily is not None:
        ratio = df_ratio_daily['close'].copy()
    else:
        # derivar del 4h
        eth = df_eth_4h.copy()
        btc = df_btc_4h.copy()
        if eth.index.tz is None:
            eth.index = eth.index.tz_localize('UTC')
        if btc.index.tz is None:
            btc.index = btc.index.tz_localize('UTC')
        eth_d = eth['close'].resample('1D').last().dropna()
        btc_d = btc['close'].resample('1D').last().dropna()
        ratio = (eth_d / btc_d).dropna()

    # Normalizar tz: cualquier indice naive a UTC
    if ratio.index.tz is None:
        ratio.index = ratio.index.tz_localize('UTC')
    ratio = ratio[ratio.index <= cutoff].sort_index()

    # Features del ratio
    ratio_ema_fast = _ema(ratio, params['ratio_ema_fast'])
    ratio_ema_slow = _ema(ratio, params['ratio_ema_slow'])
    ratio_slope = _rolling_slope(ratio, params['ratio_slope_n'])
    ratio_mean = ratio.rolling(params['ratio_z_n']).mean()
    ratio_std = ratio.rolling(params['ratio_z_n']).std()
    ratio_z = (ratio - ratio_mean) / ratio_std.replace(0, np.nan)
    ratio_high20 = ratio.rolling(params['ratio_high_n']).max()
    # uptrend: ratio > ema_slow AND ema_fast > ema_slow
    is_uptrend = ((ratio > ratio_ema_slow) & (ratio_ema_fast > ratio_ema_slow)).astype(int)
    is_accel = (ratio_slope > params['min_slope']).astype(int)
    # breakout: ratio_close > ratio_high20.shift(1)
    is_breakout = (ratio > ratio_high20.shift(1)).astype(int)
    # mean-rev oversold
    is_oversold = (ratio_z < params['mean_rev_z_max']).astype(int)

    # CRITICAL: shift(1) -> el valor del dia D solo se conoce en el dia D+1
    feat_daily = pd.DataFrame({
        'ratio': ratio.shift(1),
        'ratio_ema_fast': ratio_ema_fast.shift(1),
        'ratio_ema_slow': ratio_ema_slow.shift(1),
        'ratio_slope': ratio_slope.shift(1),
        'ratio_z': ratio_z.shift(1),
        'ratio_high20': ratio_high20.shift(1),
        'ratio_uptrend': is_uptrend.shift(1),
        'ratio_accel': is_accel.shift(1),
        'ratio_breakout': is_breakout.shift(1),
        'ratio_oversold': is_oversold.shift(1),
    })

    # Reindex al 4h con ffill (cada vela 4h del dia D usa el regimen del dia D-1)
    for col in feat_daily.columns:
        df[col] = feat_daily[col].reindex(df.index, method='ffill')

    # ---- Filtro BTC daily ----
    if df_btc_4h is not None:
        btc = df_btc_4h.copy()
        if btc.index.tz is None:
            btc.index = btc.index.tz_localize('UTC')
        btc = btc[btc.index <= cutoff].sort_index()
        btc_d = btc['close'].resample('1D').last().dropna()
        btc_ema_fast = _ema(btc_d, params['btc_ema_fast_1d'])
        btc_ema_slow = _ema(btc_d, params['btc_ema_slow_1d'])
        btc_bull = (btc_ema_fast > btc_ema_slow).astype(int).shift(1)
        df['btc_bull_1d'] = btc_bull.reindex(df.index, method='ffill')
    else:
        df['btc_bull_1d'] = 1  # sin filtro = neutral

    # ---- Funding z-score (proxy BTC, suficiente para anti-euforia) ----
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
        z = (fund_s - fmean) / fstd.replace(0, np.nan)
        z = z.shift(1)  # CRITICAL: shift
        df['funding_z'] = z.reindex(df.index, method='ffill')
    else:
        df['funding_z'] = 0.0

    df = df.dropna(subset=['atr', 'vol_ratio', 'ratio', 'ratio_ema_slow',
                           'ratio_slope', 'ratio_z', 'btc_bull_1d'])
    return df


# =============================================================================
# SIGNAL
# =============================================================================
def signal(df: pd.DataFrame, idx: int, params: dict = PARAMS) -> str | None:
    """
    LONG ETH si pasa la senal primaria (ratio uptrend + acelerando + BTC bull)
    o la secundaria (ratio oversold + BTC bull). Sin look-ahead.
    """
    if idx < params['min_bars_warmup']:
        return None
    if idx >= len(df) - 2:
        return None

    row = df.iloc[idx]

    # 0) BTC daily debe estar bull (filtro macro) - aplica a ambas senales
    if not (row.get('btc_bull_1d', 0) >= 1):
        return None

    # 1) Funding veto (anti-euforia)
    if params.get('funding_enabled', True):
        fz = row.get('funding_z', 0.0)
        if pd.notna(fz) and fz > params['funding_z_max']:
            return None

    # 2) Confirmacion de volumen ETH (suave)
    if pd.isna(row.get('vol_ratio', np.nan)) or row['vol_ratio'] < params['vol_ratio_min']:
        return None

    # 3) Senal primaria: ratio uptrend + acelerando
    primary = (row.get('ratio_uptrend', 0) >= 1) and (row.get('ratio_accel', 0) >= 1)

    # 4) Senal secundaria: ratio muy oversold (mean-rev)
    secondary = (params.get('mean_rev_enabled', True)
                 and row.get('ratio_oversold', 0) >= 1)

    if primary or secondary:
        return 'LONG'
    return None


# =============================================================================
# SIMULATE — trailing ATR, sin look-ahead intrabar
# =============================================================================
def simulate(df: pd.DataFrame, entry_bar: int, params: dict = PARAMS) -> dict:
    n = len(df)
    entry_price = float(df['close'].iloc[entry_bar])
    entry_ts = df.index[entry_bar]
    atr_pct = float(df['atr_pct'].iloc[entry_bar])
    if not np.isfinite(atr_pct) or atr_pct <= 0:
        return {'outcome': 'SKIP', 'pnl_pct': 0.0, 'bars': 0,
                'exit_price': entry_price, 'entry_ts': entry_ts,
                'exit_ts': entry_ts, 'reason': 'no_atr'}

    trail_dist = atr_pct * params['trail_atr_mult']
    trail_dist = max(params['trail_floor_pct'],
                     min(params['trail_ceiling_pct'], trail_dist))

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

        # 1) Salida contra SL ya conocido (peak/SL de velas anteriores)
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

    exit_p = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (exit_p - entry_price) / entry_price - 2 * commission
    return {'outcome': 'TIMEOUT', 'pnl_pct': pnl, 'bars': max_bars,
            'exit_price': exit_p, 'entry_ts': entry_ts,
            'exit_ts': df.index[entry_bar + max_bars], 'reason': 'max_bars'}


# =============================================================================
# RUN_BACKTEST — una posicion a la vez
# =============================================================================
def run_backtest(df: pd.DataFrame, params: dict = PARAMS,
                 start_i: int | None = None, end_i: int | None = None) -> list[dict]:
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
        # CRITICO: avanzar hasta DESPUES de la vela de cierre
        i += max(1, out['bars']) + 1
    return trades


# =============================================================================
# METRICAS (mismas que Agent A)
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
    monthly_return = (eq ** (1 / months) - 1) if eq > 0 else -1.0
    years = months / 12.0
    annual_return = ((1 + total) ** (1 / years) - 1) if years > 0 and eq > 0 else 0.0

    sl = float(pnls.mean() / pnls.std()) if pnls.std() > 0 else 0.0
    return {'n': n, 'wr': float(wr), 'pf': float(pf), 'avg_pnl': float(pnls.mean()),
            'total_return': float(total), 'max_dd': float(dd),
            'sharpe_like': sl, 'months': months,
            'monthly_return': float(monthly_return),
            'annual_return': float(annual_return)}
