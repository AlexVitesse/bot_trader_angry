"""
Agent D - BTC/USDT 1D TREND-FOLLOWING + VOL-TARGETED POSITION SIZING
=====================================================================

Filosofia (3 ideas centrales)
----------------------------
1. **Timeframe DIARIO**: menos ruido, trends limpios. La leccion de Ronda 1:
   en 4h el edge marginal es PF ~1.4. En 1D, el mismo trailing amplio captura
   movimientos de semanas, no horas. La relacion senal/ruido sube.

2. **El edge esta en la SALIDA (Chandelier amplio)**: heredamos la leccion
   del unico exito real (V7: trailing stop, no prediccion). Chandelier exit
   = peak - 3*ATR(20). Deja correr ganadores 4-12 semanas. La entrada solo
   tiene que ser razonable; el trailing hace el trabajo.

3. **VOL-TARGETING para apalancamiento defendible**:
   - tamaño = target_vol_diario / vol_realizada_20d
   - target_vol = 1.5% diario por posicion (~24% anualizado por trade)
   - cap maximo 2.5x leverage (no 3x, mas conservador)
   - Esto es el corazon de TODOS los CTAs profesionales (Winton, AHL, etc.):
     sizeas INVERSO a volatilidad. Mercados calmos -> apalancado; mercados
     volatiles -> tamaño reducido. Da retornos mas suaves Y mayores que
     leverage fijo.
   - Funding cost (8h, anualizado) RESTADO del PnL cuando hay leverage.

Decisiones a priori (sin grid search post-hoc)
----------------------------------------------
- TF 1D
- Donchian-20 daily (Turtle Sistema 1, clasico desde 1983)
- Filtro EMA 20>50 daily con shift(1) (golden cross corto plazo, no 50/200)
- Chandelier ATR(20)*3 (Charles LeBeau, 1990s)
- vol_target = 1.5%/dia, lookback 20d, leverage cap 2.5x
- LONG-only

Bugs evitados (auditoria aplicada)
----------------------------------
- Una posicion a la vez: tras cerrar trade, salta hasta despues de la vela
- Sin look-ahead intrabar: salida con SL HEREDADO; recien luego actualiza peak
- MTF (1d->1d aqui es trivial; no hace falta shift de TF)
- Donchian: rolling(N).max().shift(1) explicito
- Funding: shift(1) antes del reindex
- vol_realizada: std(returns_log).rolling(20).std().shift(1)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# =============================================================================
# PARAMETROS FROZEN (decisiones a priori - ver SELF-AUDIT en README.md)
# =============================================================================
PARAMS = {
    # --- Costes ---
    'commission': 0.0005,           # 0.05% por lado (5 bps)

    # --- Filtro de regimen daily ---
    'ema_fast': 20,                 # EMA corta diaria
    'ema_slow': 50,                 # EMA larga diaria (cruce 20/50)
    'use_ema200_floor': True,       # ademas exigir close > EMA200 (regimen bull macro)
    'ema_floor_n': 200,

    # --- Entrada Donchian breakout daily ---
    'donchian_n': 20,               # 20 dias (Turtle Sistema 1)
    'vol_ma_n': 20,                 # base de comparacion de volumen
    'vol_ratio_min': 1.1,           # 10% sobre media (suave - estamos en 1D)

    # --- Chandelier exit (trailing ATR amplio) ---
    'atr_n': 20,
    'chandelier_mult': 3.0,         # peak - 3*ATR(20)
    'max_bars': 90,                 # 90 dias maximo en trade (3 meses)

    # --- Vol-targeting ---
    # target_annual_vol = target_daily_vol * sqrt(365). BTC realized vol ~52%/yr
    # mediana. Apuntar a ~40% anual = 2.1%/dia. Esto da leverage ~0.77x media,
    # cap 2.5x. Decision a priori: 40% anual es el target CTA "agresivo high vol"
    # estandar (AHL, Winton). Para crypto da leverage moderado.
    'vol_lookback': 20,             # std de log returns 20d
    'target_daily_vol': 0.021,      # 2.1% diario -> ~40% anualizado (target trade)
    'leverage_max': 2.5,            # cap conservador
    'leverage_min': 0.5,            # piso en mercados volatiles
    # Modo alternativo (para benchmarking / spot account):
    'fixed_leverage': None,         # si no None, override del vol-targeting

    # --- Funding ---
    'funding_enabled': True,
    'funding_z_n': 28,              # 28 dias de z-score
    'funding_z_max': 2.5,           # bloquear LONG si funding muy alto (euforia)
    # Para LONG: el funding positivo es un coste; lo restamos del PnL diario

    # --- Operativos ---
    'cutoff_date': '2025-12-31',
    'min_bars_warmup': 250,         # 250 dias warmup (EMA200 necesita 200+)
    'timeframe': '1D',              # '1D' o '12h'
}


# --- PARAMS variant para 12h (escalado: 2 velas = 1 dia) ---
PARAMS_12H = {**PARAMS,
    'ema_fast': 40,         # 20 dias * 2 velas/dia
    'ema_slow': 100,        # 50 dias * 2
    'ema_floor_n': 400,     # 200 dias * 2
    'donchian_n': 40,       # 20 dias * 2
    'vol_ma_n': 40,
    'atr_n': 40,
    'vol_lookback': 40,
    'max_bars': 180,        # 90 dias * 2
    'funding_z_n': 56,      # 28 dias * 2
    'min_bars_warmup': 500,
    # vol-targeting: target_daily_vol se interpreta SIEMPRE como diario;
    # rv20 medido en log returns POR VELA y luego anualizado/escalado.
    'timeframe': '12h',
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


def _atr(h, l, c, n=20):
    return _rma(_true_range(h, l, c), n)


# =============================================================================
# PREPARE DATA
# =============================================================================
def prepare_data(df_1d: pd.DataFrame,
                 df_funding: pd.DataFrame | None = None,
                 params: dict = PARAMS) -> pd.DataFrame:
    """
    Toma OHLCV diario y adjunta todas las features.
    Aplica cutoff <= params['cutoff_date'] de inmediato.

    Sin look-ahead:
    - donchian_high = rolling(N).max().shift(1)
    - EMAs no necesitan shift (causales)
    - vol realizada: rolling std de log returns, .shift(1)
    - funding z: shift(1) explicito
    """
    cutoff = pd.Timestamp(params['cutoff_date'], tz='UTC')
    df = df_1d.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df[df.index <= cutoff].sort_index()
    # Normalizar a daily (por si vienen >1 fila por dia)
    df = df[~df.index.duplicated(keep='last')]

    h, l, c, v = df['high'], df['low'], df['close'], df['volume']

    # EMAs (causales, no requieren shift)
    df['ema_fast'] = _ema(c, params['ema_fast'])
    df['ema_slow'] = _ema(c, params['ema_slow'])
    df['ema_floor'] = _ema(c, params['ema_floor_n']) if params['use_ema200_floor'] else c
    df['bull_regime'] = (df['ema_fast'] > df['ema_slow']).astype(int)
    if params['use_ema200_floor']:
        df['bull_regime'] = ((df['ema_fast'] > df['ema_slow']) & (c > df['ema_floor'])).astype(int)

    # ATR + ATR%
    df['atr'] = _atr(h, l, c, params['atr_n'])
    df['atr_pct'] = df['atr'] / c

    # Donchian high (excluye vela actual)
    df['donchian_high'] = h.rolling(params['donchian_n']).max().shift(1)

    # Volume ratio
    df['vol_ma'] = v.rolling(params['vol_ma_n']).mean()
    df['vol_ratio'] = v / df['vol_ma'].replace(0, np.nan)

    # Realized vol (std de log returns N dias) - con shift(1)
    log_ret = np.log(c / c.shift(1))
    df['rv20'] = log_ret.rolling(params['vol_lookback']).std().shift(1)

    # Funding z-score (con shift(1))
    if df_funding is not None and params.get('funding_enabled', True):
        fund = df_funding.copy()
        if fund.index.tz is None:
            fund.index = fund.index.tz_localize('UTC')
        fund = fund[fund.index <= cutoff].sort_index()
        col = 'funding_rate' if 'funding_rate' in fund.columns else fund.columns[0]
        # Funding viene cada 8h -> resamplear a diario sumando (3 fundings/dia)
        fund_daily = fund[col].resample('1D').sum()
        n_z = params['funding_z_n']
        fmean = fund_daily.rolling(n_z).mean()
        fstd = fund_daily.rolling(n_z).std()
        z = (fund_daily - fmean) / fstd.replace(0, np.nan)
        z = z.shift(1)
        # Reindex a daily de df
        df['funding_z'] = z.reindex(df.index, method='ffill')
        # Tambien guardamos la tasa diaria (anualizada equivalente: ~3 fundings/dia)
        # para descontar costo de financiacion al trade leveraged
        df['funding_daily'] = fund_daily.reindex(df.index, method='ffill').shift(1)
    else:
        df['funding_z'] = 0.0
        df['funding_daily'] = 0.0

    # Limpieza warmup
    df = df.dropna(subset=['atr', 'donchian_high', 'vol_ratio', 'ema_slow', 'rv20'])
    return df


# =============================================================================
# SIGNAL
# =============================================================================
def signal(df: pd.DataFrame, idx: int, params: dict = PARAMS) -> str | None:
    """
    Devuelve 'LONG' si en la vela idx (ya cerrada) se cumplen:
      1) Regimen bull: EMA20 > EMA50 (y close > EMA200 si activado)
      2) Donchian breakout: close > max(high de las N velas anteriores)
      3) Volumen >= 1.1x media
      4) Funding z <= 2.5 (no euforia)
    """
    if idx < params['min_bars_warmup']:
        return None
    if idx >= len(df) - 2:
        return None

    row = df.iloc[idx]
    if not (row.get('bull_regime', 0) >= 1):
        return None
    dh = row.get('donchian_high', np.nan)
    if pd.isna(dh) or row['close'] <= dh:
        return None
    if pd.isna(row.get('vol_ratio', np.nan)) or row['vol_ratio'] < params['vol_ratio_min']:
        return None
    if params.get('funding_enabled', True):
        fz = row.get('funding_z', 0.0)
        if pd.notna(fz) and fz > params['funding_z_max']:
            return None
    return 'LONG'


# =============================================================================
# SIZE POSITION (vol-targeting)
# =============================================================================
def size_position(df: pd.DataFrame, idx: int, params: dict = PARAMS) -> float:
    """
    Devuelve fraccion de capital a poner.
    - Si fixed_leverage es no-None: usa ese valor (override).
    - Si no: vol-targeting con cap.

    Para 12h: rv20 son log returns POR VELA (12h), no por dia. Reescalamos
    target_daily_vol a target_per_bar_vol = target_daily / sqrt(bars/dia).
    """
    if params.get('fixed_leverage') is not None:
        return float(params['fixed_leverage'])
    rv = float(df['rv20'].iloc[idx])
    if not np.isfinite(rv) or rv <= 0:
        return params['leverage_min']
    # Escalado por TF
    tf = params.get('timeframe', '1D')
    bars_per_day = 1.0 if tf == '1D' else (2.0 if tf == '12h' else 1.0)
    target_per_bar = params['target_daily_vol'] / np.sqrt(bars_per_day)
    raw = target_per_bar / rv
    return float(np.clip(raw, params['leverage_min'], params['leverage_max']))


# =============================================================================
# SIMULATE - Chandelier trailing, sin look-ahead intrabar, con funding cost
# =============================================================================
def simulate(df: pd.DataFrame, entry_bar: int, params: dict = PARAMS,
             leverage: float = 1.0) -> dict:
    """
    Simula trade LONG abierto en close(entry_bar) con Chandelier exit.

    Chandelier: stop = peak - chandelier_mult * ATR(entry).
    El ATR se FIJA al entrar (no se actualiza intrabar) y se sigue
    el peak con sin-look-ahead intrabar.

    Funding cost diario (anti-overstatement de leverage):
      Si leverage > 1: cada dia el trade paga funding_daily * leverage
      (porque toda la posicion paga funding, incluso la parte prestada).
      Para LONG, funding positivo = coste; funding negativo = ingreso.

    Devuelve dict con:
      outcome, gross_pnl_pct, funding_cost_pct, net_pnl_pct, leveraged_pnl_pct,
      bars, exit_price, leverage_used.
    """
    n = len(df)
    entry_price = float(df['close'].iloc[entry_bar])
    entry_ts = df.index[entry_bar]
    atr_entry = float(df['atr'].iloc[entry_bar])
    if not np.isfinite(atr_entry) or atr_entry <= 0:
        return {'outcome': 'SKIP', 'gross_pnl_pct': 0.0, 'funding_cost_pct': 0.0,
                'net_pnl_pct': 0.0, 'leveraged_pnl_pct': 0.0,
                'bars': 0, 'exit_price': entry_price,
                'entry_ts': entry_ts, 'exit_ts': entry_ts,
                'leverage_used': leverage, 'reason': 'no_atr'}

    chand_dist = params['chandelier_mult'] * atr_entry
    # Stop inicial = entry - chand_dist (no peak yet)
    sl_price = entry_price - chand_dist
    peak = entry_price
    max_bars = params['max_bars']
    commission = params['commission']

    funding_cost_accum = 0.0  # suma diaria de funding cost (en % nominal)
    exit_p = entry_price
    exit_ts = entry_ts
    bars = 0
    outcome = 'TIMEOUT'

    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= n:
            exit_p = float(df['close'].iloc[-1])
            exit_ts = df.index[-1]
            bars = i - 1
            outcome = 'EOD'
            break

        # Funding cost del bar (acumulado sobre toda la posicion incl. leverage)
        # ATENCION: funding_daily es per-DAY. Si TF=12h, cada bar es medio dia.
        if params.get('funding_enabled', True):
            f_daily = df['funding_daily'].iloc[b]
            if pd.notna(f_daily):
                tf = params.get('timeframe', '1D')
                bars_per_day = 1.0 if tf == '1D' else (2.0 if tf == '12h' else 1.0)
                # LONG paga funding positivo. Multiplicado por leverage.
                # Dividir por bars/dia para no double-contar en 12h.
                funding_cost_accum += float(f_daily) * leverage / bars_per_day

        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])

        # 1) Salida contra SL HEREDADO (sin look-ahead intrabar)
        if lo <= sl_price:
            exit_p = sl_price
            exit_ts = df.index[b]
            bars = i
            outcome = 'TRAIL'
            break

        # 2) Actualizar peak/SL para la SIGUIENTE vela
        if hi > peak:
            peak = hi
        # Chandelier: peak - chandelier_mult * ATR(entry). Trail solo sube.
        new_sl = peak - chand_dist
        sl_price = max(sl_price, new_sl)
    else:
        # Timeout normal
        b = entry_bar + max_bars
        exit_p = float(df['close'].iloc[b])
        exit_ts = df.index[b]
        bars = max_bars
        outcome = 'TIMEOUT'

    # PnL bruto sobre el ACTIVO (sin leverage)
    gross_ret = (exit_p - entry_price) / entry_price
    # PnL neto antes de leverage: gross - 2*commission
    net_ret = gross_ret - 2 * commission
    # PnL apalancado: leverage * net_ret - funding_cost
    leveraged_ret = leverage * net_ret - funding_cost_accum

    return {
        'outcome': outcome,
        'gross_pnl_pct': gross_ret,
        'net_pnl_pct': net_ret,
        'funding_cost_pct': funding_cost_accum,
        'leveraged_pnl_pct': leveraged_ret,
        'bars': bars,
        'exit_price': exit_p,
        'entry_ts': entry_ts,
        'exit_ts': exit_ts,
        'leverage_used': leverage,
    }


# =============================================================================
# RUN BACKTEST - una posicion a la vez, no solapado
# =============================================================================
def run_backtest(df: pd.DataFrame, params: dict = PARAMS,
                 start_i: int | None = None, end_i: int | None = None,
                 use_leverage: bool = True) -> list[dict]:
    """
    Recorre velas; al abrir trade salta hasta DESPUES de la vela de cierre.
    use_leverage=True -> aplica vol-targeting; False -> leverage=1.0 (unleveraged).
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
        lev = size_position(df, i, params) if use_leverage else 1.0
        out = simulate(df, i, params, leverage=lev)
        if out['outcome'] == 'SKIP':
            i += 1
            continue
        trades.append({
            'entry_ts': out['entry_ts'],
            'exit_ts': out['exit_ts'],
            'outcome': out['outcome'],
            'gross_pnl_pct': out['gross_pnl_pct'],
            'net_pnl_pct': out['net_pnl_pct'],
            'funding_cost_pct': out['funding_cost_pct'],
            'leveraged_pnl_pct': out['leveraged_pnl_pct'],
            'leverage_used': out['leverage_used'],
            'bars': out['bars'],
            'side': sig,
        })
        # CRITICO: avanzar PAST la vela de cierre
        i += max(1, out['bars']) + 1
    return trades


# =============================================================================
# METRICAS
# =============================================================================
def metrics(trades: list[dict], pnl_key: str = 'leveraged_pnl_pct') -> dict:
    """
    Calcula metricas sobre la lista de trades.
    pnl_key: 'net_pnl_pct' (unleveraged) o 'leveraged_pnl_pct' (apalancado).
    """
    if not trades:
        return {'n': 0, 'wr': 0.0, 'pf': 0.0, 'avg_pnl': 0.0,
                'total_return': 0.0, 'max_dd': 0.0, 'sharpe_like': 0.0,
                'months': 0.0, 'monthly_return': 0.0, 'annual_return': 0.0}
    pnls = np.array([t[pnl_key] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    n = len(pnls)
    wr = len(wins) / n
    gw = float(wins.sum())
    gl = float(abs(losses.sum()))
    pf = (gw / gl) if gl > 1e-9 else (float('inf') if gw > 0 else 0.0)

    # Equity sequencial (no solapado garantizado por run_backtest)
    eq = 1.0
    peak = 1.0
    dd = 0.0
    for p in pnls:
        eq *= (1 + p)
        peak = max(peak, eq)
        dd = max(dd, (peak - eq) / max(peak, 1e-9))
    total = eq - 1.0

    t0 = pd.to_datetime(trades[0]['entry_ts'])
    tL = pd.to_datetime(trades[-1]['exit_ts'])
    days = max(1.0, (tL - t0).days)
    months = days / 30.0
    years = days / 365.25
    monthly_return = (eq ** (1 / max(months, 0.5)) - 1) if eq > 0 else -1.0
    annual_return = (eq ** (1 / max(years, 0.05)) - 1) if eq > 0 else -1.0

    sl = float(pnls.mean() / pnls.std()) if pnls.std() > 0 else 0.0
    return {'n': n, 'wr': float(wr), 'pf': float(pf), 'avg_pnl': float(pnls.mean()),
            'total_return': float(total), 'max_dd': float(dd),
            'sharpe_like': sl, 'months': months, 'years': years,
            'monthly_return': float(monthly_return),
            'annual_return': float(annual_return)}


# =============================================================================
# SHARPE ANUALIZADO (correcto, por dia)
# =============================================================================
def daily_equity_curve(trades: list[dict], df: pd.DataFrame,
                        pnl_key: str = 'leveraged_pnl_pct') -> pd.Series:
    """
    Construye curva de equity DIARIA marcando-a-mercado durante el trade.
    Esto da un Sharpe anualizado realista.
    """
    if not trades:
        return pd.Series(dtype=float)
    # Cada trade contribuye su pnl uniformemente repartido a lo largo de sus dias.
    # Es una aproximacion (no usamos los precios intermedios) pero suficiente.
    all_days = pd.date_range(df.index.min().normalize(), df.index.max().normalize(), freq='D', tz='UTC')
    daily_ret = pd.Series(0.0, index=all_days)
    for t in trades:
        t0 = pd.to_datetime(t['entry_ts']).normalize()
        t1 = pd.to_datetime(t['exit_ts']).normalize()
        days = max(1, (t1 - t0).days)
        # Convertir pnl total a daily ret compuesto: (1+r)^(1/days)-1
        p = t[pnl_key]
        if p <= -1:
            p = -0.99
        dr = (1 + p) ** (1 / days) - 1
        # Asignar dr a cada dia del rango [t0, t1)
        rng = pd.date_range(t0, t1 - pd.Timedelta(days=1), freq='D', tz='UTC')
        rng = rng[rng.isin(daily_ret.index)]
        if len(rng) > 0:
            daily_ret.loc[rng] = daily_ret.loc[rng] + dr
    return daily_ret


def annualized_sharpe(daily_ret: pd.Series) -> float:
    """Sharpe anualizado asumiendo rf=0. Usa TODOS los dias (incl. los flat)
    - eso es lo correcto: la vol fuera de mercado es 0 y debe contar."""
    if len(daily_ret) < 5 or daily_ret.std() == 0:
        return 0.0
    return float(daily_ret.mean() / daily_ret.std() * np.sqrt(365))
