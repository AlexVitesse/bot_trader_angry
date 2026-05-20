"""
Agent E — BTC/USDT 4h FUNDING-EXTREMOS MEAN-REVERSION (bidireccional).
======================================================================

Filosofia
---------
Edge crypto-especifico, NO price-action. Cuando el funding rate de los perp
futures se desvia muchas desviaciones estandar de su norma reciente, hay un
gran desequilibrio direccional entre longs y shorts apalancados que pagan
carry. Ese desequilibrio se RESUELVE en pocos dias — historicamente — con
un movimiento contrario:
  - funding muy positivo (z>>0): longs sobrecargados pagando -> probable
    pullback bajista por desapalancamiento forzado.
  - funding muy negativo (z<<0): shorts sobrecargados pagando -> probable
    short-squeeze alcista.

Es un mecanismo de microestructura documentado en la literatura (Coleman
2022, Cong et al. 2023, papers de Glassnode/CryptoQuant sobre funding rate
como predictor de reversiones a corto plazo). Distinto al edge de trend
(Agente A) y al de regimen (Agente C).

Diseno frozen *a priori*
------------------------
- **funding_z = rolling-zscore sobre 270 bars 8h (~45 dias)** — ventana
  conservadora; demasiado corta hace la z ruidosa, demasiado larga la
  insensibiliza. 45 dias es ~ 1 ciclo de gas-fee / volatilidad.
- **LONG: z < -2.0** + **vela alcista (close>=open)** -> entrar al cierre.
  -2 sigma es ~ percentil 2.3 — claramente extremo. Vela alcista evita
  capturar "el suelo" en plena caida (filtro anti-cuchillo-cayendo).
- **SHORT: z > 2.5** + **vela bajista (close<open)** -> entrar al cierre.
  +2.5 sigma mas estricto porque SHORT en BTC tiene sesgo negativo de fondo
  (proyecto: "SHORT en altcoins no funciona", solo en bear). Vela bajista
  confirma que la exhaustacion compradora ya inicio.
- **TP/SL frozen como R-multiplo 2:1**:
    LONG  TP=5%, SL=2.5% (max 36 bars = 6 dias)
    SHORT TP=4%, SL=2.0% (max 30 bars = 5 dias)
  Ratios 2:1 estandar; ligeramente mas conservadores para SHORT.
- **Carry de funding INCLUIDO en el PnL**: cada 2 velas (8h) se acumula la
  tasa de funding (positiva favorece SHORT, negativa favorece LONG).

Cero overfitting — auditorias aplicadas
---------------------------------------
- **Una posicion a la vez**: tras abrir un trade, idx se mueve hasta
  DESPUES de la vela de cierre (como `revalidate_v15.py` y el bot real).
- **Sin look-ahead intrabar**: TP/SL se evaluan UNA SOLA VEZ contra los
  high/low de la vela posterior; SL pesimista si ambos tocados (mismo
  criterio que v15_framework.py:236).
- **shift(1) en funding**: la z-score conocida en la vela t solo usa
  funding rates de periodos < t (mismo patron que
  v15_features.py:compute_sentiment_features).
- **Cutoff <= 2025-12-31** aplicado en `prepare_data`.

API publica
-----------
- PARAMS: dict frozen con todos los hiperparametros.
- prepare_data(df_4h, df_funding) -> DataFrame con features.
- signal(df, idx, params) -> 'LONG' | 'SHORT' | None
- simulate(df, entry_bar, params, side) -> dict(outcome, pnl_pct, bars, ...)
- run_backtest(df_features, params, start_i, end_i) -> list[trade]
- metrics(trades) -> dict resumen.

Autocontenido: no importa nada de src/ ni del framework V15.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# =============================================================================
# PARAMETROS FROZEN
# =============================================================================
# Cada parametro tiene una justificacion *a priori* (literatura / proyecto /
# sentido comun). Ver SELF-AUDIT en README.md.
PARAMS = {
    # --- Costes ---
    'commission': 0.0005,           # 0.05% por lado (igual que v15_framework)

    # --- Funding z-score ---
    'funding_zwindow': 270,         # 270 velas 4h = 45 dias (ventana estandar
                                    # en literatura funding-mean-reversion)

    # --- LONG: funding extremo NEGATIVO + vela alcista ---
    'long_z_max': -2.0,             # z<-2.0 = ~percentil 2.3; "extremo"
    'long_tp': 0.05,                # 5%
    'long_sl': 0.025,               # 2.5%  -> R-multiple = 2:1
    'long_max_bars': 36,            # 6 dias maximo (funding tiende a normalizar
                                    # rapido)

    # --- SHORT: funding extremo POSITIVO + vela bajista ---
    'short_z_min': 2.5,             # mas estricto que LONG (SHORT en BTC tiene
                                    # sesgo de fondo negativo; req. mas conviction)
    'short_tp': 0.04,
    'short_sl': 0.02,               # 2.0% -> R-multiple = 2:1
    'short_max_bars': 30,           # 5 dias

    # --- Operativos ---
    'cutoff_date': '2025-12-31',    # NUNCA mires datos despues de esto
    'min_bars_warmup': 300,         # warmup minimo para z-score (270 + margen)
}


# =============================================================================
# PREPARE DATA
# =============================================================================
def prepare_data(df_4h: pd.DataFrame,
                 df_funding: pd.DataFrame,
                 params: dict = PARAMS) -> pd.DataFrame:
    """
    Adjunta funding_rate (alineado a 4h, con shift(1)) y funding_zscore al df
    de precio 4h. Aplica cutoff <= params['cutoff_date'] de forma INMEDIATA.

    NO computa indicadores price-action complicados — solo lo necesario:
      - funding_rate (4h, con shift(1) -> sin look-ahead)
      - funding_zscore (ventana rolling, con shift(1))
      - bearish (close<open, dato de la vela actual ya CERRADA)
      - bullish (close>=open)

    Sin look-ahead.
    """
    cutoff = pd.Timestamp(params['cutoff_date'], tz='UTC')
    df = df_4h.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df[df.index <= cutoff].sort_index()

    # --- Funding rate procesamiento ---
    fund = df_funding.copy()
    if fund.index.tz is None:
        fund.index = fund.index.tz_localize('UTC')
    fund = fund[fund.index <= cutoff].sort_index()

    # Resamplear funding (8h) a 4h con ffill
    if 'funding_rate' in fund.columns:
        f_raw = fund['funding_rate'].resample('4h').ffill()
    else:
        f_raw = fund.iloc[:, 0].resample('4h').ffill()

    # CRITICO: shift(1). El funding rate del periodo t solo se conoce al
    # FINAL de t (cuando se paga). Por tanto al inicio de la vela t solo
    # tenemos info de hasta t-1.
    f_shifted = f_raw.shift(1)

    # z-score rolling con shift(1) implicito (la serie ya esta shifted)
    n_z = params['funding_zwindow']
    f_mean = f_shifted.rolling(n_z).mean()
    f_std = f_shifted.rolling(n_z).std()
    f_z = (f_shifted - f_mean) / f_std.replace(0, np.nan)
    f_z = f_z.clip(-5, 5)  # cap a +/-5 sigma (mismo que v15_features)

    # Alinear al indice 4h del df (ffill)
    df['funding_rate'] = f_shifted.reindex(df.index, method='ffill')
    df['funding_z'] = f_z.reindex(df.index, method='ffill')

    # --- Vela bullish / bearish (info de la vela ya CERRADA) ---
    df['bearish'] = (df['close'] < df['open']).astype(int)
    df['bullish'] = (df['close'] >= df['open']).astype(int)

    # Limpieza: nan de warmup
    df = df.dropna(subset=['funding_z'])
    return df


# =============================================================================
# SIGNAL
# =============================================================================
def signal(df: pd.DataFrame, idx: int, params: dict = PARAMS) -> str | None:
    """
    Devuelve 'LONG' | 'SHORT' | None en la vela `idx` (vela ya CERRADA).
    Sin look-ahead: usa solo info en posiciones <= idx.
    """
    if idx < params['min_bars_warmup']:
        return None
    if idx >= len(df) - 2:
        return None

    row = df.iloc[idx]
    fz = row.get('funding_z', np.nan)
    if pd.isna(fz):
        return None

    # LONG: funding extremo negativo + vela alcista (price-action confirm:
    # la caida ya se freno, hay un rebote inicial -> entrar para capturar
    # el squeeze de shorts).
    if fz < params['long_z_max']:
        if row['bullish'] == 1:
            return 'LONG'
        return None

    # SHORT: funding extremo positivo + vela bajista (la subida ya se freno,
    # hay un retroceso inicial -> entrar para capturar el deleverage de longs).
    if fz > params['short_z_min']:
        if row['bearish'] == 1:
            return 'SHORT'
        return None

    return None


# =============================================================================
# SIMULATE — TP/SL fijos + carry de funding incluido
# =============================================================================
def simulate(df: pd.DataFrame, entry_bar: int,
             params: dict = PARAMS, side: str = 'LONG') -> dict:
    """
    Simula un trade abierto en el CIERRE de `entry_bar` con TP/SL fijos.

    Carry de funding INCLUIDO:
      - cada 2 velas (8h) se acumula la tasa de funding del periodo
      - SHORT recibe funding positivo (LONG paga); LONG recibe funding negativo.

    Sin look-ahead intrabar:
      - en cada vela b > entry_bar se comprueba si HIGH/LOW tocan TP o SL
      - si ambos tocan en la misma vela, pesimista: SL gana (mismo criterio
        que v15_framework.py).

    Returns: dict(outcome, pnl_pct, bars, exit_price, entry_ts, exit_ts,
                  funding_pnl, side, entry_price)
    """
    n = len(df)
    entry_price = float(df['close'].iloc[entry_bar])
    entry_ts = df.index[entry_bar]
    commission = params['commission']
    funding_pnl = 0.0  # carry acumulado

    if side == 'LONG':
        tp_pct = params['long_tp']
        sl_pct = params['long_sl']
        max_bars = params['long_max_bars']
    elif side == 'SHORT':
        tp_pct = params['short_tp']
        sl_pct = params['short_sl']
        max_bars = params['short_max_bars']
    else:
        raise ValueError(f"side must be 'LONG' or 'SHORT', got {side!r}")

    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= n:
            # se acaba la data -> cerrar a close del ultimo bar
            exit_p = float(df['close'].iloc[-1])
            if side == 'LONG':
                pnl = (exit_p - entry_price) / entry_price
            else:
                pnl = (entry_price - exit_p) / entry_price
            pnl = pnl - 2 * commission + funding_pnl
            return {'outcome': 'EOD', 'pnl_pct': pnl, 'bars': i,
                    'exit_price': exit_p, 'entry_ts': entry_ts,
                    'exit_ts': df.index[-1], 'funding_pnl': funding_pnl,
                    'side': side, 'entry_price': entry_price}

        # --- Carry de funding (cada 2 velas = 8h) ---
        # En binance perp el funding se paga al pasar la hora de funding
        # (00:00, 08:00, 16:00 UTC). Aproximamos: cada 2 velas 4h acumulamos
        # el funding_rate de esa vela.
        if i % 2 == 0:
            f_rate = df['funding_rate'].iloc[b]
            if pd.notna(f_rate):
                if side == 'SHORT':
                    funding_pnl += float(f_rate)
                else:  # LONG
                    funding_pnl -= float(f_rate)

        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])

        if side == 'LONG':
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
            # Pesimista: SL primero si ambos tocados
            if lo <= sl_price:
                pnl = -sl_pct - 2 * commission + funding_pnl
                return {'outcome': 'SL', 'pnl_pct': pnl, 'bars': i,
                        'exit_price': sl_price, 'entry_ts': entry_ts,
                        'exit_ts': df.index[b], 'funding_pnl': funding_pnl,
                        'side': side, 'entry_price': entry_price}
            if hi >= tp_price:
                pnl = tp_pct - 2 * commission + funding_pnl
                return {'outcome': 'TP', 'pnl_pct': pnl, 'bars': i,
                        'exit_price': tp_price, 'entry_ts': entry_ts,
                        'exit_ts': df.index[b], 'funding_pnl': funding_pnl,
                        'side': side, 'entry_price': entry_price}
        else:  # SHORT
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)
            # Pesimista: SL primero si ambos tocados
            if hi >= sl_price:
                pnl = -sl_pct - 2 * commission + funding_pnl
                return {'outcome': 'SL', 'pnl_pct': pnl, 'bars': i,
                        'exit_price': sl_price, 'entry_ts': entry_ts,
                        'exit_ts': df.index[b], 'funding_pnl': funding_pnl,
                        'side': side, 'entry_price': entry_price}
            if lo <= tp_price:
                pnl = tp_pct - 2 * commission + funding_pnl
                return {'outcome': 'TP', 'pnl_pct': pnl, 'bars': i,
                        'exit_price': tp_price, 'entry_ts': entry_ts,
                        'exit_ts': df.index[b], 'funding_pnl': funding_pnl,
                        'side': side, 'entry_price': entry_price}

    # Timeout: cerrar a close del max_bars
    exit_p = float(df['close'].iloc[entry_bar + max_bars])
    if side == 'LONG':
        pnl = (exit_p - entry_price) / entry_price
    else:
        pnl = (entry_price - exit_p) / entry_price
    pnl = pnl - 2 * commission + funding_pnl
    return {'outcome': 'TIMEOUT', 'pnl_pct': pnl, 'bars': max_bars,
            'exit_price': exit_p, 'entry_ts': entry_ts,
            'exit_ts': df.index[entry_bar + max_bars],
            'funding_pnl': funding_pnl, 'side': side,
            'entry_price': entry_price}


# =============================================================================
# RUN_BACKTEST — UNA POSICION A LA VEZ
# =============================================================================
def run_backtest(df: pd.DataFrame, params: dict = PARAMS,
                 start_i: int | None = None,
                 end_i: int | None = None) -> list[dict]:
    """
    Recorre las velas [start_i, end_i). Al abrir un trade, salta hasta DESPUES
    de la vela en que cierra -> jamas solapa posiciones (mimica el bot real).
    """
    if start_i is None:
        start_i = params['min_bars_warmup']
    if end_i is None:
        end_i = len(df)

    trades = []
    i = max(start_i, params['min_bars_warmup'])
    while i < end_i:
        sig = signal(df, i, params)
        if sig is None:
            i += 1
            continue
        out = simulate(df, i, params, side=sig)
        trades.append({
            'entry_ts': out['entry_ts'],
            'exit_ts': out['exit_ts'],
            'side': sig,
            'outcome': out['outcome'],
            'pnl_pct': out['pnl_pct'],
            'bars': out['bars'],
            'funding_pnl': out['funding_pnl'],
            'entry_price': out['entry_price'],
            'exit_price': out['exit_price'],
        })
        # CRITICO: avanzar DESPUES de la vela de cierre (sin solapar)
        i += max(1, out['bars']) + 1
    return trades


# =============================================================================
# METRICAS
# =============================================================================
def metrics(trades: list[dict]) -> dict:
    if not trades:
        return {'n': 0, 'wr': 0.0, 'pf': 0.0, 'avg_pnl': 0.0,
                'total_return': 0.0, 'max_dd': 0.0, 'sharpe_like': 0.0,
                'months': 0.0, 'monthly_return': 0.0, 'annual_return': 0.0,
                'n_long': 0, 'n_short': 0,
                'sum_funding': 0.0, 'sum_total': 0.0,
                'funding_contrib_pct': 0.0, 'avg_holding_bars': 0.0}
    pnls = np.array([t['pnl_pct'] for t in trades])
    fund_pnls = np.array([t['funding_pnl'] for t in trades])
    bars = np.array([t['bars'] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    n = len(pnls)
    wr = len(wins) / n
    gw = float(wins.sum())
    gl = float(abs(losses.sum()))
    pf = (gw / gl) if gl > 1e-9 else float('inf')

    # equity sequencial (1 posicion a la vez)
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
    years = max(1.0/12.0, (tL - t0).days / 365.25)
    monthly_return = (eq ** (1 / months) - 1) if eq > 0 else -1.0
    annual_return = (eq ** (1 / years) - 1) if eq > 0 else -1.0

    sl = float(pnls.mean() / pnls.std()) if pnls.std() > 0 else 0.0

    sum_total = float(pnls.sum())
    sum_fund = float(fund_pnls.sum())
    # Contribucion del funding al total. Cuidado: si signo opuesto al PnL
    # total puede ser negativo (entonces el carry ayudo a evitar perdida).
    fc_pct = (sum_fund / abs(sum_total) * 100) if abs(sum_total) > 1e-9 else 0.0

    return {'n': n, 'wr': float(wr), 'pf': float(pf),
            'avg_pnl': float(pnls.mean()),
            'total_return': float(total), 'max_dd': float(dd),
            'sharpe_like': sl, 'months': months,
            'monthly_return': float(monthly_return),
            'annual_return': float(annual_return),
            'n_long': sum(1 for t in trades if t['side'] == 'LONG'),
            'n_short': sum(1 for t in trades if t['side'] == 'SHORT'),
            'sum_funding': sum_fund, 'sum_total': sum_total,
            'funding_contrib_pct': float(fc_pct),
            'avg_holding_bars': float(bars.mean())}
