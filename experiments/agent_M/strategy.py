"""
Agent M -- SOL/USDT 4h LEVERAGED-BETA strategy on top of A's BTC LONG signal.
==============================================================================

Filosofia
---------
SOL es estructuralmente "BTC apalancado":
  - vol ratio SOL/BTC ~= 2.06 (anual 121% vs 59%)
  - beta SOL/BTC ~= 1.29 (full sample 2020-2026)
  - rolling 168-bar correlation: mediana 0.73, >=0.5 en 86% de las barras

Tesis: NO intentamos predecir SOL. Tomamos prestada la senal de A (Donchian-55
BTC LONG + EMA daily + funding veto, p=0.070 in-sample, unica con edge demostrable
sobre BTC junto a F), y la "amplificamos" tradeando SOL en los mismos momentos
cuando la correlacion SOL-BTC esta alta.

Si BTC tiene edge (V2 lo prueba con p=0.031), entonces operar SOL EN LOS
MISMOS TIMESTAMPS deberia heredar ese edge multiplicado por beta. Si no
existe la "alpha de SOL", al menos existe el "beta a un edge real".

Lo que evitamos (errores documentados del proyecto)
---------------------------------------------------
- Trailing tight (0.8%): inflaba PnL via bug intrabar -- usamos trailing AMPLIO 2.5x ATR.
- Predecir SOL: ML SOL dedicated dio 0 trades post-2022. NO ML aqui.
- Committee BB_UPPER SHORT: DD 48%. NO SHORT.
- Re-tunear: PARAMS frozen, derivados de A (BTC) escalados por vol ratio SOL/BTC=2.

Diferencias vs A en BTC
-----------------------
1. Entrada: dispara cuando A dispara LONG en BTC (mismo timestamp 4h)
            + filtro de correlacion SOL-BTC >= 0.5
2. ATR base: SOL (mas amplio). Floor 4% (vs 2.5% A), ceiling 10% (vs 6% A)
3. Multiplicador trailing: igual a A (2.5x ATR) -- no re-tuneamos.
4. max_bars: 60 igual que A (10 dias)
5. Sin filtro ADX/volumen propio de SOL -- la señal viene de A en BTC, no de SOL

Auditorias anti-look-ahead aplicadas
------------------------------------
- shift(1) en la correlacion rolling SOL-BTC
- A.prepare_data ya aplica shift(1) en bull_1d y funding_z
- A.signal usa solo info <= idx; A.simulate sin look-ahead intrabar
- prepare_data corta a cutoff_date <= 2025-12-31 inmediatamente
- Una posicion a la vez (run_backtest salta hasta despues del exit)
- Bars de SOL no perfectamente alineados con BTC: usamos common index

API publica
-----------
- PARAMS: dict frozen.
- prepare_data(df_sol_4h, df_btc_4h, df_btc_1d, df_funding) -> (df_sol_feat, df_btc_feat_A, common_idx_pos_map)
- signal(df_sol, df_btc_A, idx_common, params) -> 'LONG' | None
- simulate(df_sol, entry_bar_sol, params) -> dict
- run_backtest(df_sol, df_btc_A, common_index, params, start_i, end_i) -> list[trade]
"""

from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Importar la estrategia A (BTC) por ruta absoluta (evita colision de modulos)
HERE = Path(__file__).parent
_A_PATH = HERE.parent / 'agent_A' / 'strategy.py'
_spec = importlib.util.spec_from_file_location('agent_A_strategy', _A_PATH)
A = importlib.util.module_from_spec(_spec)
sys.modules['agent_A_strategy'] = A
_spec.loader.exec_module(A)


# =============================================================================
# PARAMETROS FROZEN  -- derivados a priori de A y de la vol ratio SOL/BTC
# =============================================================================
PARAMS = {
    # --- Costes ---
    'commission': 0.0005,           # 0.05% por lado (igual que A)

    # --- Filtro de correlacion SOL-BTC (rolling 168 bar = 4 semanas) ---
    'corr_window': 168,             # 168 * 4h = 28 dias
    'corr_min': 0.5,                # umbral: 86% del tiempo SOL-BTC corr >= 0.5

    # --- Trailing ATR sobre SOL (mas amplio que A en BTC -- SOL es 2x volatil) ---
    # A en BTC: floor 2.5%, ceiling 6%, mult 2.5x ATR
    # SOL en SOL: floor 4%, ceiling 10%, mult 2.5x (no re-tuneamos el mult)
    'atr_n': 14,                    # igual que A
    'trail_atr_mult': 2.5,          # igual que A -- no re-tunear
    'trail_floor_pct': 0.04,        # ~ 1.6x el floor de A (vol ratio ~2x)
    'trail_ceiling_pct': 0.10,      # ~ 1.67x el ceiling de A
    'max_bars': 60,                 # igual que A (10 dias)

    # --- Operativos ---
    'cutoff_date': '2025-12-31',
    'min_bars_warmup': 250,         # warmup minimo para indicadores
}


# =============================================================================
# CARGA Y FEATURES
# =============================================================================
def _atr(h, l, c, n=14):
    """ATR Wilder -- copia de A para consistencia."""
    return A._atr(h, l, c, n)


def prepare_data(df_sol_4h: pd.DataFrame,
                 df_btc_4h: pd.DataFrame,
                 df_btc_1d: pd.DataFrame | None = None,
                 df_funding: pd.DataFrame | None = None,
                 params: dict = PARAMS) -> tuple:
    """
    Devuelve (df_sol_feat, df_btc_feat_A, common_idx)
      - df_sol_feat: SOL 4h con ATR (para el trailing) + corr_sol_btc (rolling shift1)
      - df_btc_feat_A: BTC 4h con TODAS las features de A (bull_1d, donchian, adx, vol, funding_z)
      - common_idx: pd.DatetimeIndex compartido entre SOL y BTC tras warmup

    Cortamos ambos a cutoff_date <= 2025-12-31.
    """
    cutoff = pd.Timestamp(params['cutoff_date'], tz='UTC')

    df_sol = df_sol_4h.copy()
    if df_sol.index.tz is None:
        df_sol.index = df_sol.index.tz_localize('UTC')
    df_sol = df_sol[df_sol.index <= cutoff].sort_index()

    df_btc = df_btc_4h.copy()
    if df_btc.index.tz is None:
        df_btc.index = df_btc.index.tz_localize('UTC')
    df_btc = df_btc[df_btc.index <= cutoff].sort_index()

    # === SOL features ===
    h, l, c = df_sol['high'], df_sol['low'], df_sol['close']
    df_sol['atr'] = _atr(h, l, c, params['atr_n'])
    df_sol['atr_pct'] = df_sol['atr'] / c

    # Correlacion rolling SOL-BTC sobre retornos log (con shift(1) para anti look-ahead)
    ret_sol = np.log(df_sol['close'] / df_sol['close'].shift(1))
    # Alinear BTC al indice de SOL para el calculo de corr
    btc_close_aligned = df_btc['close'].reindex(df_sol.index, method='ffill')
    ret_btc = np.log(btc_close_aligned / btc_close_aligned.shift(1))

    corr = ret_sol.rolling(params['corr_window']).corr(ret_btc)
    corr = corr.shift(1)  # CRITICAL: la corr conocida en t usa solo info <=t-1
    df_sol['corr_sol_btc'] = corr

    # === BTC features para A ===
    # Usamos los params de A (con su propio cutoff y warmup)
    paramsA = dict(A.PARAMS)
    paramsA['cutoff_date'] = params['cutoff_date']
    df_btc_A = A.prepare_data(df_btc, df_btc_1d, df_funding, paramsA)

    # === Indice comun (BTC y SOL ambos tienen features listas) ===
    # SOL empieza 2020-08-11; BTC features 2019+. SOL puede tener gaps frente a BTC 4h.
    df_sol = df_sol.dropna(subset=['atr', 'corr_sol_btc'])
    common_idx = df_sol.index.intersection(df_btc_A.index)

    df_sol = df_sol.loc[common_idx].copy()
    df_btc_A = df_btc_A.loc[common_idx].copy()

    return df_sol, df_btc_A, common_idx


# =============================================================================
# SIGNAL
# =============================================================================
def signal(df_sol: pd.DataFrame,
           df_btc_A: pd.DataFrame,
           idx: int,
           params: dict = PARAMS,
           a_signal_fn=None) -> str | None:
    """
    Devuelve 'LONG' si A dispara LONG en BTC en el mismo idx (timestamp 4h)
    Y la correlacion SOL-BTC rolling 168 >= corr_min.

    a_signal_fn: function opcional para override (usado en el cross-check random).
    Default: A.signal con A.PARAMS (cutoff respetado al preparar datos).
    """
    if idx < params['min_bars_warmup']:
        return None
    if idx >= len(df_sol) - 2:
        return None

    # 1) A's signal sobre BTC en el mismo idx
    if a_signal_fn is None:
        # signal nativo de A
        paramsA = dict(A.PARAMS)
        paramsA['cutoff_date'] = params['cutoff_date']
        sigA = A.signal(df_btc_A, idx, paramsA)
    else:
        sigA = a_signal_fn(df_btc_A, idx)

    if sigA != 'LONG':
        return None

    # 2) Filtro de correlacion SOL-BTC
    row_sol = df_sol.iloc[idx]
    corr = row_sol.get('corr_sol_btc', np.nan)
    if pd.isna(corr) or corr < params['corr_min']:
        return None

    # 3) ATR de SOL valido (necesario para el trailing)
    atr_pct = row_sol.get('atr_pct', np.nan)
    if pd.isna(atr_pct) or atr_pct <= 0:
        return None

    return 'LONG'


# =============================================================================
# SIMULATE -- trailing stop ATR amplio sobre SOL, SIN look-ahead intrabar
# =============================================================================
def simulate(df_sol: pd.DataFrame, entry_bar: int, params: dict = PARAMS) -> dict:
    """
    Trade LONG SOL abierto al CLOSE de entry_bar con trailing stop ATR SOL.

    Mismo motor honesto que A.simulate (sin look-ahead intrabar):
      Para cada vela b > entry_bar:
        1) revisar salida contra SL HEREDADO
        2) recien despues actualizar peak/SL con high[b]
    """
    n = len(df_sol)
    entry_price = float(df_sol['close'].iloc[entry_bar])
    entry_ts = df_sol.index[entry_bar]
    atr_pct = float(df_sol['atr_pct'].iloc[entry_bar])

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
            exit_p = float(df_sol['close'].iloc[-1])
            pnl = (exit_p - entry_price) / entry_price - 2 * commission
            return {'outcome': 'TIMEOUT', 'pnl_pct': pnl, 'bars': i,
                    'exit_price': exit_p, 'entry_ts': entry_ts,
                    'exit_ts': df_sol.index[-1], 'reason': 'eod'}

        hi = float(df_sol['high'].iloc[b])
        lo = float(df_sol['low'].iloc[b])

        # 1) Salida contra SL ya conocido (peak heredado de barras anteriores)
        if lo <= sl_price:
            pnl = (sl_price - entry_price) / entry_price - 2 * commission
            return {'outcome': ('TP' if sl_price > entry_price else 'SL'),
                    'pnl_pct': pnl, 'bars': i,
                    'exit_price': sl_price, 'entry_ts': entry_ts,
                    'exit_ts': df_sol.index[b], 'reason': 'trail'}

        # 2) Actualizar peak/SL para la siguiente vela
        if hi > peak:
            peak = hi
        sl_price = max(sl_price, peak * (1 - trail_dist))

    # Timeout
    exit_p = float(df_sol['close'].iloc[entry_bar + max_bars])
    pnl = (exit_p - entry_price) / entry_price - 2 * commission
    return {'outcome': 'TIMEOUT', 'pnl_pct': pnl, 'bars': max_bars,
            'exit_price': exit_p, 'entry_ts': entry_ts,
            'exit_ts': df_sol.index[entry_bar + max_bars],
            'reason': 'max_bars'}


# =============================================================================
# RUN_BACKTEST -- una posicion a la vez (no overlap)
# =============================================================================
def run_backtest(df_sol: pd.DataFrame,
                 df_btc_A: pd.DataFrame,
                 params: dict = PARAMS,
                 start_i: int | None = None,
                 end_i: int | None = None,
                 a_signal_fn=None) -> list[dict]:
    """
    Recorre [start_i, end_i). Al abrir un trade, salta hasta DESPUES de su cierre.
    df_sol y df_btc_A DEBEN compartir el mismo indice (prepare_data lo asegura).
    """
    assert len(df_sol) == len(df_btc_A), \
        f"df_sol y df_btc_A deben tener el mismo indice (len={len(df_sol)} vs {len(df_btc_A)})"

    if start_i is None:
        start_i = params['min_bars_warmup']
    if end_i is None:
        end_i = len(df_sol)

    trades = []
    i = max(start_i, params['min_bars_warmup'])
    while i < end_i:
        sig = signal(df_sol, df_btc_A, i, params, a_signal_fn=a_signal_fn)
        if sig != 'LONG':
            i += 1
            continue
        out = simulate(df_sol, i, params)
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
        i += max(1, out['bars']) + 1
    return trades


# =============================================================================
# METRICAS
# =============================================================================
def metrics(trades: list[dict]) -> dict:
    if not trades:
        return {'n': 0, 'wr': 0.0, 'pf': 0.0, 'avg_pnl': 0.0,
                'total_return': 0.0, 'max_dd': 0.0, 'sharpe_like': 0.0,
                'months': 0.0, 'monthly_return': 0.0, 'annual': 0.0}
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
    days = max(1, (tL - t0).days)
    months = max(1.0, days / 30.0)
    monthly_return = (eq ** (1 / months) - 1) if eq > 0 else -1.0
    annual = (eq ** (365.0 / days) - 1.0) if days >= 60 else float('nan')

    sl = float(pnls.mean() / pnls.std()) if pnls.std() > 0 else 0.0
    return {'n': n, 'wr': float(wr), 'pf': float(pf), 'avg_pnl': float(pnls.mean()),
            'total_return': float(total), 'max_dd': float(dd),
            'sharpe_like': sl, 'months': months,
            'monthly_return': float(monthly_return),
            'annual': float(annual) if np.isfinite(annual) else 0.0}
