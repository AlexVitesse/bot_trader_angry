"""
Agent K — V2 (A+F_BTC) + on-chain filters.
==========================================

Hipotesis a falsar
------------------
Las features on-chain (MVRV, SOPR proxy, exchange flows, active addresses)
contienen informacion FUNDAMENTAL del mercado que ni precio ni volumen
capturan. Anadirlas a V2 deberia MEJORAR el bootstrap p (baseline = 0.031).

Metricas on-chain (Coin Metrics community, gratis, sin auth)
------------------------------------------------------------
Diarias, publicadas con T+1 dia tipicamente. Aplicamos shift(2) por seguridad
ANTES de reindexar a 4h.

  - MVRV (CapMVRVCur):   Market-Value-to-Realized-Value. >3.7 historicamente
                          marca topes (sobrecomprado). <1 marca fondos.
  - Active addresses (AdrActCnt): proxy de adopcion/actividad red.
  - Exchange net flow:    FlowInExNtv - FlowOutExNtv. Positivo = BTC entrando
                          a exchanges (presion vendedora). Negativo = salida
                          (acumulacion HODL).
  - SOPR proxy: no esta en community. Usamos un proxy via ratio
                CapMVRVCur (que captura realized cap dinamica).
  - HashRate slope: actividad de mineros, proxy de confianza.

Diseno (Modo A — filtro sobre V2)
---------------------------------
1. V2 baseline: A primero (trend), F segundo (vol-breakout BTC-only). Una
   posicion a la vez.
2. Filtros on-chain (anti-toxico):
   - MVRV z-score > 2.5    -> bloquea LONG (sobrecomprado historico)
   - Exchange net inflow z > 2.0 (BTC entrando masivamente a exchanges)
     -> bloquea LONG (presion vendedora)
   - Para SHORT (F SHORT en BEAR): bloquea si MVRV z < -1.5 (capitulacion)
3. Cero re-tunear A o F. Solo filtramos a posteriori.

Thresholds a-priori (NO tuneados sobre el test):
  - mvrv_z_block_long = 2.5      (~ percentil 99 historico)
  - exch_in_z_block_long = 2.0   (~ percentil 97.5)
  - mvrv_z_unblock_short = -1.5  (~ percentil 6.7)

API
---
- PARAMS frozen.
- fetch_onchain_data() -> ya hecho en fetch_onchain.py
- prepare_onchain_features(df_4h, df_onchain) -> df con features on-chain
- onchain_veto(row, side, params) -> True si vetar entrada
- backtest_v2_plus_onchain(...) -> trades
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import importlib.util
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
EXP = ROOT / 'experiments'
ONCHAIN_PARQUET = Path(__file__).parent / 'onchain_data.parquet'


# =============================================================================
# PARAMS FROZEN  (todos elegidos A PRIORI, no tuneados sobre el test)
# =============================================================================
PARAMS = {
    # Z-score lookback (en dias daily, no 4h)
    'zscore_lookback_days': 90,      # 90 dias para z-score rolling

    # Shift on-chain (anti-look-ahead por delay de publicacion)
    'onchain_shift_days': 2,         # T+2 por seguridad

    # Filtros on-chain
    'mvrv_z_block_long': 2.5,        # MVRV z >2.5 -> bloquea LONG
    'exch_netflow_z_block_long': 2.0, # net inflow z >2.0 -> bloquea LONG
    'mvrv_z_unblock_short': -1.5,    # MVRV z <-1.5 -> bloquea SHORT (capitulacion)
    'enable_mvrv_filter': True,
    'enable_exchflow_filter': True,
    'enable_short_filter': True,

    'cutoff_date': '2027-01-01',
}


# =============================================================================
# Carga A y F y datos
# =============================================================================
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_btc_4h():
    df = pd.read_parquet(DATA / 'BTC_USDT_4h_full.parquet').sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df


def load_btc_1d():
    df = pd.read_parquet(DATA / 'btcusdt_1d_v15.parquet').sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df


def load_funding():
    df = pd.read_parquet(DATA / 'btc_v15_funding.parquet').sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df


def load_onchain():
    df = pd.read_parquet(ONCHAIN_PARQUET).sort_index()
    # Drop status columns from CM
    df = df[[c for c in df.columns if 'status' not in c]]
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    # Forward-fill the small gap of CapMrktEstUSD (21 NaNs at start)
    return df


# =============================================================================
# Features on-chain
# =============================================================================
def prepare_onchain_features(df_4h_index: pd.DatetimeIndex,
                             df_onchain: pd.DataFrame,
                             params: dict = PARAMS) -> pd.DataFrame:
    """
    Construye features on-chain con z-scores y slopes, aplicando shift y
    reindex al timeframe 4h.

    Output columns (todas alineadas a df_4h_index):
      - mvrv_raw, mvrv_z
      - active_addr_raw, active_addr_z
      - exch_netflow_btc, exch_netflow_z
      - hashrate_slope_7d
      - sply_ex_pct (% supply on exchanges)
      - sply_ex_slope_7d

    Todos los features llevan shift(onchain_shift_days) ANTES de reindex.
    """
    df = df_onchain.copy()
    shift_n = params['onchain_shift_days']
    lookback = params['zscore_lookback_days']

    # --- MVRV ---
    df['mvrv_raw'] = df['CapMVRVCur']
    mean_mvrv = df['mvrv_raw'].rolling(lookback, min_periods=30).mean()
    std_mvrv = df['mvrv_raw'].rolling(lookback, min_periods=30).std()
    df['mvrv_z'] = (df['mvrv_raw'] - mean_mvrv) / std_mvrv.replace(0, np.nan)

    # --- Active addresses ---
    df['active_addr_raw'] = df['AdrActCnt']
    mean_aa = df['active_addr_raw'].rolling(lookback, min_periods=30).mean()
    std_aa = df['active_addr_raw'].rolling(lookback, min_periods=30).std()
    df['active_addr_z'] = (df['active_addr_raw'] - mean_aa) / std_aa.replace(0, np.nan)

    # --- Exchange net flow (BTC) ---
    df['exch_netflow_btc'] = df['FlowInExNtv'] - df['FlowOutExNtv']
    mean_nf = df['exch_netflow_btc'].rolling(lookback, min_periods=30).mean()
    std_nf = df['exch_netflow_btc'].rolling(lookback, min_periods=30).std()
    df['exch_netflow_z'] = (df['exch_netflow_btc'] - mean_nf) / std_nf.replace(0, np.nan)

    # --- HashRate slope 7d ---
    df['hashrate_slope_7d'] = df['HashRate'].pct_change(7)

    # --- % supply on exchanges ---
    df['sply_ex_pct'] = df['SplyExNtv'] / df['SplyCur']
    df['sply_ex_slope_7d'] = df['sply_ex_pct'].diff(7)

    keep = ['mvrv_raw', 'mvrv_z',
            'active_addr_raw', 'active_addr_z',
            'exch_netflow_btc', 'exch_netflow_z',
            'hashrate_slope_7d',
            'sply_ex_pct', 'sply_ex_slope_7d']
    out = df[keep].copy()

    # CRITICAL: shift(shift_n) ANTES de reindex
    out = out.shift(shift_n)

    # Reindex a 4h, ffill (los datos de la fecha D-2 estan disponibles toda la
    # vela 4h del dia D)
    out = out.reindex(df_4h_index, method='ffill')
    return out


# =============================================================================
# Veto
# =============================================================================
def onchain_veto(row: pd.Series, side: str, params: dict = PARAMS) -> tuple[bool, str]:
    """
    Devuelve (True, razon) si hay que VETAR la entrada por on-chain toxico.
    """
    if side in ('LONG', 'A_LONG', 'F_LONG'):
        if params.get('enable_mvrv_filter', True):
            mvz = row.get('mvrv_z', np.nan)
            if pd.notna(mvz) and mvz > params['mvrv_z_block_long']:
                return True, f'mvrv_z={mvz:.2f}>{params["mvrv_z_block_long"]}'
        if params.get('enable_exchflow_filter', True):
            nfz = row.get('exch_netflow_z', np.nan)
            if pd.notna(nfz) and nfz > params['exch_netflow_z_block_long']:
                return True, f'exch_nf_z={nfz:.2f}>{params["exch_netflow_z_block_long"]}'
    elif side in ('SHORT', 'F_SHORT'):
        if params.get('enable_short_filter', True):
            mvz = row.get('mvrv_z', np.nan)
            if pd.notna(mvz) and mvz < params['mvrv_z_unblock_short']:
                return True, f'mvrv_z={mvz:.2f}<{params["mvrv_z_unblock_short"]} (capit.)'
    return False, ''


# =============================================================================
# Engine V2 + on-chain
# =============================================================================
def backtest_v2_plus_onchain(start_ts: pd.Timestamp,
                              end_ts: pd.Timestamp,
                              params: dict = PARAMS,
                              df_onchain_override: pd.DataFrame | None = None,
                              ) -> dict:
    """
    Replica el motor V2 (A primero, F segundo, BTC-only, una posicion a la vez)
    y aplica filtros on-chain ANTES de abrir el trade.

    Args:
        df_onchain_override: si se pasa, usa este DataFrame on-chain en lugar
            del default. Util para experimentos con random features.

    Returns:
        dict con keys: trades, trades_v2_baseline, vetoed
    """
    A = _load_module('A_strat', EXP / 'agent_A' / 'strategy.py')
    F = _load_module('F_strat', EXP / 'agent_F' / 'strategy.py')

    paramsA = dict(A.PARAMS); paramsA['cutoff_date'] = '2027-01-01'
    paramsF = dict(F.PARAMS); paramsF['cutoff_date'] = '2027-01-01'

    df_btc_4h = load_btc_4h()
    df_1d = load_btc_1d()
    df_fund = load_funding()
    df_onchain = df_onchain_override if df_onchain_override is not None else load_onchain()

    df_btc_A = A.prepare_data(df_btc_4h, df_1d, df_fund, paramsA)
    df_btc_F = F.prepare_data(df_btc_4h, df_1d, df_fund, paramsF)

    # Comun A & F
    btc_common = df_btc_A.index.intersection(df_btc_F.index)
    df_btc_A_c = df_btc_A.loc[btc_common].copy()
    df_btc_F_c = df_btc_F.loc[btc_common].copy()

    # On-chain features alineadas al indice 4h comun
    df_oc = prepare_onchain_features(btc_common, df_onchain, params)

    btc_start = int(btc_common.searchsorted(start_ts))
    btc_end = int(btc_common.searchsorted(end_ts))

    # =========================================================================
    # PASS 1: V2 baseline puro (sin filtros). Avanza POST-trade (no solape).
    # =========================================================================
    trades_v2_baseline = []
    i = btc_start
    while i < btc_end - 1:
        sigA = A.signal(df_btc_A_c, i, paramsA)
        if sigA == 'LONG':
            outA = A.simulate(df_btc_A_c, i, paramsA)
            barsA = int(outA.get('bars', 1))
            trades_v2_baseline.append({
                'ts': str(btc_common[i]),
                'side': 'A_LONG', 'strat': 'A',
                'outcome': outA.get('outcome'),
                'pnl_pct': float(outA.get('pnl_pct', 0.0)),
                'bars': barsA, 'entry_i': i,
            })
            i += barsA + 1
            continue
        sigF = F.signal(df_btc_F_c, i, paramsF)
        if sigF in ('LONG', 'SHORT'):
            outF = F.simulate(df_btc_F_c, i, paramsF, side=sigF)
            barsF = int(outF.get('bars', 1))
            pnlF = outF.get('leveraged_pnl_pct', outF.get('pnl_pct', 0.0))
            trades_v2_baseline.append({
                'ts': str(btc_common[i]),
                'side': f'F_{sigF}', 'strat': 'F',
                'outcome': outF.get('outcome'),
                'pnl_pct': float(pnlF),
                'bars': barsF, 'entry_i': i,
            })
            i += barsF + 1
            continue
        i += 1

    # =========================================================================
    # PASS 2: V2 + on-chain filter. Igual motor pero vetando entradas toxicas.
    # Si una entrada se veta, el slot queda libre y se reanuda busqueda en i+1.
    # =========================================================================
    trades_with_filter = []
    vetoed = []
    i = btc_start
    while i < btc_end - 1:
        oc_row = df_oc.iloc[i]
        sigA = A.signal(df_btc_A_c, i, paramsA)
        if sigA == 'LONG':
            veto, reason = onchain_veto(oc_row, 'A_LONG', params)
            outA = A.simulate(df_btc_A_c, i, paramsA)
            barsA = int(outA.get('bars', 1))
            trade_data = {
                'ts': str(btc_common[i]),
                'side': 'A_LONG', 'strat': 'A',
                'outcome': outA.get('outcome'),
                'pnl_pct': float(outA.get('pnl_pct', 0.0)),
                'bars': barsA, 'entry_i': i,
            }
            if veto:
                vetoed.append({**trade_data, 'veto_reason': reason})
                i += 1
                continue
            trades_with_filter.append(trade_data)
            i += barsA + 1
            continue
        sigF = F.signal(df_btc_F_c, i, paramsF)
        if sigF in ('LONG', 'SHORT'):
            veto, reason = onchain_veto(oc_row, f'F_{sigF}', params)
            outF = F.simulate(df_btc_F_c, i, paramsF, side=sigF)
            barsF = int(outF.get('bars', 1))
            pnlF = outF.get('leveraged_pnl_pct', outF.get('pnl_pct', 0.0))
            trade_data = {
                'ts': str(btc_common[i]),
                'side': f'F_{sigF}', 'strat': 'F',
                'outcome': outF.get('outcome'),
                'pnl_pct': float(pnlF),
                'bars': barsF, 'entry_i': i,
            }
            if veto:
                vetoed.append({**trade_data, 'veto_reason': reason})
                i += 1
                continue
            trades_with_filter.append(trade_data)
            i += barsF + 1
            continue
        i += 1

    return {
        'trades_v2_baseline': trades_v2_baseline,
        'trades_with_filter': trades_with_filter,
        'vetoed': vetoed,
    }


# =============================================================================
# Metricas + bootstrap (espejo de combined_AF)
# =============================================================================
def metrics(trades, weight=1.0):
    if not trades:
        return {'n': 0, 'wr': 0.0, 'pf': 0.0, 'total': 0.0,
                'max_dd': 0.0, 'monthly': 0.0, 'annual': 0.0}
    n = len(trades)
    pnls = [t['pnl_pct'] * weight for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / n
    gw = sum(wins); gl = abs(sum(losses))
    pf = (gw / gl) if gl > 1e-9 else float('inf')
    cum, peak, mdd = 1.0, 1.0, 0.0
    for t in sorted(trades, key=lambda x: pd.to_datetime(x['ts'])):
        cum *= (1.0 + t['pnl_pct'] * weight)
        peak = max(peak, cum)
        mdd = max(mdd, (peak - cum) / peak)
    ts0 = pd.to_datetime(trades[0]['ts'])
    ts1 = pd.to_datetime(trades[-1]['ts'])
    days = max(1, (ts1 - ts0).days)
    monthly = (cum - 1.0) / max(1.0, days / 30.0)
    annual = ((cum) ** (365.0 / days) - 1.0) if days >= 60 else float('nan')
    return {'n': n, 'wr': wr, 'pf': pf, 'total': cum - 1.0,
            'max_dd': mdd, 'monthly': monthly, 'annual': annual}


def bootstrap_p(trades, n_iter=3000, seed=42, weight=1.0):
    if len(trades) < 3:
        return None
    rng = np.random.default_rng(seed)
    pnls = np.array([t['pnl_pct'] * weight for t in trades])
    totals = np.empty(n_iter)
    for j in range(n_iter):
        s = rng.choice(pnls, size=len(pnls), replace=True)
        totals[j] = np.prod(1 + s) - 1
    return float(np.mean(totals <= 0))
