"""
test_combined.py -- A + F combinados con motor honesto.

Pregunta a responder: ¿La combinación A (trend BTC) + F (vol-breakout BTC+ETH)
tiene edge mejor que cada uno por separado? ¿En in-sample 2020-2025? ¿En OOS 2026?

Reglas:
- Una posición POR ACTIVO a la vez (BTC y ETH pueden tradear en paralelo).
- En BTC: A tiene prioridad (más conservador). Si A no fira, intenta F.
- En ETH: solo F (A es BTC-only).
- Motor honesto: sin overlap, sin look-ahead intrabar (los sims de A y F ya son honestos).
- Equity combinado: trades sumados en orden temporal asumiendo 50% capital sleeve por activo.

Salida: métricas in-sample + OOS, bootstrap p-value, desglose por estrategia/activo.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
EXP = ROOT / 'experiments'


def load_module(name, path):
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


def load_eth_4h():
    df = pd.read_parquet(DATA / 'ETH_USDT_4h_full.parquet').sort_index()
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


def metrics(trades, weight=1.0):
    """
    weight=1.0 -> sleeve 100% (cada trade ocupa todo el capital secuencialmente)
    weight=0.5 -> sleeve 50% (mitad del capital — útil para portfolio BTC+ETH 50/50)
    """
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


# ---------------------------------------------------------------------------
# Motor combinado: A primero en BTC; F en BTC (si A no fira) y en ETH.
# Una posición a la vez POR ACTIVO. BTC y ETH pueden estar simultáneos (sleeves).
# ---------------------------------------------------------------------------
def run_combined(start_ts, end_ts):
    A = load_module('A_strat', EXP / 'agent_A' / 'strategy.py')
    F = load_module('F_strat', EXP / 'agent_F' / 'strategy.py')

    paramsA = dict(A.PARAMS); paramsA['cutoff_date'] = '2027-01-01'
    paramsF = dict(F.PARAMS); paramsF['cutoff_date'] = '2027-01-01'

    df_btc_4h = load_btc_4h()
    df_eth_4h = load_eth_4h()
    df_1d = load_btc_1d()
    df_fund = load_funding()

    df_btc_A = A.prepare_data(df_btc_4h, df_1d, df_fund, paramsA)
    df_btc_F = F.prepare_data(df_btc_4h, df_1d, df_fund, paramsF)
    df_eth_F = F.prepare_data(df_eth_4h, df_1d, df_fund, paramsF)

    # Para BTC: usar índice común de A y F (el más restrictivo en warmup gana)
    btc_common = df_btc_A.index.intersection(df_btc_F.index)
    df_btc_A_c = df_btc_A.loc[btc_common].copy()
    df_btc_F_c = df_btc_F.loc[btc_common].copy()

    btc_start = int(btc_common.searchsorted(start_ts))
    btc_end = int(btc_common.searchsorted(end_ts))
    eth_start = int(df_eth_F.index.searchsorted(start_ts))
    eth_end = int(df_eth_F.index.searchsorted(end_ts))

    # ---- BTC: A tiene prioridad ----
    btc_trades = []
    i = btc_start
    while i < btc_end - 1:
        # 1) A primero (long-only trend)
        sigA = A.signal(df_btc_A_c, i, paramsA)
        if sigA == 'LONG':
            out = A.simulate(df_btc_A_c, i, paramsA)
            bars = int(out.get('bars', 1))
            btc_trades.append({
                'ts': str(btc_common[i]),
                'side': 'A_LONG', 'asset': 'BTC', 'strat': 'A',
                'outcome': out.get('outcome'),
                'pnl_pct': float(out.get('pnl_pct', 0.0)),
                'bars': bars,
            })
            i += bars + 1
            continue
        # 2) F si A no dispara
        sigF = F.signal(df_btc_F_c, i, paramsF)
        if sigF in ('LONG', 'SHORT'):
            out = F.simulate(df_btc_F_c, i, paramsF, side=sigF)
            bars = int(out.get('bars', 1))
            pnl = out.get('leveraged_pnl_pct', out.get('pnl_pct', 0.0))
            btc_trades.append({
                'ts': str(btc_common[i]),
                'side': f'F_{sigF}', 'asset': 'BTC', 'strat': 'F',
                'outcome': out.get('outcome'),
                'pnl_pct': float(pnl),
                'bars': bars,
            })
            i += bars + 1
            continue
        i += 1

    # ---- ETH: solo F ----
    eth_trades = []
    i = eth_start
    while i < eth_end - 1:
        sigF = F.signal(df_eth_F, i, paramsF)
        if sigF in ('LONG', 'SHORT'):
            out = F.simulate(df_eth_F, i, paramsF, side=sigF)
            bars = int(out.get('bars', 1))
            pnl = out.get('leveraged_pnl_pct', out.get('pnl_pct', 0.0))
            eth_trades.append({
                'ts': str(df_eth_F.index[i]),
                'side': f'F_{sigF}', 'asset': 'ETH', 'strat': 'F',
                'outcome': out.get('outcome'),
                'pnl_pct': float(pnl),
                'bars': bars,
            })
            i += bars + 1
            continue
        i += 1

    return btc_trades, eth_trades


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------
def variant_metrics(label, trades, weight):
    m = metrics(trades, weight=weight)
    p = bootstrap_p(trades, weight=weight) if len(trades) >= 3 else None
    p_s = f'{p:.3f}' if p is not None else '—'
    sig = '[OK]' if (p is not None and p < 0.05) else ('[~]' if (p is not None and p < 0.10) else '[X] ')
    print(f"  {sig} {label:<48}  n={m['n']:>3}  PF={m['pf']:>5.2f}  "
          f"annual={m['annual']:>+6.1%}  DD={m['max_dd']:>5.1%}  p={p_s}")
    return m, p


def report(title, btc_trades, eth_trades):
    print(f'\n{"=" * 88}\n{title}\n{"=" * 88}')
    all_trades = btc_trades + eth_trades
    all_trades.sort(key=lambda t: pd.to_datetime(t['ts']))

    # Sleeve 100% (cada trade ocupa todo el capital) — secuencial, BTC y ETH no se mezclan en capital
    m_sleeve = metrics(all_trades, weight=1.0)
    # Portfolio 50/50 (BTC y ETH cada uno 50% del capital, pueden correr en paralelo)
    m_5050 = metrics(all_trades, weight=0.5)
    p_sleeve = bootstrap_p(all_trades, weight=1.0)
    p_5050 = bootstrap_p(all_trades, weight=0.5)

    print(f"\nCombinado (assumiendo sleeve 100% por trade, secuencial):")
    print(f"  trades={m_sleeve['n']}  WR={m_sleeve['wr']:.1%}  PF={m_sleeve['pf']:.2f}")
    print(f"  total={m_sleeve['total']:+.1%}  annual={m_sleeve['annual']:+.1%}  "
          f"mensual={m_sleeve['monthly']:+.2%}  DD={m_sleeve['max_dd']:.1%}  "
          f"bootstrap p={p_sleeve:.3f}" if p_sleeve is not None else "")
    print(f"\nPortfolio 50/50 (BTC y ETH paralelos, mitad capital cada uno):")
    print(f"  total={m_5050['total']:+.1%}  annual={m_5050['annual']:+.1%}  "
          f"mensual={m_5050['monthly']:+.2%}  DD={m_5050['max_dd']:.1%}  "
          f"bootstrap p={p_5050:.3f}" if p_5050 is not None else "")

    # Desglose por componente
    A_BTC = [t for t in btc_trades if t['strat'] == 'A']
    F_BTC = [t for t in btc_trades if t['strat'] == 'F']
    F_BTC_LONG = [t for t in F_BTC if 'LONG' in t['side']]
    F_BTC_SHORT = [t for t in F_BTC if 'SHORT' in t['side']]
    F_ETH = eth_trades
    F_ETH_LONG = [t for t in F_ETH if 'LONG' in t['side']]
    F_ETH_SHORT = [t for t in F_ETH if 'SHORT' in t['side']]

    print(f'\nComponentes individuales (sleeve 100%):')
    variant_metrics('A_BTC (trend LONG)', A_BTC, 1.0)
    variant_metrics('F_BTC LONG', F_BTC_LONG, 1.0)
    variant_metrics('F_BTC SHORT', F_BTC_SHORT, 1.0)
    variant_metrics('F_ETH LONG', F_ETH_LONG, 1.0)
    variant_metrics('F_ETH SHORT', F_ETH_SHORT, 1.0)

    print(f'\nVariantes combinadas (qué descartar):')
    # V1: completo
    variant_metrics('V1: A + F (BTC+ETH, todo) - sleeve 100%', all_trades, 1.0)
    variant_metrics('V1: A + F (BTC+ETH, todo) - 50/50',       all_trades, 0.5)
    # V2: drop F_ETH
    v2 = A_BTC + F_BTC
    variant_metrics('V2: A + F_BTC (drop ETH lastre) - sleeve', v2, 1.0)
    # V3: A LONG + F SHORT only (natural division)
    v3 = A_BTC + F_BTC_SHORT + F_ETH_SHORT
    variant_metrics('V3: A_LONG + F_SHORT_only (BTC+ETH) - sleeve', v3, 1.0)
    variant_metrics('V3: A_LONG + F_SHORT_only (BTC+ETH) - 50/50', v3, 0.5)
    # V4: A LONG + F_BTC SHORT (drop ETH entirely)
    v4 = A_BTC + F_BTC_SHORT
    variant_metrics('V4: A_LONG + F_BTC_SHORT only - sleeve', v4, 1.0)

    return all_trades


def main():
    # IN-SAMPLE 2020-2025
    is_btc, is_eth = run_combined(
        pd.Timestamp('2020-01-01', tz='UTC'),
        pd.Timestamp('2026-01-01', tz='UTC'))
    is_trades = report("IN-SAMPLE 2020-2025 (los agentes diseñaron con esto)",
                       is_btc, is_eth)

    # OOS 2026
    oos_btc, oos_eth = run_combined(
        pd.Timestamp('2026-01-01', tz='UTC'),
        pd.Timestamp('2027-01-01', tz='UTC'))
    oos_trades = report("OOS 2026 (datos que NO se vieron al diseñar)",
                        oos_btc, oos_eth)

    if oos_trades:
        print("\nTrades OOS 2026:")
        for t in oos_trades:
            ts = pd.to_datetime(t['ts']).strftime('%Y-%m-%d %H:%M')
            print(f"    {ts}  {t['side']:<10s} {t['asset']:<4s}  "
                  f"{(t['outcome'] or ''):>7s}  {t['pnl_pct']:+.2%}  ({t['bars']}b)")


if __name__ == '__main__':
    main()
