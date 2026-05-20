"""
test_oos_marginals.py
=====================
Validacion OOS 2026 (Ene-Feb) de las 5 monedas MARGINAL del test_v2_all.

Pregunta: ¿cuáles de las marginales sobreviven OOS 2026 con V2 honesto?

OOS window: 2026-01-01 a fin de datos (~2026-02-27).
BTC OOS 2026 ya validado: 5 trades, PF 2.95, +12.1%, WR 60%, DD 6%.

Criterio para PASS OOS:
- Retorno total OOS >= 0
- PF >= 1.0 (no esta vendiendo dinero)
- DD <= 15% (riesgo controlado en muestra corta)
- Min N trades >= 3 (sino indeterminado)

Coins evaluadas (5 marginales + BTC referencia):
- BTC (KEEP 3/3)
- DOGE (MARGINAL 2/3, p=0.001)
- ETH (MARGINAL 2/3, p=0.472)
- XRP (MARGINAL 2/3, p=0.235)
- BNB (MARGINAL 2/3, p=0.315)
- OP  (MARGINAL 2/3, p=0.539)
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

OOS_START = pd.Timestamp('2026-01-01', tz='UTC')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = load_module('A_strat', EXP / 'agent_A' / 'strategy.py')
F = load_module('F_strat', EXP / 'agent_F' / 'strategy.py')


def load_coin_full(coin):
    """Carga TODA la data (incluyendo 2026 si existe)."""
    candidates = [
        DATA / f'{coin}_USDT_4h_full.parquet',
        DATA / f'{coin}USDT_4h_v15.parquet',
        DATA / f'{coin}_USDT_4h_history.parquet',
    ]
    for f in candidates:
        if f.exists():
            df = pd.read_parquet(f).sort_index()
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            cols = ['open', 'high', 'low', 'close', 'volume']
            missing = [c for c in cols if c not in df.columns]
            if missing:
                continue
            return df[cols].copy()
    return None


def run_V2_oos(coin):
    """Aplica V2 al coin restringido a OOS 2026."""
    df = load_coin_full(coin)
    if df is None:
        return None, 'no_data'
    if df.index[-1] < OOS_START:
        return None, 'no_oos_data'

    paramsA = dict(A.PARAMS); paramsA['cutoff_date'] = '2099-01-01'
    paramsF = dict(F.PARAMS); paramsF['cutoff_date'] = '2099-01-01'

    df_A = A.prepare_data(df, None, None, paramsA)
    df_F = F.prepare_data(df, None, None, paramsF)
    common = df_A.index.intersection(df_F.index)
    df_A_c = df_A.loc[common]
    df_F_c = df_F.loc[common]

    oos_start_i = int(common.searchsorted(OOS_START))
    if oos_start_i >= len(common):
        return None, 'no_oos_after_features'

    trades = []
    i = oos_start_i
    end_i = len(common) - 1
    while i < end_i:
        sigA = A.signal(df_A_c, i, paramsA)
        if sigA == 'LONG':
            out = A.simulate(df_A_c, i, paramsA)
            bars = int(out.get('bars', 1))
            trades.append({'ts': str(common[i]), 'strat': 'A', 'side': 'LONG',
                           'pnl_pct': float(out.get('pnl_pct', 0.0)),
                           'outcome': out.get('outcome'), 'bars': bars})
            i += bars + 1
            continue
        sigF = F.signal(df_F_c, i, paramsF)
        if sigF in ('LONG', 'SHORT'):
            out = F.simulate(df_F_c, i, paramsF, side=sigF)
            bars = int(out.get('bars', 1))
            pnl = out.get('leveraged_pnl_pct', out.get('pnl_pct', 0.0))
            trades.append({'ts': str(common[i]), 'strat': 'F', 'side': sigF,
                           'pnl_pct': float(pnl),
                           'outcome': out.get('outcome'), 'bars': bars})
            i += bars + 1
            continue
        i += 1
    return trades, common[oos_start_i], common[-1]


def metrics(trades):
    if not trades:
        return {'n': 0, 'wr': 0.0, 'pf': 0.0, 'total': 0.0, 'max_dd': 0.0}
    n = len(trades)
    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / n
    gw = sum(wins); gl = abs(sum(losses))
    pf = (gw / gl) if gl > 1e-9 else float('inf')
    cum, peak, mdd = 1.0, 1.0, 0.0
    for t in sorted(trades, key=lambda x: pd.to_datetime(x['ts'])):
        cum *= (1.0 + t['pnl_pct'])
        peak = max(peak, cum)
        mdd = max(mdd, (peak - cum) / peak)
    return {'n': n, 'wr': wr, 'pf': pf, 'total': cum - 1, 'max_dd': mdd}


COINS = ['BTC', 'DOGE', 'ETH', 'XRP', 'BNB', 'OP']

print("=" * 96)
print("OOS 2026 (Ene-Feb) — V2 sobre las 5 marginales + BTC referencia")
print("=" * 96)
print()
print(f"{'Coin':<8}{'OOS bars':<12}{'N':<5}{'WR':<8}{'PF':<8}"
      f"{'Total':<10}{'DD':<8}{'Verdict'}")
print('-' * 96)

results = {}
for coin in COINS:
    out = run_V2_oos(coin)
    if out is None or out[0] is None:
        status = out[1] if out is not None else 'load_failed'
        print(f"{coin:<8}{status}")
        results[coin] = {'status': status}
        continue
    trades, t0, t1 = out
    m = metrics(trades)

    # Verdict: positive return + PF >= 1 + DD <= 15% + n >= 3
    n_pass = m['n'] >= 3
    ret_pass = m['total'] >= 0
    pf_pass = m['pf'] >= 1.0 if m['n'] > 0 else False
    dd_pass = m['max_dd'] <= 0.15

    if n_pass and ret_pass and pf_pass and dd_pass:
        verdict = "PASS"
    elif m['n'] == 0:
        verdict = "FLAT (0 trades — defensivo OK)"
    elif n_pass and ret_pass and pf_pass:
        verdict = "PASS-marg (DD alto)"
    else:
        verdict = "FAIL"

    days = (t1 - t0).days
    pf_s = f"{m['pf']:.2f}" if np.isfinite(m['pf']) else "inf"
    print(f"{coin:<8}{days:<12d}{m['n']:<5d}{m['wr']:<8.1%}{pf_s:<8s}"
          f"{m['total']:<+10.2%}{m['max_dd']:<+8.1%}{verdict}")

    results[coin] = {
        'n_oos_days': days,
        'n_trades': m['n'],
        'wr': m['wr'],
        'pf': m['pf'] if np.isfinite(m['pf']) else None,
        'total': m['total'],
        'max_dd': m['max_dd'],
        'verdict': verdict,
        'trades': [{'ts': t['ts'][:16], 'side': t['side'], 'strat': t['strat'],
                    'outcome': t['outcome'], 'pnl': t['pnl_pct']} for t in trades],
    }

print()
print("Detalle por trade:")
for coin, r in results.items():
    if r.get('status') or r.get('n_trades', 0) == 0:
        continue
    print(f"\n  {coin} ({r['n_trades']} trades):")
    for t in r['trades']:
        print(f"    {t['ts']}  {t['side']:<5}  {t['strat']}  "
              f"{t['outcome']:>7}  {t['pnl']:+.2%}")

# Summary
print("\n" + "=" * 96)
print("RESUMEN — quien pasa OOS")
print("=" * 96)
passing = [c for c, r in results.items() if r.get('verdict', '').startswith('PASS')]
flat = [c for c, r in results.items() if r.get('verdict', '').startswith('FLAT')]
failing = [c for c, r in results.items() if r.get('verdict', '').startswith('FAIL')]

print(f"\nPASS OOS:    {', '.join(passing) if passing else 'ninguno'}")
print(f"FLAT (0 trades, defensivo): {', '.join(flat) if flat else 'ninguno'}")
print(f"FAIL OOS:    {', '.join(failing) if failing else 'ninguno'}")

# Save results
import json
out_path = ROOT / 'experiments' / 'v2_all_coins' / 'oos_2026_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nGuardado: {out_path}")
