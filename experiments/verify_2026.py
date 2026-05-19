"""
verify_2026.py -- Verificación independiente de las 3 estrategias en 2026 OOS.

Los agentes A, B, C entrenaron/diseñaron con datos <= 2025-12-31.
Aquí corro sus signal+simulate sobre Ene-Feb 2026 (datos que NO vieron) con un
motor honesto (una posición a la vez), mido WR / PF / retorno y bootstrap.

Si las estrategias eran reales, sobreviven. Si eran overfit (aunque sus auto-
reportes ya fueron honestos), aquí se ve.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent      # repo root
DATA = ROOT / 'data'
EXP = ROOT / 'experiments'

OOS_START = pd.Timestamp('2026-01-01', tz='UTC')

# ---------------------------------------------------------------------------
# Carga de datos (sin cutoff -- el cutoff lo aplica el verifier al masking)
# ---------------------------------------------------------------------------
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


def load_fng():
    p = DATA / 'fear_greed_history.parquet'
    if not p.exists():
        return None
    df = pd.read_parquet(p).sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def metrics(trades):
    if not trades:
        return {'n': 0, 'wr': 0.0, 'pf': 0.0, 'total': 0.0,
                'max_dd': 0.0, 'monthly': 0.0}
    n = len(trades)
    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / n
    gw = sum(wins)
    gl = abs(sum(losses))
    pf = (gw / gl) if gl > 1e-9 else float('inf')
    cum, peak, mdd = 1.0, 1.0, 0.0
    for t in sorted(trades, key=lambda x: pd.to_datetime(x['ts'])):
        cum *= (1.0 + t['pnl_pct'])
        peak = max(peak, cum)
        mdd = max(mdd, (peak - cum) / peak)
    ts0 = pd.to_datetime(trades[0]['ts'])
    ts1 = pd.to_datetime(trades[-1]['ts'])
    months = max(1, (ts1 - ts0).days / 30)
    return {'n': n, 'wr': wr, 'pf': pf, 'total': cum - 1.0,
            'max_dd': mdd, 'monthly': (cum - 1.0) / months}


def bootstrap_pvalue(trades, n_iter=2000, seed=42):
    if len(trades) < 3:
        return None
    rng = np.random.default_rng(seed)
    pnls = np.array([t['pnl_pct'] for t in trades])
    totals = np.empty(n_iter)
    for j in range(n_iter):
        s = rng.choice(pnls, size=len(pnls), replace=True)
        totals[j] = np.prod(1 + s) - 1
    return float(np.mean(totals <= 0))


def run_one_position(df, oos_start_i, end_i, signal_fn, simulate_fn):
    """Motor honesto: una posición a la vez. signal_fn / simulate_fn dependen del agente."""
    trades = []
    i = oos_start_i
    while i < end_i:
        sig = signal_fn(df, i)
        if sig is None:
            i += 1
            continue
        # signal puede devolver str, (str, regime), o None
        if isinstance(sig, tuple):
            side, regime = sig
        else:
            side, regime = sig, None
        if side is None:
            i += 1
            continue
        if regime is not None:
            out = simulate_fn(df, i, regime)
        else:
            out = simulate_fn(df, i)
        bars = int(out.get('bars', 1))
        trades.append({
            'ts': str(df.index[i]),
            'side': side,
            'regime': regime,
            'outcome': out.get('outcome'),
            'pnl_pct': float(out['pnl_pct']),
            'bars': bars,
        })
        i += bars + 1
    return trades


# ---------------------------------------------------------------------------
# AGENTE A
# ---------------------------------------------------------------------------
def verify_A():
    A = load_module('agent_A_strategy', EXP / 'agent_A' / 'strategy.py')
    # override cutoff para que prepare_data NO recorte 2026
    params = dict(A.PARAMS)
    params['cutoff_date'] = '2027-01-01'
    df_4h = load_btc_4h()
    df_1d = load_btc_1d()
    df_f = load_funding()
    df = A.prepare_data(df_4h, df_1d, df_f, params)
    oos_i = df.index.searchsorted(OOS_START)
    sig = lambda d, i: A.signal(d, i, params)
    sim = lambda d, i: A.simulate(d, i, params)
    trades = run_one_position(df, oos_i, len(df), sig, sim)
    return trades, df.index[oos_i], df.index[-1]


# ---------------------------------------------------------------------------
# AGENTE B
# ---------------------------------------------------------------------------
def verify_B():
    B = load_module('agent_B_strategy', EXP / 'agent_B' / 'strategy.py')
    # threshold calibrado en train
    params = dict(B.PARAMS)
    tp_path = EXP / 'agent_B' / 'trained_params.json'
    if tp_path.exists():
        params.update(json.loads(tp_path.read_text(encoding='utf-8')))
    model = joblib.load(EXP / 'agent_B' / 'model.pkl')
    scaler = joblib.load(EXP / 'agent_B' / 'scaler.pkl')
    df_4h = load_btc_4h()
    df_1d = load_btc_1d()
    df_f = load_funding()
    df_fng = load_fng()
    feats = B.build_features(df_4h, df_1d, df_f, df_fng)
    # merge features con OHLC (necesarios para simulate)
    df = df_4h.join(feats).dropna(subset=B.FEATURES)
    oos_i = df.index.searchsorted(OOS_START)
    if oos_i >= len(df):
        return [], OOS_START, df.index[-1]
    sig = lambda d, i: B.signal(d, i, params, (model, scaler))
    sim = lambda d, i: B.simulate(d, i, params)
    trades = run_one_position(df, oos_i, len(df), sig, sim)
    return trades, df.index[oos_i], df.index[-1]


# ---------------------------------------------------------------------------
# AGENTE C
# ---------------------------------------------------------------------------
def verify_C():
    C = load_module('agent_C_strategy', EXP / 'agent_C' / 'strategy.py')
    params = dict(C.PARAMS)
    df_4h = load_btc_4h()
    # Agent C usa `prepare(df_4h)` (calcula daily internamente desde 4h con shift(1))
    df = C.prepare(df_4h)
    oos_i = df.index.searchsorted(OOS_START)
    sig = lambda d, i: C.signal(d, i, params)
    sim = lambda d, i, regime: C.simulate(d, i, params, regime)
    trades = run_one_position(df, oos_i, len(df), sig, sim)
    return trades, df.index[oos_i], df.index[-1]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def report(name, trades, t0, t1):
    m = metrics(trades)
    p = bootstrap_pvalue(trades)
    p_s = f'{p:.3f}' if p is not None else '—'
    print(f'\n=== {name} === (OOS: {t0.date()} -> {t1.date()}, {len(trades)} trades)')
    print(f"  n={m['n']:>3}  WR={m['wr']:.1%}  PF={m['pf']:.2f}  "
          f"Total={m['total']:+.2%}  DD={m['max_dd']:.1%}  Mensual={m['monthly']:+.2%}")
    print(f"  bootstrap p={p_s}")
    # listar trades
    for t in trades:
        ts = pd.to_datetime(t['ts']).strftime('%Y-%m-%d %H:%M')
        rg = f"  ({t['regime']})" if t.get('regime') else ''
        print(f"    {ts}  {t['side']}{rg}  {t['outcome']:>3}  {t['pnl_pct']:+.2%}  ({t['bars']}b)")
    return {'name': name, 'trades': len(trades), **m, 'bootstrap_p': p}


def main():
    results = []
    for name, fn in [('A — Trend (Donchian)', verify_A),
                     ('B — ML GBM classifier', verify_B),
                     ('C — Regime adaptive', verify_C)]:
        try:
            trades, t0, t1 = fn()
            results.append(report(name, trades, t0, t1))
        except Exception as e:
            print(f'\n=== {name} ===  ERROR: {e}')
            import traceback
            traceback.print_exc()
    print('\n' + '=' * 70)
    print('Resumen 2026 OOS (datos que NINGÚN agente vio durante el desarrollo):')
    print('=' * 70)
    print(f"{'Agente':<28}{'N':<5}{'WR':<8}{'PF':<8}{'Mens.':<10}{'DD':<8}{'p-val'}")
    for r in results:
        if r['n'] == 0:
            print(f"{r['name']:<28}{'0':<5}{'—':<8}{'—':<8}{'—':<10}{'—':<8}{'—'}")
            continue
        p_s = f"{r['bootstrap_p']:.3f}" if r.get('bootstrap_p') is not None else '—'
        print(f"{r['name']:<28}{r['n']:<5}{r['wr']:<8.1%}{r['pf']:<8.2f}"
              f"{r['monthly']:<+10.2%}{r['max_dd']:<8.1%}{p_s}")


if __name__ == '__main__':
    main()
