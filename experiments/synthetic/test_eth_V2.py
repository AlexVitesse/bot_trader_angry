"""
test_eth_V2.py
==============
ETH-V2 candidato: combinación A (trend) + F (vol breakout) sobre ETH 4h,
mismo motor que BTC V2. Params SIN re-tunear (re-tunear = overfitting demostrado).

Hipótesis: si A y F operan en momentos distintos del ciclo ETH, la combinación
puede tener menor varianza por trade y mejor bootstrap p que cada uno por
separado. Es lo que pasó en BTC: A solo p=0.088, F solo p=0.355, combinado p=0.031.

Si la magia se repite en ETH → ETH-V2 al portfolio (junto a BTC V2).
Si no se repite → ETH se queda fuera definitivamente.

Protocolo:
- Cutoff 2025-12-31
- Engine: una posición a la vez en ETH; A primero, F después
- Real ETH 2020-2025 + 20 sintéticas + 10 null
- Veredicto si pasa los 4 criterios estándar (bootstrap p<0.05, mediana>0,
  ≥14/20 positivas, edge vs null > 5%)
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


def load_eth_4h_real():
    df = pd.read_parquet(DATA / 'ETH_USDT_4h_full.parquet').sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df.loc['2020-01-01':'2025-12-31']
    return df[['open', 'high', 'low', 'close', 'volume']].copy()


def block_bootstrap_ohlcv(df, block_size=24, n_bars=None, seed=None):
    if n_bars is None:
        n_bars = len(df)
    rng = np.random.default_rng(seed)
    max_start = len(df) - block_size
    n_blocks = (n_bars // block_size) + 2
    starts = rng.integers(0, max_start, size=n_blocks)
    parts = []
    last_close = float(df['close'].iloc[0])
    for s in starts:
        block = df.iloc[s:s + block_size].copy()
        scale = last_close / float(block['open'].iloc[0])
        block[['open', 'high', 'low', 'close']] *= scale
        parts.append(block)
        last_close = float(block['close'].iloc[-1])
    out = pd.concat(parts).iloc[:n_bars].copy()
    out.index = pd.date_range(start=df.index[0], periods=len(out),
                              freq='4h', tz='UTC')
    return out


def shuffle_returns(df, seed=None):
    rng = np.random.default_rng(seed)
    df_c = df.copy()
    log_ret = np.log(df_c['close'].values[1:] / df_c['close'].values[:-1])
    perm = rng.permutation(len(log_ret))
    log_ret_shuf = log_ret[perm]
    new_close = np.empty(len(df_c))
    new_close[0] = df_c['close'].iloc[0]
    new_close[1:] = df_c['close'].iloc[0] * np.exp(np.cumsum(log_ret_shuf))
    ratio_h_c = (df_c['high'] / df_c['close']).values[1:][perm]
    ratio_l_c = (df_c['low'] / df_c['close']).values[1:][perm]
    ratio_o_c = (df_c['open'] / df_c['close']).values[1:][perm]
    vol_shuf = df_c['volume'].values[1:][perm]
    out = pd.DataFrame({
        'open':  np.r_[df_c['open'].iloc[0], new_close[1:] * ratio_o_c],
        'high':  np.r_[df_c['high'].iloc[0], new_close[1:] * ratio_h_c],
        'low':   np.r_[df_c['low'].iloc[0], new_close[1:] * ratio_l_c],
        'close': new_close,
        'volume': np.r_[df_c['volume'].iloc[0], vol_shuf],
    }, index=df_c.index)
    return out


def run_V2_on_eth(df_eth_4h, A, F):
    """Motor combinado A+F sobre ETH. A primero, F después. Una posición a la vez."""
    paramsA = dict(A.PARAMS); paramsA['cutoff_date'] = '2099-01-01'
    paramsF = dict(F.PARAMS); paramsF['cutoff_date'] = '2099-01-01'

    df_A = A.prepare_data(df_eth_4h, None, None, paramsA)
    df_F = F.prepare_data(df_eth_4h, None, None, paramsF)
    common = df_A.index.intersection(df_F.index)
    df_A_c = df_A.loc[common]
    df_F_c = df_F.loc[common]

    trades = []
    i = 0
    end_i = len(common) - 1
    while i < end_i:
        sigA = A.signal(df_A_c, i, paramsA)
        if sigA == 'LONG':
            out = A.simulate(df_A_c, i, paramsA)
            bars = int(out.get('bars', 1))
            trades.append({
                'ts': str(common[i]), 'strat': 'A', 'side': 'LONG',
                'pnl_pct': float(out.get('pnl_pct', 0.0)),
                'outcome': out.get('outcome'), 'bars': bars,
            })
            i += bars + 1
            continue
        sigF = F.signal(df_F_c, i, paramsF)
        if sigF in ('LONG', 'SHORT'):
            out = F.simulate(df_F_c, i, paramsF, side=sigF)
            bars = int(out.get('bars', 1))
            pnl = out.get('leveraged_pnl_pct', out.get('pnl_pct', 0.0))
            trades.append({
                'ts': str(common[i]), 'strat': 'F', 'side': sigF,
                'pnl_pct': float(pnl),
                'outcome': out.get('outcome'), 'bars': bars,
            })
            i += bars + 1
            continue
        i += 1
    return trades


def metrics(trades):
    if not trades:
        return {'n': 0, 'wr': 0.0, 'pf': 0.0, 'total': 0.0,
                'annual': 0.0, 'max_dd': 0.0}
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
    annual = cum ** (1.0 / 6.0) - 1.0
    return {'n': n, 'wr': wr, 'pf': pf, 'total': cum - 1.0,
            'annual': annual, 'max_dd': mdd}


def bootstrap_p(trades, n_iter=3000, seed=42):
    if len(trades) < 3:
        return None
    rng = np.random.default_rng(seed)
    pnls = np.array([t['pnl_pct'] for t in trades])
    totals = np.empty(n_iter)
    for j in range(n_iter):
        s = rng.choice(pnls, size=len(pnls), replace=True)
        totals[j] = np.prod(1 + s) - 1
    return float(np.mean(totals <= 0))


def main():
    print("Cargando ETH real 2020-2025...")
    df_eth = load_eth_4h_real()
    print(f"  {len(df_eth)} bars 4h\n")

    A = load_module('A_strat', EXP / 'agent_A' / 'strategy.py')
    F = load_module('F_strat', EXP / 'agent_F' / 'strategy.py')

    # ===================================================================
    # Real ETH — combinado A + F
    # ===================================================================
    print("=== Real ETH 2020-2025 — Combinado A + F (sin re-tuning) ===")
    real_trades = run_V2_on_eth(df_eth, A, F)
    m_real = metrics(real_trades)
    p_real = bootstrap_p(real_trades)
    p_s = f"{p_real:.3f}" if p_real is not None else "-"
    print(f"  N={m_real['n']}  WR={m_real['wr']:.1%}  PF={m_real['pf']:.2f}  "
          f"annual={m_real['annual']:+.1%}  DD={m_real['max_dd']:.1%}  "
          f"bootstrap p={p_s}")

    # Desglose por estrategia
    A_trades = [t for t in real_trades if t['strat'] == 'A']
    F_trades = [t for t in real_trades if t['strat'] == 'F']
    m_A = metrics(A_trades); m_F = metrics(F_trades)
    p_A = bootstrap_p(A_trades) if len(A_trades) >= 3 else None
    p_F = bootstrap_p(F_trades) if len(F_trades) >= 3 else None
    p_A_s = f"{p_A:.3f}" if p_A is not None else "-"
    p_F_s = f"{p_F:.3f}" if p_F is not None else "-"
    print(f"    A_ETH:  N={m_A['n']:3d}  WR={m_A['wr']:.1%}  PF={m_A['pf']:.2f}  "
          f"annual={m_A['annual']:+.1%}  p={p_A_s}")
    print(f"    F_ETH:  N={m_F['n']:3d}  WR={m_F['wr']:.1%}  PF={m_F['pf']:.2f}  "
          f"annual={m_F['annual']:+.1%}  p={p_F_s}")

    # ===================================================================
    # Sintético — 20 series
    # ===================================================================
    N_SYNTH = 20
    print(f"\n=== {N_SYNTH} series sintéticas ETH (combinado A+F) ===")
    synth = []
    for seed in range(N_SYNTH):
        df_s = block_bootstrap_ohlcv(df_eth, block_size=24, seed=seed)
        trades = run_V2_on_eth(df_s, A, F)
        m = metrics(trades)
        synth.append(m)
        print(f"  serie {seed:2d}: N={m['n']:3d}  WR={m['wr']:.1%}  "
              f"PF={m['pf']:.2f}  annual={m['annual']:+.1%}")

    annuals = [r['annual'] for r in synth]
    print(f"\n  Distribución annual:")
    print(f"    mediana = {np.median(annuals):+.1%}")
    print(f"    media   = {np.mean(annuals):+.1%}")
    print(f"    p25-p75 = [{np.percentile(annuals,25):+.1%}, {np.percentile(annuals,75):+.1%}]")
    print(f"    p5-p95  = [{np.percentile(annuals, 5):+.1%}, {np.percentile(annuals,95):+.1%}]")
    n_pos = sum(1 for a in annuals if a > 0)
    print(f"    # series con annual > 0:  {n_pos}/{N_SYNTH}")

    # ===================================================================
    # Null
    # ===================================================================
    print("\n=== Null (shuffleados) ===")
    null_annuals = []
    for seed in range(10):
        df_null = shuffle_returns(df_eth, seed=seed)
        trades = run_V2_on_eth(df_null, A, F)
        m = metrics(trades)
        null_annuals.append(m['annual'])
        print(f"  seed {seed}: N={m['n']:3d}  annual={m['annual']:+.1%}")
    edge_vs_null = np.median(annuals) - np.median(null_annuals)
    print(f"\n  Mediana null:       {np.median(null_annuals):+.1%}")
    print(f"  Mediana sintético:  {np.median(annuals):+.1%}")
    print(f"  Edge vs null:       {edge_vs_null:+.1%}")

    # ===================================================================
    # Veredicto
    # ===================================================================
    print("\n" + "=" * 70)
    print("VEREDICTO ETH-V2 (A + F combinado, sin re-tuning)")
    print("=" * 70)
    sig = p_real is not None and p_real < 0.05
    med_pos = np.median(annuals) > 0
    n_pos_ok = n_pos >= 14
    edge_ok = edge_vs_null > 0.05
    print(f"  Bootstrap p<0.05 (real):       {'SI' if sig else 'NO'} (p={p_s})")
    print(f"  Mediana sintético > 0:         {'SI' if med_pos else 'NO'} ({np.median(annuals):+.1%})")
    print(f"  >=14/20 sintéticas positivas:  {'SI' if n_pos_ok else 'NO'} ({n_pos}/{N_SYNTH})")
    print(f"  Edge vs null > 5%:             {'SI' if edge_ok else 'NO'} ({edge_vs_null:+.1%})")
    all_pass = sig and med_pos and n_pos_ok and edge_ok
    if all_pass:
        print("\n  --> ETH-V2 APROBADO. Anadir al portfolio con BTC V2.")
    else:
        print("\n  --> ETH-V2 RECHAZADO. ETH se queda fuera. Bot opera BTC V2 only.")

    # Comparativa: ¿la combinación supera a A solo?
    print("\nComparativa A solo vs A+F combinado en ETH (in-sample real):")
    if p_A is not None and p_real is not None:
        print(f"  A solo:    p={p_A_s}, annual={m_A['annual']:+.1%}, N={m_A['n']}")
        print(f"  A+F:       p={p_s}, annual={m_real['annual']:+.1%}, N={m_real['n']}")
        if p_real < p_A:
            print(f"  --> La combinación MEJORA p ({p_A_s} -> {p_s})")
        else:
            print(f"  --> La combinación NO mejora p (F_ETH es lastre)")


if __name__ == '__main__':
    main()
