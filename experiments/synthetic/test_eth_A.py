"""
test_eth_A.py
=============
¿La estrategia A (Donchian-55 + EMA daily + ATR×2.5 trailing) que funciona
modestamente en BTC, transfiere a ETH? Aplicación honesta.

Protocolo idéntico al test sintético de BTC:
- Cutoff inviolable 2025-12-31
- A's params usados TAL CUAL (cero re-tuning específico para ETH)
- Real ETH 2020-2025 + 20 series sintéticas vía block bootstrap
- Bootstrap p-value sobre el real
- Test C: shuffle aleatorio (null hypothesis)

Veredicto:
- KEEP (añadir al portfolio): bootstrap p<0.05, mediana sintético > 0,
  ayudó en >=14/20 series
- REJECT: cualquier cosa peor
"""
from __future__ import annotations

import importlib.util
import math
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
    # cortar al cutoff
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


def run_A_on(df_4h, A):
    """A's params tal cual — sin re-tuning para ETH."""
    paramsA = dict(A.PARAMS); paramsA['cutoff_date'] = '2099-01-01'
    # A.prepare_data acepta df_1d=None (deriva daily desde 4h con shift(1))
    df = A.prepare_data(df_4h, None, None, paramsA)
    trades = []
    i = 0
    end_i = len(df) - 1
    while i < end_i:
        sig = A.signal(df, i, paramsA)
        if sig == 'LONG':
            out = A.simulate(df, i, paramsA)
            bars = int(out.get('bars', 1))
            trades.append({
                'ts': str(df.index[i]),
                'pnl_pct': float(out.get('pnl_pct', 0.0)),
                'outcome': out.get('outcome'),
                'bars': bars,
            })
            i += bars + 1
        else:
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
    print(f"  {len(df_eth)} bars 4h, {df_eth.index[0]} -> {df_eth.index[-1]}")

    A = load_module('A_strat', EXP / 'agent_A' / 'strategy.py')

    # ===================================================================
    # Real ETH
    # ===================================================================
    print("\n=== Real ETH 2020-2025 (in-sample con params de BTC) ===")
    real_trades = run_A_on(df_eth, A)
    m_real = metrics(real_trades)
    p_real = bootstrap_p(real_trades)
    p_s = f"{p_real:.3f}" if p_real is not None else "—"
    print(f"  N={m_real['n']}  WR={m_real['wr']:.1%}  PF={m_real['pf']:.2f}  "
          f"annual={m_real['annual']:+.1%}  DD={m_real['max_dd']:.1%}  "
          f"bootstrap p={p_s}")

    # ===================================================================
    # 20 sintéticas
    # ===================================================================
    N_SYNTH = 20
    print(f"\n=== {N_SYNTH} series sintéticas ETH (block bootstrap, params BTC sin tunear) ===")
    synth = []
    for seed in range(N_SYNTH):
        df_s = block_bootstrap_ohlcv(df_eth, block_size=24, seed=seed)
        trades = run_A_on(df_s, A)
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
    n_pos_10 = sum(1 for a in annuals if a > 0.10)
    print(f"    # series con annual > 0:    {n_pos}/{N_SYNTH}")
    print(f"    # series con annual > 10%:  {n_pos_10}/{N_SYNTH}")
    inside = np.percentile(annuals, 5) <= m_real['annual'] <= np.percentile(annuals, 95)
    print(f"    Real ETH ({m_real['annual']:+.1%}) está {'dentro' if inside else 'FUERA'} de p5-p95")

    # ===================================================================
    # Null hypothesis
    # ===================================================================
    print("\n=== Null (retornos shuffleados, sin estructura temporal) ===")
    null_annuals = []
    for seed in range(10):
        df_null = shuffle_returns(df_eth, seed=seed)
        trades = run_A_on(df_null, A)
        m = metrics(trades)
        null_annuals.append(m['annual'])
        print(f"  seed {seed}: N={m['n']:3d}  annual={m['annual']:+.1%}")
    print(f"\n  Mediana null: {np.median(null_annuals):+.1%}")
    print(f"  Mediana sintético - Mediana null = "
          f"{np.median(annuals) - np.median(null_annuals):+.1%}  (edge real)")

    # ===================================================================
    # Veredicto
    # ===================================================================
    print("\n" + "=" * 70)
    print("VEREDICTO A-on-ETH")
    print("=" * 70)
    sig_real = p_real is not None and p_real < 0.05
    med_pos = np.median(annuals) > 0
    n_pos_ok = n_pos >= 14
    edge_real = (np.median(annuals) - np.median(null_annuals)) > 0.05
    print(f"  Bootstrap p<0.05 (real ETH):           {'SÍ' if sig_real else 'NO'} (p={p_s})")
    print(f"  Mediana sintético > 0:                 {'SÍ' if med_pos else 'NO'} ({np.median(annuals):+.1%})")
    print(f"  >=14/20 sintéticas positivas:          {'SÍ' if n_pos_ok else 'NO'} ({n_pos}/{N_SYNTH})")
    print(f"  Edge vs null > 5%:                     {'SÍ' if edge_real else 'NO'} ({np.median(annuals)-np.median(null_annuals):+.1%})")

    all_pass = sig_real and med_pos and n_pos_ok and edge_real
    if all_pass:
        print("\n  -> ETH-A APROBADO. Añadir al portfolio junto a V2 BTC.")
    else:
        print("\n  -> ETH-A RECHAZADO. ETH se queda fuera. V2 sigue BTC-only.")


if __name__ == '__main__':
    main()
