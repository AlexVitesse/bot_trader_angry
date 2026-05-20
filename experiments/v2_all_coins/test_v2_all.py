"""
test_v2_all.py
==============
Aplica V2 (A + F combinado, motor honesto) a CADA moneda del proyecto.
Mismo protocolo que ETH/SOL tests: real + bootstrap + 10 sintéticas + 5 null.

Cutoff inviolable 2025-12-31.

Output: tabla maestra con veredicto KEEP / MARGINAL / REJECT por moneda.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
EXP = ROOT / 'experiments'
CUTOFF = pd.Timestamp('2025-12-31', tz='UTC')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = load_module('A_strat', EXP / 'agent_A' / 'strategy.py')
F = load_module('F_strat', EXP / 'agent_F' / 'strategy.py')


def load_coin(coin):
    """Carga 4h del coin, intentando varios formatos. Slice <=2025-12-31."""
    candidates = [
        DATA / f'{coin}_USDT_4h_full.parquet',
        DATA / f'{coin}USDT_4h_v15.parquet',
        DATA / f'{coin}_USDT_4h_history.parquet',
        DATA / f'{coin}USDT_4h.csv',
    ]
    for f in candidates:
        if not f.exists():
            continue
        if f.suffix == '.parquet':
            df = pd.read_parquet(f).sort_index()
        else:
            df = pd.read_csv(f)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            df = df.sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df = df.loc[:CUTOFF]
        cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in cols if c not in df.columns]
        if missing:
            continue
        return df[cols].copy()
    return None


def run_V2_on(df_4h):
    """Motor V2 = A primero, F segundo, una posicion a la vez."""
    paramsA = dict(A.PARAMS); paramsA['cutoff_date'] = '2099-01-01'
    paramsF = dict(F.PARAMS); paramsF['cutoff_date'] = '2099-01-01'
    try:
        df_A = A.prepare_data(df_4h, None, None, paramsA)
        df_F = F.prepare_data(df_4h, None, None, paramsF)
    except Exception as e:
        return [], f'prep_error: {e}'
    common = df_A.index.intersection(df_F.index)
    if len(common) < 100:
        return [], 'insufficient_after_features'
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
            trades.append({'ts': str(common[i]), 'strat': 'A', 'side': 'LONG',
                           'pnl_pct': float(out.get('pnl_pct', 0)), 'bars': bars})
            i += bars + 1
            continue
        sigF = F.signal(df_F_c, i, paramsF)
        if sigF in ('LONG', 'SHORT'):
            out = F.simulate(df_F_c, i, paramsF, side=sigF)
            bars = int(out.get('bars', 1))
            pnl = out.get('leveraged_pnl_pct', out.get('pnl_pct', 0))
            trades.append({'ts': str(common[i]), 'strat': 'F', 'side': sigF,
                           'pnl_pct': float(pnl), 'bars': bars})
            i += bars + 1
            continue
        i += 1
    return trades, 'ok'


def metrics(trades, years):
    if not trades:
        return {'n': 0, 'wr': 0.0, 'pf': 0.0, 'total': 0.0, 'annual': 0.0, 'max_dd': 0.0}
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
    annual = cum ** (1.0 / max(years, 0.1)) - 1
    return {'n': n, 'wr': wr, 'pf': pf, 'total': cum - 1,
            'annual': annual, 'max_dd': mdd}


def bootstrap_p(trades, n_iter=2000, seed=42):
    if len(trades) < 3:
        return None
    rng = np.random.default_rng(seed)
    pnls = np.array([t['pnl_pct'] for t in trades])
    totals = np.empty(n_iter)
    for j in range(n_iter):
        s = rng.choice(pnls, size=len(pnls), replace=True)
        totals[j] = np.prod(1 + s) - 1
    return float(np.mean(totals <= 0))


def block_bootstrap_ohlcv(df, block_size=24, n_bars=None, seed=None):
    if n_bars is None:
        n_bars = len(df)
    rng = np.random.default_rng(seed)
    max_start = len(df) - block_size
    if max_start < 1:
        return df.copy()
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
    log_ret = np.log(df['close'].values[1:] / df['close'].values[:-1])
    perm = rng.permutation(len(log_ret))
    log_ret_shuf = log_ret[perm]
    new_close = np.empty(len(df))
    new_close[0] = df['close'].iloc[0]
    new_close[1:] = df['close'].iloc[0] * np.exp(np.cumsum(log_ret_shuf))
    ratio_h_c = (df['high'] / df['close']).values[1:][perm]
    ratio_l_c = (df['low'] / df['close']).values[1:][perm]
    ratio_o_c = (df['open'] / df['close']).values[1:][perm]
    vol_shuf = df['volume'].values[1:][perm]
    return pd.DataFrame({
        'open':  np.r_[df['open'].iloc[0], new_close[1:] * ratio_o_c],
        'high':  np.r_[df['high'].iloc[0], new_close[1:] * ratio_h_c],
        'low':   np.r_[df['low'].iloc[0], new_close[1:] * ratio_l_c],
        'close': new_close,
        'volume': np.r_[df['volume'].iloc[0], vol_shuf],
    }, index=df.index)


# ----------------------------------------------------------------------
COINS = ['BTC', 'ETH', 'SOL', 'ADA', 'DOGE', 'LINK', 'XRP', 'AVAX',
         'DOT', 'NEAR', 'ATOM', 'INJ', 'ALGO', 'FIL', '1000SHIB',
         'BNB', 'LTC', 'ETC', 'BCH', 'UNI', 'AAVE', 'OP']

N_SYNTH = 10
N_NULL = 5

results = {}

print(f"Evaluando V2 (A+F) sobre {len(COINS)} monedas...\n")
print(f"{'Coin':<10}{'Status':<14}{'N':<5}{'PF':<8}{'Annual':<10}"
      f"{'DD':<7}{'p':<8}{'Synth':<8}{'Null':<8}{'EvsN':<8}")
print('-' * 100)

for coin in COINS:
    df = load_coin(coin)
    if df is None or len(df) < 500:
        results[coin] = {'status': 'no_data', 'n_bars': len(df) if df is not None else 0}
        print(f"{coin:<10}{'no_data':<14}")
        continue

    years = max(0.1, (df.index[-1] - df.index[0]).days / 365)
    if years < 1.5:
        results[coin] = {'status': 'too_short', 'years': round(years, 2)}
        print(f"{coin:<10}{'too_short':<14}years={years:.1f}")
        continue

    real_trades, status = run_V2_on(df)
    if status != 'ok':
        results[coin] = {'status': status}
        print(f"{coin:<10}{status:<14}")
        continue

    m_real = metrics(real_trades, years)
    p_real = bootstrap_p(real_trades)

    # Synthetic 10 series
    synth_annuals = []
    for seed in range(N_SYNTH):
        df_s = block_bootstrap_ohlcv(df, block_size=24, seed=seed)
        trades_s, st = run_V2_on(df_s)
        if st == 'ok':
            m_s = metrics(trades_s, years)
            synth_annuals.append(m_s['annual'])
    synth_med = float(np.median(synth_annuals)) if synth_annuals else 0
    synth_pos = sum(1 for a in synth_annuals if a > 0)

    # Null 5 seeds
    null_annuals = []
    for seed in range(N_NULL):
        df_n = shuffle_returns(df, seed=seed)
        trades_n, st = run_V2_on(df_n)
        if st == 'ok':
            m_n = metrics(trades_n, years)
            null_annuals.append(m_n['annual'])
    null_med = float(np.median(null_annuals)) if null_annuals else 0
    edge_vs_null = synth_med - null_med

    p_s = f"{p_real:.3f}" if p_real is not None else "NA"
    pf_s = f"{m_real['pf']:.2f}" if np.isfinite(m_real['pf']) else "inf"

    results[coin] = {
        'status': 'tested',
        'years': round(years, 2),
        'data_start': str(df.index[0])[:10],
        'data_end': str(df.index[-1])[:10],
        'n': m_real['n'],
        'wr': m_real['wr'],
        'pf': m_real['pf'] if np.isfinite(m_real['pf']) else None,
        'annual': m_real['annual'],
        'max_dd': m_real['max_dd'],
        'bootstrap_p': p_real,
        'synth_median_annual': synth_med,
        'synth_positive': f"{synth_pos}/{N_SYNTH}",
        'null_median_annual': null_med,
        'edge_vs_null': edge_vs_null,
    }
    print(f"{coin:<10}{'tested':<14}{m_real['n']:<5d}{pf_s:<8s}{m_real['annual']:<+10.2%}"
          f"{m_real['max_dd']:<7.1%}{p_s:<8s}{synth_med:<+8.2%}{null_med:<+8.2%}{edge_vs_null:<+8.2%}")

# Save results
out_path = ROOT / 'experiments' / 'v2_all_coins' / 'results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Final verdict table
print("\n\n" + "=" * 110)
print("VEREDICTO POR MONEDA (V2 = A + F combinado, motor honesto)")
print("=" * 110)
print(f"{'Coin':<10}{'p<0.05?':<10}{'Synth':<8}{'EvsN>5%':<10}{'Veredicto'}")
print('-' * 110)

for coin, r in results.items():
    if r.get('status') != 'tested':
        print(f"{coin:<10}--                                {r.get('status')}")
        continue
    p = r.get('bootstrap_p')
    sig = p is not None and p < 0.05
    synth_pos = int(r['synth_positive'].split('/')[0]) >= 7
    edge_ok = r['edge_vs_null'] > 0.05
    n_criteria = sum([sig, synth_pos, edge_ok])
    if n_criteria == 3:
        verdict = "KEEP (3/3 criterios)"
    elif n_criteria == 2:
        verdict = "MARGINAL (2/3)"
    elif n_criteria == 1:
        verdict = "WEAK (1/3)"
    else:
        verdict = "REJECT (0/3)"
    print(f"{coin:<10}{'SI' if sig else 'NO':<10}{r['synth_positive']:<8}"
          f"{'SI' if edge_ok else 'NO':<10}{verdict}")

print(f"\nResultados completos guardados en: {out_path}")
