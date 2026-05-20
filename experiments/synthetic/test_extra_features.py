"""
test_extra_features.py
======================
¿Añadir lo que los traders miran (volumen z-score, fuerza de vela, patrones)
mejora V2? Lo probamos honestamente.

Filtros A PRIORI testeados (no descubiertos en data — todos definidos antes
de ver resultados):

1. vol_zscore (>=1.5):  reemplaza vol_ratio>1.2 por z-score más estricto.
                         Solo entrar en spikes reales de volumen (~p93).

2. body_strong (>=0.6): fuerza del cuerpo de la vela = |C-O|/(H-L).
                         "Vela fuerte", no doji ni mecha larga.

3. close_strong:        close en tercio superior del rango (LONG)
                         o tercio inferior (SHORT). Cierre direccional.

4. engulfing:           patrón bullish/bearish engulfing en la vela actual.
                         Patrón clásico de reversión/confirmación.

5. CONTROL (random 50%): bloquea aleatoriamente 50% de entradas. Debería
                         REDUCIR el retorno por reducir muestra. Validación
                         de que el método discrimina.

Cada filtro se prueba SOLO (no se compone) para aislar su efecto.

Protocolo (mismo que test_learn_from_losses.py):
- Real BTC 2020-2025 in-sample
- 20 series sintéticas vía block bootstrap del BTC real
- Por filtro: delta vs V2 baseline en cada serie, mediana, # ayudó / # empeoró
- Filtro válido si mejora en >=14/20 (70%) y mediana > 0
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


def load_btc_4h_real():
    df = pd.read_parquet(DATA / 'BTC_USDT_4h_full.parquet').sort_index()
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


# =============================================================================
# FILTROS A PRIORI (todos definidos sin mirar el resultado)
# =============================================================================
def make_vol_zscore_filter(threshold=1.5, window=100):
    """Solo entrar si volumen z-score (rolling 100 bars) >= threshold."""
    def f(df, idx):
        if idx < window:
            return True
        v = df['volume'].iloc[idx - window:idx]
        if v.std() == 0:
            return True
        z = (df['volume'].iloc[idx] - v.mean()) / v.std()
        return z >= threshold
    return f


def make_body_strong_filter(min_body_ratio=0.6):
    """Cuerpo de vela >= 60% del rango total. No doji ni mecha dominante."""
    def f(df, idx):
        row = df.iloc[idx]
        rng = row['high'] - row['low']
        if rng <= 0:
            return True
        body = abs(row['close'] - row['open'])
        return (body / rng) >= min_body_ratio
    return f


def make_close_strong_filter(third=0.66, side_aware=True):
    """
    Para LONG: cierre en tercio superior del rango.
    Para SHORT: cierre en tercio inferior.
    side_aware=False: solo aplica para LONG (asume LONG-bias del strategy).
    """
    def f(df, idx, side='LONG'):
        row = df.iloc[idx]
        rng = row['high'] - row['low']
        if rng <= 0:
            return True
        close_pos = (row['close'] - row['low']) / rng
        if side == 'LONG':
            return close_pos >= third
        elif side == 'SHORT':
            return close_pos <= (1 - third)
        return True
    return f


def make_engulfing_filter():
    """
    Bullish engulfing (LONG): vela actual verde (C>O), engulle cuerpo previo
    (O actual <= C previo, C actual >= O previo), cuerpo actual > cuerpo previo.
    Bearish engulfing (SHORT): espejo.
    Si no hay vela previa o falla la condición → BLOQUEA.
    """
    def f(df, idx, side='LONG'):
        if idx < 1:
            return True
        cur = df.iloc[idx]
        prev = df.iloc[idx - 1]
        cur_body = abs(cur['close'] - cur['open'])
        prev_body = abs(prev['close'] - prev['open'])
        if side == 'LONG':
            cur_green = cur['close'] > cur['open']
            prev_red = prev['close'] < prev['open']
            engulf = (cur['open'] <= prev['close']) and (cur['close'] >= prev['open'])
            return cur_green and prev_red and engulf and (cur_body > prev_body)
        elif side == 'SHORT':
            cur_red = cur['close'] < cur['open']
            prev_green = prev['close'] > prev['open']
            engulf = (cur['open'] >= prev['close']) and (cur['close'] <= prev['open'])
            return cur_red and prev_green and engulf and (cur_body > prev_body)
        return True
    return f


def make_random_50_filter(seed=42):
    """CONTROL: bloquea aleatoriamente 50% de entradas. Debería REDUCIR retorno."""
    rng = np.random.default_rng(seed)
    def f(df, idx, side='LONG'):
        return rng.random() >= 0.5
    return f


# =============================================================================
# Engine (mismo motor que test_learn_from_losses.py, con filter side-aware)
# =============================================================================
def run_V2_on(df_synth_4h, A, F, entry_filter=None):
    paramsA = dict(A.PARAMS); paramsA['cutoff_date'] = '2099-01-01'
    paramsF = dict(F.PARAMS); paramsF['cutoff_date'] = '2099-01-01'
    df_A = A.prepare_data(df_synth_4h, None, None, paramsA)
    df_F = F.prepare_data(df_synth_4h, None, None, paramsF)
    common = df_A.index.intersection(df_F.index)
    df_A_c = df_A.loc[common]
    df_F_c = df_F.loc[common]

    trades = []
    i = 0
    end_i = len(common) - 1
    while i < end_i:
        # Evaluar señal primero, después filtro side-aware
        sigA = A.signal(df_A_c, i, paramsA)
        if sigA == 'LONG':
            # filtro evalúa con el df de F (tiene OHLCV + features comunes)
            if entry_filter is not None:
                try:
                    ok = entry_filter(df_F_c, i, side='LONG') if entry_filter.__code__.co_argcount >= 3 else entry_filter(df_F_c, i)
                except TypeError:
                    ok = entry_filter(df_F_c, i)
                if not ok:
                    i += 1
                    continue
            out = A.simulate(df_A_c, i, paramsA)
            bars = int(out.get('bars', 1))
            trades.append({'ts': str(common[i]), 'side': 'LONG', 'strat': 'A',
                           'pnl_pct': float(out.get('pnl_pct', 0.0)),
                           'bars': bars})
            i += bars + 1
            continue

        sigF = F.signal(df_F_c, i, paramsF)
        if sigF in ('LONG', 'SHORT'):
            if entry_filter is not None:
                try:
                    ok = entry_filter(df_F_c, i, side=sigF) if entry_filter.__code__.co_argcount >= 3 else entry_filter(df_F_c, i)
                except TypeError:
                    ok = entry_filter(df_F_c, i)
                if not ok:
                    i += 1
                    continue
            out = F.simulate(df_F_c, i, paramsF, side=sigF)
            bars = int(out.get('bars', 1))
            pnl = out.get('leveraged_pnl_pct', out.get('pnl_pct', 0.0))
            trades.append({'ts': str(common[i]), 'side': sigF, 'strat': 'F',
                           'pnl_pct': float(pnl), 'bars': bars})
            i += bars + 1
            continue

        i += 1
    return trades


def metrics(trades):
    if not trades:
        return {'n': 0, 'wr': 0.0, 'pf': 0.0, 'total': 0.0, 'annual': 0.0}
    n = len(trades)
    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / n
    gw = sum(wins); gl = abs(sum(losses))
    pf = (gw / gl) if gl > 1e-9 else float('inf')
    cum = 1.0
    for t in sorted(trades, key=lambda x: pd.to_datetime(x['ts'])):
        cum *= (1.0 + t['pnl_pct'])
    annual = cum ** (1.0 / 6.0) - 1.0
    return {'n': n, 'wr': wr, 'pf': pf, 'total': cum - 1.0, 'annual': annual}


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("Cargando BTC real 2020-2025...")
    df_real = load_btc_4h_real()
    A = load_module('A_strat', EXP / 'agent_A' / 'strategy.py')
    F = load_module('F_strat', EXP / 'agent_F' / 'strategy.py')

    filters = {
        'BASELINE (V2 sin filtro)': None,
        'vol_zscore >= 1.5':       make_vol_zscore_filter(1.5, 100),
        'body_strong >= 0.6':       make_body_strong_filter(0.6),
        'close_strong (tercio)':    make_close_strong_filter(0.66),
        'engulfing':                make_engulfing_filter(),
        'CONTROL: random 50%':      make_random_50_filter(42),
    }

    # 1) Real BTC
    print(f"\n{'='*88}\nReal BTC 2020-2025 (in-sample)\n{'='*88}")
    real_results = {}
    for name, filt in filters.items():
        trades = run_V2_on(df_real, A, F, entry_filter=filt)
        m = metrics(trades)
        real_results[name] = m
        print(f"  {name:<32}  N={m['n']:>4}  WR={m['wr']:.1%}  "
              f"PF={m['pf']:.2f}  annual={m['annual']:+.1%}")

    base_real = real_results['BASELINE (V2 sin filtro)']

    # 2) 20 sintéticas
    N_SYNTH = 20
    print(f"\n{'='*88}\n20 series sintéticas (block bootstrap) - delta vs BASELINE\n{'='*88}")

    # Pre-calcular todas las series una vez
    synth_series = []
    for seed in range(N_SYNTH):
        synth_series.append(block_bootstrap_ohlcv(df_real, block_size=24, seed=seed))

    synth_results = {name: [] for name in filters}
    for s_idx, df_s in enumerate(synth_series):
        sys.stdout.write(f"\r  Procesando serie {s_idx + 1}/{N_SYNTH}...")
        sys.stdout.flush()
        for name, filt in filters.items():
            trades = run_V2_on(df_s, A, F, entry_filter=filt)
            synth_results[name].append(metrics(trades))
    print()

    base_synth = synth_results['BASELINE (V2 sin filtro)']

    # 3) Comparación
    print(f"\n{'='*88}\nResumen — ¿ayuda cada filtro? (mediana annual delta sobre 20 series)\n{'='*88}\n")
    print(f"{'Filtro':<32}  {'Real dif':<10}  {'Synth med dif':<14}  "
          f"{'Ayudó':<8}  {'Empeoró':<10}  {'Veredicto'}")
    print("-" * 100)
    for name in filters:
        if name.startswith('BASELINE'):
            continue
        # Delta real
        delta_real = real_results[name]['annual'] - base_real['annual']
        # Deltas sintético
        deltas = [synth_results[name][i]['annual'] - base_synth[i]['annual']
                  for i in range(N_SYNTH)]
        med = np.median(deltas)
        n_better = sum(1 for d in deltas if d > 0)
        n_worse = N_SYNTH - n_better
        # Veredicto a priori: filtro válido si median >0 y >=14/20 mejoraron
        if med > 0 and n_better >= 14:
            verdict = '[OK] AÑADIR'
        elif med < -0.01 and n_worse >= 14:
            verdict = '[X]  DESCARTAR'
        else:
            verdict = '[~]  Sin señal'
        print(f"{name:<32}  {delta_real:+7.1%}    {med:+7.2%}        "
              f"{n_better:>2}/{N_SYNTH}    {n_worse:>2}/{N_SYNTH}      {verdict}")

    # 4) Tabla con N (trade count) para ver cuánto reduce muestra cada filtro
    print(f"\n{'='*88}\nN trades por filtro (real + mediana sintético)\n{'='*88}")
    print(f"{'Filtro':<32}  {'Real N':<10}  {'Synth N (med)':<14}  {'% del baseline'}")
    print("-" * 80)
    base_n_synth_med = int(np.median([m['n'] for m in base_synth]))
    for name in filters:
        n_real = real_results[name]['n']
        ns_synth = [m['n'] for m in synth_results[name]]
        n_synth_med = int(np.median(ns_synth))
        pct = (n_real / base_real['n'] * 100) if base_real['n'] else 0
        print(f"{name:<32}  {n_real:>5}      {n_synth_med:>5}           {pct:>5.0f}%")


if __name__ == '__main__':
    main()
