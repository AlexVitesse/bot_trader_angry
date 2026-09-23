"""
Bootstrap honesto de V2 — Fase 4 de docs/PLAN_MEJORAS_2026-09.md
=================================================================
El p=0,004 publicado es un bootstrap i.i.d. sobre ~165 trades que se agrupan
en dos regimenes alcistas, y V2 se eligio entre varias variantes sin
corregir (AUDITORIA_2026-09 §1.2, §1.3). Aqui:

  1. p i.i.d. (el de siempre) sobre los trades del simulador de cartera con
     costes nuevos (fill al open t+1, slippage 0,02%, funding historico, gap).
  2. p por BLOQUES sobre esos mismos trades: bloques de k trades
     consecutivos, bloques de calendario y episodios de regimen (bull_1d).
  3. p con NULL SINTETICO: series 4h con los bloques de velas permutados
     (misma distribucion de velas, misma deriva total, clustering de vol
     dentro de cada bloque; se rompe la estructura de regimen a mas plazo
     que el bloque). V2 completo corre en cada serie. p = fraccion de series
     con mean(r) >= observado. Version CON seleccion: en cada serie se corren
     las mismas 6 variantes entre las que se eligio V2 y se toma la mejor.

No cambia parametros. Uso:
    python experiments/bootstrap_bloques/test_bloques.py [n_series]
"""
from __future__ import annotations

import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'portfolio_sim'))

from portfolio_sim import PortfolioSim           # noqa: E402
from src import v2_engine as v2                  # noqa: E402

PAR = 'BTC/USDT'
RISK = 0.02
MIN_TRADES = 10          # variante con menos trades no compite en el max
B_BOOT = 20_000
BLOQUES_VELAS = (42, 180)    # 1 semana, 30 dias

# Las 6 variantes entre las que se eligio V2 (combined_AF: combinaciones de
# componentes; f_short_ablation: F_SHORT on/off). Todas sobre el motor V2.
NO_A = {'a_adx_min': 1e9}                        # A nunca dispara
VARIANTES = {
    'V2 (A + F_LONG)': {},
    'A + F bidir': {'f_enable_short': True},
    'A solo': {'f_enable_long': False},
    'F_LONG solo': dict(NO_A),
    'F bidir solo': {**NO_A, 'f_enable_short': True},
    'A + F_SHORT': {'f_enable_long': False, 'f_enable_short': True},
}
DF = None


def cargar_btc() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / 'data' / 'BTC_USDT_4h_full.parquet')
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df[~df.index.duplicated(keep='last')].sort_index()


def features(df: pd.DataFrame) -> pd.DataFrame:
    d1 = df.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                'close': 'last', 'volume': 'sum'}).dropna()
    return v2.build_features(df, d1, None, v2.PARAMS_V2)


def correr(feat: pd.DataFrame, extra: dict, risk=RISK, costes='nuevos'):
    params = {**v2.PARAMS_V2, **extra}
    return PortfolioSim({PAR: feat}, params=params, risk_pct=risk,
                        costes=costes).run()


def stat(trades) -> tuple[float, float]:
    """(mean(r), t = mean/sd*sqrt(n)). La t es el criterio con el que se eligio
    V2 (menor bootstrap p); mean(r) es el que pide el plan. Se reportan ambos."""
    if len(trades) < MIN_TRADES:
        return -np.inf, -np.inf
    r = np.array([t['r'] for t in trades])
    return float(r.mean()), float(r.mean() / r.std(ddof=1) * np.sqrt(len(r)))


# ---------------------------------------------------------------- sintetico
def sintetica(df: pd.DataFrame, L: int, seed: int) -> pd.DataFrame:
    """Permuta bloques de L velas. Cada vela se guarda relativa (gap, cuerpo,
    mechas) y se reconstruye el precio encadenando: misma distribucion de
    velas y misma deriva total, sin el orden de regimenes real."""
    o, h, l, c = (df[k].to_numpy() for k in ('open', 'high', 'low', 'close'))
    prev_c = np.r_[o[0], c[:-1]]
    rel = np.c_[np.log(o / prev_c), np.log(h / o), np.log(l / o),
                np.log(c / o), df['volume'].to_numpy()]
    bloques = [rel[i:i + L] for i in range(0, len(rel), L)]
    rng = np.random.default_rng(seed)
    rel = np.concatenate([bloques[k] for k in rng.permutation(len(bloques))])
    # log open_t = log open_0 + sum_{s<=t} gap_s + sum_{s<t} cuerpo_s
    lo_ = np.log(o[0]) + np.cumsum(rel[:, 0] + rel[:, 3]) - rel[:, 3]
    return pd.DataFrame({
        'open': np.exp(lo_), 'high': np.exp(lo_ + rel[:, 1]),
        'low': np.exp(lo_ + rel[:, 2]), 'close': np.exp(lo_ + rel[:, 3]),
        'volume': rel[:, 4]}, index=df.index)


def _init(df):
    # Se pasa el DataFrame en vez de leer el parquet en cada worker: leerlo en
    # paralelo dispara una carrera de registro de tipos en pyarrow.
    global DF
    DF = df


def _worker(args):
    L, seed = args
    feat = features(sintetica(DF, L, seed))
    return [stat(correr(feat, extra).trades) for extra in VARIANTES.values()]


# ---------------------------------------------------------------- bootstraps
def p_iid(r, rng):
    mu = rng.choice(r, size=(B_BOOT, len(r))).mean(axis=1)
    return float((mu <= 0).mean())


def p_bloques_trades(r, k, rng):
    """Circular block bootstrap sobre la secuencia de trades."""
    n = len(r)
    nb = int(np.ceil(n / k))
    starts = rng.integers(0, n, size=(B_BOOT, nb))
    idx = (starts[:, :, None] + np.arange(k)) % n
    mu = r[idx.reshape(B_BOOT, -1)[:, :n]].mean(axis=1)
    return float((mu <= 0).mean())


def p_grupos(r, grupos, rng):
    """Remuestrea grupos enteros (bloques de calendario o episodios)."""
    ids = np.unique(grupos)
    sumas = np.array([r[grupos == g].sum() for g in ids])
    cuentas = np.array([(grupos == g).sum() for g in ids])
    pick = rng.integers(0, len(ids), size=(B_BOOT, len(ids)))
    tot, cnt = sumas[pick].sum(axis=1), cuentas[pick].sum(axis=1)
    mu = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
    return float((mu <= 0).mean()), len(ids)


def main():
    n_series = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    rng = np.random.default_rng(7)
    df = cargar_btc()
    feat = features(df)

    # ---- expectativa (una sola tabla) ----
    print('EXPECTATIVA - BTC solo, V2, simulador de cartera')
    print(f"  {'costes':<7} {'risk':>5} {'n':>4} {'WR':>6} {'PF':>5} {'CAGR':>7} "
          f"{'DD':>6} {'p iid':>6}")
    for c in ('viejos', 'nuevos'):
        for rk in (0.02, 0.045):
            m = correr(feat, {}, risk=rk, costes=c).metricas
            print(f"  {c:<7} {rk:>5.1%} {m['n']:>4} {m['wr']:>5.1f}% "
                  f"{m['pf']:>5.2f} {m['cagr']:>+6.1f}% {m['dd']:>5.1f}% "
                  f"{m['p']:>6.3f}")

    # ---- trades observados ----
    res = correr(feat, {})
    tr = sorted(res.trades, key=lambda t: t['ts_entrada'])
    r = np.array([t['r'] for t in tr])
    ent = pd.DatetimeIndex([t['ts_entrada'] for t in tr])
    print(f"\nTRADES OBSERVADOS: n={len(r)} mean(r)={r.mean():+.4%}")

    print('\nP-VALOR SOBRE LOS TRADES (H0: mean(r) <= 0)')
    print(f"  i.i.d.                          p={p_iid(r, rng):.4f}")
    for k in (5, 10, 20):
        print(f"  bloques de {k:>2} trades            p={p_bloques_trades(r, k, rng):.4f}")
    gap_med = float(np.median(np.diff(ent.asi8)) / 86400e9)
    for dias in (max(1, round(gap_med * 5)), 30, 90, 180):
        g = ((ent - ent[0]).days // dias).to_numpy()
        p, nb = p_grupos(r, g, rng)
        print(f"  bloques de {dias:>3} dias ({nb:>3} bl.)   p={p:.4f}")
    bull = feat['bull_1d'].fillna(0).astype(int)
    epis = (bull != bull.shift()).cumsum()
    g = epis.reindex(ent, method='ffill').to_numpy()
    p, nb = p_grupos(r, g, rng)
    print(f"  episodios de regimen ({nb:>3} ep.)  p={p:.4f}   "
          f"(mediana entre trades {gap_med:.1f} dias)")

    # ---- null sintetico ----
    obs = [stat(correr(feat, extra).trades) for extra in VARIANTES.values()]
    print('\nVARIANTES OBSERVADAS:')
    for nombre, (m, t) in zip(VARIANTES, obs):
        print(f"  {nombre:<18} mean(r) {m:+.4%}   t {t:5.2f}")
    guardado = {'obs': np.array(obs)}
    for L in BLOQUES_VELAS:
        t0 = time.time()
        with Pool(initializer=_init, initargs=(df,)) as pool:
            null = np.array(pool.map(_worker, [(L, s) for s in range(n_series)],
                                     chunksize=4))       # (series, variante, stat)
        guardado[f'null_{L}'] = null
        print(f"\nNULL SINTETICO bloques de {L} velas ({L / 6:.0f} dias), "
              f"{n_series} series, {time.time() - t0:.0f}s")
        for j, nombre in enumerate(('mean(r)', 't')):
            v2n, best = null[:, 0, j], null[:, :, j].max(axis=1)
            o = obs[0][j]
            print(f"  [{nombre}] obs {o:+.4f} | mediana null V2 "
                  f"{np.median(v2n[np.isfinite(v2n)]):+.4f} | "
                  f"p sin seleccion {(v2n >= o).mean():.4f} | "
                  f"p con seleccion (max de 6) {(best >= o).mean():.4f}")
        print(f"  series con <{MIN_TRADES} trades en V2: "
              f"{(~np.isfinite(null[:, 0, 0])).sum()}")
        sys.stdout.flush()
    np.savez(HERE / 'resultados.npz', **guardado)


if __name__ == '__main__':
    main()
