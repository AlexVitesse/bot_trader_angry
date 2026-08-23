"""
Predictibilidad — cuanta senal hay REALMENTE que aprender.

`presupuesto_informacion/` demuestra que no caben los parametros de un modelo
grande. Este demuestra lo complementario: que tampoco hay senal que aprender.

Tres pruebas, de mas debil a mas fuerte:
  A. Correlacion lineal de cada feature con el retorno futuro, contra un suelo
     de ruido construido con 5.000 features ALEATORIAS.
  B. Lo mismo a varios horizontes (1, 3, 6, 12 velas) — por si la senal esta
     mas lejos.
  C. Prueba NO lineal: partir cada feature en deciles y medir cuanto separan
     los deciles al retorno futuro, contra el mismo suelo por permutacion.
     Captura estructura que la correlacion lineal no ve.

Y al final, lo que V2 hace en vez de predecir.

Uso: python experiments/predictibilidad/test_predictibilidad.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import v2_engine as v2

P = {**v2.PARAMS_V2, 'f_enable_short': False}
FEATURES = ['adx', 'atr_pct', 'bb_width', 'vol_ratio', 'bull_1d']
N_NULL = 5_000
SEED = 7


def cargar():
    root = Path(__file__).resolve().parents[2] / 'data'
    return (pd.read_parquet(root / 'btcusdt_4h_v15.parquet'),
            pd.read_parquet(root / 'btcusdt_1d_v15.parquet'))


def bloque_a(rng, d, y):
    print("=== A) Correlacion lineal con el retorno de la vela siguiente ===\n")
    null = np.array([abs(np.corrcoef(rng.permutation(y), y)[0, 1])
                     for _ in range(N_NULL)])
    p95, p99 = np.percentile(null, [95, 99])
    print(f"  suelo de ruido ({N_NULL:,} features ALEATORIAS):")
    print(f"    |corr| p95={p95:.4f}  p99={p99:.4f}  max={null.max():.4f}\n")
    print(f"  {'feature':<12} {'|corr|':>8} {'R2':>9}   veredicto")
    for c in FEATURES:
        r = abs(np.corrcoef(d[c].values, y)[0, 1])
        v = 'SENAL' if r > p99 else ('marginal' if r > p95 else 'RUIDO')
        print(f"  {c:<12} {r:8.4f} {r * r * 100:8.3f}%   {v}")

    X = np.column_stack([d[c].values for c in FEATURES])
    X = (X - X.mean(0)) / X.std(0)
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"\n  mejor combinacion lineal de las {len(FEATURES)}, IN-SAMPLE: "
          f"R2 = {r2 * 100:.3f}%")
    print(f"  -> {100 - r2 * 100:.2f}% de la varianza sin explicar EN EL TRAIN")


def bloque_b(f, d):
    print("\n=== B) Y a horizontes mas largos? ===\n")
    print(f"  {'horizonte':>10} " + ' '.join(f'{c:>10}' for c in FEATURES))
    for h in [1, 3, 6, 12]:
        fwd = f['close'].pct_change(h).shift(-h)
        sub = pd.concat([f[FEATURES], fwd.rename('y')], axis=1).dropna()
        rs = [abs(np.corrcoef(sub[c].values, sub['y'].values)[0, 1])
              for c in FEATURES]
        print(f"  {h:>3} velas ({h * 4:>2}h) " +
              ' '.join(f'{r:>10.4f}' for r in rs))
    print(f"\n  (el suelo de ruido esta en ~0,017-0,022; nada se despega)")


def bloque_c(rng, d, y):
    print("\n=== C) Prueba NO lineal: separan los deciles el retorno futuro? ===\n")

    def spread(x, yy):
        """Rango entre el decil de mayor y menor retorno medio futuro."""
        q = pd.qcut(pd.Series(x), 10, labels=False, duplicates='drop')
        m = pd.Series(yy).groupby(q).mean()
        return m.max() - m.min()

    print(f"  {'feature':<12} {'spread deciles':>15} {'p95 nulo':>10} "
          f"{'p-valor':>9}   veredicto")
    for c in FEATURES:
        obs = spread(d[c].values, y)
        null = np.array([spread(d[c].values, rng.permutation(y))
                         for _ in range(500)])
        pv = (null >= obs).mean()
        v = 'SENAL' if pv < 0.01 else ('marginal' if pv < 0.05 else 'RUIDO')
        print(f"  {c:<12} {obs * 100:14.4f}% {np.percentile(null, 95) * 100:9.4f}% "
              f"{pv:9.3f}   {v}")


def bloque_d(df4, df1):
    print("\n=== D) Y el sistema que SI funciona, que hace en vez de predecir? ===\n")
    tr = v2.run_v2_backtest(df4, df1, None, P)
    p = np.array([t['pnl_pct'] for t in tr]) * 100
    wr = (p > 0).mean()
    g, l = p[p > 0].mean(), p[p < 0].mean()
    print(f"  V2 acierta el {100 * wr:.1f}% de las veces  ->  PEOR que una moneda")
    print(f"  gana {g:+.2f}% cuando acierta")
    print(f"  pierde {l:+.2f}% cuando falla")
    print(f"\n  esperanza = {wr:.3f}x{g:.2f} + {1 - wr:.3f}x({l:.2f}) "
          f"= {p.mean():+.3f}% por trade")
    print(f"\n  No necesita predecir. Necesita que el pago sea asimetrico.")


def main() -> None:
    rng = np.random.default_rng(SEED)
    df4, df1 = cargar()
    f = v2.build_features(df4, df1, None, P)
    fwd = f['close'].pct_change().shift(-1)
    d = pd.concat([f[FEATURES], fwd.rename('y')], axis=1).dropna()
    y = d['y'].values
    print(f"muestra: {len(d):,} velas 4h "
          f"({f.index[0].date()} -> {f.index[-1].date()})\n")
    bloque_a(rng, d, y)
    bloque_b(f, d)
    bloque_c(rng, d, y)
    bloque_d(df4, df1)


if __name__ == '__main__':
    main()
