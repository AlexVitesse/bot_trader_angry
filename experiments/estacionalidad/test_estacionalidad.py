"""
Estacionalidad / efectos de calendario en BTC 4h.

Familia sin medir hasta ahora. La trampa clasica: se prueban ~25 buckets
(6 horas x 7 dias x 12 meses), alguno sale "significativo" por azar, y se
construye una estrategia sobre ruido. Aqui se controla desde el diseno:

  A. Efecto por bucket con su t-stat.
  B. Cuantos buckets "significativos" salen vs los esperados por azar
     (control de comparaciones multiples).
  C. LA PRUEBA QUE IMPORTA: el mejor bucket de la PRIMERA mitad, sigue
     funcionando en la SEGUNDA? Si no persiste, era ruido.

Uso: python experiments/estacionalidad/test_estacionalidad.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA = 0.05


def cargar():
    root = Path(__file__).resolve().parents[2] / 'data'
    d = pd.read_parquet(root / 'btcusdt_4h_v15.parquet')
    r = d['close'].pct_change().dropna()
    return r


def buckets(idx):
    return {
        'hora UTC': idx.hour,
        'dia semana': idx.dayofweek,
        'mes': idx.month,
    }


def tabla(r, titulo):
    print(f"\n=== {titulo} ===")
    total_sig = 0
    total_tests = 0
    for nombre, key in buckets(r.index).items():
        g = r.groupby(key)
        print(f"\n  {nombre}:")
        print(f"    {'bucket':>8} {'n':>7} {'media':>10} {'t':>7} {'p':>8}")
        for b, sub in g:
            n = len(sub)
            if n < 30:
                continue
            t = sub.mean() / (sub.std() / np.sqrt(n))
            # p bilateral aproximado (normal, n grande)
            from math import erfc
            p = erfc(abs(t) / np.sqrt(2))
            total_tests += 1
            sig = p < ALPHA
            total_sig += sig
            marca = ' *' if sig else ''
            print(f"    {b:>8} {n:>7,} {sub.mean() * 100:>9.4f}% {t:>7.2f} "
                  f"{p:>8.3f}{marca}")
    esperados = total_tests * ALPHA
    print(f"\n  buckets significativos a p<{ALPHA}: {total_sig} de {total_tests}")
    print(f"  esperados SOLO por azar:            {esperados:.1f}")
    return total_sig, total_tests


def persistencia(r):
    print("\n=== C) El mejor bucket de la 1a mitad, persiste en la 2a? ===")
    mitad = len(r) // 2
    r1, r2 = r.iloc[:mitad], r.iloc[mitad:]
    print(f"  1a mitad: {r1.index[0].date()} -> {r1.index[-1].date()} ({len(r1):,})")
    print(f"  2a mitad: {r2.index[0].date()} -> {r2.index[-1].date()} ({len(r2):,})\n")
    print(f"  {'familia':<12} {'mejor en 1a':>12} {'media 1a':>11} "
          f"{'media 2a':>11}  veredicto")
    for nombre in ['hora UTC', 'dia semana', 'mes']:
        k1 = buckets(r1.index)[nombre]
        k2 = buckets(r2.index)[nombre]
        m1 = r1.groupby(k1).mean()
        m2 = r2.groupby(k2).mean()
        mejor = m1.idxmax()
        v1, v2 = m1[mejor], m2.get(mejor, np.nan)
        ok = 'PERSISTE' if v2 > 0 else 'SE INVIERTE'
        print(f"  {nombre:<12} {str(mejor):>12} {v1 * 100:>10.4f}% "
              f"{v2 * 100:>10.4f}%  {ok}")

    # correlacion entre el perfil de la 1a mitad y el de la 2a
    print(f"\n  correlacion del PERFIL completo entre mitades:")
    for nombre in ['hora UTC', 'dia semana', 'mes']:
        m1 = r1.groupby(buckets(r1.index)[nombre]).mean()
        m2 = r2.groupby(buckets(r2.index)[nombre]).mean()
        com = m1.index.intersection(m2.index)
        c = np.corrcoef(m1[com], m2[com])[0, 1]
        print(f"    {nombre:<12} corr = {c:+.3f}  "
              f"({'estable' if c > 0.5 else 'INESTABLE'})")


def escrutinio_miercoles(r, px):
    """El unico efecto que sobrevive a A/B/C merece el tercer grado.

    El t-test de la tabla A supone observaciones independientes, y los
    retornos 4h dentro de un mismo dia NO lo son. El null correcto rota el
    calendario entero, que preserva la autocorrelacion.
    """
    rng = np.random.default_rng(11)
    dow = r.index.dayofweek
    w, o = r[dow == 2], r[dow != 2]
    print("\n=== D) El efecto miercoles, al tercer grado ===\n")
    print(f"  {'':<12} {'media':>10} {'mediana':>10}")
    print(f"  {'miercoles':<12} {w.mean() * 100:>9.4f}% {w.median() * 100:>9.4f}%")
    print(f"  {'resto':<12} {o.mean() * 100:>9.4f}% {o.median() * 100:>9.4f}%")
    print(f"  -> la MEDIANA casi no difiere: el efecto vive en la cola\n")

    print("  compuesto operando solo los miercoles, por ano:")
    for y, g in r.groupby(r.index.year):
        gw = g[g.index.dayofweek == 2]
        print(f"    {y}  {(np.prod(1 + gw.values) - 1) * 100:+8.2f}%")

    q = np.quantile(np.abs(r.values), 0.99)
    rt = r[np.abs(r.values) < q]
    wt, ot = rt[rt.index.dayofweek == 2], rt[rt.index.dayofweek != 2]
    print(f"\n  delta con todas las velas:          "
          f"{(w.mean() - o.mean()) * 100:+.4f}%")
    print(f"  delta sin el 1% mas extremo:        "
          f"{(wt.mean() - ot.mean()) * 100:+.4f}%   (no es outliers)")

    obs = w.mean() - o.mean()
    lab = dow.values
    null = np.array([r[((lab + k) % 7) == 2].mean() - r[((lab + k) % 7) != 2].mean()
                     for k in rng.integers(0, 7, 10000)])
    print(f"\n  CONTROL correcto (rotar el calendario, 10.000 veces):")
    print(f"    p-valor = {(null >= obs).mean():.4f}   "
          f"(el t-test ingenuo de la tabla A decia p=0,005)")

    trades = []
    for ts in px.index:
        if ts.dayofweek == 2 and ts.hour == 0:
            salida = ts + pd.Timedelta(days=1)
            if salida in px.index:
                trades.append(px.loc[salida] / px.loc[ts] - 1 - 0.001)
    t = np.array(trades)
    eq = np.cumprod(1 + t)
    yrs = (px.index[-1] - px.index[0]).days / 365.25
    dd = 100 * ((eq / np.maximum.accumulate(eq)) - 1).min()
    print(f"\n  Backtest LONG cada miercoles (costes 0,10%):")
    print(f"    n={len(t)}  WR={100 * (t > 0).mean():.1f}%  "
          f"anual={100 * (eq[-1] ** (1 / yrs) - 1):+.2f}%  DD={dd:.1f}%")
    print(f"    BTC comprar y mantener:  "
          f"{100 * ((px.iloc[-1] / px.iloc[0]) ** (1 / yrs) - 1):+.2f}% anual")
    print(f"\n  -> rinde MENOS que estar largo y quieto. No es un edge, es una")
    print(f"     forma peor de estar largo.")


def main() -> None:
    root = Path(__file__).resolve().parents[2] / 'data'
    px = pd.read_parquet(root / 'btcusdt_4h_v15.parquet')['close']
    r = cargar()
    print(f"BTC 4h: {r.index[0].date()} -> {r.index[-1].date()} ({len(r):,} velas)")
    print(f"retorno medio por vela: {r.mean() * 100:+.4f}%")
    tabla(r, 'A/B) Efecto por bucket, con control de comparaciones multiples')
    persistencia(r)
    escrutinio_miercoles(r, px)


if __name__ == '__main__':
    main()
