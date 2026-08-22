"""
max_bars sweep — septimo negativo del proyecto.

Pregunta: V2 tiene perfil de trend-following (WR 45%, ratio 2.23, skew +1.55).
El trend-following vive de "dejar correr a los ganadores". a_max_bars=60 velas
4h = 10 dias de hold maximo. Esta el reloj cortando a los ganadores?

Respuesta: NO. Solo el 2.3% de los trades muere por TIMEOUT. El trailing ATR
sale mucho antes. Subir max_bars de 60 a 240 velas no mueve las metricas.

Uso: python experiments/max_bars/test_max_bars.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import v2_engine as v2

P = {**v2.PARAMS_V2, 'f_enable_short': False}
FOLDS = [('2019-09', '2021-01'), ('2021-01', '2022-04'), ('2022-04', '2023-07'),
         ('2023-07', '2024-10'), ('2024-10', '2026-03')]


def cargar():
    root = Path(__file__).resolve().parents[2] / 'data'
    return (pd.read_parquet(root / 'btcusdt_4h_v15.parquet'),
            pd.read_parquet(root / 'btcusdt_1d_v15.parquet'))


def stats(trades, years):
    p = np.array([t['pnl_pct'] for t in trades])
    eq = np.cumprod(1 + p)
    dd = ((eq / np.maximum.accumulate(eq)) - 1).min()
    g, l = p[p > 0].sum(), -p[p < 0].sum()
    return dict(n=len(trades), wr=100 * (p > 0).mean(),
                pf=g / l if l else np.inf,
                ann=100 * (eq[-1] ** (1 / years) - 1), dd=100 * dd,
                hold=np.mean([t['bars'] for t in trades]) * 4 / 24)


def main() -> None:
    df4, df1 = cargar()
    years = (df4.index[-1] - df4.index[0]).days / 365.25
    print(f"Muestra: {df4.index[0].date()} -> {df4.index[-1].date()} "
          f"({years:.1f} anos)\n")

    trades = v2.run_v2_backtest(df4, df1, None, P)
    print("=== A) Como salen los trades con la config actual (A=60 / F=40) ===")
    for k, v in Counter(t['outcome'] for t in trades).most_common():
        sub = [t['pnl_pct'] * 100 for t in trades if t['outcome'] == k]
        print(f"  {k:12s} {v:4d} ({100 * v / len(trades):4.1f}%)  "
              f"pnl medio {np.mean(sub):+6.2f}%")
    to = [t for t in trades if t['outcome'] == 'TIMEOUT']
    if to:
        gan = sum(1 for t in to if t['pnl_pct'] > 0)
        print(f"\n  TIMEOUT = cortado por el reloj, no por el mercado.")
        print(f"  {gan}/{len(to)} iban ganando al ser cortados.")

    print("\n=== B) Barrido de max_bars, por fold (no agregado) ===")
    print(f"  {'A/F velas':>10} {'dias':>6} {'n':>4} {'WR':>6} {'PF':>6} "
          f"{'anual':>8} {'DD':>7} {'hold_d':>7}  folds+")
    for mb_a, mb_f in [(30, 20), (60, 40), (90, 60), (120, 80), (180, 120),
                       (240, 160)]:
        pp = {**P, 'a_max_bars': mb_a, 'f_max_bars': mb_f}
        t = v2.run_v2_backtest(df4, df1, None, pp)
        s = stats(t, years)
        fp = []
        for a, b in FOLDS:
            sub = [x for x in t if a <= x['ts_entry'][:7] < b]
            if len(sub) < 3:
                fp.append('.')
                continue
            fp.append('+' if np.prod([1 + x['pnl_pct'] for x in sub]) > 1 else '-')
        mark = '  <- ACTUAL' if mb_a == 60 else ''
        print(f"  {mb_a:4d}/{mb_f:<5d} {mb_a * 4 / 24:5.1f} {s['n']:4d} "
              f"{s['wr']:5.1f}% {s['pf']:6.2f} {s['ann']:+7.1f}% {s['dd']:6.1f}% "
              f"{s['hold']:6.1f}  {''.join(fp)}{mark}")

    print("\n  Conclusion: el reloj no muerde. El trailing ATR resuelve el 97.7%")
    print("  de los trades antes del limite. max_bars NO es una palanca.")


if __name__ == '__main__':
    main()
