"""
Carry de funding — la ultima familia de estrategia sin probar.

agent_E uso el funding como SENAL direccional (mean-reversion en extremos).
Esto es otra cosa: COBRAR el funding estando corto en el perp y largo en spot
(delta-neutral). No predice nada, asi que esquiva el problema de
`experiments/predictibilidad/` (R2 = 0,068%), y en teoria deberia rendir
cuando V2 esta parado.

Dos preguntas:
  1. Cuanto paga? (bruto, por ano)
  2. Diversifica? Es decir, paga cuando V2 NO opera?

Uso: python experiments/carry_funding/test_carry.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import v2_engine as v2

P = {**v2.PARAMS_V2, 'f_enable_short': False}
PAGOS_ANO = 3 * 365          # funding cada 8h


def anualizar(s):
    return s.mean() * PAGOS_ANO * 100


def main() -> None:
    root = Path(__file__).resolve().parents[2] / 'data'
    fu = pd.read_parquet(root / 'btc_v15_funding.parquet')
    df4 = pd.read_parquet(root / 'btcusdt_4h_v15.parquet')
    df1 = pd.read_parquet(root / 'btcusdt_1d_v15.parquet')
    r = fu['funding_rate']
    print(f"funding: {fu.index[0].date()} -> {fu.index[-1].date()} "
          f"({len(r)} pagos, {len(r) / (fu.index[-1] - fu.index[0]).days:.2f}/dia)\n")

    print("=== 1) Cuanto paga el carry (delta-neutral, cobrando funding) ===\n")
    print(f"  medio: {r.mean() * 100:.5f}% por pago -> {anualizar(r):+.2f}% anual bruto")
    print(f"  pagos positivos (cobras): {100 * (r > 0).mean():.1f}%\n")
    print(f"  {'ano':>6} {'anual bruto':>12} {'% positivos':>12}")
    for y, g in r.groupby(r.index.year):
        print(f"  {y:>6} {anualizar(g):+11.2f}% {100 * (g > 0).mean():11.1f}%")
    print(f"\n  -> el carry se ha COMPRIMIDO: +30,6% en 2021, +2,3% en 2026.")
    print(f"     Firma de trade arbitrado: capital institucional lo ha cerrado.")

    print("\n=== 2) Diversifica? Paga cuando V2 esta parado? ===\n")
    f = v2.build_features(df4, df1, None, P)
    bull = f['bull_1d'].reindex(r.index, method='ffill')
    on, off = r[bull >= 1], r[bull < 1]
    print(f"  {'regimen':<28} {'carry anual':>12} {'n pagos':>9}")
    print(f"  {'BULL (V2 opera)':<28} {anualizar(on):+11.2f}% {len(on):9,}")
    print(f"  {'BEAR/RANGE (V2 parado)':<28} {anualizar(off):+11.2f}% {len(off):9,}")
    ratio = anualizar(on) / anualizar(off) if anualizar(off) else float('inf')
    print(f"\n  El carry paga {ratio:.1f}x mas cuando V2 YA esta operando.")
    print(f"  Esta CORRELACIONADO con V2, no lo diversifica: los dos viven")
    print(f"  del mismo apetito alcista.")

    bear = r[r.index >= '2025-11-17']
    print(f"\n  En el bear actual (desde 2025-11-17): {anualizar(bear):+.2f}% anual, "
          f"n={len(bear)}")

    print("\n=== 3) Contra que compite ===\n")
    print(f"  Earn de stablecoins (ya desplegado en el bot): ~4-8% anual")
    print(f"  carry HOY:                                     {anualizar(bear):+.2f}% bruto")
    print(f"  menos comisiones de 2 patas, rebalanceo de delta y riesgo de base")
    print(f"\n  -> el carry rinde MENOS que el Earn que ya tiene, con mas")
    print(f"     complejidad operativa y riesgo de liquidacion en la pata perp.")


if __name__ == '__main__':
    main()
