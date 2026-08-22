"""
Veto de funding — merece la pena conectarlo en vivo?

El motor lo soporta (`a_funding_z_max=2.5`, `f_funding_z_max_long=2.0`) y hay
6 anos de funding en disco (`data/btc_v15_funding.parquet`), pero en vivo la
llamada es `get_live_signal(..., df_funding=None)`: el veto esta MUERTO desde
siempre. Es el punto 3 de "Parte 7 — Abierto" de docs/SESION_2026-08-09.md.

Antes de tocar produccion, medir cuantas senales habria vetado y con que efecto.

Uso: python experiments/funding_veto/test_funding_veto.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import v2_engine as v2

P = {**v2.PARAMS_V2, 'f_enable_short': False}


def stats(trades, years, label):
    p = np.array([t['pnl_pct'] for t in trades])
    eq = np.cumprod(1 + p)
    dd = 100 * ((eq / np.maximum.accumulate(eq)) - 1).min()
    g, l = p[p > 0].sum(), -p[p < 0].sum()
    print(f"  {label:24s} n={len(trades):4d} WR={100 * (p > 0).mean():5.1f}% "
          f"PF={g / l if l else float('inf'):5.2f} "
          f"anual={100 * (eq[-1] ** (1 / years) - 1):+6.1f}% DD={dd:6.1f}%")


def main() -> None:
    root = Path(__file__).resolve().parents[2] / 'data'
    df4 = pd.read_parquet(root / 'btcusdt_4h_v15.parquet')
    df1 = pd.read_parquet(root / 'btcusdt_1d_v15.parquet')
    fu = pd.read_parquet(root / 'btc_v15_funding.parquet')
    years = (df4.index[-1] - df4.index[0]).days / 365.25
    print(f"funding en disco: {fu.index[0].date()} -> {fu.index[-1].date()} "
          f"({len(fu)} filas)\n")

    f_con = v2.build_features(df4, df1, fu, P)
    z = f_con['funding_z']
    print(f"funding_z: media {z.mean():+.3f} std {z.std():.3f} "
          f"min {z.min():+.2f} max {z.max():+.2f}")
    print(f"  velas con z > {P['a_funding_z_max']} (veta A): "
          f"{(z > P['a_funding_z_max']).sum():4d} "
          f"({100 * (z > P['a_funding_z_max']).mean():.2f}%)")
    print(f"  velas con z > {P['f_funding_z_max_long']} (veta F): "
          f"{(z > P['f_funding_z_max_long']).sum():4d} "
          f"({100 * (z > P['f_funding_z_max_long']).mean():.2f}%)")

    print("\n=== V2 con vs sin veto ===")
    sin = v2.run_v2_backtest(df4, df1, None, P)
    stats(sin, years, 'SIN funding (actual)')
    stats(v2.run_v2_backtest(df4, df1, fu, P), years, 'CON veto de funding')

    bloq = []
    for t in sin:
        ts = pd.Timestamp(t['ts_entry'])
        if ts not in f_con.index:
            continue
        zz = f_con.loc[ts, 'funding_z']
        lim = (P['a_funding_z_max'] if t['sig_type'] == 'A_LONG'
               else P['f_funding_z_max_long'])
        if pd.notna(zz) and zz > lim:
            bloq.append((t, zz))

    print(f"\n=== Trades que el veto habria bloqueado ===")
    print(f"  {len(bloq)} de {len(sin)} en {years:.1f} anos "
          f"= {len(bloq) / years:.1f} al ano")
    if not bloq:
        return
    bp = np.array([t['pnl_pct'] for t, _ in bloq]) * 100
    print(f"  pnl: suma {bp.sum():+.2f}%  media {bp.mean():+.2f}%  "
          f"({(bp > 0).sum()} ganan / {(bp < 0).sum()} pierden)")
    for t, zz in bloq:
        print(f"    {t['ts_entry'][:16]}  {t['sig_type']:8s} z={zz:5.2f}  "
              f"pnl={t['pnl_pct'] * 100:+6.2f}%")
    print(f"\n  {len(bloq)} eventos no admiten significancia estadistica.")
    print(f"  Por el criterio de CLAUDE.md (bootstrap p + tamano de efecto),")
    print(f"  el veto NO califica para adoptarse.")


if __name__ == '__main__':
    main()
