"""
Calibracion del criterio de validacion del proyecto.

CLAUDE.md exigia ">= 7/12 folds positivos". Este script mide, por remuestreo,
cuanta capacidad de discriminacion tiene realmente ese criterio:

  - Cuantas veces lo pasa V2, que SI tiene edge (muestra empirica real)?
  - Cuantas veces lo pasa un sistema de alto win rate con edge CERO?

Resultado: el criterio deja pasar el 52.9% de los sistemas sin edge que tienen
WR alto — que es exactamente el perfil de los 5 fracasos historicos del
proyecto (V7, V9, BTC V2, SOL V2, V13.03: WR declarado 63-68%).

Uso: python experiments/criterio_validacion/test_criterio.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import v2_engine as v2

P = {**v2.PARAMS_V2, 'f_enable_short': False}
N_ITER = 20_000
SEED = 42


def folds_positivos(rng, pnl, n_folds, n_iter=N_ITER):
    """Distribucion del numero de folds positivos al remuestrear `pnl`."""
    per = len(pnl) // n_folds
    out = np.empty(n_iter, dtype=int)
    for k in range(n_iter):
        s = rng.choice(pnl, size=per * n_folds, replace=True)
        comp = np.prod(1 + s.reshape(n_folds, per), axis=1) - 1
        out[k] = int((comp > 0).sum())
    return out


def resumen(pnl):
    g, l = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return (f"n={len(pnl)} WR={100 * (pnl > 0).mean():.0f}% "
            f"PF={g / l if l else float('inf'):.2f} "
            f"EV={pnl.mean() * 100:+.3f}%/trade")


def main() -> None:
    rng = np.random.default_rng(SEED)
    root = Path(__file__).resolve().parents[2] / 'data'
    df4 = pd.read_parquet(root / 'btcusdt_4h_v15.parquet')
    df1 = pd.read_parquet(root / 'btcusdt_1d_v15.parquet')
    trades = v2.run_v2_backtest(df4, df1, None, P)
    pnl = np.array([t['pnl_pct'] for t in trades])

    print("=== Sistema CON edge real: V2 (muestra empirica) ===")
    print(f"  {resumen(pnl)} skew={pd.Series(pnl).skew():+.2f}")
    for n_folds, need in [(12, 7), (6, 4), (10, 6)]:
        d = folds_positivos(rng, pnl, n_folds)
        print(f"  {need}/{n_folds} folds+: pasa {100 * (d >= need).mean():5.1f}% "
              f"| mediana {int(np.median(d))}/{n_folds}")

    print("\n=== Sistema SIN edge pero con WR alto (el perfil que fallo 5 veces) ===")
    wr, gain = 0.68, 1.0
    loss = -gain * wr / (1 - wr)          # EV exactamente 0 por construccion
    mr = np.where(rng.random(len(pnl)) < wr, gain, loss) / 100
    print(f"  {resumen(mr)}")
    d = folds_positivos(rng, mr, 12)
    print(f"  7/12 folds+: pasa {100 * (d >= 7).mean():5.1f}% "
          f"| mediana {int(np.median(d))}/12")

    print("\n  Conclusion: contar folds no discrimina. Usar bootstrap p + "
          "tamano de efecto.")


if __name__ == '__main__':
    main()
