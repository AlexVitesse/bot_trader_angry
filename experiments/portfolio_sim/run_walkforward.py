"""
Walk-forward REAL de selección de pares
=======================================
La revisión externa señaló que los "walk-forward" previos no lo eran: elegían
el parámetro sobre toda la historia y luego lo miraban por particiones de esa
misma historia. Aquí no.

Diseño:
  - La línea temporal se corta en folds de test consecutivos.
  - Para el fold k, TRAIN = todo lo anterior al inicio del fold. TEST = el fold.
  - En TRAIN se decide qué pares entran, con una REGLA DECLARADA DE ANTEMANO
    (>= MIN_TRADES operaciones y PF >= MIN_PF). Nada más se ajusta: PARAMS_V2
    sigue frozen.
  - En TEST se corre el simulador de cartera solo con los pares elegidos.
  - Los folds de test son independientes (cada uno arranca con el mismo
    capital) para que la suerte de uno no contamine al siguiente.

El warmup se preserva: las features se construyen una sola vez sobre la
historia completa y solo se restringe el rango de fechas evaluado, así que el
motor no vuelve a descartar 220 velas en cada ventana.

Baselines para comparar: "siempre los 5 pares" y "siempre solo BTC".

Uso: python experiments/portfolio_sim/run_walkforward.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from portfolio_sim import PortfolioSim, cargar_pares
from src import v2_engine as v2

CINCO = ['BTC/USDT', 'BNB/USDT', 'DOGE/USDT', 'ETH/USDT', 'OP/USDT']
PARAMS = {**v2.PARAMS_V2, 'f_enable_short': False}
RISK = 0.02

# --- REGLA DE SELECCIÓN, declarada antes de ver ningún resultado -------------
MIN_TRADES = 15     # muestra mínima en train para opinar del par
MIN_PF = 1.20       # umbral de edge en train
# ----------------------------------------------------------------------------

N_FOLDS = 6
TRAIN_MINIMO_DIAS = 730     # 2 años antes del primer fold de test


def pf_en_train(feats, desde, hasta, params) -> tuple[int, float]:
    """PF y nº de trades del par en la ventana de train. Rápido: single-asset."""
    idx = feats.index
    start_i = max(int((idx < desde).sum()), params['min_warmup_bars'])
    end_i = int((idx < hasta).sum())
    if end_i - start_i < 50:
        return 0, 0.0
    tr = v2.run_v2_backtest(feats, None, None, params,
                            start_i=start_i, end_i=end_i)
    if len(tr) < 1:
        return 0, 0.0
    pnl = np.array([t['pnl_pct'] for t in tr])
    w, l = pnl[pnl > 0].sum(), abs(pnl[pnl <= 0].sum())
    return len(pnl), (float(w / l) if l else 99.0)


def main() -> None:
    print("Cargando features (una sola vez, historia completa)...")
    datos = cargar_pares(CINCO, PARAMS)

    # rango común de fechas
    ini = min(d.index[0] for d in datos.values())
    fin = max(d.index[-1] for d in datos.values())
    primer_test = ini + pd.Timedelta(days=TRAIN_MINIMO_DIAS)
    bordes = pd.date_range(primer_test, fin, periods=N_FOLDS + 1)
    print(f"Datos {ini.date()} -> {fin.date()} | "
          f"train inicial {TRAIN_MINIMO_DIAS}d | {N_FOLDS} folds de test")
    print(f"Regla declarada: par entra si en TRAIN tiene >={MIN_TRADES} trades "
          f"y PF >={MIN_PF}\n")

    resultados = {'seleccion_wf': [], 'siempre_5': [], 'solo_btc': []}

    for k in range(N_FOLDS):
        t0, t1 = bordes[k], bordes[k + 1]

        # ---- decisión tomada SOLO con train (todo lo anterior a t0) ----
        elegidos, detalle = [], []
        for par, feats in datos.items():
            n, pf = pf_en_train(feats, ini, t0, PARAMS)
            ok = n >= MIN_TRADES and pf >= MIN_PF
            if ok:
                elegidos.append(par)
            detalle.append(f"{par.split('/')[0]}:{n}/{pf:.2f}{'+' if ok else '-'}")

        # ---- evaluación en TEST, sin tocar nada más ----
        def correr(pares):
            if not pares:
                return 0.0
            sub = {p: datos[p] for p in pares}
            m = PortfolioSim(sub, params=PARAMS, risk_pct=RISK).run(t0, t1).metricas
            if not m.get('n'):
                return 0.0
            return (m['final'] - 1) * 100

        r_wf = correr(elegidos)
        r_5 = correr(CINCO)
        r_btc = correr(['BTC/USDT'])
        resultados['seleccion_wf'].append(r_wf)
        resultados['siempre_5'].append(r_5)
        resultados['solo_btc'].append(r_btc)

        print(f"  Fold {k+1}: test {t0.date()} -> {t1.date()}")
        print(f"    train: {' '.join(detalle)}")
        print(f"    elegidos: {[p.split('/')[0] for p in elegidos] or 'NINGUNO'}")
        print(f"    TEST  wf={r_wf:+7.1f}%   siempre5={r_5:+7.1f}%   "
              f"soloBTC={r_btc:+7.1f}%")

    print(f"\n{'=' * 70}")
    print("RESULTADO WALK-FORWARD (solo folds de test, nunca vistos al decidir)")
    print(f"{'=' * 70}")
    print(f"  {'estrategia':<18} {'folds +':>9} {'mediana':>9} {'media':>9} "
          f"{'compuesto':>11} {'peor fold':>11}")
    print("  " + "-" * 70)
    for lbl, arr in resultados.items():
        a = np.array(arr)
        comp = (np.prod(1 + a / 100) - 1) * 100
        print(f"  {lbl:<18} {int((a>0).sum())}/{len(a):<7} {np.median(a):>+8.1f}% "
              f"{a.mean():>+8.1f}% {comp:>+10.1f}% {a.min():>+10.1f}%")

    print(f"\n  Regla del proyecto: >=7/12 folds positivos (aquí {N_FOLDS}, "
          f"equivalente >=4/6).")
    print("  Si 'seleccion_wf' no supera a 'solo_btc', la vía multi-par no aporta.")


if __name__ == '__main__':
    main()
