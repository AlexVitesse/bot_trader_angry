"""
Calibracion del riesgo por trade y del kill switch
==================================================
El bot NO dimensiona por multiplo de notional, dimensiona por RIESGO:

    notional = (balance * ML_RISK_PER_TRADE) / sl_pct
    retorno sobre equity = pnl_pct * (risk_pct / sl_pct)

Como sl_pct = trail_dist varia por trade (2,5%-6%), el apalancamiento efectivo
tambien varia. Con risk 2% y SL 3% el bot corre a ~0,67x, no a 1x.

Este script barre ML_RISK_PER_TRADE en las unidades REALES del bot y calibra:
  - ML_MAX_DD_PCT       (kill switch)
  - ML_MAX_DAILY_LOSS_PCT (pausa diaria)

El kill switch debe dispararse cuando la estrategia esta ROTA, no cuando tiene
una mala racha normal. Se calibra sobre el percentil 99 de la distribucion de
drawdown obtenida reordenando los trades 3000 veces.

Uso: python experiments/aggressive/test_risk_sizing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import importlib.util
spec = importlib.util.spec_from_file_location(
    'lev', ROOT / 'experiments' / 'aggressive' / 'test_leverage.py')
lev_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lev_mod)

from src import v2_engine as v2

RNG = np.random.default_rng(42)
N_SIM = 3000
RUINA = 0.80
MAX_CONCURRENT = 3          # ML_MAX_CONCURRENT


def curva(ret: np.ndarray) -> tuple[float, bool]:
    """DD maximo de la curva. Segundo valor: True si la cuenta revienta."""
    r = 1 + ret
    if (r <= 0).any():
        return 1.0, True
    # el 1.0 inicial cuenta como primer pico: sin el, una primera operacion
    # perdedora no generaba drawdown ninguno
    eq = np.concatenate(([1.0], np.cumprod(r)))
    return float((1 - eq / np.maximum.accumulate(eq)).max()), False


def main() -> None:
    df4, df1 = lev_mod.load_btc()
    years = len(df4) * 4 / 24 / 365.25
    params = {**v2.PARAMS_V2, 'f_enable_short': False}
    trades = v2.run_v2_backtest(df4, df1, None, params)

    pnl = np.array([t['pnl_pct'] for t in trades])
    sl = np.array([t['trail_dist'] for t in trades])
    peor = float((pnl / sl).min())      # peor trade en multiplos de riesgo

    print(f"V2 sin F_SHORT | {len(pnl)} trades / {years:.1f} anos")
    print(f"SL medio (trail_dist): {sl.mean()*100:.2f}%  "
          f"[{sl.min()*100:.1f}% - {sl.max()*100:.1f}%]")
    print(f"Peor trade: {peor:.2f}x el riesgo definido "
          f"(un stop puede pasarse de largo con gaps)\n")

    print(f"  {'risk':>6} {'lev efect':>10} {'CAGR':>9} {'DD hist':>9} "
          f"{'DD p50':>8} {'DD p95':>8} {'DD p99':>8} {'P(ruina)':>9}")
    print("  " + "-" * 74)

    filas = {}
    for risk in (0.02, 0.03, 0.045, 0.06, 0.08, 0.10):
        ret = pnl * (risk / sl)                     # retorno sobre equity
        lev_ef = float((risk / sl).mean())
        dd_hist, muerto = curva(ret)
        if muerto:
            print(f"  {risk*100:>5.1f}%  LIQUIDADO por un solo trade")
            continue
        eq = float(np.prod(1 + ret))
        cagr = (eq ** (1 / years) - 1) * 100

        sims = np.array([curva(RNG.permutation(ret))[0] for _ in range(N_SIM)])
        filas[risk] = {'cagr': cagr, 'p99': float(np.percentile(sims, 99)),
                       'p95': float(np.percentile(sims, 95)),
                       'lev': lev_ef}
        print(f"  {risk*100:>5.1f}% {lev_ef:>9.2f}x {cagr:>+8.1f}% "
              f"{dd_hist*100:>8.1f}% {np.percentile(sims,50)*100:>7.1f}% "
              f"{np.percentile(sims,95)*100:>7.1f}% "
              f"{np.percentile(sims,99)*100:>7.1f}% "
              f"{(sims>=RUINA).mean()*100:>8.1f}%")

    # --- calibracion para el objetivo declarado: >= 30% anual ---------------
    objetivo = [r for r, m in filas.items() if m['cagr'] >= 30]
    if not objetivo:
        print("\n  Ninguna configuracion alcanza 30% anual.")
        return
    r = min(objetivo)                       # el menor riesgo que llega a 30%
    m = filas[r]

    print(f"\n{'=' * 74}")
    print(f"CALIBRACION PARA >= 30% ANUAL (objetivo declarado en CLAUDE.md)")
    print(f"{'=' * 74}")
    print(f"  ML_RISK_PER_TRADE   = {r:.3f}   ({r*100:.1f}% por trade, "
          f"~{m['lev']:.2f}x efectivo)")
    print(f"  CAGR esperado       = {m['cagr']:+.1f}%")
    print(f"  DD normal (p95)     = {m['p95']*100:.1f}%  <- rachas malas esperables")
    print(f"  DD p99              = {m['p99']*100:.1f}%  <- aqui SI algo va mal")
    print(f"\n  ML_MAX_DD_PCT       = {np.ceil(m['p99']*100/5)*5/100:.2f}   "
          f"(redondeado sobre p99: corta solo si supera el 99% de las rachas)")

    # pausa diaria: peor caso realista = MAX_CONCURRENT stops el mismo dia
    dia_malo = abs(peor) * r * MAX_CONCURRENT
    print(f"  ML_MAX_DAILY_LOSS_PCT = {np.ceil(dia_malo*100/5)*5/100:.2f}   "
          f"({MAX_CONCURRENT} stops simultaneos a {abs(peor):.1f}x riesgo "
          f"= {dia_malo*100:.1f}%)")
    print(f"\n  Con el kill switch actual (20%) el bot se apagaria solo: el DD")
    print(f"  p95 normal a este riesgo ya es {m['p95']*100:.1f}%.")


if __name__ == '__main__':
    main()
