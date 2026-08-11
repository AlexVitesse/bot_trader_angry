"""
Calibracion de riesgo sobre el simulador de CARTERA.
====================================================
Responde la pregunta que la simulacion de un solo activo no podia responder:
que ML_RISK_PER_TRADE es seguro para el sistema REALMENTE desplegado.

Compara tres configuraciones:
  A) la desplegada hoy   — 5 pares, F_SHORT activo, 3 concurrentes
  B) 5 pares sin F_SHORT — la ablacion aplicada
  C) BTC solo sin F_SHORT — lo que asumian los experimentos anteriores

Uso: python experiments/portfolio_sim/run_calibration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from portfolio_sim import PortfolioSim, cargar_pares
from src import v2_engine as v2

CINCO = ['BTC/USDT', 'BNB/USDT', 'DOGE/USDT', 'ETH/USDT', 'OP/USDT']
CON_SHORT = dict(v2.PARAMS_V2)
SIN_SHORT = {**v2.PARAMS_V2, 'f_enable_short': False}
RIESGOS = (0.01, 0.02, 0.03, 0.045, 0.06)


def bloque(titulo, datos, params):
    print(f"\n{titulo}")
    print(f"  {'risk':>6} {'n':>5} {'tr/año':>7} {'WR':>7} {'PF':>6} "
          f"{'CAGR':>9} {'DD':>8} {'x capital':>10} {'boot p':>8}  rechazos")
    print("  " + "-" * 96)
    for risk in RIESGOS:
        res = PortfolioSim(datos, params=params, risk_pct=risk).run()
        m = res.metricas
        if not m.get('n'):
            print(f"  {risk*100:>5.1f}%  (sin trades)")
            continue
        rech = ', '.join(f'{k}={v}' for k, v in sorted(
            res.rechazos.items(), key=lambda x: -x[1])[:2]) or '—'
        print(f"  {risk*100:>5.1f}% {m['n']:>5} {m['por_año']:>7.0f} "
              f"{m['wr']:>6.1f}% {m['pf']:>6.2f} {m['cagr']:>+8.1f}% "
              f"{m['dd']:>7.1f}% {m['final']:>9.2f}x {m['p']:>7.3f}"
              f"{' *' if m['p'] < 0.05 else '  '} {rech}")


def main():
    print("Cargando features por par...")
    d5_con = cargar_pares(CINCO, CON_SHORT)
    d5_sin = cargar_pares(CINCO, SIN_SHORT)
    d1_sin = {k: v for k, v in d5_sin.items() if k == 'BTC/USDT'}
    rango = list(d5_con.values())[0]
    print(f"Pares cargados: {len(d5_con)} | BTC {rango.index[0].date()} -> {rango.index[-1].date()}")
    print("Fill al OPEN de la vela siguiente | max_bars del motor | "
          "equity compartido | margen finito")

    bloque("A) CONFIG DESPLEGADA HOY — 5 pares, F_SHORT activo, 3 concurrentes",
           d5_con, CON_SHORT)
    bloque("B) 5 pares SIN F_SHORT (ablacion aplicada)", d5_sin, SIN_SHORT)
    bloque("C) BTC SOLO sin F_SHORT (lo que asumian los experimentos previos)",
           d1_sin, SIN_SHORT)

    print("\n  '* ' = bootstrap p < 0.05. 'x capital' = multiplo final del capital.")
    print("  DD sobre la curva de equity CON el capital inicial como primer pico.")


if __name__ == '__main__':
    main()
