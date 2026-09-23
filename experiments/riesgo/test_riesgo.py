"""Riesgo por trade: CAGR, DD y probabilidad de tocar el kill switch.

Descriptivo (no hay hipotesis que testear): apoya la decision de
ML_RISK_PER_TRADE. V2 = portfolio_sim, BTC, costes nuevos, max_concurrent=1 y
max_notional_pct=2.0 (el tope que usa el bot en vivo, ML_MAX_NOTIONAL_PCT).

Monte Carlo: bootstrap por bloques circulares de 10 trades sobre los r de cada
nivel, horizonte de 3 anos (~70 trades a 23/ano), 20.000 caminos.

Uso: python experiments/riesgo/test_riesgo.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments' / 'portfolio_sim'))
from portfolio_sim import PortfolioSim, cargar_pares  # noqa: E402

RIESGOS = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
KILL = 0.45
N_TRADES_3A = 70
CAMINOS, BLOQUE = 20_000, 10


def dd_max(eq):
    return float((1 - eq / np.maximum.accumulate(eq)).max())


def montecarlo(r, rng):
    n = len(r)
    nb = int(np.ceil(N_TRADES_3A / BLOQUE))
    idx = (rng.integers(0, n, (CAMINOS, nb))[:, :, None] + np.arange(BLOQUE)).reshape(CAMINOS, -1)
    idx = idx[:, :N_TRADES_3A] % n
    eq = np.cumprod(1 + r[idx], axis=1)
    eq = np.hstack([np.ones((CAMINOS, 1)), eq])
    dd = (1 - eq / np.maximum.accumulate(eq, axis=1)).max(axis=1)
    fin = eq[:, -1]
    return {
        'p_kill': float((dd >= KILL).mean()),
        'dd_med': float(np.median(dd)),
        'dd_p95': float(np.percentile(dd, 95)),
        'cagr_p05': float(np.percentile(fin, 5) ** (1 / 3) - 1),
        'cagr_med': float(np.median(fin) ** (1 / 3) - 1),
        'p_perdida': float((fin < 1).mean()),
    }


def main():
    datos = cargar_pares(['BTC/USDT'])
    rng = np.random.default_rng(20260923)
    print(f"{'riesgo':>6} | {'CAGR':>6} {'DD':>6} {'MAR':>5} | 3 anos MC: "
          f"{'P(kill)':>7} {'DDmed':>6} {'DDp95':>6} {'CAGRp05':>8} {'CAGRmed':>8} {'P(perd)':>7}")
    for rk in RIESGOS:
        res = PortfolioSim(datos, risk_pct=rk, max_concurrent=1,
                           max_notional_pct=2.0).run()
        m = res.metricas
        r = np.array([t['r'] for t in res.trades])
        mc = montecarlo(r, rng)
        print(f"{rk:>6.1%} | {m['cagr']:>+5.1f}% {m['dd']:>5.1f}% {m['cagr'] / m['dd']:>5.2f} | "
              f"{'':>10}{mc['p_kill']:>7.1%} {mc['dd_med']:>6.1%} {mc['dd_p95']:>6.1%} "
              f"{mc['cagr_p05']:>+8.1%} {mc['cagr_med']:>+8.1%} {mc['p_perdida']:>7.1%}")


if __name__ == '__main__':
    main()
