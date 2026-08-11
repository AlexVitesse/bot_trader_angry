"""
Bot agresivo: ¿cuanto apalancamiento aguanta V2?
================================================
Agent D concluyo que el leverage no compensa en BTC perp porque "funding ~13%
anual se come cada 1x". Ese calculo asumia estar en mercado el 100% del tiempo.
V2 esta dentro solo el 12,6% -> cada 1x cuesta ~1,8%/ano, no 13%. La conclusion
de D hay que rehacerla.

Pero el limite real del apalancamiento no es el funding: es el DRAWDOWN y la
RUINA. Este test mide ambos, y no se queda en el DD historico (que es UNA
muestra de suerte): reordena los trades 3000 veces para estimar la distribucion
de drawdown y la probabilidad de reventar la cuenta.

Uso: python experiments/aggressive/test_leverage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src import v2_engine as v2

RNG = np.random.default_rng(42)
FUNDING_8H = 0.00013
RUINA = 0.80          # -80% = cuenta practicamente muerta
N_SIM = 3000


def load_btc():
    df = pd.read_parquet(ROOT / 'data' / 'BTC_USDT_4h_full.parquet')
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    r = requests.get('https://fapi.binance.com/fapi/v1/klines',
                     params={'symbol': 'BTCUSDT', 'interval': '4h', 'limit': 1500},
                     timeout=30)
    r.raise_for_status()
    live = pd.DataFrame(r.json(), columns=['ts', 'open', 'high', 'low', 'close',
                                           'volume', 'a', 'b', 'c', 'd', 'e', 'f'])
    live['ts'] = pd.to_datetime(live['ts'], unit='ms', utc=True)
    live = live.set_index('ts')[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df4 = pd.concat([df, live])
    df4 = df4[~df4.index.duplicated(keep='last')].sort_index()
    df1 = df4.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                  'close': 'last', 'volume': 'sum'}).dropna()
    return df4, df1


def max_dd(pnl: np.ndarray, lev: float) -> float:
    """DD maximo de la curva de equity apalancada. -100% = liquidado."""
    r = 1 + lev * pnl
    if (r <= 0).any():                      # un trade se comio la cuenta entera
        return 1.0
    eq = np.cumprod(r)
    return float((1 - eq / np.maximum.accumulate(eq)).max())


def main() -> None:
    df4, df1 = load_btc()
    years = len(df4) * 4 / 24 / 365.25

    # Base: V2 sin F_SHORT (componente sin edge, ver f_short_ablation/).
    # El filtro ADX SE MANTIENE: el walk-forward dio 7/12 con vs 8/12 sin y
    # suma identica (+117.3% vs +117.2%) -> quitarlo no aporta nada, y el
    # parametro validado se queda.
    params = {**v2.PARAMS_V2, 'f_enable_short': False}
    trades = v2.run_v2_backtest(df4, df1, None, params)
    pnl = np.array([t['pnl_pct'] for t in trades])
    bars = sum(t['bars'] for t in trades)

    print(f"Base: V2 sin F_SHORT (ADX intacto) | {len(pnl)} trades / {years:.1f} anos")
    print(f"Peor trade: {pnl.min()*100:+.2f}%  |  mejor: {pnl.max()*100:+.2f}%")
    print(f"Liquidacion teorica de un solo trade a partir de "
          f"{1/abs(pnl.min()):.1f}x\n")

    print(f"  {'lev':>4} {'CAGR bruto':>11} {'funding':>8} {'CAGR neto':>10} "
          f"{'DD hist':>8} {'DD p50':>7} {'DD p95':>7} {'P(ruina)':>9}")
    print("  " + "-" * 74)

    for lev in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
        r = 1 + lev * pnl
        if (r <= 0).any():
            print(f"  {lev:>3.1f}x  LIQUIDADO por un solo trade "
                  f"({pnl.min()*100:.2f}% x {lev:.1f} <= -100%)")
            continue

        eq_final = float(np.prod(r))
        bruto = (eq_final ** (1 / years) - 1) * 100
        funding = FUNDING_8H * (bars / 2) / years * 100 * lev
        dd_hist = max_dd(pnl, lev) * 100

        # reordenar los trades: el DD historico es solo UNA de las secuencias
        # posibles. Nos interesa la cola, no la suerte que tuvimos.
        sims = np.array([max_dd(RNG.permutation(pnl), lev) for _ in range(N_SIM)])
        ruina = float((sims >= RUINA).mean() * 100)

        print(f"  {lev:>3.1f}x {bruto:>+10.1f}% {funding:>7.1f}% "
              f"{bruto - funding:>+9.1f}% {dd_hist:>7.1f}% "
              f"{np.percentile(sims, 50)*100:>6.1f}% "
              f"{np.percentile(sims, 95)*100:>6.1f}% {ruina:>8.1f}%")

    print(f"\n  'DD p95' = en el 5% de los peores reordenamientos, el DD supera esto.")
    print(f"  'P(ruina)' = % de simulaciones con DD >= {RUINA*100:.0f}% (cuenta muerta).")
    print(f"  El kill switch del bot esta en ML_MAX_DD_PCT = 20%.")


if __name__ == '__main__':
    main()
