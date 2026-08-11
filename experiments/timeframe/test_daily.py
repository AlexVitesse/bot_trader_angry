"""
¿V2 esta corriendo en la temporalidad equivocada?
=================================================
Los PARAMS_V2 (Donchian-55, EMA50/200, ATR x2.5) son los del sistema Turtle,
disenado para velas DIARIAS. V2 los aplica sobre velas de 4h, donde
Donchian-55 = 9 dias en vez de 55, y el hold medio sale de ~2 dias.

Este test corre las MISMAS reglas sobre velas diarias. Cero parametros nuevos:
solo cambia la vela. Si el edge es real y mecanico, deberia aparecer en ambas
temporalidades; si solo existe en 4h, es sospechoso de ajuste.

Uso: python experiments/timeframe/test_daily.py
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
BASE = {**v2.PARAMS_V2, 'f_enable_short': False}
FUNDING_8H = 0.00013


def load():
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
    return df4[~df4.index.duplicated(keep='last')].sort_index()


def resample(df, regla):
    return df.resample(regla).agg({'open': 'first', 'high': 'max', 'low': 'min',
                                   'close': 'last', 'volume': 'sum'}).dropna()


def metricas(trades, years, horas_vela):
    if len(trades) < 5:
        return None
    pnl = np.array([t['pnl_pct'] for t in trades])
    sl = np.array([t['trail_dist'] for t in trades])
    bars = np.array([t['bars'] for t in trades])
    eq = np.cumprod(1 + pnl)
    dd = float((1 - eq / np.maximum.accumulate(eq)).max() * 100)
    w, l = pnl[pnl > 0].sum(), abs(pnl[pnl <= 0].sum())
    means = RNG.choice(pnl, size=(2000, len(pnl)), replace=True).mean(axis=1)
    horas_dentro = bars.sum() * horas_vela
    return {
        'n': len(pnl), 'por_ano': len(pnl) / years,
        'hold_dias': float(bars.mean() * horas_vela / 24),
        'wr': float((pnl > 0).mean() * 100),
        'pf': float(w / l) if l else float('inf'),
        'cagr': float((eq[-1] ** (1 / years) - 1) * 100),
        'dd': dd, 'p': float((means <= 0).mean()),
        'tim': horas_dentro / (years * 365.25 * 24) * 100,
        'funding': FUNDING_8H * (horas_dentro / 8) / years * 100,
        'sl_medio': float(sl.mean() * 100),
    }


def fila(nombre, m):
    if m is None:
        return f"  {nombre:<22} (muestra insuficiente)"
    mk = ' *' if m['p'] < 0.05 else ''
    return (f"  {nombre:<22} {m['por_ano']:>6.0f} {m['hold_dias']:>8.1f}d "
            f"{m['wr']:>6.1f}% {m['pf']:>6.2f} {m['cagr']:>+8.1f}% "
            f"{m['funding']:>7.1f}% {m['cagr']-m['funding']:>+8.1f}% "
            f"{m['dd']:>7.1f}% {m['tim']:>6.1f}% {m['p']:>7.3f}{mk}")


def main():
    df4 = load()
    years = len(df4) * 4 / 24 / 365.25
    df1d = resample(df4, '1D')
    print(f"BTC {df4.index[0].date()} -> {df4.index[-1].date()} ({years:.1f} anos)")
    print(f"4h: {len(df4)} velas | 1d: {len(df1d)} velas\n")

    print(f"  {'config':<22} {'tr/ano':>6} {'hold':>9} {'WR':>7} {'PF':>6} "
          f"{'CAGR':>9} {'funding':>8} {'NETO':>9} {'DD':>8} {'t.merc':>7} {'p':>8}")
    print("  " + "-" * 108)

    resultados = {}

    # --- 4h: como corre hoy (regimen = daily) ---
    tr4 = v2.run_v2_backtest(df4, df1d, None, BASE)
    resultados['4h'] = metricas(tr4, years, 4)
    print(fila('4h (actual)', resultados['4h']))

    # --- 1d: mismas reglas sobre velas diarias, regimen en la misma serie ---
    tr1 = v2.run_v2_backtest(df1d, df1d, None, BASE)
    resultados['1d'] = metricas(tr1, years, 24)
    print(fila('1d (Turtle original)', resultados['1d']))

    # --- 1d con regimen semanal (escala la relacion 6:1 del original) ---
    dfw = resample(df4, '1W')
    trw = v2.run_v2_backtest(df1d, dfw, None, BASE)
    resultados['1d_sem'] = metricas(trw, years, 24)
    print(fila('1d + regimen semanal', resultados['1d_sem']))

    # --- 12h: punto intermedio ---
    df12 = resample(df4, '12h')
    tr12 = v2.run_v2_backtest(df12, df1d, None, BASE)
    resultados['12h'] = metricas(tr12, years, 12)
    print(fila('12h', resultados['12h']))

    # --- combinar 4h + 1d como sleeves independientes -----------------------
    m4, m1 = resultados['4h'], resultados['1d']
    if m4 and m1:
        print(f"\n  Solapamiento: 4h esta dentro {m4['tim']:.0f}% del tiempo, "
              f"1d el {m1['tim']:.0f}%.")
        print(f"  Correr ambos = dos sleeves; el capital ocioso baja de "
              f"{100-m4['tim']:.0f}% a ~{100-min(99, m4['tim']+m1['tim']):.0f}%.")

    print("\n  'NETO' = CAGR - funding.  (* = bootstrap p < 0.05)")
    print("  Mismos PARAMS_V2 en todas las filas: lo unico que cambia es la vela.")


if __name__ == '__main__':
    main()
