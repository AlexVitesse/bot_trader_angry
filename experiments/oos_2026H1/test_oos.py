"""
OOS 2026-03-05 -> hoy: los datos que PARAMS_V2 nunca vio.

Los .parquet con los que se ajusto V2 terminan el 2026-03-04. Este script baja
datos vivos de Binance y evalua los parametros CONGELADOS sobre la ventana
posterior, que es genuinamente fuera de muestra.

Dos preguntas distintas:
  1. Cuantos trades habria hecho V2?  -> 0 (regimen apagado el 100% del tramo)
     => el test no aporta NINGUNA informacion sobre el edge.
  2. Acerto el filtro de regimen al bloquearlos? -> SI, y esto si es medible:
     se simula el MISMO motor sin el filtro y se mira que habria pasado.

Uso: python experiments/oos_2026H1/test_oos.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import v2_engine as v2

P = {**v2.PARAMS_V2, 'f_enable_short': False}
SYM = 'BTC/USDT:USDT'
CUT = pd.Timestamp('2026-03-05', tz='UTC')   # fin de los .parquet de ajuste


def bajar(ex, tf, lim):
    r = ex.fetch_ohlcv(SYM, tf, limit=lim)
    d = pd.DataFrame(r, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    d['ts'] = pd.to_datetime(d['ts'], unit='ms', utc=True)
    return d.set_index('ts')


def main() -> None:
    import ccxt
    ex = ccxt.binanceusdm({'enableRateLimit': True})
    df4, df1 = bajar(ex, '4h', 1500), bajar(ex, '1d', 700)
    print(f"Datos vivos: {df4.index[0].date()} -> {df4.index[-1].date()}")
    print(f"PARAMS_V2 ajustados con datos hasta 2026-03-04")
    print(f"VENTANA NO VISTA: {CUT.date()} -> {df4.index[-1].date()} "
          f"({(df4.index[-1] - CUT).days} dias)\n")

    # --- 1) Cuantas senales da V2 tal cual ---
    f = v2.build_features(df4, df1, None, P)
    oos = f[f.index >= CUT]
    c = oos['close']
    print(f"=== BTC en la ventana ===")
    print(f"  {c.iloc[0]:,.0f} -> {c.iloc[-1]:,.0f} "
          f"({100 * (c.iloc[-1] / c.iloc[0] - 1):+.1f}%)  "
          f"min {c.min():,.0f}  max {c.max():,.0f}\n")

    n_sig = n_brk = n_bull = 0
    for i in range(P['min_warmup_bars'], len(f)):
        if f.index[i] < CUT:
            continue
        r = f.iloc[i]
        n_bull += int(r['bull_1d'] >= 1)
        n_brk += int(r['close'] > r['donchian_high'])
        n_sig += int(v2.detect_signal(f, i, P, live=True) is not None)
    print(f"=== V2 con parametros congelados ===")
    print(f"  velas 4h:                     {len(oos)}")
    print(f"  con bull_1d encendido:        {n_bull} ({100 * n_bull / len(oos):.0f}%)")
    print(f"  rupturas Donchian-55 brutas:  {n_brk}")
    print(f"  SENALES V2:                   {n_sig}")
    print(f"  -> informacion sobre el edge: "
          f"{'NINGUNA' if n_sig == 0 else 'alguna'}\n")

    # --- 2) Acerto el filtro? Mismo motor, sin filtro de regimen ---
    Pn = {**P, 'a_require_bull': False, 'f_require_regime': False}
    fn = v2.build_features(df4, df1, None, Pn)
    trades, i = [], P['min_warmup_bars']
    while i < len(fn) - 1:
        if fn.index[i] < CUT:
            i += 1
            continue
        s = v2.detect_signal(fn, i, Pn, live=False)
        if s is None:
            i += 1
            continue
        o = v2.simulate_trade(fn, i, Pn, sig_type=s)
        o['sig'] = s
        trades.append(o)
        i += o['bars'] + 1

    print("=== Que habria pasado SIN el filtro de regimen ===")
    if not trades:
        print("  0 trades")
        return
    p = np.array([t['pnl_pct'] for t in trades]) * 100
    eq = np.cumprod(1 + p / 100)
    dd = 100 * ((eq / np.maximum.accumulate(eq)) - 1).min()
    g, l = p[p > 0].sum(), -p[p < 0].sum()
    print(f"  trades: {len(trades)}   WR: {100 * (p > 0).mean():.1f}%   "
          f"PF: {g / l if l else float('inf'):.2f}")
    print(f"  retorno compuesto: {100 * (eq[-1] - 1):+.1f}%   DD: {dd:.1f}%")
    print(f"  mejor {p.max():+.2f}%  peor {p.min():+.2f}%  medio {p.mean():+.2f}%")
    print(f"  por tipo: {dict(Counter(t['sig'] for t in trades))}")
    print(f"\n  El filtro bloqueo estos {len(trades)} trades. Acerto? "
          f"{'SI' if eq[-1] < 1 else 'NO'}")


if __name__ == '__main__':
    main()
