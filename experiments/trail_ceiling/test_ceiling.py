"""
¿El techo del trailing esta estrangulando los trades de alta volatilidad?
=========================================================================
Hipotesis (de experiments/conviction/): el trail es

    trail = clip(atr_pct * mult, floor, ceiling)     A: 2.5, 2.5%, 6.0%
                                                     F: 2.0, 2.0%, 5.5%

Cuando atr_pct * mult supera el techo, el stop queda proporcionalmente MAS
apretado de lo que la propia volatilidad del activo exige -> salta antes de
tiempo. Eso explicaria el unico rho estable que aparecio (atr_pct vs pnl,
-0.15 en ambas mitades).

Estructura del test:
  1. VERIFICAR EL MECANISMO — comparar trades con techo activo vs sin el.
     Esto es mas fuerte que buscar un numero mejor: o el mecanismo existe o no.
  2. Barrido del techo.
  3. Walk-forward del ganador. Sin esto no se adopta nada (leccion del ADX).

Uso: python experiments/trail_ceiling/test_ceiling.py
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
    return df4, daily(df4)


def daily(df4):
    return df4.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                   'close': 'last', 'volume': 'sum'}).dropna()


def metricas(trades, years):
    if len(trades) < 5:
        return None
    pnl = np.array([t['pnl_pct'] for t in trades])
    eq = np.cumprod(1 + pnl)
    dd = float((1 - eq / np.maximum.accumulate(eq)).max() * 100)
    w, l = pnl[pnl > 0].sum(), abs(pnl[pnl <= 0].sum())
    means = RNG.choice(pnl, size=(2000, len(pnl)), replace=True).mean(axis=1)
    return {'n': len(pnl), 'wr': float((pnl > 0).mean() * 100),
            'pf': float(w / l) if l else float('inf'),
            'cagr': float((eq[-1] ** (1 / years) - 1) * 100), 'dd': dd,
            'p': float((means <= 0).mean())}


# ---------------------------------------------------------------------------
def parte1_mecanismo(df4, df1):
    print("=" * 78)
    print("1. ¿EL TECHO ESTA ACTIVO, Y HACE DANO?")
    print("=" * 78)
    feats = v2.build_features(df4, df1, None, BASE)
    trades = v2.run_v2_backtest(df4, df1, None, BASE)

    filas = []
    for t in trades:
        atr = float(feats['atr_pct'].iloc[t['idx_entry']])
        if t['sig_type'] == 'A_LONG':
            mult, floor, ceil = 2.5, BASE['a_trail_floor_pct'], BASE['a_trail_ceiling_pct']
        else:
            mult, floor, ceil = 2.0, BASE['f_trail_floor_pct'], BASE['f_trail_ceiling_pct']
        deseado = atr * mult
        filas.append({'pnl': t['pnl_pct'], 'atr': atr, 'deseado': deseado,
                      'aplicado': min(max(deseado, floor), ceil),
                      'estado': 'TECHO' if deseado > ceil else
                                ('SUELO' if deseado < floor else 'libre')})
    tab = pd.DataFrame(filas)

    print(f"\n  {'estado del trail':<18} {'n':>4} {'%':>6} {'WR':>7} "
          f"{'PnL medio':>11} {'PF':>7}  {'estrangulamiento':>18}")
    print("  " + "-" * 78)
    for est in ['SUELO', 'libre', 'TECHO']:
        s = tab[tab.estado == est]
        if not len(s):
            continue
        w, l = s.pnl[s.pnl > 0].sum(), abs(s.pnl[s.pnl <= 0].sum())
        # cuanto mas apretado queda el stop respecto a lo que pedia el ATR
        aprieto = (1 - s.aplicado / s.deseado).mean() * 100
        print(f"  {est:<18} {len(s):>4} {len(s)/len(tab)*100:>5.0f}% "
              f"{(s.pnl>0).mean()*100:>6.1f}% {s.pnl.mean()*100:>+10.2f}% "
              f"{w/l if l else float('inf'):>7.2f}  {aprieto:>17.1f}%")
    print("\n  'estrangulamiento' = cuanto se recorta el trail respecto al que")
    print("  pedia el ATR. Positivo en TECHO (se aprieta), negativo en SUELO.")


# ---------------------------------------------------------------------------
def parte2_barrido(df4, df1, years):
    print("\n" + "=" * 78)
    print("2. BARRIDO DEL TECHO (A y F a la vez, manteniendo la proporcion)")
    print("=" * 78)
    print(f"\n  {'techo A / F':<16} {'n':>4} {'WR':>7} {'PF':>7} {'CAGR':>9} "
          f"{'DD':>8} {'boot p':>8}")
    print("  " + "-" * 62)
    resultados = {}
    for a_c, f_c in [(0.06, 0.055), (0.08, 0.073), (0.10, 0.092),
                     (0.12, 0.110), (0.15, 0.138), (1.00, 1.000)]:
        p = {**BASE, 'a_trail_ceiling_pct': a_c, 'f_trail_ceiling_pct': f_c}
        m = metricas(v2.run_v2_backtest(df4, df1, None, p), years)
        if m is None:
            continue
        resultados[(a_c, f_c)] = m
        etiqueta = 'sin techo' if a_c >= 1 else f"{a_c*100:.0f}% / {f_c*100:.1f}%"
        marca = ' <- actual' if a_c == 0.06 else (' *' if m['p'] < 0.05 else '')
        print(f"  {etiqueta:<16} {m['n']:>4} {m['wr']:>6.1f}% {m['pf']:>7.2f} "
              f"{m['cagr']:>+8.1f}% {m['dd']:>7.1f}% {m['p']:>7.3f}{marca}")
    return resultados


# ---------------------------------------------------------------------------
def parte3_wf(df4, mejor):
    print("\n" + "=" * 78)
    print(f"3. WALK-FORWARD 12 folds: techo actual vs techo {mejor[0]*100:.0f}%")
    print("=" * 78)
    n = len(df4) // 12
    acum = {'actual': [], 'nuevo': []}
    print(f"\n  {'fold':<24} {'actual':>16} {'nuevo':>16}")
    print("  " + "-" * 58)
    for k in range(12):
        seg = df4.iloc[k * n:(k + 1) * n]
        if len(seg) < 400:
            continue
        d1 = daily(seg)
        out = {}
        for lbl, p in (('actual', BASE),
                       ('nuevo', {**BASE, 'a_trail_ceiling_pct': mejor[0],
                                  'f_trail_ceiling_pct': mejor[1]})):
            tr = v2.run_v2_backtest(seg, d1, None, p)
            tot = float(np.prod([1 + t['pnl_pct'] for t in tr]) - 1) * 100 if tr else 0.0
            out[lbl] = (len(tr), tot)
            acum[lbl].append(tot)
        rango = f"{seg.index[0].date()} -> {seg.index[-1].date()}"
        print(f"  {rango:<24} {out['actual'][0]:>3}tr {out['actual'][1]:>+9.1f}% "
              f"{out['nuevo'][0]:>3}tr {out['nuevo'][1]:>+9.1f}%")
    print()
    for lbl in ('actual', 'nuevo'):
        a = np.array(acum[lbl])
        print(f"  {lbl:<8} folds positivos {int((a>0).sum())}/{len(a)}  "
              f"mediana {np.median(a):+.1f}%  suma {a.sum():+.1f}%")


def main():
    df4, df1 = load_btc()
    years = len(df4) * 4 / 24 / 365.25
    print(f"BTC 4h {df4.index[0].date()} -> {df4.index[-1].date()} ({years:.1f} anos)\n")
    parte1_mecanismo(df4, df1)
    res = parte2_barrido(df4, df1, years)
    mejor = max(res, key=lambda k: res[k]['cagr'])
    parte3_wf(df4, mejor)
    print("\n  Regla del proyecto: >= 7/12 folds y mejora consistente, o no se adopta.")


if __name__ == '__main__':
    main()
