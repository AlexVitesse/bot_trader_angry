"""
Smart Money Concepts / ICT codificado — ¿tiene edge medible?
============================================================
Origen: videos de "Asimetricos Traders" (2026-10): vela de referencia,
Fibonacci reducido a la zona OTE, maximos que deja una caida como objetivo
(liquidez), y AMD (acumulacion -> manipulacion -> distribucion).

Reglas PRE-REGISTRADAS (2026-10-17, antes de correr nada). LONG; SHORT es el
espejo exacto (se corre la misma logica sobre la serie con precios negados):

  1. Liquidez debajo: nivel L0 conocido al cierre de la vela anterior.
       swing : ultimo minimo fractal confirmado (3 velas a cada lado).
       pdhl  : minimo del dia UTC anterior ("la vela de referencia").
  2. Manipulacion (barrida): low < L0 y close > L0. L = minimo de la barrida
     (se extiende si sigue bajando mientras se espera la ruptura).
  3. Ruptura de estructura (MSS) en <= 6 velas: close > ultimo maximo
     fractal interno (1 vela a cada lado) confirmado antes de la barrida.
     H = maximo desde la barrida hasta el MSS.
  4. Entrada: limite en el OTE, H - 0,705*(H-L). Valida 12 velas; se cancela
     si antes toca el objetivo. Fill a min(limite, open).
  5. Stop: L - 0,1%. Objetivo (distribucion hacia la liquidez de arriba):
       swing : el maximo fractal (3) mas cercano por encima de H formado en
               las ultimas 100 velas.
       pdhl  : maximo del dia anterior a la barrida, si esta por encima de H.
     Solo se opera si R:R >= 2.
  6. Gestion: stop primero si stop y objetivo caen en la misma vela; tope de
     30 velas. Riesgo 2% del equity, nocional <= 2,5x.
  Costes: 0,06%/lado, funding constante 0,013%/8h, gap en el stop.

Variantes: {swing, pdhl} x {long, short, ambas} = 6. El null con seleccion
toma la mejor de las 6 en cada serie sintetica. Cross-asset: ETH.

Uso:  python experiments/smc/test_smc.py [n_series]
      python experiments/smc/test_smc.py --check   (test de no look-ahead)
"""
from __future__ import annotations

import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / 'experiments' / 'bootstrap_bloques'))
from test_bloques import sintetica, p_grupos   # noqa: E402

K_SWING, K_INT = 3, 1
MSS_MAX, ORDEN_MAX, MAX_BARS, LOOKBACK_TGT = 6, 12, 30, 100
OTE, RR_MIN, BUF = 0.705, 2.0, 0.001
RISK, NOTIONAL_MAX = 0.02, 2.5
COMISION, FUNDING_8H = 0.0006, 0.00013
MIN_TRADES = 10
VARIANTES = [(liq, d) for liq in ('swing', 'pdhl') for d in ('long', 'short', 'ambas')]
DF = None


def cargar(nombre):
    df = pd.read_parquet(ROOT / 'data' / f'{nombre}_USDT_4h_full.parquet')
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df[~df.index.duplicated(keep='last')].sort_index()


def pivotes(h, l, k):
    """Precio del pivote en la vela en que queda CONFIRMADO (j+k); NaN si no."""
    n = len(h)
    ph, pl, jh = np.full(n, np.nan), np.full(n, np.nan), np.full(n, -1)
    for j in range(k, n - k):
        if h[j] == h[j - k:j + k + 1].max():
            ph[j + k], jh[j + k] = h[j], j
        if l[j] == l[j - k:j + k + 1].min():
            pl[j + k] = l[j]
    return ph, pl, jh


def ffill_prev(x):
    """Ultimo valor no-NaN conocido al cierre de la vela ANTERIOR."""
    s = pd.Series(x).ffill().shift(1)
    return s.to_numpy()


def niveles_dia(df):
    d1 = df.resample('1D').agg({'high': 'max', 'low': 'min'})
    prev = d1.shift(1).reindex(df.index.floor('1D'))
    return prev['high'].to_numpy(), prev['low'].to_numpy()


def sim_long(o, h, l, c, lvl, tgt_dia, liq, d):
    """Una posicion a la vez. Devuelve [(i_entrada, i_salida, r, R)].
    d = +1 serie real (long) / -1 serie espejo (short): solo afecta al funding."""
    n = len(c)
    ph3, pl3, jh3 = pivotes(h, l, K_SWING)
    ph1, _, _ = pivotes(h, l, K_INT)
    ref_int = ffill_prev(ph1)
    conf_hi = np.where(~np.isnan(ph3))[0]          # velas donde se confirma un maximo
    trades = []
    st = 'idle'
    for t in range(1, n):
        if st == 'pos':
            barras += 1
            salida = None
            if l[t] <= stop:
                salida = min(stop, o[t])
            elif h[t] >= tgt:
                salida = max(tgt, o[t])
            elif barras >= MAX_BARS:
                salida = c[t]
            if salida is not None:
                trades.append(_cerrar(i_ent, t, ent, salida, stop0, barras, d))
                st = 'idle'
            continue
        if st == 'orden':
            if t > hasta or h[t] >= tgt:
                st = 'idle'
                continue
            if l[t] <= ent_lim:
                ent = min(ent_lim, o[t])
                i_ent, barras, stop0 = t, 0, stop
                if l[t] <= stop:                         # stop en la vela de entrada
                    trades.append(_cerrar(t, t, ent, min(stop, ent), stop0, 1, d))
                    st = 'idle'
                else:
                    st = 'pos'
            continue
        if st == 'sweep':
            L = min(L, l[t])
            if c[t] > ref:
                H = h[i_sw:t + 1].max()
                ent_lim = H - OTE * (H - L)
                stop = L - BUF * abs(L)
                if liq == 'swing':
                    m = conf_hi[(conf_hi <= t)]
                    m = m[jh3[m] >= i_sw - LOOKBACK_TGT]
                    cand = ph3[m][ph3[m] > H]
                    tgt = cand.min() if len(cand) else np.nan
                else:
                    tgt = tgt_dia[i_sw] if tgt_dia[i_sw] > H else np.nan
                if (not np.isnan(tgt) and ent_lim > stop
                        and (tgt - ent_lim) / (ent_lim - stop) >= RR_MIN):
                    st, hasta = 'orden', t + ORDEN_MAX
                else:
                    st = 'idle'
                continue
            if t - i_sw >= MSS_MAX:
                st = 'idle'
            continue
        # idle: buscar barrida
        lv = lvl[t]
        if not np.isnan(lv) and l[t] < lv < c[t] and not np.isnan(ref_int[t]):
            st, L, i_sw, ref = 'sweep', l[t], t, ref_int[t]
    return trades


def _cerrar(i_ent, i_sal, ent, sal, stop, barras, d):
    sd = (ent - stop) / abs(ent)
    bruto = (sal - ent) / abs(ent)
    neto = bruto - 2 * COMISION - FUNDING_8H * 0.5 * barras * d
    frac = min(RISK / sd, NOTIONAL_MAX)
    return (i_ent, i_sal, frac * neto, bruto / sd)


def correr(df, liq, direccion):
    o, h, l, c = (df[k].to_numpy() for k in ('open', 'high', 'low', 'close'))
    pdh, pdl = niveles_dia(df)
    out = []
    if direccion in ('long', 'ambas'):
        lvl = ffill_prev(pivotes(h, l, K_SWING)[1]) if liq == 'swing' else pdl
        out += [(*t, 1) for t in sim_long(o, h, l, c, lvl, pdh, liq, 1)]
    if direccion in ('short', 'ambas'):
        # espejo: precios negados -> maximos pasan a ser minimos
        mo, mh, ml, mc = -o, -l, -h, -c
        lvl = ffill_prev(pivotes(mh, ml, K_SWING)[1]) if liq == 'swing' else -pdh
        out += [(*t, -1) for t in sim_long(mo, mh, ml, mc, lvl, -pdl, liq, -1)]
    out.sort()
    if direccion == 'ambas':                       # una posicion a la vez
        fil, fin = [], -1
        for t in out:
            if t[0] > fin:
                fil.append(t)
                fin = t[1]
        out = fil
    return out


def metricas(trades, df):
    if len(trades) < MIN_TRADES:
        return None
    r = np.array([t[2] for t in trades])
    R = np.array([t[3] for t in trades])
    eq = np.cumprod(1 + r)
    años = (df.index[-1] - df.index[0]).days / 365.25
    w, lo = r[r > 0].sum(), abs(r[r <= 0].sum())
    return {'n': len(r), 'por_año': len(r) / años, 'wr': (r > 0).mean() * 100,
            'pf': w / lo if lo else np.inf, 'R': R.mean(),
            'cagr': (eq[-1] ** (1 / años) - 1) * 100,
            'dd': float((1 - eq / np.maximum.accumulate(np.r_[1, eq])[1:]).max() * 100),
            'mean_r': r.mean()}


def stat(trades):
    return float(np.mean([t[2] for t in trades])) if len(trades) >= MIN_TRADES else -np.inf


def _init(df):
    global DF
    DF = df


def _null(args):
    L, seed = args
    s = sintetica(DF, L, seed)
    return [stat(correr(s, liq, d)) for liq, d in VARIANTES]


def informe(nombre, df, rng):
    print(f"\n{nombre} 4h {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"  {'variante':<13} {'n':>4} {'tr/a':>5} {'WR':>6} {'PF':>5} {'R medio':>8} "
          f"{'CAGR':>7} {'DD':>6} {'p bloq 30d':>10}")
    obs = []
    for liq, d in VARIANTES:
        tr = correr(df, liq, d)
        m = metricas(tr, df)
        obs.append(stat(tr))
        if m is None:
            print(f"  {liq + '/' + d:<13} {len(tr):>4} (muestra insuficiente)")
            continue
        r = np.array([t[2] for t in tr])
        ent = df.index[[t[0] for t in tr]]
        p, _ = p_grupos(r, ((ent - ent[0]).days // 30).to_numpy(), rng)
        print(f"  {liq + '/' + d:<13} {m['n']:>4} {m['por_año']:>5.1f} {m['wr']:>5.1f}% "
              f"{m['pf']:>5.2f} {m['R']:>+8.3f} {m['cagr']:>+6.1f}% {m['dd']:>5.1f}% {p:>10.3f}")
    return np.array(obs)


def check():
    """No look-ahead: truncar la serie no puede cambiar los trades ya cerrados."""
    df = cargar('BTC')
    corte = len(df) // 2
    for liq, d in VARIANTES:
        full = [t for t in correr(df, liq, d) if t[1] < corte - 1]
        trunc = [t for t in correr(df.iloc[:corte], liq, d) if t[1] < corte - 1]
        assert full == trunc, (liq, d, len(full), len(trunc))
    print('check OK: sin look-ahead en las 6 variantes')


def main():
    if '--check' in sys.argv:
        return check()
    n_series = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    rng = np.random.default_rng(7)
    btc, eth = cargar('BTC'), cargar('ETH')
    obs = informe('BTC', btc, rng)
    informe('ETH (cross-asset)', eth, rng)

    print(f"\nNULL SINTETICO BTC ({n_series} series; stat = mean(r) por trade)")
    for L in (42, 180):
        with Pool(initializer=_init, initargs=(btc,)) as pool:
            null = np.array(pool.map(_null, [(L, s) for s in range(n_series)], chunksize=4))
        best = null.max(axis=1)
        print(f"  bloques de {L} velas ({L // 6} dias):")
        for j, (liq, d) in enumerate(VARIANTES):
            if not np.isfinite(obs[j]):
                continue
            print(f"    {liq + '/' + d:<13} obs {obs[j]:+.4%}  mediana null "
                  f"{np.median(null[np.isfinite(null[:, j]), j]):+.4%}  "
                  f"p {(null[:, j] >= obs[j]).mean():.3f}  "
                  f"p con seleccion (max de 6) {(best >= obs[j]).mean():.3f}")
        sys.stdout.flush()


if __name__ == '__main__':
    main()
