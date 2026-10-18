"""
¿V2 funciona fuera de cripto, y diversificar sube el retorno al mismo DD?
=========================================================================
Pre-registrado (2026-10-17), antes de ver ningun resultado:

  - Activos: SPY, QQQ, IWM (bolsa EEUU), GLD (oro), TLT (bonos 20a).
    ETFs con dividendos reinvertidos (yfinance auto_adjust), velas DIARIAS
    (yfinance no da 4h con historia larga).
  - Reglas V2 congeladas, LONG-only. Regimen EMA50/200 sobre la misma serie
    diaria (como el "1d" de experiments/timeframe/).
  - Unico ajuste: suelo/techo del trailing escalados por k = ATR% mediano del
    activo (1d) / ATR% mediano de BTC 4h. Es el mismo criterio que
    timeframe/ uso para BTC diario (2,69x). Un numero por activo, sacado de la
    volatilidad, no de los retornos.
  - Costes: 0,06%/lado, gap en el stop, financiacion ~2,4%/año sobre el
    nocional (FUNDING_8H del simulador aplicado por vela).
  - Preguntas:
      1. Por activo: ¿gana dinero (bootstrap bloques 90 dias) y supera al
         null sintetico (block-shuffle 30 velas)? Corregir por 5 activos.
      2. Cartera BTC 4h (V2 desplegado) + ETFs, 2019-01 -> 2026-02:
         CAGR al MISMO DD que BTC solo. Esto es lo que diversificar compra.

Uso: python experiments/multiactivo/test_multiactivo.py [n_series]
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'portfolio_sim'))
sys.path.insert(0, str(ROOT / 'experiments' / 'bootstrap_bloques'))

from portfolio_sim import PortfolioSim                    # noqa: E402
from test_bloques import sintetica, p_grupos, cargar_btc  # noqa: E402
from src import v2_engine as v2                           # noqa: E402

ETFS = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
BASE = {**v2.PARAMS_V2, 'f_enable_short': False}
RISK = 0.02
DATOS = {}


def cargar_etf(t):
    d = pd.read_parquet(HERE / 'data' / f'{t}_1d.parquet').astype(float)
    d.index = pd.DatetimeIndex(d.index).tz_localize('UTC')
    return d[d['volume'] > 0]


def feat_diario(d, prm):
    return v2.build_features(d, d, None, prm)


def feat_4h(df):
    d1 = df.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                'close': 'last', 'volume': 'sum'}).dropna()
    return v2.build_features(df, d1, None, BASE)


def params_escalados(k):
    return {**BASE,
            'a_trail_floor_pct': BASE['a_trail_floor_pct'] * k,
            'a_trail_ceiling_pct': BASE['a_trail_ceiling_pct'] * k,
            'f_trail_floor_pct': BASE['f_trail_floor_pct'] * k,
            'f_trail_ceiling_pct': BASE['f_trail_ceiling_pct'] * k}


def correr(datos, prm_par, risk=RISK, **kw):
    return PortfolioSim(datos, params=BASE, params_por_par=prm_par,
                        risk_pct=risk, leverage=5, max_notional_pct=2.0, **kw).run()


def mean_r(res):
    r = [t['r'] for t in res.trades]
    return float(np.mean(r)) if len(r) >= 10 else -np.inf


def buy_hold(d):
    c = d['close']
    años = (c.index[-1] - c.index[0]).days / 365.25
    return ((c.iloc[-1] / c.iloc[0]) ** (1 / años) - 1) * 100, float((1 - c / c.cummax()).max() * 100)


def _init(datos):
    global DATOS
    DATOS = datos


def _null(args):
    t, k, seed = args
    prm = params_escalados(k)
    f = feat_diario(sintetica(DATOS[t], 30, seed), prm)
    return mean_r(correr({t: f}, {t: prm}))


def dd_de(res):
    eq = res.equity
    return float((1 - eq / eq.cummax()).max() * 100)


def main():
    n_series = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(7)
    btc4 = cargar_btc()
    fbtc = feat_4h(btc4)
    atr_btc = float(fbtc['atr_pct'].median())
    raw = {t: cargar_etf(t) for t in ETFS}

    ks, prms, feats = {}, {}, {}
    for t in ETFS:
        atr = float(feat_diario(raw[t], BASE)['atr_pct'].median())
        ks[t] = atr / atr_btc
        prms[t] = params_escalados(ks[t])
        feats[t] = feat_diario(raw[t], prms[t])

    # ---------------- 1. por activo ----------------
    print(f"ATR% mediano BTC 4h = {atr_btc:.2%}\n")
    print('1. POR ACTIVO (historia completa, riesgo 2%/trade)')
    print(f"  {'activo':<6} {'desde':<10} {'k':>5} {'trail A':>12} {'n':>4} {'tr/a':>5} "
          f"{'WR':>6} {'PF':>5} {'CAGR':>7} {'DD':>6} {'B&H':>7} {'DD B&H':>7} "
          f"{'p bloq':>7} {'p null':>7}")
    resumen = {}
    with Pool(initializer=_init, initargs=(raw,)) as pool:
        for t in ETFS:
            res = correr({t: feats[t]}, {t: prms[t]})
            m = res.metricas
            r = np.array([x['r'] for x in res.trades])
            ent = pd.DatetimeIndex([x['ts_entrada'] for x in res.trades])
            p_b, _ = p_grupos(r, ((ent - ent[0]).days // 90).to_numpy(), rng)
            null = np.array(pool.map(_null, [(t, ks[t], s) for s in range(n_series)]))
            p_n = float((null >= r.mean()).mean())
            bh, bhdd = buy_hold(raw[t])
            resumen[t] = {**m, 'p_b': p_b, 'p_n': p_n, 'k': ks[t]}
            print(f"  {t:<6} {str(raw[t].index[0].date()):<10} {ks[t]:>5.2f} "
                  f"{prms[t]['a_trail_floor_pct']:>5.1%}-{prms[t]['a_trail_ceiling_pct']:<5.1%} "
                  f"{m['n']:>4} {m['por_año']:>5.1f} {m['wr']:>5.1f}% {m['pf']:>5.2f} "
                  f"{m['cagr']:>+6.1f}% {m['dd']:>5.1f}% {bh:>+6.1f}% {bhdd:>6.1f}% "
                  f"{p_b:>7.3f} {p_n:>7.3f}")
            sys.stdout.flush()
    print("  (p bloq: H0 mean(r)<=0, bloques de 90 dias. p null: V2 sobre series "
          "con bloques de 30 velas barajados.\n   Bonferroni m=5 -> corte 0,01)")

    # ---------------- 2. cartera ----------------
    desde, hasta = fbtc.index[0], fbtc.index[-1]
    print(f"\n2. CARTERA {desde.date()} -> {hasta.date()} (periodo de BTC)")
    solo_btc = {'BTC/USDT': fbtc}
    cartera = {'BTC/USDT': fbtc, **feats}
    p_cart = {**prms}
    kw = dict(max_concurrent=6, max_misma_dir=6)
    print(f"  {'riesgo':>6} | {'BTC solo CAGR':>13} {'DD':>6} | {'BTC+ETFs CAGR':>13} {'DD':>6} {'n':>4}"
          f" | {'ETFs solos CAGR':>15} {'DD':>6}")
    filas = []
    for rk in (0.02, 0.03, 0.045, 0.06, 0.08, 0.10):
        a = PortfolioSim(solo_btc, params=BASE, risk_pct=rk, leverage=5,
                         max_notional_pct=2.0, **kw).run(desde, hasta)
        b = PortfolioSim(cartera, params=BASE, params_por_par=p_cart, risk_pct=rk,
                         leverage=5, max_notional_pct=2.0, **kw).run(desde, hasta)
        c = PortfolioSim(feats, params=BASE, params_por_par=p_cart, risk_pct=rk,
                         leverage=5, max_notional_pct=2.0, **kw).run(desde, hasta)
        ma, mb, mc = a.metricas, b.metricas, c.metricas
        filas.append((rk, ma, mb, mc))
        print(f"  {rk:>6.1%} | {ma['cagr']:>+12.1f}% {ma['dd']:>5.1f}% | "
              f"{mb['cagr']:>+12.1f}% {mb['dd']:>5.1f}% {mb['n']:>4} | "
              f"{mc['cagr']:>+14.1f}% {mc['dd']:>5.1f}%")
        sys.stdout.flush()

    # correlacion mensual de los sleeves con BTC
    b = PortfolioSim(cartera, params=BASE, params_por_par=p_cart, risk_pct=0.02,
                     leverage=5, max_notional_pct=2.0, **kw).run(desde, hasta)
    tr = pd.DataFrame(b.trades)
    tr['mes'] = pd.DatetimeIndex(tr['ts_salida']).to_period('M')
    pm = tr.pivot_table(index='mes', columns='par', values='r', aggfunc='sum').fillna(0)
    print("\n  Correlacion mensual del PnL de cada sleeve con el de BTC:")
    print('  ' + '  '.join(f"{c}={pm[c].corr(pm['BTC/USDT']):+.2f}"
                           for c in pm.columns if c != 'BTC/USDT'))
    print('  Contribucion por sleeve a riesgo 2% (suma de r):')
    print('  ' + '  '.join(f"{c}={v:+.1%}" for c, v in tr.groupby('par')['r'].sum().items()))

    # ETFs solos en su historia larga (2005-2026)
    largo = PortfolioSim(feats, params=BASE, params_por_par=p_cart, risk_pct=0.02,
                         leverage=5, max_notional_pct=2.0, **kw).run(pd.Timestamp('2005-01-01', tz='UTC'))
    m = largo.metricas
    print(f"\n  ETFs solos 2005 -> 2026, riesgo 2%: n={m['n']} WR {m['wr']:.1f}% "
          f"PF {m['pf']:.2f} CAGR {m['cagr']:+.1f}% DD {m['dd']:.1f}%")


if __name__ == '__main__':
    main()
