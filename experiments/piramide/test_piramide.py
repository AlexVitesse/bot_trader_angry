"""E5: piramidacion Turtle sobre V2. Pre-registro en README.md.

Uso: python experiments/piramide/test_piramide.py [n_series]
"""
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'portfolio_sim'))
sys.path.insert(0, str(ROOT / 'experiments' / 'bootstrap_bloques'))
from portfolio_sim import PortfolioSim                  # noqa: E402
from test_bloques import cargar_btc, features, sintetica  # noqa: E402

BASE, CAP = 0.045, 2.0
UNIDAD = BASE / 4
PASO, MAX_ADDS = 0.5, 3
L_VELAS = 180
DF = None


def atr20(feat):
    h, l, c = feat['high'], feat['low'], feat['close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(20).mean().shift(1)        # conocido al abrir la vela


class Piramide:
    def __init__(self, feat):
        self.N = atr20(feat).to_numpy()
        self.close = feat['close'].to_numpy()
        self.idx = {ts: i for i, ts in enumerate(feat.index)}
        self.ini = {}                              # ts_entrada -> (precio, N)
        self.expo = {}

    def __call__(self, par, ts, p, i):
        if p.ts_entrada not in self.ini:
            self.ini[p.ts_entrada] = (p.entrada, self.N[self.idx[p.ts_entrada]])
        e0, n = self.ini[p.ts_entrada]
        add = 0.0
        if p.n_adds < MAX_ADDS and n > 0:
            nivel = e0 + p.direccion * (p.n_adds + 1) * PASO * n
            if (self.close[i] - nivel) * p.direccion >= 0:
                add = p.equity_entrada * UNIDAD / p.trail
        self.expo[ts] = (p.notional, p.entrada)
        return add


class Espia:
    """Registra notional por vela sin anadir nada (para V2-m)."""
    def __init__(self):
        self.expo = {}

    def __call__(self, par, ts, p, i):
        self.expo[ts] = (p.notional, p.entrada)
        return 0.0


def correr(feat, risk, hook):
    return PortfolioSim({'BTC/USDT': feat}, risk_pct=risk, max_concurrent=1,
                        max_notional_pct=CAP, piramide=hook).run()


def expo_media(res, hook, feat):
    eq, close = res.equity, feat['close']
    x = [n * close[ts] / e / eq[ts] for ts, (n, e) in hook.expo.items() if ts in eq.index]
    return float(np.sum(x) / len(eq))


def met(res):
    m = res.metricas
    return m['cagr'], m['dd'], m['cagr'] / m['dd']


def _init(df, risk_m):
    global DF, RISK_M
    DF, RISK_M = df, risk_m


def _worker(seed):
    feat = features(sintetica(DF, L_VELAS, seed))
    return (correr(feat, UNIDAD, Piramide(feat)).metricas.get('cagr', -100),
            correr(feat, RISK_M, Espia()).metricas.get('cagr', -100))


RISK_M = None


def main():
    global RISK_M
    n_series = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    df = cargar_btc()
    feat = features(df)

    v2 = PortfolioSim({'BTC/USDT': feat}, risk_pct=BASE, max_concurrent=1, max_notional_pct=CAP).run()
    assert v2.metricas['n'] == 165 and round(v2.metricas['cagr'], 1) == 23.5, v2.metricas

    hk = Piramide(feat)
    e5 = correr(feat, UNIDAD, hk)
    x5 = expo_media(e5, hk, feat)
    risk_m = UNIDAD
    for _ in range(3):
        sp = Espia()
        vm = correr(feat, risk_m, sp)
        risk_m *= x5 / expo_media(vm, sp, feat)
    sp = Espia()
    vm = correr(feat, risk_m, sp)
    RISK_M = risk_m

    tr = pd.DataFrame(e5.trades)
    print(f"E5: trades {len(tr)}, con >=1 anadido {(tr.n_adds > 0).mean():.1%}, "
          f"anadidos por trade {tr.n_adds.mean():.2f}, con 3 anadidos {(tr.n_adds == 3).mean():.1%}")
    print(f"{'brazo':<12} {'riesgo':>7} {'expo':>6} {'PF':>5} {'CAGR':>7} {'DD':>6} {'CAGR/DD':>8}")
    for nombre, r, rk, x in (('E5', e5, UNIDAD, x5), ('V2-m', vm, risk_m, expo_media(vm, sp, feat)),
                             ('V2 4,5%', v2, BASE, None)):
        c, d, q = met(r)
        xs = f"{x:>6.3f}" if x is not None else ' ' * 6
        print(f"{nombre:<12} {rk:>7.2%} {xs} {r.metricas['pf']:>5.2f} {c:>+6.1f}% {d:>5.1f}% {q:>8.2f}")
    for k in range(4):
        g = tr[tr.n_adds == k]
        if len(g):
            print(f"  trades con {k} anadidos: {len(g):>3}  PnL total {g.pnl.sum():>+9.0f}")

    print(f"\nNull sintetico: {n_series} series, bloques de {L_VELAS} velas ...")
    with Pool(initializer=_init, initargs=(df, risk_m)) as pool:
        out = np.array(pool.map(_worker, range(n_series)))
    c5, cm = met(e5)[0], met(vm)[0]
    p5 = float((out[:, 0] >= c5).mean())
    pm = float((out[:, 1] >= cm).mean())
    print(f"  E5:   CAGR real {c5:+.1f}%  null mediana {np.median(out[:, 0]):+.1f}%  p = {p5:.3f}")
    print(f"  V2-m: CAGR real {cm:+.1f}%  null mediana {np.median(out[:, 1]):+.1f}%  p = {pm:.3f}")
    c1 = met(e5)[2] > met(vm)[2]
    c2 = p5 < 0.01
    print(f"\nH5: (1) CAGR/DD > V2-m: {c1}   (2) null p<0,01: {c2}   -> {'PASA' if c1 and c2 else 'NO pasa'}")


if __name__ == '__main__':
    main()
