"""E4: riesgo segun la caida del capital. Pre-registro en README.md.

Uso: python experiments/dd_sizing/test_dd_sizing.py [n_series]
"""
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'portfolio_sim'))
sys.path.insert(0, str(ROOT / 'experiments' / 'bootstrap_bloques'))
from portfolio_sim import PortfolioSim                                   # noqa: E402
from test_bloques import cargar_btc, features, sintetica                 # noqa: E402

BASE, KILL, PISO, CAP = 0.045, 0.45, 0.25, 2.0
REPS, BLOQUE_T, N_3A = 20_000, 10, 70
L_VELAS = 180
SEED = 20261019
DF = None


def riesgo_dd(dd):
    return BASE * np.maximum(PISO, 1 - dd / KILL)


class SizingDD:
    """Hook de PortfolioSim: guarda el pico de equities de apertura."""
    def __init__(self):
        self.pico, self.riesgos = 0.0, []

    def __call__(self, par, ts, equity, trail):
        self.pico = max(self.pico, equity)
        r = float(riesgo_dd(1 - equity / self.pico))
        self.riesgos.append(r)
        return equity * r / trail


def sim(feat, sizing=None, risk=BASE):
    return PortfolioSim({'BTC/USDT': feat}, risk_pct=risk, max_concurrent=1,
                        max_notional_pct=CAP, sizing=sizing).run()


def caminos(g, trail, regla, m_fijo):
    """g, trail: (caminos, n). Devuelve (CAGR/DD por trades, DD) para E4 y Fijo-m.
    Equity al cierre de cada trade; horizonte en trades."""
    out = {}
    for nombre in ('e4', 'fijo'):
        E = np.ones(g.shape[0])
        pico = E.copy()
        pico_ap = E.copy()
        ddmax = np.zeros_like(E)
        for j in range(g.shape[1]):
            if nombre == 'e4':
                pico_ap = np.maximum(pico_ap, E)
                rk = regla(1 - E / pico_ap)
            else:
                rk = m_fijo
            notional = np.minimum(E * rk / trail[:, j], CAP * E)
            E = E + g[:, j] * notional
            E = np.maximum(E, 1e-9)
            pico = np.maximum(pico, E)
            ddmax = np.maximum(ddmax, 1 - E / pico)
        out[nombre] = (E, ddmax)
    return out


def idx_bloques(n, reps, largo, rng):
    nb = int(np.ceil(largo / BLOQUE_T))
    i = (rng.integers(0, n, (reps, nb))[:, :, None] + np.arange(BLOQUE_T)).reshape(reps, -1)
    return i[:, :largo] % n


def _init(df):
    global DF
    DF = df


def _worker(seed):
    feat = features(sintetica(DF, L_VELAS, seed))
    res = sim(feat)
    if len(res.trades) < 10:
        return None
    g = np.array([t['pnl'] / t['notional'] for t in res.trades])
    trail = np.array([t['trail'] for t in res.trades])
    return g, trail


def main():
    n_series = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rng = np.random.default_rng(SEED)
    df = cargar_btc()
    feat = features(df)

    # ---- 1. historia, simulador real
    hook = SizingDD()
    r_e4 = sim(feat, sizing=hook)
    m_fijo = float(np.mean(hook.riesgos))
    r_m = sim(feat, risk=m_fijo)
    r_45 = sim(feat)
    print(f"Riesgo medio usado por E4: {m_fijo:.2%}  (min {min(hook.riesgos):.2%})")
    print(f"{'brazo':<14} {'n':>4} {'CAGR':>7} {'DD':>6} {'CAGR/DD':>8}")
    for nombre, r in (('E4', r_e4), (f'Fijo {m_fijo:.2%}', r_m), ('Fijo 4,50%', r_45)):
        m = r.metricas
        print(f"{nombre:<14} {m['n']:>4} {m['cagr']:>+6.1f}% {m['dd']:>5.1f}% {m['cagr'] / m['dd']:>8.2f}")

    # g y trail por trade (identicos entre brazos: mismos trades)
    g = np.array([t['pnl'] / t['notional'] for t in r_45.trades])
    trail = np.array([t['trail'] for t in r_45.trades])
    años = (r_45.equity.index[-1] - r_45.equity.index[0]).days / 365.25
    n = len(g)

    # comprobacion: la reconstruccion trade a trade reproduce el simulador
    rec = caminos(g[None], trail[None], riesgo_dd, m_fijo)
    eq_sim = r_e4.equity.iloc[-1] / r_e4.equity.iloc[0]
    print(f"\nReconstruccion E4: final {rec['e4'][0][0]:.4f} vs simulador {eq_sim:.4f}")

    # ---- 2. H4: bootstrap pareado por bloques de trades
    delta = np.empty(REPS)
    for k in range(0, REPS, 2000):
        ii = idx_bloques(n, 2000, n, rng)
        c = caminos(g[ii], trail[ii], riesgo_dd, m_fijo)
        ratio = {}
        for b, (E, dd) in c.items():
            ratio[b] = (E ** (1 / años) - 1) / np.maximum(dd, 1e-9)
        delta[k:k + 2000] = ratio['e4'] - ratio['fijo']
    p = float((delta <= 0).mean())
    print(f"\nH4: mediana delta CAGR/DD (E4 - Fijo-m) = {np.median(delta):+.3f}  "
          f"IC90 [{np.percentile(delta, 5):+.3f}, {np.percentile(delta, 95):+.3f}]  "
          f"p = P(delta<=0) = {p:.4f} -> {'PASA' if p < 0.01 else 'NO pasa'} (alpha 0,01)")

    # ---- 3. P(kill en 3 anos)
    ii = idx_bloques(n, REPS, N_3A, rng)
    c = caminos(g[ii], trail[ii], riesgo_dd, m_fijo)
    c45 = caminos(g[ii], trail[ii], riesgo_dd, BASE)['fijo']
    print("\nMonte Carlo 3 anos (70 trades, bloques de 10, DD al cierre de trade):")
    for nombre, (E, dd) in (('E4', c['e4']), (f'Fijo {m_fijo:.2%}', c['fijo']), ('Fijo 4,50%', c45)):
        print(f"  {nombre:<12} P(kill) {np.mean(dd >= KILL):>6.2%}  DD p95 {np.percentile(dd, 95):>5.1%}  "
              f"CAGR med {np.median(E) ** (1 / 3) - 1:>+6.1%}  P(perder) {np.mean(E < 1):>5.1%}")

    # ---- 4. null sintetico
    print(f"\nNull sintetico: {n_series} series, bloques de {L_VELAS} velas ...")
    with Pool(initializer=_init, initargs=(df,)) as pool:
        out = [o for o in pool.map(_worker, range(n_series)) if o is not None]
    kills = {'e4': 0, 'fijo': 0, 'f45': 0}
    deltas = []
    for g_s, tr_s in out:
        c = caminos(g_s[None], tr_s[None], riesgo_dd, m_fijo)
        c45 = caminos(g_s[None], tr_s[None], riesgo_dd, BASE)['fijo']
        kills['e4'] += c['e4'][1][0] >= KILL
        kills['fijo'] += c['fijo'][1][0] >= KILL
        kills['f45'] += c45[1][0] >= KILL
        rat = {b: (E[0] ** (1 / años) - 1) / max(dd[0], 1e-9) for b, (E, dd) in c.items()}
        deltas.append(rat['e4'] - rat['fijo'])
    ns = len(out)
    print(f"  series validas {ns}: P(kill) E4 {kills['e4'] / ns:.1%}  Fijo-m {kills['fijo'] / ns:.1%}  "
          f"Fijo 4,5% {kills['f45'] / ns:.1%}")
    print(f"  mediana delta CAGR/DD {np.median(deltas):+.3f}  fraccion delta>0 {np.mean(np.array(deltas) > 0):.1%}")


if __name__ == '__main__':
    main()
