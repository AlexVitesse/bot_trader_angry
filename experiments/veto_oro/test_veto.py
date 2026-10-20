"""E12: la alerta de caidas de E7 como veto de entradas de V2. Pre-registro en README.md.

Uso: python experiments/veto_oro/test_veto.py [n_series]
"""
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for d in ('portfolio_sim', 'bootstrap_bloques', 'especialistas'):
    sys.path.insert(0, str(ROOT / 'experiments' / d))
from portfolio_sim import PortfolioSim                        # noqa: E402
from test_bloques import cargar_btc, features, sintetica      # noqa: E402
from test_especialistas import construir, BASE, MACRO, H, AÑOS  # noqa: E402

DESDE = pd.Timestamp('2021-01-01', tz='UTC')
FEATS = BASE + MACRO
L_VELAS, SEED = 180, 20261020
DF = None


def alerta(btc4h):
    """Serie diaria booleana fuera de muestra: p_hat >= cuantil 0,80 del train de su año."""
    df = construir(btc4h)
    out = pd.Series(False, index=df.index)
    for Y in AÑOS:
        ini = pd.Timestamp(f'{Y}-01-01', tz='UTC')
        tr = df[df.index + pd.Timedelta(days=H) < ini]
        # el test usa todos los dias del año con features (la etiqueta no hace falta para alertar)
        te = df[(df.index >= ini) & (df.index < ini + pd.DateOffset(years=1))]
        if te.empty:
            continue
        mu, sd = tr[FEATS].mean(), tr[FEATS].std()
        m = LogisticRegression(C=1.0).fit((tr[FEATS] - mu) / sd, tr['y'])
        umbral = np.quantile(m.predict_proba((tr[FEATS] - mu) / sd)[:, 1], 0.80)
        out[te.index] = m.predict_proba((te[FEATS] - mu) / sd)[:, 1] >= umbral
    return out


class Veto:
    def __init__(self, al):
        self.al = al

    def __call__(self, par, ts, equity, trail):
        dia = ts.normalize() - pd.Timedelta(days=1)
        return 0.0 if bool(self.al.get(dia, False)) else None


def correr(feat, risk, veto=None):
    return PortfolioSim({'BTC/USDT': feat}, risk_pct=risk, max_concurrent=1,
                        max_notional_pct=2.0, sizing=veto).run(desde=DESDE)


def ratio(eq):
    años = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / años) - 1
    dd = float((1 - eq / eq.cummax()).max())
    return cagr, dd, cagr / dd if dd > 0 else np.nan


def expo(res):
    eq = res.equity
    x = pd.Series(0.0, index=eq.index)
    for t in res.trades:
        m = (eq.index >= t['ts_entrada']) & (eq.index < t['ts_salida'])
        x[m] = t['notional'] / eq[m]
    return float(x.mean())


def _init(df):
    global DF
    DF = df


def _worker(seed):
    try:
        s = sintetica(DF, L_VELAS, seed)
        f, al = features(s), alerta(s)
        return ratio(correr(f, 0.045, Veto(al)).equity)[2] - ratio(correr(f, 0.045).equity)[2]
    except Exception as e:
        return f'error: {e}'


def main():
    n_series = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    btc = cargar_btc()
    feat = features(btc)
    al = alerta(btc)
    al_oos = al[al.index >= DESDE]
    print(f"Dias con alerta (2021+): {al_oos.mean():.1%} de {len(al_oos)}")

    base = correr(feat, 0.045)
    # comprobacion: un veto que nunca veta reproduce V2
    nada = correr(feat, 0.045, Veto(pd.Series(False, index=al.index)))
    assert np.allclose(base.equity.values, nada.equity.values)
    con = correr(feat, 0.045, Veto(al))
    print(f"V2 trades 2021+: {len(base.trades)} | con veto: {len(con.trades)} | rechazos {con.rechazos}")

    # trades de V2 que habrian caido en alerta
    rr = np.array([t['r'] for t in base.trades])
    marc = np.array([bool(al.get(t['ts_entrada'].normalize() - pd.Timedelta(days=1), False))
                     for t in base.trades])
    v = rr[marc]
    if len(v):
        w, l = v[v > 0].sum(), -v[v <= 0].sum()
        rng = np.random.default_rng(SEED)
        n, L = len(v), 10
        nb = int(np.ceil(n / L))
        mu = np.array([v[(rng.integers(0, n, nb)[:, None] + np.arange(L)).ravel()[:n] % n].mean()
                       for _ in range(20_000)])
        p_v = float(np.mean(mu - v.mean() <= v.mean()))     # H0 mean >= 0, una cola
        print(f"Trades de V2 en dia de alerta: {n} | media r {v.mean():+.4f} | WR {(v > 0).mean():.1%} | "
              f"PF {w / l if l else np.inf:.2f} | p(mean r < 0) bloques = {p_v:.3f}")
        print(f"Resto: {(~marc).sum()} | media r {rr[~marc].mean():+.4f} | WR {(rr[~marc] > 0).mean():.1%}")

    ex_con, ex_b = expo(con), expo(base)
    risk = 0.045
    for _ in range(2):
        risk *= ex_con / ex_b
        bm = correr(feat, risk)
        ex_b = expo(bm)
    print(f"\n{'brazo':<22} {'CAGR':>7} {'DD':>6} {'CAGR/DD':>7} {'expo':>6}")
    for nombre, res_, ex in (('V2 4,5%', base, expo(base)), ('V2 + veto 4,5%', con, ex_con),
                             (f'V2 {risk:.2%} (=expo)', bm, ex_b)):
        cg, dd, rt = ratio(res_.equity)
        print(f"{nombre:<22} {cg:>+7.1%} {dd:>6.1%} {rt:>7.2f} {ex:>6.3f}")
    c1 = ratio(con.equity)[2] > ratio(bm.equity)[2]
    d_obs = ratio(con.equity)[2] - ratio(base.equity)[2]
    print(f"\n(1) CAGR/DD veto > V2 a igual expo: {'si' if c1 else 'NO'} | delta vs V2 4,5% = {d_obs:+.3f}")

    print(f"\nNull sintetico: {n_series} series ...", flush=True)
    with Pool(initializer=_init, initargs=(btc,)) as pool:
        out = pool.map(_worker, [SEED + k for k in range(n_series)])
    err = [x for x in out if isinstance(x, str)]
    d = np.array([x for x in out if not isinstance(x, str) and np.isfinite(x)])
    p2 = float((d >= d_obs).mean())
    print(f"  validas {len(d)} (errores {len(err)}{': ' + err[0] if err else ''}) | delta sint mediana "
          f"{np.median(d):+.3f} p90 {np.percentile(d, 90):+.3f} | p = {p2:.3f} -> "
          f"{'cumple' if p2 < 0.05 else 'NO cumple'}")
    print(f"\nH12: {'PASA' if c1 and p2 < 0.05 else 'NO pasa'}")


if __name__ == '__main__':
    main()
