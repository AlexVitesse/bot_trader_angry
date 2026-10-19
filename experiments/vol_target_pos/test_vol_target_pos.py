"""E2: vol objetivo con la posicion abierta (solo reduce). Pre-registro en README.md.

Uso: python experiments/vol_target_pos/test_vol_target_pos.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'portfolio_sim'))
sys.path.insert(0, str(ROOT / 'experiments' / 'bootstrap_bloques'))
from portfolio_sim import PortfolioSim               # noqa: E402
from test_bloques import cargar_btc, features        # noqa: E402

BASE, CAP, UMBRAL = 0.045, 2.0, 0.8
REPS, BLOQUE = 20_000, 30
SEED = 20261019


def sigma_previa(df4):
    """sigma GK 20 dias del dia ANTERIOR, mapeada a cada vela 4h."""
    o, h, l, c = (np.log(df4[k]) for k in ('open', 'high', 'low', 'close'))
    gk = 0.5 * (h - l) ** 2 - (2 * np.log(2) - 1) * (c - o) ** 2
    sig_d = np.sqrt(gk.resample('1D').sum().rolling(20).mean() * 365).shift(1)
    return sig_d.reindex(df4.index.normalize()).set_axis(df4.index)


class Ajuste:
    def __init__(self, sig):
        self.sig = sig
        self.entrada = {}          # ts_entrada -> (notional_entrada, sigma_entrada)
        self.n_reducciones = 0

    def __call__(self, par, ts, p):
        if p.ts_entrada not in self.entrada:
            self.entrada[p.ts_entrada] = (p.notional, self.sig.get(p.ts_entrada, np.nan))
        n0, s0 = self.entrada[p.ts_entrada]
        s = self.sig.get(ts, np.nan)
        if not (s0 > 0 and s > 0):
            return None
        obj = n0 * min(1.0, s0 / s)
        if obj < p.notional * UMBRAL:
            self.n_reducciones += 1
            return obj
        return None


def correr(feat, risk, ajuste=None):
    return PortfolioSim({'BTC/USDT': feat}, risk_pct=risk, max_concurrent=1,
                        max_notional_pct=CAP, ajuste=ajuste).run()


def exposicion_real(feat, risk, ajuste=None):
    """Exposicion por vela registrando el notional vigente con un hook espia."""
    registro = {}

    class Espia:
        def __call__(self, par, ts, p):
            obj = ajuste(par, ts, p) if ajuste else None
            registro[ts] = (obj if obj is not None else p.notional, p.entrada)
            return obj
    res = correr(feat, risk, Espia())
    eq = res.equity
    close = feat['close'].reindex(eq.index)
    expo = pd.Series(0.0, index=eq.index)
    for ts, (n, e) in registro.items():
        if ts in expo.index:
            expo[ts] = n * close[ts] / e / eq[ts]
    return res, expo


def ratio(eq):
    años = (eq.index[-1] - eq.index[0]).days / 365.25
    return ((eq.iloc[-1] / eq.iloc[0]) ** (1 / años) - 1) / (1 - eq / eq.cummax()).max()


def main():
    rng = np.random.default_rng(SEED)
    df = cargar_btc()
    feat = features(df)
    sig = sigma_previa(df).to_dict()

    # sin hook = V2 exacto
    v2 = correr(feat, BASE)
    m = v2.metricas
    assert m['n'] == 165 and round(m['cagr'], 1) == 23.5 and round(m['dd'], 1) == 42.4, m

    aj = Ajuste(sig)
    e2, expo_e2 = exposicion_real(feat, BASE, aj)
    _, expo_v2 = exposicion_real(feat, BASE)
    x_e2 = expo_e2.mean()
    risk_m = BASE * x_e2 / expo_v2.mean()
    for _ in range(2):
        v2m, expo_m = exposicion_real(feat, risk_m)
        risk_m *= x_e2 / expo_m.mean()
    v2m, expo_m = exposicion_real(feat, risk_m)

    print(f"Reducciones de E2: {aj.n_reducciones} en {len(e2.trades)} trades")
    print(f"{'brazo':<16} {'riesgo':>7} {'expo':>6} {'CAGR':>7} {'DD':>6} {'CAGR/DD':>8} {'PF':>5}")
    for nombre, r, rk, x in (('E2', e2, BASE, x_e2), ('V2-m', v2m, risk_m, expo_m.mean()),
                             ('V2 4,5%', v2, BASE, expo_v2.mean())):
        mm = r.metricas
        print(f"{nombre:<16} {rk:>7.2%} {x:>6.3f} {mm['cagr']:>+6.1f}% {mm['dd']:>5.1f}% "
              f"{mm['cagr'] / mm['dd']:>8.2f} {mm['pf']:>5.2f}")

    # H2: bootstrap pareado por bloques de calendario sobre retornos diarios
    a = e2.equity.resample('1D').last().dropna()
    b = v2m.equity.resample('1D').last().reindex(a.index).ffill()
    ra, rb = a.pct_change().dropna().to_numpy(), b.pct_change().dropna().to_numpy()
    n = len(ra)
    años = n / 365.25
    print(f"Diario: E2 {ratio(a):.2f}  V2-m {ratio(b):.2f}")
    nb = int(np.ceil(n / BLOQUE))
    delta = []
    for _ in range(REPS // 1000):
        ii = (rng.integers(0, n, (1000, nb))[:, :, None] + np.arange(BLOQUE)).reshape(1000, -1)[:, :n] % n
        rat = []
        for r in (ra, rb):
            eq = np.hstack([np.ones((1000, 1)), np.cumprod(1 + r[ii], axis=1)])
            dd = (1 - eq / np.maximum.accumulate(eq, axis=1)).max(axis=1)
            rat.append((eq[:, -1] ** (1 / años) - 1) / dd)
        delta.append(rat[0] - rat[1])
    delta = np.concatenate(delta)
    p = float((delta <= 0).mean())
    print(f"\nH2: mediana delta CAGR/DD {np.median(delta):+.3f}  IC90 [{np.percentile(delta, 5):+.3f}, "
          f"{np.percentile(delta, 95):+.3f}]  p = P(delta<=0) = {p:.4f} -> "
          f"{'PASA' if p < 0.01 else 'NO pasa'} (alpha 0,01)")


if __name__ == '__main__':
    main()
