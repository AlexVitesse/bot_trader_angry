"""E6: mezcla 50% V2 + 50% BTC spot (exposicion 0,176). Pre-registro en README.md.

Uso: python experiments/mixta_v2_btc/test_mixta.py [n_series]
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
for d in ('portfolio_sim', 'bootstrap_bloques', 'benchmark_bh'):
    sys.path.insert(0, str(ROOT / 'experiments' / d))
from portfolio_sim import PortfolioSim                  # noqa: E402
from test_bloques import cargar_btc, features, sintetica  # noqa: E402
from test_benchmark import bh, mixta                    # noqa: E402

W_BTC = 0.176
BLOQUE, REPS, L_VELAS = 30, 20_000, 180
SEED = 20261019
DF = None


def ratio(eq: pd.Series) -> float:
    años = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = float((1 - eq / eq.cummax()).max())
    return ((eq.iloc[-1] / eq.iloc[0]) ** (1 / años) - 1) / dd if dd > 0 else np.nan


def curvas(df):
    """(V2 diario, mezcla diaria, BTC-w diario) sobre una serie 4h."""
    res = PortfolioSim({'BTC/USDT': features(df)}, risk_pct=0.045, max_concurrent=1,
                       max_notional_pct=2.0).run()
    v2 = res.equity.resample('1D').last().dropna()
    close = df['close'].resample('1D').last().reindex(v2.index).ffill()
    btc = bh(close, pd.Series(W_BTC, index=v2.index))         # spot, sin funding
    return v2, mixta(btc, v2), btc


def sin_deriva(df, f):
    lc = np.log(df['close'].to_numpy())
    mu = (lc[-1] - np.log(df['open'].iloc[0])) / len(lc)
    k = np.exp(-f * mu * np.arange(len(lc)))
    out = df.copy()
    for c in ('open', 'high', 'low', 'close'):
        out[c] = df[c].to_numpy() * k
    return out


def _init(df):
    global DF
    DF = df


def _worker(seed):
    s = sintetica(DF, L_VELAS, seed)
    out = []
    for f in (0.0, 0.5, 1.0):
        v2, mx, btc = curvas(sin_deriva(s, f) if f else s)
        out.append((ratio(mx), ratio(v2), ratio(btc)))
    return out


def main():
    n_series = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rng = np.random.default_rng(SEED)
    df = cargar_btc()

    v2, mx, btc = curvas(df)
    for nombre, c in (('V2', v2), ('Mezcla', mx), (f'BTC spot {W_BTC}', btc)):
        años = (c.index[-1] - c.index[0]).days / 365.25
        cg = (c.iloc[-1] / c.iloc[0]) ** (1 / años) - 1
        dd = float((1 - c / c.cummax()).max())
        print(f"  {nombre:<16} CAGR {cg:>+6.1%}  DD {dd:>5.1%}  CAGR/DD {cg / dd:>5.2f}")
    d_obs = ratio(mx) - ratio(v2)

    # (a) bootstrap pareado por bloques de calendario
    ra = mx.pct_change().dropna().to_numpy()
    rb = v2.pct_change().dropna().to_numpy()
    n, años = len(ra), len(ra) / 365.25
    nb = int(np.ceil(n / BLOQUE))
    deltas = []
    for _ in range(REPS // 1000):
        ii = (rng.integers(0, n, (1000, nb))[:, :, None] + np.arange(BLOQUE)).reshape(1000, -1)[:, :n] % n
        rat = []
        for r in (ra, rb):
            eq = np.hstack([np.ones((1000, 1)), np.cumprod(1 + r[ii], axis=1)])
            dd = (1 - eq / np.maximum.accumulate(eq, axis=1)).max(axis=1)
            rat.append((eq[:, -1] ** (1 / años) - 1) / dd)
        deltas.append(rat[0] - rat[1])
    deltas = np.concatenate(deltas)
    p_a = float((deltas <= 0).mean())
    print(f"\n(a) delta CAGR/DD observado {d_obs:+.3f}  bootstrap mediana {np.median(deltas):+.3f}  "
          f"IC90 [{np.percentile(deltas, 5):+.3f}, {np.percentile(deltas, 95):+.3f}]  p = {p_a:.4f}  "
          f"-> {'cumple' if p_a < 0.01 else 'NO cumple'} (p<0,01)")

    # (b) + estres de deriva
    print(f"\nNull sintetico: {n_series} series, bloques de {L_VELAS} velas ...")
    with Pool(initializer=_init, initargs=(df,)) as pool:
        out = np.array(pool.map(_worker, range(n_series)))      # (series, 3 derivas, 3)
    frac_b = None
    for j, nombre in enumerate(('deriva real', 'deriva 1/2', 'deriva 0')):
        mxr, v2r, btr = out[:, j, 0], out[:, j, 1], out[:, j, 2]
        ok = ~np.isnan(mxr) & ~np.isnan(v2r)
        frac = float((mxr[ok] > v2r[ok]).mean())
        if j == 0:
            frac_b = frac
        print(f"  {nombre:<12} delta>0 en {frac:>5.1%}  mediana CAGR/DD: mezcla {np.nanmedian(mxr):+.2f}  "
              f"V2 {np.nanmedian(v2r):+.2f}  BTC {np.nanmedian(btr):+.2f}  (series {ok.sum()})")
    c_b = frac_b >= 0.90
    print(f"\n(b) delta>0 en {frac_b:.1%} de las series con deriva real -> {'cumple' if c_b else 'NO cumple'} (>=90%)")
    print(f"H6: {'PASA' if p_a < 0.01 and c_b else 'NO pasa'}")


if __name__ == '__main__':
    main()
