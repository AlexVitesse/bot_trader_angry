"""Predictibilidad por fuente. Diseno pre-registrado en README.md (10 celdas).

Ridge fijo, walk-forward expansivo de 6 tramos, IC OOS (Spearman) contra un
null de rotacion circular del target (1.000 rotaciones), Holm sobre 10 celdas.

Uso: python experiments/predictibilidad_fuentes/test_predictibilidad_fuentes.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

DATA = Path(__file__).resolve().parents[2] / 'data'
ALPHA_RIDGE = 1.0
N_ROT = 1000
N_TRAMOS = 6
COSTE = 0.0012          # ida y vuelta por pata
MIN_MONEDAS = 8


# ------------------------------------------------------------------ carga
def ohlc(path, freq=None):
    d = pd.read_parquet(path)[['open', 'high', 'low', 'close']].astype(float)
    if d.index.tz is None:
        d.index = d.index.tz_localize('UTC')
    d = d[~d.index.duplicated(keep='last')].sort_index()
    if freq:
        d = d.resample(freq).agg({'open': 'first', 'high': 'max',
                                  'low': 'min', 'close': 'last'}).dropna()
    return d


def tecnicas(d, rets, vols):
    lr = np.log(d.close).diff()
    f = {f'ret{w}': np.log(d.close / d.close.shift(w)) for w in rets}
    f.update({f'vol{w}': lr.rolling(w).std() for w in vols})
    f['rango'] = np.log(d.high / d.low)
    return pd.DataFrame(f)


def target(close, h):
    return np.log(close.shift(-h) / close)


# ------------------------------------------------------------- ridge WF
def tramos(n):
    b = np.linspace(0, n, N_TRAMOS + 1).astype(int)
    return [(b[k], b[k + 1]) for k in range(1, N_TRAMOS)]


def ridge_fit_pred(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    A1 = np.column_stack([np.ones(len(A)), A])
    reg = ALPHA_RIDGE * np.eye(A1.shape[1])
    reg[0, 0] = 0
    beta = np.linalg.solve(A1.T @ A1 + reg, A1.T @ ytr)
    return np.column_stack([np.ones(len(B)), B]) @ beta


def spearman(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]


def wf_ts(X, y, h):
    """IC OOS, R2 OOS (vs media de train), y OOS."""
    preds, ys, bench = [], [], []
    for s, e in tramos(len(y)):
        tr = slice(0, max(s - h, 1))
        preds.append(ridge_fit_pred(X[tr], y[tr], X[s:e]))
        ys.append(y[s:e])
        bench.append(np.full(e - s, y[tr].mean()))
    p, yy, bm = map(np.concatenate, (preds, ys, bench))
    r2 = 1 - ((yy - p) ** 2).sum() / ((yy - bm) ** 2).sum()
    return spearman(p, yy), r2, yy


def celda_ts(nombre, feats, y, h):
    df = pd.concat([feats, y.rename('y')], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    X, yv = df.drop(columns='y').values, df['y'].values
    ic, r2, yoos = wf_ts(X, yv, h)
    n = len(yv)
    kmin = max(10 * h, int(0.05 * n))
    null = np.array([wf_ts(X, np.roll(yv, k), h)[0]
                     for k in np.linspace(kmin, n - kmin, N_ROT).astype(int)])
    p = (np.sum(null >= ic) + 1) / (N_ROT + 1)
    sigma = yoos.std()
    return dict(celda=nombre, n=len(yoos), n_eff=len(yoos) // h, ic=ic,
                r2=r2, p=p, sigma=sigma, desde=df.index[0], hasta=df.index[-1])


# ------------------------------------------------------ cross-sectional
def panel(freq, rets, vol, h):
    """Devuelve F (D,C,p), Y (D,C) demeaned por fecha, fechas."""
    feats, ys = {}, {}
    for f in sorted(glob.glob(str(DATA / '*_4h_full.parquet'))):
        coin = Path(f).name.split('_')[0]
        d = ohlc(f, freq)
        feats[coin] = tecnicas(d, rets, [vol])
        ys[coin] = target(d.close, h)
    fechas = sorted(set().union(*[v.index for v in feats.values()]))
    idx = pd.DatetimeIndex(fechas)
    cols = list(feats[next(iter(feats))].columns)
    F = np.stack([np.stack([feats[c][col].reindex(idx).values for c in feats], 1)
                  for col in cols], 2)
    Y = np.stack([ys[c].reindex(idx).values for c in ys], 1)
    ok_f = np.isfinite(F).all(2)
    # demean por fecha (sobre monedas con features validas)
    Fm = np.where(ok_f[..., None], F, np.nan)
    F = Fm - np.nanmean(Fm, axis=1, keepdims=True)
    return F, Y, ok_f, idx


def ic_por_fecha(P, Y, M):
    """Media del Spearman por fecha (filas con >= 3 pares validos)."""
    Pm = pd.DataFrame(np.where(M, P, np.nan)).rank(axis=1).values
    Ym = pd.DataFrame(np.where(M, Y, np.nan)).rank(axis=1).values
    Pm -= np.nanmean(Pm, 1, keepdims=True)
    Ym -= np.nanmean(Ym, 1, keepdims=True)
    num = np.nansum(Pm * Ym, 1)
    den = np.sqrt(np.nansum(Pm ** 2, 1) * np.nansum(Ym ** 2, 1))
    ok = (M.sum(1) >= 3) & (den > 0)
    return np.mean(num[ok] / den[ok])


def wf_cs(F, Yraw, ok_f, h):
    # target demeaned por fecha sobre los pares validos
    M = ok_f & np.isfinite(Yraw)
    M &= (M.sum(1) >= MIN_MONEDAS)[:, None]
    Y = np.where(M, Yraw, np.nan)
    Y = Y - np.nanmean(Y, 1, keepdims=True)
    filas = np.where(M.any(1))[0]
    P = np.full(Y.shape, np.nan)
    sse = sst = 0.0
    oos = np.zeros(Y.shape[0], bool)
    for s, e in tramos(len(filas)):
        d_tr, d_te = filas[:max(s - h, 1)], filas[s:e]
        mtr = np.zeros_like(M)
        mtr[d_tr] = M[d_tr]
        mte = np.zeros_like(M)
        mte[d_te] = M[d_te]
        pr = ridge_fit_pred(F[mtr], Y[mtr], F[mte])
        P[mte] = pr
        sse += ((Y[mte] - pr) ** 2).sum()
        sst += (Y[mte] ** 2).sum()
        oos[d_te] = True
    Moos = M & oos[:, None]
    return ic_por_fecha(P, Y, Moos), 1 - sse / sst, Y, Moos


def celda_cs(nombre, freq, rets, vol, h):
    F, Yraw, ok_f, idx = panel(freq, rets, vol, h)
    ic, r2, Y, Moos = wf_cs(F, Yraw, ok_f, h)
    D = len(idx)
    kmin = max(10 * h, int(0.05 * D))
    null = np.array([wf_cs(F, np.roll(Yraw, k, axis=0), ok_f, h)[0]
                     for k in np.linspace(kmin, D - kmin, N_ROT).astype(int)])
    p = (np.sum(null >= ic) + 1) / (N_ROT + 1)
    Yo = np.where(Moos, Y, np.nan)
    sigma = np.nanmean(np.nanstd(Yo[Moos.any(1)], axis=1))
    n_fechas = int(Moos.any(1).sum())
    return dict(celda=nombre, n=int(Moos.sum()), n_eff=n_fechas // h, ic=ic,
                r2=r2, p=p, sigma=sigma, desde=idx[0], hasta=idx[-1])


# ------------------------------------------------------------ derivados
def derivados(h):
    d = ohlc(DATA / 'BTC_USDT_4h_full.parquet')
    oi = pd.read_parquet(DATA / 'deriv_oi_btcusdt_5m.parquet')['oi'].astype(float)
    if oi.index.tz is None:
        oi.index = oi.index.tz_localize('UTC')
    oi = oi[oi > 0].sort_index()
    fu = pd.read_parquet(DATA / 'btc_v15_funding.parquet')['funding_rate'].astype(float)
    if fu.index.tz is None:
        fu.index = fu.index.tz_localize('UTC')
    fu = fu.sort_index()
    fz = (fu - fu.rolling(30).mean().shift(1)) / fu.rolling(30).std().shift(1)
    cierre = d.index + pd.Timedelta(hours=4)

    def asof(s):
        return pd.Series(s.reindex(s.index.union(cierre)).ffill().reindex(cierre).values,
                         index=d.index)
    oi_c = np.log(asof(oi))
    f = pd.DataFrame({
        'oi1': oi_c.diff(1), 'oi6': oi_c.diff(6), 'oi42': oi_c.diff(42),
        'fund': asof(fu), 'fund_z': asof(fz), 'fund_chg': asof(fu.diff()),
    })
    f = f[(f.index >= oi.index[0] + pd.Timedelta(days=8))
          & (f.index < fu.index[-1])]      # sin funding posterior: no ffill infinito
    return f, target(d.close, h)


# ------------------------------------------------------------------ main
def holm(ps):
    orden = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    acum = 0
    for rank, i in enumerate(orden):
        acum = max(acum, (m - rank) * ps[i])
        adj[i] = min(acum, 1.0)
    return adj


def main():
    res = []
    b1 = ohlc(DATA / 'btcusdt_1h.parquet')
    f1 = tecnicas(b1, [1, 3, 6, 12, 24, 72, 168], [24, 168])
    for h in (1, 24):
        res.append(celda_ts(f'BTC 1h tecnicas h={h}', f1, target(b1.close, h), h))
        print(res[-1], flush=True)
    bd = ohlc(DATA / 'btcusdt_1d_v15.parquet')
    fd = tecnicas(bd, [1, 3, 5, 10, 20, 60], [10, 30])
    for h in (1, 5):
        res.append(celda_ts(f'BTC 1d tecnicas h={h}', fd, target(bd.close, h), h))
        print(res[-1], flush=True)
    for h in (1, 6):
        res.append(celda_cs(f'Panel 4h cross-sect h={h}', None,
                            [1, 6, 42, 180], 42, h))
        print(res[-1], flush=True)
    for h in (1, 5):
        res.append(celda_cs(f'Panel 1d cross-sect h={h}', '1D',
                            [1, 5, 20, 60], 20, h))
        print(res[-1], flush=True)
    for h in (1, 6):
        f, y = derivados(h)
        res.append(celda_ts(f'BTC 4h derivados h={h}', f, y, h))
        print(res[-1], flush=True)

    t = pd.DataFrame(res)
    t['p_holm'] = holm(t.p.values)
    t['edge'] = t.ic * t.sigma * np.sqrt(2 / np.pi)
    t['cubre'] = t.edge > COSTE
    t['senal'] = (t.p_holm < 0.05) & (t.r2 > 0)
    print('\n=== RESUMEN (10 celdas, Holm a 0,05) ===')
    print(f"{'celda':<28}{'N':>8}{'N_eff':>7}{'R2 OOS':>10}{'IC OOS':>9}"
          f"{'p':>8}{'p Holm':>8}{'sigma':>8}{'edge':>9}{'cubre':>7}{'senal':>7}")
    for _, r in t.iterrows():
        print(f"{r.celda:<28}{r.n:>8}{r.n_eff:>7}{r.r2 * 100:>9.3f}%{r.ic:>9.4f}"
              f"{r.p:>8.3f}{r.p_holm:>8.3f}{r.sigma * 100:>7.2f}%{r.edge * 100:>8.4f}%"
              f"{'si' if r.cubre else 'no':>7}{'SI' if r.senal else 'no':>7}")
    print(f'\ncoste ida y vuelta por pata: {COSTE:.2%}  |  edge = IC*sigma*sqrt(2/pi)')


if __name__ == '__main__':
    sys.exit(main())
