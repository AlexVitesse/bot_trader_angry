"""Meta-labeling para el sizing de V2. Diseno pre-registrado en README.md
(commit 1a16532) + adenda de implementacion escrita antes de correr.

Uso: python experiments/meta_labeling/test_meta_labeling.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments' / 'portfolio_sim'))
sys.path.insert(0, str(ROOT))
from portfolio_sim import PortfolioSim, cargar_pares, _funding_btc, FUNDING_8H  # noqa: E402
from src import v2_engine as v2  # noqa: E402

P = v2.PARAMS_V2
COM = 0.0006
MAX_BARS = 60
PURGA = pd.Timedelta(hours=4 * 60)
YEARS = range(2021, 2027)
SEED = 20260923
BAR = pd.Timedelta(hours=4)


# ------------------------------------------------------------ datos y eventos
def preparar():
    d = cargar_pares(['BTC/USDT'])['BTC/USDT'].copy()
    ema = d['close'].resample('1D').last().dropna().ewm(span=200, adjust=False).mean().shift(1)
    d['dist_ema200'] = d['close'] / ema.reindex(d.index, method='ffill') - 1
    h, c, bw = d['high'], d['close'], d['bb_width']
    ev = pd.Series(False, index=d.index)
    for n in (20, 40, 55, 100):
        ev |= c > h.rolling(n).max().shift(1)
    hi20 = c > h.rolling(20).max().shift(1)
    for q in (0.10, 0.20, 0.30):
        thr = bw.rolling(P['f_bb_window']).quantile(q).shift(1)
        ev |= (bw.shift(1) < thr) & hi20
    fs = _funding_btc()
    fund = fs.reindex(d.index.union(fs.index)).ffill().reindex(d.index)
    fund[d.index < fs.index[0]] = np.nan
    d['fund'] = fund.fillna(FUNDING_8H)
    return d, ev


def etiquetar(d, ev):
    """Trade tipo A de V2 desde cada evento: entrada open t+1, gap, funding."""
    o, h, l, c = (d[k].values for k in ('open', 'high', 'low', 'close'))
    atr, fund = d['atr_pct'].values, d['fund'].values
    idx = d.index
    filas = []
    for i in np.flatnonzero(ev.values):
        e = i + 1
        if e >= len(d):
            continue
        trail = min(max(atr[i] * P['a_trail_atr_mult'], P['a_trail_floor_pct']),
                    P['a_trail_ceiling_pct'])
        entrada = o[e]
        stop, peak, fpag = entrada * (1 - trail), entrada, 0.0
        salida = sal_i = None
        # como portfolio_sim: la vela de entrada no se chequea; barras = j - e
        for j in range(e + 1, min(e + MAX_BARS + 1, len(d))):
            fpag += fund[j] * 0.5
            if l[j] <= stop:
                salida, sal_i = min(stop, o[j]), j
                break
            if j - e >= MAX_BARS:
                salida, sal_i = c[j], j
                break
            peak = max(peak, h[j])
            stop = max(stop, peak * (1 - trail))
        if salida is None:                 # sin resolver: se descarta
            continue
        pnl = (salida - entrada) / entrada - 2 * COM - fpag
        filas.append({'t': idx[i], 'ini': idx[e], 'fin': idx[sal_i], 'y': int(pnl > 0),
                      'pnl': pnl, 'atr_pct': atr[i], 'dist_ema200': d['dist_ema200'].iloc[i],
                      'bb_width': d['bb_width'].iloc[i]})
    return pd.DataFrame(filas).dropna(subset=['atr_pct', 'dist_ema200', 'bb_width'])


FEAT = ['atr_pct', 'dist_ema200', 'bb_width']


def unicidad(tr, idx_velas):
    """Unicidad media por evento (Lopez de Prado cap. 4), solo con los spans de tr."""
    pos = pd.Series(np.arange(len(idx_velas)), index=idx_velas)
    a, b = pos[tr['ini']].values, pos[tr['fin']].values
    conc = np.zeros(len(idx_velas) + 1)
    np.add.at(conc, a, 1)
    np.add.at(conc, b + 1, -1)
    conc = np.cumsum(conc)[:-1]
    return np.array([np.mean(1.0 / conc[x:y + 1]) for x, y in zip(a, b)])


# ------------------------------------------------------------------ etapa 1
def walk_forward(evs, idx_velas):
    oos, modelos = [], {}
    for Y in YEARS:
        ini = pd.Timestamp(f'{Y}-01-01', tz='UTC')
        tr = evs[evs['fin'] < ini - PURGA]
        te = evs[(evs['t'] >= ini) & (evs['t'] < pd.Timestamp(f'{Y + 1}-01-01', tz='UTC'))]
        if te.empty:
            continue
        mu, sd = tr[FEAT].mean(), tr[FEAT].std()
        w = unicidad(tr, idx_velas)
        m = LogisticRegression(C=1.0).fit((tr[FEAT] - mu) / sd, tr['y'], sample_weight=w)
        p_tr = m.predict_proba((tr[FEAT] - mu) / sd)[:, 1]
        modelos[Y] = (m, mu, sd, np.sort(p_tr))
        te = te.assign(p=m.predict_proba((te[FEAT] - mu) / sd)[:, 1], year=Y)
        oos.append(te)
        print(f'  {Y}: train {len(tr):5d} (N_eff {w.sum():7.1f}) | test {len(te):4d} | '
              f'AUC {roc_auc_score(te["y"], te["p"]) if te["y"].nunique() > 1 else np.nan:.3f}')
    return pd.concat(oos), modelos


def auc_bloques(oos, reps=10_000):
    rng = np.random.default_rng(SEED)
    t0 = oos['t'].min()
    blq = ((oos['t'] - t0) // pd.Timedelta(days=30)).values
    ids = np.unique(blq)
    grupos = [np.flatnonzero(blq == b) for b in ids]
    y, p = oos['y'].values, oos['p'].values
    auc = roc_auc_score(y, p)
    out = np.empty(reps)
    for k in range(reps):
        sel = np.concatenate([grupos[g] for g in rng.integers(0, len(grupos), len(grupos))])
        out[k] = roc_auc_score(y[sel], p[sel]) if len(np.unique(y[sel])) > 1 else 0.5
    return auc, float(np.mean(out - auc >= auc - 0.5)), len(grupos)


# ------------------------------------------------------------------ etapa 2
def sharpe(r):
    return r.mean() / r.std(ddof=1)


def etapa2(d, modelos):
    datos = cargar_pares(['BTC/USDT'])
    mult = {}
    for ts in d.index:
        Y = ts.year
        if Y in modelos:
            m, mu, sd, ecdf = modelos[Y]
            x = ((d.loc[[ts], FEAT] - mu) / sd)
            if x.notna().all(axis=1).iloc[0]:
                p = m.predict_proba(x)[:, 1][0]
                mult[ts] = 0.5 + np.searchsorted(ecdf, p, side='right') / len(ecdf)

    def sizing(par, ts, equity, trail):
        mm = mult.get(ts - BAR)            # vela de senal
        return None if mm is None else equity * 0.02 / trail * mm

    rB = PortfolioSim(datos, risk_pct=0.02, max_concurrent=1).run()
    rM = PortfolioSim(datos, risk_pct=0.02, max_concurrent=1, sizing=sizing).run()
    eB = [(t['ts_entrada'], t['ts_salida']) for t in rB.trades]
    eM = [(t['ts_entrada'], t['ts_salida']) for t in rM.trades]
    assert eB == eM, 'los brazos deben tener los mismos trades'
    en = np.array([(t['ts_entrada'] - BAR) in mult for t in rB.trades])
    rb = np.array([t['r'] for t in rB.trades])[en]
    rm = np.array([t['r'] for t in rM.trades])[en]
    mm = np.array([mult[t['ts_entrada'] - BAR] for t in rB.trades if (t['ts_entrada'] - BAR) in mult])

    rng = np.random.default_rng(SEED)
    n, L, reps = len(rb), 10, 20_000
    nb = int(np.ceil(n / L))
    dif = np.empty(reps)
    for k in range(reps):
        ix = (rng.integers(0, n, nb)[:, None] + np.arange(L)).ravel()[:n] % n
        dif[k] = sharpe(rm[ix]) - sharpe(rb[ix])
    obs = sharpe(rm) - sharpe(rb)
    p = float(np.mean(dif - obs >= obs))

    def tramo(res):
        eq = res.equity[res.equity.index >= pd.Timestamp('2021-01-01', tz='UTC')]
        dd = float((1 - eq / eq.cummax()).max() * 100)
        años = (eq.index[-1] - eq.index[0]).days / 365.25
        cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / años) - 1) * 100
        return cagr, dd

    print('\n=== ETAPA 2: sizing meta-labeling vs V2 (mismos trades) ===')
    print(f'trades 2021-2026: {n} de {len(rB.trades)} | multiplicador medio {mm.mean():.3f} '
          f'(min {mm.min():.2f}, max {mm.max():.2f})')
    print(f'corr(m, r_V2) = {np.corrcoef(mm, rb)[0, 1]:+.3f}')
    print(f'Sharpe por trade V2 {sharpe(rb):.4f} | meta {sharpe(rm):.4f} | dif {obs:+.4f} | p = {p:.4f}')
    for nom, r_, res in (('V2', rb, rB), ('meta', rm, rM)):
        w, l = r_[r_ > 0].sum(), -r_[r_ <= 0].sum()
        cagr, dd = tramo(res)
        print(f'  {nom:5s} PF {w / l:.2f} | CAGR 2021-26 {cagr:+.1f}% | DD 2021-26 {dd:.1f}%')
    return p


if __name__ == '__main__':
    d, ev = preparar()
    evs = etiquetar(d, ev)
    print('=== ETAPA 1: AUC fuera de muestra del meta-modelo ===')
    print(f'velas-evento: {int(ev.sum())} | eventos con etiqueta: {len(evs)} | '
          f'tasa y=1: {evs["y"].mean():.3f}')
    w_all = unicidad(evs, d.index)
    print(f'N_eff (unicidad, todos los eventos): {w_all.sum():.1f}')
    oos, modelos = walk_forward(evs, d.index)
    auc, p1, nbq = auc_bloques(oos)
    print(f'AUC OOS pooled 2021-2026: {auc:.4f} | eventos {len(oos)} | bloques de 30 d: {nbq} '
          f'| p = {p1:.4f}')
    for Y, m in modelos.items():
        print(f'  coef {Y}: ' + ', '.join(f'{f} {c:+.3f}' for f, c in zip(FEAT, m[0].coef_[0])))
    if p1 >= 0.05:
        print('\nEtapa 1 NO pasa (p >= 0,05): regla de parada pre-registrada. Fin.')
        sys.exit(0)
    etapa2(d, modelos)
