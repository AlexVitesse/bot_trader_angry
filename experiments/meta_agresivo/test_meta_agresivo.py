"""E9: logistica (meta-labeling + macro E7) como decisor del agresivo en BTC.
Pre-registro en README.md.

Uso: python experiments/meta_agresivo/test_meta_agresivo.py [n_series]
"""
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for d in ('agresivo', 'portfolio_sim', 'bootstrap_bloques', 'especialistas'):
    sys.path.insert(0, str(ROOT / 'experiments' / d))
from src import agresivo_engine as ag                       # noqa: E402
from backtest import simular, DESDE                         # noqa: E402
from portfolio_sim import PortfolioSim, cargar_pares        # noqa: E402
from test_bloques import cargar_btc, sintetica              # noqa: E402
from test_especialistas import macro_diario                 # noqa: E402

PAR = 'BTC/USDT'
TEC = ['atr_pct', 'bb_width', 'dist_ema_d']
MAC = ['oro_z_d', 'dxy_20_d', 'spx_dd_d']
L_VELAS, BLOQUE, REPS = 180, 30, 10_000
SEED = 20261020
DF = None


def eventos(btc):
    f = ag.features(btc, btc)
    ev = ag.eventos_panel({PAR: f})
    c = btc['close'].resample('1D').last().dropna()
    m = macro_diario(c)
    # vela que abre en t cierra en t+4h: ultimo dia UTC completo con cierre <= t+4h
    dia = (ev['t'] + ag.BAR).dt.normalize() - pd.Timedelta(days=1)
    for k in ('oro_z', 'dxy_20', 'spx_dd'):
        ev[f'{k}_d'] = m[k].reindex(dia).to_numpy() * ev['dir'].to_numpy()
    return f, ev.reset_index(drop=True)


def _b(tr):
    win = tr.loc[tr['r_R'] > 0, 'r_R'].mean()
    loss = -tr.loc[tr['r_R'] <= 0, 'r_R'].mean()
    return float(win / loss)


def predicciones(ev, brazo):
    """p y b de cada evento con el modelo de su mes (igual que backtest.predicciones)."""
    feats = {'B': TEC + MAC, 'B0': TEC}.get(brazo)
    meses = pd.date_range(DESDE, ev['t'].max(), freq='MS', tz='UTC')
    ev = ev.copy()
    ev['p'], ev['b'] = np.nan, np.nan
    for m in meses:
        sel = (ev['t'] >= m) & (ev['t'] < m + pd.offsets.MonthBegin(1))
        if not sel.any():
            continue
        tr = ev[ev['fin'] < m - ag.PURGA_VELAS * ag.BAR]
        if brazo == 'A':
            bundle = ag.entrenar(tr)
            ev.loc[sel, 'p'] = ag.probabilidad(bundle, ev.loc[sel, ag.FEAT].values)
            ev.loc[sel, 'b'] = bundle['b']
            continue
        tr = tr.dropna(subset=feats).sort_values('t').reset_index(drop=True)
        mu, sd = tr[feats].mean(), tr[feats].std()
        mod = LogisticRegression(C=1.0).fit((tr[feats] - mu) / sd, tr['y'],
                                            sample_weight=ag.unicidad(tr))
        ev.loc[sel, 'p'] = mod.predict_proba((ev.loc[sel, feats] - mu) / sd)[:, 1]
        ev.loc[sel, 'b'] = _b(tr)
    return ev.dropna(subset=['p'])


def ratio(eq):
    años = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / años) - 1
    dd = float((1 - eq / eq.cummax()).max())
    return cagr, dd, cagr / dd if dd > 0 else np.nan


def expo(eq, tramos):
    """Exposicion media notional/equity por vela (0 fuera), tramos = (ini, fin, notional)."""
    x = pd.Series(0.0, index=eq.index)
    for ini, fin, n in tramos:
        m = (eq.index >= ini) & (eq.index < fin)
        x[m] = n / eq[m]
    return float(x.mean())


def correr_b(btc):
    f, ev = eventos(btc)
    oos = predicciones(ev, 'B')
    eq, tr = simular({PAR: btc}, {PAR: f}, oos)
    return eq, tr, oos


def v2(btc_datos, risk):
    res = PortfolioSim(btc_datos, risk_pct=risk, max_concurrent=1, max_notional_pct=2.0).run(desde=DESDE)
    tramos = [(t['ts_entrada'], t['ts_salida'], t['notional']) for t in res.trades]
    return res.equity, expo(res.equity, tramos)


def auc_boot(y, p, t, rng):
    blq = ((t - t.min()).dt.days // BLOQUE).to_numpy()
    y, p = y.to_numpy(), p.to_numpy()
    grupos = [np.flatnonzero(blq == b) for b in np.unique(blq)]
    obs = roc_auc_score(y, p)
    a = []
    for _ in range(REPS):
        ii = np.concatenate([grupos[j] for j in rng.integers(0, len(grupos), len(grupos))])
        if y[ii].min() != y[ii].max():
            a.append(roc_auc_score(y[ii], p[ii]))
    a = np.array(a)
    return obs, float(np.mean(a - obs >= obs - 0.5))


def _init(df):
    global DF
    DF = df


def _worker(seed):
    try:
        eq, _, _ = correr_b(sintetica(DF, L_VELAS, seed))
        return ratio(eq)[0]
    except Exception as e:                         # una serie rota no tumba la corrida
        return f'error: {e}'


def main():
    n_series = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rng = np.random.default_rng(SEED)
    btc = cargar_btc()
    f, ev = eventos(btc)
    print(f"Eventos BTC etiquetados: {len(ev)} | por tipo {ev['tipo'].value_counts().to_dict()} "
          f"| y=1 {ev['y'].mean():.3f}")

    # ---------------- etapa 1
    oos = {b: predicciones(ev, b) for b in ('A', 'B', 'B0')}
    comun = oos['A'].index.intersection(oos['B'].index).intersection(oos['B0'].index)
    print(f"\nEtapa 1: {len(comun)} eventos fuera de muestra ({DESDE.date()} ->)")
    for b in ('A', 'B', 'B0'):
        o = oos[b].loc[comun]
        por_año = {a: roc_auc_score(g['y'], g['p']) for a, g in o.groupby(o['t'].dt.year)
                   if g['y'].nunique() == 2}
        print(f"  {b:<3} AUC {roc_auc_score(o['y'], o['p']):.3f} | "
              + "  ".join(f"{a}:{v:.2f}" for a, v in por_año.items()))
    o = oos['B'].loc[comun]
    auc_b, p1 = auc_boot(o['y'], o['p'], o['t'], rng)
    print(f"  B: AUC {auc_b:.3f}  p(AUC<=0,5) bloques 30d = {p1:.4f} -> "
          f"{'PASA' if p1 < 0.05 else 'NO pasa'} (p<0,05)")

    # ---------------- etapa 2
    res = {}
    for b in ('A', 'B', 'B0'):
        eq, tr = simular({PAR: btc}, {PAR: f}, oos[b])
        ex = expo(eq, [(t, s, n) for t, s, n in zip(tr['ts'], tr['salida'], tr['notional'])]) if len(tr) else 0.0
        res[b] = (eq, tr, ex)
    datos = cargar_pares([PAR])
    eq_v2, ex_v2 = v2(datos, 0.045)
    risk = 0.045
    for _ in range(2):
        risk *= res['B'][2] / ex_v2 if ex_v2 else 1
        eq_v2m, ex_v2 = v2(datos, risk)
    print(f"\nEtapa 2 (cartera {DESDE.date()} -> {btc.index[-1].date()}):")
    print(f"  {'brazo':<16} {'CAGR':>7} {'DD':>6} {'CAGR/DD':>7} {'expo':>6} {'trades/año':>10} {'PF':>5} {'WR':>5}")
    filas = {}
    for nombre, (eq, tr, ex) in (('A (HGB actual)', res['A']), ('B (logistica)', res['B']),
                                 ('B0 (sin macro)', res['B0'])):
        cg, dd, rt = ratio(eq)
        años = (eq.index[-1] - eq.index[0]).days / 365.25
        r = tr['r'] if len(tr) else pd.Series(dtype=float)
        pf = r[r > 0].sum() / -r[r <= 0].sum() if (r <= 0).any() else np.nan
        filas[nombre] = rt
        print(f"  {nombre:<16} {cg:>+7.1%} {dd:>6.1%} {rt:>7.2f} {ex:>6.3f} {len(tr) / años:>10.0f} "
              f"{pf:>5.2f} {(r > 0).mean():>5.1%}")
    for nombre, eq, ex in ((f'V2 {risk:.2%} (=expo)', eq_v2m, ex_v2), ('V2 4,5%', eq_v2, None)):
        cg, dd, rt = ratio(eq)
        filas[nombre] = rt
        print(f"  {nombre:<16} {cg:>+7.1%} {dd:>6.1%} {rt:>7.2f} {ex if ex is not None else float('nan'):>6.3f}")

    tr_b = res['B'][1]
    if len(tr_b):
        for col in ('tipo', 'dir'):
            print(f"\n  B por {col}:")
            print(tr_b.groupby(col).agg(n=('r', 'size'), wr=('r', lambda x: (x > 0).mean()),
                                        suma_r=('r', 'sum'), riesgo=('riesgo', 'mean')).round(3).to_string())
        eqb = res['B'][0]
        anual = eqb.resample('YE').last()
        print("\n  B retorno por año:", (anual / anual.shift(1).fillna(eqb.iloc[0]) - 1).round(3)
              .set_axis(anual.index.year).to_dict())

    c1 = filas['B (logistica)'] > filas['A (HGB actual)']
    c2 = filas['B (logistica)'] > filas[f'V2 {risk:.2%} (=expo)']
    cagr_obs = ratio(res['B'][0])[0]
    print(f"\n  (1) CAGR/DD B > A: {'si' if c1 else 'NO'}   (2) B > V2 a igual expo: {'si' if c2 else 'NO'}")

    # ---------------- null sintetico
    print(f"\nNull sintetico: {n_series} series, bloques de {L_VELAS} velas ...", flush=True)
    with Pool(initializer=_init, initargs=(btc,)) as pool:
        out = pool.map(_worker, [SEED + k for k in range(n_series)])
    errores = [x for x in out if isinstance(x, str)]
    cg = np.array([x for x in out if not isinstance(x, str)])
    p3 = float((cg >= cagr_obs).mean())
    print(f"  series validas {len(cg)} (errores {len(errores)}{': ' + errores[0] if errores else ''})")
    print(f"  CAGR B observado {cagr_obs:+.1%} | sinteticas mediana {np.median(cg):+.1%} "
          f"p90 {np.percentile(cg, 90):+.1%}  p = {p3:.3f} -> {'cumple' if p3 < 0.05 else 'NO cumple'}")
    print(f"\nEtapa 1: {'PASA' if p1 < 0.05 else 'NO pasa'} | "
          f"H9: {'PASA' if p1 < 0.05 and c1 and c2 and p3 < 0.05 else 'NO pasa'}")


if __name__ == '__main__':
    main()
