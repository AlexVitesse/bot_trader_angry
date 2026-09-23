"""Sizing de V2 con pronostico HAR-RV. Diseno pre-registrado en README.md.

Etapa 1: HAR vs control ATR como pronostico de la varianza de los 2 dias
siguientes (QLIKE fuera de muestra, bootstrap estacionario).
Etapa 2 (solo si la 1 pasa): brazos B / A / H en portfolio_sim, bootstrap
pareado por bloques de trades sobre el Sharpe por trade.

Uso: python experiments/vol_sizing/test_vol_sizing.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments' / 'portfolio_sim'))
from portfolio_sim import PortfolioSim, cargar_pares  # noqa: E402

H = 2              # horizonte en dias (mediana de duracion de un trade)
MIN_TRAIN = 365    # dias minimos de entrenamiento
ALPHA = 0.025      # Bonferroni por 2 comparaciones (etapa 2)
SEED = 20260923


# ----------------------------------------------------------------- etapa 1
def varianza_diaria() -> pd.Series:
    """Suma diaria de la varianza Garman-Klass de las velas 4h."""
    df = pd.read_parquet(ROOT / 'data' / 'BTC_USDT_4h_full.parquet')
    df = df[['open', 'high', 'low', 'close']].astype(float)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df[~df.index.duplicated(keep='last')].sort_index()
    gk = (0.5 * np.log(df.high / df.low) ** 2
          - (2 * np.log(2) - 1) * np.log(df.close / df.open) ** 2)
    g = gk.groupby(gk.index.floor('1D'))
    rv = g.sum()[g.count() == 6]          # solo dias completos
    return rv[rv > 0]


def tabla(rv: pd.Series, atr_dia: pd.Series) -> pd.DataFrame:
    t = pd.DataFrame({'rv': rv})
    t = t.asfreq('1D')                     # huecos -> NaN, no se interpolan
    t['rv_fut'] = (t.rv.shift(-1) + t.rv.shift(-2)) / 2   # media de los 2 dias siguientes
    t['y'] = np.log(t.rv_fut)
    t['d'] = np.log(t.rv)
    t['w'] = np.log(t.rv.rolling(5).mean())
    t['m'] = np.log(t.rv.rolling(22).mean())
    t['atr'] = np.log(atr_dia.reindex(t.index) ** 2)
    return t


def pronostico_expansivo(t: pd.DataFrame, cols: list) -> pd.Series:
    """OLS expansivo con reajuste mensual. F = exp(mu + s2/2): nivel de varianza."""
    X = t[cols].copy()
    X.insert(0, 'c', 1.0)
    ok = X.notna().all(axis=1)
    out = pd.Series(np.nan, index=t.index)
    for mes in pd.date_range(t.index[0], t.index[-1], freq='MS', tz='UTC'):
        fin_mes = mes + pd.offsets.MonthBegin(1)
        # objetivo conocido: t + H dias antes del inicio del mes
        tr = ok & t.y.notna() & (t.index <= mes - pd.Timedelta(days=H + 1))
        if tr.sum() < MIN_TRAIN:
            continue
        beta, *_ = np.linalg.lstsq(X[tr].values, t.y[tr].values, rcond=None)
        res = t.y[tr].values - X[tr].values @ beta
        s2 = res.var(ddof=len(beta))
        pr = ok & (t.index >= mes) & (t.index < fin_mes)
        out[pr] = np.exp(X[pr].values @ beta + s2 / 2)
    return out


def qlike(real, f):
    q = real / f
    return q - np.log(q) - 1


def etapa1():
    rv = varianza_diaria()
    d = cargar_pares(['BTC/USDT'])['BTC/USDT']
    atr_dia = d['atr_pct'].groupby(d.index.floor('1D')).last()
    t = tabla(rv, atr_dia)
    t['F_har'] = pronostico_expansivo(t, ['d', 'w', 'm'])
    t['F_atr'] = pronostico_expansivo(t, ['atr'])
    ev = t.dropna(subset=['F_har', 'F_atr', 'rv_fut'])
    l_har, l_atr = qlike(ev.rv_fut, ev.F_har), qlike(ev.rv_fut, ev.F_atr)
    dif = (l_atr - l_har).values
    bs = StationaryBootstrap(20, dif, seed=SEED)
    medias = bs.apply(np.mean, 20_000).ravel()
    p = float(np.mean(medias - dif.mean() >= dif.mean()))
    print('=== ETAPA 1: pronostico de la varianza de los 2 dias siguientes ===')
    print(f'dias evaluados: {len(ev)} ({ev.index[0].date()} -> {ev.index[-1].date()})')
    print(f'QLIKE medio  HAR {l_har.mean():.4f} | control ATR {l_atr.mean():.4f}')
    print(f'mejora HAR: {dif.mean():.4f} ({dif.mean() / l_atr.mean():+.1%} del control) '
          f'| p = {p:.4f}')
    corr = np.corrcoef(np.log(ev.rv_fut), np.log(ev.F_har))[0, 1]
    corr_a = np.corrcoef(np.log(ev.rv_fut), np.log(ev.F_atr))[0, 1]
    print(f'corr(log real, log pronostico): HAR {corr:.3f} | ATR {corr_a:.3f}')
    return p, t


# ----------------------------------------------------------------- etapa 2
def sharpe(r):
    return r.mean() / r.std(ddof=1)


def correr(datos, fc, z, riesgo):
    """fc: Series diaria de varianza pronosticada (o None = brazo B)."""
    def sizing(par, ts, equity, trail):
        if fc is None:
            return None
        f = fc.get(ts.floor('1D') - pd.Timedelta(days=1), np.nan)  # dia anterior
        if not np.isfinite(f):
            return None                    # sin pronostico -> sizing de B
        sigma = np.sqrt(H * f)             # std del retorno a 2 dias
        return equity * riesgo / (z * sigma)
    return PortfolioSim(datos, risk_pct=riesgo, max_concurrent=1,
                        sizing=sizing).run()


def calibrar_z(datos, fc, objetivo, riesgo):
    """z tal que el notional medio / equity iguala al de B (solo escala)."""
    z = 1.0
    for _ in range(6):
        res = correr(datos, fc, z, riesgo)
        m = np.mean([t['notional'] / (t['pnl'] / t['r']) for t in res.trades])
        z *= m / objetivo
    return z, correr(datos, fc, z, riesgo)


def etapa2(t):
    datos = cargar_pares(['BTC/USDT'])
    out = {}
    for riesgo in (0.02, 0.045):
        rB = correr(datos, None, 1.0, riesgo)
        obj = np.mean([x['notional'] / (x['pnl'] / x['r']) for x in rB.trades])
        zA, rA = calibrar_z(datos, t['F_atr'], obj, riesgo)
        zH, rH = calibrar_z(datos, t['F_har'], obj, riesgo)
        out[riesgo] = {'B': rB, 'A': rA, 'H': rH, 'z': (zA, zH)}

    res = out[0.02]
    ent = [x['ts_entrada'] for x in res['B'].trades]
    assert ent == [x['ts_entrada'] for x in res['H'].trades] == \
        [x['ts_entrada'] for x in res['A'].trades], 'los brazos deben tener los mismos trades'
    fc_ok = np.array([np.isfinite(t['F_har'].get(e.floor('1D') - pd.Timedelta(days=1), np.nan))
                      and np.isfinite(t['F_atr'].get(e.floor('1D') - pd.Timedelta(days=1), np.nan))
                      for e in ent])
    r = {k: np.array([x['r'] for x in res[k].trades])[fc_ok] for k in 'BAH'}

    rng = np.random.default_rng(SEED)
    n, L, reps = len(r['B']), 10, 20_000
    nb = int(np.ceil(n / L))
    d1, d2 = np.empty(reps), np.empty(reps)
    for k in range(reps):
        idx = (rng.integers(0, n, nb)[:, None] + np.arange(L)).ravel()[:n] % n
        sB, sA, sH = sharpe(r['B'][idx]), sharpe(r['A'][idx]), sharpe(r['H'][idx])
        d1[k], d2[k] = sH - sB, sH - sA
    obs1 = sharpe(r['H']) - sharpe(r['B'])
    obs2 = sharpe(r['H']) - sharpe(r['A'])
    p1 = float(np.mean(d1 - obs1 >= obs1))
    p2 = float(np.mean(d2 - obs2 >= obs2))

    print('\n=== ETAPA 2: sizing (mismos trades, riesgo 2%) ===')
    print(f'trades con pronostico: {fc_ok.sum()} de {len(ent)}')
    print(f"Sharpe por trade  B {sharpe(r['B']):.4f} | A {sharpe(r['A']):.4f} | "
          f"H {sharpe(r['H']):.4f}")
    print(f'H1 (H > B): dif {obs1:+.4f}  p = {p1:.4f}  -> '
          f"{'PASA' if p1 < ALPHA else 'no pasa'} a {ALPHA}")
    print(f'H2 (H > A): dif {obs2:+.4f}  p = {p2:.4f}  -> '
          f"{'PASA' if p2 < ALPHA else 'no pasa'} a {ALPHA}")
    print('\nDescriptivas (historia completa, notional medio igualado a B):')
    print(f"{'riesgo':>6} {'brazo':>5} {'z':>6} {'n':>4} {'PF':>5} {'CAGR':>7} "
          f"{'DD':>6} {'peor':>7}")
    for riesgo, o in out.items():
        for k in 'BAH':
            m = o[k].metricas
            z = {'B': 1.0, 'A': o['z'][0], 'H': o['z'][1]}[k]
            peor = min(x['r'] for x in o[k].trades) * 100
            print(f"{riesgo:>6.1%} {k:>5} {z:>6.3f} {m['n']:>4} {m['pf']:>5.2f} "
                  f"{m['cagr']:>+6.1f}% {m['dd']:>5.1f}% {peor:>+6.2f}%")
    return p1, p2, out


if __name__ == '__main__':
    p, t = etapa1()
    if p >= 0.05:
        print('\nEtapa 1 NO pasa (p >= 0,05): regla de parada pre-registrada. Fin.')
        sys.exit(0)
    etapa2(t)
