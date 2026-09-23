"""Walk-forward mensual del motor agresivo (src/agresivo_engine.py) con cartera.

Diseno fijado antes de correr: experiments/agresivo/README.md.
Uso: python experiments/agresivo/backtest.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'portfolio_sim'))
from src import agresivo_engine as ag  # noqa: E402

DESDE = pd.Timestamp('2021-01-01', tz='UTC')
CAPITAL, MAX_POS, MAX_DIR, LEV, MAX_NOT = 10_000.0, 3, 2, 5, 2.5
SEED = 20260923


def cargar():
    out = {}
    for par in ag.PARES:
        df = pd.read_parquet(ROOT / 'data' / f"{par.split('/')[0]}_USDT_4h_full.parquet")
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        out[par] = df[~df.index.duplicated(keep='last')].sort_index()
    return out


def predicciones(ev):
    """p de cada evento con el modelo de su mes (entrenado solo con el pasado)."""
    meses = pd.date_range(DESDE, ev['t'].max(), freq='MS', tz='UTC')
    ev = ev.copy()
    ev['p'], ev['b'] = np.nan, np.nan
    for m in meses:
        tr = ev[ev['fin'] < m - ag.PURGA_VELAS * ag.BAR]
        sel = (ev['t'] >= m) & (ev['t'] < m + pd.offsets.MonthBegin(1))
        if not sel.any():
            continue
        bundle = ag.entrenar(tr)
        ev.loc[sel, 'p'] = ag.probabilidad(bundle, ev.loc[sel, ag.FEAT].values)
        ev.loc[sel, 'b'] = bundle['b']
    return ev.dropna(subset=['p'])


def simular(datos, feats, oos):
    idx = sorted(set().union(*[d.index[d.index >= DESDE] for d in datos.values()]))
    pos_en = {p: {ts: i for i, ts in enumerate(f.index)} for p, f in feats.items()}
    señales = {ts: g for ts, g in oos.groupby('t')}
    cash, pico = CAPITAL, CAPITAL
    abiertas, pendientes, trades, curva = {}, [], [], []
    for ts in idx:
        # 1. marcar a mercado
        unr = 0.0
        for par, q in abiertas.items():
            i = pos_en[par].get(ts)
            if i is not None:
                q['ultimo'] = feats[par]['close'].iloc[i]
            unr += q['dir'] * (q['ultimo'] - q['ent']) / q['ent'] * q['notional']
        eq = cash + unr
        curva.append((ts, eq))
        pico = max(pico, eq)
        if eq <= 0:
            break
        # 2. salidas con el stop de la vela anterior
        for par in list(abiertas):
            i = pos_en[par].get(ts)
            if i is None:
                continue
            f, q = feats[par], abiertas[par]
            o, h, l, c = (f[k].iloc[i] for k in ('open', 'high', 'low', 'close'))
            q['barras'] += 1
            q['fund'] += q['dir'] * ag.FUND_8H * 0.5 * q['notional']
            d, sal = q['dir'], None
            if (d == 1 and l <= q['stop']) or (d == -1 and h >= q['stop']):
                sal = min(q['stop'], o) if d == 1 else max(q['stop'], o)
            elif q['barras'] >= ag.MAX_BARS:
                sal = c
            if sal is None:
                if d == 1:
                    q['ext'] = max(q['ext'], h)
                    q['stop'] = max(q['stop'], q['ext'] * (1 - q['trail']))
                else:
                    q['ext'] = min(q['ext'], l)
                    q['stop'] = min(q['stop'], q['ext'] * (1 + q['trail']))
                continue
            pnl = (d * (sal - q['ent']) / q['ent'] - 2 * ag.COM) * q['notional'] - q['fund']
            cash += pnl
            trades.append({**{k: q[k] for k in ('par', 'tipo', 'dir', 'p', 'riesgo', 'ts')},
                           'salida': ts, 'pnl': pnl, 'r': pnl / q['eq_ent']})
            del abiertas[par]
        # 3. pendientes al open de esta vela
        for s in pendientes:
            par = s['par']
            i = pos_en[par].get(ts)
            if i is None or par in abiertas or len(abiertas) >= MAX_POS:
                continue
            if sum(q['dir'] == s['dir'] for q in abiertas.values()) >= MAX_DIR:
                continue
            ent = feats[par]['open'].iloc[i]
            notional = min(eq * s['riesgo'] / s['trail'], eq * MAX_NOT)
            usado = sum(q['notional'] for q in abiertas.values()) / LEV
            if notional / LEV > eq - usado:
                continue
            abiertas[par] = {'par': par, 'tipo': s['tipo'], 'dir': s['dir'], 'p': s['p'],
                             'riesgo': s['riesgo'], 'ts': ts, 'ent': ent, 'ultimo': ent,
                             'notional': notional, 'trail': s['trail'], 'ext': ent,
                             'stop': ent * (1 - s['dir'] * s['trail']), 'barras': 0,
                             'fund': 0.0, 'eq_ent': eq}
        pendientes = []
        # 4. senales en la vela cerrada (riesgo con el DD de ahora)
        dd = 1 - eq / pico
        if ts in señales:
            mejores = {}
            for s in señales[ts].itertuples():
                if s.par in abiertas:
                    continue
                r = ag.riesgo_de(s.p, s.b, dd)
                if r > 0 and r > mejores.get(s.par, {}).get('riesgo', 0):
                    mejores[s.par] = {'par': s.par, 'tipo': s.tipo, 'dir': s.dir, 'p': s.p,
                                      'riesgo': r, 'trail': s.trail}
            pendientes = sorted(mejores.values(), key=lambda x: -x['riesgo'])
    eq = pd.Series([e for _, e in curva], index=pd.DatetimeIndex([t for t, _ in curva]))
    return eq, pd.DataFrame(trades)


def metricas(eq, r):
    años = (eq.index[-1] - eq.index[0]).days / 365.25
    serie = pd.concat([pd.Series([CAPITAL]), eq.reset_index(drop=True)])
    dd = float((1 - serie / serie.cummax()).max())
    w, l = r[r > 0].sum(), -r[r <= 0].sum()
    return {'cagr': (eq.iloc[-1] / CAPITAL) ** (1 / años) - 1, 'dd': dd,
            'pf': w / l if l else np.inf, 'wr': (r > 0).mean(), 'n': len(r),
            'por_año': len(r) / años}


def p_bloques(r, L=10, reps=20_000):
    rng = np.random.default_rng(SEED)
    n = len(r)
    nb = int(np.ceil(n / L))
    mu = np.array([r[(rng.integers(0, n, nb)[:, None] + np.arange(L)).ravel()[:n] % n].mean()
                   for _ in range(reps)])
    return float(np.mean(mu - r.mean() >= r.mean()))


def main():
    datos = cargar()
    btc = datos['BTC/USDT']
    feats = {p: ag.features(d, btc) for p, d in datos.items()}
    ev = ag.eventos_panel(feats)
    print(f'eventos etiquetados: {len(ev)} | por tipo: {ev.tipo.value_counts().to_dict()} '
          f'| y=1: {ev.y.mean():.3f}')
    oos = predicciones(ev)
    auc = roc_auc_score(oos['y'], oos['p'])
    print(f'AUC OOS (todos los candidatos 2021-2026, n={len(oos)}): {auc:.4f}')
    for y_, g in oos.groupby(oos['t'].dt.year):
        print(f'  {y_}: n={len(g):5d} AUC {roc_auc_score(g.y, g.p):.3f} | p medio {g.p.mean():.3f} '
              f'| y medio {g.y.mean():.3f}')
    cal = oos.assign(bin=pd.cut(oos['p'], [0, .3, .4, .5, .6, 1]))
    print('calibracion (p pronosticada vs tasa real):')
    print(cal.groupby('bin', observed=True).agg(n=('y', 'size'), p=('p', 'mean'), real=('y', 'mean'),
                                                 rR=('r_R', 'mean')).round(3).to_string())

    eq, tr = simular(datos, feats, oos)
    r = tr['r'].values
    m = metricas(eq, r)
    print('\n=== CARTERA AGRESIVA (walk-forward 2021-01 -> fin de datos) ===')
    print(f"CAGR {m['cagr']:+.1%} | DD {m['dd']:.1%} | PF {m['pf']:.2f} | WR {m['wr']:.1%} | "
          f"trades {m['n']} ({m['por_año']:.0f}/año) | riesgo medio {tr['riesgo'].mean():.2%} | "
          f"final x{eq.iloc[-1] / CAPITAL:.2f}")
    print(f'p bootstrap bloques (mean r > 0): {p_bloques(r):.4f}')
    for col in ('tipo', 'dir'):
        print(f'\npor {col}:')
        print(tr.groupby(col).agg(n=('r', 'size'), wr=('r', lambda x: (x > 0).mean()),
                                  suma_r=('r', 'sum'), riesgo=('riesgo', 'mean')).round(3).to_string())
    print('\npor año:')
    anual = eq.resample('YE').last()
    ret = anual / anual.shift(1).fillna(CAPITAL) - 1
    ret.index = ret.index.year
    por_año = tr.groupby(tr['ts'].dt.year).agg(n=('r', 'size'), wr=('r', lambda x: (x > 0).mean()))
    print(por_año.join(ret.rename('ret'), how='outer').round(3).to_string())

    # comparaciones en la misma ventana
    from portfolio_sim import PortfolioSim, cargar_pares
    v2 = PortfolioSim(cargar_pares(['BTC/USDT']), risk_pct=0.045, max_concurrent=1).run(desde=DESDE)
    mv = v2.metricas
    c = btc['close'][btc.index >= DESDE]
    años = (c.index[-1] - c.index[0]).days / 365.25
    bh_cagr = (c.iloc[-1] / c.iloc[0]) ** (1 / años) - 1
    bh_dd = float((1 - c / c.cummax()).max())
    print('\n=== COMPARACION (misma ventana) ===')
    print(f"agresivo      CAGR {m['cagr']:+.1%} DD {m['dd']:.1%} PF {m['pf']:.2f} trades/año {m['por_año']:.0f}")
    print(f"V2 4,5%       CAGR {mv['cagr'] / 100:+.1%} DD {mv['dd'] / 100:.1%} PF {mv['pf']:.2f} "
          f"trades/año {mv['por_año']:.0f}")
    print(f"buy&hold BTC  CAGR {bh_cagr:+.1%} DD {bh_dd:.1%}")


if __name__ == '__main__':
    main()
