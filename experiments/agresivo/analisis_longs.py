"""Por que pierden los LONG del agresivo. Diagnostico descriptivo (2026-11-20).

Nivel candidato (no cartera): cada candidato 2021 -> nov-2026 con su resultado
etiquetado (r_R, en multiplos del riesgo, ya con costes) y la p del modelo de
su mes (walk-forward). Asi no lo contamina el freno por DD de la cartera.

NO es una busqueda de filtros: cortar por regimen/tipo y quedarse con la celda
buena es la busqueda de parametros que el proyecto ya cerro. Cualquier regla
que salga de aqui necesita pre-registro + null sintetico antes de usarse.
Uso: python experiments/agresivo/analisis_longs.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402
import backtest_2026 as b26  # noqa: E402
from backtest import ag  # noqa: E402


def resumen(g):
    return pd.Series({'n': len(g), 'wr': (g.r_R > 0).mean(), 'r_medio': g.r_R.mean(),
                      'suma_R': g.r_R.sum()})


def tabla(df, por):
    return df.groupby(por, observed=True).apply(resumen).round(3).to_string()


def main():
    datos = b26.completar(bt.cargar())
    btc = datos['BTC/USDT']
    feats = bt.features_todos(datos)
    ev = bt.predicciones(ag.eventos_panel(feats))          # bt.DESDE = 2021-01-01
    ev['lado'] = ev['dir'].map({1: 'LONG', -1: 'SHORT'})
    # regimen de BTC para el LADO: btc_ema_d = dist BTC-EMA200 * dir
    ev['btc_vs_ema200'] = np.where(ev['btc_ema_d'] * ev['dir'] > 0, 'BTC arriba', 'BTC abajo')
    ev['opera'] = [ag.riesgo_de(p, b, 0.0) > 0 for p, b in zip(ev.p, ev.b)]
    ev['año'] = ev['t'].dt.year
    print(f"candidatos {len(ev)} ({ev.t.min():%Y-%m} -> {ev.t.max():%Y-%m})\n")

    print('1) Todos los candidatos, por lado (r en R, con costes):')
    print(tabla(ev, 'lado'))
    print('\n2) Por lado y tipo de entrada:')
    print(tabla(ev, ['lado', 'tipo']))
    print('\n3) Por lado y regimen de BTC (EMA200 diaria):')
    print(tabla(ev, ['lado', 'btc_vs_ema200']))
    print('\n4) LONG: tipo x regimen:')
    print(tabla(ev[ev.dir == 1], ['tipo', 'btc_vs_ema200']))

    print('\n5) El modelo: discrimina? (AUC OOS) y los que SI opera (riesgo>=1% a DD 0)')
    for lado, g in ev.groupby('lado'):
        op, no = g[g.opera], g[~g.opera]
        print(f"  {lado}: AUC {roc_auc_score(g.y, g.p):.3f} | opera {len(op)} "
              f"r_medio {op.r_R.mean():+.3f} | descarta {len(no)} r_medio {no.r_R.mean():+.3f}")
    print('\n6) LONG que el modelo opera, por regimen y tipo:')
    print(tabla(ev[(ev.dir == 1) & ev.opera], ['btc_vs_ema200', 'tipo']))
    print('\n7) LONG por año (todos / los que opera):')
    lg = ev[ev.dir == 1]
    print(lg.groupby('año').apply(lambda g: pd.Series({
        'n': len(g), 'r_medio': g.r_R.mean(), 'n_opera': g.opera.sum(),
        'r_opera': g[g.opera].r_R.mean(),
        'pct_BTC_abajo': (g.btc_vs_ema200 == 'BTC abajo').mean()})).round(3).to_string())
    print('\n8) Calibracion LONG (p del modelo vs tasa real):')
    c = lg.assign(bin=pd.cut(lg.p, [0, .35, .40, .45, .50, 1]))
    print(c.groupby('bin', observed=True).agg(n=('y', 'size'), p=('p', 'mean'),
                                              real=('y', 'mean'), r=('r_R', 'mean')).round(3).to_string())


if __name__ == '__main__':
    main()
