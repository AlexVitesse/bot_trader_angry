"""Que habria hecho el bot agresivo en 2026 si arrancaba el 1-ene con $100.

Mismo motor (src/agresivo_engine.py) y misma cartera que backtest.py; solo
cambia la ventana (2026-01-01 -> HASTA) y el capital (100, DD desde cero como
en vivo). Completa los parquet (acaban 2026-02-27) con klines 4h de fapi.
Compara ademas los trades simulados con los reales del paper desde 2026-10-17
(pasar la DB del VPS: python ... data/ml_agresivo.db).

Limites: no modela el minimo de qty de Binance; los candidatos cuyo trade no
habia terminado al final de los datos no tienen etiqueta y no entran.
Uso: python experiments/agresivo/backtest_2026.py [ruta ml_agresivo.db]
"""
import sqlite3
import sys
from pathlib import Path

import ccxt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402
from backtest import ag  # noqa: E402

INICIO = pd.Timestamp('2026-01-01', tz='UTC')
HASTA = pd.Timestamp('2026-11-20', tz='UTC')        # hasta el 19-nov incluido
VIVO_DESDE = pd.Timestamp('2026-10-17', tz='UTC')


def completar(datos):
    ex = ccxt.binanceusdm()
    for par, df in datos.items():
        desde = int(df.index[-1].timestamp() * 1000) + 1
        filas = []
        while desde < HASTA.timestamp() * 1000:
            k = ex.fetch_ohlcv(par + ':USDT', '4h', since=desde, limit=1500)
            if not k:
                break
            filas += k
            desde = k[-1][0] + 1
        nuevo = pd.DataFrame(filas, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        nuevo.index = pd.to_datetime(nuevo.pop('ts'), unit='ms', utc=True)
        df = pd.concat([df, nuevo.astype(float)])
        df = df[~df.index.duplicated(keep='last')].sort_index()
        datos[par] = df[df.index < HASTA]
    return datos


def main():
    bt.DESDE, bt.CAPITAL = INICIO, 100.0
    datos = completar(bt.cargar())
    btc = datos['BTC/USDT']
    print('datos hasta:', min(d.index[-1] for d in datos.values()))
    feats = {p: ag.features(d, btc) for p, d in datos.items()}
    oos = bt.predicciones(ag.eventos_panel(feats))
    eq, tr = bt.simular(datos, feats, oos)
    tr['lado'] = tr['dir'].map({1: 'LONG', -1: 'SHORT'})
    tr['gano'] = tr['pnl'] > 0

    m = bt.metricas(eq, tr['r'].values)
    serie = pd.concat([pd.Series([100.0]), eq.reset_index(drop=True)])
    print(f"\n=== AGRESIVO 2026-01-01 -> {eq.index[-1]:%Y-%m-%d}, $100 ===")
    print(f"final ${eq.iloc[-1]:.2f} ({eq.iloc[-1] / 100 - 1:+.1%}) | DD max "
          f"{(1 - serie / serie.cummax()).max():.1%} | trades {len(tr)} | WR {tr.gano.mean():.1%} "
          f"| PF {m['pf']:.2f} | riesgo medio {tr.riesgo.mean():.2%}")
    btc26 = btc['close'][btc.index >= INICIO]
    print(f"BTC en el mismo tramo: {btc26.iloc[-1] / btc26.iloc[0] - 1:+.1%}")

    print('\npor lado:')
    print(tr.groupby('lado').agg(n=('pnl', 'size'), wr=('gano', 'mean'),
                                 pnl=('pnl', 'sum')).round(2).to_string())
    print('\npor mes (equity a fin de mes y trades abiertos ese mes):')
    mes = eq.resample('ME').last()
    ret = mes / mes.shift(1).fillna(100.0) - 1
    g = tr.groupby(tr['ts'].dt.to_period('M'))
    tabla = pd.DataFrame({'equity': mes.values.round(2), 'ret': ret.values.round(3)},
                         index=mes.index.to_period('M'))
    tabla = tabla.join(g.size().rename('trades')).join(
        g['gano'].mean().round(2).rename('wr')).join(
        tr[tr.dir == 1].groupby(tr['ts'].dt.to_period('M')).size().rename('long')).join(
        tr[tr.dir == -1].groupby(tr['ts'].dt.to_period('M')).size().rename('short'))
    print(tabla.fillna(0).to_string())
    print('\npor par:')
    print(tr.groupby('par').agg(n=('pnl', 'size'), wr=('gano', 'mean'),
                                pnl=('pnl', 'sum')).round(2).sort_values('pnl').to_string())

    # tramo en vivo: simulado vs real
    sim = tr[tr['ts'] >= VIVO_DESDE]
    print(f"\n=== tramo en vivo (desde {VIVO_DESDE:%Y-%m-%d}) simulado ===")
    print(sim[['par', 'lado', 'tipo', 'ts', 'salida', 'riesgo', 'pnl']].round(4).to_string(index=False))
    if len(sys.argv) > 1:
        c = sqlite3.connect(sys.argv[1])
        viv = pd.read_sql('select symbol par, side, entry_time, exit_time, pnl from ml_trades', c)
        print('\n=== reales (paper) ===')
        print(viv.round(2).to_string(index=False))
        print(f"\nsimulado: {len(sim)} trades, pnl ${sim.pnl.sum():+.2f} (sobre equity del "
              f"backtest) | real: {len(viv)} trades, pnl ${viv.pnl.sum():+.2f} (sobre $100)")


if __name__ == '__main__':
    main()
