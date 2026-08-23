"""
Arbitraje estadistico / cointegracion entre cripto.

Familia sin medir. La idea: si dos monedas cointegran, su spread es
estacionario y se puede operar la reversion — market-neutral, sin predecir
direccion, asi que esquiva el R2=0,068% de `predictibilidad/`.

La trampa: con 21 monedas hay 210 pares. A p<0,05, el azar solo ya produce
~10 "cointegrados". Y cointegrar en el pasado no implica cointegrar despues.
El protocolo separa las dos cosas:

  A. Cointegracion en TRAIN, contra lo esperado por azar.
  B. Los que cointegran en train, siguen cointegrando en TEST?
  C. Backtest de reversion z-score en TEST, con costes, solo sobre los
     que sobrevivieron A y B (seleccion hecha SIN mirar test).

Uso: python experiments/stat_arb/test_stat_arb.py
"""
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

warnings.filterwarnings('ignore')

ALPHA = 0.05
CORTE = '2024-06-30'      # train / test
Z_ENTRADA, Z_SALIDA = 2.0, 0.5
COSTE = 0.001             # 0,10% ida+vuelta, dos patas
VENTANA_Z = 120           # velas 4h para media/std del spread


def cargar(min_velas=3000):
    """min_velas alto = universo mas pequeno pero ventana comun mas larga."""
    root = Path(__file__).resolve().parents[2] / 'data'
    series = {}
    for f in sorted(root.glob('*_USDT_4h_full.parquet')):
        nombre = f.name.split('_')[0]
        d = pd.read_parquet(f)
        if len(d) < min_velas:
            continue
        # unos parquet vienen tz-aware y otros no; normalizar a UTC
        idx = d.index
        d.index = idx.tz_localize('UTC') if idx.tz is None else idx.tz_convert('UTC')
        series[nombre] = d['close']
    px = pd.DataFrame(series).dropna()
    return np.log(px)


def analizar(lp, etiqueta):
    barra = '=' * 70
    print(f"\n{barra}\n{etiqueta}\n{barra}")
    corte = pd.Timestamp(CORTE, tz='UTC')
    tr = lp[lp.index <= corte]
    te = lp[lp.index > corte]
    pares = list(combinations(lp.columns, 2))
    print(f"{len(lp.columns)} monedas, {len(pares)} pares")
    print(f"  train: {tr.index[0].date()} -> {tr.index[-1].date()} ({len(tr):,})")
    print(f"  test:  {te.index[0].date()} -> {te.index[-1].date()} ({len(te):,})\n")

    print("=== A) Cointegracion en TRAIN vs lo esperado por azar ===")
    coint_tr = []
    for a, b in pares:
        try:
            p = coint(tr[a], tr[b])[1]
        except Exception:
            continue
        if p < ALPHA:
            coint_tr.append((a, b, p))
    esperados = len(pares) * ALPHA
    print(f"  cointegrados a p<{ALPHA}: {len(coint_tr)} de {len(pares)}")
    print(f"  esperados SOLO por azar:  {esperados:.1f}")
    if len(coint_tr) <= esperados:
        print(f"  -> al nivel del azar. Nada que perseguir.")

    print(f"\n=== B) De esos, cuantos SIGUEN cointegrando en TEST? ===")
    persisten = []
    for a, b, p_tr in coint_tr:
        try:
            p_te = coint(te[a], te[b])[1]
        except Exception:
            continue
        if p_te < ALPHA:
            persisten.append((a, b, p_tr, p_te))
    tasa = 100 * len(persisten) / len(coint_tr) if coint_tr else 0
    print(f"  persisten: {len(persisten)} de {len(coint_tr)}  ({tasa:.0f}%)")
    print(f"  si la cointegracion fuera real, deberia persistir muy por")
    print(f"  encima del {ALPHA * 100:.0f}% que da el azar.")
    for a, b, p1, p2 in persisten[:10]:
        print(f"    {a:>5}-{b:<5} train p={p1:.4f}  test p={p2:.4f}")

    print(f"\n=== C) Backtest de reversion en TEST (seleccion hecha en train) ===")
    if not coint_tr:
        print("  sin pares que operar")
        return
    todos = []
    for a, b, _ in coint_tr:
        beta = np.polyfit(tr[b], tr[a], 1)[0]
        spread = te[a] - beta * te[b]
        m = spread.rolling(VENTANA_Z).mean()
        s = spread.rolling(VENTANA_Z).std()
        z = ((spread - m) / s).dropna()
        pos, entrada, trades = 0, 0.0, []
        for i in range(len(z)):
            zi = z.iloc[i]
            if pos == 0:
                if zi > Z_ENTRADA:
                    pos, entrada = -1, spread.loc[z.index[i]]
                elif zi < -Z_ENTRADA:
                    pos, entrada = 1, spread.loc[z.index[i]]
            elif abs(zi) < Z_SALIDA:
                sal = spread.loc[z.index[i]]
                trades.append(pos * (sal - entrada) - COSTE)
                pos = 0
        if trades:
            todos.extend(trades)
    t = np.array(todos)
    if len(t) == 0:
        print("  0 trades")
        return
    g, l = t[t > 0].sum(), -t[t < 0].sum()
    print(f"  pares operados: {len(coint_tr)}   trades: {len(t)}")
    print(f"  WR: {100 * (t > 0).mean():.1f}%   PF: {g / l if l else float('inf'):.2f}")
    print(f"  retorno medio por trade: {t.mean() * 100:+.3f}% (log-spread, "
          f"neto de costes)")
    print(f"  suma: {t.sum() * 100:+.1f}%")


def main() -> None:
    # dos universos: muchas monedas / ventana corta, y pocas / ventana larga
    analizar(cargar(3000),
             'UNIVERSO AMPLIO (21 monedas, ventana comun corta)')
    analizar(cargar(12000),
             'UNIVERSO LARGO (solo monedas con historico desde 2020)')


if __name__ == '__main__':
    main()
