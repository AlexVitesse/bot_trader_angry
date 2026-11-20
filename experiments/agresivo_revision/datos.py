"""Datos 4h de los 9 pares completados hasta 2026-11-19 (cache local, un solo fetch)."""
import sys
from pathlib import Path
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / 'agresivo'))
CACHE = AQUI / 'datos_cache.pkl'


def cargar():
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    import backtest as bt
    import backtest_2026 as b26
    datos = b26.completar(bt.cargar())
    pd.to_pickle(datos, CACHE)
    return datos


if __name__ == '__main__':
    for p, d in cargar().items():
        print(p, d.index[0], d.index[-1], len(d), round(d.close.iloc[-1], 4))
