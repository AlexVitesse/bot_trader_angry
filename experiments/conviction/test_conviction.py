"""
¿Existe "conviction" real en V2?
================================
El sizing por conviccion (mas seguro -> mas apalancado) solo funciona si algo
observable EN EL MOMENTO DE LA ENTRADA separa ganadores de perdedores. Si no,
subir tamano en los "buenos" solo aumenta la varianza sin subir el retorno
esperado — y encima concentra el riesgo justo donde no toca.

Este test coge los trades reales de V2 (sin F_SHORT) y para cada feature de
entrada parte los trades en cuartiles, mirando WR y PnL medio por cuartil.
Si una feature tiene poder predictivo, se vera monotonia (Q1 -> Q4 creciente).

AVISO DE HONESTIDAD: probar 6 features sobre 164 trades encuentra "algo" por
azar casi seguro. Por eso se reporta ademas la mitad-1 vs mitad-2 de la
historia: una relacion real aparece en ambas; una casualidad, solo en una.

Uso: python experiments/conviction/test_conviction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src import v2_engine as v2


def load_btc():
    df = pd.read_parquet(ROOT / 'data' / 'BTC_USDT_4h_full.parquet')
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    r = requests.get('https://fapi.binance.com/fapi/v1/klines',
                     params={'symbol': 'BTCUSDT', 'interval': '4h', 'limit': 1500},
                     timeout=30)
    r.raise_for_status()
    live = pd.DataFrame(r.json(), columns=['ts', 'open', 'high', 'low', 'close',
                                           'volume', 'a', 'b', 'c', 'd', 'e', 'f'])
    live['ts'] = pd.to_datetime(live['ts'], unit='ms', utc=True)
    live = live.set_index('ts')[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df4 = pd.concat([df, live])
    df4 = df4[~df4.index.duplicated(keep='last')].sort_index()
    df1 = df4.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                  'close': 'last', 'volume': 'sum'}).dropna()
    return df4, df1


def cuartiles(tab: pd.DataFrame, col: str) -> str:
    """WR y PnL medio por cuartil de `col`. Devuelve linea formateada."""
    try:
        q = pd.qcut(tab[col], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    except ValueError:
        return f"  {col:<18} (sin variacion suficiente)"
    partes = []
    for lab in ['Q1', 'Q2', 'Q3', 'Q4']:
        sub = tab[q == lab]
        if not len(sub):
            partes.append('    —    ')
            continue
        partes.append(f"{(sub.pnl > 0).mean()*100:>3.0f}%/{sub.pnl.mean()*100:>+5.2f}")
    # correlacion de rango entre la feature y el pnl: el resumen honesto
    rho = tab[col].corr(tab['pnl'], method='spearman')
    return f"  {col:<18} " + "  ".join(partes) + f"   rho={rho:>+.3f}"


def bloque(tab: pd.DataFrame, titulo: str, feats: list) -> None:
    print(f"\n{titulo}  (n={len(tab)})")
    print(f"  {'feature':<18} {'Q1 (bajo)':>11} {'Q2':>11} {'Q3':>11} "
          f"{'Q4 (alto)':>11}   {'spearman':>10}")
    print("  " + "-" * 78)
    for f in feats:
        print(cuartiles(tab, f))


def main() -> None:
    df4, df1 = load_btc()
    params = {**v2.PARAMS_V2, 'f_enable_short': False}
    feats_df = v2.build_features(df4, df1, None, params)
    trades = v2.run_v2_backtest(df4, df1, None, params)

    filas = []
    for t in trades:
        row = feats_df.iloc[t['idx_entry']]
        dh = row['donchian_high']
        filas.append({
            'pnl': t['pnl_pct'],
            'sig': t['sig_type'],
            'ts': pd.Timestamp(t['ts_entry']),
            'adx': row['adx'],
            'vol_ratio': row['vol_ratio'],
            'atr_pct': row['atr_pct'],
            'sobre_donchian': (row['close'] - dh) / dh if dh else np.nan,
            'bb_width': row['bb_width'],
        })
    tab = pd.DataFrame(filas).dropna()
    FEATS = ['adx', 'vol_ratio', 'atr_pct', 'sobre_donchian', 'bb_width']

    print(f"V2 sin F_SHORT | {len(tab)} trades | "
          f"{tab.ts.min().date()} -> {tab.ts.max().date()}")
    print("Celdas: WR% / PnL medio%")

    bloque(tab, 'HISTORIA COMPLETA', FEATS)

    mitad = tab.ts.quantile(0.5)
    bloque(tab[tab.ts <= mitad], 'PRIMERA MITAD', FEATS)
    bloque(tab[tab.ts > mitad], 'SEGUNDA MITAD', FEATS)

    print(f"\nPOR TIPO DE SETUP (la 'conviction' mas obvia)")
    print(f"  {'setup':<10} {'n':>4} {'WR':>7} {'PnL medio':>11} {'PF':>7}")
    print("  " + "-" * 44)
    for sig in ['A_LONG', 'F_LONG']:
        s = tab[tab.sig == sig]
        if not len(s):
            continue
        w, l = s.pnl[s.pnl > 0].sum(), abs(s.pnl[s.pnl <= 0].sum())
        print(f"  {sig:<10} {len(s):>4} {(s.pnl>0).mean()*100:>6.1f}% "
              f"{s.pnl.mean()*100:>+10.2f}% {w/l if l else float('inf'):>7.2f}")

    print("\nLectura: para que el sizing por conviccion tenga base, una feature")
    print("necesita spearman claramente != 0 Y el mismo signo en ambas mitades.")


if __name__ == '__main__':
    main()
