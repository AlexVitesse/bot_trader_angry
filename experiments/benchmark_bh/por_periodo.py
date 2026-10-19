"""E1 por subperiodo (descriptivo, no es test): V2 vs B&H a igual exposicion
en 4 bloques de igual duracion y por ano. Diagnostico de estacionariedad.

Uso: python experiments/benchmark_bh/por_periodo.py
"""
import numpy as np
import pandas as pd

from test_benchmark import ROOT, bh, cargar_pares, correr_v2, funding_diario, mixta, sigma_gk


def met(eq):
    años = (eq.index[-1] - eq.index[0]).days / 365.25
    c = (eq.iloc[-1] / eq.iloc[0]) ** (1 / años) - 1
    d = float((1 - eq / eq.cummax()).max())
    return c, d


def main():
    datos = cargar_pares(['BTC/USDT'])
    res, eq4, expo4 = correr_v2(datos)
    df4 = pd.read_parquet(ROOT / 'data' / 'BTC_USDT_4h_full.parquet')[['open', 'high', 'low', 'close']].astype(float)
    if df4.index.tz is None:
        df4.index = df4.index.tz_localize('UTC')
    df4 = df4[~df4.index.duplicated(keep='last')].sort_index()
    eq = eq4.resample('1D').last().dropna()
    close = df4['close'].resample('1D').last().reindex(eq.index).ffill()
    fund = funding_diario(eq.index)
    w = float(expo4.mean())
    sig = sigma_gk(df4).reindex(eq.index).shift(1)
    raw = np.minimum(1, sig.expanding().median() / sig).fillna(0)
    k = w / raw.mean()
    curvas = {
        'V2': eq,
        'BH18 spot': bh(close, pd.Series(w, index=eq.index)),
        'BH18 perp': bh(close, pd.Series(w, index=eq.index), funding=fund),
        'BHvt perp': bh(close, raw * k, umbral=0.1, escala=k, funding=fund),
    }
    curvas['Mixta'] = mixta(curvas['BH18 perp'], eq)
    curvas['BTC 100%'] = close

    tr = pd.DataFrame(res.trades)
    bordes = pd.date_range(eq.index[0], eq.index[-1], periods=5)
    tramos = [(bordes[i], bordes[i + 1]) for i in range(4)]
    tramos += [(pd.Timestamp(f'{a}-01-01', tz='UTC'), pd.Timestamp(f'{a + 1}-01-01', tz='UTC'))
               for a in range(2019, 2027)]
    print(f"{'tramo':<23} {'trades':>6} {'BTC':>7} | " +
          ' | '.join(f"{n:^15}" for n in curvas if n != 'BTC 100%') + "   (CAGR / DD)")
    for i, (a, b) in enumerate(tramos):
        if i == 4:
            print('-' * 120)
        m = (eq.index >= a) & (eq.index <= b)
        n = int(((tr['ts_entrada'] >= a) & (tr['ts_entrada'] < b)).sum())
        btc = close[m].iloc[-1] / close[m].iloc[0] - 1
        celdas = []
        for nombre, c in curvas.items():
            if nombre == 'BTC 100%':
                continue
            cg, dd = met(c[m])
            celdas.append(f"{cg:>+6.0%} / {dd:>4.0%}")
        print(f"{str(a.date()) + '->' + str(b.date()):<23} {n:>6} {btc:>+7.0%} | " + ' | '.join(celdas))


if __name__ == '__main__':
    main()
