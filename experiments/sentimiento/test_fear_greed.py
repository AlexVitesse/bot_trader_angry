"""
Fear & Greed como experto de sentimiento — etapa 1 (¿la señal existe?)
=====================================================================
PRE-REGISTRO 2026-10-17, antes de mirar resultados. Mismo protocolo que
experiments/derivados_ratios/: dos hipotesis, Bonferroni alfa = 0,025 cada una,
null por rotacion circular de la serie de sentimiento.

Datos: api.alternative.me/fng (diario desde 2018-02) -> data/fear_greed_daily.parquet
       BTC 4h data/BTC_USDT_4h_full.parquet (hasta 2026-02-27).
Sin look-ahead: el valor con timestamp D 00:00 UTC se publica al empezar el dia
D; se usa para velas que ABREN el dia D (se conoce antes de su cierre).

S1-FEAR (contrarian, "compra cuando hay miedo"):
  eventos = todos los dias; grupo = FG <= 25 (miedo extremo).
  resultado = log(close[d+7] / open[d+1]).
  estadistico = media(miedo) - media(resto). Una cola: mayor.
S1-GREED (ruptura con euforia = posicionamiento lleno):
  eventos = velas 4h con close > max(high de las 55 previas).
  grupo = FG >= 75 (euforia extrema). resultado = log(close[t+12]/open[t+1]).
  estadistico = media(euforia) - media(resto). Una cola: menor.
Null: FG rotada k dias, todos los k en [30, N-30].
Regla de parada: solo pasa a etapa 2 (filtro sobre trades de V2) la que de p < 0,025.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def cargar():
    df = pd.read_parquet(ROOT / 'data' / 'BTC_USDT_4h_full.parquet')
    df = df[['open', 'high', 'low', 'close']].astype(float)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df[~df.index.duplicated(keep='last')].sort_index()
    fg = pd.read_parquet(ROOT / 'data' / 'fear_greed_daily.parquet')['value'].astype(float)
    fg = fg.resample('1D').last().ffill()          # 2 huecos de 2-4 dias en la fuente
    return df, fg


def stat(fg_dia, dias_evento, resultado, umbral, mayor):
    v = fg_dia.reindex(dias_evento).to_numpy()
    ok = ~np.isnan(v)
    g = (v <= umbral) if mayor else (v >= umbral)
    a, b = resultado[ok & g], resultado[ok & ~g]
    return a.mean() - b.mean(), int((ok & g).sum()), int((ok & ~g).sum())


def test(nombre, fg, dias_evento, resultado, umbral, mayor):
    obs, n_g, n_r = stat(fg, dias_evento, resultado, umbral, mayor)
    vals = fg.to_numpy()
    nulls = []
    for k in range(30, len(vals) - 30):
        rot = pd.Series(np.roll(vals, k), index=fg.index)
        nulls.append(stat(rot, dias_evento, resultado, umbral, mayor)[0])
    nulls = np.array(nulls)
    p = (nulls >= obs).mean() if mayor else (nulls <= obs).mean()
    print(f"{nombre}: grupo n={n_g}  resto n={n_r}  diferencia {obs:+.4f} "
          f"({np.expm1(obs):+.2%})  null mediana {np.median(nulls):+.4f}  "
          f"rotaciones {len(nulls)}  p={p:.4f}  {'PASA' if p < 0.025 else 'no pasa'}")
    return p


def main():
    df, fg = cargar()
    fg = fg[fg.index >= df.index[0].floor('1D')]
    fg = fg[fg.index <= df.index[-1].floor('1D')]
    print(f"BTC {df.index[0].date()} -> {df.index[-1].date()} | FG {fg.index[0].date()} -> "
          f"{fg.index[-1].date()} ({len(fg)} dias)\n")

    # S1-FEAR: diario
    d1 = df.resample('1D').agg({'open': 'first', 'close': 'last'}).dropna()
    res = np.log(d1['close'].shift(-7) / d1['open'].shift(-1))
    ok = res.notna()
    test('S1-FEAR  (FG<=25, 7 dias)', fg, d1.index[ok], res[ok].to_numpy(), 25, True)

    # S1-GREED: rupturas Donchian-55 4h
    brk = df['close'] > df['high'].rolling(55).max().shift(1)
    res4 = np.log(df['close'].shift(-12) / df['open'].shift(-1))
    ev = brk & res4.notna()
    test('S1-GREED (FG>=75, rupturas 4h, 12 velas)', fg, df.index[ev].floor('1D'),
         res4[ev].to_numpy(), 75, False)


if __name__ == '__main__':
    main()
