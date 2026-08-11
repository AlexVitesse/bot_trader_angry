"""
¿Se puede mejorar el detector de régimen de V2?
==============================================
Origen: en el walk-forward real (`portfolio_sim/run_walkforward.py`), los dos
únicos folds donde V2 gana de verdad (+35% y +183%) son los dos alcistas; en
los otros cuatro pierde poco. Un sistema así no necesita más señales ni más
apalancamiento: necesita saber cuándo estar encendido.

Hoy la única puerta es `bull_1d = EMA50_1d > EMA200_1d` (con shift(1)).

DISCIPLINA DE ESTE TEST
-----------------------
- Candidatos DECLARADOS DE ANTEMANO y motivados mecánicamente, no un grid
  search. Cinco, no cincuenta.
- Evaluación con walk-forward REAL: 6 folds de test, nunca usados para elegir.
- No se toca `v2_engine`: cada candidato se inyecta sobrescribiendo la columna
  `bull_1d` de las features, que es exactamente lo que leen `_signal_a` y
  `_signal_f`.
- Todo con shift(1) sobre la serie DIARIA antes de reindexar a 4h: sin
  look-ahead.
- Si ninguno supera al base en los folds de test, el resultado es NEGATIVO y se
  documenta como tal.

AVISO: elegir el mejor de 5 candidatos mirando folds de test es, en sí, una
comparación múltiple. Si alguno gana, es una HIPÓTESIS que necesita validación
fresca, no un resultado adoptable.

Uso: python experiments/regime/test_regime.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'portfolio_sim'))

from portfolio_sim import PortfolioSim
from src import v2_engine as v2

PARAMS = {**v2.PARAMS_V2, 'f_enable_short': False}
RISK = 0.02
N_FOLDS = 6
TRAIN_MIN_DIAS = 730


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


# =============================================================================
# CANDIDATOS — declarados antes de ver un solo resultado
# =============================================================================
def candidatos(d1: pd.DataFrame) -> dict[str, pd.Series]:
    """Cada candidato devuelve una serie diaria 0/1, ya con shift(1)."""
    c = d1['close']
    ema50, ema200 = _ema(c, 50), _ema(c, 200)
    base = (ema50 > ema200)

    # 1. base: el actual. Cruce de medias.
    # 2. slope: el cruce NO dice si la tendencia se mueve; una EMA50 por encima
    #    pero plana es un mercado lateral. Exigir que la EMA50 suba.
    slope = base & (ema50 > ema50.shift(10))
    # 3. no_dd: una tendencia intacta cotiza cerca de sus máximos. Si BTC está
    #    un 25% por debajo del máximo de 200d, la tendencia está roto aunque
    #    las medias tarden en cruzarse.
    max200 = c.rolling(200).max()
    no_dd = base & (c > max200 * 0.75)
    # 4. slope_only: ¿hace falta el cruce, o basta con que la tendencia suba?
    slope_only = (ema50 > ema50.shift(20))
    # 5. vol: el trend following sufre en volatilidad alta y errática. Exigir
    #    que la vol realizada 20d no esté en su cuartil superior histórico.
    vol20 = c.pct_change().rolling(20).std()
    vol_ok = vol20 < vol20.rolling(200).quantile(0.75)
    vol = base & vol_ok

    return {n: s.astype(int).shift(1)
            for n, s in [('base (actual)', base), ('base+slope', slope),
                         ('base+no_dd', no_dd), ('slope_only', slope_only),
                         ('base+vol_baja', vol)]}


# =============================================================================
def cargar_btc():
    df = pd.read_parquet(ROOT / 'data' / 'BTC_USDT_4h_full.parquet')
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df[~df.index.duplicated(keep='last')].sort_index()
    d1 = df.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                'close': 'last', 'volume': 'sum'}).dropna()
    return df, d1


def main() -> None:
    df4, d1 = cargar_btc()
    feats_base = v2.build_features(df4, d1, None, PARAMS)
    cands = candidatos(d1)

    ini, fin = feats_base.index[0], feats_base.index[-1]
    bordes = pd.date_range(ini + pd.Timedelta(days=TRAIN_MIN_DIAS), fin,
                           periods=N_FOLDS + 1)
    print(f"BTC {ini.date()} -> {fin.date()} | {N_FOLDS} folds de test | "
          f"risk {RISK:.0%} | sin F_SHORT")
    print(f"Candidatos: {len(cands)}, declarados de antemano\n")

    # % del tiempo que cada filtro deja el sistema encendido
    print(f"  {'candidato':<16} {'% encendido':>12}")
    print("  " + "-" * 30)
    for n, s in cands.items():
        print(f"  {n:<16} {s.mean()*100:>11.1f}%")

    resultados = {}
    for nombre, serie in cands.items():
        f = feats_base.copy()
        # inyectar el régimen: es la columna que leen _signal_a / _signal_f
        f['bull_1d'] = serie.reindex(f.index, method='ffill').fillna(0)
        por_fold = []
        for k in range(N_FOLDS):
            m = PortfolioSim({'BTC/USDT': f}, params=PARAMS,
                             risk_pct=RISK).run(bordes[k], bordes[k + 1]).metricas
            por_fold.append((m['final'] - 1) * 100 if m.get('n') else 0.0)
        resultados[nombre] = por_fold

    print(f"\n  {'candidato':<16} " +
          " ".join(f"{'F'+str(i+1):>8}" for i in range(N_FOLDS)) +
          f" {'folds+':>7} {'mediana':>9} {'compuesto':>11}")
    print("  " + "-" * 106)
    for nombre, folds in resultados.items():
        a = np.array(folds)
        comp = (np.prod(1 + a / 100) - 1) * 100
        print(f"  {nombre:<16} " + " ".join(f"{x:>+7.1f}%" for x in a) +
              f" {int((a>0).sum())}/{N_FOLDS:<5} {np.median(a):>+8.1f}% "
              f"{comp:>+10.1f}%")

    base = np.array(resultados['base (actual)'])
    print(f"\n  Base: {int((base>0).sum())}/{N_FOLDS} folds, "
          f"compuesto {(np.prod(1+base/100)-1)*100:+.1f}%")
    print("  Para ser candidato real hay que ganar al base en folds+ Y en")
    print("  compuesto, y aun asi seria una hipotesis pendiente de holdout.")


if __name__ == '__main__':
    main()
