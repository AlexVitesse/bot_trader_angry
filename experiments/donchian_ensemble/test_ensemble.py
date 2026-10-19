"""E3: ensemble de Donchian (20/55/120/240/480) + F. Pre-registro en README.md.

Uso: python experiments/donchian_ensemble/test_ensemble.py [n_series]
"""
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'portfolio_sim'))
sys.path.insert(0, str(ROOT / 'experiments' / 'bootstrap_bloques'))
from portfolio_sim import PortfolioSim                  # noqa: E402
from src import v2_engine as v2                         # noqa: E402
from test_bloques import cargar_btc, sintetica          # noqa: E402

BASE, CAP = 0.045, 2.0
PERIODOS = (20, 55, 120, 240, 480)
L_VELAS = 180
NO_A = {'a_adx_min': 1e9}
DF = None


def libros(ensemble=True):
    """{clave: (params, riesgo)}"""
    if ensemble:
        out = {f'BTC/USDT#A{n}': ({**v2.PARAMS_V2, 'a_donchian_n': n, 'f_enable_long': False},
                                  BASE / len(PERIODOS)) for n in PERIODOS}
    else:   # control: V2 en dos libros
        out = {'BTC/USDT#A55': ({**v2.PARAMS_V2, 'f_enable_long': False}, BASE)}
    out['BTC/USDT#F'] = ({**v2.PARAMS_V2, **NO_A}, BASE)
    return out


def d1_de(df):
    return df.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                  'close': 'last', 'volume': 'sum'}).dropna()


def correr_libros(df, ensemble=True):
    d1 = d1_de(df)
    lib = libros(ensemble)
    datos = {k: v2.build_features(df, d1, None, prm) for k, (prm, _) in lib.items()}
    riesgo = {k: rk for k, (_, rk) in lib.items()}
    return PortfolioSim(datos, params_por_par={k: prm for k, (prm, _) in lib.items()},
                        max_concurrent=len(lib), max_misma_dir=len(lib), max_notional_pct=CAP,
                        sizing=lambda par, ts, eq, trail: eq * riesgo[par] / trail).run()


def correr_v2(df):
    feat = v2.build_features(df, d1_de(df), None, v2.PARAMS_V2)
    return PortfolioSim({'BTC/USDT': feat}, risk_pct=BASE, max_concurrent=1,
                        max_notional_pct=CAP).run()


def _init(df):
    global DF
    DF = df


def _worker(seed):
    s = sintetica(DF, L_VELAS, seed)
    return correr_libros(s).metricas.get('cagr', -100), correr_v2(s).metricas.get('cagr', -100)


def fila(nombre, res):
    m = res.metricas
    print(f"  {nombre:<26} n {m['n']:>4}  WR {m['wr']:>5.1f}%  PF {m['pf']:>5.2f}  "
          f"CAGR {m['cagr']:>+6.1f}%  DD {m['dd']:>5.1f}%  CAGR/DD {m['cagr'] / m['dd']:>5.2f}")
    return m


def main():
    n_series = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    df = cargar_btc()

    m_v2 = fila('V2 (4,5%)', correr_v2(df))
    assert m_v2['n'] == 165 and round(m_v2['cagr'], 1) == 23.5, m_v2
    fila('Control: A55 + F en 2 libros', correr_libros(df, ensemble=False))
    ens = correr_libros(df)
    m_e = fila('Ensemble 5xA + F', ens)
    tr = pd.DataFrame(ens.trades)
    print("  trades por libro:", tr.groupby('par').size().to_dict())
    print("  rechazos:", ens.rechazos)

    print(f"\nNull sintetico: {n_series} series, bloques de {L_VELAS} velas ...")
    with Pool(initializer=_init, initargs=(df,)) as pool:
        out = np.array(pool.map(_worker, range(n_series)))
    p_e = float((out[:, 0] >= m_e['cagr']).mean())
    p_v = float((out[:, 1] >= m_v2['cagr']).mean())
    print(f"  Ensemble: CAGR real {m_e['cagr']:+.1f}%  null mediana {np.median(out[:, 0]):+.1f}%  "
          f"p95 {np.percentile(out[:, 0], 95):+.1f}%  p = {p_e:.3f}")
    print(f"  V2 (ref): CAGR real {m_v2['cagr']:+.1f}%  null mediana {np.median(out[:, 1]):+.1f}%  "
          f"p95 {np.percentile(out[:, 1], 95):+.1f}%  p = {p_v:.3f}")
    c1 = p_e < 0.01
    c2 = m_e['cagr'] / m_e['dd'] >= m_v2['cagr'] / m_v2['dd']
    print(f"\nH3: (1) null p<0,01: {c1}   (2) CAGR/DD >= V2: {c2}   -> "
          f"{'PASA' if c1 and c2 else 'NO pasa'}")


if __name__ == '__main__':
    main()
