"""Ratio taker y premium index como filtro de V2. Diseno pre-registrado en
README.md. Etapa 1: senal a nivel de mercado con null por rotacion circular.
Etapa 2 (solo hipotesis que pasan): filtro sobre trades V2.

Uso: python experiments/derivados_ratios/test_derivados_ratios.py
"""
import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments' / 'derivados'))
sys.path.insert(0, str(ROOT / 'experiments' / 'portfolio_sim'))
from test_derivados import velas, rot_ks, boot_media_neg, pf, curva  # noqa: E402
from portfolio_sim import PortfolioSim, cargar_pares  # noqa: E402

METRICS_CACHE = ROOT / 'data' / 'deriv_metrics_btcusdt_5m.parquet'
PREM_CACHE = ROOT / 'data' / 'deriv_premium_btcusdt_4h.parquet'
ALPHA1 = 0.025
BAR = pd.Timedelta(hours=4)
Z_WIN = 180


# ------------------------------------------------------------------- datos
def _get_zip(u):
    for _ in range(3):
        try:
            r = requests.get(u, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            return z.open(z.namelist()[0]).read()
        except Exception:
            continue
    raise RuntimeError(f'no se pudo bajar {u}')


def _metrics_dia(d):
    raw = _get_zip('https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/'
                   f'BTCUSDT-metrics-{d:%Y-%m-%d}.zip')
    return None if raw is None else pd.read_csv(io.BytesIO(raw))


def cargar_metrics() -> pd.DataFrame:
    if not METRICS_CACHE.exists():
        dias = pd.date_range('2020-09-01', datetime.now(timezone.utc).date(), freq='D')
        with ThreadPoolExecutor(16) as ex:
            partes = [p for p in ex.map(_metrics_dia, dias) if p is not None]
        df = pd.concat(partes)
        df['ts'] = pd.to_datetime(df['create_time'], utc=True)
        df = df.drop(columns=['create_time', 'symbol']).set_index('ts')
        df = df[~df.index.duplicated()].sort_index().astype(float)
        df.to_parquet(METRICS_CACHE)
    return pd.read_parquet(METRICS_CACHE)


def _prem_mes(m):
    raw = _get_zip('https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines/'
                   f'BTCUSDT/4h/BTCUSDT-4h-{m:%Y-%m}.zip')
    if raw is None:
        return None
    df = pd.read_csv(io.BytesIO(raw), header=None).iloc[:, :5]
    df = df[pd.to_numeric(df[0], errors='coerce').notna()]   # algunos meses traen cabecera
    df.columns = ['open_time', 'open', 'high', 'low', 'close']
    return df.astype(float)


def cargar_premium() -> pd.Series:
    if not PREM_CACHE.exists():
        meses = pd.date_range('2020-01-01', datetime.now(timezone.utc).date(), freq='MS')
        with ThreadPoolExecutor(8) as ex:
            partes = [p for p in ex.map(_prem_mes, meses) if p is not None]
        df = pd.concat(partes)
        df['ts'] = pd.to_datetime(df['open_time'].astype('int64'), unit='ms', utc=True)
        df = df.set_index('ts')[['open', 'high', 'low', 'close']]
        df[~df.index.duplicated()].sort_index().to_parquet(PREM_CACHE)
    return pd.read_parquet(PREM_CACHE)['close']


# ---------------------------------------------------------------- features
def zscore(s: pd.Series) -> pd.Series:
    """z sobre las Z_WIN velas previas (sin la actual), ventana completa."""
    m = s.rolling(Z_WIN, min_periods=Z_WIN).mean().shift(1)
    sd = s.rolling(Z_WIN, min_periods=Z_WIN).std().shift(1)
    return (s - m) / sd


def taker_por_vela(met: pd.DataFrame, v: pd.DataFrame) -> pd.Series:
    """Media del ratio taker en (t-4h, t], t = cierre de la vela. <24 filas -> NaN."""
    tk = met['sum_taker_long_short_vol_ratio']
    tk = tk[tk > 0]
    # la fila con create_time exactamente en t pertenece a la vela que cierra en t
    cierre = tk.index.ceil('4h')
    g = tk.groupby(cierre)
    media = g.mean().where(g.count() >= 24)
    s = media.reindex(v.index + BAR)
    s.index = v.index
    return s


def prem_por_vela(prem: pd.Series, v: pd.DataFrame) -> pd.Series:
    return prem.reindex(v.index)          # vela con open_time t-4h cierra en t


# ------------------------------------------------------------------ etapa 1
def s1(nombre, v, z, en_senal, cola):
    fwd = np.log(v.close.shift(-12) / v.open.shift(-1)).values
    evento = (v.close > v.high.rolling(55).max().shift(1)).values
    cub = z.notna().values
    zz, fwd, evento = z.values[cub], fwd[cub], evento[cub]
    idx = v.index[cub]
    ev = np.where(evento & np.isfinite(fwd))[0]
    r = fwd[ev]

    def stat(c):
        s = en_senal(c[ev])
        return r[s].mean() - r[~s].mean()
    obs = stat(zz)
    nulos = np.array([stat(np.roll(zz, k)) for k in rot_ks(len(zz), 180)])
    p = float(np.mean(nulos >= obs) if cola == 'mayor' else np.mean(nulos <= obs))
    s = en_senal(zz[ev])
    print(f'=== {nombre}: rupturas Donchian-55 crudas, retorno 12 velas ===')
    print(f'velas con cobertura: {len(zz)} ({idx[0].date()} -> {idx[-1].date()}) | '
          f'eventos: {len(ev)} (senal {s.sum()}, resto {(~s).sum()})')
    print(f'media senal {r[s].mean():+.4%} | resto {r[~s].mean():+.4%} | '
          f'dif {obs:+.4%} | p = {p:.4f} ({len(nulos)} rotaciones)\n')
    return p


def etapa2(hip, trades, feat, fuera_fn, alpha):
    cub = feat.notna().values
    t, f = trades[cub], feat[cub].values
    fuera = fuera_fn(f)
    r_all, r_in, r_out = t.r.values, t.r.values[~fuera], t.r.values[fuera]
    print(f'\n=== ETAPA 2 {hip} (alpha {alpha}) ===')
    print(f'trades con cobertura: {len(t)} ({t.ts_entrada.iloc[0].date()} -> '
          f'{t.ts_entrada.iloc[-1].date()}) | descartados: {fuera.sum()}')
    if not fuera.any():
        print('sin trades que descartar -> no se adopta')
        return None
    p = boot_media_neg(r_out)
    for nom, r in (('completo', r_all), ('filtrado', r_in), ('descartados', r_out)):
        fin, dd = curva(r)
        print(f'  {nom:12s} n={len(r):4d} WR {np.mean(r > 0):5.1%} PF {pf(r):5.2f} '
              f'media {r.mean():+.4%} suma {r.sum():+.4f} | x{fin:.3f} DD {dd:4.1f}%')
    adopta = (p < alpha and r_in.sum() >= r_all.sum() and pf(r_in) >= pf(r_all))
    print(f'  media descartados < 0: p = {p:.4f} -> '
          f"{'ADOPTADO' if adopta else 'RECHAZADO'}")
    return p


if __name__ == '__main__':
    v = velas()
    met, prem = cargar_metrics(), cargar_premium()
    print(f'metrics 5m: {len(met)} filas {met.index[0]} -> {met.index[-1]}')
    print(f'premium 4h: {len(prem)} filas {prem.index[0]} -> {prem.index[-1]}')
    print(f'velas 4h hasta {v.index[-1]}\n')

    taker_z = zscore(taker_por_vela(met, v))
    prem_z = zscore(prem_por_vela(prem, v))
    hips = {
        'H-TAKER': (taker_z, lambda c: c > 0, 'mayor', lambda f: f <= 0),
        'H-PREM': (prem_z, lambda c: c > 1, 'menor', lambda f: f > 1),
    }
    ps = {h: s1(f'S1-{h[2:]}', v, z, sen, cola) for h, (z, sen, cola, _) in hips.items()}
    pasan = [h for h, p in ps.items() if p < ALPHA1]
    print(f'Etapa 1 (alpha {ALPHA1} por Bonferroni): pasan {pasan or "ninguna"}')
    for h, p in ps.items():
        if 0.0125 <= p < ALPHA1:
            print(f'  {h}: p={p:.4f} no sobrevive a la correccion global (4 hipotesis, 0,0125)')
    if not pasan:
        print('Regla de parada pre-registrada: el experimento termina aqui.')
        sys.exit(0)

    res = PortfolioSim(cargar_pares(['BTC/USDT']), risk_pct=0.02, max_concurrent=1).run()
    tr = pd.DataFrame(res.trades)
    alpha2 = 0.025 if len(pasan) == 2 else 0.05
    for h in pasan:
        z, _, _, fuera_fn = hips[h]
        zc = z.copy()
        zc.index = zc.index + BAR                      # indexado por cierre de vela
        feat = zc.reindex(pd.DatetimeIndex(tr.ts_entrada))
        feat.index = tr.index
        etapa2(h, tr, feat, fuera_fn, alpha2)
