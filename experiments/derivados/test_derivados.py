"""Open interest y DVOL como filtro de V2. Diseno pre-registrado en README.md
(commit 1a16532 + adenda). Etapa 1: senal a nivel de mercado con null por
rotacion circular. Etapa 2 (solo hipotesis que pasan): filtro sobre trades V2.

Uso: python experiments/derivados/test_derivados.py
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
sys.path.insert(0, str(ROOT / 'experiments' / 'portfolio_sim'))
from portfolio_sim import PortfolioSim, cargar_pares  # noqa: E402

OI_CACHE = ROOT / 'data' / 'deriv_oi_btcusdt_5m.parquet'
DVOL_CACHE = ROOT / 'data' / 'deriv_dvol_btc_1d.parquet'
N_ROT, N_BOOT, SEED = 10_000, 20_000, 20260923
ALPHA1 = 0.025
BAR = pd.Timedelta(hours=4)


# ------------------------------------------------------------------- datos
def _oi_dia(d):
    u = ('https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/'
         f'BTCUSDT-metrics-{d:%Y-%m-%d}.zip')
    for _ in range(3):
        try:
            r = requests.get(u, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            return pd.read_csv(z.open(z.namelist()[0]),
                               usecols=['create_time', 'sum_open_interest'])
        except Exception:
            continue
    raise RuntimeError(f'no se pudo bajar {u}')


def cargar_oi() -> pd.Series:
    if not OI_CACHE.exists():
        dias = pd.date_range('2020-09-01', datetime.now(timezone.utc).date(), freq='D')
        with ThreadPoolExecutor(16) as ex:
            partes = [p for p in ex.map(_oi_dia, dias) if p is not None]
        df = pd.concat(partes)
        df['ts'] = pd.to_datetime(df['create_time'], utc=True)
        s = df.set_index('ts')['sum_open_interest'].astype(float)
        s = s[~s.index.duplicated()].sort_index()
        s.to_frame('oi').to_parquet(OI_CACHE)
    oi = pd.read_parquet(OI_CACHE)['oi']
    # 473 filas con OI = 0 en el dataset de Binance (imposible en BTC): son
    # huecos, no datos. Se quitan y el asof toma el ultimo valor valido.
    return oi[oi > 0]


def cargar_dvol() -> pd.Series:
    if not DVOL_CACHE.exists():
        ms = lambda t: int(pd.Timestamp(t, tz='UTC').timestamp() * 1000)
        filas, fin = [], ms(datetime.now(timezone.utc).date())
        ini = ms('2021-01-01')
        while True:
            r = requests.get('https://www.deribit.com/api/v2/public/get_volatility_index_data',
                             params=dict(currency='BTC', start_timestamp=ini,
                                         end_timestamp=fin, resolution='1D'),
                             timeout=30).json()['result']
            filas += r['data']
            if not r.get('continuation'):
                break
            fin = r['continuation']
        df = pd.DataFrame(filas, columns=['ts', 'o', 'h', 'l', 'c'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        df = df.drop_duplicates('ts').set_index('ts').sort_index()
        df[['c']].rename(columns={'c': 'dvol'}).to_parquet(DVOL_CACHE)
    return pd.read_parquet(DVOL_CACHE)['dvol']


def velas() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / 'data' / 'BTC_USDT_4h_full.parquet')
    df = df[['open', 'high', 'low', 'close']].astype(float)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df[~df.index.duplicated(keep='last')].sort_index()


def oi_chg_en(oi: pd.Series, T: pd.DatetimeIndex) -> pd.Series:
    """OI(T)/OI(T-24h)-1 con asof (ultimo create_time <= instante)."""
    now = oi.reindex(T, method='ffill')
    prev = oi.reindex(T - pd.Timedelta(hours=24), method='ffill')
    ok = T - pd.Timedelta(hours=24) >= oi.index[0]
    out = pd.Series(now.values / prev.values - 1, index=T)
    out[~ok] = np.nan
    return out


def dvol_pos(dvol: pd.Series) -> pd.Series:
    d = dvol.asfreq('1D')
    lo, hi = d.rolling(90, min_periods=90).min(), d.rolling(90, min_periods=90).max()
    pos = (d - lo) / (hi - lo)
    pos[hi == lo] = np.nan
    return pos


def rot_ks(n, minimo):
    return np.unique(np.linspace(minimo, n - minimo, N_ROT).astype(int))


# ------------------------------------------------------------------ etapa 1
def s1_oi(v, oi):
    cierre = v.index + BAR                             # cierre de cada vela
    chg = oi_chg_en(oi, cierre)
    chg.index = v.index
    cub = chg.notna().values
    vv, chg = v[cub], chg[cub].values
    n = len(vv)
    evento = (vv.close > vv.high.rolling(55).max().shift(1)).values
    fwd = np.log(vv.close.shift(-12) / vv.open.shift(-1)).values
    ev = np.where(evento & np.isfinite(fwd))[0]
    r = fwd[ev]

    def stat(c):
        s = c[ev]
        up = s > 0
        return r[up].mean() - r[~up].mean()
    obs = stat(chg)
    nulos = np.array([stat(np.roll(chg, k)) for k in rot_ks(n, 180)])
    p = float(np.mean(nulos >= obs))
    up = chg[ev] > 0
    print('=== S1-OI: rupturas Donchian-55 crudas, retorno 12 velas ===')
    print(f'velas con cobertura: {n} ({vv.index[0].date()} -> {vv.index[-1].date()}) | '
          f'eventos: {len(ev)} (OI sube {up.sum()}, no sube {(~up).sum()})')
    print(f'media OI sube {r[up].mean():+.4%} | OI no sube {r[~up].mean():+.4%} | '
          f'dif {obs:+.4%} | p = {p:.4f} ({len(nulos)} rotaciones)')
    return p


def s1_dvol(v, dvol):
    pos = dvol_pos(dvol)
    cd = v.close.groupby(v.index.floor('1D')).last().asfreq('1D')
    fwd = np.log(cd.shift(-60) / cd)
    t = pd.DataFrame({'pos': pos, 'fwd': fwd}).dropna()
    sig = (t.pos <= 0.05).values
    f = t.fwd.values
    obs = f[sig].mean() - f[~sig].mean()
    nulos = []
    for k in rot_ks(len(t), 90):
        s = np.roll(sig, k)
        nulos.append(f[s].mean() - f[~s].mean())
    nulos = np.array(nulos)
    p = float(np.mean(nulos <= obs))
    print('\n=== S1-DVOL: dias con DVOL en el 5% inferior de su rango de 90 d, retorno 60 d ===')
    print(f'dias: {len(t)} ({t.index[0].date()} -> {t.index[-1].date()}) | '
          f'dias de senal: {sig.sum()}')
    print(f'media senal {f[sig].mean():+.4f} | resto {f[~sig].mean():+.4f} | '
          f'dif {obs:+.4f} | p = {p:.4f} ({len(nulos)} rotaciones)')
    return p, pos


# ------------------------------------------------------------------ etapa 2
def boot_media_neg(r):
    """p de H0: media >= 0, bootstrap circular por bloques de min(10, n)."""
    n = len(r)
    L = min(10, n)
    nb = int(np.ceil(n / L))
    rng = np.random.default_rng(SEED)
    idx = (rng.integers(0, n, (N_BOOT, nb))[:, :, None] + np.arange(L)).reshape(N_BOOT, -1)[:, :n] % n
    m = r[idx].mean(axis=1)
    obs = r.mean()
    return float(np.mean(m - obs <= obs))


def pf(r):
    l = -r[r <= 0].sum()
    return r[r > 0].sum() / l if l else float('inf')


def curva(r):
    eq = np.cumprod(1 + r)
    eq = np.concatenate([[1.0], eq])
    return eq[-1], float((1 - eq / np.maximum.accumulate(eq)).max() * 100)


def etapa2(hip, trades, feat, alpha):
    cub = feat.notna().values
    t, f = trades[cub], feat[cub].values
    fuera = (f <= 0) if hip == 'H-OI' else (f <= 0.05)
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
    oi, dvol = cargar_oi(), cargar_dvol()
    print(f'OI 5m: {len(oi)} filas {oi.index[0]} -> {oi.index[-1]}')
    print(f'DVOL 1d: {len(dvol)} filas {dvol.index[0].date()} -> {dvol.index[-1].date()}\n')
    p_oi = s1_oi(v, oi)
    p_dv, pos = s1_dvol(v, dvol)
    pasan = [h for h, p in (('H-OI', p_oi), ('H-DVOL', p_dv)) if p < ALPHA1]
    print(f'\nEtapa 1 (alpha {ALPHA1} por Bonferroni): pasan {pasan or "ninguna"}')
    if not pasan:
        print('Regla de parada pre-registrada: el experimento termina aqui.')
        sys.exit(0)
    res = PortfolioSim(cargar_pares(['BTC/USDT']), risk_pct=0.02, max_concurrent=1).run()
    tr = pd.DataFrame(res.trades)
    alpha2 = 0.025 if len(pasan) == 2 else 0.05
    for h in pasan:
        if h == 'H-OI':
            feat = oi_chg_en(oi, pd.DatetimeIndex(tr.ts_entrada))
        else:
            dia = pd.DatetimeIndex(tr.ts_entrada).floor('1D') - pd.Timedelta(days=1)
            feat = pos.reindex(dia)
        feat.index = tr.index
        etapa2(h, tr, feat, alpha2)
