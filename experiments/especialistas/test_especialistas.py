"""E7 (macro) y E8 (volumen): alerta de caidas de BTC. Pre-registro en README.md.

Uso: python experiments/especialistas/test_especialistas.py [n_rotaciones]
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
H, CAIDA = 10, 0.10
AÑOS = range(2021, 2027)
BLOQUE, REPS = 30, 10_000
SEED = 20261020
BASE = ['vol20', 'dist_ema200', 'ret20']
MACRO = ['oro_z', 'dxy_20', 'spx_dd']
VOL = ['vol_z', 'vol_sube', 'pct_sube']


def leer_macro(nombre):
    s = pd.read_csv(AQUI / 'data' / f'{nombre}.csv', index_col=0, parse_dates=True).iloc[:, 0]
    s.index = s.index.tz_localize('UTC') if s.index.tz is None else s.index
    return s


def macro_diario(c: pd.Series) -> pd.DataFrame:
    """Features E7 por dia UTC dado el cierre diario de BTC (real o sintetico).
    Macro en su calendario habil; fecha habil t disponible desde t+1, asi el
    asof da el ultimo valor estrictamente anterior a d."""
    oro, dxy, spx = (leer_macro(n) for n in ('oro', 'dxy', 'spx'))
    alin = lambda s: s.set_axis(s.index + pd.Timedelta(days=1)).reindex(c.index, method='ffill')  # noqa: E731
    x = np.log(c / alin(oro))
    return pd.DataFrame({'oro_z': (x - x.rolling(60).mean()) / x.rolling(60).std(),
                         'dxy_20': alin(dxy / dxy.shift(20) - 1),
                         'spx_dd': alin(spx / spx.rolling(60).max() - 1)}, index=c.index)


def construir():
    v = pd.read_parquet(ROOT / 'data' / 'BTC_USDT_4h_full.parquet').astype(float)
    if v.index.tz is None:
        v.index = v.index.tz_localize('UTC')
    d = v.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min',
                              'close': 'last', 'volume': 'sum'}).dropna()
    c = d['close']
    df = pd.DataFrame(index=d.index)

    fut_min = pd.concat([d['low'].shift(-k) for k in range(1, H + 1)], axis=1).min(axis=1, skipna=False)
    df['y'] = (fut_min <= (1 - CAIDA) * c).astype(float).where(fut_min.notna())

    lr = np.log(c).diff()
    df['vol20'] = lr.rolling(20).std()
    ema = c.ewm(span=200, adjust=False).mean()
    df['dist_ema200'] = (c / ema - 1).where(np.arange(len(c)) >= 200)
    df['ret20'] = c / c.shift(20) - 1

    df = df.join(macro_diario(c))

    # volumen sobre velas 4h, valor de la vela de las 20:00 UTC
    lv = np.log(v['volume'].clip(lower=1e-9))
    prev = lv.shift(6)
    f4 = pd.DataFrame(index=v.index)
    f4['vol_z'] = (lv.rolling(6).mean() - prev.rolling(120).mean()) / prev.rolling(120).std()
    sube = (v['close'] > v['open']).astype(float)
    f4['vol_sube'] = (v['volume'] * sube).rolling(60).sum() / v['volume'].rolling(60).sum()
    f4['pct_sube'] = sube.rolling(60).mean()
    ult = f4[f4.index.hour == 20]
    ult.index = ult.index.normalize()
    for k in VOL:
        df[k] = ult[k].reindex(d.index)
    return df.dropna()


def walk_forward(df, feats):
    p = pd.Series(np.nan, index=df.index)
    coefs = []
    for Y in AÑOS:
        ini = pd.Timestamp(f'{Y}-01-01', tz='UTC')
        tr = df[df.index + pd.Timedelta(days=H) < ini]
        te = df[(df.index >= ini) & (df.index < ini + pd.DateOffset(years=1))]
        if te.empty:
            continue
        mu, sd = tr[feats].mean(), tr[feats].std()
        m = LogisticRegression(C=1.0).fit((tr[feats] - mu) / sd, tr['y'])
        p[te.index] = m.predict_proba((te[feats] - mu) / sd)[:, 1]
        coefs.append(m.coef_[0])
    return p.dropna(), np.mean(coefs, axis=0)


def auc_por_año(y, p):
    return {Y: roc_auc_score(y[y.index.year == Y], p[p.index.year == Y])
            for Y in AÑOS if (y.index.year == Y).any() and y[y.index.year == Y].nunique() == 2}


def boot_delta(y, pa, pb, rng):
    y, pa, pb = y.to_numpy(), pa.to_numpy(), pb.to_numpy()
    blq = np.arange(len(y)) // BLOQUE
    nb = blq.max() + 1
    grupos = [np.flatnonzero(blq == b) for b in range(nb)]
    out = np.empty(REPS)
    for i in range(REPS):
        ii = np.concatenate([grupos[j] for j in rng.integers(0, nb, nb)])
        if y[ii].min() == y[ii].max():
            out[i] = np.nan
            continue
        out[i] = roc_auc_score(y[ii], pa[ii]) - roc_auc_score(y[ii], pb[ii])
    return out[~np.isnan(out)]


def main():
    n_rot = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    rng = np.random.default_rng(SEED)
    df = construir()
    print(f"Dias con features: {len(df)}  ({df.index[0].date()} -> {df.index[-1].date()})")
    print(f"Tasa base y=1: {df['y'].mean():.1%}  | por año: "
          + "  ".join(f"{a}:{g.mean():.0%}" for a, g in df['y'].groupby(df.index.year)))
    print("corr(features especialistas, base):")
    print(df[BASE + MACRO + VOL].corr().loc[MACRO + VOL, BASE].round(2).to_string())

    preds, coefs = {}, {}
    for nombre, f in (('base', BASE), ('base+macro', BASE + MACRO), ('base+vol', BASE + VOL),
                      ('solo macro', MACRO), ('solo vol', VOL)):
        preds[nombre], coefs[nombre] = walk_forward(df, f)
    y = df['y'].reindex(preds['base'].index)
    print(f"\nFuera de muestra: {len(y)} dias, y=1 {y.mean():.1%}")
    for nombre, p in preds.items():
        pa = auc_por_año(y, p)
        print(f"  {nombre:<11} AUC {roc_auc_score(y, p):.3f}  | "
              + "  ".join(f"{a}:{v:.2f}" for a, v in pa.items())
              + f"  | coef {np.round(coefs[nombre], 2)}")

    res = {}
    for h, nombre, f_esp in (('H7', 'base+macro', MACRO), ('H8', 'base+vol', VOL)):
        d_obs = roc_auc_score(y, preds[nombre]) - roc_auc_score(y, preds['base'])
        bs = boot_delta(y, preds[nombre], preds['base'], rng)
        p = float((bs <= 0).mean())
        # null por rotacion del bloque de features del especialista
        n = len(df)
        rot = []
        for k in rng.integers(90, n - 90, n_rot):
            dr = df.copy()
            dr[f_esp] = np.roll(df[f_esp].to_numpy(), k, axis=0)
            pr, _ = walk_forward(dr, BASE + f_esp)
            rot.append(roc_auc_score(y, pr.reindex(y.index)) - roc_auc_score(y, preds['base']))
        p_rot = float((np.array(rot) >= d_obs).mean())
        res[h] = p
        print(f"\n{h} ({nombre} vs base): dAUC {d_obs:+.4f}  bootstrap IC90 "
              f"[{np.percentile(bs, 5):+.4f}, {np.percentile(bs, 95):+.4f}]  p = {p:.4f} "
              f"-> {'PASA' if p < 0.025 else 'NO pasa'} (p<0,025)")
        print(f"   null rotacion ({n_rot}): mediana dAUC {np.median(rot):+.4f}  p_rot = {p_rot:.3f}")


if __name__ == '__main__':
    main()
