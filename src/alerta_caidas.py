"""
Alerta de caídas de BTC (E7 / E12) grabada en vivo. NO veta ni cambia nada:
solo deja un registro diario para medirla con muestra nueva.

Modelo: el de experiments/especialistas/ (E7, base + macro), con la etiqueta
"BTC cae ≥10% en algún momento de los próximos 10 días". Alerta activa si la
p̂ del último día cerrado ≥ cuantil 0,80 de las p̂ del train (regla de
experiments/veto_oro/, E12). E12 no pasó el null (p=0,105) y E11 mostró que el
efecto es el z-score de 60 días de BTC, no el oro. Por eso es solo registro.

Se reentrena cada día con todo lo que tiene etiqueta (ventana expansiva).
ponytail: BTC diario del perpetuo de Binance (desde 2019-09), el backtest usó
spot desde 2019-01; la diferencia de precio es despreciable.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

H, CAIDA, CUANTIL = 10, 0.10, 0.80
FEATS = ['vol20', 'dist_ema200', 'ret20', 'oro_z', 'dxy_20', 'spx_dd']
TICKERS = {'oro': 'GC=F', 'dxy': 'DX-Y.NYB', 'spx': '^GSPC'}


def _alinear(s: pd.Series, idx) -> pd.Series:
    """Fecha hábil t disponible desde t+1: último valor estrictamente anterior a d."""
    return s.set_axis(s.index + pd.Timedelta(days=1)).reindex(idx, method='ffill')


def tabla(d: pd.DataFrame, macro: dict) -> pd.DataFrame:
    """Etiqueta y features por día UTC (igual que experiments/especialistas/)."""
    c = d['close']
    t = pd.DataFrame(index=d.index)
    fut_min = pd.concat([d['low'].shift(-k) for k in range(1, H + 1)], axis=1).min(axis=1, skipna=False)
    t['y'] = (fut_min <= (1 - CAIDA) * c).astype(float).where(fut_min.notna())
    t['vol20'] = np.log(c).diff().rolling(20).std()
    t['dist_ema200'] = (c / c.ewm(span=200, adjust=False).mean() - 1).where(np.arange(len(c)) >= 200)
    t['ret20'] = c / c.shift(20) - 1
    x = np.log(c / _alinear(macro['oro'], c.index))
    t['oro_z'] = (x - x.rolling(60).mean()) / x.rolling(60).std()
    dxy, spx = macro['dxy'], macro['spx']
    t['dxy_20'] = _alinear(dxy / dxy.shift(20) - 1, c.index)
    t['spx_dd'] = _alinear(spx / spx.rolling(60).max() - 1, c.index)
    return t


def evaluar(t: pd.DataFrame) -> dict:
    """Entrena con los días etiquetados y evalúa el último día con features."""
    tr = t.dropna()
    ult = t.dropna(subset=FEATS).iloc[-1]
    mu, sd = tr[FEATS].mean(), tr[FEATS].std()
    m = LogisticRegression(C=1.0).fit((tr[FEATS] - mu) / sd, tr['y'])
    umbral = float(np.quantile(m.predict_proba((tr[FEATS] - mu) / sd)[:, 1], CUANTIL))
    p = float(m.predict_proba(((ult[FEATS] - mu) / sd).to_frame().T)[:, 1][0])
    return {'dia': ult.name.strftime('%Y-%m-%d'), 'p': p, 'umbral': umbral,
            'activa': p >= umbral, 'n_train': len(tr), 'tasa_base': float(tr['y'].mean())}


def descargar_btc(exchange) -> pd.DataFrame:
    """Velas diarias cerradas de BTC/USDT desde 2019 (paginado)."""
    filas, since = [], exchange.parse8601('2019-01-01T00:00:00Z')
    while True:
        lote = exchange.fetch_ohlcv('BTC/USDT', '1d', since=since, limit=1000)
        filas += lote
        if len(lote) < 1000:
            break
        since = lote[-1][0] + 1
    d = pd.DataFrame(filas, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    d.index = pd.to_datetime(d['ts'], unit='ms', utc=True)
    d = d[~d.index.duplicated()].drop(columns='ts').astype(float)
    hoy = pd.Timestamp(datetime.now(timezone.utc).date(), tz='UTC')
    return d[d.index < hoy]                       # sin el día en curso


def descargar_macro() -> dict:
    import yfinance as yf
    out = {}
    for k, tk in TICKERS.items():
        s = yf.download(tk, start='2018-01-01', progress=False, auto_adjust=False)['Close'].squeeze().dropna()
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).tz_localize('UTC')
        out[k] = s
    return out


def calcular(exchange) -> dict:
    return evaluar(tabla(descargar_btc(exchange), descargar_macro()))


def registrar(f: Path, res: dict):
    """Una línea JSON por día (no duplica el mismo día)."""
    f = Path(f)
    if f.exists() and any(json.loads(x).get('dia') == res['dia'] for x in f.read_text().splitlines() if x):
        return
    with f.open('a') as fh:
        fh.write(json.dumps({**res, 'calculado': datetime.now(timezone.utc).isoformat()}) + '\n')


def ultimo(f: Path) -> dict | None:
    try:
        return json.loads(Path(f).read_text().splitlines()[-1])
    except (OSError, IndexError, ValueError):
        return None


def resumen(r: dict) -> str:
    estado = '🔴 ACTIVA' if r['activa'] else '🟢 no'
    return (f"🧪 Alerta caída E7 ({r['dia']}, solo registro): {estado} | "
            f"p={r['p']:.0%} umbral={r['umbral']:.0%}")
