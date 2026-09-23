"""
Motor AGRESIVO: candidatos en todos los regimenes + ML que decide si operar y
cuanto arriesgar. Funciones puras: las usan el backtest
(experiments/agresivo/backtest.py) y el bot en vivo. Una sola fuente de verdad.

Diseno fijado ANTES de ver resultados (experiments/agresivo/README.md):
  - Candidatos en la vela CERRADA i, una por (vela, direccion), prioridad
    comp > don55 > don20 > mr:
      don20/don55  close rompe max(high)/min(low) de las N velas previas
      comp         ruptura de 20 velas con bb_width[i-1] < q20(bb_width, 100)[i-1]
      mr           close < banda inferior BB(20,2) -> LONG; > superior -> SHORT
  - Etiqueta: trade con salida tipo V2 A (trailing sin look-ahead, trail =
    clip(atr_pct*2.5, 2,5%, 6%), max 60 velas), entrada open i+1, stop con gap,
    0,06%/lado + funding constante 0,013%/8h (longs pagan, shorts cobran).
    r_R = PnL / trail (resultado en multiplos del riesgo), y = r_R > 0.
  - Modelo: HistGradientBoostingClassifier(max_depth=3, min_samples_leaf=200,
    l2=1.0, max_iter=200, lr=0.05) con sample_weight = unicidad, calibrado con
    Platt sobre el ultimo 20% (temporal) del train. Ventana EXPANSIVA.
  - Riesgo: Kelly f = p - (1-p)/b (b = win_R/loss_R medio del train);
    riesgo = clip(0,5*f, 0, 10%); no opera si < 1%; x max(0, 1 - dd/0,5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

PARES = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
         'DOGE/USDT', 'ADA/USDT', 'LINK/USDT', 'AVAX/USDT']
TIPOS = ['comp', 'don55', 'don20', 'mr']          # orden = prioridad
FEAT = ['atr_pct', 'bb_width', 'dist_ema_d', 'ret20_d', 'ret60_d', 'btc_ema_d',
        'dir'] + [f't_{t}' for t in TIPOS]

COM = 0.0006                 # comision + slippage por lado
FUND_8H = 0.00013            # funding constante (longs pagan)
TRAIL_MULT, TRAIL_MIN, TRAIL_MAX = 2.5, 0.025, 0.06
MAX_BARS = 60
PURGA_VELAS = 60
RIESGO_MAX, RIESGO_MIN, DD_CORTE = 0.10, 0.01, 0.5
BAR = pd.Timedelta(hours=4)


# ------------------------------------------------------------------ features
def _dist_ema200_diaria(df: pd.DataFrame) -> pd.Series:
    d1 = df['close'].resample('1D').last().dropna()
    ema = d1.ewm(span=200, adjust=False).mean().shift(1)      # dia anterior
    return df['close'] / ema.reindex(df.index, method='ffill') - 1


def features(df_4h: pd.DataFrame, df_btc_4h: pd.DataFrame) -> pd.DataFrame:
    """Features por vela usando solo informacion hasta el cierre de la vela."""
    df = df_4h[['open', 'high', 'low', 'close']].astype(float).copy()
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df['atr_pct'] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    df['bb_lo'], df['bb_up'] = mid - 2 * sd, mid + 2 * sd
    df['bb_width'] = (4 * sd) / mid
    df['dist_ema'] = _dist_ema200_diaria(df)
    df['ret20'] = c / c.shift(20) - 1
    df['ret60'] = c / c.shift(60) - 1
    btc = df_btc_4h[['close']].astype(float)
    df['btc_ema'] = _dist_ema200_diaria(btc).reindex(df.index, method='ffill')

    for n in (20, 55):
        df[f'hi{n}'] = h.rolling(n).max().shift(1)
        df[f'lo{n}'] = l.rolling(n).min().shift(1)
    q20 = df['bb_width'].rolling(100).quantile(0.20)
    df['comprimido'] = (df['bb_width'].shift(1) < q20.shift(1))
    return df


def candidatos(f: pd.DataFrame, i: int) -> list:
    """(tipo, direccion) en la vela cerrada i; uno por direccion."""
    r = f.iloc[i]
    c, out = r['close'], []
    for d in (1, -1):
        dispara = {
            'comp': bool(r['comprimido']) and (c > r['hi20'] if d == 1 else c < r['lo20']),
            'don55': c > r['hi55'] if d == 1 else c < r['lo55'],
            'don20': c > r['hi20'] if d == 1 else c < r['lo20'],
            'mr': c < r['bb_lo'] if d == 1 else c > r['bb_up'],
        }
        for t in TIPOS:
            if dispara[t]:
                out.append((t, d))
                break
    return out


def _candidatos_vector(f: pd.DataFrame) -> pd.DataFrame:
    """Todos los candidatos de un par (misma logica que candidatos(), vectorizada)."""
    c = f['close']
    filas = []
    for d in (1, -1):
        dispara = {
            'comp': f['comprimido'] & ((c > f['hi20']) if d == 1 else (c < f['lo20'])),
            'don55': (c > f['hi55']) if d == 1 else (c < f['lo55']),
            'don20': (c > f['hi20']) if d == 1 else (c < f['lo20']),
            'mr': (c < f['bb_lo']) if d == 1 else (c > f['bb_up']),
        }
        ya = pd.Series(False, index=f.index)
        for t in TIPOS:
            m = dispara[t].fillna(False) & ~ya
            ya |= m
            filas.append(pd.DataFrame({'i': np.flatnonzero(m.values), 'tipo': t, 'dir': d}))
    return pd.concat(filas, ignore_index=True).sort_values(['i', 'dir']).reset_index(drop=True)


def fila_modelo(f: pd.DataFrame, i: int, tipo: str, d: int) -> dict:
    r = f.iloc[i]
    x = {'atr_pct': r['atr_pct'], 'bb_width': r['bb_width'],
         'dist_ema_d': r['dist_ema'] * d, 'ret20_d': r['ret20'] * d,
         'ret60_d': r['ret60'] * d, 'btc_ema_d': r['btc_ema'] * d, 'dir': d}
    x.update({f't_{t}': float(t == tipo) for t in TIPOS})
    return x


def trail_de(atr_pct: float) -> float:
    return float(min(max(atr_pct * TRAIL_MULT, TRAIL_MIN), TRAIL_MAX))


# ------------------------------------------------------------------ etiqueta
def etiquetar(f: pd.DataFrame) -> pd.DataFrame:
    """Eventos etiquetados de un par. Salida tipo V2: stop previo primero, gap."""
    o, h, l, c = (f[k].values for k in ('open', 'high', 'low', 'close'))
    atr = f['atr_pct'].values
    cand = _candidatos_vector(f)
    filas = []
    for i, tipo, d in cand.itertuples(index=False):
        e = i + 1
        if e >= len(f) or not np.isfinite(atr[i]):
            continue
        trail = trail_de(atr[i])
        ent = o[e]
        stop, ext, sal, sal_j = ent * (1 - d * trail), ent, None, None
        for j in range(e + 1, min(e + MAX_BARS + 1, len(f))):
            if (d == 1 and l[j] <= stop) or (d == -1 and h[j] >= stop):
                sal, sal_j = (min(stop, o[j]) if d == 1 else max(stop, o[j])), j
                break
            if j - e >= MAX_BARS:
                sal, sal_j = c[j], j
                break
            if d == 1:
                ext = max(ext, h[j])
                stop = max(stop, ext * (1 - trail))
            else:
                ext = min(ext, l[j])
                stop = min(stop, ext * (1 + trail))
        if sal is None:
            continue                                   # sin resolver
        barras = sal_j - e
        pnl = d * (sal - ent) / ent - 2 * COM - d * FUND_8H * 0.5 * barras
        fila = fila_modelo(f, i, tipo, d)
        fila.update({'t': f.index[i], 'ini': f.index[e], 'fin': f.index[sal_j],
                     'tipo': tipo, 'trail': trail, 'pnl': pnl,
                     'r_R': pnl / trail, 'y': int(pnl > 0)})
        filas.append(fila)
    return pd.DataFrame(filas)


def eventos_panel(feats: dict) -> pd.DataFrame:
    evs = []
    for par, f in feats.items():
        e = etiquetar(f)
        e['par'] = par
        evs.append(e)
    return pd.concat(evs, ignore_index=True).dropna(subset=FEAT).sort_values('t')


def unicidad(ev: pd.DataFrame) -> np.ndarray:
    """Unicidad media por evento, por par (Lopez de Prado cap. 4)."""
    w = np.empty(len(ev))
    for par, g in ev.groupby('par'):
        ini = ((g['ini'] - g['ini'].min()) // BAR).astype(int).values
        fin = ((g['fin'] - g['ini'].min()) // BAR).astype(int).values
        conc = np.zeros(fin.max() + 2)
        np.add.at(conc, ini, 1)
        np.add.at(conc, fin + 1, -1)
        conc = np.cumsum(conc)
        pos = ev.index.get_indexer(g.index)
        w[pos] = [np.mean(1.0 / conc[a:b + 1]) for a, b in zip(ini, fin)]
    return w


# ------------------------------------------------------------------ modelo
def entrenar(ev: pd.DataFrame) -> dict:
    """HGB + Platt (ultimo 20% temporal del train). Devuelve el bundle."""
    ev = ev.sort_values('t').reset_index(drop=True)
    w = unicidad(ev)
    corte = int(len(ev) * 0.8)
    X, y = ev[FEAT].values, ev['y'].values
    m = HistGradientBoostingClassifier(max_depth=3, min_samples_leaf=200,
                                       l2_regularization=1.0, max_iter=200,
                                       learning_rate=0.05, random_state=0)
    m.fit(X[:corte], y[:corte], sample_weight=w[:corte])
    raw = m.predict_proba(X[corte:])[:, 1]
    platt = LogisticRegression(C=1e6).fit(_logit(raw)[:, None], y[corte:],
                                          sample_weight=w[corte:])
    win = ev.loc[ev['r_R'] > 0, 'r_R'].mean()
    loss = -ev.loc[ev['r_R'] <= 0, 'r_R'].mean()
    return {'modelo': m, 'platt': platt, 'b': float(win / loss),
            'win_R': float(win), 'loss_R': float(loss), 'n': len(ev),
            'hasta': ev['fin'].max()}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def probabilidad(bundle: dict, X: np.ndarray) -> np.ndarray:
    raw = bundle['modelo'].predict_proba(X)[:, 1]
    return bundle['platt'].predict_proba(_logit(raw)[:, None])[:, 1]


def riesgo_de(p: float, b: float, dd: float) -> float:
    f = p - (1 - p) / b
    r = min(max(0.5 * f, 0.0), RIESGO_MAX)
    if r < RIESGO_MIN:
        return 0.0
    return r * max(0.0, 1 - dd / DD_CORTE)


def decidir(bundle: dict, fila: dict, tipo: str, direccion: int, dd_actual: float):
    """None si no opera; si opera {'p', 'riesgo'}."""
    x = dict(fila)
    x.update({'dir': direccion}, **{f't_{t}': float(t == tipo) for t in TIPOS})
    X = np.array([[x[k] for k in FEAT]], dtype=float)
    if not np.isfinite(X).all():
        return None
    p = float(probabilidad(bundle, X)[0])
    r = riesgo_de(p, bundle['b'], dd_actual)
    return {'p': p, 'riesgo': r} if r > 0 else None


def reentrenar(feats: dict, t: pd.Timestamp) -> dict:
    """Ventana expansiva: todos los eventos cuya salida es anterior a t - purga."""
    ev = eventos_panel(feats)
    return entrenar(ev[ev['fin'] < t - PURGA_VELAS * BAR])


# ------------------------------------------------------------------ vivo
def señal_vivo(bundle: dict, dfs_por_par: dict, df_btc: pd.DataFrame,
               posiciones_abiertas, dd_actual: float) -> list:
    """Senales en la ultima vela CERRADA de cada par (dfs sin la vela en curso)."""
    out = []
    for par, df in dfs_por_par.items():
        if par in posiciones_abiertas:
            continue
        f = features(df, df_btc)
        i = len(f) - 1
        mejor = None
        for tipo, d in candidatos(f, i):
            dec = decidir(bundle, fila_modelo(f, i, tipo, d), tipo, d, dd_actual)
            if dec and (mejor is None or dec['riesgo'] > mejor[0]['riesgo']):
                mejor = (dec, tipo, d)
        if mejor is None:
            continue
        dec, tipo, d = mejor
        out.append({
            'pair': par, 'direction': int(d), 'side': 'LONG' if d == 1 else 'SHORT',
            'price': float(f['close'].iloc[i]), 'trail_mode': 'tight',
            'trail_fixed_dist': trail_de(f['atr_pct'].iloc[i]),
            'max_bars': MAX_BARS, 'risk_pct': float(dec['riesgo']),
            'confidence': float(dec['p']), 'setup': f'agr_{tipo}',
            'engine': 'agresivo',
        })
    return sorted(out, key=lambda s: -s['risk_pct'])
