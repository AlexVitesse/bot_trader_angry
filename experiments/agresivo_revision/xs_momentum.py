"""Momentum relativo (cross-sectional) neutral a mercado sobre los 9 pares.

PRE-REGISTRO (escrito el 2026-11-20 ANTES de correr; una sola variante, m=1).
Origen: el unico hilo que el proyecto dejo abierto (predictibilidad_fuentes:
panel 1d cross-sectional h=5, IC +0,069, p Holm 0,144). Esa celda ya se vio, asi
que la historia hasta 2026-02 NO es prueba limpia; lo nuevo es 2026-03 -> 11.
Direccion elegida por la literatura (Liu-Tsyvinski-Wu: momentum 1-4 semanas),
no mirando el signo del ridge.

Regla: cierres diarios UTC de los 9 pares del agresivo. Cada 7 dias se ordenan
por retorno de 20 dias; LARGO los 3 mejores, CORTO los 3 peores, 1/6 del equity
cada uno (bruto 1x, neto 0). Se mantiene 7 dias sin tocar. Coste 0,06%/lado
sobre el volumen de cada rebalanceo. Funding no modelado (largo paga, corto
cobra; a neto 0 se cancela en primera aproximacion). Sin stops.

Pasa si: p por bloques < 0,05 Y p contra el null sintetico < 0,05 (bloques de
7 y de 30 dias) con media semanal neta > 0.
 - p bloques: bootstrap circular de bloques de 4 semanas sobre el retorno
   semanal neto, 20.000 replicas, H0 media <= 0.
 - null sintetico: se permutan bloques de FILAS (dias) de la matriz de
   retornos diarios, la misma permutacion para los 9 pares: conserva la deriva
   de cada moneda, la correlacion entre monedas y la autocorrelacion dentro
   del bloque; rompe la persistencia del ranking. 2.000 series por longitud.
   p = fraccion con media semanal neta >= observada.
Descriptivo: por ano, tramo nuevo 2026-03 -> 11, correlacion con BTC.
Uso: python experiments/agresivo_revision/xs_momentum.py
"""
import numpy as np
import pandas as pd
from datos import cargar

LOOK, HOLD, K, COM, SEED = 20, 7, 3, 0.0006, 20261120


def estrategia(D):
    """D: retornos log diarios (T x N). Devuelve retorno neto por periodo e indices."""
    cum = np.vstack([np.zeros(D.shape[1]), np.cumsum(D, axis=0)])   # cum[t] = hasta dia t
    n = D.shape[1]
    w_prev, out, idx = np.zeros(n), [], []
    for t in range(LOOK, len(D) - HOLD + 1, HOLD):
        orden = np.argsort(cum[t] - cum[t - LOOK], kind='stable')
        w = np.zeros(n)
        w[orden[-K:]], w[orden[:K]] = 1 / (2 * K), -1 / (2 * K)
        R = np.exp(cum[t + HOLD] - cum[t]) - 1
        rp = w @ R
        out.append(rp - COM * np.abs(w - w_prev).sum())
        w_prev = w * (1 + R) / (1 + rp)                             # pesos derivados
        idx.append(t)
    return np.array(out), np.array(idx)


def barajar(D, L, rng):
    b = [D[i:i + L] for i in range(0, len(D), L)]
    return np.concatenate([b[k] for k in rng.permutation(len(b))])


def p_bloques(r, L=4, reps=20_000):
    rng, n = np.random.default_rng(SEED), len(r)
    nb = int(np.ceil(n / L))
    ini = rng.integers(0, n, (reps, nb))
    mu = r[((ini[:, :, None] + np.arange(L)) % n).reshape(reps, -1)[:, :n]].mean(axis=1)
    return float(np.mean(mu - r.mean() >= r.mean()))


def metricas(r, por_año=365.25 / HOLD):
    eq = np.cumprod(1 + r)
    dd = float((1 - np.r_[1, eq] / np.maximum.accumulate(np.r_[1, eq])).max())
    cagr = eq[-1] ** (por_año / len(r)) - 1
    return (f'n {len(r)} | CAGR {cagr:+.1%} | DD {dd:.1%} | CAGR/DD {cagr / dd:.2f} | '
            f'WR {np.mean(r > 0):.1%} | PF {r[r > 0].sum() / -r[r <= 0].sum():.2f} | '
            f'Sharpe {r.mean() / r.std() * np.sqrt(por_año):.2f} | media {r.mean():+.3%}/periodo')


def main():
    datos = cargar()
    px = pd.DataFrame({p: d['close'].resample('1D').last() for p, d in datos.items()}).dropna()
    D = np.log(px).diff().dropna()
    # autocomprobacion: con retornos identicos entre pares solo quedan costes
    r0, _ = estrategia(np.tile(D.values[:, :1], (1, 9)))
    assert (r0 <= 1e-12).all(), 'pares identicos no pueden ganar'

    r, idx = estrategia(D.values)
    fechas = D.index[idx - 1]                      # cierre del dia de la senal
    print(f'ventana {D.index[0]:%Y-%m-%d} -> {D.index[-1]:%Y-%m-%d}, {D.shape[1]} pares')
    print('historia completa:', metricas(r))
    print(f'p bloques (4 semanas, media > 0): {p_bloques(r):.4f}')
    rng = np.random.default_rng(SEED)
    for L in (7, 30):
        nulo = np.array([estrategia(barajar(D.values, L, rng))[0].mean() for _ in range(2000)])
        print(f'null sintetico bloques {L} dias: mediana {np.median(nulo):+.3%} | '
              f'p = {np.mean(nulo >= r.mean()):.4f}')
    s = pd.Series(r, index=fechas)
    print('\npor ano (suma de retornos semanales, n, WR):')
    print(s.groupby(s.index.year).agg(['sum', 'size', lambda x: (x > 0).mean()]).round(3).to_string())
    nuevo = s[s.index >= '2026-03-01']
    print('\ntramo NO visto antes (2026-03 ->):', metricas(nuevo.values))
    btc = np.array([np.exp(D['BTC/USDT'].values[t:t + HOLD].sum()) - 1 for t in idx])
    print(f'correlacion semanal con BTC: {np.corrcoef(r, btc)[0, 1]:+.2f}')


if __name__ == '__main__':
    main()
