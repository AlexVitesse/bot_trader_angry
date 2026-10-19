"""E1: V2 frente a comprar BTC con la misma exposicion media. E0: MinTRL.

Pre-registro en README.md (docs/PLAN_GITHUB_2026-10.md). alpha = 0,01.

Uso: python experiments/benchmark_bh/test_benchmark.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments' / 'portfolio_sim'))
from portfolio_sim import FUNDING_8H, PortfolioSim, _funding_btc, cargar_pares  # noqa: E402

COSTE = 0.0006          # comision + slippage por lado, costes nuevos
BLOQUE, REPS = 30, 20_000
SEED = 20261019


# ------------------------------------------------------------------ metricas
def cagr_dd(eq: pd.Series) -> tuple[float, float]:
    años = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / años) - 1
    dd = float((1 - eq / eq.cummax()).max())
    return cagr, dd


def fila(nombre, eq, expo=None):
    c, d = cagr_dd(eq)
    e = f"{expo:>6.3f}" if expo is not None else ' ' * 6
    print(f"  {nombre:<22} expo {e}  CAGR {c:>+7.1%}  DD {d:>6.1%}  CAGR/DD {c / d:>5.2f}")
    return c / d


# ------------------------------------------------------------------ V2
def correr_v2(datos):
    res = PortfolioSim(datos, risk_pct=0.045, max_concurrent=1, max_notional_pct=2.0).run()
    eq4 = res.equity
    close = datos['BTC/USDT']['close'].reindex(eq4.index)
    expo = pd.Series(0.0, index=eq4.index)
    for t in res.trades:
        m = (eq4.index >= t['ts_entrada']) & (eq4.index < t['ts_salida'])
        entrada = datos['BTC/USDT']['open'].loc[t['ts_entrada']]
        expo[m] = t['notional'] * close[m] / entrada / eq4[m]
    return res, eq4, expo


# ------------------------------------------------------------------ B&H
def sigma_gk(df4: pd.DataFrame) -> pd.Series:
    o, h, l, c = (np.log(df4[k]) for k in ('open', 'high', 'low', 'close'))
    gk = 0.5 * (h - l) ** 2 - (2 * np.log(2) - 1) * (c - o) ** 2
    var_d = gk.resample('1D').sum()
    return np.sqrt(var_d.rolling(20).mean() * 365)


def funding_diario(idx: pd.DatetimeIndex) -> pd.Series:
    f = _funding_btc().resample('1D').sum()
    f = f.reindex(idx)
    return f.fillna(3 * FUNDING_8H)      # antes de 2020: constante, como el simulador


def bh(close_d: pd.Series, objetivo: pd.Series, umbral=None, escala=1.0, funding=None):
    """objetivo = peso deseado para el dia t (fijado al cierre de t-1).
    umbral: rebalancear solo si |objetivo - mantenido| / escala > umbral."""
    R = close_d.pct_change().fillna(0).to_numpy()
    w_obj = objetivo.to_numpy()
    fu = np.zeros(len(R)) if funding is None else funding.to_numpy()
    V, w = 1.0, 0.0
    out = np.empty(len(R))
    for t in range(len(R)):
        # rebalanceo al cierre de t-1 = inicio del dia t
        if umbral is None or abs(w_obj[t] - w) / escala > umbral or (w == 0 and w_obj[t] > 0):
            V -= abs(w_obj[t] - w) * V * COSTE
            w = w_obj[t]
        V_ant = V
        V *= 1 + w * R[t]
        V -= w * V_ant * fu[t]
        w = w * (1 + R[t]) / (1 + w * R[t]) if V > 0 else 0
        out[t] = V
    return pd.Series(out, index=close_d.index)


def mixta(eq_a: pd.Series, eq_b: pd.Series) -> pd.Series:
    ra, rb = eq_a.pct_change().fillna(0), eq_b.pct_change().fillna(0)
    a, b, out = 0.5, 0.5, []
    mes = None
    for ts in ra.index:
        if ts.month != mes:                           # primer dia del mes
            tot = a + b
            a, b, mes = tot / 2, tot / 2, ts.month
        a *= 1 + ra[ts]
        b *= 1 + rb[ts]
        out.append(a + b)
    return pd.Series(out, index=ra.index)


# ------------------------------------------------------------------ tests
def idx_bloques(n, reps, rng):
    nb = int(np.ceil(n / BLOQUE))
    i = (rng.integers(0, n, (reps, nb))[:, :, None] + np.arange(BLOQUE)).reshape(reps, -1)
    return i[:, :n] % n


def test_primario(eq_v2, eq_vt, rng):
    r1 = np.log(eq_v2).diff().dropna()
    r2 = np.log(eq_vt).diff().dropna()
    d = (r1 - r2 * r1.std() / r2.std()).to_numpy()
    obs = d.mean()
    dc = d - obs
    medias = np.empty(REPS)
    for k in range(0, REPS, 2000):
        medias[k:k + 2000] = dc[idx_bloques(len(d), 2000, rng)].mean(axis=1)
    return obs * 365, float((medias >= obs).mean())


def test_secundario(eq_v2, eq_vt, rng, reps=5000):
    r1 = eq_v2.pct_change().dropna().to_numpy()
    r2 = eq_vt.pct_change().dropna().to_numpy()
    n = len(r1)
    peor = 0
    for k in range(0, reps, 500):
        ii = idx_bloques(n, 500, rng)
        ratios = []
        for r in (r1, r2):
            eq = np.cumprod(1 + r[ii], axis=1)
            eq = np.hstack([np.ones((len(ii), 1)), eq])
            dd = (1 - eq / np.maximum.accumulate(eq, axis=1)).max(axis=1)
            cagr = eq[:, -1] ** (365 / n) - 1
            ratios.append(cagr / dd)
        peor += int((ratios[0] <= ratios[1]).sum())
    return peor / reps


# ------------------------------------------------------------------ E0
def min_trl(r: np.ndarray) -> tuple[float, float, float, float]:
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    z = stats.norm.ppf(0.95)
    return 1 + (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) * (z / sr) ** 2, sr, g3, g4


# ------------------------------------------------------------------ main
def evaluar(nombre, eq_v2_d, expo_media, close_d, sig, fund, rng, tests=True):
    print(f"\n=== {nombre}: {close_d.index[0].date()} -> {close_d.index[-1].date()} ===")
    sig_ant = sig.shift(1)                            # conocido al cierre de t-1
    sig_star = sig_ant.expanding().median()
    raw = np.minimum(1, sig_star / sig_ant).fillna(0)
    # B&H-vt: exposicion media sobre todos los dias = la de V2
    k = expo_media / raw.mean()
    w_const = pd.Series(expo_media, index=close_d.index)

    eq_w = bh(close_d, w_const, funding=fund)
    eq_vt = bh(close_d, raw * k, umbral=0.1, escala=k, funding=fund)
    eq_w_spot = bh(close_d, w_const)
    eq_vt_spot = bh(close_d, raw * k, umbral=0.1, escala=k)
    eq_bh1 = bh(close_d, pd.Series(1.0, index=close_d.index))   # 100%, spot

    print(f"  k = {k:.3f}")
    fila('V2', eq_v2_d, expo_media)
    fila('B&H-w (perp)', eq_w, expo_media)
    fila('B&H-vt (perp)', eq_vt, float((raw * k).mean()))
    fila('B&H-w (spot)', eq_w_spot, expo_media)
    fila('B&H-vt (spot)', eq_vt_spot, float((raw * k).mean()))
    fila('Mixta 50/50 (perp)', mixta(eq_w, eq_v2_d))
    fila('B&H 100% spot (ref)', eq_bh1, 1.0)
    if not tests:
        return
    obs, p = test_primario(eq_v2_d, eq_vt, rng)
    print(f"\n  H1 primario: mean(d) anualizado {obs:+.2%}  p = {p:.4f}  "
          f"-> {'PASA' if p < 0.01 else 'NO pasa'} (alpha 0,01)")
    p2 = test_secundario(eq_v2_d, eq_vt, rng)
    print(f"  Secundario: P(CAGR/DD V2 <= B&H-vt) en bootstrap pareado = {p2:.3f}")


def main():
    rng = np.random.default_rng(SEED)
    datos = cargar_pares(['BTC/USDT'])
    res, eq4, expo4 = correr_v2(datos)
    m = res.metricas
    print(f"V2 (4,5%, notional 2,0): n={m['n']} WR {m['wr']:.1f}% PF {m['pf']:.2f} "
          f"CAGR {m['cagr']:+.1f}% DD {m['dd']:.1f}%")
    print(f"  tiempo en mercado: {(expo4 > 0).mean():.1%}  exposicion media: {expo4.mean():.3f}  "
          f"media con posicion: {expo4[expo4 > 0].mean():.3f}")

    df4 = pd.read_parquet(ROOT / 'data' / 'BTC_USDT_4h_full.parquet')
    df4 = df4[['open', 'high', 'low', 'close']].astype(float)
    if df4.index.tz is None:
        df4.index = df4.index.tz_localize('UTC')
    df4 = df4[~df4.index.duplicated(keep='last')].sort_index()

    eq_d = eq4.resample('1D').last().dropna()
    close_d = df4['close'].resample('1D').last().reindex(eq_d.index).ffill()
    sig = sigma_gk(df4).reindex(eq_d.index)
    fund = funding_diario(eq_d.index)
    expo_d = expo4.resample('1D').mean().reindex(eq_d.index).fillna(0)

    # primario: periodo pre-registrado (toda la curva de V2)
    evaluar('PRIMARIO (pre-registrado)', eq_d, float(expo4.mean()), close_d, sig, fund, rng)

    # sensibilidad: desde la primera vela en que V2 podia operar (warm-up del regimen)
    d = datos['BTC/USDT']
    col = next((c for c in d.columns if 'ema200' in c.lower()), None)
    t0 = d[col].first_valid_index() if col else res.trades[0]['ts_entrada']
    t0 = t0.normalize()
    m0 = eq_d.index >= t0
    m4 = eq4.index >= t0
    evaluar(f'SENSIBILIDAD (desde {t0.date()}, col {col})', eq_d[m0], float(expo4[m4].mean()),
            close_d[m0], sig[m0], fund[m0], rng, tests=False)

    # E0
    r = np.array([t['r'] for t in res.trades])
    n, sr, g3, g4 = min_trl(r)
    años = (eq4.index[-1] - eq4.index[0]).days / 365.25
    por_año = len(r) / años
    print(f"\n=== E0 MinTRL === SR/trade {sr:.3f}  skew {g3:.2f}  kurt {g4:.2f}")
    print(f"  MinTRL (95%, vs SR 0) = {n:.0f} trades = {n / por_año:.1f} años a {por_año:.1f} trades/año")


if __name__ == '__main__':
    main()
