"""
Ablation: ¿aporta algo F_SHORT a V2?
===================================
Hipotesis: NO. Segun combined_AF/README.md, F_BTC_SHORT in-sample tiene
N=30, PF 1.08, annual +0.2%, p=0.492 — estadisticamente nada. Se incluyo en
V2 por 3 trades ganadores en el OOS Ene-Feb 2026, que es exactamente el
sesgo de seleccion que VERDICTO_RONDA2 seccion 3 advirtio.

Ademas F_SHORT es el UNICO componente que puede disparar en bear (A_LONG y
F_LONG exigen bull_1d=1), asi que desde mayo 2026 es lo unico que opera.

Este test NO introduce ningun parametro nuevo: usa el flag `f_enable_short`
que ya existe en PARAMS_V2. Cero riesgo de overfitting adicional.

Uso: python experiments/f_short_ablation/test_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src import v2_engine as v2

CUTOFF = pd.Timestamp('2025-12-31', tz='UTC')   # cutoff inviolable del proyecto
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
def load_btc_4h() -> pd.DataFrame:
    """Parquet historico (2019 -> feb 2026) + Binance en vivo (-> hoy)."""
    df = pd.read_parquet(ROOT / 'data' / 'BTC_USDT_4h_full.parquet')
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    r = requests.get('https://fapi.binance.com/fapi/v1/klines',
                     params={'symbol': 'BTCUSDT', 'interval': '4h', 'limit': 1500},
                     timeout=30)
    r.raise_for_status()
    live = pd.DataFrame(r.json(), columns=['ts', 'open', 'high', 'low', 'close',
                                           'volume', 'a', 'b', 'c', 'd', 'e', 'f'])
    live['ts'] = pd.to_datetime(live['ts'], unit='ms', utc=True)
    live = live.set_index('ts')[['open', 'high', 'low', 'close', 'volume']].astype(float)

    out = pd.concat([df, live])
    out = out[~out.index.duplicated(keep='last')].sort_index()
    return out


def daily_from_4h(df4: pd.DataFrame) -> pd.DataFrame:
    """Velas diarias exactas construidas desde las 4h (6 barras por dia)."""
    return df4.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                   'close': 'last', 'volume': 'sum'}).dropna()


# ---------------------------------------------------------------------------
def stats(trades: list, label: str) -> dict:
    if not trades:
        return {'label': label, 'n': 0}
    pnl = np.array([t['pnl_pct'] for t in trades])
    eq = np.cumprod(1 + pnl)
    # el 1.0 inicial cuenta como primer pico (ver revision Codex 3.8)
    eq_dd = np.concatenate(([1.0], eq))
    dd = float((1 - eq_dd / np.maximum.accumulate(eq_dd)).max() * 100)

    t0 = pd.Timestamp(trades[0]['ts_entry'])
    t1 = pd.Timestamp(trades[-1]['ts_entry'])
    years = max((t1 - t0).days / 365.25, 0.25)
    annual = float((eq[-1] ** (1 / years) - 1) * 100)

    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() else float('inf')

    # bootstrap: prob. de que la media sea <= 0 por azar
    means = RNG.choice(pnl, size=(3000, len(pnl)), replace=True).mean(axis=1)
    p = float((means <= 0).mean())

    return {'label': label, 'n': len(pnl), 'wr': float((pnl > 0).mean() * 100),
            'pf': pf, 'total': float((eq[-1] - 1) * 100), 'annual': annual,
            'dd': dd, 'p': p}


def show(rows: list, title: str) -> None:
    print(f"\n{title}")
    print(f"  {'config':<26} {'N':>4} {'WR':>6} {'PF':>6} {'total':>9} "
          f"{'anual':>8} {'DD':>7} {'boot p':>8}")
    print("  " + "-" * 80)
    for r in rows:
        if not r['n']:
            print(f"  {r['label']:<26} {0:>4}  (sin trades)")
            continue
        flag = ' *' if r['p'] < 0.05 else ''
        print(f"  {r['label']:<26} {r['n']:>4} {r['wr']:>5.1f}% {r['pf']:>6.2f} "
              f"{r['total']:>+8.1f}% {r['annual']:>+7.1f}% {r['dd']:>6.1f}% "
              f"{r['p']:>7.3f}{flag}")


# ---------------------------------------------------------------------------
def main() -> None:
    df4 = load_btc_4h()
    df1 = daily_from_4h(df4)
    print(f"BTC 4h: {len(df4)} velas  {df4.index[0].date()} -> {df4.index[-1].date()}")

    # explicitos los dos lados: desde 2026-08-10 el default de PARAMS_V2 ya es
    # False (esta ablacion es justamente lo que lo decidio), asi que heredarlo
    # haria que las dos filas fueran identicas.
    con_short = {**v2.PARAMS_V2, 'f_enable_short': True}
    sin_short = {**v2.PARAMS_V2, 'f_enable_short': False}

    ventanas = [
        ('IN-SAMPLE 2020-01-01 -> 2025-12-31 (lo que se valido)',
         pd.Timestamp('2020-01-01', tz='UTC'), CUTOFF),
        ('OOS 2026-01-01 -> hoy (nada de esto se vio al disenar)',
         CUTOFF, df4.index[-1] + pd.Timedelta(days=1)),
    ]

    trades_full = {}
    for titulo, desde, hasta in ventanas:
        rows = []
        for label, params in (('V2 actual (A+F+F_SHORT)', con_short),
                              ('V2 sin F_SHORT', sin_short)):
            # IMPORTANTE: se pasa el df COMPLETO y se restringe solo el rango de
            # ENTRADAS. Recortar el df antes de build_features hacia que el
            # warmup de 220 velas se comiera el inicio de cada ventana (~37
            # dias), perdiendo trades reales. Bug detectado en la revision (3.1).
            feats = v2.build_features(df4, df1, None, params)
            idx = feats.index
            start_i = max(int((idx < desde).sum()), params['min_warmup_bars'])
            end_i = int((idx < hasta).sum())
            if end_i - start_i < 20:
                print(f"\n{titulo}\n  ventana demasiado corta")
                break
            tr = v2.run_v2_backtest(df4, df1, None, params,
                                    start_i=start_i, end_i=end_i)
            trades_full[(titulo, label)] = tr
            rows.append(stats(tr, label))
        show(rows, titulo)

    # --- desglose por componente sobre la historia completa -----------------
    todos = v2.run_v2_backtest(df4, df1, None, con_short)
    print(f"\nDESGLOSE POR COMPONENTE (historia completa, V2 actual)")
    rows = []
    for sig in ('A_LONG', 'F_LONG', 'F_SHORT'):
        rows.append(stats([t for t in todos if t['sig_type'] == sig], sig))
    show(rows, '')

    print("\n  (* = bootstrap p < 0.05.  'total' compone cada trade sobre el sleeve)")
    print("  Nota: F_SHORT es el UNICO que puede disparar con bull_1d=0 (bear).")


if __name__ == '__main__':
    main()
