"""
Frequency sweep: ¿que pasa si obligamos a V2 a operar mas?
==========================================================
Pregunta del usuario: "quiero que opere todos los dias, que se adapte al
mercado y no espere a que se cumplan las condiciones".

En vez de opinar, lo medimos: aflojamos los filtros de V2 uno a uno (y todos
juntos) y observamos la curva frecuencia -> retorno. Si operar mas mejora, se
vera aqui. Si lo destruye, tambien.

Costes aplicados (config/settings.py): 0.04% comision + 0.01% slippage por
lado = 0.10% ida y vuelta, ya incluido en v2_engine.COMMISSION.
El funding se estima aparte segun el tiempo real en mercado.

Uso: python experiments/frequency_sweep/test_frequency.py
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

RNG = np.random.default_rng(42)
FUNDING_8H = 0.00013          # mediana BTC perp (agent_D)


def load_btc() -> tuple[pd.DataFrame, pd.DataFrame]:
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
    df4 = pd.concat([df, live])
    df4 = df4[~df4.index.duplicated(keep='last')].sort_index()
    df1 = df4.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                  'close': 'last', 'volume': 'sum'}).dropna()
    return df4, df1


P = v2.PARAMS_V2
VARIANTES = [
    ('0. V2 tal cual (baseline)',        {}),
    ('1. sin filtro ADX',                {'a_adx_min': 0}),
    ('2. sin filtro volumen',            {'a_vol_ratio_min': 0, 'f_vol_ratio_min': 0}),
    ('3. Donchian 20 (vs 55)',           {'a_donchian_n': 20}),
    ('4. compresion laxa',               {'f_compression_pctile': 0.50,
                                          'f_compression_sustain': 1}),
    ('5. SIN filtro de regimen',         {'a_require_bull': False,
                                          'f_require_regime': False}),
    ('6. TODO suelto (max frecuencia)',  {'a_adx_min': 0, 'a_vol_ratio_min': 0,
                                          'f_vol_ratio_min': 0, 'a_donchian_n': 20,
                                          'f_compression_pctile': 0.50,
                                          'f_compression_sustain': 1,
                                          'f_breakout_n': 5,
                                          'a_require_bull': False,
                                          'f_require_regime': False,
                                          'a_max_bars': 12, 'f_max_bars': 12}),
]


def evaluar(trades: list, years: float) -> dict | None:
    if len(trades) < 5:
        return None
    pnl = np.array([t['pnl_pct'] for t in trades])
    eq = np.cumprod(1 + pnl)
    dd = float((1 - eq / np.maximum.accumulate(eq)).max() * 100)
    bruto = float((eq[-1] ** (1 / years) - 1) * 100)

    bars = sum(t['bars'] for t in trades)
    tim = bars * 4 / (years * 365.25 * 24) * 100
    funding = FUNDING_8H * (bars / 2) / years * 100

    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() else float('inf')
    means = RNG.choice(pnl, size=(2000, len(pnl)), replace=True).mean(axis=1)

    return {'n': len(pnl), 'per_year': len(pnl) / years,
            'wr': float((pnl > 0).mean() * 100), 'pf': pf,
            'bruto': bruto, 'tim': tim, 'funding': funding,
            'neto': bruto - funding, 'dd': dd,
            'coste': len(pnl) / years * 0.10,
            'p': float((means <= 0).mean())}


def main() -> None:
    df4, df1 = load_btc()
    years = len(df4) * 4 / 24 / 365.25
    print(f"BTC 4h: {df4.index[0].date()} -> {df4.index[-1].date()} ({years:.1f} anos)")
    print(f"Coste por trade: 0.10% ida y vuelta | funding {FUNDING_8H*100:.3f}%/8h\n")

    print(f"  {'variante':<30} {'trades/ano':>10} {'WR':>6} {'PF':>6} "
          f"{'bruto':>8} {'coste':>8} {'funding':>8} {'NETO':>8} {'DD':>7} {'p':>7}")
    print("  " + "-" * 108)

    for nombre, override in VARIANTES:
        params = {**P, **override}
        try:
            tr = v2.run_v2_backtest(df4, df1, None, params)
        except Exception as e:
            print(f"  {nombre:<30} error: {e}")
            continue
        m = evaluar(tr, years)
        if m is None:
            print(f"  {nombre:<30} {'(muestra insuficiente)':>10}")
            continue
        mark = ' *' if m['p'] < 0.05 else ''
        print(f"  {nombre:<30} {m['per_year']:>10.0f} {m['wr']:>5.1f}% {m['pf']:>6.2f} "
              f"{m['bruto']:>+7.1f}% {m['coste']:>7.1f}% {m['funding']:>7.1f}% "
              f"{m['neto']:>+7.1f}% {m['dd']:>6.1f}% {m['p']:>6.3f}{mark}")

    print("\n  'coste' = trades/ano x 0.10%, ya descontado dentro de 'bruto'.")
    print("  'NETO' = bruto - funding.  (* = bootstrap p < 0.05)")
    print("\n  Para operar 1 vez/dia (365 trades/ano) el coste solo de comisiones")
    print("  es 36.5%/ano, mas ~13% de funding si estas dentro todo el tiempo.")


if __name__ == '__main__':
    main()
