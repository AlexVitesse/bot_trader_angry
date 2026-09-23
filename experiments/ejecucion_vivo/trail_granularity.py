"""Trail actualizado por vela 4h (backtest) vs 1h (aprox. del bot por tick).

Mismos trades de run_v2_backtest desde 2022-01 (donde hay datos 1h): misma
entrada (close de la vela de señal), mismo trail_dist, mismo max_bars. Solo
cambia cada cuanto sube el peak: `_sim_long_trailing` sobre velas 1h con
max_bars*4 (la salida por timeout cae en el mismo close).

Uso: C:/Python/python.exe experiments/ejecucion_vivo/trail_granularity.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.v2_engine import (PARAMS_V2, _sim_long_trailing, _sim_short_trailing,
                           run_v2_backtest)

D = ROOT / 'data'
df4 = pd.read_parquet(D / 'btcusdt_4h_v15.parquet')
df1d = pd.read_parquet(D / 'btcusdt_1d_v15.parquet')
df1h = pd.read_parquet(D / 'btcusdt_1h.parquet')

trades = run_v2_backtest(df4, df1d)
h_start, h_end = df1h.index[0], df1h.index[-1]
rows = []
for t in trades:
    ts = pd.Timestamp(t['ts_entry'])
    max_bars = PARAMS_V2['a_max_bars' if t['sig_type'] == 'A_LONG' else 'f_max_bars']
    close_ts = ts + pd.Timedelta(hours=3)            # ultima vela 1h de la 4h de señal
    if ts < h_start or close_ts + pd.Timedelta(hours=4 * max_bars) > h_end:
        continue
    eb = df1h.index.get_loc(close_ts)
    sim = _sim_long_trailing if t['side'] == 'LONG' else _sim_short_trailing
    _, _, pnl1h, _ = sim(df1h, eb, t['entry_price'], t['trail_dist'],
                         max_bars * 4, PARAMS_V2['commission'])
    rows.append({'ts': ts, 'pnl_4h': t['pnl_pct'], 'pnl_1h': pnl1h})

r = pd.DataFrame(rows)


def stats(p):
    w, l = p[p > 0].sum(), -p[p <= 0].sum()
    return (len(p), f'{p.sum():+.1%}', f'{(p > 0).mean():.0%}',
            f'{w / l:.2f}', f'{p.mean():+.2%}')


print('| granularidad del trail | n | suma PnL | WR | PF | media/trade |')
print('|---|--:|--:|--:|--:|--:|')
print('| vela 4h (backtest) | %d | %s | %s | %s | %s |' % stats(r['pnl_4h']))
print('| vela 1h (aprox. vivo) | %d | %s | %s | %s | %s |' % stats(r['pnl_1h']))
changed = ((r['pnl_4h'] - r['pnl_1h']).abs() > 1e-9).sum()
flips = ((r['pnl_4h'] > 0) != (r['pnl_1h'] > 0)).sum()
print(f'\nTrades con distinto PnL: {changed}/{len(r)}  |  cambian de signo: {flips}')
print(f'Rango: {r.ts.min():%Y-%m-%d} -> {r.ts.max():%Y-%m-%d}')
