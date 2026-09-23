"""bull_1d con N velas diarias (lo que baja el bot) vs historia completa.

EMA50/200 con adjust=False arrastran el valor inicial. Para cada dia con al
menos N velas previas, recalcula el regimen con solo las N ultimas y lo
compara con el del backtest (EMA sobre toda la historia).

Uso: C:/Python/python.exe experiments/ejecucion_vivo/regime_warmup.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.v2_engine import PARAMS_V2, _ema

close = pd.read_parquet(ROOT / 'data' / 'btcusdt_1d_v15.parquet')['close']
fast, slow = PARAMS_V2['a_ema_fast_1d'], PARAMS_V2['a_ema_slow_1d']
full = _ema(close, fast) > _ema(close, slow)


def disagreements(window, first):
    bad = []
    for i in range(first, len(close)):
        c = close.iloc[i - window + 1:i + 1]
        if bool(_ema(c, fast).iloc[-1] > _ema(c, slow).iloc[-1]) != bool(full.iloc[i]):
            bad.append(close.index[i])
    return bad


print('| velas en vivo | rango evaluado | dias | desacuerdos | meses |')
print('|--:|---|--:|--:|---|')
for window, first in [(300, 299), (1000, 999), (300, 999)]:
    bad = disagreements(window, first)
    months = sorted({d.strftime('%Y-%m') for d in bad})
    print(f'| {window} | {close.index[first]:%Y-%m-%d} -> {close.index[-1]:%Y-%m-%d} '
          f'| {len(close) - first} | {len(bad)} | {", ".join(months)} |')
