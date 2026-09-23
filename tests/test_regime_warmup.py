"""bull_1d en vivo (ventana de N velas diarias) == backtest (historia completa).

EMA50/200 con adjust=False arrastran el valor inicial: con 300 velas el
regimen difiere del backtest 29 dias en 2019-2026, justo en los cruces. Con
las 1000 que baja ahora el bot, 0. AUDITORIA_2026-09 §2.2, plan 0.2.

Uso: python -m pytest tests/test_regime_warmup.py
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.v2_engine import PARAMS_V2, _ema

DATA = ROOT / 'data' / 'btcusdt_1d_v15.parquet'


def _disagreements(window: int) -> int:
    close = pd.read_parquet(DATA)['close']
    fast, slow = PARAMS_V2['a_ema_fast_1d'], PARAMS_V2['a_ema_slow_1d']
    full = _ema(close, fast) > _ema(close, slow)
    bad = 0
    for i in range(1000, len(close)):          # mismos dias para ambas ventanas
        c = close.iloc[i - window + 1:i + 1]
        bad += bool(_ema(c, fast).iloc[-1] > _ema(c, slow).iloc[-1]) != bool(full.iloc[i])
    return bad


@pytest.mark.skipif(not DATA.exists(), reason='sin parquet diario local')
def test_1000_bars_match_full_history():
    assert _disagreements(300) > 0             # el bug que se corrige existe
    assert _disagreements(1000) == 0


if __name__ == '__main__':
    print('300:', _disagreements(300), '1000:', _disagreements(1000))
