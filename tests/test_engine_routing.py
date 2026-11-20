"""El motor de cada par lo fija ML_V15_ENGINE, no la existencia de un JSON.

Antes, borrar strategies/btc_v15/models/meta_v2_paper.json resucitaba en
silencio el GBM SHORT (AUDITORIA_2026-09 §5). Ahora ningun archivo decide el motor
y un par sin motor conocido no opera. Plan 6.1.

Uso: python -m pytest tests/test_engine_routing.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import ml_strategy_v15 as mod


def _strategy(engine):
    s = object.__new__(mod.MLStrategyV15)
    s.pairs = list(engine)
    s._sizing = {p: 1.0 for p in engine}
    s._engine = dict(engine)
    return s


def test_routes_by_flag_without_any_model_files():
    calls = []
    s = _strategy({'BTC/USDT': 'v2', 'ETH/USDT': 'gbm'})
    s._generate_v2_signal = lambda pair, ex: calls.append(pair) or []
    assert s.load_models() == 1
    s.generate_signals(exchange=None)
    assert calls == ['BTC/USDT']


def test_no_ml_attributes_left():
    s = mod.MLStrategyV15()
    assert not hasattr(s, 'short_model')
    assert s.load_models() == len(s.pairs)


def test_df_4h_since_acepta_str_y_timestamp_con_tz():
    # 2026-11-20: el vivo paso un Timestamp con tz y pd.Timestamp(.., tz=) reventaba.
    import pandas as pd

    class Ex:
        def fetch_ohlcv(self, pair, tf, since=None, limit=None):
            return [[since + i * 14_400_000, 1, 1, 1, 1, 1] for i in range(3)]

    for since in ('2020-01-01', pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=750)):
        df = mod.MLStrategyV15._df_4h(None, Ex(), 'BTC/USDT', since=since)
        assert len(df) == 2                       # sin la vela en curso
