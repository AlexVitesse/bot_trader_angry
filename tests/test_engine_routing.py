"""El motor de cada par lo fija ML_V15_ENGINE, no la existencia de un JSON.

Antes, borrar strategies/btc_v15/models/meta_v2_paper.json resucitaba en
silencio el GBM SHORT (AUDITORIA_2026-09 §5). Ahora ese directorio ni existe
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
    assert not (ROOT / 'strategies').exists()
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
