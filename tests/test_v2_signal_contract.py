"""El payload de _generate_v2_signal debe cumplir el contrato que consume
ml_bot._execute_v14_signal / portfolio_manager.open_position:

  - direction: int 1 (LONG) / -1 (SHORT)   -> open_position hace `direction == 1`
  - price: float                            -> open_position lo exige posicional

Bug historico: el payload V2 traia direction='LONG' (str) y no traia 'price'.
Un LONG se habria abierto como SHORT y ademas reventaba con KeyError('price').

Uso: python tests/test_v2_signal_contract.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ml_strategy_v15 as mod


class _FakeExchange:
    """fetch_ohlcv devuelve velas planas; el engine real esta stubbeado."""
    def fetch_ohlcv(self, pair, tf, limit=None):
        return [[i * 1000, 1.0, 1.0, 1.0, 1.0, 1.0] for i in range(limit or 300)]


def _payload(side):
    strat = object.__new__(mod.MLStrategyV15)   # sin cargar modelos
    strat._sizing = {'BTC/USDT': 1.0}
    original = mod._v2_engine.get_live_signal
    mod._v2_engine.get_live_signal = lambda *a, **k: {
        'side': side, 'sig_type': f'F_{side}', 'entry_price': 65000.0,
        'trail_dist': 0.03, 'max_bars': 40, 'atr_pct': 0.012,
        'ts_entry': '2026-08-09 00:00:00+00:00', 'regime': 'BEAR/RANGE',
    }
    try:
        sigs = strat._generate_v2_signal('BTC/USDT', _FakeExchange(), None)
    finally:
        mod._v2_engine.get_live_signal = original
    assert len(sigs) == 1, f'esperaba 1 senal, obtuve {len(sigs)}'
    return sigs[0]


def demo():
    assert mod.V2_AVAILABLE, 'v2_engine no importable'

    long_sig = _payload('LONG')
    short_sig = _payload('SHORT')

    for s in (long_sig, short_sig):
        assert isinstance(s['direction'], int), \
            f"direction debe ser int, es {type(s['direction']).__name__}"
        assert 'price' in s, "falta 'price' (open_position lo exige)"
        assert isinstance(s['price'], float) and s['price'] > 0

    assert long_sig['direction'] == 1, 'LONG debe mapear a 1'
    assert short_sig['direction'] == -1, 'SHORT debe mapear a -1'
    # el bug real: 'LONG' == 1 -> False -> open_position abria un short
    assert (long_sig['direction'] == 1) is True

    # sizing por par debe viajar en el payload (ML_V15_SIZING)
    assert long_sig['sizing_mult'] == 1.0

    print('OK: contrato de senal V2 valido (direction int + price)')


if __name__ == '__main__':
    demo()
