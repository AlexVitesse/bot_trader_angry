"""El hold maximo lo manda el MOTOR, no ML_MAX_HOLD.

El payload V2 ya llevaba `max_bars` (60 para A, 40 para F) pero
`open_position` lo ignoraba y usaba `ML_MAX_HOLD` (15 velas en RANGE = 2,5
dias). Ese recorte cuesta dinero: medido sobre 6,5 anos en
`experiments/max_bars/`, 15/15 da PF 1,68 / +16,8% / DD 17,0% frente al
60/40 del motor con PF 1,83 / +19,1% / DD 14,7%.

Este test falla contra el codigo viejo (no reenviaba max_bars).

Uso: python tests/test_max_hold_propagation.py
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ml_bot
from src.portfolio_manager import PortfolioManager


class _Portfolio:
    """Captura los kwargs con los que ml_bot abre la posicion."""

    def __init__(self):
        self.kwargs = None

    def can_open(self, pair, direction):
        return True

    def open_position(self, **kwargs):
        self.kwargs = kwargs
        return False          # corta el flujo: no queremos el camino de alertas


class _Strategy:
    regime = 'RANGE'


def _bot(portfolio):
    bot = ml_bot.MLBot.__new__(ml_bot.MLBot)   # sin __init__: no toca la red
    bot.portfolio = portfolio
    bot.strategy = _Strategy()
    return bot


def _payload(**extra):
    base = {
        'pair': 'BTC/USDT', 'direction': 1, 'confidence': 1.0,
        'price': 65_000.0, 'tp_pct': 0.09, 'sl_pct': 0.045,
        'sizing_mult': 1.0, 'trail_mode': 'tight', 'trail_fixed_dist': 0.045,
    }
    base.update(extra)
    return base


def test_open_position_acepta_el_override():
    params = inspect.signature(PortfolioManager.open_position).parameters
    assert 'max_hold_override' in params, \
        'open_position debe aceptar max_hold_override'
    assert params['max_hold_override'].default is None, \
        'el override debe ser opcional (default None) para no romper V9/V14'
    print('  OK  open_position acepta max_hold_override opcional')


def test_reenvia_el_max_bars_del_motor():
    pf = _Portfolio()
    _bot(pf)._execute_v14_signal(_payload(max_bars=60))
    assert pf.kwargs is not None, 'open_position no llego a llamarse'
    got = pf.kwargs.get('max_hold_override')
    assert got == 60, f'esperaba max_hold_override=60, llego {got!r}'
    print('  OK  el max_bars del motor llega a open_position')


def test_sin_max_bars_no_rompe():
    """Las senales V9/V14 no llevan max_bars: debe pasar None y que
    open_position caiga en ML_MAX_HOLD como siempre."""
    pf = _Portfolio()
    _bot(pf)._execute_v14_signal(_payload())
    assert pf.kwargs.get('max_hold_override') is None, \
        'sin max_bars en el payload el override debe ser None'
    print('  OK  sin max_bars en el payload, se mantiene el comportamiento viejo')


if __name__ == '__main__':
    print('test_max_hold_propagation')
    test_open_position_acepta_el_override()
    test_reenvia_el_max_bars_del_motor()
    test_sin_max_bars_no_rompe()
    print('todo OK')
