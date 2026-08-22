"""Un fallo de red al pedir velas NO es "mercado quieto".

En el log del 19-22 ago 2026, 6 de 21 velas 4h quedaron ciegas por
NameResolutionError del VPS de madrugada. El bot logueaba "Sin senales en este
ciclo", indistinguible de no haber senal. Una senal perdida ahi no se recupera.

`_ohlcv` reintenta antes de rendirse. Este test falla contra el codigo viejo
(que llamaba a `exchange.fetch_ohlcv` directo, sin reintento).

Uso: python tests/test_ohlcv_retry.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ml_strategy_v15 as strat


class _Exchange:
    """Falla las primeras `n_fallos` llamadas, luego devuelve datos."""

    def __init__(self, n_fallos):
        self.n_fallos = n_fallos
        self.llamadas = 0

    def fetch_ohlcv(self, pair, timeframe, limit=None):
        self.llamadas += 1
        if self.llamadas <= self.n_fallos:
            raise ConnectionError('Temporary failure in name resolution')
        return [[0, 1.0, 2.0, 0.5, 1.5, 100.0]] * (limit or 1)


def test_reintenta_y_se_recupera():
    ex = _Exchange(n_fallos=2)
    out = strat._ohlcv(ex, 'BTC/USDT', '4h', 420, tries=3, delay=0)
    assert ex.llamadas == 3, f'esperaba 3 intentos, hubo {ex.llamadas}'
    assert len(out) == 420, f'esperaba 420 velas, llegaron {len(out)}'
    print('  OK  reintenta tras fallo transitorio y devuelve datos')


def test_no_reintenta_si_va_bien():
    ex = _Exchange(n_fallos=0)
    strat._ohlcv(ex, 'BTC/USDT', '4h', 10, tries=3, delay=0)
    assert ex.llamadas == 1, f'no debe reintentar si funciona: {ex.llamadas}'
    print('  OK  una sola llamada cuando la red va bien')


def test_propaga_si_la_red_esta_muerta():
    """Agotados los intentos, la excepcion SUBE. El caller la loguea como
    error de fetch, no como 'sin senales' — que es justo lo que enmascaro
    los 2 meses de paper trade perdidos."""
    ex = _Exchange(n_fallos=99)
    try:
        strat._ohlcv(ex, 'BTC/USDT', '4h', 420, tries=3, delay=0)
    except ConnectionError:
        assert ex.llamadas == 3, f'esperaba 3 intentos, hubo {ex.llamadas}'
        print('  OK  agota los intentos y propaga la excepcion')
        return
    raise AssertionError('debia propagar ConnectionError, no tragarsela')


if __name__ == '__main__':
    print('test_ohlcv_retry')
    test_reintenta_y_se_recupera()
    test_no_reintenta_si_va_bien()
    test_propaga_si_la_red_esta_muerta()
    print('todo OK')
