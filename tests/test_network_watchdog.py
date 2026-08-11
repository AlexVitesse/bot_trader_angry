"""El bot debe SALIR tras 3 velas 4h seguidas sin red, para que systemd
(Restart=always) lo reinicie con sockets y DNS limpios.

Sin esto el loop principal se traga las excepciones para siempre y loguea
"Sin senales en este ciclo", indistinguible de un mercado quieto: asi se
perdieron ~2 meses de paper trade (2026-05-19 -> 2026-07-13+).

Uso: python tests/test_network_watchdog.py
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ml_bot


class _Portfolio:
    def __init__(self, con_red):
        self.con_red = con_red
        self.llamadas = 0

    def refresh_balance(self):
        self.llamadas += 1
        return self.con_red


def _bot(con_red):
    """MLBot minimo: solo el camino que toca el watchdog."""
    b = object.__new__(ml_bot.MLBot)
    b.blind_candles = 0
    b.portfolio = _Portfolio(con_red)
    b.v14_mode = True
    # hoy == last_regime_date -> se salta update_regime (necesitaria red)
    b.last_regime_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    b._on_new_candle_v14 = lambda: None
    return b


def demo():
    # --- sin red: aguanta 2 velas, sale en la 3a ---
    b = _bot(con_red=False)
    b._on_new_candle()
    assert b.blind_candles == 1, b.blind_candles
    b._on_new_candle()
    assert b.blind_candles == 2, b.blind_candles

    try:
        b._on_new_candle()
    except SystemExit as e:
        assert e.code == 1, f'exit code {e.code}, esperaba 1'
    else:
        raise AssertionError('no salio tras 3 velas ciegas')

    # --- con red: nunca sale, y una vela buena resetea el contador ---
    b = _bot(con_red=True)
    b.blind_candles = 2
    b._on_new_candle()
    assert b.blind_candles == 0, 'una vela con red debe resetear el contador'

    # SystemExit no lo captura el `except Exception` del loop principal
    assert not issubclass(SystemExit, Exception)

    print('OK: watchdog sale tras 3 velas sin red y resetea al recuperarla')


if __name__ == '__main__':
    demo()
