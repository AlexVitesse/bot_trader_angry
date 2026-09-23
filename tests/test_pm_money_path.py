"""Camino del dinero de PortfolioManager contra un exchange falso.

Cubre las Fases 1 y 2.3 de docs/PLAN_MEJORAS_2026-09.md:
  1.1 sin pausa por racha de perdidas
  1.2 kill switch persistido
  1.3 posicion 'pending' -> adopcion con parametros del motor / descarte
  1.4 reemplazo del stop sin dejar sl_order_id huerfano
  2.3 trail V2 solo con velas 4h cerradas

Uso: python -m pytest tests/test_pm_money_path.py   (o python tests/...)
"""
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.portfolio_manager import PortfolioManager

PAIR = 'BTC/USDT'
BAR_MS = 4 * 3600 * 1000


class FakeExchange:
    def __init__(self, price=100_000.0):
        self.price = price
        self.position = None          # (side, contracts, entry)
        self.orders = {}              # id -> dict
        self.cancelled = []
        self.bars = []
        self.fail_market = None       # 'before' | 'after' fill
        self.fail_stop = False
        self._n = 0

    def _id(self):
        self._n += 1
        return str(self._n)

    def fetch_positions(self, symbols=None):
        if not self.position:
            return []
        side, qty, entry = self.position
        return [{'symbol': PAIR + ':USDT', 'contracts': qty, 'side': side,
                 'entryPrice': entry, 'leverage': 3}]

    def create_order(self, symbol, type, side, amount, params=None):
        if type == 'STOP_MARKET':
            if self.fail_stop:
                raise Exception('stop rechazado')
            oid = self._id()
            self.orders[oid] = {'id': oid, 'status': 'open',
                                'stopPrice': params['stopPrice']}
            return self.orders[oid]
        if self.fail_market == 'before':
            raise Exception('insufficient margin')
        if (params or {}).get('reduceOnly'):
            self.position = None
        else:
            self.position = ('long' if side == 'buy' else 'short', amount, self.price)
        if self.fail_market == 'after':
            raise Exception('read timeout')     # la orden SI entro
        oid = self._id()
        return {'id': oid, 'average': self.price, 'filled': amount}

    def cancel_order(self, oid, symbol):
        self.cancelled.append(oid)
        self.orders[oid]['status'] = 'canceled'

    def fetch_order(self, oid, symbol):
        return {'id': oid, 'average': self.price, 'status': 'closed'}

    def fetch_ticker(self, pair):
        return {'last': self.price}

    def fetch_balance(self):
        return {'USDT': {'total': 10_000.0}}

    def fetch_ohlcv(self, pair, tf, since=None, limit=None):
        return [b for b in self.bars if since is None or b[0] >= since]

    def set_leverage(self, lev, symbol):
        pass

    def amount_to_precision(self, symbol, q):
        return f'{q:.3f}'

    def price_to_precision(self, symbol, p):
        return f'{p:.1f}'


def _pm(ex, db=None):
    db = db or Path(tempfile.mkdtemp()) / 'ml.db'
    pm = PortfolioManager(ex, db_path=db)
    pm.balance = pm.peak_balance = 10_000.0
    return pm, db


def _open_v2(pm, price=100_000.0):
    return pm.open_position(PAIR, 1, 1.0, 'BULL', price, 0.02,
                            tp_pct_override=0.06, sl_pct_override=0.03,
                            trail_mode='tight', trail_fixed_dist=0.03,
                            max_hold_override=60)


def test_pending_adopted_with_engine_params():
    ex = FakeExchange()
    ex.fail_market = 'after'
    pm, db = _pm(ex)
    assert _open_v2(pm) is False
    assert pm.positions[PAIR].status == 'pending'

    ex.fail_market, ex.position = None, ('long', 0.06, 100_500.0)
    pm2, _ = _pm(ex, db)                         # "reinicio"
    pm2.sync_positions()
    pos = pm2.positions[PAIR]
    assert pos.status == 'open'
    assert pos.trail_mode == 'tight' and pos.trail_fixed_dist == 0.03
    assert pos.max_hold == 60
    assert pos.entry_price == 100_500.0 and pos.quantity == 0.06
    assert abs(pos.trail_sl - 100_500.0 * 0.97) < 1e-6
    assert pos.sl_order_id                       # stop colocado en exchange


def test_pending_discarded_when_order_never_filled():
    ex = FakeExchange()
    ex.fail_market = 'before'
    pm, db = _pm(ex)
    assert _open_v2(pm) is False
    pm.update_positions()                        # resuelve sin reiniciar
    assert PAIR not in pm.positions
    pm2, _ = _pm(ex, db)
    pm2.sync_positions()
    assert not pm2.positions


def test_duplicate_adoption_keeps_signal_params():
    ex = FakeExchange()
    ex.position = ('long', 0.05, 99_000.0)
    pm, _ = _pm(ex)
    assert _open_v2(pm) is False
    pos = pm.positions[PAIR]
    assert pos.trail_mode == 'tight' and pos.max_hold == 60
    assert pos.entry_price == 99_000.0


def test_no_pause_after_losing_streak():
    ex = FakeExchange()
    pm, _ = _pm(ex)
    for _ in range(3):
        ex.price = 100_000.0
        assert _open_v2(pm)
        ex.price = 95_000.0
        trade = pm._close_position(PAIR, 95_000.0, 'SL')
        assert trade['pnl'] < 0
    assert not pm.paused


def test_kill_switch_survives_restart():
    ex = FakeExchange()
    pm, db = _pm(ex)
    pm.balance = 5_000.0                          # DD 50% > ML_MAX_DD_PCT
    assert pm.check_risk() is False and pm.killed
    pm2, _ = _pm(ex, db)
    pm2.sync_positions()
    assert pm2.killed


def test_failed_stop_replace_keeps_live_stop():
    ex = FakeExchange()
    pm, _ = _pm(ex)
    assert _open_v2(pm)
    pos = pm.positions[PAIR]
    old_id = pos.sl_order_id
    ex.fail_stop = True
    pm._move_exchange_sl(pos, 99_000.0)
    assert pos.sl_order_id == old_id
    assert old_id not in ex.cancelled
    ex.fail_stop = False
    pm._move_exchange_sl(pos, 99_000.0)
    assert pos.sl_order_id != old_id and old_id in ex.cancelled


def test_trail_moves_only_with_closed_bars():
    ex = FakeExchange()
    pm, _ = _pm(ex)
    assert _open_v2(pm)
    pos = pm.positions[PAIR]
    sl0 = pos.trail_sl

    ex.price = 110_000.0                           # tick alto: no mueve el stop
    pm.update_positions()
    assert pos.trail_sl == sl0

    now = int(time.time() * 1000)
    pos.entry_time -= timedelta(hours=4)           # que la vela de entrada ya cerro
    entry_bar = int(pos.entry_time.timestamp() * 1000) // BAR_MS * BAR_MS
    ex.bars = [[entry_bar - BAR_MS, 0, 200_000.0, 0, 0, 0],   # antes de entrar
               [entry_bar, 0, 105_000.0, 0, 0, 0],             # cerrada
               [entry_bar + BAR_MS, 0, 120_000.0, 0, 0, 0]]    # en curso
    assert entry_bar + 2 * BAR_MS > now            # la ultima sigue abierta
    old_id = pos.sl_order_id
    pm.update_trail_on_closed_bars()
    assert pos.peak_price == 105_000.0
    assert abs(pos.trail_sl - 105_000.0 * 0.97) < 1e-6
    assert ex.orders[pos.sl_order_id]['stopPrice'] == round(105_000.0 * 0.97, 1)
    assert old_id in ex.cancelled


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('OK', name)
