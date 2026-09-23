"""El exchange de papel del perfil v2: misma interfaz que ccxt para
PortfolioManager, fills simulados con precios en vivo, estado persistente.

Uso: python -m pytest tests/test_paper_exchange.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paper_exchange import PaperExchange, SLIPPAGE, TAKER_FEE
from src.portfolio_manager import PortfolioManager

PAIR = 'BTC/USDT'


class Public:
    """ccxt publico falso: solo precios."""
    markets = {}

    def __init__(self):
        self.last = 100_000.0

    def fetch_ticker(self, s):
        return {'last': self.last}

    def fetch_ohlcv(self, *a, **k):
        return []

    def amount_to_precision(self, s, a):
        return f'{a:.3f}'

    def price_to_precision(self, s, p):
        return f'{p:.1f}'


def test_round_trip_with_portfolio_manager():
    tmp = Path(tempfile.mkdtemp())
    pub = Public()
    ex = PaperExchange(pub, tmp / 'paper.json', 10_000.0)
    pm = PortfolioManager(ex, db_path=tmp / 'ml.db')
    pm.balance = pm.peak_balance = 10_000.0
    assert pm.open_position(PAIR, 1, 1.0, 'BULL', 100_000.0, 0.02,
                            tp_pct_override=0.06, sl_pct_override=0.03,
                            trail_mode='tight', trail_fixed_dist=0.03,
                            max_hold_override=60, risk_override=0.02)
    pos = pm.positions[PAIR]
    entry = 100_000.0 * (1 + SLIPPAGE)
    assert abs(pos.entry_price - entry) < 1e-6
    assert abs(pos.notional - 10_000 * 0.02 / 0.03) < 150      # riesgo override
    assert ex.fetch_positions()[0]['contracts'] == pos.quantity
    assert pos.sl_order_id.startswith('paper-')

    pub.last = 105_000.0
    trade = pm._close_position(PAIR, 105_000.0, 'TRAIL')
    exit_px = 105_000.0 * (1 - SLIPPAGE)
    q = trade['quantity']
    esperado = 10_000.0 + (exit_px - entry) * q - (entry + exit_px) * q * TAKER_FEE
    assert abs(ex.fetch_balance()['USDT']['total'] - esperado) < 1e-6
    assert not ex.fetch_positions()

    # el income alimenta reconcile_closed_trades igual que Binance
    inc = ex.fapiPrivateGetIncome({'symbol': 'BTCUSDT'})
    assert {r['incomeType'] for r in inc} == {'REALIZED_PNL', 'COMMISSION'}
    # el estado sobrevive a un reinicio
    ex2 = PaperExchange(pub, tmp / 'paper.json', 1.0)
    assert ex2.fetch_balance()['USDT']['total'] == ex.fetch_balance()['USDT']['total']


if __name__ == '__main__':
    test_round_trip_with_portfolio_manager()
    print('OK')
