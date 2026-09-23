"""
Exchange de papel: la misma interfaz ccxt que usa PortfolioManager, pero los
fills se simulan con precios en vivo de Binance y no se manda ninguna orden.

Decisión del usuario (2026-09-23): V2 pasa a paper interno y la cuenta demo de
Binance queda para el bot agresivo con ML. Así los dos corren en paralelo sin
pisarse las posiciones.

Modelo de ejecución (deliberadamente simple):
  - market: fill al `last` del ticker ± SLIPPAGE, comisión taker.
  - limit: no soportada -> excepción; PortfolioManager cae a market
    (el perfil v2 pone ML_ENTRY_LIMIT_TIMEOUT_S = 0 de todas formas).
  - STOP_MARKET: se registra pero no se dispara. El stop lo ejecuta
    PortfolioManager al ver el precio en cada tick (30 s); si el bot está caído
    no hay stop. ponytail: sin motor de stops propio; añadirlo si el paper
    tiene que sobrevivir a caídas largas.
  - Sin funding.
Estado (balance, posiciones, órdenes, ledger de income) en un JSON.
"""
import json
import time
from pathlib import Path

TAKER_FEE = 0.0004      # Binance futures VIP0 (medido en demo-fapi 2026-09-23)
SLIPPAGE = 0.0001


class PaperExchange:
    def __init__(self, public, state_file: Path, capital: float):
        self.public = public                  # ccxt sin auth: precios y velas
        self.markets = public.markets
        self.state_file = Path(state_file)
        if self.state_file.exists():
            self.s = json.loads(self.state_file.read_text())
        else:
            self.s = {'balance': capital, 'positions': {}, 'orders': {},
                      'income': [], 'next_id': 1}
            self._save()

    # -------------------------------------------------------------- estado
    def _save(self):
        tmp = self.state_file.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.s))
        tmp.replace(self.state_file)

    def _id(self) -> str:
        i = self.s['next_id']
        self.s['next_id'] = i + 1
        return f"paper-{i}"

    def _income(self, symbol, kind, amount):
        self.s['income'].append({'symbol': symbol.replace('/', ''), 'incomeType': kind,
                                 'income': str(amount), 'time': int(time.time() * 1000)})

    # ------------------------------------------------------ datos de mercado
    def fetch_ticker(self, symbol):
        return self.public.fetch_ticker(symbol)

    def fetch_ohlcv(self, symbol, timeframe='4h', since=None, limit=None, params=None):
        return self.public.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

    def fetch_order_book(self, symbol, limit=None):
        return self.public.fetch_order_book(symbol, limit)

    def amount_to_precision(self, symbol, amount):
        return self.public.amount_to_precision(symbol, amount)

    def price_to_precision(self, symbol, price):
        return self.public.price_to_precision(symbol, price)

    def set_leverage(self, leverage, symbol):
        return None

    # --------------------------------------------------------------- cuenta
    def fetch_balance(self):
        b = self.s['balance']
        return {'USDT': {'total': b, 'free': b}}

    def fetch_positions(self, symbols=None):
        out = []
        for sym, p in self.s['positions'].items():
            if symbols and sym not in symbols:
                continue
            out.append({'symbol': f"{sym}:USDT", 'contracts': p['qty'], 'side': p['side'],
                        'entryPrice': p['entry'], 'leverage': 5})
        return out

    def fapiPrivateGetIncome(self, params):
        sym = params.get('symbol')
        t0, t1 = params.get('startTime', 0), params.get('endTime', float('inf'))
        return [r for r in self.s['income']
                if r['symbol'] == sym and t0 <= r['time'] <= t1]

    # --------------------------------------------------------------- órdenes
    def create_order(self, symbol, type, side, amount, price=None, params=None):
        params = params or {}
        amount = float(amount)
        if type == 'STOP_MARKET':
            oid = self._id()
            self.s['orders'][oid] = {'id': oid, 'status': 'open', 'type': type,
                                     'stopPrice': params.get('stopPrice')}
            self._save()
            return dict(self.s['orders'][oid])
        if type != 'market':
            raise Exception(f"paper: tipo de orden no soportado ({type})")

        last = float(self.public.fetch_ticker(symbol)['last'])
        px = last * (1 + SLIPPAGE if side == 'buy' else 1 - SLIPPAGE)
        fee = amount * px * TAKER_FEE
        self.s['balance'] -= fee
        self._income(symbol, 'COMMISSION', -fee)

        pos = self.s['positions'].get(symbol)
        if params.get('reduceOnly') or (pos and (pos['side'] == 'long') != (side == 'buy')):
            if not pos:
                raise Exception(f"paper: reduceOnly sin posicion en {symbol}")
            q = min(amount, pos['qty'])
            d = 1 if pos['side'] == 'long' else -1
            pnl = (px - pos['entry']) * q * d
            self.s['balance'] += pnl
            self._income(symbol, 'REALIZED_PNL', pnl)
            pos['qty'] = round(pos['qty'] - q, 12)
            if pos['qty'] <= 0:
                del self.s['positions'][symbol]
        elif pos:
            tot = pos['qty'] + amount
            pos['entry'] = (pos['entry'] * pos['qty'] + px * amount) / tot
            pos['qty'] = tot
        else:
            self.s['positions'][symbol] = {'side': 'long' if side == 'buy' else 'short',
                                           'qty': amount, 'entry': px}
        oid = self._id()
        self.s['orders'][oid] = {'id': oid, 'status': 'closed', 'type': 'market',
                                 'filled': amount, 'average': px}
        self._save()
        return dict(self.s['orders'][oid])

    def cancel_order(self, oid, symbol=None):
        o = self.s['orders'].get(oid)
        if o and o['status'] == 'open':
            o['status'] = 'canceled'
            self._save()
        return o

    def fetch_order(self, oid, symbol=None):
        o = self.s['orders'].get(oid)
        if o is None:
            raise Exception(f"paper: orden {oid} no existe")
        return dict(o)
