"""
Binance Futures Exchange Client
================================
Cliente para interactuar con Binance Futures API.
Soporta tanto Testnet como Live.
"""

import sys
import time
import hmac
import hashlib
import logging
import requests
from pathlib import Path

# Agregar raiz al path
sys.path.insert(0, str(Path(__file__).parent.parent))
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

from config.settings import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    BINANCE_BASE_URL,
    TRADING_MODE,
    SYMBOL,
    LEVERAGE
)


class BinanceClient:
    """Cliente para Binance Futures."""

    def __init__(self):
        self.api_key = BINANCE_API_KEY
        self.api_secret = BINANCE_API_SECRET
        self.base_url = BINANCE_BASE_URL
        self.time_offset = 0
        self._sync_time()

    def _sync_time(self):
        """Sincroniza el tiempo local con el servidor de Binance."""
        try:
            response = requests.get(f'{self.base_url}/fapi/v1/time', timeout=5)
            server_time = response.json()['serverTime']
            self.time_offset = server_time - int(time.time() * 1000)
        except Exception as e:
            logger.warning(f"[WARN] No se pudo sincronizar tiempo: {e}")
            self.time_offset = 0

    def _get_timestamp(self) -> int:
        """Obtiene timestamp ajustado."""
        return int(time.time() * 1000) + self.time_offset

    def _sign(self, params: Dict) -> str:
        """Firma los parametros con HMAC SHA256."""
        query = '&'.join([f'{k}={v}' for k, v in params.items()])
        signature = hmac.new(
            self.api_secret.encode(),
            query.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """Realiza una request a la API."""
        if params is None:
            params = {}

        headers = {'X-MBX-APIKEY': self.api_key}

        if signed:
            params['timestamp'] = self._get_timestamp()
            params['signature'] = self._sign(params)

        url = f'{self.base_url}{endpoint}'

        if method == 'GET':
            response = requests.get(url, params=params, headers=headers, timeout=10)
        elif method == 'POST':
            response = requests.post(url, params=params, headers=headers, timeout=10)
        elif method == 'DELETE':
            response = requests.delete(url, params=params, headers=headers, timeout=10)
        else:
            raise ValueError(f"Metodo no soportado: {method}")

        if response.status_code != 200:
            raise Exception(f"API Error {response.status_code}: {response.text}")

        return response.json()

    # ==================== MARKET DATA ====================

    def get_ticker(self, symbol: str = None) -> Dict:
        """Obtiene el precio actual."""
        symbol = symbol or SYMBOL.replace('/', '')
        return self._request('GET', '/fapi/v1/ticker/price', {'symbol': symbol})

    def get_klines(self, symbol: str = None, interval: str = '5m', limit: int = 100) -> List:
        """Obtiene velas historicas."""
        symbol = symbol or SYMBOL.replace('/', '')
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        return self._request('GET', '/fapi/v1/klines', params)

    def get_funding_rate(self, symbol: str = None) -> Optional[float]:
        """Obtiene el funding rate actual. Cambia cada 8 horas."""
        symbol = symbol or SYMBOL.replace('/', '')
        try:
            data = self._request('GET', '/fapi/v1/premiumIndex', {'symbol': symbol})
            return float(data.get('lastFundingRate', 0))
        except Exception as e:
            logger.warning(f"[WARN] No se pudo obtener funding rate: {e}")
            return None

    def get_open_interest(self, symbol: str = None) -> Optional[float]:
        """Obtiene el Open Interest actual (contratos abiertos en USDT)."""
        symbol = symbol or SYMBOL.replace('/', '')
        try:
            data = self._request('GET', '/fapi/v1/openInterest', {'symbol': symbol})
            return float(data.get('openInterest', 0))
        except Exception as e:
            logger.warning(f"[WARN] No se pudo obtener open interest: {e}")
            return None

    def get_long_short_ratio(self, symbol: str = None, period: str = '5m') -> Optional[float]:
        """Obtiene el Long/Short ratio global (cuentas). Actualiza cada 5 min.
        Retorna ratio: >1 = mas longs, <1 = mas shorts.
        Nota: endpoint de datos publicos, puede no estar en testnet."""
        symbol = symbol or SYMBOL.replace('/', '')
        try:
            data = self._request('GET', '/futures/data/globalLongShortAccountRatio',
                                 {'symbol': symbol, 'period': period, 'limit': 1})
            if data and len(data) > 0:
                return float(data[0].get('longShortRatio', 1.0))
        except Exception as e:
            logger.warning(f"[WARN] No se pudo obtener long/short ratio: {e}")
        return None

    # ==================== ACCOUNT ====================

    def get_balance(self) -> List[Dict]:
        """Obtiene el balance de la cuenta."""
        return self._request('GET', '/fapi/v2/balance', signed=True)

    def get_usdt_balance(self) -> float:
        """Obtiene el balance disponible en USDT."""
        balances = self.get_balance()
        usdt = next((b for b in balances if b['asset'] == 'USDT'), None)
        return float(usdt['availableBalance']) if usdt else 0.0

    def get_positions(self) -> List[Dict]:
        """Obtiene todas las posiciones."""
        return self._request('GET', '/fapi/v2/positionRisk', signed=True)

    def get_position(self, symbol: str = None) -> Optional[Dict]:
        """Obtiene la posicion de un simbolo especifico."""
        symbol = symbol or SYMBOL.replace('/', '')
        positions = self.get_positions()
        for pos in positions:
            if pos['symbol'] == symbol:
                amt = float(pos['positionAmt'])
                if amt != 0:
                    return {
                        'symbol': symbol,
                        'side': 'long' if amt > 0 else 'short',
                        'size': abs(amt),
                        'entry_price': float(pos['entryPrice']),
                        'unrealized_pnl': float(pos['unRealizedProfit']),
                        'leverage': int(pos['leverage'])
                    }
        return None

    def set_leverage(self, leverage: int, symbol: str = None) -> Dict:
        """Configura el leverage para un simbolo."""
        symbol = symbol or SYMBOL.replace('/', '')
        params = {'symbol': symbol, 'leverage': leverage}
        return self._request('POST', '/fapi/v1/leverage', params, signed=True)

    # ==================== ORDERS ====================

    def place_order(
        self,
        side: str,  # 'BUY' o 'SELL'
        quantity: float,
        order_type: str = 'MARKET',
        symbol: str = None,
        price: float = None,
        stop_price: float = None,
        reduce_only: bool = False
    ) -> Dict:
        """
        Coloca una orden.

        Args:
            side: 'BUY' o 'SELL'
            quantity: Cantidad en contratos
            order_type: 'MARKET', 'LIMIT', 'STOP_MARKET', 'TAKE_PROFIT_MARKET'
            symbol: Par de trading (default: BTC/USDT)
            price: Precio limite (solo para LIMIT)
            stop_price: Precio de stop (para STOP_MARKET)
            reduce_only: Si es True, solo reduce posicion existente
        """
        symbol = symbol or SYMBOL.replace('/', '')

        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type,
            'quantity': quantity
        }

        # Forzar respuesta con datos de fill (avgPrice, fills)
        if order_type == 'MARKET':
            params['newOrderRespType'] = 'RESULT'

        if order_type == 'LIMIT':
            params['price'] = price
            params['timeInForce'] = 'GTC'

        if stop_price and order_type in ['STOP_MARKET', 'TAKE_PROFIT_MARKET']:
            params['stopPrice'] = stop_price

        if reduce_only:
            params['reduceOnly'] = 'true'

        return self._request('POST', '/fapi/v1/order', params, signed=True)

    def market_buy(self, quantity: float, symbol: str = None) -> Dict:
        """Orden de compra a mercado (LONG)."""
        return self.place_order('BUY', quantity, 'MARKET', symbol)

    def market_sell(self, quantity: float, symbol: str = None) -> Dict:
        """Orden de venta a mercado (SHORT)."""
        return self.place_order('SELL', quantity, 'MARKET', symbol)

    def close_position(self, symbol: str = None) -> Optional[Dict]:
        """Cierra la posicion abierta."""
        position = self.get_position(symbol)
        if not position:
            return None

        side = 'SELL' if position['side'] == 'long' else 'BUY'
        return self.place_order(side, position['size'], 'MARKET', symbol, reduce_only=True)

    def place_stop_loss(self, stop_price: float, quantity: float, side: str, symbol: str = None) -> Dict:
        """Coloca una orden de Stop Loss."""
        # Para cerrar LONG, vendemos. Para cerrar SHORT, compramos.
        close_side = 'SELL' if side == 'long' else 'BUY'
        return self.place_order(
            close_side,
            quantity,
            'STOP_MARKET',
            symbol,
            stop_price=stop_price,
            reduce_only=True
        )

    def cancel_all_orders(self, symbol: str = None) -> Dict:
        """Cancela todas las ordenes abiertas."""
        symbol = symbol or SYMBOL.replace('/', '')
        return self._request('DELETE', '/fapi/v1/allOpenOrders', {'symbol': symbol}, signed=True)

    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Obtiene ordenes abiertas."""
        symbol = symbol or SYMBOL.replace('/', '')
        return self._request('GET', '/fapi/v1/openOrders', {'symbol': symbol}, signed=True)


# Singleton para usar en todo el proyecto
client = BinanceClient()


if __name__ == "__main__":
    print(f"[TEST] Conectando a Binance ({TRADING_MODE})...")

    # Test basico
    ticker = client.get_ticker()
    print(f"[OK] Precio BTC/USDT: ${float(ticker['price']):,.2f}")

    balance = client.get_usdt_balance()
    print(f"[OK] Balance USDT: ${balance:,.2f}")

    position = client.get_position()
    print(f"[OK] Posicion: {position if position else 'Sin posicion'}")
