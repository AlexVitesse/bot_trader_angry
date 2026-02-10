import json
import threading
import time
import websocket
import pandas as pd
from datetime import datetime
from typing import Callable, Optional

class BinanceWebsocketManager:
    """
    Gestor de Websocket para Binance Futures.
    Recibe actualizaciones de velas en tiempo real.
    """
    def __init__(self, symbol: str, timeframe: str, on_candle_closed: Callable, testnet: bool = False):
        self.symbol = symbol.lower().replace('/', '')
        self.interval = timeframe
        self.on_candle_closed = on_candle_closed
        self.testnet = testnet
        
        # URL base
        if self.testnet:
            self.base_url = "wss://stream.binancefuture.com/ws"
        else:
            self.base_url = "wss://fstream.binance.com/ws"
            
        self.ws: Optional[websocket.WebSocketApp] = None
        self.wst: Optional[threading.Thread] = None
        self.running = False
        self.last_candle = None

    def start(self):
        """Inicia el thread del Websocket."""
        self.running = True
        stream_url = f"{self.base_url}/{self.symbol}@kline_{self.interval}"
        
        print(f"[WS] Conectando a {stream_url}...")
        
        self.ws = websocket.WebSocketApp(
            stream_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )

        self.wst = threading.Thread(target=self.ws.run_forever)
        self.wst.daemon = True
        self.wst.start()

    def stop(self):
        """Detiene la conexion."""
        self.running = False
        if self.ws:
            self.ws.close()

    def _on_open(self, ws):
        print("[WS] Conexion establecida")

    def _on_error(self, ws, error):
        if self.running:
            print(f"[WS] Error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        print("[WS] Conexion cerrada")
        if self.running:
            print("[WS] Reconectando en 5s...")
            time.sleep(5)
            self.start()

    def _on_message(self, ws, message):
        """
        Procesa el mensaje JSON de Binance.
        Estructura kline:
        {
            "e": "kline",
            "k": {
                "t": 123400000, // Start time
                "T": 123460000, // Close time
                "o": "0.0010",  // Open
                "c": "0.0020",  // Close
                "h": "0.0025",  // High
                "l": "0.0015",  // Low
                "v": "1000",    // Volume
                "x": false      // Is closed?
            }
        }
        """
        try:
            data = json.loads(message)
            if 'k' in data:
                k = data['k']
                
                # Convertir a formato amigable
                candle = {
                    'timestamp': pd.to_datetime(k['t'], unit='ms'),
                    'open': float(k['o']),
                    'high': float(k['h']),
                    'low': float(k['l']),
                    'close': float(k['c']),
                    'volume': float(k['v']),
                    'closed': k['x'] # Booleano: True si la vela cerro
                }
                
                # Guardar ultima vela (para monitoreo en tiempo real)
                self.last_candle = candle
                
                # Si la vela cerro, notificar al callback
                if candle['closed']:
                    self.on_candle_closed(candle)
                    
        except Exception as e:
            print(f"[WS] Error procesando mensaje: {e}")

    def get_latest_price(self) -> float:
        """Retorna el ultimo precio conocido."""
        if self.last_candle:
            return self.last_candle['close']
        return 0.0
