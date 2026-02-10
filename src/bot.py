"""
Binance Scalper Bot - Loop Principal
=====================================
Bot de trading automatizado v6.7 SMART METRALLADORA (Websocket).
Ejecutar: poetry run python src/bot.py
"""

import sys
import time
import signal
import pandas as pd
from datetime import datetime
from pathlib import Path

# Agregar raiz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exchange import client
from src.strategy import strategy, Signal
from src.trader import trader
from src.websocket_manager import BinanceWebsocketManager
from config.settings import (
    SYMBOL,
    TIMEFRAME,
    TRADING_MODE,
    print_config
)


class ScalperBot:
    """Bot de trading automatizado."""

    def __init__(self):
        self.symbol = SYMBOL.replace('/', '')
        self.timeframe = TIMEFRAME
        self.running = False
        self.last_candle_time = None
        self.df_history = pd.DataFrame()
        
        # Inicializar WS Manager
        self.ws_manager = BinanceWebsocketManager(
            self.symbol, 
            self.timeframe, 
            self._on_candle_closed, 
            testnet=(TRADING_MODE == 'testnet')
        )

        # Configurar senales de sistema
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Maneja shutdown graceful."""
        print("\n[BOT] Deteniendo bot...")
        self.running = False
        self.ws_manager.stop()

    def fetch_candles(self, limit: int = 500) -> pd.DataFrame:
        """
        Obtiene velas historicas y las convierte a DataFrame.
        """
        try:
            klines = client.get_klines(self.symbol, self.timeframe, limit)

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            # Convertir a tipos correctos
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)

            # Filtrar solo velas cerradas (close_time ya paso)
            now = pd.Timestamp.now(tz='UTC').tz_localize(None)
            df = df[df['close_time'] < now]

            df.set_index('timestamp', inplace=True)
            return df

        except Exception as e:
            print(f"[ERROR] Error obteniendo velas: {e}")
            return pd.DataFrame()

    def fetch_initial_history(self, limit: int = 500):
        """Carga historial inicial para indicadores."""
        print("[BOT] Cargando historial inicial...")
        df = self.fetch_candles(limit)
        if not df.empty:
            self.df_history = df
            strategy.calculate_indicators(self.df_history)
            self.last_candle_time = self.df_history.index[-1]
            print(f"[BOT] Historial cargado. {len(df)} velas.")

    def _on_candle_closed(self, candle):
        """Callback cuando cierra una vela via Websocket."""
        row = pd.DataFrame([{
            'timestamp': candle['timestamp'],
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close'],
            'volume': candle['volume']
        }])
        row.set_index('timestamp', inplace=True)
        
        self.df_history = pd.concat([self.df_history, row])
        if len(self.df_history) > 1000:
            self.df_history = self.df_history.iloc[-1000:]
            
        self.df_history = strategy.calculate_indicators(self.df_history)
        self.last_candle_time = self.df_history.index[-1]
        
        print(f"\n[WS] Nueva vela cerrada: {self.last_candle_time}")
        self.process_candle(self.df_history)

    def process_candle(self, df: pd.DataFrame):
        """Procesa una nueva vela cerrada."""
        if df.empty: return

        last_candle = df.iloc[-1]
        current_price = last_candle['close']
        current_high = last_candle['high']
        current_low = last_candle['low']

        if trader.has_open_position():
            trader.update_position(current_high, current_low, current_price)
            pos_info = trader.get_position_info()
            if pos_info:
                self._print_position_status(pos_info, current_price)
            return

        entry_signal = strategy.check_entry_signal(df)
        if entry_signal != Signal.NONE:
            print(f"\n[SIGNAL] {entry_signal.value.upper()} detectada!")
            print(f"         Precio: ${current_price:,.2f} | StochK: {last_candle['stoch_k']:.1f}")
            success = trader.open_position(entry_signal, current_price, 0)
            if success:
                print("[BOT] Posicion abierta exitosamente")

    def _print_position_status(self, pos_info: dict, current_price: float):
        """Imprime estado de la posicion actual."""
        avg_entry = pos_info['avg_price']
        pnl_pct = ((current_price - avg_entry) / avg_entry * 100) if pos_info['side'] == 'long' else ((avg_entry - current_price) / avg_entry * 100)
        print(f"[POS] {pos_info['side'].upper()} | Avg: ${avg_entry:,.2f} | PnL: {pnl_pct:+.2f}% | DCA: {pos_info['so_count']}")

    def run(self):
        """Loop principal del bot."""
        print("\n" + "="*60)
        print("BINANCE SCALPER BOT v6.7 (WEBSOCKET)")
        print("="*60)
        print_config()

        trader.sync_with_exchange()
        self.fetch_initial_history()
        
        if self.df_history.empty: return

        print(f"[BOT] Precio: ${self.df_history['close'].iloc[-1]:,.2f}")
        self.ws_manager.start()
        print("\n[BOT] Escuchando mercado...")
        
        self.running = True
        while self.running:
            try:
                live_price = self.ws_manager.get_latest_price()
                if live_price > 0 and trader.has_open_position():
                    trader.update_position(live_price, live_price, live_price)
                time.sleep(1)
            except Exception as e:
                print(f"[ERROR] Loop: {e}")
                time.sleep(1)

        self.ws_manager.stop()

def main():
    """Punto de entrada principal."""
    try:
        ticker = client.get_ticker()
        print(f"[OK] Conectado a Binance ({TRADING_MODE})")
        bot = ScalperBot()
        bot.run()
    except Exception as e:
        print(f"[ERROR] Main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()