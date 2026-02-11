"""
Binance Scalper Bot - Loop Principal
=====================================
Bot de trading automatizado v6.7 SMART METRALLADORA (Websocket).
Ejecutar: poetry run python src/bot.py
"""

import sys
import time
import signal
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

# Agregar raiz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exchange import client
from src.strategy import strategy, Signal
from src.trader import trader
from src.database import db
from src.websocket_manager import BinanceWebsocketManager
from src.telegram_alerts import TelegramPoller, alert_bot_started, alert_status, alert_daily_summary, alert_ws_disconnected
from config.settings import (
    SYMBOL,
    TIMEFRAME,
    TRADING_MODE,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_FILE,
    print_config
)

logger = logging.getLogger(__name__)


def setup_logging():
    """Configura logging a archivo + consola."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(LOG_FORMAT)

    # Handler archivo
    file_handler = logging.FileHandler(str(LOG_FILE), encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Handler consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


class ScalperBot:
    """Bot de trading automatizado."""

    def __init__(self):
        self.symbol = SYMBOL.replace('/', '')
        self.timeframe = TIMEFRAME
        self.running = False
        self.last_candle_time = None
        self.df_history = pd.DataFrame()
        self._status_counter = 0
        self._status_interval = 12  # cada 12 ciclos de 5s = ~60 segundos
        self._pos_counter = 0
        self._pos_interval = 6     # cada 6 ciclos de 5s = ~30 segundos
        self._daily_summary_sent = ""  # fecha del ultimo resumen enviado

        # Telegram poller para /status
        self.tg_poller = TelegramPoller(status_callback=self._handle_tg_status)

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
        logger.info("[BOT] Deteniendo bot...")
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
            logger.error(f"[ERROR] Error obteniendo velas: {e}")
            return pd.DataFrame()

    def fetch_initial_history(self, limit: int = 500):
        """Carga historial inicial para indicadores."""
        logger.info("[BOT] Cargando historial inicial...")
        df = self.fetch_candles(limit)
        if not df.empty:
            self.df_history = df
            strategy.calculate_indicators(self.df_history)
            self.last_candle_time = self.df_history.index[-1]
            logger.info(f"[BOT] Historial cargado. {len(df)} velas.")

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

        # Guardar candle en SQLite
        try:
            db.save_candle({
                'timestamp': str(candle['timestamp']),
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle['volume']
            })
        except Exception:
            pass

        logger.info(f"[WS] Nueva vela cerrada: {self.last_candle_time}")
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
            logger.info(f"[SIGNAL] {entry_signal.value.upper()} detectada!")
            logger.info(f"         Precio: ${current_price:,.2f} | StochK: {last_candle['stoch_k']:.1f}")
            success = trader.open_position(entry_signal, current_price, 0)
            if success:
                logger.info("[BOT] Posicion abierta exitosamente")
                self._capture_features(df, current_price)

    def _capture_features(self, df: pd.DataFrame, price: float):
        """Captura snapshot de features al momento de abrir posicion."""
        try:
            last = df.iloc[-1]
            now = datetime.utcnow()

            # Calcular features derivados
            ema_200 = last.get('ema_trend', None)
            ema_dist_pct = ((price - ema_200) / ema_200 * 100) if ema_200 and ema_200 > 0 else None

            bb_lower = last.get('bb_lower', None)
            bb_upper = last.get('bb_upper', None)
            bb_range = (bb_upper - bb_lower) if bb_upper and bb_lower and bb_upper > bb_lower else None
            bb_position = ((price - bb_lower) / bb_range) if bb_range and bb_range > 0 else None

            atr = last.get('atr', None)
            atr_sma = last.get('atr_sma', None)
            atr_ratio = (atr / atr_sma) if atr and atr_sma and atr_sma > 0 else None

            # Volume relative
            vol = last.get('volume', None)
            vol_sma = df['volume'].rolling(20).mean().iloc[-1] if 'volume' in df.columns and len(df) >= 20 else None
            vol_relative = (vol / vol_sma) if vol and vol_sma and vol_sma > 0 else None

            # Spread
            spread = last.get('high', 0) - last.get('low', 0)
            spread_sma = (df['high'] - df['low']).rolling(20).mean().iloc[-1] if len(df) >= 20 else None
            spread_relative = (spread / spread_sma) if spread_sma and spread_sma > 0 else None

            # Returns
            closes = df['close']
            return_5 = ((closes.iloc[-1] / closes.iloc[-6]) - 1) * 100 if len(closes) > 5 else None
            return_15 = ((closes.iloc[-1] / closes.iloc[-16]) - 1) * 100 if len(closes) > 15 else None
            return_60 = ((closes.iloc[-1] / closes.iloc[-61]) - 1) * 100 if len(closes) > 60 else None

            # Racha actual
            c_wins, c_losses = 0, 0
            for t in reversed(trader.trade_history):
                if t.pnl > 0:
                    if c_losses > 0:
                        break
                    c_wins += 1
                else:
                    if c_wins > 0:
                        break
                    c_losses += 1

            features = {
                'timestamp': now.isoformat(),
                'price': price,
                'ema_200': ema_200,
                'ema_dist_pct': ema_dist_pct,
                'bb_lower': bb_lower,
                'bb_upper': bb_upper,
                'bb_position': bb_position,
                'stoch_k': last.get('stoch_k', None),
                'atr': atr,
                'atr_sma': atr_sma,
                'atr_ratio': atr_ratio,
                'volume': vol,
                'volume_sma_20': vol_sma,
                'volume_relative': vol_relative,
                'spread': spread,
                'spread_relative': spread_relative,
                'hour_utc': now.hour,
                'day_of_week': now.weekday(),
                'return_5': return_5,
                'return_15': return_15,
                'return_60': return_60,
                'consecutive_wins': c_wins,
                'consecutive_losses': c_losses,
                'outcome': ''  # Se llena cuando se cierre el trade
            }

            # El trade_id sera el proximo ID (aun no existe, se linkea al cerrar)
            # Usamos el trade count + 1 como estimacion
            trade_count = db.get_trade_count()
            db.save_features(features, trade_id=trade_count + 1)
            logger.info(f"[DB] Features capturados: EMA dist={ema_dist_pct:.2f}% | BB pos={bb_position:.2f} | ATR ratio={atr_ratio:.2f}" if ema_dist_pct and bb_position and atr_ratio else "[DB] Features capturados")

        except Exception as e:
            logger.warning(f"[WARN] Error capturando features: {e}")

    def _log_periodic_status(self, live_price: float):
        """Reporte periodico: balance, precio, stats y estado WS."""
        try:
            balance = client.get_usdt_balance()
            ws_status = "OK" if self.ws_manager.is_connected() else "DESCONECTADO"
            price_str = f"${live_price:,.2f}" if live_price > 0 else "N/A"
            stats = trader.get_stats_summary()
            pos = "SI" if trader.has_open_position() else "NO"
            logger.info(f"[STATUS] Balance: ${balance:,.2f} | Precio: {price_str} | Posicion: {pos} | WS: {ws_status} | {stats}")

            # Resumen diario a Telegram (una vez al dia, despues de las 23:55 UTC)
            now = datetime.utcnow()
            today = now.strftime('%Y-%m-%d')
            if now.hour == 23 and now.minute >= 55 and self._daily_summary_sent != today:
                self._daily_summary_sent = today
                s = trader.daily_stats
                total = s.wins + s.losses
                wr = (s.wins / total * 100) if total > 0 else 0
                alert_daily_summary(total, s.wins, s.losses, s.total_pnl, balance, wr)
        except Exception as e:
            logger.warning(f"[STATUS] No se pudo obtener balance: {e}")

    def _handle_tg_status(self):
        """Callback para comando /status de Telegram."""
        try:
            balance = client.get_usdt_balance()
            in_pos = trader.has_open_position()
            side = ""
            pnl_unreal = 0.0
            if in_pos:
                pos_info = trader.get_position_info()
                if pos_info:
                    side = pos_info['side']
                    price = self.ws_manager.get_latest_price()
                    if pos_info['avg_price'] > 0 and price > 0:
                        pnl_pct = ((price - pos_info['avg_price']) / pos_info['avg_price']) if side == 'long' else ((pos_info['avg_price'] - price) / pos_info['avg_price'])
                        pnl_unreal = pos_info['avg_price'] * pos_info.get('total_quantity', 0) * pnl_pct * trader.leverage
            s = trader.daily_stats
            alert_status(balance, in_pos, side, pnl_unreal, s.wins + s.losses)
        except Exception as e:
            logger.warning(f"[TG] Error generando status: {e}")

    def _print_position_status(self, pos_info: dict, current_price: float):
        """Imprime estado de la posicion actual."""
        avg_entry = pos_info['avg_price']
        if avg_entry <= 0 or current_price <= 0:
            logger.info(f"[POS] {pos_info['side'].upper()} | Avg: ${avg_entry:,.2f} | PnL: N/A | DCA: {pos_info['so_count']}")
            return
        pnl_pct = ((current_price - avg_entry) / avg_entry * 100) if pos_info['side'] == 'long' else ((avg_entry - current_price) / avg_entry * 100)
        logger.info(f"[POS] {pos_info['side'].upper()} | Avg: ${avg_entry:,.2f} | PnL: {pnl_pct:+.2f}% | DCA: {pos_info['so_count']}")

    def run(self):
        """Loop principal del bot."""
        logger.info("=" * 60)
        logger.info("BINANCE SCALPER BOT v6.7 (WEBSOCKET)")
        logger.info("=" * 60)
        print_config()

        trader.sync_with_exchange()
        self.fetch_initial_history()
        
        if self.df_history.empty: return

        logger.info(f"[BOT] Precio: ${self.df_history['close'].iloc[-1]:,.2f}")
        self.ws_manager.start()
        self.tg_poller.start()
        logger.info("[BOT] Escuchando mercado...")

        # Alerta de inicio en Telegram
        try:
            balance = client.get_usdt_balance()
            alert_bot_started(balance, TRADING_MODE)
        except Exception:
            alert_bot_started(0, TRADING_MODE)
        
        self.running = True
        while self.running:
            try:
                live_price = self.ws_manager.get_latest_price()

                # Reporte de posicion (~cada 30s en vez de cada 5s)
                if live_price > 0 and trader.has_open_position():
                    self._pos_counter += 1
                    if self._pos_counter >= self._pos_interval:
                        self._pos_counter = 0
                        pos_info = trader.get_position_info()
                        if pos_info:
                            self._print_position_status(pos_info, live_price)
                else:
                    self._pos_counter = 0

                # Reporte periodico de balance (~cada 60s)
                self._status_counter += 1
                if self._status_counter >= self._status_interval:
                    self._status_counter = 0
                    self._log_periodic_status(live_price)

                time.sleep(5)
            except Exception as e:
                logger.error(f"[ERROR] Loop: {e}")
                time.sleep(5)

        self.ws_manager.stop()

def main():
    """Punto de entrada principal."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        ticker = client.get_ticker()
        logger.info(f"[OK] Conectado a Binance ({TRADING_MODE})")
        logger.info(f"[DB] Trades en DB: {db.get_trade_count()} | Candles: {db.get_candle_count()}")
        bot = ScalperBot()
        bot.run()
    except Exception as e:
        logger.error(f"[ERROR] Main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()