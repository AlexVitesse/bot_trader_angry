"""
ML Bot - Trader Profesional Autonomo
=====================================
Bot de trading ML que analiza el mercado, detecta regime,
genera senales con LightGBM, y ejecuta con risk management profesional.

Ejecutar: poetry run python -m src.ml_bot
"""

import sys
import time
import logging
import ccxt
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    TRADING_MODE, BINANCE_API_KEY, BINANCE_API_SECRET,
    ML_PAIRS, ML_CHECK_INTERVAL, ML_CANDLE_HOURS, ML_LEVERAGE,
    ML_DB_FILE, MODELS_DIR, TELEGRAM_ENABLED, LOG_LEVEL,
    LOGS_DIR, INITIAL_CAPITAL,
)
from src.ml_strategy import MLStrategy
from src.portfolio_manager import PortfolioManager
from src.telegram_alerts import send_alert

logger = logging.getLogger('ml_bot')


class MLBot:
    """Bot de trading ML profesional."""

    def __init__(self):
        self.exchange_public = self._init_exchange_public()
        self.exchange = self._init_exchange()
        self.strategy = MLStrategy()
        self.portfolio = PortfolioManager(self.exchange, ML_DB_FILE)
        self.running = False
        self.last_4h_candle = None     # Timestamp de ultima vela procesada
        self.last_regime_date = None   # Fecha de ultimo regime update
        self.last_status_log = 0       # Timestamp de ultimo status log
        self.last_daily_summary = None # Fecha de ultimo resumen diario

    def _init_exchange_public(self) -> ccxt.Exchange:
        """Cliente ccxt sin auth para datos de mercado."""
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
        })
        return exchange

    def _init_exchange(self) -> ccxt.Exchange:
        """Crea cliente ccxt autenticado."""
        config = {
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
        }
        exchange = ccxt.binance(config)
        if TRADING_MODE == 'testnet':
            # Binance deprecated futures testnet - use demo trading API
            # Redirect all fapi URLs to demo
            demo = 'https://demo-fapi.binance.com'
            for key in list(exchange.urls.get('api', {}).keys()):
                if key.startswith('fapi'):
                    orig = exchange.urls['api'][key]
                    exchange.urls['api'][key] = orig.replace(
                        'https://fapi.binance.com', demo
                    )
            # Pre-load markets from public exchange to avoid
            # spot API calls (sapi) that fail with demo keys
            self.exchange_public.load_markets()
            exchange.markets = self.exchange_public.markets
            exchange.markets_by_id = self.exchange_public.markets_by_id
            exchange.currencies = self.exchange_public.currencies
            exchange.currencies_by_id = self.exchange_public.currencies_by_id
        return exchange

    def _setup_leverage(self):
        """Configura leverage para todos los pares."""
        for pair in self.strategy.pairs:
            for regime, lev in ML_LEVERAGE.items():
                try:
                    self.exchange.set_leverage(lev, pair)
                except Exception:
                    pass  # Algunos pares pueden no soportar cierto leverage
            logger.info(f"[BOT] Leverage configurado: {pair}")

    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    def run(self):
        """Loop principal del bot."""
        self._startup()

        logger.info("[BOT] Entrando en loop principal...")
        self.running = True

        while self.running:
            try:
                # 1. Verificar nueva vela 4h
                if self._is_new_4h_candle():
                    self._on_new_candle()

                # 2. Monitorear posiciones abiertas
                self._monitor_positions()

                # 3. Risk checks
                if not self.portfolio.check_risk():
                    if self.portfolio.killed:
                        self._on_kill_switch()
                        break
                    elif self.portfolio.paused:
                        self._on_pause()

                # 4. Tareas periodicas
                self._periodic_tasks()

                time.sleep(ML_CHECK_INTERVAL)

            except KeyboardInterrupt:
                logger.info("[BOT] Detenido por usuario")
                self.running = False
            except Exception as e:
                logger.error(f"[BOT] Error en loop: {e}", exc_info=True)
                time.sleep(60)

        self._shutdown()

    def _startup(self):
        """Inicializacion del bot."""
        logger.info("=" * 60)
        logger.info("ML TRADER PROFESIONAL - INICIO")
        logger.info("=" * 60)
        logger.info(f"Modo: {TRADING_MODE.upper()}")
        logger.info(f"Capital: ${INITIAL_CAPITAL}")

        # 1. Cargar modelos
        count = self.strategy.load_models()
        if count == 0:
            logger.critical("[BOT] No se encontraron modelos. "
                            "Ejecutar primero: poetry run python ml_export_models.py")
            send_alert("ERROR: No hay modelos ML. Bot no puede iniciar.")
            sys.exit(1)
        logger.info(f"[BOT] {count} modelos cargados para {len(self.strategy.pairs)} pares")

        # 2. Detectar regime
        logger.info("[BOT] Detectando regime de mercado...")
        self.strategy.update_regime(self.exchange_public)
        self.last_regime_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        logger.info(f"[BOT] Regime: {self.strategy.regime}")

        # 3. Recuperar posiciones
        self.portfolio.sync_positions()

        # 4. Actualizar balance
        self.portfolio.refresh_balance()
        logger.info(f"[BOT] Balance: ${self.portfolio.balance:.2f}")

        # 5. Setup leverage
        self._setup_leverage()

        # 6. Telegram
        regime = self.strategy.regime
        n_pos = len(self.portfolio.positions)
        send_alert(
            f"<b>ML BOT INICIADO</b>\n"
            f"Modo: {TRADING_MODE}\n"
            f"Regime: {regime}\n"
            f"Modelos: {count}\n"
            f"Posiciones: {n_pos}\n"
            f"Balance: ${self.portfolio.balance:,.2f}"
        )

        logger.info("=" * 60)
        logger.info("[BOT] Listo. Esperando senales...")

    def _shutdown(self):
        """Limpieza al cerrar."""
        logger.info("[BOT] Cerrando bot...")
        status = self.portfolio.get_status()
        send_alert(
            f"<b>ML BOT DETENIDO</b>\n"
            f"Balance: ${status['balance']:,.2f}\n"
            f"Posiciones abiertas: {status['positions']}\n"
            f"Trades hoy: {status['total_trades']}"
        )

    # =========================================================================
    # 4H CANDLE CHECK
    # =========================================================================
    def _is_new_4h_candle(self) -> bool:
        """Verifica si se cerro una nueva vela de 4h."""
        now = datetime.now(timezone.utc)
        current_hour = now.hour

        # Las velas de 4h se cierran a las 0, 4, 8, 12, 16, 20 UTC
        if current_hour not in ML_CANDLE_HOURS:
            return False

        # Esperar 2 minutos despues de la hora para asegurar cierre
        if now.minute < 2:
            return False

        # Verificar que no hayamos procesado esta vela ya
        candle_key = now.strftime('%Y-%m-%d-%H')
        if self.last_4h_candle == candle_key:
            return False

        self.last_4h_candle = candle_key
        return True

    def _on_new_candle(self):
        """Procesa nueva vela de 4h: genera senales y abre posiciones."""
        logger.info("[BOT] Nueva vela 4h - generando senales...")

        # Actualizar regime diariamente
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if today != self.last_regime_date:
            self.strategy.update_regime(self.exchange_public)
            self.last_regime_date = today

        # Actualizar balance
        self.portfolio.refresh_balance()

        # Generar senales
        open_pairs = set(self.portfolio.positions.keys())
        signals = self.strategy.generate_signals(self.exchange_public, open_pairs)

        if signals:
            logger.info(f"[BOT] {len(signals)} senales generadas:")
            for s in signals:
                side = 'LONG' if s['direction'] == 1 else 'SHORT'
                logger.info(f"  {s['pair']} {side} | conf={s['confidence']:.2f} | "
                            f"pred={s['prediction']:+.4f} | ${s['price']:,.2f}")
        else:
            logger.info("[BOT] Sin senales en este ciclo")

        # Intentar abrir posiciones
        for signal in signals:
            if not self.portfolio.can_open(signal['pair'], signal['direction']):
                continue

            success = self.portfolio.open_position(
                pair=signal['pair'],
                direction=signal['direction'],
                confidence=signal['confidence'],
                regime=self.strategy.regime,
                price=signal['price'],
                atr_pct=signal['atr_pct'],
            )

            if success:
                side = 'LONG' if signal['direction'] == 1 else 'SHORT'
                send_alert(
                    f"<b>TRADE ABIERTO</b>\n"
                    f"Par: {signal['pair']}\n"
                    f"Side: {side}\n"
                    f"Precio: ${signal['price']:,.2f}\n"
                    f"Confianza: {signal['confidence']:.2f}\n"
                    f"Regime: {self.strategy.regime}"
                )

    # =========================================================================
    # POSITION MONITORING
    # =========================================================================
    def _monitor_positions(self):
        """Monitorea posiciones abiertas cada 30s."""
        closed_trades = self.portfolio.update_positions()

        for trade in closed_trades:
            emoji = "+" if trade['pnl'] > 0 else ""
            send_alert(
                f"<b>TRADE CERRADO</b>\n"
                f"Par: {trade['symbol']}\n"
                f"Side: {trade['side'].upper()}\n"
                f"Entry: ${trade['entry_price']:,.2f}\n"
                f"Exit: ${trade['exit_price']:,.2f}\n"
                f"PnL: <b>${trade['pnl']:{emoji}.2f}</b>\n"
                f"Razon: {trade['exit_reason']}\n"
                f"Balance: ${self.portfolio.balance:,.2f}"
            )

    # =========================================================================
    # PERIODIC TASKS
    # =========================================================================
    def _periodic_tasks(self):
        """Logging periodico y resumen diario."""
        now = time.time()

        # Status log cada 10 minutos
        if now - self.last_status_log >= 600:
            self.last_status_log = now
            status = self.portfolio.get_status()
            logger.info(
                f"[STATUS] Balance=${status['balance']:.2f} | "
                f"DD={status['dd']:.1%} | "
                f"Pos={status['positions']}/{ML_LEVERAGE} | "
                f"DailyPnL=${status['daily_pnl']:+.2f} | "
                f"Regime={self.strategy.regime}"
            )
            for p in status['position_details']:
                logger.info(f"  {p['pair']} {p['side'].upper()} "
                            f"@ ${p['entry']:.2f} | trail={p['trail']}")

        # Resumen diario a las 23:55 UTC
        utcnow = datetime.now(timezone.utc)
        if utcnow.hour == 23 and utcnow.minute >= 55:
            today = utcnow.strftime('%Y-%m-%d')
            if self.last_daily_summary != today:
                self.last_daily_summary = today
                self._send_daily_summary()

    def _send_daily_summary(self):
        """Envia resumen diario por Telegram."""
        status = self.portfolio.get_status()
        trades_today = [t for t in self.portfolio.trade_log
                        if t['exit_time'].startswith(datetime.now(timezone.utc).strftime('%Y-%m-%d'))]
        wins = sum(1 for t in trades_today if t['pnl'] > 0)
        losses = len(trades_today) - wins
        wr = (wins / len(trades_today) * 100) if trades_today else 0

        send_alert(
            f"<b>RESUMEN DIARIO</b>\n"
            f"Trades: {len(trades_today)} | W:{wins} L:{losses}\n"
            f"Win Rate: {wr:.0f}%\n"
            f"PnL hoy: ${status['daily_pnl']:+.2f}\n"
            f"Balance: ${status['balance']:,.2f}\n"
            f"DD: {status['dd']:.1%}\n"
            f"Regime: {self.strategy.regime}\n"
            f"Posiciones: {status['positions']}"
        )

    def _on_kill_switch(self):
        """Maneja kill switch."""
        status = self.portfolio.get_status()
        send_alert(
            f"<b>KILL SWITCH ACTIVADO</b>\n"
            f"DD: {status['dd']:.1%} >= 20%\n"
            f"Balance: ${status['balance']:,.2f}\n"
            f"Peak: ${status['peak']:,.2f}\n"
            f"Bot DETENIDO"
        )
        logger.critical("[BOT] KILL SWITCH - Bot detenido")

    def _on_pause(self):
        """Maneja pausa por daily loss."""
        send_alert(
            f"<b>BOT PAUSADO</b>\n"
            f"Daily loss: ${self.portfolio.daily_pnl:.2f}\n"
            f"Limite: {ML_LEVERAGE}% de capital\n"
            f"Reanuda manana automaticamente"
        )
        logger.warning("[BOT] Pausado por daily loss limit")


def setup_logging():
    """Configura logging para el ML bot."""
    LOGS_DIR.mkdir(exist_ok=True)
    log_file = LOGS_DIR / "ml_bot.log"

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding='utf-8'),
    ]

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
    )

    # Silenciar logs de ccxt y urllib3
    logging.getLogger('ccxt').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def main():
    setup_logging()
    bot = MLBot()
    bot.run()


if __name__ == '__main__':
    main()
