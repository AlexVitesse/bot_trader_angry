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
    LOGS_DIR, INITIAL_CAPITAL, ML_MAX_DAILY_LOSS_PCT,
)
from src.ml_strategy import MLStrategy
from src.portfolio_manager import PortfolioManager
from src.telegram_alerts import send_alert, TelegramPoller

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
        self.last_heartbeat = 0        # Timestamp de ultimo heartbeat
        self.recent_errors = []        # Errores desde ultimo heartbeat
        self._pause_notified = False   # Evitar spam de notificacion de pausa

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
                    elif self.portfolio.paused and not self._pause_notified:
                        self._on_pause()
                        self._pause_notified = True
                elif self._pause_notified and not self.portfolio.paused:
                    # Bot reanudado (nuevo dia o comando /resume)
                    self._pause_notified = False
                    send_alert(
                        f"▶️ <b>BOT REANUDADO</b>\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"💰 Balance: ${self.portfolio.balance:,.2f}\n"
                        f"📊 Regime: {self.strategy.regime}\n"
                        f"🔄 Operando normalmente"
                    )

                # 4. Tareas periodicas
                self._periodic_tasks()

                time.sleep(ML_CHECK_INTERVAL)

            except KeyboardInterrupt:
                logger.info("[BOT] Detenido por usuario")
                self.running = False
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                logger.error(f"[BOT] Error en loop: {error_msg}", exc_info=True)
                self.recent_errors.append(error_msg)
                time.sleep(60)

        self._shutdown()

    def _startup(self):
        """Inicializacion del bot."""
        self._start_time = time.time()
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
        regime_emoji = {'BULL': '🟢🐂', 'BEAR': '🔴🐻', 'RANGE': '🟡↔️'}.get(regime, '⚪')
        n_pos = len(self.portfolio.positions)
        send_alert(
            f"🚀 <b>ML BOT INICIADO</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📊 Modo: {TRADING_MODE.upper()}\n"
            f"{regime_emoji} Regime: {regime}\n"
            f"🧠 Modelos: {count}\n"
            f"📈 Posiciones: {n_pos}\n"
            f"💰 Balance: <b>${self.portfolio.balance:,.2f}</b>"
        )

        # 7. Iniciar Telegram poller para comandos
        self.tg_poller = TelegramPoller(callbacks={
            '/status': self._cmd_status,
            '/resume': self._cmd_resume,
        })
        self.tg_poller.start()

        logger.info("=" * 60)
        logger.info("[BOT] Listo. Esperando senales...")

    def _cmd_status(self):
        """Responde al comando /status de Telegram."""
        status = self.portfolio.get_status()
        uptime_h = (time.time() - self._start_time) / 3600
        trades_today = self.portfolio.get_today_trades_from_db()
        total_pnl = sum(t['pnl'] for t in trades_today)
        pnl_emoji = '📈' if total_pnl >= 0 else '📉'
        paused_str = "\n⏸️ <b>PAUSADO</b> - usa /resume" if self.portfolio.paused else ""
        send_alert(
            f"📊 <b>STATUS</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 Balance: ${status['balance']:,.2f}\n"
            f"📈 Pos: {status['positions']}/3\n"
            f"{pnl_emoji} PnL hoy: ${total_pnl:+,.2f} ({len(trades_today)} trades)\n"
            f"📊 Regime: {self.strategy.regime}\n"
            f"⚠️ DD: {status['dd']:.1%}\n"
            f"⏱️ Uptime: {uptime_h:.1f}h"
            f"{paused_str}"
        )

    def _cmd_resume(self):
        """Responde al comando /resume de Telegram - reanuda el bot pausado."""
        if self.portfolio.paused:
            self.portfolio.paused = False
            self._pause_notified = False
            logger.info("[BOT] Reanudado via comando /resume")
            send_alert(
                f"▶️ <b>BOT REANUDADO</b>\n"
                f"━━━━━━━━━━━━━━━\n"
                f"💰 Balance: ${self.portfolio.balance:,.2f}\n"
                f"📊 Regime: {self.strategy.regime}\n"
                f"🔄 Operando normalmente"
            )
        elif self.portfolio.killed:
            send_alert("🚫 Bot en KILL SWITCH - no se puede reanudar por Telegram")
        else:
            send_alert("✅ Bot ya esta operando normalmente")

    def _shutdown(self):
        """Limpieza al cerrar."""
        logger.info("[BOT] Cerrando bot...")
        status = self.portfolio.get_status()
        send_alert(
            f"🛑 <b>ML BOT DETENIDO</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 Balance: <b>${status['balance']:,.2f}</b>\n"
            f"📈 Posiciones: {status['positions']}\n"
            f"📋 Trades hoy: {status['total_trades']}"
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
                pos = self.portfolio.positions.get(signal['pair'])
                if pos:
                    side = 'LONG' if signal['direction'] == 1 else 'SHORT'
                    side_emoji = '🟢' if signal['direction'] == 1 else '🔴'
                    conf_bar = '🔥' if signal['confidence'] > 2.0 else '⚡' if signal['confidence'] > 1.5 else '📊'
                    margin = pos.notional / pos.leverage
                    action = 'COMPRANDO' if signal['direction'] == 1 else 'VENDIENDO'
                    coin = signal['pair'].split('/')[0]
                    # Explicacion educativa
                    if signal['direction'] == 1:
                        explain = f"📖 Compra {pos.quantity} {coin} esperando que SUBA"
                        tp_dir = '↗️ sube'
                        sl_dir = '↘️ baja'
                    else:
                        explain = f"📖 Vende {pos.quantity} {coin} esperando que BAJE"
                        tp_dir = '↘️ baja'
                        sl_dir = '↗️ sube'
                    send_alert(
                        f"{side_emoji} <b>TRADE ABIERTO - {action}</b>\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"💎 {signal['pair']} → <b>{side}</b>\n"
                        f"\n"
                        f"📥 <b>Entrada</b>\n"
                        f"   Precio: ${pos.entry_price:,.2f}\n"
                        f"   Cantidad: {pos.quantity} {coin}\n"
                        f"   Notional: ${pos.notional:,.2f}\n"
                        f"   Margen: ${margin:,.2f} ({pos.leverage}x leverage)\n"
                        f"\n"
                        f"🎯 <b>Objetivos</b>\n"
                        f"   TP: ${pos.tp_price:,.2f} (si {tp_dir} {pos.tp_pct:.1%})\n"
                        f"   SL: ${pos.sl_price:,.2f} (si {sl_dir} {pos.sl_pct:.1%})\n"
                        f"   Max hold: {pos.max_hold} velas ({pos.max_hold * 4}h)\n"
                        f"\n"
                        f"{conf_bar} Confianza: {signal['confidence']:.2f}\n"
                        f"📊 Regime: {self.strategy.regime}\n"
                        f"{explain}"
                    )

    # =========================================================================
    # POSITION MONITORING
    # =========================================================================
    def _monitor_positions(self):
        """Monitorea posiciones abiertas cada 30s."""
        closed_trades = self.portfolio.update_positions()

        for trade in closed_trades:
            win = trade['pnl'] > 0
            result_emoji = '✅' if win else '❌'
            reason_emoji = {'TP': '🎯', 'SL': '🛡️', 'TRAIL': '📐', 'TIMEOUT': '⏰'}.get(trade['exit_reason'], '📌')
            reason_text = {
                'TP': 'Take Profit (objetivo alcanzado)',
                'SL': 'Stop Loss (limite de perdida)',
                'TRAIL': 'Trailing Stop (proteccion de ganancia)',
                'TIMEOUT': 'Timeout (tiempo maximo alcanzado)',
            }.get(trade['exit_reason'], trade['exit_reason'])

            # Calcular detalles
            coin = trade['symbol'].split('/')[0]
            price_change = trade['exit_price'] - trade['entry_price']
            price_change_pct = price_change / trade['entry_price']
            # Gross PnL: para long ganas si sube, para short ganas si baja
            if trade['side'] == 'long':
                gross_pnl = trade['notional'] * price_change_pct
            else:
                gross_pnl = trade['notional'] * (-price_change_pct)
            margin = trade['notional'] / trade['leverage']

            # Duracion
            try:
                entry_dt = datetime.fromisoformat(trade['entry_time'])
                exit_dt = datetime.fromisoformat(trade['exit_time'])
                duration = exit_dt - entry_dt
                hours = duration.total_seconds() / 3600
                if hours >= 24:
                    dur_str = f"{hours / 24:.1f} dias"
                else:
                    dur_str = f"{hours:.1f} horas"
            except Exception:
                dur_str = "N/A"

            # Explicacion educativa
            if trade['side'] == 'long':
                if win:
                    explain = f"📖 Compro a ${trade['entry_price']:,.2f} y vendio mas caro"
                else:
                    explain = f"📖 Compro a ${trade['entry_price']:,.2f} pero bajo"
            else:
                if win:
                    explain = f"📖 Vendio a ${trade['entry_price']:,.2f} y recompro mas barato"
                else:
                    explain = f"📖 Vendio a ${trade['entry_price']:,.2f} pero subio"

            send_alert(
                f"{result_emoji} <b>TRADE CERRADO</b>\n"
                f"━━━━━━━━━━━━━━━\n"
                f"💎 {trade['symbol']} <b>{trade['side'].upper()}</b>\n"
                f"\n"
                f"📥 <b>Entrada → Salida</b>\n"
                f"   Entry: ${trade['entry_price']:,.2f}\n"
                f"   Exit:  ${trade['exit_price']:,.2f}\n"
                f"   Cambio: {price_change_pct:+.2%} (${price_change:+,.2f})\n"
                f"\n"
                f"💰 <b>Resultado</b>\n"
                f"   Notional: ${trade['notional']:,.2f} ({trade['leverage']}x)\n"
                f"   PnL bruto: ${gross_pnl:+,.2f}\n"
                f"   Comisiones: -${trade['commission']:.2f}\n"
                f"   <b>PnL neto: ${trade['pnl']:+,.2f}</b>\n"
                f"\n"
                f"⏱️ Duracion: {dur_str}\n"
                f"{reason_emoji} {reason_text}\n"
                f"\n"
                f"💰 Balance: <b>${self.portfolio.balance:,.2f}</b>\n"
                f"{explain}"
            )

    # =========================================================================
    # PERIODIC TASKS
    # =========================================================================
    def _periodic_tasks(self):
        """Logging periodico, heartbeat y resumen diario."""
        now = time.time()

        # Heartbeat cada 2 horas por Telegram
        if now - self.last_heartbeat >= 7200:
            self.last_heartbeat = now
            self._send_heartbeat()

        # Status log cada 10 minutos
        if now - self.last_status_log >= 600:
            self.last_status_log = now
            status = self.portfolio.get_status()
            logger.info(
                f"[STATUS] Balance=${status['balance']:.2f} | "
                f"DD={status['dd']:.1%} | "
                f"Pos={status['positions']}/3 | "
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

    def _send_heartbeat(self):
        """Envia heartbeat cada 2h por Telegram para monitoreo remoto."""
        status = self.portfolio.get_status()
        uptime_h = (time.time() - self._start_time) / 3600 if hasattr(self, '_start_time') else 0
        trades_today = self.portfolio.get_today_trades_from_db()
        total_pnl = sum(t['pnl'] for t in trades_today)

        # Contador de demo: 2 semanas desde 2026-02-12 -> fin 2026-02-26
        # TODO: Borrar este contador despues del 2026-02-26 (ver plan.txt)
        demo_end = datetime(2026, 2, 26, tzinfo=timezone.utc)
        days_left = (demo_end - datetime.now(timezone.utc)).days
        if days_left > 0:
            demo_str = f"📅 Demo: {days_left} dias restantes"
        elif days_left == 0:
            demo_str = "📅 Demo: ULTIMO DIA - revisar resultados!"
        else:
            demo_str = "📅 Demo: FINALIZADA - revisar resultados!"

        if self.recent_errors:
            errors_str = "\n".join(f"  ⚠️ {e[:80]}" for e in self.recent_errors[-5:])
            n_errors = len(self.recent_errors)
            send_alert(
                f"🔴 <b>BOT ERRORES ({n_errors})</b>\n"
                f"━━━━━━━━━━━━━━━\n"
                f"{errors_str}\n"
                f"\n"
                f"💰 Balance: ${status['balance']:,.2f}\n"
                f"📈 Pos: {status['positions']}/3\n"
                f"📊 Regime: {self.strategy.regime}\n"
                f"⏱️ Uptime: {uptime_h:.1f}h\n"
                f"{demo_str}"
            )
            self.recent_errors.clear()
        else:
            pnl_emoji = '📈' if total_pnl >= 0 else '📉'
            send_alert(
                f"🟢 <b>BOT TODO OK</b>\n"
                f"━━━━━━━━━━━━━━━\n"
                f"💰 Balance: ${status['balance']:,.2f}\n"
                f"📈 Pos: {status['positions']}/3\n"
                f"{pnl_emoji} PnL hoy: ${total_pnl:+,.2f} ({len(trades_today)} trades)\n"
                f"📊 Regime: {self.strategy.regime}\n"
                f"⚠️ DD: {status['dd']:.1%}\n"
                f"⏱️ Uptime: {uptime_h:.1f}h\n"
                f"{demo_str}"
            )

    def _send_daily_summary(self):
        """Envia resumen diario por Telegram (consulta DB, sobrevive reinicios)."""
        status = self.portfolio.get_status()
        trades_today = self.portfolio.get_today_trades_from_db()
        wins = sum(1 for t in trades_today if t['pnl'] > 0)
        losses = len(trades_today) - wins
        wr = (wins / len(trades_today) * 100) if trades_today else 0
        total_pnl = sum(t['pnl'] for t in trades_today)

        pnl_emoji = '📈' if total_pnl >= 0 else '📉'
        send_alert(
            f"📊 <b>RESUMEN DIARIO</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📋 Trades: {len(trades_today)} | ✅ {wins} ❌ {losses}\n"
            f"🎯 Win Rate: {wr:.0f}%\n"
            f"{pnl_emoji} PnL hoy: <b>${total_pnl:+,.2f}</b>\n"
            f"💰 Balance: <b>${status['balance']:,.2f}</b>\n"
            f"⚠️ DD: {status['dd']:.1%}\n"
            f"📊 Regime: {self.strategy.regime}\n"
            f"📈 Posiciones: {status['positions']}"
        )

    def _on_kill_switch(self):
        """Maneja kill switch."""
        status = self.portfolio.get_status()
        send_alert(
            f"🚨🚨🚨 <b>KILL SWITCH ACTIVADO</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📉 DD: {status['dd']:.1%} >= 20%\n"
            f"💰 Balance: ${status['balance']:,.2f}\n"
            f"🏔️ Peak: ${status['peak']:,.2f}\n"
            f"🛑 Bot DETENIDO"
        )
        logger.critical("[BOT] KILL SWITCH - Bot detenido")

    def _on_pause(self):
        """Maneja pausa por daily loss."""
        send_alert(
            f"⏸️ <b>BOT PAUSADO</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📉 Daily loss: ${self.portfolio.daily_pnl:.2f}\n"
            f"🛡️ Limite: {ML_MAX_DAILY_LOSS_PCT:.0%} de capital\n"
            f"🔄 Reanuda manana automaticamente"
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
