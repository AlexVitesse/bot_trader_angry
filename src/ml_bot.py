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
import logging.handlers
import subprocess
import threading
import ccxt
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    TRADING_MODE, BINANCE_API_KEY, BINANCE_API_SECRET,
    ML_PAIRS, ML_CHECK_INTERVAL, ML_CANDLE_HOURS, ML_LEVERAGE,
    ML_DB_FILE, MODELS_DIR, TELEGRAM_ENABLED, LOG_LEVEL,
    LOGS_DIR, INITIAL_CAPITAL, ML_MAX_DAILY_LOSS_PCT,
    ML_SHADOW_ENABLED, ML_V9_ENABLED, ML_TIMEFRAME,
    ML_MAX_CONCURRENT, BOT_VERSION,
    ML_V1304_ENABLED, ML_V1304_PAIRS,
    ML_V14_ENABLED, ML_V14_EXPERTS,
)
from src.ml_strategy import MLStrategy
from src.ml_strategy_v14 import MLStrategyV14
from src.portfolio_manager import PortfolioManager
from src.shadow_portfolio_manager import ShadowPortfolioManager
from src.telegram_alerts import send_alert, send_document, TelegramPoller

logger = logging.getLogger('ml_bot')


class MLBot:
    """Bot de trading ML profesional."""

    def __init__(self):
        self.exchange_public = self._init_exchange_public()
        self.exchange = self._init_exchange()
        # Usar estrategia V14 si esta habilitada
        if ML_V14_ENABLED:
            self.strategy = MLStrategyV14()
            self.v14_mode = True
        else:
            self.strategy = MLStrategy()
            self.v14_mode = False
        self.portfolio = PortfolioManager(self.exchange, ML_DB_FILE)
        self.shadow_enabled = ML_SHADOW_ENABLED and ML_V9_ENABLED
        self.shadow_portfolio = ShadowPortfolioManager(strategy='v9_shadow') if self.shadow_enabled else None
        self.running = False
        self.last_4h_candle = None     # Timestamp de ultima vela procesada
        self.last_regime_date = None   # Fecha de ultimo regime update
        self.last_status_log = 0       # Timestamp de ultimo status log
        self.last_daily_summary = None # Fecha de ultimo resumen diario
        self.last_heartbeat = 0        # Timestamp de ultimo heartbeat
        self.recent_errors = []        # Errores desde ultimo heartbeat
        self._pause_notified = False   # Evitar spam de notificacion de pausa
        self._exit_code = 0            # Exit code for wrapper script

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
            'options': {
                'defaultType': 'future',
                'recvWindow': 10000,
                'adjustForTimeDifference': True,  # sync clock with Binance server
            },
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
            for attempt in range(5):
                try:
                    self.exchange_public.load_markets()
                    break
                except Exception as e:
                    if attempt < 4:
                        wait = 10 * (attempt + 1)
                        logger.warning(f"[BOT] load_markets intento {attempt+1}/5 fallo: {e} - reintentando en {wait}s")
                        time.sleep(wait)
                    else:
                        raise
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

        # 2b. Update macro intelligence (V8.4) - solo para estrategia legacy
        if not self.v14_mode and hasattr(self.strategy, 'v84_enabled') and self.strategy.v84_enabled:
            logger.info("[BOT] Actualizando datos macro (V8.4)...")
            self.strategy.update_macro()
            logger.info(f"[BOT] Macro: score={self.strategy.macro_score:.3f}, "
                        f"sizing={self.strategy.get_sizing_multiplier():.2f}x, "
                        f"thresh={self.strategy.get_adaptive_threshold():.2f}")

        # 3. Recuperar posiciones
        self.portfolio.sync_positions()

        # 3b. Recuperar posiciones shadow
        if self.shadow_enabled:
            self.shadow_portfolio.sync_positions()
            logger.info(f"[BOT] Shadow V9: {len(self.shadow_portfolio.positions)} posiciones")

        # 4. Actualizar balance
        self.portfolio.refresh_balance()
        logger.info(f"[BOT] Balance: ${self.portfolio.balance:.2f}")

        # 5. Setup leverage
        self._setup_leverage()

        # 6. Telegram
        regime = self.strategy.regime
        regime_emoji = {'BULL': '🟢🐂', 'BEAR': '🔴🐻', 'RANGE': '🟡↔️'}.get(regime, '⚪')
        n_pos = len(self.portfolio.positions)

        # V14 info
        if self.v14_mode:
            n_experts = len(ML_V14_EXPERTS)
            model_str = f"🤖 Ensemble Voting\n📊 {n_experts} expertos activos"
            count = n_experts
        # V13.04 info
        elif ML_V1304_ENABLED:
            v1304_pairs = [p.replace('/USDT', '') for p in ML_V1304_PAIRS]
            model_str = f"🔬 Ridge LONG_ONLY\n📊 Pares: {', '.join(v1304_pairs)}"
            count = len(ML_V1304_PAIRS)  # Override count for V13.04
        else:
            extras = []
            if hasattr(self.strategy, 'v84_enabled') and self.strategy.v84_enabled:
                extras.append(f"🌐 Macro: {self.strategy.macro_score:.2f}")
            if hasattr(self.strategy, 'v85_enabled') and self.strategy.v85_enabled:
                extras.append("🎯 Conv: ON")
            model_str = " | ".join(extras) if extras else ""

        send_alert(
            f"🚀 <b>{BOT_VERSION} INICIADO</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📊 Modo: {TRADING_MODE.upper()}\n"
            f"{regime_emoji} Regime: {regime}\n"
            f"🧠 Modelos: {count} pares\n"
            f"📈 Posiciones: {n_pos}/{ML_MAX_CONCURRENT}\n"
            f"💰 Balance: <b>${self.portfolio.balance:,.2f}</b>\n"
            f"{model_str}"
        )

        # 7. Iniciar Telegram poller para comandos
        self.tg_poller = TelegramPoller(callbacks={
            '/help': self._cmd_help,
            '/status': self._cmd_status,
            '/resume': self._cmd_resume,
            '/log': self._cmd_log,
            '/backup': self._cmd_backup,
            '/update': self._cmd_update,
            '/pull': self._cmd_pull,
            '/install': self._cmd_install,
            '/restart': self._cmd_restart,
            '/retrain': self._cmd_export_v14,  # Alias para V14
            '/export_v14': self._cmd_export_v14,
            '/clearlog': self._cmd_clearlog,
            '/resetdb': self._cmd_resetdb,
        })
        self.tg_poller.start()

        logger.info("=" * 60)
        logger.info("[BOT] Listo. Esperando senales...")

    def _cmd_help(self):
        """Responde al comando /help - lista de comandos disponibles."""
        send_alert(
            f"📋 <b>COMANDOS {BOT_VERSION}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"<b>Monitoreo:</b>\n"
            f"  /status - Estado del bot\n"
            f"  /log - Descargar archivo de log\n"
            f"  /backup - Descargar backup de BD\n"
            f"\n"
            f"<b>Control:</b>\n"
            f"  /resume - Reanudar si esta pausado\n"
            f"  /restart - Reiniciar el bot\n"
            f"\n"
            f"<b>Limpieza:</b>\n"
            f"  /clearlog - Borrar archivo de log\n"
            f"  /resetdb - Borrar trades antiguos\n"
            f"\n"
            f"<b>DevOps:</b>\n"
            f"  /update - Pull + Install + Restart\n"
            f"  /pull - git stash + pull\n"
            f"  /install - poetry install\n"
            f"  /retrain - Reentrenar modelos {BOT_VERSION}"
        )

    def _cmd_status(self):
        """Responde al comando /status de Telegram."""
        status = self.portfolio.get_status()
        uptime_h = (time.time() - self._start_time) / 3600
        trades_today = self.portfolio.get_today_trades_from_db()
        total_pnl = sum(t['pnl'] for t in trades_today)
        pnl_emoji = '📈' if total_pnl >= 0 else '📉'
        paused_str = "\n⏸️ <b>PAUSADO</b> - usa /resume" if self.portfolio.paused else ""

        # V14: Ensemble Voting
        if self.v14_mode:
            model_str = f"\n🤖 Ensemble Voting ({len(ML_V14_EXPERTS)} expertos)"
        # V13.04: Ridge LONG_ONLY
        elif ML_V1304_ENABLED:
            model_str = "\n🔬 Ridge LONG_ONLY"
        else:
            model_str = ""
            if hasattr(self.strategy, 'v84_enabled') and self.strategy.v84_enabled:
                model_str = (
                    f"\n🌐 Macro: {self.strategy.macro_score:.2f} | "
                    f"Sz: {self.strategy.get_sizing_multiplier():.2f}x"
                )
            if hasattr(self.strategy, 'v85_enabled') and self.strategy.v85_enabled:
                model_str += "\n🎯 Conv: ON"

        send_alert(
            f"📊 <b>STATUS {BOT_VERSION}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 Balance: ${status['balance']:,.2f}\n"
            f"📈 Pos: {status['positions']}/{ML_MAX_CONCURRENT}\n"
            f"{pnl_emoji} PnL hoy: ${total_pnl:+,.2f} ({len(trades_today)}t)\n"
            f"📊 Regime: {self.strategy.regime}"
            f"{model_str}\n"
            f"⚠️ DD: {status['dd']:.1%}\n"
            f"⏱️ Uptime: {uptime_h:.1f}h"
            f"{paused_str}"
        )

    def _cmd_resume(self):
        """Responde al comando /resume de Telegram - reanuda el bot pausado."""
        if self.portfolio.paused:
            self.portfolio.paused = False
            self.portfolio.daily_pnl = 0.0  # Reset para que check_risk() no re-pause
            self._pause_notified = False
            logger.info("[BOT] Reanudado via comando /resume (daily_pnl reset)")
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

    def _cmd_log(self):
        """Responde al comando /log - envia el archivo de log por Telegram."""
        log_file = LOGS_DIR / "ml_bot.log"
        if log_file.exists():
            size_kb = log_file.stat().st_size / 1024
            send_document(str(log_file), f"📄 Log ({size_kb:.0f} KB)")
        else:
            send_alert("⚠️ Archivo de log no encontrado")

    def _cmd_backup(self):
        """Responde al comando /backup - envia backup de la BD por Telegram."""
        import sqlite3 as _sqlite3
        backup_path = ML_DB_FILE.parent / 'ml_backup.db'
        try:
            src = _sqlite3.connect(str(ML_DB_FILE))
            dst = _sqlite3.connect(str(backup_path))
            src.backup(dst)
            src.close()
            dst.close()
            size_kb = backup_path.stat().st_size / 1024
            send_document(str(backup_path), f"💾 Backup DB ({size_kb:.0f} KB)")
            backup_path.unlink(missing_ok=True)
        except Exception as e:
            send_alert(f"⚠️ Error en backup: {e}")
            logger.warning(f"[BOT] Error en backup: {e}")

    def _cmd_clearlog(self):
        """Responde al comando /clearlog - borra el archivo de log."""
        log_file = LOGS_DIR / "ml_bot.log"
        try:
            if log_file.exists():
                size_kb = log_file.stat().st_size / 1024
                log_file.unlink()
                # Recrear archivo vacio
                log_file.touch()
                send_alert(f"🗑️ <b>LOG BORRADO</b>\n{size_kb:.0f} KB eliminados\nNuevo log iniciado")
                logger.info("[BOT] Log borrado via /clearlog")
            else:
                send_alert("ℹ️ Archivo de log no existe")
        except Exception as e:
            send_alert(f"⚠️ Error borrando log: {e}")
            logger.warning(f"[BOT] Error en clearlog: {e}")

    def _cmd_resetdb(self):
        """Responde al comando /resetdb - borra trades antiguos de la BD."""
        import sqlite3 as _sqlite3
        n_pos = len(self.portfolio.positions)
        if n_pos > 0:
            send_alert(
                f"⚠️ <b>NO SE PUEDE RESETEAR</b>\n"
                f"Hay {n_pos} posiciones abiertas.\n"
                f"Cierra posiciones primero."
            )
            return
        try:
            conn = _sqlite3.connect(str(ML_DB_FILE))
            cur = conn.cursor()
            # Contar trades antes
            cur.execute("SELECT COUNT(*) FROM ml_trades")
            count_before = cur.fetchone()[0]
            # Borrar trades
            cur.execute("DELETE FROM ml_trades")
            # Reset estado
            cur.execute("UPDATE ml_state SET peak = balance WHERE id = 1")
            conn.commit()
            conn.close()
            send_alert(
                f"🗑️ <b>BD RESETEADA</b>\n"
                f"━━━━━━━━━━━━━━━\n"
                f"🗂️ {count_before} trades eliminados\n"
                f"📊 Peak reseteado a balance actual\n"
                f"✅ Listo para {BOT_VERSION} limpio"
            )
            logger.info(f"[BOT] BD reseteada via /resetdb: {count_before} trades eliminados")
        except Exception as e:
            send_alert(f"⚠️ Error reseteando BD: {e}")
            logger.warning(f"[BOT] Error en resetdb: {e}")

    def _cmd_pull(self):
        """Responde al comando /pull - ejecuta git stash + git pull en background."""
        send_alert("📥 Ejecutando git pull (con stash si hay conflictos)...")
        logger.info("[BOT] Pull solicitado via /pull")
        project_root = str(Path(__file__).parent.parent)

        def _do_pull():
            try:
                # Primero stash para evitar conflictos con cambios locales
                stash_result = subprocess.run(
                    ['git', 'stash'], capture_output=True, text=True,
                    timeout=30, cwd=project_root,
                )
                stashed = "No local changes" not in stash_result.stdout

                # Luego pull
                result = subprocess.run(
                    ['git', 'pull'], capture_output=True, text=True,
                    timeout=60, cwd=project_root,
                )
                output = (result.stdout.strip() or result.stderr.strip() or "Sin output")[:400]
                ok = result.returncode == 0

                stash_msg = "\n📦 Stash: cambios locales guardados" if stashed else ""
                emoji = "✅" if ok else "❌"
                send_alert(f"{emoji} <b>GIT PULL</b>\n<code>{output}</code>{stash_msg}")
                logger.info(f"[BOT] git pull: rc={result.returncode} stashed={stashed}")
            except Exception as e:
                send_alert(f"❌ Error en git pull: {e}")
                logger.error(f"[BOT] git pull error: {e}")

        threading.Thread(target=_do_pull, daemon=True).start()

    def _cmd_install(self):
        """Responde al comando /install - ejecuta poetry install en background."""
        send_alert("📦 Ejecutando poetry install...")
        logger.info("[BOT] Install solicitado via /install")
        project_root = str(Path(__file__).parent.parent)

        def _do_install():
            try:
                result = subprocess.run(
                    ['poetry', 'install'], capture_output=True, text=True,
                    timeout=300, cwd=project_root,
                )
                output = (result.stdout.strip() or result.stderr.strip() or "Sin output")[:500]
                ok = result.returncode == 0
                emoji = "✅" if ok else "❌"
                send_alert(f"{emoji} <b>POETRY INSTALL</b>\n<code>{output}</code>")
                logger.info(f"[BOT] poetry install: rc={result.returncode} {output[:100]}")
            except Exception as e:
                send_alert(f"❌ Error en poetry install: {e}")
                logger.error(f"[BOT] poetry install error: {e}")

        threading.Thread(target=_do_install, daemon=True).start()

    def _cmd_restart(self):
        """Responde al comando /restart - reinicia el bot via wrapper."""
        n_pos = len(self.portfolio.positions)
        send_alert(
            f"🔄 <b>REINICIANDO BOT</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📈 Posiciones abiertas: {n_pos} (SL en exchange activos)\n"
            f"🔄 Reiniciara en segundos..."
        )
        logger.info("[BOT] Restart solicitado via /restart - exit code 43")
        self._exit_code = 43
        self.running = False

    def _cmd_update(self):
        """Responde al comando /update - hace pull + install + restart automatico."""
        n_pos = len(self.portfolio.positions)
        send_alert(
            f"🔄 <b>UPDATE COMPLETO INICIADO</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📈 Posiciones: {n_pos} (SL activos)\n"
            f"1️⃣ git stash + pull\n"
            f"2️⃣ poetry install\n"
            f"3️⃣ restart"
        )
        logger.info("[BOT] Update completo solicitado via /update")
        project_root = str(Path(__file__).parent.parent)

        def _do_update():
            try:
                # 1. Git stash + pull
                subprocess.run(
                    ['git', 'stash'], capture_output=True, text=True,
                    timeout=30, cwd=project_root,
                )
                pull_result = subprocess.run(
                    ['git', 'pull'], capture_output=True, text=True,
                    timeout=60, cwd=project_root,
                )
                pull_ok = pull_result.returncode == 0
                pull_out = (pull_result.stdout.strip() or pull_result.stderr.strip())[:200]

                if not pull_ok:
                    send_alert(f"❌ <b>UPDATE FALLIDO</b>\ngit pull error:\n<code>{pull_out}</code>")
                    return

                send_alert(f"✅ Pull OK\n<code>{pull_out}</code>\n\n📦 Instalando deps...")

                # 2. Poetry install
                install_result = subprocess.run(
                    ['poetry', 'install', '--no-interaction'], capture_output=True, text=True,
                    timeout=300, cwd=project_root,
                )
                install_ok = install_result.returncode == 0

                if not install_ok:
                    install_err = (install_result.stderr.strip() or install_result.stdout.strip())[:300]
                    send_alert(f"❌ <b>UPDATE FALLIDO</b>\npoetry install error:\n<code>{install_err}</code>")
                    return

                send_alert("✅ Install OK\n\n🔄 Reiniciando bot...")

                # 3. Trigger restart
                self._exit_code = 43
                self.running = False

            except Exception as e:
                send_alert(f"❌ Error en update: {e}")
                logger.error(f"[BOT] update error: {e}")

        threading.Thread(target=_do_update, daemon=True).start()

    def _cmd_export_v14(self):
        """Responde al comando /export_v14 - exporta modelos V14 Ensemble."""
        send_alert(
            f"🔬 <b>EXPORTANDO {BOT_VERSION}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📊 Modelo: Ensemble (RF+GB+LR)\n"
            f"📈 Pares: DOGE, ADA, DOT, SOL\n"
            f"⏱️ Tiempo estimado: 3-5 minutos"
        )
        logger.info(f"[BOT] Export {BOT_VERSION} solicitado via /export_v14")
        project_root = str(Path(__file__).parent.parent)

        def _do_export():
            try:
                result = subprocess.run(
                    [sys.executable, 'ml_export_v14.py', '--force'],
                    capture_output=True, text=True,
                    timeout=600, cwd=project_root,
                )
                ok = result.returncode == 0
                output = result.stdout.strip() or result.stderr.strip() or "Sin output"

                if ok:
                    # Extraer resumen
                    lines = output.split('\n')
                    summary_lines = [l for l in lines if any(s in l for s in ['DOGE:', 'ADA:', 'DOT:', 'SOL:', '✓', '✗'])]
                    if summary_lines:
                        summary = '\n'.join(summary_lines[-6:])
                        send_alert(f"✅ <b>{BOT_VERSION} EXPORTADO</b>\n<code>{summary}</code>")
                    else:
                        send_alert(f"✅ <b>{BOT_VERSION} EXPORTADO</b>\nModelos listos en strategies/*/models/")
                else:
                    error_msg = output[-300:]
                    send_alert(f"❌ <b>{BOT_VERSION} ERROR</b>\n<code>{error_msg}</code>")

                logger.info(f"[BOT] export_v14: rc={result.returncode}")
            except subprocess.TimeoutExpired:
                send_alert(f"❌ <b>{BOT_VERSION} TIMEOUT</b>\nSuperados 10 minutos")
                logger.error("[BOT] export_v14 timeout (10min)")
            except Exception as e:
                send_alert(f"❌ Error en export {BOT_VERSION}: {e}")
                logger.error(f"[BOT] export_v14 error: {e}")

        threading.Thread(target=_do_export, daemon=True).start()

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

            # V8.4: Refresh macro data daily (solo para estrategia legacy)
            if not self.v14_mode and hasattr(self.strategy, 'v84_enabled') and self.strategy.v84_enabled:
                logger.info("[BOT] Actualizando macro (V8.4)...")
                self.strategy.update_macro()
                logger.info(f"[BOT] Macro: score={self.strategy.macro_score:.3f}, "
                            f"sizing={self.strategy.get_sizing_multiplier():.2f}x, "
                            f"thresh={self.strategy.get_adaptive_threshold():.2f}")

        # Actualizar balance
        self.portfolio.refresh_balance()

        # V14: Flujo simplificado
        if self.v14_mode:
            self._on_new_candle_v14()
        # Legacy: dual-mode si V9 activo
        elif self.shadow_enabled and self.strategy.v9_enabled:
            self._on_new_candle_dual()
        else:
            self._on_new_candle_single()

    def _on_new_candle_v14(self):
        """V14: Flujo simplificado con multiples expertos."""
        open_symbols = set(self.portfolio.positions.keys())
        signals = self.strategy.generate_signals(self.exchange_public, open_symbols)

        if signals:
            logger.info(f"[V14] {len(signals)} senales generadas:")
            for s in signals:
                side = 'LONG' if s['direction'] == 1 else 'SHORT'
                logger.info(f"  {s['pair']} {side} | conf={s['confidence']:.2f} | "
                            f"${s['price']:,.2f} | {s.get('setup', '')}")
        else:
            logger.info("[V14] Sin senales en este ciclo")

        for signal in signals:
            self._execute_v14_signal(signal)

    def _on_new_candle_single(self):
        """Modo legacy: generate_signals sin shadow."""
        open_pairs = set(self.portfolio.positions.keys())
        signals = self.strategy.generate_signals(self.exchange_public, open_pairs)

        if signals:
            logger.info(f"[BOT] {len(signals)} senales generadas:")
            for s in signals:
                side = 'LONG' if s['direction'] == 1 else 'SHORT'
                sm = s.get('sizing_mult', 1.0)
                cm = s.get('conviction_mult', 1.0)
                logger.info(f"  {s['pair']} {side} | conf={s['confidence']:.2f} | "
                            f"pred={s['prediction']:+.4f} | ${s['price']:,.2f} | "
                            f"sizing={sm:.2f}x | conv={cm:.2f}x")
        else:
            logger.info("[BOT] Sin senales en este ciclo")

        for signal in signals:
            self._execute_v9_signal(signal)

    def _on_new_candle_dual(self):
        """Modo dual: V8.5 ejecuta en exchange (PROD), V9 ejecuta en shadow."""
        open_pairs_prod = set(self.portfolio.positions.keys())
        open_pairs_shadow = set(self.shadow_portfolio.positions.keys())

        # Fetch BTC 4h data once (shared by both strategies)
        try:
            btc_ohlcv = self.exchange_public.fetch_ohlcv('BTC/USDT', ML_TIMEFRAME, limit=100)
            import pandas as pd
            btc_df = pd.DataFrame(btc_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
            btc_df.set_index('timestamp', inplace=True)
        except Exception as e:
            logger.warning(f"[BOT] Error fetching BTC data: {e}")
            btc_df = None

        # NOTA: generate_dual_signals retorna (v9_filtered, v85_base)
        # Pero ahora INVERTIMOS: v85 va a PROD, v9 va a shadow
        v9_signals, v85_signals = self.strategy.generate_dual_signals(
            self.exchange_public, open_pairs_shadow, open_pairs_prod, btc_df,
        )

        # Log V8.5 signals (ahora PROD)
        if v85_signals:
            logger.info(f"[BOT] V8.5 PROD: {len(v85_signals)} senales:")
            for s in v85_signals:
                side = 'LONG' if s['direction'] == 1 else 'SHORT'
                logger.info(f"  [PROD] {s['pair']} {side} | conf={s['confidence']:.2f} | "
                            f"${s['price']:,.2f} | sizing={s.get('sizing_mult', 1.0):.2f}x")
        else:
            logger.info("[BOT] V8.5 PROD: sin senales")

        # Log V9 shadow signals
        if v9_signals:
            logger.info(f"[BOT] V9 shadow: {len(v9_signals)} senales:")
            for s in v9_signals:
                side = 'LONG' if s['direction'] == 1 else 'SHORT'
                logger.info(f"  [SHADOW] {s['pair']} {side} | conf={s['confidence']:.2f} | "
                            f"${s['price']:,.2f}")
        else:
            logger.info("[BOT] V9 shadow: sin senales")

        # Execute V8.5 signals on exchange (PROD)
        for signal in v85_signals:
            self._execute_v9_signal(signal)

        # Execute V9 signals on shadow portfolio
        for signal in v9_signals:
            self._execute_shadow_signal(signal)

    def _execute_v14_signal(self, signal):
        """Execute a V14 signal with per-pair TP/SL."""
        if not self.portfolio.can_open(signal['pair'], signal['direction']):
            return

        # V14 usa TP/SL del signal (especifico por par)
        success = self.portfolio.open_position(
            pair=signal['pair'],
            direction=signal['direction'],
            confidence=signal['confidence'],
            regime=self.strategy.regime,
            price=signal['price'],
            atr_pct=0.02,  # Default, V14 usa TP/SL fijos
            sizing_mult=1.0,
            tp_pct_override=signal.get('tp_pct'),
            sl_pct_override=signal.get('sl_pct'),
        )

        if success:
            pos = self.portfolio.positions.get(signal['pair'])
            if pos:
                side = 'LONG' if signal['direction'] == 1 else 'SHORT'
                side_emoji = '🟢' if signal['direction'] == 1 else '🔴'
                setup = signal.get('setup', 'ENSEMBLE')
                send_alert(
                    f"{side_emoji} <b>{BOT_VERSION} TRADE ABIERTO</b>\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"💎 {signal['pair']} <b>{side}</b>\n"
                    f"📥 Entry: ${pos.entry_price:,.2f}\n"
                    f"📦 Notional: ${pos.notional:,.2f} ({pos.leverage}x)\n"
                    f"🎯 TP: ${pos.tp_price:,.2f} ({pos.tp_pct:.1%})\n"
                    f"🛡️ SL: ${pos.sl_price:,.2f} ({pos.sl_pct:.1%})\n"
                    f"📊 Setup: {setup}\n"
                    f"🔮 Regime: {self.strategy.regime}"
                )

    def _execute_v9_signal(self, signal):
        """Execute a signal on the real portfolio (V9 or legacy)."""
        if not self.portfolio.can_open(signal['pair'], signal['direction']):
            return

        success = self.portfolio.open_position(
            pair=signal['pair'],
            direction=signal['direction'],
            confidence=signal['confidence'],
            regime=self.strategy.regime,
            price=signal['price'],
            atr_pct=signal['atr_pct'],
            sizing_mult=signal.get('sizing_mult', 1.0),
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
                if signal['direction'] == 1:
                    explain = f"📖 Compra {pos.quantity} {coin} esperando que SUBA"
                    tp_dir = '↗️ sube'
                    sl_dir = '↘️ baja'
                else:
                    explain = f"📖 Vende {pos.quantity} {coin} esperando que BAJE"
                    tp_dir = '↘️ baja'
                    sl_dir = '↗️ sube'
                sm = signal.get('sizing_mult', 1.0)
                cm = signal.get('conviction_mult', 1.0)
                # V14: Ensemble Voting
                if self.v14_mode:
                    model_str = f"🤖 Ensemble"
                # V13.04: Ridge LONG_ONLY, sin Macro/Conv
                elif ML_V1304_ENABLED:
                    model_str = "🔬 Ridge LONG_ONLY"
                else:
                    intel_parts = []
                    if hasattr(self.strategy, 'v84_enabled') and self.strategy.v84_enabled:
                        intel_parts.append(f"🌐 Macro: {self.strategy.macro_score:.2f}")
                    if hasattr(self.strategy, 'v85_enabled') and self.strategy.v85_enabled:
                        intel_parts.append(f"🎯 Conv: {cm:.2f}x")
                    model_str = " | ".join(intel_parts) if intel_parts else ""

                send_alert(
                    f"{side_emoji} <b>TRADE ABIERTO</b>\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"💎 {signal['pair']} <b>{side}</b>\n"
                    f"📥 Entry: ${pos.entry_price:,.2f}\n"
                    f"📦 Notional: ${pos.notional:,.2f} ({pos.leverage}x)\n"
                    f"🎯 TP: ${pos.tp_price:,.2f} ({pos.tp_pct:.1%})\n"
                    f"🛡️ SL: ${pos.sl_price:,.2f} ({pos.sl_pct:.1%})\n"
                    f"{conf_bar} Conf: {signal['confidence']:.2f}\n"
                    f"📊 {self.strategy.regime} | {model_str}"
                )

    def _execute_shadow_signal(self, signal):
        """Execute a signal on the shadow portfolio (V8.5 paper trading)."""
        self.shadow_portfolio.open_position(
            pair=signal['pair'],
            direction=signal['direction'],
            confidence=signal['confidence'],
            regime=self.strategy.regime,
            price=signal['price'],
            atr_pct=signal['atr_pct'],
            sizing_mult=signal.get('sizing_mult', 1.0),
        )

    # =========================================================================
    # POSITION MONITORING
    # =========================================================================
    def _monitor_positions(self):
        """Monitorea posiciones abiertas cada 30s."""
        closed_trades = self.portfolio.update_positions()

        # Shadow positions: fetch tickers and update
        if self.shadow_enabled and self.shadow_portfolio.positions:
            try:
                tickers = {}
                for pair in list(self.shadow_portfolio.positions.keys()):
                    try:
                        t = self.exchange_public.fetch_ticker(pair)
                        tickers[pair] = t['last']
                    except Exception:
                        pass
                shadow_closed = self.shadow_portfolio.update_positions(tickers)
                for st in shadow_closed:
                    sign = '+' if st['pnl'] > 0 else ''
                    logger.info(
                        f"[SHADOW] Cerrado {st['symbol']} {st['side'].upper()} | "
                        f"PnL: ${st['pnl']:{sign}.2f} | {st['exit_reason']}"
                    )
            except Exception as e:
                logger.warning(f"[BOT] Error updating shadow positions: {e}")

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
                f"📥 ${trade['entry_price']:,.2f} → ${trade['exit_price']:,.2f}\n"
                f"📊 Cambio: {price_change_pct:+.2%}\n"
                f"💰 <b>PnL: ${trade['pnl']:+,.2f}</b>\n"
                f"⏱️ {dur_str} | {reason_emoji} {trade['exit_reason']}\n"
                f"💵 Balance: <b>${self.portfolio.balance:,.2f}</b>"
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
            shadow_info = ""
            if self.shadow_enabled:
                ss = self.shadow_portfolio.get_summary()
                shadow_info = f" | Shadow V9: {ss['n_open']}pos ${ss['total_pnl']:+.2f}"
            logger.info(
                f"[STATUS] Balance=${status['balance']:.2f} | "
                f"DD={status['dd']:.1%} | "
                f"Pos={status['positions']}/3 | "
                f"DailyPnL=${status['daily_pnl']:+.2f} | "
                f"Regime={self.strategy.regime}{shadow_info}"
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
        trades_today = self.portfolio.get_today_trades_from_db()  # Solo V9
        total_pnl = sum(t['pnl'] for t in trades_today)

        # Contador de demo V13.04: 14 dias desde 2026-02-28 -> fin 2026-03-14
        demo_end = datetime(2026, 3, 14, tzinfo=timezone.utc)
        days_left = (demo_end - datetime.now(timezone.utc)).days
        if days_left > 0:
            demo_str = f"📅 Demo: {days_left}d restantes"
        elif days_left == 0:
            demo_str = "📅 Demo: ULTIMO DIA!"
        else:
            demo_str = "📅 Demo: FINALIZADA"

        if self.recent_errors:
            errors_str = "\n".join(f"  ⚠️ {e[:80]}" for e in self.recent_errors[-5:])
            n_errors = len(self.recent_errors)
            send_alert(
                f"🔴 <b>{BOT_VERSION} ERRORES ({n_errors})</b>\n"
                f"━━━━━━━━━━━━━━━\n"
                f"{errors_str}\n"
                f"\n"
                f"💰 Balance: ${status['balance']:,.2f}\n"
                f"📈 Pos: {status['positions']}/{ML_MAX_CONCURRENT}\n"
                f"📊 Regime: {self.strategy.regime}\n"
                f"⏱️ Uptime: {uptime_h:.1f}h\n"
                f"{demo_str}"
            )
            self.recent_errors.clear()
        else:
            pnl_emoji = '📈' if total_pnl >= 0 else '📉'

            # V14: Ensemble Voting
            if self.v14_mode:
                model_str = f"🤖 Ensemble ({len(ML_V14_EXPERTS)} expertos)\n"
            # V13.04: modelo Ridge LONG_ONLY
            elif ML_V1304_ENABLED:
                model_str = "🔬 Ridge LONG_ONLY\n"
            else:
                # Legacy V13.03 con Macro/Conviction
                if hasattr(self.strategy, 'v84_enabled') and self.strategy.v84_enabled:
                    model_str = (
                        f"🌐 Macro: {self.strategy.macro_score:.2f} | "
                        f"Sz: {self.strategy.get_sizing_multiplier():.2f}x\n"
                    )
                else:
                    model_str = ""
                if hasattr(self.strategy, 'v85_enabled') and self.strategy.v85_enabled:
                    model_str += "🎯 Conv: ON\n"

            send_alert(
                f"🟢 <b>{BOT_VERSION} OK</b>\n"
                f"━━━━━━━━━━━━━━━\n"
                f"💰 Balance: ${status['balance']:,.2f}\n"
                f"📈 Pos: {status['positions']}/{ML_MAX_CONCURRENT}\n"
                f"{pnl_emoji} PnL hoy: ${total_pnl:+,.2f} ({len(trades_today)}t)\n"
                f"📊 Regime: {self.strategy.regime}\n"
                f"{model_str}"
                f"⚠️ DD: {status['dd']:.1%}\n"
                f"⏱️ Uptime: {uptime_h:.1f}h\n"
                f"{demo_str}"
            )

    def _send_daily_summary(self):
        """Envia resumen diario por Telegram (consulta DB, sobrevive reinicios)."""
        status = self.portfolio.get_status()
        trades_today = self.portfolio.get_today_trades_from_db()  # Solo V9
        wins = sum(1 for t in trades_today if t['pnl'] > 0)
        losses = len(trades_today) - wins
        wr = (wins / len(trades_today) * 100) if trades_today else 0
        total_pnl = sum(t['pnl'] for t in trades_today)

        # Shadow summary
        shadow_str = ""
        if self.shadow_enabled:
            ss = self.shadow_portfolio.get_summary()
            shadow_str = (
                f"\n👻 <b>Shadow</b>: {ss['n_trades']}t | "
                f"${ss['total_pnl']:+,.2f} | WR {ss['win_rate']:.0f}%"
            )

        pnl_emoji = '📈' if total_pnl >= 0 else '📉'
        send_alert(
            f"📊 <b>RESUMEN DIARIO {BOT_VERSION}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📈 Trades: {len(trades_today)} | "
            f"✅ {wins} ❌ {losses} | WR {wr:.0f}%\n"
            f"{pnl_emoji} PnL: <b>${total_pnl:+,.2f}</b>"
            f"{shadow_str}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 Balance: <b>${status['balance']:,.2f}</b>\n"
            f"⚠️ DD: {status['dd']:.1%}\n"
            f"📊 Regime: {self.strategy.regime}\n"
            f"📈 Posiciones: {status['positions']}/{ML_MAX_CONCURRENT}"
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
    """Configura logging para el ML bot con rotacion automatica."""
    LOGS_DIR.mkdir(exist_ok=True)
    log_file = LOGS_DIR / "ml_bot.log"

    rotating = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB por archivo
        backupCount=5,              # Conserva ml_bot.log.1 .. .5
        encoding='utf-8',
    )

    handlers = [
        logging.StreamHandler(),
        rotating,
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
    sys.exit(bot._exit_code)


if __name__ == '__main__':
    main()
