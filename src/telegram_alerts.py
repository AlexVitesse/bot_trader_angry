"""
Telegram Alerts - Monitoreo Remoto del Bot
============================================
Envia alertas asincronas via Telegram sin bloquear el trading.
"""

import logging
import threading
import requests
from datetime import datetime

from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_ENABLED, SYMBOL

logger = logging.getLogger(__name__)

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


def _send_message(text: str):
    """Envia un mensaje via Telegram API (bloqueante, llamar desde thread)."""
    if not TELEGRAM_ENABLED:
        return
    try:
        response = requests.post(
            f"{TELEGRAM_API_URL}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        if response.status_code != 200:
            logger.warning(f"[TG] Error enviando mensaje: {response.status_code} {response.text}")
    except Exception as e:
        logger.warning(f"[TG] No se pudo enviar alerta: {e}")


def send_alert(text: str):
    """Envia alerta en un thread separado (no bloquea el bot)."""
    if not TELEGRAM_ENABLED:
        return
    t = threading.Thread(target=_send_message, args=(text,), daemon=True)
    t.start()


def _send_document(file_path: str, caption: str = ""):
    """Envia un archivo via Telegram API (bloqueante, llamar desde thread)."""
    if not TELEGRAM_ENABLED:
        return
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(
                f"{TELEGRAM_API_URL}/sendDocument",
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                files={"document": (f.name.split('/')[-1].split('\\')[-1], f)},
                timeout=30,
            )
        if response.status_code != 200:
            logger.warning(f"[TG] Error enviando archivo: {response.status_code}")
    except Exception as e:
        logger.warning(f"[TG] No se pudo enviar archivo: {e}")


def send_document(file_path: str, caption: str = ""):
    """Envia archivo en un thread separado (no bloquea el bot)."""
    if not TELEGRAM_ENABLED:
        return
    t = threading.Thread(target=_send_document, args=(file_path, caption), daemon=True)
    t.start()


# =====================================================================
# ALERTAS ESPECIFICAS
# =====================================================================

def alert_trade_opened(side: str, price: float, quantity: float, margin: float):
    """Alerta cuando se abre un trade."""
    emoji = "\U0001F7E2" if side.upper() == "LONG" else "\U0001F534"
    text = (
        f"{emoji} <b>TRADE ABIERTO</b>\n"
        f"Par: {SYMBOL}\n"
        f"Side: {side.upper()}\n"
        f"Precio: ${price:,.2f}\n"
        f"Qty: {quantity}\n"
        f"Margen: ${margin:.2f}"
    )
    send_alert(text)


def alert_trade_closed(side: str, entry_price: float, exit_price: float,
                       pnl: float, exit_reason: str, duration_min: float):
    """Alerta cuando se cierra un trade."""
    emoji = "\u2705" if pnl > 0 else "\u274C"
    text = (
        f"{emoji} <b>TRADE CERRADO</b>\n"
        f"Par: {SYMBOL} | {side.upper()}\n"
        f"Entry: ${entry_price:,.2f} -> Exit: ${exit_price:,.2f}\n"
        f"PnL: <b>${pnl:+.4f}</b>\n"
        f"Razon: {exit_reason}\n"
        f"Duracion: {duration_min:.1f} min"
    )
    send_alert(text)


def alert_dca_executed(so_num: int, price: float, new_avg: float, total_qty: float):
    """Alerta cuando se ejecuta un DCA."""
    text = (
        f"\U0001F504 <b>DCA #{so_num}</b>\n"
        f"Precio: ${price:,.2f}\n"
        f"Nuevo promedio: ${new_avg:,.2f}\n"
        f"Qty total: {total_qty}"
    )
    send_alert(text)


def alert_kill_switch(reason: str):
    """Alerta cuando se activa un kill switch."""
    text = (
        f"\U0001F6A8 <b>KILL SWITCH ACTIVADO</b>\n"
        f"Razon: {reason}\n"
        f"Bot PAUSADO - revisar manualmente"
    )
    send_alert(text)


def alert_ws_disconnected(attempt: int, max_attempts: int):
    """Alerta cuando el WebSocket se desconecta."""
    text = (
        f"\u26A0\uFE0F <b>WS DESCONECTADO</b>\n"
        f"Reintento: {attempt}/{max_attempts}"
    )
    send_alert(text)


def alert_error(error_msg: str):
    """Alerta para errores criticos."""
    text = (
        f"\U0001F4A5 <b>ERROR CRITICO</b>\n"
        f"{error_msg}"
    )
    send_alert(text)


def alert_daily_summary(trades_today: int, wins: int, losses: int,
                        pnl: float, balance: float, win_rate: float):
    """Resumen diario automatico."""
    emoji = "\U0001F4C8" if pnl >= 0 else "\U0001F4C9"
    text = (
        f"{emoji} <b>RESUMEN DIARIO</b>\n"
        f"Trades: {trades_today} | Wins: {wins} | Losses: {losses}\n"
        f"Win Rate: {win_rate:.1f}%\n"
        f"PnL: <b>${pnl:+.4f}</b>\n"
        f"Balance: ${balance:,.2f}"
    )
    send_alert(text)


def alert_bot_started(balance: float, mode: str):
    """Alerta cuando el bot arranca."""
    text = (
        f"\U0001F680 <b>BOT INICIADO</b>\n"
        f"Modo: {mode}\n"
        f"Par: {SYMBOL}\n"
        f"Balance: ${balance:,.2f}"
    )
    send_alert(text)


def alert_status(balance: float, in_position: bool, side: str,
                 pnl_unrealized: float, trades_today: int):
    """Respuesta al comando /status."""
    pos_text = f"{side.upper()} | PnL: ${pnl_unrealized:+.4f}" if in_position else "Sin posicion"
    text = (
        f"\U0001F4CA <b>STATUS</b>\n"
        f"Balance: ${balance:,.2f}\n"
        f"Posicion: {pos_text}\n"
        f"Trades hoy: {trades_today}"
    )
    send_alert(text)


# =====================================================================
# POLLING DE COMANDOS (para /status)
# =====================================================================

class TelegramPoller:
    """Escucha comandos de Telegram en background.
    Comandos soportados: /status, /resume"""

    def __init__(self, callbacks: dict = None):
        self.running = False
        self.last_update_id = 0
        self.callbacks = callbacks or {}

    def start(self):
        if not TELEGRAM_ENABLED:
            return
        self.running = True
        t = threading.Thread(target=self._poll_loop, daemon=True)
        t.start()
        logger.info("[TG] Poller de comandos iniciado (/status, /resume)")

    def stop(self):
        self.running = False

    def _poll_loop(self):
        import time as _time
        while self.running:
            try:
                response = requests.get(
                    f"{TELEGRAM_API_URL}/getUpdates",
                    params={"offset": self.last_update_id + 1, "timeout": 30},
                    timeout=35,
                )
                if response.status_code == 200:
                    data = response.json()
                    for update in data.get("result", []):
                        self.last_update_id = update["update_id"]
                        msg = update.get("message", {})
                        text = (msg.get("text", "") or "").strip()
                        cmd = text.split()[0] if text else ""
                        if cmd in self.callbacks:
                            try:
                                self.callbacks[cmd]()
                            except Exception as e:
                                logger.warning(f"[TG] Error ejecutando {cmd}: {e}")
            except Exception as e:
                logger.warning(f"[TG] Error en polling: {e}")
                _time.sleep(5)


if __name__ == "__main__":
    if TELEGRAM_ENABLED:
        print(f"[OK] Telegram configurado. Enviando test...")
        _send_message("\U0001F916 Bot de trading conectado. Alertas activas.")
        print("[OK] Mensaje enviado. Revisa tu Telegram.")
    else:
        print("[WARN] TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no configurados en .env")
