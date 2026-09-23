"""
Telegram Alerts - Monitoreo Remoto del Bot
============================================
Envia alertas asincronas via Telegram sin bloquear el trading.
"""

import logging
import threading
import requests

from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_ENABLED, BOT_VERSION

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
# POLLING DE COMANDOS (para /status)
# =====================================================================

class TelegramPoller:
    """Escucha comandos de Telegram en background.

    Comandos soportados:
    - /status: Estado actual del bot
    - /resume: Reanudar trading
    - /pull, /update, /restart, /log ... (ver ml_bot._startup)
    """

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
                        parts = text.split(None, 1) if text else []
                        cmd_raw = parts[0] if parts else ""
                        arg = parts[1] if len(parts) > 1 else ""
                        # Normalizar comando:
                        #  - quitar @botname si Telegram lo agrega (en grupos)
                        #  - lowercase para tolerancia
                        cmd = cmd_raw.split('@', 1)[0].lower()
                        if cmd_raw:
                            logger.info(f"[TG] cmd recibido: {cmd_raw!r} -> normalizado: {cmd!r}")
                        # Buscar callback por matching case-insensitive
                        cb = None
                        for k, v in self.callbacks.items():
                            if k.lower() == cmd:
                                cb = v
                                break
                        if cb is not None:
                            try:
                                import inspect
                                if arg and len(inspect.signature(cb).parameters) > 0:
                                    cb(arg)
                                else:
                                    cb()
                                logger.info(f"[TG] cmd {cmd} ejecutado OK")
                            except Exception as e:
                                logger.error(f"[TG] Error ejecutando {cmd}: {e}",
                                             exc_info=True)
                                # Intentar notificar al usuario del error
                                try:
                                    send_alert(f"❌ Error ejecutando {cmd}: {e}")
                                except Exception:
                                    pass
                        elif cmd_raw.startswith('/'):
                            logger.warning(f"[TG] comando desconocido: {cmd_raw}")
            except Exception as e:
                logger.warning(f"[TG] Error en polling: {e}")
                _time.sleep(5)


if __name__ == "__main__":
    if TELEGRAM_ENABLED:
        print(f"[OK] Telegram configurado. Enviando test...")
        _send_message(f"\U0001F916 Bot de trading {BOT_VERSION} conectado. Alertas activas.")
        print("[OK] Mensaje enviado. Revisa tu Telegram.")
    else:
        print("[WARN] TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no configurados en .env")
