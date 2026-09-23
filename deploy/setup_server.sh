#!/bin/bash
# ============================================================
# Instala el bot como servicio systemd de usuario (sobrevive a reboots).
# Uso: PYTHON=/ruta/al/python bash deploy/setup_server.sh
#   (por defecto el python del VPS condor-ia: ~/envs/deepseek/bin/python)
# Requiere linger para arrancar sin sesion abierta:
#   sudo loginctl enable-linger $(whoami)
# ============================================================
set -e

BOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$HOME/envs/deepseek/bin/python}"
UNIT_DIR="$HOME/.config/systemd/user"

[ -x "$PYTHON" ] || { echo "[ERROR] no existe $PYTHON"; exit 1; }
[ -f "$BOT_DIR/.env" ] || { echo "[ERROR] falta $BOT_DIR/.env"; exit 1; }

# Dos supervisores = dos instancias del bot (paso en agosto).
if pgrep -f run_bot.sh >/dev/null; then
    echo "[ERROR] run_bot.sh sigue corriendo. Paralo antes: pkill -f run_bot.sh"
    exit 1
fi

mkdir -p "$UNIT_DIR"
sed -e "s|@BOT_DIR@|$BOT_DIR|" -e "s|@PYTHON@|$PYTHON|" \
    "$BOT_DIR/deploy/bot-trader.service" > "$UNIT_DIR/bot-trader.service"
systemctl --user daemon-reload
systemctl --user enable bot-trader.service

loginctl show-user "$(whoami)" -p Linger | grep -q yes || \
    echo "[AVISO] linger desactivado: el bot no arrancara tras un reboot sin login."

echo "Instalado. Arrancar:  systemctl --user start bot-trader"
echo "Log del bot:          tail -f $BOT_DIR/logs/ml_bot.log"
