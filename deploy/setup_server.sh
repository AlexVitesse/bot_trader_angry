#!/bin/bash
# ============================================================
# Arranque de los dos bots tras un reboot via cron @reboot.
#   v2        -> paper interno (sin ordenes)
#   agresivo  -> cuenta demo de Binance
#
# En condor-ia no hay sudo ni bus de systemd de usuario ("Failed to connect to
# bus: No medium found"); cron @reboot si funciona (ya arranca sshd y tailscaled).
# Supervisor = run_bot.sh: relanza en crash (30 s) y con /restart (43); sale con
# 0 (Ctrl+C, kill switch) y entonces NO relanza.
#
# Uso: bash deploy/setup_server.sh     (idempotente)
# ============================================================
set -e
BOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$HOME/envs/deepseek/bin/python}"
[ -x "$PYTHON" ] || { echo "[ERROR] no existe $PYTHON"; exit 1; }

linea() {   # $1 = perfil
    echo "@reboot sleep 30 && pgrep -f 'run_bot.sh $1' >/dev/null || (cd $BOT_DIR && PYTHON=$PYTHON nohup bash run_bot.sh $1 >> logs/wrapper_$1.log 2>&1 &)"
}
( crontab -l 2>/dev/null | grep -v 'run_bot.sh'; linea v2; linea agresivo ) | crontab -
echo "cron @reboot instalado:"
crontab -l | grep run_bot.sh
