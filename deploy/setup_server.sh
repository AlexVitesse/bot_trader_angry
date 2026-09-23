#!/bin/bash
# ============================================================
# Arranque del bot tras un reboot via cron @reboot.
#
# En condor-ia no hay sudo ni bus de systemd de usuario ("Failed to connect to
# bus: No medium found"), asi que la unidad systemd del plan 5.1 no aplica.
# cron @reboot si funciona (ya arranca sshd y tailscaled).
#
# Supervisor = run_bot.sh: relanza en crash (30 s) y con /restart (43);
# sale con 0 (Ctrl+C, kill switch) y entonces NO relanza.
#
# Uso: bash deploy/setup_server.sh     (idempotente)
# ============================================================
set -e
BOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$HOME/envs/deepseek/bin/python}"
[ -x "$PYTHON" ] || { echo "[ERROR] no existe $PYTHON"; exit 1; }

LINE="@reboot sleep 30 && pgrep -f run_bot.sh >/dev/null || (cd $BOT_DIR && PYTHON=$PYTHON nohup bash run_bot.sh >> logs/wrapper.log 2>&1 &)"
( crontab -l 2>/dev/null | grep -v 'run_bot.sh'; echo "$LINE" ) | crontab -
echo "cron @reboot instalado:"
crontab -l | grep run_bot.sh
