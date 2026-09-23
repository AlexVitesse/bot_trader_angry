#!/bin/bash
# ============================================================
# Arranque tras un reboot via cron @reboot: dos bots + grabadores.
#   v2        -> cuenta demo de Binance (el de siempre)
#   agresivo  -> paper interno con $100 propios (src/paper_exchange.py)
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

linea() {   # $1 = perfil; va en la linea de comando para que pgrep distinga los wrappers
    echo "@reboot sleep 30 && pgrep -f 'run_bot[.]sh $1' >/dev/null || (cd $BOT_DIR && PYTHON=$PYTHON nohup bash run_bot.sh $1 >> logs/wrapper_$1.log 2>&1 &)"
}
# Grabador de streams (docs/GRABACION_DATOS_VIVO.md, capa B). Proceso aparte:
# si muere se relanza a los 30 s y el bot no se entera.
REC="@reboot sleep 40 && pgrep -f 'record_stream[.]py' >/dev/null || (cd $BOT_DIR && nohup bash -c 'while true; do $PYTHON scripts/record_stream.py; sleep 30; done' >> logs/record_stream.log 2>&1 &)"
# pgrep con corchete: sin el, el patron coincide con el propio 'sh -c' de cron y nunca arranca.
# Experto de noticias (experiments/sentimiento/): solo graba, usa el Ollama local.
NEWS="@reboot sleep 90 && pgrep -f 'record_news[.]py' >/dev/null || (cd $BOT_DIR && nohup bash -c 'while true; do $PYTHON scripts/record_news.py; sleep 60; done' >> logs/record_news.log 2>&1 &)"
( crontab -l 2>/dev/null | grep -v 'run_bot.sh' | grep -v 'record_stream.py' | grep -v 'record_news.py'; linea v2; linea agresivo; echo "$REC"; echo "$NEWS" ) | crontab -
echo "cron @reboot instalado:"
crontab -l | grep -E 'run_bot.sh|record_stream.py|record_news.py'
