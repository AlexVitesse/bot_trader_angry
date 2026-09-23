#!/bin/bash
# Actualiza el codigo y reinicia los dos bots (run_bot.sh relanza cada proceso
# python al morir). No hacerlo con una orden a medio ejecutar.
set -e
cd "$(dirname "$0")/.."
git pull --ff-only origin main
for pid in $(pgrep -f "python -u -m src.ml_bot"); do kill "$pid"; done
sleep 40
ps -eo pid,lstart,cmd | grep -E "[r]un_bot|[s]rc.ml_bot"
