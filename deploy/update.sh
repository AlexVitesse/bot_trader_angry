#!/bin/bash
# Actualiza el codigo y reinicia el bot (run_bot.sh lo relanza al morir el
# proceso python). No hacerlo con una posicion abierta a mitad de orden.
set -e
cd "$(dirname "$0")/.."
git pull --ff-only origin main
pkill -f "python -u -m src.ml_bot" || true
sleep 40
ps -eo pid,lstart,cmd | grep -E "run_bot|src.ml_bot" | grep -v grep
