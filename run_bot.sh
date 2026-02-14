#!/bin/bash
# =============================================================
# ML Bot Wrapper - Auto-restart on crash or /restart command
# =============================================================
# Usage: bash run_bot.sh
#
# Exit codes from bot:
#   0  = Normal stop (Ctrl+C)
#   43 = /restart  -> restart bot
#   *  = Crash -> wait 30s + restart
# =============================================================

cd "$(dirname "$0")"

echo "[WRAPPER] ML Bot Wrapper iniciado"
echo "[WRAPPER] Directorio: $(pwd)"

while true; do
    echo "[WRAPPER] Iniciando bot..."
    python -u -m src.ml_bot
    EXIT_CODE=$?

    echo "[WRAPPER] Bot termino con codigo: $EXIT_CODE"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[WRAPPER] Bot detenido normalmente. Saliendo."
        break

    elif [ $EXIT_CODE -eq 43 ]; then
        echo "[WRAPPER] Restart solicitado. Reiniciando bot..."

    else
        echo "[WRAPPER] Bot crasheo (codigo $EXIT_CODE). Reiniciando en 30s..."
        sleep 30
    fi

    echo "---"
done

echo "[WRAPPER] Wrapper finalizado."
