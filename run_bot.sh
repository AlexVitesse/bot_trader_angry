#!/bin/bash
# =============================================================
# ML Bot Wrapper - Auto-restart, update, and retrain handler
# =============================================================
# Usage: bash run_bot.sh
#
# Exit codes from bot:
#   0  = Normal stop (user pressed Ctrl+C)
#   42 = Update requested (/update) -> git pull + pip install + restart
#   43 = Restart requested (/restart) -> just restart
#   44 = Retrain requested (/retrain) -> retrain models + restart
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

    elif [ $EXIT_CODE -eq 42 ]; then
        echo "[WRAPPER] Update solicitado. Ejecutando git pull..."
        git pull
        echo "[WRAPPER] Instalando dependencias..."
        pip install -e . 2>/dev/null || pip install -r requirements.txt 2>/dev/null || true
        echo "[WRAPPER] Reiniciando bot..."

    elif [ $EXIT_CODE -eq 43 ]; then
        echo "[WRAPPER] Restart solicitado."
        echo "[WRAPPER] Reiniciando bot..."

    elif [ $EXIT_CODE -eq 44 ]; then
        echo "[WRAPPER] Retrain solicitado. Reentrenando modelos..."
        python -u ml_export_models.py
        RETRAIN_CODE=$?
        if [ $RETRAIN_CODE -eq 0 ]; then
            echo "[WRAPPER] Reentrenamiento exitoso."
        else
            echo "[WRAPPER] ERROR en reentrenamiento (codigo $RETRAIN_CODE)."
        fi
        echo "[WRAPPER] Reiniciando bot..."

    else
        echo "[WRAPPER] Bot crasheo (codigo $EXIT_CODE). Reiniciando en 30s..."
        sleep 30
    fi

    echo "---"
done

echo "[WRAPPER] Wrapper finalizado."
