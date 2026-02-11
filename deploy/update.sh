#!/bin/bash
# ============================================================
# Update Script - Actualiza el bot desde GitHub y reinicia
# ============================================================
# Uso desde el servidor: bash deploy/update.sh
# ============================================================

set -e

BOT_DIR="/home/botuser/bot_trader_angry"

echo "[1/4] Descargando cambios..."
cd "$BOT_DIR"
git pull origin main

echo "[2/4] Instalando dependencias nuevas (si hay)..."
/home/botuser/.local/bin/poetry install --no-interaction

echo "[3/4] Reiniciando bot..."
sudo systemctl restart bot-trader

echo "[4/4] Verificando estado..."
sleep 3
sudo systemctl status bot-trader --no-pager -l

echo ""
echo "[OK] Bot actualizado y reiniciado."
