#!/bin/bash
# ============================================================
# Update Script - Actualiza el bot desde GitHub y reinicia
# ============================================================
# Uso: bash deploy/update.sh
# NO requiere sudo.
# ============================================================

set -e

BOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "[1/4] Descargando cambios..."
cd "$BOT_DIR"
git pull origin main

echo "[2/4] Instalando dependencias nuevas (si hay)..."
poetry install --no-interaction

echo "[3/4] Reiniciando bot..."
systemctl --user restart bot-trader

echo "[4/4] Verificando estado..."
sleep 3
systemctl --user status bot-trader --no-pager -l

echo ""
echo "[OK] Bot actualizado y reiniciado."
