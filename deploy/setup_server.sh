#!/bin/bash
# ============================================================
# Setup Script - Bot Trader Angry (Ubuntu 24.04)
# ============================================================
# NO requiere sudo. Corre con tu usuario normal.
# Uso: bash deploy/setup_server.sh
# ============================================================

set -e

BOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
USER_SYSTEMD="$HOME/.config/systemd/user"

echo "=========================================="
echo "  Bot Trader Angry - Server Setup"
echo "=========================================="
echo "  Usuario: $(whoami)"
echo "  Directorio: $BOT_DIR"
echo ""

# 1. Verificar Python
echo "[1/5] Verificando Python..."
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "  -> Usando $(python3 --version)"
else
    echo "[ERROR] Python3 no encontrado. Pide al admin: sudo apt install python3"
    exit 1
fi
echo "  -> $($PYTHON_CMD --version) OK"

# 2. Instalar Poetry (en usuario)
echo "[2/5] Verificando Poetry..."
if ! command -v poetry &> /dev/null && ! [ -f "$HOME/.local/bin/poetry" ]; then
    echo "  -> Instalando Poetry..."
    curl -sSL https://install.python-poetry.org | $PYTHON_CMD -
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "  -> Poetry ya instalado"
fi

# Asegurar PATH
if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
    export PATH="$HOME/.local/bin:$PATH"
    # Agregar al .bashrc si no esta
    if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
        echo "  -> PATH agregado a .bashrc"
    fi
fi

# 3. Instalar dependencias
echo "[3/5] Instalando dependencias Python..."
cd "$BOT_DIR"
poetry install --no-interaction

# Crear directorio de datos
mkdir -p "$BOT_DIR/data"

# 4. Configurar servicio systemd de usuario (no requiere sudo)
echo "[4/5] Configurando servicio systemd de usuario..."
mkdir -p "$USER_SYSTEMD"

cat > "$USER_SYSTEMD/bot-trader.service" << SERVICEEOF
[Unit]
Description=Bot Trader Angry - Binance Scalper v6.7
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$BOT_DIR
ExecStart=$HOME/.local/bin/poetry run python src/bot.py
Restart=always
RestartSec=10
StartLimitIntervalSec=300
StartLimitBurst=5
EnvironmentFile=$BOT_DIR/.env

[Install]
WantedBy=default.target
SERVICEEOF

systemctl --user daemon-reload
systemctl --user enable bot-trader.service
echo "  -> Servicio de usuario configurado"

# 5. Verificar .env
echo "[5/5] Verificando configuracion..."
if [ ! -f "$BOT_DIR/.env" ]; then
    cat > "$BOT_DIR/.env" << 'ENVEOF'
# === BINANCE API ===
BINANCE_API_KEY=TU_API_KEY_AQUI
BINANCE_API_SECRET=TU_API_SECRET_AQUI
TRADING_MODE=testnet

# === TELEGRAM (opcional) ===
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
ENVEOF
    echo "  -> .env creado. EDITA con tus API keys:"
    echo "     nano $BOT_DIR/.env"
else
    echo "  -> .env ya existe"
fi

echo ""
echo "=========================================="
echo "  Setup completado!"
echo "=========================================="
echo ""
echo "  PASOS SIGUIENTES:"
echo ""
echo "  1. Editar .env con tus API keys:"
echo "     nano $BOT_DIR/.env"
echo ""
echo "  2. Iniciar el bot:"
echo "     systemctl --user start bot-trader"
echo ""
echo "  3. Ver logs en tiempo real:"
echo "     journalctl --user -u bot-trader -f"
echo ""
echo "  4. Comandos utiles:"
echo "     systemctl --user stop bot-trader      # Detener"
echo "     systemctl --user restart bot-trader   # Reiniciar"
echo "     systemctl --user status bot-trader    # Estado"
echo ""
echo "  NOTA: Si el servicio no sobrevive al logout,"
echo "  pide al admin ejecutar:"
echo "     sudo loginctl enable-linger $(whoami)"
echo "=========================================="
