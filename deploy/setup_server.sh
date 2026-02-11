#!/bin/bash
# ============================================================
# Setup Script - Bot Trader Angry (Ubuntu 24.04)
# ============================================================
# Uso: sudo bash setup_server.sh
# ============================================================

set -e

BOT_USER="botuser"
BOT_DIR="/home/$BOT_USER/bot_trader_angry"
REPO_URL="https://github.com/AlexVitesse/bot_trader_angry.git"

echo "=========================================="
echo "  Bot Trader Angry - Server Setup"
echo "=========================================="

# 1. Actualizar sistema
echo "[1/7] Actualizando sistema..."
apt update && apt upgrade -y

# 2. Instalar dependencias del sistema
echo "[2/7] Instalando dependencias..."
apt install -y python3.12 python3.12-venv python3-pip git curl ufw

# 3. Instalar Poetry
echo "[3/7] Instalando Poetry..."
if ! command -v poetry &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 -
fi

# 4. Crear usuario dedicado (si no existe)
echo "[4/7] Configurando usuario $BOT_USER..."
if ! id "$BOT_USER" &>/dev/null; then
    useradd -m -s /bin/bash "$BOT_USER"
    echo "  -> Usuario $BOT_USER creado"
else
    echo "  -> Usuario $BOT_USER ya existe"
fi

# Asegurar que Poetry esta en el PATH del botuser
su - $BOT_USER -c "curl -sSL https://install.python-poetry.org | python3 -" || true

# 5. Clonar repo e instalar dependencias
echo "[5/7] Clonando repositorio..."
if [ ! -d "$BOT_DIR" ]; then
    su - $BOT_USER -c "git clone $REPO_URL"
else
    echo "  -> Repo ya existe, haciendo pull..."
    su - $BOT_USER -c "cd $BOT_DIR && git pull"
fi

echo "  -> Instalando dependencias Python..."
su - $BOT_USER -c "cd $BOT_DIR && /home/$BOT_USER/.local/bin/poetry install --no-interaction"

# Crear directorio de datos
su - $BOT_USER -c "mkdir -p $BOT_DIR/data"

# 6. Configurar systemd
echo "[6/7] Configurando servicio systemd..."
cp "$BOT_DIR/deploy/bot-trader.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable bot-trader.service
echo "  -> Servicio habilitado (se inicia con el sistema)"

# Configurar logrotate
cp "$BOT_DIR/deploy/logrotate-bot" /etc/logrotate.d/bot-trader

# 7. Firewall
echo "[7/7] Configurando firewall..."
ufw allow OpenSSH
ufw --force enable
echo "  -> Firewall: SSH permitido, todo lo demas bloqueado (outbound libre)"

echo ""
echo "=========================================="
echo "  Setup completado!"
echo "=========================================="
echo ""
echo "  PASOS MANUALES RESTANTES:"
echo ""
echo "  1. Crear archivo .env:"
echo "     sudo -u $BOT_USER nano $BOT_DIR/.env"
echo ""
echo "     Contenido:"
echo "     BINANCE_API_KEY=tu_api_key_real"
echo "     BINANCE_API_SECRET=tu_api_secret_real"
echo "     TRADING_MODE=live"
echo "     TELEGRAM_BOT_TOKEN=tu_token"
echo "     TELEGRAM_CHAT_ID=tu_chat_id"
echo ""
echo "  2. Iniciar el bot:"
echo "     sudo systemctl start bot-trader"
echo ""
echo "  3. Ver logs en tiempo real:"
echo "     sudo journalctl -u bot-trader -f"
echo ""
echo "  4. Otros comandos utiles:"
echo "     sudo systemctl stop bot-trader     # Detener"
echo "     sudo systemctl restart bot-trader  # Reiniciar"
echo "     sudo systemctl status bot-trader   # Estado"
echo "=========================================="
