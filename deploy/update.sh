#!/bin/bash
# Actualiza el codigo y reinicia el servicio. Uso: bash deploy/update.sh
set -e
cd "$(dirname "$0")/.."
git pull --ff-only origin main
systemctl --user restart bot-trader
systemctl --user --no-pager status bot-trader | head -5
