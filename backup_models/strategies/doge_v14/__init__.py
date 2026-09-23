# DOGE V14 Expert
# Status: APPROVED - Ensemble ML Voting
# Validated: 2026-02-28
# Cross-validated: SHIB +194%, PEPE +192%

from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
VERSION = "14.0"
SYMBOL = "DOGE/USDT"
STATUS = "APPROVED"

# DIFERENCIA CLAVE vs BTC/ETH:
# - BTC/ETH usan reglas técnicas + ML para confianza
# - DOGE usa ML Ensemble Voting (3 modelos predicen directamente)
# - Razón: memecoins no siguen patrones técnicos tradicionales
