# ADA V14 Expert
# Status: APPROVED - Ensemble ML Voting
# Validated: 2026-02-28
# Walk-Forward: 11/12 folds positive

from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
VERSION = "14.0"
SYMBOL = "ADA/USDT"
STATUS = "APPROVED"

# ARQUITECTURA:
# - Similar a DOGE: ML Ensemble Voting (3 modelos votan)
# - Los setups tecnicos no funcionaron para ADA
# - ADA es altcoin de smart contracts, similar a ETH pero mas volatil
