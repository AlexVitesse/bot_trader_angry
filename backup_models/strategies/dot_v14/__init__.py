# DOT V14 Expert
# Status: APPROVED - Ensemble ML Voting
# Validated: 2026-02-28
# Walk-Forward: 6/8 folds positive

from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
VERSION = "14.0"
SYMBOL = "DOT/USDT"
STATUS = "APPROVED"

# ARQUITECTURA:
# - ML Ensemble Voting (RF + GradientBoosting)
# - TP 5% / SL 3% / Timeout 15 (configuracion optimizada)
# - DOT tiene menos historia que otros assets (desde 2020)
