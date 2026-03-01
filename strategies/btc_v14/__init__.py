# BTC V14 Ensemble Strategy
# Status: APPROVED - Cross-validated on ETH + 5 synthetic markets
# Results: 6/6 scenarios profitable

from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
VERSION = "14.0"
SYMBOL = "BTC/USDT"
STATUS = "APPROVED"

# Strategy uses: Regime Detection + 8 Setups + Ensemble ML (3 models)
