# ETH V14 Expert
# Status: APPROVED - Walk-forward validated 2026-02-28
# Setups: 3 (RSI_OVERSOLD_SHORT, VOLUME_SPIKE_UP, VOLUME_SPIKE_DOWN)
# Total PnL backtest: +1,230% combined

from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
VERSION = "14.0"
SYMBOL = "ETH/USDT"
STATUS = "APPROVED"

# Diferencias vs BTC:
# - TP/SL mas tight (4-5% / 2-2.5%)
# - 3 setups vs 8 de BTC
# - Volume spikes son clave para ETH
# - RSI oversold -> SHORT (contraintuitivo pero funciona)
