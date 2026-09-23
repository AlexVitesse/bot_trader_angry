# BTC V14 Configuration
# Validated: 2026-02-28
# Status: APPROVED FOR PRODUCTION

# =============================================================================
# RISK MANAGEMENT
# =============================================================================
BASE_RISK_PCT = 0.02          # 2% base risk per trade
MAX_PORTFOLIO_HEAT = 0.06     # 6% max total exposure
MAX_POSITIONS = 3             # Max concurrent positions

# =============================================================================
# POSITION SIZING (Confidence-based)
# =============================================================================
CONFIDENCE_THRESHOLDS = {
    'skip': 0.35,       # Below this = no trade
    'very_low': 0.45,   # 0.35-0.45 = 25% of base risk
    'low': 0.55,        # 0.45-0.55 = 50% of base risk
    'medium': 0.65,     # 0.55-0.65 = 75% of base risk
    'high': 0.75,       # 0.65-0.75 = 100% of base risk
    'very_high': 1.0,   # 0.75+ = 125% of base risk
}

# =============================================================================
# ANTI-MARTINGALE (Streak adjustment)
# =============================================================================
LOSING_STREAK_THRESHOLD = -3   # After 3 losses, reduce size
LOSING_STREAK_MULTIPLIER = 0.5 # Cut size in half
WINNING_STREAK_THRESHOLD = 3   # After 3 wins, increase size
WINNING_STREAK_MULTIPLIER = 1.25 # Increase by 25%

# =============================================================================
# REGIME DETECTION
# =============================================================================
ADX_TREND_THRESHOLD = 25       # ADX > 25 = trending
CHOP_RANGE_THRESHOLD = 61.8    # CHOP > 61.8 = range-bound
DI_DIFF_THRESHOLD = 10         # DI+ - DI- threshold for direction

# =============================================================================
# TRADE MANAGEMENT
# =============================================================================
TP_PCT = 0.06                  # 6% take profit
SL_PCT = 0.03                  # 3% stop loss (2:1 R:R)
TIMEOUT_CANDLES = 20           # Exit after 20 candles if no TP/SL

# =============================================================================
# ENSEMBLE WEIGHTS
# =============================================================================
ENSEMBLE_WEIGHTS = {
    'context': 0.4,    # Macro context model
    'momentum': 0.35,  # Momentum model
    'volume': 0.25,    # Volume model
}

# =============================================================================
# VALIDATED SETUPS (All 8 profitable in backtest)
# =============================================================================
ACTIVE_SETUPS = [
    'PULLBACK_UPTREND',
    'SUPPORT_BOUNCE',
    'CAPITULATION',
    'OVERSOLD_EXTREME',
    'RALLY_DOWNTREND',
    'RESISTANCE_REJECTION',
    'EXHAUSTION',
    'OVERBOUGHT_EXTREME',
]

# =============================================================================
# PAIR CONFIG
# =============================================================================
SYMBOL = "BTC/USDT"
TIMEFRAME = "4h"
