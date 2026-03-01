# DOGE V14 Configuration
# Validated: 2026-02-28
# Status: APPROVED - Ensemble ML Voting
# Cross-validated: SHIB +194%, PEPE +192%

SYMBOL = "DOGE/USDT"
TIMEFRAME = "4h"

# =============================================================================
# ARQUITECTURA DIFERENTE A BTC/ETH
# =============================================================================
# DOGE usa ML Ensemble Voting (3 modelos votan)
# NO usa reglas técnicas como BTC/ETH
# Razón: DOGE es memecoin, patrones técnicos no funcionan bien

MODEL_TYPE = "ENSEMBLE_VOTING"
MODELS = ["RandomForest", "GradientBoosting", "LogisticRegression"]
VOTING_THRESHOLD = 2  # Al menos 2/3 modelos deben coincidir

# =============================================================================
# FEATURES PARA ML
# =============================================================================
FEATURE_COLS = [
    'rsi', 'macd', 'adx', 'bb_pct', 'atr_pct',
    'ret_3', 'ret_5', 'ret_10', 'vol_ratio', 'trend'
]

# =============================================================================
# TRADE PARAMETERS (más conservadores que BTC/ETH)
# =============================================================================
TP_PCT = 0.06         # 6% take profit
SL_PCT = 0.04         # 4% stop loss (1.5:1 ratio)
TIMEOUT_CANDLES = 10  # Menos tiempo - DOGE es rápido

# =============================================================================
# RISK MANAGEMENT (posición reducida por volatilidad)
# =============================================================================
BASE_RISK_PCT = 0.015     # 1.5% base (vs 2% en BTC/ETH)
MAX_PORTFOLIO_HEAT = 0.04  # 4% max exposure
MAX_POSITIONS = 2          # Max concurrent

# =============================================================================
# VALIDATION RESULTS
# =============================================================================
VALIDATION = {
    'walk_forward': {'folds_positive': 7, 'folds_total': 9, 'pnl': 414},
    'cross_memecoin': {
        'SHIB': {'trades': 184, 'wr': 0.51, 'pnl': 194},
        'PEPE': {'trades': 163, 'wr': 0.52, 'pnl': 192},
    },
    'synthetic': {'positive': 4, 'total': 5},
}

# =============================================================================
# NOTA IMPORTANTE
# =============================================================================
# Este modelo también funciona para SHIB y PEPE
# Considerar crear expertos similares para otras memecoins
