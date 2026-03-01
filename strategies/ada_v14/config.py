# ADA V14 Configuration
# Validated: 2026-02-28
# Status: APPROVED - Ensemble ML Voting
# Walk-Forward: 11/12 folds positive, +458% PnL

SYMBOL = "ADA/USDT"
TIMEFRAME = "4h"

# =============================================================================
# ARQUITECTURA: ML ENSEMBLE VOTING
# =============================================================================
# Similar a DOGE - setups tecnicos no funcionaron para ADA
# 3 modelos ML votan si tradear o no

MODEL_TYPE = "ENSEMBLE_VOTING"
MODELS = ["RandomForest", "GradientBoosting", "LogisticRegression"]
VOTING_THRESHOLD = 2  # Al menos 2/3 modelos deben coincidir

# =============================================================================
# FEATURES PARA ML
# =============================================================================
FEATURE_COLS = [
    'rsi', 'macd_norm', 'adx', 'bb_pct', 'atr_pct',
    'ret_3', 'ret_5', 'ret_10', 'vol_ratio', 'trend'
]

# =============================================================================
# TRADE PARAMETERS
# =============================================================================
TP_PCT = 0.06         # 6% take profit
SL_PCT = 0.04         # 4% stop loss (1.5:1 ratio)
TIMEOUT_CANDLES = 15  # Timeout en candles

# =============================================================================
# RISK MANAGEMENT
# =============================================================================
BASE_RISK_PCT = 0.015     # 1.5% base (volatil como DOGE)
MAX_PORTFOLIO_HEAT = 0.04  # 4% max exposure
MAX_POSITIONS = 2          # Max concurrent

# =============================================================================
# VALIDATION RESULTS
# =============================================================================
VALIDATION = {
    'walk_forward': {
        'folds_positive': 11,
        'folds_total': 12,
        'pnl': 458,
        'win_rate': 0.506,
        'trades': 433,
    },
    'cross_altcoin': {
        'DOT': {'trades': 155, 'wr': 0.542, 'pnl': 220},
        'SOL': {'trades': 252, 'wr': 0.528, 'pnl': 322},
        'ATOM': {'trades': 226, 'wr': 0.544, 'pnl': 326},
    },
}

# =============================================================================
# CORRELACIONES CONOCIDAS
# =============================================================================
# ADA-ETH: ~0.75 (smart contract platforms)
# ADA-BTC: ~0.70
# Considerar en gestion de riesgo
