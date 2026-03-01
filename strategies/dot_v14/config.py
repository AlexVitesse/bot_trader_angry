# DOT V14 Configuration
# Validated: 2026-02-28
# Status: APPROVED - Ensemble ML Voting
# Walk-Forward: 6/8 folds positive, +77% PnL

SYMBOL = "DOT/USDT"
TIMEFRAME = "4h"

# =============================================================================
# ARQUITECTURA: ML ENSEMBLE VOTING (simplificado)
# =============================================================================
# DOT usa solo 2 modelos (RF + GB) porque tiene menos historia
# Menos complejo = menos overfitting

MODEL_TYPE = "ENSEMBLE_VOTING"
MODELS = ["RandomForest", "GradientBoosting"]  # Solo 2 modelos
VOTING_THRESHOLD = 2  # Ambos deben coincidir

# =============================================================================
# FEATURES PARA ML
# =============================================================================
FEATURE_COLS = [
    'rsi', 'macd_norm', 'adx', 'bb_pct', 'atr_pct',
    'ret_3', 'ret_5', 'ret_10', 'vol_ratio', 'trend'
]

# =============================================================================
# TRADE PARAMETERS (optimizados para DOT)
# =============================================================================
TP_PCT = 0.05         # 5% take profit (mas conservador)
SL_PCT = 0.03         # 3% stop loss
TIMEOUT_CANDLES = 15  # Timeout en candles

# =============================================================================
# RISK MANAGEMENT
# =============================================================================
BASE_RISK_PCT = 0.015     # 1.5% base
MAX_PORTFOLIO_HEAT = 0.04  # 4% max exposure
MAX_POSITIONS = 2          # Max concurrent

# =============================================================================
# VALIDATION RESULTS
# =============================================================================
VALIDATION = {
    'walk_forward': {
        'folds_positive': 6,
        'folds_total': 8,
        'pnl': 77,
    },
    'note': 'DOT tiene menos historia (desde 2020)',
}

# =============================================================================
# CORRELACIONES CONOCIDAS
# =============================================================================
# DOT-ADA: ~0.80 (ambos ecosistema Polkadot/smart contracts)
# DOT-ETH: ~0.75
# DOT-BTC: ~0.70
