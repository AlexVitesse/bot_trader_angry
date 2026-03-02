# Multi-Expert Trading Strategies
# Each expert is a validated ensemble for a specific coin

"""
APPROVED EXPERTS (ready for production):
- btc_v14: BTC V14 - 8 setups técnicos + ML confianza
- eth_v14: ETH V14 - 3 setups técnicos + ML confianza
- doge_v14: DOGE V14 - ML Ensemble Voting (diferente arquitectura)

PENDING EXPERTS (to be developed):
- ada_v14: ADA Expert (planned)
- dot_v14: DOT Expert (planned)

ARQUITECTURAS:
- BTC/ETH: Reglas técnicas (setups) + ML para confianza
- DOGE: ML Ensemble Voting puro (3 modelos votan)
"""

EXPERTS = {
    'btc_v14': {
        'status': 'APPROVED',
        'symbol': 'BTC/USDT',
        'validated': '2026-02-28',
        'pnl_backtest': '+1126%',
        'cross_validation': '6/6 positive',
        'architecture': 'SETUPS + ML_CONFIDENCE',
        'setups': 8,
    },
    'eth_v14': {
        'status': 'APPROVED',
        'symbol': 'ETH/USDT',
        'validated': '2026-02-28',
        'pnl_backtest': '+1230%',
        'walk_forward': '32/36 folds positive',
        'architecture': 'SETUPS + ML_CONFIDENCE',
        'setups': 3,
    },
    'doge_v14': {
        'status': 'APPROVED',
        'symbol': 'DOGE/USDT',
        'validated': '2026-02-28',
        'pnl_backtest': '+414%',
        'cross_memecoin': 'SHIB +194%, PEPE +192%',
        'architecture': 'ML_ENSEMBLE_VOTING',
        'note': 'Diferente arquitectura - memecoins no siguen técnicos',
    },
    # New experts (just trained)
    'ada_v14': {
        'status': 'APPROVED',
        'symbol': 'ADA/USDT',
        'validated': '2026-02-28',
        'pnl_backtest': '+458%',
        'walk_forward': '11/12 folds positive',
        'cross_validation': 'DOT +220%, SOL +322%, ATOM +326%',
        'architecture': 'ML_ENSEMBLE_VOTING',
    },
    'dot_v14': {
        'status': 'APPROVED',
        'symbol': 'DOT/USDT',
        'validated': '2026-02-28',
        'pnl_backtest': '+77%',
        'walk_forward': '6/8 folds positive',
        'architecture': 'ML_ENSEMBLE_VOTING',
        'note': 'Less history (since 2020), simplified 2-model ensemble',
    },
}

def get_active_experts():
    """Return list of approved experts ready for production"""
    return [k for k, v in EXPERTS.items() if v.get('status') == 'APPROVED']

def get_expert_info(expert_name):
    """Get info about a specific expert"""
    return EXPERTS.get(expert_name, None)


# Correlation Risk Management
from .correlation_risk import (
    can_open_position,
    adjust_signal_size,
    detect_market_regime,
    get_regime_adjustments,
    calculate_diversification_score,
    CORRELATIONS,
    MAX_PORTFOLIO_HEAT,
    MAX_CORRELATED_HEAT,
)
