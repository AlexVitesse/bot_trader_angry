# ETH V14 Configuration
# Validated: 2026-02-28
# Status: APPROVED - 3 setups, all walk-forward validated

SYMBOL = "ETH/USDT"
TIMEFRAME = "4h"

# =============================================================================
# RISK MANAGEMENT (diferente a BTC!)
# =============================================================================
BASE_RISK_PCT = 0.02          # 2% base risk per trade
MAX_PORTFOLIO_HEAT = 0.06     # 6% max total exposure
MAX_POSITIONS = 3             # Max concurrent positions

# =============================================================================
# SETUPS APROBADOS - ETH tiene sus propios setups
# =============================================================================
SETUPS = {
    'RSI_OVERSOLD_SHORT': {
        'direction': 'SHORT',
        'condition': 'rsi < 30',
        'tp_pct': 0.05,       # 5% TP
        'sl_pct': 0.025,      # 2.5% SL
        'validation': {
            'folds_positive': 11,
            'folds_total': 12,
            'pnl_total': 630,
            'win_rate': 0.38,
        }
    },
    'VOLUME_SPIKE_UP': {
        'direction': 'LONG',
        'condition': '(volume_ratio > 2) & (close > close_prev)',
        'tp_pct': 0.04,       # 4% TP
        'sl_pct': 0.02,       # 2% SL
        'validation': {
            'folds_positive': 12,
            'folds_total': 12,
            'pnl_total': 338,
            'win_rate': 0.43,
        }
    },
    'VOLUME_SPIKE_DOWN': {
        'direction': 'SHORT',
        'condition': '(volume_ratio > 2) & (close < close_prev)',
        'tp_pct': 0.04,       # 4% TP
        'sl_pct': 0.02,       # 2% SL
        'validation': {
            'folds_positive': 9,
            'folds_total': 12,
            'pnl_total': 262,
            'win_rate': 0.39,
        }
    },
}

# =============================================================================
# DIFERENCIAS VS BTC
# =============================================================================
# - ETH usa TP/SL mas tight (4-5%/2-2.5% vs 6%/3% de BTC)
# - Solo 3 setups (vs 8 de BTC)
# - Mas enfocado en volume spikes
# - RSI oversold -> SHORT (contrario a la intuicion!)

TIMEOUT_CANDLES = 20
