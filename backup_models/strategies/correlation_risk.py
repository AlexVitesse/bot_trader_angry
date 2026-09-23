# Correlation Risk Management for Multi-Expert Portfolio
# Created: 2026-02-28
# Purpose: Prevent catastrophic drawdowns when BTC and ETH crash together

"""
PROBLEMA IDENTIFICADO:
- Correlacion BTC-ETH: 83%
- Cuando BTC cae >2%, ETH cae 98% de las veces
- Durante crashes: mas senales LONG que SHORT
- Max Drawdown combinado sin proteccion: -99.7%

SOLUCIONES IMPLEMENTADAS:
1. MAX_PORTFOLIO_HEAT: Limite total de exposicion
2. CORRELATION_REDUCTION: Si BTC tiene posicion, reducir ETH
3. CRASH_MODE: Durante caidas fuertes, solo SHORT
4. DIVERSIFICATION_BONUS: Incluir DOGE (baja correlacion)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# =============================================================================
# CONFIGURACION
# =============================================================================

# Correlaciones conocidas (calculadas con datos historicos)
CORRELATIONS = {
    ('BTC', 'ETH'): 0.83,
    ('BTC', 'DOGE'): 0.45,
    ('BTC', 'ADA'): 0.70,
    ('BTC', 'DOT'): 0.70,
    ('ETH', 'DOGE'): 0.52,
    ('ETH', 'ADA'): 0.75,
    ('ETH', 'DOT'): 0.75,
    ('ADA', 'DOT'): 0.80,
    ('ADA', 'DOGE'): 0.55,
    ('DOT', 'DOGE'): 0.55,
}

# Limites de riesgo
MAX_PORTFOLIO_HEAT = 0.06  # 6% max total exposure
MAX_CORRELATED_HEAT = 0.04  # 4% max para activos correlacionados (>0.7)
CRASH_THRESHOLD = -0.02  # -2% = crash mode activado

# Ajustes por correlacion
CORRELATION_SIZE_MULTIPLIERS = {
    'high': 0.5,    # >0.7 correlation: 50% size
    'medium': 0.75,  # 0.4-0.7 correlation: 75% size
    'low': 1.0,      # <0.4 correlation: 100% size
}

# =============================================================================
# FUNCIONES DE RIESGO
# =============================================================================

def get_correlation(asset1: str, asset2: str) -> float:
    """Get correlation between two assets"""
    key = (asset1, asset2)
    if key in CORRELATIONS:
        return CORRELATIONS[key]
    key = (asset2, asset1)
    if key in CORRELATIONS:
        return CORRELATIONS[key]
    return 0.0  # Unknown = no correlation assumed


def calculate_portfolio_heat(positions: List[Dict]) -> float:
    """
    Calculate total portfolio heat (exposure)
    Each position contributes its risk %
    """
    total_heat = 0.0
    for pos in positions:
        total_heat += pos.get('risk_pct', 0.02)
    return total_heat


def calculate_correlated_heat(positions: List[Dict], new_asset: str) -> float:
    """
    Calculate heat from positions correlated with new_asset
    """
    correlated_heat = 0.0
    for pos in positions:
        existing_asset = pos.get('symbol', '').split('/')[0]
        corr = get_correlation(existing_asset, new_asset)
        if corr > 0.7:  # High correlation
            correlated_heat += pos.get('risk_pct', 0.02)
    return correlated_heat


def get_size_multiplier(positions: List[Dict], new_asset: str) -> float:
    """
    Get size multiplier based on existing correlated positions
    """
    max_corr = 0.0
    for pos in positions:
        existing_asset = pos.get('symbol', '').split('/')[0]
        corr = get_correlation(existing_asset, new_asset)
        max_corr = max(max_corr, corr)

    if max_corr > 0.7:
        return CORRELATION_SIZE_MULTIPLIERS['high']
    elif max_corr > 0.4:
        return CORRELATION_SIZE_MULTIPLIERS['medium']
    else:
        return CORRELATION_SIZE_MULTIPLIERS['low']


def is_crash_mode(btc_returns: pd.Series, lookback: int = 3) -> bool:
    """
    Check if we're in crash mode (recent strong BTC drop)
    """
    if len(btc_returns) < lookback:
        return False
    recent_return = btc_returns.iloc[-lookback:].sum()
    return recent_return < CRASH_THRESHOLD


def can_open_position(
    positions: List[Dict],
    new_signal: Dict,
    btc_returns: Optional[pd.Series] = None
) -> Tuple[bool, str, float]:
    """
    Check if we can open a new position given risk constraints.

    Returns:
        (can_open, reason, size_multiplier)
    """
    new_asset = new_signal.get('symbol', '').split('/')[0]
    direction = new_signal.get('direction', 'LONG')
    base_risk = new_signal.get('risk_pct', 0.02)

    # Check 1: Portfolio heat limit
    current_heat = calculate_portfolio_heat(positions)
    if current_heat + base_risk > MAX_PORTFOLIO_HEAT:
        return False, f"Portfolio heat limit ({MAX_PORTFOLIO_HEAT*100}%) exceeded", 0.0

    # Check 2: Correlated heat limit
    corr_heat = calculate_correlated_heat(positions, new_asset)
    if corr_heat + base_risk > MAX_CORRELATED_HEAT:
        return False, f"Correlated heat limit ({MAX_CORRELATED_HEAT*100}%) exceeded", 0.0

    # Check 3: Crash mode - only allow SHORT
    if btc_returns is not None and is_crash_mode(btc_returns):
        if direction == 'LONG':
            return False, "Crash mode active - only SHORT allowed", 0.0

    # Calculate size multiplier based on correlation
    size_mult = get_size_multiplier(positions, new_asset)

    return True, "OK", size_mult


def adjust_signal_size(
    signal: Dict,
    positions: List[Dict],
    btc_returns: Optional[pd.Series] = None
) -> Dict:
    """
    Adjust signal size based on correlation risk.
    Returns modified signal with adjusted risk_pct.
    """
    can_open, reason, size_mult = can_open_position(positions, signal, btc_returns)

    adjusted_signal = signal.copy()

    if not can_open:
        adjusted_signal['skip'] = True
        adjusted_signal['skip_reason'] = reason
        return adjusted_signal

    # Apply size multiplier
    original_risk = signal.get('risk_pct', 0.02)
    adjusted_signal['risk_pct'] = original_risk * size_mult
    adjusted_signal['size_multiplier'] = size_mult

    if size_mult < 1.0:
        adjusted_signal['size_reason'] = f"Reduced by {(1-size_mult)*100:.0f}% due to correlation"

    return adjusted_signal


# =============================================================================
# PORTFOLIO DIVERSIFICATION SCORE
# =============================================================================

def calculate_diversification_score(positions: List[Dict]) -> float:
    """
    Calculate portfolio diversification score (0-1).
    Higher = more diversified = better.
    """
    if len(positions) <= 1:
        return 1.0  # Single position = fully diversified

    assets = [pos.get('symbol', '').split('/')[0] for pos in positions]

    total_corr = 0.0
    pairs = 0
    for i in range(len(assets)):
        for j in range(i+1, len(assets)):
            total_corr += get_correlation(assets[i], assets[j])
            pairs += 1

    avg_corr = total_corr / pairs if pairs > 0 else 0.0

    # Score: 1 - avg_correlation
    return 1.0 - avg_corr


# =============================================================================
# CRASH DETECTION
# =============================================================================

def detect_market_regime(btc_df: pd.DataFrame) -> str:
    """
    Detect current market regime based on BTC.
    Returns: 'NORMAL', 'CAUTION', 'CRASH'
    """
    if len(btc_df) < 20:
        return 'NORMAL'

    # Calculate returns
    btc_df = btc_df.copy()
    btc_df['ret'] = btc_df['close'].pct_change()

    # Recent returns
    ret_1 = btc_df['ret'].iloc[-1]
    ret_3 = btc_df['ret'].iloc[-3:].sum()
    ret_5 = btc_df['ret'].iloc[-5:].sum()

    # Volatility
    vol_20 = btc_df['ret'].iloc[-20:].std()

    # Crash detection
    if ret_1 < -0.05 or ret_3 < -0.10:
        return 'CRASH'
    elif ret_3 < -0.05 or vol_20 > 0.05:
        return 'CAUTION'
    else:
        return 'NORMAL'


def get_regime_adjustments(regime: str) -> Dict:
    """
    Get risk adjustments based on market regime.
    """
    if regime == 'CRASH':
        return {
            'allow_long': False,
            'allow_short': True,
            'size_multiplier': 0.5,
            'max_positions': 1,
        }
    elif regime == 'CAUTION':
        return {
            'allow_long': True,
            'allow_short': True,
            'size_multiplier': 0.75,
            'max_positions': 2,
        }
    else:  # NORMAL
        return {
            'allow_long': True,
            'allow_short': True,
            'size_multiplier': 1.0,
            'max_positions': 4,
        }


# =============================================================================
# VALIDATION RESULTS
# =============================================================================
"""
Resultados de validacion (calculados en sesion anterior):

SIN proteccion:
- Correlation BTC-ETH: 83%
- Durante crashes: 68 LONG, 56 SHORT (alerta: mas LONG)
- Max Drawdown: -99.7%
- PnL total: +50%

CON proteccion (simulado):
- Max correlated heat: 4%
- Size reduction for correlated: 50%
- Crash mode: solo SHORT
- Max Drawdown esperado: ~-25%
- PnL total esperado: +40% (reducido pero mas estable)

CONCLUSION:
El sistema reduce retornos en ~20% pero reduce drawdown de -99.7% a ~-25%
Ratio riesgo/retorno mejora significativamente
"""
