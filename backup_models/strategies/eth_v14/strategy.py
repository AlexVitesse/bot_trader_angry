"""
ETH V14 Strategy
Validated: 2026-02-28
Status: APPROVED - 5/6 scenarios positive (cross-asset + synthetic)
"""

import pandas as pd
import ta

from .config import SETUPS, TIMEOUT_CANDLES

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add required indicators for ETH setups"""
    df = df.copy()
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    df['close_prev'] = df['close'].shift(1)
    return df.dropna()


def detect_setup(df: pd.DataFrame, idx: int) -> list:
    """
    Detect which setups are active at given index.
    Returns list of (setup_name, direction, tp, sl)
    """
    row = df.iloc[idx]
    active_setups = []

    # RSI_OVERSOLD_SHORT
    if row['rsi'] < 30:
        setup = SETUPS['RSI_OVERSOLD_SHORT']
        active_setups.append((
            'RSI_OVERSOLD_SHORT',
            setup['direction'],
            setup['tp_pct'],
            setup['sl_pct']
        ))

    # VOLUME_SPIKE_UP
    if row['volume_ratio'] > 2 and row['close'] > row['close_prev']:
        setup = SETUPS['VOLUME_SPIKE_UP']
        active_setups.append((
            'VOLUME_SPIKE_UP',
            setup['direction'],
            setup['tp_pct'],
            setup['sl_pct']
        ))

    # VOLUME_SPIKE_DOWN
    if row['volume_ratio'] > 2 and row['close'] < row['close_prev']:
        setup = SETUPS['VOLUME_SPIKE_DOWN']
        active_setups.append((
            'VOLUME_SPIKE_DOWN',
            setup['direction'],
            setup['tp_pct'],
            setup['sl_pct']
        ))

    return active_setups


def generate_signals(df: pd.DataFrame) -> list:
    """
    Generate trading signals from dataframe.
    Returns list of signals with entry details.
    """
    df = add_indicators(df)
    signals = []

    for i in range(len(df) - TIMEOUT_CANDLES):
        setups = detect_setup(df, i)

        for setup_name, direction, tp, sl in setups:
            signals.append({
                'timestamp': df['timestamp'].iloc[i],
                'setup': setup_name,
                'direction': direction,
                'entry_price': df['close'].iloc[i],
                'tp_pct': tp,
                'sl_pct': sl,
            })

    return signals
