"""
Strategy v6.0 - METRALLADORA (Aggressive Scalper)
=================================================
Estrategia de Alta Frecuencia (Experimental):
  - Timeframe: 1m
  - Leverage: 10x
  - Objetivo: Scalping agresivo en volatilidad.

Logica:
  - Entrada LONG: Precio <= BB_lower + RSI < 25 (Sin filtro de tendencia)
  - Entrada SHORT: Precio >= BB_upper + RSI > 75 (Sin filtro de tendencia)
  - Stop Loss: 1.5 * ATR (Ajustado)
  - Trailing Stop: Activacion 1.5 * ATR, Distancia 0.5 * ATR (Muy rapido)
"""

import sys
from pathlib import Path

# Agregar raiz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pandas_ta as ta
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
from config.settings import (
    BB_LENGTH, BB_STD, EMA_TREND_LENGTH,
    BASE_ORDER_MARGIN, DCA_STEP_PCT, MAX_SAFETY_ORDERS, 
    MARTINGALE_MULTIPLIER, TAKE_PROFIT_PCT, STOP_LOSS_CATASTROPHIC
)


class Signal(Enum):
    """Senales de trading."""
    NONE = "none"
    LONG = "long"
    SHORT = "short"
    CLOSE = "close"


@dataclass
class TradeState:
    """Estado de un trade activo."""
    side: str  # 'long' o 'short'
    avg_price: float
    total_quantity: float
    so_count: int
    trailing_active: bool = False # Legacy, kept for compatibility


class StrategyV6:
    """Estrategia Smart Metralladora v6.7 (Trend Filter + DCA)."""

    def __init__(self):
        self.trade_state: Optional[TradeState] = None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula indicadores optimizados para Scalping 1m (v6.7).
        """
        # EMA Trend Filter
        df['ema_trend'] = ta.ema(df['close'], length=EMA_TREND_LENGTH)

        # Bollinger Bands
        bbands = ta.bbands(df['close'], length=BB_LENGTH, std=BB_STD)
        df['bb_lower'] = bbands.iloc[:, 0]
        df['bb_upper'] = bbands.iloc[:, 2]
        
        # Stochastic RSI
        stoch = ta.stochrsi(df['close'], length=14, rsi_length=14, k=3, d=3)
        df['stoch_k'] = stoch.iloc[:, 0]

        return df

    def check_entry_signal(self, df: pd.DataFrame) -> Signal:
        """
        Verifica senal con Filtro de Tendencia EMA 200 (v6.7).
        """
        if len(df) < EMA_TREND_LENGTH:
            return Signal.NONE

        last = df.iloc[-1]
        prev = df.iloc[-2]
        price = last['close']
        trend_price = last['ema_trend']

        # TENDENCIA ALCISTA -> Solo Longs
        if price > trend_price:
            if prev['close'] <= prev['bb_lower'] and price > last['bb_lower']:
                if last['stoch_k'] < 20:
                    return Signal.LONG

        # TENDENCIA BAJISTA -> Solo Shorts
        elif price < trend_price:
            if prev['close'] >= prev['bb_upper'] and price < last['bb_upper']:
                if last['stoch_k'] > 80:
                    return Signal.SHORT

        return Signal.NONE

    def open_trade(self, side: str, entry_price: float, quantity: float):
        """Registra la apertura de un trade Grinder."""
        self.trade_state = TradeState(
            side=side,
            avg_price=entry_price,
            total_quantity=quantity,
            so_count=0
        )
        return self.trade_state

    def update_trade(self, high: float, low: float, close: float) -> Optional[Signal]:
        """Legacy method - functionality moved to Trader for DCA management."""
        return None

    def close_trade(self) -> Optional[TradeState]:
        """Cierra el trade actual."""
        final_state = self.trade_state
        self.trade_state = None
        return final_state

    def get_exit_reason(self) -> str:
        return "SIGNAL"

    def has_open_trade(self) -> bool:
        return self.trade_state is not None

    def get_current_stop(self) -> Optional[float]:
        if not self.trade_state: return None
        # Retornamos el SL catastrofico segun el lado
        if self.trade_state.side == 'long':
            return self.trade_state.avg_price * (1 - STOP_LOSS_CATASTROPHIC)
        return self.trade_state.avg_price * (1 + STOP_LOSS_CATASTROPHIC)



# Instancia global
strategy = StrategyV6()


if __name__ == "__main__":
    print("[TEST] Probando estrategia v6.0 (Metralladora)...")

    # Crear datos de prueba
    import numpy as np

    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=300, freq='5min')
    prices = 100 + np.cumsum(np.random.randn(300) * 0.5)

    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.rand(300) * 2,
        'low': prices - np.random.rand(300) * 2,
        'close': prices + np.random.randn(300) * 0.5,
        'volume': np.random.rand(300) * 1000
    }, index=dates)

    # Calcular indicadores
    df = strategy.calculate_indicators(df)
    print(f"[OK] Indicadores calculados")
    print(f"     Ultima vela:")
    print(f"     - Close: {df['close'].iloc[-1]:.2f}")
    print(f"     - RSI: {df['rsi'].iloc[-1]:.2f}")
    print(f"     - ATR: {df['atr'].iloc[-1]:.2f}")
    print(f"     - BB Lower/Upper: {df['bb_lower'].iloc[-1]:.2f} / {df['bb_upper'].iloc[-1]:.2f}")

    # Verificar senal
    signal = strategy.check_entry_signal(df)
    print(f"[OK] Senal de entrada: {signal.value}")
