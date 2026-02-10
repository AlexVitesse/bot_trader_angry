"""
Strategy v6.7 - SMART METRALLADORA (Trend Filter + DCA)
========================================================
Estrategia de Alta Frecuencia:
  - Timeframe: 1m
  - Leverage: 10x

Logica:
  - Entrada LONG: Precio > EMA 200 + rebote BB Lower + StochRSI < 20
  - Entrada SHORT: Precio < EMA 200 + rechazo BB Upper + StochRSI > 80
  - Gestion: DCA con Martingale (max 2 Safety Orders)
  - TP: 0.6% desde promedio | SL: 1.5% desde promedio
"""

import sys
from pathlib import Path

# Agregar raiz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pandas_ta as ta
from enum import Enum
from config.settings import (
    BB_LENGTH, BB_STD, EMA_TREND_LENGTH,
    ATR_REGIME_LENGTH, ATR_REGIME_MULT_HIGH, ATR_REGIME_MULT_LOW
)


class Signal(Enum):
    """Senales de trading."""
    NONE = "none"
    LONG = "long"
    SHORT = "short"
    CLOSE = "close"


class StrategyV6:
    """Estrategia Smart Metralladora v6.7 (Trend Filter + DCA).

    Solo se encarga de:
    1. Calcular indicadores tecnicos
    2. Generar senales de entrada
    El estado del trade y gestion de posicion es responsabilidad del Trader.
    """

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

        # ATR para filtro de regimen de volatilidad
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=ATR_REGIME_LENGTH)
        df['atr_sma'] = df['atr'].rolling(window=100).mean()

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

        # Verificar que los indicadores no sean NaN
        if pd.isna(trend_price) or pd.isna(last['bb_lower']) or pd.isna(last['stoch_k']):
            return Signal.NONE

        # Filtro de regimen de volatilidad
        if 'atr' in last.index and 'atr_sma' in last.index:
            if not pd.isna(last['atr']) and not pd.isna(last['atr_sma']) and last['atr_sma'] > 0:
                atr_ratio = last['atr'] / last['atr_sma']
                if atr_ratio > ATR_REGIME_MULT_HIGH:
                    return Signal.NONE  # Volatilidad extrema - no operar
                if atr_ratio < ATR_REGIME_MULT_LOW:
                    return Signal.NONE  # Mercado muerto - no operar

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


# Instancia global
strategy = StrategyV6()


if __name__ == "__main__":
    print("[TEST] Probando estrategia v6.7 (Smart Metralladora)...")

    import numpy as np

    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=300, freq='1min')
    prices = 100 + np.cumsum(np.random.randn(300) * 0.5)

    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.rand(300) * 2,
        'low': prices - np.random.rand(300) * 2,
        'close': prices + np.random.randn(300) * 0.5,
        'volume': np.random.rand(300) * 1000
    }, index=dates)

    df = strategy.calculate_indicators(df)
    print(f"[OK] Indicadores calculados")
    print(f"     Ultima vela:")
    print(f"     - Close: {df['close'].iloc[-1]:.2f}")
    print(f"     - EMA 200: {df['ema_trend'].iloc[-1]:.2f}")
    print(f"     - StochK: {df['stoch_k'].iloc[-1]:.2f}")
    print(f"     - BB Lower/Upper: {df['bb_lower'].iloc[-1]:.2f} / {df['bb_upper'].iloc[-1]:.2f}")

    signal = strategy.check_entry_signal(df)
    print(f"[OK] Senal de entrada: {signal.value}")
