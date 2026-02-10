"""
Test de componentes del Bot
============================
Verifica que todos los modulos funcionen correctamente.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exchange import client
from src.strategy import strategy
from src.trader import trader

print("[TEST] Verificando componentes del bot...")

# Test exchange
ticker = client.get_ticker()
price = float(ticker["price"])
print(f"[OK] Exchange: Precio BTC ${price:,.2f}")

# Test strategy
import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=100, freq="5min")
prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
df = pd.DataFrame({
    "open": prices,
    "high": prices + np.random.rand(100) * 2,
    "low": prices - np.random.rand(100) * 2,
    "close": prices + np.random.randn(100) * 0.5,
    "volume": np.random.rand(100) * 1000
}, index=dates)
df = strategy.calculate_indicators(df)
signal = strategy.check_entry_signal(df)
print(f"[OK] Strategy: Indicadores calculados, senal: {signal.value}")

# Test trader
qty = trader.calculate_position_size(price)
print(f"[OK] Trader: Position size {qty} BTC")

print("")
print("="*50)
print("BOT LISTO PARA OPERAR")
print("="*50)
print("Ejecuta: poetry run python src/bot.py")
