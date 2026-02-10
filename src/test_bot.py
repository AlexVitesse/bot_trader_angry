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
dates = pd.date_range("2024-01-01", periods=300, freq="1min")
prices = 100 + np.cumsum(np.random.randn(300) * 0.5)
df = pd.DataFrame({
    "open": prices,
    "high": prices + np.random.rand(300) * 2,
    "low": prices - np.random.rand(300) * 2,
    "close": prices + np.random.randn(300) * 0.5,
    "volume": np.random.rand(300) * 1000
}, index=dates)
df = strategy.calculate_indicators(df)
signal = strategy.check_entry_signal(df)
print(f"[OK] Strategy: Indicadores calculados, senal: {signal.value}")

# Test trader
qty = trader.calculate_base_quantity(price)
print(f"[OK] Trader: Base quantity {qty} BTC (${qty * price:,.2f} notional)")

# Test balance
balance = client.get_usdt_balance()
print(f"[OK] Balance USDT: ${balance:,.2f}")

print("")
print("="*50)
print("BOT LISTO PARA OPERAR")
print("="*50)
print("Ejecuta: poetry run python src/bot.py")
