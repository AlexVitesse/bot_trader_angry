"""
Test de Robustez - Edge Cases y Errores Criticos
==================================================
Verifica que el bot NO crashea ante respuestas inesperadas
de la API de Binance o datos corruptos en la DB.

Ejecutar: poetry run python tests/test_robustness.py
"""

import sys
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

# Agregar raiz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# =====================================================================
# RESULTADOS
# =====================================================================
passed = 0
failed = 0
errors = []


def test(name):
    """Decorator para registrar tests."""
    def wrapper(fn):
        global passed, failed
        try:
            fn()
            passed += 1
            print(f"  [PASS] {name}")
        except AssertionError as e:
            failed += 1
            errors.append(f"{name}: {e}")
            print(f"  [FAIL] {name}: {e}")
        except Exception as e:
            failed += 1
            errors.append(f"{name}: EXCEPTION {e}")
            print(f"  [FAIL] {name}: EXCEPTION {type(e).__name__}: {e}")
    return wrapper


# =====================================================================
# 1. TEST _extract_fill_price (Trader)
# =====================================================================
print("\n" + "="*60)
print("1. _extract_fill_price - Extraccion de precio de llenado")
print("="*60)

from src.trader import Trader


@test("avgPrice valido retorna el precio")
def _():
    order = {'avgPrice': '97500.50', 'fills': []}
    result = Trader._extract_fill_price(order, 99000.0)
    assert result == 97500.50, f"Esperado 97500.50, obtenido {result}"


@test("avgPrice='0' con fills calcula desde fills")
def _():
    order = {
        'avgPrice': '0',
        'fills': [
            {'qty': '0.001', 'price': '97000.0'},
            {'qty': '0.001', 'price': '97100.0'}
        ]
    }
    result = Trader._extract_fill_price(order, 99000.0)
    expected = (0.001 * 97000.0 + 0.001 * 97100.0) / 0.002
    assert abs(result - expected) < 0.01, f"Esperado {expected}, obtenido {result}"


@test("avgPrice='0' sin fills usa fallback")
def _():
    order = {'avgPrice': '0', 'fills': []}
    result = Trader._extract_fill_price(order, 98000.0)
    assert result == 98000.0, f"Esperado 98000.0, obtenido {result}"


@test("avgPrice='0.00000000' sin fills usa fallback")
def _():
    order = {'avgPrice': '0.00000000'}
    result = Trader._extract_fill_price(order, 98000.0)
    assert result == 98000.0, f"Esperado 98000.0, obtenido {result}"


@test("avgPrice ausente usa fallback")
def _():
    order = {'orderId': 12345}
    result = Trader._extract_fill_price(order, 97000.0)
    assert result == 97000.0, f"Esperado 97000.0, obtenido {result}"


@test("avgPrice negativo usa fallback")
def _():
    order = {'avgPrice': '-1'}
    result = Trader._extract_fill_price(order, 97000.0)
    assert result == 97000.0, f"Esperado 97000.0, obtenido {result}"


@test("fills con qty=0 usa fallback")
def _():
    order = {
        'avgPrice': '0',
        'fills': [{'qty': '0', 'price': '97000.0'}]
    }
    result = Trader._extract_fill_price(order, 98500.0)
    assert result == 98500.0, f"Esperado 98500.0, obtenido {result}"


@test("order dict vacio usa fallback")
def _():
    result = Trader._extract_fill_price({}, 96000.0)
    assert result == 96000.0, f"Esperado 96000.0, obtenido {result}"


# =====================================================================
# 2. TEST calculate_base_quantity (Division por cero)
# =====================================================================
print("\n" + "="*60)
print("2. calculate_base_quantity - Proteccion division por cero")
print("="*60)

# Crear un trader mock que no conecta al exchange
with patch('src.trader.client') as mock_client, \
     patch('src.trader.db') as mock_db:
    mock_client.set_leverage.return_value = {}
    mock_db.get_recent_trades.return_value = []
    mock_db.get_trade_count.return_value = 0
    test_trader = Trader()


@test("Precio normal calcula cantidad correcta")
def _():
    qty = test_trader.calculate_base_quantity(97000.0)
    expected = round((12.0 * 10) / 97000.0, 3)
    assert qty == expected, f"Esperado {expected}, obtenido {qty}"


@test("Precio 0 retorna 0 sin crashear")
def _():
    qty = test_trader.calculate_base_quantity(0.0)
    assert qty == 0.0, f"Esperado 0.0, obtenido {qty}"


@test("Precio negativo retorna 0 sin crashear")
def _():
    qty = test_trader.calculate_base_quantity(-100.0)
    assert qty == 0.0, f"Esperado 0.0, obtenido {qty}"


# =====================================================================
# 3. TEST Ciclo completo: Open -> Update -> Close (con mocks)
# =====================================================================
print("\n" + "="*60)
print("3. Ciclo completo de trade con respuestas problematicas")
print("="*60)

from src.strategy import Signal


@test("Open con avgPrice=0 usa current_price como fallback")
def _():
    with patch('src.trader.client') as mock_client, \
         patch('src.trader.db') as mock_db:
        mock_client.set_leverage.return_value = {}
        mock_db.get_recent_trades.return_value = []
        mock_db.get_trade_count.return_value = 0
        mock_db.save_active_position.return_value = None
        mock_client.market_buy.return_value = {'avgPrice': '0', 'fills': []}

        t = Trader()
        success = t.open_position(Signal.LONG, 97500.0, 0)

        assert success is True, "open_position debio retornar True"
        assert t.current_trade is not None, "current_trade no deberia ser None"
        assert t.current_trade.avg_price == 97500.0, \
            f"avg_price debio ser 97500.0 (fallback), es {t.current_trade.avg_price}"


@test("Open con avgPrice=0 y fills usa precio de fills")
def _():
    with patch('src.trader.client') as mock_client, \
         patch('src.trader.db') as mock_db:
        mock_client.set_leverage.return_value = {}
        mock_db.get_recent_trades.return_value = []
        mock_db.get_trade_count.return_value = 0
        mock_db.save_active_position.return_value = None
        mock_client.market_buy.return_value = {
            'avgPrice': '0',
            'fills': [{'qty': '0.001', 'price': '97200.0'}]
        }

        t = Trader()
        success = t.open_position(Signal.LONG, 97500.0, 0)

        assert success is True
        assert t.current_trade.avg_price == 97200.0, \
            f"avg_price debio ser 97200.0 (from fills), es {t.current_trade.avg_price}"


@test("Close con avg_price=0 no crashea (proteccion ZeroDivision)")
def _():
    with patch('src.trader.client') as mock_client, \
         patch('src.trader.db') as mock_db:
        mock_client.set_leverage.return_value = {}
        mock_db.get_recent_trades.return_value = []
        mock_db.get_trade_count.return_value = 0
        mock_db.save_trade.return_value = 1
        mock_db.update_feature_outcome.return_value = None
        mock_db.clear_active_position.return_value = None
        mock_client.close_position.return_value = {'avgPrice': '97000.0'}

        t = Trader()
        # Forzar posicion con avg_price=0 (simula el bug original)
        from src.trader import TradeRecord
        t.current_trade = TradeRecord(
            entry_time=datetime.now(),
            side='long',
            avg_price=0.0,
            total_quantity=0.002,
            safety_orders_count=0
        )

        # Esto NO debe crashear
        result = t.close_position(97000.0, "TEST")
        assert result is True, "close_position debio retornar True"
        assert t.current_trade is None, "current_trade deberia ser None tras cerrar"


@test("Update position con avg_price=0 no genera TP/SL invalido")
def _():
    with patch('src.trader.client') as mock_client, \
         patch('src.trader.db') as mock_db:
        mock_client.set_leverage.return_value = {}
        mock_db.get_recent_trades.return_value = []
        mock_db.get_trade_count.return_value = 0
        mock_db.save_trade.return_value = 1
        mock_db.update_feature_outcome.return_value = None
        mock_db.clear_active_position.return_value = None
        mock_client.close_position.return_value = {'avgPrice': '97000.0'}

        t = Trader()
        from src.trader import TradeRecord
        t.current_trade = TradeRecord(
            entry_time=datetime.now(),
            side='long',
            avg_price=0.0,  # Bug: avg_price corrupto
            total_quantity=0.002
        )

        # Con avg_price=0, TP=0*1.006=0, asi que high>=0 siempre es True
        # Esto llama close_position con tp_price=0, lo cual podria crashear
        # Verificar que NO crashea
        result = t.update_position(high=97100.0, low=96900.0, close=97000.0)
        # Deberia cerrar via TP (high >= 0.0 * 1.006 = 0.0)
        assert t.current_trade is None, "Position deberia haberse cerrado"


@test("DCA con current_price=0 no crashea")
def _():
    with patch('src.trader.client') as mock_client, \
         patch('src.trader.db') as mock_db:
        mock_client.set_leverage.return_value = {}
        mock_db.get_recent_trades.return_value = []
        mock_db.get_trade_count.return_value = 0

        t = Trader()
        from src.trader import TradeRecord
        t.current_trade = TradeRecord(
            entry_time=datetime.now(),
            side='long',
            avg_price=97000.0,
            total_quantity=0.002
        )

        result = t.execute_dca(0.0)
        assert result is False, "DCA con price=0 debio retornar False"


@test("Open position con price=0 no abre (quantity seria 0)")
def _():
    with patch('src.trader.client') as mock_client, \
         patch('src.trader.db') as mock_db:
        mock_client.set_leverage.return_value = {}
        mock_db.get_recent_trades.return_value = []
        mock_db.get_trade_count.return_value = 0

        t = Trader()
        result = t.open_position(Signal.LONG, 0.0, 0)
        assert result is False, "open_position con price=0 debio retornar False"
        assert t.current_trade is None, "No deberia haber trade abierto"


# =====================================================================
# 4. TEST Database CRUD + Edge Cases
# =====================================================================
print("\n" + "="*60)
print("4. Database - CRUD y edge cases")
print("="*60)

from src.database import Database


@test("DB inicializa correctamente en archivo temporal")
def _():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        test_db = Database(db_path=Path(f.name))
        assert test_db.get_trade_count() == 0
        assert test_db.get_candle_count() == 0
        assert test_db.get_active_position() is None


@test("DB save_trade y get_recent_trades funciona")
def _():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        test_db = Database(db_path=Path(f.name))
        trade_id = test_db.save_trade({
            'entry_time': '2026-02-10T12:00:00',
            'exit_time': '2026-02-10T12:05:00',
            'side': 'long',
            'entry_price': 97000.0,
            'exit_price': 97500.0,
            'avg_price': 97000.0,
            'total_quantity': 0.002,
            'safety_orders_count': 0,
            'pnl': 1.0,
            'pnl_pct': 0.05,
            'exit_reason': 'TAKE_PROFIT_GRINDER',
            'commission': 0.05
        })
        assert trade_id == 1, f"trade_id debio ser 1, es {trade_id}"
        assert test_db.get_trade_count() == 1

        trades = test_db.get_recent_trades(5)
        assert len(trades) == 1
        assert trades[0]['side'] == 'long'
        assert trades[0]['pnl'] == 1.0


@test("DB active_position CRUD completo")
def _():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        test_db = Database(db_path=Path(f.name))

        # Save
        test_db.save_active_position({
            'entry_time': '2026-02-10T12:00:00',
            'side': 'long',
            'avg_price': 97000.0,
            'total_quantity': 0.002,
            'safety_orders_count': 1,
            'commission': 0.05
        })

        # Read
        pos = test_db.get_active_position()
        assert pos is not None
        assert pos['side'] == 'long'
        assert pos['avg_price'] == 97000.0
        assert pos['safety_orders_count'] == 1

        # Update (replace)
        test_db.save_active_position({
            'entry_time': '2026-02-10T12:00:00',
            'side': 'long',
            'avg_price': 96800.0,
            'total_quantity': 0.006,
            'safety_orders_count': 2,
            'commission': 0.15
        })

        pos2 = test_db.get_active_position()
        assert pos2['avg_price'] == 96800.0
        assert pos2['safety_orders_count'] == 2

        # Delete
        test_db.clear_active_position()
        assert test_db.get_active_position() is None


@test("DB save_candle ignora duplicados sin error")
def _():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        test_db = Database(db_path=Path(f.name))
        candle = {
            'timestamp': '2026-02-10T12:00:00',
            'open': 97000.0, 'high': 97100.0,
            'low': 96900.0, 'close': 97050.0,
            'volume': 100.0
        }
        test_db.save_candle(candle)
        test_db.save_candle(candle)  # Duplicado - no debe crashear
        assert test_db.get_candle_count() == 1


@test("DB features save y training data query")
def _():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        test_db = Database(db_path=Path(f.name))

        # Guardar trade primero
        trade_id = test_db.save_trade({
            'entry_time': '2026-02-10T12:00:00',
            'exit_time': '2026-02-10T12:05:00',
            'side': 'long',
            'entry_price': 97000.0,
            'exit_price': 97500.0,
            'avg_price': 97000.0,
            'total_quantity': 0.002,
            'pnl': 1.0, 'pnl_pct': 0.05,
            'exit_reason': 'TP',
            'commission': 0.05
        })

        # Guardar features
        features = {
            'timestamp': '2026-02-10T12:00:00',
            'price': 97000.0,
            'ema_200': 96500.0,
            'ema_dist_pct': 0.5,
            'bb_lower': 96800.0,
            'bb_upper': 97200.0,
            'bb_position': 0.5,
            'stoch_k': 15.0,
            'atr': 50.0,
            'atr_sma': 45.0,
            'atr_ratio': 1.11,
            'volume': 100.0,
            'volume_sma_20': 90.0,
            'volume_relative': 1.11,
            'spread': 200.0,
            'spread_relative': 1.05,
            'hour_utc': 12,
            'day_of_week': 1,
            'return_5': 0.1,
            'return_15': 0.3,
            'return_60': -0.2,
            'consecutive_wins': 3,
            'consecutive_losses': 0,
            'outcome': ''
        }
        test_db.save_features(features, trade_id=trade_id)

        # Sin outcome, no aparece en training data
        training = test_db.get_training_data()
        assert len(training) == 0, "No deberia haber datos sin outcome"

        # Actualizar outcome
        test_db.update_feature_outcome(trade_id, 'WIN')
        training = test_db.get_training_data()
        assert len(training) == 1
        assert training[0]['outcome'] == 'WIN'
        assert training[0]['pnl'] == 1.0


@test("DB get_daily_stats funciona con trades")
def _():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        test_db = Database(db_path=Path(f.name))

        test_db.save_trade({
            'entry_time': '2026-02-10T12:00:00',
            'exit_time': '2026-02-10T12:05:00',
            'side': 'long',
            'entry_price': 97000.0, 'exit_price': 97500.0,
            'avg_price': 97000.0, 'total_quantity': 0.002,
            'pnl': 1.5, 'pnl_pct': 0.05,
            'exit_reason': 'TP', 'commission': 0.05
        })
        test_db.save_trade({
            'entry_time': '2026-02-10T13:00:00',
            'exit_time': '2026-02-10T13:05:00',
            'side': 'short',
            'entry_price': 97500.0, 'exit_price': 97800.0,
            'avg_price': 97500.0, 'total_quantity': 0.002,
            'pnl': -0.6, 'pnl_pct': -0.03,
            'exit_reason': 'SL', 'commission': 0.05
        })

        stats = test_db.get_daily_stats('2026-02-10')
        assert stats['total_trades'] == 2
        assert stats['wins'] == 1
        assert stats['losses'] == 1
        assert abs(stats['total_pnl'] - 0.9) < 0.01


# =====================================================================
# 5. TEST Sync with Exchange (Recovery)
# =====================================================================
print("\n" + "="*60)
print("5. sync_with_exchange - Recovery y edge cases")
print("="*60)


@test("Recovery desde DB con avg_price valido")
def _():
    with patch('src.trader.client') as mock_client, \
         patch('src.trader.db') as mock_db:
        mock_client.set_leverage.return_value = {}
        mock_db.get_recent_trades.return_value = []
        mock_db.get_trade_count.return_value = 0

        t = Trader()
        assert t.current_trade is None

        # Simular posicion en exchange + DB
        mock_client.get_position.return_value = {
            'symbol': 'BTCUSDT', 'side': 'long',
            'size': 0.002, 'entry_price': 97000.0,
            'unrealized_pnl': 0.5, 'leverage': 10
        }
        mock_db.get_active_position.return_value = {
            'entry_time': '2026-02-10T12:00:00',
            'side': 'long',
            'avg_price': 96800.0,  # Diferente del exchange (tiene DCA)
            'total_quantity': 0.006,
            'safety_orders_count': 2,
            'commission': 0.15
        }

        t.sync_with_exchange()

        assert t.current_trade is not None
        assert t.current_trade.avg_price == 96800.0, "Debio usar precio de DB"
        assert t.current_trade.safety_orders_count == 2, "Debio recuperar DCA count"


@test("Recovery con DB avg_price=0 cae a exchange fallback")
def _():
    with patch('src.trader.client') as mock_client, \
         patch('src.trader.db') as mock_db:
        mock_client.set_leverage.return_value = {}
        mock_db.get_recent_trades.return_value = []
        mock_db.get_trade_count.return_value = 0

        t = Trader()

        mock_client.get_position.return_value = {
            'symbol': 'BTCUSDT', 'side': 'long',
            'size': 0.002, 'entry_price': 97000.0,
            'unrealized_pnl': 0.5, 'leverage': 10
        }
        # DB tiene avg_price=0 (corrupto)
        mock_db.get_active_position.return_value = {
            'entry_time': '2026-02-10T12:00:00',
            'side': 'long',
            'avg_price': 0.0,
            'total_quantity': 0.002,
            'safety_orders_count': 0,
            'commission': 0.0
        }

        t.sync_with_exchange()

        assert t.current_trade is not None
        assert t.current_trade.avg_price == 97000.0, \
            f"Debio usar precio de exchange (97000), pero es {t.current_trade.avg_price}"
        mock_db.clear_active_position.assert_called()  # Debio limpiar la DB corrupta


@test("Recovery sin posicion en exchange limpia DB huerfana")
def _():
    with patch('src.trader.client') as mock_client, \
         patch('src.trader.db') as mock_db:
        mock_client.set_leverage.return_value = {}
        mock_db.get_recent_trades.return_value = []
        mock_db.get_trade_count.return_value = 0

        t = Trader()

        mock_client.get_position.return_value = None  # Sin posicion
        mock_db.get_active_position.return_value = {
            'entry_time': '2026-02-10T12:00:00',
            'side': 'long',
            'avg_price': 97000.0,
            'total_quantity': 0.002
        }

        t.sync_with_exchange()

        assert t.current_trade is None
        mock_db.clear_active_position.assert_called()


# =====================================================================
# 6. TEST Kill Switch
# =====================================================================
print("\n" + "="*60)
print("6. Kill switch - Proteccion contra perdidas")
print("="*60)


@test("3 perdidas consecutivas pausa el bot")
def _():
    with patch('src.trader.client') as mock_client, \
         patch('src.trader.db') as mock_db:
        mock_client.set_leverage.return_value = {}
        mock_client.get_usdt_balance.return_value = 100.0
        mock_db.get_recent_trades.return_value = []
        mock_db.get_trade_count.return_value = 0

        t = Trader()
        from src.trader import TradeRecord

        # Simular 3 perdidas
        for i in range(3):
            trade = TradeRecord(
                entry_time=datetime.now(), side='long',
                avg_price=97000.0, total_quantity=0.002,
                pnl=-0.5
            )
            t._update_stats(trade)

        assert t.is_paused is True, "Bot debio pausarse tras 3 losses"
        assert "consecutivas" in t.pause_reason


@test("Bot pausado no abre posiciones")
def _():
    with patch('src.trader.client') as mock_client, \
         patch('src.trader.db') as mock_db:
        mock_client.set_leverage.return_value = {}
        mock_db.get_recent_trades.return_value = []
        mock_db.get_trade_count.return_value = 0

        t = Trader()
        t.is_paused = True
        t.pause_reason = "Test"

        result = t.open_position(Signal.LONG, 97000.0, 0)
        assert result is False, "No debio abrir posicion estando pausado"


# =====================================================================
# 7. TEST Strategy - Indicadores y senales
# =====================================================================
print("\n" + "="*60)
print("7. Strategy - Edge cases en indicadores")
print("="*60)

import pandas as pd
import numpy as np
from src.strategy import strategy, Signal as StratSignal


@test("Strategy con DataFrame vacio retorna NONE")
def _():
    df = pd.DataFrame()
    signal = strategy.check_entry_signal(df)
    assert signal == StratSignal.NONE


@test("Strategy con pocas velas (<200) retorna NONE")
def _():
    dates = pd.date_range('2024-01-01', periods=50, freq='1min')
    prices = np.full(50, 97000.0)
    df = pd.DataFrame({
        'open': prices, 'high': prices + 10,
        'low': prices - 10, 'close': prices,
        'volume': np.full(50, 100.0)
    }, index=dates)
    df = strategy.calculate_indicators(df)
    signal = strategy.check_entry_signal(df)
    assert signal == StratSignal.NONE


@test("Strategy con NaN en indicadores retorna NONE")
def _():
    # Crear suficientes velas pero con datos que producen NaN
    dates = pd.date_range('2024-01-01', periods=250, freq='1min')
    prices = np.full(250, 97000.0)
    df = pd.DataFrame({
        'open': prices, 'high': prices + 10,
        'low': prices - 10, 'close': prices,
        'volume': np.full(250, 100.0)
    }, index=dates)
    df = strategy.calculate_indicators(df)

    # Forzar NaN en ema_trend de la ultima vela
    df.loc[df.index[-1], 'ema_trend'] = np.nan

    signal = strategy.check_entry_signal(df)
    assert signal == StratSignal.NONE


# =====================================================================
# 8. TEST _capture_features (bot.py) - Edge cases
# =====================================================================
print("\n" + "="*60)
print("8. _capture_features - Robustez con datos faltantes")
print("="*60)


@test("_capture_features no crashea con DataFrame minimo")
def _():
    with patch('src.trader.client') as mock_client, \
         patch('src.bot.db') as mock_db, \
         patch('src.bot.trader') as mock_trader:
        mock_db.get_trade_count.return_value = 0
        mock_db.save_features.return_value = 1
        mock_trader.trade_history = []

        from src.bot import ScalperBot
        with patch.object(ScalperBot, '__init__', lambda self: None):
            bot = ScalperBot()

            # DataFrame con datos minimos (sin indicadores)
            dates = pd.date_range('2024-01-01', periods=5, freq='1min')
            df = pd.DataFrame({
                'open': [97000]*5, 'high': [97100]*5,
                'low': [96900]*5, 'close': [97050]*5,
                'volume': [100]*5
            }, index=dates)

            # No debe crashear aunque falten indicadores
            bot._capture_features(df, 97050.0)
            # Si llego hasta aqui sin exception, paso el test


@test("_capture_features con NaN en indicadores no crashea")
def _():
    with patch('src.bot.db') as mock_db, \
         patch('src.bot.trader') as mock_trader:
        mock_db.get_trade_count.return_value = 0
        mock_db.save_features.return_value = 1
        mock_trader.trade_history = []

        from src.bot import ScalperBot
        with patch.object(ScalperBot, '__init__', lambda self: None):
            bot = ScalperBot()

            dates = pd.date_range('2024-01-01', periods=25, freq='1min')
            df = pd.DataFrame({
                'open': [97000]*25, 'high': [97100]*25,
                'low': [96900]*25, 'close': [97050]*25,
                'volume': [100]*25,
                'ema_trend': [np.nan]*25,
                'bb_lower': [np.nan]*25,
                'bb_upper': [np.nan]*25,
                'stoch_k': [np.nan]*25,
                'atr': [np.nan]*25,
                'atr_sma': [np.nan]*25
            }, index=dates)

            # No debe crashear con NaN
            bot._capture_features(df, 97050.0)


# =====================================================================
# 9. TEST PnL Calculation Edge Cases
# =====================================================================
print("\n" + "="*60)
print("9. PnL calculation - Edge cases")
print("="*60)


@test("PnL LONG correcto: compra 97000, cierra 97582 (TP 0.6%)")
def _():
    with patch('src.trader.client') as mock_client, \
         patch('src.trader.db') as mock_db:
        mock_client.set_leverage.return_value = {}
        mock_db.get_recent_trades.return_value = []
        mock_db.get_trade_count.return_value = 0
        mock_db.save_trade.return_value = 1
        mock_db.update_feature_outcome.return_value = None
        mock_db.clear_active_position.return_value = None
        mock_client.close_position.return_value = {'avgPrice': '97582.0'}

        t = Trader()
        from src.trader import TradeRecord
        t.current_trade = TradeRecord(
            entry_time=datetime.now(),
            side='long',
            avg_price=97000.0,
            total_quantity=0.002,
            commission=0.002 * 97000.0 * 0.0004  # Entry commission
        )

        t.close_position(97582.0, "TAKE_PROFIT")

        # Verificar que se guardo un trade con PnL positivo
        save_call = mock_db.save_trade.call_args[0][0]
        assert save_call['pnl'] > 0, f"PnL debio ser positivo, es {save_call['pnl']}"
        assert save_call['exit_reason'] == 'TAKE_PROFIT'


@test("PnL SHORT correcto: vende 97000, cierra 96418 (TP 0.6%)")
def _():
    with patch('src.trader.client') as mock_client, \
         patch('src.trader.db') as mock_db:
        mock_client.set_leverage.return_value = {}
        mock_db.get_recent_trades.return_value = []
        mock_db.get_trade_count.return_value = 0
        mock_db.save_trade.return_value = 1
        mock_db.update_feature_outcome.return_value = None
        mock_db.clear_active_position.return_value = None
        mock_client.close_position.return_value = {'avgPrice': '96418.0'}

        t = Trader()
        from src.trader import TradeRecord
        t.current_trade = TradeRecord(
            entry_time=datetime.now(),
            side='short',
            avg_price=97000.0,
            total_quantity=0.002,
            commission=0.002 * 97000.0 * 0.0004
        )

        t.close_position(96418.0, "TAKE_PROFIT")

        save_call = mock_db.save_trade.call_args[0][0]
        assert save_call['pnl'] > 0, f"PnL SHORT debio ser positivo, es {save_call['pnl']}"


# =====================================================================
# RESUMEN
# =====================================================================
print("\n" + "="*60)
total = passed + failed
print(f"RESULTADO: {passed}/{total} tests pasaron")
if failed > 0:
    print(f"\n[ERRORES] {failed} tests fallaron:")
    for e in errors:
        print(f"  - {e}")
    print("="*60)
    sys.exit(1)
else:
    print("[OK] Todos los tests pasaron - Bot robusto!")
    print("="*60)
    sys.exit(0)
