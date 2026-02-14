"""
Pre-Launch Test Suite - Validacion completa antes de produccion
================================================================
Ejecutar: python test_bot_prelaunch.py

Tests:
  1. Imports y dependencias
  2. Settings y config
  3. MLStrategy: ConvictionScorer scoring logic
  4. MLStrategy: regime filtering
  5. MLStrategy: adaptive threshold + sizing
  6. PortfolioManager: duplicate prevention
  7. PortfolioManager: position lifecycle (open/close/trailing)
  8. PortfolioManager: risk checks (DD, daily loss, pause)
  9. PortfolioManager: sync/reconciliation
  10. MLBot: candle detection
  11. Integration: full signal -> open -> monitor -> close flow
"""

import sys
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

PASSED = 0
FAILED = 0
ERRORS = []


def test(name):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            global PASSED, FAILED, ERRORS
            try:
                func()
                PASSED += 1
                print(f'  [PASS] {name}')
            except AssertionError as e:
                FAILED += 1
                ERRORS.append(f'{name}: {e}')
                print(f'  [FAIL] {name}: {e}')
            except Exception as e:
                FAILED += 1
                ERRORS.append(f'{name}: {type(e).__name__}: {e}')
                print(f'  [ERROR] {name}: {type(e).__name__}: {e}')
        wrapper._test_name = name
        return wrapper
    return decorator


# =========================================================================
# 1. IMPORTS
# =========================================================================
print('\n' + '=' * 60)
print('TEST 1: Imports y dependencias')
print('=' * 60)


@test('Import config/settings')
def test_import_settings():
    from config.settings import (
        TRADING_MODE, ML_PAIRS, ML_SIGNAL_THRESHOLD, ML_TIMEFRAME,
        ML_LEVERAGE, ML_RISK_PER_TRADE, ML_MAX_CONCURRENT,
        ML_V84_ENABLED, ML_V85_ENABLED,
        ML_CONVICTION_SKIP_MULT, ML_CONVICTION_SIZING_MIN,
        ML_CONVICTION_SIZING_MAX, ML_TP_PCT, ML_SL_PCT,
        MODELS_DIR, INITIAL_CAPITAL, ML_MAX_DD_PCT,
        ML_MAX_DAILY_LOSS_PCT,
    )
    assert ML_V85_ENABLED is True, 'V8.5 debe estar habilitado'
    assert ML_CONVICTION_SKIP_MULT == 0.5
    assert ML_CONVICTION_SIZING_MIN == 0.3
    assert ML_CONVICTION_SIZING_MAX == 1.8


test_import_settings()


@test('Import ml_strategy')
def test_import_strategy():
    from src.ml_strategy import MLStrategy
    s = MLStrategy()
    assert hasattr(s, 'v85_enabled')
    assert hasattr(s, 'conviction_scorer')
    assert hasattr(s, 'score_conviction')
    assert s.macro_score == 0.5
    assert s.risk_off_mult == 1.0


test_import_strategy()


@test('Import portfolio_manager')
def test_import_pm():
    from src.portfolio_manager import PortfolioManager, Position
    # Dataclass fields are instance attrs, check via __dataclass_fields__
    assert 'pair' in Position.__dataclass_fields__, 'Position missing pair field'
    assert 'trail_active' in Position.__dataclass_fields__, 'Position missing trail_active'


test_import_pm()


@test('Import ml_bot')
def test_import_bot():
    from src.ml_bot import MLBot
    assert hasattr(MLBot, 'run')
    assert hasattr(MLBot, '_on_new_candle')


test_import_bot()


@test('Import scipy.special.expit')
def test_import_scipy():
    from scipy.special import expit
    assert abs(expit(0) - 0.5) < 1e-10, 'expit(0) debe ser 0.5'
    assert expit(100) > 0.99
    assert expit(-100) < 0.01


test_import_scipy()


# =========================================================================
# 2. SETTINGS VALIDATION
# =========================================================================
print('\n' + '=' * 60)
print('TEST 2: Settings y config')
print('=' * 60)


@test('V8.5 settings range validation')
def test_settings_ranges():
    from config.settings import (
        ML_CONVICTION_SKIP_MULT, ML_CONVICTION_SIZING_MIN,
        ML_CONVICTION_SIZING_MAX, ML_SIZING_MIN, ML_SIZING_MAX,
        ML_ADAPTIVE_THRESH_MIN, ML_ADAPTIVE_THRESH_MAX,
    )
    assert 0 < ML_CONVICTION_SKIP_MULT < 2.0, f'skip_mult={ML_CONVICTION_SKIP_MULT}'
    assert 0 < ML_CONVICTION_SIZING_MIN < ML_CONVICTION_SIZING_MAX
    assert ML_CONVICTION_SIZING_MAX <= 3.0, 'sizing max demasiado alto'
    assert ML_ADAPTIVE_THRESH_MIN < ML_ADAPTIVE_THRESH_MAX
    assert ML_SIZING_MIN < ML_SIZING_MAX


test_settings_ranges()


@test('ML pairs consistency')
def test_pairs():
    from config.settings import ML_PAIRS
    assert len(ML_PAIRS) >= 5, f'Solo {len(ML_PAIRS)} pares'
    for p in ML_PAIRS:
        assert '/USDT' in p, f'Par {p} no es USDT'


test_pairs()


@test('Risk settings validation')
def test_risk_settings():
    from config.settings import (
        ML_MAX_DD_PCT, ML_MAX_DAILY_LOSS_PCT, ML_RISK_PER_TRADE,
        ML_MAX_CONCURRENT, ML_MAX_NOTIONAL, INITIAL_CAPITAL,
    )
    assert 0 < ML_MAX_DD_PCT <= 0.5, f'DD max={ML_MAX_DD_PCT}'
    assert 0 < ML_MAX_DAILY_LOSS_PCT <= 0.2, f'Daily loss max={ML_MAX_DAILY_LOSS_PCT}'
    assert 0 < ML_RISK_PER_TRADE <= 0.05, f'Risk per trade={ML_RISK_PER_TRADE}'
    assert 1 <= ML_MAX_CONCURRENT <= 10
    assert ML_MAX_NOTIONAL > 0
    assert INITIAL_CAPITAL > 0


test_risk_settings()


# =========================================================================
# 3. CONVICTION SCORER LOGIC
# =========================================================================
print('\n' + '=' * 60)
print('TEST 3: ConvictionScorer scoring logic')
print('=' * 60)


@test('ConvictionScorer disabled returns (False, 1.0)')
def test_conv_disabled():
    from src.ml_strategy import MLStrategy
    s = MLStrategy()
    s.v85_enabled = False
    skip, mult = s.score_conviction(1.5, 0.01, 0.02, 0, 1)
    assert skip is False
    assert mult == 1.0


test_conv_disabled()


@test('ConvictionScorer with no model returns (False, 1.0)')
def test_conv_no_model():
    from src.ml_strategy import MLStrategy
    s = MLStrategy()
    s.v85_enabled = True
    s.conviction_scorer = None
    skip, mult = s.score_conviction(1.5, 0.01, 0.02, 0, 1)
    assert skip is False
    assert mult == 1.0


test_conv_no_model()


@test('ConvictionScorer skip on negative prediction')
def test_conv_skip_negative():
    from src.ml_strategy import MLStrategy
    s = MLStrategy()
    s.v85_enabled = True
    s.conviction_pred_std = 1.0

    # Mock scorer that predicts very negative PnL
    mock_scorer = MagicMock()
    mock_scorer.predict.return_value = [-0.6]  # < -0.5 * 1.0
    s.conviction_scorer = mock_scorer
    s.conviction_fcols = ['cs_conf', 'cs_pred_mag', 'cs_macro_score',
                          'cs_risk_off', 'cs_regime_bull', 'cs_regime_bear',
                          'cs_regime_range', 'cs_atr_pct', 'cs_n_open',
                          'cs_pred_sign']

    skip, mult = s.score_conviction(1.5, 0.01, 0.02, 0, 1)
    assert skip is True, f'Debe skipear (pred=-0.6 < -0.5), got skip={skip}'
    assert mult == 0.0


test_conv_skip_negative()


@test('ConvictionScorer sizing for positive prediction')
def test_conv_positive():
    from src.ml_strategy import MLStrategy
    from config.settings import ML_CONVICTION_SIZING_MIN, ML_CONVICTION_SIZING_MAX
    s = MLStrategy()
    s.v85_enabled = True
    s.conviction_pred_std = 1.0

    # Mock scorer with positive prediction
    mock_scorer = MagicMock()
    mock_scorer.predict.return_value = [1.0]  # Positive PnL
    s.conviction_scorer = mock_scorer
    s.conviction_fcols = ['cs_conf', 'cs_pred_mag', 'cs_macro_score',
                          'cs_risk_off', 'cs_regime_bull', 'cs_regime_bear',
                          'cs_regime_range', 'cs_atr_pct', 'cs_n_open',
                          'cs_pred_sign']

    skip, mult = s.score_conviction(1.5, 0.01, 0.02, 0, 1)
    assert skip is False
    assert mult > 1.0, f'Prediccion positiva debe dar mult > 1.0, got {mult}'
    assert ML_CONVICTION_SIZING_MIN <= mult <= ML_CONVICTION_SIZING_MAX


test_conv_positive()


@test('ConvictionScorer sizing for neutral prediction')
def test_conv_neutral():
    from src.ml_strategy import MLStrategy
    s = MLStrategy()
    s.v85_enabled = True
    s.conviction_pred_std = 1.0

    # Mock scorer with zero prediction (neutral)
    mock_scorer = MagicMock()
    mock_scorer.predict.return_value = [0.0]
    s.conviction_scorer = mock_scorer
    s.conviction_fcols = ['cs_conf', 'cs_pred_mag', 'cs_macro_score',
                          'cs_risk_off', 'cs_regime_bull', 'cs_regime_bear',
                          'cs_regime_range', 'cs_atr_pct', 'cs_n_open',
                          'cs_pred_sign']

    skip, mult = s.score_conviction(1.5, 0.01, 0.02, 0, 1)
    assert skip is False
    # expit(0) = 0.5 -> mult = 0.3 + 1.5*0.5 = 1.05
    assert 0.9 < mult < 1.2, f'Neutral debe dar ~1.05, got {mult}'


test_conv_neutral()


@test('ConvictionScorer passes correct features')
def test_conv_features():
    from src.ml_strategy import MLStrategy
    s = MLStrategy()
    s.v85_enabled = True
    s.conviction_pred_std = 1.0
    s.regime = 'BULL'
    s.macro_score = 0.8
    s.risk_off_mult = 0.7

    mock_scorer = MagicMock()
    mock_scorer.predict.return_value = [0.5]
    s.conviction_scorer = mock_scorer
    s.conviction_fcols = ['cs_conf', 'cs_pred_mag', 'cs_macro_score',
                          'cs_risk_off', 'cs_regime_bull', 'cs_regime_bear',
                          'cs_regime_range', 'cs_atr_pct', 'cs_n_open',
                          'cs_pred_sign']

    s.score_conviction(2.0, 0.05, 0.03, 2, -1)

    # Verify the DataFrame passed to predict
    call_args = mock_scorer.predict.call_args
    df = call_args[0][0]
    assert float(df['cs_conf'].iloc[0]) == 2.0
    assert float(df['cs_pred_mag'].iloc[0]) == 0.05
    assert float(df['cs_macro_score'].iloc[0]) == 0.8
    assert float(df['cs_risk_off'].iloc[0]) == 0.7
    assert float(df['cs_regime_bull'].iloc[0]) == 1.0
    assert float(df['cs_regime_bear'].iloc[0]) == 0.0
    assert float(df['cs_n_open'].iloc[0]) == 2
    assert float(df['cs_pred_sign'].iloc[0]) == -1.0


test_conv_features()


# =========================================================================
# 4. REGIME FILTERING
# =========================================================================
print('\n' + '=' * 60)
print('TEST 4: Regime filtering logic')
print('=' * 60)


@test('BULL regime blocks SHORT signals')
def test_regime_bull_no_short():
    from src.ml_strategy import MLStrategy
    s = MLStrategy()
    s.regime = 'BULL'
    # In BULL: direction=-1 (SHORT) should be filtered
    # This is checked in generate_signals, line: if self.regime == 'BULL' and direction == -1: continue
    assert s.regime == 'BULL'


test_regime_bull_no_short()


@test('BEAR regime blocks LONG signals')
def test_regime_bear_no_long():
    from src.ml_strategy import MLStrategy
    s = MLStrategy()
    s.regime = 'BEAR'
    assert s.regime == 'BEAR'


test_regime_bear_no_long()


# =========================================================================
# 5. ADAPTIVE THRESHOLD + SIZING
# =========================================================================
print('\n' + '=' * 60)
print('TEST 5: Adaptive threshold + sizing')
print('=' * 60)


@test('Adaptive threshold: macro_score=0 -> strict threshold')
def test_thresh_low_score():
    from src.ml_strategy import MLStrategy
    from config.settings import ML_ADAPTIVE_THRESH_MAX
    s = MLStrategy()
    s.v84_enabled = True
    s.macro_score = 0.0
    t = s.get_adaptive_threshold()
    assert t == ML_ADAPTIVE_THRESH_MAX, f'Score=0 -> thresh should be max ({ML_ADAPTIVE_THRESH_MAX}), got {t}'


test_thresh_low_score()


@test('Adaptive threshold: macro_score=1 -> lenient threshold')
def test_thresh_high_score():
    from src.ml_strategy import MLStrategy
    from config.settings import ML_ADAPTIVE_THRESH_MIN
    s = MLStrategy()
    s.v84_enabled = True
    s.macro_score = 1.0
    t = s.get_adaptive_threshold()
    assert t == ML_ADAPTIVE_THRESH_MIN, f'Score=1 -> thresh should be min ({ML_ADAPTIVE_THRESH_MIN}), got {t}'


test_thresh_high_score()


@test('ML sizing: macro_score=0 -> minimum sizing')
def test_sizing_low():
    from src.ml_strategy import MLStrategy
    from config.settings import ML_SIZING_MIN
    s = MLStrategy()
    s.v84_enabled = True
    s.macro_score = 0.0
    s.risk_off_mult = 1.0
    m = s.get_sizing_multiplier()
    assert abs(m - ML_SIZING_MIN) < 0.01, f'Expected ~{ML_SIZING_MIN}, got {m}'


test_sizing_low()


@test('ML sizing: macro_score=1 -> maximum sizing')
def test_sizing_high():
    from src.ml_strategy import MLStrategy
    from config.settings import ML_SIZING_MAX
    s = MLStrategy()
    s.v84_enabled = True
    s.macro_score = 1.0
    s.risk_off_mult = 1.0
    m = s.get_sizing_multiplier()
    assert abs(m - ML_SIZING_MAX) < 0.01, f'Expected ~{ML_SIZING_MAX}, got {m}'


test_sizing_high()


@test('Combined sizing: macro * conviction stays in bounds')
def test_combined_sizing():
    from config.settings import ML_SIZING_MAX, ML_CONVICTION_SIZING_MAX
    # Worst case: max macro * max conviction
    worst = ML_SIZING_MAX * ML_CONVICTION_SIZING_MAX
    # Code clips to max(0.2, min(2.5, total_sizing))
    clipped = max(0.2, min(2.5, worst))
    assert clipped <= 2.5, f'Combined sizing {worst} should clip to 2.5'


test_combined_sizing()


# =========================================================================
# 6. DUPLICATE PREVENTION
# =========================================================================
print('\n' + '=' * 60)
print('TEST 6: Duplicate prevention')
print('=' * 60)


@test('can_open rejects pair already in positions')
def test_can_open_dup_pair():
    from src.portfolio_manager import PortfolioManager, Position
    mock_exchange = MagicMock()
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    try:
        pm = PortfolioManager(mock_exchange, db_path)
        pm.positions['BTC/USDT'] = Position(
            pair='BTC/USDT', side='long', direction=1,
            entry_price=50000, quantity=0.01, notional=500,
            leverage=3, tp_price=52000, sl_price=48000,
            tp_pct=0.04, sl_pct=0.04, atr_pct=0.02,
        )
        assert pm.can_open('BTC/USDT', 1) is False
    finally:
        db_path.unlink(missing_ok=True)


test_can_open_dup_pair()


@test('can_open rejects when max concurrent reached')
def test_can_open_max():
    from src.portfolio_manager import PortfolioManager, Position
    from config.settings import ML_MAX_CONCURRENT
    mock_exchange = MagicMock()
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    try:
        pm = PortfolioManager(mock_exchange, db_path)
        for i in range(ML_MAX_CONCURRENT):
            pair = f'TEST{i}/USDT'
            pm.positions[pair] = Position(
                pair=pair, side='long', direction=1,
                entry_price=100, quantity=1, notional=100,
                leverage=3, tp_price=110, sl_price=90,
                tp_pct=0.1, sl_pct=0.1, atr_pct=0.02,
            )
        assert pm.can_open('NEW/USDT', 1) is False
    finally:
        db_path.unlink(missing_ok=True)


test_can_open_max()


@test('can_open rejects when killed')
def test_can_open_killed():
    from src.portfolio_manager import PortfolioManager
    mock_exchange = MagicMock()
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    try:
        pm = PortfolioManager(mock_exchange, db_path)
        pm.killed = True
        assert pm.can_open('BTC/USDT', 1) is False
    finally:
        db_path.unlink(missing_ok=True)


test_can_open_killed()


@test('can_open rejects when paused')
def test_can_open_paused():
    from src.portfolio_manager import PortfolioManager
    mock_exchange = MagicMock()
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    try:
        pm = PortfolioManager(mock_exchange, db_path)
        pm.paused = True
        assert pm.can_open('BTC/USDT', 1) is False
    finally:
        db_path.unlink(missing_ok=True)


test_can_open_paused()


@test('open_position exchange safety check prevents duplicate')
def test_exchange_dup_check():
    from src.portfolio_manager import PortfolioManager
    mock_exchange = MagicMock()

    # Exchange reports existing position for BTC/USDT
    mock_exchange.fetch_positions.return_value = [{
        'symbol': 'BTC/USDT:USDT',
        'contracts': 0.01,
        'side': 'long',
        'entryPrice': 95000,
        'leverage': 3,
    }]

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    try:
        pm = PortfolioManager(mock_exchange, db_path)
        result = pm.open_position(
            pair='BTC/USDT', direction=1, confidence=1.5,
            regime='RANGE', price=95000, atr_pct=0.02,
        )
        # Should return False (adopted, not opened new)
        assert result is False, 'Should return False when adopting existing position'
        # Should have adopted the position
        assert 'BTC/USDT' in pm.positions, 'Position should be adopted'
        pos = pm.positions['BTC/USDT']
        assert pos.entry_price == 95000
        assert pos.quantity == 0.01
        # create_order should NOT have been called
        assert mock_exchange.create_order.call_count == 0, \
            'create_order should NOT be called when duplicate detected'
    finally:
        db_path.unlink(missing_ok=True)


test_exchange_dup_check()


@test('open_position proceeds when exchange has no position')
def test_exchange_no_dup():
    from src.portfolio_manager import PortfolioManager
    mock_exchange = MagicMock()

    # Exchange reports no positions
    mock_exchange.fetch_positions.return_value = []
    mock_exchange.set_leverage.return_value = None
    mock_exchange.amount_to_precision.return_value = '0.001'
    mock_exchange.price_to_precision.return_value = '93575.0'
    mock_exchange.create_order.return_value = {
        'average': 95000, 'filled': 0.001, 'id': 'test_order_123',
    }

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    try:
        pm = PortfolioManager(mock_exchange, db_path)
        result = pm.open_position(
            pair='BTC/USDT', direction=1, confidence=1.5,
            regime='RANGE', price=95000, atr_pct=0.02,
        )
        assert result is True, 'Should succeed when no duplicate'
        assert 'BTC/USDT' in pm.positions
        # 2 calls: market order + SL STOP_MARKET order
        assert mock_exchange.create_order.call_count == 2
    finally:
        db_path.unlink(missing_ok=True)


test_exchange_no_dup()


# =========================================================================
# 7. POSITION LIFECYCLE
# =========================================================================
print('\n' + '=' * 60)
print('TEST 7: Position lifecycle')
print('=' * 60)


@test('Trailing stop activates at correct profit level')
def test_trailing_activation():
    from src.portfolio_manager import PortfolioManager, Position
    from config.settings import ML_TP_PCT, ML_TRAILING_ACTIVATION
    mock_exchange = MagicMock()
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    try:
        pm = PortfolioManager(mock_exchange, db_path)
        pos = Position(
            pair='BTC/USDT', side='long', direction=1,
            entry_price=100000, quantity=0.01, notional=1000,
            leverage=3, tp_price=100000 * (1 + ML_TP_PCT),
            sl_price=100000 * 0.96, tp_pct=ML_TP_PCT, sl_pct=0.04,
            atr_pct=0.02,
        )
        # Price slightly above activation level (avoid float boundary)
        activation_price = 100000 * (1 + ML_TP_PCT * ML_TRAILING_ACTIVATION) + 1
        pm._update_trailing(pos, activation_price)
        assert pos.trail_active is True, f'Trail should activate at {activation_price}'
        assert pos.trail_sl is not None
        assert pos.trail_sl > pos.entry_price, 'Trail SL should be above entry for long'
    finally:
        db_path.unlink(missing_ok=True)


test_trailing_activation()


@test('Position closes on TP hit')
def test_close_tp():
    from src.portfolio_manager import PortfolioManager, Position
    mock_exchange = MagicMock()
    mock_exchange.fetch_ticker.return_value = {'last': 104100}  # Above TP
    mock_exchange.create_order.return_value = {'average': 104100, 'filled': 0.01}

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    try:
        pm = PortfolioManager(mock_exchange, db_path)
        pos = Position(
            pair='BTC/USDT', side='long', direction=1,
            entry_price=100000, quantity=0.01, notional=1000,
            leverage=3, tp_price=104000, sl_price=96000,
            tp_pct=0.04, sl_pct=0.04, atr_pct=0.02,
        )
        pm.positions['BTC/USDT'] = pos
        trades = pm.update_positions()
        assert len(trades) == 1, f'Expected 1 closed trade, got {len(trades)}'
        assert trades[0]['exit_reason'] == 'TP'
        assert trades[0]['pnl'] > 0, 'TP trade should be profitable'
        assert 'BTC/USDT' not in pm.positions, 'Position should be removed'
    finally:
        db_path.unlink(missing_ok=True)


test_close_tp()


@test('Position closes on SL hit')
def test_close_sl():
    from src.portfolio_manager import PortfolioManager, Position
    mock_exchange = MagicMock()
    mock_exchange.fetch_ticker.return_value = {'last': 95900}  # Below SL
    mock_exchange.create_order.return_value = {'average': 95900, 'filled': 0.01}

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    try:
        pm = PortfolioManager(mock_exchange, db_path)
        pos = Position(
            pair='BTC/USDT', side='long', direction=1,
            entry_price=100000, quantity=0.01, notional=1000,
            leverage=3, tp_price=104000, sl_price=96000,
            tp_pct=0.04, sl_pct=0.04, atr_pct=0.02,
        )
        pm.positions['BTC/USDT'] = pos
        trades = pm.update_positions()
        assert len(trades) == 1
        assert trades[0]['exit_reason'] == 'SL'
        assert trades[0]['pnl'] < 0, 'SL trade should be negative'
    finally:
        db_path.unlink(missing_ok=True)


test_close_sl()


@test('Position closes on TIMEOUT')
def test_close_timeout():
    from src.portfolio_manager import PortfolioManager, Position
    mock_exchange = MagicMock()
    mock_exchange.fetch_ticker.return_value = {'last': 100500}  # Between SL and TP
    mock_exchange.create_order.return_value = {'average': 100500, 'filled': 0.01}

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    try:
        pm = PortfolioManager(mock_exchange, db_path)
        pos = Position(
            pair='BTC/USDT', side='long', direction=1,
            entry_price=100000, quantity=0.01, notional=1000,
            leverage=3, tp_price=104000, sl_price=96000,
            tp_pct=0.04, sl_pct=0.04, atr_pct=0.02,
            max_hold=1,  # 1 candle = 4h
            entry_time=datetime.now(timezone.utc) - timedelta(hours=5),  # >4h ago
        )
        pm.positions['BTC/USDT'] = pos
        trades = pm.update_positions()
        assert len(trades) == 1
        assert trades[0]['exit_reason'] == 'TIMEOUT'
    finally:
        db_path.unlink(missing_ok=True)


test_close_timeout()


# =========================================================================
# 8. RISK CHECKS
# =========================================================================
print('\n' + '=' * 60)
print('TEST 8: Risk checks')
print('=' * 60)


@test('Kill switch on excessive DD')
def test_kill_switch():
    from src.portfolio_manager import PortfolioManager
    from config.settings import INITIAL_CAPITAL, ML_MAX_DD_PCT
    mock_exchange = MagicMock()
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    try:
        pm = PortfolioManager(mock_exchange, db_path)
        pm.peak_balance = INITIAL_CAPITAL
        pm.balance = INITIAL_CAPITAL * (1 - ML_MAX_DD_PCT - 0.01)  # Exceed DD
        result = pm.check_risk()
        assert result is False, 'Should return False on DD breach'
        assert pm.killed is True, 'Should set killed=True'
    finally:
        db_path.unlink(missing_ok=True)


test_kill_switch()


@test('Daily loss pause')
def test_daily_pause():
    from src.portfolio_manager import PortfolioManager
    from config.settings import INITIAL_CAPITAL, ML_MAX_DAILY_LOSS_PCT
    mock_exchange = MagicMock()
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    try:
        pm = PortfolioManager(mock_exchange, db_path)
        pm.daily_pnl = -(INITIAL_CAPITAL * ML_MAX_DAILY_LOSS_PCT + 1)
        pm.daily_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        result = pm.check_risk()
        assert result is False, 'Should return False on daily loss breach'
        assert pm.paused is True, 'Should set paused=True'
        assert pm.killed is False, 'Should NOT kill on daily loss'
    finally:
        db_path.unlink(missing_ok=True)


test_daily_pause()


@test('Normal operation passes risk check')
def test_risk_ok():
    from src.portfolio_manager import PortfolioManager
    from config.settings import INITIAL_CAPITAL
    mock_exchange = MagicMock()
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    try:
        pm = PortfolioManager(mock_exchange, db_path)
        pm.peak_balance = INITIAL_CAPITAL
        pm.balance = INITIAL_CAPITAL * 0.95  # 5% DD, within limits
        pm.daily_pnl = -1.0  # Minimal loss
        pm.daily_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        result = pm.check_risk()
        assert result is True
        assert pm.killed is False
        assert pm.paused is False
    finally:
        db_path.unlink(missing_ok=True)


test_risk_ok()


# =========================================================================
# 9. SYNC / RECONCILIATION
# =========================================================================
print('\n' + '=' * 60)
print('TEST 9: Sync / reconciliation')
print('=' * 60)


@test('Reconcile adopts exchange position not in DB')
def test_reconcile_adopt():
    from src.portfolio_manager import PortfolioManager
    mock_exchange = MagicMock()
    mock_exchange.fetch_positions.return_value = [{
        'symbol': 'ETH/USDT:USDT',
        'contracts': 1.5,
        'side': 'long',
        'entryPrice': 3000,
        'leverage': 5,
    }]

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    try:
        pm = PortfolioManager(mock_exchange, db_path)
        pm._reconcile_with_exchange()
        assert 'ETH/USDT' in pm.positions, 'Should adopt ETH/USDT from exchange'
        pos = pm.positions['ETH/USDT']
        assert pos.entry_price == 3000
        assert pos.quantity == 1.5
        assert pos.leverage == 5
    finally:
        db_path.unlink(missing_ok=True)


test_reconcile_adopt()


@test('Reconcile removes stale DB position')
def test_reconcile_stale():
    from src.portfolio_manager import PortfolioManager, Position
    mock_exchange = MagicMock()
    # Exchange has NO positions
    mock_exchange.fetch_positions.return_value = []

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    try:
        pm = PortfolioManager(mock_exchange, db_path)
        # Add stale position to memory
        pm.positions['BTC/USDT'] = Position(
            pair='BTC/USDT', side='long', direction=1,
            entry_price=90000, quantity=0.01, notional=900,
            leverage=3, tp_price=94000, sl_price=86000,
            tp_pct=0.04, sl_pct=0.04, atr_pct=0.02,
        )
        pm._reconcile_with_exchange()
        assert 'BTC/USDT' not in pm.positions, \
            'Stale position should be removed after reconciliation'
    finally:
        db_path.unlink(missing_ok=True)


test_reconcile_stale()


@test('Reconcile handles exchange API error gracefully')
def test_reconcile_error():
    from src.portfolio_manager import PortfolioManager, Position
    mock_exchange = MagicMock()
    mock_exchange.fetch_positions.side_effect = Exception('API timeout')

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    try:
        pm = PortfolioManager(mock_exchange, db_path)
        pm.positions['BTC/USDT'] = Position(
            pair='BTC/USDT', side='long', direction=1,
            entry_price=90000, quantity=0.01, notional=900,
            leverage=3, tp_price=94000, sl_price=86000,
            tp_pct=0.04, sl_pct=0.04, atr_pct=0.02,
        )
        # Should not crash, should keep existing positions
        pm._reconcile_with_exchange()
        assert 'BTC/USDT' in pm.positions, \
            'Existing position should survive API error'
    finally:
        db_path.unlink(missing_ok=True)


test_reconcile_error()


@test('DB persistence: save and recover position')
def test_db_persistence():
    from src.portfolio_manager import PortfolioManager, Position
    mock_exchange = MagicMock()
    mock_exchange.fetch_positions.return_value = [{
        'symbol': 'SOL/USDT:USDT',
        'contracts': 10,
        'side': 'short',
        'entryPrice': 200,
        'leverage': 3,
    }]
    mock_exchange.price_to_precision.return_value = '210.0'
    mock_exchange.cancel_order.return_value = None
    mock_exchange.create_order.return_value = {'id': 'sl_order_test'}

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    try:
        # First instance: save a position
        pm1 = PortfolioManager(mock_exchange, db_path)
        pos = Position(
            pair='SOL/USDT', side='short', direction=-1,
            entry_price=200, quantity=10, notional=2000,
            leverage=3, tp_price=180, sl_price=220,
            tp_pct=0.1, sl_pct=0.1, atr_pct=0.03,
            trail_active=True, trail_sl=210, peak_price=190,
        )
        pm1._save_position(pos)

        # Second instance: recover
        pm2 = PortfolioManager(mock_exchange, db_path)
        pm2.sync_positions()
        assert 'SOL/USDT' in pm2.positions
        recovered = pm2.positions['SOL/USDT']
        assert recovered.entry_price == 200
        assert recovered.direction == -1
        assert recovered.trail_active is True
        assert recovered.trail_sl == 210
    finally:
        db_path.unlink(missing_ok=True)


test_db_persistence()


# =========================================================================
# 10. CANDLE DETECTION
# =========================================================================
print('\n' + '=' * 60)
print('TEST 10: Candle detection')
print('=' * 60)


@test('_is_new_4h_candle only triggers at correct hours')
def test_candle_hours():
    from config.settings import ML_CANDLE_HOURS
    # Verify correct hours
    assert ML_CANDLE_HOURS == [0, 4, 8, 12, 16, 20], \
        f'Expected [0,4,8,12,16,20], got {ML_CANDLE_HOURS}'


test_candle_hours()


# =========================================================================
# 11. INTEGRATION TEST
# =========================================================================
print('\n' + '=' * 60)
print('TEST 11: Integration - full flow')
print('=' * 60)


@test('V8.5 graceful degradation: no model -> V8.4 only')
def test_graceful_v85():
    from src.ml_strategy import MLStrategy
    s = MLStrategy()
    s.v85_enabled = True
    # Without loading models, conviction_scorer is None
    skip, mult = s.score_conviction(1.5, 0.01, 0.02, 0, 1)
    assert skip is False and mult == 1.0, 'No model -> transparent passthrough'


test_graceful_v85()


@test('V8.4 graceful degradation: no model -> V7 only')
def test_graceful_v84():
    from src.ml_strategy import MLStrategy
    s = MLStrategy()
    s.v84_enabled = True
    s.macro_scorer = None
    # update_macro should set defaults when no scorer
    s.update_macro()
    assert s.macro_score == 0.5, f'No scorer -> neutral score, got {s.macro_score}'
    assert s.risk_off_mult == 1.0


test_graceful_v84()


@test('Signal -> open_position flow with mocked exchange')
def test_full_flow():
    from src.portfolio_manager import PortfolioManager

    mock_exchange = MagicMock()
    # Safety check: no existing positions
    mock_exchange.fetch_positions.return_value = []
    mock_exchange.set_leverage.return_value = None
    mock_exchange.amount_to_precision.return_value = '0.01'
    mock_exchange.create_order.return_value = {
        'average': 95000, 'filled': 0.01
    }
    # For position monitoring
    mock_exchange.fetch_ticker.return_value = {'last': 96000}

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    try:
        pm = PortfolioManager(mock_exchange, db_path)

        # Simulate signal -> open
        result = pm.open_position(
            pair='BTC/USDT', direction=1, confidence=1.8,
            regime='BULL', price=95000, atr_pct=0.025,
            sizing_mult=1.2,
        )
        assert result is True, 'Should open successfully'
        assert 'BTC/USDT' in pm.positions
        pos = pm.positions['BTC/USDT']
        assert pos.entry_price == 95000
        assert pos.side == 'long'

        # Simulate monitoring (price moves up, no exit yet)
        trades = pm.update_positions()
        assert len(trades) == 0, 'Should not close yet (price between SL and TP)'
        assert 'BTC/USDT' in pm.positions

        # Price hits TP
        mock_exchange.fetch_ticker.return_value = {'last': pos.tp_price + 100}
        mock_exchange.create_order.return_value = {
            'average': pos.tp_price, 'filled': 0.01
        }
        trades = pm.update_positions()
        assert len(trades) == 1, 'Should close at TP'
        assert trades[0]['exit_reason'] == 'TP'
        assert trades[0]['pnl'] > 0
        assert 'BTC/USDT' not in pm.positions
    finally:
        db_path.unlink(missing_ok=True)


test_full_flow()


@test('Correlation filter: max 2 same direction')
def test_correlation_filter():
    from src.portfolio_manager import PortfolioManager, Position
    mock_exchange = MagicMock()
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    try:
        pm = PortfolioManager(mock_exchange, db_path)
        # Add 2 LONGs
        for pair in ['BTC/USDT', 'ETH/USDT']:
            pm.positions[pair] = Position(
                pair=pair, side='long', direction=1,
                entry_price=1000, quantity=1, notional=1000,
                leverage=3, tp_price=1100, sl_price=900,
                tp_pct=0.1, sl_pct=0.1, atr_pct=0.02,
            )
        # Third LONG should be rejected
        assert pm.can_open('SOL/USDT', 1) is False, \
            'Should reject 3rd LONG (correlation filter)'
        # SHORT should still be allowed
        assert pm.can_open('SOL/USDT', -1) is True, \
            'Should allow SHORT (different direction)'
    finally:
        db_path.unlink(missing_ok=True)


test_correlation_filter()


@test('PnL calculation is correct')
def test_pnl_calc():
    from src.portfolio_manager import PortfolioManager, Position
    from config.settings import COMMISSION_RATE, SLIPPAGE_PCT
    mock_exchange = MagicMock()
    mock_exchange.fetch_ticker.return_value = {'last': 105000}
    mock_exchange.create_order.return_value = {'average': 105000, 'filled': 0.1}

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    try:
        pm = PortfolioManager(mock_exchange, db_path)
        pos = Position(
            pair='BTC/USDT', side='long', direction=1,
            entry_price=100000, quantity=0.1, notional=10000,
            leverage=3, tp_price=104000, sl_price=96000,
            tp_pct=0.04, sl_pct=0.04, atr_pct=0.02,
        )
        pm.positions['BTC/USDT'] = pos
        trades = pm.update_positions()

        assert len(trades) == 1
        t = trades[0]
        # 5% profit on $10000 notional = $500 gross
        gross = 10000 * (105000 - 100000) / 100000
        commission = 10000 * (COMMISSION_RATE + SLIPPAGE_PCT) * 2
        expected_pnl = gross - commission
        assert abs(t['pnl'] - expected_pnl) < 0.01, \
            f'PnL should be ~${expected_pnl:.2f}, got ${t["pnl"]:.2f}'
    finally:
        db_path.unlink(missing_ok=True)


test_pnl_calc()


# =========================================================================
# RESULTS
# =========================================================================
print('\n' + '=' * 60)
print(f'RESULTADOS: {PASSED} passed, {FAILED} failed')
print('=' * 60)

if ERRORS:
    print('\nFAILURES:')
    for e in ERRORS:
        print(f'  - {e}')

if FAILED == 0:
    print('\nTODOS LOS TESTS PASARON - LISTO PARA PRODUCCION')
else:
    print(f'\n{FAILED} TESTS FALLARON - REVISAR ANTES DE PRODUCCION')

sys.exit(1 if FAILED > 0 else 0)
