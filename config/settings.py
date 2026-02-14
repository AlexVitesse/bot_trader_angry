"""
Binance Scalper Bot - Configuracion Central
============================================
Estrategia: v6.7 Smart Metralladora (Trend Filter + DCA)
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# =============================================================================
# RUTAS DEL PROYECTO
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"

# Crear directorios si no existen
LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# =============================================================================
# CONFIGURACION DE BINANCE
# =============================================================================
TRADING_MODE = os.getenv("TRADING_MODE", "testnet")  # "testnet" o "live"

# API Keys segun modo
if TRADING_MODE == "testnet":
    BINANCE_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "")
    BINANCE_BASE_URL = "https://demo-fapi.binance.com"
else:
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    BINANCE_BASE_URL = "https://fapi.binance.com"

# =============================================================================
# CONFIGURACION DE TRADING
# =============================================================================
SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
LEVERAGE = 10  # Volvemos a 10x para la v6.7 ganadora

# Capital
INITIAL_CAPITAL = 100.0  # Recomendado para esta configuracion
MAX_POSITION_SIZE_PCT = 0.95

# =============================================================================
# ESTRATEGIA v6.7 - SMART METRALLADORA (Trend Filter + DCA)
# =============================================================================

# Indicadores
BB_LENGTH = 20
BB_STD = 2.0
EMA_TREND_LENGTH = 200  # Filtro maestro de tendencia

# Gestion de Riesgo Grinder (DCA)
POSITION_SIZE_PCT = 0.12      # 12% del balance como margen por trade
MIN_ORDER_MARGIN = 12.0       # Minimo $12 para cumplir $100 notional con 10x
BASE_ORDER_MARGIN = 12.0      # Fallback fijo (usado si no se puede obtener balance)
DCA_STEP_PCT = 0.008         # 0.8% distancia (Mas seguro)
MAX_SAFETY_ORDERS = 2        # Solo 2 recompras
MARTINGALE_MULTIPLIER = 2.0 
TAKE_PROFIT_PCT = 0.006      # 0.6% TP (6% ROE con 10x)
STOP_LOSS_CATASTROPHIC = 0.015 # 1.5% SL desde el promedio

# Comisiones (Binance Futures)
COMMISSION_RATE = 0.0004  # 0.04% taker fee

# Slippage estimado (diferencia entre precio esperado y precio real de ejecucion)
SLIPPAGE_PCT = 0.0001    # 0.01% estimado para BTC/USDT (alta liquidez)

# =============================================================================
# LIMITES DE SEGURIDAD (KILL SWITCH)
# =============================================================================
MAX_DAILY_LOSS_PCT = 0.20       # 20% perdida diaria maxima -> pausar bot
MAX_CONSECUTIVE_LOSSES = 3      # 3 perdidas seguidas -> pausar bot (antes era 5)
MAX_TRADES_PER_DAY = 50         # Maximo de trades por dia

# Rolling Win Rate Protection (DESACTIVADO - modo agresivo)
ROLLING_WR_WINDOW = 20          # Ventana de ultimos N trades para calcular WR
ROLLING_WR_MIN = 0.0            # 0% = desactivado (breakeven real es ~77%)
ROLLING_WR_RESUME = 0.82        # Threshold para reanudar si se reactiva

# Volatility Regime Filter (DESACTIVADO - modo agresivo)
ATR_REGIME_LENGTH = 14          # Periodo del ATR para detectar regimen
ATR_REGIME_MULT_HIGH = 99.0     # 99x = efectivamente desactivado
ATR_REGIME_MULT_LOW = 0.0       # 0x = efectivamente desactivado

# =============================================================================
# TELEGRAM (Alertas)
# =============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

# =============================================================================
# BASE DE DATOS (SQLite)
# =============================================================================
DB_FILE = DATA_DIR / "bot_trades.db"

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FILE = LOGS_DIR / "bot.log"

# =============================================================================
# METRICAS OBJETIVO (para monitoreo)
# =============================================================================
TARGET_WIN_RATE = 0.35          # 35% minimo aceptable
TARGET_PROFIT_FACTOR = 1.20     # 1.20 minimo aceptable
TARGET_RR_RATIO = 2.0           # 2:1 minimo aceptable


# =============================================================================
# ML TRADER PROFESIONAL (Fase 3 - V7)
# =============================================================================
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

ML_DB_FILE = DATA_DIR / "ml_bot.db"

ML_PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT',
    'NEAR/USDT',
]

ML_TIMEFRAME = '4h'
ML_HORIZON = 5              # Predecir retorno 5 velas adelante (20h)
ML_SIGNAL_THRESHOLD = 0.7   # Confianza minima para senal

# Risk Management
ML_MAX_CONCURRENT = 3       # Max posiciones simultaneas
ML_MAX_DD_PCT = 0.20        # 20% DD -> kill switch
ML_MAX_DAILY_LOSS_PCT = 0.05  # 5% daily loss -> pausar
ML_RISK_PER_TRADE = 0.02    # 2% capital at risk
ML_MAX_NOTIONAL = 300.0     # Cap notional por trade

# Leverage por regimen
ML_LEVERAGE = {'BULL': 5, 'BEAR': 4, 'RANGE': 3}

# TP/SL fijos (ganador en backtest - ATR causaba kill switch)
ML_TP_PCT = 0.03            # 3% TP
ML_SL_PCT = 0.015           # 1.5% SL

# Trailing Stop
ML_TRAILING_ACTIVATION = 0.5  # Activar al 50% del TP (1.5% profit)
ML_TRAILING_LOCK = 0.4        # Proteger 40% de ganancia desde peak

# Max hold (velas de 4h)
ML_MAX_HOLD = {'BULL': 30, 'BEAR': 30, 'RANGE': 15}

# Timing
ML_CHECK_INTERVAL = 30      # Segundos entre checks de posiciones
ML_CANDLE_HOURS = [0, 4, 8, 12, 16, 20]  # 4h candle close hours UTC


# =============================================================================
# V8.4 MACRO INTELLIGENCE
# =============================================================================
ML_V84_ENABLED = True       # Feature flag: enable V8.4 macro layer

# Adaptive Threshold: adjusts V7 confidence threshold by macro score
# thresh = THRESH_MAX - (THRESH_MAX - THRESH_MIN) * macro_score
# score=0.0 -> thresh=0.90 (very selective)
# score=0.5 -> thresh=0.70 (V7 default)
# score=1.0 -> thresh=0.50 (accept more signals)
ML_ADAPTIVE_THRESH_MIN = 0.50
ML_ADAPTIVE_THRESH_MAX = 0.90

# ML Sizing: scale position size by macro score
# sizing_mult = SIZING_MIN + (SIZING_MAX - SIZING_MIN) * macro_score
ML_SIZING_MIN = 0.3
ML_SIZING_MAX = 1.8

# Soft Risk-Off: reduce sizing on extreme macro days (not regime override)
ML_RISKOFF_ENABLED = True


def validate_config() -> bool:
    """Valida que la configuracion este completa."""
    errors = []

    if not BINANCE_API_KEY:
        errors.append("BINANCE_API_KEY no configurada")
    if not BINANCE_API_SECRET:
        errors.append("BINANCE_API_SECRET no configurada")
    if LEVERAGE < 1 or LEVERAGE > 20:
        errors.append(f"LEVERAGE fuera de rango: {LEVERAGE}")
    if BASE_ORDER_MARGIN <= 0:
        errors.append("BASE_ORDER_MARGIN debe ser > 0")
    if POSITION_SIZE_PCT <= 0 or POSITION_SIZE_PCT > 0.5:
        errors.append(f"POSITION_SIZE_PCT fuera de rango seguro: {POSITION_SIZE_PCT} (debe ser 0-50%)")
    if MIN_ORDER_MARGIN <= 0:
        errors.append("MIN_ORDER_MARGIN debe ser > 0")
    if TAKE_PROFIT_PCT <= 0:
        errors.append("TAKE_PROFIT_PCT debe ser > 0")
    if STOP_LOSS_CATASTROPHIC <= 0:
        errors.append("STOP_LOSS_CATASTROPHIC debe ser > 0")

    if errors:
        _logger = logging.getLogger(__name__)
        _logger.error("[CONFIG ERROR] Errores de configuracion:")
        for e in errors:
            _logger.error(f"  - {e}")
        return False

    return True


def print_config():
    """Imprime la configuracion actual."""
    _logger = logging.getLogger(__name__)
    _logger.info("=" * 60)
    _logger.info("CONFIGURACION DEL BOT")
    _logger.info("=" * 60)
    _logger.info(f"Modo: {TRADING_MODE.upper()}")
    _logger.info(f"Simbolo: {SYMBOL}")
    _logger.info(f"Timeframe: {TIMEFRAME}")
    _logger.info(f"Leverage: {LEVERAGE}x")
    _logger.info(f"Capital: ${INITIAL_CAPITAL}")
    _logger.info(f"Estrategia: v6.7 SMART METRALLADORA (Trend Filter + DCA)")
    _logger.info(f"  - BB({BB_LENGTH}, {BB_STD}) | EMA Trend: {EMA_TREND_LENGTH}")
    _logger.info(f"  - Position Sizing: {POSITION_SIZE_PCT*100}% del balance (min ${MIN_ORDER_MARGIN})")
    _logger.info(f"  - DCA: {MAX_SAFETY_ORDERS} Recompras cada {DCA_STEP_PCT*100}%")
    _logger.info(f"  - Take Profit: {TAKE_PROFIT_PCT*100}% | SL: {STOP_LOSS_CATASTROPHIC*100}%")
    _logger.info(f"Seguridad:")
    _logger.info(f"  - Max Daily Loss: {MAX_DAILY_LOSS_PCT*100}%")
    _logger.info(f"  - Max Consecutive Losses: {MAX_CONSECUTIVE_LOSSES}")
    _logger.info("=" * 60)


if __name__ == "__main__":
    print_config()
    if validate_config():
        print("[OK] Configuracion valida")
    else:
        print("[ERROR] Configuracion invalida")
