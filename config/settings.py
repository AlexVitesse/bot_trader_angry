"""
Configuracion central del bot (motor V2, BTC/USDT, Binance Futures 4h).

Limpiado 2026-09 (PLAN_MEJORAS_2026-09 Fase 6.2): se borraron los bloques del
scalper v6.7, V8.4, V8.5, V9, V13.03/04 y V14. Siguen en el historial de git.
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

# Capital inicial: solo para el log de arranque y el balance antes del primer
# fetch. El sizing usa el balance real del exchange.
INITIAL_CAPITAL = 100.0

# Comisiones (Binance Futures)
COMMISSION_RATE = 0.0004  # 0.04% taker fee

# Slippage estimado (diferencia entre precio esperado y precio real de ejecucion)
SLIPPAGE_PCT = 0.0001    # 0.01% estimado para BTC/USDT (alta liquidez)

# =============================================================================
# TELEGRAM (Alertas)
# =============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# =============================================================================
# BOT EN VIVO
# =============================================================================
ML_DB_FILE = DATA_DIR / "ml_bot.db"

# Risk Management
ML_MAX_CONCURRENT = 1       # BTC solo: una posicion. El multi-par se rechazo
                            # (experiments/portfolio_sim/README.md resultado 3)
# =============================================================================
# RIESGO — PERFIL AGRESIVO, decision del usuario 2026-08-10
# =============================================================================
# Calibrado con el simulador de CARTERA (experiments/portfolio_sim/), que ya
# modela fills al open siguiente, max_bars del motor, equity compartido y
# margen finito. Config asumida por la calibracion, y por tanto OBLIGATORIA
# mas abajo: BTC solo + f_enable_short=False.
#
# Expectativa realista (walk-forward, 6 folds de test 2021-2026):
#   +12.6%/año | 2/6 folds positivos | peor fold -12.5% | DD max por fold 29.3%
#   Historia completa 2019-2026: +29.0%/año, DD 40.4% (optimista: incluye train)
#
# El DD cede por decision explicita: se prioriza retorno sobre drawdown.
ML_MAX_DD_PCT = 0.45        # por encima del DD historico (40.4%): corta solo si
                            # es peor que cualquier cosa vista, no en mala racha
ML_MAX_DAILY_LOSS_PCT = 0.10  # peor trade ~1.05x riesgo = ~4.7%; 10% da margen
ML_RISK_PER_TRADE = 0.045   # 4.5% -> notional 0.75x-1.8x del equity
# Backstop contra stops patologicamente estrechos, NO un limite operativo: con
# risk 4.5% y trail_dist 2.5-6% el notional pide 0.75x-1.8x del equity.
ML_MAX_NOTIONAL_PCT = 2.0   # Cap notional por trade, en multiplos del balance

# Leverage por regimen. OJO a un punto que se confunde facil: con sizing basado
# en riesgo el notional lo fija ML_RISK_PER_TRADE/SL, NO el leverage. El leverage
# solo decide cuanto MARGEN hay que bloquear. Subirlo no aumenta la exposicion,
# libera margen. Con notional hasta 1.8x del equity hace falta >=5x para que el
# margen (36%) quepa en el buffer del yield manager (40%).
ML_LEVERAGE = {'BULL': 5, 'BEAR': 5, 'RANGE': 5}

# TP/SL del modo 'default' (legacy). V2 manda los suyos en la senal; esto solo
# aplica a una posicion del exchange sin registro en la DB (adopcion huerfana).
ML_TP_PCT = 0.03
ML_SL_PCT = 0.015

# Trailing del modo 'default' (legacy; V2 usa trail_mode='tight')
ML_TRAILING_ACTIVATION = 0.5  # Activar al 50% del TP (1.5% profit)
ML_TRAILING_LOCK = 0.4        # Proteger 40% de ganancia desde peak

# Max hold (velas de 4h) si la senal no trae max_bars
ML_MAX_HOLD = {'BULL': 30, 'BEAR': 30, 'RANGE': 15}

# Timing
ML_CHECK_INTERVAL = 30      # Segundos entre checks de posiciones
ML_CANDLE_HOURS = [0, 4, 8, 12, 16, 20]  # 4h candle close hours UTC

# Entrada maker: limit post-only (GTX) al mejor precio del libro; lo que no se
# llene en este tiempo se cancela y va a market. Binance futures: maker 0,02% vs
# taker 0,04% (demo-fapi, fapiPrivateGetCommissionRate, 2026-09-23). 0 = market.
ML_ENTRY_LIMIT_TIMEOUT_S = 60


# =============================================================================
# VERSION DEL BOT (centralizado para Telegram y logs)
# =============================================================================
BOT_VERSION = "V15"  # Cambiar aqui para actualizar todos los mensajes

# ============================================================================
# V2 paper trading 3 meses (2026-05-19 start) — ver docs/PLAN_PAPER_3MESES.md
# ============================================================================
# Tier 1 (KEEP 3/3 in-sample + PASS OOS sólido): BTC, BNB
# Tier 2 (MARGINAL in-sample, indeterminado OOS por muestra pequeña):
#         DOGE, ETH, OP
# 17 monedas rechazadas (REJECT 0/3 o WEAK 1/3) — fuera del bot.
# Motor: V2 = A (Donchian trend LONG) + F (vol-compression breakout bidir)
# 2026-08-10: reducido a BTC. El walk-forward REAL por par
# (experiments/portfolio_sim/run_walkforward.py) mostro que la ventaja multi-par
# venia ENTERA del fold de 2021: excluyendolo, BTC solo (+40.7%) supera a la
# seleccion multi-par (+34.2%). Con correlacion media 0.69 entre los 5 pares las
# posiciones concurrentes son una sola apuesta apalancada, no diversificacion.
# Ademas 4 de los 5 fallaron significancia individual en experiments/v2_all_coins
# (BNB p=0.315, ETH p=0.472, OP p=0.539, DOGE falla edge-vs-null).
# Detalle: experiments/portfolio_sim/README.md resultado 3.
ML_V15_PAIRS = [
    'BTC/USDT',     # el unico par que paso 3/3 en v2_all_coins
]
ML_V15_SIZING = {
    'BTC/USDT':  1.0,
}
# Motor por par. Antes el enrutado a V2 dependia de que existiera
# strategies/btc_v15/models/meta_v2_paper.json: borrarlo resucitaba en silencio
# el GBM SHORT (AUDITORIA_2026-09 §5). Un par sin motor aqui no opera.
ML_V15_ENGINE = {
    'BTC/USDT': 'v2',
}

# ============================================================================
# YIELD MANAGER — estilo Mercado Pago (capital ocioso genera yield)
# ============================================================================
# V2 es selectivo (~95% del capital ocioso). Mientras espera, el capital se
# va a Binance Simple Earn Flexible USDT (~2-5% APY). Cuando V2 fira, el
# rebalanceo automatico devuelve capital al futures wallet.
#
# Testnet: simula yield virtualmente (testnet no tiene Earn). Mainnet: opera
# con la API real de Binance.
# Apagado 2026-09-22 (AUDITORIA_2026-09 §3.3): nunca probado en mainnet, mueve
# capital real y el sizing no ve el saldo en Earn. Reactivar solo tras la
# Fase 5.4 de docs/PLAN_MEJORAS_2026-09.md.
YIELD_MANAGER_ENABLED = False
YIELD_CONFIG = {
    'enabled': YIELD_MANAGER_ENABLED,
    'simulate_mode': None,            # None = auto-detect (True si testnet)
    # Subido 2026-08-10: con risk 4.5% el notional llega a 1.8x del equity, o
    # sea 36% de margen a 5x. Con el buffer al 20% el trade fallaba por falta de
    # margen (y open_position NO rescata de Earn: Fase 4.1 sin hacer).
    # El coste en yield es minimo: mover 20% extra del Earn al wallet cuesta
    # ~0.6% del capital al ano, y evita perder senales validas.
    'buffer_target_pct': 0.40,        # 40% en futures wallet (cubre 36% de margen)
    'buffer_max_pct': 0.50,           # >50% en futures -> sweep excess a Earn
    'buffer_min_pct': 0.30,           # <30% en futures -> redeem de Earn
    'rebalance_interval_s': 600,      # rebalance cada 10 min
    'simulate_apy': 0.03,             # 3% APY USDT (Binance Earn flexible tipico)
    'min_sweep_amount': 50.0,         # no sweep si <$50 (overhead)
    'state_file': 'data/yield_state.json',
    'earn_asset': 'USDT',
    'earn_product_id': 'USDT001',     # Binance Earn flexible USDT product id
}

def validate_config() -> bool:
    """Valida que la configuracion este completa."""
    errors = []
    if not BINANCE_API_KEY:
        errors.append("BINANCE_API_KEY no configurada")
    if not BINANCE_API_SECRET:
        errors.append("BINANCE_API_SECRET no configurada")
    if not 0 < ML_RISK_PER_TRADE <= 0.10:
        errors.append(f"ML_RISK_PER_TRADE fuera de rango: {ML_RISK_PER_TRADE}")
    if not 0 < ML_MAX_DD_PCT < 1:
        errors.append(f"ML_MAX_DD_PCT fuera de rango: {ML_MAX_DD_PCT}")
    for pair in ML_V15_PAIRS:
        if ML_V15_ENGINE.get(pair) != 'v2':
            errors.append(f"{pair} sin motor en ML_V15_ENGINE")
    for e in errors:
        logging.getLogger(__name__).error(f"[CONFIG] {e}")
    return not errors


def print_config():
    """Imprime la configuracion actual."""
    _logger = logging.getLogger(__name__)
    _logger.info(f"Modo: {TRADING_MODE.upper()} | Pares: {ML_V15_PAIRS} | "
                 f"Motor: {ML_V15_ENGINE}")
    _logger.info(f"Riesgo/trade: {ML_RISK_PER_TRADE:.1%} | Kill DD: "
                 f"{ML_MAX_DD_PCT:.0%} | Daily loss: {ML_MAX_DAILY_LOSS_PCT:.0%} | "
                 f"Leverage: {ML_LEVERAGE}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_config()
    print("[OK] Configuracion valida" if validate_config()
          else "[ERROR] Configuracion invalida")
