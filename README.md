# Bot de Trading ML - Binance Futures

Bot de trading automatizado para Binance Futures usando Machine Learning.

## Estrategia Actual: V13 CORE

Sistema de trading basado en:
- **Modelos V7**: LightGBM regressors por par (prediccion de retorno 4h)
- **V8.5 ConvictionScorer**: Evalua calidad de cada trade
- **Filtros**: Choppiness Index, conviction minima

### Pares activos
```
XRP/USDT, ETH/USDT, DOGE/USDT, ADA/USDT
AVAX/USDT, LINK/USDT, DOT/USDT, NEAR/USDT
```

### Configuracion
| Parametro | Valor |
|-----------|-------|
| Timeframe | 4h |
| Max posiciones | 3 |
| TP / SL | 3% / 1.5% |
| Trailing Stop | 50% activacion, 40% lock |
| Risk por trade | 2% |

## Requisitos

- Python 3.11+
- Poetry
- Cuenta Binance Futures (testnet o live)

## Instalacion

```bash
# Clonar repo
git clone https://github.com/AlexVitesse/bot_trader_angry.git
cd bot_trader_angry

# Instalar dependencias
poetry install

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys
```

## Configuracion .env

```env
# Modo: testnet o live
TRADING_MODE=testnet

# Binance Testnet
BINANCE_TESTNET_API_KEY=tu_api_key
BINANCE_TESTNET_API_SECRET=tu_api_secret

# Binance Live (cuando estes listo)
BINANCE_API_KEY=
BINANCE_API_SECRET=

# Telegram (opcional)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

## Uso

```bash
# Ejecutar bot ML
poetry run python -m src.ml_bot

# Backtest
poetry run python backtest_max_positions.py

# Exportar modelos (reentrenamiento)
poetry run python ml_export_models.py
```

## Comandos Telegram

| Comando | Descripcion |
|---------|-------------|
| `/help` | Lista de comandos |
| `/status` | Estado actual del bot |
| `/resume` | Reanudar si esta pausado |
| `/log` | Ultimas lineas del log |
| `/backup` | Backup de BD y modelos |
| `/clearlog` | Borrar archivo de log |
| `/resetdb` | Borrar trades antiguos |
| `/update` | Pull + Install + Restart |
| `/pull` | Git pull (con stash) |
| `/restart` | Reiniciar bot |
| `/retrain` | Reentrenar modelos |

## Estructura del Proyecto

```
├── config/
│   └── settings.py      # Configuracion central
├── src/
│   ├── ml_bot.py        # Bot principal
│   ├── ml_strategy.py   # Estrategia ML
│   ├── portfolio_manager.py
│   └── telegram_alerts.py
├── models/              # Modelos entrenados (.pkl)
├── data/                # Datos OHLCV (.parquet)
├── logs/                # Logs del bot
├── docs/                # Documentacion
└── archive_scripts/     # Scripts obsoletos
```

## Metricas Objetivo

| Metrica | Minimo | Target |
|---------|--------|--------|
| Win Rate | 50% | 60%+ |
| Profit Factor | 1.5 | 2.0+ |
| Max Drawdown | <25% | <15% |

## Backtest V13 (Feb 2026)

```
Periodo: 14 dias
Trades: 161
Win Rate: 64%
Profit Factor: 3.41
Max Drawdown: 18.3%
```

## Roadmap

- [x] V13 CORE en produccion
- [ ] Validacion 14 dias (hasta Mar 12, 2026)
- [ ] Macro LLM Bot (proyecto futuro)

## Disclaimer

Este bot es para uso educativo y experimental. Trading de criptomonedas con apalancamiento conlleva riesgo significativo de perdida. No inviertas mas de lo que puedes permitirte perder.

## Licencia

Privado - Uso personal
