# Bot de Trading ML — Binance Futures

Bot de trading automatizado para Binance Futures (4h, futuros perpetuos) usando
Machine Learning + reglas. **Estado: paper trading en testnet.**

> **Fuentes de verdad del proyecto:**
> - [`CLAUDE.md`](CLAUDE.md) — estado, arquitectura, reglas de trabajo.
> - [`docs/AUDITORIA_2026-05.md`](docs/AUDITORIA_2026-05.md) — auditoría: qué
>   funciona, qué no, y validación por par.

## Estrategia actual: V15 Multi-Par

| Tipo | Pares | Lógica |
|------|-------|--------|
| ML | BTC | GBM SHORT + Breakout / Pullback EMA20 LONG |
| Reglas | ETH | BTC-follower + Breakout + SHORT multi-confirmación |
| Reglas trailing | 20 altcoins | BTC-follower LONG + BTC-breakdown SHORT, trailing stop tight |

`config/settings.py` define los 22 pares activos en `ML_V15_PAIRS`.

> ⚠️ **Solo BTC, ETH, ADA y SOL tienen validación documentada creíble.** Los otros
> 18 pares tienen backtest con firma de overfitting (PF 7–20, DD 1–4%) y están
> pendientes de re-validación. No mover capital real a ellos. Ver la auditoría.

### Configuración base

| Parámetro | Valor |
|-----------|-------|
| Timeframe | 4h |
| Máx. posiciones | 3 (`ML_MAX_CONCURRENT`) |
| Sizing | BTC 1.0x · ETH 0.5x · resto 0.3x |
| Máx. drawdown | 20% (`ML_MAX_DD_PCT`) |
| Máx. pérdida diaria | 20% (`ML_MAX_DAILY_LOSS_PCT`) |

### Métricas objetivo

| Métrica | Mínimo | Target |
|---------|--------|--------|
| Retorno anual | 30% | 50–100% |
| Win Rate | 50% | 55–60% |
| Profit Factor | 1.3 | 1.5+ |
| Max Drawdown | < 25% | < 15% |

> Expectativas realistas: WR 50–55%, PF 1.2–1.5. Un backtest con WR 70%+ o PF > 3
> debe tratarse como overfitting hasta demostrar lo contrario.

## Requisitos

- Python 3.12 + Poetry
- Cuenta Binance Futures (testnet o live)

## Instalación

```bash
git clone https://github.com/AlexVitesse/bot_trader_angry.git
cd bot_trader_angry
poetry install
cp .env.example .env   # editar con tus API keys
```

## Configuración .env

```env
TRADING_MODE=testnet
BINANCE_TESTNET_API_KEY=tu_api_key
BINANCE_TESTNET_API_SECRET=tu_api_secret
BINANCE_API_KEY=
BINANCE_API_SECRET=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

## Uso

```bash
poetry run python -m src.ml_bot          # ejecutar el bot
poetry run python train_v15_prod.py      # reentrenar modelo SHORT de BTC
poetry run python evaluate_new_pairs_v15.py  # evaluar pares
```

## Comandos Telegram

| Comando | Descripción |
|---------|-------------|
| `/status` | Balance, posición, trades del día |
| `/log` | Últimas líneas del log (`/log 1` para el rotado) |
| `/resume` | Reanudar bot pausado |
| `/update` | Pull + install + restart |
| `/retrain` | Reentrenar modelos |

## Estructura del proyecto

```
config/settings.py     # Configuración central (BOT_VERSION, ML_V15_PAIRS, ...)
src/
  ml_bot.py            # Bot principal (loop 30s, señales 4h)
  ml_strategy_v15.py   # Motor de señales V15 (ACTIVO)
  ml_strategy_v14.py   # Motor V14 (desactivado, preservado)
  portfolio_manager.py # Gestión de posiciones + trailing stop
  telegram_alerts.py   # Alertas + comandos Telegram
strategies/{coin}_v15/ # meta_v15.json + modelos por par
docs/                  # Documentación V15 + auditoría
docs/archive/          # Documentación histórica (V12-V14)
archive_scripts/       # Scripts de experimentos y versiones previas
```

## Disclaimer

Uso educativo y experimental. El trading de criptomonedas con apalancamiento
conlleva riesgo significativo de pérdida. No inviertas más de lo que puedes
permitirte perder.

## Licencia

Privado — uso personal.
