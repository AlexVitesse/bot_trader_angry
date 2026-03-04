# CLAUDE.md - Bot de Trading ML V14

## Objetivo del Proyecto

Bot de trading automatizado en Binance Futures (4h timeframe).
**Estado actual: Paper trading en testnet** — validando antes de capital real.

### Métricas mínimas para ir a producción real
| Métrica | Mínimo | Target |
|---------|--------|--------|
| Win Rate | 50% | 60%+ |
| Profit Factor | 1.5 | 2.0+ |
| Max Drawdown | < 25% | < 15% |

---

## Arquitectura V14 (APROBADA)

### BTC: Reglas técnicas + ML confianza
```
Datos → Detectar Régimen → Detectar Setup → Ensemble ML confianza → Sizing
```
- Régimen: TREND_UP / TREND_DOWN / RANGE / VOLATILE (ADX, CHOP, BB)
- 8 setups técnicos (pullback, rally, bounce, rejection, breakout...)
- Ensemble = 3 modelos (context/momentum/volume) calculan confianza
- **Umbral de confianza**: 0.35 mínimo para ejecutar (skip si < 0.35)
- TP 3% / SL 1.5% — validado: 12/12 folds positivos, +1126% PnL

### ETH: PENDIENTE — ver sección estado actual
- Los modelos `ethusdt_v14` tienen **status: NEEDS_REVIEW**, total_pnl -996
- Conectados en `feature/v14.1-bidirectional` pero NO validados
- NO mergear a main hasta reentrenar con cross-asset validation

### DOGE/ADA/DOT/SOL: ML Ensemble Voting (LONG only — APROBADO)
```
Datos → Features → 3 modelos votan → Tradear si 2/3 coinciden
```
- RF + GradientBoosting + LogisticRegression
- Diseñado LONG only: memecoins/altcoins tienen sesgo alcista
- Resultados validados LONG: DOGE 7/9 folds+, ADA 11/12, DOT 6/8
- Cross-pairs (ATOM/AVAX/POL, PEPE/FLOKI, LINK/ALGO/FIL/NEAR): heredan modelo base
- TP 6% / SL 4% (DOGE/ADA), 5% / 3% (DOT)

### SHORT en ensemble: NO APROBADO
- Modelos SHORT (rama `feature/v14.1-bidirectional`) fallaron walk-forward:
  - DOGE: 4/12 folds+ | ADA: 5/12 (-222%) | DOT: 4/12 (-52%) | SOL: 4/12 (-200%)
- **Sin validación cruzada** (DOGE necesita SHIB/PEPE, ADA necesita DOT/SOL/ATOM)
- NO usar en producción hasta reentrenar y validar cross-asset

---

## Reglas de Validación — OBLIGATORIAS antes de producción

Toda nueva estrategia/modelo/dirección DEBE pasar:

1. **Walk-forward**: ≥ 7/12 folds positivos (58% mínimo, 70% objetivo)
2. **Cross-asset validation**: probar modelo de un activo en activos correlacionados
   - DOGE SHORT → validar con SHIB, PEPE, FLOKI
   - ADA SHORT → validar con DOT, SOL, ATOM
   - Todos deben ser positivos (PnL > 0)
3. **Resultados documentados** antes de mergear a `main`

**Regla clave de METODOLOGIA_TESTING.md:**
> "Aplicar filtros generales es fórmula para fracasar"
> Un modelo con WR < 40% NO se arregla poniendo umbral más estricto. Se rechaza o se reentrena.

---

## Estado Actual (2026-03-03)

### Ramas
| Rama | Estado | Descripción |
|------|--------|-------------|
| `main` | Producción (testnet) | V14 con BTC + ensemble LONG |
| `feature/v14.1-bidirectional` | Experimental | ETH ML + SHORT ensemble (NO validados) |

### Modelos aprobados en `main`
| Expert | Tipo | Walk-Forward | Cross-Asset | Aprobado |
|--------|------|-------------|-------------|----------|
| BTC V14 | Setups + ML | 12/12 | ETH +2829% | ✓ |
| DOGE V14 | Ensemble LONG | 7/9 | SHIB/PEPE + | ✓ |
| ADA V14 | Ensemble LONG | 11/12 | DOT/SOL/ATOM + | ✓ |
| DOT V14 | Ensemble LONG | 6/8 | — | ✓ |
| SOL V14 | Ensemble LONG | Validado | — | ✓ |
| ETH V14 | ML | 5/12 FAIL | — | ✗ NEEDS_REVIEW |
| SHORT ensemble | — | 4/12 FAIL | NO hecha | ✗ NO APROBADO |

### Problema activo: 0 trades en producción
- El bot corre pero no genera trades
- Posibles causas: setup conditions BTC muy estrictas, ETH simple rules raramente disparan
- Debug: revisar logs con `/log` (Telegram) — buscar líneas `[V14]`

---

## Estructura del Proyecto

```
src/
  ml_bot.py            # Bot principal (loop 30s, señales 4h)
  ml_strategy_v14.py   # Motor de señales V14
  portfolio_manager.py # Gestión posiciones + trailing stop
  telegram_alerts.py   # Alertas + TelegramPoller (comandos /status etc)

config/
  settings.py          # BOT_VERSION, ML_V14_EXPERTS, ML_V14_FEATURES

strategies/
  btc_v14/models/      # context/momentum/volume _long/_short.pkl
  eth_v14/models/      # ensemble_*_long/short.pkl + regime_detector.pkl (NEEDS_REVIEW)
  doge_v14/models/     # rf.pkl, gb.pkl, lr.pkl, scaler.pkl (+ _short — NO validados)
  ada_v14/models/      # ídem
  dot_v14/models/      # ídem (sin lr.pkl)
  sol_v14/models/      # ídem

scripts de entrenamiento:
  train_btc_v14.py         # BTC (LONG+SHORT, ambos validados)
  train_expert.py          # ETH (NEEDS_REVIEW)
  train_ensemble_voting.py # DOGE/ADA/DOT/SOL LONG
  train_ensemble_short.py  # SHORT (NO validados — no usar en producción)
  ml_export_v14.py         # Export/reentrenamiento mensual

logs/
  ml_bot.log           # RotatingFileHandler 5MB x 5 archivos

data/
  {PAIR}USDT_4h.csv    # Datos históricos descargados
```

---

## Entornos Python — CRÍTICO

```
Claude Code bash:     C:\Python\python.exe (sklearn 1.6.1)
Bot producción:       C:\Users\pcdec\AppData\Local\pypoetry\Cache\
                      virtualenvs\binance-scalper-bot-ofXWUGOe-py3.12\
                      Scripts\python.exe (sklearn 1.8.0)
```

- **Entrenar modelos siempre con el venv de producción** (sklearn 1.8.0)
- Pickle incompatible entre versiones — modelos del venv pueden dar warning en Claude bash
- Para instalar deps en producción: usar pip.exe del venv, NO poetry desde bash de Claude

---

## Comandos Telegram

| Comando | Acción |
|---------|--------|
| `/status` | Balance, posición, trades hoy |
| `/log` | Últimas líneas del log principal |
| `/log 1` | Log rotado (ml_bot.log.1) |
| `/resume` | Reanudar bot pausado |
| `/export_v14` | Reentrenar modelos V14 (~5 min) |

---

## Reglas para Claude

### ANTES de cualquier cambio de código
1. Leer los archivos relevantes — nunca asumir el estado del código
2. Si el usuario dice "REVISA BIEN" — hacer revisión exhaustiva antes de opinar
3. Confirmar cuál es el objetivo ANTES de proponer una solución

### Sobre modelos ML
- Un modelo con métricas malas NO se "arregla" con umbrales más estrictos
- Cualquier nuevo modelo/dirección DEBE pasar walk-forward (7+/12) + cross-asset
- Si un modelo falla validación: rechazarlo o reentrenarlo, no conectarlo igual

### Sobre SHORT para ensemble
- Los modelos SHORT actuales (DOGE/ADA/DOT/SOL) NO están aprobados
- Para aprobarlos se necesita: mejores WR (>40%) + cross-asset validation
- Considerar que el sesgo alcista de altcoins hace SHORT estructuralmente difícil

### Sobre ETH
- Los modelos ETH ML (ethusdt_v14) fallaron validación con -996 PnL
- AUC ~0.5 = esencialmente aleatorio
- Opciones: reentrenar con mejores features/parámetros, o mantener como reglas simples
- NO conectar modelos fallidos con "umbral estricto" como workaround

### Git workflow
- Ramas de feature: `feature/v14.1-*`
- Mergear a `main` solo cuando walk-forward + cross-asset OK
- El archivo `nul` (Windows reserved name) bloquea git add — usar `git add --all -- ":!nul"`

---

## Próximos pasos reales

1. **Debugging 0 trades**: confirmar que el bot en producción genera señales (revisar `/log`)
2. **ETH**: decidir — reentrenar modelos con mejores features O mantener reglas simples
3. **SHORT ensemble**: solo proceder si:
   - Walk-forward ≥ 7/12 folds
   - Cross-asset validation positiva
   - Win Rate > 40% (break-even con TP 6% / SL 4%)
4. **Paper trading**: acumular 2-4 semanas de datos antes de evaluar
