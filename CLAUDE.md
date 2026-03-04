# CLAUDE.md - Bot de Trading ML V14

## Objetivo del Proyecto

### Filosofía Central

> "La idea es crear un bot tomando lo que hacen muchos pero buscando un enfoque diferente. No buscamos la riqueza sino más bien la rentabilidad — que no sea una ganancia de 10% anual porque por el riesgo no vale la pena. Debemos buscar lo que hacen bien los traders y aplicarlo."

**Lo que esto significa concretamente:**

1. **El umbral mínimo es 30-40%+ anual** — crypto tiene riesgo significativo. Un 10% anual lo da un ETF de S&P 500 sin riesgo de liquidación, hacks o volatilidad extrema. Si el bot no supera ampliamente eso, no tiene sentido operarlo.

2. **Inspirado en traders reales, no en ML teórico** — los mejores traders no predicen el futuro con ML. Identifican: confluencia de señales técnicas, gestión de riesgo disciplinada, regímenes de mercado claros, y size apropiado al conviction level.

3. **Enfoque diferente al típico bot ML** — no buscar el "mejor AUC" ni el backtest más brillante. Buscar consistencia: pocos trades de alta calidad, gestión de drawdown estricta, y adaptación al régimen de mercado.

4. **El riesgo debe ser recompensado** — cada trade debe tener una razón clara basada en confluencia técnica + régimen + tamaño de posición proporcional al edge.

---

Bot de trading automatizado en Binance Futures (4h timeframe, futuros perpetuos).
**Estado actual: Paper trading en testnet** — acumulando datos reales antes de capital real.

### Métricas mínimas para producción real
| Métrica | Mínimo | Target |
|---------|--------|--------|
| Retorno Anual | 30% | 50-100% |
| Win Rate | 50% | 55-60% |
| Profit Factor | 1.3 | 1.5+ |
| Max Drawdown | < 25% | < 15% |

> **Expectativas realistas** (según historial): WR 50-55%, PF 1.2-1.5. Si el backtest muestra 70%+ WR → sospechar overfitting. Si el retorno real es < 30% anual → el riesgo crypto no lo justifica.

---

## Historia Crítica: Patrón de Overfitting Repetido

**CADA versión con backtest brillante ha fallado en producción:**

| Versión | Backtest | Producción Real | Problema |
|---------|----------|-----------------|----------|
| V7 Original | "Bueno" | 33-42% WR | Falló |
| V9 LossDetector | 68% WR | 41.4% WR | Falló |
| BTC V2 | 65.7% WR | 43.8% WR (2026) | Degradación |
| SOL V2 | 63.6% WR | 12.5% WR | Catástrofe |
| V13.03 | 67.3% WR | ??? | Sin validar |

**Por qué sucede:** Correlación train/test cae 89-94% en todos los pares. El modelo memoriza 2020-2024, pero el mercado de 2025-2026 es diferente. Grid search de TP/SL con datos históricos completos = look-ahead bias clásico.

**Lección clave de LOW_OVERFIT_MODEL_RESULTS.md:** Solo BTC (0.1% drop) y ADA (4.3% drop) tienen overfitting bajo. El resto: DOGE 79.9%, DOT 95.8%, NEAR 120.6%.

---

## ETH: Históricamente Excluido

ETH ha fallado en TODAS las versiones:
- V13: excluido de ML_PAIRS
- Low-Overfit experiments: "ETH: No funciona con este modelo" (WR < 40%)
- V14 Expert (ethusdt_v14): entrenado pero **status: NEEDS_REVIEW**, total_pnl -996, AUC ~0.5
- SHORT trades históricamente: 0% WR en período alcista

**Consecuencia:** Conectar los modelos ethusdt_v14 al bot con cualquier umbral no agrega valor — AUC ~0.5 = predicción aleatoria.

---

## SHORT Direction: No Aprobado para Altcoins

Múltiples documentos confirman:
- SHORT trades = 0% WR en períodos alcistas (mercado crypto tiene sesgo alcista)
- Low-Overfit experiments: "LONG_ONLY es obligatorio para evitar pérdidas"
- SHORT ensemble entrenado en esta sesión (DOGE/ADA/DOT/SOL): 4-5/12 folds positivos, WR 35-43%, ADA -222%, SOL -200% → **no aprobado**
- Sin validación cruzada entre activos (requisito obligatorio)

---

## Arquitectura V14 Aprobada (en main)

### BTC: Reglas técnicas + ML confianza
```
Datos → Régimen → Setup → Ensemble ML confianza → Position sizing
```
- 4 regímenes: TREND_UP / TREND_DOWN / RANGE / VOLATILE
- 8 setups técnicos por régimen (pullback, rally, bounce, rejection, breakout...)
- Ensemble confianza: 3 modelos (context/momentum/volume), umbral skip < 0.35
- TP 3% / SL 1.5% — **validado: 12/12 folds, cross-asset ETH +2829%**

### ETH: Setups simples (reglas, SIN ML)
- Regla: RSI<30 o vol_ratio>2
- **BUG CONOCIDO en main**: RSI<30 genera señal SHORT (debería ser LONG — oversold = comprar)
- ETH dispara raramente — contribuye a 0 trades
- No conectar modelos ethusdt_v14 (AUC ~0.5, status NEEDS_REVIEW)

### DOGE/ADA/DOT/SOL: Ensemble Voting LONG only
```
Datos → Features → 3 modelos votan → Trade si 2/3 coinciden
```
- RandomForest + GradientBoosting + LogisticRegression
- **LONG only**: memecoins/altcoins tienen sesgo alcista
- Validados: DOGE 7/9 folds+ (cross: SHIB/PEPE ✓), ADA 11/12 (cross: DOT/SOL/ATOM ✓), DOT 6/8
- Features: rsi, macd_norm, adx, bb_pct, atr_pct, ret_3, ret_5, ret_10, vol_ratio, trend
- TP 6%/SL 4% (DOGE/ADA), TP 5%/SL 3% (DOT)

### Cross-pairs (aprobados con walk-forward propio)
| Modelo base | Cross-pairs validados | WF folds |
|------------|----------------------|----------|
| ADA | SOL 9/10, ATOM 8/10, AVAX 8/10, POL 8/10 | ✓ |
| DOGE | SHIB 6/10, PEPE 6/10, FLOKI 6/10 | ✓ |
| DOT | LINK 7/10, ALGO 7/10, FIL 6/10, NEAR 6/10 | ✓ |

**Rechazados** (< 60% folds): INJ, BONK, WIF.

---

## Estado de Ramas

| Rama | Usar | Descripción |
|------|------|-------------|
| `main` | ✓ Producción | V14 validado: BTC+ETH setups + ensemble LONG |
| `feature/v14.1-bidirectional` | ✗ Experimental | ETH ML (AUC~0.5) + SHORT ensemble (fallido) — NO mergear |

### Qué tiene `feature/v14.1-bidirectional` que NO va a main
1. ETH conectado a ethusdt_v14 models — AUC ~0.5, no predice nada útil
2. Modelos SHORT DOGE/ADA/DOT/SOL — WR < 40%, sin cross-asset validation
3. `train_ensemble_short.py` — útil como referencia, pero los modelos necesitan pasar validación

---

## Problema Activo: 0 Trades en Producción

El bot corre pero no genera trades. Causas conocidas:

1. **BTC setups muy estrictos**: rsi14<40 AND bb_pct<0.3 AND ema200_dist>0 simultáneamente
2. **ETH RSI bug**: RSI<30 → SHORT (invertido), vol_ratio>2 raro
3. **Filtro ML confianza BTC**: si models no están cargando bien → skip todos
4. **Log level**: confirmar que los `[V14]` INFO messages son visibles

**Diagnóstico**: `/log` en Telegram y buscar líneas `[V14]` — debe haber al menos un log por par por vela 4h.

---

## Requisitos de Validación — Obligatorios

Antes de agregar cualquier modelo/dirección a main:

1. **Walk-forward**: ≥ 7/12 folds positivos (o 6/10 para cross-pairs)
2. **Cross-asset**: probar modelo en activos correlacionados, todos positivos
   - DOGE → SHIB, PEPE (o similares)
   - ADA → DOT, SOL, ATOM
   - SHORT → equivalentes SHORT de los mismos activos
3. **Win Rate > break-even**: WR > SL/(TP+SL)
   - Con TP 6% / SL 4%: necesita WR > 40%
   - Con TP 3% / SL 1.5%: necesita WR > 33%
4. **Documentar resultados** antes de mergear

**Regla de METODOLOGIA_TESTING.md:**
> Un modelo con métricas malas NO se arregla con umbral más estricto. Se rechaza o se reentrena con mejor metodología.

---

## Estructura del Proyecto

```
src/
  ml_bot.py              # Bot principal (loop 30s, señales 4h)
  ml_strategy_v14.py     # Motor de señales V14 (archivo crítico)
  portfolio_manager.py   # Gestión posiciones + trailing stop
  telegram_alerts.py     # Alertas + TelegramPoller

config/
  settings.py            # BOT_VERSION, ML_V14_EXPERTS, ML_V14_FEATURES

strategies/
  btc_v14/models/        # context/momentum/volume _long/_short.pkl (aprobados)
  eth_v14/models/        # NEEDS_REVIEW — no usar en bot
  doge_v14/models/       # rf/gb/lr + scaler (LONG aprobados) + _short (NO aprobados)
  ada_v14/models/        # ídem
  dot_v14/models/        # ídem (sin lr.pkl)
  sol_v14/models/        # ídem

docs/                    # LEER ANTES DE CAMBIOS
  ARQUITECTURA_V14.md
  ANALISIS_CRITICO_OVERFITTING.md   ← crítico
  LOW_OVERFIT_MODEL_RESULTS.md      ← crítico
  WALKFORWARD_VALIDATION_RESULTS.md ← crítico
  METODOLOGIA_TESTING.md            ← crítico

METODOLOGIA_TESTING.md  # En raíz — proceso obligatorio de validación

Scripts de entrenamiento:
  train_btc_v14.py          # BTC (LONG+SHORT, ambos validados)
  train_expert.py           # ETH expert (generó ethusdt_v14 — NEEDS_REVIEW)
  train_ensemble_voting.py  # DOGE/ADA/DOT/SOL LONG (aprobados)
  train_ensemble_short.py   # SHORT (entrenados pero NO validados cross-asset)
  ml_export_v14.py          # Export/reentrenamiento mensual
  validate_new_pairs.py     # Validación WF + sintéticos para nuevos pares
```

---

## Entornos Python — Crítico

```
Claude Code bash:    C:\Python\python.exe (sklearn 1.6.1)
Bot producción:      C:\Users\pcdec\AppData\Local\pypoetry\Cache\
                     virtualenvs\binance-scalper-bot-ofXWUGOe-py3.12\
                     Scripts\python.exe (sklearn 1.8.0)
```

- **Entrenar modelos SIEMPRE con el venv de producción** (sklearn 1.8.0)
- Modelos del venv dan InconsistentVersionWarning en Claude bash (no es error)
- `poetry install` desde Claude bash instala en entorno INCORRECTO
- Para instalar deps: usar pip.exe del venv completo

---

## Comandos Telegram

| Comando | Acción |
|---------|--------|
| `/status` | Balance, posición, trades hoy |
| `/log` | Últimas líneas del log principal |
| `/log 1` | Log rotado (ml_bot.log.1) |
| `/resume` | Reanudar bot pausado |
| `/export_v14` | Reentrenar modelos V14 |

---

## Reglas para Claude

### Principio guía
El objetivo no es el mejor modelo técnico posible, sino un sistema que genere **30%+ anual con drawdown controlado**, inspirado en lo que hacen los traders exitosos: confluencia de señales, disciplina en el sizing, respeto del régimen de mercado. Un modelo técnicamente sofisticado que no cumple esta meta es inútil.

### Antes de cualquier cambio
1. **Leer TODA la documentación relevante** antes de opinar o proponer
   - Mínimo: ARQUITECTURA_V14.md, METODOLOGIA_TESTING.md, ANALISIS_CRITICO_OVERFITTING.md
   - No asumir el estado del código — leerlo
2. **Preguntar el objetivo** si no está claro antes de implementar
3. **No proponer código sin leer los archivos** que se van a modificar

### Sobre modelos ML
- Si un modelo tiene AUC ~0.5 → no conectarlo, no "filtrarlo mejor" → descartarlo
- Si un modelo falla walk-forward → rechazarlo o reentrenar, no usarlo con umbral estricto
- Cualquier modelo nuevo: walk-forward ≥ 7/12 + cross-asset + documentar
- ETH ML models (ethusdt_v14) = NEEDS_REVIEW, no conectar al bot sin reentrenamiento

### Sobre dirección SHORT
- El historial del proyecto muestra que SHORT en altcoins/memecoins no funciona
- SHORT necesita mercados bajistas sostenidos — en crypto el sesgo es alcista
- Para aprobar SHORT: WR > break-even + walk-forward 7/12 + cross-asset validation

### Sobre ETH
- ETH ha sido excluido en TODOS los experimentos históricos
- Los setups simples actuales (main) tienen un BUG: RSI<30 genera SHORT (debería ser LONG)
- Corregir el bug es válido. Conectar los modelos ML fallidos no lo es.

### Git workflow
- **main**: solo modelos validados. No mergear código experimental.
- Feature branches para experimentos. Documentar validación antes del merge.
- `nul` (Windows reserved name) bloquea git add → usar `git add --all -- ":!nul"`

---

## Próximos Pasos Prioritarios

1. **Diagnóstico 0 trades** (prioridad máxima)
   - Revisar `/log` en Telegram — buscar líneas `[V14]`
   - ¿Llegan señales de BTC? ¿Se filtran por confianza?
   - ¿El ensemble de DOGE/ADA/DOT/SOL genera votes?

2. **Fix bug ETH** (opcional, bajo riesgo)
   - En `check_eth_setups()`: cambiar RSI<30 de SHORT a LONG
   - Es una línea en main directamente (no requiere rama)

3. **NO hacer hasta tener trades reales que analizar**
   - No agregar más pares
   - No habilitar SHORT
   - No conectar ETH ML
   - El paper trading es la única validación real

4. **Reentrenamiento mensual** (cuando corresponda)
   - Usar `ml_export_v14.py`
   - Verificar walk-forward antes de deploy
