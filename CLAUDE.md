# CLAUDE.md - Bot de Trading ML V15

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

## Arquitectura V15 Activa (Multi-Par — 22 pares)

> Estado real al 2026-05-19 (rama `v15/multi-pair`). Ver `docs/AUDITORIA_2026-05.md`
> para el análisis crítico de validación.

V15 opera **22 pares** en paper trading testnet (`ML_V15_PAIRS` en `settings.py`).
Tres tipos de motor según el par:

| Tipo | Pares | Lógica | model_type |
|------|-------|--------|------------|
| **ML** | BTC | GBM SHORT (threshold 0.60) + Breakout B / Pullback EMA20 LONG | `GradientBoostingClassifier` |
| **Reglas ETH** | ETH | BTC-follower LONG + Breakout + SHORT multi-conf/BB | `rule_based` |
| **Reglas trailing** | ADA, SOL, DOGE, LINK, AVAX, DOT, NEAR, XRP, ATOM, INJ, ALGO, FIL, 1000SHIB, BNB, LTC, ETC, BCH, UNI, AAVE, OP (20) | BTC-follower LONG + BTC-breakdown SHORT, ambos con trailing stop tight | `rule_based_trailing` |

- **Régimen**: EMA20/EMA50 diario por par, 2% dead zone + recovery filter (close>EMA200)
- **Funding veto**: z-score > 2.0 bloquea LONG, < -1.5 bloquea SHORT
- **Sizing**: BTC 1.0x, ETH 0.5x, resto 0.3x (`ML_V15_SIZING`)
- **Trailing**: `_generate_alt_trailing_signals()` en `ml_strategy_v15.py`; el
  `portfolio_manager` soporta trailing inmediato (`trail_mode='tight'`)

### Estado de validación por par — IMPORTANTE

| Nivel | Pares | Evidencia |
|-------|-------|-----------|
| **Validado (creíble)** | BTC, ETH, ADA, SOL | Doc dedicado + WF + métricas modestas (PF 1.3–2.9, DD 7–43%) |
| **Sin documento — backtest sospechoso** | Los otros 18 | Solo `meta_v15.json`. WF ~100%, PF 7–20, DD 1–4% = firma de overfitting |

> ⚠️ Los 18 pares sin documento **NO cumplen** los requisitos de validación del
> proyecto. Su backtest declara PF de 13–20 (BTC validado: PF 1.35) — métricas
> imposibles que indican overfitting / sesgo de selección. Operarlos en paper
> trading testnet es aceptable (acumula datos), pero **no se debe mover capital
> real** a ellos sin re-validación con WF purgado + bootstrap. Detalle completo
> en `docs/AUDITORIA_2026-05.md` sección B.

### V14 (desactivada, preservada)
V14 sigue en el código pero ML_V14_ENABLED=False. Se puede reactivar si necesario.

---

## Estado de Ramas

| Rama | Usar | Descripción |
|------|------|-------------|
| `main` | ✓ Producción | V14 validado (preservado, ML_V14_ENABLED=False) |
| `v15/multi-pair` | ✓ Deploy actual | V15 multi-par (22 pares) en paper trading |
| `feature/v14.1-bidirectional` | ✗ Experimental | ETH ML (AUC~0.5) + SHORT ensemble (fallido) — NO mergear |

### Flujo de deploy V15
1. `v15/multi-pair` contiene el código V15 multi-par en paper trading testnet
2. Mergear a `main` **solo** cuando: (a) paper trading confirme señales correctas
   y (b) los pares se hayan re-validado con metodología purgada (ver auditoría)
3. V14 preservado y reactivable cambiando flags en settings.py

---

## V15 Deployment Status

- **V15 desplegado para 22 pares** en paper trading testnet (`ML_V15_PAIRS`)
- Solo BTC, ETH, ADA, SOL tienen validación documentada creíble
- Los otros 18 pares: backtest con firma de overfitting — pendientes de re-validar
- V14 desactivado (ML_V14_ENABLED=False)
- Modelos entrenados con sklearn 1.8.0 (producción)
- **Diagnóstico**: `/log` en Telegram y buscar líneas `[V15]` — debe haber logs cada vela 4h
- **Auditoría completa**: `docs/AUDITORIA_2026-05.md`

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
  ml_strategy_v15.py     # Motor de señales V15 Expert Committee (ACTIVO)
  ml_strategy_v14.py     # Motor de señales V14 (desactivado, preservado)
  portfolio_manager.py   # Gestión posiciones + trailing stop
  telegram_alerts.py     # Alertas + TelegramPoller

config/
  settings.py            # BOT_VERSION="V15", ML_V15_ENABLED=True

strategies/
  {coin}_v15/models/     # 22 pares: meta_v15.json por par (BTC además: short_gbm.pkl)
  btc_v14/models/        # V14 (preservado)

V15 Scripts:
  train_v15_prod.py        # Entrenar SHORT model BTC para producción
  v15_framework.py          # Framework compartido (sim, features, WF)
  evaluate_new_pairs_v15.py # Evaluación masiva de pares (ver auditoría)

docs/
  AUDITORIA_2026-05.md     # Auditoría — estado real, overfitting, inspiración GitHub
  V15_COMMITTEE_results.md # Resultados del comité BTC validado
  archive/                 # Documentación de versiones previas (V12-V14)
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

> Basado en `docs/AUDITORIA_2026-05.md` (2026-05-19).

1. **Re-validar los 18 pares sin documento** (prioridad máxima)
   - Su backtest actual (PF 7–20, DD 1–4%, WF ~100%) tiene firma de overfitting
   - Re-correr WF **con gap de purga** entre train/test + test bootstrap (estilo Jesse)
   - Esperar caída a PF 1.2–2.0; rechazar los que no sobrevivan
   - **No mover capital real** a estos pares hasta entonces

2. **Paper trade V15** (en curso)
   - Verificar que el bot genera señales V15 en testnet para los 22 pares
   - `/log` debe mostrar `[V15]` con régimen y evaluación de setups cada 4h
   - Acumular trades reales — comparar con el backtest declarado

3. **Adoptar metodología de validación robusta**
   - Bootstrap / Monte Carlo como gate obligatorio antes de aprobar un par
   - Baseline simple (buy&hold / DCA) como referencia mínima a superar
   - Considerar migrar la capa de backtest a Jesse (sin look-ahead por diseño)

4. **Mergear v15/multi-pair a main**
   - Solo tras re-validación creíble y confirmación de señales en paper trading
   - Verificar que portfolio_manager ejecuta trades V15 correctamente

5. **Reentrenamiento mensual SHORT model BTC**
   - Usar `train_v15_prod.py` con production Python
   - Actualizar cutoff date y verificar in-sample metrics
   - Recalibrar threshold si distribución de probabilidades cambia
