# Plan V13.05 - Adaptive Expert System

## Fecha: 2026-02-28

---

## Filosofia: Pensar como Trader Profesional

Un trader profesional NO usa una sola estrategia para todo. El dice:

> "Hoy BTC esta muy volatil, no entro"
> "Parece que va a caer por X y Y, pero el mercado esta en modo Z, asi que ajusto mi estrategia"
> "Esta moneda esta en rango, voy a jugar mean reversion"
> "Hay tendencia clara, voy a seguirla con pullbacks"

**Clave**: Cambian la estrategia dependiendo de la situacion.

---

## Arquitectura Propuesta

```
                    +------------------+
                    |  MARKET CONTEXT  |
                    |  (Regime + Mood) |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
        +-----v-----+  +-----v-----+  +-----v-----+
        |  TRENDING |  |  RANGING  |  |  VOLATILE |
        | STRATEGY  |  | STRATEGY  |  |  NO TRADE |
        +-----------+  +-----------+  +-----------+
              |              |              |
              +--------------+--------------+
                             |
                    +--------v---------+
                    |  ENTRY SIGNAL    |
                    |  (si aplica)     |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  RISK MANAGER    |
                    |  (sizing, TP/SL) |
                    +------------------+
```

---

## Componentes del Sistema

### 1. Regime Detector (ML)

**Objetivo**: Clasificar el mercado en regimenes discretos.

**Regimenes**:
- `BULL_TREND`: Tendencia alcista clara
- `BEAR_TREND`: Tendencia bajista clara
- `RANGE_NARROW`: Lateral con poca volatilidad
- `RANGE_WIDE`: Lateral con alta volatilidad
- `BREAKOUT`: Salida de rango (direccion incierta)
- `CHOPPY`: Sin direccion clara, peligroso

**Features para detectar regimen**:
- ADX (fuerza de tendencia)
- Choppiness Index
- BB Width (volatilidad)
- EMA stack (20/50/200 alignment)
- ATR percentile (vs historico)
- Precio vs EMAs

**Modelo**: Clasificador (RandomForest o LightGBM)
- Target: Regimen (etiquetado manual o por reglas)
- Importante: NO predice precio, solo clasifica estado actual

**Validacion**: El regimen detectado debe ser estable (no cambiar cada vela)

---

### 2. Strategy Selector (Reglas)

**Por cada regimen, una estrategia diferente**:

| Regimen | Estrategia | Direccion | TP/SL |
|---------|------------|-----------|-------|
| BULL_TREND | Pullback a EMA20 | LONG only | 4%/2% |
| BEAR_TREND | NO TRADE o Short | - | - |
| RANGE_NARROW | Mean Reversion | LONG/SHORT | 2%/1% |
| RANGE_WIDE | Breakout | LONG/SHORT | 3%/1.5% |
| BREAKOUT | Seguir momentum | Direccion del break | 4%/2% |
| CHOPPY | NO TRADE | - | - |

**Clave**: Algunas condiciones = NO TRADEAR. Esto es critico.

---

### 3. Entry Signal Generator (ML + Reglas)

**Depende de la estrategia seleccionada**:

#### Estrategia Pullback (BULL_TREND):
- RSI < 40 (pero > 25)
- Precio toca EMA20 o EMA50
- Volumen normal o bajo (no panico)
- ML: Confirma probabilidad de rebote > 55%

#### Estrategia Mean Reversion (RANGE):
- RSI < 30 o > 70
- Precio en extremos de BB
- Volumen spike (posible capitulacion)
- ML: Confirma probabilidad de reversion > 55%

#### Estrategia Breakout (RANGE_WIDE):
- Precio rompe BB upper/lower
- Volumen > 1.5x promedio
- ADX subiendo
- ML: Confirma momentum > 55%

---

### 4. Risk Manager (Reglas + ML)

**Sizing dinamico**:
- Base: 10% del capital por trade
- Ajuste por regimen: CHOPPY = 0%, TREND = 100%, RANGE = 70%
- Ajuste por confianza del ML: <60% = 50% size, >80% = 100% size

**TP/SL dinamico**:
- Depende del regimen (tabla arriba)
- Ajuste por ATR actual (mercado volatil = mas espacio)

---

## Plan de Implementacion

### Fase 1: Regime Detector (Semana 1)

1. **Etiquetar datos historicos**:
   - Usar reglas para etiquetar regimenes en historico
   - Ej: ADX > 25 + EMA stack aligned = TREND
   - Ej: ADX < 20 + BB narrow = RANGE_NARROW
   - Ej: Choppiness > 60 = CHOPPY

2. **Entrenar clasificador**:
   - Features: ADX, Chop, BB_width, EMA_distances, ATR_pct
   - Target: Regimen (multi-clase)
   - Validacion: Accuracy + Confusion matrix

3. **Validar estabilidad**:
   - El regimen no debe cambiar cada vela
   - Usar smoothing o confirmacion (2+ velas en mismo regimen)

**Entregable**: `regime_detector.pkl` + `regime_meta.json`

---

### Fase 2: Backtest por Regimen (Semana 1-2)

1. **Para cada regimen, probar estrategias**:
   - BULL_TREND: Pullback vs Breakout vs Hold
   - RANGE: Mean Reversion vs Grid vs No trade
   - etc.

2. **Encontrar la mejor estrategia por regimen**:
   - Metricas: WR, PnL, PF, MaxDD
   - Usar train/validation split (NO test)

3. **Documentar reglas claras**:
   - Si regimen = X, entonces estrategia = Y con params Z

**Entregable**: `strategies_by_regime.json`

---

### Fase 3: Entry Signal Models (Semana 2-3)

1. **Para cada estrategia, entrenar modelo de entrada**:
   - Pullback model: P(rebote exitoso | pullback detectado)
   - Mean reversion model: P(reversion exitosa | extremo detectado)
   - Breakout model: P(continuation | breakout detectado)

2. **Features especificos por estrategia**:
   - Pullback: Profundidad del pullback, volumen, RSI
   - Mean reversion: Distancia a media, tiempo en extremo
   - Breakout: Fuerza del break, volumen, momentum

3. **Clasificadores binarios**:
   - Target: WIN (1) o LOSS (0) del trade
   - Threshold calibrado para precision

**Entregable**: `pullback_model.pkl`, `meanrev_model.pkl`, `breakout_model.pkl`

---

### Fase 4: Integracion (Semana 3)

1. **Pipeline completo**:
   ```python
   def generate_signal(df):
       # 1. Detectar regimen
       regime = regime_detector.predict(features)

       # 2. Si regimen es tradeable
       if regime in ['CHOPPY', 'BEAR_TREND']:
           return None  # No trade

       # 3. Seleccionar estrategia
       strategy = STRATEGIES[regime]

       # 4. Buscar setup
       setup = strategy.detect_setup(df)
       if setup is None:
           return None

       # 5. Confirmar con ML
       prob = strategy.model.predict_proba(setup)
       if prob < 0.55:
           return None

       # 6. Generar senal con sizing
       return Signal(
           direction=setup.direction,
           tp=strategy.tp,
           sl=strategy.sl,
           size=calculate_size(regime, prob)
       )
   ```

2. **Backtest completo**:
   - Train: 2019-2024
   - Validation: 2025-01 a 2025-08
   - Test: 2025-09 a 2026-02 (UNA vez al final)

**Entregable**: `ml_strategy_v1305.py`

---

### Fase 5: Evaluacion Final (Semana 4)

1. **Backtest en TEST set** (una sola vez):
   - Si pasa: Deploy a produccion
   - Si falla: Analizar y volver a Fase 1

2. **Criterios de exito**:
   - WR > 50%
   - PF > 1.2
   - Mejor que random por > 10%
   - MaxDD < 20%

3. **Paper trading** (2 semanas):
   - Antes de dinero real
   - Comparar con backtest

---

## Diferencias con Enfoques Anteriores

| Aspecto | V13.04 (fallido) | V13.05 (propuesto) |
|---------|------------------|-------------------|
| Modelo | Uno para todo | Multiples especializados |
| Regimen | Ignorado | Detectado y usado |
| Estrategia | Siempre igual | Adaptativa |
| No tradear | Nunca | Si, cuando corresponde |
| Target | Predecir precio | Clasificar setup |
| Validacion | Contaminada | Train/Val/Test estricto |

---

## Empezar con BTC

1. BTC es el mas liquido y con mas datos
2. Si funciona en BTC, replicar a otros pares
3. Un "experto BTC" completo antes de agregar mas

---

## Riesgos y Mitigaciones

| Riesgo | Mitigacion |
|--------|------------|
| Overfitting del regime detector | Features simples, poca profundidad |
| Regimen cambia muy rapido | Confirmacion de 2+ velas |
| Estrategia equivocada | Backtest extensivo por regimen |
| Pocos trades | Aceptable si son de calidad |
| Complejidad del sistema | Modularidad, testing unitario |

---

## Proximos Pasos Inmediatos

1. [ ] Crear script de etiquetado de regimenes
2. [ ] Entrenar regime detector
3. [ ] Backtest por regimen en BTC
4. [ ] Identificar mejor estrategia por regimen
5. [ ] Entrenar modelos de entrada por estrategia
6. [ ] Integrar pipeline completo
7. [ ] Evaluar en test set

---

*Documento creado: 2026-02-28*
*Version: V13.05 Plan Inicial*
