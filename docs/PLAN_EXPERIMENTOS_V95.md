# Plan de Experimentos V9.5+

## Baseline Actual
```
V9.5: 2159 trades | WR=41.4% | PnL=$626 | PF=1.08 | Return=125%
```

---

## FASE 1: Ajustes Rapidos (sin re-entrenar)

### Exp 1A: Threshold mas alto
**Hipotesis**: Thresholds 0.45-0.55 son muy permisivos, subir a 0.55-0.65
**Implementacion**: Modificar thresholds en backtest
**Variantes**:
- 1A.1: Todos +0.10 (0.45->0.55, 0.50->0.60, 0.55->0.65)
- 1A.2: Todos +0.15
- 1A.3: Solo pares debiles +0.15 (SOL, BTC, AVAX)

### Exp 1B: Confidence mas alta
**Hipotesis**: conf=0.7 es bajo para mercado choppy
**Implementacion**: Subir threshold de confianza
**Variantes**:
- 1B.1: conf >= 0.8
- 1B.2: conf >= 1.0
- 1B.3: conf >= 1.2
- 1B.4: conf >= 1.5

### Exp 1C: Combinacion A+B
**Hipotesis**: Threshold alto + Confidence alta = mejor filtrado
**Variantes**:
- 1C.1: thresh+0.10 AND conf>=1.0
- 1C.2: thresh+0.15 AND conf>=1.2

---

## FASE 2: Choppiness Index (nuevo feature)

### Exp 2A: Choppiness Filter
**Hipotesis**: No operar cuando mercado es lateral
**Implementacion**: Calcular CHOP index, filtrar cuando CHOP > threshold
**Formula**: CHOP = 100 * LOG10(SUM(ATR,14) / (MAX(HIGH,14) - MIN(LOW,14))) / LOG10(14)
**Variantes**:
- 2A.1: No trade si CHOP > 55
- 2A.2: No trade si CHOP > 60
- 2A.3: No trade si CHOP > 65

### Exp 2B: Choppiness como Feature del LossDetector
**Hipotesis**: El modelo aprende cuando CHOP alto = perdida probable
**Implementacion**: Agregar CHOP a las 21 features, re-entrenar
**Requiere**: Re-entrenamiento de LossDetectors

---

## FASE 3: Reduccion de Features (anti-overfitting)

### Exp 3A: Top 10 Features por importancia
**Hipotesis**: Menos features = menos overfitting = mejor generalizacion
**Implementacion**:
1. Extraer feature importance de cada LossDetector
2. Quedarse con top 10-12
3. Re-entrenar solo con esas features
**Requiere**: Re-entrenamiento

### Exp 3B: Features simplificados
**Hipotesis**: Algunas features son redundantes
**Implementacion**: Remover features correlacionadas
- Remover: ld_pair_ret_20 (correlacionado con ret_5)
- Remover: cs_regime_range (derivado de bull+bear)
- Remover: ld_hour (poco impacto?)
**Requiere**: Re-entrenamiento

---

## FASE 4: Datos mas Recientes

### Exp 4A: Solo 2024-2026
**Hipotesis**: Patrones recientes son mas relevantes
**Implementacion**: Re-entrenar V7 y LossDetector solo con datos 2024-2026
**Requiere**: Re-entrenamiento completo

### Exp 4B: Weighted Training (mas peso a datos recientes)
**Hipotesis**: Datos recientes importan mas pero antiguos dan contexto
**Implementacion**: sample_weight en LightGBM, peso exponencial por fecha
**Requiere**: Modificar ml_train_v95.py

---

## FASE 5: Features Nuevos (alto esfuerzo)

### Exp 5A: Funding Rate
**Hipotesis**: Funding extremo predice reversiones
**Datos**: Binance Futures API - funding rate historico
**Implementacion**:
- funding_rate_8h
- funding_rate_ma_24h
- funding_rate_zscore

### Exp 5B: Open Interest
**Hipotesis**: OI alto + caida precio = liquidaciones = mas caida
**Datos**: Binance Futures API - open interest historico
**Implementacion**:
- oi_change_24h
- oi_price_divergence

### Exp 5C: Liquidaciones
**Hipotesis**: Cascadas de liquidaciones predicen movimientos
**Datos**: Coinglass API o similar
**Implementacion**:
- liquidations_long_24h
- liquidations_short_24h
- liquidation_ratio

---

## FASE 6: Ensemble

### Exp 6A: Voting Ensemble
**Hipotesis**: Combinar V9 generico + V9.5 por par
**Implementacion**: Skip trade si AMBOS dicen skip
**Variantes**:
- 6A.1: AND (ambos deben aprobar)
- 6A.2: OR (uno aprueba es suficiente)
- 6A.3: Weighted average de probabilidades

### Exp 6B: Stacking
**Hipotesis**: Meta-modelo aprende cuando confiar en cada LossDetector
**Implementacion**:
1. Entrenar modelos base
2. Entrenar meta-modelo con outputs de base
**Requiere**: Arquitectura compleja

---

## ORDEN DE EJECUCION RECOMENDADO

| Orden | Experimento | Esfuerzo | Impacto Esperado |
|-------|-------------|----------|------------------|
| 1     | 1A + 1B + 1C | Bajo    | Alto             |
| 2     | 2A (CHOP filter) | Bajo | Alto            |
| 3     | 3A (top features) | Medio | Medio-Alto    |
| 4     | 4A (datos recientes) | Medio | Medio       |
| 5     | 2B (CHOP como feature) | Medio | Alto      |
| 6     | 5A-C (features nuevos) | Alto | Alto       |
| 7     | 6A-B (ensemble) | Alto | Medio            |

---

## METRICAS DE EXITO

Para considerar un experimento exitoso debe superar V9.5 base en:
- WR >= 45% (vs 41.4% actual)
- PF >= 1.15 (vs 1.08 actual)
- Return >= 125% (mantener o mejorar)

Idealmente:
- WR >= 50%
- PF >= 1.5
- Menos trades pero de mejor calidad

---

## SIGUIENTE PASO

Crear `backtest_experiments.py` que ejecute FASE 1 automaticamente
y genere tabla comparativa de resultados.
