# Metodologia de Testing de Modelos V14

## Principio Fundamental

> "Aplicar filtros generales es formula para fracasar"

Cada modelo fue entrenado con datos y parametros especificos. Antes de aplicar cualquier mejora, debemos verificar que el modelo funciona TAL CUAL fue validado.

---

## Proceso de Testing

### Paso 1: Validar Configuracion Original

1. Identificar los parametros originales del modelo:
   - TP/SL usados en validacion
   - Logica de senales (setups vs ensemble)
   - Features requeridos

2. Ejecutar backtest con parametros EXACTOS
3. Comparar con resultados documentados
4. Si no coincide -> hay un problema en la implementacion

### Paso 2: Analizar Problemas (si los hay)

Si un par muestra resultados diferentes a la validacion:
1. Verificar que los datos son los mismos
2. Verificar que la logica de senales es identica
3. Verificar TP/SL correctos
4. NO aplicar filtros hasta entender la diferencia

### Paso 3: Aplicar Filtros (solo si necesario)

Una vez confirmado que el modelo funciona como fue validado:
1. Analizar filtros POR MODELO (no generales)
2. Probar filtros que mantengan al menos 20% de trades
3. Evaluar mejora en WR vs reduccion de trades
4. Aplicar solo si la mejora es significativa (>3% WR)

---

## Resultados de Validacion (2025-03-01)

### Modelos de Setups (BTC V14)

| Par | Trades | WR% | PnL% | TP/SL | Status |
|-----|--------|-----|------|-------|--------|
| BTCUSDT | 5768 | 37.2% | +3465.5% | 3%/1.5% | OK |
| ETHUSDT | 5710 | 37.7% | +4142.0% | 3%/1.5% | OK |

**Nota**: BTC y ETH usan la misma logica de setups tecnicos (8 patrones).
El WR de ~37% es normal - la rentabilidad viene del ratio TP:SL (2:1).

#### Detalle por Setup (BTC)

| Setup | Trades | WR% | PnL% |
|-------|--------|-----|------|
| PULLBACK_UPTREND | 527 | 46.1% | +527.1% |
| RALLY_DOWNTREND | 586 | 46.6% | +628.2% |
| CAPITULATION | 395 | 41.8% | +382.8% |
| OVERSOLD_EXTREME | 368 | 40.5% | +388.3% |
| SUPPORT_BOUNCE | 1091 | 40.3% | +923.0% |
| RESISTANCE_REJECTION | 1701 | 32.8% | +509.6% |
| EXHAUSTION | 475 | 30.1% | +72.8% |
| OVERBOUGHT_EXTREME | 625 | 27.5% | +33.6% |

### Modelos Ensemble

| Modelo | Par | Trades | WR% | PnL% | TP/SL | Status |
|--------|-----|--------|-----|------|-------|--------|
| ADA | ADAUSDT | 107 | 86.0% | +735.2% | 6%/4% | OK |
| ADA | ATOMUSDT | 117 | 59.0% | +413.9% | 6%/4% | OK (menor WR) |
| ADA | AVAXUSDT | 120 | 46.7% | +257.6% | 6%/4% | OK (menor WR) |
| DOGE | DOGEUSDT | 121 | 67.8% | +988.3% | 6%/4% | OK |
| DOT | DOTUSDT | 33 | 84.8% | +210.7% | 5%/3% | OK |

**Observaciones Cross-Pairs (ATOM, AVAX)**:
- Usan modelo ADA pero tienen WR menor (59%, 46% vs 86%)
- AUN ASI son rentables (+413%, +257%)
- Candidatos para filtros modelo-especificos

---

## Filtros Recomendados por Modelo

Basado en analisis con `analyze_filters_per_model.py`:

### Modelos que NO necesitan filtros:
- **BTC/ETH**: 37% WR es normal, rentabilidad viene de ratio TP:SL

### Filtros IMPLEMENTADOS (basados en analisis de trades reales):
- **DOT**: `vol_ratio < 4.15` mejora 84.8% -> 92.6% WR (retiene 82% trades)
- **DOGE**: `bb_pct < 0.7` mejora WR
- **ADA**: `vol_ratio > 2.0` mejora WR
- **SOL**: `ret_3 > -0.03` mejora WR
- **ATOM/AVAX**: Usan filtro ADA (mismo modelo base)

---

## Errores Comunes a Evitar

1. **Usar TP/SL incorrectos**
   - BTC validation usa 3%/1.5%, no 4%/1.5%
   - Siempre verificar contra archivo de validacion original

2. **Agregar filtros ML que no estaban en validacion**
   - BTC validation NO usa filtro de confianza ML
   - El filtro ML REDUCE rendimiento en BTC

3. **Aplicar filtros generales**
   - Cada modelo necesita filtros especificos
   - Lo que funciona para ADA puede destruir DOT

4. **No probar configuracion original primero**
   - Siempre validar que funciona como fue entrenado
   - Solo despues explorar mejoras

---

## Scripts de Testing

- `test_models_as_validated.py`: Prueba modelos con configuracion original
- `analyze_filters_per_model.py`: Analiza filtros por modelo
- `analyze_all_models.py`: Genera datos de trades para analisis

---

## Flujo Correcto de Testing

```
1. Cargar modelo con config original
         |
         v
2. Ejecutar backtest sin modificaciones
         |
         v
3. Comparar con resultados documentados
         |
    [Coincide?]
      |       \
     Si       No
      |        |
      v        v
4. Analizar   Investigar
   filtros    diferencia
      |
      v
5. Probar filtros MODELO-ESPECIFICOS
      |
      v
6. Seleccionar filtro con mejor balance WR/trades
      |
      v
7. Implementar en produccion
```

---

## Compatibilidad de Cross-Pairs

Analisis de si los cross-pairs generan senales con su modelo base:

### Modelo DOGE -> Memecoins

| Par | Signal Rate | Prob Avg | Status |
|-----|-------------|----------|--------|
| DOGEUSDT | 1.5% | RF=0.28, GB=0.28 | OK (nativo) |
| 1000SHIBUSDT | **0.0%** | RF=0.24, GB=0.24 | **INCOMPATIBLE** |
| 1000PEPEUSDT | 1.2% | RF=0.29, GB=0.29 | OK |
| 1000FLOKIUSDT | 1.0% | RF=0.28, GB=0.28 | OK |

**SHIB deshabilitado** - El modelo DOGE nunca genera senales para SHIB.

### Modelo ADA -> Smart Contracts

| Par | Signal Rate | Prob Avg | Status |
|-----|-------------|----------|--------|
| ADAUSDT | 1.0% | RF=0.29, GB=0.29 | OK (nativo) |
| ATOMUSDT | 1.0% | RF=0.29, GB=0.29 | OK |
| AVAXUSDT | 1.3% | RF=0.30, GB=0.30 | OK |
| POLUSDT | 0.9% | RF=0.27, GB=0.27 | BAJO |

### Modelo DOT -> Infraestructura

| Par | Signal Rate | Prob Avg | Status |
|-----|-------------|----------|--------|
| DOTUSDT | 0.4% | RF=0.30, GB=0.30 | Muy selectivo (nativo) |
| LINKUSDT | 0.4% | RF=0.31, GB=0.32 | Muy selectivo |
| ALGOUSDT | 0.4% | RF=0.32, GB=0.32 | Muy selectivo |
| FILUSDT | 0.7% | RF=0.31, GB=0.31 | OK |
| NEARUSDT | 0.4% | RF=0.32, GB=0.33 | Muy selectivo |

**Nota**: El modelo DOT es muy selectivo (0.4% signal rate) lo cual explica los pocos trades (33) pero alto WR (84.8%).

---

## Pares Aprobados para Paper Trading (15 de 16)

1. BTC - setups
2. ETH - setups
3. DOGE - ensemble nativo
4. ADA - ensemble nativo
5. DOT - ensemble nativo
6. SOL - ensemble dedicado
7. ATOM - modelo ADA
8. AVAX - modelo ADA
9. POL - modelo ADA
10. PEPE - modelo DOGE
11. FLOKI - modelo DOGE
12. LINK - modelo DOT
13. ALGO - modelo DOT
14. FIL - modelo DOT
15. NEAR - modelo DOT

**Excluido**: SHIB (incompatible con modelo DOGE)

---

*Documento generado: 2026-03-01*
*Datos de: analysis/validation_test_results.json*
