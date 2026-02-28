# Análisis de Errores V13.04

## Fecha: 2026-02-28

## Resumen

V13.04 fue desplegado a producción con errores metodológicos graves que invalidan los resultados documentados.

## Errores Identificados

### 1. Data Leakage en Grid Search

**Archivo**: `multi_pair_low_overfit.py` líneas 145-173

```python
def grid_search_best_config(model, scaler, X_test, df_test, pred_std):
    tp_range = [0.02, 0.025, 0.03, 0.04, 0.05]
    sl_range = [0.01, 0.015, 0.02, 0.025]
    conv_range = [0.3, 0.5, 0.7, 1.0]
    # Prueba TODAS las combinaciones en TEST SET
```

**Problema**: Los parámetros "óptimos" (TP/SL/conviction) fueron encontrados haciendo grid search EN el test set. Esto es data leakage - los parámetros están sobreajustados a ese periodo específico.

**Consecuencia**: Los resultados documentados (58% WR, +$321 para BTC) no son reproducibles en datos nuevos.

### 2. Entrenamiento en 100% de Datos

**Archivo**: `ml_export_v1304.py` línea 116 (original)

```python
# Train on ALL data (no split - we already validated with walk-forward)
```

**Problema**: El modelo de producción fue entrenado en 100% de los datos, incluyendo el periodo de "test".

**Consecuencia**: No hay validación out-of-sample real. El modelo "vio" todos los datos durante entrenamiento.

### 3. Walk-Forward ≠ Modelo de Producción

**Confusión**: Walk-forward validation RE-ENTRENA el modelo en cada ventana. Los buenos resultados son de 5 modelos diferentes, no del modelo desplegado.

| Walk-Forward | Producción |
|--------------|------------|
| 5 modelos diferentes | 1 modelo |
| Re-entrena cada ventana | Entrenado una vez |
| Buenos resultados | Resultados desconocidos |

### 4. Parámetros Genéricos vs Documentados

**Archivo**: `ml_export_v1304.py` líneas 40-46 (original)

| Par | Documentado | Desplegado |
|-----|-------------|------------|
| DOGE | SL 1% | SL 2% |
| ADA | SL 1.5% | SL 2% |
| DOT | TP 2.5% | TP 2% |

### 5. Lógica de Backtest Diferente

**Script original**: Una posición a la vez (`position = None`)
**Mi backtest**: Múltiples posiciones simultáneas

Resultado: 775 trades vs 192 documentados para BTC.

## Verificación de Discrepancias

### Mismo Periodo que Documentación

| Par | Doc Trades | Real Trades | Doc WR | Real WR | Doc PnL | Real PnL |
|-----|------------|-------------|--------|---------|---------|----------|
| BTC | 192 | 775 | 58.3% | 54.1% | +$321 | -$520 |
| DOGE | 177 | 249 | 59.9% | 32.5% | +$335 | -$2,790 |

### Correlación Out-of-Sample Real (80/20)

| Par | Corr (100% datos) | Corr (out-of-sample) | Drop |
|-----|-------------------|----------------------|------|
| DOGE | 0.2307 | 0.0489 | -79% |
| ADA | 0.1322 | 0.1232 | -7% |
| DOT | 0.0925 | 0.0042 | -95% |
| XRP | 0.0813 | 0.0839 | +3% |
| BTC | 0.0916 | 0.0921 | +1% |

**DOGE y DOT tenían overfitting severo**.

## Impacto en Producción

El modelo en producción (main branch) tiene:
- Entrenamiento en 100% datos (sin out-of-sample)
- Parámetros genéricos 2%/2% (no los documentados)
- Resultados esperados: **PÉRDIDAS**

## Correcciones Realizadas (esta rama)

1. `ml_export_v1304.py`: Cambiado a 80/20 split
2. `ml_export_v1304.py`: Parámetros individuales por par
3. `backtest_v1304_params.py`: Script de verificación

## Metodología Correcta (TODO)

Para tener resultados válidos:

1. **Split temporal estricto**:
   - Train: 2019-2024
   - Validation: 2025 (para elegir params)
   - Test: 2026 (NUNCA tocar hasta el final)

2. **Grid search en VALIDATION, no test**:
   ```python
   # Encontrar mejor TP/SL en validation
   best_params = grid_search(X_val, y_val)
   # Evaluar UNA VEZ en test
   final_result = backtest(X_test, best_params)
   ```

3. **Una posición a la vez**:
   - Igual que el bot real
   - No trades superpuestos

4. **Modelo único**:
   - Entrenar en train+validation
   - Testear en test
   - Desplegar ese mismo modelo

## Próximos Pasos

1. [ ] Implementar split train/validation/test correcto
2. [ ] Grid search SOLO en validation
3. [ ] Backtest con lógica idéntica al bot (1 posición)
4. [ ] Evaluar en test UNA vez
5. [ ] Si pasa, desplegar a producción
6. [ ] Si no pasa, considerar abandonar ML

---

*Documento generado: 2026-02-28*
