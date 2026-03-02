# Plan de Entrenamiento - Expertos V14

## Resumen
Cada experto sigue el mismo pipeline de 6 pasos. BTC ya completado, sirve de plantilla.

## Pipeline por Experto

```
[1] DATOS → [2] REGIMEN → [3] ENSEMBLE → [4] BACKTEST → [5] VALIDACION → [6] EXPORT
```

### Paso 1: Descargar Datos
- Fuente: Binance via CCXT
- Timeframe: 4h
- Periodo: 2018-2026 (o máximo disponible)
- Output: `data/{SYMBOL}_4h.csv`

### Paso 2: Detector de Régimen
- Entrenar clasificador de régimen (TREND_UP, TREND_DOWN, RANGE, VOLATILE)
- Features: ADX, CHOP, DI+, DI-, volatilidad
- Output: `models/{symbol}_regime_detector.pkl`

### Paso 3: Ensemble de 3 Modelos
- **Context Model**: RSI, BB%, tendencia macro
- **Momentum Model**: ROC, MACD, momentum
- **Volume Model**: OBV, volume ratio, accumulation
- Output: `models/{symbol}_ensemble_{context|momentum|volume}.pkl`

### Paso 4: Backtest Walk-Forward
- 12+ folds, train 6 meses, test 2 meses
- Métricas: PnL, WR, PF, MaxDD
- Criterio: 80%+ folds positivos

### Paso 5: Validación Cruzada
- Test en datos sintéticos (5 mercados)
- Test en otro asset similar (sin modificar reglas)
- Criterio: 5/6+ escenarios positivos

### Paso 6: Exportar a Producción
- Copiar modelos a `strategies/{symbol}_v14/models/`
- Actualizar config.py con parámetros optimizados
- Marcar como APPROVED en `strategies/__init__.py`

---

## Estado de Expertos

| Experto | Paso 1 | Paso 2 | Paso 3 | Paso 4 | Paso 5 | Paso 6 | Status |
|---------|--------|--------|--------|--------|--------|--------|--------|
| BTC V14 | ✅ | ✅ | ✅ | ✅ | ✅ | ⏳ | FALTA EXPORT |
| ETH V14 | ⏳ | - | - | - | - | - | SIGUIENTE |
| DOGE V14 | - | - | - | - | - | - | PENDIENTE |
| ADA V14 | - | - | - | - | - | - | PENDIENTE |
| DOT V14 | - | - | - | - | - | - | PENDIENTE |

---

## Orden de Prioridad

1. **BTC** - Completar export de modelos
2. **ETH** - Correlación alta con BTC, mercado líquido
3. **DOGE** - Alta volatilidad, buen para momentum
4. **ADA** - Diferente comportamiento, diversifica
5. **DOT** - Ecosistema diferente, diversifica más

---

## Tiempo Estimado por Experto

- Paso 1 (Datos): 5 min
- Paso 2 (Régimen): 10 min
- Paso 3 (Ensemble): 15 min
- Paso 4 (Backtest): 20 min
- Paso 5 (Validación): 15 min
- Paso 6 (Export): 5 min

**Total por experto: ~70 min**

---

## Archivos a Generar por Experto

```
strategies/{symbol}_v14/
├── __init__.py           # Exports
├── config.py             # Parámetros específicos del asset
├── strategy.py           # Lógica (heredada de BTC con ajustes)
├── validation.py         # Script de validación
├── models/
│   ├── regime_detector.pkl
│   ├── ensemble_context.pkl
│   ├── ensemble_momentum.pkl
│   ├── ensemble_volume.pkl
│   └── meta.json         # Métricas, fecha, versión
└── results/
    ├── backtest_results.json
    └── validation_results.json
```

---

## Script de Entrenamiento Unificado

```bash
# Entrenar un experto completo
python train_expert.py --symbol ETH/USDT --output strategies/eth_v14/

# Validar un experto
python validate_expert.py --symbol ETH/USDT --strategy strategies/eth_v14/

# Comparar expertos
python compare_experts.py --experts btc_v14,eth_v14
```
