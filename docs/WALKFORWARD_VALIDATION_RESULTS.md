# Walk-Forward Validation Results

## Fecha: 2026-02-27

## Metodologia

**Opcion B: Walk-Forward Validation REAL**

```
Datos: 2020 ──────────── 2024 │ 2025 ────── Sep │ Oct 2025 ── Feb 2026
       │                      │                 │                    │
       └──── TRAIN ───────────┘                 │                    │
                               └── VALIDATION ──┘                    │
                                                 └──── TEST SAGRADO ─┘
                                                      (NO TOCADO)
```

- **Train**: 2020-01-01 a 2024-12-31
- **Validation**: 2025-01-01 a 2025-09-30
- **Test**: 2025-10-01 a 2026-02-28 (SAGRADO - no usado)

## Resultados BTC/USDT

| Metrica | Train | Validation | Cambio |
|---------|-------|------------|--------|
| Correlation | 0.4550 | -0.0533 | -111.7% |
| Predicciones | 13,038 | 1,638 | - |
| Pred std | 0.0076 | 0.0041 | -46% |

### Distribucion de Conviction en Validation

| Threshold | Count | % |
|-----------|-------|---|
| >= 0.5 | 208 | 12.7% |
| >= 1.0 | 59 | 3.6% |
| >= 2.0 | 25 | 1.5% |

## Interpretacion

### El modelo NO generaliza

1. **Correlation negativa** en validation (-0.053) significa que el modelo es PEOR que aleatorio
2. **Correlation drop** de 111.7% confirma overfitting severo
3. Los patrones aprendidos en 2020-2024 NO aplican a 2025

### Por que V13.03 parece funcionar en backtest

V13.03 fue entrenado y optimizado usando TODOS los datos (incluyendo 2025-2026):
- El modelo "ve" el futuro durante el entrenamiento
- Grid search de TP/SL usa datos historicos completos
- Esto es **look-ahead bias** - no se puede replicar en produccion

### Implicacion para produccion

Si V13.03 se pone en produccion:
- **Expectativa realista**: 45-50% WR (no 67%)
- **PnL esperado**: $50-100/año (no $400)
- **El modelo puede perder dinero** porque los patrones de 2020-2024 ya no aplican

## Conclusion

Walk-Forward Validation confirma que:

1. **Los modelos ML para crypto tienen vida util limitada**
2. **Backtest optimista != Produccion rentable**
3. **La unica validacion real es el paper trading o produccion con poco capital**

## Recomendacion

### Para V13.03 actual:
- Correr en produccion con capital minimo ($100)
- Monitorear WR real vs backtest
- Si WR < 50% despues de 50 trades, revisar estrategia

### Para futuro:
- Reentrenar modelos cada 3-6 meses
- Usar datos mas recientes para training
- Considerar modelos mas simples (menos overfitting)
- O abandonar ML y usar estrategias rules-based

---

*Documento generado: 2026-02-27*
*Script: train_walkforward_vbeta.py, debug_walkforward.py*
