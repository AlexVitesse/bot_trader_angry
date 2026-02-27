# Experimento: Modelos SOL y BNB V2
## Fecha: 27 Feb 2026

---

## Contexto

SOL y BNB fueron excluidos de V13 por bajo rendimiento:
- **SOL/USDT**: 42% WR en últimos 14 días, -$31 (el peor)
- **BNB/USDT**: volátil, rendimiento marginal

**Objetivo:** Aplicar el mismo proceso que BTC V2 para intentar rehabilitar estas monedas.

---

## Metodología

Mismo proceso que BTC:
1. Descargar datos históricos completos
2. Entrenar modelo GradientBoosting con 54 features
3. Optimizar TP/SL en grid search
4. Validar en últimos 3 meses (Dic 2025 - Feb 2026)
5. Analizar por régimen, dirección y conviction

---

## SOL/USDT

### Datos
| Campo | Valor |
|-------|-------|
| Periodo | 2020-08-11 a 2026-02-27 |
| Velas | 12,161 |
| Train | 10,691 samples |
| Val | 1,104 samples |
| Test | 162 samples |

### Modelo
```python
GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
)
```

### Optimización TP/SL

| TP% | SL% | Ratio | Trades | WR | PnL | PF |
|-----|-----|-------|--------|-----|-----|-----|
| **5.0%** | **2.50%** | 2:1 | 998 | 63.6% | +2265% | 3.50 |
| 4.5% | 2.50% | 1.8:1 | 998 | 64.3% | +1997% | 3.24 |
| 5.0% | 2.00% | 2.5:1 | 998 | 54.5% | +1809% | 2.99 |
| 4.0% | 2.50% | 1.6:1 | 998 | 65.1% | +1728% | 2.99 |
| 3.0% | 1.50% | 2:1 | 998 | 48.1% | +663% | 1.85 |

**Mejor config: TP=5%, SL=2.5%** (+1602% vs baseline)

### Rendimiento Histórico por Año
| Año | Trades | WR | PnL |
|-----|--------|-----|-----|
| 2020 | 135 | 66.7% | +337% |
| 2021 | 458 | 58.1% | +850% |
| 2022 | 145 | 62.1% | +312% |
| 2023 | 137 | 86.1% | +542% |
| 2024 | 55 | 56.4% | +95% |
| 2025 | 64 | 60.9% | +132% |
| **2026** | **4** | **25.0%** | **-5%** |

### Rendimiento Histórico por Régimen
| Régimen | Trades | WR | PnL |
|---------|--------|-----|-----|
| BULL | 344 | 58.7% | +655% |
| BEAR | 169 | 62.7% | +370% |
| RANGE | 485 | 67.4% | +1240% |

### Validación Últimos 3 Meses (Dic 2025 - Feb 2026)

| Mes | Trades | WR | PnL |
|-----|--------|-----|-----|
| 2025-12 | 4 | 0.0% | -$10.00 |
| 2026-01 | 1 | 100.0% | +$2.47 |
| 2026-02 | 3 | 0.0% | -$7.50 |
| **TOTAL** | **8** | **12.5%** | **-$15.03** |

### Análisis del Problema

| Condición | Trades | WR | PnL |
|-----------|--------|-----|-----|
| BEAR | 2 | 50.0% | -$0.03 |
| RANGE | 6 | 0.0% | -$15.00 |
| LONG | 6 | 0.0% | -$15.00 |
| SHORT | 2 | 50.0% | -$0.03 |

**Diagnóstico:**
- Solo 8 trades con conviction >= 1.0 (muy pocos)
- 7 de 8 terminan en Stop Loss
- LONGs en RANGE no funcionan en absoluto
- El modelo no generaliza al mercado actual de SOL

### Test de Conviction Thresholds

| Conv Min | Trades | WR | PnL |
|----------|--------|-----|-----|
| 0.5 | 72 | 37.5% | +$11.25 |
| 0.7 | 30 | 36.7% | +$1.04 |
| 1.0 | 8 | 12.5% | -$15.03 |
| 1.2 | 5 | 20.0% | -$7.53 |
| 1.5 | 1 | 0.0% | -$2.50 |

**Observación:** Con conviction bajo (0.5) genera más trades y es rentable, pero WR muy bajo (37.5%).

### Conclusión SOL

**ESTADO: NO HABILITAR**

El modelo V2 funciona históricamente pero no generaliza al mercado actual de SOL. Posibles causas:
1. SOL ha cambiado su comportamiento en 2025-2026
2. El modelo está sobreajustado a patrones históricos
3. Necesita features diferentes o enfoque alternativo

### Ideas para Mejorar SOL
- [ ] Entrenar solo con datos 2024-2026
- [ ] Agregar features de funding rate
- [ ] Usar modelo de clasificación en vez de regresión
- [ ] Probar ensemble con múltiples timeframes
- [ ] Filtrar solo trades SHORT
- [ ] Usar meta-filtro como BTC

---

## BNB/USDT

### Datos
| Campo | Valor |
|-------|-------|
| Periodo | 2019-01-01 a 2026-02-27 |
| Velas | 15,685 |
| Train | 14,215 samples |
| Val | 1,104 samples |
| Test | 162 samples |

### Modelo
```python
GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
)
```

### Optimización TP/SL

| TP% | SL% | Ratio | Trades | WR | PnL | PF |
|-----|-----|-------|--------|-----|-----|-----|
| **5.0%** | **2.50%** | 2:1 | 970 | 70.9% | +2708% | 4.87 |
| 5.0% | 2.00% | 2.5:1 | 970 | 64.6% | +2437% | 4.55 |
| 4.5% | 2.50% | 1.8:1 | 970 | 71.2% | +2391% | 4.46 |
| 4.0% | 2.50% | 1.6:1 | 970 | 72.1% | +2105% | 4.13 |
| 3.0% | 1.50% | 2:1 | 970 | 57.1% | +1035% | 2.66 |

**Mejor config: TP=5%, SL=2.5%** (+1673% vs baseline)

### Rendimiento Histórico por Año
| Año | Trades | WR | PnL |
|-----|--------|-----|-----|
| 2019 | 207 | 77.8% | +691% |
| 2020 | 137 | 70.8% | +385% |
| 2021 | 339 | 70.5% | +945% |
| 2022 | 78 | 87.2% | +315% |
| 2023 | 28 | 82.1% | +100% |
| 2024 | 52 | 86.5% | +207% |
| **2025** | **79** | **35.4%** | **-8%** |
| **2026** | **50** | **54.0%** | **+72%** |

### Rendimiento Histórico por Régimen
| Régimen | Trades | WR | PnL |
|---------|--------|-----|-----|
| BULL | 268 | 72.4% | +782% |
| BEAR | 261 | 70.5% | +726% |
| RANGE | 441 | 70.3% | +1199% |

### Validación Últimos 3 Meses (Dic 2025 - Feb 2026)

| Mes | Trades | WR | PnL |
|-----|--------|-----|-----|
| 2025-12 | 8 | 62.5% | +$3.30 |
| 2026-01 | 1 | 0.0% | -$2.50 |
| 2026-02 | 49 | 55.1% | +$74.93 |
| **TOTAL** | **58** | **55.2%** | **+$75.72** |

### Análisis por Condición

| Condición | Trades | WR | PnL |
|-----------|--------|-----|-----|
| **BEAR** | **30** | **73.3%** | **+$88.49** |
| RANGE | 28 | 35.7% | -$12.77 |
| **SHORT** | **40** | **72.5%** | **+$101.79** |
| LONG | 18 | 16.7% | -$26.07 |

**Diagnóstico:**
- BNB funciona EXCELENTE para SHORTs (72.5% WR)
- BNB funciona MAL para LONGs (16.7% WR)
- BEAR regime: muy bueno (73.3% WR)
- RANGE regime: malo (35.7% WR)

### Test de Conviction Thresholds

| Conv Min | Trades | WR | PnL |
|----------|--------|-----|-----|
| 0.5 | 114 | 47.4% | +$60.49 |
| 0.7 | 88 | 44.3% | +$30.05 |
| **1.0** | **58** | **55.2%** | **+$75.72** |
| 1.2 | 46 | 52.2% | +$65.60 |
| 1.5 | 33 | 57.6% | +$60.60 |
| 2.0 | 18 | 66.7% | +$45.00 |

**Mejor config: conviction >= 1.0** (mejor balance trades/WR/PnL)

### Conclusión BNB

**ESTADO: HABILITAR CON RESTRICCIONES**

BNB V2 es rentable (+$76 en 3 meses) pero tiene debilidades claras:
- LONGs pierden consistentemente
- RANGE regime no funciona

**Configuración recomendada si se habilita:**
```python
ML_BNB_CONFIG = {
    'model_file': 'bnb_usdt_v2_gradientboosting.pkl',
    'tp_pct': 0.05,           # 5% TP
    'sl_pct': 0.025,          # 2.5% SL
    'conv_min': 1.0,
    'only_short': True,       # Solo permitir SHORTs
    # O alternativamente:
    'only_bear': True,        # Solo operar en BEAR
}
```

### Ideas para Mejorar BNB
- [ ] Filtrar solo SHORTs (ya validado: +$102 vs +$76 total)
- [ ] Filtrar solo BEAR regime (+$88)
- [ ] Combinar: SHORT + BEAR solamente
- [ ] Entrenar modelo separado para LONGs
- [ ] Agregar features de correlación con BTC

---

## Comparación BTC vs SOL vs BNB

| Métrica | BTC V2 | SOL V2 | BNB V2 |
|---------|--------|--------|--------|
| Hist. PnL | +2319% | +2265% | +2708% |
| Hist. WR | 67.7% | 63.6% | 70.9% |
| 3M PnL | +$34* | -$15 | +$76 |
| 3M WR | 59%* | 12.5% | 55.2% |
| TP/SL | 4%/2% | 5%/2.5% | 5%/2.5% |
| **Estado** | **HABILITADO** | **NO HABILITAR** | **PROBAR** |

*BTC estimado del backtest integrado

---

## Archivos Generados

### Modelos
- `models/sol_usdt_v2_gradientboosting.pkl`
- `models/bnb_usdt_v2_gradientboosting.pkl`

### Datos
- `data/SOL_USDT_4h_full.parquet`
- `data/BNB_USDT_4h_full.parquet`

### Scripts
- `train_sol_bnb_v2.py` - Entrenamiento y optimización
- `validate_sol_bnb_recent.py` - Validación últimos 3 meses

---

## Próximos Pasos

### Para SOL
1. [ ] Probar modelo con datos solo 2024-2026
2. [ ] Probar features de funding rate / open interest
3. [ ] Probar clasificador binario WIN/LOSS
4. [ ] Probar solo SHORTs
5. [ ] Evaluar si SOL tiene comportamiento diferente post-FTX

### Para BNB
1. [ ] Implementar filtro solo-SHORT
2. [ ] Probar filtro solo-BEAR
3. [ ] Evaluar modelo separado para cada dirección
4. [ ] Probar correlación inversa con BTC como feature

---

*Documento creado: 27 Feb 2026*
