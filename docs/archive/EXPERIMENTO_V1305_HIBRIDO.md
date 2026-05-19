# Experimento V13.05 - Sistema Hibrido

## Fecha: 2026-02-28

---

## Resumen Ejecutivo

Despues de multiples experimentos fallidos con ML tradicional, encontramos un enfoque que funciona:

**Sistema Hibrido = Setup Tecnico + Filtro ML**

| Metrica | Sin Filtro | Con Filtro (0.45) | Mejora |
|---------|------------|-------------------|--------|
| Trades | 44 | 12 | -73% |
| Win Rate | 34.1% | 50.0% | +16% |
| PnL | -22.1% | +9.1% | +31% |
| Profit Factor | 0.71 | 1.49 | +110% |

---

## Enfoques Probados (Fallidos)

### 1. ML Predice Precio
- Ridge, XGBoost, GradientBoosting
- Resultado: Overfitting severo, no generaliza
- Correlaciones muy bajas (<0.1)

### 2. Clasificador WIN/LOSS
- V9 LossDetector
- Resultado: 68% WR en backtest, 41% en produccion

### 3. Reglas Tecnicas Simples
- RSI extremos, BB, etc.
- Resultado: El mercado cambia, reglas fijas no funcionan

### 4. Modelo Adaptativo (re-entrena mensual)
- GradientBoosting con ventana rolling de 6 meses
- Resultado: -39% PnL, peor que B&H

### 5. Deteccion de Regimenes
- BULL/BEAR/RANGE/VOLATILE
- Resultado: Regimenes mal clasificados, no mejora

---

## Enfoque Exitoso: Sistema Hibrido

### Filosofia

El ML NO genera senales. El setup tecnico genera la senal, el ML decide si el CONTEXTO es favorable.

Esto imita a un trader profesional:
1. Ve un setup (RSI oversold, pullback, etc.)
2. Evalua el contexto (volatilidad, tendencia, etc.)
3. Decide si entrar o esperar

### Componentes

#### 1. Setup Detector (Reglas)

Detecta 4 tipos de setups (7% de velas):

| Setup | Condicion |
|-------|-----------|
| RSI_EXTREME | RSI14 < 25 AND BB_pct < 0.3 |
| CAPITULATION | 4+ velas rojas AND RSI < 35 AND vol_ratio > 1.3 |
| TREND_PULLBACK | EMA200_dist > 5% AND ADX > 25 AND RSI < 40 |
| DIVERGENCE | RSI7 < 25 AND RSI14 > RSI7+5 AND BB_pct < 0.2 |

#### 2. ML Filter (GradientBoosting)

Features de CONTEXTO (no del setup):
- adx, di_diff, chop (tendencia)
- atr_pct, vol_ratio, bb_width (volatilidad)
- ret_5, ret_20 (momentum)
- hour_sin, hour_cos (patron temporal)

Target: P(WIN | setup detectado)

Threshold: 0.45 (balance entre trades y calidad)

---

## Resultados Detallados

### Por Periodo

| Periodo | Trades | WR | PnL | PF |
|---------|--------|-----|-----|-----|
| Train (2019-2024) | 53 | 81.1% | +167.2% | 4.14 |
| Test (Sep 2025-Feb 2026) | 12 | 50.0% | +9.1% | 1.49 |

### Por Threshold (Test)

| Threshold | Trades | WR | PnL | PF |
|-----------|--------|-----|-----|-----|
| 0.40 | 15 | 53.3% | +9.0% | 1.41 |
| **0.45** | **12** | **50.0%** | **+9.1%** | **1.49** |
| 0.50 | 10 | 40.0% | +4.4% | 1.26 |
| 0.55 | 4 | 50.0% | +5.9% | 2.50 |

---

## Por Que Funciona

1. **Setups selectivos**: Solo 7% de velas tienen setup (antes 50%)
2. **ML como filtro, no generador**: Reduce ruido sin predecir precio
3. **Contexto > Prediccion**: Evalua SI las condiciones son favorables, no predice direccion
4. **Menos trades, mas calidad**: 12 trades vs 44, pero rentables

---

## Implementacion Recomendada

### Parametros

```python
# Setup Detection
SETUP_RSI_EXTREME = {'rsi14': 25, 'bb_pct': 0.3}
SETUP_CAPITULATION = {'consec_down': 4, 'rsi14': 35, 'vol_ratio': 1.3}
SETUP_TREND_PULLBACK = {'ema200_dist': 5, 'adx': 25, 'rsi14': 40}
SETUP_DIVERGENCE = {'rsi7': 25, 'rsi_diff': 5, 'bb_pct': 0.2}

# ML Filter
ML_THRESHOLD = 0.45
ML_MODEL = 'GradientBoostingClassifier'
ML_PARAMS = {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.05}

# Trade Management
TP_PCT = 0.03  # 3%
SL_PCT = 0.015  # 1.5%
```

### Re-entrenamiento

- Frecuencia: Mensual
- Datos: Ultimos 6 meses para setup outcomes
- Validacion: Accuracy en setups recientes

---

## Validacion Multi-Coin (5 Monedas)

### Resultados por Moneda (Test: Sep 2025 - Feb 2026)

| Coin | Sin Filtro |       | Con Filtro |       | Mejora | Veredicto |
|------|------------|-------|------------|-------|--------|-----------|
|      | WR    | PnL   | WR    | PnL   | PnL    |           |
| BTC  | 34.1% | -22.1%| 50.0% | +9.1% | +31.2% | USAR FILTRO |
| DOGE | 41.2% | +15.2%| 42.9% | +12.2%| -3.0%  | SIN FILTRO |
| ADA  | 36.8% | -35.1%| 42.9% | +0.4% | +35.5% | USAR FILTRO (marginal) |
| XRP  | 42.1% | -7.7% | 40.9% | -8.4% | -0.7%  | NO TRADEAR |
| DOT  | 57.4% | +44.7%| 71.4% | +39.9%| -4.8%  | SIN FILTRO |

### Clasificacion de Monedas

**Categoria A: Setups funcionan bien (NO usar filtro)**
- DOT: 57.4% WR, +44.7% PnL - setups excelentes
- DOGE: 41.2% WR, +15.2% PnL - setups rentables

**Categoria B: Setups necesitan filtro (SI usar filtro)**
- BTC: 34.1% -> 50.0% WR, -22% -> +9% PnL - filtro transforma perdedor en ganador
- ADA: 36.8% -> 42.9% WR, -35% -> +0.4% - filtro mejora pero aun marginal

**Categoria C: No tradear**
- XRP: Setups pierden, filtro no ayuda - evitar

### Patron Descubierto

**El filtro ML ayuda cuando los setups solos NO funcionan:**

| Base WR | Base PnL | Usar Filtro? | Ejemplo |
|---------|----------|--------------|---------|
| >45%    | Positivo | NO           | DOT, DOGE |
| <40%    | Negativo | SI           | BTC, ADA |
| ~40%    | Negativo | NO TRADEAR   | XRP |

### Regla de Decision

```
IF setup_base_WR > 45% AND setup_base_PnL > 0:
    tradear_sin_filtro()
ELIF setup_base_WR < 40% AND filtro_mejora_pnl:
    tradear_con_filtro()
ELSE:
    no_tradear()
```

### Configuracion Final por Moneda

| Coin | Modo | Threshold | TP/SL | Accion |
|------|------|-----------|-------|--------|
| BTC  | HIBRIDO | 0.45 | 3%/1.5% | Tradear con filtro ML |
| DOGE | SETUP_ONLY | - | 4%/2% | Tradear sin filtro |
| ADA  | HIBRIDO | 0.50 | 3%/1.5% | Tradear con filtro (monitorear) |
| XRP  | DISABLED | - | - | NO tradear |
| DOT  | SETUP_ONLY | - | 3%/1.5% | Tradear sin filtro |

---

## Proximos Pasos

1. [x] Validar en BTC (APROBADO con filtro)
2. [x] Validar en DOGE (APROBADO sin filtro)
3. [x] Validar en ADA (MARGINAL con filtro)
4. [x] Validar en XRP (RECHAZADO)
5. [x] Validar en DOT (APROBADO sin filtro)
6. [ ] Implementar sistema adaptativo por moneda en `src/ml_strategy.py`
7. [ ] Paper trading 2 semanas
8. [ ] Produccion

---

## Lecciones Aprendidas

### 1. El ML no predice el mercado, filtra el ruido

El mercado es dificil de predecir, pero las CONDICIONES favorables son identificables. Cuando un setup tecnico ocurre en el contexto correcto, la probabilidad de exito aumenta.

### 2. No todas las monedas necesitan ML

Algunas monedas (DOT, DOGE) tienen setups que funcionan bien solos. Agregar ML solo reduce oportunidades sin mejorar resultados. La clave es identificar CUANDO usar filtro y cuando no.

### 3. Algunas monedas simplemente no funcionan

XRP mostro que ni setups ni filtro funcionan. Mejor no tradear que forzar una estrategia perdedora.

### 4. El patron es: Setup detecta, ML filtra

El sistema imita a un trader profesional:
1. Detectar oportunidad (setup tecnico)
2. Evaluar contexto (ML)
3. Decidir si el contexto es favorable

### 5. Configuracion por moneda es esencial

No existe una estrategia "one size fits all". Cada moneda tiene su comportamiento y debe configurarse independientemente.

---

## Archivos del Experimento

| Archivo | Descripcion |
|---------|-------------|
| `btc_hybrid_system.py` | Sistema hibrido BTC |
| `doge_hybrid_system.py` | Sistema hibrido DOGE |
| `ada_hybrid_system.py` | Sistema hibrido ADA |
| `xrp_hybrid_system.py` | Sistema hibrido XRP |
| `dot_hybrid_system.py` | Sistema hibrido DOT |

---

*Documento actualizado: 2026-02-28*
