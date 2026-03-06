# V15 — Comparacion de Estrategias (Exploracion Multi-Rama)

**Fecha**: 2026-03-06
**Contexto**: Exploracion de 4 estrategias trend-following puras como alternativa a V14 (countertrend, OOS PF=0.86, -3%/yr)
**Resultado**: Ninguna aprobada standalone. Mejor candidato: Estrategia B (Breakout) para integrar al comite expertos V15.

---

## Por que se creo V15

V14 BTC tenia un problema fundamental: era COUNTERTREND (RSI<40 oversold).

- En bear market (2022, 2025), los bounces son efimeros → WR cae a 27%
- OOS 2022-2026: WR=26.9%, PF=0.86, -3%/yr (el sistema pierde dinero)
- Mejoras intentadas (1H confirmation, macro filter, trailing stop) → todas RECHAZADAS
- Diagnostico: el setup en si es malo, no sus parametros

**Hipotesis V15**: ir A FAVOR de la tendencia, como hacen los traders exitosos.

---

## Las 4 Estrategias Probadas

### Estrategia A — EMA Trend Following
**Rama**: `v15/ema-trend`
**Script**: `validate_v15_a_ema_trend.py`
**Idea**: Entrar cuando EMA20>EMA50>EMA200 alineadas, precio cerca de EMA20, RSI 38-62 (pausa en tendencia, no oversold). RR 2:1.

**Resultado**: WF **4/12** RECHAZADO

| Metrica | Valor |
|---------|-------|
| Folds OK | 4/12 |
| OOS WR | 38.4% |
| OOS PF | 0.73 |
| Trades/mes | 12.1 |
| Retorno anual | -50% |

**Por que falla**: Demasiadas seniales en condiciones mixtas. 2022-2024 tiene WR < 35% en casi todos los semestres. Solo 2020 y 2025 funcionan bien (tendencias fuertes claras).

**Diferencia con V14**: Genera mas trades pero con menor calidad. El problema no es la frecuencia sino que EMA alignment no es suficiente discriminador.

---

### Estrategia B — Momentum Breakout ⭐ (Mejor candidato)
**Rama**: `v15/momentum-breakout`
**Scripts**: `validate_v15_b_breakout.py`, `v2`, `v3`
**Idea**: BB estrecho (consolidacion) + ruptura del maximo de 20 barras + volumen confirma. SL bajo minimo de consolidacion (logica de trader real). RR 1.5:1.

**Resultados (3 iteraciones)**:

| Version | WF | WR | PF | t/m | Notas |
|---------|----|----|----|----|-------|
| v1 (original) | 4/12 | 59.2% | 1.90 | 1.2 | Calidad excelente, frecuencia insuficiente |
| v2 (relajado) | 5/12 | 49.5% | 1.37 | 2.4 | Frecuencia sube pero calidad baja |
| v3 (+macro) | 5/12 | 61.3% | 2.87 | 0.9 | Calidad excelente, frecuencia aun peor |

**Breakdown clave de v2** (explica todo):
- Seniales nuevas (vol<1.8): N=30, WR=36.7%, PF=0.87 → danan la calidad
- Seniales originales (vol>=1.8): N=65, WR=55.4%, PF=1.67 → excelentes

**Conclusion**: La calidad del setup es genuina (WR=55-61% en multiples tests). El problema es que solo genera 1-2 trades/mes de alta calidad — insuficiente para cumplir WF (>3 trades por fold de 6 meses).

**Lo que se aprendio**: El breakout B es el mejor GENERADOR DE SETUPS encontrado. No puede funcionar solo pero es ideal como capa "setup" en el comite expertos.

---

### Estrategia C — Multi-Timeframe Alignment
**Rama**: `v15/mtf-alignment`
**Script**: `validate_v15_c_mtf.py`
**Idea**: Alinear daily (EMA20>EMA50) + 4H (EMA20>EMA50>EMA200) + RSI 42-65. TP=3xATR, SL=1.5xATR. RR 2:1.

**Resultado**: WF **3/12** RECHAZADO

| Metrica | Valor |
|---------|-------|
| Folds OK | 3/12 |
| OOS WR | 37.5% |
| OOS PF | 0.91 |
| Trades/mes | 29.2 |
| Retorno anual | -39% |

**Por que falla**: Genera demasiadas seniales (29/mes) pero WR < 40% en casi todos los semestres. El ATR-based TP/SL ayuda en volatilidad pero no resuelve el problema de discriminacion.

**Diferencia con A**: Similar nivel de fallo pero con mas ruido (mas trades = mas perdidas).

---

### Estrategia D — Funding Rate Sentiment
**Rama**: `v15/funding-sentiment`
**Script**: `validate_v15_d_funding.py`
**Idea**: Usar funding rate z-score < -1.5 como trigger (mercado overshorted = squeeze potencial). Requiere macro alcista (EMA20>EMA50 diario). TP=4xATR, SL=2xATR.

**Resultado**: WF **3/10** RECHAZADO (solo 10 folds desde 2021 cuando hay datos de funding)

| Metrica | Valor |
|---------|-------|
| Folds OK | 3/10 |
| OOS WR | 42.0% |
| OOS PF | 0.73 |
| Trades/mes | 4.4 |
| Retorno anual | -19% |

**Por que falla**: El funding extremo no es suficientemente predictivo por si solo. En 2022, funding muy negativo pero el mercado seguia bajando (no hay squeeze si el bear es estructural). 2025-H2 fue catastrofico (WR=20%).

**Lo que se aprendio**: Funding como trigger principal no funciona. Pero como FILTRO ADICIONAL en el comite expertos tiene valor (no entrar si funding es extremadamente positivo = overshorted).

---

## Resumen Comparativo

| Estrategia | WF | OOS WR | OOS PF | t/m | Retorno/yr |
|------------|----|----|----|----|-----------|
| A - EMA Trend | 4/12 | 38.4% | 0.73 | 12.1 | -50% |
| B - Breakout v1 | 4/12 | 59.2% | 1.90 | 1.2 | +26% |
| B - Breakout v2 | 5/12 | 49.5% | 1.37 | 2.4 | +11% |
| B - Breakout v3 | 5/12 | 61.3% | 2.87 | 0.9 | +15% |
| C - MTF | 3/12 | 37.5% | 0.91 | 29.2 | -39% |
| D - Funding | 3/10 | 42.0% | 0.73 | 4.4 | -19% |
| **V14 BTC (baseline)** | - | 26.9% | 0.86 | 3.0 | -3% |

> **Criterio de aprobacion**: WF>=7/12, WR>=50%, PF>=1.3, >=3 trades/mes

---

## Por que Ninguna Pasa el WF

El WF usa 12 semestres (ventanas de 6 meses). Para aprobar un fold necesita: WR>40%, PF>1.2, n>=3.

El 2022 es el asesino de todas las estrategias:
- 2022-H1: BTC cayo -75%. LONG-only + bear market = perdidas sistematicas.
- 2022-H2: recuperacion parcial, pero WR sigue bajo.
- Con solo EMA/breakout/MTF → no hay forma de detectar que el mercado estructuralmente esta bajando.

Strategies B v1/v2/v3 tienen 4-5 folds OK pero casi siempre fallan en 2021 (choppy), 2022-H1 (bear) y 2025 (rango bajista).

---

## Conclusion y Siguiente Paso

**Lo que funciona de B**: cuando el breakout ocurre con condiciones correctas (vol>=1.8, consolidacion real, ADX bajo), la senial es genuinamente predictiva: WR=55-61%, PF=1.67-2.87.

**El problema es estructural**: una sola capa (setup tecnico) no puede distinguir:
- Breakout valido en bull market → TP
- Falso breakout en bear market → SL

**Solucion**: Comite de expertos V15 (plan ya definido):
1. **Macro Context** (reglas): ¿BULL/BEAR/RANGE? → Estrategia B como setup
2. **Price Action Setup** (ML): Usar el detector de B como feature base
3. **Funding Sentiment** (reglas): Veto si funding extremo en contra
4. **Volume & Order Flow** (ML): Confirmar que hay conviccion

La estrategia B por si sola genera 55-61% WR cuando dispara. Con el filtro macro + funding veto, el objetivo es mantener esa calidad y alcanzar 3-5 trades/mes en regimenes favorables.

---

## Estado de Ramas

| Rama | Estado | Archivo Principal |
|------|--------|-------------------|
| `v15/ema-trend` | RECHAZADA, archivada | `validate_v15_a_ema_trend.py` |
| `v15/momentum-breakout` | MEJOR CANDIDATO, 3 iteraciones | `validate_v15_b_breakout_v3.py` |
| `v15/mtf-alignment` | RECHAZADA, archivada | `validate_v15_c_mtf.py` |
| `v15/funding-sentiment` | RECHAZADA, funding como filtro util | `validate_v15_d_funding.py` |

---

## Archivos Creados en esta Exploracion

```
v15_framework.py                      # Framework compartido: load/features/sim/metrics/WF
validate_v15_a_ema_trend.py           # Estrategia A
validate_v15_b_breakout.py            # Estrategia B v1
validate_v15_b_breakout_v2.py         # Estrategia B v2 (vol relajado)
validate_v15_b_breakout_v3.py         # Estrategia B v3 (+macro filter)
validate_v15_c_mtf.py                 # Estrategia C
validate_v15_d_funding.py             # Estrategia D

docs/V15_A_EMA_TREND_results.md       # Resultados A
docs/V15_B_BREAKOUT_results.md        # Resultados B v1
docs/V15_B_BREAKOUT_v2_results.md     # Resultados B v2
docs/V15_B_BREAKOUT_v3_results.md     # Resultados B v3
docs/V15_C_MTF_results.md             # Resultados C
docs/V15_D_FUNDING_results.md         # Resultados D
docs/V15_COMPARACION.md               # Este archivo (resumen general)
```
