# Sizing de V2 con un pronóstico de volatilidad (HAR-RV)

> **Pre-registro escrito el 2026-09-23, ANTES de correr nada.** El diseño, las
> hipótesis, los umbrales y el número de comparaciones quedan fijados aquí.
> Los resultados se añaden debajo, sin tocar esta sección.
>
> Origen: la revisión de GitHub del 2026-09-23 (idea 2 del ranking). Es el
> único uso de ML con señal medida en los datos del proyecto:
> `predictibilidad/` encontró que `atr_pct` predice la *magnitud* del
> movimiento, no la dirección.

## Qué NO cambia

Las reglas de V2 (`PARAMS_V2`), las entradas, las salidas y el trailing no
cambian. El conjunto de trades es **idéntico** en todos los brazos: solo cambia
el tamaño de cada posición. Por eso no hace falta el null sintético: este
experimento no toca el timing.

## Por qué no se entrena sobre los trades

El modelo se ajusta sobre **todas las velas** (unos 2.600 días de varianza
realizada), no sobre los 165 trades. Tiene 4 parámetros. El presupuesto de
información (`presupuesto_informacion/`) no aplica.

## Etapa 1: ¿el pronóstico es mejor? (con muchos datos)

- **Varianza realizada diaria:** suma de la varianza Garman-Klass de las 6
  velas 4h del día UTC (`data/BTC_USDT_4h_full.parquet`).
- **Objetivo:** varianza realizada media de los **2 días siguientes** (mediana
  de duración de un trade de V2: 9 velas; media: 12,4).
- **HAR (el modelo de ML):** `log RV(t+1..t+2) ~ β0 + βd·log RV_d + βw·log RV_5d + βm·log RV_22d`.
  Mínimos cuadrados con ventana **expansiva**, reajuste mensual y al menos
  365 días de entrenamiento. Corrección de sesgo log-normal: `exp(μ + s²/2)`.
- **Control (estimador tosco):** `log RV(t+1..t+2) ~ a + b·log(atr_pct²)`, con
  el mismo esquema expansivo y el `atr_pct` de V2 al cierre del día.
- **Métrica:** QLIKE fuera de muestra, `RV/F − log(RV/F) − 1`, desde el primer
  día con 365 días de entrenamiento.
- **Test:** diferencia media de QLIKE (control − HAR), con bootstrap
  estacionario (`arch`, bloques de ~20 días) y H0: diferencia ≤ 0.
- **Regla de parada:** si p ≥ 0,05, **el experimento acaba aquí** con
  resultado negativo. Un mejor pronóstico es condición necesaria.

## Etapa 2: ¿el sizing es mejor? (con los 165 trades)

Tres brazos con los mismos trades de `portfolio_sim` (BTC, costes nuevos, 2% de
riesgo, `max_concurrent=1`):

| brazo | notional |
|---|---|
| **B** (actual) | `equity · riesgo / trail` |
| **A** (vol-target con el control) | `equity · riesgo / (z · σ̂_ATR)` |
| **H** (vol-target con HAR) | `equity · riesgo / (z · σ̂_HAR)` |

- `σ̂` es el pronóstico de volatilidad a 2 días del **día anterior** a la
  entrada: sin look-ahead.
- `z` es una sola constante de escala, igual para A y H, que iguala el
  notional medio al de B. Como la métrica primaria no depende de la escala,
  `z` no cambia el test. Se reporta para que CAGR y DD sean comparables.
- Se mantiene el tope `max_notional_pct = 2,5`.

**Métrica primaria:** Sharpe por trade, `mean(r)/std(r)` con
`r = pnl / equity_entrada`. No depende de `z`.

**Hipótesis (2 comparaciones, Bonferroni α = 0,025 cada una):**
- **H1:** Sharpe(H) > Sharpe(B).
- **H2:** Sharpe(H) > Sharpe(A), es decir, el valor del pronóstico HAR frente
  al estimador tosco.

**Test:** bootstrap **pareado** por bloques de trades (bloques circulares de 10
trades, 20.000 réplicas). Se remuestrean los mismos índices en todos los
brazos. p = P(ΔSharpe ≤ 0).

**Criterio de adopción:** se adopta solo si H1 **y** H2 pasan a α = 0,025, y
además el DD máximo de H con escala `z` no es peor que el de B. Si pasa solo
H1, la mejora viene del vol-targeting y no del ML, y se reporta así.

**Secundarias (descriptivas, sin test):** CAGR, DD máximo, PF y peor trade por
brazo, también al 4,5% de riesgo.

**Adenda al pre-registro (2026-09-23, también antes de correr):** los trades
anteriores al primer pronóstico disponible (el primer año sirve para entrenar)
usan el sizing de B en los tres brazos. Los tests de H1 y H2 se calculan solo
sobre los trades **con** pronóstico. El conjunto de trades sigue siendo
idéntico en los tres brazos.

## Riesgos conocidos antes de empezar

- V2 ya dimensiona por el inverso del trail (ATR×k), así que B ya es un sizing
  por volatilidad. La mejora marginal esperada es pequeña.
- `tonykark1/btc-realized-volatility` encontró que un mejor pronóstico de
  volatilidad en BTC no produjo una mejor estrategia.
- Con 165 trades, un bootstrap pareado detecta solo diferencias de Sharpe
  grandes.

---

## Resultados

> Corrida el 2026-09-23 · salida completa en `salida.txt`.

### Etapa 1: el pronóstico HAR **no** es mejor. Se aplica la regla de parada.

2.126 días evaluados fuera de muestra (2020-05-01 → 2026-02-24).

| pronóstico | QLIKE medio | corr(log real, log pronóstico) |
|---|--:|--:|
| **HAR** (4 parámetros: diario, semanal, mensual) | 0,3120 | 0,684 |
| **Control ATR** (2 parámetros) | **0,3072** | **0,690** |

Mejora de HAR sobre el control: **−0,0048 (−1,6%)**, **p = 0,84**. El HAR
pronostica un poco *peor* que el `atr_pct` que V2 ya usa.

**El experimento termina aquí.** La etapa 2 no se corrió, como exige el
pre-registro: si el pronóstico no es mejor, no hay nada que el sizing pueda
aprovechar.

### Lectura

- La volatilidad de BTC sí es predecible (correlación ~0,69 con la realizada),
  pero **el ATR de 14 velas 4h ya captura esa predictibilidad**. Añadir las
  componentes semanal y mensual del HAR no aporta a 2 días vista.
- Encaja con lo que ya decía `predictibilidad/`: `atr_pct` es la única feature
  por encima del ruido, y V2 ya dimensiona con ella (`notional = riesgo / trail`,
  con `trail = ATR × k`). **V2 ya hace el sizing por volatilidad que el ML podría
  aportar.**
- También coincide con `tonykark1/btc-realized-volatility`: en BTC, los mejores
  pronósticos de volatilidad no se tradujeron en mejores estrategias.

### Lo que NO se probó, a propósito

El brazo A (vol-targeting con el control ATR frente al sizing por trail) sería
una hipótesis nueva, y además no es ML. Correrlo ahora, después de ver este
resultado, sería una búsqueda post-hoc. Si se quiere probar, debe tener su
propio pre-registro.

### Veredicto

**RECHAZADO.** El ML de volatilidad no mejora el insumo que V2 ya usa para
dimensionar. Es el octavo negativo de ajustes sobre V2, y el primero en
aplicar pre-registro y regla de parada.
