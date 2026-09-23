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

## Riesgos conocidos antes de empezar

- V2 ya dimensiona por el inverso del trail (ATR×k), así que B ya es un sizing
  por volatilidad. La mejora marginal esperada es pequeña.
- `tonykark1/btc-realized-volatility` encontró que un mejor pronóstico de
  volatilidad en BTC no produjo una mejor estrategia.
- Con 165 trades, un bootstrap pareado detecta solo diferencias de Sharpe
  grandes.

---

## Resultados

*(se completan después de correr `test_vol_sizing.py`)*
