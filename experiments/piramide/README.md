# E5 — Piramidación tipo Turtle

> **Pre-registro escrito el 2026-10-19, ANTES de correr nada.** Viene de
> `docs/PLAN_GITHUB_2026-10.md` (E5), con las precisiones de implementación.
> Familia de 5 tests → **α = 0,01**. Cambia la exposición y el momento en que
> existe, así que se exige el null sintético. Es la octava búsqueda de
> parámetros: una sola configuración, y se acepta el no.

## Regla (valores por defecto de las Tortugas, sin tunear)

- **Unidad** = 4,5% / 4 = **1,125%** de riesgo: `notional = equity_entrada × 1,125% / trail`.
- **Entrada:** 1 unidad con la señal de V2 (A o F), con el mismo trail y el
  mismo `max_bars`.
- **Añadidos:** 1 unidad cada vez que el close supera
  `entrada_inicial + k × 0,5 × N` (k = 1, 2, 3), con N = ATR(20) de velas 4h
  (media simple del true range) **fijado en la entrada**. Máximo 4 unidades.
- **Ejecución del añadido:** decidido al close, llenado al **open** de la vela
  siguiente (la misma política que las entradas de V2). La posición pasa a
  tener precio medio. Comisión y slippage sobre cada añadido.
- **Stop común** para todas las unidades: el trailing ATR de V2, que sigue el
  máximo desde la entrada.
- **Topes:** notional total ≤ 2,0 × equity y margen con apalancamiento 4.
- Se usa la **equity de entrada** para dimensionar los añadidos (no la
  vigente). Es una simplificación declarada.
- Implementado como hook opcional `piramide` en `PortfolioSim`. Sin hook, V2 da
  exactamente sus cifras (lo comprueba un `assert`).

## Comparación justa

**V2-m**: V2 a riesgo fijo escalado para igualar la **exposición media** de E5
(notional/equity por vela, 0 fuera de posición). Se itera 3 veces.

## Hipótesis y test

**H5** (deben cumplirse las dos):
1. **CAGR/DD de E5 > CAGR/DD de V2-m** en la historia.
2. **Null sintético:** 200 series `sintetica()` con bloques de 180 velas.
   Estadístico: CAGR de E5. `p` = fracción de series con CAGR ≥ el observado.
   **p < 0,01.**

Como referencia se reporta también el p de V2-m con las mismas series.

**Expectativa honesta:** hace la ganancia más convexa en tendencias largas,
pero en las rupturas fallidas se paga más. Con trades de ~2 días de mediana,
muchos no llegarán a +0,5 N antes de salir.

---

## Resultados (2026-10-19) — `test_piramide.py`, salida en `salida.txt`

En 2 de cada 3 trades se añade al menos una unidad; el 43,6% llega a las 4.

| brazo | riesgo | exposición | PF (PnL $) | CAGR | DD | CAGR/DD |
|---|--:|--:|--:|--:|--:|--:|
| **E5** | 1,125% por unidad | 0,097 | 1,49 | +13,7% | 30,8% | **0,44** |
| V2-m | 2,64% | 0,097 | 1,65 | +14,5% | 27,6% | **0,52** |
| V2 4,5% | 4,50% | — | 1,65 | +23,5% | 42,4% | 0,55 |

(El "PF 1,65" que imprime la salida para E5 está calculado sobre `r`. Coincide
con V2 por azar; sobre el PnL en dólares es 1,49. Lo comprobé: los trades son
los mismos y los `r` son distintos.)

PnL de E5 según cuántas unidades se añadieron:

| añadidos | trades | PnL |
|--:|--:|--:|
| 0 | 55 | −10.285 |
| 1 | 19 | −3.769 |
| 2 | 19 | −3.502 |
| 3 | 72 | +32.537 |

**Null sintético:** E5 p = **0,130**; V2-m p = 0,085.

**H5: NO pasa** en ninguna de las dos condiciones.

### Lectura

1. **A igual exposición, piramidar es peor que entrar con todo de una vez**
   (CAGR/DD 0,44 contra 0,52), y queda más cerca del null.
2. Las tendencias que funcionan (los 72 trades con 4 unidades) sí pagan. Pero
   como la entrada es de 1/4 de tamaño, la ganancia temprana de cada ruptura se
   cobra pequeña. Mecanismo probable, no medido: con las 4 unidades el precio
   medio sube, y la vuelta desde el máximo hasta el trail devuelve más.
3. Los trades sin añadidos pierden poco cada uno, pero son 55. El neto no
   compensa lo que se deja de ganar al entrar con 1/4.
4. Confirma lo que ya mostraban E3 y E4: el retorno de V2 está en **entrar con
   todo al romper**, una posición a la vez, y salir con el trail. Repartir la
   entrada, repartir el periodo o recortar tras pérdidas lo empeoran.
