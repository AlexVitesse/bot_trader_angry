# Riesgo por trade: qué compra cada punto de riesgo

> 2026-09-23 · `test_riesgo.py` · salida en `salida.txt`. Es un estudio
> **descriptivo**, no un test: no cambia la señal, solo muestra el reparto
> entre retorno y drawdown para decidir `ML_RISK_PER_TRADE`. No toca el bot.

**Configuración de V2:** `portfolio_sim` con BTC, costes nuevos (Fase 4),
`max_concurrent=1` y **`max_notional_pct=2,0`**, el tope que usa el bot en vivo
(`ML_MAX_NOTIONAL_PCT`). Por ese tope las cifras al 4,5% difieren un poco de las
de `bootstrap_bloques/` (2,5×).

**Monte Carlo:** bootstrap por bloques circulares de 10 trades sobre los `r` de
cada nivel. Horizonte de 3 años (~70 trades), 20.000 caminos. Kill switch al
45% de DD (`ML_MAX_DD_PCT`).

## Resultados

| riesgo | CAGR (historia) | DD (historia) | CAGR/DD | P(kill en 3 años) | DD mediano 3 años | DD p95 3 años | CAGR p5 3 años | P(perder en 3 años) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1,0% | +5,6% | 11,2% | 0,50 | 0,0% | 6,3% | 11,5% | −0,6% | 7,1% |
| 1,5% | +8,4% | 16,5% | 0,51 | 0,0% | 9,3% | 16,8% | −0,9% | 7,1% |
| 2,0% | +11,1% | 21,5% | 0,51 | 0,0% | 12,3% | 21,8% | −1,2% | 7,0% |
| 2,5% | +13,7% | 26,3% | 0,52 | 0,0% | 15,1% | 26,5% | −1,8% | 7,5% |
| 3,0% | +16,4% | 30,9% | 0,53 | 0,2% | 17,9% | 31,3% | −2,3% | 8,0% |
| 3,5% | +18,9% | 35,3% | 0,54 | 0,7% | 20,6% | 35,6% | −3,2% | 8,5% |
| 4,0% | +21,4% | 39,5% | 0,54 | 1,8% | 23,2% | 39,6% | −3,9% | 8,7% |
| **4,5% (desplegado)** | **+23,5%** | **42,4%** | 0,55 | **3,0%** | 25,1% | 42,5% | −4,3% | 8,8% |
| 5,0% | +25,3% | 45,3% | 0,56 | 4,6% | 26,8% | 44,5% | −4,8% | 9,1% |

## Lectura

1. **El riesgo es apalancamiento puro.** El cociente CAGR/DD se mantiene entre
   0,50 y 0,56 en todos los niveles. Cada punto de riesgo compra retorno y
   drawdown casi en la misma proporción. No hay un nivel "óptimo" oculto: la
   elección depende solo de cuánto drawdown se tolera.
2. **Al 4,5% el sistema vive pegado al kill switch.** El DD histórico (42,4%)
   queda a 2,6 puntos del corte del 45%. En 3 años la probabilidad de tocarlo
   es del 3,0% según el remuestreo. El remuestreo **subestima** esa
   probabilidad, porque rompe las rachas largas de régimen, y es justo ahí
   donde nacen los DD grandes.
3. **La probabilidad de acabar 3 años en pérdida (~7-9%) casi no depende del
   riesgo.** La marca la calidad de la señal, no el tamaño.
4. **Las cifras de CAGR son un techo.** Salen de la historia completa (con la
   selección de la variante incluida), y la Fase 4 mostró que el timing de V2
   no supera a un null con la misma deriva.

## Opciones (decisión del usuario, no aplicada)

| opción | riesgo | CAGR hist. | DD hist. | comentario |
|---|--:|--:|--:|---|
| Mantener | 4,5% | +23,5% | 42,4% | Cerca del objetivo del 30% pero con DD de 42% y el kill switch a 2,6 pts |
| Intermedio | 3,0-3,5% | +16-19% | 31-35% | ~10 pts de margen hasta el kill switch; P(kill) < 1% |
| Conservador | 2,0% | +11,1% | 21,5% | Cumple el DD < 25% del `CLAUDE.md`; retorno por debajo del mínimo del 30% |

Ningún nivel cumple a la vez los dos mínimos de `CLAUDE.md` (retorno ≥ 30% y
DD < 25%). Con esta señal, subir el riesgo para acercarse al 30% obliga a
aceptar un DD de más del 40%.
