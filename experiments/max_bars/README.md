# `max_bars` — séptimo negativo

> Fecha: 2026-08-22 · `test_max_bars.py` · BTC 2019-09 → 2026-03 (6,5 años)

## La hipótesis

V2 tiene perfil de trend-following medido: WR 45%, ratio ganancia/pérdida 2,23,
skew +1,55, y los 10 mejores trades aportan el 45% de la ganancia bruta.

El trend-following vive de *dejar correr a los ganadores*. Pero `a_max_bars=60`
sobre velas de 4h son **10 días** de hold máximo, y `f_max_bars=40` son 6,7.
Los Turtles mantenían posiciones meses. ¿Está el reloj decapitando a los
ganadores antes de que sean grandes?

## El resultado: no

| salida | n | % | pnl medio |
|---|--:|--:|--:|
| SL | 72 | 55,0% | −2,13% |
| TP | 56 | 42,7% | +4,42% |
| **TIMEOUT** | **3** | **2,3%** | **+11,03%** |

Solo el 2,3% de los trades muere por reloj. El trailing ATR resuelve el 97,7%
antes de llegar al límite. **El reloj no muerde.**

Barrido completo, con desglose por fold para no mirar solo el agregado:

| A/F velas | días | n | WR | PF | anual | DD | hold real | folds+ |
|---|--:|--:|--:|--:|--:|--:|--:|:--|
| 30/20 | 5,0 | 142 | 46,5% | 1,82 | +20,0% | −14,7% | 1,9 d | `+-+++` |
| **60/40** | **10,0** | **131** | **45,0%** | **1,83** | **+19,1%** | **−14,7%** | **2,3 d** | `+-+++` |
| 90/60 | 15,0 | 130 | 45,4% | 1,82 | +18,4% | −14,7% | 2,3 d | `+-+++` |
| 120/80 | 20,0 | 130 | 45,4% | 1,82 | +18,4% | −14,7% | 2,3 d | `+-+++` |
| 180/120 | 30,0 | 130 | 45,4% | 1,82 | +18,4% | −14,7% | 2,3 d | `+-+++` |
| 240/160 | 40,0 | 130 | 45,4% | 1,82 | +18,4% | −14,7% | 2,3 d | `+-+++` |

De 60 velas en adelante los números son **idénticos**: mismo n, mismo PF, mismo
DD. Cuadruplicar el límite no cambia un solo trade. `max_bars` no es una
palanca; está muy por encima de donde el trailing ya ha cerrado todo.

## Lo que sí queda de aquí

**1. V2 no es trend-following, es swing trading.** El hold medio real son
**2,3 días**, no semanas. Y `a_donchian_n=55` sobre velas de 4h es una ruptura
de **9,2 días**, no de 55 — más rápida que Turtle S1 (20 d), no más lenta. El
proyecto venía describiendo V2 como "Turtle clásico"; la medida dice otra cosa.
Eso no lo invalida, pero cambia la clase de comparación.

**2. Divergencia vivo-vs-backtest abierta.** `ML_MAX_HOLD = {'BULL': 30,
'BEAR': 30, 'RANGE': 15}` corta a **2,5 días en RANGE**, mientras el motor
asume 60 velas (10 d). Con hold medio de 13,8 velas, ese tope de 15 muerde
justo en la cola larga — y los 3 TIMEOUT de la muestra promedian **+11,03%**.
El efecto agregado es pequeño (la fila 30/20 rinde +20,0% vs +19,1%), pero es
divergencia real. Sigue siendo el punto 2 de "Parte 7 — Abierto" de
`docs/SESION_2026-08-09.md`.

## Marcador

Séptima búsqueda de retorno, séptimo rechazo. `PARAMS_V2` sigue en su óptimo
local. Las seis anteriores: filtro ADX, techo del trail, suelo del trail,
temporalidad diaria, detector de régimen, conviction sizing.
