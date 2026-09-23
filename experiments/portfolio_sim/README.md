# Simulador de cartera — resultados

> Fecha: 2026-08-10 · `portfolio_sim.py` + `run_calibration.py`
> Construido tras la revisión externa, que señaló que toda la calibración
> previa simulaba un BTC secuencial idealizado y no el sistema desplegado.

## Qué modela, y qué corrige

Replica el sistema **efectivo**: varios pares simultáneos, hasta
`ML_MAX_CONCURRENT` posiciones, tope de 2 en la misma dirección, **equity
compartido** (las posiciones concurrentes compiten por el capital), margen
finito, y correlación real (implícita al usar las series simultáneas).

Corrige tres divergencias vivo-vs-backtest:

1. **Fill posterior al cierre.** La señal se detecta con el `close` de la vela
   t; la entrada se ejecuta al **`open` de t+1**. El backtest anterior entraba
   al mismo close que generaba la señal — un fill imposible en vivo.
2. **`max_bars` del motor** (A=60, F=40), no `ML_MAX_HOLD` (15/30).
3. **DD con el capital inicial como primer pico.**

Sin look-ahead intrabar: el stop se comprueba con el nivel de la vela anterior
y se actualiza después.

## Resultado 1 — la config defendible es peor de lo que se había reportado

BTC solo, sin `F_SHORT`, 165 trades, 2019-2026, PF 1,77 estable (costes
**viejos**, ver abajo):

| risk/trade | CAGR | DD |
|--:|--:|--:|
| 1,0% | +6,6% | 10,2% |
| 2,0% | +13,2% | 19,7% |
| 3,0% | +19,6% | 28,5% |
| **4,5%** | **+29,0%** | **40,4%** |
| 6,0% | +36,6% | 49,1% |

**bootstrap p (i.i.d.) = 0,004 — un solo número, no uno por nivel de riesgo.**

> ⚠️ Corrección 2026-09 (`docs/AUDITORIA_2026-09.md` §1.1). Esta tabla
> mostraba antes una columna de p por nivel de riesgo, todas ≈0,004, y eso se
> leyó como "p estable en todos los niveles de riesgo". Es una tautología:
> `notional = equity·risk/trail` y `r = pnl/equity_entrada`, así que
> `r ∝ risk_pct` y escalar `r` por una constante no cambia el signo de su
> media. Los cinco p eran el mismo número (el 0,005 a 6% solo aparecía porque
> muerde el tope de notional 2,5×). La robustez del p frente a la dependencia
> temporal y a la selección de variante se mide en
> `experiments/bootstrap_bloques/`.

### Con costes nuevos (desde 2026-09)

`PortfolioSim(..., costes='nuevos')` es ahora el **default**;
`costes='viejos'` reproduce las cifras de arriba. Cambios (§1.7 de la
auditoría): slippage 0,02%/lado (antes 0,01%), funding histórico real de BTC
en vez de la mediana constante, y stop llenado a `min(stop, open)` cuando la
vela abre con gap por debajo del stop.

| costes | risk | n | PF | CAGR | DD | p i.i.d. |
|---|--:|--:|--:|--:|--:|--:|
| viejos | 2,0% | 165 | 1,77 | +13,2% | 19,7% | 0,004 |
| **nuevos** | **2,0%** | 165 | **1,65** | **+11,1%** | **21,5%** | 0,008 |
| viejos | 4,5% | 165 | 1,77 | +29,0% | 40,4% | 0,004 |
| **nuevos** | **4,5%** | 165 | **1,65** | **+23,9%** | **43,4%** | 0,008 |

Descomposición a 2% (cada coste añadido solo sobre los viejos): slippage
−0,3 pts de CAGR, funding histórico −0,2 pts (+1,4 pts de DD), **gap del stop
−1,6 pts** (PF 1,77 → 1,68). El gap es el coste que faltaba y el mayor.

Comparado con la calibración del 2026-08-09 (que daba +32,7% con "DD p95 39%"
a 4,5%): **el fill honesto cuesta ~3,7 puntos de CAGR** y el DD real es peor.

**Para llegar al 30% anual en BTC hacen falta ~4,6% de riesgo por trade y hay
que aceptar un drawdown en torno al 41%.** Ése es el precio real.

## Resultado 2 — los números multi-par NO son creíbles

| config (risk 2%) | n | tr/año | PF | CAGR | DD |
|---|--:|--:|--:|--:|--:|
| 5 pares, `F_SHORT` activo | 688 | 96 | 1,65 | **+65,9%** | 25,1% |
| 5 pares, sin `F_SHORT` | 508 | 71 | 1,77 | **+57,3%** | 31,8% |
| BTC solo, sin `F_SHORT` | 165 | 23 | 1,77 | +13,2% | 19,7% |

Un 5x de retorno por 1,3x de drawdown sería una comida gratis. No lo es. Tres
razones para desconfiar:

1. **4 de los 5 pares fallaron significancia individualmente** en
   `v2_all_coins/`: BNB p=0,315 · ETH p=0,472 · OP p=0,539 · DOGE falla
   edge-vs-null por drift natural. Solo BTC pasa 3/3. Agregar cuatro edges no
   significativos y obtener p=0,000 es la trampa de comparaciones múltiples.
2. **Los 5 pares se seleccionaron de entre 22** en base a pruebas previas. La
   cartera hereda ese sesgo de selección completo.
3. **Correlación media 0,69** entre los 5 pares (BTC-ETH: 0,83). Con hasta 2
   posiciones en la misma dirección sobre activos que se mueven juntos,
   *"2% de riesgo por trade"* es en realidad ~4% sobre un único factor. El
   retorno extra es apalancamiento disfrazado, no diversificación.

Es exactamente la firma que `CLAUDE.md` enseña a desconfiar, y la misma que
tumbó a los 18 pares en la auditoría de mayo.

## Dato operativo nuevo

`max_misma_direccion` rechazó **177-211 señales** según config. El tope
cross-pair de `can_open` es muy activo y ninguna simulación anterior lo
modelaba: el backtest asumía que todas las señales se ejecutaban.

## Limitaciones que quedan

- ~~Funding constante~~ y ~~sin gaps~~: corregido en 2026-09 (`costes='nuevos'`).
  El funding histórico solo cubre BTC desde 2020-01; antes y en otros pares
  se usa la constante.
- **DD es un único camino histórico**, no una distribución. La crítica de la
  revisión sobre las permutaciones sigue aplicando: permutar preserva el
  retorno final, así que no estima ruina futura.
- **`OP/USDT` solo tiene datos desde 2022-06**, así que la cartera no es
  homogénea en el tiempo.
- Los parquets terminan en **2026-02-27**; falta refrescar datos.

## Resultado 3 — walk-forward real: la vía multi-par queda RECHAZADA

`run_walkforward.py`. Selección de pares decidida **solo con train** (regla
declarada de antemano: ≥15 trades y PF ≥1,20), 6 folds de test independientes
nunca usados para decidir. Warmup preservado.

| estrategia | folds + | mediana | compuesto | peor fold |
|---|--:|--:|--:|--:|
| selección WF | 4/6 | +7,6% | +279,3% | −10,9% |
| siempre 5 pares | 4/6 | +12,1% | +599,3% | −13,8% |
| solo BTC | **2/6** | −2,6% | +37,9% | −6,8% |

Parece que el multi-par gana. **No.** Por fold:

| fold | test | selección WF | solo BTC |
|---|---|--:|--:|
| **1** | 2021-01 → 2021-11 | **+182,8%** | −2,0% |
| 2 | 2021-11 → 2022-10 | −7,4% | −3,2% |
| 3 | 2022-10 → 2023-08 | −10,9% | −6,8% |
| 4 | 2023-08 → 2024-06 | +40,5% | +35,1% |
| 5 | 2024-06 → 2025-05 | +10,7% | +22,3% |
| 6 | 2025-05 → 2026-03 | +4,6% | −5,6% |

**Toda la ventaja está en el fold 1, el bull run de 2021.** Excluyéndolo:
selección WF **+34,2%** vs solo BTC **+40,7%** — BTC solo gana.

Cinco altcoins con correlación 0,69 en un bull market suben juntas; con sizing
sobre equity, el compounding de posiciones correlacionadas produce ese +267%.
Es beta apalancada sobre un régimen que no se repite, no edge.

Tres corolarios:

1. **La regla de selección no aporta**: es peor que usar siempre los 5 pares
   (+279% vs +599%). Si seleccionar con información de train no mejora nada,
   lo que hay no es una señal seleccionable.
2. **BTC solo pasa 2/6 folds.** El requisito de `CLAUDE.md` es ≥7/12 (≈4/6).
   **V2 falla su propio walk-forward con fills honestos.** Los "WF 7/12"
   históricos venían de particiones in-sample del mismo ajuste.
3. El payoff de BTC (pierde poco a menudo, gana mucho de vez en cuando) es la
   forma clásica de un trend follower: expectativa positiva (+6,6%/año
   compuesto) con solo 33% de folds ganadores.

## Conclusión

1. La calibración honesta para BTC sin `F_SHORT` es **2% → +13,2% anual con DD
   19,7%**. El 30% cuesta ~41% de drawdown. *(2026-09, costes nuevos: 2% →
   +11,1% / DD 21,5%; 4,5% → +23,9% / DD 43,4%. El p resiste el bootstrap por
   bloques pero no el null sintético con selección: ver
   `experiments/bootstrap_bloques/`.)*
2. **La vía multi-par está rechazada**: su ventaja es un artefacto de 2021.
3. **V2 no supera el walk-forward del propio proyecto** (2/6 folds). No es
   candidato a capital real en su forma actual.

Lo que queda vivo: `A_LONG` sigue siendo el componente con mejor evidencia
(PF 1,91 aislado) y sigue **sin probarse en una ventana alcista en vivo**. Los
folds 1 y 4 —los dos alcistas— son los únicos donde el sistema gana de verdad,
lo que es coherente con un trend follower que necesita tendencia. Eso apunta a
que la pregunta correcta no es "cómo saco más retorno" sino "cómo sé que estoy
en un régimen donde esto funciona".
