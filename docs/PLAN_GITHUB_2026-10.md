# Plan: 5 ideas de la revisión de GitHub (2026-10-18)

> Origen: segunda revisión de GitHub (unos 30 repos y papers, sin repetir los 25
> de `INVESTIGACION_GITHUB_ML_2026-09.md`). Ninguno publica resultados en vivo
> auditables; las cifras citadas son las que declaran sus autores.
> **Este plan no toca el bot en vivo.** Todo corre en `experiments/` con
> `portfolio_sim` y costes nuevos.

## Estado

| # | estado | resultado |
|---|---|---|
| E0 | hecho (2026-10-19) | MinTRL = 65 trades ≈ 2,8 años (suelo). `experiments/benchmark_bh/` |
| E1 | hecho (2026-10-19) | **H1 no pasa** (p=0,33). V2 CAGR/DD 0,58 contra 0,39-0,53 de B&H a igual exposición. Vol objetivo empeora a B&H. La Mixta 50/50 da 0,72 (descriptivo, sin testear) |
| E4 | hecho (2026-10-19) | **H4 no pasa** (p=0,44); en el null sintético resta en el 80% de las series. Dato colateral: 4,5% fijo toca el kill en el 7,5% de las series sintéticas de 7 años. `experiments/dd_sizing/` |
| E2 | hecho (2026-10-19) | **No aplica**: 0 reducciones en 165 trades (la vol de 20 días no se mueve en trades de ~2 días). `experiments/vol_target_pos/` |
| E3 | hecho (2026-10-19) | **H3 no pasa**: CAGR/DD 0,30 contra 0,55 de V2, DD 53,7%, null p=0,21. `experiments/donchian_ensemble/` |
| E5 | hecho (2026-10-19) | **H5 no pasa**: CAGR/DD 0,44 contra 0,52 a igual exposición, null p=0,13. `experiments/piramide/` |

**Balance: 0 de 4 tests pasan** (E2 no llegó a activarse). Lo único que sale con mejor CAGR/DD que V2 es la mezcla 50/50 con BTC (0,72), y es descriptivo, sin testear.

## Reglas comunes (pre-registro)

1. **Familia de 5 tests → α = 0,01 por test** (Bonferroni 0,05/5). Se declara
   ahora; no se añaden variantes después de ver resultados.
2. **Una sola configuración por idea**, fijada en este documento. Si falla, se
   acepta el no. No hay "probemos con otro parámetro".
3. **Comparar siempre a igual exposición media.** `experiments/riesgo/` mostró
   que el riesgo es apalancamiento puro (CAGR/DD ≈ 0,50-0,56 en todos los
   niveles). Cualquier regla que reduzca la exposición baja el DD "gratis", y
   eso no es una mejora. **Métrica principal: CAGR/DD** (o DD a igual CAGR).
4. **Si la idea cambia el timing** (qué trades existen), se exige además el
   null sintético de `bootstrap_bloques/sintetica()`. Si solo cambia el
   tamaño, basta con el bootstrap pareado por bloques (como `vol_sizing/`).
5. Un README por experimento: el pre-registro primero y los resultados debajo,
   sin tocar el pre-registro.

## Orden y por qué

| # | idea | cambia | cambio en el simulador | coste |
|---|---|---|---|---|
| E1 | Benchmark B&H a igual exposición | nada (es la vara de medir) | ninguno (usa `Resultado.equity`) | bajo |
| E4 | Riesgo según la caída del capital | tamaño | ninguno (hook `sizing` al entrar) | bajo |
| E2 | Vol objetivo con la posición abierta | tamaño | **sí**: reajustar el notional por vela | medio |
| E3 | Ensemble de periodos de Donchian | timing | variante del motor (sub-posiciones) | medio |
| E5 | Piramidación Turtle | timing | **sí**: varias entradas por posición | alto |

E1 va primero porque **decide si el resto importa**: si un B&H de BTC con vol
objetivo empata o gana a V2 en CAGR/DD, mejorar V2 deja de ser la pregunta.

---

## E0 (complemento, sin test) — cuántos trades de paper hacen falta

Calcular el **Minimum Track Record Length** (Bailey y López de Prado; ref.
`tschm/jsharpe`) con el Sharpe por trade, la asimetría y la curtosis de los 165
trades, para un PSR del 95% contra Sharpe 0. Sustituye o confirma el "≥30
trades" de CLAUDE.md. Ojo: el MinTRL es paramétrico y asume trades
independientes, así que es un **suelo**, no una garantía.

## E1 — V2 frente a comprar BTC con la misma exposición

**Fuente:** `naplesblue/riskbound` (benchmark a igual exposición).
Responde al punto 4 pendiente de CLAUDE.md.

**Brazos** (todos sobre 2019-01 → 2026-02, curva de equity diaria):

| brazo | definición |
|---|---|
| V2 | `portfolio_sim`, BTC, 4,5%, `max_notional_pct=2,0` (config viva) |
| B&H-w | BTC long siempre con exposición constante `w` = exposición media de V2 (notional/equity promediado en **todos** los días, contando 0 fuera de posición). Rebalanceo diario, costes de rebalanceo incluidos |
| B&H-vt | BTC long siempre con `w_t = min(1, σ*/σ̂_t)`, σ̂ = vol realizada 20 días (GK de las velas 4h), σ* = mediana **expandente** de σ̂ (sin look-ahead), reescalado al final para que la exposición media sea la de V2. Rebalanceo solo si `|Δw| > 0,1` |
| Mixta | 50% B&H-w + 50% V2, rebalanceo mensual |

**Hipótesis H1:** CAGR/DD de V2 > CAGR/DD de B&H-vt.
**Test:** bootstrap por bloques de calendario (30 días, circular) sobre la
diferencia de retornos diarios log entre V2 y B&H-vt, ajustados a igual
volatilidad. p < 0,01.

**Lectura pre-acordada:**
- Si V2 no gana a B&H-vt → **decisión del usuario**: V2 añade riesgo de
  ejecución sin ganar a la regla más simple. Se documenta; no se cambia nada
  en vivo sin que el usuario lo decida.
- Si V2 gana → primera evidencia de que el timing aporta algo **en relación con
  el riesgo**, aunque no supere al null en retorno.
- La Mixta es descriptiva: se reporta la frontera CAGR/DD, no se testea.

## E4 — Bajar el riesgo según la caída del capital

**Fuente:** portfolio insurance (CPPI).
**Regla única:** `riesgo_t = 4,5% × max(0,25, 1 − DD_t / 45%)`, con `DD_t` la
caída desde el pico de equity al abrir el trade. Se aplica con el hook
`sizing` de `PortfolioSim`. Los trades son idénticos a V2.

**Comparación justa:** V2 a riesgo fijo **igual al riesgo medio que usó E4**
(no al 4,5%).

**Métricas:**
- DD histórico y CAGR/DD.
- P(kill en 3 años) por Monte Carlo. Para que la regla actúe sobre el camino,
  se remuestrean los retornos **por unidad de riesgo** (`r / riesgo`) en
  bloques de 10 trades y se aplica la regla en cada camino.
- Repetido sobre 200 series sintéticas (`sintetica`), porque `riesgo/` ya
  avisó de que el remuestreo de trades rompe las rachas de régimen y
  subestima el DD.

**Hipótesis H4:** CAGR/DD de E4 > V2 a igual riesgo medio (bootstrap pareado
por bloques de trades, p < 0,01).
**Expectativa honesta:** solo ayuda si las pérdidas vienen en rachas. Si no, la
regla se queda pequeña justo antes de la recuperación (el edge de V2 se
concentra en pocos regímenes). La evidencia práctica de los filtros sobre la
curva de capital es negativa.

## E2 — Vol objetivo con la posición abierta

**Fuente:** `OctopusTakopi/crypto-trend-following` (DD 43,1% → 30,1% en su
libro, con el efecto sobre el retorno no resuelto).
**Diferencia con `vol_sizing/` (rechazado):** allí se probaba si un
*pronóstico* HAR mejoraba el ATR al entrar. Aquí no se pronostica nada: se
reajusta el tamaño **durante** el trade con la vol realizada.

**Regla única:** en cada vela 4h con posición abierta,
`notional_t = notional_entrada × min(1, σ_entrada / σ̂_t)`, con σ̂ = vol
realizada 20 días. Solo reduce, nunca añade. Se rebalancea si
`|Δnotional| > 20%`, pagando comisión taker y slippage sobre el delta.

**Cambio en el simulador:** `PortfolioSim.run()` necesita reajustar
`Pos.notional` por vela y cobrar el coste del delta. Va en una rama/flag
nueva, sin alterar el camino por defecto, con un test que verifique que, sin
la flag, las cifras de V2 salen idénticas.

**Comparación justa:** V2 a riesgo fijo con la misma exposición media.
**Hipótesis H2:** DD menor a igual CAGR (bootstrap pareado por bloques de
calendario, p < 0,01). Los trades son los mismos, así que no hace falta el null.

## E3 — Promediar varios periodos de Donchian

**Fuente:** Zarattini, Pagani y Barbon, "Catching Crypto Trends" (SSRN
5209907). Sin código público; el PDF no se pudo leer (403), así que la
adaptación es nuestra.

**Regla única:**
- Periodos en velas 4h: **{20, 55, 120, 240, 480}**.
- 5 sub-señales independientes. Cada una entra con la ruptura de su canal y
  sale con el **mismo trailing ATR de V2**: se cambia una sola cosa.
- Riesgo por sub-posición = 4,5% / 5. El filtro de régimen EMA50/200 diario y
  la componente F no cambian.

**Motivo:** quitar el grado de libertad "55" sin buscar el mejor periodo. Se
declara como **una variante más** en la familia de V2: sumarla a la cuenta de
selección (m pasa de 36 a 37).

**Hipótesis H3:** el ensemble supera al null sintético (200 series, bloques
como en `bootstrap_bloques/`, p < 0,01) **y** su CAGR/DD ≥ V2.
**Expectativa honesta:** incluye el 55 y las sub-señales estarán muy
correlacionadas. Lo más probable es que también capture solo la deriva. Si es
así, se cierra igual que las otras siete búsquedas de parámetros.

## E5 — Piramidación tipo Turtle

**Regla única** (valores por defecto de las Tortugas, sin tunear):
- Unidad = 4,5% / 4 de riesgo.
- Entrada inicial: 1 unidad con la señal de V2.
- Añadir 1 unidad cada `0,5 × ATR(20)` a favor, máximo 4 unidades.
- Stop común para todas: el trailing ATR de V2.
- Tope de notional total `max_notional_pct=2,0`.

**Cambio en el simulador:** varias entradas por posición, con precio medio y
comisión por cada añadido (fill al open t+1, como las entradas).

**Comparación justa:** V2 a riesgo fijo con la misma exposición media.
**Hipótesis H5:** CAGR/DD > V2 **y** supera al null sintético (p < 0,01),
porque cambia qué exposición existe y cuándo.
**Expectativa honesta:** cambia la forma de la ganancia (más convexa en
tendencias largas), pero en las rupturas fallidas se paga más. Es la octava
búsqueda de parámetros: una sola configuración y aceptar el no.

---

## Qué pasa con los resultados

- Ninguna idea pasa a vivo por pasar su test: requiere además la decisión
  explícita del usuario y la regla de CLAUDE.md (paper trade antes de capital
  real).
- Cada experimento, pase o no, se añade al mapa de familias de
  `experiments/carry_funding/README.md` y a la tabla de "Qué pasó con cada
  idea".

## Descartado en esta revisión (no se planifica)

- **Basis trimestral y arbitraje de funding entre exchanges:** familia carry ya
  rechazada; el basis rinde menos que el bono de EE. UU. a 2 años desde febrero
  de 2026.
- **Grid, DCA y martingala** (NostalgiaForInfinity, passivbot): WR alto y ruina
  oculta, la firma de los cinco fracasos históricos.
- **Filtro sobre la curva de capital:** evidencia práctica negativa. E4 cubre la
  versión con fundamento.
- **Cascadas de liquidación:** K33 midió una mediana de −1,1% a 30 días.
  Revisar de forma descriptiva cuando `record_stream.py` tenga 6-12 meses.
- **Opciones en Deribit:** el histórico cuesta dinero y recortan las ganancias
  justo en los rallies.

## Fuentes

- `naplesblue/riskbound` (MIT): benchmark a igual exposición, vol objetivo,
  Holm p=0,002 sobre el DD en acciones de EE. UU.
- `OctopusTakopi/crypto-trend-following` (MIT): CTA en 788 perps de Binance con
  ablaciones y bootstrap estacionario.
- Zarattini, Pagani y Barbon (2025), "Catching Crypto Trends", SSRN 5209907.
- `pst-group/pysystemtrade` (GPL-3): pronóstico continuo y vol objetivo (Carver).
- `tschm/jsharpe` (MIT): PSR, DSR y MinTRL.
- K33 sobre liquidaciones (vía CryptoSlate); Glassnode sobre el basis (vía TFTC).
