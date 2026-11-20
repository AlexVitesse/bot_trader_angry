# Bot AGRESIVO con ML: candidatos en todos los regímenes y riesgo adaptativo

> **Diseño fijado el 2026-09-23, ANTES de correr el backtest.** Decisión del
> usuario: un bot agresivo con ML, con reglas para cada caso (alcista, bajista
> y rango) y un riesgo que decide el propio modelo. Corre en la **cuenta demo**;
> V2 pasa a paper interno. El código es `src/agresivo_engine.py`, que es la
> única fuente de verdad del backtest y del vivo.

## Diseño (fijo)

**Universo.** BTC, ETH, SOL, BNB, XRP, DOGE, ADA, LINK y AVAX (perpetuos USDT),
en velas de 4h desde 2020.

**Candidatos.** Se evalúan en la vela cerrada `i`. Hay como mucho uno por
dirección y vela, con prioridad `comp > don55 > don20 > mr`. No se aplica
filtro de régimen: lo decide el modelo.

| tipo | LONG | SHORT |
|---|---|---|
| `don20` / `don55` | `close > max(high)` de las N velas previas | `close < min(low)` de las N previas |
| `comp` | ruptura de 20 velas con `bb_width[i-1] < q20(bb_width, 100)[i-1]` | ídem, a la baja |
| `mr` (rango) | `close < banda inferior BB(20,2)` | `close > banda superior` |

**Etiqueta.**
- La salida es la de V2 A: trailing sin look-ahead (se comprueba primero el
  stop de la vela anterior), con `trail = clip(atr_pct·2,5; 2,5%; 6%)` y un
  máximo de 60 velas.
- La entrada es al open de `i+1`, y el stop se llena con gap.
- Costes: 0,06% por lado más funding. Hasta 2026-11-20 era constante, de
  0,013% cada 8h (los longs pagan y los shorts cobran). Ahora es el
  **histórico de cada par** (ver "Funding real" al final).
- `r_R` es el PnL dividido por el trail, en múltiplos de R. `y = r_R > 0`.

**Features (11).** Todas son generalizables entre pares y llevan el signo de la
dirección cuando aplica:
- `atr_pct`, `bb_width`;
- `dist_ema200_diaria·d`, `ret20·d`, `ret60·d`;
- distancia de BTC a su EMA200 diaria `·d` (régimen del mercado);
- `dir` y el tipo de disparador en one-hot.

La EMA diaria usa un shift de un día.

**Modelo.**
- `HistGradientBoostingClassifier(max_depth=3, min_samples_leaf=200, l2=1.0, max_iter=200, lr=0.05)`.
- `sample_weight` = unicidad media por par (López de Prado).
- Se calibra con Platt sobre el último 20% temporal del train.
- Un solo modelo para todo el panel. Ventana **expansiva**; en el walk-forward
  se reentrena **cada mes** con los eventos cuya salida es anterior al inicio
  del mes menos 60 velas.

**Riesgo adaptativo.**
- `b` = ganancia media en R entre pérdida media en R (del train).
- Kelly: `f = p − (1−p)/b`.
- `riesgo = clip(0,5·f; 0; 10%)`.
- No opera si el riesgo queda por debajo del 1%.
- Se multiplica por `max(0, 1 − DD/50%)`.

**Cartera del backtest.**
- Equity compartido, máximo 3 posiciones, máximo 2 en la misma dirección y una por par.
- `notional = equity·riesgo/trail`, con tope de 2,5× equity y margen a 5×.
- Fill al open siguiente. Si hay varias señales, se toman por riesgo descendente.

**Ventana OOS.** De 2021-01 al final de los datos (2026-02).

**Comparaciones.** V2 (`portfolio_sim`, BTC, 4,5%) y buy & hold de BTC en la
misma ventana. p por bootstrap de bloques de 10 trades sobre `mean(r) > 0`. AUC
fuera de muestra sobre todos los candidatos.

---

## Resultados

> Corrida el 2026-09-23 · salida completa en `salida.txt`. No se ajustó nada
> después de ver los resultados: es una sola variante.

### El modelo no discrimina

- 25.657 eventos etiquetados en los 9 pares, con 37,2% de ganadores. Por tipo:
  `mr` 14.577, `don55` 5.285, `don20` 3.017, `comp` 2.778.
- **AUC fuera de muestra: 0,497** sobre 21.208 candidatos de 2021-2026. Es azar.
  Por año va de 0,478 a 0,527.
- **Calibración:** el modelo es sobreconfiado. Donde pronostica 40-50% de
  aciertos, la tasa real es 37%.

| p pronosticada | n | p media | tasa real | r_R medio |
|---|--:|--:|--:|--:|
| ≤ 0,3 | 677 | 0,275 | 0,343 | −0,135 |
| 0,3-0,4 | 13.240 | 0,368 | 0,377 | −0,004 |
| 0,4-0,5 | 6.960 | 0,427 | 0,370 | −0,055 |
| 0,5-0,6 | 309 | 0,531 | 0,382 | −0,017 |
| > 0,6 | 22 | 0,624 | 0,455 | +0,149 |

- Con `b = 1,61` (ganancia media 1,11 R, pérdida media 0,69 R), Kelly abre
  riesgo en 8.142 de los 21.208 candidatos (riesgo medio 3,6%), y el r_R medio
  de esos candidatos es **−0,045**. El sizing adaptativo apuesta fuerte
  justo donde no hay ventaja.

### La cartera pierde la mitad en el primer año

| sistema (2021-01 → 2026-02) | CAGR | DD máx | PF | trades/año |
|---|--:|--:|--:|--:|
| **Agresivo con ML** | **−12,6%** | **50,5%** | 0,32 | 381 |
| V2 al 4,5% | +7,5% | 40,9% | 1,26 | 22 |
| Buy & hold BTC | +16,8% | 77,0% | — | — |

- p (bootstrap por bloques, mean r > 0) = **0,98**.
- **2021: −50%** con 664 trades. A partir de ahí el throttle por drawdown
  (`riesgo × max(0, 1 − DD/50%)`) deja el riesgo en casi cero y la cuenta queda
  congelada en ×0,50 hasta 2026. Sin ese throttle habría seguido cayendo.
- Por tipo, la reversión a la media (`mr`) es el grueso de los trades (1.519) y
  de la pérdida (suma de r −0,66). `don55` es el único con WR cercana al 50%,
  pero con muestra pequeña.
- Por dirección, los shorts (1.136) concentran la pérdida (−0,70). Los longs
  quedan en +0,03.

### Lectura honesta

1. **Es peor que V2 y peor que comprar y mantener BTC**, en retorno y en
   riesgo de ruina.
2. **Cubrir todos los regímenes con ML no crea ventaja.** Los candidatos de
   rango y bajistas tienen expectativa negativa después de costes, y el modelo
   no sabe separarlos (AUC 0,50). Es el mismo resultado que ya dieron los
   experimentos de mean-reversion (`agent_I`, PF 0,66), SHORT (PF 0,88) y los
   clasificadores ML (AUC 0,51-0,52), ahora con 9 pares y 25.000 eventos.
3. **El riesgo adaptativo con Kelly amplifica el problema.** Kelly solo
   funciona si `p` está bien estimada. Con un modelo sin poder de
   discriminación y sobreconfiado, convierte ruido en apuestas grandes.

### Veredicto

**RECHAZADO para capital real.** No recomiendo desplegarlo tal cual ni
siquiera en demo como sustituto de V2.

## Para producción (si aun así se quiere correr en demo)

- **Entrenamiento en vivo:** klines 4h de los 9 pares desde 2020 vía fapi
  (unas 14.000 velas por par, ~10 llamadas de 1.500 por par). Calcular
  `features` + `eventos_panel` y `entrenar` tarda ~2 s en este equipo. Se
  reentrena una vez al mes con `reentrenar(feats, t)`.
- **Bundle:** ~100 KB (pickle): el modelo HGB, Platt y `b`.
- **Contrato de `señal_vivo(bundle, dfs_por_par, df_btc, posiciones_abiertas, dd_actual)`:**
  - Los `dfs` son OHLCV 4h **sin la vela en curso**, y `df_btc` igual.
  - Devuelve una lista ordenada por `risk_pct` descendente de:
    `{'pair', 'direction' (int 1/-1), 'side' ('LONG'/'SHORT'),
    'price' (close de la vela de señal), 'trail_mode': 'tight',
    'trail_fixed_dist', 'max_bars': 60, 'risk_pct' (0,01-0,10),
    'confidence' (p), 'setup': 'agr_<tipo>', 'engine': 'agresivo'}`.
  - El bot tiene que usar `risk_pct` de la señal como riesgo por trade (hoy
    `open_position` usa `ML_RISK_PER_TRADE × sizing_mult`).

## 2026 desde $100 (corrido el 2026-11-20, `backtest_2026.py`)

Mismo motor y cartera, arrancando el 1-ene-2026 con $100 y DD desde cero.
Datos completados con fapi hasta el 19-nov. Salida: `salida_2026.txt`.

| tramo | final | DD máx | trades | WR | PF | LONG | SHORT |
|---|--:|--:|--:|--:|--:|--:|--:|
| 2026-01-01 → 11-19, **funding real** | **$60,39 (−39,6%)** | 47,1% | 289 | 35,3% | 0,66 | 150, −$38,2 | 139, −$1,5 |
| 2026-01-01 → 11-19, funding constante (antes) | $67,66 (−32,3%) | 45,1% | 273 | 37,7% | 0,79 | 128, −$40,7 | 145, +$8,4 |
| 2026-10-17 → 11-19 (mismo arranque que el vivo, funding constante) | $105,77 (+5,8%) | — | 22 | 45% | — | 6, −$21,8 | 16, +$27,6 |
| **Vivo (paper)**, mismo tramo | $114,5 (+14,5%) | — | 18 | 38,9% | 1,58 | 5, −$7,8 | 13, +$22,3 |

BTC −29,5% en el año. Pierde en 9 de 11 meses; el riesgo se va frenando solo
con el DD (de ~3,5% a ~0,3%), por eso a partir de junio casi no se mueve.
El vivo comparte la forma (shorts ganan, longs pierden en un mercado que
cae) y ~12 de sus 18 trades coinciden con el simulado; va +9 puntos por encima.
Las diferencias vienen de que casi todas las señales tienen p ≈ 0,40-0,45 y el
umbral de operar (riesgo ≥ 1%) cae en p ≈ 0,40: cualquier diferencia pequeña
cambia qué trades entran. Dos fuentes medidas (vela 2026-10-18 16:00):
- **EMA200 diaria con 1500 velas 4h (250 días) en vivo** contra la historia
  completa en el backtest: la EMA no ha convergido. `btc_ema_d` +0,098 en vivo
  contra +0,066; `dist_ema_d` de XRP 0,039 contra 0,085.
- **Fecha de corte del modelo**: el vivo reentrena en cada reinicio del
  proceso (el bundle vive en memoria) y el backtest el día 1 de cada mes.
(Una primera versión de esta tabla decía −2,0% y 14 trades: `predicciones`
arrancaba en el primer día de mes posterior a `DESDE` y se saltó octubre.)

## Por qué pierden los LONG (2026-11-20, `analisis_longs.py`)

Diagnóstico descriptivo sobre 23.928 candidatos 2021 → nov-2026, con la p
walk-forward de su mes y r en R con costes (`salida_longs.txt`). **No es una
búsqueda de filtros**: las celdas buenas se eligieron mirando. Las cifras de
abajo son con el funding constante; con el real (`salida_longs.txt` actual)
cambia lo que se dice en "Funding real".

- Dos fuentes de pérdida: los `mr` LONG (comprar la caída bajo la BB inferior:
  6.509 candidatos, −457 R) y los LONG con **BTC bajo su EMA200 diaria**
  (−395 R). Las rupturas LONG con BTC arriba ganan (`don55` +0,090 R,
  `don20` +0,100 R): es la lógica de V2.
- El modelo no aprende el régimen: opera 1.997 LONG con BTC abajo, todos con
  r negativo. AUC LONG 0,512; p 0,47-0,54 → acierto real 0,377 (sobreconfiado).
  En SHORT es peor: AUC 0,478, los que opera rinden −0,098 R y los que
  descarta +0,019 R.
- Los años malos de los LONG son 2022 (−0,29 R operados, 97% de candidatos
  con BTC abajo) y 2026 (−0,14 R, 76%). Los 5 LONG en vivo de nov-2026 son
  `mr` con BTC abajo: la peor celda.
- Quitar `mr` y exigir BTC > EMA200 en LONG convierte el agresivo en V2
  multipar, que ya se rechazó. Para probarlo habría que pre-registrarlo y
  pasarlo por el null sintético.

## Funding real (2026-11-20)

La revisión de `experiments/agresivo_revision/` encontró que el funding
constante (0,013%/8h ≈ 14%/año a favor de los shorts) no se parece al real.
Ahora `agresivo_engine.features(df, btc, funding)` guarda en la columna `fund`
lo cobrado en el open de cada vela (00/08/16 UTC), las etiquetas lo suman
entre la entrada y la salida, y `backtest.simular` lo cobra vela a vela.
Datos: `fapi/v1/fundingRate` (cache en `data/*_USDT_funding.parquet`). El bot
en vivo lo descarga al reentrenar; sin red usa la constante de antes.

Funding real anual (suma de tasas, %): BTC 30,6 en 2021, 4,2 en 2022 y 3,0 en
2026; SOL −38,0 en 2022 (los shorts **pagaron**); BNB negativo de 2022 a 2025.

| | constante | real |
|---|--:|--:|
| Cartera 2021 → 2026: CAGR / DD | −12,6% / 50,5% | −12,6% / 51,1% |
| AUC OOS | 0,497 | 0,495 |
| 2026 desde $100 | −32,3% | **−39,6%** |
| 2026, SHORT | 145 trades, +$8,4 | 139, **−$1,5** |
| Candidatos SHORT con BTC bajo EMA200 (r medio) | +0,019 R | **+0,003 R** |
| Candidatos LONG, r medio | −0,026 R | −0,017 R |
| SHORT que el modelo opera / descarta | −0,098 / +0,019 R | −0,098 / +0,004 R |

La cartera 2021-26 no cambia porque toca el freno por DD en 2021 y queda casi
congelada. **Lo único de los shorts que parecía ganar era el funding
regalado.** Con el real, ni en 2026 (BTC −29,5%) los shorts dejan dinero.
