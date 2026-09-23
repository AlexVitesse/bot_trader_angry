# Predictibilidad por fuente: ¿hay algún sitio con más señal que BTC 4h?

> **Pre-registro escrito el 2026-09-23, ANTES de correr nada.** Los resultados
> se añaden debajo sin tocar esta sección.
>
> **Pregunta:** `predictibilidad/` midió un R² in-sample de 0,068% para BTC 4h
> con features técnicas. ¿Existe alguna fuente, horizonte o familia de features
> con los datos locales donde un modelo simple prediga **fuera de muestra** lo
> suficiente como para justificar un proyecto nuevo de ML?
>
> Es un **estudio de predictibilidad, no una estrategia.** No hay backtest de
> trading. La traducción a costes es un cálculo aproximado y explícito.

## Celdas: lista cerrada (10)

| # | fuente | datos | familia | horizontes |
|--:|---|---|---|---|
| 1-2 | BTC 1h | `btcusdt_1h.parquet` (2022-01 → 2026-03) | (a) técnicas | h = 1 vela, h = 24 velas |
| 3-4 | BTC 1d | `btcusdt_1d_v15.parquet` (2019-09 → 2026-03) | (a) técnicas | h = 1 día, h = 5 días |
| 5-6 | Panel 4h (21 monedas) | `*_4h_full.parquet` | (b) cross-sectional | h = 1 vela, h = 6 velas |
| 7-8 | Panel 1d (21 monedas) | `*_4h_full.parquet` remuestreado a 1D UTC | (b) cross-sectional | h = 1 día, h = 5 días |
| 9-10 | BTC 4h + derivados | `BTC_USDT_4h_full` + `deriv_oi_btcusdt_5m` + `btc_v15_funding` (2020-09 → 2026-02) | (c) derivados | h = 1 vela, h = 6 velas |

No se descarga nada: todos los datos están en local.

## Features (fijas)

Todos los retornos son logarítmicos. Cada feature se calcula con información
disponible al cierre de la vela `t`.

- **(a) técnicas, BTC 1h:**
  - retorno pasado de 1, 3, 6, 12, 24, 72 y 168 velas;
  - volatilidad (std de retornos de 1 vela) a 24 y 168 velas;
  - `log(high/low)` de la vela.
- **(a) técnicas, BTC 1d:**
  - retorno pasado de 1, 3, 5, 10, 20 y 60 días;
  - volatilidad a 10 y 30 días;
  - `log(high/low)`.
- **(b) cross-sectional:** por moneda y fecha, se toman las features y a cada
  una se le resta la **media de ese momento entre todas las monedas**.
  - **Panel 4h:** retorno pasado de 1, 6, 42 y 180 velas, volatilidad a 42
    velas y `log(high/low)`.
  - **Panel 1d:** retorno pasado de 1, 5, 20 y 60 días, volatilidad a 20 días
    y `log(high/low)`.
  - Solo cuentan las fechas con al menos 8 monedas con datos.
- **(c) derivados, BTC 4h:**
  - cambio log del OI a 1, 6 y 42 velas;
  - funding vigente;
  - z-score del funding sobre sus 30 observaciones previas;
  - cambio del funding frente a la observación anterior.

  OI y funding se toman "asof" el cierre de la vela (`ts + 4h`). Las filas con
  OI = 0 se descartan (defecto conocido del dataset).

## Target

- **Técnicas y derivados:** `log(close[t+h] / close[t])`.
- **Cross-sectional:** el mismo retorno menos su media entre monedas en la
  fecha `t`.

## Modelo (fijo)

- **Ridge** con α = 1,0 sobre features estandarizadas con los datos de train.
  Sin búsqueda de hiperparámetros.
- En el panel es un modelo **agrupado**: un solo β para todas las monedas.
- **Walk-forward expansivo:** las fechas se parten en 6 tramos iguales en el
  tiempo. En el fold k = 1…5 se entrena con los tramos anteriores (quitando las
  últimas h fechas como purga) y se predice el tramo k. Fuera de muestra quedan
  los tramos 1 a 5.

## Métricas

- **Primaria, IC fuera de muestra:**
  - técnicas y derivados: Spearman entre predicción y target sobre todas las
    observaciones fuera de muestra;
  - cross-sectional: la **media del Spearman por fecha**.
- **R² fuera de muestra:**
  - técnicas y derivados: `1 − Σ(y − ŷ)² / Σ(y − ȳ_train)²` (benchmark: la
    media histórica);
  - cross-sectional: benchmark 0, porque el target ya está demeaned.
- **N y N_eff:** N_eff = N fuera de muestra / h (corrección por solape).
  En el panel se cuentan fechas, no pares fecha-moneda, porque las monedas
  están correlacionadas.

## Null y significancia

- **Rotación circular del target frente a las features.** En el panel se
  rotan fechas enteras, así que la estructura cross-sectional se conserva.
- Se reentrena y re-predice en cada rotación.
- **Desplazamiento mínimo:** `max(10·h, 5% de N)`. Se usan 1.000 rotaciones
  equiespaciadas.
- **p (una cola):** fracción de rotaciones con IC ≥ IC observado.
- **Corrección:** Holm sobre las 10 celdas, a 0,05.

## Criterios

- **"Hay señal":** p de Holm < 0,05 **y** R² fuera de muestra > 0.
- **"Cubre costes":** el margen esperado por operación aproximado supera el
  coste de ida y vuelta de 0,12% (0,05% de comisión + 0,01% de slippage por
  lado).
  - `edge ≈ IC · σ_y · √(2/π)`: retorno esperado por período de una regla
    que opera según el signo de la predicción, bajo normalidad bivariada.
  - σ_y es la std del target fuera de muestra. En el panel es la std
    cross-sectional media por fecha, con edge y coste **por pata**.
  - Supuesto conservador: rotación completa cada h velas.
- **"Vale un proyecto nuevo de ML":** hay señal **y** cubre costes.

## Adenda de implementación (2026-09-23, escrita ANTES de correr)

1. **Rotación en el panel.** Se rota la matriz de targets (fechas × monedas)
   entera a lo largo de las fechas. Los pares cuyo target rotado es NaN (por
   ejemplo, una moneda que aún no existía en la fecha de origen) se descartan
   en esa réplica.
   - El filtro de "≥ 8 monedas" y el demeaning del target se recalculan sobre
     los pares válidos de cada réplica, igual que en el observado.
   - Los tramos del walk-forward se definen sobre las fechas con algún par
     válido.
2. **IC por fecha.** Solo cuentan las fechas con al menos 3 pares válidos
   fuera de muestra.
3. **p con corrección +1:** `p = (#{null ≥ obs} + 1) / (1.000 + 1)`.
4. **Derivados.**
   - Se empieza 8 días después del primer OI, para que exista `oi42`.
   - Se corta en el último funding disponible (2026-03-04), para no arrastrar
     un funding viejo con forward-fill.
   - El z-score del funding usa media y std de las 30 observaciones previas,
     sin incluir la actual.
5. **BTC 1d:** se usa `btcusdt_1d_v15.parquet` tal cual, con el close diario de
   Binance.

6. **Corrección posterior a la primera corrida (bug, no diseño).** En la
   primera corrida las celdas de derivados dieron IC = NaN, con un p espurio
   de 0,001. La causa: con funding constante 30 observaciones, la std es 0 y
   `fund_z` es infinito, y `dropna()` no quita los infinitos.
   - Corrección: los infinitos se tratan como NaN y esas filas se descartan.
   - Se re-corrieron las 10 celdas. Las 8 que no son de derivados no tienen
     infinitos y dan idéntico resultado.

## Riesgos conocidos antes de empezar

- En el panel de 21 monedas la correlación media es alta (~0,69 en el
  proyecto), así que el N efectivo es mucho menor que monedas × fechas.
- El panel tiene sesgo de supervivencia: son monedas que siguen listadas hoy.
- BTC 1h solo cubre desde 2022 con los datos locales.

---

## Resultados

> Corrida el 2026-09-23 · salida completa en `salida.txt`.

| celda | N OOS | N_eff | R² OOS | IC OOS | p | p Holm | σ_y | edge | ¿cubre 0,12%? | ¿hay señal? |
|---|--:|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|
| BTC 1h técnicas h=1 | 30.355 | 30.355 | −0,114% | +0,021 | 0,001 | **0,010** | 0,51% | 0,008% | no | no (R² < 0) |
| BTC 1h técnicas h=24 | 30.336 | 1.264 | −0,881% | +0,006 | 0,413 | 1,000 | 2,46% | 0,012% | no | no |
| BTC 1d técnicas h=1 | 1.925 | 1.925 | −1,976% | +0,022 | 0,083 | 0,663 | 3,10% | 0,053% | no | no |
| BTC 1d técnicas h=5 | 1.921 | 384 | −7,170% | −0,099 | 0,884 | 1,000 | 6,87% | −0,541% | no | no |
| Panel 4h cross-sect h=1 | 216.732 | 11.229 | −0,027% | −0,001 | 0,701 | 1,000 | 1,07% | −0,001% | no | no |
| Panel 4h cross-sect h=6 | 216.644 | 1.870 | −0,275% | +0,026 | 0,277 | 1,000 | 2,70% | 0,057% | no | no |
| Panel 1d cross-sect h=1 | 35.567 | 1.846 | −0,182% | +0,035 | 0,094 | 0,663 | 2,65% | 0,073% | no | no |
| Panel 1d cross-sect h=5 | 35.500 | 368 | −0,766% | +0,069 | 0,016 | 0,144 | 6,07% | **0,334%** | sí | no |
| BTC 4h derivados h=1 | 9.967 | 9.967 | −0,143% | −0,005 | 0,551 | 1,000 | 1,11% | −0,004% | no | no |
| BTC 4h derivados h=6 | 9.963 | 1.660 | −0,335% | +0,003 | 0,302 | 1,000 | 2,75% | 0,007% | no | no |

`edge = IC · σ_y · √(2/π)` por operación (por pata en el panel), frente a un
coste de ida y vuelta de 0,12%.

### Lectura

- **Ninguna de las 10 celdas tiene señal según el criterio pre-registrado.**
  El R² fuera de muestra es **negativo en todas**: un ridge fijo predice peor
  que la media histórica en cualquier fuente y horizonte probados.
- **BTC 1h, siguiente vela.** Es la única celda con IC significativo tras
  Holm (0,021, p Holm = 0,010), con 30.000 observaciones. Pero el edge
  aproximado es 0,008% por operación, **15 veces menor que el coste**. Hay
  una estructura detectable (microestructura / reversión de corto plazo a 1h)
  que no se puede monetizar con costes de taker.
- **Panel 1d cross-sectional a 5 días.** Es la única celda cuyo edge nominal
  (0,33% por pata) supera el coste. No es significativa tras la corrección
  (p Holm = 0,144, N_eff = 368 fechas), su R² es negativo, y el panel tiene
  sesgo de supervivencia y monedas con correlación ~0,69. Sería el único
  hilo que merecería una réplica pre-registrada con otro universo (monedas
  deslistadas incluidas), no un proyecto.
- **Derivados** (cambio de OI, funding y su z-score): nada, a 4h ni a 24h.
  Coincide con `derivados/`.
- **BTC 1d a 5 días:** el IC es negativo (−0,10). El modelo lineal extrapola
  mal entre regímenes.

### Recomendación

**No vale la pena un proyecto nuevo de ML sobre ninguna de estas fuentes con
estos datos.** El patrón es el mismo que en `predictibilidad/`, ahora en 1h,
1d, panel y derivados:

- donde hay significancia, el tamaño del efecto no cubre costes;
- donde el efecto nominal cubriría costes, no es significativo.

Lo único que queda abierto, y con probabilidad baja: replicar el reversal /
momentum cross-sectional diario a 5 días **fuera de esta muestra**, con un
universo sin sesgo de supervivencia. Requiere datos que el proyecto no tiene
(histórico de monedas deslistadas).
