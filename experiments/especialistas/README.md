# E7 y E8: especialistas macro y de volumen como alerta de caídas de BTC

> **Pre-registro escrito el 2026-10-20, ANTES de descargar datos o correr nada.**
> Viene de `docs/PLAN_ESPECIALISTAS_2026-10.md`, con las precisiones de
> implementación. **Familia de 2 tests → α = 0,025 cada uno.** Una sola
> configuración por especialista: si falla, se acepta el no.

## Pregunta

¿El macro (oro, dólar, bolsa) o el volumen avisan de una caída de BTC **más
allá de lo que ya avisa BTC por sí mismo** (su volatilidad, tendencia y
momentum)? Las caídas vienen tras volatilidad alta, así que cualquier cosa
correlacionada con la volatilidad parece predecirlas sin aportar nada. Por eso
el test es **incremental**, sobre una base.

## Datos

- **BTC:** `data/BTC_USDT_4h_full.parquet`. El diario se construye agregando
  las velas 4h por día UTC: `open` la primera, `high` el máximo, `low` el
  mínimo, `close` la última, `volume` la suma.
- **Macro:** yfinance, diario desde 2018-01-01, columna `Close`:
  - oro `GC=F`;
  - dólar `DX-Y.NYB`;
  - S&P 500 `^GSPC`.
  Se guarda en `data/` la primera vez y no se vuelve a descargar.

## Etiqueta (común)

Para el día `d` (cierre UTC):

`y_d = 1` si `min(low_{d+1}, …, low_{d+10}) ≤ 0,90 · close_d`

Es decir, **BTC cae al menos un 10% en algún momento de los próximos 10 días**.
Los últimos 10 días sin horizonte completo se descartan. Se informa la tasa
base.

## Features (todas conocidas al cierre del día `d`)

**Base (BTC sobre sí mismo):**
- `vol20`: desviación estándar de los retornos log diarios de los últimos 20 días.
- `dist_ema200`: `close_d / EMA200_d − 1`, EMA del cierre diario con `adjust=False`.
- `ret20`: `close_d / close_{d−20} − 1`.

**E7, macro.** Cada serie se calcula en su propio calendario de días hábiles.
Luego se lleva al calendario diario de BTC con el **valor del último día hábil
estrictamente anterior a `d`** (desplazado 1 día y rellenado hacia delante).
- `oro_z`: z-score de 60 días naturales de `log(close_BTC / oro)`, con el oro
  ya alineado.
- `dxy_20`: retorno de 20 días hábiles del dólar.
- `spx_dd`: `SPX / max(SPX de los últimos 60 días hábiles) − 1`.

**E8, volumen.** Se calcula sobre las velas 4h y se toma el valor de la última
vela del día (la de las 20:00 UTC, que cierra a las 24:00).
- `vol_z`: `lv = log(volume)` por vela;
  `(media de lv en las últimas 6 velas − media de lv en las 120 anteriores) / desviación de esas 120`.
- `vol_sube`: volumen de las velas con `close > open` dividido entre el
  volumen total, en las últimas 60 velas.
- `pct_sube`: fracción de las últimas 60 velas con `close > open`.

Se descartan los días con alguna feature sin valor. La EMA200 necesita 200
días de calentamiento: se usa desde el día 200 de la serie.

## Modelo y walk-forward

- Regresión logística de sklearn con `C=1,0`. Features estandarizadas con la
  media y la desviación del train de cada año.
- Para cada año de test Y ∈ {2021, …, 2026}:
  - **train:** días `d` con `d + 10 días < 1-ene-Y`. La purga es igual al
    horizonte de la etiqueta.
  - **test:** días del año Y.
- Las predicciones fuera de muestra de todos los años se juntan.
- Modelos:
  - **base**;
  - **base+macro** (E7);
  - **base+vol** (E8);
  - **solo macro** y **solo vol** (estos dos son descriptivos).

## Test

**H7:** `ΔAUC₇ = AUC(base+macro) − AUC(base) > 0`.
**H8:** `ΔAUC₈ = AUC(base+vol) − AUC(base) > 0`.

- **p:** bootstrap pareado por bloques de 30 días de calendario sobre los días
  fuera de muestra, 10.000 réplicas. Los bloques se cuentan desde el primer día
  de test y se remuestrean con reemplazo hasta el mismo número de bloques.
  `p = P(ΔAUC* ≤ 0)`.
- **Pasa si p < 0,025.**

**Descriptivo (no decide):**
- AUC por año de cada modelo y coeficientes medios.
- **Null por rotación:** 1.000 desplazamientos circulares (≥ 90 días) del
  bloque de features del especialista contra el resto. En cada uno se repite el
  walk-forward completo. Se reporta `p_rot = P(ΔAUC_rot ≥ ΔAUC_obs)`.

## Si pasa

La feature entra en E9, el decisor de meta-labeling del agresivo. Si no pasa,
no entra, sin variantes.

## Riesgos conocidos antes de empezar

- Las caídas de ≥10% en 10 días vienen en racimos (2021, 2022). El N efectivo
  es mucho menor que el número de días.
- La correlación entre cripto y bolsa cambia de régimen (casi nula antes de
  2020, alta en 2022). Un efecto que solo exista en 2022 puede dar un AUC
  global decente y no servir después. Por eso se informa el AUC por año.

---

## Resultados (2026-10-20): `test_especialistas.py`, salida en `salida.txt`

Hay 2.405 días con features (2019-07-20 → 2026-02-17, fin del parquet) y 1.874
fuera de muestra (2021 → 2026-02). **Tasa base de caída ≥10% en 10 días: 24%.**
Varía mucho por año: 42% en 2021, 8% en 2023.

| modelo | AUC fuera de muestra | 2021 | 2022 | 2023 | 2024 | 2025 | 2026* |
|---|--:|--:|--:|--:|--:|--:|--:|
| base (BTC solo) | 0,475 | 0,57 | 0,50 | 0,48 | 0,52 | 0,58 | 0,63 |
| **base + macro** | **0,568** | 0,54 | 0,51 | 0,60 | 0,59 | 0,65 | 0,88 |
| base + volumen | 0,526 | 0,52 | 0,54 | 0,39 | 0,63 | 0,48 | 0,54 |
| solo macro | 0,453 | | | | | | |
| solo volumen | 0,491 | | | | | | |

\* 2026 son solo 48 días.

| | ΔAUC frente a base | IC90 bootstrap | p | rotación p_rot | veredicto |
|---|--:|---|--:|--:|---|
| **H7 macro** | +0,094 | [+0,020, +0,170] | **0,016** | 0,025 | **PASA** (p < 0,025) |
| H8 volumen | +0,051 | [−0,011, +0,111] | 0,089 | 0,058 | **NO pasa** |

### Diagnóstico post-hoc (`diag_e7.py`, descriptivo, no cambia el veredicto)

| | AUC |
|---|--:|
| base + `oro_z` sola | **0,588** (coef `oro_z` −0,52) |
| base + `dxy_20` sola | 0,479 |
| base + `spx_dd` sola | 0,441 |

AUC absoluto de base+macro: 0,568, con IC90 [0,496; 0,638] y p(AUC ≤ 0,5) = 0,056.

### Lectura

1. **E7 pasa según el pre-registro**, y la rotación lo confirma (p_rot = 0,025).
   Según la regla, las tres features macro entran en E9.
2. **Todo el efecto viene de la paridad con el oro.** El dólar y el S&P no
   aportan nada (incluso empeoran). Con un coeficiente negativo en `oro_z` y
   positivo en `ret20`, la señal es: **si BTC sube pero el oro sube más que BTC,
   aumenta el riesgo de caída.** Se parece a una alerta de "huida a refugio".
3. **Cautelas, y son serias:**
   - **No ayuda en los años de grandes caídas.** En 2021 baja frente a la base
     (0,54 contra 0,57) y en 2022 empata (0,51). La mejora está en 2023-2026,
     que coincide con el rally histórico del oro (`multiactivo/` ya lo marcó
     como excepcional).
   - **Parte de la mejora es corregir una base mala.** La base sale en 0,475 en
     el conjunto, aunque supera 0,5 en casi todos los años por separado; los
     cambios de nivel entre años penalizan el AUC conjunto. Por sí solo, el
     modelo con macro supera al azar solo al límite (p = 0,056).
   - `oro_z` correlaciona 0,77 con `ret20`. El modelo usa la diferencia entre
     las dos, y esa combinación es frágil.
4. **E8 (volumen) no pasa.** No entra en E9. `vol_sube` tiene un coeficiente
   grande (−0,7), pero no se distingue del ruido.
5. **Qué significa para E9:** la paridad con el oro entra como candidata, pero
   E9 la tiene que justificar con su propio test, sobre los trades del agresivo
   y contra el null sintético. Pasar E7 **no** significa que mejore el dinero.
