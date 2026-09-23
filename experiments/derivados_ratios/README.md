# Ratio taker y premium index como filtro de V2

> **Pre-registro escrito el 2026-09-23, ANTES de descargar el premium index o
> correr nada.** Los resultados se añaden debajo sin tocar esta sección.
>
> Origen: `docs/FUENTES_DE_DATOS_2026-09.md` §5-6. Son las dos únicas fuentes
> gratuitas de derivados con historia desde 2020 que no se han medido. Hay dos
> hipótesis, y solo dos. Los ratios largo/corto (cuentas y top traders) del
> mismo fichero **no se prueban**: ya los rechazó `pavel-shkliar` y añadirlos
> solo bajaría el α.

## Datos

| serie | fuente | desde | resolución |
|---|---|---|---|
| `sum_taker_long_short_vol_ratio` BTCUSDT | `data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/` (mismos zips que `derivados/`) | 2020-09-01 | 5 min |
| Premium index BTCUSDT | `data.binance.vision/data/futures/um/monthly/premiumIndexKlines/BTCUSDT/4h/` | 2020-01 | 4h (open, high, low, close del premium) |

- **Velas:** `data/BTC_USDT_4h_full.parquet` (hasta 2026-02-27, sin refrescar,
  para que siga siendo comparable con `derivados/`).
- **Trades de V2:** `experiments/portfolio_sim/` (BTC, costes nuevos, riesgo
  2%, `max_concurrent=1`).
- **Caché:** `data/deriv_metrics_btcusdt_5m.parquet` (esta vez con **todas**
  las columnas del fichero `metrics`) y `data/deriv_premium_btcusdt_4h.parquet`.
- **Sin look-ahead:** para una vela o trade cuya señal cierra en `t`, solo se
  usan observaciones con timestamp ≤ `t`.

## Definiciones (fijas)

- **`taker`**: media del `sum_taker_long_short_vol_ratio` de las 48 filas de
  5 min cuyo `create_time` cae dentro de la vela de señal `(t − 4h, t]`. Si
  hay menos de 24 filas válidas (> 0), la vela queda **sin cobertura**.
  - **`taker_z`**: z-score de `taker` sobre las **180 velas previas** (30 días),
    sin incluir la actual, ventana completa obligatoria.
  - Hipótesis de trader: una ruptura con compradores agresivos por encima de
    lo normal es demanda real, no un barrido de stops.
- **`prem`**: `close` del premium index de la vela 4h que cierra en `t` (es
  `(mark − index) / index`, adimensional).
  - **`prem_z`**: z-score sobre las 180 velas previas, igual que arriba.
  - Hipótesis de trader: una ruptura con el perpetuo muy por encima del
    índice es posicionamiento largo apalancado ya metido; rinde menos. Es la
    misma lógica que el veto de funding, que en `funding_veto/` solo disparó
    6 veces en 6,5 años; el premium a 4h debería disparar más.

## Etapa 1: ¿la señal existe a nivel de mercado? (muchos datos)

Con **dos** tests, Bonferroni da α = 0,025 a cada uno.

**Eventos (comunes a los dos):** todas las velas 4h con
`close > max(high de las 55 velas previas)`, sin ningún otro filtro. Son
rupturas Donchian-55 crudas, igual que en `derivados/` S1-OI. Solo cuentan los
eventos con la feature definida y con 12 velas futuras.

**Resultado:** `log(close[t+12] / open[t+1])`.

**S1-TAKER.**
- **Estadístico:** media del resultado en los eventos con `taker_z > 0` menos
  la media en los eventos con `taker_z ≤ 0`.
- **Null:** la serie `taker_z` calculada en **todas** las velas con cobertura
  se rota circularmente k velas, con k de 180 (30 días) a N − 180, 10.000 k
  equiespaciados, y se toman sus valores en las velas-evento.
- **p (una cola):** fracción de rotaciones con estadístico ≥ observado.

**S1-PREM.**
- **Estadístico:** media del resultado en los eventos con `prem_z > 1` menos
  la media en el resto.
- **Null:** misma rotación circular sobre `prem_z`.
- **p (una cola):** fracción de rotaciones con estadístico ≤ observado. La
  hipótesis es que el retorno es **menor**.

**Regla de parada:** solo pasa a la etapa 2 la hipótesis con p < 0,025. Si
ninguna pasa, el experimento termina y se documenta el negativo.

## Etapa 2: ¿mejora V2? (trades con cobertura)

Filtros sobre los trades de V2:

| hipótesis | se **descartan** los trades con | cobertura |
|---|---|---|
| **H-TAKER** | `taker_z ≤ 0` en la vela de señal | entradas desde 2020-10-01 (~125) |
| **H-PREM** | `prem_z > 1` en la vela de señal | entradas desde 2020-02-01 (~150) |

- **Test:** la media de `r` de los trades **descartados** es < 0. Bootstrap
  por bloques circulares de `min(10, n)` trades en orden temporal, 20.000
  réplicas, una cola (H0: media ≥ 0; p = fracción de réplicas centradas
  ≤ `media_obs`).
  - Un filtro solo aporta si lo que quita pierde dinero.
- **α:** 0,025 si llegan las dos hipótesis, 0,05 si llega una.
- **Adopción:** p < α, y además la suma de `r` y el PF de los trades
  filtrados ≥ los de V2 completo sobre los mismos trades con cobertura.
- **Descriptivas:** n descartados, WR y PF de cada grupo, CAGR y DD del V2
  filtrado frente al completo en el tramo con cobertura.
- Si no hay trades que descartar, se reporta "sin trades que descartar" y no
  se adopta.

## Riesgos conocidos antes de empezar

- `pavel-shkliar/Trading-research` rechazó el OI y los ratios largo/corto de
  este mismo fichero. El ratio taker es la columna que no probó, pero es de la
  misma familia.
- El premium index es el insumo del funding, y el funding ya se midió dos
  veces sin resultado. Lo único nuevo es la resolución 4h en vez de 8h y el
  umbral por z-score.
- Con ~125-150 trades en la etapa 2, la potencia es mínima. Por eso la etapa
  1 es la que decide.
- Es la **tercera** pareja de hipótesis de derivados del mes (OI, DVOL, y
  ahora estas dos). Si se cuentan las cuatro juntas, el α efectivo es 0,0125.
  Si algo sale con p entre 0,0125 y 0,025 se reporta como "no sobrevive a la
  corrección global".

## Ejecución

- Script: `test_derivados_ratios.py`, copiado de `derivados/test_derivados.py`
  y con las dos features nuevas. Reutiliza `cargar_oi` para bajar los
  `metrics` (ya en caché los zips no: se rebajan, ~2.200 ficheros, ~10 min).
- Tiempo estimado: medio día de trabajo, una corrida.
- Salida en `salida.txt`, resultados en la sección de abajo.

---

## Resultados

> Corrida el 2026-09-23 · salida completa en `salida.txt`.
>
> - **Datos:** 636.710 filas de `metrics` a 5 min (2020-09-01 → 2026-09-22)
>   y 14.562 velas 4h del premium index (2020-01 → 2026-08).
> - **Velas 4h hasta 2026-02-27**, sin refrescar, igual que `derivados/`.
> - **Notas de implementación (sin cambios de diseño):**
>   - El ratio taker se asigna a la vela por `ceil('4h')` del `create_time`:
>     una fila con `create_time` justo en `t` es de la vela que cierra en `t`.
>   - Eventos y retorno a 12 velas se calculan sobre la serie completa, y
>     luego se restringen a las velas con z definido. En `derivados/` se
>     filtraba antes: con huecos raros el resultado es el mismo.
>   - S1-TAKER usa 9.976 rotaciones: son las distintas que salen de 10.000 k
>     equiespaciados.

### Etapa 1: no pasa ninguna hipótesis. Se aplica la regla de parada.

| test | eventos (señal / resto) | señal | resto | diferencia | p (una cola) | ¿pasa a 0,025? |
|---|--:|--:|--:|--:|--:|:--:|
| **S1-TAKER:** `taker_z > 0` vs `≤ 0` | 351 (158 / 193) | +1,24% | +0,97% | +0,27% | **0,310** | no |
| **S1-PREM:** `prem_z > 1` vs resto | 405 (154 / 251) | +0,77% | +0,98% | −0,22% | **0,426** | no |

Las dos diferencias van en la dirección que predice la hipótesis de trader.
Pero son de un cuarto de punto sobre retornos a 12 velas con desviación de
varios puntos, y la rotación circular las deja en p = 0,31 y 0,43. El azar
produce diferencias así una de cada tres o cuatro veces.

**El experimento termina aquí.** La etapa 2 no se corrió, como exige el
pre-registro.

### Veredicto

**RECHAZADO**, en las dos hipótesis:
- ni el ratio taker compra/venta,
- ni el premium index a 4h.

Con esto las **cuatro** hipótesis de derivados del mes (OI, DVOL, taker,
premium) quedan rechazadas en la etapa 1, a nivel de mercado y con miles de
velas. Siguiendo `docs/FUENTES_DE_DATOS_2026-09.md` §6.3, **no se pagan
datos de la misma familia** (liquidaciones históricas de Coinglass/Tardis).
Lo que queda es medir la ejecución con la grabación en vivo
(`docs/GRABACION_DATOS_VIVO.md`).
