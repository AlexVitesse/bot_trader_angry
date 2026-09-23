# Datos de derivados como filtro de V2: open interest y DVOL

> **Pre-registro escrito el 2026-09-23, ANTES de descargar datos o correr nada.**
> Los resultados se añaden debajo sin tocar esta sección.
>
> Origen: idea 4 de la revisión de GitHub del 2026-09-23. Hay dos hipótesis,
> y solo dos.

## Datos

| serie | fuente | desde | resolución |
|---|---|---|---|
| Open interest BTCUSDT (`sum_open_interest`) | `data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/` | 2020-09-01 | 5 min |
| DVOL BTC | Deribit `public/get_volatility_index_data`, `resolution=1D` (se usa el close diario) | 2021-03-24 | 1 día |

- **Velas:** `data/BTC_USDT_4h_full.parquet`.
- **Trades de V2:** `experiments/portfolio_sim/` (BTC, costes nuevos, riesgo 2%, `max_concurrent=1`).
- **Sin look-ahead:** para una vela o un trade que entra en `t`, solo se usa
  OI observado **hasta el cierre de la vela de señal** y DVOL del **día UTC
  anterior**.

## Definiciones (fijas)

- **`oi_chg`**: `OI(cierre de la vela de señal) / OI(24 h antes) − 1`.
  La hipótesis de trader es que una ruptura con OI subiendo es posicionamiento
  nuevo, no cierre de cortos.
- **`dvol_pos`**: `(DVOL − min90) / (max90 − min90)`, con min y max de los 90
  días previos, incluido el día. Replica la definición de
  `pavel-shkliar/Trading-research`: "5% inferior de su rango de 90 días".
  - **Señal "DVOL bajo"**: `dvol_pos ≤ 0,05`.

## Etapa 1: ¿la señal existe a nivel de mercado? (muchos datos)

Con **dos** tests, Bonferroni da α = 0,025 a cada uno.

**S1-OI.**
- **Eventos:** todas las velas 4h con `close > max(high de las 55 velas previas)`,
  sin ningún otro filtro. Son rupturas Donchian-55 crudas.
- **Resultado:** retorno log de las **12 velas siguientes**, medido desde el
  open de la vela siguiente.
- **Estadístico:** media del retorno en los eventos con `oi_chg > 0` menos la
  media en los eventos con `oi_chg ≤ 0`.
- **Null:** rotación circular de la serie `oi_chg` contra las velas (desplazamiento
  mínimo de 30 días, 10.000 rotaciones). Conserva la autocorrelación de ambas
  series.
- **p (una cola):** fracción de rotaciones con estadístico ≥ observado.

**S1-DVOL.** Es la réplica del hallazgo de `pavel-shkliar`, que usaba un t-test
sobre ventanas solapadas, inválido según `CLAUDE.md`.
- **Eventos:** días con `dvol_pos ≤ 0,05`.
- **Resultado:** retorno log de BTC de los **60 días siguientes**, desde el
  close del día.
- **Estadístico:** media en los días de señal menos media en el resto.
- **Null:** rotación circular de la serie de señal (desplazamiento mínimo de 90
  días, 10.000 rotaciones).
- **p (una cola):** fracción de rotaciones con estadístico ≤ observado. La
  hipótesis es que el retorno es menor.

**Regla de parada:** solo pasa a la etapa 2 la hipótesis con p < 0,025 en la
etapa 1. Si ninguna pasa, el experimento termina.

## Etapa 2: ¿mejora V2? (trades con cobertura)

Filtros sobre los trades de V2:

| hipótesis | se **descartan** los trades con | trades con cobertura |
|---|---|---|
| **H-OI** | `oi_chg ≤ 0` en la vela de señal | entradas desde 2020-09-02 (~128) |
| **H-DVOL** | `dvol_pos ≤ 0,05` el día anterior a la entrada | entradas desde 2021-06-22 (~100) |

- **Test:** la media de `r` de los trades **descartados** es < 0. Bootstrap por
  bloques circulares de 10 trades, 20.000 réplicas, una cola.
  - Un filtro solo aporta si lo que quita pierde dinero.
- **α:** 0,025 si pasan las dos hipótesis a esta etapa, 0,05 si solo pasa una.
- **Adopción:** p < α, además de que la suma de `r` y el PF de los trades
  filtrados sean ≥ que los de V2 completo sobre los mismos trades con cobertura.
- **Descriptivas:** n descartados, WR y PF de cada grupo, y CAGR y DD del V2
  filtrado frente al completo en el tramo con cobertura.

## Riesgos conocidos antes de empezar

- `pavel-shkliar` ya probó el OI y el ratio largo/corto y los **rechazó**.
- Con ~100-128 trades, y posiblemente muy pocos con DVOL bajo, la potencia es
  mínima.
- DVOL solo cubre unos 4 episodios de régimen.

## Adenda al pre-registro (2026-09-23, escrita antes de descargar datos)

Interpretaciones de lo ambiguo, elegidas antes de ver ningún número:

1. **OI en un instante `T`:** último `sum_open_interest` con `create_time ≤ T`
   (asof). La vela de señal cierra en `ts_entrada`, así que
   `oi_chg = OI(ts_entrada) / OI(ts_entrada − 24 h) − 1`. Vela o trade sin OI
   en ambos extremos → sin cobertura, se excluye.
2. **S1-OI:** solo cuentan los eventos con `oi_chg` definido (desde
   2020-09-02). El retorno es `log(close[t+12] / open[t+1])` y se excluyen los
   eventos sin 12 velas futuras.
   - **Rotación:** `oi_chg` se calcula en **todas** las velas 4h con cobertura.
     Esa serie se rota circularmente k velas, con k de 180 (30 días) a N − 180,
     y se toman sus valores en las velas-evento. Se usan 10.000 k
     equiespaciados.
3. **S1-DVOL:**
   - `dvol_pos` usa los 90 valores diarios hasta el día incluido, con ventana
     completa obligatoria.
   - El close diario sale del último close 4h del día UTC.
   - Solo cuentan los días con `dvol_pos` y retorno a 60 días definidos.
   - **Rotación:** la serie booleana de señal se rota k días, con k de 90 a
     N − 90, y se usan 10.000 k equiespaciados.
   - **Rango degenerado:** si `max90 = min90`, `dvol_pos` queda sin definir.
4. **Etapa 2:**
   - H-DVOL usa `dvol_pos` del día UTC anterior al de `ts_entrada`.
   - Un trade "con cobertura" es el que tiene la feature definida.
   - **Bootstrap de los descartados:** se hace sobre los trades descartados en
     orden temporal, con bloques circulares de `min(10, n)`.
   - **p (una cola, H0: media ≥ 0):** fracción de réplicas centradas
     (`media* − media_obs`) que son ≤ `media_obs`.
   - Si no hay ningún trade descartado, la hipótesis se reporta como "sin
     trades que descartar" y no se adopta.
5. **Caché de datos:** va en `data/deriv_oi_btcusdt_5m.parquet` y
   `data/deriv_dvol_btc_1d.parquet`. No se usa `data/derivados/` porque el
   patrón `data/*.parquet` de `.gitignore` no ignora subdirectorios.

---

## Resultados

> Corrida el 2026-09-23 · salida completa en `salida.txt`.
>
> - **Datos:** 636.237 filas de OI a 5 min (2020-09-01 → 2026-09-22) y 2.010
>   días de DVOL (2021-03-24 → 2026-09-23).
> - **Velas 4h hasta 2026-02-27.** El parquet no se refrescó, para que todos
>   los experimentos sigan siendo comparables.

### Corrección de datos detectada al correr (post-hoc, documentada)

El dataset de Binance tiene **473 filas con `sum_open_interest = 0`**. En BTC
eso es imposible: son huecos, no datos. En la primera corrida producían 13
`oi_chg` infinitos y varios de −100%.

- **Corrección:** se descartan las filas con OI ≤ 0 y el asof toma el último
  valor válido.
- **Qué cambia:** la conclusión es **idéntica** en las dos corridas. La primera
  se conserva en `salida_con_oi_cero.txt`: diferencia −1,33 %, p = 0,992.

### Etapa 1: no pasa ninguna hipótesis. Se aplica la regla de parada.

| test | eventos | grupo con señal | resto | diferencia | p (una cola) | ¿pasa a 0,025? |
|---|--:|--:|--:|--:|--:|:--:|
| **S1-OI:** rupturas D55, retorno 12 velas, OI sube vs no sube | 395 (294 / 101) | +0,65% | +2,04% | **−1,39%** | **0,993** | no |
| **S1-DVOL:** días con DVOL en el 5% inferior de su rango, retorno 60 d | 1.653 días, 208 con señal | +1,79% | +2,95% | −1,16% | **0,406** | no |

- **S1-OI:** 10.000 rotaciones, desplazamiento de 30 días a N − 30 días.
- **S1-DVOL:** solo existen **1.474** rotaciones circulares distintas con
  desplazamiento ≥ 90 días en 1.653 días, así que se usaron todas en vez de
  10.000.

**El experimento termina aquí.** La etapa 2 (filtros sobre los trades de V2)
no se corrió, como exige el pre-registro.

### Lectura

- **OI:** la hipótesis de trader ("ruptura con OI subiendo = posicionamiento
  nuevo = mejor") no solo falla, sino que el signo sale **al revés**. Las
  rupturas con OI **bajando o plano** rindieron más a 12 velas (+2,04% frente a
  +0,65%). Coincide con que `pavel-shkliar` también rechazó el OI.
  - Esto **no** es un hallazgo aprovechable. Invertir el filtro después de ver
    el resultado sería una hipótesis post-hoc: con el test a dos colas sería
    p ≈ 0,014 sin corregir, y aquí ya hubo 2 hipótesis. Si alguien quiere
    probar "rupturas con OI cayendo", necesita un pre-registro nuevo y datos
    que no sean estos.
- **DVOL:** la señal de `pavel-shkliar` (DVOL en mínimos → peor retorno a 60
  días) apunta en la dirección que él dice (−1,16 puntos), pero con un null
  que respeta la autocorrelación **no se distingue del azar (p = 0,41)**. Su
  p = 0,0086 venía de un t-test sobre ventanas solapadas de 60 días: el mismo
  error que `CLAUDE.md` documenta en `estacionalidad/` (p = 0,005 → 0,14).
  Es otra instancia del mismo error.

### Veredicto

**RECHAZADO**, en las dos hipótesis:
- ni el open interest de Binance;
- ni el DVOL de Deribit.

Ninguno de los dos aporta una señal a nivel de mercado que justifique filtrar
V2. No se conecta nada al bot.
