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

---

## Resultados

*(se completan después de correr `test_derivados.py`)*
