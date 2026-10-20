# E9: meta-labeling + paridad oro como decisor del agresivo, solo BTC

> **Pre-registro escrito el 2026-10-20, ANTES de correr nada.** Viene de
> `docs/PLAN_ESPECIALISTAS_2026-10.md` (E9), con las precisiones de
> implementación. α = 0,05 por etapa.
>
> ⚠️ **Selección declarada:**
> - Las 3 features técnicas se reutilizan de `meta_labeling/`, cuyo AUC 0,61 ya
>   se vio.
> - Las 3 macro entran porque E7 pasó (`especialistas/`), y su diagnóstico ya
>   mostró que solo aporta `oro_z`. **Aun así entran las tres**, como dice el
>   plan; no se elige `oro_z` sola.
> - No se prueba ninguna otra combinación con capacidad de decidir.

## Brazos

| brazo | decisor | features |
|---|---|---|
| **A** | HGB + Platt del motor actual (`src/agresivo_engine.py`, sin cambios) | las 11 de `FEAT` |
| **B** | regresión logística, `C=1,0`, features estandarizadas con el train | `atr_pct`, `bb_width`, `dist_ema_d`, `oro_z_d`, `dxy_20_d`, `spx_dd_d` |
| B0 (descriptivo) | la misma logística | solo las 3 técnicas, para ver qué aporta el macro |
| **C** | V2 (`portfolio_sim`, BTC, costes nuevos) | — |

- **`_d` = feature × dirección** (se invierte en cortos), la misma convención que
  `dist_ema_d` en el motor.
- Las features macro son las de E7 (`experiments/especialistas/`), calculadas
  con el BTC de la serie (real o sintética) y el oro, dólar y S&P reales por
  fecha.
- Para un evento en la vela 4h que **cierra** en `T`, se usa la fila diaria del
  último día UTC completo cuyo cierre es ≤ `T`. La vela de las 20:00 cierra a
  las 24:00 y usa ese mismo día; cualquier otra usa el día anterior.

## Eventos y etiqueta

Los candidatos del agresivo en **BTC 4h** (`comp`, `don55`, `don20`, `mr`, long
y short), con la etiqueta del motor: `ag.etiquetar`, con trailing, costes y
`y = 1` si el PnL es mayor que 0. Pesos por unicidad (`ag.unicidad`). No se
cambia ninguna regla de entrada ni de salida.

## Walk-forward (igual para A, B y B0)

- **Mensual**, como `experiments/agresivo/backtest.py`.
- Para cada mes desde 2021-01, el modelo se entrena con los eventos cuya salida
  es anterior al inicio del mes menos 60 velas de purga, y predice los eventos
  de ese mes.
- `b` = ganancia media en R / pérdida media en R del train (igual que el motor).

## Decisión y cartera (igual para A, B y B0)

- `riesgo = ag.riesgo_de(p̂, b, dd)`: Kelly/2, entre 1% y 10%, con el freno por
  DD actual.
- Simulación con `simular()` de `experiments/agresivo/backtest.py`, con los datos
  limitados a BTC: una posición, fill al open siguiente, `LEV` 5, `MAX_NOT` 2,5.
- Ventana: 2021-01 → fin del parquet (2026-02).

## Etapa 1: ¿B discrimina?

- **Estadístico:** AUC de p̂_B sobre los eventos fuera de muestra.
- **Test:** bootstrap por bloques de 30 días de calendario, 10.000 réplicas.
  `p = P(AUC* − AUC ≥ AUC − 0,5)`, como en `meta_labeling/`.
- **Pasa si p < 0,05.** Si no pasa, se para aquí.
- **Descriptivo:** AUC de A y de B0 sobre los mismos eventos, y AUC por año.

## Etapa 2: ¿B gana más por unidad de riesgo? H9, deben cumplirse las tres

1. **CAGR/DD(B) > CAGR/DD(A)** en la historia.
2. **CAGR/DD(B) > CAGR/DD(C)** en la misma ventana, con V2 a igual exposición
   media que B:
   - exposición = notional / equity por vela, 0 fuera de posición;
   - el riesgo de V2 se escala por `expo_B / expo_V2` y se itera 2 veces.
3. **Null sintético:** 200 series `sintetica()` de BTC con bloques de 180 velas.
   En cada una se corre **todo** B: features, eventos, reentreno mensual y
   cartera. El macro se queda en sus fechas reales, así que el null también
   rompe la relación entre el oro y BTC.
   - Estadístico: CAGR de B.
   - `p = P(CAGR_sint ≥ CAGR_obs)`. **p < 0,05.**

**Descriptivo:** PF, WR, trades/año y resultados por tipo, dirección y año, para
los brazos A, B y B0. También V2 al 4,5% sin escalar.

## Expectativa honesta

- El meta-labeling aprendió "largo por encima de la EMA200 con volatilidad
  baja", que es casi el filtro de V2. B probablemente se parecerá a V2 con más
  trades.
- La paridad con el oro no ayudó en 2021-2022 (E7).
- V2 desde 2021 tiene CAGR/DD bajo (≈0,18 al 4,5% en `agresivo/`), así que el
  criterio 2 no es exigente. **El que pesa es el null sintético.**

---

## Resultados (2026-10-20): `test_meta_agresivo.py`, salida en `salida.txt`

Eventos de BTC etiquetados: **3.362**, de ellos 1.979 de reversión (`mr`).
`y = 1`: 36,8%. Fuera de muestra (2021 → 2026-02): **2.401**.

### Etapa 1: NO pasa → según el pre-registro, el experimento termina aquí

| brazo | AUC | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|--:|--:|--:|--:|--:|--:|--:|
| A (HGB actual) | 0,509 | 0,50 | 0,47 | 0,50 | 0,51 | 0,56 | 0,56 |
| **B (logística + macro)** | **0,475** | 0,48 | 0,40 | 0,50 | 0,42 | 0,51 | 0,56 |
| B0 (logística sin macro) | 0,493 | 0,46 | 0,50 | 0,50 | 0,51 | 0,44 | 0,54 |

B: **p = 0,87** (bloques de 30 días). No discrimina. Queda incluso por debajo
de 0,5, y el macro lo empeora frente a B0.

### Lo demás es solo descriptivo (la etapa 2 no decide porque la 1 no pasó)

| brazo, 2021-01 → 2026-02 | CAGR | DD | CAGR/DD | exposición | trades/año | PF |
|---|--:|--:|--:|--:|--:|--:|
| A (HGB actual, **solo BTC**) | +5,6% | 35,2% | 0,16 | 0,276 | 65 | 1,15 |
| B | −9,2% | 47,6% | −0,19 | 0,032 | 86 | 0,49 |
| B0 | −8,5% | 46,3% | −0,18 | 0,036 | 93 | 0,53 |
| V2 al 0,77% (exposición igual a B) | +1,7% | 8,0% | 0,21 | 0,032 | | |
| V2 al 4,5% | +7,2% | 39,9% | 0,18 | | | |

- El null sintético **no se corrió** (en `salida.txt` hay 2 series de prueba;
  no deciden). La regla de parada lo hace innecesario.
- AUC por tipo, post-hoc (`comp` / `don20` / `don55` / `mr`):
  - A: 0,50 / 0,57 / 0,53 / 0,49;
  - B0: 0,55 / 0,56 / 0,44 / 0,48;
  - B: 0,54 / 0,49 / 0,44 / 0,47.

### Lectura

1. **El meta-labeling no se traslada al agresivo.** En `meta_labeling/` los
   eventos eran solo rupturas largas, y ahí "por encima de la EMA200 con
   volatilidad baja" sí separaba ganadores de perdedores (AUC 0,61). Aquí el 59%
   de los eventos son de reversión (`mr`), además de cortos. Con las mismas
   tres features, el modelo no separa nada (0,49).
2. **La paridad con el oro no sirve como decisor de trades.** Avisaba algo
   sobre caídas del 10% en 10 días (E7). Pero los trades del agresivo duran
   horas o pocos días y salen por trailing, y para eso no aporta: empeora el
   AUC (0,475 contra 0,493).
3. **El HGB actual en BTC solo no es el desastre del agresivo de 9 pares**
   (+5,6%/año, PF 1,15, frente a −12,6%), pero tampoco tiene edge: AUC 0,509.
   Su CAGR/DD (0,16) queda por debajo del de V2 (0,18). El agresivo pierde
   sobre todo por las altcoins y los cortos.
4. **H9: NO pasa.** No se conecta nada al bot (E10 queda descartado).
