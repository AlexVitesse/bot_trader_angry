# E3 — Promediar varios periodos de Donchian

> **Pre-registro escrito el 2026-10-19, ANTES de correr nada.** Viene de
> `docs/PLAN_GITHUB_2026-10.md` (E3), con las precisiones de implementación.
> Familia de 5 tests → **α = 0,01**. Es una variante más de la familia de V2:
> la cuenta de selección pasa de m=36 a **m=37**.
> Fuente: Zarattini, Pagani y Barbon, "Catching Crypto Trends" (SSRN 5209907).
> No hay código público; la adaptación a BTC 4h es nuestra.

## Construcción (única)

**Sub-libros** sobre la misma serie BTC, que pueden estar abiertos a la vez:

| libro | señal | riesgo por trade |
|---|---|--:|
| A20, A55, A120, A240, A480 | señal A de V2 con `a_donchian_n` = 20, 55, 120, 240, 480. **Todo lo demás de A igual**: filtro EMA50/200 diario, ADX, volumen, funding, trailing ATR, `max_bars` | 4,5% / 5 = **0,9%** cada uno |
| F | señal F de V2 sin cambios (A desactivada en ese libro) | **4,5%** |

- Simulador: `PortfolioSim` con `params_por_par`, un "par" por libro,
  `max_concurrent=6`, `max_misma_dir=6`, `max_notional_pct=2,0` por posición,
  apalancamiento 4 (el margen compartido limita), costes nuevos.
- **El funding histórico se aplica a todos los libros.** El simulador lo
  aplicaba solo a la clave `'BTC/USDT'`; se cambia a `startswith('BTC/USDT')`,
  que no altera nada para V2.
- Diferencia inevitable con V2: en V2, F solo entra si no hay posición abierta.
  Aquí F es un libro propio y puede coincidir con los libros A. Se reporta
  también el **control** "V2 en dos libros" (A55 al 4,5% + F al 4,5%), para
  separar el efecto de "varios periodos" del de "libros simultáneos".

## Hipótesis y test

**H3** (deben cumplirse las dos):
1. **Null sintético:** 200 series `sintetica()` con bloques de 180 velas (30
   días), como `bootstrap_bloques/`. En cada una se corre el ensemble completo.
   Estadístico: **CAGR de la cartera**. `p` = fracción de series con CAGR ≥ el
   observado. **p < 0,01.**
2. **CAGR/DD del ensemble ≥ CAGR/DD de V2** (4,5%, `max_notional_pct=2,0`) en
   la historia.

Como referencia se reporta el p de V2 con el mismo estadístico y las mismas
series.

**Expectativa honesta:** el ensemble incluye el 55 y las sub-señales están muy
correlacionadas. Lo más probable es que también capture solo la deriva y que
falle (1). Si es así, se cierra como las otras siete búsquedas de parámetros.

---

## Resultados (2026-10-19) — `test_ensemble.py`, salida en `salida.txt`

| sistema | trades | WR | PF | CAGR | DD | CAGR/DD |
|---|--:|--:|--:|--:|--:|--:|
| V2 (4,5%) | 165 | 44,2% | 1,65 | +23,5% | 42,4% | **0,55** |
| Control: A55 + F en 2 libros | 211 | 42,7% | 1,53 | +23,4% | 47,5% | 0,49 |
| **Ensemble 5×A + F** | 592 | 41,6% | 1,41 | +16,2% | **53,7%** | **0,30** |

Trades por libro: A20 154, A55 116, A120 89, A240 74, A480 64, F 95.

**Null sintético** (200 series, estadístico CAGR):

| | CAGR real | mediana null | p95 null | p |
|---|--:|--:|--:|--:|
| Ensemble | +16,2% | +11,6% | +23,3% | **0,210** |
| V2 (referencia) | +23,5% | +14,3% | +24,7% | 0,090 |

**H3: NO pasa** en ninguna de las dos condiciones.

### Lectura

1. **Promediar periodos no suaviza, empeora.** PF por libro (dentro del
   ensemble): A20 1,35, **A55 1,53**, A120 1,28, A240 1,34, A480 1,10, F 1,13.
   Todos los periodos ganan algo y ninguno compensa a otro.
2. **El DD sube (53,7%)** porque las sub-señales disparan juntas en las mismas
   rupturas: es un solo trade repartido en cinco con costes multiplicados. El
   control muestra que solo dejar a F y A coincidir ya empeora V2 (0,55 → 0,49).
   La regla de V2 de "una posición a la vez" estaba evitando ese riesgo
   apilado.
3. **Que A55 salga el mejor libro no prueba nada.** El 55 se eligió mirando
   esta misma historia, así que in-sample tiene que salir el mejor. Lo que sí
   dice el test es que repartir el riesgo entre periodos no acerca el sistema
   a un edge: el ensemble queda **más cerca del null** que V2 (p 0,21 contra
   0,09).
4. Dato colateral: con el estadístico CAGR, V2 da p = 0,09 contra el null, en
   línea con el 0,13-0,20 de `bootstrap_bloques/` (que usaba la media por
   trade).
