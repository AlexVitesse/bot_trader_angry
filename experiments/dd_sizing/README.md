# E4 — Bajar el riesgo según la caída del capital

> **Pre-registro escrito el 2026-10-19, ANTES de correr nada.** Viene de
> `docs/PLAN_GITHUB_2026-10.md` (E4), con las precisiones de implementación.
> Familia de 5 tests del plan → **α = 0,01**. Los trades son los mismos que en
> V2: solo cambia el tamaño.

## Regla (única, no se tunea)

`riesgo_t = 4,5% × max(0,25, 1 − DD_t / 45%)`

- `DD_t` = caída de la equity al abrir el trade respecto al máximo de las
  equities de apertura anteriores. Con una posición a la vez, eso es el
  máximo de la equity realizada: lo que el bot en vivo puede calcular.
- Se aplica con el hook `sizing` de `PortfolioSim`. El tope
  `max_notional_pct=2,0` sigue igual.

## Comparación justa

**Fijo-m**: V2 a riesgo fijo igual al **riesgo medio por trade que usó E4** en
la historia. También se reporta V2 al 4,5% como referencia.

## Implementación del remuestreo

El PnL de cada trade es lineal en el notional (comisión, slippage y funding son
proporcionales). Por eso, de una corrida se guarda `g = pnl/notional` y `trail`
de cada trade, y cualquier regla de sizing se reproduce trade a trade:
`notional = min(E·riesgo/trail, 2·E)`, `E += g·notional`. El DD de estos
caminos se mide sobre la equity al cierre de cada trade, igual para los dos
brazos. Es menor que el DD marcado a mercado, y por eso solo se usa para
comparar.

## Métricas e hipótesis

1. **Historia (simulador real):** CAGR, DD y CAGR/DD de E4, Fijo-m y Fijo 4,5%.
2. **H4 (test):** CAGR/DD de E4 > Fijo-m. Bootstrap pareado por bloques
   circulares de 10 trades sobre la secuencia histórica (165 trades, mismos
   índices para los dos brazos), 20.000 réplicas, horizonte = la historia.
   `p = P(Δ ≤ 0)`, con `Δ = CAGR/DD(E4) − CAGR/DD(Fijo-m)`. **Pasa si p < 0,01.**
3. **P(kill en 3 años)** (descriptivo): igual que `riesgo/`, 70 trades, bloques
   de 10, 20.000 caminos, kill al 45%.
4. **Null sintético** (descriptivo): 200 series con `sintetica()` (bloques de
   180 velas = 30 días, como `bootstrap_bloques/`). V2 corre en cada una y se
   aplican las dos reglas. Se reporta P(kill en la serie completa) y la mediana
   de Δ. Esto responde a la advertencia de `riesgo/`: el remuestreo de trades
   rompe las rachas de régimen.

**Expectativa honesta:** solo ayuda si las pérdidas vienen en rachas. Si no, la
regla se queda pequeña justo antes de la recuperación. Esto encaja con la
lectura de E1: lo que aporta V2 se concentra en pocos periodos alcistas.

---

## Resultados (2026-10-19) — `test_dd_sizing.py`, salida en `salida.txt`

La reconstrucción trade a trade coincide con el simulador (múltiplo final
3,4159 en los dos).

| brazo | riesgo medio | CAGR | DD | CAGR/DD |
|---|--:|--:|--:|--:|
| E4 | 3,42% (mín. 1,59%) | +18,8% | 32,6% | 0,58 |
| Fijo-m | 3,42% | +18,5% | 34,6% | 0,53 |
| Fijo 4,5% | 4,50% | +23,5% | 42,4% | 0,55 |

**H4: NO pasa.** Δ CAGR/DD mediano +0,014, IC90 [−0,118, +0,212], **p = 0,44**.

| | P(kill) | DD p95 | CAGR mediano | P(perder) |
|---|--:|--:|--:|--:|
| MC 3 años, E4 | 0,0% | 30,2% | +18,4% | **10,9%** |
| MC 3 años, Fijo-m | 0,5% | 34,7% | +18,2% | 8,2% |
| MC 3 años, Fijo 4,5% | 2,9% | 42,0% | +23,1% | 8,8% |
| Null sintético (7 años), E4 | 0,0% | | | |
| Null sintético (7 años), Fijo-m | 1,5% | | | |
| Null sintético (7 años), **Fijo 4,5%** | **7,5%** | | | |

En el null sintético, E4 tiene **peor** CAGR/DD que Fijo-m en el **80,5%** de
las series (mediana Δ −0,057).

### Lectura

1. **Rechazado como mejora.** La ventaja histórica (0,58 contra 0,53) es
   suerte de esa secuencia concreta. Con otro orden de los mismos trades
   desaparece, y en series con otro orden de regímenes casi siempre resta.
   Ocurre lo que anticipaba el pre-registro: la regla reduce el tamaño justo
   antes de la recuperación. Por eso P(perder en 3 años) sube del 8,2% al
   10,9%.
2. **Que P(kill) baje a 0 es mecánico, no una mejora.** Con DD del 34% el
   riesgo ya está en el piso (1,1%), así que llegar al 45% es casi imposible.
   No evita perder dinero: lo reparte en más tiempo.
3. **Dato nuevo para la decisión de riesgo:** con series que conservan las
   rachas de régimen, el **4,5% fijo toca el kill switch en el 7,5% de las
   historias de 7 años**. `riesgo/` daba un 3% a 3 años con remuestreo de
   trades, y ya avisaba de que subestimaba. A 3,42% fijo la cifra baja al 1,5%.
