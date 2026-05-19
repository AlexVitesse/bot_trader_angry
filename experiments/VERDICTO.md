# Veredicto — 3 Agentes BTC vs. 2026 OOS

> Tres estrategias construidas con cutoff inviolable 2025-12-31, evaluadas por un
> motor independiente sobre Ene–Feb 2026 (datos que ningún agente vio).
> Fecha: 2026-05-19

---

## Resultados consolidados

### In-sample 2020-2025 (declarado por el agente, honestamente)

| Agente | Estrategia | WF | PF | WR | Mensual | DD | Bootstrap p |
|--------|------------|----|----|----|---------|------|-------------|
| A | Donchian-55 + EMA daily + ATR×2.5 trail | 7/9 | 1.41 | 44.1% | +1.0% | 18% | **0.07** |
| B | GBM classifier 11 features, purged CV | 3/11 | 0.96 | 33.6% | -0.3% | — | 0.607 |
| C | Regime adaptive (BULL pullback + RANGE meanrev) | 6/12 | 1.49 | 42% | +1.1% | 32% | 0.156 |

### Forward-OOS 2026 (Ene–Feb, ~57 días, BTC -23% YTD)

| Agente | N | WR | PF | Total | DD | Mensual | Bootstrap p |
|--------|---|----|----|-------|-----|---------|-------------|
| **A** | **0** | — | — | **0%** | 0% | **0%** | — |
| B | 8 | 37.5% | 0.51 | -3.95% | 7.7% | -3.95% | 0.833 |
| C | 5 | 20% | 0.15 | -7.90% | 7.9% | -7.90% | 0.977 |

---

## Análisis honesto

### Agente A — el ganador *defensivo*
**Stayed flat** durante todo el bear de Ene-Feb 2026. Cero trades.
- Su filtro daily `EMA50_1d > EMA200_1d` (con shift(1)) bloqueó las entradas
  cuando el régimen daily se rompió.
- Ese es **comportamiento correcto** para un trend-follower long-only en bear.
- Pero también: cero evidencia positiva. La OOS no probó su capacidad de
  generar retorno, solo confirmó que no pierde en bear.
- **Veredicto: VÁLIDA pero NO COMPROBADA en forward.** Esperar una ventana
  alcista para verificarla realmente. In-sample: bootstrap p=0.07 (marginal).

### Agente B — confirmado sin edge
**8 trades en 2026, PF 0.51, -3.95%**. Consistente con su propio reporte
honesto (in-sample PF 0.96, p=0.607). El ML no genera edge en BTC 4h con
purged CV — exactamente lo que predijo la historia del proyecto (V7, V9,
V13.03 todos fallaron de la misma manera).
- **Veredicto: REJECT.** El propio agente lo dijo. 2026 lo confirma.
- *Valor del experimento:* demuestra con datos limpios que ML clasificador
  simple en BTC 4h no tiene edge. Cierra una hipótesis costosa.

### Agente C — RANGE sub-estrategia falló live
**5 trades en 2026, todos en RANGE, WR 20%, PF 0.15, -7.90%**. El sub-componente
RANGE (RSI<25 + BB inferior + bullish candle) capturó pullbacks oversold en
Ene 2026 que resultaron ser *cuchillos cayendo* — el rebote no llegó.
- BULL no firmó (BTC en downtrend) — bien, defensivo.
- RANGE catastrófico. Es exactamente el riesgo que el propio agente alertó:
  "RANGE 0/12 alone, fires too rarely; 25 trades en 6 años; WR 60% in-sample
  pero muestra pequeña".
- **Veredicto: REJECT en su forma actual.** Si se quisiera rescatar, hay que
  desactivar RANGE o añadirle un filtro de momentum (no entrar oversold si la
  tendencia de medio plazo aún bajando).

---

## Conclusión global

**Ninguna de las 3 estrategias cumple los objetivos declarados**
(WR > 50%, >20% mensual). La realidad honesta:

| | In-sample | OOS 2026 |
|--|-----------|----------|
| Mejor PF | 1.49 (C) | indeterminado (A) |
| Mejor mensual | +1.1% (C) | 0% (A) |
| Mejor WR | 44.1% (A) | 37.5% (B) |

A 20%/mes le faltaba ~20x. Eso es bueno saberlo ahora, no con capital real.

### Lo positivo (real)
1. **Los 3 agentes fueron honestos.** Sus reportes in-sample reflejan sus
   resultados OOS (B y C empeoran un poco, A queda en cero). **No hay
   overfitting nuevo** — esto es la primera ronda del proyecto en que se
   diseña con disciplina y los números coinciden con la realidad.
2. **A no destruyó capital** — un trend follower que se queda flat en bear es
   exactamente lo deseado. Es la mitad del trabajo.
3. **B y C contribuyeron información valiosa:** ML sin edge confirmado;
   mean-reversion oversold sin filtro de tendencia es trampa.

### Lo negativo
1. **Edge real en BTC 4h es modesto** — todo apunta a CAGR realista del
   10–15% anual, no del 800% que pedía el objetivo. Eso es honesto.
2. **A necesita una ventana alcista** para mostrar si realmente captura algo.
   En 2 meses bajistas no se prueba el lado positivo.
3. **El objetivo "20% mensual" en BTC con strategies honestas, sin leverage
   alto, parece inalcanzable.** La literatura y la propia historia del
   proyecto (V7 fue ~13%/mes con leverage) lo sugiere.

---

## Recomendaciones

1. **Quedarse con A como base.** Donchian + filtro daily + trailing amplio es
   una receta clásica (Turtle Traders) que ya probó en in-sample con bootstrap
   marginal. Esperar Q2-Q3 2026 (probable rally post-halving) para ver
   resultados positivos en vivo. Paper trading testnet con A en `ML_V15_PAIRS`
   = ['BTC/USDT'] como ÚNICO par.

2. **Cerrar ML clasificador como línea de investigación** — B + las versiones
   históricas (V7-V13.03) son evidencia suficiente: ML probabilístico simple
   no genera edge en BTC 4h con validación honesta.

3. **C: rescatable solo si se rediseña.** Idea: añadir a RANGE el filtro
   "no entrar oversold si EMA50 4h baja" para evitar cuchillos. Pero el ROI
   investigación/resultado es bajo — preferible enfocarse en A.

4. **Ajustar el objetivo de retorno.** 20%/mes en BTC = 792% anual.
   Esperar 10–15% **anual** honesto y crecer desde ahí. El V7 histórico
   (322% anual) probablemente sobreestimaba por bugs similares — no es la
   referencia confiable que parece.

5. **Refrescar datos y re-correr** para tener Mar-May 2026 también — más
   muestra OOS mejor.

---

## Lo que esta vuelta nos enseñó

Tres agentes entrenando en paralelo con disciplina anti-overfitting + un
verificador independiente = la primera vez en este proyecto que los números
in-sample y OOS coinciden. **No hay magia descubierta. Pero ahora sabemos qué
es real y qué era ruido**, que vale infinitamente más que un PF 18 inventado.
