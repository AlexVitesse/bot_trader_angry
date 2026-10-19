# E6 — Mezcla V2 + BTC spot

> **Pre-registro escrito el 2026-10-19, ANTES de correr nada.** Test nuevo,
> fuera de la familia de 5 del plan: **α = 0,01** por separado.
>
> ⚠️ **Selección declarada.** Esta mezcla ya se vio, como dato descriptivo, en
> E1 (`benchmark_bh/`: CAGR/DD 0,72 contra 0,58 de V2). Se testea **exactamente
> esa configuración** (m=1), sin elegir pesos ni exposición de nuevo. Aun así,
> la historia ya no es una prueba limpia. El peso lo llevan el null sintético y
> el estrés de deriva.

## Definición (única)

- **50% del capital en V2**: `PortfolioSim`, BTC, 4,5%, `max_notional_pct=2,0`,
  costes nuevos.
- **50% del capital en BTC spot con exposición constante 0,176** (la
  exposición media de V2 medida en E1). El resto de esa mitad queda en efectivo,
  sin rendimiento. Rebalanceo diario de la exposición con coste 0,06% sobre el
  volumen, sin funding.
- **Rebalanceo entre mitades** el primer día de cada mes, sin coste (es
  efectivo interno).
- En total: unos 0,5 × V2 + 8,8% de BTC + efectivo.

La única diferencia con la Mixta de E1 es que la pata BTC es **spot** y no
perpetuo. Ya en E1 se dijo que lo correcto para "mantener" es spot. La
exposición 0,176 queda **fija** en todas las series sintéticas (no se
recalcula).

## Hipótesis

**H6:** CAGR/DD de la mezcla > CAGR/DD de V2 solo (al 4,5%). Según
`riesgo/`, el CAGR/DD de V2 casi no depende del riesgo, así que comparar con
V2 al 4,5% no favorece a la mezcla por tener menos exposición.

**Pasa si se cumplen (a) y (b):**

- **(a) Historia:** bootstrap pareado por bloques circulares de calendario de
  30 días sobre los retornos diarios conjuntos (mezcla, V2), 20.000 réplicas.
  `p = P(Δ ≤ 0)`, con `Δ = CAGR/DD(mezcla) − CAGR/DD(V2)`. **p < 0,01.**
- **(b) Null sintético con la deriva real:** 200 series `sintetica()` con
  bloques de 180 velas. En cada una se corren V2 y la mezcla. **Δ > 0 en al
  menos el 90% de las series.**

  Aquí no se busca un p frente al null. El beneficio de mezclar es
  diversificación (V2 está fuera el 87% del tiempo), no timing, así que debería
  aparecer también en series sin timing. La pregunta es si es **robusto** a otro
  orden de regímenes.

**Estrés de deriva (descriptivo, no decide):** las mismas 200 series con la
deriva log por vela recortada a la **mitad** y a **cero**: se multiplica toda la
serie OHLC por `exp(−f·μ·t)`, con μ la deriva log media por vela y f = 0,5 o 1.
Se reporta la fracción de series con Δ > 0 y las medianas de CAGR/DD. Responde
a la duda principal de E1: si la ventaja depende de que BTC siga subiendo al
50% anual.

---

## Resultados (2026-10-19) — `test_mixta.py`, salida en `salida.txt`

| (historia, diario) | CAGR | DD | CAGR/DD |
|---|--:|--:|--:|
| V2 (4,5%) | +23,5% | 40,6% | 0,58 |
| **Mezcla** (pata BTC spot) | **+17,7%** | **20,9%** | **0,85** |
| Solo BTC spot al 0,176 | +10,5% | 20,0% | 0,53 |

(La mezcla de E1 daba 0,72 porque su pata BTC era perpetuo y pagaba funding.)

**(a) Historia: NO cumple.** Δ observado +0,27. En el bootstrap pareado la
mediana es +0,16, con IC90 [−0,18, +0,57] y **p = 0,18**.

**(b) Null sintético: cumple.**

| deriva | Δ > 0 | mediana mezcla | mediana V2 | mediana BTC 0,176 |
|---|--:|--:|--:|--:|
| real | **99,0%** | 0,74 | 0,46 | 0,64 |
| mitad (estrés) | 94,0% | 0,38 | 0,22 | 0,34 |
| cero (estrés) | 85,5% | 0,15 | 0,09 | 0,12 |

**H6: NO pasa**, porque falla (a). Se respeta el pre-registro: la mezcla
**no queda validada**.

### Lectura

1. **Las dos pruebas dicen cosas distintas, y las dos son ciertas.**
   - La historia real tiene pocos episodios de régimen (11). Remuestreada por
     bloques de 30 días, la ventaja de la mezcla en ese único camino es
     demasiado ruidosa para p < 0,01.
   - Las 200 historias sintéticas dicen que la ventaja es **estructural**:
     aparece en el 99% de los órdenes de régimen posibles. Y no depende de la
     deriva: sigue en el 85% de las series **sin subida de BTC**.
   - Tiene lógica: V2 está fuera el 87% del tiempo, y una pata pequeña de BTC
     siempre dentro diversifica sin necesitar timing.
2. **La mezcla no tiene edge.** Es diversificación, no predicción. Mejora el
   CAGR/DD, no el retorno: con el mismo DD que V2 al 4,5% habría que
   apalancarla, y eso no se ha medido.
3. **En las series sintéticas, BTC al 17,6% solo tiene mejor CAGR/DD que V2**
   (mediana 0,64 contra 0,46). Coincide con E1 y con `bootstrap_bloques/`: sin
   el orden de regímenes real, el timing de V2 no aporta sobre la deriva.
4. **Para decidir:** si el objetivo es retorno por unidad de DD, mantener una
   pata pequeña de BTC spot junto a V2 es la mejora más robusta que ha salido
   en el proyecto. **Pero no superó el criterio pre-registrado** y no es lo
   desplegado. Usarlo es una decisión del usuario, no un resultado validado.
