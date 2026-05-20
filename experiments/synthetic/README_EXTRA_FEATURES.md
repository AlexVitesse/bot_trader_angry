# Test — ¿Añadir features de velas/volumen mejora V2?

> Respuesta empírica a: "los traders ven velas y volumen, ¿no deberíamos verlo
> mejor?" Probamos 4 filtros a priori derivados del trading discrecional clásico
> + un control aleatorio.
> Fecha: 2026-05-19 · `test_extra_features.py`

---

## Diseño

Cada filtro se aplica COMO CAPA encima de V2 (A + F_BTC). Si la señal
original de V2 dispara, el filtro decide si la deja pasar o la bloquea.
Filtros definidos A PRIORI con números estándar — no descubiertos en data.

| Filtro | Lógica | Origen |
|--------|--------|--------|
| vol_zscore ≥ 1.5 | Volumen actual a ≥1.5σ sobre rolling 100 | Estadística estándar |
| body_strong ≥ 0.6 | \|close-open\| / (high-low) ≥ 60% | Trading discrecional clásico |
| close_strong (tercio) | LONG: cierre en tercio superior; SHORT: inferior | Closing strength |
| engulfing | Bullish/bearish engulfing alineado con dirección | Patrón Bulkowski |
| **CONTROL: random 50%** | Bloquea 50% de entradas aleatoriamente | Validar el protocolo |

Criterio para AÑADIR un filtro:
- Mediana annual delta > 0 sobre 20 series sintéticas
- Ayudó en ≥14/20 (70%)

---

## Resultado en BTC real 2020-2025 (in-sample)

| Filtro | N trades | WR | PF | Annual | Δ vs V2 |
|--------|---------:|---:|---:|-------:|--------:|
| **V2 baseline** | **166** | **44.6%** | **1.64** | **+23.4%** | — |
| vol_zscore ≥ 1.5 | 93 | 39.8% | 1.60 | +11.8% | **-11.6%** |
| body_strong ≥ 0.6 | 134 | 42.5% | 1.65 | +18.5% | -4.9% |
| close_strong (tercio) | 150 | 44.0% | 1.64 | +19.9% | -3.5% |
| engulfing | 40 | 35.0% | 0.91 | **-1.3%** | **-24.7%** |
| **CONTROL random 50%** | 117 | 42.7% | 1.43 | +11.0% | **-12.4%** |

### Interpretación clave

**El control aleatorio (random 50%) dio -12.4%.** Esa es la magnitud esperable
del daño que causa **simplemente perder muestra** sin aportar información.

Mirando los filtros relativos al control:

| Filtro | Δ vs V2 | Δ vs CONTROL random | Significado |
|--------|--------:|--------------------:|-------------|
| vol_zscore ≥ 1.5 | -11.6% | +0.8% | **= a tirar moneda** — no aporta info |
| close_strong (tercio) | -3.5% | +8.9% | Marginal — algo de info pero no mucho |
| body_strong ≥ 0.6 | -4.9% | +7.5% | Marginal — algo de info |
| engulfing | -24.7% | -12.3% | **Peor que random** — costo de muestra demasiado alto, edge negativo |

**vol_zscore se comporta como un filtro aleatorio** — el "z-score más estricto"
no contiene información direccional adicional sobre lo que vol_ratio>1.2 ya
capturaba. Solo recorta muestra.

**engulfing es peor que el control aleatorio**: 40 trades en 6 años es muestra
ínfima, y los pocos engulfings que disparan son fundamentalmente neutros o
negativos. **Confirma los estudios formales de Bulkowski**: patrones de velas
tienen edge insignificante post-costes.

**body_strong y close_strong** muestran señales débiles (delta vs random ~+8%)
pero no superan baseline. Reducen muestra y annual a la vez — net negativo.

---

## Resultado sintético (20 series)

| Filtro | Real Δ | Synth mediana Δ | Ayudó | Empeoró | Veredicto |
|--------|-------:|----------------:|------:|--------:|-----------|
| vol_zscore ≥ 1.5 | -11.6% | -0.48% | 10/20 | 10/20 | Sin señal — = azar |
| body_strong ≥ 0.6 | -4.9% | +0.07% | 10/20 | 10/20 | Sin señal — neutro |
| close_strong (tercio) | -3.5% | +1.51% | 12/20 | 8/20 | Sin señal — borderline |
| engulfing | -24.7% | -3.60% | 6/20 | 14/20 | **DESCARTAR** |
| **CONTROL random 50%** | -12.4% | -2.50% | 7/20 | 13/20 | (esperado negativo) |

### Trades retenidos por filtro (real BTC)
| Filtro | N real | N synth (mediana) | % baseline |
|--------|-------:|------------------:|-----------:|
| BASELINE | 166 | 157 | 100% |
| vol_zscore | 93 | 67 | 56% |
| body_strong | 134 | 127 | 81% |
| close_strong | 150 | 139 | 90% |
| engulfing | 40 | 41 | 24% |
| CONTROL random | 117 | 103 | 70% |

### Lectura crucial — el real vs sintético es informativo
- En **REAL**, vol_zscore dio -11.6% (cercano al control). En **sintético**, dio -0.48%.
- En universos paralelos, vol_zscore a veces ayuda, a veces estorba → promedio ≈ 0.
- En el universo REAL (BTC 2020-2025), justo no funcionó.
- **El sintético es más informativo que el real** para evaluar si un filtro tiene
  edge sistemático — el real es solo UNA observación, el sintético son 20.

### El protocolo funciona (validación con control)
CONTROL random 50% salió en 13/20 worse, mediana -2.5%. Eso confirma que el
test **discrimina correctamente filtros sin información** de filtros con
información. Si alguno de los filtros reales hubiera salido como random,
sabemos que es indistinguible de ruido.

### Conclusión del sintético
**Ningún filtro pasa el criterio "AÑADIR" (≥14/20 + mediana>0).** El menos malo
(close_strong) llega solo a 12/20. El peor (engulfing) cae a 6/20 con mediana
-3.6% — peor que el control aleatorio.

---

## Conclusión

**Ningún filtro de los probados mejora V2.**

La intuición "los traders miran velas y volumen mejor, agreguemos eso" es
**falsa empíricamente** sobre BTC 4h:

1. **V2 ya usa velas y volumen** en su forma útil (Donchian = patrón compuesto;
   vol_ratio = filtro de volumen). Las versiones "más sofisticadas" no aportan.
2. **El "z-score más estricto" se comporta como filtro aleatorio** —
   confirmando que vol_ratio>1.2 ya capturaba todo lo capturable.
3. **Patrones de velas (engulfing) son peores que random** en BTC 4h —
   coincide con la literatura formal (Bulkowski et al.).
4. **El control aleatorio (-12% annual) define el "ruido baseline"** — cualquier
   filtro que no supere claramente esa cota es ruido.

**Implicación operativa**: V2 permanece **FROZEN como está**. Cualquier
mejora futura debe venir de:
- Diseñar una nueva estrategia con mecanismo distinto (no más filtros)
- Aplicar la misma metodología a otro mercado (BTC daily, ETH, futuros basis)
- NO añadir capas de "inteligencia" a V2

---

## Por qué los traders humanos creen ver edge donde no lo hay

Los humanos somos máquinas de detectar patrones. Cuando un trader ve un
engulfing seguido de un movimiento al alza, **lo recuerda**. Cuando ve un
engulfing seguido de nada o caída, **lo olvida**. Tras meses, su cerebro
"ha visto que engulfing predice subidas" — sesgo de memoria, no edge real.
Los tests formales (este, Bulkowski, papers académicos) demuestran que el
edge real es indistinguible de cero después de costes.

Por eso un sistema honesto vale más que años de intuición — quita el sesgo
de la ecuación.

---

## Script
- `test_extra_features.py` — runner del experimento
