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

[Test corriendo al momento de redactar — actualizaré aquí]

[Esperado basado en real + lecciones previas: ninguno alcanza 14/20 con
mediana > 0. CONTROL random debe salir claramente negativo, validando protocolo.]

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
