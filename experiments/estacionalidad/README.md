# Estacionalidad — el efecto miércoles y por qué no es un edge

> Fecha: 2026-08-22 · `test_estacionalidad.py` · BTC 4h 2019-09 → 2026-03
> (14.214 velas)

Familia sin medir hasta ahora. Y sirve de caso de estudio: es el experimento
donde el test ingenuo dice **p=0,005** y el test correcto dice **p=0,14**.

---

## A/B) Efecto por bucket, con control de comparaciones múltiples

Se prueban 25 buckets (6 horas UTC × 7 días × 12 meses). Tres salen
significativos a p<0,05:

| bucket | media por vela | t | p |
|---|--:|--:|--:|
| hora 20 UTC | +0,0695% | 2,57 | 0,010 |
| **día 2 (miércoles)** | **+0,0838%** | **2,80** | **0,005** |
| mes 10 (octubre) | +0,0862% | 3,01 | 0,003 |

```
significativos a p<0,05:  3 de 25
esperados SOLO por azar:  1,2
```

Tres contra 1,2 esperados. No es escandaloso, pero tampoco despreciable — vale
la pena seguir mirando en vez de descartarlo de entrada.

## C) ¿Persiste entre mitades?

| familia | mejor en 1ª mitad | media 1ª | media 2ª | |
|---|---|--:|--:|---|
| hora UTC | 20 | +0,0721% | +0,0668% | persiste |
| **día semana** | **2 (miércoles)** | **+0,0844%** | **+0,0833%** | **persiste** |
| mes | 10 | +0,1056% | +0,0602% | persiste |

Y la correlación del **perfil completo** entre mitades:

| familia | corr | |
|---|--:|---|
| hora UTC | +0,228 | inestable |
| **día semana** | **+0,849** | **estable** |
| mes | +0,122 | inestable |

El efecto miércoles no solo persiste: el perfil entero de días de la semana
tiene correlación 0,849 entre las dos mitades de la muestra. En este punto
parecía un hallazgo.

---

## D) El tercer grado — y ahí se cae

**No es un problema de outliers.** Quitando el 1% de velas más extremas el
efecto apenas se mueve (delta +0,0646% vs +0,0719%). **No es un solo año**:
positivo en 7 de 8. Ambas cosas apuntaban a que era real.

Pero:

**1. La mediana no difiere.**

| | media | mediana |
|---|--:|--:|
| miércoles | +0,0838% | **+0,0157%** |
| resto | +0,0119% | **+0,0203%** |

El miércoles típico **no** es mejor que el día típico. Todo el efecto vive en
la cola derecha: unos pocos miércoles muy buenos arrastran la media.

**2. El t-test estaba mal calibrado.** Supone observaciones independientes, y
seis velas de 4h del mismo día no lo son. El null correcto **rota el calendario
entero**, lo que preserva la autocorrelación:

```
delta observado:                    +0,0719%
p-valor (10.000 rotaciones):         0,1405
p-valor del t-test ingenuo:          0,005
```

**El 14% de las rotaciones del calendario producen un "efecto día" tan grande
como el real.** Con el test correcto, el miércoles no se distingue del azar.

**3. Y aunque fuera real, no serviría.** Backtest de comprar cada miércoles a
las 00:00 UTC y vender 24h después, con 0,10% de costes:

| | anual | DD |
|---|--:|--:|
| LONG cada miércoles (338 trades, WR 52,1%) | **+15,07%** | −36,8% |
| BTC comprar y mantener | **+35,23%** | — |

Rinde **menos de la mitad** que estar largo y quieto. No es un edge: es una
forma peor de estar largo, que captura una parte del drift de BTC estando en
mercado 1/7 del tiempo.

---

## Veredicto: REJECT

No hay estacionalidad explotable en BTC 4h. El único candidato serio —
miércoles — sobrevive a persistencia y a la eliminación de outliers, pero
muere en el control de rotación de calendario (p=0,14) y no llega a
buy-and-hold ni con viento a favor.

## La lección metodológica, que vale más que el resultado

Es el mismo patrón que `criterio_validacion/`: **el test estándar aplicado a
datos autocorrelacionados da falsos positivos.**

| | dice |
|---|---|
| t-test sobre 2.033 velas de miércoles | p = 0,005 → "significativo" |
| rotación de calendario (null correcto) | p = 0,14 → ruido |

La diferencia son dos órdenes de magnitud, y viene entera de asumir
independencia donde no la hay. Cualquier futuro test sobre buckets temporales
en este proyecto debe usar el null por rotación, no el t-test.

Es exactamente el mismo error que hacía que `N_eff` de `adx` fuera 35 y no
15.187 (`presupuesto_informacion/`), y que el `≥7/12 folds` dejara pasar
sistemas sin edge (`criterio_validacion/`). Tres apariciones del mismo bicho
en la misma sesión.
