# Predictibilidad — no es que falten parámetros, es que no hay señal

> Fecha: 2026-08-22 · `test_predictibilidad.py` · BTC 2019-09 → 2026-03,
> 14.159 velas 4h

`presupuesto_informacion/` demuestra que **no caben** los parámetros de un
modelo grande. Éste demuestra lo complementario y más fuerte: que **tampoco hay
señal que aprender**. Juntos cierran la pregunta "¿y si usamos modelos más
complejos?".

---

## A) Correlación lineal, contra un suelo de ruido honesto

La trampa habitual es mirar una correlación de 0,02 y decir "poco pero algo
hay". Para saber si es algo, hace falta saber cuánto da el azar. Suelo
construido con **5.000 features aleatorias** (permutaciones del target):

```
|corr| p95 = 0,0167    p99 = 0,0222    max = 0,0300
```

| feature | \|corr\| | R² | veredicto |
|---|--:|--:|---|
| `adx` | 0,0084 | 0,007% | **RUIDO** |
| `atr_pct` | 0,0237 | 0,056% | señal |
| `bb_width` | 0,0159 | 0,025% | **RUIDO** |
| `vol_ratio` | 0,0084 | 0,007% | **RUIDO** |
| `bull_1d` | 0,0036 | 0,001% | **RUIDO** |

**Cuatro de cinco features predicen menos que una columna de números
aleatorios.** La única que supera el umbral es `atr_pct`, y predice *cuánto* se
va a mover el precio, no *hacia dónde*: es clustering de volatilidad, un efecto
bien conocido que no da dirección y por tanto no da dinero por sí solo.

### El techo optimista

```
mejor combinación lineal de las 5, medida EN EL PROPIO TRAIN:
  R² = 0,068%
  -> 99,93% de la varianza sin explicar, en los datos que el modelo ya vio
```

En un problema donde el techo **in-sample** es 0,07%, el test no puede ser otra
cosa que AUC ≈ 0,5. El `train 0,805 → test 0,513` de `agent_G/` no era mala
suerte ni mala validación: era el único resultado posible.

---

## B) ¿Y a horizontes más largos?

| horizonte | `adx` | `atr_pct` | `bb_width` | `vol_ratio` | `bull_1d` |
|---|--:|--:|--:|--:|--:|
| 1 vela (4h) | 0,0084 | 0,0237 | 0,0159 | 0,0084 | 0,0036 |
| 3 velas (12h) | 0,0128 | 0,0325 | 0,0239 | 0,0086 | 0,0074 |
| 6 velas (24h) | 0,0161 | 0,0428 | 0,0268 | 0,0177 | 0,0121 |
| 12 velas (48h) | 0,0201 | 0,0457 | 0,0252 | 0,0171 | 0,0175 |

A simple vista las correlaciones **suben** con el horizonte. Es un espejismo:
las ventanas se solapan, así que el número de observaciones independientes cae
y **el suelo de ruido sube con ellas**.

El suelo p95 de una correlación con `n` observaciones independientes es
≈ `1,96/√n` (a 1 vela: `1,96/√14.159 = 0,0165`, que reproduce el 0,0167
medido). Corrigiendo por solape:

| horizonte | ventanas independientes | suelo p95 | mejor feature | ¿supera? |
|---|--:|--:|--:|:--|
| 1 vela | 14.159 | 0,0165 | 0,0237 | apenas |
| 3 velas | 4.720 | 0,0285 | 0,0325 | apenas |
| 6 velas | 2.360 | 0,0403 | 0,0428 | apenas |
| 12 velas | 1.180 | **0,0571** | 0,0457 | **no** |

Una vez descontado el solape, **nada se despega del ruido a ningún horizonte**,
y a 48h la mejor feature queda por debajo del suelo.

---

## C) La prueba no lineal — la que de verdad cierra el caso

Las dos anteriores solo ven relaciones lineales. Un árbol de decisión no
necesita linealidad, así que hace falta una prueba que capture estructura
arbitraria. Se parte cada feature en **deciles** y se mide cuánto separan el
retorno futuro, contra el mismo suelo por permutación (500 iteraciones):

| feature | spread entre deciles | p95 nulo | p-valor | veredicto |
|---|--:|--:|--:|---|
| `adx` | 0,0751% | 0,1526% | 0,878 | **RUIDO** |
| `atr_pct` | 0,1476% | 0,1481% | 0,052 | **RUIDO** |
| `bb_width` | 0,1034% | 0,1538% | 0,518 | **RUIDO** |
| `vol_ratio` | 0,1314% | 0,1546% | 0,178 | **RUIDO** |
| `bull_1d` | — | — | — | (binaria, test vacío) |

**Las cinco fallan.** Ni siquiera `atr_pct`, que pasaba la prueba lineal,
separa los deciles mejor que barajar el target al azar (p=0,052). No hay
estructura no lineal escondida que un modelo más flexible pudiera encontrar.

> Caveat honesto: `bull_1d` es binaria, así que `qcut` en 10 deciles colapsa y
> el test no dice nada sobre ella. Su correlación lineal (0,0036) ya la sitúa
> muy por debajo del ruido, pero conviene saber que C no la evalúa.

---

## D) Entonces, ¿por qué gana V2?

```
V2 acierta el 45,0% de las veces   ->   PEOR que una moneda
gana  +4,76%  cuando acierta
pierde −2,13%  cuando falla

esperanza = 0,450 × 4,76 + 0,550 × (−2,13) = +0,971% por trade
```

**V2 se equivoca más veces de las que acierta y gana dinero igual.**

No predice nada. Espera a que la ruptura **ya haya ocurrido**, se sube, y deja
que un trailing corte rápido lo que sale mal y deje correr lo que sale bien.
Eso es asimetría de pago, no predicción.

Y ahí está la razón de fondo por la que el ML no aplica aquí: **un clasificador
optimiza acierto, y el acierto es la variable equivocada.** Subir del 45% al
48% es imposible en un dominio con R² de 0,068%; y aunque se lograra, aportaría
menos que el ratio 2,23 que el mecanismo ya tiene gratis.

---

## Por qué este dominio es distinto de un problema de ML normal

| | predicción convencional | mercado líquido |
|---|---|---|
| señal/ruido | alta | **R² ≈ 0,07%** |
| estacionariedad | estable | se mueve (vol 60,8% → 38,8% en 5 meses) |
| ¿reacciona a tu modelo? | **no** | **sí** — si es rentable y detectable, se arbitra |
| muestra | miles de casos independientes | 11 episodios, 131 trades |

El tercer punto no tiene arreglo técnico. Una radiografía no cambia porque
hayas aprendido a leerla; un patrón de mercado sí. Es el único dominio de ML
donde **los datos te responden**.

## Conclusión

No hay que probar más modelos. Se probaron once familias, incluida LSTM, y dos
con purged CV (`agent_B/`, `agent_G/`). Lo que este experimento añade es que
**ninguna podía haber funcionado**: con R² in-sample de 0,068% y cero
estructura no lineal detectable, no hay arquitectura que extraiga lo que no
está.

El sitio donde queda margen no es el modelo, es la ejecución (slippage, fills),
el sizing, y —si algún día se paga— información ortogonal al OHLCV. Ver
`presupuesto_informacion/` § Conclusión.
