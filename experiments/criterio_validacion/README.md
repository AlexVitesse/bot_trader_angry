# El criterio de validación estaba mal calibrado

> Fecha: 2026-08-22 · `test_criterio.py` · 20.000 remuestreos, semilla 42

## Qué se midió

`CLAUDE.md` exigía **≥ 7/12 folds positivos** para aprobar cualquier modelo.
Nunca se había comprobado qué discrimina ese criterio. Se mide por remuestreo:
se toma una distribución de trades, se barajan, se agrupan en folds y se cuenta
cuántas veces pasaría el filtro.

## Resultado

**Sistema CON edge real** — V2, muestra empírica (n=131, WR 45%, PF 1,83,
EV +0,971%/trade, skew +1,55):

| criterio | pasa | mediana |
|---|--:|--:|
| 7/12 folds+ | 90,2% | 9/12 |
| 4/6 folds+ | 91,4% | 5/6 |
| 6/10 folds+ | 92,3% | 8/10 |

**Sistema SIN edge, con win rate alto** — WR 66%, PF 0,93, EV −0,050%/trade
(cero edge por construcción):

| criterio | pasa | mediana |
|---|--:|--:|
| **7/12 folds+** | **52,9%** | 7/12 |

## Las dos conclusiones

### 1. El criterio deja pasar basura la mitad de las veces

Un sistema con **cero edge** aprueba 7/12 folds en el **52,9%** de los
remuestreos, con solo tener win rate alto. Contar folds mira únicamente el
signo del fold y tira la magnitud, así que un sistema que gana muchas veces
poco y pierde pocas veces mucho lo engaña sistemáticamente.

Y ése es exactamente el perfil de los cinco fracasos de `CLAUDE.md`:

| versión | WR backtest | producción |
|---|--:|---|
| V7 Original | — | 33-42% WR |
| V9 LossDetector | 68% | 41,4% WR |
| BTC V2 | 65,7% | 43,8% WR |
| SOL V2 | 63,6% | 12,5% WR |
| V13.03 | 67,3% | sin validar |

**El filtro que se adoptó para evitar el overfitting era ciego precisamente al
perfil que ya había fallado cinco veces.**

### 2. El 2/6 de V2 no es mala suerte — es información

Si V2 barajado pasa 4/6 folds el **91,4%** de las veces y el walk-forward real
dio **2/6**, la diferencia no es ruido de muestreo. Significa que el edge de V2
**no está distribuido uniformemente en el tiempo, se concentra**.

Eso confirma por una vía independiente el discriminador que ya identificaba
`docs/SESION_2026-08-09.md`: V2 gana en tendencia limpia y pierde en tendencia
entrecortada. El fold-count no estaba diciendo "esto no funciona", estaba
diciendo "esto funciona a ratos" — que es una afirmación distinta y más útil.

## Qué se cambió

`CLAUDE.md` § *Requisitos de Validación*, criterio 1:

- **Antes**: `≥ 7/12 folds positivos` como aprobado/suspenso.
- **Ahora**: **bootstrap p < 0,05 + tamaño de efecto** como criterio primario.
  El conteo de folds se sigue reportando, pero como **diagnóstico de
  estacionariedad** (¿cuándo funciona?), no como puerta.

El bootstrap ya estaba construido (`portfolio_sim/`, `v2_all_coins/`) y sí
discrimina: V2 da **p=0,004 estable en todos los niveles de riesgo**, y BTC
p=0,006 en la evaluación de 22 monedas donde 12 fueron rechazadas.

## Aviso metodológico

El sistema sin edge se simula con retornos i.i.d. de dos valores; los sistemas
reales tienen autocorrelación y volatilidad variable, que si acaso **empeoran**
la capacidad discriminante del conteo de folds. El 52,9% es un límite superior
optimista, no un peor caso.
