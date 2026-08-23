# Arbitraje estadístico — las cripto no cointegran

> Fecha: 2026-08-22 · `test_stat_arb.py` · 21 monedas, dos universos

Familia sin medir. La idea era buena sobre el papel: si dos monedas cointegran,
el spread es estacionario y se opera la reversión — market-neutral, sin
predecir dirección, así que esquiva el `R² = 0,068%` de `predictibilidad/`.

## El protocolo

La trampa evidente: con 21 monedas hay **210 pares**, y a p<0,05 el azar solo
ya produce ~10 "cointegrados". Y cointegrar en el pasado no implica cointegrar
después. Se separan las dos cosas, con la selección hecha **sin mirar test**:

- **A**. Cointegración (Engle-Granger) en TRAIN, contra lo esperado por azar.
- **B**. ¿Los que cointegran en train siguen cointegrando en TEST?
- **C**. Backtest de reversión z-score en TEST, con costes, solo sobre los
  elegidos en A.

Se corren **dos universos** porque hay un compromiso real: incluir monedas
recientes (SUI, ARB, FET) amplía el número de pares pero recorta la ventana
común a 2023-05.

---

## A) Ni siquiera hay cointegración al nivel del azar

| universo | pares | cointegran (p<0,05) | esperados por azar |
|---|--:|--:|--:|
| Amplio (21 monedas, train desde 2023-05) | 210 | **3** | 10,5 |
| Largo (14 monedas, train desde 2020-09) | 91 | **4** | 4,5 |

En el universo amplio hay **menos** cointegración que la que produciría ruido
puro. En el largo, exactamente la del azar. No hay nada que perseguir: el
mecanismo no existe en este universo.

Tiene sentido económico: las cripto no están ligadas por un arbitraje que
fuerce a sus precios a converger. Se mueven juntas por **factor común**
(correlación 0,69, ya medida en `portfolio_sim/`), que es otra cosa —
correlación alta no es cointegración, y de hecho es lo peor de ambos mundos:
te da riesgo compartido sin darte spread estacionario que operar.

## B) Y de los pocos que salen, ninguno persiste

```
universo amplio:  0 de 3 persisten en test  (0%)
universo largo:   0 de 4 persisten en test  (0%)
```

Si la cointegración fuera un rasgo real del par, debería reaparecer en test muy
por encima del 5% que da el azar. Sale 0%.

## C) El backtest, y la ironía

| universo | pares | trades | **WR** | **PF** | suma |
|---|--:|--:|--:|--:|--:|
| Amplio | 3 | 99 | **69,7%** | **0,88** | −39,3% |
| Largo | 4 | 136 | **65,4%** | **0,80** | −107,8% |

**Win rate del 65-70% y pierde dinero en los dos universos.**

Es la ilustración perfecta de lo que mide `criterio_validacion/`: un sistema
que acierta dos de cada tres veces y aun así tiene PF por debajo de 1, porque
gana poco muchas veces y pierde mucho pocas veces. Es exactamente el perfil de
los cinco fracasos históricos del proyecto (V7, V9, BTC V2, SOL V2, V13.03 —
WR declarado 63-68%), y exactamente el que el antiguo criterio `≥7/12 folds`
dejaba pasar el 53% de las veces.

Si este experimento se hubiera hecho con la metodología vieja, el "WR 69,7%"
habría ido al README como titular.

---

## Veredicto: REJECT

No hay cointegración explotable entre cripto. No es que la estrategia esté mal
calibrada: **el fenómeno sobre el que se apoya no está presente en los datos**,
en ninguno de los dos universos y en ninguna de las dos ventanas.

---

## Apéndice: market making, el tercer pendiente

No se mide porque no es medible con estos datos, y conviene decir por qué en
vez de dejarlo abierto:

```
rango intravela 4h mediano de BTC:    1,399%
spread típico de BTC perp:            ~0,010-0,020%
comisión del bot (taker, ida+vuelta):  0,100%
```

Un market maker captura el **spread**, no el rango. A 0,015% por vuelta contra
0,100% de comisión taker, **cada ronda pierde 0,085%**. Para que salga hace
falta fee tier con rebate de maker (VIP) y latencia de milisegundos.

El bot corre un loop de 30s sobre velas de 4h en testnet. Market making no es
un parámetro que tocar aquí — es otro producto, con otra infraestructura y otra
estructura de comisiones. Queda como **no aplicable**, no como pendiente.

---

## Estado del mapa de familias

Con este documento y `estacionalidad/`, el mapa de `carry_funding/README.md`
queda cerrado: **no quedan familias de estrategia sin medir** dentro de lo que
la arquitectura del bot permite. Lo único que sigue sin explorar son **fuentes
de datos** (order book, open interest, liquidaciones), no estrategias.
