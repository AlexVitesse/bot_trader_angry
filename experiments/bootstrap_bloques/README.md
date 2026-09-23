# Bootstrap honesto de V2 — Fase 4 del plan 2026-09

> Fecha: 2026-09-23 · `test_bloques.py` · salida completa en `salida.txt`,
> distribuciones nulas en `resultados.npz`.
> Responde a `docs/AUDITORIA_2026-09.md` §1.2 (bootstrap i.i.d. sobre trades
> agrupados), §1.3 (selección de variante sin corregir) y §1.7 (costes).
> **No cambia ningún parámetro de V2.**

## Respuesta corta

| pregunta | respuesta |
|---|---|
| ¿Es V2 rentable en la historia, respetando la dependencia temporal? | **Sí.** p entre 0,004 y 0,033 en todas las versiones por bloques. |
| ¿Tiene V2 un edge **por encima** de lo que da cualquier regla parecida sobre BTC con la misma deriva y la misma estructura de corto plazo? | **No demostrado.** p = 0,13–0,20 sin corregir por selección; 0,33–0,68 corrigiendo. |
| ¿Sobrevive la selección entre las variantes probadas? | Con 6 variantes, Bonferroni sobre el p i.i.d. da 0,053: **raya**. Con las ~36 evaluaciones documentadas, no. |
| Expectativa con costes realistas | **2% → +11,1%/año, DD 21,5%. 4,5% → +23,9%/año, DD 43,4%.** |

Lectura: los trades de V2 ganan dinero de forma robusta a la agrupación en
regímenes, pero gran parte de esa ganancia es **la deriva alcista de BTC
capturada por una regla long-only**. Una serie con las mismas velas en otro
orden (deriva idéntica) ya le da a V2 una expectativa por trade de +0,34%
(mediana), frente al +0,49% real. La diferencia no es significativa.

## 1. Expectativa — una sola tabla (costes nuevos)

BTC solo, V2 congelado (`f_enable_short=False`), simulador de cartera
(`experiments/portfolio_sim/`), fill al open t+1, 2019-01 → 2026-02.

| risk/trade | n | WR | PF | CAGR | DD máx |
|--:|--:|--:|--:|--:|--:|
| **2,0%** | 165 | 44,2% | **1,65** | **+11,1%** | **21,5%** |
| **4,5%** (desplegado) | 165 | 44,2% | **1,65** | **+23,9%** | **43,4%** |

Costes nuevos frente a los viejos (`costes='viejos'` los reproduce):

| costes | PF | CAGR 2% | DD 2% | CAGR 4,5% | DD 4,5% |
|---|--:|--:|--:|--:|--:|
| viejos | 1,77 | +13,2% | 19,7% | +29,0% | 40,4% |
| nuevos | 1,65 | +11,1% | 21,5% | +23,9% | 43,4% |

Qué cambia (§1.7): slippage 0,02%/lado (antes 0,01%); funding histórico real
de BTC (`data/btc_v15_funding.parquet`, desde 2020-01; constante antes); stop
llenado a `min(stop, open)` si la vela abre con gap por debajo. Descomposición
a 2%: slippage −0,3 pts CAGR, funding −0,2 pts (+1,4 pts DD), **gap −1,6 pts**.

Al riesgo desplegado (4,5%) el sistema **no llega al 30% anual** del objetivo
del proyecto y el DD histórico ya roza el kill switch (45%).

## 2. p-valor sobre los trades observados (H0: mean(r) ≤ 0)

20.000 remuestreos. `r` = PnL del trade / equity a la entrada.

| remuestreo | bloques | p |
|---|--:|--:|
| i.i.d. (el publicado, con costes nuevos) | 165 | 0,009 |
| bloques circulares de 5 trades | — | 0,009 |
| bloques de 10 trades | — | 0,008 |
| bloques de 20 trades | — | 0,004 |
| calendario, 40 días (= 5 × mediana entre trades, 8,1 d) | 46 | 0,005 |
| calendario, 30 días | 62 | 0,013 |
| calendario, 90 días | 24 | 0,006 |
| calendario, 180 días | 12 | 0,017 |
| **episodios de régimen `bull_1d`** | **6** | **0,033** |

Los bloques de calendario cuentan solo los bloques que contienen trades.

Respetar la dependencia temporal **sube el p pero no lo saca de 0,05**. El caso
más duro, remuestrear episodios de régimen enteros, da 0,033 — con solo 6
episodios, así que es un test de muy baja resolución. El p=0,004 publicado
era optimista; el rango honesto es 0,005–0,033.

Esto responde "¿los trades de V2 tienen media positiva?", no "¿V2 tiene
habilidad?". Un sistema long-only sobre un activo que se multiplicó por 20 en
el periodo puede tener media positiva sin habilidad de timing. Para eso está
la sección 3.

## 3. Null sintético — ¿V2 le gana a su propia regla sobre series sin su historia?

Cada serie sintética permuta **bloques de velas 4h** de BTC (cada vela guardada
como gap + cuerpo + mechas relativos, precio reconstruido encadenando).
Conserva: la distribución de velas, la volatilidad agrupada dentro del
bloque, la **deriva total** (el precio final es el mismo) y la estructura de
corto plazo. Rompe: el orden de los regímenes a plazo mayor que el bloque, que
es justo lo que el filtro diario EMA50/200 y el Donchian-55 dicen explotar.

En cada serie se corre V2 completo (features, señales, simulador de cartera,
costes nuevos, riesgo 2%). **800 series por longitud de bloque** (medido
~20 s de CPU por serie con las 6 variantes; 1000 no cabía en 40 min en esta
máquina). Estadísticos: `mean(r)` (el que pide el plan) y `t = mean/sd·√n`
(el criterio real con el que se eligió V2: menor bootstrap p).

**Con selección:** en cada serie se corren las 6 variantes y se toma la mejor;
p = fracción de series donde la mejor iguala o supera a V2 observado. Es la
forma de meter la selección *dentro* del null (Reality Check de White), en
vez de corregir el p después.

| bloque | estadístico | V2 observado | mediana null | p sin selección | p con selección |
|---|---|--:|--:|--:|--:|
| 42 velas (7 días) | mean(r) | +0,49% | +0,34% | 0,201 | 0,519 |
| 42 velas (7 días) | t | 2,31 | 1,67 | 0,163 | 0,329 |
| 180 velas (30 días) | mean(r) | +0,49% | +0,36% | 0,183 | 0,679 |
| 180 velas (30 días) | t | 2,31 | 1,69 | 0,134 | 0,424 |

Ninguna serie sintética dejó a V2 con menos de 10 trades.

**Lectura.** La mediana del null ya es claramente positiva: la regla gana
dinero sobre cualquier reordenación de las velas de BTC porque BTC sube y la
regla es long-only. V2 sobre la historia real está por encima de esa mediana,
pero **entre una de cada cinco y una de cada ocho reordenaciones aleatorias
lo iguala o supera** — y entre un tercio y dos tercios si se permite elegir la
mejor de las 6 variantes, como se hizo. El edge de *timing* de régimen, que es la tesis de V2, **no se
distingue del azar** con estos datos.

Limitación importante: el null conserva la estructura de hasta 1-4 semanas.
Si parte del edge de trend-following vive en ese plazo, el null lo contiene y
el test es conservador en esa parte. Lo que sí mide limpiamente es el valor
del filtro de régimen de largo plazo.

## 4. Variantes contadas (selección)

| fuente | evaluaciones | qué |
|---|--:|---|
| `combined_AF/README.md` | 6 | V1 sleeve, V1 50/50, V2, V3 sleeve, V3 50/50, V4 — V2 elegida por menor p |
| `agent_F/explore_params.py` | 29 | barrido uno-a-uno: dirección 3, compression percentile 4, breakout_n 4, min bars 4, filtro régimen 2, target vol 4, trail ATR 4, vol ratio 4 |
| `f_short_ablation/` | 1 adicional | F_SHORT on/off sobre la historia completa (se apagó) |
| ronda de familias `agent_A..O` | ~14 familias | A y F elegidas entre ellas (capa exterior, no cuantificada aquí) |

Las 6 variantes del null (sección 3) son las **combinaciones de componentes**
reproducibles con el motor V2: V2, A+F bidir, A solo, F_LONG solo, F bidir
solo, A+F_SHORT. No incluyen el barrido de `agent_F` (29) ni la elección de
familia, así que **el p con selección es una cota inferior**.

Corrección sobre el p i.i.d. (0,009, costes nuevos):

| m (comparaciones) | Bonferroni | ¿< 0,05? |
|--:|--:|:--:|
| 6 | 0,053 | raya, no |
| 36 (6 + 29 + 1) | 0,32 | no |

## Conclusión

1. **La expectativa honesta** es +11,1%/año a 2% de riesgo y +23,9% a 4,5%
   con DD 43%. Por debajo del umbral de 30% del proyecto.
2. **Rentable sí, robusto a la agrupación temporal** (p ≤ 0,033 en todos los
   bloques).
3. **Habilidad de timing no demostrada**: frente a series con la misma deriva,
   p ≈ 0,13–0,20; incluyendo la selección, 0,33–0,68.
4. Nada de esto cambia la recomendación: seguir en paper trade. **El paper
   trade es la evidencia que falta, no el backtest.** Un sistema cuyo edge
   sobre el null no es significativo necesita que la ventana alcista real
   (A_LONG en vivo) entregue lo que predice.

## Reproducir

```
python experiments/bootstrap_bloques/test_bloques.py 800
```

~57 min en 12 hilos (32 + 24 min). `resultados.npz`: `obs` (6 variantes × [mean, t]) y
`null_42`, `null_180` (series × variantes × [mean, t]).
