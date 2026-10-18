# Smart Money Concepts / ICT codificado

> Fecha: 2026-10-17 · `test_smc.py` · salida completa en `salida.txt`
> **Estado: RECHAZADO.** 0/6 variantes ganan dinero de forma significativa y
> ninguna sobrevive al null con corrección por selección. ETH tampoco.

## Origen

Videos de TikTok de "Asimétricos Traders" que trajo el usuario. Los cuatro son
SMC/ICT: la vela de referencia, Fibonacci reducido a la zona OTE, los máximos
que deja una caída como objetivo (liquidez) y AMD (acumulación → manipulación
→ distribución). Era la única familia popular de "trading de traders" que el
proyecto no había medido.

## Reglas (pre-registradas en el docstring antes de correr)

LONG; SHORT es el espejo exacto, la misma lógica sobre precios negados.

1. **Liquidez:** `swing` = último mínimo fractal confirmado (3 velas a cada
   lado), o `pdhl` = mínimo del día anterior ("la vela de referencia").
2. **Manipulación:** low < nivel y close > nivel (barrida de stops).
3. **Ruptura de estructura** en ≤ 6 velas: close > último máximo interno.
4. **Entrada:** límite en el OTE 0,705 del tramo barrida → máximo, válida 12
   velas. Fill a min(límite, open).
5. **Stop** bajo la barrida (−0,1%). **Objetivo** en la liquidez de arriba:
   máximo fractal más cercano por encima, o máximo del día anterior. Solo si
   R:R ≥ 2.
6. Stop primero si stop y objetivo caen en la misma vela; tope de 30 velas.
   Riesgo 2%. Costes 0,06% por lado + funding + gap.

`--check` verifica que truncar la serie no cambia los trades ya cerrados (sin
look-ahead). Pasa en las 6 variantes.

## Resultados — BTC 4h, 2019-01 → 2026-02

| variante | n | tr/año | WR | PF | R medio | CAGR | DD | p bloques 30d |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| swing/long | 51 | 7,1 | 21,6% | 0,77 | −0,07 | −3,1% | 39,3% | 0,78 |
| swing/short | 52 | 7,3 | 25,0% | 0,93 | +0,00 | −1,2% | 18,4% | 0,61 |
| swing/ambas | 101 | 14,1 | 22,8% | 0,83 | −0,05 | −4,5% | 42,9% | 0,78 |
| pdhl/long | 23 | 3,2 | 21,7% | 0,63 | −0,10 | −2,1% | 16,4% | 0,83 |
| pdhl/short | 24 | 3,4 | 33,3% | 1,38 | +0,29 | +1,4% | 13,7% | 0,25 |
| pdhl/ambas | 45 | 6,3 | 28,9% | 1,03 | +0,15 | −0,1% | 25,4% | 0,48 |

**ETH (cross-asset):** 5/6 variantes con PF < 1. La mejor, pdhl/short, da PF
1,07 y p=0,47.

**Null sintético** (300 series, BTC): la única variante interesante es
pdhl/short, con p=0,023 en bloques de 7 días. Pero con selección (la mejor de 6)
sube a **p=0,35**, en bloques de 30 días a **p=0,48**, y no gana dinero de forma
significativa (p=0,25 por bloques). Con 24 trades en 7 años, además, no hay
muestra. Es el positivo que el azar da solo cuando se miran 6 variantes.

## Lectura

- **El win rate real es 22-33%.** Con R:R ≥ 2 el break-even es ~33%. Los
  gráficos de los videos muestran los setups que llegaron al objetivo; la regla
  mecánica también toma todos los que no llegan.
- **La "barrida + ruptura" no anticipa la dirección.** El null, que conserva la
  forma de las velas pero baraja su orden, produce setups igual de buenos (o
  malos).
- **SMC es discrecional por diseño** ("un mapa, no una estrategia", dice uno de
  los títulos). Otra codificación podría dar otros números, pero cada
  codificación nueva es otra variante que hay que pagar en la corrección por
  selección. Ésta usa las definiciones canónicas y no se ajustó.

No re-tunear (OTE, R:R, k del fractal): sería la búsqueda de parámetros número
ocho. El mapa de familias en `carry_funding/README.md` gana una entrada más.
