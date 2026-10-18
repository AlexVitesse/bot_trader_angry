# V2 en oro y bolsa: ¿diversificar sube el retorno al mismo DD?

> Fecha: 2026-10-17 · `test_multiactivo.py` · salida completa en `salida.txt`
> **Estado: RECHAZADO.** Ningún ETF pasa el null sintético y la cartera no
> mejora a BTC solo a igual drawdown.

## Pregunta

El usuario pidió llegar al 60% anual diversificando (más activos, oro, bolsa).
El mecanismo legítimo es éste: si hay sleeves con edge propio y poco
correlacionados, el DD de la cartera baja y se puede subir el riesgo por trade
al mismo DD. Se mide eso, no el CAGR suelto.

## Diseño (pre-registrado en el docstring antes de correr)

- SPY, QQQ, IWM, GLD y TLT, en velas **diarias** de yfinance con dividendos
  reinvertidos. yfinance no da 4h con historia larga.
- Reglas V2 congeladas, LONG-only, régimen EMA50/200 sobre la misma serie.
- Único ajuste: suelo y techo del trailing × k, donde k = ATR% mediano del
  activo / ATR% de BTC 4h. Es el mismo criterio que usó `timeframe/` para BTC
  diario.
- Costes: 0,06% por lado, gap en el stop, financiación ~2,4% al año sobre el
  nocional.
- `portfolio_sim.py` gana `params_por_par` y un arreglo para mezclar
  temporalidades (ver abajo).

## 1. Por activo (historia completa, 2% de riesgo)

| activo | desde | n | tr/año | WR | PF | CAGR | DD | B&H | p bloques 90d | p null |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SPY | 1993 | 54 | 1,6 | 46,3% | 1,30 | +0,3% | 11,0% | +10,8% | 0,217 | 0,370 |
| QQQ | 1999 | 49 | 1,8 | 36,7% | 0,96 | −0,1% | 16,8% | +10,7% | 0,549 | 0,790 |
| IWM | 2000 | 42 | 1,6 | 33,3% | 1,17 | +0,2% | 11,1% | +8,4% | 0,351 | 0,210 |
| GLD | 2004 | 67 | 3,1 | 52,2% | 1,92 | +1,8% | 11,5% | +10,3% | **0,007** | 0,240 |
| TLT | 2002 | 57 | 2,4 | 45,6% | 1,28 | +0,5% | 13,8% | +3,4% | 0,196 | 0,210 |

Corte con Bonferroni (m=5): p < 0,01. **Solo GLD gana dinero de forma
significativa (p=0,007), pero no supera al null (p=0,24).** Es decir, captura la
deriva alcista del oro y no hace timing. Es la misma conclusión que V2 en BTC
(`bootstrap_bloques/`).

Los ~2 trades al año por activo hacen que, incluso con un edge real, el sleeve
pese muy poco.

## 2. Cartera, 2019-01 → 2026-02 (periodo con datos de BTC)

| riesgo | BTC solo CAGR / DD | BTC+ETFs CAGR / DD | ETFs solos CAGR / DD |
|--:|--:|--:|--:|
| 2,0% | +11,1% / 21,5% | +15,6% / 28,1% | +4,4% / 17,1% |
| 3,0% | +16,4% / 30,9% | +22,2% / 39,6% | +5,7% / 24,7% |
| 4,5% | +23,5% / 42,4% | +30,5% / 49,9% | +6,3% / 36,9% |
| 6,0% | +27,2% / 50,0% | +31,1% / 51,7% | +7,8% / 37,1% |
| 8,0% | +28,7% / 56,7% | +36,5% / 55,8% | +8,9% / 39,5% |

Los ETFs **suben CAGR y DD a la vez**, casi en la misma proporción que subir el
riesgo de BTC:

- a ~40% de DD: BTC solo da ~+21% (riesgo 4%) y la cartera +22,2%;
- a ~50% de DD: BTC solo da +27,2% y la cartera +30,5%.

Unos 1-3 puntos, y casi todo viene del oro en 2019-2026 (contribución GLD
+25,3% contra +1,5% de SPY). Fue uno de los mejores tramos del oro en décadas,
así que es **selección de periodo**, no edge. En la historia larga (2005-2026)
los ETFs juntos dan **+2,5% al año con 34% de DD**.

La correlación mensual con BTC es baja (−0,24 a +0,18). La diversificación
existe, pero **diversificar sleeves sin edge no añade retorno**: solo añade
ruido no correlacionado.

**El 60% no aparece en ninguna fila.** Por encima del 6% de riesgo el CAGR se
aplana (margen y tope de nocional) y el DD pasa del kill switch del 45%.

## Bug de simulador encontrado (y arreglado)

`PortfolioSim` descartaba la orden pendiente si la vela siguiente del índice
maestro no era del mismo par. Con BTC en 4h y los ETFs en diario, toda señal
diaria se perdía a las 04:00 y la cartera daba exactamente lo mismo que BTC
solo. Ahora la orden espera a la siguiente vela de su par. **Las cifras de
referencia de BTC solo no cambian** (165 trades, +23,5% / 42,4% al 4,5%).
Ningún experimento anterior mezclaba temporalidades.

## Qué no se probó

- Oro y bolsa en 4h (sin datos largos gratis). La literatura de trend
  following en futuros diversificados (CTAs) da ~5-10% al año, coherente con
  esto.
- Otros activos (petróleo, divisas). Con lo de arriba no hay motivo para
  esperar otra cosa de las mismas reglas.
