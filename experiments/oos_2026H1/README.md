# OOS 2026-03-05 → 08-22 — ¿hace falta reentrenar con los datos no vistos?

> Fecha: 2026-08-22 · `test_oos.py` · datos vivos de Binance

## La pregunta

Los `.parquet` con los que se congeló `PARAMS_V2` terminan el **2026-03-04**.
Desde entonces hay **170 días** que los parámetros nunca vieron. ¿Hay que
reajustarlos con esos datos?

## Respuesta corta: no hay nada que reentrenar, y tampoco nada que aprender

### 1. No existe ningún modelo entrenado en el camino de ejecución

V2 es `src/v2_engine.py`: reglas puras con parámetros congelados (Donchian-55,
ADX>18, vol>1,2, trailing ATR). **No tiene pesos.** No hay nada que reentrenar.

`short_gbm.pkl` **sí** se carga al arrancar — de ahí la línea
`[V15] BTC: SHORT GBM loaded | threshold=0.6` del log — pero es **código
muerto**: `_generate_signals` comprueba `meta_v2_paper.json` antes, enruta a
`_generate_v2_signal` y hace `continue`. La rama del GBM nunca se alcanza.

### 2. La ventana no vista no contiene información sobre el edge

```
velas 4h en la ventana:        1026
con bull_1d encendido:            0   (0%)
rupturas Donchian-55 brutas:     32
SEÑALES V2:                       0
```

El tramo completo cae dentro del bear de 279 días. V2 no habría abierto ni un
trade. **Un test OOS sobre estos datos devuelve 0 trades = 0 información.**
Refrescar los `.parquet` (punto 4 de "Próximos Pasos") sigue siendo higiene
correcta, pero no va a resolver ninguna duda sobre el edge.

> Corolario incómodo: el proyecto **no puede aprender nada nuevo sobre V2
> hasta que el régimen se dé la vuelta**. Con BTC a 77.000 y EMA50 a 65.900,
> el cruce está a ~24 días vista al precio actual.

## Lo que sí se pudo validar fuera de muestra: el filtro de régimen

Es la única pregunta contestable con estos datos. Se simula el **mismo motor
sin el filtro** y se mira qué habría pasado con los 32 breakouts vetados:

| | |
|---|--:|
| trades que el filtro bloqueó | **21** |
| WR | **19,0%** |
| PF | **0,69** |
| retorno compuesto | **−8,5%** |
| DD | **−20,8%** |
| mejor / peor trade | +8,78% / −2,87% |
| reparto | 10 `A_LONG` · 11 `F_LONG` |
| BTC buy & hold mismo periodo | +6,3% |

**El filtro acertó.** Bloqueó 21 trades que habrían perdido un 8,5% con un
drawdown del 20,8%, en un tramo donde BTC subió un 6,3% — es decir, subió, pero
a tirones, que es exactamente el régimen donde `docs/SESION_2026-08-09.md` ya
había identificado que V2 pierde (*"tendencia limpia vs entrecortada"*).

Esto es una validación OOS genuina, con parámetros congelados, sobre datos
posteriores al ajuste. Es el primer resultado fuera de muestra limpio del
proyecto y **confirma el mecanismo**, no solo el resultado.

## Consecuencia práctica

1. **No reentrenar.** No hay modelo; y reajustar los parámetros con estos 170
   días sería exactamente el patrón de overfitting que `CLAUDE.md` documenta.
2. **Refrescar los `.parquet`** por higiene, sabiendo que no aportará señal.
3. **Borrar o marcar el GBM muerto** — carga 4 `.pkl` al arranque para nada y
   confunde el log (`SHORT GBM loaded` sugiere que ML está activo, y no lo está).
4. La única fuente de información nueva es **el régimen alcista**, y ahí lo que
   se prueba es `A_LONG`, el hueco abierto desde `VERDICTO.md`.
