# Parámetros de salida de V2 — resultado NEGATIVO

> Fecha: 2026-08-09 · `test_ceiling.py`
> **Conclusión: no hay nada que ganar tocando el trailing de V2. No repetir.**

## Hipótesis

De `experiments/conviction/`: `atr_pct` era la única feature con correlación
estable contra el PnL (rho −0,14 / −0,16 en ambas mitades). Explicación
propuesta: el trail es `clip(atr_pct × mult, floor, ceiling)`, así que en alta
volatilidad el techo deja el stop proporcionalmente más apretado de lo que la
volatilidad exige → salta antes de tiempo.

## 1. El mecanismo no existe

| estado del trail | n | % | WR | PnL medio | PF | estrangulamiento |
|---|--:|--:|--:|--:|--:|--:|
| SUELO (stop ensanchado) | 32 | 20% | 53,1% | +1,27% | **2,58** | −23,7% |
| libre | 113 | 69% | 41,6% | +0,94% | 1,76 | 0% |
| TECHO (stop apretado) | 19 | 12% | 47,4% | +1,17% | 1,72 | +17,7% |

Los trades con el techo activo **no rinden peor** que los libres — rinden algo
mejor. La hipótesis del estrangulamiento queda falsada por los datos.

## 2. Barrido del techo: el valor actual ya es óptimo

| techo A/F | n | PF | CAGR | DD |
|---|--:|--:|--:|--:|
| **6% / 5,5% (actual)** | 164 | **1,86** | **+21,8%** | **19,6%** |
| 8% / 7,3% | 163 | 1,80 | +20,6% | 27,1% |
| 10% / 9,2% | 162 | 1,84 | +21,4% | 22,6% |
| 12% / 11% | 162 | 1,83 | +21,3% | 23,5% |
| sin techo | 162 | 1,83 | +21,3% | 23,5% |

El techo frozen gana en CAGR, PF y DD a la vez.

## 3. Barrido del suelo: gana in-sample, muere en walk-forward

El grupo SUELO (PF 2,58) sugería que ensanchar ayuda. Probado:

| suelo A/F | n | PF | CAGR | DD |
|---|--:|--:|--:|--:|
| 2,5% / 2,0% (actual) | 164 | 1,86 | +21,8% | 19,6% |
| 3,5% / 3,0% | 153 | 1,91 | **+23,0%** | 19,6% |
| 4,5% / 4,0% | 145 | 1,62 | +16,7% | 20,4% |
| 5,5% / 5,0% | 132 | 1,78 | +21,1% | 23,6% |

Curva **no monótona** (3,5 mejor, 4,5 peor, 5,5 mejor) = firma de ruido.
Walk-forward 12 folds del 3,5% contra el actual:

```
actual   7/12 folds   mediana +2.7%   suma +117.3%
nuevo    7/12 folds   mediana +1.2%   suma +114.1%
```

La ventaja in-sample se invierte fuera de muestra. **Rechazado.**

## Lectura para el proyecto

Tres búsquedas de parámetros sobre los mismos 164 trades en esta sesión
(filtro ADX, techo del trail, suelo del trail). Las tres murieron en
walk-forward. Eso es **buena noticia**: `PARAMS_V2` está en un óptimo local
razonable y no está sobreajustado a base de tunear.

También acota el hallazgo histórico *"el edge está en la salida"* (V7, comité
SOL/ADA): se refiere a **usar trailing stop en vez de TP/SL fijo** — que V2 ya
hace — no a afinar su anchura. Ahí no queda margen.

**Corolario: el retorno de V2 no va a subir tocando parámetros.** Sube
quitando el componente sin edge (`f_short_ablation/`), corrigiendo el sizing
y aceptando más drawdown vía apalancamiento (`aggressive/`).
