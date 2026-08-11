# Multi-temporalidad: 4h + 1d como dos sleeves del mismo motor

> Fecha: 2026-08-09 · `test_daily.py` · BTC 2019-01 → 2026-08 (7,6 años)
> **Estado: PROMETEDOR, NO VALIDADO.** Falta walk-forward de la combinación.

## Origen

Los `PARAMS_V2` (Donchian-55, EMA50/200, ATR×2.5) son los del sistema Turtle,
diseñado para velas **diarias**. V2 los aplica sobre 4h, donde Donchian-55 son
9 días en vez de 55 y el hold medio sale de 2,1 días. Hipótesis: V2 podría
estar corriendo en la temporalidad equivocada.

## 1. Primer intento — inválido

Correr V2 tal cual sobre velas diarias da +6,7% CAGR y p=0,086. Parecía que el
diario no funcionaba. **El test estaba mal**: el techo del trailing (6%) está
calibrado para 4h.

| | ATR% mediano | el techo 6% muerde en |
|---|--:|--:|
| 4h | 1,55% | 17% de las velas |
| 1d | 4,18% | **95% de las velas** |

Ratio de ATR medido: **2,69x**. Aplicar un tope de 4h sobre datos diarios
estrangula prácticamente todos los trades.

## 2. Con el trail escalado por el ratio de ATR

| trail 1d (suelo/techo) | tr/año | hold | WR | PF | CAGR | DD | p |
|---|--:|--:|--:|--:|--:|--:|--:|
| 2,5% / 6% (el de 4h) | 5 | 6,5d | 48,6% | 1,91 | +6,8% | 15,2% | 0,086 |
| **6% / 15% (escalado)** | 4 | **14,0d** | **51,9%** | **2,42** | +11,7% | 18,3% | **0,045** ✅ |
| 6% / 25% | 4 | 14,0d | 51,9% | 2,36 | +11,1% | 18,9% | 0,062 |
| sin límites | 4 | 14,0d | 51,9% | 2,37 | +11,1% | 18,9% | 0,061 |

El hold pasa de 6,5 a 14 días — lo esperable de un Turtle. **No es un
parámetro de filo de navaja**: 15%, 25% y sin techo dan resultados casi
idénticos, lo que indica que lo relevante era *no* aplicar un tope de 4h, no
acertar un número.

## 3. Combinación de sleeves

| sleeve | n | tr/año | WR | PF | CAGR | DD | bootstrap p |
|---|--:|--:|--:|--:|--:|--:|--:|
| solo 4h | 164 | 22 | 44,5% | 1,86 | +21,8% | 19,6% | 0,003 |
| solo 1d escalado | 27 | 4 | 51,9% | 2,42 | +11,7% | 18,3% | 0,049 |
| 4h + 1d (sleeve 100%) | 191 | 25 | 45,5% | **2,02** | +36,1% | 28,2% | 0,002 |
| **4h + 1d (50/50)** | 191 | 25 | 45,5% | **2,02** | +18,1% | **14,5%** | **0,000** |

**Tiempo en mercado: 4h 13%, 1d 14%, suma 26%.** Apenas se solapan → capturan
movimientos distintos. Es el mismo mecanismo por el que A+F superaba a A y a F
por separado (ver `combined_AF/README.md`).

El resultado clave no es el CAGR sino el **DD del 50/50: 14,5% con p=0,000**,
por debajo del de 4h solo (19,6%) y con mejor significancia. Bajar el DD es lo
que compra margen para subir el sizing.

## Advertencias — leer antes de creerse nada

1. **El +36,1% del sleeve 100% es apalancamiento encubierto.** Asume capital
   completo en ambos sleeves simultáneamente, y sí se solapan. El número
   honesto sin apalancar es el 50/50: **+18,1%**.
2. **El diario tiene 27 trades y p=0,049.** Justo en el filo del corte.
3. **El escalado 2,69x lo eligió el analista** midiendo el ratio de ATR. Es
   principiado, no ajustado, y la insensibilidad al techo lo respalda — pero
   es un parámetro que se tocó.
4. **FALTA EL WALK-FORWARD de la combinación.** En esta misma sesión el WF
   tumbó tres candidatos que parecían buenos in-sample (filtro ADX, techo del
   trail, suelo del trail). Es el gate, y este resultado no ha pasado por él.

## Qué implicaría si sobrevive al walk-forward

- Dos instancias del mismo motor, no dos modelos. Sin parámetros nuevos salvo
  el escalado del trail por ATR.
- 25 trades/año y 26% de tiempo en mercado, contra 22 y 13%. Más actividad
  **con mejores métricas** — lo contrario de lo que ocurre al relajar filtros
  (ver `frequency_sweep/`).
- El capital ocioso baja del 87% al 74%, lo que reduce la dependencia del
  yield manager.
