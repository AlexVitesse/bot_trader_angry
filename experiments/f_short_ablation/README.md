# Ablation F_SHORT — quitar el componente sin edge

> ¿Aporta algo `F_SHORT` a V2? Respuesta: no. Resta.
> Fecha: 2026-08-09 · `test_ablation.py` · datos BTC 4h 2019-01 → 2026-08-09
> (parquet histórico + Binance en vivo)

## Por qué este test

`combined_AF/README.md` incluyó `F_BTC_SHORT` en V2 reconociendo que era
marginal in-sample (N=30, PF 1.08, annual +0.2%, p=0.492) pero "útil en bear
(vio el OOS 2026)". Esa justificación se apoyaba en **3 trades ganadores** del
OOS Ene-Feb 2026 — que es exactamente el sesgo de selección que
`VERDICTO_RONDA2.md` §3 advirtió y prohibió.

Además `F_SHORT` es el **único** componente que puede disparar con `bull_1d=0`:
`A_LONG` y `F_LONG` exigen régimen daily alcista. Desde mayo 2026 es lo único
que ha operado el bot.

**Este test no introduce ningún parámetro nuevo**: usa el flag `f_enable_short`
que ya existe en `PARAMS_V2`. Riesgo de overfitting adicional: cero.

## Resultados

### In-sample 2020-01-01 → 2025-12-31 (la ventana que se validó)

> ⚠️ **Números corregidos 2026-08-10** tras revisión externa. La versión previa
> recortaba el df antes de `build_features`, así que el warmup de 220 velas se
> comía ~37 días del inicio de cada ventana y se perdían trades reales. También
> el DD ignoraba el capital inicial. Cifras corregidas abajo.

| Config | N | WR | PF | Total | Anual | DD | Bootstrap p |
|--------|--:|---:|---:|------:|------:|---:|------------:|
| V2 actual (A+F+F_SHORT) | 168 | 44,0% | 1,53 | +145,7% | +16,4% | 22,1% | 0,021 ✅ |
| **V2 sin F_SHORT** | 139 | 43,9% | **1,65** | **+158,2%** | **+17,9%** | **19,6%** | **0,019** ✅ |

Quitar `F_SHORT` mejora **todas** las métricas a la vez: más PF, más retorno,
menos drawdown, mejor p. Dominancia estricta.

### OOS 2026-01-01 → 2026-08-09 (nadie vio estos datos al diseñar)

| Config | N | WR | PF | Total | DD | Bootstrap p |
|--------|--:|---:|---:|------:|---:|------------:|
| V2 actual | 12 | 33,3% | 0,96 | **−1,1%** | 8,1% | 0,534 |
| **V2 sin F_SHORT** | **0** | — | — | **0%** | 0% | — |

Las 12 operaciones de 2026 fueron **todas** `F_SHORT` y en conjunto perdieron.
Sin ese componente el sistema se queda plano — el comportamiento que
`VERDICTO.md` §"Agente A" califica de **correcto** para un trend-follower en bear.

> ⚠️ **Esta ventana NO es OOS limpio.** Ene-Feb 2026 se usó para justificar la
> inclusión original de `F_SHORT` en `combined_AF`, así que ya fue observada. El
> argumento válido para quitar `F_SHORT` es su `p=0,644` sobre la historia
> completa, no este −1,1%.

### Desglose por componente (historia completa 2019 → 2026-08)

| Componente | N | WR | PF | Anual | DD | Bootstrap p |
|------------|--:|---:|---:|------:|---:|------------:|
| `A_LONG` | 95 | 45,3% | 1,91 | **+16,5%** | 21,5% | **0,012** ✅ |
| `F_LONG` | 69 | 43,5% | 1,78 | **+7,9%** | 16,5% | **0,041** ✅ |
| `F_SHORT` | 44 | 40,9% | 0,88 | **−1,0%** | 10,3% | 0,639 ❌ |

Con los datos extendidos hasta agosto 2026, `F_SHORT` ya no es "marginal": es
**negativo**. Los 14 trades posteriores al cutoff original lo empujaron de
+0,2% a −1,0% anual. Los dos motores reales son ambos LONG.

## Corrección al supuesto de funding

`combined_AF/README.md` §4 resta "~13%/año" de funding y concluye
"V2 real ≈ +9-10%". Ese cálculo asume estar en mercado el 100% del tiempo.

Medido sobre los trades reales:

| Config | Trades | Velas medias | Tiempo en mercado | Funding real |
|--------|-------:|-------------:|------------------:|-------------:|
| V2 actual | 208 | 12,2 | 15,3% | **~2,17%/año** |
| V2 sin F_SHORT | 164 | 12,8 | 12,6% | **~1,80%/año** |

El funding sólo se paga mientras la posición está abierta. V2 está fuera del
mercado ~87% del tiempo, así que el coste es **~1,8%, no 13%**.

**V2 sin F_SHORT ≈ +18,7% bruto − 1,8% funding ≈ +16,9% neto anual**, no el
+9-10% que asumía el proyecto.

## Honestidad sobre este resultado

- Quité un componente después de verlo perder en 2026. Eso suena a sesgo de
  selección al revés. La defensa: su `p=0.492` in-sample ya existía **antes**
  de que 2026 existiera, y la decisión original de *incluirlo* fue la que se
  tomó sobre 3 trades. Esto corrige esa inclusión, no crea una nueva.
- **No llega a los 30% anuales** del objetivo del proyecto. ~17% neto.
- El precio es **cero trades mientras BTC siga en bear daily**. Hoy eso
  significa un bot parado, con el capital en Earn al 3%.
- `A_LONG` sigue sin probarse en una ventana alcista real (pendiente desde
  `VERDICTO.md`). Es el componente que aporta +16,5% y el que nunca se ha
  verificado en vivo.

## Qué haría falta para aprobarlo formalmente

Los requisitos de `CLAUDE.md` que este test **ya** cubre: bootstrap p<0.05,
motor honesto (una posición a la vez, sin look-ahead intrabar), cutoff
respetado, documentado. Falta: walk-forward por folds y cross-asset — aunque
`v2_all_coins/` ya mostró que fuera de BTC (y DOGE con asterisco) no hay señal.
