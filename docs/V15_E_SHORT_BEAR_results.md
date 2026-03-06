# V15 Estrategia E: SHORT Bear Market (Failed Bounce)

**Rama**: `v15/momentum-breakout`
**Fecha**: 2026-03-06
**Veredicto**: RECHAZADO

## Por que B-SHORT fallo

El patron 'BB estrecho + ruptura bajo low20' es de bull market.
En 2022-H1 (bear -75%), solo 10 barras cumplian todas las condiciones.

## Setup E: Failed Bounce

| Condicion | Valor |
|-----------|-------|
| Macro | EMA20_1d < EMA50_1d (BEAR) |
| 4H trend | EMA20_4h < EMA50_4h |
| RSI reciente | subio a 42-72 (hubo rebote) |
| RSI actual | 30-62 (rechazo sin oversold extremo) |
| Entry | close < EMA20_4h (rechazado en resistencia) |
| Bar move | < 2.5% |
| SL | max(high[-3:]) * 1.003 |
| TP | SL_pct * 2.0 (RR 2.0:1) |

## Walk-forward

| Periodo | N | WR | PF | Anual | OK |
|---------|---|----|----|-------|----|\n| 2020-01/06 | 3 | 0.0% | 0.00 | -17% | NO |
| 2020-07/12 | 5 | 0.0% | 0.00 | -20% | NO |
| 2021-01/06 | 2 | 100.0% | inf | 25% | NO |
| 2021-07/12 | 9 | 44.4% | 1.32 | 8% | SI |
| 2022-01/06 | 31 | 29.0% | 0.63 | -36% | NO |
| 2022-07/12 | 41 | 41.5% | 0.89 | -10% | NO |
| 2023-01/06 | 10 | 30.0% | 0.46 | -13% | NO |
| 2023-07/12 | 9 | 33.3% | 0.58 | -8% | NO |
| 2024-01/06 | 6 | 0.0% | 0.00 | -25% | NO |
| 2024-07/12 | 7 | 14.3% | 0.31 | -21% | NO |
| 2025-01/06 | 8 | 12.5% | 0.32 | -24% | NO |
| 2025-07/12 | 21 | 23.8% | 0.29 | -48% | NO |

**Folds OK**: 1/12

## OOS | N=134 | WR=29.9% | PF=0.55 | ~-21%/yr

## Veredicto: RECHAZADO
