# V15 Estrategia B-SHORT: Momentum Breakdown

**Rama**: `v15/momentum-breakout`
**Direccion**: SHORT (equivalente inverso de B-LONG)
**Fecha**: 2026-03-06
**Veredicto**: RECHAZADO

## Setup

| Condicion | LONG | SHORT |
|-----------|------|-------|
| Ruptura | close > high20 | close < low20 |
| Macro | EMA20>EMA50 (BULL) | EMA20<EMA50 (BEAR) |
| Vol | >= 1.8x | >= 1.8x |
| BB | <4.5% (estrecho) | <4.5% (estrecho) |
| SL | bajo min consolidacion | sobre max consolidacion |
| RR | 1.5:1 | 1.5:1 |

## Walk-forward

| Periodo | N | WR | PF | Anual | OK |
|---------|---|----|----|-------|----|\n| 2020-01/06 | 0 | - | - | - | NO |
| 2020-07/12 | 2 | 100.0% | inf | 6% | NO |
| 2021-01/06 | 0 | - | - | - | NO |
| 2021-07/12 | 0 | - | - | - | NO |
| 2022-01/06 | 0 | - | - | - | NO |
| 2022-07/12 | 1 | 0.0% | 0.00 | -6% | NO |
| 2023-01/06 | 0 | - | - | - | NO |
| 2023-07/12 | 1 | 0.0% | 0.00 | -8% | NO |
| 2024-01/06 | 2 | 100.0% | inf | 19% | NO |
| 2024-07/12 | 1 | 100.0% | inf | 11% | NO |
| 2025-01/06 | 2 | 50.0% | 1.19 | 1% | NO |
| 2025-07/12 | 1 | 100.0% | inf | 1% | NO |

**Folds OK**: 0/12

## OOS | N=8 | WR=62.5% | PF=1.85 | ~3%/yr

## Veredicto: RECHAZADO
Criterios: WF>=7/12, WR>=50%, PF>=1.2, >=2.5 trades/mes
