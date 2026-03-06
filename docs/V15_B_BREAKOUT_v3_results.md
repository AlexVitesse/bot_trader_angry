# V15 Estrategia B-v3: Momentum Breakout + Macro Filter

**Rama**: `v15/momentum-breakout`
**Iteracion**: 3 (filtro macro EMA20_1d > EMA50_1d)
**Fecha**: 2026-03-06
**Veredicto**: RECHAZADO

## Hipotesis

v2 mostro que vol<1.8 degradaba calidad. El filtro macro elimina trades en 2025 bajista (WR=35-39% -> no entrar).

## Parametros v3

| Parametro | v1 | v2 | v3 |
|-----------|----|----|----|
| vol_ratio min | 1.8 | 1.3 | 1.8 |
| bb_width max | 4.0% | 5.5% | 4.5% |
| bar_move max | 2.5% | 3.5% | 2.8% |
| adx_prev max | 28 | 30 | 28 |
| macro filter | NO | NO | SI (EMA20>EMA50 diario) |

## Walk-forward

| Periodo | N | WR | PF | Anual | OK |
|---------|---|----|----|-------|----|\n| 2020-01/2020-06 | 0 | - | - | - | NO |
| 2020-07/2020-12 | 5 | 60.0% | 1.41 | 6% | SI |
| 2021-01/2021-06 | 0 | - | - | - | NO |
| 2021-07/2021-12 | 0 | - | - | - | NO |
| 2022-01/2022-06 | 0 | - | - | - | NO |
| 2022-07/2022-12 | 1 | 0.0% | 0.00 | -13% | NO |
| 2023-01/2023-06 | 2 | 100.0% | inf | 77% | NO |
| 2023-07/2023-12 | 10 | 50.0% | 1.76 | 15% | SI |
| 2024-01/2024-06 | 4 | 100.0% | inf | 36% | SI |
| 2024-07/2024-12 | 4 | 75.0% | 102.83 | 102% | SI |
| 2025-01/2025-06 | 5 | 20.0% | 0.54 | -52% | NO |
| 2025-07/2025-12 | 4 | 100.0% | inf | 159% | SI |

**Folds OK**: 5/12

## OOS | N=31 | WR=61.3% | PF=2.87 | ~15%/yr

## Veredicto: RECHAZADO
Criterios: WF>=7/12, WR>=50%, PF>=1.2, >=2.5 trades/mes
