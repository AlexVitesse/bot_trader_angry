# V15 Estrategia B-v2: Momentum Breakout (Optimizado)

**Rama**: `v15/momentum-breakout`
**Iteracion**: 2 (optimizacion de frecuencia)
**Fecha**: 2026-03-06
**Veredicto**: RECHAZADO

## Cambios vs v1

| Parametro | v1 | v2 |
|-----------|----|----|----|
| vol_ratio min | 1.8 | 1.3 |
| bb_width max | 4.0% | 5.5% |
| bar_move max | 2.5% | 3.5% |
| adx_prev max | 28 | 30 |

## Comparacion v1 vs v2

| Metrica | v1 | v2 |
|---------|----|----|----|
| N OOS | 49 | 95 |
| WR | 59.2% | 49.5% |
| PF | 1.90 | 1.37 |
| Trades/mes | 1.2 | 2.4 |
| WF folds OK | 4/12 | 5/12 |

## Walk-forward

| Periodo | N | WR | PF | Anual | OK |
|---------|---|----|----|-------|----|
| 2020-01/2020-06 | 3 | 66.7% | 3.36 | 20% | SI |
| 2020-07/2020-12 | 10 | 60.0% | 1.98 | 22% | SI |
| 2021-01/2021-06 | 1 | 0.0% | 0.00 | -43% | NO |
| 2021-07/2021-12 | 1 | 0.0% | 0.00 | -44% | NO |
| 2022-01/2022-06 | 0 | - | - | - | NO |
| 2022-07/2022-12 | 6 | 83.3% | 16.51 | 120% | SI |
| 2023-01/2023-06 | 8 | 50.0% | 1.06 | 2% | NO |
| 2023-07/2023-12 | 19 | 42.1% | 0.86 | -6% | NO |
| 2024-01/2024-06 | 8 | 75.0% | 3.44 | 39% | SI |
| 2024-07/2024-12 | 7 | 57.1% | 3.68 | 41% | SI |
| 2025-01/2025-06 | 17 | 35.3% | 0.67 | -28% | NO |
| 2025-07/2025-12 | 23 | 39.1% | 0.80 | -10% | NO |

**Folds OK**: 5/12

## OOS | N=95 | WR=49.5% | PF=1.37 | ~11%/yr

## Veredicto: RECHAZADO
Criterios: WF>=7/12, WR>=50%, PF>=1.2, >=2.5 trades/mes
