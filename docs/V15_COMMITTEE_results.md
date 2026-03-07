# V15 Expert Committee — Backtest Combinado

**Fecha**: 2026-03-06
**Veredicto**: APROBADO

## Componentes

| Regimen | Estrategia | Descripcion |
|---------|-----------|-------------|
| BULL | Breakout B LONG | vol>=1.8, BB estrecho, close>high20 |
| BEAR | SHORT ML (GBM) | Entrenado solo en BEAR, threshold=0.55 |
| RANGE | No operar | Esperar |

## Walk-forward

| Periodo | N | L | S | WR | PF | Anual | OK |
|---------|---|---|---|----|----|-------|----|\n| 2020-01/06 | 58 | 31 | 27 | 39.7% | 1.11 | 17% | SI |
| 2020-07/12 | 40 | 40 | 0 | 67.5% | 2.94 | 106% | SI |
| 2021-01/06 | 46 | 26 | 20 | 54.3% | 2.07 | 129% | SI |
| 2021-07/12 | 18 | 18 | 0 | 27.8% | 0.61 | -18% | NO |
| 2022-01/06 | 21 | 5 | 16 | 52.4% | 2.70 | 76% | SI |
| 2022-07/12 | 71 | 1 | 70 | 69.0% | 3.24 | 127% | SI |
| 2023-01/06 | 53 | 44 | 9 | 32.1% | 0.73 | -31% | NO |
| 2023-07/12 | 57 | 56 | 1 | 49.1% | 1.31 | 25% | SI |
| 2024-01/06 | 34 | 34 | 0 | 52.9% | 1.78 | 38% | SI |
| 2024-07/12 | 35 | 31 | 4 | 51.4% | 1.67 | 36% | SI |
| 2025-01/06 | 44 | 38 | 6 | 27.3% | 0.40 | -63% | NO |
| 2025-07/12 | 29 | 21 | 8 | 41.4% | 0.76 | -14% | NO |

**Folds OK**: 8/12

## OOS | N=344 | WR=48.0% | PF=1.35 | 7.3t/m

### LONG: N=230 | WR=45.7% | PF=1.20
### SHORT: N=114 | WR=52.6% | PF=1.64

## Equity: $1000 -> $7116 (611.6%)

## Veredicto: APROBADO
Criterios: WF>=7/12, WR>=38%, PF>=1.2, >=1.5 trades/mes
