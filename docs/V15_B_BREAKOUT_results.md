# V15 Estrategia B: Momentum Breakout

**Rama**: `v15/momentum-breakout`
**Fecha**: 2026-03-05
**Veredicto**: RECHAZADO

## Lo nuevo vs V14
- V14 BREAKOUT_UP requeria bb_pct>1.0 en VOLATILE -> casi nunca disparaba
- Esta detecta consolidacion real (BB estrecho 3/5 bars) + ruptura con volumen
- SL bajo minimo de consolidacion (logica real)

## Walk-forward

| Periodo | N | WR | PF | Anual | OK |
|---------|---|----|----|-------|----|
| 2020-01/2020-06 | 0 | - | - | - | NO |
| 2020-07/2020-12 | 3 | 33.3% | 0.58 | -20% | NO |
| 2021-01/2021-06 | 0 | - | - | - | NO |
| 2021-07/2021-12 | 0 | - | - | - | NO |
| 2022-01/2022-06 | 0 | - | - | - | NO |
| 2022-07/2022-12 | 4 | 75.0% | 9.23 | 66% | SI |
| 2023-01/2023-06 | 2 | 50.0% | 1.03 | 1% | NO |
| 2023-07/2023-12 | 14 | 50.0% | 1.39 | 10% | SI |
| 2024-01/2024-06 | 4 | 100.0% | inf | 36% | SI |
| 2024-07/2024-12 | 3 | 66.7% | 73.47 | 73% | SI |
| 2025-01/2025-06 | 8 | 37.5% | 0.53 | -25% | NO |
| 2025-07/2025-12 | 9 | 55.6% | 1.09 | 2% | NO |

**Folds OK**: 4/12

## OOS | N=49 | WR=59.2% | PF=1.90 | ~11%/yr

## Veredicto: RECHAZADO
