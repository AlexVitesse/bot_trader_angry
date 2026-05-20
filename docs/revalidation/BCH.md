# Re-validación V15 — BCH/USDT

> Generado por `revalidate_v15.py` (motor corregido, una posición a la vez). Fecha: 2026-05-19

## VEREDICTO: **PAPER-ONLY**

WF ok pero falta forward-OOS (refrescar datos)

## Declarado en meta_v15.json vs. motor corregido

| Métrica | Declarado (motor viejo) | Motor corregido |
|---------|------------------------|-----------------|
| WF LONG | 12/12 | 7/12 (LONG+SHORT combinado) |
| PF LONG | 7.86 | ver capa B abajo |
| PF SHORT | 20.13 | — |
| DD declarado | 0.035 | — |

## Capa A — Forward-OOS (datos posteriores al cutoff)

⚠️ **No disponible.** Cutoff = 2026-03-25, los datos locales terminan en 2026-03-25 (5 velas post-cutoff). Refrescar datos con `download_new_pairs.py` para activar esta capa.

## Capa B — Walk-forward (motor corregido)

**7/12 folds aprobados** (criterio: n≥5, PF finito ≥1.2, retorno>0)

| Semestre | N | WR | PF | Retorno | OK |
|----------|---|----|----|---------|----|
| 2020-01 | 34 | 21% | 2.18 | +23.8% | ✅ |
| 2020-07 | 58 | 24% | 0.70 | -9.4% | ❌ |
| 2021-01 | 46 | 17% | 0.55 | -14.3% | ❌ |
| 2021-07 | 37 | 32% | 0.82 | -3.4% | ❌ |
| 2022-01 | 43 | 19% | 0.82 | -5.7% | ❌ |
| 2022-07 | 28 | 36% | 1.96 | +12.1% | ✅ |
| 2023-01 | 39 | 26% | 2.05 | +19.2% | ✅ |
| 2023-07 | 45 | 31% | 1.86 | +19.5% | ✅ |
| 2024-01 | 34 | 47% | 1.64 | +8.4% | ✅ |
| 2024-07 | 46 | 33% | 2.33 | +30.4% | ✅ |
| 2025-01 | 54 | 20% | 0.87 | -4.7% | ❌ |
| 2025-07 | 48 | 35% | 1.72 | +14.6% | ✅ |

## Capa C — Bootstrap de significancia

- p-value (retorno ≤ 0 por azar): **0.083** — ❌ NO significativo
- Retorno mediano re-muestreado: +99.9%
- Percentil 5: -12.1%
