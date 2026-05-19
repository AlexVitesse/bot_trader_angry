# Re-validación V15 — ALGO/USDT

> Generado por `revalidate_v15.py` (motor corregido, una posición a la vez). Fecha: 2026-05-19

## VEREDICTO: **REJECT (provisional)**

falla WF con motor corregido

## Declarado en meta_v15.json vs. motor corregido

| Métrica | Declarado (motor viejo) | Motor corregido |
|---------|------------------------|-----------------|
| WF LONG | 12/12 | 4/12 (LONG+SHORT combinado) |
| PF LONG | 6.62 | ver capa B abajo |
| PF SHORT | 15.36 | — |
| DD declarado | 0.03 | — |

## Capa A — Forward-OOS (datos posteriores al cutoff)

⚠️ **No disponible.** Cutoff = 2026-03-25, los datos locales terminan en 2026-03-01 (0 velas post-cutoff). Refrescar datos con `download_new_pairs.py` para activar esta capa.

## Capa B — Walk-forward (motor corregido)

**4/12 folds aprobados** (criterio: n≥5, PF finito ≥1.2, retorno>0)

| Semestre | N | WR | PF | Retorno | OK |
|----------|---|----|----|---------|----|
| 2020-01 | 29 | 24% | 0.79 | -4.2% | ❌ |
| 2020-07 | 30 | 20% | 0.93 | -2.0% | ❌ |
| 2021-01 | 41 | 12% | 0.48 | -19.0% | ❌ |
| 2021-07 | 34 | 24% | 0.70 | -6.6% | ❌ |
| 2022-01 | 44 | 16% | 0.50 | -17.3% | ❌ |
| 2022-07 | 32 | 34% | 2.28 | +20.3% | ✅ |
| 2023-01 | 25 | 40% | 1.31 | +3.2% | ✅ |
| 2023-07 | 33 | 15% | 0.65 | -6.7% | ❌ |
| 2024-01 | 40 | 35% | 1.54 | +10.3% | ✅ |
| 2024-07 | 48 | 29% | 0.69 | -8.0% | ❌ |
| 2025-01 | 43 | 23% | 1.05 | +0.6% | ❌ |
| 2025-07 | 47 | 32% | 1.70 | +17.4% | ✅ |

## Capa C — Bootstrap de significancia

- p-value (retorno ≤ 0 por azar): **0.575** — ❌ NO significativo
- Retorno mediano re-muestreado: -7.7%
- Percentil 5: -48.4%
