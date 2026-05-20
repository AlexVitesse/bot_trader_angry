# Re-validación V15 — ADA/USDT

> Generado por `revalidate_v15.py` (motor corregido, una posición a la vez). Fecha: 2026-05-19

## VEREDICTO: **REJECT (provisional)**

falla WF con motor corregido

## Declarado en meta_v15.json vs. motor corregido

| Métrica | Declarado (motor viejo) | Motor corregido |
|---------|------------------------|-----------------|
| WF LONG | 10/12 | 5/12 (LONG+SHORT combinado) |
| PF LONG | 2.86 | ver capa B abajo |
| PF SHORT | 13.51 | — |
| DD declarado | 0.071 | — |

## Capa A — Forward-OOS (datos posteriores al cutoff)

⚠️ **No disponible.** Cutoff = 2026-03-24, los datos locales terminan en 2026-02-27 (0 velas post-cutoff). Refrescar datos con `download_new_pairs.py` para activar esta capa.

## Capa B — Walk-forward (motor corregido)

**5/12 folds aprobados** (criterio: n≥5, PF finito ≥1.2, retorno>0)

| Semestre | N | WR | PF | Retorno | OK |
|----------|---|----|----|---------|----|
| 2020-01 | 27 | 26% | 1.45 | +7.0% | ✅ |
| 2020-07 | 55 | 33% | 1.93 | +26.9% | ✅ |
| 2021-01 | 38 | 26% | 1.04 | +0.4% | ❌ |
| 2021-07 | 49 | 31% | 1.05 | +0.6% | ❌ |
| 2022-01 | 46 | 11% | 0.24 | -26.0% | ❌ |
| 2022-07 | 29 | 28% | 0.73 | -4.5% | ❌ |
| 2023-01 | 31 | 23% | 0.94 | -1.2% | ❌ |
| 2023-07 | 34 | 24% | 0.77 | -4.3% | ❌ |
| 2024-01 | 42 | 36% | 0.95 | -1.2% | ❌ |
| 2024-07 | 52 | 38% | 1.24 | +5.5% | ✅ |
| 2025-01 | 43 | 16% | 1.72 | +16.4% | ✅ |
| 2025-07 | 51 | 27% | 1.22 | +5.1% | ✅ |

## Capa C — Bootstrap de significancia

- p-value (retorno ≤ 0 por azar): **0.326** — ❌ NO significativo
- Retorno mediano re-muestreado: +22.8%
- Percentil 5: -40.0%
