# Re-validación V15 — ATOM/USDT

> Generado por `revalidate_v15.py` (motor corregido, una posición a la vez). Fecha: 2026-05-19

## VEREDICTO: **REJECT (provisional)**

falla WF con motor corregido

## Declarado en meta_v15.json vs. motor corregido

| Métrica | Declarado (motor viejo) | Motor corregido |
|---------|------------------------|-----------------|
| WF LONG | 12/12 | 4/12 (LONG+SHORT combinado) |
| PF LONG | 7.85 | ver capa B abajo |
| PF SHORT | 14.81 | — |
| DD declarado | 0.029 | — |

## Capa A — Forward-OOS (datos posteriores al cutoff)

⚠️ **No disponible.** Cutoff = 2026-03-25, los datos locales terminan en 2026-03-01 (0 velas post-cutoff). Refrescar datos con `download_new_pairs.py` para activar esta capa.

## Capa B — Walk-forward (motor corregido)

**4/12 folds aprobados** (criterio: n≥5, PF finito ≥1.2, retorno>0)

| Semestre | N | WR | PF | Retorno | OK |
|----------|---|----|----|---------|----|
| 2020-01 | 38 | 18% | 1.64 | +11.7% | ✅ |
| 2020-07 | 46 | 26% | 0.98 | -1.3% | ❌ |
| 2021-01 | 37 | 19% | 0.43 | -19.9% | ❌ |
| 2021-07 | 36 | 11% | 0.39 | -17.9% | ❌ |
| 2022-01 | 44 | 20% | 0.71 | -9.7% | ❌ |
| 2022-07 | 30 | 37% | 2.85 | +29.0% | ✅ |
| 2023-01 | 31 | 39% | 2.17 | +16.6% | ✅ |
| 2023-07 | 33 | 15% | 0.65 | -6.9% | ❌ |
| 2024-01 | 39 | 33% | 1.13 | +2.1% | ❌ |
| 2024-07 | 36 | 44% | 2.24 | +20.4% | ✅ |
| 2025-01 | 45 | 20% | 0.74 | -7.1% | ❌ |
| 2025-07 | 48 | 27% | 0.86 | -4.1% | ❌ |

## Capa C — Bootstrap de significancia

- p-value (retorno ≤ 0 por azar): **0.548** — ❌ NO significativo
- Retorno mediano re-muestreado: -5.8%
- Percentil 5: -53.5%
