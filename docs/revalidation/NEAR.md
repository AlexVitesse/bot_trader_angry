# Re-validación V15 — NEAR/USDT

> Generado por `revalidate_v15.py` (motor corregido, una posición a la vez). Fecha: 2026-05-19

## VEREDICTO: **REJECT (provisional)**

falla WF con motor corregido

## Declarado en meta_v15.json vs. motor corregido

| Métrica | Declarado (motor viejo) | Motor corregido |
|---------|------------------------|-----------------|
| WF LONG | 11/11 | 5/11 (LONG+SHORT combinado) |
| PF LONG | 11.05 | ver capa B abajo |
| PF SHORT | 16.8 | — |
| DD declarado | 0.025 | — |

## Capa A — Forward-OOS (datos posteriores al cutoff)

⚠️ **No disponible.** Cutoff = 2026-03-25, los datos locales terminan en 2026-02-27 (0 velas post-cutoff). Refrescar datos con `download_new_pairs.py` para activar esta capa.

## Capa B — Walk-forward (motor corregido)

**5/11 folds aprobados** (criterio: n≥5, PF finito ≥1.2, retorno>0)

| Semestre | N | WR | PF | Retorno | OK |
|----------|---|----|----|---------|----|
| 2020-01 | — | — | — | sin datos | — |
| 2020-07 | 10 | 50% | 3.07 | +9.1% | ✅ |
| 2021-01 | 36 | 14% | 0.53 | -18.4% | ❌ |
| 2021-07 | 35 | 17% | 0.70 | -8.9% | ❌ |
| 2022-01 | 42 | 26% | 1.46 | +14.0% | ✅ |
| 2022-07 | 26 | 27% | 1.25 | +3.8% | ✅ |
| 2023-01 | 32 | 22% | 1.23 | +3.8% | ✅ |
| 2023-07 | 22 | 5% | 0.03 | -13.7% | ❌ |
| 2024-01 | 38 | 24% | 0.91 | -2.8% | ❌ |
| 2024-07 | 51 | 24% | 0.96 | -2.0% | ❌ |
| 2025-01 | 44 | 23% | 1.12 | +2.6% | ❌ |
| 2025-07 | 44 | 30% | 1.41 | +8.9% | ✅ |

## Capa C — Bootstrap de significancia

- p-value (retorno ≤ 0 por azar): **0.440** — ❌ NO significativo
- Retorno mediano re-muestreado: +6.4%
- Percentil 5: -44.0%
