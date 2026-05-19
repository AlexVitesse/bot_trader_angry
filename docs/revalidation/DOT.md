# Re-validación V15 — DOT/USDT

> Generado por `revalidate_v15.py` (motor corregido, una posición a la vez). Fecha: 2026-05-19

## VEREDICTO: **REJECT (provisional)**

falla WF con motor corregido

## Declarado en meta_v15.json vs. motor corregido

| Métrica | Declarado (motor viejo) | Motor corregido |
|---------|------------------------|-----------------|
| WF LONG | 11/11 | 4/11 (LONG+SHORT combinado) |
| PF LONG | 7.62 | ver capa B abajo |
| PF SHORT | 13.9 | — |
| DD declarado | 0.027 | — |

## Capa A — Forward-OOS (datos posteriores al cutoff)

⚠️ **No disponible.** Cutoff = 2026-03-25, los datos locales terminan en 2026-02-27 (0 velas post-cutoff). Refrescar datos con `download_new_pairs.py` para activar esta capa.

## Capa B — Walk-forward (motor corregido)

**4/11 folds aprobados** (criterio: n≥5, PF finito ≥1.2, retorno>0)

| Semestre | N | WR | PF | Retorno | OK |
|----------|---|----|----|---------|----|
| 2020-01 | — | — | — | sin datos | — |
| 2020-07 | 38 | 26% | 1.46 | +9.0% | ✅ |
| 2021-01 | 47 | 19% | 0.80 | -9.4% | ❌ |
| 2021-07 | 41 | 29% | 1.19 | +4.2% | ❌ |
| 2022-01 | 43 | 12% | 0.69 | -10.9% | ❌ |
| 2022-07 | 25 | 44% | 1.91 | +9.5% | ✅ |
| 2023-01 | 38 | 26% | 0.62 | -7.1% | ❌ |
| 2023-07 | 30 | 33% | 1.07 | +0.7% | ❌ |
| 2024-01 | 41 | 39% | 1.21 | +3.5% | ✅ |
| 2024-07 | 40 | 20% | 0.78 | -5.8% | ❌ |
| 2025-01 | 45 | 16% | 0.69 | -9.0% | ❌ |
| 2025-07 | 51 | 31% | 1.47 | +11.6% | ✅ |

## Capa C — Bootstrap de significancia

- p-value (retorno ≤ 0 por azar): **0.560** — ❌ NO significativo
- Retorno mediano re-muestreado: -5.6%
- Percentil 5: -48.0%
