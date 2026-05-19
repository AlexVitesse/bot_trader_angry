# Re-validación V15 — AVAX/USDT

> Generado por `revalidate_v15.py` (motor corregido, una posición a la vez). Fecha: 2026-05-19

## VEREDICTO: **REJECT (provisional)**

falla WF con motor corregido

## Declarado en meta_v15.json vs. motor corregido

| Métrica | Declarado (motor viejo) | Motor corregido |
|---------|------------------------|-----------------|
| WF LONG | 10/11 | 4/11 (LONG+SHORT combinado) |
| PF LONG | 8.33 | ver capa B abajo |
| PF SHORT | 15.43 | — |
| DD declarado | 0.031 | — |

## Capa A — Forward-OOS (datos posteriores al cutoff)

⚠️ **No disponible.** Cutoff = 2026-03-25, los datos locales terminan en 2026-02-27 (0 velas post-cutoff). Refrescar datos con `download_new_pairs.py` para activar esta capa.

## Capa B — Walk-forward (motor corregido)

**4/11 folds aprobados** (criterio: n≥5, PF finito ≥1.2, retorno>0)

| Semestre | N | WR | PF | Retorno | OK |
|----------|---|----|----|---------|----|
| 2020-01 | — | — | — | sin datos | — |
| 2020-07 | 5 | 20% | 1.69 | +3.0% | ✅ |
| 2021-01 | 34 | 6% | 0.47 | -19.6% | ❌ |
| 2021-07 | 35 | 31% | 1.16 | +3.3% | ❌ |
| 2022-01 | 47 | 15% | 0.66 | -13.7% | ❌ |
| 2022-07 | 27 | 15% | 0.45 | -9.6% | ❌ |
| 2023-01 | 31 | 16% | 0.58 | -7.6% | ❌ |
| 2023-07 | 29 | 17% | 0.59 | -7.4% | ❌ |
| 2024-01 | 44 | 41% | 1.78 | +15.6% | ✅ |
| 2024-07 | 49 | 31% | 1.23 | +5.4% | ✅ |
| 2025-01 | 50 | 20% | 0.83 | -5.6% | ❌ |
| 2025-07 | 50 | 34% | 1.60 | +14.7% | ✅ |

## Capa C — Bootstrap de significancia

- p-value (retorno ≤ 0 por azar): **0.713** — ❌ NO significativo
- Retorno mediano re-muestreado: -18.9%
- Percentil 5: -54.7%
