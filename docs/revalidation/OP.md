# Re-validación V15 — OP/USDT

> Generado por `revalidate_v15.py` (motor corregido, una posición a la vez). Fecha: 2026-05-19

## VEREDICTO: **REJECT (provisional)**

falla WF con motor corregido

## Declarado en meta_v15.json vs. motor corregido

| Métrica | Declarado (motor viejo) | Motor corregido |
|---------|------------------------|-----------------|
| WF LONG | 7/7 | 4/7 (LONG+SHORT combinado) |
| PF LONG | 14.91 | ver capa B abajo |
| PF SHORT | 13.13 | — |
| DD declarado | 0.014 | — |

## Capa A — Forward-OOS (datos posteriores al cutoff)

⚠️ **No disponible.** Cutoff = 2026-03-25, los datos locales terminan en 2026-03-25 (5 velas post-cutoff). Refrescar datos con `download_new_pairs.py` para activar esta capa.

## Capa B — Walk-forward (motor corregido)

**4/7 folds aprobados** (criterio: n≥5, PF finito ≥1.2, retorno>0)

| Semestre | N | WR | PF | Retorno | OK |
|----------|---|----|----|---------|----|
| 2020-01 | — | — | — | sin datos | — |
| 2020-07 | — | — | — | sin datos | — |
| 2021-01 | — | — | — | sin datos | — |
| 2021-07 | — | — | — | sin datos | — |
| 2022-01 | — | — | — | sin datos | — |
| 2022-07 | 20 | 30% | 2.56 | +24.8% | ✅ |
| 2023-01 | 38 | 26% | 0.95 | -1.9% | ❌ |
| 2023-07 | 33 | 18% | 0.37 | -11.3% | ❌ |
| 2024-01 | 45 | 13% | 0.39 | -17.8% | ❌ |
| 2024-07 | 44 | 30% | 1.28 | +6.9% | ✅ |
| 2025-01 | 32 | 19% | 1.20 | +3.6% | ✅ |
| 2025-07 | 51 | 33% | 1.97 | +27.2% | ✅ |

## Capa C — Bootstrap de significancia

- p-value (retorno ≤ 0 por azar): **0.199** — ❌ NO significativo
- Retorno mediano re-muestreado: +34.9%
- Percentil 5: -24.5%
