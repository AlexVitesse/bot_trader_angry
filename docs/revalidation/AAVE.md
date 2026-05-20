# Re-validación V15 — AAVE/USDT

> Generado por `revalidate_v15.py` (motor corregido, una posición a la vez). Fecha: 2026-05-19

## VEREDICTO: **REJECT (provisional)**

falla WF con motor corregido

## Declarado en meta_v15.json vs. motor corregido

| Métrica | Declarado (motor viejo) | Motor corregido |
|---------|------------------------|-----------------|
| WF LONG | 11/11 | 4/11 (LONG+SHORT combinado) |
| PF LONG | 6.99 | ver capa B abajo |
| PF SHORT | 19.76 | — |
| DD declarado | 0.027 | — |

## Capa A — Forward-OOS (datos posteriores al cutoff)

⚠️ **No disponible.** Cutoff = 2026-03-25, los datos locales terminan en 2026-03-25 (5 velas post-cutoff). Refrescar datos con `download_new_pairs.py` para activar esta capa.

## Capa B — Walk-forward (motor corregido)

**4/11 folds aprobados** (criterio: n≥5, PF finito ≥1.2, retorno>0)

| Semestre | N | WR | PF | Retorno | OK |
|----------|---|----|----|---------|----|
| 2020-01 | — | — | — | sin datos | — |
| 2020-07 | 14 | 36% | 1.73 | +7.2% | ✅ |
| 2021-01 | 47 | 11% | 0.38 | -25.8% | ❌ |
| 2021-07 | 38 | 21% | 0.51 | -11.9% | ❌ |
| 2022-01 | 40 | 15% | 0.48 | -18.4% | ❌ |
| 2022-07 | 26 | 23% | 0.81 | -3.3% | ❌ |
| 2023-01 | 35 | 23% | 1.28 | +4.6% | ✅ |
| 2023-07 | 33 | 33% | 1.43 | +6.3% | ✅ |
| 2024-01 | 38 | 34% | 1.15 | +2.5% | ❌ |
| 2024-07 | 42 | 24% | 0.84 | -4.1% | ❌ |
| 2025-01 | 39 | 18% | 0.96 | -1.6% | ❌ |
| 2025-07 | 53 | 38% | 1.74 | +19.6% | ✅ |

## Capa C — Bootstrap de significancia

- p-value (retorno ≤ 0 por azar): **0.808** — ❌ NO significativo
- Retorno mediano re-muestreado: -27.6%
- Percentil 5: -58.5%
