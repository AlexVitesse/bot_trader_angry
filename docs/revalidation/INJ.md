# Re-validación V15 — INJ/USDT

> Generado por `revalidate_v15.py` (motor corregido, una posición a la vez). Fecha: 2026-05-19

## VEREDICTO: **REJECT (provisional)**

falla WF con motor corregido

## Declarado en meta_v15.json vs. motor corregido

| Métrica | Declarado (motor viejo) | Motor corregido |
|---------|------------------------|-----------------|
| WF LONG | 10/10 | 5/11 (LONG+SHORT combinado) |
| PF LONG | 8.89 | ver capa B abajo |
| PF SHORT | 18.96 | — |
| DD declarado | 0.025 | — |

## Capa A — Forward-OOS (datos posteriores al cutoff)

⚠️ **No disponible.** Cutoff = 2026-03-25, los datos locales terminan en 2026-03-01 (0 velas post-cutoff). Refrescar datos con `download_new_pairs.py` para activar esta capa.

## Capa B — Walk-forward (motor corregido)

**5/11 folds aprobados** (criterio: n≥5, PF finito ≥1.2, retorno>0)

| Semestre | N | WR | PF | Retorno | OK |
|----------|---|----|----|---------|----|
| 2020-01 | — | — | — | sin datos | — |
| 2020-07 | 8 | 25% | 0.51 | -4.6% | ❌ |
| 2021-01 | 34 | 18% | 0.50 | -17.7% | ❌ |
| 2021-07 | 33 | 18% | 0.77 | -6.4% | ❌ |
| 2022-01 | 30 | 37% | 2.54 | +35.7% | ✅ |
| 2022-07 | 19 | 26% | 1.24 | +2.3% | ✅ |
| 2023-01 | 25 | 28% | 1.36 | +5.2% | ✅ |
| 2023-07 | 39 | 31% | 1.07 | +0.9% | ❌ |
| 2024-01 | 40 | 18% | 0.65 | -9.6% | ❌ |
| 2024-07 | 50 | 24% | 0.99 | -0.9% | ❌ |
| 2025-01 | 41 | 27% | 1.21 | +4.8% | ✅ |
| 2025-07 | 46 | 35% | 1.41 | +10.4% | ✅ |

## Capa C — Bootstrap de significancia

- p-value (retorno ≤ 0 por azar): **0.386** — ❌ NO significativo
- Retorno mediano re-muestreado: +13.7%
- Percentil 5: -39.6%
