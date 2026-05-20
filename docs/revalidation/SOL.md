# Re-validación V15 — SOL/USDT

> Generado por `revalidate_v15.py` (motor corregido, una posición a la vez). Fecha: 2026-05-19

## VEREDICTO: **REJECT (provisional)**

falla WF con motor corregido

## Declarado en meta_v15.json vs. motor corregido

| Métrica | Declarado (motor viejo) | Motor corregido |
|---------|------------------------|-----------------|
| WF LONG | 8/10 | 6/11 (LONG+SHORT combinado) |
| PF LONG | 2.56 | ver capa B abajo |
| PF SHORT | 15.04 | — |
| DD declarado | 0.096 | — |

## Capa A — Forward-OOS (datos posteriores al cutoff)

⚠️ **No disponible.** Cutoff = 2026-03-24, los datos locales terminan en 2026-02-27 (0 velas post-cutoff). Refrescar datos con `download_new_pairs.py` para activar esta capa.

## Capa B — Walk-forward (motor corregido)

**6/11 folds aprobados** (criterio: n≥5, PF finito ≥1.2, retorno>0)

| Semestre | N | WR | PF | Retorno | OK |
|----------|---|----|----|---------|----|
| 2020-01 | — | — | — | sin datos | — |
| 2020-07 | 5 | 20% | 0.09 | -4.4% | ❌ |
| 2021-01 | 32 | 31% | 2.99 | +54.1% | ✅ |
| 2021-07 | 34 | 29% | 1.64 | +13.0% | ✅ |
| 2022-01 | 43 | 9% | 0.30 | -24.9% | ❌ |
| 2022-07 | 31 | 23% | 0.63 | -10.0% | ❌ |
| 2023-01 | 29 | 28% | 1.35 | +5.2% | ✅ |
| 2023-07 | 35 | 29% | 1.62 | +11.8% | ✅ |
| 2024-01 | 36 | 44% | 2.65 | +25.7% | ✅ |
| 2024-07 | 49 | 33% | 1.01 | -0.1% | ❌ |
| 2025-01 | 47 | 21% | 0.81 | -5.3% | ❌ |
| 2025-07 | 55 | 35% | 1.39 | +9.4% | ✅ |

## Capa C — Bootstrap de significancia

- p-value (retorno ≤ 0 por azar): **0.155** — ❌ NO significativo
- Retorno mediano re-muestreado: +56.7%
- Percentil 5: -22.7%
