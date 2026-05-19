# Re-validación V15 — DOGE/USDT

> Generado por `revalidate_v15.py` (motor corregido, una posición a la vez). Fecha: 2026-05-19

## VEREDICTO: **REJECT (provisional)**

falla WF con motor corregido

## Declarado en meta_v15.json vs. motor corregido

| Métrica | Declarado (motor viejo) | Motor corregido |
|---------|------------------------|-----------------|
| WF LONG | 12/12 | 4/12 (LONG+SHORT combinado) |
| PF LONG | 7.51 | ver capa B abajo |
| PF SHORT | 12.73 | — |
| DD declarado | 0.044 | — |

## Capa A — Forward-OOS (datos posteriores al cutoff)

⚠️ **No disponible.** Cutoff = 2026-03-24, los datos locales terminan en 2026-02-27 (0 velas post-cutoff). Refrescar datos con `download_new_pairs.py` para activar esta capa.

## Capa B — Walk-forward (motor corregido)

**4/12 folds aprobados** (criterio: n≥5, PF finito ≥1.2, retorno>0)

| Semestre | N | WR | PF | Retorno | OK |
|----------|---|----|----|---------|----|
| 2020-01 | 30 | 30% | 0.99 | -0.5% | ❌ |
| 2020-07 | 29 | 41% | 1.73 | +9.4% | ✅ |
| 2021-01 | 22 | 0% | 0.00 | -23.5% | ❌ |
| 2021-07 | 40 | 20% | 0.62 | -10.0% | ❌ |
| 2022-01 | 40 | 25% | 0.96 | -1.7% | ❌ |
| 2022-07 | 25 | 12% | 0.45 | -8.2% | ❌ |
| 2023-01 | 28 | 36% | 2.02 | +13.1% | ✅ |
| 2023-07 | 30 | 33% | 1.10 | +1.0% | ❌ |
| 2024-01 | 37 | 43% | 2.79 | +26.8% | ✅ |
| 2024-07 | 45 | 24% | 1.17 | +4.0% | ❌ |
| 2025-01 | 44 | 23% | 1.21 | +4.6% | ✅ |
| 2025-07 | 50 | 30% | 1.02 | +0.1% | ❌ |

## Capa C — Bootstrap de significancia

- p-value (retorno ≤ 0 por azar): **0.346** — ❌ NO significativo
- Retorno mediano re-muestreado: +16.7%
- Percentil 5: -36.0%
