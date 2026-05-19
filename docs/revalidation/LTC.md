# Re-validación V15 — LTC/USDT

> Generado por `revalidate_v15.py` (motor corregido, una posición a la vez). Fecha: 2026-05-19

## VEREDICTO: **REJECT (provisional)**

falla WF con motor corregido

## Declarado en meta_v15.json vs. motor corregido

| Métrica | Declarado (motor viejo) | Motor corregido |
|---------|------------------------|-----------------|
| WF LONG | 12/12 | 5/12 (LONG+SHORT combinado) |
| PF LONG | 7.1 | ver capa B abajo |
| PF SHORT | 18.21 | — |
| DD declarado | 0.026 | — |

## Capa A — Forward-OOS (datos posteriores al cutoff)

⚠️ **No disponible.** Cutoff = 2026-03-25, los datos locales terminan en 2026-03-25 (5 velas post-cutoff). Refrescar datos con `download_new_pairs.py` para activar esta capa.

## Capa B — Walk-forward (motor corregido)

**5/12 folds aprobados** (criterio: n≥5, PF finito ≥1.2, retorno>0)

| Semestre | N | WR | PF | Retorno | OK |
|----------|---|----|----|---------|----|
| 2020-01 | 25 | 28% | 2.28 | +16.7% | ✅ |
| 2020-07 | 58 | 31% | 1.54 | +16.4% | ✅ |
| 2021-01 | 48 | 8% | 0.19 | -28.4% | ❌ |
| 2021-07 | 40 | 15% | 0.27 | -16.5% | ❌ |
| 2022-01 | 45 | 24% | 1.24 | +5.9% | ✅ |
| 2022-07 | 24 | 25% | 0.81 | -2.6% | ❌ |
| 2023-01 | 39 | 31% | 1.10 | +1.5% | ❌ |
| 2023-07 | 39 | 28% | 0.81 | -3.5% | ❌ |
| 2024-01 | 39 | 41% | 2.18 | +17.5% | ✅ |
| 2024-07 | 47 | 30% | 1.18 | +3.5% | ❌ |
| 2025-01 | 39 | 26% | 1.22 | +4.2% | ✅ |
| 2025-07 | 54 | 24% | 0.57 | -11.9% | ❌ |

## Capa C — Bootstrap de significancia

- p-value (retorno ≤ 0 por azar): **0.563** — ❌ NO significativo
- Retorno mediano re-muestreado: -6.5%
- Percentil 5: -50.4%
