# Re-validación V15 — LINK/USDT

> Generado por `revalidate_v15.py` (motor corregido, una posición a la vez). Fecha: 2026-05-19

## VEREDICTO: **REJECT (provisional)**

falla WF con motor corregido

## Declarado en meta_v15.json vs. motor corregido

| Métrica | Declarado (motor viejo) | Motor corregido |
|---------|------------------------|-----------------|
| WF LONG | 12/12 | 3/12 (LONG+SHORT combinado) |
| PF LONG | 7.4 | ver capa B abajo |
| PF SHORT | 15.02 | — |
| DD declarado | 0.031 | — |

## Capa A — Forward-OOS (datos posteriores al cutoff)

⚠️ **No disponible.** Cutoff = 2026-03-25, los datos locales terminan en 2026-02-27 (0 velas post-cutoff). Refrescar datos con `download_new_pairs.py` para activar esta capa.

## Capa B — Walk-forward (motor corregido)

**3/12 folds aprobados** (criterio: n≥5, PF finito ≥1.2, retorno>0)

| Semestre | N | WR | PF | Retorno | OK |
|----------|---|----|----|---------|----|
| 2020-01 | 28 | 25% | 1.36 | +5.3% | ✅ |
| 2020-07 | 50 | 22% | 0.75 | -8.4% | ❌ |
| 2021-01 | 47 | 19% | 0.48 | -19.3% | ❌ |
| 2021-07 | 42 | 24% | 0.60 | -10.2% | ❌ |
| 2022-01 | 42 | 10% | 0.31 | -23.6% | ❌ |
| 2022-07 | 27 | 41% | 2.43 | +16.6% | ✅ |
| 2023-01 | 32 | 25% | 0.70 | -4.7% | ❌ |
| 2023-07 | 39 | 23% | 0.45 | -11.7% | ❌ |
| 2024-01 | 41 | 32% | 1.10 | +1.7% | ❌ |
| 2024-07 | 50 | 30% | 1.06 | +1.1% | ❌ |
| 2025-01 | 40 | 18% | 0.69 | -7.9% | ❌ |
| 2025-07 | 52 | 33% | 1.67 | +17.1% | ✅ |

## Capa C — Bootstrap de significancia

- p-value (retorno ≤ 0 por azar): **0.958** — ❌ NO significativo
- Retorno mediano re-muestreado: -43.0%
- Percentil 5: -66.3%
