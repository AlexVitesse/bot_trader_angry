# Re-validación V15 — BNB/USDT

> Generado por `revalidate_v15.py` (motor corregido, una posición a la vez). Fecha: 2026-05-19

## VEREDICTO: **REJECT (provisional)**

falla WF con motor corregido

## Declarado en meta_v15.json vs. motor corregido

| Métrica | Declarado (motor viejo) | Motor corregido |
|---------|------------------------|-----------------|
| WF LONG | 11/12 | 4/12 (LONG+SHORT combinado) |
| PF LONG | 2.85 | ver capa B abajo |
| PF SHORT | 8.64 | — |
| DD declarado | 0.045 | — |

## Capa A — Forward-OOS (datos posteriores al cutoff)

⚠️ **No disponible.** Cutoff = 2026-03-25, los datos locales terminan en 2026-02-27 (0 velas post-cutoff). Refrescar datos con `download_new_pairs.py` para activar esta capa.

## Capa B — Walk-forward (motor corregido)

**4/12 folds aprobados** (criterio: n≥5, PF finito ≥1.2, retorno>0)

| Semestre | N | WR | PF | Retorno | OK |
|----------|---|----|----|---------|----|
| 2020-01 | 46 | 13% | 0.37 | -16.3% | ❌ |
| 2020-07 | 63 | 38% | 1.11 | +2.6% | ❌ |
| 2021-01 | 38 | 16% | 0.64 | -10.6% | ❌ |
| 2021-07 | 46 | 35% | 1.42 | +8.6% | ✅ |
| 2022-01 | 47 | 17% | 0.68 | -9.0% | ❌ |
| 2022-07 | 33 | 36% | 0.97 | -0.6% | ❌ |
| 2023-01 | 37 | 41% | 1.37 | +5.2% | ✅ |
| 2023-07 | 34 | 35% | 0.84 | -1.9% | ❌ |
| 2024-01 | 34 | 59% | 2.34 | +12.5% | ✅ |
| 2024-07 | 54 | 30% | 0.61 | -8.9% | ❌ |
| 2025-01 | 40 | 28% | 0.39 | -9.4% | ❌ |
| 2025-07 | 52 | 42% | 1.20 | +3.5% | ✅ |

## Capa C — Bootstrap de significancia

- p-value (retorno ≤ 0 por azar): **0.831** — ❌ NO significativo
- Retorno mediano re-muestreado: -23.7%
- Percentil 5: -49.7%
