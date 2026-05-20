# Re-validación V15 — XRP/USDT

> Generado por `revalidate_v15.py` (motor corregido, una posición a la vez). Fecha: 2026-05-19

## VEREDICTO: **REJECT (provisional)**

falla WF con motor corregido

## Declarado en meta_v15.json vs. motor corregido

| Métrica | Declarado (motor viejo) | Motor corregido |
|---------|------------------------|-----------------|
| WF LONG | 12/12 | 4/12 (LONG+SHORT combinado) |
| PF LONG | 4.36 | ver capa B abajo |
| PF SHORT | 18.8 | — |
| DD declarado | 0.037 | — |

## Capa A — Forward-OOS (datos posteriores al cutoff)

⚠️ **No disponible.** Cutoff = 2026-03-25, los datos locales terminan en 2026-02-27 (0 velas post-cutoff). Refrescar datos con `download_new_pairs.py` para activar esta capa.

## Capa B — Walk-forward (motor corregido)

**4/12 folds aprobados** (criterio: n≥5, PF finito ≥1.2, retorno>0)

| Semestre | N | WR | PF | Retorno | OK |
|----------|---|----|----|---------|----|
| 2020-01 | 36 | 17% | 0.34 | -13.6% | ❌ |
| 2020-07 | 61 | 30% | 1.15 | +3.7% | ❌ |
| 2021-01 | 26 | 8% | 0.29 | -14.0% | ❌ |
| 2021-07 | 47 | 34% | 1.34 | +7.7% | ✅ |
| 2022-01 | 41 | 27% | 1.13 | +2.2% | ❌ |
| 2022-07 | 32 | 34% | 1.27 | +3.8% | ✅ |
| 2023-01 | 38 | 18% | 0.53 | -9.4% | ❌ |
| 2023-07 | 25 | 24% | 0.84 | -2.0% | ❌ |
| 2024-01 | 37 | 32% | 0.72 | -5.0% | ❌ |
| 2024-07 | 46 | 35% | 1.49 | +10.1% | ✅ |
| 2025-01 | 40 | 22% | 1.30 | +5.5% | ✅ |
| 2025-07 | 58 | 26% | 1.07 | +1.3% | ❌ |

## Capa C — Bootstrap de significancia

- p-value (retorno ≤ 0 por azar): **0.623** — ❌ NO significativo
- Retorno mediano re-muestreado: -11.1%
- Percentil 5: -51.2%
