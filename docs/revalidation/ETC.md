# Re-validación V15 — ETC/USDT

> Generado por `revalidate_v15.py` (motor corregido, una posición a la vez). Fecha: 2026-05-19

## VEREDICTO: **PAPER-ONLY**

WF ok pero falta forward-OOS (refrescar datos)

## Declarado en meta_v15.json vs. motor corregido

| Métrica | Declarado (motor viejo) | Motor corregido |
|---------|------------------------|-----------------|
| WF LONG | 12/12 | 7/12 (LONG+SHORT combinado) |
| PF LONG | 8.27 | ver capa B abajo |
| PF SHORT | 17.12 | — |
| DD declarado | 0.03 | — |

## Capa A — Forward-OOS (datos posteriores al cutoff)

⚠️ **No disponible.** Cutoff = 2026-03-25, los datos locales terminan en 2026-03-25 (5 velas post-cutoff). Refrescar datos con `download_new_pairs.py` para activar esta capa.

## Capa B — Walk-forward (motor corregido)

**7/12 folds aprobados** (criterio: n≥5, PF finito ≥1.2, retorno>0)

| Semestre | N | WR | PF | Retorno | OK |
|----------|---|----|----|---------|----|
| 2020-01 | 27 | 30% | 3.57 | +43.8% | ✅ |
| 2020-07 | 50 | 30% | 1.11 | +2.1% | ❌ |
| 2021-01 | 38 | 26% | 1.61 | +15.8% | ✅ |
| 2021-07 | 44 | 25% | 0.53 | -12.0% | ❌ |
| 2022-01 | 46 | 24% | 1.67 | +21.8% | ✅ |
| 2022-07 | 25 | 24% | 1.12 | +1.5% | ❌ |
| 2023-01 | 36 | 25% | 1.48 | +8.3% | ✅ |
| 2023-07 | 37 | 14% | 0.47 | -11.2% | ❌ |
| 2024-01 | 45 | 38% | 1.43 | +8.2% | ✅ |
| 2024-07 | 45 | 38% | 1.28 | +5.4% | ✅ |
| 2025-01 | 43 | 23% | 0.81 | -5.0% | ❌ |
| 2025-07 | 49 | 33% | 1.64 | +13.9% | ✅ |

## Capa C — Bootstrap de significancia

- p-value (retorno ≤ 0 por azar): **0.050** — ❌ NO significativo
- Retorno mediano re-muestreado: +121.6%
- Percentil 5: +0.2%
