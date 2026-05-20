# Re-validación V15 — UNI/USDT

> Generado por `revalidate_v15.py` (motor corregido, una posición a la vez). Fecha: 2026-05-19

## VEREDICTO: **PAPER-ONLY**

WF ok pero falta forward-OOS (refrescar datos)

## Declarado en meta_v15.json vs. motor corregido

| Métrica | Declarado (motor viejo) | Motor corregido |
|---------|------------------------|-----------------|
| WF LONG | 11/11 | 7/11 (LONG+SHORT combinado) |
| PF LONG | 8.34 | ver capa B abajo |
| PF SHORT | 20.17 | — |
| DD declarado | 0.019 | — |

## Capa A — Forward-OOS (datos posteriores al cutoff)

⚠️ **No disponible.** Cutoff = 2026-03-25, los datos locales terminan en 2026-03-25 (5 velas post-cutoff). Refrescar datos con `download_new_pairs.py` para activar esta capa.

## Capa B — Walk-forward (motor corregido)

**7/11 folds aprobados** (criterio: n≥5, PF finito ≥1.2, retorno>0)

| Semestre | N | WR | PF | Retorno | OK |
|----------|---|----|----|---------|----|
| 2020-01 | — | — | — | sin datos | — |
| 2020-07 | 14 | 29% | 1.91 | +8.6% | ✅ |
| 2021-01 | 44 | 11% | 0.24 | -29.7% | ❌ |
| 2021-07 | 45 | 40% | 1.67 | +15.4% | ✅ |
| 2022-01 | 43 | 14% | 0.56 | -15.2% | ❌ |
| 2022-07 | 30 | 30% | 1.26 | +3.9% | ✅ |
| 2023-01 | 36 | 36% | 1.46 | +6.9% | ✅ |
| 2023-07 | 32 | 25% | 0.67 | -5.3% | ❌ |
| 2024-01 | 31 | 39% | 1.88 | +12.3% | ✅ |
| 2024-07 | 46 | 37% | 1.91 | +21.6% | ✅ |
| 2025-01 | 40 | 18% | 0.67 | -8.9% | ❌ |
| 2025-07 | 41 | 34% | 1.37 | +7.6% | ✅ |

## Capa C — Bootstrap de significancia

- p-value (retorno ≤ 0 por azar): **0.390** — ❌ NO significativo
- Retorno mediano re-muestreado: +11.1%
- Percentil 5: -38.5%
