# Re-validación V15 — FIL/USDT

> Generado por `revalidate_v15.py` (motor corregido, una posición a la vez). Fecha: 2026-05-19

## VEREDICTO: **REJECT (provisional)**

falla WF con motor corregido

## Declarado en meta_v15.json vs. motor corregido

| Métrica | Declarado (motor viejo) | Motor corregido |
|---------|------------------------|-----------------|
| WF LONG | 9/10 | 5/11 (LONG+SHORT combinado) |
| PF LONG | 9.88 | ver capa B abajo |
| PF SHORT | 18.43 | — |
| DD declarado | 0.028 | — |

## Capa A — Forward-OOS (datos posteriores al cutoff)

⚠️ **No disponible.** Cutoff = 2026-03-25, los datos locales terminan en 2026-03-01 (0 velas post-cutoff). Refrescar datos con `download_new_pairs.py` para activar esta capa.

## Capa B — Walk-forward (motor corregido)

**5/11 folds aprobados** (criterio: n≥5, PF finito ≥1.2, retorno>0)

| Semestre | N | WR | PF | Retorno | OK |
|----------|---|----|----|---------|----|
| 2020-01 | — | — | — | sin datos | — |
| 2020-07 | 3 | 67% | 2.69 | +1.2% | ❌ |
| 2021-01 | 24 | 12% | 0.63 | -8.0% | ❌ |
| 2021-07 | 34 | 21% | 0.83 | -4.1% | ❌ |
| 2022-01 | 40 | 15% | 0.79 | -7.2% | ❌ |
| 2022-07 | 20 | 40% | 2.66 | +16.4% | ✅ |
| 2023-01 | 35 | 34% | 1.57 | +10.0% | ✅ |
| 2023-07 | 37 | 19% | 0.43 | -11.7% | ❌ |
| 2024-01 | 33 | 27% | 0.77 | -4.2% | ❌ |
| 2024-07 | 44 | 36% | 1.58 | +12.9% | ✅ |
| 2025-01 | 43 | 28% | 1.54 | +12.7% | ✅ |
| 2025-07 | 35 | 31% | 1.21 | +3.5% | ✅ |

## Capa C — Bootstrap de significancia

- p-value (retorno ≤ 0 por azar): **0.435** — ❌ NO significativo
- Retorno mediano re-muestreado: +5.3%
- Percentil 5: -39.1%
