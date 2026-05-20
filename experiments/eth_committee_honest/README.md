# Comité ETH V15 "aprobado" — Re-test con motor honesto

> Reimplementación EXACTA de las reglas del comité ETH que fueron declaradas
> "APROBADO" en `docs/V15_ETH_evaluation.md` (PF 1.28, WF 8/12, annual ~30%,
> OOS 2026 +16.2%), aplicada con motor honesto.
> Fecha: 2026-05-19 · `test_committee.py`

---

## Veredicto

**RECHAZADO con motor honesto.** El "edge" declarado era artefacto del simulador.

| Criterio | Resultado |
|----------|-----------|
| Bootstrap p<0.05 (real) | NO (p=0.305) |
| WF ≥ 7/12 | NO (6/12) |
| Mediana sintético > 0 | NO (-7.0%) |
| ≥14/20 sintéticas positivas | NO (**1/20**) |
| Edge vs null > 5% | NO (**-2.6%**, peor que null) |

---

## Lo que se reprodujo exactamente

Reglas idénticas a `archive_scripts/evaluate_eth_v2.py` y `evaluate_eth_short_v5.py`:

**LONG (no BEAR)**:
- `BRK_ETH`: close>high20, vol≥1.3, BB<5.5 en 2/5 bars, ADX<32, bar<3.5%
- `FOLLOW_BRK_BTC`: BTC rompe (vol≥1.8, BB<4 en 3/5, ADX<28, bar<2.5%) + ETH-BTC corr≥0.5
- `FOLLOW_PB_BTC`: BTC pullback EMA20 (dist -0.5% a +1.5%, ADX≥15, RSI 33-58) + corr≥0.5

**SHORT (BEAR only)**:
- `MULTI_CONF`: bearish + RSI≥60 + BB_pct≥0.75 + vol≥1.0
- `BB_UPPER`: bearish + BB_pct≥0.90

**TP/SL adaptive**:
- TP = max(min(atr_pct × 2.5, 0.08), 0.025)
- SL = max(min(atr_pct × 1.5, 0.05), 0.015)
- MaxBars: 18 LONG, 16 SHORT

**Diferencia única con el original**: **una posición a la vez** (avanzar idx tras
cierre del trade). Sin esa diferencia, el original abría nuevo trade en cada vela
con señal — generando trades solapados.

---

## Resultados detallados

### Real ETH 2020-2025

| Métrica | Declarado | Honesto | Cambio |
|---------|----------:|--------:|-------:|
| N trades | 467+ | 345 | -26% |
| WR | 49% | 45% | -4pp |
| PF | 1.28 | 1.11 | **-13%** |
| Annual | ~30% | **+6.5%** | **-78%** |
| Equity $1K | $4,820 | $1,461 | **-70%** |
| MaxDD | 42.7% | 46.7% | similar |
| WF folds | 8/12 | 6/12 | -2 |
| Bootstrap p | — | 0.305 | NO sig |

### Desglose por setup (real)

| Setup | N | WR | PF | Total | p |
|-------|--:|----|---:|------:|--:|
| BRK_ETH | 34 | 47% | 1.38 | +14.1% | 0.208 |
| FOLLOW_BRK_BTC | 13 | 62% | 1.99 | +13.8% | 0.157 |
| **FOLLOW_PB_BTC** | **220** | **42%** | **1.04** | **-3.1%** | **0.521** |
| MULTI_CONF (SHORT) | 39 | 51% | 1.24 | +12.4% | 0.302 |
| BB_UPPER (SHORT) | 39 | 46% | 1.09 | +3.3% | 0.457 |

**El culpable principal**: `FOLLOW_PB_BTC` era el 64% del volumen de trades del
comité y es **prácticamente neutro** (PF 1.04, total -3%). Sin el solape inflando
el contador, su contribución real desaparece.

### Walk-forward (6/12 vs declarado 8/12)

| Semestre | N | WR | PF | Total | OK |
|----------|--:|----|---:|------:|----|
| 2020-01 | 26 | 42% | 0.97 | -4.7% | — |
| 2020-07 | 34 | 41% | 1.03 | -0.5% | — |
| 2021-01 | 25 | 52% | 1.50 | +25.8% | OK |
| 2021-07 | 23 | 30% | 0.78 | -13.2% | — |
| 2022-01 | 19 | 58% | 1.73 | +17.3% | OK |
| 2022-07 | 29 | 35% | 0.58 | -25.6% | — |
| 2023-01 | 34 | 35% | 0.84 | -8.0% | — |
| 2023-07 | 36 | 50% | 1.24 | +7.7% | OK |
| 2024-01 | 25 | 40% | 0.76 | -9.3% | — |
| 2024-07 | 35 | 54% | 1.60 | +26.2% | OK |
| 2025-01 | 26 | 50% | 1.33 | +11.9% | OK |
| 2025-07 | 32 | 53% | 1.83 | +34.5% | OK |

### Sintético (20 series block bootstrap)

| Métrica | Valor |
|---------|------:|
| Mediana annual | **-7.0%** |
| Media annual | -6.5% |
| p25-p75 | -9.6% a -2.9% |
| **Series positivas** | **1/20 (5%)** |

### Null hypothesis (shuffle)

| | Annual |
|--|-------:|
| Sintético (estructura preservada) | -7.0% |
| Null (sin estructura) | -4.4% |
| **Edge vs null** | **-2.6% (NEGATIVO)** |

**El comité es PEOR que el shuffle aleatorio.** Cuando una estrategia pierde
contra el null, no tiene edge — captura ruido específico del sample.

---

## Por qué el declarado "funcionaba"

Cuatro causas combinadas inflaban el resultado original:

1. **Trades solapados**: cada vela con señal abría trade nuevo sin guard de
   posición. FOLLOW_PB_BTC dispara en ráfagas (pullbacks duran 3-5 velas) →
   se contaban duplicadamente.

2. **WF threshold demasiado laxo**: criterio original (n≥3, WR>0.38, PF>1.0)
   pasaba folds con muestras minúsculas y rachas afortunadas.

3. **Sample favorable**: 2020-2025 BTC tuvo bull ciclos limpios. Cuando el
   comité se aplica a sintéticas (mismas estadísticas, distinto orden), pierde.
   El sintético es el control que el método original no aplicó.

4. **OOS 2026 favorable (suerte)**: 7 trades en Ene-Mar (4 LONG + 3 SHORT) con
   WR 71% y PF 5.45. **Sample tan pequeño que es indistinguible de azar** —
   bootstrap sobre 7 trades no significativo.

---

## Comparación con los 20 alts del V15 original

Idéntico patrón:

| | 20 alts | Comité ETH |
|--|---------|------------|
| Declarado en `meta_v15.json` | PF 7-20, DD 1-4% | PF 1.28, DD 42% |
| Real con motor honesto | PF 0.75-1.72 | PF 1.11 |
| Sintético | ~50% positivas | **5% positivas** |
| Edge vs null | ~0% | **-2.6%** |
| Causa | trades solapados + trail intrabar | trades solapados solo |

ETH committee es **peor que los 20 alts** en el sintético — 1/20 positivas vs
los 50% típico de los alts.

---

## Conclusión final

**8 enfoques ETH ya rechazados** con motor honesto:
1. ETH-A (trend Donchian) — marginal p=0.103
2. ETH-F (vol breakout) — lastre
3. ETH-V2 (A+F combinado) — sample desafortunado
4. Agent G (ML LightGBM + ETH/BTC ratio) — overfit p=0.808
5. Agent H (ETH/BTC ratio rotation) — control random
6. Agent I (mean-reversion RANGE) — 3 trades/año
7. Agent J (ETH 1D timeframe) — edge vs null negativo
8. **Comité ETH original "aprobado"** (este) — edge vs null negativo

**ETH no es operable con métodos algorítmicos 4h convencionales.**

La búsqueda está empíricamente agotada. Sumar ETH al portfolio del bot bajo
cualquiera de las 8 configuraciones probadas pondría capital en algo que NO
tiene edge real. La única opción honesta para exposición ETH es spot DCA o
staking (no direccional).

---

## Script
- `test_committee.py` — implementación completa con motor honesto
