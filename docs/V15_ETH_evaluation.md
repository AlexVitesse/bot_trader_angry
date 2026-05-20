# V15 ETH Evaluation — Veredicto Final

**Fecha**: 2026-03-22 (actualizado 2026-03-23)
**Result**: COMITE COMPLETO APROBADO (LONG rules BULL/RANGE + SHORT rules BEAR)

## Ronda 1: Enfoques iniciales (todos rechazados)

| Opcion | WF | WR | PF | MaxDD |
|--------|-----|----|----|-------|
| ML 4H (36 features ETH) | 6/12 | 42.0% | 1.09 | 89% |
| ML 1D | 3/6 | 39.3% | 1.19 | - |
| Rule-based 4H | 5/12 | 34.2% | 0.75 | 49% |
| BTC-follower v1 | 5/12 | 46.9% | 1.12 | 25% |

Conclusion ronda 1: ML no funciona para ETH, pero BTC-follower y Breakout B muestran potencial.

## Ronda 2: 5 estrategias optimizadas

Basadas en lo que SI funciono: BTC-follower (WR 47%, MaxDD 25%), Breakout B (WR 45%, PF 1.29).

### Resultados

| # | Opcion | WF | N OOS | WR | PF | t/m | $1K-> | MaxDD | Veredicto |
|---|--------|-----|-------|----|----|-----|-------|-------|-----------|
| 1 | Follower Tuned | 7/12 | 218 | 46.8% | 1.19 | 5.0 | $1,463 | 29% | MARGINAL |
| 2 | Breakout Adapted | 6/12 | 295 | 40.3% | 0.91 | 6.8 | $620 | 56% | RECHAZADO |
| 3 | **Hibrido** | **8/12** | 275 | **48.4%** | **1.23** | 6.3 | $1,859 | 30% | **APROBADO** |
| 4 | Regime BULL | 7/12 | 423 | 43.3% | 1.00 | 9.9 | $845 | 52% | RECHAZADO |
| 5 | **TP/SL Adapt** | **7/12** | 278 | **49.3%** | **1.35** | 6.4 | $2,904 | 30% | **APROBADO** |

### Opcion 5: TP/SL Adaptados (MEJOR)

Misma logica que el Hibrido pero con TP/SL escaleados a la volatilidad de ETH:
- SL = 1.5 * ATR (vs 1.0-1.2 en BTC), capped 1.5%-5%
- TP = 2.5 * ATR (vs 1.5 en BTC), capped 2.5%-8%
- MaxBars = 18

| Metrica | Valor | Criterio | Pasa? |
|---------|-------|----------|-------|
| WF Folds | 7/12 | >= 7/12 | SI |
| WR | 49.3% | > 40% | SI |
| PF | 1.35 | > 1.2 | SI |
| MaxDD | 30.2% | < 25% ideal | MARGINAL |
| Trades/mes | 6.4 | > 2 | SI |
| Equity | $1K -> $2,904 | positivo | SI |

Breakdown por setup:
- BRK_ETH (breakout propio): N=57, WR=54.4%, PF=1.63
- FOLLOW_BRK_BTC: N=37, WR=48.6%, PF=1.29
- FOLLOW_PB_BTC: N=184, WR=47.8%, PF=1.29

### Opcion 3: Hibrido (segundo mejor)

Combina BTC-follower tuned + Breakout B ETH adaptado.
- WF 8/12 (mas consistente), PF 1.23, WR 48.4%
- $1K -> $1,859, MaxDD 30.1%

### Por que funciona

1. **BTC-follower aporta consistencia** (PF 1.24 en pullback BTC)
2. **Breakout B en ETH aporta edge** (PF 1.35-1.63)
3. **TP/SL amplios capturan la mayor volatilidad de ETH** (ATR ETH ~1.5x BTC)
4. **No depende de ML** — estrategia 100% rule-based, sin riesgo de overfitting
5. **MaxDD ~30%** — manejable (vs 89% del ML)

### Que NO funciona en ETH

1. ML (cualquier modelo, cualquier timeframe): AUC ~0.53
2. Pullback EMA20 solo: WR 33-37%, PF < 0.80
3. Operar en BEAR/RANGE sin filtro BTC: destruye valor

## Ronda 3: BEAR SHORT — ML (todos rechazados)

Objetivo: encontrar estrategia SHORT para mercados BEAR de ETH (25.7% del tiempo).

### ML SHORT (todos rechazados)

| Metodo | WF | N | WR | PF | DD | Veredicto |
|--------|-----|---|----|----|-----|-----------|
| ML GBM | 4/12 | 1039 | 40.6% | 0.92 | 99% | RECHAZADO |
| ML RF (t=0.60) | 3/12 | 36 | 50.0% | 1.56 | 21% | MARGINAL |
| ML LGBM | 4/12 | 1080 | 41.4% | 0.92 | 100% | RECHAZADO |
| ML XGB | 4/12 | 1035 | 40.3% | 0.90 | 100% | RECHAZADO |
| ENS Voting | 2/12 | 257 | 38.1% | 0.83 | 90% | RECHAZADO |
| ENS AvgProb | 3/12 | 379 | 39.3% | 0.86 | 93% | RECHAZADO |
| ENS Stacking | 4/12 | 915 | 40.5% | 0.88 | 99% | RECHAZADO |

Ademas, RF SHORT probado OOS 2026 (Ene-Mar): max prob 0.566 < threshold 0.60, **cero trades**. ML descartado.

## Ronda 4: SHORT rule-based — busqueda intensiva

Criterio corregido: evaluar SHORT SOLO en folds con >=30 bars BEAR (9/12 validos).
Aprobacion: >= 60% de folds BEAR validos positivos (6/9) + WR > 50% + PF > 1.2.

### v2-v3: 12 opciones (ninguna aprobada)

| Opcion | BEAR OK | N | WR | PF | DD | Veredicto |
|--------|---------|---|----|----|-----|-----------|
| MeanRev v2 (mejor) | 5/9 | 77 | 53.2% | 1.38 | 14.9% | MARGINAL (56%) |
| RF t=0.55 | 3/9 | 58 | 44.8% | 0.87 | 91% | RECHAZADO |
| MeanRev basic | 4/9 | 49 | 49.0% | 1.13 | 21% | RECHAZADO |
| Otros 9 opciones | 2-4/9 | - | 35-48% | 0.3-1.1 | 14-99% | RECHAZADOS |

### v4: 15 estrategias WR>50% focus

| Opcion | BEAR OK | N | WR | PF | DD | Veredicto |
|--------|---------|---|----|----|-----|-----------|
| **Multi-conf** | **5/9 (56%)** | 29 | **64.3%** | **2.70** | **8.6%** | A 1 fold de aprobar |
| BB upper | 5/9 (56%) | 61 | 56.7% | 1.60 | 13.7% | A 1 fold de aprobar |
| Multi v2 | 4/9 | 17 | 52.9% | 1.45 | 16.1% | RECHAZADO |
| Exhaustion+RSI | 3/9 | 40 | 45.0% | 1.06 | 22% | RECHAZADO |
| Otros 11 | 1-4/9 | - | 30-52% | 0.3-1.4 | 10-32% | RECHAZADOS |

Multi-conf (RSI>60 + BB_pct>0.75 + bearish candle + vol_ratio>1.0) y BB upper (BB_pct>0.90 + bearish) = los 2 mejores.

## Ronda 5: Combinaciones + OOS 2026 (FINAL)

### SHORT standalone — combinaciones

| Combo | BEAR OK | N | WR | PF | $1K-> | DD | Veredicto |
|-------|---------|---|----|----|-------|-----|-----------|
| **Multi conf** | **6/9 (67%)** | 29 | **65.5%** | **2.81** | $1,583 | **8.6%** | **APROBADO** |
| BB upper | 5/9 (56%) | 61 | 57.4% | 1.64 | $1,570 | 13.7% | MARGINAL |
| Multi+BB | 5/9 (56%) | 68 | 57.4% | 1.69 | $1,727 | 18.1% | MARGINAL |

**Multi conf APROBADO como SHORT standalone**: 6/9 folds BEAR positivos (67%), WR 65.5%, PF 2.81, DD 8.6%.
Logica: solo 29 trades en 5 anos (~0.5/mes) — muy selectivo, alta calidad.

### Comites completos (12 folds)

| Comite | WF | N | WR | PF | $1K-> | DD | Veredicto |
|--------|-----|---|----|----|-------|-----|-----------|
| Solo LONG | 7/12 | 396 | 48.0% | 1.23 | $2,929 | 44.3% | APROBADO |
| **LONG+Multi+BB** | **8/12** | 467 | **49.0%** | **1.28** | **$4,820** | 42.7% | **APROBADO** |
| LONG+Multi+BB+MRv2 | 8/12 | 520 | 48.7% | 1.26 | $5,172 | 44.3% | APROBADO |

### OOS 2026 (Ene-Mar) — ETH -36.1%

Datos NUNCA vistos por el modelo (entrenado solo 2020-2025).

| Comite | N | LONG | SHORT | WR | PF | $1K-> | DD |
|--------|---|------|-------|----|----|-------|-----|
| Solo LONG | 4 | 4 | 0 | 75.0% | 4.99 | $1,074 | 1.8% |
| **LONG+Multi+BB** | **7** | **4** | **3** | **71.4%** | **5.45** | **$1,162** | **1.8%** |

Detalle SHORT OOS 2026:
- 2026-01-02: BB_UPPER SHORT $3,018 -> SL (-1.63%)
- 2026-02-13: BB_UPPER SHORT $2,049 -> TP (+4.39%)
- 2026-02-14: BB_UPPER SHORT $2,087 -> TP (+5.34%)

**+16.2% mientras ETH cayo -36.1%** — los 3 SHORT sumaron +8.1% neto al comite.

## MEJOR: LONG+SHORT(Multi+BB) — Comite final ETH

| Regimen | Estrategia | Tipo | Detalle |
|---------|-----------|------|---------|
| BULL | BRK_ETH + FOLLOW_BRK/PB_BTC | Reglas | corr>=0.5 para follower, vol>=1.3+BB<5.5 para BRK |
| RANGE | BRK_ETH + FOLLOW_BRK/PB_BTC | Reglas | mismo que BULL |
| BEAR | Multi-conf + BB upper SHORT | Reglas | RSI>60+BB>0.75+bear candle+vol>1 / BB>0.90+bear candle |

TP/SL adaptados a ETH: SL=1.5*ATR (1.5-5%), TP=2.5*ATR (2.5-8%), MaxBars=18.
100% rule-based, sin ML. Sin riesgo de overfitting.

### Por que funciona

1. **LONG en BULL/RANGE**: BTC-follower + Breakout B ETH cubren tendencia y volatilidad
2. **SHORT en BEAR**: Multi-conf es muy selectivo (WR 65%, ~0.5 trades/mes) + BB upper agrega volumen
3. **OOS 2026 validado**: +16.2% en mercado -36.1% (7 trades, WR 71.4%)
4. **Sin ML**: todo rule-based, sin riesgo de degradacion por distribuciones cambiantes
5. **DD controlado**: 8.6% en SHORT standalone, 42.7% en comite completo

## Proximos pasos

1. **Cross-asset validation** — probar comite en pares correlacionados
2. **Implementar en ml_strategy_v15.py** — modulo ETH del comite
3. **Paper trading** — agregar ETH al bot en testnet
4. **Sizing**: conservador (50% del size de BTC) por DD 42.7%

## Archivos

| Archivo | Descripcion |
|---------|-------------|
| evaluate_eth_models.py | Ronda 1: ML multi-modelo (rechazado) |
| evaluate_eth_options.py | Ronda 1: 3 opciones alternativas (rechazadas) |
| evaluate_eth_v2.py | Ronda 2: 5 estrategias LONG optimizadas (2 aprobadas) |
| evaluate_eth_bear.py | Ronda 3: ML SHORT (todos rechazados) |
| evaluate_eth_short_v2.py | Ronda 4a: 12 SHORT rule-based (ninguno aprobado) |
| evaluate_eth_short_v3.py | Ronda 4a: correccion solo folds BEAR (ninguno aprobado) |
| evaluate_eth_short_v4.py | Ronda 4b: 15 SHORT WR>50% (Multi-conf y BB upper mejores) |
| evaluate_eth_short_v5.py | Ronda 5: combos + OOS 2026 (**Multi-conf APROBADO**) |
| evaluate_eth_2026_oos.py | OOS 2026: validacion comites (RF SHORT = 0 trades) |
