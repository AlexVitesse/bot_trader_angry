# Agent M — SOL/USDT 4h as "BTC apalancado" (SOL beta-leveraged on A's signal)

**Veredicto: REJECT** — la hipótesis "SOL es BTC apalancado" no sobrevive al test sintético honesto.

## Tesis

SOL tiene beta ≈ 1.29 a BTC y vol ratio ≈ 2.06 (anual 121% vs 59%). Rolling 168-bar
correlación: mediana 0.73, ≥0.5 en 86% de barras. Si BTC tiene edge demostrable
(A's Donchian-55 + EMA daily, p=0.07 in-sample; A+F=V2 con p=0.031), entonces
operar SOL en los MISMOS timestamps cuando A dispara LONG sobre BTC debería
**heredar** ese edge amplificado por la beta, sin pretender predecir SOL.

NO buscamos alpha propia de SOL (ya rechazada en V14/V15: ML SOL dedicado dio
0 trades post-2022; committee con BB_UPPER SHORT colapsó con DD 48%).
Buscamos **beta a un edge real**.

## Mecanismo (FROZEN — sin re-tuneo)

1. **Trigger**: cuando `A.signal(df_btc_4h_features, idx) == 'LONG'` (BTC).
2. **Filtro de correlación**: SOL-BTC rolling 168-bar (shift 1) ≥ 0.5.
3. **Entrada**: LONG SOL al close de la vela donde A dispara.
4. **Trailing stop**: 2.5×ATR_SOL, floor 4%, ceiling 10% (escalado de A en BTC por vol ratio).
5. **max_bars**: 60 (10 días).
6. **Una posición a la vez**, sin overlap; sin look-ahead intrabar.

Todos los parámetros de SOL se derivaron **a priori** de A's (en BTC) y de la
ratio de volatilidad SOL/BTC; no hubo grid search.

## Resultados (cutoff 2025-12-31)

### Capa 1 — Real SOL in-sample 2020-09 → 2025-12

| Métrica | Valor |
|---|---:|
| N trades | 83 |
| WR | 42.2% |
| PF | 1.33 |
| Total return | +49.6% |
| Annual | **+8.4%** |
| Max DD | 28.4% |
| Sharpe-like | 0.11 |
| **Bootstrap p (3000 iter)** | **0.243** ← no significativo |
| Resampled p5 / p50 / p95 | −40% / +44% / +284% |

**Walk-forward (12 semestres con purga 14d)**: 3 de 9 evaluados pasan
(`PF≥1.2 ∧ total>0 ∧ n≥3`):

| Período | N | WR | PF | Total | DD | Status |
|---|---:|---:|---:|---:|---:|---|
| 2020-01 | — | — | — | — | — | sin datos (SOL no existía) |
| 2020-07 | 5 | 20.0% | 0.84 | −4.1% | 16.4% | [−] |
| 2021-01 | 6 | 33.3% | 1.11 | +0.1% | 15.5% | [−] |
| 2021-07 | 6 | 50.0% | 0.97 | −1.4% | 13.8% | [−] |
| 2022-01 | 0 | — | — | — | — | [no-signal] (BTC BEAR macro) |
| 2022-07 | 0 | — | — | — | — | [no-signal] |
| 2023-01 | 7 | 42.9% | 1.31 | +3.3% | 6.4% | **[+]** |
| 2023-07 | 10 | 60.0% | 1.06 | −0.3% | 12.6% | [−] |
| 2024-01 | 11 | 45.5% | 2.20 | +24.5% | 11.9% | **[+]** |
| 2024-07 | 13 | 46.2% | 2.73 | +28.4% | 8.7% | **[+]** |
| 2025-01 | 11 | 36.4% | 1.13 | +0.7% | 19.9% | [−] |
| 2025-07 | 7 | 28.6% | 0.31 | −9.9% | 9.9% | [−] |

El edge se concentra en 2024 (bull cycle BTC con beta SOL alta). Sin él, el
resultado global es negativo.

### Capa 2 — 20 series sintéticas (joint block bootstrap SOL+BTC, preservando corr)

| Métrica | Valor |
|---|---:|
| Mediana annual | **−1.3%** |
| Media annual | +3.0% |
| p5 / p95 | −16.2% / +58.6% |
| # series con annual > 0 | **10/20** |
| Real SOL (+8.4%) | **dentro** de p5-p95 (no es outlier) |

> Crítico: el real +8.4% NO es outlier; está cerca de la mediana de la
> distribución sintética. La hipótesis del beta-leveraged en SOL produce
> resultados igualmente negativos en la mitad de los universos paralelos.

**Block bootstrap conjunto**: sample bloques de 24 barras consecutivas
SOL+BTC con los MISMOS timestamps → preserva la correlación local SOL-BTC
(crítica para el filtro de corr y para la hipótesis de la beta).

### Capa 3 — Null (shuffle SOL retornos, corr filter relajado)

| Métrica | Valor |
|---|---:|
| Mediana annual null | −4.8% |
| **Edge synth − null** | **+3.5%** ← no supera el umbral +5% |

Nota: en null la `corr_min` se relaja porque la corr SOL-BTC con SOL shuffleado
es ≈0 y bloquearía 100% de trades. Esto mide "si SOL es ruido temporal puro,
¿qué pasa con A's señal + trailing en SOL?". Sale negativo, lo cual es
correcto y esperado, pero el edge del sintético sobre este null es marginal.

### Capa 4 — Cross-check: reemplazar A's signal por random uniforme

A's signal real reemplazada por una función random uniforme con la MISMA tasa
de disparo (signals/bars) que la real (0.729%).

| Métrica | Valor |
|---|---:|
| Mediana annual random | −7.0% |
| **Edge real (+8.4%) − random** | **+15.4%** ← supera +5% |

A's signal es genuinamente mejor que disparar al azar al mismo ritmo. Pero
este test **solo prueba que A.signal aporta** algo; **NO prueba que el edge
sobrevive en universos paralelos**, que es lo que falla en Capa 2.

## Criterios KEEP (todos requeridos)

| Criterio | Umbral | Resultado | Pass |
|---|---|---|:---:|
| Bootstrap p<0.05 (real) | <0.05 | 0.243 | ❌ |
| Mediana sintético > 0 | >0 | −1.3% | ❌ |
| ≥14/20 sintéticas positivas | ≥14 | 10/20 | ❌ |
| Edge synth − null > 5% | >+5% | +3.5% | ❌ |
| Edge real − random > 5% | >+5% | +15.4% | ✅ |

**4/5 fallidos → REJECT.**

## Interpretación honesta

1. **No hay evidencia de que la hipótesis "SOL = BTC apalancado" añada valor
   sistemáticamente**. La beta existe físicamente (correlación 0.73, vol 2×), pero
   tradear SOL en los timestamps de A no produce edge robusto.

2. **El edge real ~+8.4% annual en 2020-2025 es indistinguible del azar
   de la muestra histórica**:
   - Bootstrap p=0.24 (muy lejos de 0.05).
   - La mediana de 20 mundos sintéticos paralelos: −1.3%.
   - Real cae cerca de p55-p60 de la distribución sintética.

3. **El edge está concentrado en 2024** (PF 2.20 y 2.73 en H1/H2). Si SOL
   continúa lateral o bear (como 2022 y H1-H2 2026 hasta la fecha), la estrategia
   pierde. No es un edge transferible al futuro con confianza.

4. **El cross-check vs random (Capa 4) es engañoso si se mira solo**: A's
   signal en BTC es mejor que random (eso es lo que mide Capa 4), pero ese
   "mejor" no se traduce en edge transferible a SOL cuando se evalúa contra
   universos paralelos (Capa 2).

5. **Confirma el patrón documentado del proyecto**: 22 pares revalidados,
   17 REJECT, 3 marginales (ETC, BCH, UNI), 0 KEEP. SOL en este test fue
   ortodoxo (params derivados de A, no re-tuneados, validación 4-capas) y
   sale REJECT como todos los demás altcoins. **El único par con edge real
   demostrable sigue siendo BTC vía V2 (A+F)**.

## SELF-AUDIT — chequeos anti-overfitting aplicados

- ✅ Cutoff 2025-12-31 aplicado inmediatamente al cargar SOL, BTC 4h, BTC 1d y funding.
- ✅ Una posición a la vez: `run_backtest` salta a `i + bars + 1` tras cada trade.
- ✅ Sin look-ahead intrabar: `simulate` chequea SL contra peak heredado ANTES de
  actualizar peak con la vela actual. Espejo exacto del simulador honesto de A.
- ✅ MTF anti look-ahead: `bull_1d` con `.shift(1)` antes de `reindex(method='ffill')`
  (heredado de A.prepare_data).
- ✅ Rolling corr SOL-BTC con `.shift(1)` antes de usarse en signal.
- ✅ Donchian high con `.shift(1)` (en A.prepare_data).
- ✅ Funding z-score con `.shift(1)` (en A.prepare_data).
- ✅ PARAMS frozen: corr_min=0.5, trail_atr_mult=2.5, floor=4%, ceiling=10%,
  max_bars=60. Derivados a priori de A y de la vol ratio. **No hubo grid search**.
- ✅ Cross-check random signal con la MISMA tasa de disparo: protocolo correcto.
- ✅ Synthetic conjunto SOL+BTC para preservar la correlación local (preserva
  la hipótesis y evita el degenerate case de corr=0 sobre SOL shuffleado).
- ✅ Null relajado en corr_min para que el shuffle de SOL pueda generar trades
  (de otro modo serían 0 trades por construcción).
- ✅ Walk-forward 12 semestres con purga 14d (gap ≥ max_bars*4h=10d).
- ✅ Bootstrap 3000 iter sobre pnl por trade.
- ✅ SOL no existía pre-2020-09 → fold 2020-H1 marcado `sin datos`, no inflado.

## Archivos

- `strategy.py` — implementación auto-contenida (PARAMS, prepare_data, signal, simulate, run_backtest).
- `train.py` — pipeline 4-capas (WF + bootstrap + sintético conjunto + null + random).
- `results.json` — resultados completos serializados.

## Conclusión

El proyecto ya determinó (`VERDICTO_FINAL.md`) que el único edge demostrable
es **V2 BTC (A + F_BTC)**, con bootstrap p=0.031. Agent M intentó extender ese
edge a SOL por la vía del **beta-leveraged** (no predecir SOL, solo amplificar
A vía la correlación SOL-BTC). El intento **no sobrevive el test sintético
honesto**: los retornos son indistinguibles del azar de la muestra histórica.

Resultado **consistente con todos los demás intentos de añadir altcoins al
portfolio**: edge BTC no se transfiere mediante beta. Si el usuario quiere
exposición SOL, lo defendible sigue siendo **spot DCA** sin algoritmo, no esta
estrategia ni ninguna otra de las probadas.
