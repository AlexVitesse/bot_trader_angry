# Agent N — SOL Vol-Compression Breakout, WIDE Trail (scaled to SOL vol)

> Ronda 5 — SOL. Adaptación del mecanismo F (BTC vol-compression breakout) a SOL
> con trailing **proporcional a la volatilidad real de SOL** (~2x BTC, ~1.5x ETH).
> Cutoff inviolable 2025-12-31. Motor honesto (sin look-ahead intrabar, una
> posición a la vez).
> Fecha: 2026-05-19

---

## TL;DR honesto

| Capa | Métrica | Resultado | Pasa? |
|------|---------|-----------|-------|
| Real SOL — WF (12 sem) | folds OK | 7/11 (datos), 7/12 total | **Sí** (≥6/10) |
| Real SOL — backtest global | annual / PF / DD / N | +9.8% / 1.50 / 16.8% / 112 | OK direccionalmente |
| Bootstrap p (real) | p<0.05 | **p=0.078** | **NO** (marginal) |
| Sintéticas 20 series | ≥14/20 positivas | 11/20 | **NO** |
| Sintéticas — mediana annual | >0 | +1.1% | Sí (apenas) |
| Edge vs null (sintético vs shuffle) | >5% | **-0.7%** | **NO** |
| Cross-check tight vs wide | wide > tight | Tight gana p=0.047 vs Wide p=0.078 | (ver §4) |

**Veredicto: REJECT.** El mecanismo vol-compression breakout en SOL tiene
métricas sample-positivas (PF 1.50, annual +9.8%, WF 7/11) pero NO supera las
4 capas de validación. Específicamente:

1. El bootstrap p=0.078 es marginal — no se distingue del azar al umbral 0.05.
2. Las sintéticas dan mediana ~0 — el edge desaparece cuando se rebarajan
   bloques de mercado.
3. **Edge vs null = -0.7%** — el sistema gana lo mismo (o peor) sobre returns
   shuffleados que sobre el mercado real. Es ruido.
4. **El trailing wide NO mejora sobre el trailing tight** — la hipótesis del
   prompt (que "wide es proporcional a SOL vol") no se sostiene en los datos.

**Conclusión principal: el mecanismo nunca tuvo edge real en SOL. El trailing
tight 0.8% en el documento original era bug-inflado; el wide 2.5% honesto
tampoco genera edge significativo.** Ambos versiones operan en la zona de
ruido. La diferencia entre tight y wide es pequeña y no aporta veredicto
diferenciado — es lo mismo al final.

---

## 1. Diseño y parámetros

### 1.1 Tesis (heredada de F + ajuste de escala)

Si el mecanismo vol-compression-breakout tiene edge en BTC (PF 1.33, F),
SOL podría heredarlo SI:

1. La microestructura de SOL respeta el patrón (BB squeeze → expansion).
2. El trailing está calibrado a la vol REAL de SOL (no a la de BTC).

**Hipótesis sometida a test**: trail_atr_factor=0.50, trail_floor=0.025 (vs
F's 2.0 y 0.020) es la calibración proporcional para SOL.

> Nota sobre "vs F's 0.20": el prompt usa "factor 0.50 vs F's 0.20".
> Reviewando agent_F/strategy.py, F usa `trail_atr_mult=2.0` (no 0.20).
> Interpretación: el prompt se refiere a la "expansión efectiva" del SL
> (~ATR_pct × multiplier ≈ 2% × 2.0 = 4% en BTC, donde el piso 2% domina).
> La intención clara es: SL más amplio en SOL para no morir de ruido. Adopté
> `trail_atr_factor=0.50` con piso=0.025 y techo=0.080.

### 1.2 Parámetros FROZEN (a priori, no tuneados a SOL)

```python
PARAMS = {
    'commission': 0.0005,
    # Compresión (idéntica a F)
    'bb_n': 20, 'bb_k': 2.0,
    'percentile_lookback': 100,
    'compression_percentile': 0.20,
    'compression_min_bars': 3,
    # Breakout
    'breakout_n': 12,
    'vol_ratio_n': 20,
    'vol_ratio_min': 1.0,           # ligeramente más laxo que F (1.2)
                                    # porque SOL tiene vol baseline +alta
    # Régimen daily
    'ema_fast_1d': 50, 'ema_slow_1d': 200,
    'regime_filter_enabled': True,
    # Trailing WIDE — ESCALADO A SOL
    'atr_n': 14,
    'trail_atr_factor': 0.50,       # vs F's 2.0 (interpretado, ver arriba)
    'trail_floor_pct': 0.025,       # 2.5% piso (vs F 2.0%)
    'trail_ceiling_pct': 0.080,     # 8% techo (vs F 5.5%)
    'max_bars': 30,                 # 5 días (vs F's 48 = 8 días)
    # Direccionalidad
    'enable_long': True, 'enable_short': True,
    'cutoff_date': '2025-12-31',
    'min_bars_warmup': 250,
}
```

### 1.3 Diferencias clave vs F BTC

| Param | F BTC | N SOL | Justificación |
|-------|-------|-------|---------------|
| `trail_atr_factor` (mult) | 2.0 | 0.50 | F's ATR × 2.0 ≈ 4% en BTC; piso domina. En SOL queremos que el ATR no infle tanto el SL (3.55% × 0.5 = 1.78%, piso 2.5% domina) |
| `trail_floor_pct` | 0.020 | 0.025 | Escalado +25% por vol-ratio SOL/BTC |
| `trail_ceiling_pct` | 0.055 | 0.080 | Escalado +45% (espacio para SOL spikes) |
| `vol_ratio_min` | 1.2 | 1.0 | SOL tiene volatilidad de fondo alta, exigir vol_ratio>=1.2 dropea breakouts genuinos |
| `max_bars` | 48 | 30 | SOL se mueve más rápido; menos tiempo de exposición innecesaria |

**Ningún param fue elegido por grid search en SOL.** Cada variación tiene
justificación a priori basada en propiedades estructurales de SOL (vol, BB
width, time-to-resolve).

---

## 2. Resultados — Capa 1 (Real SOL 2020-2025)

### 2.1 Diagnóstico de datos

- ATR% mean: **3.63%** (vs BTC ~2%, ETH ~2.4%) → confirma SOL 1.8x BTC, 1.5x ETH
- BB width mean: **15.28%** (vs BTC/ETH ~7%) → SOL tiene bandas mucho más anchas
- Bars en compresión sostenida: 2123 / 11689 (18.2%) — la señal de compresión
  dispara con frecuencia razonable

### 2.2 Walk-forward 12 semestres (purga 14d)

| Período | N | WR | PF | Total | DD | OK |
|---------|---|----|-----|-------|-----|-----|
| 2020-H1 | 0 | — | — | — | — | [no-sig] |
| 2020-H2 | 3 | 33.3% | 0.42 | -3.1% | 5.1% | [small_sample] |
| 2021-H1 | 16 | 50.0% | 1.30 | +5.3% | 9.0% | + |
| 2021-H2 | 6 | 33.3% | 2.06 | +8.3% | 7.7% | + |
| 2022-H1 | 11 | 45.5% | 1.48 | +3.0% | 4.1% | + |
| 2022-H2 | 8 | 37.5% | 0.46 | -5.2% | 5.3% | - |
| 2023-H1 | 10 | 40.0% | 1.60 | +5.0% | 8.0% | + |
| 2023-H2 | 8 | 62.5% | 1.55 | +3.3% | 3.4% | + |
| 2024-H1 | 12 | 50.0% | 1.16 | +1.6% | 5.1% | - |
| 2024-H2 | 10 | 40.0% | 3.23 | +13.8% | 2.4% | + |
| 2025-H1 | 9 | 44.4% | 1.19 | +1.4% | 4.7% | - |
| 2025-H2 | 11 | 36.4% | 1.46 | +4.6% | 4.9% | + |

**WF 7/11 folds con datos.** Cumple criterio ≥6/10 para cross-pairs.

(SOL no tiene datos antes de 2020-08, por eso el primer fold 2020-H1 está
vacío y se excluye del denominador.)

### 2.3 Backtest global 2020-01 → 2025-12

```
N=112  WR=44.6%  PF=1.50  total=+62.1%  annual=+9.8%  DD=16.8%
sharpe-like=0.14  months=62.2
```

### 2.4 Por dirección

| Dir | N | WR | PF | Total | avg_pnl |
|-----|---|----|----|-------|---------|
| **LONG** | 82 | **47.6%** | **1.78** | **+77.8%** | +0.77% |
| SHORT | 30 | 36.7% | 0.70 | -8.8% | -0.29% |

**LONG tiene edge real (PF 1.78). SHORT no funciona** (PF 0.70). Esto es
**consistente con el historial de SOL en todas las versiones**: SHORT en SOL
no ha funcionado nunca. La hipótesis del prompt ("SHORT en BEAR captura
caídas con trail wide") **NO se valida**. El BEAR de SOL es violento pero
los bounces matan stops igual con trail 2.5%.

### 2.5 Bootstrap (3000 iter)

```
p-value(retorno <= 0): 0.0783    -> NO SIGNIFICATIVO
retorno mediano resampled: +59.8%
retorno p5: -6.3%   p95: +191.8%
```

p=0.078 está por encima del umbral 0.05. El CI [-6%, +192%] es ENORME — la
estrategia podría perder 6% o ganar 192% en 112 trades por puro azar.

---

## 3. Capa 2 — 20 sintéticas (block bootstrap 24-bar)

Distribución de annual_return en 20 series sintéticas:

```
mediana = +1.1%
media   = +1.0%
p25-p75 = [-2.9%, +3.7%]
p5-p95  = [-6.0%, +8.5%]
# series con annual > 0:    11/20    <-- necesita >=14
# series con annual > 10%:  1/20     <-- necesita varias
```

**Solo 11/20 sintéticas son positivas.** Necesita ≥14/20 para considerar que
el edge es robusto al reshuffling de bloques. **Falla esta capa.**

El real SOL (+9.8%) cae fuera del p95 sintético (+8.5%), pero esto es síntoma
de SAMPLE LUCKY, no de edge:
- El BTC V2 (único KEEP del proyecto) también cae por encima del p95 sintético.
- La diferencia es que V2 tiene mediana sintética **+7.9%** mientras N tiene **+1.1%**.

**La distribución sintética de N está centrada en 0 — el edge no existe en la
estructura de SOL, solo en este particular sample 2020-2025.**

---

## 4. Capa 3 — Null hypothesis (shuffle returns)

Aplicamos la estrategia a 10 series con returns shuffleados de SOL (preserva
distribución, destruye estructura temporal):

| seed | N | annual | WR |
|------|---|--------|-----|
| 0-9 | 70-101 | -11.5% a +13.2% | 28.7%-44.1% |

Mediana null annual: **+1.8%**
**Edge sintético vs null: -0.7%**

**El sistema gana lo MISMO (o ligeramente menos) sobre returns aleatorios
shuffleados que sobre el mercado real.** Esto es la prueba más demoledora:
la estrategia no extrae información del orden temporal de SOL. Cualquier
"edge" observado en el real es coincidencia.

---

## 5. Capa 4 — Cross-check tight vs wide (la pregunta clave)

| Variante | trail_atr_factor | trail_floor | N | WR | PF | annual | DD | bootstrap p |
|----------|-----------------:|-----------:|--:|---:|---:|-------:|---:|------------:|
| **TIGHT** | 0.30 | 0.008 (0.8%) | 122 | 40.2% | 1.54 | +7.8% | 14.5% | **0.047 ✅** |
| **WIDE** | 0.50 | 0.025 (2.5%) | 112 | 44.6% | 1.50 | +9.8% | 16.8% | 0.078 |

### Hallazgo crítico

La hipótesis del prompt era: **"trail wide funciona mejor en SOL"**.

**La hipótesis NO se sostiene**. Tight 0.8% genera p=0.047 (apenas
significativo), wide 2.5% queda en p=0.078. Tight tiene **menos DD** (14.5%
vs 16.8%) y **más trades** (122 vs 112), aunque el wide tiene WR superior y
PF marginalmente parecido.

### ¿Significa esto que tight 0.8% es el correcto?

**NO**. Tres razones para NO declarar "tight gana":

1. **El test sintético/null no se hizo con tight** — si el wide falla las
   capas 2-3, no hay motivo para creer que tight sobrevivirá. Hicimos
   cross-check de PARAMS, no de validación completa.

2. **Tight 0.8% sigue estando en el régimen del bug original** — la
   diferencia tight vs wide en producción real (sin look-ahead intrabar) es
   marginal. El tight ya pierde el "halo" que tenía en el doc original
   (PF 2.56, WR 54.6%, DD 9.6% — todo eso era bug-inflado).

3. **Multiple comparison**: probamos 2 variantes; el "ganador" tight
   p=0.047 tiene Bonferroni adjusted p ≈ 0.094. **Vuelve a no significativo
   con corrección estándar de multiple testing.**

### La interpretación honesta del cross-check

> El mecanismo vol-compression breakout en SOL produce métricas
> sample-positive (PF 1.4-1.5, annual +8-10%, WR 40-45%) **independientemente
> de la calibración del trailing** entre 0.8% y 2.5%. La elección del
> trailing no es la fuente del edge; es decorativa.
>
> Como el mecanismo no supera las capas sintéticas/null, **no hay edge
> robusto que defender — sea con trail tight o wide.**

---

## 6. SELF-AUDIT

### 6.1 Bugs prohibidos — chequeo

| Bug | Test | Resultado |
|-----|------|-----------|
| Trades solapados | iterar trades ordenados, verificar `entry[i] > exit[i-1]` | **0 violaciones** |
| Look-ahead intrabar trailing | en cada vela: salida vs SL HEREDADO antes de update peak | implementado y verificado |
| MTF sin shift(1) | EMA50/200 daily | `.shift(1)` explícito |
| Percentiles con look-ahead | BB_width.rolling(100).rank(pct=True) | `.shift(1)` |
| N-bar high/low | rolling.max/min().shift(1) | OK |
| Cutoff respetado | df.index.max() <= 2025-12-31 | sol_feat.index.max() = 2025-12-31 |
| Selection bias params | params FROZEN a priori | sí (justificación arriba) |

### 6.2 Sanity checks

| Métrica | Valor | Veredicto |
|---------|-------|-----------|
| PF (wide) | 1.50 | OK (1.2-1.5 honesto) |
| WR (wide) | 44.6% | OK (40-50% típico de trend) |
| DD (wide) | 16.8% | OK (<25% objetivo) |
| Sharpe | 0.14 | Bajo |
| Boot p | 0.078 | NO significativo |

### 6.3 Selection bias a nivel estrategia

Esta estrategia (vol-compression breakout SOL wide trail) fue **dada por el
prompt** — no la elegí entre N alternativas. No hay selection bias a nivel
ESTRATEGIA. El cross-check tight vs wide es 1 comparación adicional; el
"ganador" tight pierde significancia con Bonferroni (×2 → p≈0.094).

### 6.4 Limitaciones honestas

1. **SOL tiene 5.5 años de datos (2020-08 → 2025-12)** — menos historia que
   BTC. Solo 11 folds WF con datos, en lugar de 12.
2. **N=112 trades es modesto** — CI bootstrap muy ancho.
3. **SHORT NO funciona en SOL** (PF 0.70) — confirma el historial del
   proyecto. Una versión LONG-only sería más limpia, pero ya es selection
   bias (elegir el subset que ganó).
4. **2022-H2, 2024-H1, 2025-H1 todos perdedores** — la estrategia tiene
   periodos malos sustanciales.

---

## 7. Conclusión

### El mecanismo NO tiene edge real en SOL

Conforme a las 4 capas obligatorias:

| Capa | Resultado |
|------|-----------|
| 1. Real WF + bootstrap | WF 7/11 OK pero p=0.078 marginal |
| 2. 20 sintéticas | 11/20 positivas (necesita 14) |
| 3. Null hypothesis | edge vs null **-0.7%** |
| 4. Tight vs wide | tight gana marginal pero pierde Bonferroni |

**Conclusión: REJECT.** El mecanismo vol-compression breakout en SOL no
distingue del azar bajo medición honesta. Las métricas sample-positivas
(PF 1.50, annual +9.8%) son consistentes con varianza muestral, no con edge
estructural.

### Separando "mecanismo no funciona" vs "trail mal calibrado"

> El prompt pidió distinguir entre:
> (a) "el mecanismo funciona pero el trail estaba mal"
> (b) "el mecanismo nunca funcionó"

La respuesta es **(b)**: el mecanismo nunca funcionó en SOL.

Evidencia:
1. Tight 0.8% y wide 2.5% generan métricas casi idénticas (PF 1.54 vs 1.50)
   en el motor honesto. La calibración del trail NO es el problema.
2. El edge vs null negativo (-0.7%) demuestra que la información temporal
   de SOL no es extraída por la señal vol-compression breakout.
3. Las sintéticas centran en 0 — no hay edge estructural transferible.

### ¿Qué del prompt SÍ se confirmó?

1. El trailing tight 0.8% original del doc V15 ERA bug-inflado (el motor
   honesto da PF 1.54, no 2.56-15.04 declarado).
2. SOL tiene vol 2x BTC y 1.5x ETH (confirmado: ATR 3.63% vs 2% / 2.4%).
3. La hipótesis "calibrar trail proporcional" es razonable a priori — pero
   no resuelve el problema porque el problema no era el trail.

### Mensaje al user

> **SOL no es operable a 4h con vol-compression breakout, ni con trail tight
> ni wide**. Esto **confirma** los veredictos del proyecto: solo BTC V2
> tiene edge estadísticamente defendible. Cualquier intento de incluir SOL
> en el portfolio directional requeriría:
> - Mecanismo genuinamente NUEVO (no vol-breakout, no breakout simple,
>   no ML — todos probados y rechazados)
> - Mismo protocolo 4 capas
> - Bootstrap p<0.05 reproducible + edge vs null > 5%

---

## 8. Resultados JSON

```json
{
  "agent": "N",
  "strategy_name": "SOL vol breakout WIDE trail (scaled to SOL volatility)",
  "in_sample_wf": "7/11",
  "in_sample_pf": 1.50,
  "in_sample_wr": 0.446,
  "annual_return": 0.098,
  "bootstrap_pvalue": 0.078,
  "synth_median_annual": 0.011,
  "synth_n_positive": "11/20",
  "edge_vs_null": -0.007,
  "trail_tight_vs_wide_comparison": "tight 0.8%: PF 1.54 p=0.047 | wide 2.5%: PF 1.50 p=0.078",
  "n_trades": 112,
  "max_dd": 0.168,
  "key_insight": "El mecanismo vol-compression breakout en SOL NO tiene edge real. Tight y wide trail dan métricas casi idénticas (PF 1.54 vs 1.50, p=0.047 vs 0.078). El 'edge' observado en el sample es consistente con varianza muestral — edge vs null -0.7%, sintéticas 11/20 positivas. No es el trail el problema; es el mecanismo. Tight 0.8% del doc original estaba bug-inflado (PF 2.56 declarado vs PF 1.54 honesto).",
  "deliverable_files": ["strategy.py", "train.py", "results.json", "README.md"]
}
```

---

## 9. Archivos

- `strategy.py` — código auto-contenido (PARAMS, prepare_data, signal, simulate, run_backtest, metrics + PARAMS_TIGHT)
- `train.py` — pipeline 4 capas (WF, bootstrap, sintéticas, null, tight vs wide cross-check)
- `results.json` — métricas serializadas
- `README.md` — este documento
