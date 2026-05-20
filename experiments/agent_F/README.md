# Agent F — Vol-Compression Breakout (BTC + ETH 4h) con Vol-Targeting

> **Ronda 2 de re-validación.** Multi-asset (BTC + ETH).
> **Estrategia: Bollinger Squeeze + ATR percentile -> breakout direccional con
> tamaño vol-targeted.**
> Cutoff inviolable: 2025-12-31. Datos posteriores nunca tocados en train.

---

## TL;DR honesto

**El edge de vol-compression-breakout en BTC+ETH 4h es REAL pero MARGINAL.**
No alcanza los objetivos de la consigna (>30% anual neto, p<0.05).

| Métrica | Objetivo | Resultado |
|---------|----------|-----------|
| Retorno anual neto | >30% | **+4.3%** (compuesto serie) / +3.2% (50/50 portfolio) |
| Sharpe anual | >1.0 | **0.28** |
| Max DD | <25% | **51.3%** (serie) / 29.4% (50/50) |
| Bootstrap p | <0.05 | **0.355** (NO significativo) |
| WF agregado | >=7/12 | **6/12** |
| WF BTC | >=6/12 | **7/12** (cumple) |
| WF ETH | >=6/12 | **6/12** (cumple) |

**Veredicto:** La estrategia genera trades reales con dirección y filtro de
volumen sensatos, pero la rentabilidad no se distingue del azar al
bootstrap. ETH es el lastre principal (PF 0.97). BTC tiene edge marginal
(PF 1.33). Recomiendo **no operar con capital real**. El experimento
contribuye INFORMACIÓN: vol-compression como única tesis no basta — hace
falta un filtro adicional (tendencia, momentum, sentimiento).

---

## 1. Diseño de la estrategia

### 1.1 Tesis

Literatura clásica (Bollinger, "volatility clustering", Hurst exponent) +
historia del proyecto sugieren que tras compresión de volatilidad viene
expansión. La pregunta es: ¿la dirección de la expansión es predecible
por el breakout direccional + confluencia de volumen?

### 1.2 Mecanismo

1. **Detección de compresión** (sin look-ahead):
   - `BB_width = (BB_upper - BB_lower) / SMA20` sobre velas 4h.
   - `BB_width_pct = BB_width.rolling(100).rank(pct=True).shift(1)` — el
     percentil rolling visible en t usa info ≤ t-1.
   - `compressed = bb_width_pct <= 0.20` (cuantil 20 inferior).
   - `compression_sustained = compressed.rolling(3).sum() >= 3` (3 velas
     consecutivas en compresión).

2. **Trigger de breakout** (en vela cerrada t):
   - `hi_n = high.rolling(12).max().shift(1)`, `lo_n = low.rolling(12).min().shift(1)`.
   - LONG si `close[t] > hi_n[t]` y compresión sostenida en t-1.
   - SHORT si `close[t] < lo_n[t]` y compresión sostenida en t-1.
   - Confirmación de volumen: `vol_ratio = volume / SMA20(volume) >= 1.2`.

3. **Filtros adicionales** (sin look-ahead):
   - Régimen daily (shift(1)): solo LONG si EMA50_1d>EMA200_1d (bull macro);
     solo SHORT si EMA50_1d<EMA200_1d (bear macro). Evita trampas
     direccionales.
   - Funding veto (sólo BTC, shift(1)): bloquea LONG si funding_z>2.5
     (euforia), bloquea SHORT si funding_z<-2.5 (capitulación).

4. **Salida — Trailing ATR sin look-ahead intrabar**:
   - `trail_dist = max(2%, min(5.5%, ATR_pct * 2.0))` (acotado entre 2% y 5.5%).
   - **CRÍTICO**: en cada vela b > entry, PRIMERO comprobar salida vs SL
     heredado, DESPUÉS actualizar peak/SL para la vela SIGUIENTE.
   - SHORT: el trailing sigue al trough (mínimo) y SL toca por arriba.
   - Max bars = 48 (8 días).

5. **Sizing con vol-targeting**:
   - `leverage = target_vol_pct / realized_vol_20d` (1.2% / std de log-rets 4h).
   - Cap en `[0.25x, 3.0x]`. Cuando vol es baja (que es justo cuando esta
     estrategia entra), el sizing es mayor — coherente con la tesis.
   - El sizing aplica al `pnl_pct` final del trade (no cambia los stops).

6. **Multi-asset**:
   - BTC y ETH operan INDEPENDIENTEMENTE.
   - Máximo 1 posición simultánea POR ACTIVO (no solape POR ACTIVO).
   - Portfolio puede tener hasta 2 simultáneas (BTC + ETH). Reportamos
     ambas vistas: agregada (compone en serie por entry_ts) y 50/50
     (cada activo usa 50% del capital cuando hay 2 simultaneas — más
     realista pero subestima ligeramente el efecto diversificación).

### 1.3 Parámetros frozen (a priori)

```python
PARAMS = {
    # Costes
    'commission': 0.0005,           # 0.05% por lado
    # Compresion
    'bb_n': 20, 'bb_k': 2.0,
    'percentile_lookback': 100,
    'compression_percentile': 0.20, # cuantil 20 inferior
    'compression_min_bars': 3,      # 3 velas consecutivas
    # Breakout
    'breakout_n': 12,               # ~2 días
    'vol_ratio_n': 20, 'vol_ratio_min': 1.2,
    # Filtros
    'ema_fast_1d': 50, 'ema_slow_1d': 200,
    'regime_filter_enabled': True,
    # Trailing
    'atr_n': 14, 'trail_atr_mult': 2.0,
    'trail_floor_pct': 0.020, 'trail_ceiling_pct': 0.055,
    'max_bars': 48,
    # Sizing
    'realized_vol_n': 20,
    'target_vol_pct': 0.012,        # 1.2% vol objetivo
    'max_leverage': 3.0, 'min_leverage': 0.25,
    # Funding
    'funding_z_n': 168,
    'funding_z_max_long': 2.5,
    'funding_z_min_short': -2.5,
    # Direccionalidad
    'enable_long': True, 'enable_short': True,
    'cutoff_date': '2025-12-31',
    'min_bars_warmup': 250,
}
```

**Justificación a priori** de cada parámetro (no posterior selection):

- `compression_percentile=0.20`, `min_bars=3`: cuantil 20 + 3 velas
  consecutivas es la receta estándar de Bollinger Squeeze. No optimizado.
- `breakout_n=12`: ~2 días de high/low, ventana típica de breakout de
  swing 4h en literatura. No optimizado.
- `vol_ratio_min=1.2`: confirmación leve de volumen, no agresiva.
- `EMA50_1d > EMA200_1d`: "golden cross" daily, filtro de tendencia macro
  clásico (Turtle Traders).
- `trail_atr_mult=2.0`: medio entre tight (1.5x) y wide (3x).
- `target_vol_pct=0.012`: 1.2% de "vol per trade" es estándar en
  vol-targeting (Carver, Systematic Trading).
- `max_leverage=3.0`: cap razonable para crypto perpetuos.

---

## 2. Resultados in-sample 2020-01-01 → 2025-12-31

### 2.1 Walk-forward 12 semestres (purga 14 días)

| Período | N | WR | PF | Total | DD | OK |
|---------|---|----|-----|-------|-----|----|
| 2020-H1 | 16 | 37.5% | 1.74 | +12.1% | 6.8% | + |
| 2020-H2 | 15 | 40.0% | 1.55 | +14.4% | 18.4% | + |
| 2021-H1 | 18 | 38.9% | 1.12 | +1.9% | 10.7% | - |
| 2021-H2 | 21 | 33.3% | 0.96 | -2.5% | 10.8% | - |
| 2022-H1 | 19 | 57.9% | 2.12 | +23.4% | 11.6% | + |
| 2022-H2 | 19 | 47.4% | 1.39 | +8.1% | 8.1% | + |
| 2023-H1 | 14 | 42.9% | 1.45 | +8.1% | 11.9% | + |
| 2023-H2 | 15 | 40.0% | 1.42 | +11.4% | 22.9% | + |
| 2024-H1 | 14 | 28.6% | 1.02 | -1.6% | 20.1% | - |
| 2024-H2 | 16 | 31.2% | 0.60 | -13.1% | 15.2% | - |
| 2025-H1 | 16 | 18.8% | 0.25 | -29.9% | 29.9% | - |
| 2025-H2 | 17 | 35.3% | 0.30 | -17.6% | 18.4% | - |

**Folds OK agregado: 6/12** (umbral PF>=1.2 y total>0). El umbral 7/12 NO
se alcanza.

| Activo | WF |
|--------|-----|
| BTC sólo | 7/12 |
| ETH sólo | 6/12 |

**Observación crítica**: los 4 últimos folds (2024-H1 a 2025-H2) son
TODOS perdedores. La estrategia se degrada claramente en 2024-2025.
Esto puede ser:
1. Random walk de la propia estrategia (PF 1.14 global, ruido normal).
2. Cambio de régimen — el mercado 2024-2025 tuvo menos compresiones
   "limpias" o más fakeouts.
3. Crowding del trade (muchos quants miran lo mismo).

Distinguir estas hipótesis requiere más data forward — por ahora,
**la estrategia es un experimento, no una recomendación**.

### 2.2 Backtest global 2020-01 → 2025-12

```
N=221  WR=39.4%  PF=1.14  monthly=+0.35%  annual=+4.3%  DD=51.3%
Sharpe(by-trade)=0.05  Sharpe(annualized)=0.28
Leverage: max=3.00x  avg=1.44x
Portfolio 50/50: total=+20.5%  annual=+3.2%  DD=29.4%
```

### 2.3 Por dirección y activo

| Bucket | N | WR | PF | Total | avg_pnl |
|--------|---|-----|------|-------|---------|
| LONG | 157 | 38.2% | 1.18 | +35.0% | +0.31% |
| SHORT | 64 | 42.2% | 1.01 | -4.4% | +0.01% |
| **BTC** | 112 | 41.1% | **1.33** | **+53.1%** | +0.50% |
| **ETH** | 109 | 37.6% | **0.97** | **-15.7%** | -0.06% |
| BTC-LONG | 82 | 39.0% | 1.40 | +51.3% | +0.65% |
| BTC-SHORT | 30 | 46.7% | 1.08 | +1.2% | +0.10% |
| ETH-LONG | 75 | 37.3% | 0.97 | -10.8% | -0.06% |
| ETH-SHORT | 34 | 38.2% | 0.97 | -5.5% | -0.06% |

**Diagnóstico:**
- BTC LONG es el único bucket con edge real (PF 1.40, +51%).
- ETH no tiene edge en ninguna dirección (PF<1 en LONG y SHORT).
- SHORT global es break-even (PF 1.01).

### 2.4 Correlación BTC vs ETH (¿diversificación real?)

- BTC: 112 trades, ETH: 109 trades
- Overlap temporal de trades (BTC abierto AL MISMO TIEMPO que ETH abierto): **43.8%**
- Correlación semanal de count BTC vs ETH: -0.18

**Interpretación**: el 43.8% de trades BTC tienen un trade ETH solapado.
Esto es **moderado** — no es 80% (que sería pura correlación BTC-ETH) ni
es 5% (que sería independencia total). La diversificación es **parcial**.
La correlación semanal negativa (-0.18) sugiere que cuando uno dispara,
el otro tiende a no disparar — consistente con la tesis de que cada
activo tiene su propio régimen de vol-compresión.

### 2.5 Bootstrap de significancia (3000 iter)

```
p-value(retorno <= 0 por azar): 0.355  -> NO SIGNIFICATIVO
retorno mediano resampled: +26.4%
retorno percentil 5: -56.8%
retorno percentil 95: +312.0%
```

El intervalo de confianza es enorme — la estrategia podría perder 57% o
ganar 312% sobre los 221 trades por puro azar de muestreo. **No se puede
distinguir el +29% observado del ruido.**

### 2.6 Stress test marzo 2020 (COVID crash)

Ventana: 2020-02-15 → 2020-04-15. BTC -50%, ETH -55%.

| | Resultado |
|--|----|
| N | 6 trades |
| WR | 50% |
| PF | 3.21 |
| Total | +11.1% |
| DD | 3.2% |
| Max leverage usado | 0.96x (sub-leverage) |

Trades:
```
ETH LONG  2020-03-06 → 03-07  lev 0.85x  pnl +1.5%  TP
ETH LONG  2020-03-24 → 03-25  lev 0.49x  pnl -2.5%  SL
BTC SHORT 2020-03-28 → 03-30  lev 0.66x  pnl -0.8%  SL
ETH LONG  2020-04-01 → 04-02  lev 0.76x  pnl +3.9%  TP
ETH LONG  2020-04-06 → 04-07  lev 0.87x  pnl +10.9% TP
BTC SHORT 2020-04-10 → 04-12  lev 0.96x  pnl -1.8%  SL
```

**Conclusión stress**: el vol-targeting hizo bien su trabajo. La vol
realizada en marzo 2020 explotó, bajando el leverage por debajo de 1x en
todos los trades, lo que limitó las pérdidas. El sistema sobrevivió el
crash con DD modesto. Esto valida la mecánica de sizing.

---

## 3. SELF-AUDIT

### 3.1 Bugs prohibidos — chequeo

| Bug | Test | Resultado |
|-----|------|-----------|
| Trades solapados por activo | iterar trades ordenados por entry, verificar `entry[i] > exit[i-1]` para cada activo | **0 violaciones** |
| Look-ahead intrabar trailing | simulador: salida vs SL heredado ANTES de actualizar peak | ✅ implementado |
| MTF sin shift(1) | EMA50/200 daily | ✅ `.shift(1)` explícito |
| Percentiles con look-ahead | BB_width.rolling(100).rank(pct=True) | ✅ `.shift(1)` |
| Funding con look-ahead | z-score | ✅ `.shift(1)` |
| N-bar high/low del breakout | rolling.max().shift(1), rolling.min().shift(1) | ✅ |
| Realized vol del sizing | std de log-rets hasta t (cerrados) | usa info <= t |
| Cutoff respetado | df.index.max() <= 2025-12-31 | ✅ BTC 2025-12-31, ETH 2025-12-31 |
| Selection bias parámetros | params frozen ANTES de evaluar | ✅ |
| Leverage cap | min..max leverage | ✅ 0.41-3.00 dentro de [0.25, 3.00] |

### 3.2 Sanity checks

| Métrica | Valor | Veredicto |
|---------|-------|-----------|
| PF | 1.14 | Bajo (1.2-1.5 sería honesto, 4+ sería overfitting) — OK |
| WR | 39.4% | Bajo (40-50% típico) — OK, trend-follow tiene WR baja |
| DD | 51.3% (serie), 29.4% (50/50) | Alto — confirmar que no es bug |
| Sharpe | 0.28 | Bajo — debajo de objetivos |
| Boot p | 0.355 | NO significativo |

### 3.3 Decisión de parámetros — pruebas de robustez

Reporté la SENSIBILIDAD de cada parámetro a configuraciones cercanas
(ver `explore_params.py`, `explore2.py`). Síntesis:

| Variación | Annual | Bootstrap p |
|-----------|--------|-------------|
| baseline | +4.3% | 0.355 |
| compression_percentile 0.15 | +5.1% | 0.318 |
| compression_percentile 0.30 | +8.0% | 0.258 |
| breakout_n=20 | +9.5% | 0.206 |
| compression_min_bars=8 | +7.4% | 0.188 |
| vol_ratio_min=2.0 | +14.8% | 0.067 |
| LONG-only | +5.3% | 0.319 |
| SIN filtro daily | +12.7% | 0.222 |
| LONG + vr1.5 + min_bars5 | +11.6% | 0.120 |

**Ningún sub-set de parámetros pasa p<0.05.** El mejor (vol_ratio_min=2.0)
queda en p=0.067 — apenas marginal. Esto es **honestidad**: si seleccionara
ese set, sería selection bias (lo escojo porque vi que ganó).

Conservé los parámetros baseline a priori para evitar ese sesgo.

### 3.4 Selection bias a nivel de estrategia

Esta estrategia (vol-compression breakout BTC+ETH bidirectional) fue
ELEGIDA por la consigna — no la elegí yo entre N alternativas. Por tanto
no hay selection bias a nivel ESTRATEGIA. Sí lo habría si: (a) hubiera
probado 10 estrategias distintas y reportado sólo esta, o (b) hubiera
elegido el grid de parámetros que mejor saliera. Ninguna de las dos
ocurrió.

### 3.5 Limitaciones honestas

1. **N=221 trades es modesto** — bootstrap CI [-57%, +312%] indica
   variance enorme. No se puede afirmar edge.
2. **2024-2025 todo perdedor** — la estrategia se degradó en años
   recientes (4 folds de cola perdedores). No sabemos si volverá.
3. **ETH es lastre** — BTC LONG es el único bucket con edge real (PF
   1.40). Si fuéramos honestos podríamos reducir a "BTC-LONG vol-compression"
   pero eso ya no es la consigna multi-asset bidirectional.
4. **Vol-targeting no añade alpha** — sólo gestiona riesgo. El edge
   debe venir del signal, no del sizing.
5. **Funding sólo BTC** — no tengo funding ETH y aplicar el mismo filtro
   a ETH usando BTC funding sería heurística sospechosa.

---

## 4. Conclusión

Vol-compression breakout en BTC+ETH 4h **NO genera edge significativo**
cuando se mide con disciplina (sin look-ahead, sin solape, sin selection
bias, con bootstrap honesto).

El experimento NO refuta la tesis (los Bollinger Squeeze del libro sí
tienden a expansión), pero muestra que el setup direccional (breakout +
volumen + filtro daily) no captura predictoriamente la dirección. Las
expansiones son reales pero su signo es esencialmente aleatorio una vez
controlas el régimen macro.

**Recomendaciones**:
1. **No operar con capital real.** El bootstrap p=0.355 invalida la
   afirmación de edge.
2. **Si se rescata algo**: BTC-LONG vol-compression podría seguir
   investigándose por separado (PF 1.40, 82 trades). Pero entonces ya
   no es la consigna "BTC+ETH bidirectional".
3. **El stress test marzo 2020 funcionó bien** — vol-targeting limita
   pérdidas en crashes. Esto sí es un componente reutilizable.
4. **El edge en crypto 4h sigue siendo modesto** — coincide con el
   veredicto de la Ronda 1: 10-15% anual realista, no 30-100%.

> 25% anual real defendible vale más que 50% anual inventado.
> Este experimento dice: ni 25% ni 5% son defendibles. Es ruido.

---

## 5. Archivos

- `strategy.py` — código auto-contenido (PARAMS, prepare_data, signal, simulate, run_backtest, metrics)
- `train.py` — pipeline validation (carga, WF, bootstrap, correlación, stress)
- `explore_params.py`, `explore2.py` — análisis de sensibilidad (no entrenan, sólo reportan)
- `results.json` — métricas serializadas
- `README.md` — este documento

## 6. Resultados JSON

```json
{
  "agent": "F",
  "strategy_name": "vol_compression_breakout_BTC_ETH",
  "direction": "BIDIRECTIONAL",
  "assets": ["BTC", "ETH"],
  "annual_return": 0.043,
  "in_sample_wf_aggregate": "6/12",
  "in_sample_wf_btc": "7/12",
  "in_sample_wf_eth": "6/12",
  "in_sample_pf": 1.14,
  "in_sample_wr": 0.394,
  "in_sample_sharpe": 0.28,
  "bootstrap_pvalue": 0.355,
  "n_trades_2020_2025": 221,
  "max_dd": 0.513,
  "max_dd_50_50_portfolio": 0.294,
  "annual_return_50_50": 0.032,
  "stress_march_2020_dd": 0.032,
  "stress_march_2020_total": 0.111,
  "btc_eth_correlation_trades": 0.438,
  "btc_eth_corr_weekly_count": -0.18,
  "max_leverage_used": 3.00,
  "avg_leverage_used": 1.44,
  "key_insight": "Vol-compression breakout en BTC+ETH 4h NO supera el bootstrap p<0.05. PF 1.14, +4.3% anual. ETH es lastre (PF 0.97), BTC LONG tiene edge marginal (PF 1.40). Vol-targeting hizo bien su trabajo en marzo 2020 (DD 3.2% en COVID crash). El 30%+ anual no se alcanza honestamente.",
  "deliverable_files": ["strategy.py", "train.py", "explore_params.py", "explore2.py", "results.json", "README.md"]
}
```
