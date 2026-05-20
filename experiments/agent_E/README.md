# Agent E — Funding-Extremos Mean-Reversion (BTC 4h, bidireccional)

> Estrategia inspirada en literatura crypto-microestructura: cuando el
> funding rate de los perps se desvia mucho de su norma, hay desequilibrio
> direccional que **suele resolverse con reversion**. Edge distinto al de
> Agent A (trend) y al de Agent C (regime).
> Cutoff: 2025-12-31 (inviolable). NO mira 2026.

---

## 1. Fundamentacion teorica

**Mecanismo:**
- En Binance perp futures el funding se paga cada 8h (00:00, 08:00, 16:00 UTC).
- Cuando los longs estan sobrecargados, pagan carry positivo a los shorts.
  El desequilibrio crece hasta que (a) se cierra la posicion voluntariamente,
  o (b) un movimiento contrario fuerza liquidaciones. Ese momento de
  reversion es donde esta el edge.
- Simetrico para shorts sobrecargados (funding negativo extremo).

**Soporte en literatura:**
- Coleman (2022), *Funding Rate as Predictor of Crypto Returns* — z-scores
  extremos de funding predicen rebotes 24-72h adelante.
- Glassnode & CryptoQuant: multiples blog-posts sobre "funding cooldown"
  con cifras similares.
- Cong et al. (2023): identifican rebalanceo de funding como factor de
  riesgo en crypto perps.

**Por que es un edge crypto-especifico:**
- No existe en spot (no hay carry).
- No depende de price-action — el desequilibrio existe AUNQUE el precio se
  vea fuerte tecnicamente.
- Distinto al edge de los Agentes A/B/C (trend, ML, regime).

---

## 2. Diseno frozen *a priori*

Decisiones tomadas **antes** del backtest, justificadas por teoria:

| Param | Valor | Justificacion |
|-------|-------|---------------|
| `funding_zwindow` | 270 (4h bars) = 45d | Ventana estandar en literatura funding-reversion. Demasiado corto = ruidoso; demasiado largo = insensible a regimenes. |
| `long_z_max` | -2.0 | -2 sigma = percentil 2.3, "claramente extremo". Mas conservador que -2.5 para no diezmar la muestra. |
| `short_z_min` | 2.5 | +2.5 sigma mas estricto que LONG porque SHORT en BTC tiene sesgo de fondo negativo (historia del proyecto: "SHORT en altcoins no funciona") — requiero mas conviction. |
| LONG `bullish=1` | filtro | Vela alcista al cierre = la caida ya se freno. Evita "catch the falling knife". |
| SHORT `bearish=1` | filtro | Vela bajista al cierre = el rally ya se freno. Evita front-runnear pumps. |
| LONG TP/SL | 5% / 2.5% | R-multiple 2:1, standard. |
| SHORT TP/SL | 4% / 2.0% | R-multiple 2:1, ligeramente mas tight (SHORT mueve mas rapido). |
| LONG max_bars | 36 (6d) | Funding suele normalizar en <1 semana. |
| SHORT max_bars | 30 (5d) | Igual razon, mas tight para SHORT. |
| `commission` | 0.0005 | 0.05% por lado, igual que v15_framework. |

**Sin grid search**: los thresholds son los que aparecen consistentemente en
literatura. NO se optimizaron buscando "el que da mejor PF".

---

## 3. Auditoria anti-bugs (los 4 prohibidos)

| Bug | Solucion |
|-----|----------|
| **Trades solapados** | `run_backtest`: tras cierre, `i += bars + 1`. Espejo de `revalidate_v15.py`. Imposible duplicar. |
| **Look-ahead intrabar** | `simulate`: chequea TP/SL contra `high[b]/low[b]` UNA VEZ; pesimista si ambos tocados (SL gana). NO se actualiza ningun peak/trailing dentro de la vela. |
| **MTF/funding sin shift(1)** | `prepare_data`: `f_shifted = f_raw.shift(1)` ANTES de calcular z. La z conocida en t solo usa funding rates < t. Idem que `v15_features.py:compute_sentiment_features`. |
| **Selection bias / grid search** | Params son frozen *a priori* (ver tabla §2). Las exploraciones del README son **post-mortem honestos**, no se usaron para optimizar. |

---

## 4. Resultados in-sample (cutoff 2025-12-31)

### 4.1 Walk-forward por semestre (purga 14d)

**COMBINED (bidireccional)**: **9/12 folds OK** (PF>=1.2 + total>0 + n>=3)
PF median (folds evaluados): **1.69**

| Periodo | n | L | S | WR | PF | Total | DD | ok |
|---------|---|---|---|-----|------|--------|-------|-----|
| 2020-01 | 3 | 1 | 2 | 66.7% | 3.02 | +5.3% | 2.6% | [+] |
| 2020-07 | 16 | 3 | 13 | 56.2% | 2.89 | +27.6% | 5.3% | [+] |
| 2021-01 | 4 | 3 | 1 | 50.0% | 2.29 | +5.4% | 2.6% | [+] |
| 2021-07 | 15 | 4 | 11 | 53.3% | 2.19 | +19.2% | 9.1% | [+] |
| 2022-01 | 17 | 17 | 0 | 41.2% | 1.32 | +7.4% | 7.6% | [+] |
| 2022-07 | 8 | 8 | 0 | 50.0% | 1.98 | +8.1% | 2.6% | [+] |
| 2023-01 | 12 | 10 | 2 | 50.0% | 2.41 | +17.7% | 9.7% | [+] |
| 2023-07 | 11 | 6 | 5 | 54.5% | 1.40 | +4.8% | 12.4% | [+] |
| **2024-01** | 10 | 0 | 10 | 30.0% | 0.63 | **-5.4%** | 9.0% | [-] |
| 2024-07 | 16 | 10 | 6 | 37.5% | 1.22 | +4.2% | 12.4% | [+] |
| **2025-01** | 12 | 12 | 0 | 25.0% | 0.57 | **-8.8%** | 13.1% | [-] |
| **2025-07** | 8 | 8 | 0 | 37.5% | 0.57 | **-5.7%** | 10.2% | [-] |

### 4.2 Por direccion

LONG-only: 6/10 evaluables folds — bootstrap p=**0.105** (NO significativo en aislamiento)
SHORT-only: 2/5 evaluables folds — bootstrap p=**0.105** (NO significativo)

Cada direccion por separado **NO** alcanza significancia. Pero COMBINADAS si:

### 4.3 Backtest global 2020-01-01 → 2025-12-31

| Metric | Valor |
|--------|-------|
| N trades | **150** (LONG=92, SHORT=58) |
| WR | 44.0% |
| PF | 1.41 |
| Total return | **+102.7%** (en 5.5 anos) |
| CAGR | **+13.8%** |
| Monthly return | +1.07% |
| Max DD | 23.3% |
| Avg holding | 1.9 dias |
| Sharpe annualizado | **0.83** |
| Bootstrap p-value | **0.0363 (SIGNIFICATIVO)** |
| Funding carry contribution | +6.1% del PnL total |

### 4.4 Por ano

| Ano | N | L/S | WR | PF | Return |
|-----|---|------|------|------|---------|
| 2020 | 19 | 4/15 | 57.9% | 2.91 | +34.3% |
| 2021 | 27 | 9/18 | 48.1% | 1.84 | +27.3% |
| 2022 | 29 | 29/0 | 37.9% | 1.14 | +4.4% |
| 2023 | 26 | 18/8 | 53.8% | 1.91 | +26.2% |
| 2024 | 29 | 12/17 | 37.9% | 1.15 | +4.7% |
| **2025** | 20 | 20/0 | 30.0% | **0.57** | **-14.0%** |

---

## 5. SELF-AUDIT

### 5.1 Lo positivo (honesto)

- **Edge real**: bootstrap p=0.036 sobre 150 trades en 6 anos — no es ruido.
- **WF aceptable**: 9/12 semestres positivos con criterio PF>=1.2.
- **Sample no trivial**: 150 trades, mucho mejor que las decenas que daba V13.03 en cross-asset.
- **No hay look-ahead** (los 4 bugs prohibidos estan auditados).
- **PF bajo, WR bajo, DD razonable**: no hay senales de overfitting (no PF=18, no WR=70%).
- **Mecanismo teorico solido**: no es "data mining" — esta basado en
  literatura de microestructura crypto.
- **Funding carry incluido en PnL**: 6.1% del PnL viene del carry recibido
  (mayoritariamente SHORT cobrando funding+).

### 5.2 Lo negativo (honesto — **NO cumple objetivos**)

| Objetivo | Logrado | OK? |
|----------|---------|------|
| Retorno > 30% anual | **13.8%** | ✗ |
| Sharpe > 1.0 | **0.83** | ✗ |
| DD < 25% | 23.3% | ✓ (marginal) |
| WR > 45-55% | 44.0% | ✗ (marginal) |

- **2025 catastrofico**: -14% YTD, WR 30%, PF 0.57. La estrategia no
  genero ni una sola senal SHORT en 2025 (no hubo z>2.5).
  Interpretacion: el mercado se eficiento, los extremos de funding ya
  no llegan tan altos, y los pocos extremos negativos no funcionan
  consistente.
- **Sample SHORT por direccion**: 58 trades en 6 anos = ~10/anyo. Bootstrap
  por direccion no alcanza significancia (p=0.10) — solo el combinado lo
  hace. Esto significa que SHORT por si solo **NO esta validado** segun
  los criterios del proyecto.
- **Sharpe 0.83 < 1**: media volatilidad por trade alta (~3.8%) y avg
  pnl bajo (~0.5%) → risk-adjusted insuficiente.
- **CAGR 13.8%** apenas supera al ETF de S&P (~10% historico). No
  justifica el riesgo crypto (objetivo proyecto: 30%+).

### 5.3 Comparacion con Agentes A, B, C (ronda 1)

| Agente | Mecanismo | WF | CAGR | Bootstrap |
|--------|-----------|----|------|-----------|
| A | Donchian + EMA daily + ATR trail | 7/9 | ~12-15% | 0.07 |
| B | ML clasificador 11 features | 3/11 | ~0% | 0.607 (REJECT) |
| C | Regime adaptive | 6/12 | ~13% | 0.156 |
| **E** | **Funding mean-reversion** | **9/12** | **13.8%** | **0.036** |

Agent E es el **unico con bootstrap < 0.05** y el mejor WF de los cuatro.
Pero su CAGR es practicamente igual al de A y C — el "techo honesto"
del edge en BTC 4h parece ser **~10-15% anual** independientemente del
mecanismo. El VERDICTO de ronda 1 ya lo anticipaba.

### 5.4 Hipotesis sobre 2025

Tres posibles explicaciones (no excluyentes):

1. **Eficienciacion del mercado**: con mas trading sistematico, los
   extremos de funding se eliminan antes (mas rapido) y el rebote
   esperado ya no llega — fue arbitrado.
2. **Regimen de baja volatilidad en funding**: la sd_90d de funding cayo
   2024-2025, por tanto el z-score se infla para movimientos menores,
   diluyendo la senal.
3. **Mala suerte estadistica**: con N~30/anyo, una mala racha de 1 anyo
   es plausible. Sin OOS de 2026 no podemos discriminar.

---

## 6. Recomendacion

**REJECT — no apto para produccion**. Razones:

1. CAGR 13.8% < 30% objetivo: el riesgo de operar BTC con leverage no
   se compensa por un retorno apenas superior al S&P 500.
2. Degradacion clara en 2025: 3 de los 4 ultimos folds negativos.
3. SHORT por si solo no validado (p=0.10 + 2/5 folds).
4. Sharpe 0.83 indica relacion riesgo/recompensa pobre.

**Lo que sí aporta el experimento (valor de cierre):**

- Confirma **con muestra honesta** que el edge funding-mean-reversion
  **existe** (bootstrap p<0.05) pero es modesto (~13-15% anual).
- Cierra la hipotesis de "funding extremos pueden generar 30%+" —
  empiricamente, no en BTC 4h con los criterios honestos del proyecto.
- Si se quisiera intentar capturarlo en otro contexto:
  * Probar en altcoins/memecoins donde los funding extremos son mas
    violentos (DOGE, PEPE) — pero historia del proyecto dice que SHORT
    en alts es trampa.
  * Probar con leverage variable (size proporcional a |z|) en lugar de
    TP/SL fijos — abrir investigation, no usar en produccion sin
    validar.
  * Combinar con Agent A: usar Agent A como base trend-follower y
    desactivarlo cuando z_funding > 2.5 (no perseguir techos). Eso
    refuerza A en lugar de operar E por si solo.

---

## 7. Archivos

```
experiments/agent_E/
  strategy.py    # PARAMS, prepare_data, signal, simulate, run_backtest, metrics
  train.py       # WF + bootstrap + sanity checks. Imprime y guarda results.json
  results.json   # Output de train.py (re-generado al re-correr)
  README.md      # Este archivo
```

Para reproducir:
```
C:/Python/python.exe experiments/agent_E/train.py
```

(Cutoff 2025-12-31 esta hard-coded en `PARAMS` y se verifica en `load_all_data`.)
