# Veredicto Consolidado SOL — Estructuralmente no operable a 4h

> Tras 5+ enfoques distintos con motor honesto, SOL/USDT 4h NO genera edge
> estadísticamente significativo con ningún mecanismo convencional.
> Fecha: 2026-05-19

---

## Tabla maestra — todos los enfoques SOL probados

| # | Enfoque | Bootstrap p | Veredicto |
|---|---------|------------:|-----------|
| 1 | SOL committee original (BRK + BB_UPPER SHORT) | — | REJECT (DD 48%, OOS catástrofe) |
| 2 | SOL tight trailing "APROBADO" | 0.155 (revalidación) | REJECT (era bug-inflado, PF 15 declarado) |
| 3 | ML dedicated SOL→SOL (V14-style) | — | REJECT (0 trades post-2022) |
| 4 | V14 ADA→SOL cross-asset | — | REJECT (originalmente declarado 9/10 era bug) |
| 5 | **Agent M — SOL leveraged BTC** | **0.243** | **REJECT** |
| 6 | **Agent N — Vol breakout WIDE trail** | **0.078** | **REJECT (marginal)** |
| 7 | **Agent O — ML cross-asset (BTC+ETH+SOL)** | **0.983** | **REJECT (PEOR que azar)** |

---

## Hallazgos por agente

### Agent M — Hipótesis "SOL es BTC apalancado"

**Resultado**: WF 3/9, PF 1.33, annual +8.4%, p=0.243

- Cross-check vs random: A's BTC signal SÍ supera random (+15.4% de mejora)
- **Pero**: synth 10/20 positivas, mediana -1.3% — el "+15% mejora" no
  transfiere en universos paralelos
- **Edge concentrado en 2024 BTC bull**: 2024-H1 PF 2.20, 2024-H2 PF 2.73; en
  2025 degrada (H1 PF 1.13, H2 PF 0.31)
- 2022 folds = 0 trades (filtro daily bloqueó BEAR correctamente)

**Lección**: SOL's beta 1.5 a BTC NO se traduce automáticamente en edge
direccional sobre SOL. La hipótesis "leveraged BTC" no se sostiene
estadísticamente.

### Agent N — Vol breakout con trailing WIDE escalado

**Resultado**: WF 7/11, PF 1.50, annual +9.8%, p=0.078

- **El descubrimiento crítico**: tight (0.8%) vs wide (2.5%) dan PF casi idénticos
  (1.54 vs 1.50, p=0.047 vs 0.078)
- **El problema NUNCA fue el trail width**. El mecanismo vol-breakout no
  tiene edge en SOL — ni con tight ni con wide
- Edge vs null: **-0.7%** (peor que shuffleado)
- Synth 11/20 positivas (debajo del 14/20 requerido)
- SHORT específicamente catastrófico: PF 0.70, -8.8% — confirma "SOL bounces matan stops"

**Lección**: el "PF 2.56-15.04" declarado en docs era 100% bug-inflado
(look-ahead intrabar). El mecanismo subyacente es ruido.

### Agent O — ML cross-asset (16 features BTC+ETH+SOL)

**Resultado**: WF 2/11, PF 0.86, annual **-44%**, p=**0.983**

- **Peor resultado ML del proyecto** — bootstrap p casi 1.0 (peor que azar)
- Train AUC 0.79 → Test AUC 0.499 — gap 0.29 (igual de malo que G en ETH)
- Synth **0/20 positivas**
- Features TOP-3 importance: ETH/BTC z-score, BTC daily ratio, SOL-BTC corr
  → dominan in-sample pero fallan OOS
- **3 ML independientes convergen**:
  - B (BTC, 11 own features): test AUC 0.520, p=0.607
  - G (ETH, 16 features incl. ETH/BTC): test AUC 0.513, p=0.808
  - O (SOL, 16 cross-asset): test AUC 0.499, p=0.983
- **Degradación monotónica**: p bootstrap subió 0.607 → 0.808 → 0.983 a medida
  que crecía el feature set. **Más features = más overfitting, no más edge**.

**Lección DEFINITIVA del proyecto**:
> ML clasificador con labels binarios TP/SL en crypto 4h con purged CV
> **NO produce edge estadísticamente significativo**. Tres experimentos
> independientes (BTC, ETH, SOL) con diferentes features confirman.

---

## ¿Por qué SOL es tan difícil?

| Característica | SOL | BTC | ETH |
|----------------|----:|----:|----:|
| ATR% 4h promedio | 3.55% | ~2.0% | ~2.4% |
| BB width % promedio | 14.99% | ~5% | ~7% |
| Vol realizada 90d | 80% | 41% | 60% |
| Beta a BTC | 1.5 | 1.0 | 1.2 |
| Datos desde | 2020-09 | 2017+ | 2017+ |
| % tiempo BEAR | 32.6% | ~25% | ~25% |

Combinación letal:
1. **Vol 2x BTC** → cualquier stop razonable se ejecuta en ruido
2. **BB width 3x BTC** → señales BB pierden significado
3. **Bouncing violento en bear** → SHORT trades stopped repetidamente
4. **Datos limitados** (5.4 años vs 8+ BTC) → menos sample size = menos significancia
5. **High beta** → todo movimiento BTC amplificado, dos veces más ruido

---

## Implicación final

**SOL queda DEFINITIVAMENTE fuera de cualquier estrategia algorítmica 4h en este proyecto.**

Igual que ETH, la única exposición SOL defendible:
- **Spot DCA** (acumulación sin algo direccional)
- **SOL staking** (~6-8% APR via validators) — más alto que ETH por beta + inflación
- **Lending DeFi en SOL** (Solend, MarginFi) — 3-5% APY

---

## Lecciones meta acumuladas (3 activos × múltiples enfoques cada uno)

Después de testear honestamente:
- **BTC**: 6 agentes (A-F) → 1 candidato (V2)
- **ETH**: 8 enfoques → 0 candidatos
- **SOL**: 7+ enfoques → 0 candidatos

**Solo BTC V2 (A + F_BTC) tiene edge bootstrap-significativo en TODO el proyecto.**

Lecciones definitivas:
1. **ML 4h crypto no funciona** (B + G + O agree, gap train→test ~0.29 consistente)
2. **El "edge" cross-asset (ETH/BTC ratio, SOL-BTC corr) NO se sostiene OOS** —
   parece feature importante in-sample pero el mercado se eficientiza
3. **Trailing tight era bug, no estrategia** — vol breakout WIDE en SOL tampoco
   funciona, confirmando que el mecanismo nunca tuvo edge
4. **Beta apalancada no transfiere edge** — SOL como "BTC apalancado" no funciona
5. **Sintético es el control definitivo** — 0/20 positivas en O es prueba
   irrefutable de overfitting

---

## Camino a partir de aquí

El proyecto ha **agotado las opciones razonables** en BTC, ETH, SOL a 4h. La
evidencia es overwhelming.

Opciones honestas remaining:

1. **Aceptar el portfolio combinado** (recomendado):
   - BTC V2 algo (único edge real demostrado)
   - ETH + SOL spot DCA o staking (no direccional)
   - Yield DeFi stables

2. **Cambiar de timeframe** (probado con J ETH 1D, no mejoró):
   - 1W BTC: muy poca muestra
   - Intradía 1h o 15m: requiere infraestructura distinta, alto fee impact

3. **Cambiar de instrumento**:
   - Opciones BTC (Deribit) — distinto problema
   - Basis trading (cash and carry)
   - Spread BTC/ETH pairs trading

4. **Pausar exploración** — los datos dicen que el mercado es eficiente al
   nivel que probamos. Acumular datos reales de paper trade BTC V2 es más
   informativo que más backtests.

---

## Archivos

- `experiments/agent_M/` — SOL leveraged BTC
- `experiments/agent_N/` — Vol breakout WIDE
- `experiments/agent_O/` — ML cross-asset
- `docs/V15_SOL_evaluation.md` — historia previa (incluye "APROBADO" debunked)
- `docs/revalidation/SOL.md` — revalidación con motor honesto
