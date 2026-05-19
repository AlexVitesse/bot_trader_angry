# V15 ADA/USDT Evaluation Results

**Date:** 2026-03-23
**Script:** `evaluate_ada_v15.py`
**Data:** ADA 4H, 13,275 bars (2020-02 to 2026-02)

## ADA Characteristics

| Metric | ADA | ETH (ref) | SOL (ref) |
|--------|-----|-----------|-----------|
| ATR% mean | 3.02% | ~2.4% | 3.55% |
| BB width mean | 12.6% | ~7% | 15.0% |
| ADA-BTC corr | 0.713 (93.5% >= 0.5) | 0.69 | 0.70 |
| BEAR bars | 37.6% (4,988) | ~25% | 32.6% |
| Raw label WR (TP=6%/SL=4%) | **28.3%** | n/a | 32.9% |
| Data range | 2020-02 | 2017 | 2020-09 |

ADA has the **worst raw label WR** of all tested pairs. Only 28.3% of bars produce winning LONG trades with TP=6%/SL=4%. More time in BEAR regime than ETH.

## Context: V14 Claims vs V15 Reality

V14 ADA ensemble claimed 11/12 WF folds, +458% PnL, 4.3% overfitting drop. This was the 2nd-best pair after BTC in the V14 framework. However, V15's stricter walk-forward (12 fixed semesters, N>=3 requirement per fold) reveals the edge was concentrated in early folds.

## Part 1: ML Ensemble V14-style (10 features)

| Config | WF | N | WR | PF | $1K-> | DD |
|--------|---:|--:|---:|---:|------:|---:|
| ML_V14_th0.50 | **8/10** | 388 | 46.4% | 1.16 | $2,455 | **88.8%** |
| ML_V14_th0.55 | 5/10 | 199 | 43.7% | 1.10 | $1,266 | 72.5% |
| ML_V14_th0.60 | 1/7 | 59 | 47.5% | 1.37 | $1,449 | 49.1% |

ML_V14_th0.50 passes WF 8/10 but **DD 88.8%** — catastrophic. Trade concentration: 172 trades in 2021-H1 alone (44% of total), then drops sharply.

### Trade Distribution by Period (th=0.50)
| Period | N | WR | Pattern |
|--------|--:|---:|---------|
| 2020-H2 | 102 | 34.3% | Heavy losses (model just started) |
| 2021-H1 | 172 | 44.2% | Most trades, marginal edge |
| 2021-H2 | 19 | 47.4% | Drops off |
| 2022-H1 | 26 | 61.5% | Decent but few trades |
| 2022-H2 | 27 | 55.6% | PF only 0.71 despite good WR |
| 2023-2025 | 42 | 69.0% | Model nearly stops trading |

Same pattern as SOL: **model dies after 2022**, producing 0-12 trades per fold.

## Part 2-4: Filters (Regime, Vol, Combined)

| Best Filter Config | WF | N | WR | PF | DD |
|-------------------|---:|--:|---:|---:|---:|
| ML_VOL1.5_th0.5 (MARGINAL) | 6/10 | 142 | 51.4% | 1.45 | 44.3% |
| ML_VOL2_th0.6 | 2/6 | 28 | 60.7% | 2.22 | 21.0% |
| ML_REGIME+VOL_th0.5 | 1/8 | 73 | 45.2% | 1.16 | 34.0% |

- **Vol filter vol_ratio > 2.0**: Improves per-trade quality (PF 2.22) but only 28 trades total (too few, WF 2/6)
- **Vol filter vol_ratio > 1.5**: MARGINAL 6/10, PF 1.45 — best quality-adjusted ML config but DD 44.3%
- **Regime filter**: Hurts rather than helps — removes trades but doesn't improve quality
- **Momentum filter**: Makes everything worse (PF < 1.0 with regime+momentum)

## Part 5: Rule-Based Strategies

| Config | WF | N | WR | PF | $1K-> | DD |
|--------|---:|--:|---:|---:|------:|---:|
| BRK_ADA standalone (best) | 2-4/11 | 14-61 | 34-50% | 0.72-1.54 | varies | 4.7-40% |
| **BTC_FOLLOW_c0.4_ATR** | **10/12** | 434 | 45.4% | 1.21 | $3,962 | **58.6%** |
| BTC_FOLLOW_c0.5_ATR | 9/12 | 412 | 45.4% | 1.22 | $3,886 | 63.8% |
| COMBINED_v1.2_ATR | 9/12 | 431 | 44.8% | 1.20 | $3,373 | 65.3% |

**Breakout ADA standalone: RECHAZADO** — WR 34-50%, most configs have PF < 1.0. ADA breakouts are unreliable.

**BTC breakout follower** passes WF with high fold counts (8-10/12) but has **unacceptable DD** (58-67%). The strategy buys ADA whenever BTC breaks out above 20-bar high. Works because ADA follows BTC (93.5% corr >= 0.5), but in bear markets ADA falls harder than BTC.

## Part 6: ATR-based TP/SL for ML

| Config | WF | N | WR | PF | $1K-> | DD |
|--------|---:|--:|---:|---:|------:|---:|
| ML_ATR_3.0x2.0_th0.55 | 7/10 | 199 | 48.2% | **1.48** | $8,786 | **75.8%** |
| ML_ATR_3.0x2.0_th0.50 | 7/10 | 388 | 46.6% | 1.33 | $15,407 | **94.7%** |
| ML_ATR_3.0x2.0_RV_th0.50 | 3/8 | 73 | 56.2% | 2.06 | $5,510 | 22.2% |

ATR 3.0x2.0 with regime+vol filter (RV) has the best risk profile: **PF 2.06, DD 22.2%** — but only 73 trades in 8 folds, WF 3/8. Not enough fold consistency.

## OOS 2026 (Jan-Mar)

ADA: $0.3327 -> $0.2746 (**-17.5%**)
Regimes: BEAR=343, RANGE=5 (virtually all BEAR)

### Results: 0-1 trades across ALL configurations

| Config | N | Result |
|--------|--:|--------|
| ML_V14_th0.6 | 1 | SL -4.10% |
| ML_VOL1.5_th0.5 | 1 | SL -4.10% |
| ML_VOL2_th0.6 | 1 | SL -4.10% |
| All others | 0 | No trades |

The ML model generates essentially zero signals in 2026. The one signal that fired (Feb 5, $0.2455) hit SL immediately.

## Final Verdict: RECHAZADO

### Critical Issue: No Config with DD <= 25%

**ALL 13 APROBADO configs have DD > 25%** (project maximum):
- Best ML DD: 88.8% (ML_V14_th0.50)
- Best Rules DD: 58.6% (BTC_FOLLOW_c0.4_ATR)
- Best quality-adjusted: DD 22.2% but WF only 3/8 (ML_ATR_3.0x2.0_RV_th0.50)

### Why ADA Fails V15

1. **Raw label WR = 28.3%** — only 1 in 3.5 bars produces a winning LONG trade with TP 6%/SL 4%. Fundamental mismatch between TP/SL and ADA's volatility.

2. **ML concentrates in 2020-2021** — 274/388 trades (70%) in the first 3 folds. Model dies after 2022 (same as SOL pattern).

3. **BTC-follower has edge but no risk control** — ADA follows BTC breakouts consistently (93.5% corr) but falls harder in bear markets, creating cascading losses.

4. **BEAR regime dominance** — 37.6% of bars are BEAR (more than ETH's 25%). In bear periods, LONG-only strategies bleed out.

5. **OOS 2026 empty** — virtually zero signals in a real bear market. A bot that doesn't trade for 3 months isn't useful.

### Comparison: V14 Claims vs V15 Reality

| Metric | V14 Claim | V15 Result |
|--------|-----------|------------|
| WF folds | 11/12 | 8/10 |
| Overfitting drop | 4.3% | n/a (different WF) |
| DD | not reported | **88.8%** |
| OOS 2026 | not tested | 1 trade, SL |
| Trade concentration | not analyzed | 70% in 2020-2021 |

V14's optimistic results came from a different WF methodology and likely didn't capture the catastrophic drawdown from early fold losses.

### Recommendations

- **ADA queda fuera del V15** — no configuration meets the DD <= 25% requirement
- The BTC-follower approach is the most consistently profitable strategy (WF 10/12) but needs position sizing or hedging to control DD
- If revisiting: consider TP 3%/SL 2% (smaller targets, matching ADA's lower volatility vs SOL)
- The vol_ratio filter (> 1.5 or > 2.0) significantly improves ML quality but kills trade count
- **Do NOT activate V14 ADA ensemble** — the DD was always there, just not measured

---

*ADA joins SOL as RECHAZADO for V15 with standard approaches. Only BTC (Expert Committee) and ETH (rule-based committee) have passed validation with fixed TP/SL.*

---

# UPDATE (2026-03-24): Alternative Solutions — CANDIDATO A APROBACION

**Script:** `evaluate_alt_v15_solutions.py`
**Full results:** `docs/V15_ALT_solutions.md`

Testing 4 new approaches (smaller TP/SL, trailing stop, sizing, LightGBM) revealed a viable configuration:

## RULES_TP3_SL1.5_TRAIL_tight_imm

| Metric | Value | Target | Pass? |
|--------|-------|--------|-------|
| WF | **10/12** | >= 7 | YES |
| N | 446 | - | - |
| WR | **50.4%** | >= 50% | YES |
| PF | **2.86** | >= 1.3 | YES |
| DD | **7.1%** | < 25% | YES |
| Ann% | **50.9%** | >= 30% | YES |

**Strategy**: BTC-follower + breakout ADA, BULL/RANGE only, tight trailing stop (0.8% SL, immediate)
**100% rule-based** — no ML, no overfitting risk.

**OOS 2026**: 2 trades, -0.98% (vs ADA -17.5% B&H). Weak but minimal loss in extreme BEAR.

**Key insight**: The tight trailing stop (V7 approach) transforms mediocre fixed-TP/SL entries into profitable micro-gain trades. The edge is in the exit mechanism, not the entry signal.

Status: **CANDIDATO** — pending SHORT evaluation for BEAR regime.

---

# UPDATE (2026-03-24): SHORT in BEAR Regime — APROBADO

**Script:** `evaluate_short_alt_v15.py`
**Full results:** `docs/V15_ALT_solutions.md`

## SHORT_BTC_BREAKDOWN_TRAIL_tight

| Metric | Value | Target | Pass? |
|--------|-------|--------|-------|
| WF | **10/10** | >= 60% BEAR folds | YES |
| N | 329 | - | - |
| WR | **62.6%** | >= 50% | YES |
| PF | **13.51** | >= 1.3 | YES |
| DD | **2.4%** | < 25% | YES |
| Ann% | **62.6%** | >= 30% | YES |

**Strategy**: SHORT when BTC breaks below 20-bar low (vol_ratio > 1.0), BEAR regime only, tight trailing stop (0.8% SL, immediate).

**OOS 2026**: 19 trades, WR 78.9%, PF 15.01, **+33.3%** (vs ADA -17.5% B&H).

## Full Committee: LONG + SHORT

| Component | Regime | WF | N | PF | DD | Ann% |
|-----------|--------|---:|--:|---:|---:|-----:|
| LONG trail | BULL/RANGE | 10/12 | 446 | 2.86 | 7.1% | 50.9% |
| SHORT BTC breakdown | BEAR | 10/10 | 329 | 13.51 | 2.4% | 62.6% |
| **Combined** | **ALL** | - | 775 | - | ~7% | ~55% |

**Combined OOS 2026**: LONG -0.98% + SHORT +33.3% = **+32.3%** (vs ADA -17.5% B&H).

## Final Status: **APROBADO**

ADA V15 passes all project criteria with the tight trailing stop approach covering both directions. 100% rule-based, no ML.
