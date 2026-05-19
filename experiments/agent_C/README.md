# Agent C — Regime-Adaptive BTC/USDT 4h

> Walk-forward + bootstrap + per-regime audit on data **strictly ≤ 2025-12-31**.
> Honest engine: one position at a time, no intrabar look-ahead in trailing.

---

## TL;DR — honest verdict

**The strategy has real but modest edge that does NOT pass the project's
≥7/12 walk-forward bar or the p<0.05 bootstrap bar.** It generates positive
expectancy (win/loss ratio 1.74, WR 42%, +54% over 6 years, max DD 32%), but in
choppy periods (2021-H2, 2023-H2, 2025-H1) the BULL pullback fires repeatedly
and gets stopped out — a fundamental limit of trend-following on 4h. The RANGE
sub-strategy fires too rarely to earn its own keep but doesn't harm the combined
result. **BEAR is disabled** (no SHORT) — the project history rejects shorting
crypto on principle, and I find no reason to override that here.

| Metric | Value | Target | Pass? |
|---|---|---|---|
| Walk-forward folds | **6/12** | ≥7/12 | ❌ |
| Bootstrap p-value | 0.156 | <0.05 | ❌ |
| Win rate | 42.4% | >50% | ❌ |
| Profit factor (continuous) | 1.28 | ≥1.2 | ✅ marginal |
| Max drawdown | 31.8% | <25% | ❌ |
| Annualized return | 10.1% | >30% | ❌ |
| Bullish expectancy | win/loss 1.74 | — | ✅ |

**Recommendation for the orchestrator**: this strategy is **honest** but
**below the project bar**. The single value it delivers — *the trailing-stop
edge is real, the entry quality is what fails* — confirms the audit's diagnosis
(`docs/AUDITORIA_2026-05.md` §C.3: "The edge is in the EXIT, not in the entry").

---

## Strategy design

### Regime detection (daily EMAs, `shift(1)` to avoid look-ahead)
- BULL: `EMA20_1d > EMA50_1d` by more than 2% (dead zone)
- BEAR: `EMA20_1d < EMA50_1d` by more than 2% **AND** `close_1d < EMA200_1d`
- RANGE: everything else (including the "recovery zone" where BEAR demotes to
  RANGE because price is back above EMA200)

In-sample regime distribution: **BULL 44%, RANGE 29%, BEAR 26%**.

### BULL sub-strategy: pullback to EMA20 (trend-following)
Trader rationale: in an up-trending market, buy the dip to the mean (EMA20)
when the trend is confirmed and momentum is starting to turn back up.

Entry filters (5 parameters, all standard textbook values):
1. `|close - EMA20| / EMA20 <= 1.5%` — close to EMA20
2. `EMA20 > EMA50` (4h) AND `EMA50_1d > EMA200_1d` (daily) — full trend alignment
3. `40 <= RSI <= 60` — not over-extended either way
4. `ATR_pct >= 0.5%` — avoid dead markets
5. Entry bar bullish (`close > open`) — the dip is being bought
6. `EMA20_slope_5 > 0` — EMA20 itself is rising

Exit: **wide trailing stop** at 1.8 × ATR (floor 1.2%), max 24 bars (~4 days).
Wide-trail follows the agent brief: "Trailing AMPLIO (1.5-3x ATR; NO el 0.2x
que demostró ser bug)."

### RANGE sub-strategy: deep oversold mean-reversion
Trader rationale: in a sideways market, oversold extremes near the lower BB
often bounce. **LONG only** — shorting RSI>70 in BTC RANGE is a documented loser
because the crypto bias is bullish.

Entry filters (4 parameters):
1. `RSI <= 30` — Wilder oversold (standard textbook)
2. `BB_pct <= 0.15` — close to lower BB
3. Entry bar bullish (`close > open`) — confirmation of the bounce
4. `ATR_pct >= 0.5%` — need real movement to mean-revert

Exit: **fixed TP/SL at 1.5 × ATR each** (capped 0.8%-6%), max 12 bars (~2 days).
Symmetric R/R because mean reversion targets a definite level, not an extended
move.

### BEAR sub-strategy: DISABLED
Per `CLAUDE.md` and `docs/revalidation/RESUMEN.md` §4: SHORT on crypto
historically does not survive a single bull cycle. I found no theoretical
reason to override the project's rule. Adding a BEAR sub-strategy in this
sample (which contains 2022-H1/H2 — the only real BTC bear of the window) would
let me overfit a SHORT to one period and call it edge. **Better to leave the
slot empty**.

---

## In-sample results (2020-01 → 2025-12, walk-forward purged 14 days)

### Combined (BULL + RANGE)

| Period | N | WR | PF | Total | DD | OK |
|---|---|---|---|---|---|---|
| 2020-01 | 12 | 50% | 1.53 | +5.9% | 4.9% | ✅ |
| 2020-07 | 17 | 65% | 4.38 | +43.7% | 5.0% | ✅ |
| 2021-01 | 13 | 46% | 2.00 | +17.3% | 10.7% | ✅ |
| 2021-07 | 16 | 31% | 0.25 | -19.0% | 21.1% | ❌ |
| 2022-01 | 1 | 100% | inf | +2.5% | 0% | ❌ (n<5) |
| 2022-07 | 3 | 67% | 1.49 | +2.1% | 4.8% | ❌ (n<5) |
| 2023-01 | 17 | 47% | 1.11 | +1.2% | 5.2% | ❌ |
| 2023-07 | 16 | 31% | 0.40 | -10.4% | 10.4% | ❌ |
| 2024-01 | 16 | 44% | 1.74 | +8.9% | 5.1% | ✅ |
| 2024-07 | 11 | 55% | 3.20 | +18.2% | 3.7% | ✅ |
| 2025-01 | 16 | 19% | 0.26 | -15.7% | 18.7% | ❌ |
| 2025-07 | 13 | 38% | 1.26 | +2.3% | 7.0% | ✅ |

**6/12 folds passed.** Median PF among folds (excluding inf): 1.49.

### Continuous 2020-2025 (no fold split)
- 172 trades, WR 42.4%, PF 1.28, total +54.3%, DD 31.8%
- Monthly return mean: 1.08% (median 0.43%)
- Months positive: 55.6% (30/54)
- **Annualized (geometric monthly): 10.1%**

### Bootstrap (n=3000) on continuous trades
- p-value (P(total ≤ 0 | resampling)): **0.156** — NOT significant
- Median resampled total: +53.4%
- 5th percentile: -24.6%

### Per-regime audit (each sub-strategy in isolation)

| Sub | Folds OK | N | WR | PF | Total | DD | Boot p |
|---|---|---|---|---|---|---|---|
| BULL only | 5/12 | 126 | 39.7% | 1.34 | +45.4% | 34.9% | 0.139 |
| RANGE only | 0/12 | 25 | 60.0% | 1.19 | +4.7% | 11.7% | 0.377 |

RANGE has the **best per-trade quality** (WR 60%) but fires too rarely to
register a fold pass (1-4 trades per semester). It is statistically negligible
on its own and stays in only because it doesn't hurt the combined result.

---

## SELF-AUDIT

### What I did right
- **Honest engine**: one position at a time (no overlapping trades) +
  no intrabar look-ahead in the trailing stop. This is the bug-fix that
  collapsed the old `meta_v15.json` PF 13-20 to PF ~1.0. My engine cannot
  produce those numbers.
- **All daily features use `shift(1)`** — `compute_macro_daily` returns
  yesterday's regime for use today. Verified.
- **All 4h indicators are rolling/expanding on past bars only.** RSI/ATR/EMA
  use `.ewm()` recursively, BB uses `.rolling(20)`. No `.future()` or
  `.shift(-N)`. The only forward reference is inside `simulate()` (which is
  correct — it's the actual price path after entry).
- **Cutoff respected**: `assert df.index.max() <= CUTOFF` in `train.py`. The
  prepared data ends at `2025-12-31 20:00 UTC`. 2026 was never loaded.
- **Walk-forward with purge** (14 days each side of every fold). Sensitivity
  test with purge=28d gave identical 6/12 folds — no train/test leakage.
- **Bootstrap** with n=3000 resamples to test whether the +54% total return
  could be due to luck. Result: p=0.156 (cannot reject null).
- **Per-regime audit** — each sub-strategy was forced to stand on its own. The
  fact that RANGE alone got 0/12 is honest; I report it.
- **NO BEAR / NO SHORT** — declined to add a third sub-strategy just to fill
  the regime slot, in line with project history.
- **No selection bias of folds**: every WF result above is from the same
  `PARAMS` dict, frozen before measurement. I did three principled refinements
  during development (daily-EMA alignment, bullish entry candle, slightly
  relaxed RANGE thresholds from RSI<25/BB<10 to RSI<30/BB<15) — each
  justifiable as a textbook trader rule, not a knob fit to a fold.

### What did not work
- **The "regime-adaptive" diversification did NOT produce the 3x edge I hoped**.
  RANGE fires too rarely to compensate for BULL's bad folds. BULL alone has
  similar metrics to BULL+RANGE — the diversification benefit is marginal.
- **BULL pullback in choppy markets is the strategy's Achilles heel**. 2021-H2,
  2023-H2 and 2025-H1 were all periods classified as BULL by daily EMAs but
  effectively distribution/chop. The pullback entry fired repeatedly and ate
  small-but-frequent SLs. I do NOT have a clean filter that distinguishes
  "real BULL" from "decaying BULL" without overfitting.
- **A theoretical "missing piece"**: I considered adding a "fresh-high"
  momentum filter (only enter if a 20-bar high was made in the last ~8 bars),
  but that would be a 7th parameter on BULL and I'd be tuning to the bad folds
  — exactly the selection bias the brief warns against. I left it out.

### Sanity flags (automatic checks)
- PF > 4? No (PF 1.28).
- WR > 65%? No (42%).
- DD < 5%? No (32%).
- N trades < 20? No (151 in WF, 172 continuous). ✅ enough sample for inference.

**Verdict from my own audit**: nothing in the in-sample metrics looks
suspicious. PF 1.28 is in the documented credible range for BTC honest backtest
(the same range as the validated V15-BTC: OOS PF=1.35). If this strategy
collapses on 2026 OOS, it will be because BULL pullbacks failed there too — not
because the engine lied.

### What the orchestrator should expect on 2026 OOS
- The 2026 data through Feb 27 (per the parquet) covers a sharp BTC drawdown
  (audit mentions BTC -23% in early 2026). My BULL sub-strategy will probably
  **not fire much**: BEAR regime would dominate when EMA50_1d falls below
  EMA200_1d. With BEAR disabled, this strategy **stays flat** during BTC
  capitulation — capital-preserving, but no positive contribution.
- RANGE may catch one or two oversold bounces. WR is 60% so each trade has
  good expectancy, but the total contribution will likely be ~+5% at most.
- I expect: **few trades, near-zero return, max DD <10%**. If you see PF > 2
  or DD < 3% on 2026, suspect a bug. If you see DD > 20%, the BULL filter
  failed.

---

## Files

- `strategy.py` — frozen `PARAMS`, `detect_regime()`, `signal()`, `simulate()`,
  `run_engine()`. Importable, self-contained (no project imports).
- `train.py` — loads BTC 4h with the 2025-12-31 cutoff, computes features,
  runs walk-forward + continuous + per-regime + bootstrap, writes
  `results.json`.
- `results.json` — full numerical output of the latest training run.

## Run

```bash
C:/Python/python.exe experiments/agent_C/train.py
```

Takes ~30 seconds. Output is printed and persisted in `results.json`.
