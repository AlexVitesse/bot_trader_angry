# Agent I — ETH/USDT 4h RANGE Mean-Reversion

> Walk-forward + bootstrap on data **strictly ≤ 2025-12-31**.
> Honest engine: one position at a time, no intrabar look-ahead in TP/SL.
> Round 3 of the project (ETH-specific). Goal: cover the mean-reversion
> mechanism that A (trend), F (vol-breakout) and H (rotation) do not test.

---

## TL;DR — honest verdict

**REJECT.** ETH RANGE mean-reversion as a standalone strategy does not work
with the discipline this project requires. The setup is structurally too rare
(3.2 trades/year) and the per-trade expectancy is negative (PF 0.66, total
-6.1% over 6 years).

The mechanism's win rate is reasonable (52-60%) but losses are larger than
wins. Even when filters are stripped to the bare minimum (no candle
confirmation, no volume confirmation), the strategy converges to PF ≈ 1.0 with
zero edge. **The hypothesis "ETH RANGE bounces are more violent than BTC's
and therefore more tradeable" is not supported by data.**

| Metric | Value | Target | Pass? |
|---|---|---|---|
| Walk-forward folds (n≥5, PF≥1.2, total>0) | **1/12** | ≥7/12 | NO |
| Bootstrap p-value (return ≤ 0 by chance) | 0.760 | <0.05 | NO |
| Total trades 2020-2025 | **19** | ≥60 | NO |
| Trades per year | 3.2 | ≥10 | NO |
| PF continuous 2020-2025 | 0.66 | ≥1.2 | NO |
| WR continuous | 52.6% | >50% | YES |
| Total return 6y | -6.1% | >0 | NO |
| Annualized (geom monthly) | -5.3% | >30% | NO |
| Max DD | 9.8% | <25% | YES |

**Recommendation for the orchestrator:** **don't use this**. It also gives a
clean structural answer: *RANGE windows in ETH 4h are too short and too
infrequent to support a standalone mean-reversion strategy.* This confirms the
direction in Agent C's BTC finding ("RANGE fires too rarely; 0/12 folds") and
extends it to ETH.

---

## Strategy design (frozen `PARAMS`)

### RANGE regime detection (multi-condition)
All three must hold simultaneously. Each is a textbook "weak trend" threshold:
- `ADX_14 < 25` — Wilder's "weak trend" cutoff (not the stricter <20).
- `BB_width percentile_250 < 40` — current Bollinger Band width is in the
  bottom 40% of the past 250 bars (compressed but not extreme).
- `|EMA20_1d − EMA50_1d| / EMA50_1d ≤ 3%` — daily EMAs close together.

In-sample regime distribution (ETH, 2020-2025):
- BULL  54.3% · BEAR 35.9% · **RANGE 9.7%**

For reference, BTC under the same definition (the relevant comparison) has
**RANGE ≈ 29%** (per Agent C). ETH spends 3× less time in RANGE than BTC —
the structural beta difference shows up here: ETH oscillates between trending
phases faster, with shorter consolidations.

### LONG entry (fade oversold) — only fires when regime == RANGE
- `RSI_14 ≤ 30` **OR** `BB_pct ≤ 0.10` (OR-logic — this is the relaxation
  vs Agent C's RSI AND BB which only fired 25 times in BTC)
- AND entry candle is bullish (`close > open`) — confirm bounce, avoid knife
- AND `vol[i] / SMA20(vol)[i-1] ≥ 1.2` — confirm participation
- AND `atr_pct ≥ 0.4%` — avoid dead market

### SHORT entry (fade overbought) — only fires when regime == RANGE
Symmetric to LONG: RSI ≥ 70 OR BB_pct ≥ 0.90, bearish candle, vol, ATR.

### Exit
- **TP** = 1.5 × ATR (clamped 1.0%–4.0%) — mean-reversion target
- **SL** = 2.0 × ATR (clamped 1.2%–5.0%) — wider stop allows for noise
- **Max bars** = 10 (~40 hours) — mean-rev thesis is local
- **Regime-change exit**: close the position if regime becomes the *opposite*
  direction (LONG closes on BEAR; SHORT closes on BULL). RANGE→BULL with an
  open LONG is supportive, not contradictory.
- Conservative TP/SL on same bar → assume SL.

### What we did NOT do (to avoid Agent C's trap and the project's overfitting history)
- Did NOT add a momentum filter ("only enter oversold if EMA50 4h slope > 0") —
  exactly the overfit Agent C warned about.
- Did NOT use trailing stop on mean-rev exits. Mean reversion targets a level;
  trailing turns it into trend-following in disguise.
- Did NOT add a daily funding filter. ETH funding patterns differ from BTC and
  re-tuning thresholds per asset is the overfit road.

---

## In-sample results (2020-01 → 2025-12, walk-forward purged 14 days)

### Combined (LONG + SHORT in RANGE)

| Period | N | WR | PF | Total | DD | OK |
|---|---|---|---|---|---|---|
| 2020-01 | 0 | — | — | — | — | (no fire) |
| 2020-07 | 5 | 60% | 3.32 | +1.4% | 0.6% | ✅ |
| 2021-01 | 1 | 0% | — | -5.1% | 5.1% | ❌ (n<5) |
| 2021-07 | 0 | — | — | — | — | (no fire) |
| 2022-01 | 1 | 100% | inf | +2.3% | 0% | ❌ (n<5) |
| 2022-07 | 1 | 100% | inf | +2.1% | 0% | ❌ (n<5) |
| 2023-01 | 1 | 100% | inf | +1.3% | 0% | ❌ (n<5) |
| 2023-07 | 3 | 67% | 151 | +1.1% | 0% | ❌ (n<5) |
| 2024-01 | 3 | 33% | 0.06 | -4.7% | 5.0% | ❌ (n<5) |
| 2024-07 | 0 | — | — | — | — | (no fire) |
| 2025-01 | 0 | — | — | — | — | (no fire) |
| 2025-07 | 0 | — | — | — | — | (no fire) |

**1/12 folds passed**. Most folds have n<5, which fails the fold-quality bar
automatically. Half the folds had zero firings — the setup is just too rare.

### Continuous 2020-2025
- **19 trades**, WR 52.6%, PF 0.66, total -6.1%, DD 9.8%
- Annualized (geometric monthly): **-5.3%**
- 14 months had any trades; 50% positive months
- **Trade frequency: 3.2 trades/year** — well below the 10/yr operability bar

### Bootstrap (n=3000)
- p-value (return ≤ 0 by chance): **0.760** — not significant
- Median resampled total: -6.2% (centered on negative)
- 95th percentile: +7.5% (best plausible outcome still tiny)

### Per-direction audit

| Direction | Folds OK | N | WR | PF | Total | Bootstrap p |
|---|---|---|---|---|---|---|
| LONG  | 0/12 | 7 | 57.1% | 0.73 | -1.7% | 0.586 |
| SHORT | 0/12 | 8 | 62.5% | 0.97 | -0.2% | 0.511 |

Neither direction earns its keep alone. SHORT WR is interesting (62.5%) but PF
just under 1 — high WR cancelled by deep losers.

### Exit reason breakdown (continuous, 19 trades)
- TP: 32% · SL: 26% · Regime change: 26% · Timeout: 16%

The regime-change exit closed 26% of trades — when ETH transitions from RANGE
to a clear direction the position is unwound. This is correct behavior (thesis
broken) but it limits the upside.

---

## Diagnostic: would the strategy work if we relaxed more?

This is **not** a re-tune — the validation above used the frozen PARAMS. But
the brief asks us to confirm or descard the hypothesis honestly, so I checked
what would happen if each restrictive filter were dropped (as a structural
diagnostic, see `diagnose2.py`):

| Variant | N | WR | PF | Total |
|---|---|---|---|---|
| baseline (frozen PARAMS) | 19 | 52.6% | **0.66** | -6.1% |
| no volume confirmation | 46 | 58.7% | 0.95 | -3.0% |
| no regime-change exit | 19 | 52.6% | 0.73 | -6.4% |
| no candle confirmation | 108 | 52.8% | 1.10 | **+6.6%** |
| broader RANGE (ADX<30, BBW<50, sep<4%) | 32 | 59.4% | 0.92 | -3.3% |
| no vol AND no candle | 136 | 52.9% | 1.01 | -2.6% |
| LONG-only | 8 | 50.0% | 0.46 | -4.9% |

**Reading**: even the most permissive variant ("no candle filter", PF 1.10,
+6.6%) is below the project's PF ≥ 1.2 bar. Stripping ALL filters converges
to PF 1.00 — no edge. **The mechanism is structurally weak in ETH**, not
something that filter tuning can rescue. This is consistent with the project's
audit (`docs/AUDITORIA_2026-05.md`) — *the edge is in the EXIT not the entry*,
and a mean-rev exit (fixed TP/SL) cannot inherit the trailing-stop edge that
made trend strategies positive.

---

## Why this differs from Agent C (BTC RANGE, also failed)

| | Agent C (BTC) | Agent I (ETH) |
|---|---|---|
| RANGE definition | EMA20_1d / EMA50_1d dead-zone only | ADX<25 AND BBW<40 AND sep<3% |
| RANGE % of time | 29% | **9.7%** ← much rarer |
| LONG extreme | RSI≤30 AND BB≤0.15 AND bullish | RSI≤30 OR BB≤0.10 AND bullish + vol |
| Total RANGE trades | 25 | 19 |
| RANGE WR | 60% | 52.6% |
| RANGE PF | 1.19 | 0.66 |
| Verdict | "fires too rarely" | "fires too rarely AND PF<1" |

**ETH is worse than BTC for this mechanism.** Both fail the frequency bar, but
BTC at least had PF > 1 on its few RANGE trades; ETH does not. The structural
beta hypothesis ("ETH bounces harder in extremes") is **falsified** by the
data over 2020-2025.

---

## SELF-AUDIT

### What I did right
- **Honest engine**: one-position-at-a-time, no intrabar look-ahead in TP/SL,
  conservative SL-first on same-bar conflicts.
- **All daily features shifted by 1 day** — `compute_macro_daily` returns
  yesterday's daily EMAs for today's use. Verified.
- **All 4h indicators are rolling on past bars only** (RSI/ATR/EMA/BB/ADX use
  `.ewm()` / `.rolling()`; no `.shift(-N)` or future references).
- **`bb_width_pctile` uses `shift(1)` before the rolling percentile** — today's
  percentile is computed against bars strictly before today.
- **`vol_sma20` uses `shift(1)`** — vol ratio compares today to a baseline
  ending yesterday.
- **CUTOFF respected**: `assert df.index.max() <= CUTOFF` in `train.py`. The
  prepared data ends at 2025-12-31 20:00 UTC.
- **Walk-forward with 14-day purge** on each side of each fold.
- **Bootstrap (n=3000)** on the continuous trades.
- **Per-direction audit** — both LONG-only and SHORT-only reported honestly.
- **No selection bias of folds**: every metric above came from the same frozen
  `PARAMS`. I made *one* principled threshold relaxation between the initial
  draft (ADX<20, BBW<30, sep<2%) and the final PARAMS (ADX<25, BBW<40, sep<3%)
  — justified by the diagnostic that showed the original RANGE was only 3.8% of
  bars and produced just 12 trades. This relaxation was applied to all metrics
  uniformly; no subsequent per-fold tuning.
- **`exit_only_on_opposite_regime`** was added in the same pass — the original
  "any regime change" closed 58% of trades; the relaxed "only opposite" closes
  26% which is more reasonable. Both are textbook choices, not fit knobs.
- **I report REJECT honestly** — the brief explicitly said this was a valid
  outcome and the data supports it cleanly.

### What did not work (and why we accept it)
- **The hypothesis itself**: "ETH RANGE bounces are more violent → more
  tradeable than BTC RANGE bounces." The data shows the opposite: ETH spends
  3× less time in RANGE than BTC, and within that time the mean-reversion
  setups produce PF<1.
- **The OR-logic relaxation (vs C's AND-logic)** did NOT lift trade frequency
  to operable levels. The throttling came from the regime filter itself, not
  the extreme definition.
- **The candle confirmation** was the most restrictive filter on frequency
  (108→19 trades). But removing it gives WR≈53% and PF≈1.10 — barely an edge
  and below the project bar. So we don't pretend that "no candle filter" is a
  discovered solution. It's a diagnostic that confirms there's no edge to mine.
- **2024-H1 was the worst fold** (3 trades, -4.7%) — coincided with a sharp
  ETH oscillation that triggered our oversold signals just before deeper drops.
  This is the classic mean-rev failure mode that mean-rev strategies cannot
  fix without becoming trend-followers (which is exactly Agent A's territory).

### Sanity flags (automatic checks)
- PF > 4? No (PF 0.66).
- WR > 70%? No (52.6%).
- DD < 3%? No (9.8%).
- N trades < 30? **Yes (19) — sanity flag tripped → too rare to operate.**

The sanity flag is doing its job: warning that the strategy is too rare for
inference. The flag together with PF < 1 is unambiguous: REJECT.

### What the orchestrator should expect on 2026 OOS
Given how rare the firings are (19 in 6 years = 3.2/yr), 2026 OOS Jan-Feb is
likely to produce **0–1 trades**. Whatever the outcome is, it will not be
statistically meaningful. The OOS will mostly confirm that **the strategy
stays flat**, which is consistent with the in-sample diagnosis.

If the OOS unexpectedly produces 3+ trades and they all happen to win, that
would still be lottery-level evidence (n=3 means nothing). Do not promote on
that basis.

---

## Files

- `strategy.py` — frozen `PARAMS`, `detect_regime()`, `signal()`, `simulate()`,
  `run_engine()`. Importable, self-contained (no project imports).
- `train.py` — loads ETH 4h with the 2025-12-31 cutoff, computes features,
  runs walk-forward + continuous + per-direction + bootstrap, writes
  `results.json`.
- `results.json` — full numerical output of the latest run.
- `diagnose.py` — RANGE-frequency exploration (informed the one principled
  threshold relaxation; documented above).
- `diagnose2.py` — "what if we drop each filter" structural diagnostic
  (informed the REJECT verdict but did not change the validated PARAMS).

## Run

```bash
C:/Python/python.exe experiments/agent_I/train.py
```

Takes ~30 seconds. Output is printed and persisted in `results.json`.
