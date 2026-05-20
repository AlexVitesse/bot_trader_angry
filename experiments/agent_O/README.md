# Agent O — ML LightGBM **cross-asset** Classifier (SOL/USDT 4h)

## TL;DR — Honest verdict

**Outcome: NEGATIVE. The LightGBM cross-asset (BTC + ETH + SOL) classifier
does NOT have a statistically significant edge on SOL/USDT 4h.**

This is the THIRD independent honest-validation experiment on 4h crypto
classification to fail with the same fingerprint (B BTC, G ETH, O SOL).
The three results, taken together, **close the line "ML 4h classification
in crypto" for this project** as the brief anticipated.

| Metric | Value |
|---|---|
| Purged CV folds passing (>=3 trades, PF>=1.2, total>0) | **2 / 11** |
| Mean train AUC | **0.791** |
| Mean test AUC | **0.499** |
| Aggregate test trades | 723 |
| Aggregate WR | 36.4% |
| Aggregate PF | 0.86 |
| Aggregate total PnL across folds | **-170.3%** |
| Annualized return on WF trades (linear) | **-44.1%** |
| Compound equity final (start = 1.0) | **0.128** (i.e. -87.2%) |
| Multiplicative MaxDD on WF equity | **-90.4%** |
| Bootstrap p-value (H0: mean<=0) | **0.983** (one-sided; very strong reject of any edge) |
| 95% CI on mean PnL/trade | [-0.460%, -0.011%] |

The train→test AUC gap of **0.292** matches Agent G (worst-in-project) and
exceeds Agent B's 0.230 — adding BTC + ETH context did NOT mitigate the
overfitting. The cross-asset hypothesis is empirically refuted.

---

## What was built

### Hypothesis
- Agents B (BTC, 11 own-features) and G (ETH, 16 features inc. ETH/BTC ratio)
  both failed: train AUC ~0.75-0.81 collapsing to test AUC ~0.51-0.52.
- New idea: SOL has beta ~1.5 to BTC. Maybe SOL is predictable from the
  **broader** market state (BTC + ETH context), capturing a non-linearity
  that rules can't see.
- The brief explicitly framed this as the last test: if cross-asset ML
  doesn't work on SOL either, close the ML line definitively.

### Model
- `lightgbm.LGBMClassifier`, at the brief's regularization bound:
  - `max_depth=4`, `num_leaves=15`, `n_estimators=200`, `min_data_in_leaf=80`,
    `feature_fraction=0.8`, `bagging_fraction=0.8`, `reg_alpha=1.0`,
    `reg_lambda=1.0`, `learning_rate=0.03`, `random_state=42`.
- `StandardScaler` on inputs; class balance via `sample_weight`.

### Features (16 — every one justified, ≤20 budget)

| # | Feature | Group | Rationale |
|---|---|---|---|
| 1 | `sol_rsi14_4h` | SOL 4h | own momentum / OB-OS |
| 2 | `sol_bb_pct_4h` | SOL 4h | own mean-rev position |
| 3 | `sol_atr_pct_4h` | SOL 4h | own volatility regime |
| 4 | `sol_vol_ratio_4h` | SOL 4h | own volume thrust |
| 5 | `sol_ret_5_4h` | SOL 4h | own 5-bar momentum |
| 6 | `btc_ema20_50_ratio_1d` | BTC daily | broader trend (shift 1d) |
| 7 | `btc_rsi14_4h` | BTC 4h | leading 4h momentum |
| 8 | `btc_ret_5_4h` | BTC 4h | lead/lag with SOL |
| 9 | `btc_vol_ratio_4h` | BTC 4h | broad market thrust |
| 10 | `btc_regime_1d` | BTC daily | encoded -1/0/+1 (shift 1d) |
| 11 | `ethbtc_zscore_90d` | ETH/BTC daily | alt rotation signal |
| 12 | `eth_regime_1d` | ETH daily | alt market regime |
| 13 | `eth_ret_5_4h` | ETH 4h | alt sentiment lead/lag |
| 14 | `sol_btc_corr_168` | cross | rolling 168-bar SOL-BTC corr |
| 15 | `sol_btc_ret_div_5` | cross | SOL minus BTC 5-bar return |
| 16 | `funding_zscore` | macro | BTC funding z-score (shift 1) |

All multi-timeframe / daily / external inputs are `.shift(1)`-ed BEFORE
reindex to the SOL 4h grid (`build_features` in `strategy.py`). **No
look-ahead.**

### Label
Binary: `1` if entering LONG at `close[i]` hits **+4% (TP)** before **-2.5% (SL)**
within 12 bars (48h). Pessimistic tie-break (both hit in same bar -> 0).
Uses only bars `i+1..i+max_bars`. Label balance: 36.5% (vs 33-40% on B/G).

> Note: TP/SL scaled UP from B's 3%/1.5% to **4%/2.5%** to match SOL's higher
> volatility (ATR% ~3.55% vs BTC ~1.2%). Reward:risk = 1.6:1 (vs B's 2:1) —
> realistic given SOL's wider expected range.

### Exit (live + WF backtest)
**FIXED TP/SL** (no trailing). Explicitly chosen over trailing because the
project's PASO 0 audit documented that the trailing simulator hides a
sub-bar look-ahead bug that inflated metrics. With fixed TP/SL the test is
clean: did the classifier identify entries that resolve into a +4% before
-2.5% move? The model is the only thing being tested.

- Pessimistic intrabar order: if a single bar touches both SL and TP, count
  it as SL (-2.5%). Conservative.
- Timeout at max_bars=12 -> exit at that bar's close.
- Commissions: 0.05% per side -> 0.10% round trip.

### Backtest engine
Strictly sequential, no overlap: a new trade only opens AFTER the previous
one closes (matches `agent_B` / `agent_G` engines and the project's PASO 0
guard). The "open multiple trades in the same direction" bug from
`evaluate_new_pairs_v15.py` is NOT present.

### Cutoff
**`<= 2025-12-31` enforced on every source**: SOL 4h, BTC 4h, ETH 4h, BTC 1d,
ETH 1d, ETH/BTC daily, BTC funding. The inviolable rule.

---

## Validation protocol

| Requirement | Implementation | Result |
|---|---|---|
| CUTOFF 2025-12-31 | Hard slice on every parquet load | 11,813 SOL 4h bars; 11,113 aligned samples |
| Purged time-series CV | Custom: expanding train, 84-bar gap (=14 days >= label horizon), 12 chronological splits | 11 valid folds |
| Class balance | `sample_weight` inverse to class frequency | Done |
| Threshold calibration | TRAIN ONLY; lowest thr in [0.50, 0.70] with train precision >= 0.50 and signal_rate >= 0.02 | thr 0.50 every fold |
| Bootstrap significance | 5000 resamples on 723 aggregate test trades, H0: mean<=0 | p = **0.983** |
| Synthetic SOL | 20 block-bootstrap SOL series + REAL BTC/ETH context | 0/20 positive; real percentile 100% |
| Feature stability check | Top-5 importance count per fold | High (see below) |
| No grid search of TP/SL on data | Label thresholds fixed a priori | OK |
| No test-set re-tuning | Threshold frozen per fold; no post-hoc rescue | OK |

---

## Per-fold results

```
Fold  1: train 2020-09-08->2021-12-11 test 2021-12-25->2022-05-02 AUC_tr=0.817 AUC_te=0.554 thr=0.50 n= 30 wr=43.33% pf=1.15 tot_pnl= +6.5% --
Fold  2: train 2020-09-08->2022-04-18 test 2022-05-03->2022-09-06 AUC_tr=0.810 AUC_te=0.505 thr=0.50 n= 24 wr=37.50% pf=0.90 tot_pnl= -3.9% --
Fold  3: train 2020-09-08->2022-08-23 test 2022-09-06->2023-01-18 AUC_tr=0.803 AUC_te=0.448 thr=0.50 n= 32 wr=25.00% pf=0.50 tot_pnl=-31.2% --
Fold  4: train 2020-09-08->2023-01-04 test 2023-01-18->2023-06-05 AUC_tr=0.807 AUC_te=0.463 thr=0.50 n=105 wr=27.62% pf=0.57 tot_pnl=-84.5% --
Fold  5: train 2020-09-08->2023-05-15 test 2023-06-05->2023-10-28 AUC_tr=0.805 AUC_te=0.542 thr=0.50 n= 65 wr=38.46% pf=0.94 tot_pnl= -6.5% --
Fold  6: train 2020-09-08->2023-10-14 test 2023-10-29->2024-03-03 AUC_tr=0.798 AUC_te=0.518 thr=0.50 n=121 wr=47.93% pf=1.38 tot_pnl=+62.4% OK
Fold  7: train 2020-09-08->2024-02-18 test 2024-03-04->2024-07-13 AUC_tr=0.789 AUC_te=0.505 thr=0.50 n= 95 wr=34.74% pf=0.80 tot_pnl=-32.5% --
Fold  8: train 2020-09-08->2024-06-28 test 2024-07-13->2024-11-23 AUC_tr=0.784 AUC_te=0.479 thr=0.50 n= 73 wr=32.88% pf=0.73 tot_pnl=-33.8% --
Fold  9: train 2020-09-08->2024-11-09 test 2024-11-23->2025-04-03 AUC_tr=0.773 AUC_te=0.442 thr=0.50 n= 79 wr=29.11% pf=0.62 tot_pnl=-55.9% --
Fold 10: train 2020-09-08->2025-03-18 test 2025-04-03->2025-08-15 AUC_tr=0.761 AUC_te=0.520 thr=0.50 n= 56 wr=46.43% pf=1.30 tot_pnl=+23.4% OK
Fold 11: train 2020-09-08->2025-07-31 test 2025-08-15->2025-12-30 AUC_tr=0.751 AUC_te=0.516 thr=0.50 n= 43 wr=34.88% pf=0.80 tot_pnl=-14.3% --

=== WF Summary: 2/11 folds pass
```

- Only 2 of 11 folds (6 and 10) clear the bare-minimum bar (PF>=1.2, total>0).
  Both have moderate WR (~47%) and modest PF (~1.3) — quintessential "lucky
  fold" pattern, not edge.
- Catastrophic folds: 4 (-84.5%, 105 trades), 9 (-55.9%, 79 trades), 8 (-33.8%).
  The model produces lots of trades in mid-2023 and late-2024 that lose money
  systematically — i.e., it's confident and wrong in those regimes.
- Test AUC dips BELOW 0.5 in folds 3, 4, 8, 9 — the model is **actively worse
  than random** in those windows.

## Feature stability (count in top-5 across folds)

```
sol_atr_pct_4h                top5_in: 11/11   imp_mean=217.2 std=13.2
btc_ema20_50_ratio_1d         top5_in: 11/11   imp_mean=193.5 std=25.3
ethbtc_zscore_90d             top5_in: 11/11   imp_mean=203.9 std=20.2
sol_btc_corr_168              top5_in: 10/11   imp_mean=201.7 std=55.0
btc_rsi14_4h                  top5_in:  4/11   imp_mean=140.1 std=24.5
funding_zscore                top5_in:  4/11   imp_mean=143.6 std=28.3
sol_rsi14_4h                  top5_in:  3/11   imp_mean=139.2 std=18.8
btc_ret_5_4h                  top5_in:  1/11   imp_mean=119.8 std=14.4
...
```

The cross-asset features I added (`ethbtc_zscore_90d`, `sol_btc_corr_168`,
`btc_ema20_50_ratio_1d`) are **the most stable / most important features in
the model** — exactly the literature prior. SOL's own ATR% rounds out the
top 4. **And yet test AUC = 0.499** — confirming once more that high
importance in-sample is not the same as OOS predictive value on 4h crypto.

Note: `btc_regime_1d` and `eth_regime_1d` (encoded -1/0/+1) get near-zero
importance — the tree can read the same info from the continuous
`*_ema20_50_ratio_1d` features. Not a feature engineering mistake; just
redundancy that LightGBM ignored.

---

## Aggregate bootstrap (5000 resamples, seed 42)

```
Total test trades across folds: 723
Mean PnL/trade: -0.236%  (total: -170.3%)
WR: 36.38%  PF: 0.86  MaxDD on cum equity (linear): -204.1%
Multiplicative compound: 1.0 -> 0.128  (-87.2%)  Multiplicative MaxDD: -90.4%
Annualized return on WF trades (linear): -44.1%

Bootstrap (N=5000) one-sided p (H0: mean<=0): 0.9830
Bootstrap two-sided p: 0.0338
95% CI on mean PnL: [-0.460%, -0.011%]
```

The 95% CI is **entirely below zero**. The one-sided p-value of 0.983 means
that under the null "true mean = 0" the observed mean is more negative than
98.3% of resamples — i.e., this is **significantly negative**. The two-sided
p of 0.034 is significant at α=0.05 but in the WRONG direction. **There is
no edge to be salvaged here.**

---

## Synthetic SOL test (20 block-bootstrap universes)

For each synth run: shuffle SOL log-returns in 30-bar blocks (preserves short-
range autocorrelation but destroys SOL's specific structure), keep BTC/ETH/
funding context REAL, and run the trained model + threshold.

```
REAL SOL: n=1191  total=+833.8%  wr=50.97%  pf=1.56  annual=+154.8%
Synth median: -65.2% annual
Synth positive: 0/20
Real percentile: 100% (real beats every synth)
```

**Interpretation matters here.** The +154.8% "annual" on real data is
the **in-sample backtest** with the final retrained model — same in-sample
behaviour that Agent B's threshold-sweep showed (PF goes up at higher
thresholds in-sample). The fact that real >> synth tells us the model HAS
learned something specific to SOL's real price path (it would, with full
training data). What it DOES NOT tell us is whether that learning generalizes:
that's what the purged CV measured, and the purged CV says **no, it does not.**

Reading the synth correctly: real-SOL in-sample +155% vs synth median -65%
vs real-SOL OOS WF -44%. The 200-point gap between in-sample real and OOS
WF is the bias the model brings to its predictions. The synth test confirms
the model is memorizing real SOL — it's not arbitrarily overfit (otherwise
synths might score positive too) — but the memorization is non-stationary,
which is the same fingerprint as Agents B and G.

---

## Self-audit (every decision and whether it peeked at test data)

| Decision | When made | Looked at test data? |
|---|---|---|
| 16 cross-asset features | A priori, derived from CLAUDE.md (SOL beta 1.5 to BTC), V15_SOL_evaluation.md, Agent G's importance ranking, and the brief | No |
| Model = LightGBM at brief's caps (max_depth=4, etc.) | Brief constraints, no tuning | No |
| Label TP=4%, SL=2.5%, 12 bars | A priori (project says SOL ATR 1.5x ETH; B used 3%/1.5%; scaling) | No |
| Exit = FIXED TP/SL (no trailing) | A priori (PASO 0 documents trailing bug) | No |
| Threshold calibration target precision=0.50, signal_rate>=0.02 | A priori (50% beats the 36.5% base rate; matches Agent B's >=0.55 absolute floor philosophy, relaxed for SOL's lower base rate) | No |
| Threshold per fold (range 0.50-0.50 — calibration floor binding) | Calibrated on TRAIN ONLY | No |
| Bootstrap N=5000, seed=42 | A priori (matches B, G) | No |
| Min train = 2000 bars, gap=84 bars (=14 days), 12 splits | A priori | No |
| Synth N=20, block=30 | A priori | No |
| Regime filter: skip LONG if BTC AND ETH daily regime <= -1 | A priori (conservative; let the model handle moderate BEAR) | No |
| Funding veto: skip if funding_z > 2.0 | A priori (contrarian to crowded-long extremes) | No |

I did **NOT** re-tune any param after seeing CV results. I did not adjust
the feature set, the LightGBM hyperparams, the label thresholds, the
calibration floor, or the exit rules. The version reported here is the one
I committed to a priori.

I considered (and rejected) running a post-hoc threshold sweep like Agent B
did, because:
- B already documented that "rescue at threshold >=0.58" is test-set
  data snooping.
- The synth test result alone is sufficient evidence the model has no edge.

---

## Comparison with Agents B and G

| Experiment | Asset | Features | Train AUC | Test AUC | Gap | Aggregate PF | Bootstrap p (1-sided) |
|------------|-------|----:|----------:|---------:|----:|-------------:|----------------------:|
| Agent B (BTC) | BTC | 11 own | 0.751 | 0.520 | 0.231 | 0.96 | 0.607 |
| Agent G (ETH) | ETH | 16 (inc. ETH/BTC, BTC daily) | 0.805 | 0.513 | 0.292 | 0.92 | 0.808 |
| **Agent O (SOL)** | **SOL** | **16 cross-asset (SOL+BTC+ETH)** | **0.791** | **0.499** | **0.292** | **0.86** | **0.983** |

**Key observations:**

1. Cross-asset features do **NOT** reduce the train→test AUC gap. O's gap
   (0.292) ties G's (worst-in-project) and exceeds B's (0.231).
2. Test AUC is **worst** for SOL (0.499 — literally indistinguishable from
   random; both B and G were marginally above 0.5).
3. Bootstrap p-value is **worst** for SOL (0.983 — model is significantly
   anti-edge in the wrong direction).
4. The aggregate PF degrades monotonically: 0.96 (BTC, weakest negative) ->
   0.92 (ETH) -> 0.86 (SOL, strongest negative).
5. The literature priors (ETH/BTC ratio, SOL-BTC corr, BTC daily regime) get
   high feature importance every time AND fail to translate to OOS edge
   every time. This is a robust finding, not an artifact of one model.

**The cross-asset hypothesis (richer feature space about the broader market
state should help SOL specifically) is empirically falsified.** Adding more
context did not help — if anything it gave the model more rope with which
to overfit (gap matches G's worst-in-project).

---

## Key insight (the negative result IS the finding)

After Agents B, G, and now O — across three independent assets, two model
families, with own-features, ETH-specific features, and cross-asset
features respectively — the conclusion is:

**ML classifiers on 4h crypto with theoretically-grounded features under
honest validation reliably produce train AUC ~0.75-0.81 collapsing to test
AUC ~0.50.** This is structural, not a quirk of any one configuration.

This **definitively closes the line "ML 4h classification with binary TP/SL
labels in crypto"** for this project, exactly as the brief anticipated.

What this means for the project:
- The edge documented in `experiments/VERDICTO_FINAL.md` lives in the EXIT
  (trailing, volatility-targeted) and in REGIME-AWARE rule systems
  (V2 = A + F_BTC with bootstrap p=0.031), not in entry probability.
- Continuing to ML-engineer the entry side is not a productive path.
- `MEMORY.md`'s SOL approval (`RULES_TP3_SL1.5_TRAIL_tight_imm`, PF 2.56,
  WF 8/10) was generated by the broken evaluation engine (`evaluate_new_pairs_v15.py`)
  documented in PASO 0. PASO 0 already showed that with the engine fixed,
  ADA's WF collapses from 10/12 to 5/12 — SOL's would too. Agent O's
  result corroborates that **no ML can rescue SOL** either.

---

## Recommendation

**Do NOT deploy this model. Specifically:**
1. Do not lower the threshold to "get more trades" — at thr=0.50 we already
   have 723 trades and PF 0.86. Lower would be worse.
2. Do not raise the threshold post-hoc — synthesizing edge from B's
   diagnostic threshold sweep was explicitly rejected as data snooping.
3. Do not add more features — going from 11 (B) to 16 own+exogenous (G)
   to 16 cross-asset (O) monotonically WORSENED the bootstrap p-value.
   The relationship between feature count and OOS edge is, if anything,
   slightly NEGATIVE in this study.
4. Do not relax the LightGBM caps — looser models will overfit harder, not
   less; Agent G's `test_learn_from_losses` style empirically showed re-
   tuning makes 14/20 universes WORSE.
5. **Close ML classifiers as a research direction for SOL 4h LONG-only**,
   matching the closures by B (BTC) and G (ETH).

The honest path forward for SOL remains:
- Don't run SOL standalone. The honest evaluation in V15_SOL_evaluation.md
  (with the broken engine bugs) was already shaky; with fixed engines it
  collapses (see ADA PASO 0 result, which generalizes).
- The only operable crypto strategy in this project remains **BTC V2 =
  Agent A (Donchian breakout) + Agent F_BTC (vol-compression breakout)**,
  bootstrap p=0.031, expected ~10% annual.
- Capital exposure to SOL, if any, via spot DCA — NOT algorithmic 4h
  classification.

---

## Files

| File | Purpose |
|---|---|
| `strategy.py` | `PARAMS`, `FEATURES`, `build_features()`, `create_labels()`, `signal()`, `simulate()` (fixed TP/SL), `backtest()`, `load_model()` |
| `train.py` | Load 7 sources (cutoff 2025-12-31), purged CV, retrain on all data, save artifacts |
| `bootstrap_test.py` | 5000-resample bootstrap on aggregate WF trades |
| `synth_test.py` | 20 block-bootstrap synth SOL universes; uses REAL BTC/ETH context |
| `model.pkl` | LGBMClassifier retrained on ALL <=2025-12-31 data |
| `scaler.pkl` | StandardScaler fit on ALL <=2025-12-31 features |
| `trained_params.json` | PARAMS with final threshold (0.500) |
| `cv_results.csv` | Per-fold AUC/WR/PF/PnL |
| `cv_summary.json` | Headline metrics + feature stability |
| `bootstrap_result.json` | p-value, CI, PF, WR over 723 aggregate trades |
| `synth_results.csv` / `synth_summary.json` | 20 synth universes |
| `wf_trades.csv` | All 723 walk-forward test trades |

---

## How to reproduce

```bash
cd C:/Users/pcdec/OneDrive/Documentos/MIS\ EMPRENDIMIENTOS/BOTDETRADINGAGRESIVO
C:/Python/python.exe -u experiments/agent_O/train.py            # ~60s
C:/Python/python.exe -u experiments/agent_O/bootstrap_test.py   # ~10s
C:/Python/python.exe -u experiments/agent_O/synth_test.py       # ~300s (20 universes)
```

`model.pkl` was trained with sklearn 1.8.0 + lightgbm 4.6.0 (same as the
production bot's venv).

---

## Conclusion

A LightGBM classifier on 16 cross-asset (BTC + ETH + SOL) features,
validated with strict purged time-series CV, FIXED TP/SL simulation (no
intra-bar trailing bug), a 5000-iteration bootstrap, and a 20-universe
synthetic test, **does NOT show a statistically significant edge on
SOL/USDT 4h LONG:**

- **2 / 11** folds pass the minimum bar
- Train AUC **0.791** -> Test AUC **0.499** (gap **0.292**, matches G as
  worst-in-project)
- Aggregate 723 trades: PF **0.86**, WR **36.4%**, total **-170.3%**, multiplicative
  equity 1.0 -> 0.128, max DD **-90%**
- Linear annual on WF trades: **-44.1%**
- Bootstrap one-sided p = **0.983** (significantly worse than null)
- Cross-asset features dominate importance (top 4 of 5 stable across folds)
  AND fail to generalize — same fingerprint as G's ETH/BTC ratio

**Closes the hypothesis "cross-asset (BTC + ETH) features rescue SOL where
own-features failed."** Together with Agents B and G, **closes the line
"ML classifier with binary TP/SL label in 4h crypto"** for this project.

The brief asked: "if cross-asset ML tampoco funciona en SOL (probable, dado
B y G), REPORTAR NEGATIVO. Importante: confirmar empíricamente si añadir
contexto BTC+ETH mueve la aguja en SOL — si no, cierra definitivamente la
línea ML en crypto 4h." The answer is unambiguous: **NEGATIVE. The needle
did NOT move. The line is closed.**
