# Agent G — ML Classifier (LightGBM) on ETH/USDT 4h with Anti-Overfitting Protocol

## TL;DR — Honest verdict

**Outcome: NEGATIVE. The LightGBM classifier does NOT have a statistically
significant edge on ETH/USDT 4h, even with ETH-specific features
(notably ETH/BTC ratio) that the literature flagged as the most likely
to be informative.**

This **mirrors and reinforces** Agent B's finding on BTC. ML probabilistic
classifiers with honest validation appear to lack edge on both BTC and ETH
4h with this label structure (LONG, TP=3%, SL=1.5%, 12-bar horizon).

| Metric | Value |
|---|---|
| Purged CV folds passing (>=3 trades, PF>=1.2, total>0) | **2 / 11** |
| Mean train AUC | **0.805** |
| Mean test AUC | **0.513** |
| Aggregate test trades | 983 |
| Aggregate WR | 32.04% |
| Aggregate PF | 0.92 |
| Aggregate total PnL across folds | **-72.1%** |
| Annualized return on WF trades | **-20.9%** |
| Max DD on cumulative WF equity | 83.1% |
| Bootstrap p-value (H0: mean<=0) | **0.808** (one-sided) |
| 95% CI on mean PnL/trade | [-0.232%, +0.094%] |

The train→test AUC gap of **0.292** is the largest seen in the project's
honest-validation rounds — actually *worse* overfitting than Agent B's BTC
GBM (gap 0.23). Despite ETH/BTC ratio being the #1 most stable and most
important feature (matching the literature's prior), it does **not**
translate to OOS edge.

---

## What was built

A LightGBM binary classifier on a 16-feature matrix focused on the
hypothesis "ETH alpha is in the ETH/BTC ratio + BTC daily regime."

### Model (within brief's constraints)
- `lightgbm.LGBMClassifier` with:
  - `max_depth=4`, `num_leaves=15`, `n_estimators=200`,
    `min_data_in_leaf=80`, `feature_fraction=0.8`, `bagging_fraction=0.8`,
    `reg_alpha=1.0`, `reg_lambda=1.0`, `learning_rate=0.03`.
- `StandardScaler` on inputs; class balance via `sample_weight`.

### Features (16 — each justified, no data mining)

| # | Feature | Rationale |
|---|---|---|
| 1 | `rsi14_4h` | 4h momentum / OB-OS |
| 2 | `rsi_slope_4h` | RSI acceleration |
| 3 | `bb_pct_4h` | Mean-reversion position |
| 4 | `atr_pct_4h` | Current volatility |
| 5 | `vol_ratio_4h` | Volume thrust vs 20-bar mean |
| 6 | `ret_5_4h` | 5-bar (20h) momentum |
| 7 | `dist_high20_4h` | Extension from 20-bar high |
| 8 | `dist_low20_4h` | Proximity to 20-bar low (support) |
| 9 | `ema20_50_ratio_1d_eth` | ETH daily trend |
| 10 | `close_above_ema200_1d_eth` | ETH macro structure |
| 11 | `ema20_50_ratio_1d_btc` | BTC daily trend (ETH follows BTC) |
| 12 | `btc_ret_5_4h` | BTC 5-bar return (lead/lag) |
| 13 | `btc_vol_ratio_4h` | BTC volume thrust |
| 14 | `ethbtc_slope_30d` | 30-day slope of ETH/BTC ratio ★ |
| 15 | `ethbtc_zscore_90d` | 90-day z-score of ETH/BTC ratio ★ |
| 16 | `funding_zscore` | BTC funding sentiment |

★ = the literature-flagged "signal #1 for ETH". MTF inputs (`*_1d`, funding,
ETH/BTC daily) are `.shift(1)`-ed before reindexing to 4h. **No look-ahead.**

### Label
Binary: `1` if entering LONG at `close[i]` hits +3% before -1.5% within 12
bars (48h). Pessimistic tie-break (both hit -> 0). Uses ONLY bars `i+1..i+12`.

### Exit (live + WF backtest)
Trailing stop, mirror of Agent B for direct comparability:
- Initial SL = `entry * (1 - 0.015)`
- Trailing activates once `close >= entry * 1.01`; trail = 1.5% from
  the peak CLOSE (not high, conservative).
- **Intrabar safety**: on each bar we first check `low <= stop` using the
  stop computed from prior closes; only then update peak/stop with the new
  close. Eliminates the look-ahead trailing bug from PASO0.

### Backtest engine
Strictly sequential: a new trade only opens AFTER the previous one closes.
No overlapping positions (the PASO0 bug is not present).

### Cutoff
**`<= 2025-12-31` enforced on every load** — applied to ETH 4h, ETH 1d,
BTC 4h, BTC 1d, ETH/BTC daily, and BTC funding parquet.

---

## Validation protocol

| Requirement | Implementation | Result |
|---|---|---|
| CUTOFF 2025-12-31 | Hard slice on every parquet load | 13,151 ETH 4h bars; 12,197 aligned samples |
| Purged time-series CV | Custom: expanding train, 84-bar gap (=14 days >= label horizon), 12 chronological splits | 11 valid folds |
| Class balance | `sample_weight` inverse to class frequency | Done |
| Threshold calibration | TRAIN ONLY; lowest thr in [0.50, 0.70] with train precision >= 0.55 and signal_rate >= 0.02 | thr ~0.50-0.51 per fold |
| Bootstrap significance | 5000 resamples on 983 aggregate test trades, H0: mean<=0 | p = **0.808** |
| Feature stability check | Top-5 importance count per fold | High (see below) |
| No grid search of TP/SL on data | Label thresholds fixed a priori | OK |
| No test-set re-tuning | Threshold frozen per fold | OK |
| No model retuning after seeing results | Single LightGBM config | OK |

---

## Per-fold results

```
Fold  1: train 2020-01-21->2021-05-21 test 2021-06-04->2021-10-23 AUC_tr=0.855 AUC_te=0.526 thr=0.50 n=101 wr=29.70% pf=1.10 tot_pnl= +9.7% --
Fold  2: train 2020-01-21->2021-10-09 test 2021-10-23->2022-03-15 AUC_tr=0.835 AUC_te=0.525 thr=0.50 n= 98 wr=27.55% pf=0.73 tot_pnl=-27.3% --
Fold  3: train 2020-01-21->2022-02-28 test 2022-03-15->2022-08-05 AUC_tr=0.818 AUC_te=0.568 thr=0.50 n= 46 wr=36.96% pf=0.83 tot_pnl= -6.9% --
Fold  4: train 2020-01-21->2022-07-22 test 2022-08-05->2023-01-02 AUC_tr=0.801 AUC_te=0.431 thr=0.50 n= 76 wr=27.63% pf=0.60 tot_pnl=-31.4% --
Fold  5: train 2020-01-21->2022-12-12 test 2023-01-02->2023-06-11 AUC_tr=0.798 AUC_te=0.495 thr=0.51 n=120 wr=31.67% pf=0.91 tot_pnl=-10.8% --
Fold  6: train 2020-01-21->2023-05-23 test 2023-06-11->2023-12-19 AUC_tr=0.798 AUC_te=0.490 thr=0.51 n= 81 wr=20.99% pf=0.79 tot_pnl=-18.6% --
Fold  7: train 2020-01-21->2023-12-03 test 2023-12-19->2024-05-15 AUC_tr=0.804 AUC_te=0.496 thr=0.50 n=115 wr=35.65% pf=0.87 tot_pnl=-14.1% --
Fold  8: train 2020-01-21->2024-04-30 test 2024-05-15->2024-10-14 AUC_tr=0.797 AUC_te=0.504 thr=0.50 n= 99 wr=33.33% pf=1.05 tot_pnl= +4.7% --
Fold  9: train 2020-01-21->2024-09-29 test 2024-10-14->2025-03-09 AUC_tr=0.788 AUC_te=0.515 thr=0.51 n=116 wr=31.90% pf=0.77 tot_pnl=-25.1% --
Fold 10: train 2020-01-21->2025-02-23 test 2025-03-10->2025-08-01 AUC_tr=0.783 AUC_te=0.528 thr=0.51 n= 81 wr=35.80% pf=1.45 tot_pnl=+32.8% OK
Fold 11: train 2020-01-21->2025-07-18 test 2025-08-01->2025-12-28 AUC_tr=0.775 AUC_te=0.567 thr=0.51 n= 50 wr=50.00% pf=1.38 tot_pnl=+15.0% OK

=== WF Summary: 2/11 folds pass
```

- Only 2 of 11 folds clear the bare-minimum bar (PF>=1.2, total>0): the most
  recent two (Mar-Aug 2025 and Aug-Dec 2025). Every prior fold loses money.
- The **late-period success** is suspicious: it could be the model finally
  learning a real pattern (good news), or could be a recent regime that
  resembles training data more closely (selection survival).
- Three big losing folds (2, 4, 9) drag the aggregate to -72%.

## Feature stability (count in top-5 across folds)

```
ethbtc_zscore_90d             top5_in: 11/11   imp_mean=228.9 std=28.4
ethbtc_slope_30d              top5_in: 10/11   imp_mean=171.0 std=26.5
ema20_50_ratio_1d_btc         top5_in:  9/11   imp_mean=168.8 std=36.6
atr_pct_4h                    top5_in:  7/11   imp_mean=157.7 std=35.1
ema20_50_ratio_1d_eth         top5_in:  6/11   imp_mean=153.3 std=18.8
funding_zscore                top5_in:  6/11   imp_mean=151.8 std=36.4
dist_low20_4h                 top5_in:  5/11   imp_mean=156.3 std=21.8
btc_vol_ratio_4h              top5_in:  1/11   imp_mean=112.2 std=17.9
dist_high20_4h                top5_in:  0/11   imp_mean=124.2 std=11.9
ret_5_4h                      top5_in:  0/11   imp_mean= 89.6 std=14.8
vol_ratio_4h                  top5_in:  0/11   imp_mean= 79.5 std=14.4
bb_pct_4h                     top5_in:  0/11   imp_mean= 99.3 std=16.0
rsi14_4h                      top5_in:  0/11   imp_mean=107.4 std=17.5
rsi_slope_4h                  top5_in:  0/11   imp_mean=101.3 std=18.5
btc_ret_5_4h                  top5_in:  0/11   imp_mean= 89.4 std=17.6
close_above_ema200_1d_eth     top5_in:  0/11   imp_mean=  3.5 std= 3.3
```

**The literature was right about WHICH feature matters: `ethbtc_zscore_90d` is
the #1 most stable and most important feature (top-5 in all 11 folds, mean
importance 229).** The literature was wrong about whether knowing this feature
is enough to extract OOS edge with a tree-based classifier on 4h bars.

Final model (retrained on ALL <= cutoff data) feature importance:
1. `ethbtc_zscore_90d` (292) ★
2. `ethbtc_slope_30d` (224) ★
3. `ema20_50_ratio_1d_btc` (202)
4. `atr_pct_4h` (197)
5. `dist_low20_4h` (184)
6. `funding_zscore` (176)
...

ETH/BTC ratio features rank **#1 and #2 of 16** — directly confirming the
hypothesis they should be the most predictive — yet test AUC stays at 0.513.

---

## Aggregate bootstrap

```
Total test trades across folds: 983
Observed mean PnL: -0.073%/trade  sum: -72.1%  WR: 32.04%  PF: 0.92  MaxDD: 83.1%
Bootstrap (N=5000) one-sided p-value (H0: mean<=0): 0.8078
Bootstrap two-sided p-value: 0.3768
95% CI on mean PnL: [-0.232%, +0.094%]
```

The 95% CI straddles zero with a slightly negative center. **No edge.** The
one-sided p-value of 0.808 is the "probability of seeing >= observed mean if
true edge were zero" — at 81% it means the observed mean is meaningfully
*below* the null distribution mean (i.e. worse than random).

---

## Self-audit (every decision and whether it peeked at test data)

| Decision | When made | Looked at test data? |
|---|---|---|
| 16-feature list (focused on ETH/BTC + BTC context) | A-priori, before any model run, derived from literature claim and the project's `MEMORY.md` | No |
| Model = LightGBM, max_depth=4, num_leaves=15, n_est=200, min_data_in_leaf=80, lr=0.03, reg_alpha=reg_lambda=1.0 | Constraints in the brief (kept at the bounds) | No |
| Label thresholds TP=3%, SL=1.5%, 12 bars | A-priori (matches Agent B for comparability) | No |
| Trailing exit (sl=1.5%, trail=1.5%, activate=+1%, max 24 bars) | A-priori (mirrors Agent B) | No |
| Threshold calibration target precision=0.55, signal_rate>=0.02 | A-priori | No |
| Threshold per fold | TRAIN ONLY | No |
| Bootstrap N=5000, seed=42 | A-priori | No |
| Min train = 2000 bars, gap = 84 bars (= label horizon), 12 splits | A-priori | No |
| Regime filter: block LONG if BOTH eth and btc daily ratios < -3% | A-priori (conservative, looser than Agent B's -2% single-asset) | No |

I did **NOT** re-tune any param after seeing CV results. I did not adjust
the feature set, the model hyperparams, the threshold floor, or the trailing
exit after observing the disappointing results. The version reported here
is the one I committed to a priori.

I also did NOT try LightGBM with looser caps to "rescue" the result.
`test_learn_from_losses.py` already demonstrated empirically that re-tuning
after results empirically worsens 14/20 universes — exactly the trap to
avoid here.

---

## Key insight (the negative result IS the finding)

**Three independent honest-validation experiments now converge on the same
conclusion for 4h crypto classification with a TP=3%/SL=1.5% label**:

| Experiment | Asset | Model | Train AUC | Test AUC | Aggregate PF | Bootstrap p |
|------------|-------|-------|----------:|---------:|-------------:|------------:|
| Agent B (BTC) | BTC | GBM (sklearn) | 0.751 | 0.520 | 0.96 | 0.607 |
| **Agent G (ETH)** | **ETH** | **LightGBM** | **0.805** | **0.513** | **0.92** | **0.808** |
| Project history (V7, V9, V13.03) | mixed | various ML | high | — | — | unrecoverable |

**The pattern is structural, not a quirk of one model or one asset:**
- Feature space is rich enough to MEMORIZE the train period (high train AUC).
- The features are STABLE across folds (top features don't shuffle randomly).
- Yet OOS AUC collapses to ~0.51 — barely better than coin flip.
- ETH/BTC ratio — the literature's claimed silver bullet — is the single
  most important feature in this model, AND it doesn't help OOS.

**Two distinct facts emerge:**
1. The 16 features used DO contain information about the *training period's*
   conditional distribution P(win | features). The model learns it
   (AUC 0.805 in-sample). But that conditional distribution is non-stationary
   on 4h crypto — what predicted wins in 2020-2023 does not predict wins in
   2024-2025 (test AUC 0.513).
2. The single literature-prior feature most expected to add ETH-specific
   edge (`ethbtc_zscore_90d`) gets used heavily but provides no OOS lift.
   This is direct empirical refutation of the claim that ETH/BTC dynamics
   give ETH structural ML alpha at 4h.

This is **consistent with `CLAUDE.md`'s position**: "edge lives in the exit
(trailing stop), not in the entry probability." Agent A's exit-driven
trend follower achieved comparable in-sample numbers without any ML at
all; adding ML to the entry side does not move the needle.

---

## Recommendation

**Do NOT deploy this model.** Specifically:
1. Do not lower the threshold to "get more trades" — at lower thresholds
   the WR is worse.
2. Do not raise the threshold to "be more selective" — Agent B's threshold
   sweep showed the high-tail trades are too few to be statistically
   significant, and re-tuning to that threshold post hoc would be
   test-set data snooping.
3. Do not re-tune LightGBM hyperparams — the constraints in the brief are
   already at the regularization bound, and looser models will overfit
   harder, not less.
4. Do not add more features — the 16 here already include the literature's
   top recommendation (ETH/BTC ratio) and the model still doesn't generalize.
5. **Close ML classifiers as a research direction for ETH 4h LONG-only**, the
   same way Agent B closed it for BTC 4h LONG-only.

The honest path forward for ETH:
- Stick with the existing rule-based V15 ETH strategy
  (BTC-follower + breakout + multi-conf SHORT) that paper trades
  WF 8/12, PF 1.28. Its edge is modest but real.
- Or accept that ETH at 4h has structural alpha similar to BTC
  (~10-15% annual, not the 30%+ the project aims for).

---

## Files in this directory

| File | Purpose |
|---|---|
| `strategy.py` | Self-contained: `PARAMS`, `FEATURES`, `build_features()`, `create_labels()`, `signal()`, `simulate()`, `backtest()`, `load_model()` |
| `train.py` | Loads data (cutoff 2025-12-31), runs purged CV, retrains on full data, saves artifacts |
| `bootstrap_test.py` | 5000-resample bootstrap on aggregate test trades |
| `model.pkl` | Final `LGBMClassifier` trained on ALL <=2025-12-31 data |
| `scaler.pkl` | `StandardScaler` fit on ALL <=2025-12-31 features |
| `trained_params.json` | `PARAMS` with the median-of-passing-folds threshold (0.510) |
| `cv_results.csv` | Per-fold AUC/WR/PF/PnL |
| `cv_summary.json` | Headline metrics + feature stability |
| `bootstrap_result.json` | p-value, CI, PF, WR over the 983 aggregate trades |
| `wf_trades.csv` | All 983 walk-forward test trades |

## How to reproduce

```bash
cd C:/Users/pcdec/OneDrive/Documentos/MIS\ EMPRENDIMIENTOS/BOTDETRADINGAGRESIVO
C:/Python/python.exe -u experiments/agent_G/train.py            # ~60s
C:/Python/python.exe -u experiments/agent_G/bootstrap_test.py   # ~60s
```

`model.pkl` was trained with sklearn 1.6.1 + lightgbm 4.6.0. The bot's
production venv has sklearn 1.8.0 — joblib generally tolerates a minor
version mismatch on `LGBMClassifier` but verify if loading in production.

---

## Conclusion

A LightGBM classifier on 16 ETH-focused features, validated with strict
purged time-series CV, an honest sim (no overlap, no intra-bar look-ahead),
and a 5000-iteration bootstrap, **does not show a statistically significant
edge on ETH/USDT 4h LONG**:

- **2 / 11** folds pass
- Train AUC **0.805** → Test AUC **0.513** (gap 0.292 — worse than Agent B's
  0.23 on BTC)
- Aggregate 983 trades: PF **0.92**, WR **32%**, total **-72%**, max DD 83%
- Annual on WF trades: **-20.9%**
- Bootstrap one-sided p = **0.808** (not significant; worse than chance)
- The single most important feature (ETH/BTC z-score 90d), promoted by the
  literature, is heavily used in-sample but adds NO OOS edge

**Closes the hypothesis "LightGBM with ETH/BTC ratio features beats Agent B
on ETH"** with the same rigor that Agent B closed it for BTC. Both negative
results, taken together, are strong evidence that the project should stop
investing in 4h LONG-only ML classifiers and concentrate research effort on
exit logic / trend-following / regime-aware rule systems where there's at
least marginal evidence of edge (Agent A on BTC, V15 ETH rule-based).
