# Agent B — ML Classifier (BTC/USDT 4h) with Anti-Overfitting Protocol

## TL;DR — Honest verdict

**The ML classifier does NOT have a statistically significant edge on BTC/USDT 4h with these constraints.**
With the protocol's required calibration (no look-ahead, no test-set tuning):

| Metric | Value |
|---|---|
| Purged CV folds passing (>=3 trades, PF>=1.2, total>0) | **3 / 11** |
| Median PF (test) | 1.10 |
| Median WR (test) | 36.4% |
| Mean test AUC | 0.520 |
| Aggregate test trades | 452 |
| Aggregate PnL across folds | **-16.0%** |
| Aggregate WR | 33.6% |
| Aggregate PF | 0.96 |
| Bootstrap p-value (H0: mean<=0) | **0.607** |
| 95% CI on mean PnL/trade | [-0.26%, +0.21%] |

Train AUC ~0.75 vs test AUC ~0.52 is the classic overfitting fingerprint: the
model learns *something* on past data that does not generalize. The aggregate
results across 11 purged folds — across diverse market regimes 2021-06 to
2025-12 — are statistically indistinguishable from random.

**This is a valid and informative result.** It confirms the project's lesson
in `CLAUDE.md`: small-feature ML classifiers on BTC 4h with honest validation
do not produce the edge that flashy in-sample backtests suggested.

---

## What was built

A minimal, theory-driven binary classifier:

- **Target.** Binary label = 1 if a LONG entry at `close[i]` hits +3% (TP)
  before -1.5% (SL) within 12 bars (48h). Pessimistic tie-break: if a single
  bar touches BOTH TP and SL, label = 0. Uses ONLY future bars (i+1..i+12).
- **Model.** `GradientBoostingClassifier` (`sklearn`) with **aggressive
  regularization**:
  - `max_depth=3`, `n_estimators=100`, `learning_rate=0.05`,
    `min_samples_leaf=50`, `subsample=0.8`
  - StandardScaler on inputs; class-weight balancing via `sample_weight`
- **Features (11 — all theoretically justified, no data mining).**

| Feature | Rationale |
|---|---|
| `ema20_50_ratio_1d` | Daily-trend regime (BULL when >0) |
| `rsi14_4h` | 4h momentum / OB-OS |
| `rsi_slope_4h` | RSI acceleration |
| `bb_pct_4h` | Mean-reversion position |
| `atr_pct_4h` | Current volatility |
| `vol_ratio_4h` | Current volume vs 20-bar mean |
| `ret_5_4h` | 5-bar momentum |
| `dist_high20_4h` | Extension from 20-bar high |
| `dist_low20_4h` | Proximity to 20-bar low |
| `funding_zscore` | Contrarian sentiment |
| `fng_value` | Macro Fear & Greed (daily) |

All multi-timeframe inputs (`*_1d`, funding, F&G) are `.shift(1)`-ed before
reindex to 4h — no look-ahead.

- **Exit rules in `simulate()`** (used both for production AND for the test-fold
  backtest):
  - Initial SL = -1.5% from entry.
  - Once close >= entry * (1 + 0.01), trailing activates.
    Trail uses **peak of CLOSES** (not highs) — strictly conservative.
  - **Intrabar safety:** on each new bar we check exit against the stop value
    from PRIOR closes FIRST, then update peak/stop AFTER the bar closes. This
    eliminates the intra-bar look-ahead bug documented in
    `docs/revalidation/PASO0_lookahead.md`.
  - Max holding = 24 bars (96h); timeout exits at close.

- **Backtest engine in `strategy.backtest()`**: strictly sequential — once a
  trade opens we advance the cursor past its exit bar before evaluating the
  next signal. **NO overlapping trades** (the PASO0 bug is not present).

- **Commissions:** 0.05% per side → 0.1% round trip (matches project default).

---

## Validation protocol followed

| Requirement | Implementation | Result |
|---|---|---|
| CUTOFF 2025-12-31 | Hard slice on load (`df.index <= '2025-12-31'`) | 13,838 4h bars |
| Purged time-series CV | Custom `purged_ts_split()`: expanding train, 84-bar (=14 days) gap, 11 chronological splits | Done |
| Class balance | `sample_weight` inverse to class frequency | Done |
| Threshold calibration | On TRAIN ONLY; LOWEST threshold >=0.50 with train-precision >=0.55 and signal-rate >=2% | Per-fold (range 0.50-0.52) |
| Bootstrap significance | 5000 resamples on 452 aggregate test trades | p = 0.607 (not significant) |
| Feature stability check | Top-5 importance count per fold | High stability (see below) |
| Sanity check (PF>4 / WR>65% suspicious) | n/a — observed PF<1.5 in all folds | OK |
| No grid search of TP/SL on data | Label thresholds fixed a-priori | OK |
| No test-set re-tuning | Threshold frozen per fold; threshold_sweep.csv only diagnostic | OK |

---

## Per-fold results

```
Fold  1: train 2020-01-21->2021-06-11 test 2021-06-25->2021-10-30 AUC_tr=0.802 AUC_te=0.561 thr=0.50 n= 54 wr=40.74% pf=1.24 tot_pnl=+12.5% OK
Fold  2: train 2020-01-21->2021-10-16 test 2021-10-30->2022-03-08 AUC_tr=0.775 AUC_te=0.493 thr=0.51 n= 27 wr=22.22% pf=0.62 tot_pnl=-10.6% --
Fold  3: train 2020-01-21->2022-02-22 test 2022-03-08->2022-07-16 AUC_tr=0.765 AUC_te=0.441 thr=0.51 n= 18 wr=44.44% pf=1.22 tot_pnl= +3.3% OK
Fold  4: train 2020-01-21->2022-07-02 test 2022-07-16->2022-12-07 AUC_tr=0.755 AUC_te=0.546 thr=0.52 n= 22 wr=13.64% pf=0.37 tot_pnl=-14.9% --
Fold  5: train 2020-01-21->2022-11-15 test 2022-12-07->2023-05-15 AUC_tr=0.755 AUC_te=0.551 thr=0.52 n= 63 wr=39.68% pf=1.17 tot_pnl= +9.6% --
Fold  6: train 2020-01-21->2023-04-29 test 2023-05-15->2023-11-27 AUC_tr=0.754 AUC_te=0.571 thr=0.52 n= 49 wr=40.82% pf=1.10 tot_pnl= +4.6% --
Fold  7: train 2020-01-21->2023-11-09 test 2023-11-27->2024-04-17 AUC_tr=0.750 AUC_te=0.540 thr=0.52 n= 61 wr=36.07% pf=1.35 tot_pnl=+18.4% OK
Fold  8: train 2020-01-21->2024-04-03 test 2024-04-17->2024-09-04 AUC_tr=0.735 AUC_te=0.379 thr=0.52 n= 48 wr=18.75% pf=0.25 tot_pnl=-42.7% --
Fold  9: train 2020-01-21->2024-08-20 test 2024-09-04->2025-01-28 AUC_tr=0.735 AUC_te=0.534 thr=0.52 n= 42 wr=38.10% pf=1.19 tot_pnl= +6.5% --
Fold 10: train 2020-01-21->2025-01-14 test 2025-01-28->2025-07-09 AUC_tr=0.727 AUC_te=0.517 thr=0.52 n= 33 wr=36.36% pf=1.10 tot_pnl= +3.3% --
Fold 11: train 2020-01-21->2025-06-15 test 2025-07-09->2025-12-27 AUC_tr=0.726 AUC_te=0.583 thr=0.52 n= 35 wr=25.71% pf=0.86 tot_pnl= -5.8% --

=== WF Summary: 3/11 folds pass (>=3 trades, PF>=1.2, total>0)
```

Three big losing folds (4, 8) and several mild losers swamp the modest winners.
Fold 8 is particularly damning: high-trade-count, train AUC 0.735, test AUC
**0.379** — the model was actively wrong on this segment of 2024.

## Feature stability

```
ema20_50_ratio_1d          top5_in: 11/11   imp_mean=0.185 std=0.028
atr_pct_4h                 top5_in: 11/11   imp_mean=0.111 std=0.011
funding_zscore             top5_in: 11/11   imp_mean=0.132 std=0.020
dist_high20_4h             top5_in:  9/11   imp_mean=0.099 std=0.022
dist_low20_4h              top5_in:  6/11   imp_mean=0.107 std=0.029
rsi14_4h                   top5_in:  5/11   imp_mean=0.081 std=0.013
fng_value                  top5_in:  2/11   imp_mean=0.072 std=0.015
vol_ratio_4h               top5_in:  0/11   imp_mean=0.063 std=0.013
bb_pct_4h                  top5_in:  0/11   imp_mean=0.055 std=0.014
ret_5_4h                   top5_in:  0/11   imp_mean=0.054 std=0.006
rsi_slope_4h               top5_in:  0/11   imp_mean=0.043 std=0.012
```

Feature stability is **high** — the same five features dominate every fold
(daily regime, ATR, funding z-score, dist-to-high20, dist-to-low20). That's
good news: the model isn't structurally unstable. **The problem is that those
stable, theoretically-sound features simply do not separate winners from
losers reliably on 4h BTC**. AUC ~0.52 means the predicted probability and the
true outcome are nearly independent on out-of-sample data.

---

## Diagnostic (NOT used for production threshold)

After completing the honest validation above, I also ran an a-posteriori
`threshold_sweep.py` to see whether a *more selective* threshold would have
recovered an edge. This is **diagnostic only** — using these results to pick a
threshold would be look-ahead.

```
thr      n     wr     pf      mean    total
0.50   568  31.51%   0.88  -0.120%  -68.0%
0.52   436  33.49%   0.97  -0.033%  -14.3%
0.55   272  33.82%   1.04  +0.043%  +11.7%
0.58   176  37.50%   1.25  +0.221%  +38.9%
0.60   123  40.65%   1.67  +0.580%  +71.3%
0.62   105  39.05%   1.53  +0.472%  +49.6%
0.65    73  41.10%   1.42  +0.383%  +28.0%
0.70    33  51.52%   2.50  +1.100%  +36.3%
```

So the classifier carries SOME signal at the high-probability tail. But the
calibration procedure on train (precision >=0.55) never selects a threshold
that high because train precision saturates quickly; without an a-priori
reason to demand threshold >=0.58 we'd be tuning on test. **Honest reading:
the right move is not to "fix" the threshold but to accept that this model
shape doesn't carry a robust edge.** Several other agents have observed the
same on BTC 4h.

---

## Self-audit (decisions and whether they peeked at data)

| Decision | When made | Looked at test data? |
|---|---|---|
| 11-feature list | Before any model run; chosen from `CLAUDE.md` reasoning + V14/V15 documents | No |
| Model = sklearn GBM, max_depth=3, n=100, lr=0.05, leaf=50 | Constraints in the brief | No |
| Label thresholds TP=3%, SL=1.5%, 12 bars | A-priori (asymmetric R:R=2:1) | No |
| Trailing stop config (sl=1.5%, trail=1.5%, activate=+1%) | A-priori (mirrors V7's edge-on-exit principle) | No |
| Calibration target precision=0.55, signal_rate>=0.02 | A-priori before running CV | No |
| Threshold per fold | Calibrated on TRAIN ONLY | No |
| Bootstrap N=5000, alpha=0.05 | A-priori | No |
| Min train size = 2000 bars, gap=84 bars (14 days), 12 splits | A-priori (purge >= label horizon) | No |
| `threshold_sweep.csv` | AFTER CV, AFTER bootstrap | Yes — DIAGNOSTIC ONLY, results not used for production model |
| `min_threshold=0.58` (NOT applied) | Considered then rejected because it would be data-snooping | No (revoked) |

I considered changing `calibrate_threshold` to force `threshold >=0.58` after
seeing the threshold_sweep table. **I did not apply that change** because that
is precisely the kind of test-set tuning the protocol forbids. The honest
result is the one above.

---

## Files

| File | Purpose |
|---|---|
| `strategy.py` | Self-contained: `PARAMS`, `FEATURES`, `build_features()`, `create_labels()`, `signal()`, `simulate()`, `backtest()`, `load_model()` |
| `train.py` | Loads data (cutoff 2025-12-31), runs purged CV, retrains on full data, saves artifacts |
| `bootstrap_test.py` | 5000-resample bootstrap on aggregate test trades |
| `threshold_sweep.py` | DIAGNOSTIC only (post-hoc threshold sensitivity) |
| `model.pkl` | Final `GradientBoostingClassifier` trained on ALL <=2025-12-31 data |
| `scaler.pkl` | StandardScaler fit on ALL <=2025-12-31 features |
| `trained_params.json` | `PARAMS` with the median-of-passing-folds threshold |
| `cv_results.csv` | Per-fold AUC/WR/PF/PnL |
| `cv_summary.json` | Headline metrics |
| `bootstrap_result.json` | p-value, CI, PF, WR over aggregate 452 trades |
| `threshold_sweep.csv` | Diagnostic threshold sensitivity |
| `wf_trades.csv` | All 452 walk-forward test trades (entry, exit, pnl, fold) |

---

## How to reproduce

```bash
cd C:/Users/pcdec/OneDrive/Documentos/MIS\ EMPRENDIMIENTOS/BOTDETRADINGAGRESIVO
C:/Python/python.exe -u experiments/agent_B/train.py        # ~30s — fits 11 folds
C:/Python/python.exe -u experiments/agent_B/bootstrap_test.py  # ~30s — bootstrap
C:/Python/python.exe -u experiments/agent_B/threshold_sweep.py # ~30s — diagnostic only
```

`model.pkl` was trained with sklearn 1.8.0 (same as the production bot's venv),
so it should load both in Claude bash and the bot env.

---

## Conclusion

A theoretically-grounded, properly regularized ML classifier on BTC 4h, with
an honest purged walk-forward and intrabar-safe simulation, **does not show a
statistically significant edge**:

- 3/11 folds pass the bare minimum bar (PF>=1.2, total>0)
- Aggregate 452 trades: PF 0.96, WR 33.6%, total -16.0%
- Bootstrap p = 0.607 (utterly not significant)
- Train AUC 0.75 vs test AUC 0.52 = textbook overfitting in feature space

This **confirms** the project's hard-learned lesson (V7, V9, V13.03, BTC V2,
SOL V2 all failed similarly) that ML classifiers in this configuration tend
to memorize past regimes without generalizing. The recommendation: do NOT
deploy this model. The edge in this project — if there is one — appears to
live in the EXIT (trailing stop), not in the entry probability — exactly as
the V15 documents and `CLAUDE.md` already argue.
