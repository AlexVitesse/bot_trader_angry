"""
Bootstrap significance test on aggregated test-fold trades.
H0: mean_pnl_per_trade = 0 (no edge).
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import strategy  # noqa
from train import load_data, build_dataset, purged_ts_split, make_model, calibrate_threshold, evaluate_fold

CUTOFF = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
N_BOOT = 5000
np.random.seed(42)


def main():
    df4h, df1d, fund, fng = load_data()
    X, y = build_dataset(df4h, df1d, fund, fng)

    all_trades = []
    for k, (tr_idx, te_idx) in enumerate(purged_ts_split(len(X)), 1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te = X.iloc[te_idx]
        scaler = StandardScaler().fit(X_tr.values)
        cnt = np.bincount(y_tr.values)
        w = np.where(y_tr.values == 1, 1.0/cnt[1], 1.0/cnt[0])
        w = w / w.mean()
        model = make_model()
        model.fit(scaler.transform(X_tr.values), y_tr.values, sample_weight=w)
        thr = calibrate_threshold(model, scaler, X_tr, y_tr)
        sim = evaluate_fold(df4h, X, model, scaler, thr, X_te.index)
        if sim["n"] > 0:
            trs = sim["trades"].copy()
            trs["fold"] = k
            all_trades.append(trs)
    all_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if len(all_trades) == 0:
        print("No trades generated.")
        return
    pnls = all_trades["pnl_pct"].values
    n = len(pnls)
    obs_mean = pnls.mean()
    obs_sum = pnls.sum()
    wr = (pnls > 0).mean()
    gw = pnls[pnls > 0].sum()
    gl = -pnls[pnls <= 0].sum()
    pf = gw / gl if gl > 0 else float("inf")
    print(f"Total test trades across folds: {n}")
    print(f"Observed mean PnL: {obs_mean*100:+.3f}%/trade  sum: {obs_sum*100:+.1f}%  WR: {wr:.2%}  PF: {pf:.2f}")

    # Bootstrap: sample n trades with replacement N_BOOT times, count how many have mean<=0
    rng = np.random.default_rng(42)
    means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sample = rng.choice(pnls, size=n, replace=True)
        means[i] = sample.mean()
    # H0 mean = 0 ; we shift sample distribution to be centered at 0
    centered = pnls - obs_mean
    null_means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sample = rng.choice(centered, size=n, replace=True)
        null_means[i] = sample.mean()
    # one-sided p-value: P(null_mean >= obs_mean)
    p_one_sided = (null_means >= obs_mean).mean()
    p_two_sided = (np.abs(null_means) >= abs(obs_mean)).mean()
    print(f"Bootstrap (N={N_BOOT}) one-sided p-value (H0: mean<=0): {p_one_sided:.4f}")
    print(f"Bootstrap two-sided p-value: {p_two_sided:.4f}")
    # 95% CI on observed mean (from bootstrap sample distribution)
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    print(f"95% CI on mean PnL: [{ci_low*100:+.3f}%, {ci_high*100:+.3f}%]")

    # Save trade-level CSV
    all_trades.to_csv(HERE / "wf_trades.csv", index=False)
    import json
    with open(HERE / "bootstrap_result.json", "w") as f:
        json.dump({
            "n_trades": int(n),
            "observed_mean_pct": float(obs_mean * 100),
            "observed_sum_pct": float(obs_sum * 100),
            "wr": float(wr),
            "pf": float(pf) if pf != float("inf") else None,
            "p_one_sided": float(p_one_sided),
            "p_two_sided": float(p_two_sided),
            "ci95_low_pct": float(ci_low * 100),
            "ci95_high_pct": float(ci_high * 100),
        }, f, indent=2)
    print("Saved: wf_trades.csv, bootstrap_result.json")


if __name__ == "__main__":
    main()
