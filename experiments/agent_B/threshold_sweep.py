"""
Threshold sweep on the test folds (per fold, all use same swept t)
to check if a MORE selective threshold improves edge.

WARNING: doing this AFTER seeing test results = look-ahead.
This is purely diagnostic — does NOT update the production threshold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import strategy  # noqa
from train import load_data, build_dataset, purged_ts_split, make_model, evaluate_fold


def main():
    df4h, df1d, fund, fng = load_data()
    X, y = build_dataset(df4h, df1d, fund, fng)

    fold_models = []
    for k, (tr_idx, te_idx) in enumerate(purged_ts_split(len(X)), 1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te = X.iloc[te_idx]
        scaler = StandardScaler().fit(X_tr.values)
        cnt = np.bincount(y_tr.values)
        w = np.where(y_tr.values == 1, 1.0/cnt[1], 1.0/cnt[0])
        w = w / w.mean()
        model = make_model()
        model.fit(scaler.transform(X_tr.values), y_tr.values, sample_weight=w)
        fold_models.append((k, model, scaler, X_te.index))

    print(f"{'thr':>5} {'n':>5} {'wr':>7} {'pf':>6} {'mean':>8} {'total':>8}")
    rows = []
    for thr in [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]:
        all_pnls = []
        for k, m, s, idx in fold_models:
            sim = evaluate_fold(df4h, X, m, s, thr, idx)
            if sim["n"] > 0:
                all_pnls.extend(sim["trades"]["pnl_pct"].tolist())
        if not all_pnls:
            print(f"{thr:>5.2f}  (no trades)")
            continue
        a = np.array(all_pnls)
        wr = (a > 0).mean()
        gw = a[a > 0].sum()
        gl = -a[a <= 0].sum()
        pf = gw / gl if gl > 0 else float("inf")
        print(f"{thr:>5.2f} {len(a):>5d} {wr*100:>6.2f}% {pf:>6.2f} {a.mean()*100:>+7.3f}% {a.sum()*100:>+7.1f}%")
        rows.append({"thr": thr, "n": len(a), "wr": wr, "pf": pf,
                     "mean": a.mean(), "total": a.sum()})
    pd.DataFrame(rows).to_csv(HERE / "threshold_sweep.csv", index=False)
    print("Saved: threshold_sweep.csv")


if __name__ == "__main__":
    main()
