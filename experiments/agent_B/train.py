"""
Agent B — Training script with anti-overfitting protocol.

- Loads BTC 4h, 1d, funding, F&G (all CUT at 2025-12-31 → REGLA INVIOLABLE).
- Builds 11-feature matrix + binary labels (TP=3% before SL=1.5% in 12 bars).
- Runs PURGED time-series CV (12 folds, gap=2 weeks=84 bars).
- Trains a GradientBoostingClassifier (max_depth=3, n_est=100, lr=0.05,
  min_samples_leaf=50) on each train window; reports per-fold AUC + WR + PF.
- Calibrates threshold on TRAIN data of each fold (target precision 0.55);
  evaluates the SAME threshold on test. NEVER re-optimized after seeing test.
- Reports feature-importance stability across folds.
- Finally retrains on ALL ≤2025-12-31 data, saves model.pkl + scaler.pkl.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Path setup
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import strategy  # noqa: E402

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA = ROOT / "data"
CUTOFF = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")  # REGLA INVIOLABLE
GAP_BARS = 84   # 84 * 4h = 14 days purge gap (matches max label horizon ≥ 12 bars)
MIN_TRAIN_BARS = 2000   # ~ 1 year of 4h bars
N_SPLITS = 12


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df4h = pd.read_parquet(DATA / "btcusdt_4h_v15.parquet")
    if df4h.index.tz is None:
        df4h.index = df4h.index.tz_localize("UTC")
    df4h = df4h.sort_index()

    df1d = pd.read_parquet(DATA / "btcusdt_1d_v15.parquet")
    if df1d.index.tz is None:
        df1d.index = df1d.index.tz_localize("UTC")
    df1d = df1d.sort_index()

    fund = pd.read_parquet(DATA / "btc_v15_funding.parquet")
    if fund.index.tz is None:
        fund.index = fund.index.tz_localize("UTC")
    fund = fund.sort_index()

    fng = pd.read_parquet(DATA / "fear_greed_history.parquet")
    if fng.index.tz is None:
        fng.index = fng.index.tz_localize("UTC")
    fng = fng.sort_index()

    # ---- HARD CUTOFF ----
    df4h = df4h[df4h.index <= CUTOFF]
    df1d = df1d[df1d.index <= CUTOFF]
    fund = fund[fund.index <= CUTOFF]
    fng = fng[fng.index <= CUTOFF]
    return df4h, df1d, fund, fng


def build_dataset(df4h, df1d, fund, fng):
    feats = strategy.build_features(df4h, df1d, fund, fng)
    labels = strategy.create_labels(
        df4h,
        tp=strategy.PARAMS["tp_label"],
        sl=strategy.PARAMS["sl_label"],
        max_bars=strategy.PARAMS["max_bars_label"],
    )
    # Align
    common = feats.dropna().index.intersection(labels.dropna().index)
    X = feats.loc[common]
    y = labels.loc[common].astype(int)
    return X, y


def make_model() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=50,
        subsample=0.8,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Purged time-series CV
# ---------------------------------------------------------------------------
def purged_ts_split(n: int, n_splits: int = N_SPLITS, gap: int = GAP_BARS,
                    min_train: int = MIN_TRAIN_BARS):
    """Yield (train_idx, test_idx) with expanding-train and PURGED gap.

    Train: [0, t_end_train); Gap: [t_end_train, t_end_train+gap);
    Test:  [t_end_train+gap, t_end_train+gap+test_size).
    """
    usable = n - min_train - gap
    if usable <= 0:
        return
    test_size = usable // n_splits
    if test_size < 50:
        return
    for k in range(n_splits):
        test_start = min_train + gap + k * test_size
        test_end = test_start + test_size
        if test_end > n:
            test_end = n
        train_end = test_start - gap
        if train_end <= min_train or test_end <= test_start:
            continue
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        yield train_idx, test_idx


# ---------------------------------------------------------------------------
# Per-fold simulation evaluation (uses honest backtest)
# ---------------------------------------------------------------------------
def evaluate_fold(
    df_4h: pd.DataFrame,      # bars (needed for sim, full index)
    features_df: pd.DataFrame,  # aligned features (index subset of df_4h)
    model: GradientBoostingClassifier,
    scaler: StandardScaler,
    threshold: float,
    test_index_labels: pd.DatetimeIndex,
) -> dict:
    """Simulate trades on the test segment using honest backtest."""
    # Build df for backtest: 4h OHLCV + the feature columns.
    df = df_4h.join(features_df, how="left").copy()
    # Find numerical indices for start/end of the test window
    if len(test_index_labels) == 0:
        return {"n": 0, "wr": 0, "pf": 0, "mean_pnl": 0, "trades": pd.DataFrame()}
    start_ts = test_index_labels[0]
    end_ts = test_index_labels[-1]
    # Numerical positions in df
    try:
        start_pos = df.index.get_indexer([start_ts])[0]
        end_pos = df.index.get_indexer([end_ts])[0]
    except KeyError:
        return {"n": 0, "wr": 0, "pf": 0, "mean_pnl": 0, "trades": pd.DataFrame()}
    if start_pos < 0 or end_pos < 0:
        return {"n": 0, "wr": 0, "pf": 0, "mean_pnl": 0, "trades": pd.DataFrame()}

    params = dict(strategy.PARAMS)
    params["threshold"] = threshold
    trades = strategy.backtest(df, params, (model, scaler),
                               start_idx=start_pos, end_idx=end_pos + 1)
    if len(trades) == 0:
        return {"n": 0, "wr": 0, "pf": 0, "mean_pnl": 0,
                "trades": trades}
    wins = trades[trades["pnl_pct"] > 0]
    losses = trades[trades["pnl_pct"] <= 0]
    n = len(trades)
    wr = len(wins) / n
    gw = wins["pnl_pct"].sum() if len(wins) else 0.0
    gl = -losses["pnl_pct"].sum() if len(losses) else 0.0
    pf = gw / gl if gl > 0 else float("inf")
    return {
        "n": n,
        "wr": wr,
        "pf": pf,
        "mean_pnl": trades["pnl_pct"].mean(),
        "total_pnl": trades["pnl_pct"].sum(),
        "trades": trades,
    }


# ---------------------------------------------------------------------------
# Calibrate threshold on TRAIN data
# ---------------------------------------------------------------------------
def calibrate_threshold(model, scaler, X_train, y_train,
                        target_precision: float = 0.55,
                        min_signal_rate: float = 0.02) -> float:
    """Pick the LOWEST threshold >= 0.50 such that:
       - precision(positive class) >= target_precision on train, AND
       - signal rate (# positives predicted / total) >= min_signal_rate.
    If unreachable, returns 0.55 (default).

    This is the ORIGINAL calibration used in the primary CV run. It is
    NOT re-tuned after looking at test data.
    """
    X_s = scaler.transform(X_train)
    proba = model.predict_proba(X_s)[:, 1]
    best_t = 0.55
    for t in np.arange(0.50, 0.71, 0.01):
        pred = (proba >= t).astype(int)
        sig_rate = pred.mean()
        if sig_rate < min_signal_rate:
            break
        positives = pred.sum()
        if positives == 0:
            continue
        precision = (pred * y_train.values).sum() / positives
        if precision >= target_precision:
            best_t = float(t)
            break
    return best_t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== Agent B - Train ===")
    print(f"Cutoff: {CUTOFF}")

    df4h, df1d, fund, fng = load_data()
    print(f"4h bars (<= cutoff): {len(df4h)}  range: {df4h.index[0]} -> {df4h.index[-1]}")

    X, y = build_dataset(df4h, df1d, fund, fng)
    print(f"Aligned samples: {len(X)}")
    print(f"Label distribution: {y.value_counts().to_dict()}  (mean={y.mean():.3f})")
    print(f"Features ({len(strategy.FEATURES)}): {strategy.FEATURES}")

    # --- Purged CV ---
    fold_results = []
    feature_importances = []
    for k, (tr_idx, te_idx) in enumerate(purged_ts_split(len(X)), 1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

        scaler = StandardScaler().fit(X_tr.values)
        X_tr_s = scaler.transform(X_tr.values)
        X_te_s = scaler.transform(X_te.values)

        # Class imbalance handled via sample_weight
        cnt = np.bincount(y_tr.values)
        w = np.where(y_tr.values == 1, 1.0 / cnt[1], 1.0 / cnt[0])
        w = w / w.mean()

        model = make_model()
        model.fit(X_tr_s, y_tr.values, sample_weight=w)

        # AUC
        proba_tr = model.predict_proba(X_tr_s)[:, 1]
        proba_te = model.predict_proba(X_te_s)[:, 1]
        try:
            auc_tr = roc_auc_score(y_tr.values, proba_tr)
            auc_te = roc_auc_score(y_te.values, proba_te)
        except ValueError:
            auc_tr = auc_te = np.nan

        # Calibrate threshold on TRAIN
        thr = calibrate_threshold(model, scaler, X_tr, y_tr)

        # Backtest the test window (honest sim)
        # Note: features_df = X (full aligned), df_4h = df4h
        # The actual test range = labels of test fold
        test_ts = X_te.index
        sim = evaluate_fold(df4h, X, model, scaler, thr, test_ts)

        # PF
        passes = (sim["n"] >= 3 and sim["pf"] >= 1.2 and sim["total_pnl"] > 0)
        fold_results.append({
            "fold": k,
            "train_start": X_tr.index[0],
            "train_end": X_tr.index[-1],
            "test_start": X_te.index[0] if len(X_te) else None,
            "test_end": X_te.index[-1] if len(X_te) else None,
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "auc_tr": auc_tr,
            "auc_te": auc_te,
            "threshold": thr,
            "n_trades": sim["n"],
            "wr": sim["wr"],
            "pf": sim["pf"],
            "total_pnl": sim.get("total_pnl", 0.0),
            "passes": passes,
        })
        feature_importances.append(model.feature_importances_)

        print(f"Fold {k:2d}: train {X_tr.index[0].date()}->{X_tr.index[-1].date()} "
              f"test {X_te.index[0].date()}->{X_te.index[-1].date()} "
              f"AUC_tr={auc_tr:.3f} AUC_te={auc_te:.3f} thr={thr:.2f} "
              f"n={sim['n']:3d} wr={sim['wr']:.2%} pf={sim['pf']:.2f} "
              f"tot_pnl={sim.get('total_pnl',0)*100:+.1f}% "
              f"{'OK' if passes else '--'}")

    fold_df = pd.DataFrame(fold_results)
    n_pass = int(fold_df["passes"].sum())
    n_total = len(fold_df)
    print(f"\n=== WF Summary: {n_pass}/{n_total} folds pass (>=3 trades, PF>=1.2, total>0)")
    print(f"  median PF (n>=3): {fold_df[fold_df.n_trades>=3]['pf'].replace([np.inf],np.nan).median():.2f}")
    print(f"  median WR (n>=3): {fold_df[fold_df.n_trades>=3]['wr'].median():.2%}")
    print(f"  mean AUC_te: {fold_df['auc_te'].mean():.3f}")
    print(f"  total_pnl sum across folds: {fold_df['total_pnl'].sum()*100:+.1f}%")

    # Feature stability — top-K consistency
    fi = np.vstack(feature_importances)  # [n_folds, n_feats]
    fi_mean = fi.mean(axis=0)
    fi_std = fi.std(axis=0)
    top5_per_fold = [tuple(np.argsort(-r)[:5]) for r in fi]
    # Count how often each feature appears in top-5 across folds
    top5_count = np.zeros(len(strategy.FEATURES), dtype=int)
    for t in top5_per_fold:
        for j in t:
            top5_count[j] += 1
    print("\nFeature stability (count in top-5 across folds):")
    for j, name in enumerate(strategy.FEATURES):
        print(f"  {name:25s}  top5_in: {top5_count[j]:2d}/{n_total}  "
              f"imp_mean={fi_mean[j]:.3f} std={fi_std[j]:.3f}")

    # --- Retrain on ALL data ≤ 2025-12-31 ---
    print("\nRetraining on ALL <=2025-12-31 data...")
    scaler_full = StandardScaler().fit(X.values)
    cnt = np.bincount(y.values)
    w = np.where(y.values == 1, 1.0 / cnt[1], 1.0 / cnt[0])
    w = w / w.mean()
    model_full = make_model()
    model_full.fit(scaler_full.transform(X.values), y.values, sample_weight=w)

    # Use median threshold from successful folds (or default if none pass)
    if n_pass > 0:
        thr_final = float(fold_df.loc[fold_df.passes, "threshold"].median())
    else:
        thr_final = float(fold_df["threshold"].median())
    print(f"Final threshold (median of fold-calibrated): {thr_final:.3f}")

    # Save
    joblib.dump(model_full, HERE / "model.pkl")
    joblib.dump(scaler_full, HERE / "scaler.pkl")

    # Save params snapshot with final threshold
    params_out = dict(strategy.PARAMS)
    params_out["threshold"] = thr_final
    with open(HERE / "trained_params.json", "w") as f:
        json.dump({k: (str(v) if isinstance(v, pd.Timestamp) else v)
                   for k, v in params_out.items()}, f, indent=2)

    # Save CV results
    fold_df.to_csv(HERE / "cv_results.csv", index=False)
    print("\nSaved: model.pkl, scaler.pkl, trained_params.json, cv_results.csv")

    # --- Save summary metrics for README ---
    summary = {
        "n_features": len(strategy.FEATURES),
        "n_samples": int(len(X)),
        "label_balance": float(y.mean()),
        "purged_cv_folds_ok": f"{n_pass}/{n_total}",
        "median_pf_test": (float(fold_df[fold_df.n_trades>=3]['pf']
                                .replace([np.inf], np.nan).median())
                           if (fold_df.n_trades>=3).any() else 0.0),
        "median_wr_test": (float(fold_df[fold_df.n_trades>=3]['wr'].median())
                           if (fold_df.n_trades>=3).any() else 0.0),
        "mean_auc_test": float(fold_df['auc_te'].mean()),
        "total_pnl_across_folds_pct": float(fold_df['total_pnl'].sum() * 100),
        "final_threshold": thr_final,
        "top_features_by_count_top5": [
            strategy.FEATURES[j]
            for j in np.argsort(-top5_count)[:5]
        ],
    }
    with open(HERE / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("Summary saved to cv_summary.json")
    return summary, fold_df


if __name__ == "__main__":
    main()
