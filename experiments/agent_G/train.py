"""
Agent G — Training script (LightGBM + purged time-series CV).

- Loads ETH 4h, ETH 1d, BTC 4h, BTC 1d, ETH/BTC daily ratio, BTC funding
  — ALL slices applied at <=2025-12-31 (REGLA INVIOLABLE).
- Builds the 16-feature matrix and binary labels (TP=3% before SL=1.5% in 12 bars).
- Runs PURGED time-series CV (12 folds, gap=84 bars=14 days).
- LightGBM classifier with anti-overfitting caps; threshold calibrated on TRAIN.
- Reports per-fold AUC + WR + PF + total PnL; aggregate metrics; feature
  importance stability.
- Retrains on ALL <=cutoff data and saves model.pkl + scaler.pkl + meta.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# LightGBM (literature: #1 ranked for ETH 4h)
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import strategy  # noqa: E402

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
DATA = ROOT / "data"
CUTOFF = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
GAP_BARS = 84            # 14 days at 4h (>= label horizon of 12 bars)
MIN_TRAIN_BARS = 2000    # ~ 1 year at 4h
N_SPLITS = 12

# LightGBM constrained params — within the brief's hard caps
LGB_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.03,
    max_depth=4,
    num_leaves=15,
    min_data_in_leaf=80,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_alpha=1.0,
    reg_lambda=1.0,
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


def _ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def load_data():
    """Load and CUT all sources to <=cutoff."""
    eth_4h = _ensure_utc(pd.read_parquet(DATA / "ETH_USDT_4h_full.parquet"))
    eth_1d = _ensure_utc(pd.read_parquet(DATA / "ETH_USDT_1d_history.parquet"))
    btc_4h = _ensure_utc(pd.read_parquet(DATA / "BTC_USDT_4h_full.parquet"))
    btc_1d = _ensure_utc(pd.read_parquet(DATA / "btcusdt_1d_v15.parquet"))
    ethbtc = _ensure_utc(pd.read_parquet(DATA / "ethbtc_daily_history.parquet"))
    fund = _ensure_utc(pd.read_parquet(DATA / "btc_v15_funding.parquet"))

    # ---- HARD CUTOFF (inviolable) ----
    eth_4h = eth_4h[eth_4h.index <= CUTOFF]
    eth_1d = eth_1d[eth_1d.index <= CUTOFF]
    btc_4h = btc_4h[btc_4h.index <= CUTOFF]
    btc_1d = btc_1d[btc_1d.index <= CUTOFF]
    ethbtc = ethbtc[ethbtc.index <= CUTOFF]
    fund = fund[fund.index <= CUTOFF]
    return eth_4h, eth_1d, btc_4h, btc_1d, ethbtc, fund


def build_dataset(eth_4h, eth_1d, btc_4h, btc_1d, ethbtc, fund):
    feats = strategy.build_features(eth_4h, eth_1d, btc_4h, btc_1d, ethbtc, fund)
    labels = strategy.create_labels(
        eth_4h,
        tp=strategy.PARAMS["tp_label"],
        sl=strategy.PARAMS["sl_label"],
        max_bars=strategy.PARAMS["max_bars_label"],
    )
    common = feats.dropna().index.intersection(labels.dropna().index)
    X = feats.loc[common]
    y = labels.loc[common].astype(int)
    return X, y


def make_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**LGB_PARAMS)


def purged_ts_split(n: int, n_splits: int = N_SPLITS, gap: int = GAP_BARS,
                    min_train: int = MIN_TRAIN_BARS):
    """Expanding-train purged CV. Train: [0,t); Gap: [t,t+gap); Test:[t+gap,...)."""
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
        yield np.arange(0, train_end), np.arange(test_start, test_end)


def evaluate_fold(
    df_4h: pd.DataFrame,
    features_df: pd.DataFrame,
    model,
    scaler: StandardScaler,
    threshold: float,
    test_index_labels: pd.DatetimeIndex,
) -> dict:
    """Run honest backtest over the test fold window."""
    df = df_4h.join(features_df, how="left").copy()
    if len(test_index_labels) == 0:
        return {"n": 0, "wr": 0, "pf": 0, "mean_pnl": 0, "total_pnl": 0,
                "trades": pd.DataFrame()}
    start_ts = test_index_labels[0]
    end_ts = test_index_labels[-1]
    start_pos = df.index.get_indexer([start_ts])[0]
    end_pos = df.index.get_indexer([end_ts])[0]
    if start_pos < 0 or end_pos < 0:
        return {"n": 0, "wr": 0, "pf": 0, "mean_pnl": 0, "total_pnl": 0,
                "trades": pd.DataFrame()}

    params = dict(strategy.PARAMS)
    params["threshold"] = threshold
    trades = strategy.backtest(df, params, (model, scaler),
                               start_idx=start_pos, end_idx=end_pos + 1)
    if len(trades) == 0:
        return {"n": 0, "wr": 0, "pf": 0, "mean_pnl": 0, "total_pnl": 0,
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


def calibrate_threshold(model, scaler, X_train, y_train,
                        target_precision: float = 0.55,
                        min_signal_rate: float = 0.02) -> float:
    """Lowest threshold >=0.50 with train precision >= target and signal_rate
    >= min. If unreachable, fall back to 0.55. NEVER touches test data."""
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


def main():
    print("=== Agent G — Train (LightGBM, ETH 4h) ===")
    print(f"Cutoff: {CUTOFF}")

    eth_4h, eth_1d, btc_4h, btc_1d, ethbtc, fund = load_data()
    print(f"ETH 4h bars (<= cutoff): {len(eth_4h)}  "
          f"range: {eth_4h.index[0]} -> {eth_4h.index[-1]}")
    print(f"BTC 4h bars: {len(btc_4h)}  ETH/BTC daily: {len(ethbtc)}")

    X, y = build_dataset(eth_4h, eth_1d, btc_4h, btc_1d, ethbtc, fund)
    print(f"Aligned samples: {len(X)}")
    print(f"Label distribution: {y.value_counts().to_dict()}  (mean={y.mean():.3f})")
    print(f"Features ({len(strategy.FEATURES)}): {strategy.FEATURES}")

    fold_results = []
    feature_importances = []
    for k, (tr_idx, te_idx) in enumerate(purged_ts_split(len(X)), 1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

        scaler = StandardScaler().fit(X_tr.values)
        X_tr_s = scaler.transform(X_tr.values)
        X_te_s = scaler.transform(X_te.values)

        # Class imbalance via sample_weight
        cnt = np.bincount(y_tr.values)
        if len(cnt) < 2 or cnt[1] == 0 or cnt[0] == 0:
            continue
        w = np.where(y_tr.values == 1, 1.0 / cnt[1], 1.0 / cnt[0])
        w = w / w.mean()

        model = make_model()
        model.fit(X_tr_s, y_tr.values, sample_weight=w)

        proba_tr = model.predict_proba(X_tr_s)[:, 1]
        proba_te = model.predict_proba(X_te_s)[:, 1]
        try:
            auc_tr = roc_auc_score(y_tr.values, proba_tr)
            auc_te = roc_auc_score(y_te.values, proba_te)
        except ValueError:
            auc_tr = auc_te = np.nan

        thr = calibrate_threshold(model, scaler, X_tr, y_tr)

        sim = evaluate_fold(eth_4h, X, model, scaler, thr, X_te.index)

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
            "total_pnl": sim["total_pnl"],
            "passes": passes,
        })
        feature_importances.append(model.feature_importances_)

        print(f"Fold {k:2d}: train {X_tr.index[0].date()}->{X_tr.index[-1].date()} "
              f"test {X_te.index[0].date()}->{X_te.index[-1].date()} "
              f"AUC_tr={auc_tr:.3f} AUC_te={auc_te:.3f} thr={thr:.2f} "
              f"n={sim['n']:3d} wr={sim['wr']:.2%} pf={sim['pf']:.2f} "
              f"tot_pnl={sim['total_pnl']*100:+.1f}% "
              f"{'OK' if passes else '--'}")

    fold_df = pd.DataFrame(fold_results)
    n_pass = int(fold_df["passes"].sum())
    n_total = len(fold_df)
    print(f"\n=== WF Summary: {n_pass}/{n_total} folds pass (>=3 trades, PF>=1.2, total>0)")
    if (fold_df.n_trades >= 3).any():
        print(f"  median PF (n>=3): "
              f"{fold_df[fold_df.n_trades>=3]['pf'].replace([np.inf],np.nan).median():.2f}")
        print(f"  median WR (n>=3): "
              f"{fold_df[fold_df.n_trades>=3]['wr'].median():.2%}")
    print(f"  mean AUC_tr: {fold_df['auc_tr'].mean():.3f}")
    print(f"  mean AUC_te: {fold_df['auc_te'].mean():.3f}")
    print(f"  total_pnl sum across folds: {fold_df['total_pnl'].sum()*100:+.1f}%")

    # Feature stability
    fi = np.vstack(feature_importances)
    fi_mean = fi.mean(axis=0)
    fi_std = fi.std(axis=0)
    top5_per_fold = [tuple(np.argsort(-r)[:5]) for r in fi]
    top5_count = np.zeros(len(strategy.FEATURES), dtype=int)
    for t in top5_per_fold:
        for j in t:
            top5_count[j] += 1
    print("\nFeature stability (count in top-5 across folds):")
    order = np.argsort(-top5_count)
    feature_stats = []
    for j in order:
        name = strategy.FEATURES[j]
        line = (f"  {name:28s}  top5_in: {top5_count[j]:2d}/{n_total}  "
                f"imp_mean={fi_mean[j]:.1f} std={fi_std[j]:.1f}")
        print(line)
        feature_stats.append({
            "feature": name,
            "top5_in": int(top5_count[j]),
            "imp_mean": float(fi_mean[j]),
            "imp_std": float(fi_std[j]),
        })

    # === Retrain on ALL <= cutoff data ===
    print("\nRetraining on ALL <=2025-12-31 data...")
    scaler_full = StandardScaler().fit(X.values)
    cnt = np.bincount(y.values)
    w = np.where(y.values == 1, 1.0 / cnt[1], 1.0 / cnt[0])
    w = w / w.mean()
    model_full = make_model()
    model_full.fit(scaler_full.transform(X.values), y.values, sample_weight=w)

    if n_pass > 0:
        thr_final = float(fold_df.loc[fold_df.passes, "threshold"].median())
    else:
        thr_final = float(fold_df["threshold"].median())
    print(f"Final threshold (median of fold-calibrated): {thr_final:.3f}")

    joblib.dump(model_full, HERE / "model.pkl")
    joblib.dump(scaler_full, HERE / "scaler.pkl")

    params_out = dict(strategy.PARAMS)
    params_out["threshold"] = thr_final
    with open(HERE / "trained_params.json", "w") as f:
        json.dump(
            {k: (str(v) if isinstance(v, pd.Timestamp) else v)
             for k, v in params_out.items()},
            f, indent=2,
        )

    fold_df.to_csv(HERE / "cv_results.csv", index=False)
    print("\nSaved: model.pkl, scaler.pkl, trained_params.json, cv_results.csv")

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
        "mean_auc_train": float(fold_df['auc_tr'].mean()),
        "mean_auc_test": float(fold_df['auc_te'].mean()),
        "total_pnl_across_folds_pct": float(fold_df['total_pnl'].sum() * 100),
        "final_threshold": thr_final,
        "feature_stability_ordered": feature_stats,
    }
    with open(HERE / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("Summary saved to cv_summary.json")
    return summary, fold_df


if __name__ == "__main__":
    main()
