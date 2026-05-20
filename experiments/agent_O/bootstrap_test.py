"""
Agent O — Bootstrap significance test on aggregate WF trades.

H0: mean PnL/trade <= 0 (i.e., no edge).
We resample with replacement N times and compute the proportion of resamples
with mean <= 0 -> one-sided p-value.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
N_BOOT = 5000
SEED = 42


def main():
    trades_path = HERE / "wf_trades.csv"
    if not trades_path.exists():
        raise FileNotFoundError(
            f"{trades_path} not found. Run train.py first."
        )
    trades = pd.read_csv(trades_path)
    if len(trades) == 0:
        print("No trades — cannot bootstrap.")
        return

    pnl = trades["pnl_pct"].values
    n = len(pnl)
    mean_obs = pnl.mean()
    total_pnl = pnl.sum()
    wins = (pnl > 0).sum()
    losses = (pnl <= 0).sum()
    wr = wins / n
    gw = pnl[pnl > 0].sum() if wins > 0 else 0.0
    gl = -pnl[pnl <= 0].sum() if losses > 0 else 0.0
    pf = gw / gl if gl > 0 else float("inf")

    # Max DD on cumulative equity (no compounding to keep linear with PnL%)
    equity = pnl.cumsum()
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = float(dd.min())

    rng = np.random.default_rng(SEED)
    boot_means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sample = rng.choice(pnl, size=n, replace=True)
        boot_means[i] = sample.mean()

    # One-sided p-value: P(boot_mean <= 0 | observed)
    # If true mean is 0, what fraction of resamples land >= mean_obs?
    # Standard formulation: p = P(boot_mean - mean_obs >= mean_obs)
    # Equivalent shifted-bootstrap:
    centered = boot_means - mean_obs
    p_one_sided = float((centered >= mean_obs).mean())
    # Two-sided
    p_two_sided = float((np.abs(centered) >= abs(mean_obs)).mean())
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    print(f"Total WF trades: {n}")
    print(f"Mean PnL/trade: {mean_obs*100:+.3f}%  (total: {total_pnl*100:+.1f}%)")
    print(f"WR: {wr:.2%}  PF: {pf:.2f}  MaxDD on cum equity: {max_dd*100:.1f}%")
    print(f"Bootstrap (N={N_BOOT}) one-sided p (H0: mean<=0): {p_one_sided:.4f}")
    print(f"Bootstrap two-sided p: {p_two_sided:.4f}")
    print(f"95% CI on mean PnL: [{ci_lo*100:+.3f}%, {ci_hi*100:+.3f}%]")

    # Annualized return: weeks_count = total bars / (24/4 * 7) — approx
    # Simpler: estimate avg trades per year and annual mean*N_trades_year
    # Use linear PnL sum / span_years
    if "entry_ts" in trades.columns:
        ts = pd.to_datetime(trades["entry_ts"], utc=True)
        span_years = (ts.max() - ts.min()).days / 365.25
        annual_ret = total_pnl / span_years if span_years > 0 else 0.0
    else:
        annual_ret = 0.0

    result = {
        "n_trades": int(n),
        "mean_pnl_pct": float(mean_obs * 100),
        "total_pnl_pct": float(total_pnl * 100),
        "wr": float(wr),
        "pf": float(pf),
        "max_dd_cum_pct": float(max_dd * 100),
        "annual_return_pct": float(annual_ret * 100),
        "bootstrap_p_one_sided": p_one_sided,
        "bootstrap_p_two_sided": p_two_sided,
        "ci_95_mean_pnl_pct": [ci_lo * 100, ci_hi * 100],
        "n_boot": N_BOOT,
    }
    with open(HERE / "bootstrap_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved bootstrap_result.json  annual_return_estimate={annual_ret*100:+.1f}%")


if __name__ == "__main__":
    main()
