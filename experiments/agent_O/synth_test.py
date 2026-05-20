"""
Agent O — Synthetic SOL test (block-bootstrap from real SOL returns).

Builds 20 synthetic SOL price series by block-bootstrapping real SOL log-returns
in 30-bar blocks (preserves short-range autocorrelation). For each synthetic
universe we run the strategy and count how many synth series produce positive
annual return. If the real edge is solid, we should see real SOL near the
median or higher of the synth distribution.

Notes:
- We synthesize ONLY SOL price; BTC/ETH context comes from the REAL data
  (the bot would not see synth crypto market). This is the conservative test:
  could the SOL classifier alone produce edge even when SOL price is shuffled
  but BTC/ETH context is real?
- This is more demanding than B's/G's bootstrap because it forces SOL features
  to lose their dependence with BTC/ETH context — if the model has real
  cross-asset edge, the synth distribution should be much lower than real SOL.
- If real SOL annual return is in the top quartile of synth -> evidence of edge.
  If it's around the median -> the result is in the null distribution.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import strategy  # noqa: E402

warnings.filterwarnings("ignore")

DATA = ROOT / "data"
CUTOFF = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
N_SYNTH = 20
BLOCK_SIZE = 30
SEED = 42


def _ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def load_all():
    sol_4h = _ensure_utc(pd.read_parquet(DATA / "SOL_USDT_4h_full.parquet"))
    btc_4h = _ensure_utc(pd.read_parquet(DATA / "BTC_USDT_4h_full.parquet"))
    eth_4h = _ensure_utc(pd.read_parquet(DATA / "ETH_USDT_4h_full.parquet"))
    btc_1d = _ensure_utc(pd.read_parquet(DATA / "btcusdt_1d_v15.parquet"))
    eth_1d = _ensure_utc(pd.read_parquet(DATA / "ETH_USDT_1d_history.parquet"))
    ethbtc = _ensure_utc(pd.read_parquet(DATA / "ethbtc_daily_history.parquet"))
    fund = _ensure_utc(pd.read_parquet(DATA / "btc_v15_funding.parquet"))
    sol_4h = sol_4h[sol_4h.index <= CUTOFF]
    btc_4h = btc_4h[btc_4h.index <= CUTOFF]
    eth_4h = eth_4h[eth_4h.index <= CUTOFF]
    btc_1d = btc_1d[btc_1d.index <= CUTOFF]
    eth_1d = eth_1d[eth_1d.index <= CUTOFF]
    ethbtc = ethbtc[ethbtc.index <= CUTOFF]
    fund = fund[fund.index <= CUTOFF]
    return sol_4h, btc_4h, eth_4h, btc_1d, eth_1d, ethbtc, fund


def synth_sol(sol_4h: pd.DataFrame, rng: np.random.Generator,
              block_size: int = BLOCK_SIZE) -> pd.DataFrame:
    """Block-bootstrap SOL close to make a synthetic price series with the
    same index and same length. Keeps OHLC structure roughly intact by scaling
    open/high/low by the realized 4h return."""
    c = sol_4h["close"].values
    h = sol_4h["high"].values
    l = sol_4h["low"].values
    o = sol_4h["open"].values
    v = sol_4h["volume"].values
    n = len(c)
    log_ret = np.diff(np.log(c))

    # Per-bar relative offsets so we can rebuild OHLC from synthetic closes:
    # ratio of (open|high|low) to close at each bar.
    r_o = o / c
    r_h = h / c
    r_l = l / c

    # Build synthetic log-return sequence by sampling blocks
    n_blocks = (n - 1) // block_size + 1
    blocks = []
    for _ in range(n_blocks):
        start = rng.integers(0, max(1, len(log_ret) - block_size))
        blocks.append(log_ret[start:start + block_size])
    synth_lr = np.concatenate(blocks)[:n - 1]

    # Rebuild prices starting from real first close
    synth_c = np.empty(n)
    synth_c[0] = c[0]
    synth_c[1:] = c[0] * np.exp(np.cumsum(synth_lr))

    synth = pd.DataFrame({
        "open": synth_c * r_o,
        "high": synth_c * r_h,
        "low": synth_c * r_l,
        "close": synth_c,
        "volume": v,  # keep original volume for realism
    }, index=sol_4h.index)
    return synth


def simulate_run(
    df_sol: pd.DataFrame,
    df_btc_4h, df_eth_4h, df_btc_1d, df_eth_1d, df_ethbtc, df_fund,
    model, scaler, threshold: float,
):
    feats = strategy.build_features(
        df_sol, df_btc_4h, df_eth_4h, df_btc_1d, df_eth_1d, df_ethbtc, df_fund
    )
    df = df_sol.join(feats, how="left")
    params = dict(strategy.PARAMS)
    params["threshold"] = threshold
    trades = strategy.backtest(df, params, (model, scaler))
    if len(trades) == 0:
        return {"n_trades": 0, "total_pnl_pct": 0.0, "wr": 0.0,
                "pf": 0.0, "annual_pct": 0.0}
    pnl = trades["pnl_pct"].values
    n = len(pnl)
    wins = (pnl > 0).sum()
    losses = (pnl <= 0).sum()
    gw = pnl[pnl > 0].sum() if wins > 0 else 0.0
    gl = -pnl[pnl <= 0].sum() if losses > 0 else 0.0
    span_years = (df.index.max() - df.index.min()).days / 365.25
    annual = pnl.sum() / span_years if span_years > 0 else 0.0
    return {
        "n_trades": int(n),
        "total_pnl_pct": float(pnl.sum() * 100),
        "wr": float(wins / n),
        "pf": float(gw / gl) if gl > 0 else float("inf"),
        "annual_pct": float(annual * 100),
    }


def main():
    sol_4h, btc_4h, eth_4h, btc_1d, eth_1d, ethbtc, fund = load_all()
    model = joblib.load(HERE / "model.pkl")
    scaler = joblib.load(HERE / "scaler.pkl")
    with open(HERE / "trained_params.json") as f:
        params = json.load(f)
    threshold = float(params["threshold"])
    print(f"Synthetic SOL test (block-bootstrap), threshold={threshold:.3f}, "
          f"block={BLOCK_SIZE}, n_synth={N_SYNTH}")

    # Real result
    real = simulate_run(sol_4h, btc_4h, eth_4h, btc_1d, eth_1d, ethbtc, fund,
                        model, scaler, threshold)
    print(f"\nREAL SOL: n={real['n_trades']}  total={real['total_pnl_pct']:+.1f}%  "
          f"wr={real['wr']:.2%}  pf={real['pf']:.2f}  annual={real['annual_pct']:+.1f}%")

    rng = np.random.default_rng(SEED)
    rows = [{"run": "real", **real}]
    print("\nSynth runs:")
    for i in range(N_SYNTH):
        synth_sol_df = synth_sol(sol_4h, rng)
        r = simulate_run(synth_sol_df, btc_4h, eth_4h, btc_1d, eth_1d, ethbtc,
                         fund, model, scaler, threshold)
        rows.append({"run": f"synth_{i:02d}", **r})
        print(f"  synth_{i:02d}: n={r['n_trades']:3d}  "
              f"total={r['total_pnl_pct']:+7.1f}%  "
              f"wr={r['wr']:.2%}  pf={r['pf']:.2f}  "
              f"annual={r['annual_pct']:+.1f}%")

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "synth_results.csv", index=False)

    synth_df = df[df.run != "real"]
    n_positive = int((synth_df["annual_pct"] > 0).sum())
    med_annual = float(synth_df["annual_pct"].median())
    real_annual = real["annual_pct"]
    # Where does real fall in the synth distribution?
    pct_below_real = float((synth_df["annual_pct"] < real_annual).mean())

    print("\n=== Synthetic Summary ===")
    print(f"Synth median annual: {med_annual:+.1f}%")
    print(f"Synth positive: {n_positive}/{N_SYNTH}")
    print(f"Real SOL annual: {real_annual:+.1f}%")
    print(f"Real percentile among synths: {pct_below_real*100:.0f}% "
          f"(higher = better; >75% suggests edge over null)")

    summary = {
        "real_annual_pct": real_annual,
        "real_n_trades": real["n_trades"],
        "synth_median_annual_pct": med_annual,
        "synth_n_positive": n_positive,
        "synth_total": N_SYNTH,
        "real_percentile_in_synth": pct_below_real * 100,
        "block_size": BLOCK_SIZE,
        "seed": SEED,
    }
    with open(HERE / "synth_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved synth_results.csv and synth_summary.json")


if __name__ == "__main__":
    main()
