"""
Agent G — ML Classifier for ETH/USDT 4h (LightGBM, anti-overfitting protocol).

Goal: build an ML classifier that, unlike Agent B on BTC, captures a real edge
on ETH/USDT 4h. Hypothesis: ETH has structural features (notably ETH/BTC ratio
dynamics and stronger beta to BTC trend) that may be more learnable than BTC's
own price action.

Strategy:
- LightGBM classifier on a SMALL (<=20), theoretically-justified feature set
  centred on ETH-specific drivers (ETH/BTC ratio, BTC daily regime, funding,
  4h price action).
- Target: binary label = 1 if entering LONG at close[i] hits +3% (TP) before
  -1.5% (SL) within 12 bars (48h). Pessimistic tie-break (both hit -> 0).
- Entry: ML prob >= threshold AND not in strong BEAR daily regime.
- Exit: trailing stop, honest (no intra-bar look-ahead). Mirror of Agent B's
  simulator for direct comparability.

Anti-overfitting constraints:
- LightGBM with hard caps: max_depth=4, num_leaves=15, n_estimators=200,
  min_data_in_leaf=80, reg_alpha=1.0, reg_lambda=1.0, learning_rate=0.03.
- 16 features (<=20), each with explicit trader rationale.
- Class balance via sample_weight.
- Threshold calibrated on TRAIN of each fold, frozen for test.
- Cutoff 2025-12-31 enforced at every load.

Files in this directory:
- strategy.py     (this file)
- train.py        (purged CV + retrain on all data <= cutoff)
- bootstrap_test.py
- model.pkl, scaler.pkl, trained_params.json, cv_*.json, wf_trades.csv
"""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Frozen parameters (the threshold is set by train.py to the median of
# in-sample fold-calibrated thresholds).
# ---------------------------------------------------------------------------
PARAMS = {
    # Label horizon (forward) used at training time
    "tp_label": 0.03,
    "sl_label": 0.015,
    "max_bars_label": 12,

    # Live simulation rules (mirror Agent B for direct comparability)
    "sl_init": 0.015,
    "trail_dist": 0.015,
    "trail_activate": 0.01,
    "max_bars_sim": 24,
    "commission": 0.0005,

    # Inference threshold (overwritten by train.py with the calibrated value)
    "threshold": 0.55,

    # Regime filter (defensive: avoid LONG in deep BEAR daily regime)
    "use_regime_filter": True,
}

# 16 ETH-focused features (<=20). All justified by trader logic.
FEATURES = [
    # --- ETH 4h price action ---
    "rsi14_4h",          # 4h momentum / OB-OS
    "rsi_slope_4h",      # RSI acceleration
    "bb_pct_4h",         # mean-reversion position inside Bollinger
    "atr_pct_4h",        # current volatility
    "vol_ratio_4h",      # current volume vs 20-bar mean
    "ret_5_4h",          # 5-bar momentum (20h)
    "dist_high20_4h",    # extension from 20-bar high
    "dist_low20_4h",     # proximity to 20-bar low

    # --- ETH daily regime (causal, shift(1)) ---
    "ema20_50_ratio_1d_eth",     # ETH daily trend
    "close_above_ema200_1d_eth", # ETH macro structure

    # --- BTC context (the "exogenous driver" of ETH) ---
    "ema20_50_ratio_1d_btc",     # BTC daily trend (literature: ETH follows BTC)
    "btc_ret_5_4h",              # BTC 5-bar return (lead/lag with ETH)
    "btc_vol_ratio_4h",          # BTC volume thrust

    # --- ETH/BTC ratio dynamics (literature: signal #1 for ETH) ---
    "ethbtc_slope_30d",          # 30-day slope of ETH/BTC ratio
    "ethbtc_zscore_90d",         # 90-day z-score of ETH/BTC ratio

    # --- Macro sentiment ---
    "funding_zscore",            # BTC funding z-score (proxy for crypto risk-on)
]


# ---------------------------------------------------------------------------
# Causal indicator helpers
# ---------------------------------------------------------------------------
def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1.0 / length, adjust=False).mean()
    roll_dn = down.ewm(alpha=1.0 / length, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False).mean()


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------
def build_features(
    df_eth_4h: pd.DataFrame,
    df_eth_1d: pd.DataFrame,
    df_btc_4h: pd.DataFrame,
    df_btc_1d: pd.DataFrame,
    df_ethbtc_1d: pd.DataFrame,
    df_funding: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build the 16-feature matrix. NO look-ahead — all MTF inputs shift(1)-ed.

    All DataFrames must be tz-aware UTC and sorted ascending.
    """
    out = pd.DataFrame(index=df_eth_4h.index)

    c4 = df_eth_4h["close"]
    h4 = df_eth_4h["high"]
    l4 = df_eth_4h["low"]
    v4 = df_eth_4h["volume"]

    # === ETH 4h indicators (causal) ===
    rsi = _rsi(c4, 14)
    out["rsi14_4h"] = rsi
    out["rsi_slope_4h"] = rsi.diff(3)

    mid = c4.rolling(20).mean()
    std = c4.rolling(20).std()
    bb_up = mid + 2 * std
    bb_low = mid - 2 * std
    bb_range = (bb_up - bb_low).replace(0, np.nan)
    out["bb_pct_4h"] = (c4 - bb_low) / bb_range

    atr = _atr(df_eth_4h, 14)
    out["atr_pct_4h"] = (atr / c4) * 100.0

    vol_ma = v4.rolling(20).mean()
    out["vol_ratio_4h"] = (v4 / vol_ma.replace(0, np.nan)).clip(0, 10)

    out["ret_5_4h"] = c4.pct_change(5) * 100.0

    high20 = h4.rolling(20).max().shift(1)
    low20 = l4.rolling(20).min().shift(1)
    out["dist_high20_4h"] = (high20 - c4) / c4 * 100.0
    out["dist_low20_4h"] = (c4 - low20) / c4 * 100.0

    # === ETH daily regime (shift(1) BEFORE reindex) ===
    c_eth_d = df_eth_1d["close"]
    ema20_d = _ema(c_eth_d, 20)
    ema50_d = _ema(c_eth_d, 50)
    ema200_d = _ema(c_eth_d, 200)
    eth_ema_ratio = ((ema20_d / ema50_d) - 1.0).shift(1)
    eth_above_200 = ((c_eth_d > ema200_d).astype(float)).shift(1)
    out["ema20_50_ratio_1d_eth"] = eth_ema_ratio.reindex(df_eth_4h.index, method="ffill")
    out["close_above_ema200_1d_eth"] = eth_above_200.reindex(df_eth_4h.index, method="ffill")

    # === BTC daily regime (shift(1) BEFORE reindex) ===
    c_btc_d = df_btc_1d["close"]
    ema20_btc = _ema(c_btc_d, 20)
    ema50_btc = _ema(c_btc_d, 50)
    btc_ratio = ((ema20_btc / ema50_btc) - 1.0).shift(1)
    out["ema20_50_ratio_1d_btc"] = btc_ratio.reindex(df_eth_4h.index, method="ffill")

    # === BTC 4h context (ETH typically follows BTC) ===
    # Align BTC 4h to ETH 4h index via reindex method='ffill' (no look-ahead:
    # uses last BTC close at or before each ETH bar; same timestamp = same
    # closed bar for both).
    btc_4h_aligned = df_btc_4h.reindex(df_eth_4h.index, method="ffill")
    btc_c4 = btc_4h_aligned["close"]
    btc_v4 = btc_4h_aligned["volume"]
    out["btc_ret_5_4h"] = btc_c4.pct_change(5) * 100.0
    btc_vol_ma = btc_v4.rolling(20).mean()
    out["btc_vol_ratio_4h"] = (btc_v4 / btc_vol_ma.replace(0, np.nan)).clip(0, 10)

    # === ETH/BTC ratio dynamics (literature: key driver of ETH alpha) ===
    eb = df_ethbtc_1d["close"].copy()
    # 30-day slope (% change over 30 days) and 90-day rolling z-score
    eb_slope_30d = (eb / eb.shift(30) - 1.0) * 100.0
    eb_mean_90d = eb.rolling(90).mean()
    eb_std_90d = eb.rolling(90).std()
    eb_z_90d = ((eb - eb_mean_90d) / eb_std_90d.replace(0, np.nan)).clip(-5, 5)
    # shift(1) BEFORE reindex — the daily ETH/BTC at day D is known after D
    eb_slope_30d = eb_slope_30d.shift(1)
    eb_z_90d = eb_z_90d.shift(1)
    out["ethbtc_slope_30d"] = eb_slope_30d.reindex(df_eth_4h.index, method="ffill")
    out["ethbtc_zscore_90d"] = eb_z_90d.reindex(df_eth_4h.index, method="ffill")

    # === BTC funding rate z-score (crypto risk sentiment) ===
    if df_funding is not None and len(df_funding) > 0:
        fr = df_funding["funding_rate"].copy()
        fr_sh = fr.shift(1)
        fr_4h = fr_sh.reindex(df_eth_4h.index, method="ffill")
        win = 540  # ~90 days at 4h
        mean = fr_4h.rolling(win, min_periods=120).mean()
        std = fr_4h.rolling(win, min_periods=120).std()
        out["funding_zscore"] = ((fr_4h - mean) / std.replace(0, np.nan)).clip(-5, 5)
    else:
        out["funding_zscore"] = 0.0

    out = out[FEATURES]
    return out


# ---------------------------------------------------------------------------
# Label (forward-looking, used only for training targets)
# ---------------------------------------------------------------------------
def create_labels(
    df_4h: pd.DataFrame,
    tp: float = 0.03,
    sl: float = 0.015,
    max_bars: int = 12,
) -> pd.Series:
    """Binary label: 1 if entering LONG at close[i] hits +tp before -sl within
    max_bars subsequent bars. 0 otherwise. NaN if not enough future to resolve.

    Pessimistic tie-break: a single bar hitting BOTH levels -> 0 (SL).
    Uses ONLY bars i+1..i+max_bars (no look-ahead at the entry timestamp).
    """
    closes = df_4h["close"].values
    highs = df_4h["high"].values
    lows = df_4h["low"].values
    n = len(closes)
    labels = np.full(n, np.nan)

    for i in range(n - 1):
        entry = closes[i]
        tp_price = entry * (1 + tp)
        sl_price = entry * (1 - sl)
        max_j = min(i + max_bars, n - 1)
        outcome = np.nan
        for j in range(i + 1, max_j + 1):
            hi = highs[j]
            lo = lows[j]
            if lo <= sl_price:
                outcome = 0
                break
            if hi >= tp_price:
                outcome = 1
                break
        if not np.isnan(outcome):
            labels[i] = outcome

    return pd.Series(labels, index=df_4h.index, name="label")


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_model(path: Path | None = None) -> tuple[object, object]:
    path = path or ROOT
    model = joblib.load(path / "model.pkl")
    scaler = joblib.load(path / "scaler.pkl")
    return model, scaler


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
def signal(
    df: pd.DataFrame,
    idx: int,
    params: dict,
    model_pack: tuple,
) -> str | None:
    """Return 'LONG' or None for bar `idx`. df must include FEATURES columns."""
    if idx < 0 or idx >= len(df):
        return None
    row = df.iloc[idx]
    feats = row[FEATURES]
    if feats.isna().any():
        return None

    # Defensive regime filter
    if params.get("use_regime_filter", True):
        eth_ratio = row.get("ema20_50_ratio_1d_eth", 0.0)
        btc_ratio = row.get("ema20_50_ratio_1d_btc", 0.0)
        # Block LONG if BOTH ETH and BTC daily trends are deeply bearish.
        # Using -3% threshold (slightly more permissive than Agent B's -2%)
        # because ETH is more volatile; we still let "moderate BEAR" through
        # and rely on the model probability.
        if eth_ratio < -0.03 and btc_ratio < -0.03:
            return None

    model, scaler = model_pack
    X = feats.values.reshape(1, -1)
    X_s = scaler.transform(X)
    p = float(model.predict_proba(X_s)[0, 1])
    if p >= params["threshold"]:
        return "LONG"
    return None


# ---------------------------------------------------------------------------
# Honest simulator — no overlap, no intra-bar look-ahead trailing
# (Mirror of Agent B for direct comparability)
# ---------------------------------------------------------------------------
def simulate(df: pd.DataFrame, entry_bar: int, params: dict) -> dict:
    """Simulate ONE LONG trade entered at the close of `entry_bar`.

    Exit rules:
      - initial stop = entry * (1 - sl_init)
      - if close[bar] >= entry * (1 + trail_activate): trailing activates
            stop = max(stop, peak_close * (1 - trail_dist))
            peak_close = max CLOSE seen so far (NOT high — strictly conservative)
      - **Intrabar safety**: exit check compares bar's LOW against the stop
        as it was AT THE END OF THE PRIOR BAR. Only afterwards do we update
        peak_close / stop using close[b]. Prevents the look-ahead intrabar bug.
      - max_bars_sim reached -> exit at that bar's close (TIMEOUT).
    """
    sl_init = params["sl_init"]
    trail_dist = params["trail_dist"]
    trail_act = params["trail_activate"]
    max_bars = params["max_bars_sim"]
    comm = params["commission"]

    n = len(df)
    if entry_bar >= n - 1:
        return {"outcome": "INVALID", "exit_bar": entry_bar,
                "exit_price": np.nan, "pnl_pct": 0.0, "bars": 0,
                "entry_price": np.nan}

    entry_price = float(df["close"].iloc[entry_bar])
    stop = entry_price * (1 - sl_init)
    peak_close = entry_price
    trailing_on = False

    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= n:
            break
        lo = float(df["low"].iloc[b])
        # 1) Exit FIRST using stop set in the prior iteration
        if lo <= stop:
            pnl = (stop - entry_price) / entry_price - 2 * comm
            outcome = "SL" if stop <= entry_price else "TRAIL"
            return {"outcome": outcome, "exit_bar": b, "exit_price": stop,
                    "pnl_pct": pnl, "bars": i, "entry_price": entry_price}
        # 2) Update peak/stop using close[b] for the NEXT iteration
        cl = float(df["close"].iloc[b])
        if cl > peak_close:
            peak_close = cl
        if not trailing_on and peak_close >= entry_price * (1 + trail_act):
            trailing_on = True
        if trailing_on:
            new_stop = peak_close * (1 - trail_dist)
            if new_stop > stop:
                stop = new_stop

    # Timeout
    b = min(entry_bar + max_bars, n - 1)
    exit_price = float(df["close"].iloc[b])
    pnl = (exit_price - entry_price) / entry_price - 2 * comm
    return {
        "outcome": "TIMEOUT",
        "exit_bar": b,
        "exit_price": exit_price,
        "pnl_pct": pnl,
        "bars": b - entry_bar,
        "entry_price": entry_price,
    }


# ---------------------------------------------------------------------------
# Honest backtest with no overlapping trades
# ---------------------------------------------------------------------------
def backtest(
    df: pd.DataFrame,
    params: dict,
    model_pack: tuple,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> pd.DataFrame:
    n = len(df)
    end_idx = end_idx if end_idx is not None else n
    i = start_idx
    trades = []
    while i < end_idx - 1:
        sig = signal(df, i, params, model_pack)
        if sig == "LONG":
            res = simulate(df, i, params)
            if res["outcome"] == "INVALID":
                break
            res["entry_bar"] = i
            res["entry_ts"] = df.index[i]
            res["exit_ts"] = df.index[res["exit_bar"]]
            trades.append(res)
            i = res["exit_bar"] + 1
        else:
            i += 1
    return pd.DataFrame(trades)
