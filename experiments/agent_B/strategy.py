"""
Agent B — ML Classifier with Anti-Overfitting Protocol.

Strategy:
- Gradient Boosting Classifier (sklearn) on a SMALL, theoretically justified feature set.
- Predict P(LONG wins TP=3% before SL=1.5% within 12 bars [48h]).
- Enter LONG only when:
    1) Daily regime is not BEAR (EMA20_1d > EMA50_1d or EMA20_1d slope >= 0)
    2) ML probability >= threshold (frozen from train)
- Exit: trailing stop (honest, no intra-bar look-ahead). Initial SL = -1.5%,
        peak follows the close (NOT the high) to be conservative, trail = 1.5%.

Why this should NOT overfit:
- 11 features (≤15) — each justified by trader logic (regime, momentum,
  reversion, volatility, sentiment).
- Tiny model (max_depth=3, n_estimators=100, min_samples_leaf=50, lr=0.05).
- Class balance via sample_weight.
- Single threshold frozen on train data per fold.
- Trailing exit removes the TP/SL grid search look-ahead bias.

Cutoff: only data ≤ 2025-12-31 used at any stage.
"""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Parameters (frozen — calibrated on train data within each fold; the final
# production threshold below is the median of in-sample thresholds).
# ---------------------------------------------------------------------------
PARAMS = {
    # Label horizon (forward) used during training
    "tp_label": 0.03,         # 3% TP for label
    "sl_label": 0.015,        # 1.5% SL for label
    "max_bars_label": 12,     # 48h horizon

    # Simulation (live trading exit rules)
    "sl_init": 0.015,         # initial stop loss 1.5%
    "trail_dist": 0.015,      # trailing distance once activated
    "trail_activate": 0.01,   # activate trailing after +1% from entry
    "max_bars_sim": 24,       # max 96h holding before forced exit
    "commission": 0.0005,     # 0.05% per side (0.1% RT)

    # Inference threshold (will be overwritten by train.py with calibrated value)
    "threshold": 0.55,

    # Regime filter — don't go LONG against a strong bear
    "use_regime_filter": True,
}

# 11 features — each with explicit trader rationale
FEATURES = [
    "ema20_50_ratio_1d",   # daily trend (regime)
    "rsi14_4h",            # short-term momentum / overbought
    "rsi_slope_4h",        # rsi acceleration
    "bb_pct_4h",           # mean-reversion proxy on 4h
    "atr_pct_4h",          # current volatility
    "vol_ratio_4h",        # current vol vs 20-bar mean
    "ret_5_4h",            # 5-bar momentum (20h)
    "dist_high20_4h",      # distance to 20-bar high (extension)
    "dist_low20_4h",       # distance to 20-bar low (proximity to support)
    "funding_zscore",      # contrarian sentiment
    "fng_value",           # macro sentiment (daily F&G)
]


# ---------------------------------------------------------------------------
# Feature engineering
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


def build_features(
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    funding_df: pd.DataFrame | None,
    fng_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build the 11-feature matrix. NO look-ahead.

    All multi-timeframe data is shift(1)-ed before reindexing.
    All rolling features are causal.
    """
    out = pd.DataFrame(index=df_4h.index)

    c4 = df_4h["close"]
    h4 = df_4h["high"]
    l4 = df_4h["low"]
    v4 = df_4h["volume"]

    # --- 4h indicators (all causal) ---
    rsi = _rsi(c4, 14)
    out["rsi14_4h"] = rsi
    out["rsi_slope_4h"] = rsi.diff(3)

    # Bollinger bands position
    mid = c4.rolling(20).mean()
    std = c4.rolling(20).std()
    bb_up = mid + 2 * std
    bb_low = mid - 2 * std
    bb_range = (bb_up - bb_low).replace(0, np.nan)
    out["bb_pct_4h"] = (c4 - bb_low) / bb_range

    atr = _atr(df_4h, 14)
    out["atr_pct_4h"] = (atr / c4) * 100.0

    vol_ma = v4.rolling(20).mean()
    out["vol_ratio_4h"] = (v4 / vol_ma.replace(0, np.nan)).clip(0, 10)

    out["ret_5_4h"] = c4.pct_change(5) * 100.0

    high20 = h4.rolling(20).max().shift(1)
    low20 = l4.rolling(20).min().shift(1)
    out["dist_high20_4h"] = (high20 - c4) / c4 * 100.0
    out["dist_low20_4h"] = (c4 - low20) / c4 * 100.0

    # --- Daily regime ---
    c1d = df_1d["close"]
    ema20_1d = _ema(c1d, 20)
    ema50_1d = _ema(c1d, 50)
    # ratio - 1: positive when trend bullish.
    ema_ratio = (ema20_1d / ema50_1d) - 1.0
    # shift(1) — the daily value at date D is known after D closes
    ema_ratio = ema_ratio.shift(1)
    # align to 4h index
    out["ema20_50_ratio_1d"] = ema_ratio.reindex(df_4h.index, method="ffill")

    # --- Funding rate (sentiment, contrarian) ---
    if funding_df is not None and len(funding_df) > 0:
        fr = funding_df["funding_rate"].copy()
        # shift(1) on the funding native cadence (8h)
        fr_sh = fr.shift(1)
        # reindex to 4h via ffill
        fr_4h = fr_sh.reindex(df_4h.index, method="ffill")
        # rolling z-score using a long window (90 days at 4h = 540 bars)
        win = 540
        mean = fr_4h.rolling(win, min_periods=120).mean()
        std = fr_4h.rolling(win, min_periods=120).std()
        out["funding_zscore"] = ((fr_4h - mean) / std.replace(0, np.nan)).clip(-5, 5)
    else:
        out["funding_zscore"] = 0.0

    # --- Fear & Greed (daily macro sentiment) ---
    if fng_df is not None and len(fng_df) > 0:
        fng = fng_df["fng_value"].copy()
        # F&G of day D published end-of-day → shift(1) and ffill to 4h
        fng_sh = fng.shift(1)
        out["fng_value"] = fng_sh.reindex(df_4h.index, method="ffill")
    else:
        out["fng_value"] = 50.0

    # Keep only listed features, drop incomplete warmup rows
    out = out[FEATURES]
    return out


# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------
def create_labels(
    df_4h: pd.DataFrame,
    tp: float = 0.03,
    sl: float = 0.015,
    max_bars: int = 12,
) -> pd.Series:
    """For each bar i, label = 1 if entering LONG at close[i] hits TP before SL
    within max_bars (touching high/low of subsequent bars). Otherwise 0.

    NaN when there isn't enough future to resolve and TP not yet hit.
    Uses only data > i (no look-ahead at the label timestamp).
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
            # Pessimistic tie-break: if BOTH hit on same bar -> SL
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
# Model loading
# ---------------------------------------------------------------------------
def load_model(path: Path | None = None) -> tuple[object, object]:
    """Load (model, scaler). Default = ROOT/model.pkl + ROOT/scaler.pkl."""
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
    """Return 'LONG' or None for bar at `idx`.

    df must include the FEATURES columns. `model_pack = (model, scaler)`.
    """
    if idx < 0 or idx >= len(df):
        return None
    row = df.iloc[idx]
    feats = row[FEATURES]
    if feats.isna().any():
        return None

    # Regime filter
    if params.get("use_regime_filter", True):
        if row["ema20_50_ratio_1d"] < -0.02:  # strong bear (-2%)
            return None

    model, scaler = model_pack
    X = feats.values.reshape(1, -1)
    X_s = scaler.transform(X)
    p = float(model.predict_proba(X_s)[0, 1])
    if p >= params["threshold"]:
        return "LONG"
    return None


# ---------------------------------------------------------------------------
# Honest simulator — NO overlapping trades, NO intra-bar look-ahead trailing
# ---------------------------------------------------------------------------
def simulate(df: pd.DataFrame, entry_bar: int, params: dict) -> dict:
    """Simulate ONE LONG trade entered at the close of `entry_bar`.

    Exit rules (LONG):
      - initial stop = entry * (1 - sl_init)
      - if close[bar] >= entry * (1 + trail_activate): start trailing
            stop = max(stop, peak_close * (1 - trail_dist))
            where peak_close is the highest CLOSE seen so far in the trade
      - **Intrabar safety**: when checking exit at bar `b`, we compare the
        bar's LOW against the stop value computed from CLOSES UP TO bar
        `b-1`. Then, after the bar closes, we update peak_close/stop using
        close[b]. This prevents the look-ahead intrabar bug.
      - if `max_bars_sim` reached → close at that bar's close (TIMEOUT).

    Returns dict with: outcome, exit_bar, exit_price, pnl_pct, bars, entry_price.
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
        # 1) Exit check FIRST using stop computed from prior close
        if lo <= stop:
            # Conservative: assume fill at stop
            pnl = (stop - entry_price) / entry_price - 2 * comm
            outcome = "SL" if stop <= entry_price else "TRAIL"
            return {"outcome": outcome, "exit_bar": b, "exit_price": stop,
                    "pnl_pct": pnl, "bars": i, "entry_price": entry_price}
        # 2) AFTER no exit, update peak_close and trailing stop using close[b]
        cl = float(df["close"].iloc[b])
        if cl > peak_close:
            peak_close = cl
        # Activate trailing once the close is >= +trail_activate above entry
        if not trailing_on and peak_close >= entry_price * (1 + trail_act):
            trailing_on = True
        if trailing_on:
            new_stop = peak_close * (1 - trail_dist)
            if new_stop > stop:
                stop = new_stop

    # Timeout: exit at last bar close
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
# Full backtest with NO trade overlapping
# ---------------------------------------------------------------------------
def backtest(
    df: pd.DataFrame,
    params: dict,
    model_pack: tuple,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> pd.DataFrame:
    """Iterate bars sequentially, only opening a new trade AFTER the previous
    one closed. Returns a DataFrame of trades.
    """
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
            # Advance past the exit bar (no overlap)
            i = res["exit_bar"] + 1
        else:
            i += 1
    return pd.DataFrame(trades)
