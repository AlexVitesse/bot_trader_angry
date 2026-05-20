"""
Agent O — ML Classifier for SOL/USDT 4h with CROSS-ASSET features
(LightGBM + anti-overfitting protocol).

Hypothesis (different from Agents B and G):
- Agent B (BTC, 11 BTC-only features): train AUC 0.75 -> test AUC 0.52 (no edge)
- Agent G (ETH, 16 features inc. ETH/BTC ratio): train 0.81 -> test 0.51
  (worst overfitting gap of the project; ETH/BTC ratio is most important
  feature but adds no OOS edge).
- This run (SOL): test whether COMBINING the BTC + ETH + SOL state space
  (cross-asset features) lets a tree model find a non-linear interaction
  that is more stable OOS than single-asset features. SOL beta ~1.5 to BTC
  -> its return should be driven by the BROADER market state, not its own
  history alone. If this also fails, the line "ML 4h classification in
  crypto" is closed definitively.

Strategy:
- LightGBM classifier on 16 features (<=20 budget):
  * 5 SOL 4h features (own price action)
  * 5 BTC features (4h + daily, including encoded regime)
  * 3 ETH features (daily regime + ret_5 + ETH/BTC ratio)
  * 3 cross-asset features (SOL-BTC corr/divergence, funding)
- Target: binary label = 1 if LONG at close[i] hits +4% TP before -2.5% SL
  within 12 bars (label scaled to SOL's higher volatility vs BTC's TP=3%/SL=1.5%).
- Entry: ML prob >= threshold AND not in deep BEAR daily regime (SOL+BTC)
  AND funding not catastrophic.
- Exit: FIXED TP/SL (no trailing) — explicitly avoids the intra-bar look-ahead
  trailing bug; TP=4%, SL=2.5%, max_bars=12.

Anti-overfitting constraints:
- LightGBM caps at the brief's regularization bound:
  max_depth=4, num_leaves=15, n_estimators=200, min_data_in_leaf=80,
  reg_alpha=1.0, reg_lambda=1.0, learning_rate=0.03, feat/bag fraction 0.8.
- 16 features (<=20), each with explicit trader rationale (no data mining).
- Class balance via sample_weight.
- Threshold calibrated on TRAIN of each fold, frozen for test.
- Cutoff 2025-12-31 enforced at every load.

Files in this directory:
  strategy.py     (this file)
  train.py        (purged CV + retrain on all data <= cutoff)
  bootstrap_test.py
  synth_test.py
  model.pkl, scaler.pkl, trained_params.json, cv_*.json, wf_trades.csv
"""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Frozen parameters (threshold overwritten by train.py to median-of-passing-folds)
# ---------------------------------------------------------------------------
PARAMS = {
    # Label horizon (forward) used at training time — scaled for SOL volatility
    "tp_label": 0.04,
    "sl_label": 0.025,
    "max_bars_label": 12,

    # Live sim rules — FIXED TP/SL (no trailing; trailing has the documented
    # intra-bar look-ahead failure mode that broke 12 ADA folds). Honest test
    # of pure-classifier edge.
    "tp_sim": 0.04,
    "sl_sim": 0.025,
    "max_bars_sim": 12,
    "commission": 0.0005,  # 0.05% per side -> 0.10% round trip

    # Inference threshold (overwritten by train.py)
    "threshold": 0.55,

    # Regime defensive filters
    "use_regime_filter": True,
    "use_funding_veto": True,
}

# 16 cross-asset features (<=20 budget). Each justified.
FEATURES = [
    # === SOL 4h price action (5) ===
    "sol_rsi14_4h",         # SOL 14-period RSI (momentum / OB-OS)
    "sol_bb_pct_4h",        # SOL position inside Bollinger Bands (mean rev)
    "sol_atr_pct_4h",       # SOL volatility regime
    "sol_vol_ratio_4h",     # SOL volume vs 20-bar mean (thrust)
    "sol_ret_5_4h",         # SOL 5-bar momentum

    # === BTC features (5) — clave por beta 1.5 ===
    "btc_ema20_50_ratio_1d",   # BTC daily trend (shift 1d) — broader regime
    "btc_rsi14_4h",            # BTC 4h momentum
    "btc_ret_5_4h",            # BTC 5-bar return (lead/lag)
    "btc_vol_ratio_4h",        # BTC volume thrust
    "btc_regime_1d",           # BTC daily regime encoded: -1 BEAR / 0 RANGE / +1 BULL

    # === ETH features (3) ===
    "ethbtc_zscore_90d",    # ETH/BTC 90d z-score (broad alt rotation signal)
    "eth_regime_1d",        # ETH daily regime encoded -1/0/+1
    "eth_ret_5_4h",         # ETH 5-bar return (alt sentiment)

    # === Cross-asset (3) ===
    "sol_btc_corr_168",     # SOL-BTC rolling Pearson corr over 168 bars (~28 days)
    "sol_btc_ret_div_5",    # SOL ret_5 minus BTC ret_5 (divergence)
    "funding_zscore",       # BTC funding z-score (shift 1; risk sentiment)
]


# ---------------------------------------------------------------------------
# Causal indicator helpers (manual; no pandas_ta — brief says it's broken)
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


def _daily_regime(df_1d: pd.DataFrame) -> pd.Series:
    """Encode daily regime: -1 BEAR / 0 RANGE / +1 BULL using EMA20 vs EMA50
    with a small dead zone (project-standard 2%). Causal (shift 1) is applied
    outside this helper (before reindex)."""
    c = df_1d["close"]
    ema20 = _ema(c, 20)
    ema50 = _ema(c, 50)
    ratio = (ema20 / ema50) - 1.0  # >0 BULL, <0 BEAR
    regime = pd.Series(0.0, index=c.index)
    regime[ratio > 0.02] = 1.0
    regime[ratio < -0.02] = -1.0
    return regime


# ---------------------------------------------------------------------------
# Feature builder — cross-asset
# ---------------------------------------------------------------------------
def build_features(
    df_sol_4h: pd.DataFrame,
    df_btc_4h: pd.DataFrame,
    df_eth_4h: pd.DataFrame,
    df_btc_1d: pd.DataFrame,
    df_eth_1d: pd.DataFrame,
    df_ethbtc_1d: pd.DataFrame,
    df_funding: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build the 16-feature cross-asset matrix. NO look-ahead — all MTF/daily
    inputs are .shift(1)-ed BEFORE reindex to 4h.
    All inputs must be tz-aware UTC and sorted ascending.
    """
    out = pd.DataFrame(index=df_sol_4h.index)

    # ---------- SOL 4h ----------
    c4 = df_sol_4h["close"]
    h4 = df_sol_4h["high"]
    l4 = df_sol_4h["low"]
    v4 = df_sol_4h["volume"]

    out["sol_rsi14_4h"] = _rsi(c4, 14)

    mid = c4.rolling(20).mean()
    std = c4.rolling(20).std()
    bb_up = mid + 2 * std
    bb_low = mid - 2 * std
    bb_range = (bb_up - bb_low).replace(0, np.nan)
    out["sol_bb_pct_4h"] = (c4 - bb_low) / bb_range

    atr = _atr(df_sol_4h, 14)
    out["sol_atr_pct_4h"] = (atr / c4) * 100.0

    vol_ma = v4.rolling(20).mean()
    out["sol_vol_ratio_4h"] = (v4 / vol_ma.replace(0, np.nan)).clip(0, 10)
    out["sol_ret_5_4h"] = c4.pct_change(5) * 100.0

    # ---------- BTC 4h ----------
    btc_4h_aligned = df_btc_4h.reindex(df_sol_4h.index, method="ffill")
    btc_c4 = btc_4h_aligned["close"]
    btc_v4 = btc_4h_aligned["volume"]
    out["btc_rsi14_4h"] = _rsi(btc_c4, 14)
    out["btc_ret_5_4h"] = btc_c4.pct_change(5) * 100.0
    btc_vol_ma = btc_v4.rolling(20).mean()
    out["btc_vol_ratio_4h"] = (btc_v4 / btc_vol_ma.replace(0, np.nan)).clip(0, 10)

    # ---------- BTC 1d (shift 1 BEFORE reindex) ----------
    c_btc_d = df_btc_1d["close"]
    ema20_btc = _ema(c_btc_d, 20)
    ema50_btc = _ema(c_btc_d, 50)
    btc_ratio = ((ema20_btc / ema50_btc) - 1.0).shift(1)
    out["btc_ema20_50_ratio_1d"] = btc_ratio.reindex(df_sol_4h.index, method="ffill")

    btc_regime = _daily_regime(df_btc_1d).shift(1)
    out["btc_regime_1d"] = btc_regime.reindex(df_sol_4h.index, method="ffill")

    # ---------- ETH features ----------
    eth_4h_aligned = df_eth_4h.reindex(df_sol_4h.index, method="ffill")
    out["eth_ret_5_4h"] = eth_4h_aligned["close"].pct_change(5) * 100.0

    eth_regime = _daily_regime(df_eth_1d).shift(1)
    out["eth_regime_1d"] = eth_regime.reindex(df_sol_4h.index, method="ffill")

    eb = df_ethbtc_1d["close"]
    eb_mean_90d = eb.rolling(90).mean()
    eb_std_90d = eb.rolling(90).std()
    eb_z_90d = ((eb - eb_mean_90d) / eb_std_90d.replace(0, np.nan)).clip(-5, 5)
    eb_z_90d = eb_z_90d.shift(1)
    out["ethbtc_zscore_90d"] = eb_z_90d.reindex(df_sol_4h.index, method="ffill")

    # ---------- Cross-asset (3) ----------
    # SOL-BTC 168-bar rolling Pearson corr on log returns (~28 days)
    sol_logret = np.log(c4 / c4.shift(1))
    btc_logret = np.log(btc_c4 / btc_c4.shift(1))
    out["sol_btc_corr_168"] = sol_logret.rolling(168).corr(btc_logret).clip(-1, 1)

    # Divergence: SOL outperformance vs BTC over 5 bars
    out["sol_btc_ret_div_5"] = (c4.pct_change(5) - btc_c4.pct_change(5)) * 100.0

    # BTC funding z-score (shift 1 BEFORE reindex)
    if df_funding is not None and len(df_funding) > 0:
        fr = df_funding["funding_rate"].copy()
        fr_sh = fr.shift(1)
        fr_4h = fr_sh.reindex(df_sol_4h.index, method="ffill")
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
    tp: float = 0.04,
    sl: float = 0.025,
    max_bars: int = 12,
) -> pd.Series:
    """Binary label: 1 if entering LONG at close[i] hits +tp before -sl within
    max_bars subsequent bars. 0 otherwise. NaN if not enough future to resolve.

    Pessimistic tie-break: a single bar hitting BOTH levels -> 0 (SL).
    Uses ONLY bars i+1..i+max_bars (no look-ahead at entry timestamp).
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
            # Pessimistic: if same bar touches both, treat as SL
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
# Signal — defensive filters + threshold
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

    # Defensive: block LONG if BOTH SOL/BTC are in deep BEAR daily regime.
    # We do NOT use sol_regime explicitly (not in feature set); we use the
    # BTC + ETH regimes already in features as a proxy for "broad market BEAR".
    if params.get("use_regime_filter", True):
        btc_reg = row.get("btc_regime_1d", 0.0)
        eth_reg = row.get("eth_regime_1d", 0.0)
        # Both regimes BEAR -> skip (avoid catching falling knife)
        if btc_reg <= -1.0 and eth_reg <= -1.0:
            return None

    # Funding veto: extreme positive funding = crowded long, contrarian risk
    if params.get("use_funding_veto", True):
        fz = row.get("funding_zscore", 0.0)
        if not np.isnan(fz) and fz > 2.0:
            return None

    model, scaler = model_pack
    X = feats.values.reshape(1, -1)
    X_s = scaler.transform(X)
    p = float(model.predict_proba(X_s)[0, 1])
    if p >= params["threshold"]:
        return "LONG"
    return None


# ---------------------------------------------------------------------------
# Honest simulator — FIXED TP/SL, no overlap, no intra-bar look-ahead
# ---------------------------------------------------------------------------
def simulate(df: pd.DataFrame, entry_bar: int, params: dict) -> dict:
    """Simulate ONE LONG trade entered at close of `entry_bar` with FIXED TP/SL.

    Exit rules (deterministic):
      - SL at entry*(1-sl_sim)
      - TP at entry*(1+tp_sim)
      - On every subsequent bar i+1..i+max_bars: check low <= sl (pessimistic
        first), then high >= tp; if both in same bar -> SL.
      - Timeout at max_bars -> exit at that bar's close.

    Commissions: 2*commission round-trip.
    """
    tp = params["tp_sim"]
    sl = params["sl_sim"]
    max_bars = params["max_bars_sim"]
    comm = params["commission"]

    n = len(df)
    if entry_bar >= n - 1:
        return {"outcome": "INVALID", "exit_bar": entry_bar,
                "exit_price": np.nan, "pnl_pct": 0.0, "bars": 0,
                "entry_price": np.nan}

    entry_price = float(df["close"].iloc[entry_bar])
    tp_price = entry_price * (1 + tp)
    sl_price = entry_price * (1 - sl)

    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= n:
            break
        hi = float(df["high"].iloc[b])
        lo = float(df["low"].iloc[b])
        # Pessimistic order: check SL first (if both hit in same bar -> SL)
        if lo <= sl_price:
            pnl = (sl_price - entry_price) / entry_price - 2 * comm
            return {"outcome": "SL", "exit_bar": b, "exit_price": sl_price,
                    "pnl_pct": pnl, "bars": i, "entry_price": entry_price}
        if hi >= tp_price:
            pnl = (tp_price - entry_price) / entry_price - 2 * comm
            return {"outcome": "TP", "exit_bar": b, "exit_price": tp_price,
                    "pnl_pct": pnl, "bars": i, "entry_price": entry_price}

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
