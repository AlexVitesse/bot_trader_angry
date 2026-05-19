"""
agent_C/strategy.py
===================
Regime-Adaptive BTC/USDT 4h strategy. Self-contained.

Philosophy
----------
- BULL: pullback to EMA20 with up-slope (trend-following). Wide trailing (1.8x ATR).
- RANGE: deep oversold mean-reversion (RSI<25 + lower BB + bullish candle).
- BEAR: STAY FLAT. The project's own history (CLAUDE.md) shows SHORT in crypto
        4h does NOT generalize. Adding a SHORT just to fill BEAR adds parameters
        without edge.

Anti-overfitting discipline:
- Each sub-strategy has 4-5 parameters max.
- All thresholds are round/standard (RSI 25/30/70, EMA 20/50/200, ATR ×1.5/×1.8) —
  not optimized to 2 decimal places.
- One position at a time, no intrabar look-ahead trailing.

Use with the honest engine convention from revalidate_v15.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# FROZEN PARAMETERS — committed once, then NEVER re-tuned per-fold
# -----------------------------------------------------------------------------
PARAMS = {
    # ----- regime detection (daily EMAs, shifted by 1 day -> no look-ahead) ---
    'regime_dead_zone': 0.02,    # 2% band between EMA20/EMA50 -> RANGE
    'regime_use_ema200_filter': True,  # close>EMA200 demotes BEAR -> RANGE

    # ----- BULL: pullback to EMA20 -----------------------------------------
    'bull_pullback_ema20_max_dist': 0.015,   # close within ±1.5% of EMA20
    'bull_pullback_require_above_ema50': True,  # ema20 > ema50 (4h)
    'bull_pullback_require_daily_alignment': True,  # ema50_1d > ema200_1d -> "real" uptrend
    'bull_pullback_rsi_min': 40,             # not over-extended
    'bull_pullback_rsi_max': 60,             # not over-extended
    'bull_pullback_min_atr_pct': 0.5,        # avoid dead market (>=0.5% ATR)
    'bull_pullback_require_bullish_entry': True,  # entry bar close > open (dip being bought)

    'bull_trail_atr_mult': 1.8,              # WIDE trail per agent brief
    'bull_trail_floor': 0.012,               # 1.2% min trail
    'bull_max_bars': 24,                     # ~4 days timeout

    # ----- RANGE: deep oversold mean reversion ------------------------------
    # Standard Wilder oversold = RSI<30. BB_pct<0.15 = within 15% of lower band.
    'range_rsi_max': 30,                     # Wilder oversold
    'range_bb_pct_max': 0.15,                # close to lower BB
    'range_require_bullish_candle': True,    # close > open on the entry bar
    'range_min_atr_pct': 0.5,                # need some volatility

    'range_tp_atr_mult': 1.5,                # quick target
    'range_sl_atr_mult': 1.5,                # symmetric SL
    'range_max_bars': 12,                    # ~2 days timeout

    # ----- BEAR: stay flat (no SHORT) ---------------------------------------
    'bear_active': False,

    # ----- universal ---------------------------------------------------------
    'commission_one_way': 0.0005,            # 0.05% per side (0.1% RT)
    'min_history_bars': 220,                 # for EMA200 / daily features
}


# -----------------------------------------------------------------------------
# FEATURE COMPUTATION (4h) — no look-ahead, all rolling on past bars
# -----------------------------------------------------------------------------
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def _bbands(close: pd.Series, n: int = 20, k: float = 2.0):
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    pct = (close - lower) / (upper - lower).replace(0, np.nan)
    return lower, ma, upper, pct


def compute_features(df_4h: pd.DataFrame) -> pd.DataFrame:
    """4h indicators. All rolling = no look-ahead."""
    df = df_4h.copy()
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    df['ema20'] = _ema(c, 20)
    df['ema50'] = _ema(c, 50)
    df['ema200'] = _ema(c, 200)
    df['rsi14'] = _rsi(c, 14)
    atr = _atr(h, l, c, 14)
    df['atr14'] = atr
    df['atr_pct'] = atr / c * 100
    _, _, _, bb_pct = _bbands(c, 20, 2.0)
    df['bb_pct'] = bb_pct
    df['ema20_slope_5'] = df['ema20'].pct_change(5) * 100
    return df


def compute_macro_daily(df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Daily EMAs for regime detection. shift(1) -> today uses yesterday's regime.
    NO look-ahead.
    """
    daily = df_4h['close'].resample('1D').last().dropna()
    out = pd.DataFrame(index=daily.index)
    out['ema20_1d'] = daily.ewm(span=20, adjust=False).mean()
    out['ema50_1d'] = daily.ewm(span=50, adjust=False).mean()
    out['ema200_1d'] = daily.ewm(span=200, adjust=False).mean()
    out['close_1d'] = daily
    return out.shift(1).dropna()


def merge_daily_to_4h(df_4h: pd.DataFrame, df_daily: pd.DataFrame) -> pd.DataFrame:
    df_d = df_daily.copy()
    if df_d.index.tz is None:
        df_d.index = df_d.index.tz_localize('UTC')
    out = df_4h.copy()
    for col in df_d.columns:
        out[col] = df_d[col].reindex(df_4h.index, method='ffill')
    return out


def prepare(df_4h: pd.DataFrame) -> pd.DataFrame:
    """Full prep: 4h features + daily regime info merged."""
    df = compute_features(df_4h)
    macro = compute_macro_daily(df_4h)
    df = merge_daily_to_4h(df, macro)
    return df.dropna(subset=['ema200', 'ema200_1d', 'rsi14', 'atr14']).copy()


# -----------------------------------------------------------------------------
# REGIME DETECTION (daily EMAs, shifted -> no look-ahead)
# -----------------------------------------------------------------------------
def detect_regime(df: pd.DataFrame, idx: int, params: dict = PARAMS) -> str:
    """
    BULL: EMA20>EMA50 by more than dead_zone (and ideally EMA50>=EMA200)
    BEAR: EMA20<EMA50 by more than dead_zone (and close<EMA200 to confirm)
    RANGE: otherwise
    """
    row = df.iloc[idx]
    e20 = row.get('ema20_1d', np.nan)
    e50 = row.get('ema50_1d', np.nan)
    e200 = row.get('ema200_1d', np.nan)
    close_1d = row.get('close_1d', np.nan)
    if not np.isfinite(e20) or not np.isfinite(e50):
        return 'RANGE'
    dist = (e20 - e50) / e50 if e50 > 0 else 0.0
    dz = params.get('regime_dead_zone', 0.02)
    if dist > dz:
        return 'BULL'
    if dist < -dz:
        # demote BEAR -> RANGE if still above EMA200 (recovery zone)
        if params.get('regime_use_ema200_filter', True):
            if np.isfinite(e200) and np.isfinite(close_1d) and close_1d > e200:
                return 'RANGE'
        return 'BEAR'
    return 'RANGE'


# -----------------------------------------------------------------------------
# SIGNAL ROUTERS
# -----------------------------------------------------------------------------
def _signal_bull(df: pd.DataFrame, idx: int, p: dict):
    """Pullback to EMA20 in an uptrend. Returns 'LONG' or None."""
    row = df.iloc[idx]
    c = float(row['close'])
    e20 = float(row['ema20'])
    e50 = float(row['ema50'])
    rsi = float(row['rsi14'])
    atr_p = float(row['atr_pct'])
    if not (np.isfinite(e20) and np.isfinite(e50) and np.isfinite(rsi)):
        return None
    if p.get('bull_pullback_require_above_ema50', True) and not (e20 > e50):
        return None
    # daily alignment filter: ema50_1d > ema200_1d
    if p.get('bull_pullback_require_daily_alignment', True):
        e50_d = row.get('ema50_1d', np.nan)
        e200_d = row.get('ema200_1d', np.nan)
        if not (np.isfinite(e50_d) and np.isfinite(e200_d) and e50_d > e200_d):
            return None
    dist_to_ema20 = abs(c - e20) / e20 if e20 > 0 else 1.0
    if dist_to_ema20 > p['bull_pullback_ema20_max_dist']:
        return None
    if not (p['bull_pullback_rsi_min'] <= rsi <= p['bull_pullback_rsi_max']):
        return None
    if atr_p < p['bull_pullback_min_atr_pct']:
        return None
    # require ema20 itself trending up (slope over 5 bars)
    slope = float(row.get('ema20_slope_5', 0.0))
    if not np.isfinite(slope) or slope <= 0:
        return None
    # require bullish candle on entry (dip is being bought)
    if p.get('bull_pullback_require_bullish_entry', True):
        if float(row['close']) <= float(row['open']):
            return None
    return 'LONG'


def _signal_range(df: pd.DataFrame, idx: int, p: dict):
    """Deep oversold mean reversion. Returns 'LONG' or None."""
    row = df.iloc[idx]
    rsi = float(row['rsi14'])
    bb_pct = row.get('bb_pct', np.nan)
    atr_p = float(row['atr_pct'])
    if rsi > p['range_rsi_max']:
        return None
    if not np.isfinite(bb_pct) or bb_pct > p['range_bb_pct_max']:
        return None
    if atr_p < p['range_min_atr_pct']:
        return None
    if p.get('range_require_bullish_candle', True):
        # close > open on entry bar — confirmation we're not catching a falling knife
        if float(row['close']) <= float(row['open']):
            return None
    return 'LONG'


def signal(df: pd.DataFrame, idx: int, params: dict = PARAMS):
    """
    Route the signal based on regime.
    Returns ('LONG'|'SHORT'|None, regime).
    """
    regime = detect_regime(df, idx, params)
    if regime == 'BULL':
        s = _signal_bull(df, idx, params)
        return s, regime
    if regime == 'RANGE':
        s = _signal_range(df, idx, params)
        return s, regime
    # BEAR -> stay flat
    if params.get('bear_active', False):
        # placeholder for future BEAR sub-strategy. Default: disabled.
        return None, regime
    return None, regime


# -----------------------------------------------------------------------------
# SIMULATORS (one position at a time; NO intrabar look-ahead in trailing)
# -----------------------------------------------------------------------------
def _sim_long_trailing(df, entry_bar, entry_price, trail_dist, max_bars, commission):
    """
    LONG trailing stop, honest:
      1. Check exit against the stop that was already set BEFORE this bar.
      2. Only AFTER that, update peak/stop using THIS bar's high (for the next bar).
    """
    sl_price = entry_price * (1 - trail_dist)
    peak = entry_price
    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df):
            exit_p = float(df['close'].iloc[-1])
            pnl = (exit_p - entry_price) / entry_price - 2 * commission
            return ('TP' if exit_p > entry_price else 'SL'), pnl, i, exit_p
        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        # 1) exit vs. previously known stop
        if lo <= sl_price:
            exit_p = sl_price
            pnl = (exit_p - entry_price) / entry_price - 2 * commission
            return ('TP' if exit_p > entry_price else 'SL'), pnl, i, exit_p
        # 2) update peak/stop AFTER for next bar
        if hi > peak:
            peak = hi
        sl_price = max(sl_price, peak * (1 - trail_dist))
    exit_p = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (exit_p - entry_price) / entry_price - 2 * commission
    return ('TP' if exit_p > entry_price else 'SL'), pnl, max_bars, exit_p


def _sim_fixed_tp_sl_long(df, entry_bar, entry_price, tp_pct, sl_pct, max_bars, commission):
    """LONG fixed TP/SL, conservative: if both touched same bar, assume SL."""
    tp = entry_price * (1 + tp_pct)
    sl = entry_price * (1 - sl_pct)
    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df):
            exit_p = float(df['close'].iloc[-1])
            pnl = (exit_p - entry_price) / entry_price - 2 * commission
            return ('TP' if exit_p > entry_price else 'SL'), pnl, i, exit_p
        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        # conservative: SL checked first; if both in same bar, SL wins
        if lo <= sl:
            pnl = -sl_pct - 2 * commission
            return 'SL', pnl, i, sl
        if hi >= tp:
            pnl = tp_pct - 2 * commission
            return 'TP', pnl, i, tp
    exit_p = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (exit_p - entry_price) / entry_price - 2 * commission
    return ('TP' if exit_p > entry_price else 'SL'), pnl, max_bars, exit_p


def simulate(df: pd.DataFrame, entry_bar: int, params: dict, regime: str) -> dict:
    """
    Simulate the trade according to the regime that produced the signal.
    Returns dict: outcome, pnl_pct, bars, exit_price, side, regime.
    """
    p = params
    entry = float(df['close'].iloc[entry_bar])
    atr = float(df['atr14'].iloc[entry_bar])
    atr_pct = atr / entry if entry > 0 else 0.0
    commission = p.get('commission_one_way', 0.0005)

    if regime == 'BULL':
        trail_dist = max(p['bull_trail_floor'], atr_pct * p['bull_trail_atr_mult'])
        outcome, pnl, bars, exit_p = _sim_long_trailing(
            df, entry_bar, entry, trail_dist, p['bull_max_bars'], commission)
        side = 'LONG'
    elif regime == 'RANGE':
        tp_pct = atr_pct * p['range_tp_atr_mult']
        sl_pct = atr_pct * p['range_sl_atr_mult']
        # caps to avoid degenerate sizing
        tp_pct = min(max(tp_pct, 0.008), 0.06)
        sl_pct = min(max(sl_pct, 0.006), 0.05)
        outcome, pnl, bars, exit_p = _sim_fixed_tp_sl_long(
            df, entry_bar, entry, tp_pct, sl_pct, p['range_max_bars'], commission)
        side = 'LONG'
    else:
        # BEAR / unknown -> shouldn't be called by the engine
        return {'outcome': 'NONE', 'pnl_pct': 0.0, 'bars': 0,
                'exit_price': entry, 'side': None, 'regime': regime}

    return {'outcome': outcome, 'pnl_pct': pnl, 'bars': bars,
            'exit_price': exit_p, 'side': side, 'regime': regime,
            'entry_price': entry, 'ts': df.index[entry_bar]}


# -----------------------------------------------------------------------------
# ENGINE: one-position-at-a-time. Same convention used by revalidate_v15.py
# -----------------------------------------------------------------------------
def run_engine(df: pd.DataFrame, params: dict, start_i: int, end_i: int) -> list:
    """
    Walks bars [start_i, end_i). On a signal, opens ONE trade, simulates until
    it closes, then jumps past the close bar. Never overlaps positions.
    Returns list of trade dicts.
    """
    trades = []
    i = max(start_i, params.get('min_history_bars', 220))
    end_i = min(end_i, len(df) - 2)
    while i < end_i:
        s, regime = signal(df, i, params)
        if s is None:
            i += 1
            continue
        t = simulate(df, i, params, regime)
        if t['outcome'] == 'NONE':
            i += 1
            continue
        trades.append(t)
        # jump to after the exit bar -> never overlap
        i += int(t['bars']) + 1
    return trades
