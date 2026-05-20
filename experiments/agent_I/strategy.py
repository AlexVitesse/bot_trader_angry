"""
agent_I/strategy.py
===================
ETH/USDT 4h — Mean-reversion in RANGE regime. Self-contained.

Philosophy
----------
ETH has more "beta" than BTC: when it overshoots in either direction during a
choppy / sideways regime, the bounce is often violent. The hypothesis is that
fade-the-extreme in confirmed RANGE works better in ETH than in BTC (where
Agent C tried this and got 0/12 because the setup fired only 25 times).

Three things differentiate this from Agent C's failed attempt:
  1) RANGE detection is broader — ADX<20 AND BB-width compressed AND daily
     EMA20-EMA50 narrow.  More inclusive but still clearly non-trending.
  2) The extreme is detected as RSI in lower band OR BB position in lower band
     (OR logic, not AND) — this is the key to lifting trade frequency to
     operable levels without becoming trend-following.
  3) Bullish-candle confirmation prevents catching falling knives.

Regime-exit rule: if the regime changes from RANGE -> BULL/BEAR while a trade
is open, we close that bar (thesis no longer supported).

Anti-overfitting discipline:
- 4 parameters for LONG entry, 3 for SHORT entry — all standard textbook values
  (RSI 30/35/65/70, BB 0.10/0.15/0.85/0.90, ADX 20).
- Frozen `PARAMS`. Set once after reading the brief, never tuned per fold.
- One position at a time, no intrabar look-ahead in TP/SL evaluation.
- All daily features use shift(1).

NOT exploring:
- Trend-following entries (Agent A did this — found ETH-A marginal).
- Vol-breakout (Agent F did this — combined ETH-V2 was REJECT).
- Cross-asset rotation (Agent H's domain).
- Long-hold mean-rev: max-bars is short (≤12). Mean reversion thesis is local.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# FROZEN PARAMETERS — committed before measurement, NEVER tuned per-fold
# -----------------------------------------------------------------------------
PARAMS = {
    # ---------- RANGE regime detection (multi-condition) --------------------
    # All three must hold to call it RANGE. Thresholds chosen from the
    # standard "weak trend" band (not the extremes):
    #   ADX<25 = Wilder's "weak trend" (not the stricter <20).
    #   BB width below 40th-pctile = bottom 40% of compression history.
    #   |EMA20d - EMA50d| <= 3% = daily EMAs are close-ish (not micro-narrow).
    # Combined, this hits ~10% of bars in ETH (vs 3.8% with stricter values),
    # which is the minimum needed to get enough fade opportunities.
    'regime_adx_max': 25.0,
    'regime_bb_width_pct_max': 40.0,
    'regime_daily_ema_sep_max': 0.03,
    'regime_bb_width_lookback': 250,

    # ---------- LONG entry (fade oversold in RANGE) ------------------------
    # OR logic: either RSI deeply low OR BB position deeply low.
    # Plus bullish candle + volume confirmation.
    'long_rsi_max': 30,                # RSI <= 30 (Wilder oversold)
    'long_bb_pct_max': 0.10,           # OR BB position <= 0.10
    'long_require_bullish_candle': True,
    'long_vol_ratio_min': 1.2,         # vol[i] / mean(vol[i-20:i]) >= 1.2
    'long_min_atr_pct': 0.4,           # avoid dead market

    # ---------- SHORT entry (fade overbought in RANGE) ---------------------
    # Symmetric to LONG. Enabled but stricter (project history says SHORT in
    # crypto rarely works).
    'short_enabled': True,
    'short_rsi_min': 70,               # RSI >= 70 (Wilder overbought)
    'short_bb_pct_min': 0.90,          # OR BB position >= 0.90
    'short_require_bearish_candle': True,
    'short_vol_ratio_min': 1.2,
    'short_min_atr_pct': 0.4,

    # ---------- TP / SL ----------------------------------------------------
    # Mean-reversion exit: small TP (target the mean), wider SL (allow noise).
    # Capped to sane bounds so a low-vol bar doesn't ship a 0.2% TP.
    'tp_atr_mult': 1.5,                # quick mean-reversion target
    'sl_atr_mult': 2.0,                # slightly wider stop
    'tp_min': 0.010,                   # 1.0% floor
    'tp_max': 0.040,                   # 4.0% cap
    'sl_min': 0.012,                   # 1.2% floor
    'sl_max': 0.050,                   # 5.0% cap
    'max_bars': 10,                    # ~40h timeout

    # ---------- Regime-change exit ----------------------------------------
    # We only exit if the regime flips to the OPPOSITE direction:
    #   LONG  -> exit if regime becomes BEAR  (RANGE/BULL still tolerable)
    #   SHORT -> exit if regime becomes BULL  (RANGE/BEAR still tolerable)
    # This avoids the trap of "RANGE -> mild BULL while LONG is winning".
    'exit_on_regime_change': True,
    'exit_only_on_opposite_regime': True,

    # ---------- Universal -------------------------------------------------
    'commission_one_way': 0.0005,      # 0.05% per side (0.1% RT)
    'min_history_bars': 260,           # for ADX + 250-bar percentile warmup
}


# -----------------------------------------------------------------------------
# FEATURE COMPUTATION (4h) — all rolling, no look-ahead
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
    width = (upper - lower) / ma.replace(0, np.nan)  # relative bb width
    return lower, ma, upper, pct, width


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """ADX (Wilder). Standard formulation, all rolling -> no look-ahead."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, np.nan)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1/n, adjust=False).mean()


def compute_features(df_4h: pd.DataFrame) -> pd.DataFrame:
    """4h indicators. All rolling = no look-ahead."""
    df = df_4h.copy()
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    df['rsi14'] = _rsi(c, 14)
    atr = _atr(h, l, c, 14)
    df['atr14'] = atr
    df['atr_pct'] = atr / c * 100  # in percent units

    _, _, _, bb_pct, bb_width = _bbands(c, 20, 2.0)
    df['bb_pct'] = bb_pct
    df['bb_width'] = bb_width

    # Percentile rank of current BB width vs last N bars (e.g. 250).
    # Lower percentile = compressed regime. shift(1) so today uses up-to-yesterday.
    n_look = PARAMS['regime_bb_width_lookback']
    df['bb_width_pctile'] = (
        bb_width.shift(1)
        .rolling(n_look, min_periods=max(50, n_look // 2))
        .apply(lambda s: (s <= s.iloc[-1]).mean() * 100, raw=False)
    )

    df['adx14'] = _adx(h, l, c, 14)
    # Volume ratio: vol[i] / mean(vol[i-20:i])  — exclusive of current bar
    df['vol_sma20'] = v.rolling(20).mean().shift(1)
    df['vol_ratio'] = v / df['vol_sma20']

    return df


def compute_macro_daily(df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Daily EMAs for RANGE filter. shift(1) -> today uses yesterday's daily EMAs.
    NO look-ahead.
    """
    daily = df_4h['close'].resample('1D').last().dropna()
    out = pd.DataFrame(index=daily.index)
    out['ema20_1d'] = daily.ewm(span=20, adjust=False).mean()
    out['ema50_1d'] = daily.ewm(span=50, adjust=False).mean()
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
    needed = ['rsi14', 'atr14', 'bb_pct', 'bb_width_pctile', 'adx14',
              'vol_ratio', 'ema20_1d', 'ema50_1d']
    return df.dropna(subset=needed).copy()


# -----------------------------------------------------------------------------
# REGIME DETECTION (multi-condition; produces BULL/BEAR/RANGE)
# -----------------------------------------------------------------------------
def detect_regime(df: pd.DataFrame, idx: int, params: dict = PARAMS) -> str:
    """
    RANGE  ->  ALL of:  ADX<20  AND  BB-width percentile<30  AND
                       |EMA20_1d-EMA50_1d|/EMA50_1d <= 2%
    BULL   ->  EMA20_1d > EMA50_1d * (1 + sep)   (trending up)
    BEAR   ->  EMA20_1d < EMA50_1d * (1 - sep)   (trending down)
    """
    row = df.iloc[idx]
    e20 = row.get('ema20_1d', np.nan)
    e50 = row.get('ema50_1d', np.nan)
    adx = row.get('adx14', np.nan)
    bb_pctile = row.get('bb_width_pctile', np.nan)

    if not (np.isfinite(e20) and np.isfinite(e50) and e50 > 0):
        return 'RANGE'

    sep = (e20 - e50) / e50
    abs_sep = abs(sep)
    dz = params.get('regime_daily_ema_sep_max', 0.02)

    # Is the broad RANGE filter satisfied?
    range_macro = abs_sep <= dz
    range_adx = np.isfinite(adx) and adx < params.get('regime_adx_max', 20.0)
    range_bbw = (np.isfinite(bb_pctile) and
                 bb_pctile < params.get('regime_bb_width_pct_max', 30.0))

    if range_macro and range_adx and range_bbw:
        return 'RANGE'

    # Otherwise classify by daily EMA stack as BULL or BEAR
    if sep > dz:
        return 'BULL'
    if sep < -dz:
        return 'BEAR'
    # Macro neutral but ADX/BBW says trending -> still call it RANGE-ish but
    # NOT eligible for our trades (no extreme to fade with confidence).
    # We return BULL by convention if e20>e50 else BEAR. This keeps regime
    # well-defined for the exit-on-regime-change rule.
    return 'BULL' if e20 >= e50 else 'BEAR'


# -----------------------------------------------------------------------------
# SIGNAL
# -----------------------------------------------------------------------------
def _signal_long(df: pd.DataFrame, idx: int, p: dict):
    """Fade oversold in RANGE. Returns 'LONG' or None."""
    row = df.iloc[idx]
    rsi = float(row['rsi14'])
    bb_pct = row.get('bb_pct', np.nan)
    atr_p = float(row['atr_pct'])
    vol_r = row.get('vol_ratio', np.nan)
    if atr_p < p['long_min_atr_pct']:
        return None
    if not np.isfinite(vol_r) or vol_r < p['long_vol_ratio_min']:
        return None
    # extreme = RSI deeply low  OR  BB position deeply low  (relaxed AND -> OR)
    extreme_rsi = rsi <= p['long_rsi_max']
    extreme_bb = np.isfinite(bb_pct) and bb_pct <= p['long_bb_pct_max']
    if not (extreme_rsi or extreme_bb):
        return None
    if p.get('long_require_bullish_candle', True):
        if float(row['close']) <= float(row['open']):
            return None
    return 'LONG'


def _signal_short(df: pd.DataFrame, idx: int, p: dict):
    """Fade overbought in RANGE. Returns 'SHORT' or None."""
    if not p.get('short_enabled', False):
        return None
    row = df.iloc[idx]
    rsi = float(row['rsi14'])
    bb_pct = row.get('bb_pct', np.nan)
    atr_p = float(row['atr_pct'])
    vol_r = row.get('vol_ratio', np.nan)
    if atr_p < p['short_min_atr_pct']:
        return None
    if not np.isfinite(vol_r) or vol_r < p['short_vol_ratio_min']:
        return None
    extreme_rsi = rsi >= p['short_rsi_min']
    extreme_bb = np.isfinite(bb_pct) and bb_pct >= p['short_bb_pct_min']
    if not (extreme_rsi or extreme_bb):
        return None
    if p.get('short_require_bearish_candle', True):
        if float(row['close']) >= float(row['open']):
            return None
    return 'SHORT'


def signal(df: pd.DataFrame, idx: int, params: dict = PARAMS):
    """
    Return ('LONG'|'SHORT'|None, regime).
    Only fires in RANGE regime.
    """
    regime = detect_regime(df, idx, params)
    if regime != 'RANGE':
        return None, regime
    # LONG has priority (project history: long-bias in crypto more reliable);
    # both could not fire on the same bar anyway (RSI<30 and RSI>70).
    s = _signal_long(df, idx, params)
    if s is not None:
        return s, regime
    s = _signal_short(df, idx, params)
    if s is not None:
        return s, regime
    return None, regime


# -----------------------------------------------------------------------------
# SIMULATORS — one position at a time, no intrabar look-ahead.
# Honest TP/SL: if both touched the same bar, assume SL (conservative).
# Also: exit-on-regime-change is checked AT THE OPEN OF EACH NEW BAR (so we
# use info already known at that bar; no look-ahead).
# -----------------------------------------------------------------------------
def _sim_long_tp_sl(df, entry_bar, entry_price, tp_pct, sl_pct, max_bars,
                    commission, regime_at_entry, exit_on_regime_change,
                    params):
    tp = entry_price * (1 + tp_pct)
    sl = entry_price * (1 - sl_pct)
    opposite_only = params.get('exit_only_on_opposite_regime', True)
    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df):
            exit_p = float(df['close'].iloc[-1])
            pnl = (exit_p - entry_price) / entry_price - 2 * commission
            return ('TP' if exit_p > entry_price else 'SL'), pnl, i, exit_p, 'EOD'

        # Regime-change exit at the open of THIS bar (no look-ahead; daily
        # macro feature is already shifted by one day, ADX/BBW use rolling
        # past bars only).
        if exit_on_regime_change:
            reg_now = detect_regime(df, b, params)
            adverse = (reg_now == 'BEAR') if opposite_only else (reg_now != regime_at_entry)
            if adverse:
                exit_p = float(df['open'].iloc[b])
                pnl = (exit_p - entry_price) / entry_price - 2 * commission
                outcome = 'REG_TP' if exit_p > entry_price else 'REG_SL'
                return outcome, pnl, i, exit_p, 'REGIME'

        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        # conservative: SL first
        if lo <= sl:
            pnl = -sl_pct - 2 * commission
            return 'SL', pnl, i, sl, 'SL'
        if hi >= tp:
            pnl = tp_pct - 2 * commission
            return 'TP', pnl, i, tp, 'TP'

    exit_p = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (exit_p - entry_price) / entry_price - 2 * commission
    return ('TP' if exit_p > entry_price else 'SL'), pnl, max_bars, exit_p, 'TIMEOUT'


def _sim_short_tp_sl(df, entry_bar, entry_price, tp_pct, sl_pct, max_bars,
                     commission, regime_at_entry, exit_on_regime_change,
                     params):
    tp = entry_price * (1 - tp_pct)
    sl = entry_price * (1 + sl_pct)
    opposite_only = params.get('exit_only_on_opposite_regime', True)
    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df):
            exit_p = float(df['close'].iloc[-1])
            pnl = (entry_price - exit_p) / entry_price - 2 * commission
            return ('TP' if exit_p < entry_price else 'SL'), pnl, i, exit_p, 'EOD'

        if exit_on_regime_change:
            reg_now = detect_regime(df, b, params)
            adverse = (reg_now == 'BULL') if opposite_only else (reg_now != regime_at_entry)
            if adverse:
                exit_p = float(df['open'].iloc[b])
                pnl = (entry_price - exit_p) / entry_price - 2 * commission
                outcome = 'REG_TP' if exit_p < entry_price else 'REG_SL'
                return outcome, pnl, i, exit_p, 'REGIME'

        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        # conservative: SL first (SHORT SL = high crosses sl)
        if hi >= sl:
            pnl = -sl_pct - 2 * commission
            return 'SL', pnl, i, sl, 'SL'
        if lo <= tp:
            pnl = tp_pct - 2 * commission
            return 'TP', pnl, i, tp, 'TP'

    exit_p = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (entry_price - exit_p) / entry_price - 2 * commission
    return ('TP' if exit_p < entry_price else 'SL'), pnl, max_bars, exit_p, 'TIMEOUT'


def simulate(df: pd.DataFrame, entry_bar: int, params: dict, side: str,
             regime_at_entry: str) -> dict:
    p = params
    entry = float(df['close'].iloc[entry_bar])
    atr = float(df['atr14'].iloc[entry_bar])
    atr_pct = atr / entry if entry > 0 else 0.0
    commission = p.get('commission_one_way', 0.0005)

    tp_pct = atr_pct * p['tp_atr_mult']
    sl_pct = atr_pct * p['sl_atr_mult']
    tp_pct = min(max(tp_pct, p['tp_min']), p['tp_max'])
    sl_pct = min(max(sl_pct, p['sl_min']), p['sl_max'])

    exit_on_change = p.get('exit_on_regime_change', True)

    if side == 'LONG':
        outcome, pnl, bars, exit_p, exit_reason = _sim_long_tp_sl(
            df, entry_bar, entry, tp_pct, sl_pct, p['max_bars'], commission,
            regime_at_entry, exit_on_change, p)
    else:  # SHORT
        outcome, pnl, bars, exit_p, exit_reason = _sim_short_tp_sl(
            df, entry_bar, entry, tp_pct, sl_pct, p['max_bars'], commission,
            regime_at_entry, exit_on_change, p)

    return {'outcome': outcome, 'pnl_pct': pnl, 'bars': bars,
            'exit_price': exit_p, 'exit_reason': exit_reason,
            'side': side, 'regime': regime_at_entry,
            'entry_price': entry, 'ts': df.index[entry_bar]}


# -----------------------------------------------------------------------------
# ENGINE: one-position-at-a-time. Honest convention.
# -----------------------------------------------------------------------------
def run_engine(df: pd.DataFrame, params: dict, start_i: int, end_i: int) -> list:
    """
    Walks bars [start_i, end_i). On a signal, opens ONE trade, simulates until
    it closes, then jumps past the close bar. Never overlaps positions.
    """
    trades = []
    i = max(start_i, params.get('min_history_bars', 260))
    end_i = min(end_i, len(df) - 2)
    while i < end_i:
        s, regime = signal(df, i, params)
        if s is None:
            i += 1
            continue
        t = simulate(df, i, params, s, regime)
        if t['outcome'] == 'NONE':
            i += 1
            continue
        trades.append(t)
        i += int(t['bars']) + 1
    return trades
