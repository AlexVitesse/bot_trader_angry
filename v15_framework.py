"""
v15_framework.py
================
Framework compartido para las 4 estrategias V15.
Cada rama importa este archivo o lo copia localmente.

Diferencias clave vs V14:
- V14: COUNTERTREND (RSI<40, oversold, support bounce) → PF=0.86 OOS
- V15: TREND FOLLOWING + SENTIMIENTO (entrar EN la tendencia, no contra ella)
- Walk-forward: 12 semestres 2020-2025 (rule-based, sin training)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pandas_ta as pta
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent
DATA = ROOT / 'data'

# 12 semestres para walk-forward (2020-H1 a 2025-H2)
WF_FOLDS = [
    ('2020-01-01', '2020-06-30'),
    ('2020-07-01', '2020-12-31'),
    ('2021-01-01', '2021-06-30'),
    ('2021-07-01', '2021-12-31'),
    ('2022-01-01', '2022-06-30'),
    ('2022-07-01', '2022-12-31'),
    ('2023-01-01', '2023-06-30'),
    ('2023-07-01', '2023-12-31'),
    ('2024-01-01', '2024-06-30'),
    ('2024-07-01', '2024-12-31'),
    ('2025-01-01', '2025-06-30'),
    ('2025-07-01', '2025-12-31'),
]
OOS_START = '2022-01-01'
OOS_END   = '2026-01-31'
COMMISSION = 0.0005  # 0.05% por lado → 0.1% round trip


# ============================================================
# CARGA DE DATOS
# ============================================================
def load_btc_4h() -> pd.DataFrame:
    """OHLCV 4H desde CSV (2018-2026), SIN taker vol."""
    df = pd.read_csv(DATA / 'BTCUSDT_4h.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    return df.set_index('timestamp').sort_index()


def load_btc_4h_v15() -> pd.DataFrame:
    """OHLCV 4H con taker_buy_vol (2019-2026)."""
    df = pd.read_parquet(DATA / 'btcusdt_4h_v15.parquet')
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df.sort_index()


def load_btc_1d() -> pd.DataFrame:
    """OHLCV diario (2019-2026)."""
    df = pd.read_parquet(DATA / 'btcusdt_1d_v15.parquet')
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df.sort_index()


def load_funding() -> pd.Series:
    """Funding rate 8H, resampled a 4H con ffill."""
    f = pd.read_parquet(DATA / 'btc_v15_funding.parquet')
    if f.index.tz is None:
        f.index = f.index.tz_localize('UTC')
    return f['funding_rate'].resample('4h').ffill()


def load_fng() -> pd.Series:
    """Fear & Greed diario, resampled a 4H con ffill."""
    f = pd.read_parquet(DATA / 'fear_greed_history.parquet')
    if f.index.tz is None:
        f.index = f.index.tz_localize('UTC')
    return f['fng_value'].resample('4h').ffill()


# ============================================================
# FEATURES 4H
# ============================================================
def compute_features_4h(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    h, l, c, v = df['high'], df['low'], df['close'], df['volume']

    # EMAs
    for n in [20, 50, 200]:
        df[f'ema{n}'] = pta.ema(c, length=n)
    df['ema20_slope'] = df['ema20'].pct_change(5) * 100
    df['ema50_slope'] = df['ema50'].pct_change(10) * 100
    df['ema200_dist'] = (c - df['ema200']) / df['ema200'] * 100

    # RSI
    df['rsi14'] = pta.rsi(c, length=14)

    # ATR
    atr = pta.atr(h, l, c, length=14)
    df['atr14']  = atr
    df['atr_pct'] = atr / c * 100

    # Bollinger Bands
    bb = pta.bbands(c, length=20)
    if bb is not None:
        bb_low, bb_mid, bb_up = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
        df['bb_pct']   = (c - bb_low) / (bb_up - bb_low).replace(0, np.nan)
        df['bb_width'] = (bb_up - bb_low) / bb_mid * 100
    else:
        df['bb_pct'] = 0.5; df['bb_width'] = 5.0

    # ADX
    adx_df = pta.adx(h, l, c, length=14)
    if adx_df is not None:
        df['adx14']   = adx_df.iloc[:, 0]
        df['di_plus'] = adx_df.iloc[:, 1]
        df['di_minus']= adx_df.iloc[:, 2]
        df['di_diff'] = df['di_plus'] - df['di_minus']
    else:
        df['adx14'] = 20.0; df['di_diff'] = 0.0

    # Volume ratio
    vol_ma = v.rolling(20).mean()
    df['vol_ratio'] = v / vol_ma.replace(0, np.nan)

    # Rolling high/low (20 bars)
    df['high20'] = h.rolling(20).max().shift(1)  # shift 1: sin look-ahead
    df['low20']  = l.rolling(20).min().shift(1)
    df['range_pos'] = (c - df['low20']) / (df['high20'] - df['low20']).replace(0, np.nan)

    # Returns
    df['ret_1'] = c.pct_change(1) * 100
    df['ret_5'] = c.pct_change(5) * 100

    return df.dropna(subset=['ema20', 'ema50', 'ema200', 'rsi14', 'atr14', 'adx14'])


def compute_macro_daily(df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Resamplea 4H a diario, calcula EMAs diarias.
    Shift 1 dia para evitar look-ahead.
    """
    daily = df_4h['close'].resample('1D').last().dropna()
    out = pd.DataFrame(index=daily.index)
    out['ema20_1d']  = daily.ewm(span=20, adjust=False).mean()
    out['ema50_1d']  = daily.ewm(span=50, adjust=False).mean()
    out['ema200_1d'] = daily.ewm(span=200, adjust=False).mean()
    out['bull_1d']   = (out['ema20_1d'] > out['ema50_1d']).astype(int)
    # Shift 1 dia (hoy usa info de ayer)
    return out.shift(1).dropna()


def merge_daily_to_4h(df_4h: pd.DataFrame, df_daily: pd.DataFrame) -> pd.DataFrame:
    """Merge macro diario a cada barra 4H con asof (forward fill)."""
    df_daily_tz = df_daily.copy()
    if df_daily_tz.index.tz is None:
        df_daily_tz.index = df_daily_tz.index.tz_localize('UTC')
    result = df_4h.copy()
    for col in df_daily_tz.columns:
        result[col] = df_daily_tz[col].reindex(df_4h.index, method='ffill')
    return result


# ============================================================
# SIMULACION
# ============================================================
def sim_trade_fixed(df: pd.DataFrame, entry_bar: int, entry_price: float,
                    tp_pct: float, sl_pct: float,
                    max_bars: int = 18) -> tuple:
    """
    Trade con TP y SL fijos.
    Retorna (outcome, exit_price, pnl_pct, bars)
    outcome: 'TP' | 'SL' | 'TIMEOUT'
    pnl_pct: % ganancia/perdida (sin comision)
    """
    tp = entry_price * (1 + tp_pct)
    sl = entry_price * (1 - sl_pct)
    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df):
            ep = float(df['close'].iloc[-1])
            pnl = (ep - entry_price) / entry_price - 2 * COMMISSION
            return ('TP' if ep > entry_price else 'SL'), ep, pnl, i
        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        if lo <= sl:
            pnl = -sl_pct - 2 * COMMISSION
            if hi >= tp and df['close'].iloc[b] > (sl + tp) / 2:
                pnl = tp_pct - 2 * COMMISSION
                return 'TP', tp, pnl, i
            return 'SL', sl, pnl, i
        if hi >= tp:
            pnl = tp_pct - 2 * COMMISSION
            return 'TP', tp, pnl, i
    ep  = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (ep - entry_price) / entry_price - 2 * COMMISSION
    return ('TP' if ep > entry_price else 'SL'), ep, pnl, max_bars


def sim_trade_atr(df: pd.DataFrame, entry_bar: int, entry_price: float,
                  atr: float, tp_mult: float = 3.0, sl_mult: float = 1.5,
                  max_bars: int = 18) -> tuple:
    """Trade con TP y SL basados en ATR."""
    tp_pct = atr * tp_mult / entry_price
    sl_pct = atr * sl_mult / entry_price
    # Cap SL at 5% to avoid extreme sizing
    sl_pct = min(sl_pct, 0.05)
    tp_pct = min(tp_pct, 0.15)
    return sim_trade_fixed(df, entry_bar, entry_price, tp_pct, sl_pct, max_bars)


# ============================================================
# METRICAS
# ============================================================
def metrics(trades: list, label: str = '') -> dict:
    if not trades:
        return {'label': label, 'n': 0, 'wr': 0, 'pf': 0,
                'avg_pnl': 0, 'trades_pm': 0, 'annual_pct': 0,
                'ok': False}
    n     = len(trades)
    wins  = [t for t in trades if t['outcome'] == 'TP']
    losses= [t for t in trades if t['outcome'] == 'SL']
    wr    = len(wins) / n

    gross_win  = sum(t['pnl_pct'] for t in wins)
    gross_loss = sum(abs(t['pnl_pct']) for t in losses)
    pf         = gross_win / gross_loss if gross_loss > 0 else float('inf')

    avg_pnl = sum(t['pnl_pct'] for t in trades) / n

    # Periodo: de primer a ultimo trade
    if trades:
        t0 = pd.to_datetime(trades[0]['ts'])
        t1 = pd.to_datetime(trades[-1]['ts'])
        months = max(1, (t1 - t0).days / 30)
    else:
        months = 1
    trades_pm = n / months

    # Retorno anual estimado (2% riesgo por trade como proxy)
    annual_pct = avg_pnl * trades_pm * 12 * (1 / 0.02) * 2

    ok = (wr >= 0.40 and pf >= 1.2 and n >= 3)

    return {
        'label': label, 'n': n, 'wr': wr, 'pf': pf,
        'avg_pnl': avg_pnl, 'trades_pm': trades_pm,
        'annual_pct': annual_pct, 'ok': ok,
    }


def print_metrics(m: dict, baseline: dict = None):
    mark = '[+]' if m['ok'] else '[-]'
    wr_s = f"{m['wr']:.1%}"
    pf_s = f"{m['pf']:.2f}"
    if baseline:
        wr_s += f" ({m['wr']-baseline['wr']:+.1%})"
        pf_s += f" ({m['pf']-baseline['pf']:+.2f})"
    print(f"  {mark} {m['label']:<40} | N={m['n']:>4} | "
          f"WR={wr_s:<14} | PF={pf_s:<10} | "
          f"~{m['annual_pct']:.0f}%/yr | {m['trades_pm']:.1f}t/m")


# ============================================================
# WALK-FORWARD (rule-based: 12 semestres)
# ============================================================
def walk_forward(df: pd.DataFrame, detect_fn, simulate_fn,
                 min_trades: int = 3) -> dict:
    """
    df: DataFrame con features ya calculadas (index = timestamp UTC)
    detect_fn(df, i) -> trade_dict | None
    simulate_fn(df, i, trade_dict) -> (outcome, exit_price, pnl_pct, bars)
    """
    results = []
    for start, end in WF_FOLDS:
        mask  = (df.index >= start) & (df.index <= end)
        df_f  = df[mask]
        if len(df_f) < 100:
            results.append({'period': f'{start[:7]}/{end[:7]}',
                            'ok': False, 'n': 0, 'wr': 0, 'pf': 0, 'note': 'sin datos'})
            continue

        trades = []
        in_trade = False
        for i in range(len(df_f)):
            if in_trade:
                continue
            trade_def = detect_fn(df_f, i)
            if trade_def is None:
                continue
            global_i = df.index.get_loc(df_f.index[i])
            out = simulate_fn(df, global_i, trade_def)
            trades.append({
                'outcome': out[0], 'exit_price': out[1],
                'pnl_pct': out[2], 'bars': out[3],
                'ts': df_f.index[i],
            })
        m = metrics(trades, f'{start[:7]}/{end[:7]}')
        results.append({
            'period': f'{start[:7]}/{end[:7]}',
            'ok': m['ok'] and m['n'] >= min_trades,
            'n': m['n'], 'wr': m['wr'], 'pf': m['pf'],
            'annual_pct': m['annual_pct'], 'note': '',
        })

    folds_ok = sum(1 for r in results if r['ok'])
    return {'folds': results, 'folds_ok': folds_ok,
            'folds_total': len(results),
            'approved': folds_ok >= 7}
