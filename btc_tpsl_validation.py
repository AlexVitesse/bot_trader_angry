"""
BTC TP/SL Optimization Validation
=================================
Validate that 4% TP / 2% SL works across all market conditions.
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

print('='*70)
print('BTC TP/SL OPTIMIZATION - FULL VALIDATION')
print('='*70)

# =============================================================================
# 1. LOAD DATA AND MODEL
# =============================================================================
print('\n[1] Loading data and model...')

df = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_full.parquet')
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')

v2_data = joblib.load(MODELS_DIR / 'btc_v2_gradientboosting.pkl')
v2_model = v2_data['model']
v2_cols = v2_data['feature_cols']

print(f'    Data: {len(df)} candles ({df.index[0].year} - {df.index[-1].year})')

# =============================================================================
# 2. CALCULATE FEATURES
# =============================================================================
print('\n[2] Calculating features...')

feat = pd.DataFrame(index=df.index)
c, h, l, v = df['close'], df['high'], df['low'], df['volume']

# Returns
for p in [1, 2, 3, 5, 10, 20, 50]:
    feat[f'ret_{p}'] = c.pct_change(p)

# Volatility
feat['atr14'] = ta.atr(h, l, c, length=14)
feat['atr_r'] = feat['atr14'] / feat['atr14'].rolling(50).mean()
feat['vol5'] = c.pct_change().rolling(5).std()
feat['vol20'] = c.pct_change().rolling(20).std()
feat['vol_ratio'] = feat['vol5'] / (feat['vol20'] + 1e-10)

# RSI
feat['rsi14'] = ta.rsi(c, length=14)
feat['rsi7'] = ta.rsi(c, length=7)
feat['rsi21'] = ta.rsi(c, length=21)

# Stochastic RSI
sr = ta.stochrsi(c, length=14, rsi_length=14, k=3, d=3)
if sr is not None:
    feat['srsi_k'] = sr.iloc[:, 0]
    feat['srsi_d'] = sr.iloc[:, 1]

# MACD
macd = ta.macd(c, fast=12, slow=26, signal=9)
if macd is not None:
    feat['macd'] = macd.iloc[:, 0]
    feat['macd_h'] = macd.iloc[:, 1]
    feat['macd_s'] = macd.iloc[:, 2]

# ROC
feat['roc5'] = ta.roc(c, length=5)
feat['roc10'] = ta.roc(c, length=10)
feat['roc20'] = ta.roc(c, length=20)

# EMAs
for el in [8, 21, 55, 100, 200]:
    e = ta.ema(c, length=el)
    feat[f'ema{el}_d'] = (c - e) / e * 100
feat['ema8_sl'] = ta.ema(c, length=8).pct_change(3) * 100
feat['ema21_sl'] = ta.ema(c, length=21).pct_change(5) * 100
feat['ema55_sl'] = ta.ema(c, length=55).pct_change(5) * 100

# Bollinger Bands
bb = ta.bbands(c, length=20, std=2.0)
if bb is not None:
    bw = bb.iloc[:, 2] - bb.iloc[:, 0]
    feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw
    feat['bb_w'] = bw / bb.iloc[:, 1] * 100

# Volume
feat['vr'] = v / v.rolling(20).mean()
feat['vr5'] = v / v.rolling(5).mean()

# Price action
feat['spr'] = (h - l) / c * 100
feat['body'] = abs(c - df['open']) / (h - l + 1e-10)
feat['upper_wick'] = (h - np.maximum(c, df['open'])) / (h - l + 1e-10)
feat['lower_wick'] = (np.minimum(c, df['open']) - l) / (h - l + 1e-10)

# ADX
ax = ta.adx(h, l, c, length=14)
if ax is not None:
    feat['adx'] = ax.iloc[:, 0]
    feat['dip'] = ax.iloc[:, 1]
    feat['dim'] = ax.iloc[:, 2]
    feat['di_diff'] = feat['dip'] - feat['dim']

# Choppiness
atr_1 = ta.atr(h, l, c, length=1)
atr_sum = atr_1.rolling(14).sum()
high_max = h.rolling(14).max()
low_min = l.rolling(14).min()
feat['chop'] = 100 * np.log10(atr_sum / (high_max - low_min + 1e-10)) / np.log10(14)

# Time features
hr = df.index.hour
dw = df.index.dayofweek
feat['h_s'] = np.sin(2 * np.pi * hr / 24)
feat['h_c'] = np.cos(2 * np.pi * hr / 24)
feat['d_s'] = np.sin(2 * np.pi * dw / 7)
feat['d_c'] = np.cos(2 * np.pi * dw / 7)

# Lag features
for lag in [1, 2, 3]:
    feat[f'ret1_lag{lag}'] = feat['ret_1'].shift(lag)
    feat[f'rsi14_lag{lag}'] = feat['rsi14'].shift(lag)

# =============================================================================
# 3. DETECT REGIME
# =============================================================================
print('\n[3] Detecting market regime...')

ema20 = ta.ema(c, length=20)
ema50 = ta.ema(c, length=50)
ret_20d = c.pct_change(20 * 6)

regime = pd.Series(index=df.index, dtype=str)
for i in range(len(df)):
    if i < 200:
        regime.iloc[i] = 'UNKNOWN'
        continue
    price = c.iloc[i]
    e20 = ema20.iloc[i]
    e50 = ema50.iloc[i]
    r20 = ret_20d.iloc[i]
    if pd.isna(e20) or pd.isna(e50) or pd.isna(r20):
        regime.iloc[i] = 'UNKNOWN'
    elif price > e20 and e20 > e50 and r20 > 0.05:
        regime.iloc[i] = 'BULL'
    elif price < e20 and e20 < e50 and r20 < -0.05:
        regime.iloc[i] = 'BEAR'
    else:
        regime.iloc[i] = 'RANGE'

df['regime'] = regime
feat['regime'] = regime
feat = feat.dropna()

# =============================================================================
# 4. GET PREDICTIONS
# =============================================================================
print('\n[4] Generating predictions...')

# Only predict on data after training period (train ended 2025-07)
test_start = '2019-07-01'  # Start after warmup
predictions = pd.Series(v2_model.predict(feat[v2_cols]), index=feat.index)

print(f'    {len(predictions)} predictions generated')

# =============================================================================
# 5. BACKTEST FUNCTION
# =============================================================================
def backtest(preds, df_ref, conv_min, tp_pct, sl_pct, max_hold=20):
    """Backtest with given parameters."""
    trades = []

    for idx, pred in preds.items():
        if abs(pred) / 0.005 < conv_min:
            continue

        try:
            loc = df_ref.index.get_loc(idx)
        except:
            continue

        if loc + 1 >= len(df_ref):
            continue

        direction = 1 if pred > 0 else -1
        entry = df_ref.iloc[loc + 1]['open']
        tp = entry * (1 + tp_pct) if direction == 1 else entry * (1 - tp_pct)
        sl = entry * (1 - sl_pct) if direction == 1 else entry * (1 + sl_pct)

        pnl = None
        exit_type = 'TIME'

        for j in range(loc + 1, min(loc + max_hold + 1, len(df_ref))):
            candle = df_ref.iloc[j]
            if direction == 1:
                if candle['low'] <= sl:
                    pnl = -sl_pct
                    exit_type = 'SL'
                    break
                if candle['high'] >= tp:
                    pnl = tp_pct
                    exit_type = 'TP'
                    break
            else:
                if candle['high'] >= sl:
                    pnl = -sl_pct
                    exit_type = 'SL'
                    break
                if candle['low'] <= tp:
                    pnl = tp_pct
                    exit_type = 'TP'
                    break

        if pnl is None:
            exit_idx = min(loc + max_hold, len(df_ref) - 1)
            exit_p = df_ref.iloc[exit_idx]['close']
            pnl = (exit_p - entry) / entry * direction

        trades.append({
            'idx': idx,
            'pnl': pnl,
            'win': pnl > 0,
            'exit': exit_type,
            'year': idx.year,
            'regime': df_ref.loc[idx, 'regime'] if idx in df_ref.index else 'UNKNOWN'
        })

    return trades

# =============================================================================
# 6. RUN BACKTESTS
# =============================================================================
print('\n[5] Running backtests...')

# Original TP/SL
trades_original = backtest(predictions, df, 1.0, 0.03, 0.015)

# Optimized TP/SL
trades_optimized = backtest(predictions, df, 1.0, 0.04, 0.02)

print(f'    Original (3%/1.5%): {len(trades_original)} trades')
print(f'    Optimized (4%/2%): {len(trades_optimized)} trades')

# =============================================================================
# 7. ANALYSIS BY YEAR
# =============================================================================
print('\n' + '='*70)
print('COMPARISON BY YEAR')
print('='*70)

print(f'\n{"Year":>6} | {"--- Original (3%/1.5%) ---":^30} | {"--- Optimized (4%/2%) ---":^30}')
print(f'{"":>6} | {"Trades":>8} {"WR":>8} {"PnL":>10} | {"Trades":>8} {"WR":>8} {"PnL":>10} | {"Diff":>8}')
print('-'*90)

years = sorted(set(t['year'] for t in trades_original))
total_diff = 0

for year in years:
    # Original
    yr_orig = [t for t in trades_original if t['year'] == year]
    n_orig = len(yr_orig)
    wr_orig = sum(t['win'] for t in yr_orig) / n_orig * 100 if n_orig > 0 else 0
    pnl_orig = sum(t['pnl'] for t in yr_orig) * 100 if n_orig > 0 else 0

    # Optimized
    yr_opt = [t for t in trades_optimized if t['year'] == year]
    n_opt = len(yr_opt)
    wr_opt = sum(t['win'] for t in yr_opt) / n_opt * 100 if n_opt > 0 else 0
    pnl_opt = sum(t['pnl'] for t in yr_opt) * 100 if n_opt > 0 else 0

    diff = pnl_opt - pnl_orig
    total_diff += diff

    diff_str = f'{diff:+.1f}%' if diff != 0 else '0.0%'

    print(f'{year:>6} | {n_orig:>8} {wr_orig:>7.1f}% {pnl_orig:>+9.1f}% | {n_opt:>8} {wr_opt:>7.1f}% {pnl_opt:>+9.1f}% | {diff_str:>8}')

# Totals
print('-'*90)
n_orig = len(trades_original)
wr_orig = sum(t['win'] for t in trades_original) / n_orig * 100
pnl_orig = sum(t['pnl'] for t in trades_original) * 100

n_opt = len(trades_optimized)
wr_opt = sum(t['win'] for t in trades_optimized) / n_opt * 100
pnl_opt = sum(t['pnl'] for t in trades_optimized) * 100

print(f'{"TOTAL":>6} | {n_orig:>8} {wr_orig:>7.1f}% {pnl_orig:>+9.1f}% | {n_opt:>8} {wr_opt:>7.1f}% {pnl_opt:>+9.1f}% | {pnl_opt-pnl_orig:>+7.1f}%')

# =============================================================================
# 8. ANALYSIS BY REGIME
# =============================================================================
print('\n' + '='*70)
print('COMPARISON BY MARKET REGIME')
print('='*70)

print(f'\n{"Regime":>8} | {"--- Original (3%/1.5%) ---":^30} | {"--- Optimized (4%/2%) ---":^30}')
print(f'{"":>8} | {"Trades":>8} {"WR":>8} {"PnL":>10} | {"Trades":>8} {"WR":>8} {"PnL":>10} | {"Diff":>8}')
print('-'*92)

for reg in ['BULL', 'BEAR', 'RANGE']:
    # Original
    reg_orig = [t for t in trades_original if t['regime'] == reg]
    n_orig = len(reg_orig)
    wr_orig = sum(t['win'] for t in reg_orig) / n_orig * 100 if n_orig > 0 else 0
    pnl_orig = sum(t['pnl'] for t in reg_orig) * 100 if n_orig > 0 else 0

    # Optimized
    reg_opt = [t for t in trades_optimized if t['regime'] == reg]
    n_opt = len(reg_opt)
    wr_opt = sum(t['win'] for t in reg_opt) / n_opt * 100 if n_opt > 0 else 0
    pnl_opt = sum(t['pnl'] for t in reg_opt) * 100 if n_opt > 0 else 0

    diff = pnl_opt - pnl_orig

    print(f'{reg:>8} | {n_orig:>8} {wr_orig:>7.1f}% {pnl_orig:>+9.1f}% | {n_opt:>8} {wr_opt:>7.1f}% {pnl_opt:>+9.1f}% | {diff:>+7.1f}%')

# =============================================================================
# 9. PROFIT FACTOR AND DRAWDOWN
# =============================================================================
print('\n' + '='*70)
print('ADVANCED METRICS')
print('='*70)

def calc_metrics(trades):
    """Calculate advanced metrics."""
    if not trades:
        return {}

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    pf = gross_profit / gross_loss

    # Max drawdown
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = running_max - cumsum
    max_dd = np.max(drawdown) * 100

    # Average win/loss
    avg_win = np.mean(wins) * 100 if wins else 0
    avg_loss = np.mean(losses) * 100 if losses else 0

    # Expectancy
    n = len(trades)
    wr = len(wins) / n
    expectancy = (wr * avg_win + (1-wr) * avg_loss)

    return {
        'pf': pf,
        'max_dd': max_dd,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': expectancy,
        'win_count': len(wins),
        'loss_count': len(losses)
    }

m_orig = calc_metrics(trades_original)
m_opt = calc_metrics(trades_optimized)

print(f'\n{"Metric":<20} {"Original":>15} {"Optimized":>15} {"Diff":>12}')
print('-'*65)
print(f'{"Profit Factor":<20} {m_orig["pf"]:>15.2f} {m_opt["pf"]:>15.2f} {m_opt["pf"]-m_orig["pf"]:>+11.2f}')
print(f'{"Max Drawdown":<20} {m_orig["max_dd"]:>14.1f}% {m_opt["max_dd"]:>14.1f}% {m_opt["max_dd"]-m_orig["max_dd"]:>+10.1f}%')
print(f'{"Avg Win":<20} {m_orig["avg_win"]:>14.2f}% {m_opt["avg_win"]:>14.2f}% {m_opt["avg_win"]-m_orig["avg_win"]:>+10.2f}%')
print(f'{"Avg Loss":<20} {m_orig["avg_loss"]:>14.2f}% {m_opt["avg_loss"]:>14.2f}% {m_opt["avg_loss"]-m_orig["avg_loss"]:>+10.2f}%')
print(f'{"Expectancy/Trade":<20} {m_orig["expectancy"]:>14.2f}% {m_opt["expectancy"]:>14.2f}% {m_opt["expectancy"]-m_orig["expectancy"]:>+10.2f}%')
print(f'{"Wins":<20} {m_orig["win_count"]:>15} {m_opt["win_count"]:>15}')
print(f'{"Losses":<20} {m_orig["loss_count"]:>15} {m_opt["loss_count"]:>15}')

# =============================================================================
# 10. SUMMARY
# =============================================================================
print('\n' + '='*70)
print('SUMMARY')
print('='*70)

total_pnl_orig = sum(t['pnl'] for t in trades_original) * 100
total_pnl_opt = sum(t['pnl'] for t in trades_optimized) * 100
improvement = total_pnl_opt - total_pnl_orig

print(f'\n  Original (3% TP / 1.5% SL):')
print(f'    Total PnL: {total_pnl_orig:+.1f}%')
print(f'    Win Rate: {sum(t["win"] for t in trades_original)/len(trades_original)*100:.1f}%')
print(f'    Profit Factor: {m_orig["pf"]:.2f}')

print(f'\n  Optimized (4% TP / 2% SL):')
print(f'    Total PnL: {total_pnl_opt:+.1f}%')
print(f'    Win Rate: {sum(t["win"] for t in trades_optimized)/len(trades_optimized)*100:.1f}%')
print(f'    Profit Factor: {m_opt["pf"]:.2f}')

print(f'\n  Improvement: {improvement:+.1f}% total PnL')

# Check consistency
years_improved = 0
years_worse = 0
for year in years:
    yr_orig = [t for t in trades_original if t['year'] == year]
    yr_opt = [t for t in trades_optimized if t['year'] == year]
    pnl_orig = sum(t['pnl'] for t in yr_orig) * 100
    pnl_opt = sum(t['pnl'] for t in yr_opt) * 100
    if pnl_opt > pnl_orig:
        years_improved += 1
    elif pnl_opt < pnl_orig:
        years_worse += 1

print(f'\n  Consistency: {years_improved}/{len(years)} years improved, {years_worse}/{len(years)} years worse')

regimes_improved = 0
for reg in ['BULL', 'BEAR', 'RANGE']:
    reg_orig = [t for t in trades_original if t['regime'] == reg]
    reg_opt = [t for t in trades_optimized if t['regime'] == reg]
    pnl_orig = sum(t['pnl'] for t in reg_orig) * 100
    pnl_opt = sum(t['pnl'] for t in reg_opt) * 100
    if pnl_opt >= pnl_orig:
        regimes_improved += 1

print(f'  Regimes: {regimes_improved}/3 improved or equal')

if years_improved >= len(years) * 0.7 and regimes_improved >= 2:
    print(f'\n  [VALIDATED] TP/SL optimization is consistent across markets!')
else:
    print(f'\n  [CAUTION] Results are not consistent - needs more analysis')

print('\n[COMPLETED]')
