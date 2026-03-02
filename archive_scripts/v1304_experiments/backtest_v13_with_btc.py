"""
Backtest V13 + BTC Integration
==============================
Compare V13 current (without BTC) vs V13 with BTC V2 model enabled.

V13 Config:
- 8 pairs with 3% TP / 1.5% SL
- Conviction >= 0.5

BTC V2 Config:
- BTC with 4% TP / 2% SL (optimized)
- Conviction >= 1.0
- Uses btc_v2_gradientboosting.pkl model
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import ccxt

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

# V13 Configuration
V13_PAIRS = [
    'XRP/USDT', 'NEAR/USDT', 'DOT/USDT', 'ETH/USDT',
    'DOGE/USDT', 'AVAX/USDT', 'LINK/USDT', 'ADA/USDT'
]
V13_TP_PCT = 0.03
V13_SL_PCT = 0.015
V13_CONV_MIN = 0.5
V13_MAX_HOLD = 20

# BTC V2 Configuration
BTC_TP_PCT = 0.04
BTC_SL_PCT = 0.02
BTC_CONV_MIN = 1.0
BTC_MAX_HOLD = 20

# Portfolio settings
MAX_CONCURRENT = 3
CAPITAL = 100  # $100 starting capital

print('='*70)
print('BACKTEST: V13 vs V13 + BTC')
print('='*70)

# =============================================================================
# 1. LOAD MODELS
# =============================================================================
print('\n[1] Loading models...')

# Load V7 metadata (feature columns)
import json
v7_meta_path = MODELS_DIR / 'v7_meta.json'
if v7_meta_path.exists():
    with open(v7_meta_path) as f:
        v7_meta = json.load(f)
    v7_feature_cols = v7_meta['feature_cols']
    print(f'    V7 metadata: {len(v7_feature_cols)} features')
else:
    print('    ERROR: V7 metadata not found!')
    exit(1)

# BTC V2 model (optimized)
btc_model_path = MODELS_DIR / 'btc_v2_gradientboosting.pkl'
if btc_model_path.exists():
    btc_data = joblib.load(btc_model_path)
    btc_model = btc_data['model']
    btc_cols = btc_data['feature_cols']
    print(f'    BTC/USDT: loaded V2 optimized ({len(btc_cols)} features)')
else:
    print('    ERROR: BTC V2 model not found!')
    exit(1)

# Load per-pair V7 models
v7_models = {}
for pair in V13_PAIRS:
    safe_name = pair.replace('/', '_')
    # Try v7_XRP_USDT.pkl format first
    model_path = MODELS_DIR / f'v7_{safe_name}.pkl'
    if not model_path.exists():
        # Try v95_v7_XRPUSDT.pkl format
        safe_name2 = pair.replace('/', '')
        model_path = MODELS_DIR / f'v95_v7_{safe_name2}.pkl'

    if model_path.exists():
        model = joblib.load(model_path)
        v7_models[pair] = {
            'model': model,
            'feature_cols': v7_feature_cols
        }
        print(f'    {pair}: loaded')
    else:
        print(f'    {pair}: MODEL NOT FOUND')

# =============================================================================
# 2. DOWNLOAD DATA
# =============================================================================
print('\n[2] Downloading data...')

exchange = ccxt.binance({'enableRateLimit': True})

# Test period: Last 3 months
end_date = datetime.utcnow()
start_date = datetime(2025, 12, 1)  # 3 months of data

def download_ohlcv(symbol, timeframe='4h', since=None):
    """Download OHLCV data from Binance."""
    all_data = []
    since_ts = int(since.timestamp() * 1000) if since else None

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=1000)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since_ts = ohlcv[-1][0] + 1
            if since_ts >= int(end_date.timestamp() * 1000):
                break
        except Exception as e:
            print(f'    Error downloading {symbol}: {e}')
            break

    if not all_data:
        return None

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    return df

# Download all pairs
all_pairs = V13_PAIRS + ['BTC/USDT']
pair_data = {}

for pair in all_pairs:
    safe_name = pair.replace('/', '_')
    cache_file = DATA_DIR / f'{safe_name}_4h_backtest.parquet'

    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
    else:
        print(f'    Downloading {pair}...')
        df = download_ohlcv(pair, '4h', start_date)
        if df is not None:
            df.to_parquet(cache_file)

    if df is not None and len(df) > 200:
        pair_data[pair] = df
        print(f'    {pair}: {len(df)} candles')
    else:
        print(f'    {pair}: SKIPPED (insufficient data)')

# =============================================================================
# 3. FEATURE CALCULATION
# =============================================================================
print('\n[3] Calculating features...')

def calculate_features(df):
    """Calculate all features for a pair."""
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

    return feat.dropna()

# Calculate features for all pairs
pair_features = {}
for pair, df in pair_data.items():
    feat = calculate_features(df)
    pair_features[pair] = feat
    print(f'    {pair}: {len(feat)} samples')

# =============================================================================
# 4. GENERATE SIGNALS
# =============================================================================
print('\n[4] Generating signals...')

def get_signals(pair, features, df):
    """Generate trading signals for a pair."""
    if pair == 'BTC/USDT':
        model = btc_model
        cols = btc_cols
        conv_min = BTC_CONV_MIN
        tp_pct = BTC_TP_PCT
        sl_pct = BTC_SL_PCT
    else:
        if pair not in v7_models:
            return []
        model = v7_models[pair]['model']
        cols = v7_models[pair]['feature_cols']
        conv_min = V13_CONV_MIN
        tp_pct = V13_TP_PCT
        sl_pct = V13_SL_PCT

    # Ensure all required columns exist
    missing = [c for c in cols if c not in features.columns]
    if missing:
        print(f'    Warning: {pair} missing features: {missing[:3]}...')
        return []

    predictions = model.predict(features[cols])

    signals = []
    for i, (idx, pred) in enumerate(zip(features.index, predictions)):
        conviction = abs(pred) / 0.005
        if conviction >= conv_min:
            signals.append({
                'timestamp': idx,
                'pair': pair,
                'prediction': pred,
                'conviction': conviction,
                'direction': 1 if pred > 0 else -1,
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
            })

    return signals

all_signals = []
for pair in pair_features:
    signals = get_signals(pair, pair_features[pair], pair_data[pair])
    all_signals.extend(signals)
    print(f'    {pair}: {len(signals)} signals')

# Sort by timestamp
all_signals.sort(key=lambda x: x['timestamp'])
print(f'\n    Total signals: {len(all_signals)}')

# =============================================================================
# 5. SIMULATE TRADES
# =============================================================================
print('\n[5] Simulating trades...')

def simulate_trade(signal, df):
    """Simulate a single trade."""
    idx = signal['timestamp']
    direction = signal['direction']
    tp_pct = signal['tp_pct']
    sl_pct = signal['sl_pct']

    try:
        loc = df.index.get_loc(idx)
    except:
        return None

    if loc + 1 >= len(df):
        return None

    entry_time = df.index[loc + 1]
    entry = df.iloc[loc + 1]['open']
    tp = entry * (1 + tp_pct) if direction == 1 else entry * (1 - tp_pct)
    sl = entry * (1 - sl_pct) if direction == 1 else entry * (1 + sl_pct)

    max_hold = BTC_MAX_HOLD if signal['pair'] == 'BTC/USDT' else V13_MAX_HOLD

    pnl = None
    exit_type = 'TIME'
    exit_time = None

    for j in range(loc + 1, min(loc + max_hold + 1, len(df))):
        candle = df.iloc[j]
        if direction == 1:
            if candle['low'] <= sl:
                pnl = -sl_pct
                exit_type = 'SL'
                exit_time = df.index[j]
                break
            if candle['high'] >= tp:
                pnl = tp_pct
                exit_type = 'TP'
                exit_time = df.index[j]
                break
        else:
            if candle['high'] >= sl:
                pnl = -sl_pct
                exit_type = 'SL'
                exit_time = df.index[j]
                break
            if candle['low'] <= tp:
                pnl = tp_pct
                exit_type = 'TP'
                exit_time = df.index[j]
                break

    if pnl is None:
        exit_idx = min(loc + max_hold, len(df) - 1)
        exit_p = df.iloc[exit_idx]['close']
        pnl = (exit_p - entry) / entry * direction
        exit_time = df.index[exit_idx]

    return {
        'pair': signal['pair'],
        'entry_time': entry_time,
        'exit_time': exit_time,
        'direction': direction,
        'conviction': signal['conviction'],
        'pnl': pnl,
        'win': pnl > 0,
        'exit_type': exit_type,
    }

# Simulate all trades
all_trades = []
for signal in all_signals:
    df = pair_data[signal['pair']]
    trade = simulate_trade(signal, df)
    if trade:
        all_trades.append(trade)

print(f'    Total trades simulated: {len(all_trades)}')

# =============================================================================
# 6. PORTFOLIO SIMULATION WITH MAX_CONCURRENT
# =============================================================================
print('\n[6] Portfolio simulation (max {MAX_CONCURRENT} concurrent)...')

def simulate_portfolio(trades, max_concurrent=3):
    """Simulate portfolio with position limits."""
    if not trades:
        return []

    # Sort by entry time
    trades_sorted = sorted(trades, key=lambda x: x['entry_time'])

    executed = []
    open_positions = []

    for trade in trades_sorted:
        # Close expired positions
        open_positions = [p for p in open_positions if p['exit_time'] > trade['entry_time']]

        # Check if we can open new position
        if len(open_positions) < max_concurrent:
            executed.append(trade)
            open_positions.append(trade)

    return executed

# =============================================================================
# 7. COMPARE SCENARIOS
# =============================================================================
print('\n' + '='*70)
print('RESULTS')
print('='*70)

# Scenario 1: V13 without BTC
v13_trades = [t for t in all_trades if t['pair'] != 'BTC/USDT']
v13_executed = simulate_portfolio(v13_trades, MAX_CONCURRENT)

# Scenario 2: V13 + BTC
v13_btc_trades = all_trades  # All trades including BTC
v13_btc_executed = simulate_portfolio(v13_btc_trades, MAX_CONCURRENT)

# BTC only trades
btc_only = [t for t in all_trades if t['pair'] == 'BTC/USDT']

def calc_stats(trades):
    if not trades:
        return {'n': 0, 'wr': 0, 'pnl': 0, 'pf': 0}

    n = len(trades)
    wins = sum(t['win'] for t in trades)
    wr = wins / n * 100

    pnl_total = sum(t['pnl'] for t in trades) * 100
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999

    return {'n': n, 'wr': wr, 'pnl': pnl_total, 'pf': pf}

stats_v13 = calc_stats(v13_executed)
stats_v13_btc = calc_stats(v13_btc_executed)
stats_btc = calc_stats(btc_only)

print(f'\n{"Scenario":<30} {"Trades":>8} {"WR":>8} {"PnL":>10} {"PF":>8}')
print('-'*70)
print(f'{"V13 (sin BTC)":<30} {stats_v13["n"]:>8} {stats_v13["wr"]:>7.1f}% {stats_v13["pnl"]:>+9.1f}% {stats_v13["pf"]:>7.2f}')
print(f'{"V13 + BTC (hibrido)":<30} {stats_v13_btc["n"]:>8} {stats_v13_btc["wr"]:>7.1f}% {stats_v13_btc["pnl"]:>+9.1f}% {stats_v13_btc["pf"]:>7.2f}')
print(f'{"BTC solo (referencia)":<30} {stats_btc["n"]:>8} {stats_btc["wr"]:>7.1f}% {stats_btc["pnl"]:>+9.1f}% {stats_btc["pf"]:>7.2f}')

# Difference
diff_trades = stats_v13_btc['n'] - stats_v13['n']
diff_pnl = stats_v13_btc['pnl'] - stats_v13['pnl']

print(f'\n{"Diferencia (V13+BTC vs V13)":<30} {diff_trades:>+8} {"":>8} {diff_pnl:>+9.1f}%')

# =============================================================================
# 8. BREAKDOWN BY PAIR
# =============================================================================
print('\n' + '='*70)
print('BREAKDOWN BY PAIR')
print('='*70)

print(f'\n{"Pair":<12} {"Trades":>8} {"WR":>8} {"PnL":>10} {"TP%":>8} {"SL%":>8}')
print('-'*60)

for pair in all_pairs:
    pair_trades = [t for t in v13_btc_executed if t['pair'] == pair]
    if pair_trades:
        n = len(pair_trades)
        wr = sum(t['win'] for t in pair_trades) / n * 100
        pnl = sum(t['pnl'] for t in pair_trades) * 100

        tp_pct = BTC_TP_PCT * 100 if pair == 'BTC/USDT' else V13_TP_PCT * 100
        sl_pct = BTC_SL_PCT * 100 if pair == 'BTC/USDT' else V13_SL_PCT * 100

        marker = ' [NEW]' if pair == 'BTC/USDT' else ''
        print(f'{pair:<12} {n:>8} {wr:>7.1f}% {pnl:>+9.1f}% {tp_pct:>7.1f}% {sl_pct:>7.1f}%{marker}')

# =============================================================================
# 9. MONTHLY BREAKDOWN
# =============================================================================
print('\n' + '='*70)
print('MONTHLY COMPARISON')
print('='*70)

def monthly_stats(trades):
    if not trades:
        return {}

    monthly = {}
    for t in trades:
        month = t['entry_time'].strftime('%Y-%m')
        if month not in monthly:
            monthly[month] = []
        monthly[month].append(t)

    result = {}
    for month, tr in sorted(monthly.items()):
        n = len(tr)
        wr = sum(t['win'] for t in tr) / n * 100
        pnl = sum(t['pnl'] for t in tr) * 100
        result[month] = {'n': n, 'wr': wr, 'pnl': pnl}

    return result

monthly_v13 = monthly_stats(v13_executed)
monthly_v13_btc = monthly_stats(v13_btc_executed)

print(f'\n{"Month":<10} | {"--- V13 sin BTC ---":^25} | {"--- V13 + BTC ---":^25} | {"Diff":>8}')
print(f'{"":>10} | {"Trades":>8} {"WR":>8} {"PnL":>8} | {"Trades":>8} {"WR":>8} {"PnL":>8} |')
print('-'*80)

all_months = sorted(set(list(monthly_v13.keys()) + list(monthly_v13_btc.keys())))

for month in all_months:
    v13_m = monthly_v13.get(month, {'n': 0, 'wr': 0, 'pnl': 0})
    v13_btc_m = monthly_v13_btc.get(month, {'n': 0, 'wr': 0, 'pnl': 0})
    diff = v13_btc_m['pnl'] - v13_m['pnl']

    print(f'{month:<10} | {v13_m["n"]:>8} {v13_m["wr"]:>7.1f}% {v13_m["pnl"]:>+7.1f}% | {v13_btc_m["n"]:>8} {v13_btc_m["wr"]:>7.1f}% {v13_btc_m["pnl"]:>+7.1f}% | {diff:>+7.1f}%')

# =============================================================================
# 10. RECOMMENDATION
# =============================================================================
print('\n' + '='*70)
print('RECOMMENDATION')
print('='*70)

if stats_v13_btc['pnl'] > stats_v13['pnl'] and stats_v13_btc['wr'] >= stats_v13['wr'] - 5:
    print(f'\n  [RECOMMENDED] Enable BTC with V2 model')
    print(f'    - Extra PnL: {diff_pnl:+.1f}%')
    print(f'    - Extra trades: {diff_trades}')
    print(f'    - Win Rate maintained: {stats_v13_btc["wr"]:.1f}% vs {stats_v13["wr"]:.1f}%')
elif stats_v13_btc['pnl'] > stats_v13['pnl']:
    print(f'\n  [CONSIDER] BTC adds PnL but reduces WR')
    print(f'    - Extra PnL: {diff_pnl:+.1f}%')
    print(f'    - WR change: {stats_v13_btc["wr"]:.1f}% vs {stats_v13["wr"]:.1f}%')
else:
    print(f'\n  [NOT RECOMMENDED] BTC does not improve overall performance')

print('\n[COMPLETED]')
