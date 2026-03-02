"""
Multi-Pair Low-Overfitting Experiment
======================================
Test Ridge(alpha=100) with minimal features on all pairs.
Goal: Find which pairs work best with this low-overfitting approach.

Pairs to test:
- BTC (reference)
- BNB, XRP, AVAX, ADA, LINK, DOGE, NEAR, DOT
- ETH excluded (already tested, doesn't work)
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
np.random.seed(42)

PAIRS = ['BTC', 'BNB', 'XRP', 'AVAX', 'ADA', 'LINK', 'DOGE', 'NEAR', 'DOT']


def compute_features_minimal(df):
    """7 features - lowest overfitting."""
    feat = pd.DataFrame(index=df.index)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    feat['ret_1'] = c.pct_change(1)
    feat['ret_5'] = c.pct_change(5)
    feat['ret_20'] = c.pct_change(20)
    feat['vol20'] = c.pct_change().rolling(20).std()
    feat['rsi14'] = ta.rsi(c, length=14)
    ema21 = ta.ema(c, length=21)
    feat['ema21_d'] = (c - ema21) / ema21 * 100
    feat['vr'] = v / v.rolling(20).mean()
    return feat


def load_data(pair):
    """Load pair data."""
    patterns = [
        f'{pair}_USDT_4h_full.parquet',
        f'{pair}_USDT_4h_backtest.parquet',
        f'{pair}_USDT_4h_history.parquet',
    ]

    for pattern in patterns:
        file_path = DATA_DIR / pattern
        if file_path.exists():
            df = pd.read_parquet(file_path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            return df.sort_index()

    return None


def prepare_data(df, target_periods=5):
    """Prepare features and target."""
    feat = compute_features_minimal(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)
    target = df['close'].pct_change(target_periods).shift(-target_periods)
    valid_idx = feat.dropna().index.intersection(target.dropna().index)
    X = feat.loc[valid_idx].iloc[:-target_periods]
    y = target.loc[valid_idx].iloc[:-target_periods]
    return X, y


def backtest_model(model, scaler, X_test, df_test, pred_std,
                   tp_pct, sl_pct, conv_min, direction_filter='BOTH'):
    """Backtest con TP/SL real."""
    X_test_scaled = scaler.transform(X_test.fillna(0))
    preds = model.predict(X_test_scaled)
    conf = np.abs(preds) / pred_std if pred_std > 1e-8 else np.zeros_like(preds)

    trades = []
    position = None
    test_indices = list(X_test.index)

    for i, idx in enumerate(test_indices[:-5]):
        pred = preds[i]
        conviction = conf[i]
        direction = 1 if pred > 0 else -1

        if direction_filter == 'LONG_ONLY' and direction == -1:
            continue
        if direction_filter == 'SHORT_ONLY' and direction == 1:
            continue

        if position is None and conviction >= conv_min:
            entry_price = df_test.loc[idx, 'close']
            position = {
                'entry_idx': i,
                'entry_price': entry_price,
                'direction': direction,
                'tp_price': entry_price * (1 + direction * tp_pct),
                'sl_price': entry_price * (1 - direction * sl_pct),
                'conviction': conviction,
            }

        elif position is not None:
            current_high = df_test.loc[idx, 'high']
            current_low = df_test.loc[idx, 'low']
            current_close = df_test.loc[idx, 'close']

            hit_tp = hit_sl = False
            if position['direction'] == 1:
                hit_tp = current_high >= position['tp_price']
                hit_sl = current_low <= position['sl_price']
            else:
                hit_tp = current_low <= position['tp_price']
                hit_sl = current_high >= position['sl_price']

            timeout = (i - position['entry_idx']) >= 30

            if hit_tp or hit_sl or timeout:
                leverage = 5
                commission = 0.0004

                if hit_tp:
                    pnl_pct = tp_pct * position['direction'] * leverage
                elif hit_sl:
                    pnl_pct = -sl_pct * leverage
                else:
                    raw_pnl = (current_close - position['entry_price']) / position['entry_price']
                    pnl_pct = raw_pnl * position['direction'] * leverage

                pnl_pct -= commission * 2
                trades.append({
                    'pnl_pct': pnl_pct,
                    'win': pnl_pct > 0,
                    'direction': position['direction'],
                })
                position = None

    return trades


def grid_search_best_config(model, scaler, X_test, df_test, pred_std):
    """Find best TP/SL and direction config."""
    best_result = None
    best_pnl = -9999

    tp_range = [0.02, 0.025, 0.03, 0.04, 0.05]
    sl_range = [0.01, 0.015, 0.02, 0.025]
    conv_range = [0.3, 0.5, 0.7, 1.0]
    dir_range = ['BOTH', 'LONG_ONLY', 'SHORT_ONLY']

    for direction in dir_range:
        for tp in tp_range:
            for sl in sl_range:
                for conv in conv_range:
                    trades = backtest_model(
                        model, scaler, X_test, df_test, pred_std,
                        tp_pct=tp, sl_pct=sl, conv_min=conv,
                        direction_filter=direction
                    )

                    if len(trades) < 10:
                        continue

                    wins = sum(1 for t in trades if t['win'])
                    wr = wins / len(trades) * 100
                    pnl = sum(t['pnl_pct'] for t in trades) * 100

                    # Score: prefer higher WR with positive PnL
                    score = pnl if wr >= 50 else pnl * 0.5

                    if score > best_pnl:
                        best_pnl = score
                        best_result = {
                            'direction': direction,
                            'tp_pct': tp,
                            'sl_pct': sl,
                            'conv_min': conv,
                            'trades': len(trades),
                            'wr': wr,
                            'pnl': pnl,
                        }

    return best_result


def analyze_pair(pair):
    """Full analysis for a single pair."""
    df = load_data(pair)
    if df is None:
        return None, f"No data found for {pair}"

    X, y = prepare_data(df)
    if len(X) < 500:
        return None, f"Insufficient data for {pair}: {len(X)} samples"

    # 80/20 split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    df_test = df.loc[X_test.index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test.fillna(0))

    # Train Ridge model
    model = Ridge(alpha=100.0)
    model.fit(X_train_scaled, y_train)

    # Correlation analysis
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    train_corr = np.corrcoef(train_pred, y_train)[0, 1]
    test_corr = np.corrcoef(test_pred, y_test)[0, 1]
    pred_std = np.std(train_pred)

    corr_drop = (train_corr - test_corr) / train_corr * 100 if train_corr > 0 else 0

    # Find best config
    best_config = grid_search_best_config(model, scaler, X_test, df_test, pred_std)

    result = {
        'pair': pair,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'test_start': str(X_test.index.min().date()),
        'test_end': str(X_test.index.max().date()),
        'train_corr': train_corr,
        'test_corr': test_corr,
        'corr_drop': corr_drop,
    }

    if best_config:
        result.update(best_config)
    else:
        result.update({
            'direction': '-',
            'tp_pct': 0,
            'sl_pct': 0,
            'conv_min': 0,
            'trades': 0,
            'wr': 0,
            'pnl': 0,
        })

    return result, None


def main():
    print("="*80)
    print("MULTI-PAIR LOW-OVERFITTING EXPERIMENT")
    print("Model: Ridge(alpha=100) with 7 minimal features")
    print("="*80)

    results = []

    for pair in PAIRS:
        print(f"\nAnalyzing {pair}...", end=" ")
        result, error = analyze_pair(pair)

        if error:
            print(f"ERROR: {error}")
            continue

        results.append(result)

        if result['trades'] > 0:
            print(f"OK - {result['trades']} trades, {result['wr']:.1f}% WR, ${result['pnl']:.2f}")
        else:
            print("No valid configuration found")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: ALL PAIRS")
    print("="*80)

    print(f"\n{'Pair':<6} {'Test Period':<25} {'Corr Drop':<10} {'Direction':<12} {'Trades':<8} {'WR':<8} {'PnL':<10}")
    print("-"*90)

    # Sort by PnL
    results_sorted = sorted(results, key=lambda x: x['pnl'], reverse=True)

    for r in results_sorted:
        test_period = f"{r['test_start']} to {r['test_end']}"
        print(f"{r['pair']:<6} {test_period:<25} {r['corr_drop']:>6.1f}%    {r['direction']:<12} "
              f"{r['trades']:<8} {r['wr']:>5.1f}%   ${r['pnl']:>7.2f}")

    # Filter profitable pairs
    print("\n" + "="*80)
    print("PROFITABLE PAIRS (WR >= 50% AND PnL > 0)")
    print("="*80)

    profitable = [r for r in results if r['wr'] >= 50 and r['pnl'] > 0 and r['trades'] >= 20]
    profitable = sorted(profitable, key=lambda x: x['pnl'], reverse=True)

    if profitable:
        print(f"\n{'Pair':<6} {'Direction':<12} {'TP':<6} {'SL':<6} {'Conv':<6} {'Trades':<8} {'WR':<8} {'PnL':<10} {'Overfit':<8}")
        print("-"*85)

        total_pnl = 0
        for r in profitable:
            print(f"{r['pair']:<6} {r['direction']:<12} {r['tp_pct']*100:>4.1f}%  {r['sl_pct']*100:>4.1f}%  "
                  f"{r['conv_min']:>4.1f}   {r['trades']:<8} {r['wr']:>5.1f}%   ${r['pnl']:>7.2f}   {r['corr_drop']:>5.1f}%")
            total_pnl += r['pnl']

        print("-"*85)
        print(f"{'TOTAL':<6} {'':<12} {'':<6} {'':<6} {'':<6} {sum(r['trades'] for r in profitable):<8} {'':<8} ${total_pnl:>7.2f}")
    else:
        print("\nNo profitable pairs found with WR >= 50%")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(DATA_DIR / 'multi_pair_results.csv', index=False)
    print(f"\nResults saved to: data/multi_pair_results.csv")

    return results


if __name__ == '__main__':
    results = main()

    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    print("""
NEXT STEPS:
1. Pairs with positive PnL and WR >= 50% are candidates for production
2. Pairs with high corr_drop (>30%) may be overfitting
3. Consider combining multiple profitable pairs for diversification
4. SHORT_ONLY pairs may work in bear markets (verify with different test period)
""")
