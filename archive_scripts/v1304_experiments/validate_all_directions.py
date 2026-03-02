"""
Validate All Directions: LONG_ONLY, SHORT_ONLY, BOTH
=====================================================
Test if models work in different market conditions.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
np.random.seed(42)

PAIRS = ['BTC', 'ETH', 'BNB', 'XRP', 'AVAX', 'ADA', 'LINK', 'DOGE', 'NEAR', 'DOT']


def compute_features_minimal(df):
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
    feat = compute_features_minimal(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)
    target = df['close'].pct_change(target_periods).shift(-target_periods)
    valid_idx = feat.dropna().index.intersection(target.dropna().index)
    X = feat.loc[valid_idx].iloc[:-target_periods]
    y = target.loc[valid_idx].iloc[:-target_periods]
    return X, y, df


def detect_market_regime(df, window=20*6):
    """Detect if market is BULL, BEAR, or SIDEWAYS."""
    returns = df['close'].pct_change(window)

    regimes = []
    for i, ret in enumerate(returns):
        if pd.isna(ret):
            regimes.append('UNKNOWN')
        elif ret > 0.10:  # >10% gain
            regimes.append('BULL')
        elif ret < -0.10:  # >10% loss
            regimes.append('BEAR')
        else:
            regimes.append('SIDEWAYS')

    return pd.Series(regimes, index=df.index)


def backtest_direction(model, scaler, X_test, df_test, pred_std,
                       tp_pct=0.02, sl_pct=0.02, conv_min=1.0,
                       direction_filter='BOTH'):
    """Backtest with specific direction filter."""
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

        # Direction filter
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
            }
        elif position is not None:
            current_high = df_test.loc[idx, 'high']
            current_low = df_test.loc[idx, 'low']
            current_close = df_test.loc[idx, 'close']

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
                    'direction': position['direction']
                })
                position = None

    if not trades:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'longs': 0, 'shorts': 0}

    wins = sum(1 for t in trades if t['win'])
    longs = sum(1 for t in trades if t['direction'] == 1)
    shorts = sum(1 for t in trades if t['direction'] == -1)

    return {
        'trades': len(trades),
        'wr': wins / len(trades) * 100,
        'pnl': sum(t['pnl_pct'] for t in trades) * 100,
        'longs': longs,
        'shorts': shorts,
    }


def analyze_pair_all_directions(pair):
    """Analyze a pair with all direction filters."""
    df = load_data(pair)
    if df is None:
        return None

    X, y, df_full = prepare_data(df)
    if len(X) < 500:
        return None

    # 80/20 split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    df_test = df_full.loc[X_test.index]

    # Detect market regime in test period
    regimes = detect_market_regime(df_full)
    test_regimes = regimes.loc[X_test.index]
    regime_counts = test_regimes.value_counts()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = Ridge(alpha=100.0)
    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    pred_std = np.std(train_pred)

    results = {
        'pair': pair,
        'test_start': str(X_test.index.min().date()),
        'test_end': str(X_test.index.max().date()),
        'regime_bull': regime_counts.get('BULL', 0),
        'regime_bear': regime_counts.get('BEAR', 0),
        'regime_sideways': regime_counts.get('SIDEWAYS', 0),
    }

    # Test all directions
    for direction in ['LONG_ONLY', 'SHORT_ONLY', 'BOTH']:
        bt = backtest_direction(
            model, scaler, X_test, df_test, pred_std,
            tp_pct=0.02, sl_pct=0.02, conv_min=1.0,
            direction_filter=direction
        )

        results[f'{direction}_trades'] = bt['trades']
        results[f'{direction}_wr'] = bt['wr']
        results[f'{direction}_pnl'] = bt['pnl']

        if direction == 'BOTH':
            results['both_longs'] = bt['longs']
            results['both_shorts'] = bt['shorts']

    return results


def main():
    print("="*90)
    print("VALIDATION: ALL DIRECTIONS (LONG_ONLY, SHORT_ONLY, BOTH)")
    print("="*90)

    all_results = []

    for pair in PAIRS:
        print(f"\nAnalyzing {pair}...", end=" ")
        result = analyze_pair_all_directions(pair)

        if result:
            all_results.append(result)
            print(f"OK")
        else:
            print(f"SKIP")

    # Summary table
    print("\n" + "="*90)
    print("MARKET REGIME IN TEST PERIOD")
    print("="*90)
    print(f"\n{'Pair':<6} {'Test Period':<25} {'BULL':<8} {'BEAR':<8} {'SIDEWAYS':<10}")
    print("-"*65)

    for r in all_results:
        print(f"{r['pair']:<6} {r['test_start']} to {r['test_end']}  "
              f"{r['regime_bull']:<8} {r['regime_bear']:<8} {r['regime_sideways']:<10}")

    # Direction comparison
    print("\n" + "="*90)
    print("DIRECTION COMPARISON")
    print("="*90)

    print(f"\n{'Pair':<6} | {'LONG_ONLY':<25} | {'SHORT_ONLY':<25} | {'BOTH':<25}")
    print(f"{'':6} | {'Trades':<7} {'WR':<7} {'PnL':<10} | {'Trades':<7} {'WR':<7} {'PnL':<10} | {'Trades':<7} {'WR':<7} {'PnL':<10}")
    print("-"*90)

    for r in all_results:
        long_info = f"{r['LONG_ONLY_trades']:<7} {r['LONG_ONLY_wr']:>5.1f}%  ${r['LONG_ONLY_pnl']:>7.2f}"
        short_info = f"{r['SHORT_ONLY_trades']:<7} {r['SHORT_ONLY_wr']:>5.1f}%  ${r['SHORT_ONLY_pnl']:>7.2f}"
        both_info = f"{r['BOTH_trades']:<7} {r['BOTH_wr']:>5.1f}%  ${r['BOTH_pnl']:>7.2f}"

        print(f"{r['pair']:<6} | {long_info} | {short_info} | {both_info}")

    # Find best direction per pair
    print("\n" + "="*90)
    print("BEST DIRECTION PER PAIR")
    print("="*90)

    print(f"\n{'Pair':<6} {'Best Direction':<15} {'Trades':<8} {'WR':<8} {'PnL':<10} {'Reason':<30}")
    print("-"*85)

    recommendations = []

    for r in all_results:
        directions = [
            ('LONG_ONLY', r['LONG_ONLY_trades'], r['LONG_ONLY_wr'], r['LONG_ONLY_pnl']),
            ('SHORT_ONLY', r['SHORT_ONLY_trades'], r['SHORT_ONLY_wr'], r['SHORT_ONLY_pnl']),
            ('BOTH', r['BOTH_trades'], r['BOTH_wr'], r['BOTH_pnl']),
        ]

        # Filter directions with enough trades
        valid = [(d, t, w, p) for d, t, w, p in directions if t >= 10]

        if not valid:
            print(f"{r['pair']:<6} {'N/A':<15} {'-':<8} {'-':<8} {'-':<10} {'Not enough trades':<30}")
            continue

        # Score: prioritize WR >= 50% with positive PnL
        def score(x):
            d, t, w, p = x
            if w >= 50 and p > 0:
                return p + (w - 50) * 10  # Bonus for WR above 50
            elif p > 0:
                return p * 0.5  # Penalize low WR
            else:
                return p  # Negative stays negative

        best = max(valid, key=score)
        d, t, w, p = best

        if w >= 50 and p > 0:
            reason = "WR>=50% and profitable"
            status = "PRODUCTION"
        elif p > 0:
            reason = f"Profitable but WR={w:.1f}%"
            status = "CAUTION"
        else:
            reason = "Not profitable"
            status = "EXCLUDE"

        print(f"{r['pair']:<6} {d:<15} {t:<8} {w:>5.1f}%   ${p:>7.2f}   {reason:<30}")

        recommendations.append({
            'pair': r['pair'],
            'direction': d,
            'trades': t,
            'wr': w,
            'pnl': p,
            'status': status,
        })

    # Final recommendations
    print("\n" + "="*90)
    print("FINAL RECOMMENDATIONS")
    print("="*90)

    prod = [r for r in recommendations if r['status'] == 'PRODUCTION']
    caution = [r for r in recommendations if r['status'] == 'CAUTION']
    exclude = [r for r in recommendations if r['status'] == 'EXCLUDE']

    print(f"\nPRODUCTION READY:")
    if prod:
        for r in sorted(prod, key=lambda x: x['pnl'], reverse=True):
            print(f"  {r['pair']}: {r['direction']}, {r['wr']:.1f}% WR, ${r['pnl']:.2f}")
    else:
        print("  None")

    print(f"\nUSE WITH CAUTION:")
    if caution:
        for r in sorted(caution, key=lambda x: x['pnl'], reverse=True):
            print(f"  {r['pair']}: {r['direction']}, {r['wr']:.1f}% WR, ${r['pnl']:.2f}")
    else:
        print("  None")

    print(f"\nEXCLUDE:")
    if exclude:
        for r in exclude:
            print(f"  {r['pair']}: Best was {r['direction']} with ${r['pnl']:.2f}")
    else:
        print("  None")

    # Save
    pd.DataFrame(all_results).to_csv(DATA_DIR / 'direction_comparison.csv', index=False)
    print(f"\nResults saved to: data/direction_comparison.csv")


if __name__ == '__main__':
    main()
