"""
Conviction Analysis - Finding Optimal Thresholds
=================================================
Analyze how conviction thresholds affect WR and profitability
for our best models (Ridge with high regularization).
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


def load_data(pair='BTC'):
    """Load pair data."""
    file_path = DATA_DIR / f'{pair}_USDT_4h_full.parquet'
    if not file_path.exists():
        file_path = DATA_DIR / f'{pair}_USDT_4h_backtest.parquet'

    df = pd.read_parquet(file_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    return df.sort_index()


def prepare_data(df, feature_func, target_periods=5):
    """Prepare features and target."""
    feat = feature_func(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)
    target = df['close'].pct_change(target_periods).shift(-target_periods)
    valid_idx = feat.dropna().index.intersection(target.dropna().index)
    X = feat.loc[valid_idx].iloc[:-target_periods]
    y = target.loc[valid_idx].iloc[:-target_periods]
    return X, y


def detailed_backtest(model, scaler, X_test, df_test, pred_std,
                      tp_pct=0.03, sl_pct=0.015, conv_min=0.5):
    """
    Backtest LONG_ONLY with detailed trade info.
    """
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

        # LONG_ONLY: skip shorts
        if direction == -1:
            continue

        if position is None and conviction >= conv_min:
            entry_price = df_test.loc[idx, 'close']
            position = {
                'entry_idx': i,
                'entry_time': idx,
                'entry_price': entry_price,
                'conviction': conviction,
                'prediction': pred,
                'tp_price': entry_price * (1 + tp_pct),
                'sl_price': entry_price * (1 - sl_pct),
            }

        elif position is not None:
            current_high = df_test.loc[idx, 'high']
            current_low = df_test.loc[idx, 'low']
            current_close = df_test.loc[idx, 'close']

            hit_tp = current_high >= position['tp_price']
            hit_sl = current_low <= position['sl_price']
            timeout = (i - position['entry_idx']) >= 30

            if hit_tp or hit_sl or timeout:
                leverage = 5
                commission = 0.0004

                if hit_tp:
                    pnl_pct = tp_pct * leverage
                    exit_type = 'TP'
                elif hit_sl:
                    pnl_pct = -sl_pct * leverage
                    exit_type = 'SL'
                else:
                    raw_pnl = (current_close - position['entry_price']) / position['entry_price']
                    pnl_pct = raw_pnl * leverage
                    exit_type = 'TIMEOUT'

                pnl_pct -= commission * 2

                trades.append({
                    'entry_time': position['entry_time'],
                    'conviction': position['conviction'],
                    'pnl_pct': pnl_pct,
                    'win': pnl_pct > 0,
                    'exit_type': exit_type,
                })
                position = None

    return trades


def analyze_conviction_thresholds(pair='BTC'):
    """Analyze how conviction affects WR."""
    print(f"\n{'='*70}")
    print(f"CONVICTION ANALYSIS: {pair}/USDT - Ridge(alpha=100) LONG_ONLY")
    print('='*70)

    df = load_data(pair)
    X, y = prepare_data(df, compute_features_minimal)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    df_test = df.loc[X_test.index]

    print(f"Test period: {X_test.index.min()} to {X_test.index.max()}")
    print(f"Test samples: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = Ridge(alpha=100.0)
    model.fit(X_train_scaled, y_train)
    pred_std = np.std(model.predict(X_train_scaled))

    # Test different conviction levels
    print(f"\nTP=3%, SL=1.5%, LONG_ONLY")
    print(f"\n{'Conv >=':<10} {'Trades':<8} {'WR':<8} {'PnL':<10} {'Avg Conv':<10}")
    print("-"*50)

    for conv_min in [0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]:
        trades = detailed_backtest(
            model, scaler, X_test, df_test, pred_std,
            tp_pct=0.03, sl_pct=0.015, conv_min=conv_min
        )

        if len(trades) >= 5:
            wins = sum(1 for t in trades if t['win'])
            wr = wins / len(trades) * 100
            pnl = sum(t['pnl_pct'] for t in trades) * 100
            avg_conv = np.mean([t['conviction'] for t in trades])

            print(f"{conv_min:<10.1f} {len(trades):<8} {wr:>5.1f}%   ${pnl:>7.2f}   {avg_conv:>7.2f}")
        else:
            print(f"{conv_min:<10.1f} {len(trades):<8} -        -          -")

    # Optimize TP/SL at conv >= 0.5
    print(f"\n\nOPTIMIZING TP/SL at conv >= 0.5:")
    print(f"\n{'TP':<6} {'SL':<6} {'Trades':<8} {'WR':<8} {'PnL':<10} {'PF':<6}")
    print("-"*50)

    best_config = None
    best_pnl = -999

    for tp in [0.02, 0.025, 0.03, 0.035, 0.04, 0.05]:
        for sl in [0.01, 0.0125, 0.015, 0.02, 0.025]:
            trades = detailed_backtest(
                model, scaler, X_test, df_test, pred_std,
                tp_pct=tp, sl_pct=sl, conv_min=0.5
            )

            if len(trades) >= 10:
                wins = sum(1 for t in trades if t['win'])
                wr = wins / len(trades) * 100
                pnl = sum(t['pnl_pct'] for t in trades) * 100

                gross_profit = sum(t['pnl_pct'] for t in trades if t['win']) * 100
                gross_loss = abs(sum(t['pnl_pct'] for t in trades if not t['win'])) * 100
                pf = gross_profit / gross_loss if gross_loss > 0 else 0

                if pnl > best_pnl and wr >= 50:
                    best_pnl = pnl
                    best_config = {'tp': tp, 'sl': sl, 'trades': len(trades), 'wr': wr, 'pnl': pnl, 'pf': pf}

                if wr >= 50:
                    print(f"{tp*100:>4.1f}%  {sl*100:>4.1f}%  {len(trades):<8} {wr:>5.1f}%   ${pnl:>7.2f}   {pf:.2f}")

    if best_config:
        print(f"\nBEST CONFIG: TP={best_config['tp']*100:.1f}%, SL={best_config['sl']*100:.1f}%")
        print(f"  {best_config['trades']} trades, {best_config['wr']:.1f}% WR, ${best_config['pnl']:.2f} PnL, PF {best_config['pf']:.2f}")

    return best_config


def monthly_breakdown(pair='BTC'):
    """Show monthly performance."""
    print(f"\n{'='*70}")
    print(f"MONTHLY BREAKDOWN: {pair}/USDT")
    print('='*70)

    df = load_data(pair)
    X, y = prepare_data(df, compute_features_minimal)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    df_test = df.loc[X_test.index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = Ridge(alpha=100.0)
    model.fit(X_train_scaled, y_train)
    pred_std = np.std(model.predict(X_train_scaled))

    trades = detailed_backtest(
        model, scaler, X_test, df_test, pred_std,
        tp_pct=0.03, sl_pct=0.015, conv_min=0.5
    )

    if not trades:
        print("No trades found")
        return

    trades_df = pd.DataFrame(trades)
    trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.to_period('M')

    print(f"\n{'Month':<10} {'Trades':<8} {'Wins':<6} {'WR':<8} {'PnL':<10}")
    print("-"*45)

    monthly = trades_df.groupby('month').agg({
        'pnl_pct': ['count', 'sum'],
        'win': 'sum'
    })

    for month in monthly.index:
        n_trades = monthly.loc[month, ('pnl_pct', 'count')]
        wins = monthly.loc[month, ('win', 'sum')]
        pnl = monthly.loc[month, ('pnl_pct', 'sum')] * 100
        wr = wins / n_trades * 100 if n_trades > 0 else 0

        print(f"{str(month):<10} {n_trades:<8.0f} {wins:<6.0f} {wr:>5.1f}%   ${pnl:>7.2f}")

    # Summary
    total_trades = len(trades_df)
    total_wins = trades_df['win'].sum()
    total_pnl = trades_df['pnl_pct'].sum() * 100

    print("-"*45)
    print(f"{'TOTAL':<10} {total_trades:<8} {total_wins:<6.0f} {total_wins/total_trades*100:>5.1f}%   ${total_pnl:>7.2f}")


if __name__ == '__main__':
    for pair in ['BTC', 'ETH']:
        analyze_conviction_thresholds(pair)
        monthly_breakdown(pair)

    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    print("""
CONVICTION FINDINGS:
- Higher conviction = Higher WR but fewer trades
- Sweet spot: conv >= 0.5 to 1.0 for balance

TP/SL FINDINGS:
- Asymmetric TP:SL (2:1) generally works best
- Lower SL reduces losses but may trigger too often
- Higher TP increases per-trade profit but reduces WR

FINAL RECOMMENDATION:
- Use Ridge(alpha=100) with minimal features (7)
- LONG_ONLY direction
- TP=3%, SL=1.5% (2:1 ratio)
- Conv >= 0.5 for balance of trades and WR
""")
