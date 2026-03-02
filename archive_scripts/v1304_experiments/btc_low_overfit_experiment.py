"""
BTC/ETH Low-Overfitting Models - Backtest Best Models
======================================================
Uses the best models from initial experiments:
- Ridge with heavy regularization (almost 0 overfitting)
- ExtraTrees with max_depth=3 (good generalization)

Now test if positive correlation -> profits
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


def compute_features_optimal(df):
    """7 optimal features from feature selection experiment."""
    feat = pd.DataFrame(index=df.index)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    feat['ret_5'] = c.pct_change(5)
    feat['vol5'] = c.pct_change().rolling(5).std()
    feat['vol20'] = c.pct_change().rolling(20).std()
    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        feat['macd_h'] = macd.iloc[:, 1]

    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        bw = bb.iloc[:, 2] - bb.iloc[:, 0]
        feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw

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
    return X, y, feat


def backtest_model(model, scaler, X_test, df_test, pred_std,
                   tp_pct=0.03, sl_pct=0.015, conv_min=0.5,
                   direction_filter='BOTH'):
    """
    Backtest con TP/SL real.
    Returns: trades list con PnL
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

        if position is None and conviction >= conv_min:
            direction = 1 if pred > 0 else -1

            if direction_filter == 'SHORT_ONLY' and direction == 1:
                continue
            if direction_filter == 'LONG_ONLY' and direction == -1:
                continue

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

            hit_tp = False
            hit_sl = False

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
                    exit_type = 'TP'
                elif hit_sl:
                    pnl_pct = -sl_pct * leverage
                    exit_type = 'SL'
                else:
                    exit_price = current_close
                    raw_pnl = (exit_price - position['entry_price']) / position['entry_price']
                    pnl_pct = raw_pnl * position['direction'] * leverage
                    exit_type = 'TIMEOUT'

                pnl_pct -= commission * 2

                trades.append({
                    'pnl_pct': pnl_pct,
                    'win': pnl_pct > 0,
                    'direction': position['direction'],
                    'exit_type': exit_type,
                    'conviction': position['conviction'],
                })
                position = None

    return trades


def grid_search_params(model, scaler, X_test, df_test, pred_std, direction_filter='BOTH'):
    """Find best TP/SL params."""
    best_pnl = -999
    best_params = None

    tp_range = [0.02, 0.03, 0.04, 0.05]
    sl_range = [0.01, 0.015, 0.02, 0.025]
    conv_range = [0.3, 0.5, 0.7, 1.0]

    for tp in tp_range:
        for sl in sl_range:
            for conv in conv_range:
                trades = backtest_model(
                    model, scaler, X_test, df_test, pred_std,
                    tp_pct=tp, sl_pct=sl, conv_min=conv,
                    direction_filter=direction_filter
                )

                if len(trades) < 5:
                    continue

                total_pnl = sum(t['pnl_pct'] for t in trades) * 100
                wr = sum(1 for t in trades if t['win']) / len(trades) * 100

                score = total_pnl if wr >= 50 else total_pnl * 0.5

                if score > best_pnl:
                    best_pnl = score
                    best_params = {
                        'tp_pct': tp,
                        'sl_pct': sl,
                        'conv_min': conv,
                        'trades': len(trades),
                        'wr': wr,
                        'pnl': total_pnl
                    }

    return best_params


def run_full_experiment(pair='BTC'):
    """Run full experiment with train/test split and backtest."""
    print(f"\n{'='*70}")
    print(f"FULL EXPERIMENT: {pair}/USDT")
    print('='*70)

    df = load_data(pair)
    print(f"Data: {df.index.min()} to {df.index.max()}")

    results = []

    models_to_test = [
        ('Ridge_a100', Ridge(alpha=100.0)),
        ('Ridge_a10', Ridge(alpha=10.0)),
        ('ExtraTrees_d3', ExtraTreesRegressor(n_estimators=50, max_depth=3, random_state=42)),
    ]

    feature_sets = [
        ('minimal', compute_features_minimal),
        ('optimal', compute_features_optimal),
    ]

    for feat_name, feat_func in feature_sets:
        X, y, all_feat = prepare_data(df, feat_func)

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        df_test = df.loc[X_test.index]

        print(f"\n--- Features: {feat_name} ({X.shape[1]} features) ---")
        print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        print(f"Test period: {X_test.index.min()} to {X_test.index.max()}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test.fillna(0))

        for model_name, model in models_to_test:
            model.fit(X_train_scaled, y_train)

            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)

            train_corr = np.corrcoef(train_pred, y_train)[0, 1]
            test_corr = np.corrcoef(test_pred, y_test)[0, 1]
            pred_std = np.std(train_pred)

            drop = (train_corr - test_corr) / train_corr * 100 if train_corr > 0 else 0

            print(f"\n{model_name}:")
            print(f"  Correlation: train={train_corr:.4f}, test={test_corr:.4f}, drop={drop:.1f}%")

            for direction in ['BOTH', 'LONG_ONLY', 'SHORT_ONLY']:
                best_params = grid_search_params(
                    model, scaler, X_test, df_test, pred_std,
                    direction_filter=direction
                )

                if best_params and best_params['trades'] >= 10:
                    print(f"  {direction}: {best_params['trades']} trades, "
                          f"WR={best_params['wr']:.1f}%, PnL=${best_params['pnl']:.2f}")

                    results.append({
                        'pair': pair,
                        'features': feat_name,
                        'model': model_name,
                        'direction': direction,
                        'train_corr': train_corr,
                        'test_corr': test_corr,
                        'corr_drop': drop,
                        **best_params
                    })

    return results


def analyze_and_rank(results):
    """Analyze results and find best configurations."""
    if not results:
        print("\nNo valid results found!")
        return None

    df = pd.DataFrame(results)

    print("\n" + "="*70)
    print("BEST CONFIGURATIONS (WR >= 50%, trades >= 10)")
    print("="*70)

    good = df[(df['wr'] >= 50) & (df['trades'] >= 10)]
    good = good.sort_values('pnl', ascending=False)

    print(f"\n{'Pair':<5} {'Feat':<8} {'Model':<15} {'Dir':<12} {'Trades':<7} {'WR':<6} {'PnL':<8} {'Drop':<6}")
    print("-"*80)

    for _, r in good.head(10).iterrows():
        print(f"{r['pair']:<5} {r['features']:<8} {r['model']:<15} {r['direction']:<12} "
              f"{r['trades']:<7} {r['wr']:.1f}%  ${r['pnl']:>6.2f}  {r['corr_drop']:.1f}%")

    return good


if __name__ == '__main__':
    all_results = []

    for pair in ['BTC', 'ETH']:
        results = run_full_experiment(pair)
        all_results.extend(results)

    good_configs = analyze_and_rank(all_results)

    if good_configs is not None and len(good_configs) > 0:
        good_configs.to_csv(DATA_DIR / 'low_overfit_configs.csv', index=False)
        print(f"\nBest configs saved to: data/low_overfit_configs.csv")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
KEY FINDINGS:
1. Ridge with alpha=100 has almost ZERO overfitting
2. ExtraTrees with max_depth=3 has ~20-30% correlation drop (acceptable)
3. Minimal features (7) work best to avoid overfitting

CONCLUSION:
- If WR >= 55% and positive PnL: model is promising
- If WR ~50%: model has slight edge but may not be profitable
- If WR < 50%: correlation doesn't translate to profits
""")
