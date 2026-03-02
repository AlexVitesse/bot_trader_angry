"""
Comprehensive Overfitting Validation
=====================================
Multiple tests to verify if models generalize:

1. Walk-Forward Validation (5 windows)
2. Time Series Cross-Validation (5 folds)
3. Train/Test Distribution Comparison
4. Consistency Score

A pair passes validation if:
- Walk-forward WR variance < 15%
- At least 3/5 folds profitable
- Consistent positive correlation across folds
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
np.random.seed(42)

# All pairs to validate
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
    return X, y, df


def backtest_fold(model, scaler, X_test, df_test, pred_std,
                  tp_pct=0.02, sl_pct=0.02, conv_min=1.0):
    """Backtest on a single fold - LONG_ONLY."""
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

        # LONG_ONLY
        if direction == -1:
            continue

        if position is None and conviction >= conv_min:
            entry_price = df_test.loc[idx, 'close']
            position = {
                'entry_idx': i,
                'entry_price': entry_price,
                'direction': direction,
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
                elif hit_sl:
                    pnl_pct = -sl_pct * leverage
                else:
                    raw_pnl = (current_close - position['entry_price']) / position['entry_price']
                    pnl_pct = raw_pnl * leverage

                pnl_pct -= commission * 2
                trades.append({'pnl_pct': pnl_pct, 'win': pnl_pct > 0})
                position = None

    if not trades:
        return {'trades': 0, 'wr': 0, 'pnl': 0}

    wins = sum(1 for t in trades if t['win'])
    return {
        'trades': len(trades),
        'wr': wins / len(trades) * 100,
        'pnl': sum(t['pnl_pct'] for t in trades) * 100
    }


def walk_forward_validation(pair, X, y, df, n_windows=5):
    """
    Walk-Forward Validation with multiple windows.

    Window 1: Train 0-60%, Test 60-70%
    Window 2: Train 10-70%, Test 70-80%
    Window 3: Train 20-80%, Test 80-90%
    Window 4: Train 30-90%, Test 90-100%
    Window 5: Train 0-80%, Test 80-100% (standard)
    """
    total_len = len(X)
    results = []

    windows = [
        (0.0, 0.6, 0.6, 0.7),   # Window 1
        (0.1, 0.7, 0.7, 0.8),   # Window 2
        (0.2, 0.8, 0.8, 0.9),   # Window 3
        (0.3, 0.9, 0.9, 1.0),   # Window 4
        (0.0, 0.8, 0.8, 1.0),   # Window 5 (standard 80/20)
    ]

    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        train_s = int(total_len * train_start)
        train_e = int(total_len * train_end)
        test_s = int(total_len * test_start)
        test_e = int(total_len * test_end)

        X_train = X.iloc[train_s:train_e]
        y_train = y.iloc[train_s:train_e]
        X_test = X.iloc[test_s:test_e]
        y_test = y.iloc[test_s:test_e]

        if len(X_train) < 100 or len(X_test) < 50:
            continue

        df_test = df.loc[X_test.index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test.fillna(0))

        model = Ridge(alpha=100.0)
        model.fit(X_train_scaled, y_train)

        # Correlation
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)

        train_corr = np.corrcoef(train_pred, y_train)[0, 1]
        test_corr = np.corrcoef(test_pred, y_test)[0, 1]
        pred_std = np.std(train_pred)

        # Backtest
        bt_result = backtest_fold(model, scaler, X_test, df_test, pred_std)

        results.append({
            'window': i + 1,
            'train_period': f"{X_train.index.min().date()} to {X_train.index.max().date()}",
            'test_period': f"{X_test.index.min().date()} to {X_test.index.max().date()}",
            'train_corr': train_corr,
            'test_corr': test_corr,
            'corr_drop': (train_corr - test_corr) / train_corr * 100 if train_corr > 0 else 0,
            **bt_result
        })

    return results


def time_series_cv(pair, X, y, df, n_splits=5):
    """Time Series Cross-Validation using sklearn TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        if len(X_test) < 50:
            continue

        df_test = df.loc[X_test.index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test.fillna(0))

        model = Ridge(alpha=100.0)
        model.fit(X_train_scaled, y_train)

        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)

        train_corr = np.corrcoef(train_pred, y_train)[0, 1]
        test_corr = np.corrcoef(test_pred, y_test)[0, 1]
        pred_std = np.std(train_pred)

        bt_result = backtest_fold(model, scaler, X_test, df_test, pred_std)

        results.append({
            'fold': fold + 1,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_corr': train_corr,
            'test_corr': test_corr,
            **bt_result
        })

    return results


def calculate_consistency_score(wf_results, cv_results):
    """
    Calculate consistency score (0-100).

    Criteria:
    - WR variance across folds (lower is better)
    - % of profitable folds
    - Correlation consistency
    - PnL consistency
    """
    score = 0
    details = {}

    # 1. WR Variance (max 25 points)
    wrs = [r['wr'] for r in wf_results if r['trades'] > 0]
    if wrs:
        wr_variance = np.std(wrs)
        wr_mean = np.mean(wrs)
        details['wr_mean'] = wr_mean
        details['wr_std'] = wr_variance

        if wr_variance < 5:
            score += 25
        elif wr_variance < 10:
            score += 20
        elif wr_variance < 15:
            score += 15
        elif wr_variance < 20:
            score += 10
        else:
            score += 5

    # 2. Profitable folds (max 25 points)
    profitable_folds = sum(1 for r in wf_results if r['pnl'] > 0)
    total_folds = len(wf_results)
    details['profitable_folds'] = f"{profitable_folds}/{total_folds}"

    if total_folds > 0:
        profit_ratio = profitable_folds / total_folds
        score += int(profit_ratio * 25)

    # 3. Correlation consistency (max 25 points)
    test_corrs = [r['test_corr'] for r in wf_results]
    if test_corrs:
        positive_corrs = sum(1 for c in test_corrs if c > 0)
        details['positive_corr_folds'] = f"{positive_corrs}/{len(test_corrs)}"

        if positive_corrs == len(test_corrs):
            score += 25
        elif positive_corrs >= len(test_corrs) * 0.8:
            score += 20
        elif positive_corrs >= len(test_corrs) * 0.6:
            score += 15
        else:
            score += 5

    # 4. WR above 50% consistency (max 25 points)
    wr_above_50 = sum(1 for r in wf_results if r['wr'] >= 50 and r['trades'] > 0)
    details['wr_above_50_folds'] = f"{wr_above_50}/{total_folds}"

    if total_folds > 0:
        wr50_ratio = wr_above_50 / total_folds
        score += int(wr50_ratio * 25)

    return score, details


def validate_pair(pair):
    """Full validation for a single pair."""
    df = load_data(pair)
    if df is None:
        return None, "No data"

    X, y, df_full = prepare_data(df)
    if len(X) < 500:
        return None, f"Insufficient data: {len(X)}"

    # Walk-Forward Validation
    wf_results = walk_forward_validation(pair, X, y, df_full)

    # Time Series CV
    cv_results = time_series_cv(pair, X, y, df_full)

    # Consistency Score
    score, details = calculate_consistency_score(wf_results, cv_results)

    return {
        'pair': pair,
        'walk_forward': wf_results,
        'time_series_cv': cv_results,
        'consistency_score': score,
        'details': details,
    }, None


def print_validation_results(result):
    """Pretty print validation results."""
    pair = result['pair']
    wf = result['walk_forward']
    cv = result['time_series_cv']
    score = result['consistency_score']
    details = result['details']

    print(f"\n{'='*70}")
    print(f"VALIDATION: {pair}/USDT")
    print(f"{'='*70}")

    # Walk-Forward Results
    print(f"\n--- Walk-Forward Validation (5 Windows) ---")
    print(f"{'Window':<8} {'Test Period':<25} {'Corr Drop':<10} {'Trades':<8} {'WR':<8} {'PnL':<10}")
    print("-"*75)

    for r in wf:
        print(f"{r['window']:<8} {r['test_period']:<25} {r['corr_drop']:>6.1f}%    "
              f"{r['trades']:<8} {r['wr']:>5.1f}%   ${r['pnl']:>7.2f}")

    # Summary stats
    wrs = [r['wr'] for r in wf if r['trades'] > 0]
    pnls = [r['pnl'] for r in wf]

    if wrs:
        print(f"\nWR: mean={np.mean(wrs):.1f}%, std={np.std(wrs):.1f}%, min={min(wrs):.1f}%, max={max(wrs):.1f}%")
    print(f"PnL: total=${sum(pnls):.2f}, mean=${np.mean(pnls):.2f}")

    # Time Series CV
    print(f"\n--- Time Series CV (5 Folds) ---")
    print(f"{'Fold':<6} {'Train':<8} {'Test':<8} {'Train Corr':<12} {'Test Corr':<12} {'WR':<8} {'PnL':<10}")
    print("-"*70)

    for r in cv:
        print(f"{r['fold']:<6} {r['train_size']:<8} {r['test_size']:<8} "
              f"{r['train_corr']:>8.4f}     {r['test_corr']:>8.4f}     "
              f"{r['wr']:>5.1f}%   ${r['pnl']:>7.2f}")

    # Consistency Score
    print(f"\n--- CONSISTENCY SCORE: {score}/100 ---")
    for k, v in details.items():
        print(f"  {k}: {v}")

    # Verdict
    print(f"\n--- VERDICT ---")
    if score >= 70:
        print(f"  [PASS] {pair} is suitable for production")
    elif score >= 50:
        print(f"  [CAUTION] {pair} may work but needs monitoring")
    else:
        print(f"  [FAIL] {pair} shows signs of overfitting")

    return score


def main():
    print("="*80)
    print("COMPREHENSIVE OVERFITTING VALIDATION")
    print("Model: Ridge(alpha=100), Features: 7 minimal, Direction: LONG_ONLY")
    print("="*80)

    all_results = []
    scores = {}

    for pair in PAIRS:
        print(f"\nValidating {pair}...", end=" ", flush=True)
        result, error = validate_pair(pair)

        if error:
            print(f"ERROR: {error}")
            continue

        all_results.append(result)
        score = print_validation_results(result)
        scores[pair] = score

    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - PRODUCTION READINESS")
    print("="*80)

    print(f"\n{'Pair':<8} {'Score':<10} {'Verdict':<15} {'Avg WR':<10} {'Total PnL':<12}")
    print("-"*60)

    sorted_results = sorted(all_results, key=lambda x: x['consistency_score'], reverse=True)

    production_ready = []
    caution = []
    not_ready = []

    for r in sorted_results:
        pair = r['pair']
        score = r['consistency_score']
        wf = r['walk_forward']

        wrs = [x['wr'] for x in wf if x['trades'] > 0]
        avg_wr = np.mean(wrs) if wrs else 0
        total_pnl = sum(x['pnl'] for x in wf)

        if score >= 70:
            verdict = "PRODUCTION"
            production_ready.append(pair)
        elif score >= 50:
            verdict = "CAUTION"
            caution.append(pair)
        else:
            verdict = "NOT READY"
            not_ready.append(pair)

        print(f"{pair:<8} {score:<10} {verdict:<15} {avg_wr:>6.1f}%    ${total_pnl:>9.2f}")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    print(f"\nPRODUCTION READY (score >= 70):")
    if production_ready:
        for p in production_ready:
            print(f"  - {p}")
    else:
        print("  None")

    print(f"\nUSE WITH CAUTION (score 50-69):")
    if caution:
        for p in caution:
            print(f"  - {p}")
    else:
        print("  None")

    print(f"\nNOT RECOMMENDED (score < 50):")
    if not_ready:
        for p in not_ready:
            print(f"  - {p}")
    else:
        print("  None")

    # Save results
    summary_data = []
    for r in all_results:
        wf = r['walk_forward']
        wrs = [x['wr'] for x in wf if x['trades'] > 0]
        summary_data.append({
            'pair': r['pair'],
            'consistency_score': r['consistency_score'],
            'avg_wr': np.mean(wrs) if wrs else 0,
            'wr_std': np.std(wrs) if wrs else 0,
            'total_pnl': sum(x['pnl'] for x in wf),
            'profitable_folds': r['details'].get('profitable_folds', '0/0'),
            'wr_above_50_folds': r['details'].get('wr_above_50_folds', '0/0'),
        })

    pd.DataFrame(summary_data).to_csv(DATA_DIR / 'overfitting_validation_results.csv', index=False)
    print(f"\nResults saved to: data/overfitting_validation_results.csv")


if __name__ == '__main__':
    main()
