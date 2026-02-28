"""
BTC/ETH Model Experiments - Finding Models That Generalize
===========================================================
Goal: Find ML models that DON'T overfit using 80/20 split

Strategies to avoid overfitting:
1. Simpler models (Ridge, Lasso, ElasticNet)
2. Heavy regularization
3. Feature selection (fewer features)
4. Early stopping
5. Cross-validation
6. Ensemble with bagging
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    BaggingRegressor,
    ExtraTreesRegressor
)
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
np.random.seed(42)


def compute_features_minimal(df):
    """Minimal feature set - less prone to overfitting."""
    feat = pd.DataFrame(index=df.index)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    # Only basic momentum
    feat['ret_1'] = c.pct_change(1)
    feat['ret_5'] = c.pct_change(5)
    feat['ret_20'] = c.pct_change(20)

    # Volatility
    feat['vol20'] = c.pct_change().rolling(20).std()

    # RSI
    feat['rsi14'] = ta.rsi(c, length=14)

    # Single EMA distance
    ema21 = ta.ema(c, length=21)
    feat['ema21_d'] = (c - ema21) / ema21 * 100

    # Volume ratio
    feat['vr'] = v / v.rolling(20).mean()

    return feat  # Only 7 features


def compute_features_medium(df):
    """Medium feature set - balanced."""
    feat = pd.DataFrame(index=df.index)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    # Returns
    for p in [1, 5, 10, 20]:
        feat[f'ret_{p}'] = c.pct_change(p)

    # Volatility
    feat['vol5'] = c.pct_change().rolling(5).std()
    feat['vol20'] = c.pct_change().rolling(20).std()

    # RSI
    feat['rsi14'] = ta.rsi(c, length=14)

    # MACD
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        feat['macd_h'] = macd.iloc[:, 1]

    # EMAs
    for el in [8, 21, 55]:
        e = ta.ema(c, length=el)
        feat[f'ema{el}_d'] = (c - e) / e * 100

    # BB
    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        bw = bb.iloc[:, 2] - bb.iloc[:, 0]
        feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw

    # Volume
    feat['vr'] = v / v.rolling(20).mean()

    return feat  # ~15 features


def compute_features_full(df):
    """Full feature set - 54 features."""
    feat = pd.DataFrame(index=df.index)
    c, h, l, o, v = df['close'], df['high'], df['low'], df['open'], df['volume']

    for p in [1, 2, 3, 5, 10, 20, 50]:
        feat[f'ret_{p}'] = c.pct_change(p)

    feat['atr14'] = ta.atr(h, l, c, length=14)
    feat['vol5'] = c.pct_change().rolling(5).std()
    feat['vol20'] = c.pct_change().rolling(20).std()
    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        feat['macd_h'] = macd.iloc[:, 1]

    for el in [8, 21, 55, 100]:
        e = ta.ema(c, length=el)
        feat[f'ema{el}_d'] = (c - e) / e * 100

    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        bw = bb.iloc[:, 2] - bb.iloc[:, 0]
        feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw

    feat['vr'] = v / v.rolling(20).mean()

    return feat


def load_data(pair='BTC'):
    """Load BTC or ETH data."""
    file_path = DATA_DIR / f'{pair}_USDT_4h_full.parquet'
    if not file_path.exists():
        file_path = DATA_DIR / f'{pair}_USDT_4h_backtest.parquet'

    df = pd.read_parquet(file_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    df = df.sort_index()
    return df


def prepare_data(df, feature_func, target_periods=5):
    """Prepare features and target."""
    feat = feature_func(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # Target: future returns
    target = df['close'].pct_change(target_periods).shift(-target_periods)

    # Align
    valid_idx = feat.dropna().index.intersection(target.dropna().index)
    X = feat.loc[valid_idx].iloc[:-target_periods]
    y = target.loc[valid_idx].iloc[:-target_periods]

    return X, y


def evaluate_model(model, X_train, X_test, y_train, y_test, scaler):
    """Evaluate model and return metrics."""
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_corr = np.corrcoef(train_pred, y_train)[0, 1]
    test_corr = np.corrcoef(test_pred, y_test)[0, 1]

    # Correlation drop
    if train_corr > 0:
        corr_drop = (train_corr - test_corr) / train_corr * 100
    else:
        corr_drop = 0

    return {
        'train_corr': train_corr,
        'test_corr': test_corr,
        'corr_drop': corr_drop,
        'train_std': np.std(train_pred),
        'test_std': np.std(test_pred),
    }


def run_experiments(pair='BTC'):
    """Run all experiments for a pair."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENTS FOR {pair}/USDT")
    print('='*70)

    df = load_data(pair)
    print(f"Data range: {df.index.min()} to {df.index.max()}")
    print(f"Total rows: {len(df)}")

    results = []

    # Feature sets to try
    feature_sets = [
        ('minimal', compute_features_minimal),
        ('medium', compute_features_medium),
        ('full', compute_features_full),
    ]

    # Models to try
    models = [
        ('Ridge_a1', Ridge(alpha=1.0)),
        ('Ridge_a10', Ridge(alpha=10.0)),
        ('Ridge_a100', Ridge(alpha=100.0)),
        ('Lasso_a0.01', Lasso(alpha=0.01)),
        ('Lasso_a0.1', Lasso(alpha=0.1)),
        ('ElasticNet', ElasticNet(alpha=0.1, l1_ratio=0.5)),
        ('RF_d3_n50', RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)),
        ('RF_d5_n100', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)),
        ('GB_d2_n50', GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)),
        ('GB_d3_n100', GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)),
        ('ExtraTrees_d3', ExtraTreesRegressor(n_estimators=50, max_depth=3, random_state=42)),
        ('Bagging_Ridge', BaggingRegressor(estimator=Ridge(alpha=10), n_estimators=20, random_state=42)),
    ]

    for feat_name, feat_func in feature_sets:
        print(f"\n--- Feature Set: {feat_name} ---")

        X, y = prepare_data(df, feat_func)
        print(f"Features: {X.shape[1]}, Samples: {len(X)}")

        # 80/20 split - CHRONOLOGICAL (no shuffle for time series!)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"Train period: {X_train.index.min()} to {X_train.index.max()}")
        print(f"Test period: {X_test.index.min()} to {X_test.index.max()}")

        scaler = StandardScaler()

        for model_name, model in models:
            try:
                metrics = evaluate_model(model, X_train, X_test, y_train, y_test, scaler)

                results.append({
                    'pair': pair,
                    'features': feat_name,
                    'model': model_name,
                    **metrics
                })

                # Print if promising (low correlation drop)
                if metrics['corr_drop'] < 50 and metrics['test_corr'] > 0:
                    print(f"  {model_name}: train={metrics['train_corr']:.4f}, "
                          f"test={metrics['test_corr']:.4f}, drop={metrics['corr_drop']:.1f}% ✓")

            except Exception as e:
                print(f"  {model_name}: ERROR - {e}")

    return results


def analyze_results(results):
    """Analyze and rank results."""
    df = pd.DataFrame(results)

    print("\n" + "="*70)
    print("TOP MODELS BY TEST CORRELATION (with low overfitting)")
    print("="*70)

    # Filter: positive test correlation and <80% drop
    good = df[(df['test_corr'] > 0) & (df['corr_drop'] < 80)]

    if len(good) == 0:
        print("\nNo models found with positive test correlation and <80% drop")
        print("\nAll results:")
        print(df.sort_values('test_corr', ascending=False).head(20).to_string())
        return df

    # Sort by test correlation
    good = good.sort_values('test_corr', ascending=False)

    print(f"\n{'Pair':<6} {'Features':<10} {'Model':<20} {'Train':<8} {'Test':<8} {'Drop':<8}")
    print("-"*70)

    for _, row in good.head(15).iterrows():
        print(f"{row['pair']:<6} {row['features']:<10} {row['model']:<20} "
              f"{row['train_corr']:.4f}   {row['test_corr']:.4f}   {row['corr_drop']:.1f}%")

    return df


def run_feature_selection_experiment(pair='BTC'):
    """Try feature selection to reduce overfitting."""
    print(f"\n{'='*70}")
    print(f"FEATURE SELECTION EXPERIMENT - {pair}")
    print('='*70)

    df = load_data(pair)
    X, y = prepare_data(df, compute_features_full)

    # 80/20 chronological split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # Try different numbers of features
    for k in [3, 5, 7, 10, 15]:
        selector = SelectKBest(f_regression, k=k)
        X_train_sel = selector.fit_transform(X_train_scaled, y_train)
        X_test_sel = selector.transform(X_test_scaled)

        selected_features = X_train.columns[selector.get_support()].tolist()

        # Try Ridge on selected features
        model = Ridge(alpha=10.0)
        model.fit(X_train_sel, y_train)

        train_pred = model.predict(X_train_sel)
        test_pred = model.predict(X_test_sel)

        train_corr = np.corrcoef(train_pred, y_train)[0, 1]
        test_corr = np.corrcoef(test_pred, y_test)[0, 1]
        drop = (train_corr - test_corr) / train_corr * 100 if train_corr > 0 else 0

        results.append({
            'k': k,
            'train_corr': train_corr,
            'test_corr': test_corr,
            'drop': drop,
            'features': selected_features
        })

        print(f"K={k}: train={train_corr:.4f}, test={test_corr:.4f}, drop={drop:.1f}%")
        print(f"   Features: {selected_features}")

    return results


def run_cross_validation_experiment(pair='BTC'):
    """Use time-series cross-validation."""
    print(f"\n{'='*70}")
    print(f"TIME SERIES CV EXPERIMENT - {pair}")
    print('='*70)

    from sklearn.model_selection import TimeSeriesSplit

    df = load_data(pair)
    X, y = prepare_data(df, compute_features_medium)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=5)

    models = [
        ('Ridge_a10', Ridge(alpha=10.0)),
        ('Ridge_a100', Ridge(alpha=100.0)),
        ('GB_d2', GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)),
    ]

    print(f"\nUsing TimeSeriesSplit with 5 folds")
    print(f"Features: {X.shape[1]}, Samples: {len(X)}")

    for name, model in models:
        fold_scores = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_scaled[train_idx], y.iloc[train_idx])
            pred = model_copy.predict(X_scaled[test_idx])
            corr = np.corrcoef(pred, y.iloc[test_idx])[0, 1]
            fold_scores.append(corr)

        mean_corr = np.mean(fold_scores)
        std_corr = np.std(fold_scores)
        print(f"{name}: CV mean={mean_corr:.4f} (+/- {std_corr:.4f}), folds={fold_scores}")


if __name__ == '__main__':
    all_results = []

    # Run for BTC and ETH
    for pair in ['BTC', 'ETH']:
        results = run_experiments(pair)
        all_results.extend(results)

    # Analyze
    results_df = analyze_results(all_results)

    # Save results
    results_df.to_csv(DATA_DIR / 'model_experiments_results.csv', index=False)
    print(f"\nResults saved to: data/model_experiments_results.csv")

    # Additional experiments
    for pair in ['BTC', 'ETH']:
        run_feature_selection_experiment(pair)
        run_cross_validation_experiment(pair)

    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    print("""
Key findings will show:
1. Which models have lowest correlation drop (least overfitting)
2. Which feature sets work best
3. Whether simpler models (Ridge) beat complex ones (GradientBoosting)
4. Which features are most predictive

Look for:
- Models with test_corr > 0.05 (some predictive power)
- Models with corr_drop < 50% (reasonable generalization)
- Consistency across BTC and ETH
""")
