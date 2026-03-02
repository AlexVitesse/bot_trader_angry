"""
BTC Experiments - Probando alternativas
========================================
1. XGBoost con mas features
2. Target 20 periodos (en vez de 5)
3. Features macro aproximadas (BTC dominance proxy, volatility regime)

Metodologia correcta:
  - Train: 2019-2024
  - Validation: 2025-01 a 2025-08 (grid search)
  - Test: 2025-09 a 2026-02 (evaluacion final)
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# Intentar importar XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost no instalado, usando GradientBoosting de sklearn")
    from sklearn.ensemble import GradientBoostingRegressor

DATA_DIR = Path('data')
TRAIN_END = '2024-12-31'
VALIDATION_END = '2025-08-31'


def load_data():
    df = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_full.parquet')
    return df.sort_index()


def compute_features_minimal(df):
    """7 features originales."""
    feat = pd.DataFrame(index=df.index)
    c, v = df['close'], df['volume']
    feat['ret_1'] = c.pct_change(1)
    feat['ret_5'] = c.pct_change(5)
    feat['ret_20'] = c.pct_change(20)
    feat['vol20'] = c.pct_change().rolling(20).std()
    feat['rsi14'] = ta.rsi(c, length=14)
    ema21 = ta.ema(c, length=21)
    feat['ema21_d'] = (c - ema21) / ema21 * 100
    feat['vr'] = v / v.rolling(20).mean()
    return feat


def compute_features_extended(df):
    """Features extendidas para XGBoost."""
    feat = pd.DataFrame(index=df.index)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    # Retornos multiples
    for p in [1, 2, 3, 5, 10, 20, 50]:
        feat[f'ret_{p}'] = c.pct_change(p)

    # Volatilidad multiples ventanas
    for p in [5, 10, 20, 50]:
        feat[f'vol_{p}'] = c.pct_change().rolling(p).std()

    # RSI multiples
    for p in [7, 14, 21]:
        feat[f'rsi_{p}'] = ta.rsi(c, length=p)

    # EMAs y distancias
    for p in [9, 21, 50, 100]:
        ema = ta.ema(c, length=p)
        feat[f'ema{p}_d'] = (c - ema) / ema * 100

    # Bollinger Bands
    bb = ta.bbands(c, length=20)
    if bb is not None and len(bb.columns) >= 3:
        bbl = bb.iloc[:, 0]  # Lower band
        bbm = bb.iloc[:, 1]  # Middle band
        bbu = bb.iloc[:, 2]  # Upper band
        feat['bb_pct'] = (c - bbl) / (bbu - bbl)
        feat['bb_width'] = (bbu - bbl) / bbm

    # MACD
    macd = ta.macd(c)
    if macd is not None and len(macd.columns) >= 3:
        feat['macd'] = macd.iloc[:, 0]
        feat['macd_signal'] = macd.iloc[:, 1]
        feat['macd_hist'] = macd.iloc[:, 2]

    # ATR
    atr = ta.atr(h, l, c, length=14)
    feat['atr_pct'] = atr / c * 100

    # Volume features
    feat['vr'] = v / v.rolling(20).mean()
    feat['vr_5'] = v / v.rolling(5).mean()

    # High/Low range
    feat['hl_range'] = (h - l) / c
    feat['hl_range_20'] = feat['hl_range'].rolling(20).mean()

    return feat


def compute_features_macro(df):
    """Features macro aproximadas (sin datos externos)."""
    feat = compute_features_extended(df)
    c = df['close']

    # Volatility regime (proxy para Fear&Greed)
    vol_20 = c.pct_change().rolling(20).std()
    vol_100 = c.pct_change().rolling(100).std()
    feat['vol_regime'] = vol_20 / vol_100  # >1 = alta vol, <1 = baja vol

    # Trend strength (proxy para momentum macro)
    ema50 = ta.ema(c, length=50)
    ema200 = ta.ema(c, length=200)
    feat['trend_strength'] = (ema50 - ema200) / ema200 * 100

    # Drawdown from ATH (rolling)
    rolling_max = c.rolling(252*6).max()  # ~1 year of 4h candles
    feat['dd_from_ath'] = (c - rolling_max) / rolling_max * 100

    # Mean reversion signal
    sma_100 = c.rolling(100).mean()
    feat['mean_rev'] = (c - sma_100) / sma_100 * 100

    # Consecutive up/down days
    daily_ret = c.pct_change()
    feat['consec_up'] = (daily_ret > 0).rolling(10).sum()
    feat['consec_down'] = (daily_ret < 0).rolling(10).sum()

    return feat


def backtest_single_position(df, preds, tp_pct, sl_pct, conv_min, pred_std):
    """Backtest con una posicion a la vez."""
    trades = []
    position = None

    for i, idx in enumerate(df.index):
        price = df.loc[idx, 'close']
        pred = preds[i]
        conviction = abs(pred) / pred_std if pred_std > 0 else 0

        if position is not None:
            entry_price = position['entry_price']
            pnl_pct = (price - entry_price) / entry_price

            if pnl_pct >= tp_pct:
                trades.append({'pnl_pct': pnl_pct, 'result': 'TP'})
                position = None
            elif pnl_pct <= -sl_pct:
                trades.append({'pnl_pct': pnl_pct, 'result': 'SL'})
                position = None

        if position is None and pred > 0 and conviction >= conv_min:
            position = {'entry_price': price}

    if position is not None:
        final_price = df.iloc[-1]['close']
        pnl_pct = (final_price - position['entry_price']) / position['entry_price']
        trades.append({'pnl_pct': pnl_pct, 'result': 'TIMEOUT'})

    if not trades:
        return {'n_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'pf': 0}

    trades_df = pd.DataFrame(trades)
    n_trades = len(trades_df)
    wins = (trades_df['pnl_pct'] > 0).sum()
    total_pnl = trades_df['pnl_pct'].sum() * 100

    gross_profit = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
    gross_loss = abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 999

    return {
        'n_trades': n_trades,
        'win_rate': wins / n_trades * 100,
        'total_pnl': total_pnl,
        'pf': pf
    }


def grid_search(df_val, preds_val, pred_std):
    """Grid search rapido."""
    tp_range = [0.02, 0.03, 0.04]
    sl_range = [0.01, 0.015, 0.02]
    conv_range = [0.5, 1.0]

    best_score, best_params, best_result = -999, None, None

    for tp in tp_range:
        for sl in sl_range:
            for conv in conv_range:
                result = backtest_single_position(df_val, preds_val, tp, sl, conv, pred_std)
                score = result['total_pnl'] if result['n_trades'] >= 10 else -999
                if score > best_score:
                    best_score = score
                    best_params = {'tp': tp, 'sl': sl, 'conv': conv}
                    best_result = result

    return best_params, best_result


def run_experiment(name, df, feat_func, target_periods, model_type='ridge'):
    """Ejecutar un experimento completo."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO: {name}")
    print(f"{'='*60}")

    # Features
    feat = feat_func(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # Target
    target = df['close'].pct_change(target_periods).shift(-target_periods)

    # Valid indices
    valid_idx = feat.dropna().index.intersection(target.dropna().index)
    X_all = feat.loc[valid_idx]
    y_all = target.loc[valid_idx]

    # Split
    train_mask = X_all.index <= TRAIN_END
    val_mask = (X_all.index > TRAIN_END) & (X_all.index <= VALIDATION_END)
    test_mask = X_all.index > VALIDATION_END

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    df_val = df.loc[X_val.index]
    df_test = df.loc[X_test.index]

    print(f"  Features: {len(feat.columns)}")
    print(f"  Target: {target_periods} periodos")
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Model
    if model_type == 'xgboost':
        if HAS_XGBOOST:
            model = XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
    else:
        model = Ridge(alpha=100.0)

    model.fit(X_train_s, y_train)

    # Predictions
    preds_train = model.predict(X_train_s)
    preds_val = model.predict(X_val_s)
    preds_test = model.predict(X_test_s)

    pred_std = float(np.std(preds_train))

    # Correlations
    corr_train = np.corrcoef(preds_train, y_train)[0, 1]
    corr_val = np.corrcoef(preds_val, y_val)[0, 1]
    corr_test = np.corrcoef(preds_test, y_test)[0, 1]

    print(f"  Corr: train={corr_train:.4f} | val={corr_val:.4f} | test={corr_test:.4f}")

    # Grid search on validation
    best_params, val_result = grid_search(df_val, preds_val, pred_std)

    if best_params is None:
        print(f"  [FAIL] No encontro params validos en validation")
        return None

    print(f"  Val: {val_result['n_trades']} trades, {val_result['win_rate']:.1f}% WR, {val_result['total_pnl']:.1f}% PnL")

    # Test
    test_result = backtest_single_position(
        df_test, preds_test,
        best_params['tp'], best_params['sl'], best_params['conv'],
        pred_std
    )

    print(f"  Test: {test_result['n_trades']} trades, {test_result['win_rate']:.1f}% WR, {test_result['total_pnl']:.1f}% PnL, PF={test_result['pf']:.2f}")

    # Verdict
    passed = (test_result['total_pnl'] > 0 and
              test_result['win_rate'] >= 45 and
              test_result['pf'] >= 1.0 and
              test_result['n_trades'] >= 10)

    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status}")

    return {
        'name': name,
        'n_features': len(feat.columns),
        'target_periods': target_periods,
        'model_type': model_type,
        'corr_train': corr_train,
        'corr_val': corr_val,
        'corr_test': corr_test,
        'val_pnl': val_result['total_pnl'],
        'test_trades': test_result['n_trades'],
        'test_wr': test_result['win_rate'],
        'test_pnl': test_result['total_pnl'],
        'test_pf': test_result['pf'],
        'passed': passed,
        'best_params': best_params
    }


def main():
    print("="*60)
    print("BTC EXPERIMENTS - Probando 3 alternativas")
    print("="*60)

    df = load_data()
    print(f"Datos: {len(df):,} candles, {df.index.min().date()} a {df.index.max().date()}")

    results = []

    # Baseline: Ridge 7 features, target 5
    r = run_experiment(
        "Baseline (Ridge, 7 feat, target=5)",
        df, compute_features_minimal, 5, 'ridge'
    )
    if r: results.append(r)

    # Experimento 1: XGBoost con mas features
    r = run_experiment(
        "XGBoost + 25 features",
        df, compute_features_extended, 5, 'xgboost'
    )
    if r: results.append(r)

    # Experimento 2: Target 20 periodos
    r = run_experiment(
        "Ridge target=20 periodos",
        df, compute_features_minimal, 20, 'ridge'
    )
    if r: results.append(r)

    # Experimento 2b: XGBoost + target 20
    r = run_experiment(
        "XGBoost target=20 periodos",
        df, compute_features_extended, 20, 'xgboost'
    )
    if r: results.append(r)

    # Experimento 3: Features macro
    r = run_experiment(
        "XGBoost + Features Macro",
        df, compute_features_macro, 5, 'xgboost'
    )
    if r: results.append(r)

    # Experimento 3b: Macro + target 20
    r = run_experiment(
        "XGBoost + Macro + target=20",
        df, compute_features_macro, 20, 'xgboost'
    )
    if r: results.append(r)

    # Summary
    print("\n" + "="*60)
    print("RESUMEN DE EXPERIMENTOS")
    print("="*60)

    print(f"\n{'Experimento':<35} {'Test PnL':>10} {'Test WR':>10} {'PF':>8} {'Status':>8}")
    print("-"*75)

    for r in results:
        status = "PASS" if r['passed'] else "FAIL"
        print(f"{r['name']:<35} {r['test_pnl']:>9.1f}% {r['test_wr']:>9.1f}% {r['test_pf']:>7.2f} {status:>8}")

    # Best result
    passed_results = [r for r in results if r['passed']]
    if passed_results:
        best = max(passed_results, key=lambda x: x['test_pnl'])
        print(f"\n[MEJOR RESULTADO]: {best['name']}")
        print(f"  PnL: {best['test_pnl']:.1f}%")
        print(f"  Params: TP={best['best_params']['tp']:.1%} SL={best['best_params']['sl']:.1%}")
    else:
        print("\n[NINGUN EXPERIMENTO PASO]")
        print("Considerar:")
        print("  1. LSTM para capturar patrones temporales")
        print("  2. Ensemble de multiples modelos")
        print("  3. Reinforcement Learning")
        print("  4. Senales tecnicas clasicas sin ML")
        print("  5. Datos alternativos (order flow, funding rates)")

    # Save results
    import json
    with open('btc_experiments_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResultados guardados en: btc_experiments_results.json")


if __name__ == '__main__':
    main()
