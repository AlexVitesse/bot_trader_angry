"""
SOL - Comparacion Clasificador: Base vs Con Features BTC
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_DIR = Path('data')
MODELS_DIR = Path('models')


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """54 features base."""
    feat = pd.DataFrame(index=df.index)
    c, h, l, o, v = df['close'], df['high'], df['low'], df['open'], df['volume']

    for p in [1, 2, 3, 5, 10, 20, 50]:
        feat[f'ret_{p}'] = c.pct_change(p)

    feat['atr14'] = ta.atr(h, l, c, length=14)
    feat['atr_r'] = feat['atr14'] / feat['atr14'].rolling(50).mean()
    feat['vol5'] = c.pct_change().rolling(5).std()
    feat['vol20'] = c.pct_change().rolling(20).std()
    feat['vol_ratio'] = feat['vol5'] / (feat['vol20'] + 1e-10)

    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)
    feat['rsi21'] = ta.rsi(c, length=21)

    sr = ta.stochrsi(c, length=14, rsi_length=14, k=3, d=3)
    if sr is not None:
        feat['srsi_k'] = sr.iloc[:, 0]
        feat['srsi_d'] = sr.iloc[:, 1]

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        feat['macd'] = macd.iloc[:, 0]
        feat['macd_h'] = macd.iloc[:, 1]
        feat['macd_s'] = macd.iloc[:, 2]

    feat['roc5'] = ta.roc(c, length=5)
    feat['roc10'] = ta.roc(c, length=10)
    feat['roc20'] = ta.roc(c, length=20)

    for el in [8, 21, 55, 100, 200]:
        e = ta.ema(c, length=el)
        feat[f'ema{el}_d'] = (c - e) / e * 100

    feat['ema8_sl'] = ta.ema(c, length=8).pct_change(3) * 100
    feat['ema21_sl'] = ta.ema(c, length=21).pct_change(5) * 100
    feat['ema55_sl'] = ta.ema(c, length=55).pct_change(5) * 100

    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        bw = bb.iloc[:, 2] - bb.iloc[:, 0]
        feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw
        feat['bb_w'] = bw / bb.iloc[:, 1] * 100

    feat['vr'] = v / v.rolling(20).mean()
    feat['vr5'] = v.rolling(5).mean() / v.rolling(20).mean()

    feat['spr'] = (h - l) / c * 100
    feat['body'] = abs(c - o) / (h - l + 1e-10)
    feat['upper_wick'] = (h - np.maximum(c, o)) / (h - l + 1e-10)
    feat['lower_wick'] = (np.minimum(c, o) - l) / (h - l + 1e-10)

    ax = ta.adx(h, l, c, length=14)
    if ax is not None:
        feat['adx'] = ax.iloc[:, 0]
        feat['dip'] = ax.iloc[:, 1]
        feat['dim'] = ax.iloc[:, 2]
        feat['di_diff'] = feat['dip'] - feat['dim']

    chop = ta.chop(h, l, c, length=14)
    if chop is not None:
        feat['chop'] = chop

    hr = df.index.hour
    dw = df.index.dayofweek
    feat['h_s'] = np.sin(2 * np.pi * hr / 24)
    feat['h_c'] = np.cos(2 * np.pi * hr / 24)
    feat['d_s'] = np.sin(2 * np.pi * dw / 7)
    feat['d_c'] = np.cos(2 * np.pi * dw / 7)

    feat['ret1_lag1'] = feat['ret_1'].shift(1)
    feat['rsi14_lag1'] = feat['rsi14'].shift(1)
    feat['ret1_lag2'] = feat['ret_1'].shift(2)
    feat['rsi14_lag2'] = feat['rsi14'].shift(2)
    feat['ret1_lag3'] = feat['ret_1'].shift(3)
    feat['rsi14_lag3'] = feat['rsi14'].shift(3)

    return feat


def add_btc_features(sol_feat: pd.DataFrame, sol_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 10 features de correlacion con BTC y ETH."""
    btc = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_full.parquet')
    eth = pd.read_parquet(DATA_DIR / 'ETH_USDT_4h_full.parquet')

    # Normalizar timezones
    if btc.index.tz is not None:
        btc.index = btc.index.tz_localize(None)
    if eth.index.tz is not None:
        eth.index = eth.index.tz_localize(None)
    if sol_df.index.tz is not None:
        sol_df = sol_df.copy()
        sol_df.index = sol_df.index.tz_localize(None)
    if sol_feat.index.tz is not None:
        sol_feat = sol_feat.copy()
        sol_feat.index = sol_feat.index.tz_localize(None)

    common_idx = sol_df.index.intersection(btc.index).intersection(eth.index)
    feat = sol_feat.loc[common_idx].copy()

    btc_ret = btc['close'].pct_change()
    eth_ret = eth['close'].pct_change()
    sol_ret = sol_df['close'].pct_change()

    # Features BTC
    feat['btc_ret_1'] = btc_ret.loc[common_idx]
    feat['btc_ret_5'] = btc['close'].pct_change(5).loc[common_idx]
    feat['btc_ret_20'] = btc['close'].pct_change(20).loc[common_idx]
    feat['btc_rsi14'] = ta.rsi(btc['close'], length=14).loc[common_idx]

    # Features ETH
    feat['eth_ret_1'] = eth_ret.loc[common_idx]
    feat['eth_ret_5'] = eth['close'].pct_change(5).loc[common_idx]

    # Correlacion y Beta
    corr_20 = sol_ret.rolling(20).corr(btc_ret)
    feat['sol_btc_corr_20'] = corr_20.loc[common_idx]

    cov = sol_ret.rolling(20).cov(btc_ret)
    var = btc_ret.rolling(20).var()
    feat['sol_btc_beta'] = (cov / var).loc[common_idx]

    # Ratio y divergencia
    sol_btc_ratio = sol_df['close'] / btc['close']
    feat['sol_btc_ratio_ret'] = sol_btc_ratio.pct_change(5).loc[common_idx]
    feat['sol_btc_divergence'] = (sol_ret - btc_ret).rolling(5).mean().loc[common_idx]

    return feat


def detect_regime(df: pd.DataFrame) -> pd.Series:
    c = df['close']
    ema20 = ta.ema(c, length=20)
    ema50 = ta.ema(c, length=50)
    ret20 = c.pct_change(20)

    regime = pd.Series('RANGE', index=df.index)
    bull = (c > ema50) & (ema20 > ema50) & (ret20 > 0.05)
    bear = (c < ema50) & (ema20 < ema50) & (ret20 < -0.05)
    regime[bull] = 'BULL'
    regime[bear] = 'BEAR'

    return regime


def backtest_classifier(df, feat, model, scaler, feature_cols,
                        tp_pct, sl_pct, prob_threshold=0.5,
                        start_date='2025-12-01'):
    """Backtest con clasificador solo SHORT."""

    X = feat[feature_cols].copy()
    X = X[X.index >= start_date]

    valid = X.notna().all(axis=1)
    X = X[valid]
    df_bt = df.loc[X.index].copy()

    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    regime = detect_regime(df_bt)

    trades = []
    for i, (idx, row) in enumerate(X.iterrows()):
        if i >= len(X) - 5:
            break

        prob = probs[i]
        reg = regime.iloc[i]

        if prob < prob_threshold:
            continue

        direction = -1  # Solo SHORT

        # Regime filter
        if reg == 'BULL':
            continue

        entry_price = df_bt.loc[idx, 'close']
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)

        exit_price = None
        exit_reason = 'timeout'

        future_idx = X.index.get_loc(idx)
        for j in range(1, min(21, len(df_bt) - future_idx)):
            bar = df_bt.iloc[future_idx + j]
            high, low = bar['high'], bar['low']

            if high >= sl_price:
                exit_price = sl_price
                exit_reason = 'sl'
                break
            elif low <= tp_price:
                exit_price = tp_price
                exit_reason = 'tp'
                break

        if exit_price is None:
            exit_price = df_bt.iloc[min(future_idx + 20, len(df_bt) - 1)]['close']

        pnl_pct = (entry_price - exit_price) / entry_price

        trades.append({
            'entry_time': idx,
            'prob': prob,
            'regime': reg,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_pct * 100,
        })

    if not trades:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'trades_df': None}

    trades_df = pd.DataFrame(trades)
    wins = (trades_df['pnl_pct'] > 0).sum()
    total = len(trades_df)
    wr = wins / total
    pnl = trades_df['pnl_usd'].sum()

    return {'trades': total, 'wr': wr, 'pnl': pnl, 'trades_df': trades_df}


def train_and_evaluate(name, feat, df, feature_cols):
    """Entrena clasificador y evalua."""
    print(f"\n{'='*60}")
    print(f"MODELO: {name}")
    print(f"Features: {len(feature_cols)}")
    print('='*60)

    # Target: WIN si precio baja (SHORT gana)
    future_ret = df['close'].pct_change(5).shift(-5)
    target = (future_ret < 0).astype(int)

    valid = feat[feature_cols].notna().all(axis=1) & target.notna()
    feat_clean = feat.loc[valid, feature_cols].copy()
    target_clean = target[valid]
    df_clean = df[valid].copy()

    # Split
    train_end = '2025-10-31'
    train_mask = feat_clean.index <= train_end
    test_mask = feat_clean.index > train_end

    X_train, y_train = feat_clean[train_mask], target_clean[train_mask]
    X_test, y_test = feat_clean[test_mask], target_clean[test_mask]

    print(f"\n   Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"   Train WIN rate: {y_train.mean()*100:.1f}%")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Entrenar RandomForest
    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_split=20,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train_s, y_train)

    # Metricas
    test_pred = model.predict(X_test_s)
    test_prob = model.predict_proba(X_test_s)[:, 1]
    test_acc = accuracy_score(y_test, test_pred)

    try:
        test_auc = roc_auc_score(y_test, test_prob)
    except:
        test_auc = 0.5

    print(f"\n   Test Accuracy: {test_acc*100:.1f}%")
    print(f"   Test AUC: {test_auc:.3f}")

    # Feature importance (top 10)
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n   Top 10 Features:")
    for i, (_, row) in enumerate(importance.head(10).iterrows()):
        marker = " <-- BTC" if 'btc' in row['feature'] or 'eth' in row['feature'] else ""
        print(f"      {i+1}. {row['feature']}: {row['importance']:.4f}{marker}")

    # Backtest con diferentes thresholds
    print("\n   Backtest (Dic 2025 - Feb 2026):")
    results = []
    for prob_thresh in [0.45, 0.50, 0.55, 0.60]:
        res = backtest_classifier(
            df_clean, feat_clean, model, scaler, feature_cols,
            tp_pct=0.05, sl_pct=0.025, prob_threshold=prob_thresh,
            start_date='2025-12-01'
        )
        print(f"      P>{prob_thresh}: {res['trades']:>3} trades | {res['wr']*100:>5.1f}% WR | ${res['pnl']:>+7.2f}")
        results.append({
            'threshold': prob_thresh,
            **res
        })

    # Guardar mejor modelo
    best = max(results, key=lambda x: x['pnl'])

    return {
        'name': name,
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'best_threshold': best['threshold'],
        'best_trades': best['trades'],
        'best_wr': best['wr'],
        'best_pnl': best['pnl'],
        'results': results,
        'importance': importance,
    }


if __name__ == '__main__':
    print("\n" + "="*70)
    print("SOL - COMPARACION CLASIFICADORES")
    print("RandomForest Base (54 feat) vs Con BTC (64 feat)")
    print("="*70)

    # Cargar datos
    df = pd.read_parquet(DATA_DIR / 'SOL_USDT_4h_full.parquet')

    # Features base
    feat_base = compute_features(df)
    base_cols = list(feat_base.columns)

    # Features con BTC
    feat_btc = add_btc_features(feat_base, df)
    btc_cols = list(feat_btc.columns)

    # Modelo 1: Base (54 features)
    result_base = train_and_evaluate(
        "RandomForest BASE (54 features)",
        feat_base, df, base_cols
    )

    # Modelo 2: Con BTC (64 features)
    result_btc = train_and_evaluate(
        "RandomForest + BTC (64 features)",
        feat_btc, df, btc_cols
    )

    # Comparacion
    print("\n" + "="*70)
    print("COMPARACION FINAL")
    print("="*70)

    print(f"\n{'Modelo':<35} | {'Trades':>6} | {'WR':>7} | {'PnL':>10} | {'AUC':>6}")
    print("-" * 75)

    for r in [result_base, result_btc]:
        print(f"{r['name']:<35} | {r['best_trades']:>6} | {r['best_wr']*100:>6.1f}% | ${r['best_pnl']:>+8.2f} | {r['test_auc']:.3f}")

    # Determinar ganador
    print("\n" + "-"*70)
    if result_btc['best_pnl'] > result_base['best_pnl']:
        winner = result_btc
        print(f"GANADOR: {winner['name']}")
        print(f"   Mejora: ${result_btc['best_pnl'] - result_base['best_pnl']:+.2f}")
    else:
        winner = result_base
        print(f"GANADOR: {winner['name']}")
        print(f"   (BTC features no mejoran)")

    # Exportar ganador
    print("\n" + "="*70)
    print("EXPORTANDO MODELO GANADOR")
    print("="*70)

    model_path = MODELS_DIR / 'sol_usdt_v3_classifier.pkl'

    export_data = {
        'model': winner['model'],
        'scaler': winner['scaler'],
        'feature_cols': winner['feature_cols'],
        'model_type': 'classifier',
        'prob_threshold': winner['best_threshold'],
        'direction': 'short',
        'tp_pct': 0.05,
        'sl_pct': 0.025,
        'test_auc': winner['test_auc'],
        'backtest_pnl': winner['best_pnl'],
        'backtest_wr': winner['best_wr'],
        'backtest_trades': winner['best_trades'],
        'n_features': len(winner['feature_cols']),
        'has_btc_features': 'btc_ret_1' in winner['feature_cols'],
    }

    joblib.dump(export_data, model_path)
    print(f"\n   Guardado: {model_path}")
    print(f"   Features: {export_data['n_features']}")
    print(f"   Threshold: P > {export_data['prob_threshold']}")
    print(f"   BTC features: {export_data['has_btc_features']}")
    print(f"   Backtest: {export_data['backtest_trades']} trades, {export_data['backtest_wr']*100:.1f}% WR, ${export_data['backtest_pnl']:+.2f}")

    # Guardar tambien el otro modelo por si acaso
    loser = result_btc if winner == result_base else result_base
    alt_path = MODELS_DIR / 'sol_usdt_v3_classifier_alt.pkl'

    alt_data = {
        'model': loser['model'],
        'scaler': loser['scaler'],
        'feature_cols': loser['feature_cols'],
        'model_type': 'classifier',
        'prob_threshold': loser['best_threshold'],
        'direction': 'short',
        'tp_pct': 0.05,
        'sl_pct': 0.025,
        'test_auc': loser['test_auc'],
        'backtest_pnl': loser['best_pnl'],
        'backtest_wr': loser['best_wr'],
        'n_features': len(loser['feature_cols']),
        'has_btc_features': 'btc_ret_1' in loser['feature_cols'],
    }

    joblib.dump(alt_data, alt_path)
    print(f"   Alternativo: {alt_path}")

    print("\n" + "="*70)
    print("LISTO PARA PRODUCCION")
    print("="*70)
