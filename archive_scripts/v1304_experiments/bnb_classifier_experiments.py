"""
BNB - Experimentos de Mejora
Baseline: Solo SHORT + TP=6%/SL=3% → 80% WR, $+139
Probar: Clasificador, diferentes configs
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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


def backtest_regressor(df, feat, model, scaler, feature_cols, pred_std,
                       tp_pct, sl_pct, conv_min=1.0, only_short=True,
                       start_date='2025-12-01'):
    """Backtest con modelo regresor (V2 actual)."""

    X = feat[feature_cols].copy()
    X = X[X.index >= start_date]

    valid = X.notna().all(axis=1)
    X = X[valid]
    df_bt = df.loc[X.index].copy()

    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    conv = np.abs(preds) / pred_std
    regime = detect_regime(df_bt)

    trades = []
    for i, (idx, row) in enumerate(X.iterrows()):
        if i >= len(X) - 5:
            break

        pred = preds[i]
        c = conv[i]
        reg = regime.iloc[i]

        if c < conv_min:
            continue

        direction = 1 if pred > 0 else -1

        if only_short and direction == 1:
            continue

        if reg == 'BULL' and direction == -1:
            continue
        if reg == 'BEAR' and direction == 1:
            continue

        entry_price = df_bt.loc[idx, 'close']

        if direction == 1:
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
        else:
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)

        exit_price = None
        exit_reason = 'timeout'

        future_idx = X.index.get_loc(idx)
        for j in range(1, min(21, len(df_bt) - future_idx)):
            bar = df_bt.iloc[future_idx + j]
            high, low = bar['high'], bar['low']

            if direction == 1:
                if low <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'sl'
                    break
                elif high >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'
                    break
            else:
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

        if direction == 1:
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        trades.append({
            'entry_time': idx,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'conviction': c,
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


def backtest_classifier(df, feat, model, scaler, feature_cols,
                        tp_pct, sl_pct, prob_threshold=0.5,
                        only_short=True, start_date='2025-12-01'):
    """Backtest con clasificador."""

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

        direction = -1 if only_short else (1 if prob > 0.5 else -1)

        # Regime filter
        if reg == 'BULL' and direction == -1:
            continue
        if reg == 'BEAR' and direction == 1:
            continue

        entry_price = df_bt.loc[idx, 'close']

        if direction == 1:
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
        else:
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)

        exit_price = None
        exit_reason = 'timeout'

        future_idx = X.index.get_loc(idx)
        for j in range(1, min(21, len(df_bt) - future_idx)):
            bar = df_bt.iloc[future_idx + j]
            high, low = bar['high'], bar['low']

            if direction == 1:
                if low <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'sl'
                    break
                elif high >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'
                    break
            else:
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

        if direction == 1:
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        trades.append({
            'entry_time': idx,
            'direction': 'LONG' if direction == 1 else 'SHORT',
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


if __name__ == '__main__':
    print("\n" + "="*70)
    print("BNB - EXPERIMENTOS DE MEJORA")
    print("="*70)

    # Cargar datos
    df = pd.read_parquet(DATA_DIR / 'BNB_USDT_4h_full.parquet')
    feat = compute_features(df)
    feature_cols = list(feat.columns)

    # Cargar modelo V2 actual
    model_data = joblib.load(MODELS_DIR / 'bnb_usdt_v2_gradientboosting.pkl')

    # =================================================================
    # BASELINE: V2 Solo SHORT con diferentes TP/SL
    # =================================================================
    print("\n" + "="*60)
    print("BASELINE: Modelo V2 Regresor (Solo SHORT)")
    print("="*60)

    print("\n   Config actual vs alternativas:")
    print(f"   {'TP/SL':<10} | {'Conv':>5} | {'Trades':>6} | {'WR':>7} | {'PnL':>10}")
    print("   " + "-"*50)

    best_v2 = None
    for tp, sl in [(0.05, 0.025), (0.06, 0.03), (0.04, 0.02), (0.07, 0.035)]:
        for conv in [0.7, 1.0, 1.2]:
            res = backtest_regressor(
                df, feat, model_data['model'], model_data['scaler'],
                model_data['feature_cols'], model_data['pred_std'],
                tp_pct=tp, sl_pct=sl, conv_min=conv,
                only_short=True, start_date='2025-12-01'
            )
            if res['trades'] > 0:
                print(f"   {tp*100:.0f}%/{sl*100:.1f}%   | {conv:>5.1f} | {res['trades']:>6} | {res['wr']*100:>6.1f}% | ${res['pnl']:>+8.2f}")
                if best_v2 is None or res['pnl'] > best_v2['pnl']:
                    best_v2 = {'tp': tp, 'sl': sl, 'conv': conv, **res}

    print(f"\n   MEJOR V2: TP={best_v2['tp']*100:.0f}%/SL={best_v2['sl']*100:.1f}%, Conv>={best_v2['conv']}")
    print(f"            {best_v2['trades']} trades, {best_v2['wr']*100:.1f}% WR, ${best_v2['pnl']:+.2f}")

    # =================================================================
    # EXPERIMENTO 1: Clasificador RandomForest
    # =================================================================
    print("\n" + "="*60)
    print("EXPERIMENTO 1: Clasificador RandomForest")
    print("="*60)

    # Target: WIN si precio baja (SHORT gana)
    future_ret = df['close'].pct_change(5).shift(-5)
    target = (future_ret < 0).astype(int)

    valid = feat.notna().all(axis=1) & target.notna()
    feat_clean = feat[valid].copy()
    target_clean = target[valid]
    df_clean = df[valid].copy()

    train_end = '2025-10-31'
    train_mask = feat_clean.index <= train_end

    X_train, y_train = feat_clean[train_mask], target_clean[train_mask]

    print(f"\n   Train: {len(X_train)} samples")
    print(f"   Train WIN rate: {y_train.mean()*100:.1f}%")

    scaler_clf = StandardScaler()
    X_train_s = scaler_clf.fit_transform(X_train)

    # Entrenar RandomForest
    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_split=20,
        random_state=42, n_jobs=-1
    )
    rf_clf.fit(X_train_s, y_train)

    print("\n   Backtest Clasificador RF (Solo SHORT):")
    print(f"   {'TP/SL':<10} | {'Prob':>5} | {'Trades':>6} | {'WR':>7} | {'PnL':>10}")
    print("   " + "-"*50)

    best_clf = None
    for tp, sl in [(0.05, 0.025), (0.06, 0.03), (0.04, 0.02)]:
        for prob in [0.45, 0.50, 0.55]:
            res = backtest_classifier(
                df_clean, feat_clean, rf_clf, scaler_clf, feature_cols,
                tp_pct=tp, sl_pct=sl, prob_threshold=prob,
                only_short=True, start_date='2025-12-01'
            )
            if res['trades'] > 0:
                print(f"   {tp*100:.0f}%/{sl*100:.1f}%   | {prob:>5.2f} | {res['trades']:>6} | {res['wr']*100:>6.1f}% | ${res['pnl']:>+8.2f}")
                if best_clf is None or res['pnl'] > best_clf['pnl']:
                    best_clf = {'tp': tp, 'sl': sl, 'prob': prob, **res}

    if best_clf:
        print(f"\n   MEJOR CLF: TP={best_clf['tp']*100:.0f}%/SL={best_clf['sl']*100:.1f}%, P>{best_clf['prob']}")
        print(f"             {best_clf['trades']} trades, {best_clf['wr']*100:.1f}% WR, ${best_clf['pnl']:+.2f}")

    # =================================================================
    # EXPERIMENTO 2: Clasificador GradientBoosting
    # =================================================================
    print("\n" + "="*60)
    print("EXPERIMENTO 2: Clasificador GradientBoosting")
    print("="*60)

    gb_clf = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=42
    )
    gb_clf.fit(X_train_s, y_train)

    print("\n   Backtest Clasificador GB (Solo SHORT):")
    print(f"   {'TP/SL':<10} | {'Prob':>5} | {'Trades':>6} | {'WR':>7} | {'PnL':>10}")
    print("   " + "-"*50)

    best_gb = None
    for tp, sl in [(0.05, 0.025), (0.06, 0.03), (0.04, 0.02)]:
        for prob in [0.45, 0.50, 0.55, 0.60]:
            res = backtest_classifier(
                df_clean, feat_clean, gb_clf, scaler_clf, feature_cols,
                tp_pct=tp, sl_pct=sl, prob_threshold=prob,
                only_short=True, start_date='2025-12-01'
            )
            if res['trades'] > 0:
                print(f"   {tp*100:.0f}%/{sl*100:.1f}%   | {prob:>5.2f} | {res['trades']:>6} | {res['wr']*100:>6.1f}% | ${res['pnl']:>+8.2f}")
                if best_gb is None or res['pnl'] > best_gb['pnl']:
                    best_gb = {'tp': tp, 'sl': sl, 'prob': prob, **res}

    if best_gb:
        print(f"\n   MEJOR GB: TP={best_gb['tp']*100:.0f}%/SL={best_gb['sl']*100:.1f}%, P>{best_gb['prob']}")
        print(f"            {best_gb['trades']} trades, {best_gb['wr']*100:.1f}% WR, ${best_gb['pnl']:+.2f}")

    # =================================================================
    # EXPERIMENTO 3: V2 con filtro adicional RSI
    # =================================================================
    print("\n" + "="*60)
    print("EXPERIMENTO 3: V2 + Filtro RSI (SHORT cuando RSI > 60)")
    print("="*60)

    # Backtest manual con filtro RSI
    X = feat[feature_cols].copy()
    X = X[X.index >= '2025-12-01']
    valid = X.notna().all(axis=1)
    X = X[valid]
    df_bt = df.loc[X.index].copy()

    X_scaled = model_data['scaler'].transform(X)
    preds = model_data['model'].predict(X_scaled)
    conv = np.abs(preds) / model_data['pred_std']
    regime = detect_regime(df_bt)

    print("\n   V2 + RSI Filter (Solo SHORT cuando RSI > threshold):")
    print(f"   {'RSI Min':<10} | {'Trades':>6} | {'WR':>7} | {'PnL':>10}")
    print("   " + "-"*40)

    for rsi_min in [50, 55, 60, 65, 70]:
        trades = []
        for i, (idx, row) in enumerate(X.iterrows()):
            if i >= len(X) - 5:
                break

            pred = preds[i]
            c = conv[i]
            reg = regime.iloc[i]
            rsi = feat.loc[idx, 'rsi14']

            if c < 1.0:
                continue
            if pred > 0:  # Solo SHORT
                continue
            if reg == 'BULL':
                continue
            if rsi < rsi_min:  # Filtro RSI
                continue

            entry_price = df_bt.loc[idx, 'close']
            tp_price = entry_price * (1 - 0.06)
            sl_price = entry_price * (1 + 0.03)

            exit_price = None
            future_idx = X.index.get_loc(idx)
            for j in range(1, min(21, len(df_bt) - future_idx)):
                bar = df_bt.iloc[future_idx + j]
                if bar['high'] >= sl_price:
                    exit_price = sl_price
                    break
                elif bar['low'] <= tp_price:
                    exit_price = tp_price
                    break

            if exit_price is None:
                exit_price = df_bt.iloc[min(future_idx + 20, len(df_bt) - 1)]['close']

            pnl_pct = (entry_price - exit_price) / entry_price
            trades.append({'pnl_pct': pnl_pct, 'pnl_usd': pnl_pct * 100})

        if trades:
            trades_df = pd.DataFrame(trades)
            wins = (trades_df['pnl_pct'] > 0).sum()
            wr = wins / len(trades_df)
            pnl = trades_df['pnl_usd'].sum()
            print(f"   RSI > {rsi_min:<4} | {len(trades):>6} | {wr*100:>6.1f}% | ${pnl:>+8.2f}")

    # =================================================================
    # COMPARACION FINAL
    # =================================================================
    print("\n" + "="*70)
    print("COMPARACION FINAL")
    print("="*70)

    print(f"\n{'Modelo':<35} | {'Trades':>6} | {'WR':>7} | {'PnL':>10}")
    print("-" * 65)
    print(f"{'V2 Regresor (mejor config)':<35} | {best_v2['trades']:>6} | {best_v2['wr']*100:>6.1f}% | ${best_v2['pnl']:>+8.2f}")
    if best_clf:
        print(f"{'Clasificador RF (mejor config)':<35} | {best_clf['trades']:>6} | {best_clf['wr']*100:>6.1f}% | ${best_clf['pnl']:>+8.2f}")
    if best_gb:
        print(f"{'Clasificador GB (mejor config)':<35} | {best_gb['trades']:>6} | {best_gb['wr']*100:>6.1f}% | ${best_gb['pnl']:>+8.2f}")

    # Determinar ganador
    candidates = [('V2 Regresor', best_v2)]
    if best_clf:
        candidates.append(('Clasificador RF', best_clf))
    if best_gb:
        candidates.append(('Clasificador GB', best_gb))

    winner_name, winner = max(candidates, key=lambda x: x[1]['pnl'])

    print(f"\n" + "-"*65)
    print(f"GANADOR: {winner_name}")
    print(f"   Config: TP={winner['tp']*100:.0f}%/SL={winner['sl']*100:.1f}%")
    print(f"   {winner['trades']} trades, {winner['wr']*100:.1f}% WR, ${winner['pnl']:+.2f}")

    # Exportar si es clasificador
    if 'Clasificador' in winner_name:
        print("\n" + "="*60)
        print("EXPORTANDO MODELO CLASIFICADOR BNB")
        print("="*60)

        model_to_save = rf_clf if 'RF' in winner_name else gb_clf

        export_data = {
            'model': model_to_save,
            'scaler': scaler_clf,
            'feature_cols': feature_cols,
            'model_type': 'classifier',
            'prob_threshold': winner['prob'],
            'direction': 'short',
            'tp_pct': winner['tp'],
            'sl_pct': winner['sl'],
            'backtest_pnl': winner['pnl'],
            'backtest_wr': winner['wr'],
            'backtest_trades': winner['trades'],
        }

        model_path = MODELS_DIR / 'bnb_usdt_v3_classifier.pkl'
        joblib.dump(export_data, model_path)
        print(f"\n   Guardado: {model_path}")
    else:
        print("\n   V2 Regresor sigue siendo el mejor - no se exporta nuevo modelo")
        print(f"   Usar: models/bnb_usdt_v2_gradientboosting.pkl")
        print(f"   Config: Solo SHORT, TP={winner['tp']*100:.0f}%/SL={winner['sl']*100:.1f}%, Conv>={winner['conv']}")
