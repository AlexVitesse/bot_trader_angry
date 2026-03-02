"""
SOL V3 - Evaluacion Detallada del Clasificador
Mismo analisis que BNB: drawdown, rachas, consistencia
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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


def run_backtest(df, feat, model, scaler, feature_cols,
                 tp_pct, sl_pct, model_type='classifier',
                 conv_min=1.0, prob_threshold=0.5, pred_std=None,
                 start_date='2025-12-01'):
    """Run backtest and return detailed trades."""

    X = feat[feature_cols].copy()
    X = X[X.index >= start_date]

    valid = X.notna().all(axis=1)
    X = X[valid]
    df_bt = df.loc[X.index].copy()

    X_scaled = scaler.transform(X)

    if model_type == 'regressor':
        preds = model.predict(X_scaled)
        conv = np.abs(preds) / pred_std
        signals = conv >= conv_min
        directions = np.where(preds < 0, -1, 1)
    else:
        probs = model.predict_proba(X_scaled)[:, 1]
        signals = probs >= prob_threshold
        directions = np.full(len(probs), -1)  # Solo SHORT

    regime = detect_regime(df_bt)

    trades = []
    for i, (idx, row) in enumerate(X.iterrows()):
        if i >= len(X) - 5:
            break

        if not signals[i]:
            continue

        direction = directions[i]
        reg = regime.iloc[i]

        # Solo SHORT
        if direction == 1:
            continue

        # Regime filter
        if reg == 'BULL':
            continue

        entry_price = df_bt.loc[idx, 'close']
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)

        exit_price = None
        exit_reason = 'timeout'
        exit_bars = 0

        future_idx = X.index.get_loc(idx)
        for j in range(1, min(21, len(df_bt) - future_idx)):
            bar = df_bt.iloc[future_idx + j]
            exit_bars = j

            if bar['high'] >= sl_price:
                exit_price = sl_price
                exit_reason = 'sl'
                break
            elif bar['low'] <= tp_price:
                exit_price = tp_price
                exit_reason = 'tp'
                break

        if exit_price is None:
            exit_price = df_bt.iloc[min(future_idx + 20, len(df_bt) - 1)]['close']

        pnl_pct = (entry_price - exit_price) / entry_price

        trades.append({
            'entry_time': idx,
            'regime': reg,
            'exit_reason': exit_reason,
            'exit_bars': exit_bars,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_pct * 100,
            'win': pnl_pct > 0,
        })

    return pd.DataFrame(trades) if trades else pd.DataFrame()


def analyze_trades(trades_df, name):
    """Analyze trades in detail."""
    if len(trades_df) == 0:
        return None

    total = len(trades_df)
    wins = trades_df['win'].sum()
    losses = total - wins
    wr = wins / total
    total_pnl = trades_df['pnl_usd'].sum()

    # Profit Factor
    gross_profit = trades_df[trades_df['pnl_usd'] > 0]['pnl_usd'].sum()
    gross_loss = abs(trades_df[trades_df['pnl_usd'] < 0]['pnl_usd'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Drawdown
    cumulative = trades_df['pnl_usd'].cumsum()
    running_max = cumulative.cummax()
    drawdown = running_max - cumulative
    max_dd = drawdown.max()

    # Average win/loss
    avg_win = trades_df[trades_df['win']]['pnl_usd'].mean() if wins > 0 else 0
    avg_loss = trades_df[~trades_df['win']]['pnl_usd'].mean() if losses > 0 else 0

    # Consecutive losses
    loss_streaks = []
    current_streak = 0
    for w in trades_df['win']:
        if not w:
            current_streak += 1
        else:
            if current_streak > 0:
                loss_streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        loss_streaks.append(current_streak)
    max_loss_streak = max(loss_streaks) if loss_streaks else 0

    # Monthly
    trades_df['month'] = trades_df['entry_time'].dt.strftime('%Y-%m')
    monthly = trades_df.groupby('month').agg({
        'pnl_usd': 'sum',
        'win': ['sum', 'count']
    })
    monthly.columns = ['pnl', 'wins', 'total']
    monthly['wr'] = monthly['wins'] / monthly['total'] * 100

    losing_months = (monthly['pnl'] < 0).sum()

    return {
        'name': name,
        'trades': total,
        'wins': wins,
        'losses': losses,
        'wr': wr,
        'total_pnl': total_pnl,
        'profit_factor': pf,
        'max_drawdown': max_dd,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_loss_streak': max_loss_streak,
        'monthly': monthly,
        'losing_months': losing_months,
        'trades_df': trades_df,
    }


if __name__ == '__main__':
    print("\n" + "="*70)
    print("SOL V3 - EVALUACION DETALLADA")
    print("Clasificador RF vs V2 Regresor (baseline)")
    print("="*70)

    # Cargar datos
    df = pd.read_parquet(DATA_DIR / 'SOL_USDT_4h_full.parquet')
    feat = compute_features(df)
    feature_cols = list(feat.columns)

    # Cargar modelo V2 (baseline)
    model_data = joblib.load(MODELS_DIR / 'sol_usdt_v2_gradientboosting.pkl')

    # Preparar datos para clasificador
    future_ret = df['close'].pct_change(5).shift(-5)
    target = (future_ret < 0).astype(int)
    valid = feat.notna().all(axis=1) & target.notna()
    feat_clean = feat[valid].copy()
    target_clean = target[valid]
    df_clean = df[valid].copy()

    train_mask = feat_clean.index <= '2025-10-31'
    X_train, y_train = feat_clean[train_mask], target_clean[train_mask]

    scaler_clf = StandardScaler()
    X_train_s = scaler_clf.fit_transform(X_train)

    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_split=20,
        random_state=42, n_jobs=-1
    )
    rf_clf.fit(X_train_s, y_train)

    # =================================================================
    # OPCION 1: V2 Regresor (baseline) - diferentes configs
    # =================================================================
    print("\n" + "-"*70)
    print("V2 REGRESOR - Probando diferentes configs")
    print("-"*70)

    print(f"\n   {'TP/SL':<10} | {'Conv':>5} | {'Trades':>6} | {'WR':>7} | {'PnL':>10} | {'MaxDD':>8} | {'MaxLoss':>8}")
    print("   " + "-"*75)

    best_v2 = None
    for tp, sl in [(0.05, 0.025), (0.06, 0.03), (0.07, 0.035), (0.04, 0.02)]:
        for conv in [0.3, 0.5, 0.7, 1.0]:
            trades = run_backtest(
                df, feat, model_data['model'], model_data['scaler'],
                model_data['feature_cols'], tp_pct=tp, sl_pct=sl,
                model_type='regressor', conv_min=conv,
                pred_std=model_data['pred_std'], start_date='2025-12-01'
            )
            if len(trades) > 0:
                result = analyze_trades(trades, f"V2 {tp*100:.0f}/{sl*100:.1f}")
                if result and result['trades'] >= 5:  # Minimo 5 trades
                    print(f"   {tp*100:.0f}%/{sl*100:.1f}%   | {conv:>5.1f} | {result['trades']:>6} | {result['wr']*100:>6.1f}% | ${result['total_pnl']:>+8.2f} | ${result['max_drawdown']:>7.2f} | {result['max_loss_streak']:>8}")
                    if best_v2 is None or result['total_pnl'] > best_v2['total_pnl']:
                        best_v2 = {**result, 'tp': tp, 'sl': sl, 'conv': conv}

    if best_v2:
        print(f"\n   MEJOR V2: TP={best_v2['tp']*100:.0f}%/SL={best_v2['sl']*100:.1f}%, Conv>={best_v2['conv']}")
        print(f"            {best_v2['trades']} trades, {best_v2['wr']*100:.1f}% WR, ${best_v2['total_pnl']:+.2f}")

    # =================================================================
    # OPCION 2: Clasificador RF - diferentes configs
    # =================================================================
    print("\n" + "-"*70)
    print("CLASIFICADOR RF - Probando diferentes configs")
    print("-"*70)

    print(f"\n   {'TP/SL':<10} | {'Prob':>5} | {'Trades':>6} | {'WR':>7} | {'PnL':>10} | {'MaxDD':>8} | {'MaxLoss':>8}")
    print("   " + "-"*75)

    best_clf = None
    for tp, sl in [(0.05, 0.025), (0.06, 0.03), (0.07, 0.035), (0.04, 0.02)]:
        for prob in [0.45, 0.50, 0.55]:
            trades = run_backtest(
                df_clean, feat_clean, rf_clf, scaler_clf, feature_cols,
                tp_pct=tp, sl_pct=sl, model_type='classifier',
                prob_threshold=prob, start_date='2025-12-01'
            )
            if len(trades) > 0:
                result = analyze_trades(trades, f"CLF {tp*100:.0f}/{sl*100:.1f}")
                if result and result['trades'] >= 5:
                    print(f"   {tp*100:.0f}%/{sl*100:.1f}%   | {prob:>5.2f} | {result['trades']:>6} | {result['wr']*100:>6.1f}% | ${result['total_pnl']:>+8.2f} | ${result['max_drawdown']:>7.2f} | {result['max_loss_streak']:>8}")
                    if best_clf is None or result['total_pnl'] > best_clf['total_pnl']:
                        best_clf = {**result, 'tp': tp, 'sl': sl, 'prob': prob}

    if best_clf:
        print(f"\n   MEJOR CLF: TP={best_clf['tp']*100:.0f}%/SL={best_clf['sl']*100:.1f}%, P>{best_clf['prob']}")
        print(f"             {best_clf['trades']} trades, {best_clf['wr']*100:.1f}% WR, ${best_clf['total_pnl']:+.2f}")

    # =================================================================
    # COMPARACION DETALLADA
    # =================================================================
    print("\n" + "="*70)
    print("COMPARACION DETALLADA: MEJOR V2 vs MEJOR CLF")
    print("="*70)

    if best_v2 and best_clf:
        print(f"\n{'Metrica':<25} | {'V2 Regresor':>15} | {'Clasificador RF':>15} | {'Mejor':>12}")
        print("-"*75)

        metrics = [
            ('Trades', best_v2['trades'], best_clf['trades'], 'neutral'),
            ('Win Rate', f"{best_v2['wr']*100:.1f}%", f"{best_clf['wr']*100:.1f}%", 'v2' if best_v2['wr'] > best_clf['wr'] else 'clf'),
            ('PnL Total', f"${best_v2['total_pnl']:+.2f}", f"${best_clf['total_pnl']:+.2f}", 'v2' if best_v2['total_pnl'] > best_clf['total_pnl'] else 'clf'),
            ('Profit Factor', f"{best_v2['profit_factor']:.2f}", f"{best_clf['profit_factor']:.2f}", 'v2' if best_v2['profit_factor'] > best_clf['profit_factor'] else 'clf'),
            ('Max Drawdown', f"${best_v2['max_drawdown']:.2f}", f"${best_clf['max_drawdown']:.2f}", 'v2' if best_v2['max_drawdown'] < best_clf['max_drawdown'] else 'clf'),
            ('Avg Win', f"${best_v2['avg_win']:+.2f}", f"${best_clf['avg_win']:+.2f}", 'v2' if best_v2['avg_win'] > best_clf['avg_win'] else 'clf'),
            ('Avg Loss', f"${best_v2['avg_loss']:.2f}", f"${best_clf['avg_loss']:.2f}", 'v2' if abs(best_v2['avg_loss']) < abs(best_clf['avg_loss']) else 'clf'),
            ('Max Racha Perdidas', best_v2['max_loss_streak'], best_clf['max_loss_streak'], 'v2' if best_v2['max_loss_streak'] < best_clf['max_loss_streak'] else 'clf'),
            ('Meses Perdedores', best_v2['losing_months'], best_clf['losing_months'], 'v2' if best_v2['losing_months'] < best_clf['losing_months'] else 'clf'),
        ]

        v2_wins = 0
        clf_wins = 0

        for name, v2_val, clf_val, winner in metrics:
            winner_str = ""
            if winner == 'v2':
                winner_str = "<-- V2"
                v2_wins += 1
            elif winner == 'clf':
                winner_str = "<-- CLF"
                clf_wins += 1
            print(f"{name:<25} | {str(v2_val):>15} | {str(clf_val):>15} | {winner_str:>12}")

        # Desglose mensual
        print("\n" + "-"*70)
        print("DESGLOSE MENSUAL")
        print("-"*70)

        print(f"\n   V2 Regresor:")
        for month, row in best_v2['monthly'].iterrows():
            print(f"      {month}: {int(row['total'])} trades, {row['wr']:.1f}% WR, ${row['pnl']:+.2f}")

        print(f"\n   Clasificador RF:")
        for month, row in best_clf['monthly'].iterrows():
            print(f"      {month}: {int(row['total'])} trades, {row['wr']:.1f}% WR, ${row['pnl']:+.2f}")

        # Riesgo
        print("\n" + "-"*70)
        print("ANALISIS DE RIESGO")
        print("-"*70)

        v2_risk_adj = best_v2['total_pnl'] / best_v2['max_drawdown'] if best_v2['max_drawdown'] > 0 else 0
        clf_risk_adj = best_clf['total_pnl'] / best_clf['max_drawdown'] if best_clf['max_drawdown'] > 0 else 0

        print(f"\n   Retorno/Riesgo (PnL/MaxDD):")
        print(f"   V2 Regresor: {v2_risk_adj:.2f}")
        print(f"   Clasificador RF: {clf_risk_adj:.2f}")

        # Scoring
        print("\n" + "="*70)
        print("RECOMENDACION FINAL SOL")
        print("="*70)

        print(f"\n   Metricas ganadas: V2={v2_wins} vs CLF={clf_wins}")

        # Score ponderado
        score_v2 = 0
        score_clf = 0

        if best_v2['total_pnl'] > best_clf['total_pnl']: score_v2 += 3
        else: score_clf += 3

        if best_v2['profit_factor'] > best_clf['profit_factor']: score_v2 += 2
        else: score_clf += 2

        if best_v2['max_drawdown'] < best_clf['max_drawdown']: score_v2 += 2
        else: score_clf += 2

        if best_v2['wr'] > best_clf['wr']: score_v2 += 1
        else: score_clf += 1

        if best_v2['max_loss_streak'] < best_clf['max_loss_streak']: score_v2 += 2
        else: score_clf += 2

        print(f"\n   Score Ponderado:")
        print(f"   V2 Regresor: {score_v2} puntos")
        print(f"   Clasificador RF: {score_clf} puntos")

        if score_clf > score_v2:
            print(f"\n   >>> RECOMENDACION: Clasificador RF")
            print(f"   Config: TP={best_clf['tp']*100:.0f}%/SL={best_clf['sl']*100:.1f}%, P>{best_clf['prob']}")
            print(f"   {best_clf['trades']} trades, {best_clf['wr']*100:.1f}% WR, ${best_clf['total_pnl']:+.2f}")

            # Guardar modelo
            export_data = {
                'model': rf_clf,
                'scaler': scaler_clf,
                'feature_cols': feature_cols,
                'model_type': 'classifier',
                'prob_threshold': best_clf['prob'],
                'direction': 'short',
                'tp_pct': best_clf['tp'],
                'sl_pct': best_clf['sl'],
                'backtest_pnl': best_clf['total_pnl'],
                'backtest_wr': best_clf['wr'],
                'backtest_trades': best_clf['trades'],
                'max_drawdown': best_clf['max_drawdown'],
                'max_loss_streak': best_clf['max_loss_streak'],
                'profit_factor': best_clf['profit_factor'],
            }
            joblib.dump(export_data, MODELS_DIR / 'sol_usdt_v3_classifier.pkl')
            print(f"\n   Modelo guardado: models/sol_usdt_v3_classifier.pkl")

        elif score_v2 > score_clf:
            print(f"\n   >>> RECOMENDACION: V2 Regresor")
            print(f"   Config: TP={best_v2['tp']*100:.0f}%/SL={best_v2['sl']*100:.1f}%, Conv>={best_v2['conv']}")
            print(f"   {best_v2['trades']} trades, {best_v2['wr']*100:.1f}% WR, ${best_v2['total_pnl']:+.2f}")
        else:
            print(f"\n   >>> EMPATE - Revisar manualmente")

        # Comparacion final con baseline original
        print("\n" + "-"*70)
        print("vs BASELINE ORIGINAL (V2 conv>=1.0, 5%/2.5%)")
        print("-"*70)

        baseline_trades = run_backtest(
            df, feat, model_data['model'], model_data['scaler'],
            model_data['feature_cols'], tp_pct=0.05, sl_pct=0.025,
            model_type='regressor', conv_min=1.0,
            pred_std=model_data['pred_std'], start_date='2025-12-01'
        )
        if len(baseline_trades) > 0:
            baseline = analyze_trades(baseline_trades, "Baseline")
            print(f"\n   Baseline: {baseline['trades']} trades, {baseline['wr']*100:.1f}% WR, ${baseline['total_pnl']:+.2f}")

            winner = best_clf if score_clf >= score_v2 else best_v2
            print(f"   Ganador:  {winner['trades']} trades, {winner['wr']*100:.1f}% WR, ${winner['total_pnl']:+.2f}")
            print(f"\n   Mejora: ${winner['total_pnl'] - baseline['total_pnl']:+.2f} ({(winner['total_pnl']/baseline['total_pnl']-1)*100:+.1f}%)")
