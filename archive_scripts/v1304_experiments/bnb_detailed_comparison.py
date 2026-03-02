"""
BNB - Comparacion Detallada: V2 Regresor vs Clasificador RF
Analisis: Drawdown, consistencia mensual, racha de perdidas
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
                 tp_pct, sl_pct, model_type='regressor',
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
        directions = np.where(preds < 0, -1, 1)  # SHORT if pred < 0
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

    # Basic metrics
    total = len(trades_df)
    wins = trades_df['win'].sum()
    losses = total - wins
    wr = wins / total
    total_pnl = trades_df['pnl_usd'].sum()

    # Profit Factor
    gross_profit = trades_df[trades_df['pnl_usd'] > 0]['pnl_usd'].sum()
    gross_loss = abs(trades_df[trades_df['pnl_usd'] < 0]['pnl_usd'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Drawdown calculation
    cumulative = trades_df['pnl_usd'].cumsum()
    running_max = cumulative.cummax()
    drawdown = running_max - cumulative
    max_dd = drawdown.max()

    # Average win/loss
    avg_win = trades_df[trades_df['win']]['pnl_usd'].mean() if wins > 0 else 0
    avg_loss = trades_df[~trades_df['win']]['pnl_usd'].mean() if losses > 0 else 0

    # Consecutive losses
    trades_df['loss_streak'] = (~trades_df['win']).astype(int)
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

    # Monthly breakdown
    trades_df['month'] = trades_df['entry_time'].dt.strftime('%Y-%m')
    monthly = trades_df.groupby('month').agg({
        'pnl_usd': 'sum',
        'win': ['sum', 'count']
    })
    monthly.columns = ['pnl', 'wins', 'total']
    monthly['wr'] = monthly['wins'] / monthly['total'] * 100

    # Months with loss
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
    print("BNB - COMPARACION DETALLADA")
    print("V2 Regresor (87.5% WR) vs Clasificador RF (59% WR)")
    print("="*70)

    # Cargar datos
    df = pd.read_parquet(DATA_DIR / 'BNB_USDT_4h_full.parquet')
    feat = compute_features(df)
    feature_cols = list(feat.columns)

    # Cargar modelo V2
    model_data = joblib.load(MODELS_DIR / 'bnb_usdt_v2_gradientboosting.pkl')

    # Entrenar clasificador RF (mismo que antes)
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
    # OPCION 1: V2 Regresor TP=7%/SL=3.5%
    # =================================================================
    print("\n" + "-"*70)
    print("OPCION 1: V2 Regresor (TP=7%/SL=3.5%, Conv>=1.0)")
    print("-"*70)

    trades_v2 = run_backtest(
        df, feat, model_data['model'], model_data['scaler'],
        model_data['feature_cols'], tp_pct=0.07, sl_pct=0.035,
        model_type='regressor', conv_min=1.0,
        pred_std=model_data['pred_std'], start_date='2025-12-01'
    )

    result_v2 = analyze_trades(trades_v2, "V2 Regresor")

    if result_v2:
        print(f"\n   Trades: {result_v2['trades']} ({result_v2['wins']}W / {result_v2['losses']}L)")
        print(f"   Win Rate: {result_v2['wr']*100:.1f}%")
        print(f"   PnL Total: ${result_v2['total_pnl']:+.2f}")
        print(f"   Profit Factor: {result_v2['profit_factor']:.2f}")
        print(f"   Max Drawdown: ${result_v2['max_drawdown']:.2f}")
        print(f"   Avg Win: ${result_v2['avg_win']:+.2f}")
        print(f"   Avg Loss: ${result_v2['avg_loss']:.2f}")
        print(f"   Max Racha Perdedora: {result_v2['max_loss_streak']} trades")
        print(f"   Meses Perdedores: {result_v2['losing_months']}")

        print(f"\n   Por Mes:")
        print(f"   {'Mes':<10} | {'Trades':>6} | {'WR':>7} | {'PnL':>10}")
        print("   " + "-"*40)
        for month, row in result_v2['monthly'].iterrows():
            print(f"   {month:<10} | {int(row['total']):>6} | {row['wr']:>6.1f}% | ${row['pnl']:>+8.2f}")

    # =================================================================
    # OPCION 2: Clasificador RF TP=6%/SL=3%
    # =================================================================
    print("\n" + "-"*70)
    print("OPCION 2: Clasificador RF (TP=6%/SL=3%, P>0.5)")
    print("-"*70)

    trades_clf = run_backtest(
        df_clean, feat_clean, rf_clf, scaler_clf, feature_cols,
        tp_pct=0.06, sl_pct=0.03, model_type='classifier',
        prob_threshold=0.5, start_date='2025-12-01'
    )

    result_clf = analyze_trades(trades_clf, "Clasificador RF")

    if result_clf:
        print(f"\n   Trades: {result_clf['trades']} ({result_clf['wins']}W / {result_clf['losses']}L)")
        print(f"   Win Rate: {result_clf['wr']*100:.1f}%")
        print(f"   PnL Total: ${result_clf['total_pnl']:+.2f}")
        print(f"   Profit Factor: {result_clf['profit_factor']:.2f}")
        print(f"   Max Drawdown: ${result_clf['max_drawdown']:.2f}")
        print(f"   Avg Win: ${result_clf['avg_win']:+.2f}")
        print(f"   Avg Loss: ${result_clf['avg_loss']:.2f}")
        print(f"   Max Racha Perdedora: {result_clf['max_loss_streak']} trades")
        print(f"   Meses Perdedores: {result_clf['losing_months']}")

        print(f"\n   Por Mes:")
        print(f"   {'Mes':<10} | {'Trades':>6} | {'WR':>7} | {'PnL':>10}")
        print("   " + "-"*40)
        for month, row in result_clf['monthly'].iterrows():
            print(f"   {month:<10} | {int(row['total']):>6} | {row['wr']:>6.1f}% | ${row['pnl']:>+8.2f}")

    # =================================================================
    # COMPARACION LADO A LADO
    # =================================================================
    print("\n" + "="*70)
    print("COMPARACION LADO A LADO")
    print("="*70)

    print(f"\n{'Metrica':<25} | {'V2 Regresor':>15} | {'Clasificador RF':>15} | {'Mejor':>12}")
    print("-"*75)

    metrics = [
        ('Trades', result_v2['trades'], result_clf['trades'], 'neutral'),
        ('Win Rate', f"{result_v2['wr']*100:.1f}%", f"{result_clf['wr']*100:.1f}%", 'v2' if result_v2['wr'] > result_clf['wr'] else 'clf'),
        ('PnL Total', f"${result_v2['total_pnl']:+.2f}", f"${result_clf['total_pnl']:+.2f}", 'v2' if result_v2['total_pnl'] > result_clf['total_pnl'] else 'clf'),
        ('Profit Factor', f"{result_v2['profit_factor']:.2f}", f"{result_clf['profit_factor']:.2f}", 'v2' if result_v2['profit_factor'] > result_clf['profit_factor'] else 'clf'),
        ('Max Drawdown', f"${result_v2['max_drawdown']:.2f}", f"${result_clf['max_drawdown']:.2f}", 'v2' if result_v2['max_drawdown'] < result_clf['max_drawdown'] else 'clf'),
        ('Avg Win', f"${result_v2['avg_win']:+.2f}", f"${result_clf['avg_win']:+.2f}", 'v2' if result_v2['avg_win'] > result_clf['avg_win'] else 'clf'),
        ('Avg Loss', f"${result_v2['avg_loss']:.2f}", f"${result_clf['avg_loss']:.2f}", 'v2' if abs(result_v2['avg_loss']) < abs(result_clf['avg_loss']) else 'clf'),
        ('Max Racha Perdidas', result_v2['max_loss_streak'], result_clf['max_loss_streak'], 'v2' if result_v2['max_loss_streak'] < result_clf['max_loss_streak'] else 'clf'),
        ('Meses Perdedores', result_v2['losing_months'], result_clf['losing_months'], 'v2' if result_v2['losing_months'] < result_clf['losing_months'] else 'clf'),
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

    # =================================================================
    # ANALISIS DE RIESGO
    # =================================================================
    print("\n" + "="*70)
    print("ANALISIS DE RIESGO")
    print("="*70)

    # Simulacion: que pasa si tenemos una mala racha?
    print("\n   Escenario: 5 trades consecutivos perdedores")
    v2_loss_5 = 5 * abs(result_v2['avg_loss'])
    clf_loss_5 = 5 * abs(result_clf['avg_loss'])
    print(f"   V2 Regresor: -${v2_loss_5:.2f}")
    print(f"   Clasificador RF: -${clf_loss_5:.2f}")

    # Recovery: cuantos wins necesitas para recuperar?
    v2_recovery = v2_loss_5 / result_v2['avg_win'] if result_v2['avg_win'] > 0 else 0
    clf_recovery = clf_loss_5 / result_clf['avg_win'] if result_clf['avg_win'] > 0 else 0
    print(f"\n   Trades ganadores para recuperar:")
    print(f"   V2 Regresor: {v2_recovery:.1f} trades")
    print(f"   Clasificador RF: {clf_recovery:.1f} trades")

    # Risk-adjusted return (PnL / MaxDD)
    v2_risk_adj = result_v2['total_pnl'] / result_v2['max_drawdown'] if result_v2['max_drawdown'] > 0 else 0
    clf_risk_adj = result_clf['total_pnl'] / result_clf['max_drawdown'] if result_clf['max_drawdown'] > 0 else 0
    print(f"\n   Retorno Ajustado al Riesgo (PnL/MaxDD):")
    print(f"   V2 Regresor: {v2_risk_adj:.2f}")
    print(f"   Clasificador RF: {clf_risk_adj:.2f}")

    # =================================================================
    # RECOMENDACION FINAL
    # =================================================================
    print("\n" + "="*70)
    print("RECOMENDACION FINAL")
    print("="*70)

    print(f"\n   Metricas ganadas: V2={v2_wins} vs CLF={clf_wins}")

    # Scoring
    score_v2 = 0
    score_clf = 0

    # PnL (peso 3)
    if result_v2['total_pnl'] > result_clf['total_pnl']:
        score_v2 += 3
    else:
        score_clf += 3

    # Profit Factor (peso 2)
    if result_v2['profit_factor'] > result_clf['profit_factor']:
        score_v2 += 2
    else:
        score_clf += 2

    # Max Drawdown (peso 2)
    if result_v2['max_drawdown'] < result_clf['max_drawdown']:
        score_v2 += 2
    else:
        score_clf += 2

    # Win Rate (peso 1)
    if result_v2['wr'] > result_clf['wr']:
        score_v2 += 1
    else:
        score_clf += 1

    # Max Loss Streak (peso 2)
    if result_v2['max_loss_streak'] < result_clf['max_loss_streak']:
        score_v2 += 2
    else:
        score_clf += 2

    print(f"\n   Score Ponderado:")
    print(f"   V2 Regresor: {score_v2} puntos")
    print(f"   Clasificador RF: {score_clf} puntos")

    if score_v2 > score_clf:
        print(f"\n   >>> RECOMENDACION: V2 Regresor (TP=7%/SL=3.5%)")
        print(f"   Razones:")
        print(f"   - Mayor WR ({result_v2['wr']*100:.1f}%) = menos estres")
        print(f"   - Menor drawdown = protege capital")
        print(f"   - Menor racha perdedora = mas estable")
    elif score_clf > score_v2:
        print(f"\n   >>> RECOMENDACION: Clasificador RF (TP=6%/SL=3%)")
        print(f"   Razones:")
        print(f"   - Mayor PnL total (${result_clf['total_pnl']:+.2f})")
        print(f"   - Mas trades = mas oportunidades")
    else:
        print(f"\n   >>> EMPATE - Ambos son buenos, depende de tu preferencia:")
        print(f"   - V2: Menos trades, mas conservador")
        print(f"   - CLF: Mas trades, mas agresivo")
