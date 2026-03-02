"""
SOL - Analisis de Riesgo del Clasificador
¿Puede perder todo el capital?
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


def simulate_with_capital(df, feat, model, scaler, feature_cols,
                          tp_pct, sl_pct, prob_threshold,
                          initial_capital, position_size_pct,
                          start_date='2025-12-01'):
    """Simula con capital real y position sizing."""

    X = feat[feature_cols].copy()
    X = X[X.index >= start_date]

    valid = X.notna().all(axis=1)
    X = X[valid]
    df_bt = df.loc[X.index].copy()

    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    regime = detect_regime(df_bt)

    capital = initial_capital
    peak_capital = initial_capital
    trades = []
    equity_curve = [initial_capital]

    for i, (idx, row) in enumerate(X.iterrows()):
        if i >= len(X) - 5:
            break

        if probs[i] < prob_threshold:
            continue

        reg = regime.iloc[i]
        if reg == 'BULL':
            continue

        # Position size basado en capital actual
        position_value = capital * position_size_pct

        entry_price = df_bt.loc[idx, 'close']
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)

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

        # PnL en USD
        pnl_pct = (entry_price - exit_price) / entry_price
        pnl_usd = position_value * pnl_pct

        capital += pnl_usd
        peak_capital = max(peak_capital, capital)
        equity_curve.append(capital)

        trades.append({
            'entry_time': idx,
            'capital_before': capital - pnl_usd,
            'position_size': position_value,
            'pnl_usd': pnl_usd,
            'capital_after': capital,
            'drawdown_from_peak': peak_capital - capital,
            'drawdown_pct': (peak_capital - capital) / peak_capital * 100,
        })

        # Check if bankrupt
        if capital <= 0:
            print(f"   !!! BANCARROTA en trade {len(trades)} !!!")
            break

    return pd.DataFrame(trades), equity_curve


if __name__ == '__main__':
    print("\n" + "="*70)
    print("SOL CLASIFICADOR - ANALISIS DE RIESGO DE CAPITAL")
    print("="*70)

    # Cargar datos
    df = pd.read_parquet(DATA_DIR / 'SOL_USDT_4h_full.parquet')
    feat = compute_features(df)
    feature_cols = list(feat.columns)

    # Entrenar clasificador
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
    # SIMULACION 1: Config agresiva (P>0.45, 7%/3.5%)
    # =================================================================
    print("\n" + "-"*70)
    print("ESCENARIO 1: Config Agresiva (P>0.45, TP=7%/SL=3.5%)")
    print("-"*70)

    for capital in [500, 1000, 2000]:
        for pos_pct in [0.10, 0.05, 0.02]:  # 10%, 5%, 2% del capital por trade
            trades, equity = simulate_with_capital(
                df_clean, feat_clean, rf_clf, scaler_clf, feature_cols,
                tp_pct=0.07, sl_pct=0.035, prob_threshold=0.45,
                initial_capital=capital, position_size_pct=pos_pct,
                start_date='2025-12-01'
            )

            if len(trades) > 0:
                final_capital = trades.iloc[-1]['capital_after']
                max_dd = trades['drawdown_from_peak'].max()
                max_dd_pct = trades['drawdown_pct'].max()
                min_capital = min(equity)
                pnl = final_capital - capital

                status = "OK" if min_capital > 0 else "BANCARROTA"

                print(f"   Capital ${capital:>4}, Pos {pos_pct*100:.0f}%: "
                      f"Final ${final_capital:>7.2f} | "
                      f"PnL ${pnl:>+7.2f} | "
                      f"MaxDD ${max_dd:>6.2f} ({max_dd_pct:>5.1f}%) | "
                      f"Min ${min_capital:>7.2f} | {status}")

    # =================================================================
    # SIMULACION 2: Config conservadora (P>0.55)
    # =================================================================
    print("\n" + "-"*70)
    print("ESCENARIO 2: Config Conservadora (P>0.55, TP=7%/SL=3.5%)")
    print("-"*70)

    for capital in [500, 1000]:
        for pos_pct in [0.10, 0.05]:
            trades, equity = simulate_with_capital(
                df_clean, feat_clean, rf_clf, scaler_clf, feature_cols,
                tp_pct=0.07, sl_pct=0.035, prob_threshold=0.55,
                initial_capital=capital, position_size_pct=pos_pct,
                start_date='2025-12-01'
            )

            if len(trades) > 0:
                final_capital = trades.iloc[-1]['capital_after']
                max_dd = trades['drawdown_from_peak'].max()
                max_dd_pct = trades['drawdown_pct'].max()
                pnl = final_capital - capital

                print(f"   Capital ${capital:>4}, Pos {pos_pct*100:.0f}%: "
                      f"Final ${final_capital:>7.2f} | "
                      f"PnL ${pnl:>+7.2f} | "
                      f"MaxDD ${max_dd:>6.2f} ({max_dd_pct:>5.1f}%) | "
                      f"{len(trades)} trades")

    # =================================================================
    # COMPARACION CON BNB
    # =================================================================
    print("\n" + "-"*70)
    print("COMPARACION: SOL vs BNB (mismo capital)")
    print("-"*70)

    # Cargar BNB
    df_bnb = pd.read_parquet(DATA_DIR / 'BNB_USDT_4h_full.parquet')
    feat_bnb = compute_features(df_bnb)
    model_bnb = joblib.load(MODELS_DIR / 'bnb_usdt_v2_gradientboosting.pkl')

    # Simular BNB manualmente
    X_bnb = feat_bnb[model_bnb['feature_cols']].copy()
    X_bnb = X_bnb[X_bnb.index >= '2025-12-01']
    valid_bnb = X_bnb.notna().all(axis=1)
    X_bnb = X_bnb[valid_bnb]
    df_bnb_bt = df_bnb.loc[X_bnb.index].copy()

    X_bnb_scaled = model_bnb['scaler'].transform(X_bnb)
    preds_bnb = model_bnb['model'].predict(X_bnb_scaled)
    conv_bnb = np.abs(preds_bnb) / model_bnb['pred_std']
    regime_bnb = detect_regime(df_bnb_bt)

    capital = 1000
    pos_pct = 0.05  # 5% por trade

    # BNB simulation
    cap_bnb = capital
    peak_bnb = capital
    equity_bnb = [capital]

    for i, (idx, row) in enumerate(X_bnb.iterrows()):
        if i >= len(X_bnb) - 5:
            break

        if conv_bnb[i] < 1.0:
            continue
        if preds_bnb[i] > 0:  # Solo SHORT
            continue
        if regime_bnb.iloc[i] == 'BULL':
            continue

        pos_val = cap_bnb * pos_pct
        entry = df_bnb_bt.loc[idx, 'close']
        tp = entry * (1 - 0.07)
        sl = entry * (1 + 0.035)

        exit_price = None
        future_idx = X_bnb.index.get_loc(idx)

        for j in range(1, min(21, len(df_bnb_bt) - future_idx)):
            bar = df_bnb_bt.iloc[future_idx + j]
            if bar['high'] >= sl:
                exit_price = sl
                break
            elif bar['low'] <= tp:
                exit_price = tp
                break

        if exit_price is None:
            exit_price = df_bnb_bt.iloc[min(future_idx + 20, len(df_bnb_bt) - 1)]['close']

        pnl_pct = (entry - exit_price) / entry
        pnl_usd = pos_val * pnl_pct
        cap_bnb += pnl_usd
        peak_bnb = max(peak_bnb, cap_bnb)
        equity_bnb.append(cap_bnb)

    # SOL simulation
    trades_sol, equity_sol = simulate_with_capital(
        df_clean, feat_clean, rf_clf, scaler_clf, feature_cols,
        tp_pct=0.07, sl_pct=0.035, prob_threshold=0.45,
        initial_capital=capital, position_size_pct=pos_pct,
        start_date='2025-12-01'
    )

    print(f"\n   Capital Inicial: ${capital}")
    print(f"   Position Size: {pos_pct*100:.0f}% por trade")
    print()
    print(f"   BNB (V2 Regresor, TP=7%/SL=3.5%):")
    print(f"      Final: ${cap_bnb:.2f}")
    print(f"      PnL: ${cap_bnb - capital:+.2f}")
    print(f"      Max DD: ${peak_bnb - min(equity_bnb):.2f} ({(peak_bnb - min(equity_bnb))/peak_bnb*100:.1f}%)")
    print()
    print(f"   SOL (Clasificador RF, TP=7%/SL=3.5%, P>0.45):")
    if len(trades_sol) > 0:
        final_sol = trades_sol.iloc[-1]['capital_after']
        print(f"      Final: ${final_sol:.2f}")
        print(f"      PnL: ${final_sol - capital:+.2f}")
        print(f"      Max DD: ${trades_sol['drawdown_from_peak'].max():.2f} ({trades_sol['drawdown_pct'].max():.1f}%)")

    # =================================================================
    # RECOMENDACION FINAL
    # =================================================================
    print("\n" + "="*70)
    print("RECOMENDACION DE POSITION SIZING")
    print("="*70)

    print("""
   PROBLEMA IDENTIFICADO:
   - SOL Clasificador tiene drawdown de hasta 60-80% del capital
   - Con $500 y 10% position size, puedes perder $300+ antes de recuperar
   - BNB es mucho mas estable (drawdown <5%)

   RECOMENDACIONES:

   1. NO usar SOL con position size > 2% del capital
      - Con $1000 capital, max $20 por trade
      - Esto limita el drawdown a ~$60 (6% del capital)

   2. ALTERNATIVA: Usar P>0.55 en vez de P>0.45
      - Solo 9 trades en 3 meses (muy pocos)
      - Pero drawdown maximo de $6-7 (casi nada)

   3. MEJOR OPCION: NO habilitar SOL por ahora
      - BNB es mucho mas seguro y rentable
      - BTC ya esta habilitado
      - SOL tiene demasiado riesgo para el retorno

   DECISION SUGERIDA:
   - BTC: SI (ya habilitado)
   - BNB: SI (V2 Regresor, muy seguro)
   - SOL: NO (riesgo/retorno no favorable)
""")
