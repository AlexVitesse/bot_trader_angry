"""
Experimentos de mejora para SOL y BNB
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path('data')
MODELS_DIR = Path('models')


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """54 features."""
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


def backtest(df, feat, model, scaler, feature_cols, pred_std,
             tp_pct, sl_pct, conv_min=1.0,
             only_short=False, only_long=False,
             only_bear=False, only_bull=False,
             start_date=None):
    """Backtest con filtros opcionales."""

    X = feat[feature_cols].copy()
    if start_date:
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

        # Filtros de direccion
        if only_short and direction == 1:
            continue
        if only_long and direction == -1:
            continue

        # Filtros de regimen
        if only_bear and reg != 'BEAR':
            continue
        if only_bull and reg != 'BULL':
            continue

        # Regime filter estandar
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
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_pct * 100,
        })

    if not trades:
        return {'trades': 0, 'wr': 0, 'pnl': 0}

    trades_df = pd.DataFrame(trades)
    wins = (trades_df['pnl_pct'] > 0).sum()
    total = len(trades_df)
    wr = wins / total
    pnl = trades_df['pnl_usd'].sum()

    return {
        'trades': total,
        'wr': wr,
        'pnl': pnl,
        'trades_df': trades_df,
    }


def load_model(pair: str):
    """Carga modelo V2."""
    safe = pair.replace('/', '_').lower()
    data = joblib.load(MODELS_DIR / f'{safe}_v2_gradientboosting.pkl')
    return data['model'], data['scaler'], data['feature_cols'], data['pred_std']


def experiment_bnb():
    """Experimentos de mejora para BNB."""
    print("\n" + "="*70)
    print("BNB/USDT - EXPERIMENTOS DE MEJORA")
    print("="*70)

    # Cargar datos y modelo
    df = pd.read_parquet(DATA_DIR / 'BNB_USDT_4h_full.parquet')
    feat = compute_features(df)
    model, scaler, feature_cols, pred_std = load_model('BNB/USDT')

    start = '2025-12-01'
    tp, sl = 0.05, 0.025

    print(f"\nPeriodo: {start} a hoy")
    print(f"TP={tp*100}%, SL={sl*100}%")

    # Baseline
    print("\n[1] BASELINE (sin filtros)")
    res = backtest(df, feat, model, scaler, feature_cols, pred_std,
                   tp, sl, conv_min=1.0, start_date=start)
    print(f"    {res['trades']} trades | {res['wr']*100:.1f}% WR | ${res['pnl']:+.2f}")
    baseline_pnl = res['pnl']

    # Solo SHORT
    print("\n[2] SOLO SHORT")
    res = backtest(df, feat, model, scaler, feature_cols, pred_std,
                   tp, sl, conv_min=1.0, only_short=True, start_date=start)
    print(f"    {res['trades']} trades | {res['wr']*100:.1f}% WR | ${res['pnl']:+.2f}")
    if res['pnl'] > baseline_pnl:
        print(f"    [OK] Mejora: +${res['pnl'] - baseline_pnl:.2f}")

    # Solo BEAR
    print("\n[3] SOLO BEAR REGIME")
    res = backtest(df, feat, model, scaler, feature_cols, pred_std,
                   tp, sl, conv_min=1.0, only_bear=True, start_date=start)
    print(f"    {res['trades']} trades | {res['wr']*100:.1f}% WR | ${res['pnl']:+.2f}")
    if res['pnl'] > baseline_pnl:
        print(f"    [OK] Mejora: +${res['pnl'] - baseline_pnl:.2f}")

    # SHORT + conviction variado
    print("\n[4] SOLO SHORT con diferentes conviction")
    for conv in [0.7, 1.0, 1.2, 1.5]:
        res = backtest(df, feat, model, scaler, feature_cols, pred_std,
                       tp, sl, conv_min=conv, only_short=True, start_date=start)
        print(f"    Conv >= {conv}: {res['trades']} trades | {res['wr']*100:.1f}% WR | ${res['pnl']:+.2f}")

    # SHORT + diferentes TP/SL
    print("\n[5] SOLO SHORT con diferentes TP/SL")
    for tp_test, sl_test in [(0.04, 0.02), (0.05, 0.025), (0.06, 0.03), (0.04, 0.025)]:
        res = backtest(df, feat, model, scaler, feature_cols, pred_std,
                       tp_test, sl_test, conv_min=1.0, only_short=True, start_date=start)
        print(f"    TP={tp_test*100}%/SL={sl_test*100}%: {res['trades']} trades | {res['wr']*100:.1f}% WR | ${res['pnl']:+.2f}")


def experiment_sol():
    """Experimentos de mejora para SOL."""
    print("\n" + "="*70)
    print("SOL/USDT - EXPERIMENTOS DE MEJORA")
    print("="*70)

    # Cargar datos y modelo
    df = pd.read_parquet(DATA_DIR / 'SOL_USDT_4h_full.parquet')
    feat = compute_features(df)
    model, scaler, feature_cols, pred_std = load_model('SOL/USDT')

    start = '2025-12-01'
    tp, sl = 0.05, 0.025

    print(f"\nPeriodo: {start} a hoy")
    print(f"TP={tp*100}%, SL={sl*100}%")

    # Baseline
    print("\n[1] BASELINE (conv >= 1.0)")
    res = backtest(df, feat, model, scaler, feature_cols, pred_std,
                   tp, sl, conv_min=1.0, start_date=start)
    print(f"    {res['trades']} trades | {res['wr']*100:.1f}% WR | ${res['pnl']:+.2f}")
    baseline_pnl = res['pnl']

    # Solo SHORT
    print("\n[2] SOLO SHORT")
    res = backtest(df, feat, model, scaler, feature_cols, pred_std,
                   tp, sl, conv_min=1.0, only_short=True, start_date=start)
    print(f"    {res['trades']} trades | {res['wr']*100:.1f}% WR | ${res['pnl']:+.2f}")

    # Conviction mas bajo
    print("\n[3] CONVICTION MAS BAJO")
    for conv in [0.3, 0.5, 0.7]:
        res = backtest(df, feat, model, scaler, feature_cols, pred_std,
                       tp, sl, conv_min=conv, start_date=start)
        print(f"    Conv >= {conv}: {res['trades']} trades | {res['wr']*100:.1f}% WR | ${res['pnl']:+.2f}")

    # Conviction bajo + solo SHORT
    print("\n[4] CONV BAJO + SOLO SHORT")
    for conv in [0.3, 0.5, 0.7]:
        res = backtest(df, feat, model, scaler, feature_cols, pred_std,
                       tp, sl, conv_min=conv, only_short=True, start_date=start)
        print(f"    Conv >= {conv} SHORT: {res['trades']} trades | {res['wr']*100:.1f}% WR | ${res['pnl']:+.2f}")

    # Diferentes TP/SL
    print("\n[5] DIFERENTES TP/SL (conv >= 0.5)")
    for tp_test, sl_test in [(0.03, 0.015), (0.04, 0.02), (0.06, 0.03), (0.08, 0.04)]:
        res = backtest(df, feat, model, scaler, feature_cols, pred_std,
                       tp_test, sl_test, conv_min=0.5, start_date=start)
        print(f"    TP={tp_test*100}%/SL={sl_test*100}%: {res['trades']} trades | {res['wr']*100:.1f}% WR | ${res['pnl']:+.2f}")

    # Solo BEAR
    print("\n[6] SOLO BEAR REGIME")
    res = backtest(df, feat, model, scaler, feature_cols, pred_std,
                   tp, sl, conv_min=0.5, only_bear=True, start_date=start)
    print(f"    {res['trades']} trades | {res['wr']*100:.1f}% WR | ${res['pnl']:+.2f}")

    # Entrenar modelo solo con datos recientes
    print("\n[7] MODELO ENTRENADO SOLO 2024-2026")
    train_recent_model_sol(df)


def train_recent_model_sol(df):
    """Entrena modelo SOL solo con datos 2024-2026."""
    # Filtrar datos recientes
    df_recent = df[df.index >= '2024-01-01'].copy()
    feat = compute_features(df_recent)

    target = df_recent['close'].pct_change(5).shift(-5)
    valid = feat.notna().all(axis=1) & target.notna()
    feat = feat[valid]
    target = target[valid]

    # Split
    train_end = '2025-10-31'
    train_mask = feat.index <= train_end
    test_mask = feat.index > train_end

    X_train, y_train = feat[train_mask], target[train_mask]
    X_test, y_test = feat[test_mask], target[test_mask]

    print(f"    Train: {len(X_train)} | Test: {len(X_test)}")

    # Entrenar
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train_s, y_train)

    pred_train = model.predict(X_train_s)
    pred_std = np.std(pred_train)

    # Backtest con modelo reciente
    df_test = df[df.index > train_end].copy()
    feat_test = compute_features(df_test)

    feature_cols = list(feat.columns)

    for tp, sl in [(0.05, 0.025), (0.04, 0.02), (0.03, 0.015)]:
        res = backtest(df_test, feat_test, model, scaler, feature_cols, pred_std,
                       tp, sl, conv_min=0.5)
        print(f"    TP={tp*100}%/SL={sl*100}%: {res['trades']} trades | {res['wr']*100:.1f}% WR | ${res['pnl']:+.2f}")


if __name__ == '__main__':
    experiment_bnb()
    experiment_sol()

    print("\n" + "="*70)
    print("EXPERIMENTOS COMPLETADOS")
    print("="*70)
