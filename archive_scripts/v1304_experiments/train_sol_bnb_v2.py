"""
Entrenar modelos V2 para SOL y BNB
Mismo proceso que BTC: GradientBoosting con 54 features + optimizacion TP/SL
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import ccxt
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

# =============================================================================
# FUNCIONES
# =============================================================================

def download_data(pair: str, timeframe: str = '4h', limit: int = 5000) -> pd.DataFrame:
    """Descarga datos historicos de Binance."""
    print(f"\n[1/5] Descargando datos {pair}...")

    exchange = ccxt.binance({'enableRateLimit': True})

    all_data = []
    since = exchange.parse8601('2019-01-01T00:00:00Z')

    while True:
        ohlcv = exchange.fetch_ohlcv(pair, timeframe, since=since, limit=1000)
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < 1000:
            break

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')].sort_index()

    # Guardar
    safe_pair = pair.replace('/', '_')
    df.to_parquet(DATA_DIR / f'{safe_pair}_4h_full.parquet')
    print(f"   Descargadas {len(df)} velas desde {df.index[0]} hasta {df.index[-1]}")

    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """54 features igual que BTC V2."""
    feat = pd.DataFrame(index=df.index)
    c, h, l, o, v = df['close'], df['high'], df['low'], df['open'], df['volume']

    # Returns (7)
    for p in [1, 2, 3, 5, 10, 20, 50]:
        feat[f'ret_{p}'] = c.pct_change(p)

    # ATR (2)
    feat['atr14'] = ta.atr(h, l, c, length=14)
    feat['atr_r'] = feat['atr14'] / feat['atr14'].rolling(50).mean()

    # Volatility (3)
    feat['vol5'] = c.pct_change().rolling(5).std()
    feat['vol20'] = c.pct_change().rolling(20).std()
    feat['vol_ratio'] = feat['vol5'] / (feat['vol20'] + 1e-10)

    # RSI (3)
    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)
    feat['rsi21'] = ta.rsi(c, length=21)

    # StochRSI (2)
    sr = ta.stochrsi(c, length=14, rsi_length=14, k=3, d=3)
    if sr is not None:
        feat['srsi_k'] = sr.iloc[:, 0]
        feat['srsi_d'] = sr.iloc[:, 1]

    # MACD (3)
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        feat['macd'] = macd.iloc[:, 0]
        feat['macd_h'] = macd.iloc[:, 1]
        feat['macd_s'] = macd.iloc[:, 2]

    # ROC (3)
    feat['roc5'] = ta.roc(c, length=5)
    feat['roc10'] = ta.roc(c, length=10)
    feat['roc20'] = ta.roc(c, length=20)

    # EMA distance (5)
    for el in [8, 21, 55, 100, 200]:
        e = ta.ema(c, length=el)
        feat[f'ema{el}_d'] = (c - e) / e * 100

    # EMA slopes (3)
    feat['ema8_sl'] = ta.ema(c, length=8).pct_change(3) * 100
    feat['ema21_sl'] = ta.ema(c, length=21).pct_change(5) * 100
    feat['ema55_sl'] = ta.ema(c, length=55).pct_change(5) * 100

    # Bollinger Bands (2)
    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        bw = bb.iloc[:, 2] - bb.iloc[:, 0]
        feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw
        feat['bb_w'] = bw / bb.iloc[:, 1] * 100

    # Volume (2)
    feat['vr'] = v / v.rolling(20).mean()
    feat['vr5'] = v.rolling(5).mean() / v.rolling(20).mean()

    # Candle patterns (4)
    feat['spr'] = (h - l) / c * 100
    feat['body'] = abs(c - o) / (h - l + 1e-10)
    feat['upper_wick'] = (h - np.maximum(c, o)) / (h - l + 1e-10)
    feat['lower_wick'] = (np.minimum(c, o) - l) / (h - l + 1e-10)

    # ADX (4)
    ax = ta.adx(h, l, c, length=14)
    if ax is not None:
        feat['adx'] = ax.iloc[:, 0]
        feat['dip'] = ax.iloc[:, 1]
        feat['dim'] = ax.iloc[:, 2]
        feat['di_diff'] = feat['dip'] - feat['dim']

    # Choppiness
    chop = ta.chop(h, l, c, length=14)
    if chop is not None:
        feat['chop'] = chop

    # Time features (4)
    hr = df.index.hour
    dw = df.index.dayofweek
    feat['h_s'] = np.sin(2 * np.pi * hr / 24)
    feat['h_c'] = np.cos(2 * np.pi * hr / 24)
    feat['d_s'] = np.sin(2 * np.pi * dw / 7)
    feat['d_c'] = np.cos(2 * np.pi * dw / 7)

    # Lag features (6)
    feat['ret1_lag1'] = feat['ret_1'].shift(1)
    feat['rsi14_lag1'] = feat['rsi14'].shift(1)
    feat['ret1_lag2'] = feat['ret_1'].shift(2)
    feat['rsi14_lag2'] = feat['rsi14'].shift(2)
    feat['ret1_lag3'] = feat['ret_1'].shift(3)
    feat['rsi14_lag3'] = feat['rsi14'].shift(3)

    return feat


def detect_regime(df: pd.DataFrame) -> pd.Series:
    """Detecta regimen de mercado."""
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


def train_model(pair: str, df: pd.DataFrame):
    """Entrena modelo GradientBoosting para un par."""
    print(f"\n[2/5] Entrenando modelo {pair}...")

    # Features
    feat = compute_features(df)

    # Target: retorno 5 velas adelante
    target = df['close'].pct_change(5).shift(-5)

    # Limpiar
    valid = feat.notna().all(axis=1) & target.notna()
    feat = feat[valid]
    target = target[valid]
    df_clean = df[valid].copy()

    # Split temporal
    train_end = '2025-07-31'
    val_end = '2026-01-31'

    train_mask = feat.index <= train_end
    val_mask = (feat.index > train_end) & (feat.index <= val_end)
    test_mask = feat.index > val_end

    X_train, y_train = feat[train_mask], target[train_mask]
    X_val, y_val = feat[val_mask], target[val_mask]
    X_test, y_test = feat[test_mask], target[test_mask]

    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Escalar
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Entrenar
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train_s, y_train)

    # Predicciones
    pred_train = model.predict(X_train_s)
    pred_val = model.predict(X_val_s)
    pred_test = model.predict(X_test_s)

    pred_std = np.std(pred_train)

    # Guardar modelo
    safe_pair = pair.replace('/', '_').lower()
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_cols': list(feat.columns),
        'pred_std': pred_std,
        'model_type': 'GradientBoostingRegressor',
    }
    joblib.dump(model_data, MODELS_DIR / f'{safe_pair}_v2_gradientboosting.pkl')

    return model, scaler, feat.columns.tolist(), pred_std, df_clean, feat, target


def backtest_with_tpsl(pair: str, df: pd.DataFrame, feat: pd.DataFrame,
                       target: pd.Series, model, scaler, feature_cols,
                       pred_std: float, tp_pct: float, sl_pct: float,
                       conv_min: float = 1.0) -> dict:
    """Backtest con TP/SL especificos."""

    # Predicciones
    X = feat[feature_cols].copy()
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    # Conviction
    conv = np.abs(preds) / pred_std

    # Regime
    regime = detect_regime(df.loc[X.index])

    # Simular trades
    trades = []
    for i, (idx, row) in enumerate(X.iterrows()):
        if i >= len(X) - 5:  # No trades en ultimas 5 velas
            break

        pred = preds[i]
        c = conv[i]
        reg = regime.iloc[i]

        if c < conv_min:
            continue

        direction = 1 if pred > 0 else -1

        # Regime filter
        if reg == 'BULL' and direction == -1:
            continue
        if reg == 'BEAR' and direction == 1:
            continue

        # Simular trade
        entry_price = df.loc[idx, 'close']

        if direction == 1:
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
        else:
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)

        # Buscar exit en siguientes 20 velas
        exit_price = None
        exit_reason = 'timeout'

        future_idx = X.index.get_loc(idx)
        for j in range(1, min(21, len(df) - future_idx)):
            bar = df.iloc[future_idx + j]
            high, low, close = bar['high'], bar['low'], bar['close']

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
            exit_price = df.iloc[min(future_idx + 20, len(df) - 1)]['close']

        # Calcular PnL
        if direction == 1:
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        trades.append({
            'entry_time': idx,
            'direction': direction,
            'conviction': c,
            'regime': reg,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'year': idx.year,
        })

    if not trades:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'pf': 0}

    trades_df = pd.DataFrame(trades)
    wins = (trades_df['pnl_pct'] > 0).sum()
    total = len(trades_df)
    wr = wins / total if total > 0 else 0
    total_pnl = trades_df['pnl_pct'].sum() * 100

    gross_profit = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
    gross_loss = abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        'trades': total,
        'wr': wr,
        'pnl': total_pnl,
        'pf': pf,
        'trades_df': trades_df,
    }


def optimize_tpsl(pair: str, df: pd.DataFrame, feat: pd.DataFrame,
                  target: pd.Series, model, scaler, feature_cols, pred_std: float):
    """Optimiza TP/SL para un par."""
    print(f"\n[4/5] Optimizando TP/SL para {pair}...")

    # Grid de TP/SL a probar
    tp_options = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    sl_options = [0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025]

    results = []

    for tp in tp_options:
        for sl in sl_options:
            res = backtest_with_tpsl(
                pair, df, feat, target, model, scaler, feature_cols,
                pred_std, tp, sl, conv_min=1.0
            )
            results.append({
                'tp': tp,
                'sl': sl,
                'ratio': tp / sl,
                'trades': res['trades'],
                'wr': res['wr'],
                'pnl': res['pnl'],
                'pf': res['pf'],
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('pnl', ascending=False)

    print("\n   Top 10 configuraciones:")
    print("   TP% | SL% | Ratio | Trades | WR% | PnL% | PF")
    print("   " + "-" * 55)
    for _, r in results_df.head(10).iterrows():
        print(f"   {r['tp']*100:.1f}% | {r['sl']*100:.2f}% | {r['ratio']:.1f}:1 | "
              f"{r['trades']:4.0f} | {r['wr']*100:.1f}% | {r['pnl']:+.1f}% | {r['pf']:.2f}")

    best = results_df.iloc[0]
    return best['tp'], best['sl'], results_df


def analyze_by_year_regime(pair: str, df: pd.DataFrame, feat: pd.DataFrame,
                           target: pd.Series, model, scaler, feature_cols,
                           pred_std: float, tp: float, sl: float):
    """Analiza rendimiento por ano y regimen."""
    print(f"\n[5/5] Analisis por ano/regimen para {pair}...")

    res = backtest_with_tpsl(
        pair, df, feat, target, model, scaler, feature_cols,
        pred_std, tp, sl, conv_min=1.0
    )

    if res['trades'] == 0:
        print("   Sin trades")
        return

    trades_df = res['trades_df']

    # Por ano
    print("\n   Por Ano:")
    print("   Ano  | Trades | WR% | PnL%")
    print("   " + "-" * 35)
    for year in sorted(trades_df['year'].unique()):
        yr_trades = trades_df[trades_df['year'] == year]
        wins = (yr_trades['pnl_pct'] > 0).sum()
        wr = wins / len(yr_trades) if len(yr_trades) > 0 else 0
        pnl = yr_trades['pnl_pct'].sum() * 100
        print(f"   {year} | {len(yr_trades):6} | {wr*100:5.1f}% | {pnl:+7.1f}%")

    # Por regimen
    print("\n   Por Regimen:")
    print("   Reg   | Trades | WR% | PnL%")
    print("   " + "-" * 35)
    for reg in ['BULL', 'BEAR', 'RANGE']:
        reg_trades = trades_df[trades_df['regime'] == reg]
        if len(reg_trades) == 0:
            continue
        wins = (reg_trades['pnl_pct'] > 0).sum()
        wr = wins / len(reg_trades) if len(reg_trades) > 0 else 0
        pnl = reg_trades['pnl_pct'].sum() * 100
        print(f"   {reg:5} | {len(reg_trades):6} | {wr*100:5.1f}% | {pnl:+7.1f}%")


def process_pair(pair: str):
    """Procesa un par completo."""
    print(f"\n{'='*60}")
    print(f"PROCESANDO {pair}")
    print('='*60)

    # 1. Descargar datos
    df = download_data(pair)

    # 2. Entrenar modelo
    model, scaler, feature_cols, pred_std, df_clean, feat, target = train_model(pair, df)

    # 3. Backtest baseline (3%/1.5%)
    print(f"\n[3/5] Backtest baseline {pair} (TP=3%, SL=1.5%)...")
    baseline = backtest_with_tpsl(
        pair, df_clean, feat, target, model, scaler, feature_cols,
        pred_std, tp_pct=0.03, sl_pct=0.015, conv_min=1.0
    )
    print(f"   Baseline: {baseline['trades']} trades, {baseline['wr']*100:.1f}% WR, "
          f"{baseline['pnl']:+.1f}% PnL, PF={baseline['pf']:.2f}")

    # 4. Optimizar TP/SL
    best_tp, best_sl, results_df = optimize_tpsl(
        pair, df_clean, feat, target, model, scaler, feature_cols, pred_std
    )

    # 5. Backtest con mejor TP/SL
    optimized = backtest_with_tpsl(
        pair, df_clean, feat, target, model, scaler, feature_cols,
        pred_std, tp_pct=best_tp, sl_pct=best_sl, conv_min=1.0
    )

    print(f"\n   MEJOR CONFIG: TP={best_tp*100:.1f}%, SL={best_sl*100:.2f}%")
    print(f"   Optimizado: {optimized['trades']} trades, {optimized['wr']*100:.1f}% WR, "
          f"{optimized['pnl']:+.1f}% PnL, PF={optimized['pf']:.2f}")

    # 6. Comparacion
    print(f"\n   COMPARACION:")
    print(f"   {'Metrica':<15} | {'Baseline':>10} | {'Optimizado':>10} | {'Cambio':>10}")
    print(f"   " + "-" * 55)
    print(f"   {'Trades':<15} | {baseline['trades']:>10} | {optimized['trades']:>10} | {optimized['trades'] - baseline['trades']:>+10}")
    print(f"   {'Win Rate':<15} | {baseline['wr']*100:>9.1f}% | {optimized['wr']*100:>9.1f}% | {(optimized['wr'] - baseline['wr'])*100:>+9.1f}%")
    print(f"   {'PnL':<15} | {baseline['pnl']:>+9.1f}% | {optimized['pnl']:>+9.1f}% | {optimized['pnl'] - baseline['pnl']:>+9.1f}%")
    print(f"   {'Profit Factor':<15} | {baseline['pf']:>10.2f} | {optimized['pf']:>10.2f} | {optimized['pf'] - baseline['pf']:>+10.2f}")

    # 7. Analisis detallado
    analyze_by_year_regime(
        pair, df_clean, feat, target, model, scaler, feature_cols,
        pred_std, best_tp, best_sl
    )

    return {
        'pair': pair,
        'baseline': baseline,
        'optimized': optimized,
        'best_tp': best_tp,
        'best_sl': best_sl,
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'pred_std': pred_std,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    pairs = ['SOL/USDT', 'BNB/USDT']

    results = {}
    for pair in pairs:
        try:
            results[pair] = process_pair(pair)
        except Exception as e:
            print(f"\n[ERROR] {pair}: {e}")
            import traceback
            traceback.print_exc()

    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)

    print(f"\n{'Par':<12} | {'Baseline PnL':>12} | {'Optimizado PnL':>14} | {'Mejor TP/SL':>12} | {'Recomendacion':>15}")
    print("-" * 75)

    for pair, res in results.items():
        baseline_pnl = res['baseline']['pnl']
        opt_pnl = res['optimized']['pnl']
        tp_sl = f"{res['best_tp']*100:.1f}/{res['best_sl']*100:.1f}%"

        # Recomendacion
        if opt_pnl > 20 and res['optimized']['wr'] > 0.5:
            rec = "HABILITAR"
        elif opt_pnl > 0:
            rec = "PROBAR"
        else:
            rec = "NO HABILITAR"

        print(f"{pair:<12} | {baseline_pnl:>+11.1f}% | {opt_pnl:>+13.1f}% | {tp_sl:>12} | {rec:>15}")

    print("\n[DONE]")
