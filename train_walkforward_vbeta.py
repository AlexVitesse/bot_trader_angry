"""
Walk-Forward Training - V13 Beta Models
========================================
Implementa validacion walk-forward REAL:
- Train: 2020-01-01 hasta 2024-12-31
- Validation: 2025-01-01 hasta 2025-09-30 (para optimizar TP/SL)
- Test: 2025-10-01 hasta 2026-02-27 (SAGRADO - no tocar hasta el final)

Los modelos se guardan como *_vbeta_*.pkl
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from pathlib import Path
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Configuracion de fechas ESTRICTAS
TRAIN_START = '2020-01-01'
TRAIN_END = '2024-12-31'
VAL_START = '2025-01-01'
VAL_END = '2025-09-30'
TEST_START = '2025-10-01'
TEST_END = '2026-02-28'

# Pares a entrenar
PAIRS = [
    'BTC_USDT', 'BNB_USDT', 'XRP_USDT', 'ETH_USDT', 'AVAX_USDT',
    'ADA_USDT', 'LINK_USDT', 'DOGE_USDT', 'NEAR_USDT', 'DOT_USDT'
]

# Configuracion de backtest
CAPITAL = 100.0
LEVERAGE = 5
COMMISSION = 0.0004

# Grid de parametros para validacion (mas pequeno para evitar overfitting)
TP_GRID = [0.03, 0.04, 0.05, 0.06, 0.07]  # Menos opciones
SL_GRID = [0.015, 0.02, 0.025, 0.03, 0.035]  # Menos opciones
CONV_GRID = [0.5, 1.0]  # Solo 2 opciones

DATA_DIR = Path('data')
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)


def load_pair_data(pair: str) -> pd.DataFrame:
    """Carga datos de un par desde parquet."""
    # Try different file patterns
    patterns = [
        f'{pair}_4h_full.parquet',
        f'{pair}_4h_backtest.parquet',
        f'{pair}_4h_history.parquet',
    ]

    for pattern in patterns:
        file_path = DATA_DIR / pattern
        if file_path.exists():
            df = pd.read_parquet(file_path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            df = df[~df.index.duplicated(keep='first')].sort_index()
            print(f"  Loaded: {file_path.name}")
            return df

    print(f"  [!] No data file for {pair}")
    return None


def compute_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Computa 54 features para modelos V2."""
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


def train_model_walkforward(pair: str, df: pd.DataFrame) -> dict:
    """
    Entrena modelo usando SOLO datos de TRAIN period (2020-2024).
    Retorna modelo, scaler, feature_cols, pred_std.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING {pair} - Walk-Forward")
    print(f"{'='*60}")

    # Filtrar SOLO periodo de train
    train_df = df[TRAIN_START:TRAIN_END].copy()
    print(f"Train period: {TRAIN_START} to {TRAIN_END}")
    print(f"Train samples: {len(train_df)}")

    if len(train_df) < 500:
        print(f"  [!] Insuficientes datos de train")
        return None

    # Compute features
    feat = compute_features_v2(train_df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # Target: retorno 5 velas adelante
    target = train_df['close'].pct_change(5).shift(-5)

    # Alinear y limpiar
    valid_idx = feat.dropna().index.intersection(target.dropna().index)
    X = feat.loc[valid_idx]
    y = target.loc[valid_idx]

    # Eliminar ultimas 5 filas (no tienen target real)
    X = X.iloc[:-5]
    y = y.iloc[:-5]

    print(f"Training samples after cleanup: {len(X)}")

    feature_cols = list(X.columns)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=feature_cols)

    # Train model
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_scaled, y)

    # Compute pred_std from training predictions
    train_preds = model.predict(X_scaled)
    pred_std = np.std(train_preds)

    # Correlation on train set
    train_corr = np.corrcoef(train_preds, y)[0, 1]
    print(f"Train correlation: {train_corr:.4f}")
    print(f"Pred std: {pred_std:.6f}")

    return {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'pred_std': pred_std,
        'train_corr': train_corr,
        'train_samples': len(X),
    }


def backtest_validation(pair: str, df: pd.DataFrame, model_data: dict,
                        tp_pct: float, sl_pct: float, conv_min: float,
                        direction_filter: str = 'BOTH') -> dict:
    """
    Backtest en periodo de VALIDACION (Ene-Sep 2025).
    NO usa datos de test (Oct 2025+).
    """
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['feature_cols']
    pred_std = model_data['pred_std']

    # Need lookback for features - get data INCLUDING some history before validation
    # Handle timezone issues
    try:
        if df.index.tz is not None:
            val_start_ts = pd.Timestamp(VAL_START, tz=df.index.tz)
            val_end_ts = pd.Timestamp(VAL_END, tz=df.index.tz)
        else:
            val_start_ts = pd.Timestamp(VAL_START)
            val_end_ts = pd.Timestamp(VAL_END)

        val_start_idx = df.index.get_indexer([val_start_ts], method='nearest')[0]
        val_end_idx = df.index.get_indexer([val_end_ts], method='nearest')[0]
    except:
        # Fallback: use string filtering
        val_mask = (df.index >= VAL_START) & (df.index <= VAL_END)
        if not val_mask.any():
            return {'trades': 0, 'wr': 0, 'pnl': 0}
        val_start_idx = val_mask.argmax()
        val_end_idx = len(df) - val_mask[::-1].argmax() - 1

    # Get 250 candles before validation start for feature computation
    start_idx = max(0, val_start_idx - 250)
    work_df = df.iloc[start_idx:val_end_idx + 1].copy()

    if len(work_df) < 300:
        return {'trades': 0, 'wr': 0, 'pnl': 0}

    # Compute features on full working data
    feat = compute_features_v2(work_df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # Align with available columns
    cols = [c for c in feature_cols if c in feat.columns]

    # Calculate which indices are in validation period (after lookback)
    # The lookback is 250 candles, so the first valid index for signals is around 200
    # and all rows from val_start_idx onward (in work_df) are validation period
    val_offset = 250  # lookback used

    # Simulate trading
    trades = []
    position = None
    predictions = []  # Debug: collect all predictions

    for i in range(200, len(work_df) - 5):

        row_feat = feat[cols].iloc[i:i+1].fillna(0)

        if len(row_feat.dropna(axis=1)) < len(cols) * 0.5:
            continue

        # Scale and predict
        try:
            row_scaled = scaler.transform(row_feat)
            pred = model.predict(row_scaled)[0]
        except:
            continue

        conf = abs(pred) / pred_std if pred_std > 1e-8 else 0
        predictions.append({'pred': pred, 'conf': conf})

        if position is None and conf >= conv_min:
            direction = 1 if pred > 0 else -1

            # Apply direction filter
            if direction_filter == 'SHORT_ONLY' and direction == 1:
                continue
            if direction_filter == 'LONG_ONLY' and direction == -1:
                continue

            entry_price = work_df['close'].iloc[i]
            position = {
                'entry_idx': i,
                'entry_price': entry_price,
                'direction': direction,
                'tp_price': entry_price * (1 + direction * tp_pct),
                'sl_price': entry_price * (1 - direction * sl_pct),
            }

        elif position is not None:
            current_high = work_df['high'].iloc[i]
            current_low = work_df['low'].iloc[i]
            current_close = work_df['close'].iloc[i]

            hit_tp = False
            hit_sl = False

            if position['direction'] == 1:  # LONG
                hit_tp = current_high >= position['tp_price']
                hit_sl = current_low <= position['sl_price']
            else:  # SHORT
                hit_tp = current_low <= position['tp_price']
                hit_sl = current_high >= position['sl_price']

            # Timeout: 30 candles
            timeout = (i - position['entry_idx']) >= 30

            if hit_tp or hit_sl or timeout:
                if hit_tp:
                    pnl_pct = tp_pct * position['direction'] * LEVERAGE
                elif hit_sl:
                    pnl_pct = -sl_pct * LEVERAGE
                else:
                    exit_price = current_close
                    raw_pnl = (exit_price - position['entry_price']) / position['entry_price']
                    pnl_pct = raw_pnl * position['direction'] * LEVERAGE

                pnl_pct -= COMMISSION * 2  # Entry + exit commission

                trades.append({
                    'pnl_pct': pnl_pct,
                    'win': pnl_pct > 0,
                    'direction': position['direction'],
                })
                position = None

    # Debug: show prediction stats (only for first call per pair)
    if predictions and conv_min == 0.5 and direction_filter == 'BOTH':
        confs = [p['conf'] for p in predictions]
        print(f"    Prediction stats: {len(predictions)} predictions, "
              f"conf range [{min(confs):.3f}, {max(confs):.3f}], "
              f"mean conf {np.mean(confs):.3f}, "
              f">0.5: {sum(1 for c in confs if c >= 0.5)}, "
              f">1.0: {sum(1 for c in confs if c >= 1.0)}")

    if not trades:
        return {'trades': 0, 'wr': 0, 'pnl': 0}

    wins = sum(1 for t in trades if t['win'])
    total_pnl = sum(t['pnl_pct'] * CAPITAL for t in trades)

    return {
        'trades': len(trades),
        'wr': wins / len(trades) * 100,
        'pnl': total_pnl,
        'wins': wins,
        'losses': len(trades) - wins,
    }


def optimize_on_validation(pair: str, df: pd.DataFrame, model_data: dict) -> dict:
    """
    Optimiza TP/SL usando SOLO el periodo de validacion.
    Retorna mejores parametros.
    """
    print(f"\nOptimizing {pair} on VALIDATION set ({VAL_START} to {VAL_END})...")

    best_result = None
    best_params = None
    all_results = []

    # Test all directions first to see which works best
    for direction in ['BOTH', 'LONG_ONLY', 'SHORT_ONLY']:
        for tp in TP_GRID:
            for sl in SL_GRID:
                for conv in CONV_GRID:
                    result = backtest_validation(
                        pair, df, model_data,
                        tp_pct=tp, sl_pct=sl, conv_min=conv,
                        direction_filter=direction
                    )

                    all_results.append({
                        'tp': tp, 'sl': sl, 'conv': conv, 'dir': direction,
                        **result
                    })

                    # Relax minimum trades requirement
                    if result['trades'] < 5:
                        continue

                    # Score: any positive PnL is good, prioritize WR
                    score = result['pnl'] + result['wr'] * 0.5

                    if best_result is None or score > best_result.get('score', -999):
                        best_result = {**result, 'score': score}
                        best_params = {
                            'tp_pct': tp,
                            'sl_pct': sl,
                            'conv_min': conv,
                            'direction': direction,
                        }

    # Debug: show best results
    if all_results:
        sorted_results = sorted(all_results, key=lambda x: x.get('pnl', -999), reverse=True)[:5]
        print(f"  Top 5 results by PnL:")
        for r in sorted_results:
            print(f"    TP={r['tp']*100:.0f}%, SL={r['sl']*100:.1f}%, Conv={r['conv']}, "
                  f"Dir={r['dir']}: {r['trades']} trades, {r['wr']:.1f}% WR, ${r['pnl']:.2f}")

    if best_params:
        print(f"  Best: TP={best_params['tp_pct']*100}%, SL={best_params['sl_pct']*100}%, "
              f"Conv={best_params['conv_min']}, Dir={best_params['direction']}")
        print(f"  Validation: {best_result['trades']} trades, "
              f"{best_result['wr']:.1f}% WR, ${best_result['pnl']:.2f} PnL")
    else:
        # If no valid params, use most conservative as fallback
        if all_results:
            fallback = max(all_results, key=lambda x: x.get('pnl', -999))
            if fallback['trades'] > 0:
                best_result = fallback
                best_params = {
                    'tp_pct': fallback['tp'],
                    'sl_pct': fallback['sl'],
                    'conv_min': fallback['conv'],
                    'direction': fallback['dir'],
                }
                print(f"  Fallback params: TP={best_params['tp_pct']*100}%, SL={best_params['sl_pct']*100}%")

    return {
        'params': best_params,
        'validation_result': best_result,
    }


def save_vbeta_model(pair: str, model_data: dict, opt_result: dict):
    """Guarda modelo vbeta con su configuracion."""
    safe_pair = pair.lower()
    model_file = MODELS_DIR / f'{safe_pair}_vbeta_gradientboosting.pkl'

    save_data = {
        'model': model_data['model'],
        'scaler': model_data['scaler'],
        'feature_cols': model_data['feature_cols'],
        'pred_std': model_data['pred_std'],
        'train_corr': model_data['train_corr'],
        'train_samples': model_data['train_samples'],
        'train_period': f'{TRAIN_START} to {TRAIN_END}',
        'validation_period': f'{VAL_START} to {VAL_END}',
        'optimized_params': opt_result['params'],
        'validation_result': opt_result['validation_result'],
        'created': datetime.now().isoformat(),
    }

    joblib.dump(save_data, model_file)
    print(f"  Saved: {model_file}")

    return model_file


def main():
    print("="*70)
    print("WALK-FORWARD TRAINING - V13 BETA MODELS")
    print("="*70)
    print(f"\nTrain period:      {TRAIN_START} to {TRAIN_END}")
    print(f"Validation period: {VAL_START} to {VAL_END}")
    print(f"Test period:       {TEST_START} to {TEST_END} (SACRED - NOT USED)")
    print()

    results = []

    for pair in PAIRS:
        # Load data
        df = load_pair_data(pair)
        if df is None:
            continue

        print(f"\nData range: {df.index.min()} to {df.index.max()}")
        print(f"Total rows: {len(df)}")

        # Step 1: Train on 2020-2024 ONLY
        model_data = train_model_walkforward(pair, df)
        if model_data is None:
            continue

        # Step 2: Optimize on validation (Jan-Sep 2025) ONLY
        opt_result = optimize_on_validation(pair, df, model_data)
        if opt_result['params'] is None:
            print(f"  [!] No valid params found for {pair}")
            continue

        # Step 3: Save vbeta model
        model_file = save_vbeta_model(pair, model_data, opt_result)

        results.append({
            'pair': pair,
            'train_corr': model_data['train_corr'],
            'train_samples': model_data['train_samples'],
            **opt_result['params'],
            **opt_result['validation_result'],
        })

    # Summary
    print("\n" + "="*70)
    print("WALK-FORWARD TRAINING SUMMARY")
    print("="*70)
    print(f"\n{'Pair':<12} {'Train Corr':<12} {'Val Trades':<12} {'Val WR':<10} {'Val PnL':<10} {'TP':<6} {'SL':<6} {'Dir':<12}")
    print("-"*90)

    for r in results:
        print(f"{r['pair']:<12} {r['train_corr']:.4f}       {r['trades']:<12} {r['wr']:.1f}%     ${r['pnl']:>7.2f}   "
              f"{r['tp_pct']*100:.0f}%   {r['sl_pct']*100:.1f}%  {r['direction']:<12}")

    # Save summary
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(DATA_DIR / 'vbeta_walkforward_summary.csv', index=False)
    print(f"\nSummary saved to: data/vbeta_walkforward_summary.csv")

    print("\n" + "="*70)
    print("NEXT STEP: Run backtest_vbeta_test.py to evaluate on TEST set")
    print("TEST SET IS SACRED - Only run ONCE for final evaluation!")
    print("="*70)


if __name__ == '__main__':
    main()
