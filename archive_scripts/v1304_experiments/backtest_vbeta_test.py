"""
Backtest V-Beta on TEST SET (SACRED)
====================================
ADVERTENCIA: Este script solo debe correrse UNA VEZ.
El test set (Oct 2025 - Feb 2026) es SAGRADO.

Este script:
1. Carga los modelos vbeta entrenados con walk-forward
2. Ejecuta backtest en el periodo de TEST
3. Compara con los resultados de V13.03 (entrenado con todos los datos)
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from pathlib import Path
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# Test period - SACRED
TEST_START = '2025-10-01'
TEST_END = '2026-02-28'

# Backtest config
CAPITAL = 100.0
LEVERAGE = 5
COMMISSION = 0.0004
MAX_POSITIONS = 3

PAIRS = [
    'BTC_USDT', 'BNB_USDT', 'XRP_USDT', 'ETH_USDT', 'AVAX_USDT',
    'ADA_USDT', 'LINK_USDT', 'DOGE_USDT', 'NEAR_USDT', 'DOT_USDT'
]

DATA_DIR = Path('data')
MODELS_DIR = Path('models')


def load_pair_data(pair: str) -> pd.DataFrame:
    """Carga datos de un par desde parquet."""
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
            return df

    return None


def compute_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Computa 54 features para modelos V2."""
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


def load_vbeta_model(pair: str) -> dict:
    """Carga modelo vbeta y su configuracion."""
    safe_pair = pair.lower()
    model_file = MODELS_DIR / f'{safe_pair}_vbeta_gradientboosting.pkl'

    if not model_file.exists():
        return None

    return joblib.load(model_file)


def backtest_pair_on_test(pair: str, df: pd.DataFrame, model_data: dict) -> dict:
    """
    Backtest de un par en el TEST SET.
    """
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['feature_cols']
    pred_std = model_data['pred_std']
    params = model_data['optimized_params']

    tp_pct = params['tp_pct']
    sl_pct = params['sl_pct']
    conv_min = params['conv_min']
    direction_filter = params['direction']

    # Filtrar SOLO periodo de TEST
    test_df = df[TEST_START:TEST_END].copy()

    if len(test_df) < 50:
        return {'trades': 0, 'wr': 0, 'pnl': 0}

    # Need some history for features
    start_idx = df.index.get_loc(test_df.index[0])
    lookback = max(0, start_idx - 200)
    full_df = df.iloc[lookback:].copy()

    feat = compute_features_v2(full_df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    cols = [c for c in feature_cols if c in feat.columns]

    # Only trade in test period
    test_indices = feat.index.intersection(test_df.index)

    trades = []
    position = None

    for i, idx in enumerate(test_indices[:-5]):
        row_feat = feat[cols].loc[[idx]].fillna(0)

        try:
            row_scaled = scaler.transform(row_feat)
            pred = model.predict(row_scaled)[0]
        except:
            continue

        conf = abs(pred) / pred_std if pred_std > 1e-8 else 0

        if position is None and conf >= conv_min:
            direction = 1 if pred > 0 else -1

            if direction_filter == 'SHORT_ONLY' and direction == 1:
                continue
            if direction_filter == 'LONG_ONLY' and direction == -1:
                continue

            entry_price = test_df.loc[idx, 'close']
            position = {
                'entry_idx': i,
                'entry_price': entry_price,
                'direction': direction,
                'tp_price': entry_price * (1 + direction * tp_pct),
                'sl_price': entry_price * (1 - direction * sl_pct),
            }

        elif position is not None:
            current_high = test_df.loc[idx, 'high']
            current_low = test_df.loc[idx, 'low']
            current_close = test_df.loc[idx, 'close']

            hit_tp = False
            hit_sl = False

            if position['direction'] == 1:
                hit_tp = current_high >= position['tp_price']
                hit_sl = current_low <= position['sl_price']
            else:
                hit_tp = current_low <= position['tp_price']
                hit_sl = current_high >= position['sl_price']

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

                pnl_pct -= COMMISSION * 2

                trades.append({
                    'pnl_pct': pnl_pct,
                    'win': pnl_pct > 0,
                    'direction': position['direction'],
                })
                position = None

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


def main():
    print("="*70)
    print("BACKTEST V-BETA ON TEST SET (SACRED)")
    print("="*70)
    print(f"\nTest period: {TEST_START} to {TEST_END}")
    print("\n*** ADVERTENCIA: Este test solo debe correrse UNA VEZ ***")
    print("*** Los resultados son la VERDADERA evaluacion del modelo ***\n")

    # Confirm
    confirm = input("Escriba 'CONFIRMO' para continuar: ")
    if confirm != 'CONFIRMO':
        print("Cancelado.")
        return

    results = []

    for pair in PAIRS:
        # Load model
        model_data = load_vbeta_model(pair)
        if model_data is None:
            print(f"[!] No vbeta model for {pair}")
            continue

        # Load data
        df = load_pair_data(pair)
        if df is None:
            continue

        # Backtest on TEST set
        result = backtest_pair_on_test(pair, df, model_data)

        params = model_data['optimized_params']
        val_result = model_data['validation_result']

        results.append({
            'pair': pair,
            'val_trades': val_result['trades'],
            'val_wr': val_result['wr'],
            'val_pnl': val_result['pnl'],
            'test_trades': result['trades'],
            'test_wr': result['wr'],
            'test_pnl': result['pnl'],
            'tp_pct': params['tp_pct'],
            'sl_pct': params['sl_pct'],
            'direction': params['direction'],
        })

        print(f"{pair}: VAL {val_result['wr']:.1f}% WR, ${val_result['pnl']:.2f} -> "
              f"TEST {result['wr']:.1f}% WR, ${result['pnl']:.2f}")

    # Summary
    print("\n" + "="*70)
    print("COMPARISON: VALIDATION vs TEST")
    print("="*70)
    print(f"\n{'Pair':<12} {'Val Trades':<12} {'Val WR':<10} {'Val PnL':<10} {'Test Trades':<12} {'Test WR':<10} {'Test PnL':<10} {'Drop':<8}")
    print("-"*100)

    total_val_pnl = 0
    total_test_pnl = 0

    for r in results:
        wr_drop = r['test_wr'] - r['val_wr']
        print(f"{r['pair']:<12} {r['val_trades']:<12} {r['val_wr']:.1f}%     ${r['val_pnl']:>7.2f}   "
              f"{r['test_trades']:<12} {r['test_wr']:.1f}%     ${r['test_pnl']:>7.2f}   {wr_drop:+.1f}%")
        total_val_pnl += r['val_pnl']
        total_test_pnl += r['test_pnl']

    print("-"*100)
    print(f"{'TOTAL':<12} {'':<12} {'':<10} ${total_val_pnl:>7.2f}   {'':<12} {'':<10} ${total_test_pnl:>7.2f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(DATA_DIR / 'vbeta_test_results.csv', index=False)
    print(f"\nResults saved to: data/vbeta_test_results.csv")

    # Comparison with V13.03
    print("\n" + "="*70)
    print("INTERPRETACION")
    print("="*70)
    print("""
Si TEST WR es similar a VAL WR (+/- 5%):
  -> El modelo generaliza bien, NO hay overfitting severo

Si TEST WR es mucho menor que VAL WR (>10% drop):
  -> Hay overfitting, incluso con walk-forward

Si TEST PnL es positivo:
  -> El modelo es potencialmente rentable en produccion

Si TEST PnL es negativo:
  -> El modelo NO funciona en datos nuevos
""")


if __name__ == '__main__':
    main()
