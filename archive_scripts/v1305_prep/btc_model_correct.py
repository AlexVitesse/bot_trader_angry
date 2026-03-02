"""
BTC Model - Metodologia Correcta
================================
Split temporal estricto:
  - Train: 2019-01-01 a 2024-12-31 (6 años)
  - Validation: 2025-01-01 a 2025-08-31 (8 meses) - grid search aqui
  - Test: 2025-09-01 a 2026-02-28 (6 meses) - NUNCA tocar hasta final

Grid search: SOLO en validation
Backtest: Una posicion a la vez (igual que el bot)
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')

# Fechas de corte
TRAIN_END = '2024-12-31'
VALIDATION_END = '2025-08-31'
# Test: 2025-09-01 en adelante

FEATURE_COLS = ['ret_1', 'ret_5', 'ret_20', 'vol20', 'rsi14', 'ema21_d', 'vr']


def compute_features(df):
    """7 features minimas."""
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


def load_data():
    """Cargar datos BTC."""
    df = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_full.parquet')
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    return df.sort_index()


def backtest_single_position(df, preds, tp_pct, sl_pct, conv_min, pred_std):
    """
    Backtest con UNA posicion a la vez (igual que el bot real).

    Returns:
        dict con metricas
    """
    trades = []
    position = None  # {'entry_idx': idx, 'entry_price': price, 'direction': 'LONG'}

    for i, idx in enumerate(df.index):
        price = df.loc[idx, 'close']
        pred = preds[i]

        # Calcular conviction
        conviction = abs(pred) / pred_std if pred_std > 0 else 0

        # Si hay posicion abierta, checkear TP/SL
        if position is not None:
            entry_price = position['entry_price']

            if position['direction'] == 'LONG':
                pnl_pct = (price - entry_price) / entry_price

                # Check TP/SL
                if pnl_pct >= tp_pct:  # TP hit
                    trades.append({
                        'entry_time': position['entry_idx'],
                        'exit_time': idx,
                        'direction': 'LONG',
                        'entry_price': entry_price,
                        'exit_price': price,
                        'pnl_pct': pnl_pct,
                        'result': 'TP'
                    })
                    position = None
                elif pnl_pct <= -sl_pct:  # SL hit
                    trades.append({
                        'entry_time': position['entry_idx'],
                        'exit_time': idx,
                        'direction': 'LONG',
                        'entry_price': entry_price,
                        'exit_price': price,
                        'pnl_pct': pnl_pct,
                        'result': 'SL'
                    })
                    position = None

        # Si no hay posicion, buscar señal
        if position is None:
            # Solo LONG si pred > 0 y conviction >= threshold
            if pred > 0 and conviction >= conv_min:
                position = {
                    'entry_idx': idx,
                    'entry_price': price,
                    'direction': 'LONG'
                }

    # Cerrar posicion abierta al final (si existe)
    if position is not None:
        entry_price = position['entry_price']
        final_price = df.iloc[-1]['close']
        pnl_pct = (final_price - entry_price) / entry_price
        trades.append({
            'entry_time': position['entry_idx'],
            'exit_time': df.index[-1],
            'direction': 'LONG',
            'entry_price': entry_price,
            'exit_price': final_price,
            'pnl_pct': pnl_pct,
            'result': 'TIMEOUT'
        })

    # Calcular metricas
    if not trades:
        return {'n_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0, 'pf': 0}

    trades_df = pd.DataFrame(trades)
    n_trades = len(trades_df)
    wins = (trades_df['pnl_pct'] > 0).sum()
    win_rate = wins / n_trades * 100
    total_pnl = trades_df['pnl_pct'].sum() * 100  # En %
    avg_pnl = total_pnl / n_trades

    gross_profit = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
    gross_loss = abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 999

    return {
        'n_trades': n_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'pf': pf,
        'trades': trades_df
    }


def grid_search_validation(df_val, preds_val, pred_std):
    """
    Grid search de TP/SL SOLO en validation set.
    """
    tp_range = [0.015, 0.02, 0.025, 0.03, 0.04]
    sl_range = [0.01, 0.015, 0.02, 0.025, 0.03]
    conv_range = [0.5, 0.7, 1.0, 1.5]

    best_result = None
    best_params = None
    best_score = -999

    print("\nGrid Search en VALIDATION set...")
    print("-" * 70)

    for tp in tp_range:
        for sl in sl_range:
            for conv in conv_range:
                result = backtest_single_position(
                    df_val, preds_val, tp, sl, conv, pred_std
                )

                # Score: priorizamos PnL pero penalizamos pocos trades
                if result['n_trades'] < 10:
                    score = -999
                else:
                    score = result['total_pnl']

                if score > best_score:
                    best_score = score
                    best_params = {'tp': tp, 'sl': sl, 'conv': conv}
                    best_result = result

    print(f"Mejor config: TP={best_params['tp']:.1%} SL={best_params['sl']:.1%} Conv={best_params['conv']}")
    print(f"  Trades: {best_result['n_trades']}")
    print(f"  WinRate: {best_result['win_rate']:.1f}%")
    print(f"  PnL: {best_result['total_pnl']:.2f}%")
    print(f"  PF: {best_result['pf']:.2f}")

    return best_params, best_result


def main():
    print("=" * 70)
    print("BTC MODEL - METODOLOGIA CORRECTA")
    print("=" * 70)

    # Cargar datos
    print("\n[1/5] Cargando datos BTC...")
    df = load_data()
    print(f"  Total samples: {len(df):,}")
    print(f"  Rango: {df.index.min().date()} a {df.index.max().date()}")

    # Computar features
    print("\n[2/5] Calculando features...")
    feat = compute_features(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # Target: retorno 5 periodos adelante
    target = df['close'].pct_change(5).shift(-5)

    # Indices validos
    valid_idx = feat.dropna().index.intersection(target.dropna().index)
    X_all = feat.loc[valid_idx]
    y_all = target.loc[valid_idx]

    print(f"  Samples validos: {len(X_all):,}")

    # Split temporal estricto
    print("\n[3/5] Split temporal estricto...")

    # Train: hasta 2024-12-31
    train_mask = X_all.index <= TRAIN_END
    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    df_train = df.loc[X_train.index]

    # Validation: 2025-01-01 a 2025-08-31
    val_mask = (X_all.index > TRAIN_END) & (X_all.index <= VALIDATION_END)
    X_val = X_all[val_mask]
    y_val = y_all[val_mask]
    df_val = df.loc[X_val.index]

    # Test: 2025-09-01 en adelante
    test_mask = X_all.index > VALIDATION_END
    X_test = X_all[test_mask]
    y_test = y_all[test_mask]
    df_test = df.loc[X_test.index]

    print(f"  Train: {len(X_train):,} samples ({X_train.index.min().date()} a {X_train.index.max().date()})")
    print(f"  Validation: {len(X_val):,} samples ({X_val.index.min().date()} a {X_val.index.max().date()})")
    print(f"  Test: {len(X_test):,} samples ({X_test.index.min().date()} a {X_test.index.max().date()})")

    # Entrenar modelo SOLO con datos de train
    print("\n[4/5] Entrenando modelo...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = Ridge(alpha=100.0)
    model.fit(X_train_scaled, y_train)

    # Predecir en train
    preds_train = model.predict(X_train_scaled)
    pred_std = float(np.std(preds_train))
    corr_train = np.corrcoef(preds_train, y_train)[0, 1]

    # Predecir en validation (para grid search)
    X_val_scaled = scaler.transform(X_val)
    preds_val = model.predict(X_val_scaled)
    corr_val = np.corrcoef(preds_val, y_val)[0, 1]

    print(f"  Correlacion train: {corr_train:.4f}")
    print(f"  Correlacion validation: {corr_val:.4f}")
    print(f"  pred_std: {pred_std:.6f}")

    # Drop de correlacion
    corr_drop = (corr_train - corr_val) / corr_train * 100
    print(f"  Drop correlacion: {corr_drop:.1f}%")

    if corr_drop > 50:
        print("\n  WARNING: Drop > 50% indica posible overfitting")

    # Grid search SOLO en validation
    best_params, val_result = grid_search_validation(df_val, preds_val, pred_std)

    # Ahora TEST - evaluacion FINAL (una sola vez)
    print("\n" + "=" * 70)
    print("[5/5] EVALUACION FINAL EN TEST SET (una sola vez)")
    print("=" * 70)

    X_test_scaled = scaler.transform(X_test)
    preds_test = model.predict(X_test_scaled)
    corr_test = np.corrcoef(preds_test, y_test)[0, 1]

    print(f"\nCorrelacion test: {corr_test:.4f}")

    test_result = backtest_single_position(
        df_test, preds_test,
        best_params['tp'], best_params['sl'], best_params['conv'],
        pred_std
    )

    print(f"\nBacktest TEST con params de validation:")
    print(f"  TP={best_params['tp']:.1%} SL={best_params['sl']:.1%} Conv={best_params['conv']}")
    print(f"  Trades: {test_result['n_trades']}")
    print(f"  WinRate: {test_result['win_rate']:.1f}%")
    print(f"  PnL: {test_result['total_pnl']:.2f}%")
    print(f"  PF: {test_result['pf']:.2f}")

    # Veredicto
    print("\n" + "=" * 70)
    print("VEREDICTO")
    print("=" * 70)

    passed = True
    reasons = []

    if test_result['total_pnl'] <= 0:
        passed = False
        reasons.append(f"PnL negativo: {test_result['total_pnl']:.2f}%")

    if test_result['win_rate'] < 50:
        passed = False
        reasons.append(f"WinRate < 50%: {test_result['win_rate']:.1f}%")

    if test_result['pf'] < 1.0:
        passed = False
        reasons.append(f"Profit Factor < 1: {test_result['pf']:.2f}")

    if test_result['n_trades'] < 10:
        passed = False
        reasons.append(f"Muy pocos trades: {test_result['n_trades']}")

    if corr_test < 0.05:
        passed = False
        reasons.append(f"Correlacion test muy baja: {corr_test:.4f}")

    if passed:
        print("\n[APROBADO] MODELO LISTO - Listo para produccion")
        print(f"\nParametros a usar:")
        print(f"  direction: LONG_ONLY")
        print(f"  tp_pct: {best_params['tp']}")
        print(f"  sl_pct: {best_params['sl']}")
        print(f"  conv_min: {best_params['conv']}")
        print(f"  pred_std: {pred_std}")
    else:
        print("\n[RECHAZADO] MODELO NO VIABLE")
        print("\nRazones:")
        for reason in reasons:
            print(f"  - {reason}")
        print("\nOpciones:")
        print("  1. Probar otros modelos (XGBoost, LSTM)")
        print("  2. Agregar mas features")
        print("  3. Considerar abandonar ML para este par")

    # Guardar resumen
    summary = {
        'pair': 'BTC',
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'train_range': f"{X_train.index.min().date()} to {X_train.index.max().date()}",
        'val_range': f"{X_val.index.min().date()} to {X_val.index.max().date()}",
        'test_range': f"{X_test.index.min().date()} to {X_test.index.max().date()}",
        'corr_train': float(corr_train),
        'corr_val': float(corr_val),
        'corr_test': float(corr_test),
        'best_params': best_params,
        'test_result': {
            'n_trades': test_result['n_trades'],
            'win_rate': test_result['win_rate'],
            'total_pnl': test_result['total_pnl'],
            'pf': test_result['pf']
        },
        'passed': passed,
        'reasons': reasons if not passed else []
    }

    import json
    with open('btc_model_result.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResultados guardados en: btc_model_result.json")


if __name__ == '__main__':
    main()
