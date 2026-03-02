"""
Backtest V13.04 - Parametros Incorrectos vs Individualizados
==========================================================
Compara el rendimiento de V13.04 con:
- Parametros actuales: 2%/2% TP/SL para todos
- Parametros optimizados: Individuales por par (de docs OPTIMIZACION_*.md)

Uso: python backtest_v1304_params.py
"""

import json
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

# Parametros INCORRECTOS (lo que estaba en prod)
PARAMS_WRONG = {
    'DOGE': {'tp_pct': 0.02, 'sl_pct': 0.02, 'direction': 'LONG_ONLY'},
    'ADA':  {'tp_pct': 0.02, 'sl_pct': 0.02, 'direction': 'LONG_ONLY'},
    'DOT':  {'tp_pct': 0.02, 'sl_pct': 0.02, 'direction': 'LONG_ONLY'},
    'XRP':  {'tp_pct': 0.02, 'sl_pct': 0.02, 'direction': 'LONG_ONLY'},
    'BTC':  {'tp_pct': 0.02, 'sl_pct': 0.02, 'direction': 'LONG_ONLY'},
}

# Parametros CORRECTOS (de docs/LOW_OVERFIT_MODEL_RESULTS.md)
PARAMS_CORRECT = {
    'BTC':  {'tp_pct': 0.02,  'sl_pct': 0.02,  'direction': 'LONG_ONLY'},  # Score 100
    'DOGE': {'tp_pct': 0.02,  'sl_pct': 0.01,  'direction': 'LONG_ONLY'},  # Score 90, SL 1%
    'ADA':  {'tp_pct': 0.02,  'sl_pct': 0.015, 'direction': 'LONG_ONLY'},  # Score 90, SL 1.5%
    'XRP':  {'tp_pct': 0.02,  'sl_pct': 0.02,  'direction': 'LONG_ONLY'},  # Score 85
    'DOT':  {'tp_pct': 0.025, 'sl_pct': 0.02,  'direction': 'LONG_ONLY'},  # Score 70
}

FEATURE_COLS = ['ret_1', 'ret_5', 'ret_20', 'vol20', 'rsi14', 'ema21_d', 'vr']

# Backtest config
INITIAL_CAPITAL = 5000
LEVERAGE = 5
POSITION_SIZE = 0.20  # 20% per trade
COMMISSION = 0.001  # 0.1% (0.05% entry + 0.05% exit)
CONV_MIN = 1.0  # Minimum conviction to trade


def compute_features_minimal(df):
    """7 features - same as V13.04 training."""
    feat = pd.DataFrame(index=df.index)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    feat['ret_1'] = c.pct_change(1)
    feat['ret_5'] = c.pct_change(5)
    feat['ret_20'] = c.pct_change(20)
    feat['vol20'] = c.pct_change().rolling(20).std()
    feat['rsi14'] = ta.rsi(c, length=14)
    ema21 = ta.ema(c, length=21)
    feat['ema21_d'] = (c - ema21) / ema21 * 100
    feat['vr'] = v / v.rolling(20).mean()
    return feat


def load_data(pair):
    """Load pair data from parquet."""
    patterns = [
        f'{pair}_USDT_4h_full.parquet',
        f'{pair}_USDT_4h_backtest.parquet',
        f'{pair}_USDT_4h_history.parquet',
    ]
    for pattern in patterns:
        file_path = DATA_DIR / pattern
        if file_path.exists():
            df = pd.read_parquet(file_path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            return df.sort_index()
    return None


def load_v1304_model(pair):
    """Load V13.04 Ridge model and scaler."""
    model_path = MODELS_DIR / f'v1304_{pair}.pkl'
    scaler_path = MODELS_DIR / f'v1304_{pair}_scaler.pkl'

    if not model_path.exists() or not scaler_path.exists():
        return None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def get_pred_std(pair):
    """Get pred_std from metadata."""
    meta_path = MODELS_DIR / 'v1304_meta.json'
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if pair in meta.get('pairs', {}):
            return meta['pairs'][pair].get('pred_std', 0.01)
    return 0.01


def backtest_pair(pair, params, test_start='2025-01-01', test_end='2026-02-26'):
    """Backtest single pair with given params."""
    df = load_data(pair)
    if df is None:
        return None

    model, scaler = load_v1304_model(pair)
    if model is None:
        return None

    pred_std = get_pred_std(pair)

    # Filter test period
    df = df.loc[test_start:test_end].copy()
    if len(df) < 50:
        return None

    # Compute features
    feat = compute_features_minimal(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    tp_pct = params['tp_pct']
    sl_pct = params['sl_pct']
    direction = params['direction']

    trades = []

    for i in range(30, len(df) - 5):
        row_feat = feat.iloc[i:i+1][FEATURE_COLS]
        if row_feat.isna().any().any():
            continue

        # Get prediction
        X_scaled = scaler.transform(row_feat)
        pred = model.predict(X_scaled)[0]

        # Conviction (normalized prediction)
        conviction = abs(pred) / pred_std if pred_std > 0 else 0

        if conviction < CONV_MIN:
            continue

        # Direction filter
        if direction == 'LONG_ONLY' and pred <= 0:
            continue
        if direction == 'SHORT_ONLY' and pred >= 0:
            continue

        side = 'LONG' if pred > 0 else 'SHORT'
        entry_price = df['close'].iloc[i]
        entry_time = df.index[i]

        # Simulate trade (look ahead up to 20 bars)
        exit_price = None
        exit_reason = None
        exit_time = None

        for j in range(1, min(21, len(df) - i)):
            future_bar = df.iloc[i + j]
            high = future_bar['high']
            low = future_bar['low']
            close = future_bar['close']

            if side == 'LONG':
                tp_price = entry_price * (1 + tp_pct)
                sl_price = entry_price * (1 - sl_pct)

                # Check SL first (conservative)
                if low <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'SL'
                    exit_time = df.index[i + j]
                    break
                # Check TP
                if high >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'TP'
                    exit_time = df.index[i + j]
                    break
            else:  # SHORT
                tp_price = entry_price * (1 - tp_pct)
                sl_price = entry_price * (1 + sl_pct)

                if high >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'SL'
                    exit_time = df.index[i + j]
                    break
                if low <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'TP'
                    exit_time = df.index[i + j]
                    break

        # Timeout exit
        if exit_price is None:
            exit_price = df['close'].iloc[min(i + 20, len(df) - 1)]
            exit_reason = 'TIMEOUT'
            exit_time = df.index[min(i + 20, len(df) - 1)]

        # Calculate PnL
        if side == 'LONG':
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        notional = INITIAL_CAPITAL * POSITION_SIZE * LEVERAGE
        pnl = notional * pnl_pct - notional * COMMISSION * 2

        trades.append({
            'pair': pair,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'conviction': conviction,
        })

    return trades


def analyze_results(trades, label):
    """Analyze backtest results."""
    if not trades:
        return None

    df = pd.DataFrame(trades)

    total_pnl = df['pnl'].sum()
    n_trades = len(df)
    wins = len(df[df['pnl'] > 0])
    losses = n_trades - wins
    wr = wins / n_trades * 100 if n_trades > 0 else 0

    gross_wins = df[df['pnl'] > 0]['pnl'].sum()
    gross_losses = abs(df[df['pnl'] <= 0]['pnl'].sum())
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    avg_win = df[df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
    avg_loss = df[df['pnl'] <= 0]['pnl'].mean() if losses > 0 else 0

    # Calculate max drawdown
    cumulative = df['pnl'].cumsum() + INITIAL_CAPITAL
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max * 100
    max_dd = drawdown.min()

    return {
        'label': label,
        'trades': n_trades,
        'wins': wins,
        'losses': losses,
        'wr': wr,
        'pnl': total_pnl,
        'pf': pf,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_dd': max_dd,
    }


def main():
    print('=' * 80)
    print('BACKTEST V13.04 - PARAMETROS INCORRECTOS vs CORRECTOS')
    print('=' * 80)
    print(f'\nPeriodo: 2025-01-01 a 2026-02-26 (test out-of-sample)')
    print(f'Capital: ${INITIAL_CAPITAL:,} | Leverage: {LEVERAGE}x | Position: {POSITION_SIZE:.0%}')
    print(f'Conviction min: {CONV_MIN}')

    pairs = list(PARAMS_WRONG.keys())

    # ========================================================================
    # BACKTEST CON PARAMETROS INCORRECTOS (2%/2%)
    # ========================================================================
    print('\n' + '=' * 80)
    print('BACKTEST 1: PARAMETROS INCORRECTOS (TP 2% / SL 2%)')
    print('=' * 80)

    all_trades_wrong = []
    results_wrong = {}

    for pair in pairs:
        print(f'\n  {pair}...', end=' ')
        trades = backtest_pair(pair, PARAMS_WRONG[pair])
        if trades:
            all_trades_wrong.extend(trades)
            result = analyze_results(trades, pair)
            results_wrong[pair] = result
            print(f'{result["trades"]} trades | WR {result["wr"]:.1f}% | PnL ${result["pnl"]:+,.2f}')
        else:
            print('No data')

    total_wrong = analyze_results(all_trades_wrong, 'TOTAL GENERIC')

    # ========================================================================
    # BACKTEST CON PARAMETROS CORRECTOS
    # ========================================================================
    print('\n' + '=' * 80)
    print('BACKTEST 2: PARAMETROS CORRECTOS (de docs/OPTIMIZACION_*.md)')
    print('=' * 80)

    for pair, params in PARAMS_CORRECT.items():
        print(f'  {pair}: TP {params["tp_pct"]:.1%} / SL {params["sl_pct"]:.1%}')

    all_trades_correct = []
    results_correct = {}

    for pair in pairs:
        print(f'\n  {pair}...', end=' ')
        trades = backtest_pair(pair, PARAMS_CORRECT[pair])
        if trades:
            all_trades_correct.extend(trades)
            result = analyze_results(trades, pair)
            results_correct[pair] = result
            print(f'{result["trades"]} trades | WR {result["wr"]:.1f}% | PnL ${result["pnl"]:+,.2f}')
        else:
            print('No data')

    total_correct = analyze_results(all_trades_correct, 'TOTAL OPTIMIZED')

    # ========================================================================
    # COMPARACION
    # ========================================================================
    print('\n' + '=' * 80)
    print('COMPARACION POR PAR')
    print('=' * 80)
    print(f"\n{'Par':<6} | {'--- INCORRECTO 2%/2% ---':<28} | {'--- CORRECTO ---':<28} | {'MEJOR':<8}")
    print(f"{'':6} | {'Trades':>6} {'WR':>6} {'PnL':>12} | {'Trades':>6} {'WR':>6} {'PnL':>12} |")
    print('-' * 85)

    for pair in pairs:
        gen = results_wrong.get(pair, {})
        opt = results_correct.get(pair, {})

        gen_str = f"{gen.get('trades', 0):>6} {gen.get('wr', 0):>5.1f}% ${gen.get('pnl', 0):>+10,.2f}"
        opt_str = f"{opt.get('trades', 0):>6} {opt.get('wr', 0):>5.1f}% ${opt.get('pnl', 0):>+10,.2f}"

        gen_pnl = gen.get('pnl', 0)
        opt_pnl = opt.get('pnl', 0)
        mejor = 'OPT' if opt_pnl > gen_pnl else 'GEN' if gen_pnl > opt_pnl else 'IGUAL'

        print(f"{pair:<6} | {gen_str} | {opt_str} | {mejor:<8}")

    print('-' * 85)

    # Total row
    if total_wrong and total_correct:
        gen_str = f"{total_wrong['trades']:>6} {total_wrong['wr']:>5.1f}% ${total_wrong['pnl']:>+10,.2f}"
        opt_str = f"{total_correct['trades']:>6} {total_correct['wr']:>5.1f}% ${total_correct['pnl']:>+10,.2f}"
        mejor = 'OPT' if total_correct['pnl'] > total_wrong['pnl'] else 'GEN'
        print(f"{'TOTAL':<6} | {gen_str} | {opt_str} | {mejor:<8}")

    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    print('\n' + '=' * 80)
    print('RESUMEN FINAL')
    print('=' * 80)

    if total_wrong and total_correct:
        print(f"\n{'Metrica':<20} | {'Incorrecto':<15} | {'Correcto':<15} | {'Diferencia':<15}")
        print('-' * 70)
        print(f"{'Trades':<20} | {total_wrong['trades']:<15} | {total_correct['trades']:<15} | {total_correct['trades'] - total_wrong['trades']:>+15}")
        print(f"{'Win Rate':<20} | {total_wrong['wr']:<14.1f}% | {total_correct['wr']:<14.1f}% | {total_correct['wr'] - total_wrong['wr']:>+14.1f}%")
        print(f"{'PnL Total':<20} | ${total_wrong['pnl']:<13,.2f} | ${total_correct['pnl']:<13,.2f} | ${total_correct['pnl'] - total_wrong['pnl']:>+13,.2f}")
        print(f"{'Profit Factor':<20} | {total_wrong['pf']:<15.2f} | {total_correct['pf']:<15.2f} | {total_correct['pf'] - total_wrong['pf']:>+15.2f}")
        print(f"{'Max Drawdown':<20} | {total_wrong['max_dd']:<14.1f}% | {total_correct['max_dd']:<14.1f}% | {total_correct['max_dd'] - total_wrong['max_dd']:>+14.1f}%")

        pnl_diff = total_correct['pnl'] - total_wrong['pnl']
        pnl_diff_pct = (total_correct['pnl'] / total_wrong['pnl'] - 1) * 100 if total_wrong['pnl'] != 0 else 0

        print('\n' + '=' * 80)
        if pnl_diff > 0:
            print(f'CONCLUSION: Parametros CORRECTOS generan ${pnl_diff:+,.2f} mas ({pnl_diff_pct:+.1f}%)')
            print('RECOMENDACION: Actualizar V13.04 con parametros individualizados')
        else:
            print(f'CONCLUSION: Parametros INCORRECTOS generan ${-pnl_diff:+,.2f} mas ({-pnl_diff_pct:+.1f}%)')
            print('RECOMENDACION: Mantener parametros genericos 2%/2%')
        print('=' * 80)


if __name__ == '__main__':
    main()
