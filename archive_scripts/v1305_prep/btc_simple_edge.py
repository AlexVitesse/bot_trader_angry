"""
BTC Simple Edge - Reglas tecnicas estrictas
============================================
Objetivo: NO predecir precio, sino encontrar setups de alta probabilidad.

Estrategia: Solo entrar cuando MULTIPLES condiciones se alinean.
Menos trades, mejor calidad.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path

DATA_DIR = Path('data')
TEST_START = '2025-09-01'  # Test set intocable


def load_data():
    df = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_full.parquet')
    return df.sort_index()


def compute_indicators(df):
    """Indicadores tecnicos clasicos."""
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    ind = pd.DataFrame(index=df.index)
    ind['close'] = c

    # RSI
    ind['rsi14'] = ta.rsi(c, length=14)
    ind['rsi7'] = ta.rsi(c, length=7)

    # Bollinger Bands
    bb = ta.bbands(c, length=20)
    if bb is not None and len(bb.columns) >= 3:
        bbl, bbm, bbu = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
        ind['bb_pct'] = (c - bbl) / (bbu - bbl)  # 0-1, donde esta el precio
        ind['bb_width'] = (bbu - bbl) / bbm * 100  # Ancho de bandas

    # Volume
    ind['vol_ratio'] = v / v.rolling(20).mean()

    # Volatilidad
    ind['atr14'] = ta.atr(h, l, c, length=14)
    ind['atr_pct'] = ind['atr14'] / c * 100

    # Tendencia
    ema20 = ta.ema(c, length=20)
    ema50 = ta.ema(c, length=50)
    ema200 = ta.ema(c, length=200)
    ind['above_ema20'] = (c > ema20).astype(int)
    ind['above_ema50'] = (c > ema50).astype(int)
    ind['above_ema200'] = (c > ema200).astype(int)
    ind['ema_trend'] = ind['above_ema20'] + ind['above_ema50'] + ind['above_ema200']  # 0-3

    # Momentum
    ind['ret_1'] = c.pct_change(1)
    ind['ret_5'] = c.pct_change(5)

    # Velas consecutivas
    ind['consec_green'] = (ind['ret_1'] > 0).rolling(5).sum()
    ind['consec_red'] = (ind['ret_1'] < 0).rolling(5).sum()

    return ind.dropna()


def backtest_strategy(ind, strategy_func, tp_pct, sl_pct, name="Strategy"):
    """Backtest con una posicion a la vez."""
    trades = []
    position = None

    for i, idx in enumerate(ind.index):
        row = ind.loc[idx]
        price = row['close']

        # Check posicion abierta
        if position is not None:
            entry_price = position['entry_price']
            pnl_pct = (price - entry_price) / entry_price

            if pnl_pct >= tp_pct:
                trades.append({'pnl_pct': pnl_pct, 'result': 'TP', 'hold': i - position['entry_i']})
                position = None
            elif pnl_pct <= -sl_pct:
                trades.append({'pnl_pct': pnl_pct, 'result': 'SL', 'hold': i - position['entry_i']})
                position = None

        # Buscar entrada
        if position is None:
            signal = strategy_func(row, ind, i)
            if signal == 'LONG':
                position = {'entry_price': price, 'entry_i': i}

    # Cerrar posicion final
    if position is not None:
        final_price = ind.iloc[-1]['close']
        pnl_pct = (final_price - position['entry_price']) / position['entry_price']
        trades.append({'pnl_pct': pnl_pct, 'result': 'TIMEOUT', 'hold': len(ind) - position['entry_i']})

    if not trades:
        return {'name': name, 'n_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'pf': 0, 'avg_hold': 0}

    trades_df = pd.DataFrame(trades)
    n_trades = len(trades_df)
    wins = (trades_df['pnl_pct'] > 0).sum()
    total_pnl = trades_df['pnl_pct'].sum() * 100
    avg_hold = trades_df['hold'].mean()

    gross_profit = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
    gross_loss = abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 999

    return {
        'name': name,
        'n_trades': n_trades,
        'win_rate': wins / n_trades * 100,
        'total_pnl': total_pnl,
        'pf': pf,
        'avg_hold': avg_hold
    }


# ============================================================
# ESTRATEGIAS A PROBAR
# ============================================================

def strategy_random(row, ind, i):
    """Baseline: entrada random 10% del tiempo."""
    import random
    return 'LONG' if random.random() < 0.10 else None


def strategy_rsi_extreme(row, ind, i):
    """RSI muy extremo (<25) = oversold."""
    if row['rsi14'] < 25:
        return 'LONG'
    return None


def strategy_rsi_bb(row, ind, i):
    """RSI oversold + precio bajo en BB."""
    if row['rsi14'] < 30 and row['bb_pct'] < 0.2:
        return 'LONG'
    return None


def strategy_rsi_bb_vol(row, ind, i):
    """RSI + BB + volumen alto (capitulacion)."""
    if row['rsi14'] < 30 and row['bb_pct'] < 0.2 and row['vol_ratio'] > 1.5:
        return 'LONG'
    return None


def strategy_trend_pullback(row, ind, i):
    """Tendencia alcista + pullback a EMA."""
    if row['ema_trend'] >= 2 and row['rsi14'] < 40 and row['bb_pct'] < 0.3:
        return 'LONG'
    return None


def strategy_capitulation(row, ind, i):
    """3+ velas rojas + RSI extremo + volumen = capitulacion."""
    if row['consec_red'] >= 3 and row['rsi14'] < 30 and row['vol_ratio'] > 1.3:
        return 'LONG'
    return None


def strategy_strict_oversold(row, ind, i):
    """MUY estricto: RSI<20 + BB<0.1 + Vol alto."""
    if row['rsi14'] < 20 and row['bb_pct'] < 0.1 and row['vol_ratio'] > 1.5:
        return 'LONG'
    return None


def strategy_mean_reversion(row, ind, i):
    """Mean reversion extremo: 5 velas rojas + RSI<25."""
    if row['consec_red'] >= 4 and row['rsi14'] < 25:
        return 'LONG'
    return None


def strategy_bb_squeeze_breakout(row, ind, i):
    """BB muy estrecho + breakout con volumen."""
    if i < 5:
        return None
    # BB estrecho (baja volatilidad)
    if row['bb_width'] < 3:  # Bandas muy juntas
        # Y ahora subiendo con volumen
        if row['ret_1'] > 0.01 and row['vol_ratio'] > 1.5:
            return 'LONG'
    return None


def strategy_multi_confirm(row, ind, i):
    """
    Estrategia con MULTIPLES confirmaciones:
    - Tendencia general alcista (EMA200)
    - Pullback (RSI < 40)
    - Soporte en BB (bb_pct < 0.3)
    - Volumen normal o alto
    """
    if (row['above_ema200'] == 1 and  # Tendencia macro alcista
        row['rsi14'] < 40 and          # Pullback en RSI
        row['rsi14'] > 25 and          # Pero no extremo (evitar cuchillo cayendo)
        row['bb_pct'] < 0.35 and       # Cerca del soporte BB
        row['vol_ratio'] > 0.8):       # Volumen al menos normal
        return 'LONG'
    return None


def main():
    print("=" * 70)
    print("BTC SIMPLE EDGE - Reglas Tecnicas Estrictas")
    print("=" * 70)

    df = load_data()
    ind = compute_indicators(df)

    # Split
    ind_train = ind[ind.index < TEST_START]
    ind_test = ind[ind.index >= TEST_START]

    print(f"\nTrain: {len(ind_train):,} candles ({ind_train.index.min().date()} a {ind_train.index.max().date()})")
    print(f"Test:  {len(ind_test):,} candles ({ind_test.index.min().date()} a {ind_test.index.max().date()})")

    strategies = [
        (strategy_random, "Random 10%"),
        (strategy_rsi_extreme, "RSI < 25"),
        (strategy_rsi_bb, "RSI<30 + BB<0.2"),
        (strategy_rsi_bb_vol, "RSI+BB+Vol"),
        (strategy_trend_pullback, "Trend Pullback"),
        (strategy_capitulation, "Capitulacion"),
        (strategy_strict_oversold, "Strict Oversold"),
        (strategy_mean_reversion, "Mean Reversion"),
        (strategy_bb_squeeze_breakout, "BB Squeeze"),
        (strategy_multi_confirm, "Multi Confirm"),
    ]

    tp_sl_configs = [
        (0.02, 0.01, "2:1"),
        (0.03, 0.015, "2:1"),
        (0.04, 0.02, "2:1"),
        (0.03, 0.01, "3:1"),
    ]

    print("\n" + "=" * 70)
    print("RESULTADOS EN TRAIN (para encontrar estrategia)")
    print("=" * 70)

    best_train = None
    best_train_score = -999

    for tp, sl, ratio in tp_sl_configs:
        print(f"\n--- TP={tp:.1%} SL={sl:.1%} ({ratio}) ---")
        print(f"{'Estrategia':<20} {'Trades':>8} {'WR':>8} {'PnL':>10} {'PF':>8}")
        print("-" * 60)

        for strat_func, strat_name in strategies:
            result = backtest_strategy(ind_train, strat_func, tp, sl, strat_name)

            if result['n_trades'] > 0:
                print(f"{strat_name:<20} {result['n_trades']:>8} {result['win_rate']:>7.1f}% {result['total_pnl']:>9.1f}% {result['pf']:>7.2f}")

                # Score: PnL pero penaliza pocos trades
                score = result['total_pnl'] if result['n_trades'] >= 20 else result['total_pnl'] * 0.5
                if score > best_train_score and result['win_rate'] > 50:
                    best_train_score = score
                    best_train = (strat_func, strat_name, tp, sl)
            else:
                print(f"{strat_name:<20} {'No trades':>8}")

    if best_train is None:
        print("\n[WARNING] Ninguna estrategia supero 50% WR en train")
        # Tomar la mejor por PnL aunque no supere 50%
        best_train_score = -999
        for tp, sl, ratio in tp_sl_configs:
            for strat_func, strat_name in strategies:
                result = backtest_strategy(ind_train, strat_func, tp, sl, strat_name)
                if result['n_trades'] >= 10 and result['total_pnl'] > best_train_score:
                    best_train_score = result['total_pnl']
                    best_train = (strat_func, strat_name, tp, sl)

    # Test con la mejor estrategia
    print("\n" + "=" * 70)
    print("EVALUACION EN TEST (out-of-sample)")
    print("=" * 70)

    if best_train:
        strat_func, strat_name, tp, sl = best_train
        print(f"\nMejor estrategia de train: {strat_name} (TP={tp:.1%}, SL={sl:.1%})")

        train_result = backtest_strategy(ind_train, strat_func, tp, sl, strat_name)
        test_result = backtest_strategy(ind_test, strat_func, tp, sl, strat_name)

        print(f"\n{'Set':<10} {'Trades':>8} {'WR':>8} {'PnL':>10} {'PF':>8}")
        print("-" * 50)
        print(f"{'Train':<10} {train_result['n_trades']:>8} {train_result['win_rate']:>7.1f}% {train_result['total_pnl']:>9.1f}% {train_result['pf']:>7.2f}")
        print(f"{'Test':<10} {test_result['n_trades']:>8} {test_result['win_rate']:>7.1f}% {test_result['total_pnl']:>9.1f}% {test_result['pf']:>7.2f}")

        # Comparar con random
        print("\n--- Comparacion con Random ---")
        random_results = []
        for _ in range(10):
            r = backtest_strategy(ind_test, strategy_random, tp, sl, "Random")
            if r['n_trades'] > 0:
                random_results.append(r)

        if random_results:
            avg_random_wr = np.mean([r['win_rate'] for r in random_results])
            avg_random_pnl = np.mean([r['total_pnl'] for r in random_results])
            print(f"Random promedio (10 runs): WR={avg_random_wr:.1f}%, PnL={avg_random_pnl:.1f}%")
            print(f"Estrategia:                WR={test_result['win_rate']:.1f}%, PnL={test_result['total_pnl']:.1f}%")

            edge_wr = test_result['win_rate'] - avg_random_wr
            edge_pnl = test_result['total_pnl'] - avg_random_pnl
            print(f"\nEdge sobre random: WR={edge_wr:+.1f}%, PnL={edge_pnl:+.1f}%")

        # Veredicto
        print("\n" + "=" * 70)
        if test_result['total_pnl'] > 0 and test_result['win_rate'] > 45:
            print("[VIABLE] Estrategia tiene edge positivo en test")
        elif test_result['total_pnl'] > 0:
            print("[MARGINAL] PnL positivo pero WR bajo")
        else:
            print("[NO VIABLE] Estrategia pierde dinero en test")
    else:
        print("No se encontro estrategia viable")

    # Probar TODAS las estrategias en test
    print("\n" + "=" * 70)
    print("TODAS LAS ESTRATEGIAS EN TEST (para comparar)")
    print("=" * 70)

    tp, sl = 0.03, 0.015  # Config estandar
    print(f"\nConfig: TP={tp:.1%} SL={sl:.1%}")
    print(f"{'Estrategia':<20} {'Trades':>8} {'WR':>8} {'PnL':>10} {'PF':>8}")
    print("-" * 60)

    for strat_func, strat_name in strategies:
        result = backtest_strategy(ind_test, strat_func, tp, sl, strat_name)
        if result['n_trades'] > 0:
            print(f"{strat_name:<20} {result['n_trades']:>8} {result['win_rate']:>7.1f}% {result['total_pnl']:>9.1f}% {result['pf']:>7.2f}")
        else:
            print(f"{strat_name:<20} {'No trades':>8}")


if __name__ == '__main__':
    main()
