"""
Discover Setups - Encuentra los setups que funcionan para un asset
Uso: python discover_setups.py --symbol ETH/USDT
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import ccxt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ta

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACION
# =============================================================================
TIMEFRAME = '4h'
TP_PCT = 0.06
SL_PCT = 0.03
TIMEOUT_CANDLES = 20

# =============================================================================
# DATOS
# =============================================================================
def load_data(symbol: str) -> pd.DataFrame:
    """Carga o descarga datos"""
    symbol_clean = symbol.replace('/', '')
    csv_path = f'data/{symbol_clean}_4h.csv'

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        print(f"Cargado: {len(df)} candles")
        return df

    print(f"Descargando {symbol}...")
    exchange = ccxt.binance()
    since = exchange.parse8601('2018-01-01T00:00:00Z')
    all_ohlcv = []

    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < 1000:
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    Path('data').mkdir(exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Guardado: {len(df)} candles")
    return df

# =============================================================================
# INDICADORES
# =============================================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todos los indicadores tecnicos"""
    df = df.copy()

    # Tendencia
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_extreme_oversold'] = (df['rsi'] < 20).astype(int)
    df['rsi_extreme_overbought'] = (df['rsi'] > 80).astype(int)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['bb_squeeze'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_mid']).rolling(20).rank(pct=True)

    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
    df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)

    # ADX y DI
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df['di_plus'] = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=14)
    df['di_minus'] = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=14)
    df['di_cross_up'] = ((df['di_plus'] > df['di_minus']) & (df['di_plus'].shift(1) <= df['di_minus'].shift(1))).astype(int)
    df['di_cross_down'] = ((df['di_plus'] < df['di_minus']) & (df['di_plus'].shift(1) >= df['di_minus'].shift(1))).astype(int)

    # Volumen
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    df['volume_spike'] = (df['volume_ratio'] > 2).astype(int)
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['obv_sma'] = df['obv'].rolling(20).mean()
    df['obv_trend_up'] = (df['obv'] > df['obv_sma']).astype(int)

    # Velas
    df['body'] = abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    df['body_pct'] = df['body'] / (df['range'] + 1e-10)
    df['bullish_candle'] = (df['close'] > df['open']).astype(int)
    df['bearish_candle'] = (df['close'] < df['open']).astype(int)
    df['big_candle'] = (df['body'] > df['body'].rolling(20).mean() * 1.5).astype(int)

    # Momentum
    df['roc_5'] = df['close'].pct_change(5)
    df['roc_10'] = df['close'].pct_change(10)
    df['roc_20'] = df['close'].pct_change(20)
    df['momentum_up'] = (df['roc_5'] > 0).astype(int)
    df['momentum_down'] = (df['roc_5'] < 0).astype(int)

    # Retornos
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_3'] = df['close'].pct_change(3)
    df['green_streak'] = df['bullish_candle'].rolling(3).sum()
    df['red_streak'] = df['bearish_candle'].rolling(3).sum()

    # Soporte/Resistencia (rolling high/low)
    df['resistance_20'] = df['high'].rolling(20).max()
    df['support_20'] = df['low'].rolling(20).min()
    df['near_resistance'] = (df['close'] > df['resistance_20'] * 0.98).astype(int)
    df['near_support'] = (df['close'] < df['support_20'] * 1.02).astype(int)
    df['breakout_up'] = (df['close'] > df['resistance_20'].shift(1)).astype(int)
    df['breakout_down'] = (df['close'] < df['support_20'].shift(1)).astype(int)

    # ATR
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['atr_pct'] = df['atr'] / df['close']
    df['high_volatility'] = (df['atr_pct'] > df['atr_pct'].rolling(50).quantile(0.8)).astype(int)

    # Regimen
    df['uptrend'] = ((df['close'] > df['sma_50']) & (df['sma_50'] > df['sma_200'])).astype(int)
    df['downtrend'] = ((df['close'] < df['sma_50']) & (df['sma_50'] < df['sma_200'])).astype(int)
    df['ranging'] = ((~df['uptrend'].astype(bool)) & (~df['downtrend'].astype(bool))).astype(int)

    return df.dropna()

# =============================================================================
# CREAR LABELS
# =============================================================================
def create_trade_labels(df: pd.DataFrame, direction: str) -> pd.Series:
    """Crea labels basados en si el trade ganaria"""
    labels = []

    for i in range(len(df) - TIMEOUT_CANDLES - 1):
        entry = df['close'].iloc[i]
        future = df.iloc[i+1:i+TIMEOUT_CANDLES+1]

        won = False
        lost = False

        if direction == 'LONG':
            tp = entry * (1 + TP_PCT)
            sl = entry * (1 - SL_PCT)
            for _, row in future.iterrows():
                if row['high'] >= tp:
                    won = True
                    break
                if row['low'] <= sl:
                    lost = True
                    break
        else:
            tp = entry * (1 - TP_PCT)
            sl = entry * (1 + SL_PCT)
            for _, row in future.iterrows():
                if row['low'] <= tp:
                    won = True
                    break
                if row['high'] >= sl:
                    lost = True
                    break

        if won:
            labels.append(1)
        elif lost:
            labels.append(0)
        else:
            # Timeout - check final price
            final = future['close'].iloc[-1] if len(future) > 0 else entry
            if direction == 'LONG':
                labels.append(1 if final > entry else 0)
            else:
                labels.append(1 if final < entry else 0)

    return pd.Series(labels, index=df.index[:len(labels)])

# =============================================================================
# DESCUBRIR SETUPS
# =============================================================================
def discover_setups(df: pd.DataFrame, direction: str, min_trades: int = 50) -> list:
    """Descubre combinaciones de condiciones que funcionan"""

    labels = create_trade_labels(df, direction)
    df_labeled = df.iloc[:len(labels)].copy()
    df_labeled['win'] = labels.values

    # Condiciones a probar
    conditions = {
        # RSI
        'rsi_oversold': 'rsi_oversold == 1',
        'rsi_overbought': 'rsi_overbought == 1',
        'rsi_extreme_oversold': 'rsi_extreme_oversold == 1',
        'rsi_extreme_overbought': 'rsi_extreme_overbought == 1',
        'rsi_neutral': '(rsi > 40) & (rsi < 60)',

        # Bollinger
        'bb_low': 'bb_pct < 0.2',
        'bb_high': 'bb_pct > 0.8',
        'bb_middle': '(bb_pct > 0.4) & (bb_pct < 0.6)',
        'bb_squeeze': 'bb_squeeze < 0.2',

        # MACD
        'macd_cross_up': 'macd_cross_up == 1',
        'macd_cross_down': 'macd_cross_down == 1',
        'macd_positive': 'macd_hist > 0',
        'macd_negative': 'macd_hist < 0',

        # ADX/DI
        'trending': 'adx > 25',
        'ranging': 'adx < 20',
        'di_bullish': 'di_plus > di_minus',
        'di_bearish': 'di_plus < di_minus',
        'di_cross_up': 'di_cross_up == 1',
        'di_cross_down': 'di_cross_down == 1',

        # Volume
        'volume_spike': 'volume_spike == 1',
        'obv_up': 'obv_trend_up == 1',
        'obv_down': 'obv_trend_up == 0',
        'high_volume': 'volume_ratio > 1.5',

        # Candles
        'bullish_candle': 'bullish_candle == 1',
        'bearish_candle': 'bearish_candle == 1',
        'big_candle': 'big_candle == 1',
        'green_streak': 'green_streak >= 3',
        'red_streak': 'red_streak >= 3',

        # Momentum
        'momentum_up': 'momentum_up == 1',
        'momentum_down': 'momentum_down == 1',
        'strong_momentum_up': 'roc_5 > 0.05',
        'strong_momentum_down': 'roc_5 < -0.05',

        # Position
        'near_support': 'near_support == 1',
        'near_resistance': 'near_resistance == 1',
        'breakout_up': 'breakout_up == 1',
        'breakout_down': 'breakout_down == 1',

        # Trend
        'uptrend': 'uptrend == 1',
        'downtrend': 'downtrend == 1',
        'above_sma20': 'close > sma_20',
        'below_sma20': 'close < sma_20',
        'above_sma50': 'close > sma_50',
        'below_sma50': 'close < sma_50',

        # Volatility
        'high_volatility': 'high_volatility == 1',
        'low_volatility': 'high_volatility == 0',
    }

    results = []
    base_wr = df_labeled['win'].mean()

    print(f"\nBase win rate ({direction}): {base_wr:.1%}")
    print(f"\nProbando condiciones individuales...")

    # Probar condiciones individuales
    for name, cond in conditions.items():
        try:
            mask = df_labeled.eval(cond)
            subset = df_labeled[mask]

            if len(subset) >= min_trades:
                wr = subset['win'].mean()
                edge = wr - base_wr

                if edge > 0.05:  # Al menos 5% mejor que base
                    results.append({
                        'name': name,
                        'condition': cond,
                        'trades': len(subset),
                        'win_rate': wr,
                        'edge': edge
                    })
        except Exception as e:
            pass

    # Ordenar por edge
    results.sort(key=lambda x: x['edge'], reverse=True)

    return results

def combine_top_setups(df: pd.DataFrame, direction: str, top_setups: list, min_trades: int = 30) -> list:
    """Combina los mejores setups para encontrar combinaciones ganadoras"""

    labels = create_trade_labels(df, direction)
    df_labeled = df.iloc[:len(labels)].copy()
    df_labeled['win'] = labels.values

    combined_results = []
    base_wr = df_labeled['win'].mean()

    print(f"\nCombinando top {len(top_setups)} setups...")

    # Probar combinaciones de 2
    for i, setup1 in enumerate(top_setups[:10]):
        for setup2 in top_setups[i+1:10]:
            try:
                combined_cond = f"({setup1['condition']}) & ({setup2['condition']})"
                mask = df_labeled.eval(combined_cond)
                subset = df_labeled[mask]

                if len(subset) >= min_trades:
                    wr = subset['win'].mean()
                    edge = wr - base_wr

                    if edge > 0.10:  # Al menos 10% mejor
                        combined_results.append({
                            'name': f"{setup1['name']} + {setup2['name']}",
                            'condition': combined_cond,
                            'trades': len(subset),
                            'win_rate': wr,
                            'edge': edge
                        })
            except:
                pass

    combined_results.sort(key=lambda x: x['edge'], reverse=True)
    return combined_results

# =============================================================================
# BACKTEST SETUP
# =============================================================================
def backtest_setup(df: pd.DataFrame, condition: str, direction: str) -> dict:
    """Hace backtest de un setup especifico"""

    df = df.copy()
    labels = create_trade_labels(df, direction)
    df = df.iloc[:len(labels)]
    df['win'] = labels.values

    # Walk-forward
    n_folds = 10
    fold_size = len(df) // (n_folds + 1)

    results = []
    for fold in range(n_folds):
        test_start = (fold + 1) * fold_size
        test_end = test_start + fold_size

        if test_end > len(df):
            break

        test_df = df.iloc[test_start:test_end]

        try:
            mask = test_df.eval(condition)
            trades = test_df[mask]

            if len(trades) > 0:
                wins = trades['win'].sum()
                total = len(trades)
                wr = wins / total
                pnl = wins * TP_PCT - (total - wins) * SL_PCT

                results.append({
                    'fold': fold,
                    'trades': total,
                    'wr': wr,
                    'pnl': pnl * 100
                })
        except:
            pass

    if not results:
        return None

    positive_folds = len([r for r in results if r['pnl'] > 0])
    total_pnl = sum(r['pnl'] for r in results)
    avg_wr = np.mean([r['wr'] for r in results])

    return {
        'folds': len(results),
        'positive_folds': positive_folds,
        'win_rate': positive_folds / len(results) if results else 0,
        'total_pnl': total_pnl,
        'avg_wr': avg_wr
    }

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, required=True)
    args = parser.parse_args()

    print("=" * 70)
    print(f"DESCUBRIENDO SETUPS PARA: {args.symbol}")
    print("=" * 70)

    df = load_data(args.symbol)
    df = compute_indicators(df)

    print(f"\nDatos preparados: {len(df)} candles")
    print(f"Periodo: {df['timestamp'].min()} a {df['timestamp'].max()}")

    final_setups = []

    for direction in ['LONG', 'SHORT']:
        print(f"\n{'='*70}")
        print(f"DIRECCION: {direction}")
        print("=" * 70)

        # Descubrir setups individuales
        top_setups = discover_setups(df, direction)

        if top_setups:
            print(f"\nTop 10 condiciones individuales:")
            for i, setup in enumerate(top_setups[:10]):
                print(f"  {i+1}. {setup['name']}: WR {setup['win_rate']:.1%}, "
                      f"Edge +{setup['edge']:.1%}, Trades: {setup['trades']}")

            # Combinar
            combined = combine_top_setups(df, direction, top_setups)

            if combined:
                print(f"\nTop 5 combinaciones:")
                for i, setup in enumerate(combined[:5]):
                    print(f"  {i+1}. {setup['name']}: WR {setup['win_rate']:.1%}, "
                          f"Edge +{setup['edge']:.1%}, Trades: {setup['trades']}")

                # Backtest top 3
                print(f"\nBacktest walk-forward de top 3:")
                for setup in combined[:3]:
                    bt = backtest_setup(df, setup['condition'], direction)
                    if bt:
                        status = "OK" if bt['win_rate'] >= 0.7 else "REVISAR"
                        print(f"  {setup['name']}: {bt['positive_folds']}/{bt['folds']} folds+, "
                              f"PnL {bt['total_pnl']:+.1f}% [{status}]")

                        if bt['win_rate'] >= 0.7 and bt['total_pnl'] > 0:
                            final_setups.append({
                                'direction': direction,
                                'name': setup['name'],
                                'condition': setup['condition'],
                                'backtest': bt
                            })

    # Resumen
    print("\n" + "=" * 70)
    print("SETUPS APROBADOS")
    print("=" * 70)

    if final_setups:
        for setup in final_setups:
            print(f"\n{setup['direction']} - {setup['name']}")
            print(f"  Condicion: {setup['condition']}")
            print(f"  Backtest: {setup['backtest']['positive_folds']}/{setup['backtest']['folds']} folds+, "
                  f"PnL {setup['backtest']['total_pnl']:+.1f}%")
    else:
        print("\nNo se encontraron setups que pasen el backtest walk-forward.")
        print("Considera ajustar los thresholds o usar mas datos.")

if __name__ == '__main__':
    main()
