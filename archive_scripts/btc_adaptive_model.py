"""
BTC Adaptive Model - V13.05
===========================
Modelo que se re-entrena periodicamente con datos recientes.

Simula lo que haria un trader profesional:
- Usa datos recientes (ultimos 6 meses para entrenar)
- Re-entrena cada mes
- Se adapta a las condiciones actuales del mercado
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')

# Configuracion del modelo adaptativo
TRAIN_WINDOW_MONTHS = 6  # Entrenar con ultimos 6 meses
RETRAIN_FREQUENCY = '1M'  # Re-entrenar cada mes
MIN_TRADES_PER_MONTH = 5  # Minimo de trades para validar


def load_data():
    df = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_full.parquet')
    return df.sort_index()


def compute_features(df):
    """Features para el modelo."""
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    feat = pd.DataFrame(index=df.index)

    # RSI
    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)

    # Bollinger
    bb = ta.bbands(c, length=20)
    if bb is not None:
        feat['bb_pct'] = (c - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])

    # EMAs
    ema20 = ta.ema(c, length=20)
    ema50 = ta.ema(c, length=50)
    feat['ema20_dist'] = (c - ema20) / ema20 * 100
    feat['ema50_dist'] = (c - ema50) / ema50 * 100

    # ADX
    adx = ta.adx(h, l, c, length=14)
    if adx is not None:
        feat['adx'] = adx.iloc[:, 0]
        feat['di_diff'] = adx.iloc[:, 1] - adx.iloc[:, 2]

    # Volatilidad
    feat['atr_pct'] = ta.atr(h, l, c, length=14) / c * 100

    # Volumen
    feat['vol_ratio'] = v / v.rolling(20).mean()

    # Retornos
    feat['ret_1'] = c.pct_change(1) * 100
    feat['ret_5'] = c.pct_change(5) * 100

    # Target: Retorno positivo en proximas 5 velas (clasificacion binaria)
    feat['target'] = (c.shift(-5) > c).astype(int)

    return feat


def generate_signals(feat, model, threshold=0.55):
    """Generar senales de trading."""
    feature_cols = ['rsi14', 'rsi7', 'bb_pct', 'ema20_dist', 'ema50_dist',
                    'adx', 'di_diff', 'atr_pct', 'vol_ratio', 'ret_1', 'ret_5']

    available_cols = [c for c in feature_cols if c in feat.columns]
    X = feat[available_cols].dropna()

    if len(X) == 0:
        return pd.Series(index=feat.index, data=False)

    probs = model.predict_proba(X)[:, 1]
    signals = pd.Series(index=X.index, data=probs > threshold)

    return signals


def backtest_period(df, signals, tp_pct=0.03, sl_pct=0.015):
    """Backtest de un periodo."""
    trades = []
    position = None

    signal_indices = signals[signals == True].index

    for idx in df.index:
        price = df.loc[idx, 'close']

        # Check posicion abierta
        if position is not None:
            pnl_pct = (price - position['entry']) / position['entry']
            if pnl_pct >= tp_pct:
                trades.append({'pnl': pnl_pct, 'result': 'TP'})
                position = None
            elif pnl_pct <= -sl_pct:
                trades.append({'pnl': pnl_pct, 'result': 'SL'})
                position = None

        # Nueva entrada
        if position is None and idx in signal_indices:
            position = {'entry': price}

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    n = len(trades_df)
    wins = (trades_df['pnl'] > 0).sum()
    total_pnl = trades_df['pnl'].sum() * 100

    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())

    return {
        'trades': n,
        'wins': wins,
        'wr': wins / n * 100 if n > 0 else 0,
        'pnl': total_pnl,
        'pf': gross_profit / gross_loss if gross_loss > 0 else 999
    }


def simulate_adaptive_trading(df, feat, start_date='2025-01-01'):
    """
    Simula trading adaptativo:
    - Cada mes, entrena con los ultimos 6 meses
    - Tradea el mes siguiente
    - Re-entrena al inicio del proximo mes
    """
    print("\n" + "=" * 70)
    print("SIMULACION DE TRADING ADAPTATIVO")
    print("=" * 70)
    print(f"\nConfiguracion:")
    print(f"  Ventana de entrenamiento: {TRAIN_WINDOW_MONTHS} meses")
    print(f"  Re-entrenamiento: Mensual")
    print(f"  Inicio simulacion: {start_date}")

    # Generar fechas de re-entrenamiento
    start = pd.Timestamp(start_date, tz='UTC')
    end = df.index.max()
    if end.tzinfo is None:
        end = end.tz_localize('UTC')

    retrain_dates = pd.date_range(start=start, end=end, freq='MS')  # Inicio de cada mes

    all_results = []
    cumulative_pnl = 0

    feature_cols = ['rsi14', 'rsi7', 'bb_pct', 'ema20_dist', 'ema50_dist',
                    'adx', 'di_diff', 'atr_pct', 'vol_ratio', 'ret_1', 'ret_5']

    print(f"\n{'Mes':<12} {'Trades':>8} {'WR':>8} {'PnL':>10} {'Cum PnL':>12}")
    print("-" * 55)

    for i, retrain_date in enumerate(retrain_dates[:-1]):
        # Periodo de trading: este mes
        trade_start = retrain_date
        trade_end = retrain_dates[i + 1] if i + 1 < len(retrain_dates) else end

        # Periodo de entrenamiento: 6 meses antes
        train_end = retrain_date - timedelta(days=1)
        train_start = train_end - timedelta(days=TRAIN_WINDOW_MONTHS * 30)

        # Datos de entrenamiento
        train_mask = (feat.index >= train_start) & (feat.index <= train_end)
        feat_train = feat[train_mask].dropna()

        if len(feat_train) < 500:
            continue

        # Features y target
        available_cols = [c for c in feature_cols if c in feat_train.columns]
        X_train = feat_train[available_cols]
        y_train = feat_train['target']

        # Entrenar modelo
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Datos de trading
        trade_mask = (feat.index >= trade_start) & (feat.index < trade_end)
        feat_trade = feat[trade_mask]
        df_trade = df[trade_mask]

        if len(feat_trade) < 10:
            continue

        # Generar senales
        signals = generate_signals(feat_trade, model, threshold=0.55)

        # Backtest
        result = backtest_period(df_trade, signals)

        month_str = trade_start.strftime('%Y-%m')

        if result and result['trades'] > 0:
            cumulative_pnl += result['pnl']
            all_results.append({
                'month': month_str,
                'trades': result['trades'],
                'wr': result['wr'],
                'pnl': result['pnl'],
                'pf': result['pf']
            })
            print(f"{month_str:<12} {result['trades']:>8} {result['wr']:>7.1f}% {result['pnl']:>9.1f}% {cumulative_pnl:>11.1f}%")
        else:
            print(f"{month_str:<12} {'0':>8} {'-':>8} {'-':>10} {cumulative_pnl:>11.1f}%")

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN TRADING ADAPTATIVO")
    print("=" * 70)

    if all_results:
        results_df = pd.DataFrame(all_results)
        total_trades = results_df['trades'].sum()
        avg_wr = results_df['wr'].mean()
        total_pnl = results_df['pnl'].sum()
        months_positive = (results_df['pnl'] > 0).sum()
        months_total = len(results_df)

        print(f"\n  Meses simulados: {months_total}")
        print(f"  Meses positivos: {months_positive} ({months_positive/months_total*100:.0f}%)")
        print(f"  Total trades: {total_trades}")
        print(f"  WR promedio: {avg_wr:.1f}%")
        print(f"  PnL total: {total_pnl:.1f}%")
        print(f"  PnL mensual promedio: {total_pnl/months_total:.1f}%")

        # Comparar con buy & hold
        bh_start = df.loc[df.index >= start_date].iloc[0]['close']
        bh_end = df.iloc[-1]['close']
        bh_return = (bh_end - bh_start) / bh_start * 100

        print(f"\n  Buy & Hold mismo periodo: {bh_return:.1f}%")
        print(f"  Diferencia vs B&H: {total_pnl - bh_return:+.1f}%")

        # Veredicto
        print("\n" + "=" * 70)
        if total_pnl > 0 and avg_wr > 45 and months_positive >= months_total * 0.5:
            print("[VIABLE] El modelo adaptativo muestra edge consistente")
        elif total_pnl > 0:
            print("[MARGINAL] PnL positivo pero inconsistente")
        else:
            print("[NO VIABLE] El modelo adaptativo no funciona")

        return results_df
    else:
        print("\nNo hay resultados suficientes")
        return None


def main():
    print("=" * 70)
    print("BTC ADAPTIVE MODEL - V13.05")
    print("=" * 70)

    # Cargar datos
    print("\n[1/3] Cargando datos...")
    df = load_data()
    print(f"  Total: {len(df):,} candles ({df.index.min().date()} a {df.index.max().date()})")

    # Features
    print("\n[2/3] Calculando features...")
    feat = compute_features(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # Simulacion adaptativa
    print("\n[3/3] Simulando trading adaptativo...")
    results = simulate_adaptive_trading(df, feat, start_date='2025-01-01')

    if results is not None:
        # Guardar resultados
        results.to_csv('btc_adaptive_results.csv', index=False)
        print(f"\nResultados guardados en: btc_adaptive_results.csv")


if __name__ == '__main__':
    main()
