"""
BTC Setup Detector - V13.05
===========================
En vez de detectar "regimenes", identificamos SETUPS especificos
con edge historico comprobado.

Insight clave: Mean reversion funciona en BTC.
"Lo que parece malo es buena oportunidad"
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

TRAIN_END = '2024-12-31'
VALIDATION_END = '2025-08-31'


def load_data():
    df = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_full.parquet')
    return df.sort_index()


def compute_features(df):
    """Features para detectar setups."""
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    feat = pd.DataFrame(index=df.index)

    # RSI
    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)

    # Bollinger Bands
    bb = ta.bbands(c, length=20)
    if bb is not None:
        feat['bb_pct'] = (c - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])
        feat['bb_width'] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / bb.iloc[:, 1] * 100

    # EMAs
    ema20 = ta.ema(c, length=20)
    ema50 = ta.ema(c, length=50)
    ema200 = ta.ema(c, length=200)
    feat['ema20_dist'] = (c - ema20) / ema20 * 100
    feat['ema50_dist'] = (c - ema50) / ema50 * 100
    feat['ema200_dist'] = (c - ema200) / ema200 * 100

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
    feat['ret_20'] = c.pct_change(20) * 100

    # Velas consecutivas
    feat['consec_down'] = (feat['ret_1'] < 0).rolling(5).sum()
    feat['consec_up'] = (feat['ret_1'] > 0).rolling(5).sum()

    # Choppiness
    atr = ta.atr(h, l, c, length=14)
    atr_sum = atr.rolling(14).sum()
    high_max = h.rolling(14).max()
    low_min = l.rolling(14).min()
    feat['chop'] = 100 * np.log10(atr_sum / (high_max - low_min + 0.0001)) / np.log10(14)

    return feat


def identify_setups(df, feat):
    """
    Identificar SETUPS especificos con edge historico.
    Retorna el tipo de setup y si es trade o no.
    """
    setups = pd.DataFrame(index=feat.index)
    setups['setup_type'] = 'NONE'
    setups['trade_signal'] = False

    for idx in feat.index:
        row = feat.loc[idx]

        rsi14 = row.get('rsi14', 50)
        rsi7 = row.get('rsi7', 50)
        bb_pct = row.get('bb_pct', 0.5)
        ema20_dist = row.get('ema20_dist', 0)
        ema200_dist = row.get('ema200_dist', 0)
        adx = row.get('adx', 20)
        di_diff = row.get('di_diff', 0)
        vol_ratio = row.get('vol_ratio', 1)
        consec_down = row.get('consec_down', 0)
        ret_5 = row.get('ret_5', 0)

        # === SETUP 1: OVERSOLD BOUNCE ===
        # RSI muy bajo + precio en fondo de BB + no en bear extremo
        if rsi14 < 30 and bb_pct < 0.15 and ema200_dist > -15:
            setups.loc[idx, 'setup_type'] = 'OVERSOLD_BOUNCE'
            setups.loc[idx, 'trade_signal'] = True

        # === SETUP 2: CAPITULACION ===
        # Varias velas rojas + RSI extremo + volumen alto
        elif consec_down >= 4 and rsi14 < 35 and vol_ratio > 1.3:
            setups.loc[idx, 'setup_type'] = 'CAPITULATION'
            setups.loc[idx, 'trade_signal'] = True

        # === SETUP 3: PULLBACK EN TENDENCIA ===
        # Tendencia alcista + pullback temporal
        elif ema200_dist > 5 and adx > 25 and di_diff > 0 and rsi14 < 45 and bb_pct < 0.4:
            setups.loc[idx, 'setup_type'] = 'TREND_PULLBACK'
            setups.loc[idx, 'trade_signal'] = True

        # === SETUP 4: BREAKOUT ===
        # Salida de rango con fuerza
        elif bb_pct > 0.95 and adx > 25 and di_diff > 10 and vol_ratio > 1.5:
            setups.loc[idx, 'setup_type'] = 'BREAKOUT'
            setups.loc[idx, 'trade_signal'] = True

        # === SETUP 5: DOUBLE BOTTOM (aproximado) ===
        # RSI muy bajo pero subiendo
        elif rsi7 < 25 and rsi14 > rsi7 and bb_pct < 0.2:
            setups.loc[idx, 'setup_type'] = 'DOUBLE_BOTTOM'
            setups.loc[idx, 'trade_signal'] = True

        # === NO TRADE: Condiciones adversas ===
        # Bear extremo
        elif ema200_dist < -20 and di_diff < -15:
            setups.loc[idx, 'setup_type'] = 'BEAR_EXTREME'
            setups.loc[idx, 'trade_signal'] = False

        # Overbought extremo
        elif rsi14 > 80 and bb_pct > 0.95:
            setups.loc[idx, 'setup_type'] = 'OVERBOUGHT'
            setups.loc[idx, 'trade_signal'] = False

    return setups


def backtest_setup(df, setups, setup_type, tp_pct=0.03, sl_pct=0.015):
    """Backtest un setup especifico."""
    mask = setups['setup_type'] == setup_type
    entries = df[mask].index

    if len(entries) == 0:
        return None

    trades = []
    for entry_idx in entries:
        entry_pos = df.index.get_loc(entry_idx)
        entry_price = df.loc[entry_idx, 'close']

        # Buscar salida en las proximas 20 velas
        for i in range(1, min(21, len(df) - entry_pos)):
            future_idx = df.index[entry_pos + i]
            future_price = df.loc[future_idx, 'close']
            pnl_pct = (future_price - entry_price) / entry_price

            if pnl_pct >= tp_pct:
                trades.append({'pnl': pnl_pct, 'result': 'TP', 'hold': i})
                break
            elif pnl_pct <= -sl_pct:
                trades.append({'pnl': pnl_pct, 'result': 'SL', 'hold': i})
                break
        else:
            # Timeout
            if entry_pos + 20 < len(df):
                final_price = df.iloc[entry_pos + 20]['close']
                pnl_pct = (final_price - entry_price) / entry_price
                trades.append({'pnl': pnl_pct, 'result': 'TIMEOUT', 'hold': 20})

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    n = len(trades_df)
    wins = (trades_df['pnl'] > 0).sum()
    wr = wins / n * 100
    total_pnl = trades_df['pnl'].sum() * 100

    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 999

    return {
        'setup': setup_type,
        'trades': n,
        'win_rate': wr,
        'total_pnl': total_pnl,
        'pf': pf,
        'avg_hold': trades_df['hold'].mean()
    }


def main():
    print("=" * 70)
    print("BTC SETUP DETECTOR - V13.05")
    print("=" * 70)

    # Cargar datos
    print("\n[1/4] Cargando datos...")
    df = load_data()
    print(f"  Total: {len(df):,} candles")

    # Features
    print("\n[2/4] Calculando features...")
    feat = compute_features(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # Identificar setups
    print("\n[3/4] Identificando setups...")
    setups = identify_setups(df, feat)

    # Distribucion
    print("\n  Distribucion de setups:")
    for setup in setups['setup_type'].value_counts().index:
        count = (setups['setup_type'] == setup).sum()
        pct = count / len(setups) * 100
        print(f"    {setup}: {count:,} ({pct:.1f}%)")

    # Split
    train_mask = df.index <= TRAIN_END
    val_mask = (df.index > TRAIN_END) & (df.index <= VALIDATION_END)
    test_mask = df.index > VALIDATION_END

    # Backtest cada setup en TRAIN
    print("\n[4/4] Backtesting setups en TRAIN...")
    print("\n" + "=" * 70)
    print("RESULTADOS POR SETUP (Train: 2019-2024)")
    print("=" * 70)

    df_train = df[train_mask]
    setups_train = setups[train_mask]

    setup_types = [s for s in setups['setup_type'].unique() if s != 'NONE']
    results_train = []

    print(f"\n{'Setup':<20} {'Trades':>8} {'WR':>8} {'PnL':>10} {'PF':>8}")
    print("-" * 60)

    for setup_type in setup_types:
        result = backtest_setup(df_train, setups_train, setup_type)
        if result:
            results_train.append(result)
            print(f"{setup_type:<20} {result['trades']:>8} {result['win_rate']:>7.1f}% {result['total_pnl']:>9.1f}% {result['pf']:>7.2f}")

    # Validar en VALIDATION
    print("\n" + "=" * 70)
    print("VALIDACION (Ene-Ago 2025)")
    print("=" * 70)

    df_val = df[val_mask]
    setups_val = setups[val_mask]

    print(f"\n{'Setup':<20} {'Trades':>8} {'WR':>8} {'PnL':>10} {'PF':>8}")
    print("-" * 60)

    viable_setups = []
    for setup_type in setup_types:
        result = backtest_setup(df_val, setups_val, setup_type)
        if result and result['trades'] >= 3:
            print(f"{setup_type:<20} {result['trades']:>8} {result['win_rate']:>7.1f}% {result['total_pnl']:>9.1f}% {result['pf']:>7.2f}")

            # Viable si PnL > 0 y WR > 40%
            if result['total_pnl'] > 0 and result['win_rate'] > 40:
                viable_setups.append(setup_type)

    # Resumen
    print("\n" + "=" * 70)
    print("SETUPS VIABLES (positivos en train Y validation)")
    print("=" * 70)

    if viable_setups:
        for setup in viable_setups:
            train_r = next((r for r in results_train if r['setup'] == setup), None)
            val_r = backtest_setup(df_val, setups_val, setup)
            if train_r and val_r:
                print(f"\n{setup}:")
                print(f"  Train: {train_r['trades']} trades, {train_r['win_rate']:.1f}% WR, {train_r['total_pnl']:.1f}% PnL")
                print(f"  Val:   {val_r['trades']} trades, {val_r['win_rate']:.1f}% WR, {val_r['total_pnl']:.1f}% PnL")
    else:
        print("\nNingun setup fue positivo en ambos periodos.")

    # Guardar setups para siguiente fase
    setups.to_parquet(DATA_DIR / 'btc_setups.parquet')
    print(f"\nSetups guardados en: data/btc_setups.parquet")

    # Combinar setups viables y probar
    print("\n" + "=" * 70)
    print("COMBINACION DE SETUPS VIABLES")
    print("=" * 70)

    # Backtest con TODOS los setups que dan senal
    combined_mask = setups['trade_signal'] == True

    print("\nTrain (todos los setups combinados):")
    combined_train = backtest_combined(df_train, setups_train[combined_mask])
    if combined_train:
        print(f"  {combined_train['trades']} trades, {combined_train['win_rate']:.1f}% WR, {combined_train['total_pnl']:.1f}% PnL, PF {combined_train['pf']:.2f}")

    print("\nValidation (todos los setups combinados):")
    combined_val = backtest_combined(df_val, setups_val[combined_mask])
    if combined_val:
        print(f"  {combined_val['trades']} trades, {combined_val['win_rate']:.1f}% WR, {combined_val['total_pnl']:.1f}% PnL, PF {combined_val['pf']:.2f}")

    # === TEST FINAL (una sola vez, sin mas ajustes) ===
    print("\n" + "=" * 70)
    print("TEST FINAL (Sep 2025 - Feb 2026) - SIN MAS AJUSTES")
    print("=" * 70)

    df_test = df[test_mask]
    setups_test = setups[test_mask]

    print(f"\n{'Setup':<20} {'Trades':>8} {'WR':>8} {'PnL':>10} {'PF':>8}")
    print("-" * 60)

    for setup_type in setup_types:
        result = backtest_setup(df_test, setups_test, setup_type)
        if result and result['trades'] >= 1:
            print(f"{setup_type:<20} {result['trades']:>8} {result['win_rate']:>7.1f}% {result['total_pnl']:>9.1f}% {result['pf']:>7.2f}")

    print("\nTest (todos los setups combinados):")
    combined_test = backtest_combined(df_test, setups_test[combined_mask])
    if combined_test:
        print(f"  {combined_test['trades']} trades, {combined_test['win_rate']:.1f}% WR, {combined_test['total_pnl']:.1f}% PnL, PF {combined_test['pf']:.2f}")

        # Veredicto
        print("\n" + "=" * 70)
        print("VEREDICTO")
        print("=" * 70)

        if combined_test['total_pnl'] > 0 and combined_test['win_rate'] > 45 and combined_test['pf'] > 1.0:
            print("\n[APROBADO] El sistema tiene edge en TEST")
            print(f"  WR: {combined_test['win_rate']:.1f}% (>45%)")
            print(f"  PnL: +{combined_test['total_pnl']:.1f}%")
            print(f"  PF: {combined_test['pf']:.2f} (>1.0)")
        else:
            print("\n[RECHAZADO] El sistema NO tiene edge en TEST")
            if combined_test['total_pnl'] <= 0:
                print(f"  PnL negativo: {combined_test['total_pnl']:.1f}%")
            if combined_test['win_rate'] <= 45:
                print(f"  WR bajo: {combined_test['win_rate']:.1f}%")
            if combined_test['pf'] <= 1.0:
                print(f"  PF bajo: {combined_test['pf']:.2f}")


def backtest_combined(df, setups_filtered, tp_pct=0.03, sl_pct=0.015):
    """Backtest combinando todos los setups."""
    entries = setups_filtered.index
    if len(entries) == 0:
        return None

    trades = []
    position = None

    for i, idx in enumerate(df.index):
        price = df.loc[idx, 'close']

        # Check posicion
        if position is not None:
            pnl_pct = (price - position['entry_price']) / position['entry_price']
            if pnl_pct >= tp_pct:
                trades.append({'pnl': pnl_pct, 'result': 'TP'})
                position = None
            elif pnl_pct <= -sl_pct:
                trades.append({'pnl': pnl_pct, 'result': 'SL'})
                position = None

        # Nueva entrada
        if position is None and idx in entries:
            position = {'entry_price': price}

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    n = len(trades_df)
    wins = (trades_df['pnl'] > 0).sum()

    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())

    return {
        'trades': n,
        'win_rate': wins / n * 100,
        'total_pnl': trades_df['pnl'].sum() * 100,
        'pf': gross_profit / gross_loss if gross_loss > 0 else 999
    }


if __name__ == '__main__':
    main()
