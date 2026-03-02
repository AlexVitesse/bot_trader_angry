"""
BTC Hybrid System - V13.05
==========================
Enfoque hibrido: Setup tecnico + Filtro ML

Logica:
1. SETUP TECNICO genera senal de entrada potencial
2. ML FILTER decide si el contexto es favorable
3. Solo entramos si ambos coinciden

Esto imita a un trader profesional:
- Ve un setup (RSI oversold, soporte, etc.)
- Evalua el contexto (volatilidad, tendencia, etc.)
- Decide si entrar o esperar mejor oportunidad
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from datetime import timedelta
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

# Splits temporales
TRAIN_END = '2024-12-31'
VALIDATION_END = '2025-08-31'


def load_data():
    df = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_full.parquet')
    return df.sort_index()


def compute_features(df):
    """Features para el modelo."""
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    feat = pd.DataFrame(index=df.index)

    # === INDICADORES PARA SETUPS ===
    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)

    bb = ta.bbands(c, length=20)
    if bb is not None:
        feat['bb_pct'] = (c - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])
        feat['bb_width'] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / bb.iloc[:, 1] * 100

    ema20 = ta.ema(c, length=20)
    ema50 = ta.ema(c, length=50)
    ema200 = ta.ema(c, length=200)
    feat['ema20_dist'] = (c - ema20) / ema20 * 100
    feat['ema50_dist'] = (c - ema50) / ema50 * 100
    feat['ema200_dist'] = (c - ema200) / ema200 * 100

    # === INDICADORES PARA CONTEXTO ===
    adx = ta.adx(h, l, c, length=14)
    if adx is not None:
        feat['adx'] = adx.iloc[:, 0]
        feat['di_diff'] = adx.iloc[:, 1] - adx.iloc[:, 2]

    feat['atr_pct'] = ta.atr(h, l, c, length=14) / c * 100
    feat['vol_ratio'] = v / v.rolling(20).mean()

    feat['ret_1'] = c.pct_change(1) * 100
    feat['ret_5'] = c.pct_change(5) * 100
    feat['ret_20'] = c.pct_change(20) * 100

    # Choppiness
    atr = ta.atr(h, l, c, length=14)
    atr_sum = atr.rolling(14).sum()
    high_max = h.rolling(14).max()
    low_min = l.rolling(14).min()
    feat['chop'] = 100 * np.log10(atr_sum / (high_max - low_min + 0.0001)) / np.log10(14)

    # Velas consecutivas
    feat['consec_down'] = (feat['ret_1'] < 0).rolling(5).sum()
    feat['consec_up'] = (feat['ret_1'] > 0).rolling(5).sum()

    # Hora del dia (patron ciclico)
    feat['hour'] = df.index.hour
    feat['hour_sin'] = np.sin(2 * np.pi * feat['hour'] / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * feat['hour'] / 24)

    return feat


def detect_setups(feat):
    """
    Detecta setups tecnicos MAS SELECTIVOS.
    Menos setups = mayor calidad potencial.
    """
    setups = pd.DataFrame(index=feat.index)
    setups['has_setup'] = False
    setups['setup_type'] = 'NONE'

    for idx in feat.index:
        row = feat.loc[idx]

        rsi14 = row.get('rsi14', 50)
        rsi7 = row.get('rsi7', 50)
        bb_pct = row.get('bb_pct', 0.5)
        ema200_dist = row.get('ema200_dist', 0)
        ema20_dist = row.get('ema20_dist', 0)
        consec_down = row.get('consec_down', 0)
        vol_ratio = row.get('vol_ratio', 1)
        adx = row.get('adx', 20)

        # SETUP 1: RSI Extremo (mas estricto: < 25)
        if rsi14 < 25 and bb_pct < 0.3:
            setups.loc[idx, 'has_setup'] = True
            setups.loc[idx, 'setup_type'] = 'RSI_EXTREME'

        # SETUP 2: Capitulacion (selloff + volumen + RSI bajo)
        elif consec_down >= 4 and rsi14 < 35 and vol_ratio > 1.3:
            setups.loc[idx, 'has_setup'] = True
            setups.loc[idx, 'setup_type'] = 'CAPITULATION'

        # SETUP 3: Pullback en tendencia alcista
        elif ema200_dist > 5 and adx > 25 and rsi14 < 40 and ema20_dist < -1:
            setups.loc[idx, 'has_setup'] = True
            setups.loc[idx, 'setup_type'] = 'TREND_PULLBACK'

        # SETUP 4: Double bottom (RSI divergencia)
        elif rsi7 < 25 and rsi14 > rsi7 + 5 and bb_pct < 0.2:
            setups.loc[idx, 'has_setup'] = True
            setups.loc[idx, 'setup_type'] = 'DIVERGENCE'

    return setups


def label_outcomes(df, setups, tp_pct=0.03, sl_pct=0.015, max_hold=20):
    """
    Para cada setup, determinar si fue WIN o LOSS.
    Esto sera el target del ML filter.
    """
    outcomes = pd.Series(index=setups.index, dtype='object')
    outcomes[:] = 'NO_SETUP'

    setup_indices = setups[setups['has_setup'] == True].index

    for idx in setup_indices:
        entry_pos = df.index.get_loc(idx)
        entry_price = df.loc[idx, 'close']

        outcome = 'LOSS'  # Default

        for i in range(1, min(max_hold + 1, len(df) - entry_pos)):
            future_idx = df.index[entry_pos + i]
            future_price = df.loc[future_idx, 'close']
            pnl = (future_price - entry_price) / entry_price

            if pnl >= tp_pct:
                outcome = 'WIN'
                break
            elif pnl <= -sl_pct:
                outcome = 'LOSS'
                break

        outcomes[idx] = outcome

    return outcomes


def train_ml_filter(feat, setups, outcomes, train_mask):
    """
    Entrena el ML filter que predice si un setup sera WIN o LOSS.
    """
    # Features de contexto (NO del setup en si, sino del MERCADO)
    context_cols = ['adx', 'di_diff', 'chop', 'atr_pct', 'vol_ratio',
                    'ret_5', 'ret_20', 'ema200_dist', 'bb_width',
                    'hour_sin', 'hour_cos']

    available_cols = [c for c in context_cols if c in feat.columns]

    # Solo entrenar con setups que ocurrieron
    setup_mask = setups['has_setup'] == True
    combined_mask = train_mask & setup_mask

    X = feat.loc[combined_mask, available_cols]
    y = (outcomes[combined_mask] == 'WIN').astype(int)

    # Remover NaN
    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid]

    if len(X) < 100:
        print("  Insuficientes datos para entrenar")
        return None, available_cols

    print(f"  Training samples: {len(X)}")
    print(f"  Win rate en train: {y.mean()*100:.1f}%")

    # Entrenar modelo conservador
    model = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=42
    )
    model.fit(X, y)

    # Train accuracy
    train_pred = model.predict(X)
    train_acc = (train_pred == y).mean()
    print(f"  Train accuracy: {train_acc:.1%}")

    return model, available_cols


def backtest_hybrid(df, feat, setups, outcomes, ml_filter, context_cols, mask, threshold=0.5):
    """
    Backtest del sistema hibrido:
    - Setup tecnico detecta oportunidad
    - ML filter decide si entrar
    """
    df_period = df[mask]
    feat_period = feat[mask]
    setups_period = setups[mask]

    trades = []
    position = None

    for idx in df_period.index:
        price = df_period.loc[idx, 'close']

        # Check posicion abierta
        if position is not None:
            pnl = (price - position['entry']) / position['entry']
            if pnl >= 0.03:  # TP 3%
                trades.append({'pnl': pnl, 'result': 'TP', 'filtered': position['filtered']})
                position = None
            elif pnl <= -0.015:  # SL 1.5%
                trades.append({'pnl': pnl, 'result': 'SL', 'filtered': position['filtered']})
                position = None

        # Buscar entrada
        if position is None and setups_period.loc[idx, 'has_setup']:
            # Hay setup tecnico, consultar ML filter
            if ml_filter is not None:
                context = feat_period.loc[[idx], context_cols]
                if context.notna().all().all():
                    prob = ml_filter.predict_proba(context)[0, 1]
                    if prob >= threshold:
                        position = {'entry': price, 'filtered': True}
            else:
                # Sin filtro, entrar en todos los setups
                position = {'entry': price, 'filtered': False}

    # Cerrar posicion final
    if position is not None:
        final_price = df_period.iloc[-1]['close']
        pnl = (final_price - position['entry']) / position['entry']
        trades.append({'pnl': pnl, 'result': 'TIMEOUT', 'filtered': position['filtered']})

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
        'wr': wins / n * 100,
        'pnl': total_pnl,
        'pf': gross_profit / gross_loss if gross_loss > 0 else 999
    }


def main():
    print("=" * 70)
    print("BTC HYBRID SYSTEM - V13.05")
    print("Setup Tecnico + ML Filter")
    print("=" * 70)

    # Cargar datos
    print("\n[1/6] Cargando datos...")
    df = load_data()
    print(f"  Total: {len(df):,} candles")

    # Features
    print("\n[2/6] Calculando features...")
    feat = compute_features(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # Detectar setups
    print("\n[3/6] Detectando setups tecnicos...")
    setups = detect_setups(feat)
    n_setups = setups['has_setup'].sum()
    print(f"  Setups detectados: {n_setups:,} ({n_setups/len(setups)*100:.1f}%)")

    for setup_type in setups['setup_type'].value_counts().index:
        if setup_type != 'NONE':
            count = (setups['setup_type'] == setup_type).sum()
            print(f"    {setup_type}: {count}")

    # Etiquetar outcomes
    print("\n[4/6] Etiquetando outcomes (WIN/LOSS)...")
    outcomes = label_outcomes(df, setups)
    setup_outcomes = outcomes[setups['has_setup'] == True]
    win_rate_raw = (setup_outcomes == 'WIN').mean() * 100
    print(f"  Win rate SIN filtro: {win_rate_raw:.1f}%")

    # Split temporal
    train_mask = feat.index <= TRAIN_END
    val_mask = (feat.index > TRAIN_END) & (feat.index <= VALIDATION_END)
    test_mask = feat.index > VALIDATION_END

    # Entrenar ML filter
    print("\n[5/6] Entrenando ML filter...")
    ml_filter, context_cols = train_ml_filter(feat, setups, outcomes, train_mask)

    # Backtest
    print("\n[6/6] Backtesting...")

    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)

    # Sin filtro (baseline)
    print("\n--- SIN FILTRO ML (solo setup tecnico) ---")

    result_train_nf = backtest_hybrid(df, feat, setups, outcomes, None, context_cols, train_mask)
    result_val_nf = backtest_hybrid(df, feat, setups, outcomes, None, context_cols, val_mask)
    result_test_nf = backtest_hybrid(df, feat, setups, outcomes, None, context_cols, test_mask)

    print(f"{'Periodo':<12} {'Trades':>8} {'WR':>8} {'PnL':>10} {'PF':>8}")
    print("-" * 50)
    if result_train_nf:
        print(f"{'Train':<12} {result_train_nf['trades']:>8} {result_train_nf['wr']:>7.1f}% {result_train_nf['pnl']:>9.1f}% {result_train_nf['pf']:>7.2f}")
    if result_val_nf:
        print(f"{'Validation':<12} {result_val_nf['trades']:>8} {result_val_nf['wr']:>7.1f}% {result_val_nf['pnl']:>9.1f}% {result_val_nf['pf']:>7.2f}")
    if result_test_nf:
        print(f"{'Test':<12} {result_test_nf['trades']:>8} {result_test_nf['wr']:>7.1f}% {result_test_nf['pnl']:>9.1f}% {result_test_nf['pf']:>7.2f}")

    # Con filtro ML
    if ml_filter is not None:
        print("\n--- CON FILTRO ML (threshold=0.50) ---")

        result_train = backtest_hybrid(df, feat, setups, outcomes, ml_filter, context_cols, train_mask, threshold=0.50)
        result_val = backtest_hybrid(df, feat, setups, outcomes, ml_filter, context_cols, val_mask, threshold=0.50)
        result_test = backtest_hybrid(df, feat, setups, outcomes, ml_filter, context_cols, test_mask, threshold=0.50)

        print(f"{'Periodo':<12} {'Trades':>8} {'WR':>8} {'PnL':>10} {'PF':>8}")
        print("-" * 50)
        if result_train:
            print(f"{'Train':<12} {result_train['trades']:>8} {result_train['wr']:>7.1f}% {result_train['pnl']:>9.1f}% {result_train['pf']:>7.2f}")
        if result_val:
            print(f"{'Validation':<12} {result_val['trades']:>8} {result_val['wr']:>7.1f}% {result_val['pnl']:>9.1f}% {result_val['pf']:>7.2f}")
        if result_test:
            print(f"{'Test':<12} {result_test['trades']:>8} {result_test['wr']:>7.1f}% {result_test['pnl']:>9.1f}% {result_test['pf']:>7.2f}")

        # Probar diferentes thresholds
        print("\n--- COMPARACION DE THRESHOLDS (Test) ---")
        print(f"{'Threshold':<12} {'Trades':>8} {'WR':>8} {'PnL':>10} {'PF':>8}")
        print("-" * 50)

        for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
            result = backtest_hybrid(df, feat, setups, outcomes, ml_filter, context_cols, test_mask, threshold=thresh)
            if result and result['trades'] > 0:
                print(f"{thresh:<12} {result['trades']:>8} {result['wr']:>7.1f}% {result['pnl']:>9.1f}% {result['pf']:>7.2f}")

    # Veredicto
    print("\n" + "=" * 70)
    print("VEREDICTO")
    print("=" * 70)

    if result_test_nf and result_test:
        improvement_wr = result_test['wr'] - result_test_nf['wr'] if result_test else 0
        improvement_pnl = result_test['pnl'] - result_test_nf['pnl'] if result_test else 0

        print(f"\nMejora del filtro ML en TEST:")
        print(f"  WR: {improvement_wr:+.1f}%")
        print(f"  PnL: {improvement_pnl:+.1f}%")

        if result_test and result_test['pnl'] > 0:
            print("\n[VIABLE] El sistema hibrido genera PnL positivo en test")
        elif result_test and result_test['pnl'] > result_test_nf['pnl']:
            print("\n[MEJORA] El filtro ML mejora vs baseline pero aun negativo")
        else:
            print("\n[NO VIABLE] El filtro ML no mejora el sistema")


if __name__ == '__main__':
    main()
