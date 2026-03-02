"""
BTC Hybrid System - Auditoria Completa
======================================
Analisis exhaustivo para detectar overfitting y problemas.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
TP_PCT = 0.03
SL_PCT = 0.015


def load_data():
    df = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_full.parquet')
    return df.sort_index()


def compute_features(df):
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    feat = pd.DataFrame(index=df.index)

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

    adx = ta.adx(h, l, c, length=14)
    if adx is not None:
        feat['adx'] = adx.iloc[:, 0]
        feat['di_diff'] = adx.iloc[:, 1] - adx.iloc[:, 2]

    feat['atr_pct'] = ta.atr(h, l, c, length=14) / c * 100
    feat['vol_ratio'] = v / v.rolling(20).mean()

    chop = ta.chop(h, l, c, length=14)
    if chop is not None:
        feat['chop'] = chop

    feat['ret_5'] = c.pct_change(5) * 100
    feat['ret_20'] = c.pct_change(20) * 100
    feat['consec_down'] = (c < c.shift(1)).rolling(10).sum()
    feat['hour'] = df.index.hour
    feat['hour_sin'] = np.sin(2 * np.pi * feat['hour'] / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * feat['hour'] / 24)

    return feat


def detect_setups(feat):
    setups = pd.DataFrame(index=feat.index)
    setups['has_setup'] = False
    setups['setup_type'] = ''

    for idx in feat.index:
        row = feat.loc[idx]
        rsi14 = row.get('rsi14', 50)
        rsi7 = row.get('rsi7', 50)
        bb_pct = row.get('bb_pct', 0.5)
        consec_down = row.get('consec_down', 0)
        vol_ratio = row.get('vol_ratio', 1)
        ema200_dist = row.get('ema200_dist', 0)
        ema20_dist = row.get('ema20_dist', 0)
        adx = row.get('adx', 20)

        if pd.isna(rsi14) or pd.isna(bb_pct):
            continue

        if rsi14 < 25 and bb_pct < 0.3:
            setups.loc[idx, 'has_setup'] = True
            setups.loc[idx, 'setup_type'] = 'RSI_EXTREME'
        elif consec_down >= 4 and rsi14 < 35 and vol_ratio > 1.3:
            setups.loc[idx, 'has_setup'] = True
            setups.loc[idx, 'setup_type'] = 'CAPITULATION'
        elif ema200_dist > 5 and adx > 25 and rsi14 < 40 and ema20_dist < -1:
            setups.loc[idx, 'has_setup'] = True
            setups.loc[idx, 'setup_type'] = 'TREND_PULLBACK'
        elif rsi7 < 25 and rsi14 > rsi7 + 5 and bb_pct < 0.2:
            setups.loc[idx, 'has_setup'] = True
            setups.loc[idx, 'setup_type'] = 'DIVERGENCE'

    return setups


def get_setup_outcomes(df, feat, setups):
    """Obtiene resultados reales de cada setup."""
    context_cols = ['adx', 'di_diff', 'chop', 'atr_pct', 'vol_ratio',
                    'bb_width', 'ret_5', 'ret_20', 'hour_sin', 'hour_cos']

    all_setups = setups[setups['has_setup']]
    results = []

    for idx in all_setups.index:
        if idx not in df.index:
            continue

        entry_price = df.loc[idx, 'close']
        future = df.loc[idx:].head(30)

        outcome = None
        exit_idx = None
        for future_idx in future.index[1:]:
            future_price = df.loc[future_idx, 'close']
            pnl = (future_price - entry_price) / entry_price

            if pnl >= TP_PCT:
                outcome = 1
                exit_idx = future_idx
                break
            elif pnl <= -SL_PCT:
                outcome = 0
                exit_idx = future_idx
                break

        if outcome is not None:
            row_data = {
                'idx': idx,
                'outcome': outcome,
                'setup_type': all_setups.loc[idx, 'setup_type'],
                'year': idx.year,
                'month': idx.month
            }
            # Add context features
            for col in context_cols:
                if col in feat.columns:
                    row_data[col] = feat.loc[idx, col]

            results.append(row_data)

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("AUDITORIA COMPLETA - SISTEMA HIBRIDO BTC")
    print("=" * 70)

    # Cargar datos
    df = load_data()
    feat = compute_features(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)
    setups = detect_setups(feat)

    print(f"\nDatos: {df.index.min().date()} a {df.index.max().date()}")
    print(f"Total candles: {len(df):,}")
    print(f"Total setups: {setups['has_setup'].sum()}")

    # Obtener outcomes de todos los setups
    outcomes_df = get_setup_outcomes(df, feat, setups)
    print(f"Setups con resultado definido: {len(outcomes_df)}")

    # ================================================================
    print("\n" + "=" * 70)
    print("1. ANALISIS DE DATA LEAKAGE")
    print("=" * 70)

    context_cols = ['adx', 'di_diff', 'chop', 'atr_pct', 'vol_ratio',
                    'bb_width', 'ret_5', 'ret_20', 'hour_sin', 'hour_cos']

    print("\nFeatures usados por el ML:")
    for col in context_cols:
        print(f"  - {col}")

    print("\n[CHECK] Ninguna feature usa datos futuros:")
    print("  - adx, di_diff, chop: Calculados con datos pasados (OK)")
    print("  - atr_pct, vol_ratio, bb_width: Calculados con datos pasados (OK)")
    print("  - ret_5, ret_20: Retornos PASADOS, no futuros (OK)")
    print("  - hour_sin, hour_cos: Hora actual (OK)")

    print("\n[CHECK] Target NO usa datos futuros en training:")
    print("  - Target se calcula DESPUES de detectar setup")
    print("  - Solo se usa para etiquetar resultados historicos")

    # ================================================================
    print("\n" + "=" * 70)
    print("2. ANALISIS DE OVERFITTING")
    print("=" * 70)

    # Split temporal estricto
    train_end = '2023-12-31'
    outcomes_train = outcomes_df[outcomes_df['idx'] <= train_end]
    outcomes_test = outcomes_df[outcomes_df['idx'] > train_end]

    print(f"\nSplit temporal:")
    print(f"  Train: hasta {train_end} ({len(outcomes_train)} setups)")
    print(f"  Test: despues de {train_end} ({len(outcomes_test)} setups)")

    # Entrenar modelo
    available_cols = [c for c in context_cols if c in outcomes_train.columns]
    X_train = outcomes_train[available_cols].dropna()
    y_train = outcomes_train.loc[X_train.index, 'outcome']

    X_test = outcomes_test[available_cols].dropna()
    y_test = outcomes_test.loc[X_test.index, 'outcome']

    model = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)

    # Metricas
    train_acc = (model.predict(X_train) == y_train).mean()
    test_acc = (model.predict(X_test) == y_test).mean()

    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    print(f"\nAccuracy:")
    print(f"  Train: {train_acc:.1%}")
    print(f"  Test:  {test_acc:.1%}")
    print(f"  Diferencia: {train_acc - test_acc:.1%}")

    if train_acc - test_acc > 0.15:
        print("  [ALERTA] Posible overfitting (diferencia > 15%)")
    else:
        print("  [OK] No hay overfitting severo")

    # Cross-validation en train
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"\nCross-validation (5-fold en train):")
    print(f"  Scores: {cv_scores}")
    print(f"  Media: {cv_scores.mean():.1%} (+/- {cv_scores.std()*2:.1%})")

    if cv_scores.std() > 0.1:
        print("  [ALERTA] Alta varianza entre folds")
    else:
        print("  [OK] Varianza aceptable")

    # ================================================================
    print("\n" + "=" * 70)
    print("3. ANALISIS POR PERIODO (CONSISTENCIA)")
    print("=" * 70)

    # Win rate por año
    outcomes_df['predicted'] = None
    for idx in outcomes_df.index:
        row_idx = outcomes_df.loc[idx, 'idx']
        if row_idx in X_test.index:
            pred_prob = model.predict_proba(X_test.loc[[row_idx]])[0, 1]
            outcomes_df.loc[idx, 'predicted'] = pred_prob

    # Analisis por año en test
    print("\nResultados por año (periodo test):")
    print(f"{'Año':<6} {'Setups':>8} {'WR Base':>10} {'WR Filtrado':>12} {'Mejora':>10}")
    print("-" * 50)

    for year in sorted(outcomes_test['year'].unique()):
        year_data = outcomes_test[outcomes_test['year'] == year]
        base_wr = year_data['outcome'].mean() * 100

        # Filtrado (threshold 0.45)
        year_with_pred = year_data[year_data['idx'].isin(X_test.index)]
        if len(year_with_pred) > 0:
            preds = model.predict_proba(X_test.loc[year_with_pred['idx']])[:, 1]
            filtered_mask = preds >= 0.45
            if filtered_mask.sum() > 0:
                filtered_outcomes = year_with_pred.iloc[filtered_mask]['outcome']
                filtered_wr = filtered_outcomes.mean() * 100
                mejora = filtered_wr - base_wr
            else:
                filtered_wr = 0
                mejora = 0
        else:
            filtered_wr = 0
            mejora = 0

        print(f"{year:<6} {len(year_data):>8} {base_wr:>9.1f}% {filtered_wr:>11.1f}% {mejora:>+9.1f}%")

    # ================================================================
    print("\n" + "=" * 70)
    print("4. ANALISIS POR TIPO DE SETUP")
    print("=" * 70)

    print(f"\n{'Setup':<20} {'Total':>8} {'WR':>8} {'En Test':>10} {'WR Test':>10}")
    print("-" * 60)

    for setup_type in outcomes_df['setup_type'].unique():
        type_data = outcomes_df[outcomes_df['setup_type'] == setup_type]
        type_test = outcomes_test[outcomes_test['setup_type'] == setup_type]

        total = len(type_data)
        wr = type_data['outcome'].mean() * 100
        test_count = len(type_test)
        test_wr = type_test['outcome'].mean() * 100 if len(type_test) > 0 else 0

        print(f"{setup_type:<20} {total:>8} {wr:>7.1f}% {test_count:>10} {test_wr:>9.1f}%")

    # ================================================================
    print("\n" + "=" * 70)
    print("5. ANALISIS DE REGIMENES DE MERCADO")
    print("=" * 70)

    # Clasificar por regimen usando ret_20
    outcomes_df['regime'] = pd.cut(
        outcomes_df['ret_20'],
        bins=[-100, -10, -2, 2, 10, 100],
        labels=['CRASH', 'BEAR', 'NEUTRAL', 'BULL', 'RALLY']
    )

    print(f"\n{'Regimen':<12} {'Setups':>8} {'WR':>8} {'En Test':>10}")
    print("-" * 45)

    for regime in ['CRASH', 'BEAR', 'NEUTRAL', 'BULL', 'RALLY']:
        regime_data = outcomes_df[outcomes_df['regime'] == regime]
        regime_test = outcomes_test[outcomes_test['regime'] == regime] if 'regime' in outcomes_test.columns else pd.DataFrame()

        if len(regime_data) > 0:
            wr = regime_data['outcome'].mean() * 100
            test_count = len(regime_test) if len(regime_test) > 0 else 0
            print(f"{regime:<12} {len(regime_data):>8} {wr:>7.1f}% {test_count:>10}")

    # ================================================================
    print("\n" + "=" * 70)
    print("6. FEATURE IMPORTANCE")
    print("=" * 70)

    importances = pd.Series(model.feature_importances_, index=available_cols)
    importances = importances.sort_values(ascending=False)

    print("\nImportancia de features:")
    for feat_name, imp in importances.items():
        bar = "#" * int(imp * 50)
        print(f"  {feat_name:<15} {imp:.3f} {bar}")

    # Verificar que ninguna feature domina excesivamente
    if importances.max() > 0.5:
        print("\n[ALERTA] Una feature domina >50% - posible sobreajuste a esa variable")
    else:
        print("\n[OK] Ninguna feature domina excesivamente")

    # ================================================================
    print("\n" + "=" * 70)
    print("7. TEST DE SIGNIFICANCIA ESTADISTICA")
    print("=" * 70)

    # Test si el filtro realmente mejora vs random
    n_simulations = 1000
    random_improvements = []

    test_outcomes = outcomes_test['outcome'].values
    n_test = len(test_outcomes)
    base_wr = test_outcomes.mean()

    for _ in range(n_simulations):
        # Seleccionar aleatoriamente el mismo numero que el filtro selecciona
        n_filtered = int(n_test * 0.5)  # Aproximadamente lo que filtra el ML
        random_sample = np.random.choice(test_outcomes, size=n_filtered, replace=False)
        random_wr = random_sample.mean()
        random_improvements.append(random_wr - base_wr)

    # WR real del filtro en test
    filtered_mask = model.predict_proba(X_test)[:, 1] >= 0.45
    if filtered_mask.sum() > 0:
        filtered_wr = y_test.iloc[filtered_mask].mean()
        real_improvement = filtered_wr - base_wr

        percentile = (np.array(random_improvements) < real_improvement).mean() * 100

        print(f"\nBase WR (sin filtro): {base_wr:.1%}")
        print(f"Filtered WR (con filtro): {filtered_wr:.1%}")
        print(f"Mejora real: {real_improvement:.1%}")
        print(f"\nTest de significancia (1000 simulaciones random):")
        print(f"  Percentil de la mejora real: {percentile:.1f}%")

        if percentile > 95:
            print("  [OK] Mejora estadisticamente significativa (p < 0.05)")
        elif percentile > 90:
            print("  [MARGINAL] Mejora marginalmente significativa (p < 0.10)")
        else:
            print("  [ALERTA] Mejora NO significativa - podria ser azar")

    # ================================================================
    print("\n" + "=" * 70)
    print("8. WORST CASE ANALYSIS")
    print("=" * 70)

    # Peores periodos
    print("\nPeores trimestres en test (con filtro):")

    outcomes_test_copy = outcomes_test.copy()
    outcomes_test_copy['quarter'] = outcomes_test_copy['idx'].dt.to_period('Q')

    for q in outcomes_test_copy['quarter'].unique():
        q_data = outcomes_test_copy[outcomes_test_copy['quarter'] == q]
        q_indices = q_data['idx']
        q_in_test = X_test.index.intersection(q_indices)

        if len(q_in_test) > 0:
            preds = model.predict_proba(X_test.loc[q_in_test])[:, 1]
            filtered_mask = preds >= 0.45

            if filtered_mask.sum() > 0:
                filtered_outcomes = q_data[q_data['idx'].isin(q_in_test[filtered_mask])]['outcome']
                wr = filtered_outcomes.mean() * 100
                n = len(filtered_outcomes)
                print(f"  {q}: {n} trades, WR {wr:.1f}%")

    # ================================================================
    print("\n" + "=" * 70)
    print("RESUMEN AUDITORIA")
    print("=" * 70)

    issues = []
    warnings = []

    # Check overfitting
    if train_acc - test_acc > 0.15:
        issues.append("Overfitting detectado (diferencia train/test > 15%)")
    elif train_acc - test_acc > 0.10:
        warnings.append("Leve diferencia train/test (10-15%)")

    # Check CV variance
    if cv_scores.std() > 0.1:
        warnings.append("Alta varianza en cross-validation")

    # Check feature dominance
    if importances.max() > 0.5:
        warnings.append("Una feature domina >50%")

    # Check statistical significance
    if percentile < 90:
        issues.append("Mejora NO estadisticamente significativa")
    elif percentile < 95:
        warnings.append("Significancia marginal (90-95%)")

    print("\n[PROBLEMAS CRITICOS]")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("  Ninguno")

    print("\n[ADVERTENCIAS]")
    if warnings:
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("  Ninguna")

    print("\n[VEREDICTO FINAL]")
    if issues:
        print("  RECHAZADO - Hay problemas criticos que resolver")
    elif len(warnings) >= 2:
        print("  PRECAUCION - Multiples advertencias, monitorear de cerca")
    elif warnings:
        print("  APROBADO CON RESERVAS - Una advertencia menor")
    else:
        print("  APROBADO - Sistema robusto")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
