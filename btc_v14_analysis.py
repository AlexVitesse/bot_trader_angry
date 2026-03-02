"""
BTC V14 - Analisis Completo
===========================
1. Analisis de setups (cuales funcionan)
2. Walk-forward en multiples periodos
3. Feature importance

Sin overfitting: validamos en MULTIPLES periodos, no optimizamos para uno.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')


def load_data():
    df = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_full.parquet')
    return df.sort_index()


def compute_features(df):
    """Features completas."""
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    feat = pd.DataFrame(index=df.index)

    # Trend
    adx_df = ta.adx(h, l, c, length=14)
    if adx_df is not None:
        feat['adx'] = adx_df.iloc[:, 0]
        feat['di_plus'] = adx_df.iloc[:, 1]
        feat['di_minus'] = adx_df.iloc[:, 2]
        feat['di_diff'] = feat['di_plus'] - feat['di_minus']

    chop = ta.chop(h, l, c, length=14)
    feat['chop'] = chop if chop is not None else 50

    # EMAs
    feat['ema20'] = ta.ema(c, length=20)
    feat['ema50'] = ta.ema(c, length=50)
    feat['ema200'] = ta.ema(c, length=200)
    feat['ema20_dist'] = (c - feat['ema20']) / feat['ema20'] * 100
    feat['ema50_dist'] = (c - feat['ema50']) / feat['ema50'] * 100
    feat['ema200_dist'] = (c - feat['ema200']) / feat['ema200'] * 100
    feat['ema20_slope'] = feat['ema20'].pct_change(5) * 100

    # Volatility
    feat['atr_pct'] = ta.atr(h, l, c, length=14) / c * 100
    bb = ta.bbands(c, length=20)
    if bb is not None:
        feat['bb_upper'] = bb.iloc[:, 2]
        feat['bb_lower'] = bb.iloc[:, 0]
        feat['bb_mid'] = bb.iloc[:, 1]
        feat['bb_width'] = (feat['bb_upper'] - feat['bb_lower']) / feat['bb_mid'] * 100
        feat['bb_pct'] = (c - feat['bb_lower']) / (feat['bb_upper'] - feat['bb_lower'])

    # Momentum
    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)
    stoch = ta.stoch(h, l, c, k=14, d=3)
    if stoch is not None:
        feat['stoch_k'] = stoch.iloc[:, 0]

    feat['ret_1'] = c.pct_change(1) * 100
    feat['ret_5'] = c.pct_change(5) * 100
    feat['ret_20'] = c.pct_change(20) * 100

    # Volume
    feat['vol_ratio'] = v / v.rolling(20).mean()
    feat['vol_trend'] = v.rolling(5).mean() / v.rolling(20).mean()

    # Structure
    feat['high_20'] = h.rolling(20).max()
    feat['low_20'] = l.rolling(20).min()
    feat['range_pos'] = (c - feat['low_20']) / (feat['high_20'] - feat['low_20'])
    feat['consec_down'] = (c < c.shift(1)).rolling(10).sum()
    feat['consec_up'] = (c > c.shift(1)).rolling(10).sum()

    return feat


def detect_regime(row):
    """Detecta regimen."""
    adx = row.get('adx', 20)
    di_diff = row.get('di_diff', 0)
    chop = row.get('chop', 50)
    atr_pct = row.get('atr_pct', 2)
    ema20_slope = row.get('ema20_slope', 0)

    if pd.isna(adx):
        return 'RANGE'

    if atr_pct > 4:
        return 'VOLATILE'
    if adx > 25 and chop < 50:
        if di_diff > 5:
            return 'TREND_UP'
        elif di_diff < -5:
            return 'TREND_DOWN'
    if chop > 55 or adx < 20:
        return 'RANGE'

    return 'RANGE'


# =============================================================================
# SETUP DEFINITIONS - Vamos a probar cada uno
# =============================================================================

SETUP_DEFINITIONS = {
    # LONG setups
    'PULLBACK_UPTREND': {
        'direction': 'long',
        'conditions': lambda r: r['rsi14'] < 40 and r['bb_pct'] < 0.3 and r['ema200_dist'] > 0,
        'regime': ['TREND_UP']
    },
    'OVERSOLD_EXTREME': {
        'direction': 'long',
        'conditions': lambda r: r['rsi14'] < 25 and r['bb_pct'] < 0.2,
        'regime': ['RANGE', 'TREND_UP']
    },
    'SUPPORT_BOUNCE': {
        'direction': 'long',
        'conditions': lambda r: r['range_pos'] < 0.15 and r['rsi14'] < 35,
        'regime': ['RANGE']
    },
    'CAPITULATION': {
        'direction': 'long',
        'conditions': lambda r: r['consec_down'] >= 4 and r['rsi14'] < 30 and r['vol_ratio'] > 1.5,
        'regime': ['RANGE', 'TREND_DOWN']
    },

    # SHORT setups
    'RALLY_DOWNTREND': {
        'direction': 'short',
        'conditions': lambda r: r['rsi14'] > 60 and r['bb_pct'] > 0.7 and r['ema200_dist'] < 0,
        'regime': ['TREND_DOWN']
    },
    'OVERBOUGHT_EXTREME': {
        'direction': 'short',
        'conditions': lambda r: r['rsi14'] > 75 and r['bb_pct'] > 0.8,
        'regime': ['RANGE', 'TREND_DOWN']
    },
    'RESISTANCE_REJECTION': {
        'direction': 'short',
        'conditions': lambda r: r['range_pos'] > 0.85 and r['rsi14'] > 65,
        'regime': ['RANGE']
    },
    'EXHAUSTION': {
        'direction': 'short',
        'conditions': lambda r: r['consec_up'] >= 4 and r['rsi14'] > 70 and r['vol_ratio'] > 1.5,
        'regime': ['RANGE', 'TREND_UP']
    },
}


def detect_all_setups(df, feat):
    """Detecta todos los setups definidos."""
    setups = []

    for idx in feat.index:
        row = feat.loc[idx]
        regime = detect_regime(row)

        # Check NaN in critical fields
        if pd.isna(row.get('rsi14')) or pd.isna(row.get('bb_pct')):
            continue

        for setup_name, setup_def in SETUP_DEFINITIONS.items():
            # Check regime
            if regime not in setup_def['regime']:
                continue

            # Check conditions
            try:
                if setup_def['conditions'](row):
                    setups.append({
                        'idx': idx,
                        'setup': setup_name,
                        'direction': setup_def['direction'],
                        'regime': regime
                    })
            except:
                continue

    return pd.DataFrame(setups) if setups else None


def get_setup_outcome(df, idx, direction, tp=0.03, sl=0.015):
    """Obtiene resultado de un setup."""
    if idx not in df.index:
        return None

    entry_price = df.loc[idx, 'close']
    future = df.loc[idx:].head(50)

    for future_idx in future.index[1:]:
        future_price = df.loc[future_idx, 'close']

        if direction == 'long':
            pnl = (future_price - entry_price) / entry_price
        else:
            pnl = (entry_price - future_price) / entry_price

        if pnl >= tp:
            return {'outcome': 1, 'pnl': pnl}
        elif pnl <= -sl:
            return {'outcome': 0, 'pnl': -sl}

    return None


def analyze_setups(df, feat, setups_df):
    """
    Analisis 1: Cual setup funciona mejor?
    Sin ML, solo resultados crudos.
    """
    print("\n" + "=" * 70)
    print("ANALISIS 1: RENDIMIENTO DE CADA SETUP (Sin ML)")
    print("=" * 70)

    results = []

    for setup_name in setups_df['setup'].unique():
        subset = setups_df[setups_df['setup'] == setup_name]
        direction = subset.iloc[0]['direction']

        outcomes = []
        for idx in subset['idx']:
            result = get_setup_outcome(df, idx, direction)
            if result:
                outcomes.append(result)

        if len(outcomes) >= 10:
            outcomes_df = pd.DataFrame(outcomes)
            n = len(outcomes_df)
            wr = outcomes_df['outcome'].mean() * 100
            total_pnl = outcomes_df['pnl'].sum() * 100

            results.append({
                'setup': setup_name,
                'direction': direction,
                'trades': n,
                'wr': wr,
                'pnl': total_pnl
            })

    results_df = pd.DataFrame(results).sort_values('pnl', ascending=False)

    print(f"\n{'Setup':<25} {'Dir':<6} {'Trades':>7} {'WR':>7} {'PnL':>8}")
    print("-" * 60)

    for _, row in results_df.iterrows():
        status = "[OK]" if row['pnl'] > 0 else "[BAD]"
        print(f"{row['setup']:<25} {row['direction']:<6} {row['trades']:>7} {row['wr']:>6.1f}% {row['pnl']:>+7.1f}% {status}")

    return results_df


def walk_forward_analysis(df, feat, setups_df, train_months=12, test_months=6):
    """
    Analisis 2: Walk-forward en multiples periodos.
    """
    print("\n" + "=" * 70)
    print("ANALISIS 2: WALK-FORWARD VALIDATION")
    print("=" * 70)
    print(f"Train: {train_months} meses, Test: {test_months} meses")

    # Generate folds
    start_date = df.index.min()
    end_date = df.index.max()
    first_test = start_date + pd.DateOffset(months=train_months)

    folds = []
    current = first_test

    while current + pd.DateOffset(months=test_months) <= end_date:
        folds.append({
            'train_end': current - pd.Timedelta(days=1),
            'test_start': current,
            'test_end': current + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        })
        current += pd.DateOffset(months=test_months)

    print(f"Total folds: {len(folds)}")

    # Test each fold
    fold_results = []

    print(f"\n{'Fold':<5} {'Test Period':<25} {'Trades':>7} {'WR':>7} {'PnL':>8}")
    print("-" * 60)

    for i, fold in enumerate(folds):
        # Get setups in test period
        test_setups = setups_df[
            (setups_df['idx'] >= fold['test_start']) &
            (setups_df['idx'] <= fold['test_end'])
        ]

        if len(test_setups) == 0:
            continue

        # Calculate outcomes
        outcomes = []
        for _, row in test_setups.iterrows():
            result = get_setup_outcome(df, row['idx'], row['direction'])
            if result:
                outcomes.append(result)

        if len(outcomes) > 0:
            outcomes_df = pd.DataFrame(outcomes)
            n = len(outcomes_df)
            wr = outcomes_df['outcome'].mean() * 100
            pnl = outcomes_df['pnl'].sum() * 100

            fold_results.append({
                'fold': i + 1,
                'period': f"{fold['test_start'].strftime('%Y-%m')} to {fold['test_end'].strftime('%Y-%m')}",
                'trades': n,
                'wr': wr,
                'pnl': pnl
            })

            print(f"{i+1:<5} {fold_results[-1]['period']:<25} {n:>7} {wr:>6.1f}% {pnl:>+7.1f}%")

    # Summary
    if fold_results:
        results_df = pd.DataFrame(fold_results)
        positive_folds = (results_df['pnl'] > 0).sum()
        total_pnl = results_df['pnl'].sum()
        avg_wr = results_df['wr'].mean()

        print(f"\n{'='*60}")
        print(f"RESUMEN WALK-FORWARD")
        print(f"  Folds positivos: {positive_folds}/{len(results_df)} ({positive_folds/len(results_df)*100:.0f}%)")
        print(f"  WR promedio: {avg_wr:.1f}%")
        print(f"  PnL total: {total_pnl:+.1f}%")
        print(f"  PnL promedio por fold: {total_pnl/len(results_df):+.1f}%")

        return results_df

    return None


def feature_importance_analysis(df, feat, setups_df):
    """
    Analisis 3: Que features predicen mejor WIN/LOSS?
    """
    print("\n" + "=" * 70)
    print("ANALISIS 3: FEATURE IMPORTANCE")
    print("=" * 70)

    # Get outcomes for all setups
    all_outcomes = []
    feature_cols = ['adx', 'di_diff', 'chop', 'atr_pct', 'bb_width', 'bb_pct',
                    'rsi14', 'rsi7', 'stoch_k', 'ret_5', 'ret_20',
                    'vol_ratio', 'vol_trend', 'range_pos']

    for _, row in setups_df.iterrows():
        result = get_setup_outcome(df, row['idx'], row['direction'])
        if result and row['idx'] in feat.index:
            feat_row = feat.loc[row['idx']]
            data = {'outcome': result['outcome'], 'direction': row['direction']}
            for col in feature_cols:
                if col in feat_row:
                    data[col] = feat_row[col]
            all_outcomes.append(data)

    if len(all_outcomes) < 50:
        print("No hay suficientes datos")
        return

    outcomes_df = pd.DataFrame(all_outcomes)

    # Train model to get feature importance
    available_cols = [c for c in feature_cols if c in outcomes_df.columns]
    X = outcomes_df[available_cols].dropna()
    y = outcomes_df.loc[X.index, 'outcome']

    if len(y) < 50:
        print("No hay suficientes datos despues de dropna")
        return

    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X, y)

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=available_cols).sort_values(ascending=False)

    print(f"\nFeature Importance (RandomForest):")
    print("-" * 50)
    for feat_name, imp in importance.items():
        bar = "#" * int(imp * 40)
        print(f"  {feat_name:<15} {imp:.3f} {bar}")

    # Correlation with outcome
    print(f"\nCorrelacion con Outcome:")
    print("-" * 50)

    correlations = []
    for col in available_cols:
        if col in outcomes_df.columns:
            corr = outcomes_df[col].corr(outcomes_df['outcome'])
            correlations.append({'feature': col, 'corr': corr})

    corr_df = pd.DataFrame(correlations).sort_values('corr', key=abs, ascending=False)

    for _, row in corr_df.iterrows():
        direction = "+" if row['corr'] > 0 else "-"
        bar = "#" * int(abs(row['corr']) * 30)
        print(f"  {row['feature']:<15} {row['corr']:>+.3f} {bar}")

    # Analisis por direccion
    print(f"\n\nANALISIS POR DIRECCION:")
    print("=" * 50)

    for direction in ['long', 'short']:
        subset = outcomes_df[outcomes_df['direction'] == direction]
        if len(subset) > 20:
            print(f"\n{direction.upper()}:")
            X_dir = subset[available_cols].dropna()
            y_dir = subset.loc[X_dir.index, 'outcome']

            if len(y_dir) > 20:
                model_dir = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
                model_dir.fit(X_dir, y_dir)

                imp_dir = pd.Series(model_dir.feature_importances_, index=available_cols).sort_values(ascending=False)
                for feat_name, imp in imp_dir.head(5).items():
                    print(f"  {feat_name:<15} {imp:.3f}")

    return importance


def analyze_by_market_condition(df, feat, setups_df):
    """
    Analisis adicional: Como funcionan los setups en diferentes condiciones?
    """
    print("\n" + "=" * 70)
    print("ANALISIS 4: RENDIMIENTO POR CONDICION DE MERCADO")
    print("=" * 70)

    # Add market condition to setups
    conditions = []
    for idx in setups_df['idx']:
        if idx in feat.index:
            ret_20 = feat.loc[idx, 'ret_20']
            atr = feat.loc[idx, 'atr_pct']

            if pd.isna(ret_20):
                condition = 'UNKNOWN'
            elif ret_20 > 10:
                condition = 'STRONG_BULL'
            elif ret_20 > 2:
                condition = 'BULL'
            elif ret_20 < -10:
                condition = 'STRONG_BEAR'
            elif ret_20 < -2:
                condition = 'BEAR'
            else:
                condition = 'NEUTRAL'

            conditions.append(condition)
        else:
            conditions.append('UNKNOWN')

    setups_df = setups_df.copy()
    setups_df['market_condition'] = conditions

    # Analyze by condition
    print(f"\n{'Condicion':<15} {'Dir':<6} {'Trades':>7} {'WR':>7} {'PnL':>8}")
    print("-" * 50)

    for condition in ['STRONG_BULL', 'BULL', 'NEUTRAL', 'BEAR', 'STRONG_BEAR']:
        for direction in ['long', 'short']:
            subset = setups_df[(setups_df['market_condition'] == condition) &
                              (setups_df['direction'] == direction)]

            if len(subset) < 5:
                continue

            outcomes = []
            for _, row in subset.iterrows():
                result = get_setup_outcome(df, row['idx'], direction)
                if result:
                    outcomes.append(result)

            if len(outcomes) >= 5:
                outcomes_df = pd.DataFrame(outcomes)
                n = len(outcomes_df)
                wr = outcomes_df['outcome'].mean() * 100
                pnl = outcomes_df['pnl'].sum() * 100

                status = "[OK]" if pnl > 0 else ""
                print(f"{condition:<15} {direction:<6} {n:>7} {wr:>6.1f}% {pnl:>+7.1f}% {status}")


def main():
    print("=" * 70)
    print("BTC V14 - ANALISIS COMPLETO")
    print("=" * 70)

    # Load data
    print("\nCargando datos...")
    df = load_data()
    print(f"  {len(df):,} candles ({df.index.min().date()} a {df.index.max().date()})")

    # Features
    print("\nCalculando features...")
    feat = compute_features(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # Detect all setups
    print("\nDetectando setups...")
    setups_df = detect_all_setups(df, feat)

    if setups_df is None or len(setups_df) == 0:
        print("No se detectaron setups")
        return

    print(f"  Total setups: {len(setups_df)}")
    print(f"  LONG: {len(setups_df[setups_df['direction'] == 'long'])}")
    print(f"  SHORT: {len(setups_df[setups_df['direction'] == 'short'])}")

    # Analisis 1: Setup performance
    setup_results = analyze_setups(df, feat, setups_df)

    # Analisis 2: Walk-forward
    wf_results = walk_forward_analysis(df, feat, setups_df)

    # Analisis 3: Feature importance
    feature_imp = feature_importance_analysis(df, feat, setups_df)

    # Analisis 4: Market condition
    analyze_by_market_condition(df, feat, setups_df)

    # Conclusiones
    print("\n" + "=" * 70)
    print("CONCLUSIONES")
    print("=" * 70)

    if setup_results is not None:
        good_setups = setup_results[setup_results['pnl'] > 0]['setup'].tolist()
        bad_setups = setup_results[setup_results['pnl'] <= 0]['setup'].tolist()

        print(f"\nSetups RENTABLES:")
        for s in good_setups:
            row = setup_results[setup_results['setup'] == s].iloc[0]
            print(f"  - {s}: WR {row['wr']:.1f}%, PnL {row['pnl']:+.1f}%")

        print(f"\nSetups NO RENTABLES (considerar eliminar):")
        for s in bad_setups:
            row = setup_results[setup_results['setup'] == s].iloc[0]
            print(f"  - {s}: WR {row['wr']:.1f}%, PnL {row['pnl']:+.1f}%")

    if wf_results is not None:
        consistency = (wf_results['pnl'] > 0).mean() * 100
        if consistency >= 60:
            print(f"\n[OK] Sistema consistente: {consistency:.0f}% de periodos positivos")
        else:
            print(f"\n[WARN] Baja consistencia: {consistency:.0f}% de periodos positivos")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
