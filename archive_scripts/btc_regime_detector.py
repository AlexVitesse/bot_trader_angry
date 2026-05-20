"""
BTC Regime Detector - Fase 1 V13.05
===================================
Detecta el regimen de mercado actual para adaptar estrategia.

Regimenes:
- BULL_TREND: Tendencia alcista clara
- BEAR_TREND: Tendencia bajista clara
- RANGE: Mercado lateral
- VOLATILE: Alta volatilidad sin direccion clara

Metodologia:
- Train: 2019-2024
- Validation: 2025-01 a 2025-08
- Test: 2025-09+ (intocable hasta evaluacion final)
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)

# Splits temporales
TRAIN_END = '2024-12-31'
VALIDATION_END = '2025-08-31'


def load_data():
    """Cargar datos BTC."""
    df = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_full.parquet')
    return df.sort_index()


def compute_regime_features(df):
    """
    Features para detectar regimen de mercado.
    Enfoque: indicadores de tendencia, volatilidad y momentum.
    """
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    feat = pd.DataFrame(index=df.index)

    # === TENDENCIA ===

    # ADX - Fuerza de tendencia (alto = tendencia fuerte)
    adx_data = ta.adx(h, l, c, length=14)
    if adx_data is not None:
        feat['adx'] = adx_data.iloc[:, 0]  # ADX
        feat['dmp'] = adx_data.iloc[:, 1]  # DI+
        feat['dmn'] = adx_data.iloc[:, 2]  # DI-
        feat['di_diff'] = feat['dmp'] - feat['dmn']  # Direccion

    # EMA Stack - Alineacion de medias
    ema20 = ta.ema(c, length=20)
    ema50 = ta.ema(c, length=50)
    ema200 = ta.ema(c, length=200)

    feat['ema20_dist'] = (c - ema20) / ema20 * 100
    feat['ema50_dist'] = (c - ema50) / ema50 * 100
    feat['ema200_dist'] = (c - ema200) / ema200 * 100

    # EMA alignment score (-3 a +3)
    feat['ema_align'] = (
        (c > ema20).astype(int) +
        (c > ema50).astype(int) +
        (c > ema200).astype(int) +
        (ema20 > ema50).astype(int) +
        (ema50 > ema200).astype(int) +
        (ema20 > ema200).astype(int)
    ) - 3  # Centrado en 0

    # === VOLATILIDAD ===

    # ATR y ATR percentile
    atr = ta.atr(h, l, c, length=14)
    feat['atr_pct'] = atr / c * 100
    feat['atr_percentile'] = feat['atr_pct'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )

    # Bollinger Band Width
    bb = ta.bbands(c, length=20)
    if bb is not None and len(bb.columns) >= 3:
        bbl, bbm, bbu = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
        feat['bb_width'] = (bbu - bbl) / bbm * 100
        feat['bb_pct'] = (c - bbl) / (bbu - bbl)

    # Volatilidad historica
    feat['hvol_20'] = c.pct_change().rolling(20).std() * 100
    feat['hvol_50'] = c.pct_change().rolling(50).std() * 100
    feat['vol_ratio'] = feat['hvol_20'] / feat['hvol_50']  # >1 = vol subiendo

    # === MOMENTUM ===

    # RSI
    feat['rsi14'] = ta.rsi(c, length=14)

    # ROC (Rate of Change)
    feat['roc_10'] = ta.roc(c, length=10)
    feat['roc_20'] = ta.roc(c, length=20)

    # Retornos
    feat['ret_5'] = c.pct_change(5) * 100
    feat['ret_20'] = c.pct_change(20) * 100

    # === VOLUMEN ===
    feat['vol_ma_ratio'] = v / v.rolling(20).mean()

    # === CHOPPINESS INDEX ===
    # Mide si el mercado es trending o choppy
    # Alto (>61.8) = choppy, Bajo (<38.2) = trending
    atr_sum = atr.rolling(14).sum()
    high_max = h.rolling(14).max()
    low_min = l.rolling(14).min()
    feat['chop'] = 100 * np.log10(atr_sum / (high_max - low_min)) / np.log10(14)

    return feat


def label_regimes_rules(df, feat):
    """
    Etiquetar regimenes usando reglas claras.

    Enfoque simplificado basado en el analisis de datos:
    - BEAR_TREND: 35% pos rate, -2% ret -> NO TRADE
    - RANGE baja vol: 52% pos -> Cuidado
    - VOLATILE/BULL: 54-57% pos -> TRADE

    Simplificamos a: FAVORABLE vs UNFAVORABLE
    """

    labels = pd.Series(index=feat.index, dtype='object')

    for idx in feat.index:
        row = feat.loc[idx]

        # Valores
        adx = row.get('adx', 20)
        di_diff = row.get('di_diff', 0)
        ema200_dist = row.get('ema200_dist', 0)
        ema50_dist = row.get('ema50_dist', 0)
        chop = row.get('chop', 50)
        rsi14 = row.get('rsi14', 50)
        ret_20 = row.get('ret_20', 0)
        bb_pct = row.get('bb_pct', 0.5)

        # === UNFAVORABLE: No tradear ===

        # BEAR_TREND claro: Precio bajo EMAs + momentum bajista
        if ema200_dist < -5 and ema50_dist < -3 and di_diff < -5:
            labels[idx] = 'UNFAVORABLE'

        # Sobrecompra extrema: RSI muy alto + precio muy arriba
        elif rsi14 > 75 and bb_pct > 0.95:
            labels[idx] = 'UNFAVORABLE'

        # Mercado muy choppy sin direccion
        elif chop > 65 and abs(di_diff) < 5 and adx < 20:
            labels[idx] = 'UNFAVORABLE'

        # === FAVORABLE: Tradear ===

        # Tendencia alcista clara
        elif adx > 25 and di_diff > 5 and ema200_dist > 0:
            labels[idx] = 'FAVORABLE'

        # Oversold con tendencia no bajista (rebote probable)
        elif rsi14 < 35 and ema200_dist > -10:
            labels[idx] = 'FAVORABLE'

        # Volatilidad + momentum positivo (breakout)
        elif di_diff > 3 and ret_20 > 2:
            labels[idx] = 'FAVORABLE'

        # === NEUTRAL: Esperar mejor setup ===
        else:
            labels[idx] = 'NEUTRAL'

    return labels


def smooth_labels(labels, min_periods=3):
    """
    Suavizar etiquetas para evitar cambios muy rapidos.
    Un regimen debe mantenerse al menos min_periods velas.
    """
    smoothed = labels.copy()
    current_regime = labels.iloc[0]
    regime_count = 1

    for i in range(1, len(labels)):
        if labels.iloc[i] == current_regime:
            regime_count += 1
        else:
            if regime_count < min_periods:
                # Mantener regimen anterior
                for j in range(i - regime_count, i):
                    smoothed.iloc[j] = labels.iloc[max(0, i - regime_count - 1)]
            current_regime = labels.iloc[i]
            regime_count = 1

    return smoothed


def analyze_regime_performance(df, labels):
    """
    Analizar que pasa en cada regimen (retornos futuros).
    """
    analysis = df.copy()
    analysis['regime'] = labels
    analysis['future_ret_5'] = df['close'].pct_change(5).shift(-5) * 100
    analysis['future_ret_20'] = df['close'].pct_change(20).shift(-20) * 100

    print("\n=== ANALISIS DE REGIMENES ===\n")

    regime_stats = []
    for regime in labels.unique():
        if pd.isna(regime):
            continue
        mask = analysis['regime'] == regime
        count = mask.sum()
        pct = count / len(analysis) * 100

        future_5 = analysis.loc[mask, 'future_ret_5'].dropna()
        future_20 = analysis.loc[mask, 'future_ret_20'].dropna()

        stats = {
            'regime': regime,
            'count': count,
            'pct': pct,
            'avg_ret_5': future_5.mean() if len(future_5) > 0 else 0,
            'avg_ret_20': future_20.mean() if len(future_20) > 0 else 0,
            'pos_rate_5': (future_5 > 0).mean() * 100 if len(future_5) > 0 else 0,
            'pos_rate_20': (future_20 > 0).mean() * 100 if len(future_20) > 0 else 0,
        }
        regime_stats.append(stats)

        print(f"{regime}:")
        print(f"  Frecuencia: {count:,} velas ({pct:.1f}%)")
        print(f"  Ret 5 velas: {stats['avg_ret_5']:.2f}% (pos: {stats['pos_rate_5']:.1f}%)")
        print(f"  Ret 20 velas: {stats['avg_ret_20']:.2f}% (pos: {stats['pos_rate_20']:.1f}%)")
        print()

    return pd.DataFrame(regime_stats)


def train_regime_classifier(feat, labels, train_mask, val_mask):
    """
    Entrenar clasificador de regimenes.
    """
    # Features a usar
    feature_cols = [
        'adx', 'di_diff', 'ema_align', 'ema20_dist', 'ema50_dist',
        'atr_pct', 'bb_width', 'vol_ratio', 'chop', 'rsi14',
        'roc_10', 'roc_20', 'ret_5', 'ret_20', 'vol_ma_ratio'
    ]

    # Filtrar features disponibles
    available_cols = [c for c in feature_cols if c in feat.columns]

    X = feat[available_cols]
    y = labels

    # Indices validos
    valid_mask = X.notna().all(axis=1) & y.notna()

    X_train = X[train_mask & valid_mask]
    y_train = y[train_mask & valid_mask]
    X_val = X[val_mask & valid_mask]
    y_val = y[val_mask & valid_mask]

    print(f"\nEntrenando clasificador...")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Validation: {len(X_val):,} samples")
    print(f"  Features: {len(available_cols)}")

    # Entrenar RandomForest (poco overfitting)
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,  # Limitado para evitar overfitting
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    # Evaluar en train
    train_pred = clf.predict(X_train)
    train_acc = (train_pred == y_train).mean()

    # Evaluar en validation
    val_pred = clf.predict(X_val)
    val_acc = (val_pred == y_val).mean()

    print(f"\n  Train Accuracy: {train_acc:.1%}")
    print(f"  Val Accuracy: {val_acc:.1%}")
    print(f"  Drop: {(train_acc - val_acc) / train_acc * 100:.1f}%")

    # Classification report
    print("\n=== VALIDATION CLASSIFICATION REPORT ===")
    print(classification_report(y_val, val_pred))

    # Feature importance
    importance = pd.DataFrame({
        'feature': available_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n=== FEATURE IMPORTANCE ===")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']:<20} {row['importance']:.4f}")

    return clf, available_cols, importance


def check_stability(labels, window=10):
    """
    Verificar que los regimenes sean estables (no cambien cada vela).
    """
    changes = (labels != labels.shift(1)).sum()
    total = len(labels)
    change_rate = changes / total * 100

    # Duracion promedio de regimen
    regime_lengths = []
    current_regime = labels.iloc[0]
    current_length = 1

    for i in range(1, len(labels)):
        if labels.iloc[i] == current_regime:
            current_length += 1
        else:
            regime_lengths.append(current_length)
            current_regime = labels.iloc[i]
            current_length = 1
    regime_lengths.append(current_length)

    avg_length = np.mean(regime_lengths)

    print(f"\n=== ESTABILIDAD DE REGIMENES ===")
    print(f"  Total cambios: {changes:,} de {total:,} velas")
    print(f"  Tasa de cambio: {change_rate:.1f}%")
    print(f"  Duracion promedio: {avg_length:.1f} velas ({avg_length * 4:.0f} horas)")

    return change_rate, avg_length


def main():
    print("=" * 70)
    print("BTC REGIME DETECTOR - Fase 1 V13.05")
    print("=" * 70)

    # Cargar datos
    print("\n[1/6] Cargando datos BTC...")
    df = load_data()
    print(f"  Total: {len(df):,} candles ({df.index.min().date()} a {df.index.max().date()})")

    # Calcular features
    print("\n[2/6] Calculando features de regimen...")
    feat = compute_regime_features(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)
    print(f"  Features calculados: {len(feat.columns)}")

    # Etiquetar regimenes
    print("\n[3/6] Etiquetando regimenes con reglas...")
    labels_raw = label_regimes_rules(df, feat)
    labels = smooth_labels(labels_raw, min_periods=3)

    # Distribucion de regimenes
    print("\n  Distribucion de regimenes:")
    for regime in labels.value_counts().index:
        count = (labels == regime).sum()
        pct = count / len(labels) * 100
        print(f"    {regime}: {count:,} ({pct:.1f}%)")

    # Verificar estabilidad
    print("\n[4/6] Verificando estabilidad...")
    check_stability(labels)

    # Split temporal
    train_mask = feat.index <= TRAIN_END
    val_mask = (feat.index > TRAIN_END) & (feat.index <= VALIDATION_END)
    test_mask = feat.index > VALIDATION_END

    # Analizar performance por regimen (solo en train)
    print("\n[5/6] Analizando regimenes (train)...")
    df_train = df[train_mask]
    labels_train = labels[train_mask]
    regime_stats = analyze_regime_performance(df_train, labels_train)

    # Entrenar clasificador
    print("\n[6/6] Entrenando clasificador...")
    clf, feature_cols, importance = train_regime_classifier(
        feat, labels, train_mask, val_mask
    )

    # Guardar modelo y metadata
    print("\n=== GUARDANDO MODELO ===")

    model_path = MODELS_DIR / 'btc_regime_detector.pkl'
    joblib.dump(clf, model_path)
    print(f"  Modelo: {model_path}")

    meta = {
        'version': 'v13.05',
        'pair': 'BTC',
        'feature_cols': feature_cols,
        'regimes': list(labels.unique()),
        'regime_stats': regime_stats.to_dict('records'),
        'train_samples': int(train_mask.sum()),
        'val_samples': int(val_mask.sum()),
        'train_end': TRAIN_END,
        'validation_end': VALIDATION_END,
    }

    meta_path = MODELS_DIR / 'btc_regime_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {meta_path}")

    # Guardar labels para usar en siguiente fase
    labels_df = pd.DataFrame({
        'timestamp': feat.index,
        'regime': labels,
        'regime_raw': labels_raw
    })
    labels_path = DATA_DIR / 'btc_regimes.parquet'
    labels_df.to_parquet(labels_path)
    print(f"  Labels: {labels_path}")

    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)

    print(f"""
Regime Detector entrenado para BTC.

Regimenes detectados:
  - BULL_TREND: Tendencia alcista (entrar en pullbacks)
  - BEAR_TREND: Tendencia bajista (NO TRADE o short)
  - RANGE: Mercado lateral (mean reversion)
  - VOLATILE: Alta volatilidad (reducir size o NO TRADE)

Siguiente paso: Fase 2 - Backtest por regimen para encontrar
la mejor estrategia en cada tipo de mercado.
""")


if __name__ == '__main__':
    main()
