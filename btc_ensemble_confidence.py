"""
BTC Ensemble con Confianza - V13.06
===================================
Multiples modelos especializados que votan.
La confianza combinada determina tamaño de posicion.

Filosofia: No importa ganar siempre, importa no perder dinero.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')

# Trade management
TP_PCT = 0.03
SL_PCT = 0.015

# Position sizing basado en confianza
CONFIDENCE_THRESHOLDS = {
    'skip': 0.40,      # < 40% = no entrar
    'small': 0.55,     # 40-55% = 0.5x
    'normal': 0.70,    # 55-70% = 1.0x
    'large': 0.85      # 70-85% = 1.5x, >85% = 2.0x
}

POSITION_SIZES = {
    'skip': 0.0,
    'small': 0.5,
    'normal': 1.0,
    'large': 1.5,
    'max': 2.0
}


def load_data():
    df = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_full.parquet')
    return df.sort_index()


def compute_all_features(df):
    """Calcula TODAS las features, luego las separamos por modelo."""
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    feat = pd.DataFrame(index=df.index)

    # === CONTEXTO (tendencia, volatilidad general) ===
    feat['adx'] = ta.adx(h, l, c, length=14).iloc[:, 0]
    adx_df = ta.adx(h, l, c, length=14)
    feat['di_plus'] = adx_df.iloc[:, 1]
    feat['di_minus'] = adx_df.iloc[:, 2]
    feat['di_diff'] = feat['di_plus'] - feat['di_minus']

    chop = ta.chop(h, l, c, length=14)
    feat['chop'] = chop if chop is not None else 50

    feat['atr_pct'] = ta.atr(h, l, c, length=14) / c * 100

    bb = ta.bbands(c, length=20)
    if bb is not None:
        feat['bb_width'] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / bb.iloc[:, 1] * 100

    # === MOMENTUM (RSI, retornos, divergencias) ===
    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)
    feat['rsi_diff'] = feat['rsi14'] - feat['rsi7']

    feat['ret_1'] = c.pct_change(1) * 100
    feat['ret_5'] = c.pct_change(5) * 100
    feat['ret_20'] = c.pct_change(20) * 100

    # Momentum del momentum
    feat['rsi_change'] = feat['rsi14'].diff(3)

    # === VOLUMEN (liquidez, presion) ===
    feat['vol_ratio'] = v / v.rolling(20).mean()
    feat['vol_trend'] = v.rolling(5).mean() / v.rolling(20).mean()

    # OBV simplificado
    obv = (np.sign(c.diff()) * v).cumsum()
    feat['obv_pct'] = obv.pct_change(10) * 100

    # Volumen en velas alcistas vs bajistas
    up_vol = v.where(c > c.shift(1), 0).rolling(10).sum()
    down_vol = v.where(c < c.shift(1), 0).rolling(10).sum()
    feat['vol_pressure'] = (up_vol - down_vol) / (up_vol + down_vol + 1)

    # === ESTRUCTURA (posicion en rango, EMAs) ===
    ema20 = ta.ema(c, length=20)
    ema50 = ta.ema(c, length=50)
    ema200 = ta.ema(c, length=200)

    feat['ema20_dist'] = (c - ema20) / ema20 * 100
    feat['ema50_dist'] = (c - ema50) / ema50 * 100
    feat['ema200_dist'] = (c - ema200) / ema200 * 100

    if bb is not None:
        feat['bb_pct'] = (c - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])

    # Consecutivas bajistas
    feat['consec_down'] = (c < c.shift(1)).rolling(10).sum()

    # Hora
    feat['hour'] = df.index.hour
    feat['hour_sin'] = np.sin(2 * np.pi * feat['hour'] / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * feat['hour'] / 24)

    return feat


def detect_setups(feat):
    """Detecta setups tecnicos."""
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


# Features por modelo especializado
CONTEXT_FEATURES = ['adx', 'di_diff', 'chop', 'atr_pct', 'bb_width']
MOMENTUM_FEATURES = ['rsi14', 'rsi7', 'rsi_diff', 'ret_5', 'ret_20', 'rsi_change']
VOLUME_FEATURES = ['vol_ratio', 'vol_trend', 'obv_pct', 'vol_pressure']


def get_setup_outcomes(df, setups):
    """Obtiene resultado de cada setup."""
    results = []
    setup_indices = setups[setups['has_setup']].index

    for idx in setup_indices:
        if idx not in df.index:
            continue

        entry_price = df.loc[idx, 'close']
        future = df.loc[idx:].head(30)

        outcome = None
        for future_idx in future.index[1:]:
            future_price = df.loc[future_idx, 'close']
            pnl = (future_price - entry_price) / entry_price

            if pnl >= TP_PCT:
                outcome = 1
                break
            elif pnl <= -SL_PCT:
                outcome = 0
                break

        if outcome is not None:
            results.append({'idx': idx, 'outcome': outcome})

    return pd.DataFrame(results).set_index('idx') if results else None


def train_ensemble(feat, outcomes, train_mask):
    """Entrena 3 modelos especializados."""
    train_outcomes = outcomes[outcomes.index.isin(feat[train_mask].index)]

    if len(train_outcomes) < 50:
        return None

    models = {}

    # Modelo 1: Contexto (GradientBoosting - bueno para relaciones no lineales)
    ctx_cols = [c for c in CONTEXT_FEATURES if c in feat.columns]
    X_ctx = feat.loc[train_outcomes.index, ctx_cols].dropna()
    y_ctx = train_outcomes.loc[X_ctx.index, 'outcome']

    if len(y_ctx) >= 30:
        models['context'] = {
            'model': GradientBoostingClassifier(
                n_estimators=30, max_depth=2, learning_rate=0.1, random_state=42
            ),
            'features': ctx_cols
        }
        models['context']['model'].fit(X_ctx, y_ctx)

    # Modelo 2: Momentum (LogisticRegression - simple, menos overfitting)
    mom_cols = [c for c in MOMENTUM_FEATURES if c in feat.columns]
    X_mom = feat.loc[train_outcomes.index, mom_cols].dropna()
    y_mom = train_outcomes.loc[X_mom.index, 'outcome']

    if len(y_mom) >= 30:
        models['momentum'] = {
            'model': LogisticRegression(C=0.1, random_state=42, max_iter=1000),
            'features': mom_cols
        }
        models['momentum']['model'].fit(X_mom, y_mom)

    # Modelo 3: Volumen (RandomForest - robusto a outliers)
    vol_cols = [c for c in VOLUME_FEATURES if c in feat.columns]
    X_vol = feat.loc[train_outcomes.index, vol_cols].dropna()
    y_vol = train_outcomes.loc[X_vol.index, 'outcome']

    if len(y_vol) >= 30:
        models['volume'] = {
            'model': RandomForestClassifier(
                n_estimators=30, max_depth=3, random_state=42
            ),
            'features': vol_cols
        }
        models['volume']['model'].fit(X_vol, y_vol)

    return models if len(models) >= 2 else None


def get_ensemble_confidence(feat_row, models):
    """
    Obtiene confianza combinada de todos los modelos.
    Retorna: probabilidad promedio y votos individuales.
    """
    votes = {}
    probs = []

    for name, model_dict in models.items():
        model = model_dict['model']
        features = model_dict['features']

        X = feat_row[features].values.reshape(1, -1)

        if np.isnan(X).any():
            continue

        prob = model.predict_proba(X)[0, 1]
        votes[name] = prob
        probs.append(prob)

    if not probs:
        return 0.0, {}

    # Confianza = promedio de probabilidades
    confidence = np.mean(probs)

    return confidence, votes


def get_position_size(confidence):
    """Determina tamaño de posicion basado en confianza."""
    if confidence < CONFIDENCE_THRESHOLDS['skip']:
        return 0.0, 'skip'
    elif confidence < CONFIDENCE_THRESHOLDS['small']:
        return POSITION_SIZES['small'], 'small'
    elif confidence < CONFIDENCE_THRESHOLDS['normal']:
        return POSITION_SIZES['normal'], 'normal'
    elif confidence < CONFIDENCE_THRESHOLDS['large']:
        return POSITION_SIZES['large'], 'large'
    else:
        return POSITION_SIZES['max'], 'max'


def backtest_ensemble(df, feat, setups, models, test_mask):
    """Backtest con position sizing basado en confianza."""
    trades = []
    position = None

    test_setups = setups[test_mask & setups['has_setup']]

    for idx in df[test_mask].index:
        price = df.loc[idx, 'close']

        # Check posicion abierta
        if position is not None:
            pnl_pct = (price - position['entry']) / position['entry']

            if pnl_pct >= TP_PCT:
                # PnL ajustado por tamaño
                adjusted_pnl = pnl_pct * position['size']
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': idx,
                    'pnl_raw': pnl_pct,
                    'pnl_adjusted': adjusted_pnl,
                    'size': position['size'],
                    'size_label': position['size_label'],
                    'confidence': position['confidence'],
                    'result': 'WIN'
                })
                position = None

            elif pnl_pct <= -SL_PCT:
                adjusted_pnl = pnl_pct * position['size']
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': idx,
                    'pnl_raw': pnl_pct,
                    'pnl_adjusted': adjusted_pnl,
                    'size': position['size'],
                    'size_label': position['size_label'],
                    'confidence': position['confidence'],
                    'result': 'LOSS'
                })
                position = None

        # Nueva entrada
        if position is None and idx in test_setups.index:
            feat_row = feat.loc[idx]
            confidence, votes = get_ensemble_confidence(feat_row, models)
            size, size_label = get_position_size(confidence)

            if size > 0:
                position = {
                    'entry': price,
                    'entry_time': idx,
                    'size': size,
                    'size_label': size_label,
                    'confidence': confidence,
                    'votes': votes
                }

    return pd.DataFrame(trades) if trades else None


def backtest_fixed_size(df, feat, setups, models, test_mask, threshold=0.45):
    """Backtest con tamaño fijo (como antes) para comparar."""
    trades = []
    position = None

    test_setups = setups[test_mask & setups['has_setup']]

    for idx in df[test_mask].index:
        price = df.loc[idx, 'close']

        if position is not None:
            pnl_pct = (price - position['entry']) / position['entry']
            if pnl_pct >= TP_PCT:
                trades.append({'pnl': pnl_pct, 'result': 'WIN'})
                position = None
            elif pnl_pct <= -SL_PCT:
                trades.append({'pnl': pnl_pct, 'result': 'LOSS'})
                position = None

        if position is None and idx in test_setups.index:
            feat_row = feat.loc[idx]
            confidence, _ = get_ensemble_confidence(feat_row, models)

            if confidence >= threshold:
                position = {'entry': price}

    return pd.DataFrame(trades) if trades else None


def main():
    print("=" * 70)
    print("BTC ENSEMBLE CON CONFIANZA - V13.06")
    print("=" * 70)

    # Cargar datos
    print("\n[1/5] Cargando datos...")
    df = load_data()
    print(f"  {len(df):,} candles ({df.index.min().date()} a {df.index.max().date()})")

    # Features
    print("\n[2/5] Calculando features...")
    feat = compute_all_features(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # Setups
    print("\n[3/5] Detectando setups...")
    setups = detect_setups(feat)
    print(f"  {setups['has_setup'].sum()} setups detectados")

    # Outcomes
    outcomes = get_setup_outcomes(df, setups)
    print(f"  {len(outcomes)} con resultado definido")

    # Split
    train_end = '2024-06-30'
    test_start = '2024-07-01'
    train_mask = df.index <= train_end
    test_mask = df.index >= test_start

    print(f"\n  Train: hasta {train_end}")
    print(f"  Test: desde {test_start}")

    # Entrenar ensemble
    print("\n[4/5] Entrenando ensemble...")
    models = train_ensemble(feat, outcomes, train_mask)

    if models is None:
        print("  [ERROR] No se pudo entrenar")
        return

    print(f"  Modelos entrenados: {list(models.keys())}")

    # Backtest
    print("\n[5/5] Backtesting...")

    # Con position sizing
    trades_sized = backtest_ensemble(df, feat, setups, models, test_mask)

    # Con tamaño fijo (comparacion)
    trades_fixed = backtest_fixed_size(df, feat, setups, models, test_mask)

    # Resultados
    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)

    if trades_fixed is not None and len(trades_fixed) > 0:
        n = len(trades_fixed)
        wins = (trades_fixed['result'] == 'WIN').sum()
        pnl = trades_fixed['pnl'].sum() * 100

        print(f"\n[TAMAÑO FIJO] (threshold 0.45, siempre 1x)")
        print(f"  Trades: {n}")
        print(f"  Win Rate: {wins/n*100:.1f}%")
        print(f"  PnL: {pnl:+.1f}%")

    if trades_sized is not None and len(trades_sized) > 0:
        n = len(trades_sized)
        wins = (trades_sized['result'] == 'WIN').sum()
        pnl_raw = trades_sized['pnl_raw'].sum() * 100
        pnl_adjusted = trades_sized['pnl_adjusted'].sum() * 100

        print(f"\n[POSITION SIZING] (basado en confianza)")
        print(f"  Trades: {n}")
        print(f"  Win Rate: {wins/n*100:.1f}%")
        print(f"  PnL (sin ajuste): {pnl_raw:+.1f}%")
        print(f"  PnL (con sizing): {pnl_adjusted:+.1f}%")

        # Desglose por tamaño
        print(f"\n  Desglose por confianza:")
        for label in ['small', 'normal', 'large', 'max']:
            subset = trades_sized[trades_sized['size_label'] == label]
            if len(subset) > 0:
                sub_wins = (subset['result'] == 'WIN').sum()
                sub_pnl = subset['pnl_adjusted'].sum() * 100
                avg_conf = subset['confidence'].mean()
                print(f"    {label:>8}: {len(subset):>3} trades, WR {sub_wins/len(subset)*100:>5.1f}%, PnL {sub_pnl:>+6.1f}%, Conf avg {avg_conf:.2f}")

        # Estadisticas de confianza
        print(f"\n  Estadisticas de confianza:")
        print(f"    Promedio: {trades_sized['confidence'].mean():.2f}")
        print(f"    Min: {trades_sized['confidence'].min():.2f}")
        print(f"    Max: {trades_sized['confidence'].max():.2f}")

        # Trades skipped
        skipped = setups[test_mask & setups['has_setup']].shape[0] - n
        print(f"\n  Setups skipped (confianza < 40%): {skipped}")

    # Comparacion final
    print("\n" + "=" * 70)
    print("COMPARACION")
    print("=" * 70)

    if trades_fixed is not None and trades_sized is not None:
        pnl_fixed = trades_fixed['pnl'].sum() * 100
        pnl_sized = trades_sized['pnl_adjusted'].sum() * 100

        print(f"\n  PnL Tamaño Fijo:    {pnl_fixed:+.1f}%")
        print(f"  PnL Position Sizing: {pnl_sized:+.1f}%")
        print(f"  Diferencia:          {pnl_sized - pnl_fixed:+.1f}%")

        # Analisis de riesgo
        losses_fixed = trades_fixed[trades_fixed['result'] == 'LOSS']['pnl'].sum() * 100
        losses_sized = trades_sized[trades_sized['result'] == 'LOSS']['pnl_adjusted'].sum() * 100

        print(f"\n  Perdidas Tamaño Fijo:    {losses_fixed:.1f}%")
        print(f"  Perdidas Position Sizing: {losses_sized:.1f}%")
        print(f"  Reduccion de perdidas:    {losses_fixed - losses_sized:.1f}%")

    # Veredicto
    print("\n" + "=" * 70)
    if trades_sized is not None:
        pnl_sized = trades_sized['pnl_adjusted'].sum() * 100
        if pnl_sized > 0:
            print("[OK] Sistema rentable con position sizing")
        else:
            print("[WARN] Sistema no rentable, requiere ajustes")

    print("=" * 70)


if __name__ == '__main__':
    main()
