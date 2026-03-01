"""
Train BTC V14 - Entrena y guarda los modelos exactos del backtest
"""
import warnings
import numpy as np
import pandas as pd
import pandas_ta as pta
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
from enum import Enum

warnings.filterwarnings('ignore')

# Copiado de strategy.py para evitar problemas de import

class Regime(Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    VOLATILE = "VOLATILE"

class Strategy(Enum):
    TREND_FOLLOW_LONG = "TREND_FOLLOW_LONG"
    TREND_FOLLOW_SHORT = "TREND_FOLLOW_SHORT"
    MEAN_REVERSION_LONG = "MEAN_REVERSION_LONG"
    MEAN_REVERSION_SHORT = "MEAN_REVERSION_SHORT"
    BREAKOUT_LONG = "BREAKOUT_LONG"
    BREAKOUT_SHORT = "BREAKOUT_SHORT"

STRATEGY_PARAMS = {
    Strategy.TREND_FOLLOW_LONG: {'tp': 0.04, 'sl': 0.015},
    Strategy.TREND_FOLLOW_SHORT: {'tp': 0.04, 'sl': 0.015},
    Strategy.MEAN_REVERSION_LONG: {'tp': 0.025, 'sl': 0.012},
    Strategy.MEAN_REVERSION_SHORT: {'tp': 0.025, 'sl': 0.012},
    Strategy.BREAKOUT_LONG: {'tp': 0.05, 'sl': 0.02},
    Strategy.BREAKOUT_SHORT: {'tp': 0.05, 'sl': 0.02},
}

CONTEXT_FEATURES = ['adx', 'di_diff', 'chop', 'atr_pct', 'bb_width']
MOMENTUM_FEATURES = ['rsi14', 'rsi7', 'stoch_k', 'ret_5', 'ret_20']
VOLUME_FEATURES = ['vol_ratio', 'vol_trend', 'obv_slope']

def detect_regime(feat_row):
    """Detecta régimen de mercado"""
    adx = feat_row.get('adx', 20)
    di_diff = feat_row.get('di_diff', 0)
    chop = feat_row.get('chop', 50)
    atr_pct = feat_row.get('atr_pct', 2)
    bb_width = feat_row.get('bb_width', 5)
    ema20_slope = feat_row.get('ema20_slope', 0)
    ema50_slope = feat_row.get('ema50_slope', 0)

    if pd.isna(adx) or pd.isna(chop):
        return Regime.RANGE

    if atr_pct > 4 and bb_width > 8:
        return Regime.VOLATILE

    if adx > 25 and chop < 50:
        if di_diff > 5 and ema20_slope > 0:
            return Regime.TREND_UP
        elif di_diff < -5 and ema20_slope < 0:
            return Regime.TREND_DOWN

    if chop > 55 or adx < 20:
        return Regime.RANGE

    if ema50_slope > 0.5:
        return Regime.TREND_UP
    elif ema50_slope < -0.5:
        return Regime.TREND_DOWN

    return Regime.RANGE

DATA_DIR = Path('data')
MODEL_DIR = Path('strategies/btc_v14/models')

# =============================================================================
# FUNCIONES ADAPTADAS
# =============================================================================

def load_data():
    """Carga desde CSV en lugar de parquet"""
    df = pd.read_csv(DATA_DIR / 'BTCUSDT_4h.csv', parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    return df.sort_index()


def compute_features(df):
    """Calcula features usando pandas_ta"""
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    feat = pd.DataFrame(index=df.index)

    # TREND
    adx_df = pta.adx(h, l, c, length=14)
    if adx_df is not None:
        feat['adx'] = adx_df.iloc[:, 0]
        feat['di_plus'] = adx_df.iloc[:, 1]
        feat['di_minus'] = adx_df.iloc[:, 2]
        feat['di_diff'] = feat['di_plus'] - feat['di_minus']

    chop = pta.chop(h, l, c, length=14)
    feat['chop'] = chop if chop is not None else 50

    feat['ema20'] = pta.ema(c, length=20)
    feat['ema50'] = pta.ema(c, length=50)
    feat['ema200'] = pta.ema(c, length=200)
    feat['ema20_dist'] = (c - feat['ema20']) / feat['ema20'] * 100
    feat['ema50_dist'] = (c - feat['ema50']) / feat['ema50'] * 100
    feat['ema200_dist'] = (c - feat['ema200']) / feat['ema200'] * 100
    feat['ema20_slope'] = feat['ema20'].pct_change(5) * 100
    feat['ema50_slope'] = feat['ema50'].pct_change(10) * 100

    # VOLATILITY
    feat['atr'] = pta.atr(h, l, c, length=14)
    feat['atr_pct'] = feat['atr'] / c * 100

    bb = pta.bbands(c, length=20)
    if bb is not None:
        feat['bb_upper'] = bb.iloc[:, 2]
        feat['bb_lower'] = bb.iloc[:, 0]
        feat['bb_mid'] = bb.iloc[:, 1]
        feat['bb_width'] = (feat['bb_upper'] - feat['bb_lower']) / feat['bb_mid'] * 100
        feat['bb_pct'] = (c - feat['bb_lower']) / (feat['bb_upper'] - feat['bb_lower'])

    # MOMENTUM
    feat['rsi14'] = pta.rsi(c, length=14)
    feat['rsi7'] = pta.rsi(c, length=7)
    feat['rsi_diff'] = feat['rsi14'] - feat['rsi7']

    stoch = pta.stoch(h, l, c, k=14, d=3)
    if stoch is not None:
        feat['stoch_k'] = stoch.iloc[:, 0]
        feat['stoch_d'] = stoch.iloc[:, 1]

    feat['ret_1'] = c.pct_change(1) * 100
    feat['ret_5'] = c.pct_change(5) * 100
    feat['ret_20'] = c.pct_change(20) * 100

    # VOLUME
    feat['vol_ratio'] = v / v.rolling(20).mean()
    feat['vol_trend'] = v.rolling(5).mean() / v.rolling(20).mean()
    obv = (np.sign(c.diff()) * v).cumsum()
    feat['obv_slope'] = obv.pct_change(10) * 100

    # STRUCTURE
    feat['high_20'] = h.rolling(20).max()
    feat['low_20'] = l.rolling(20).min()
    feat['range_pos'] = (c - feat['low_20']) / (feat['high_20'] - feat['low_20'])
    feat['consec_up'] = (c > c.shift(1)).rolling(10).sum()
    feat['consec_down'] = (c < c.shift(1)).rolling(10).sum()

    return feat


def detect_setups(df, feat):
    """Detecta setups por régimen"""
    setups = pd.DataFrame(index=df.index)
    setups['regime'] = None
    setups['strategy'] = None
    setups['setup_type'] = ''
    setups['has_setup'] = False

    for idx in feat.index:
        row = feat.loc[idx]
        regime = detect_regime(row)
        setups.loc[idx, 'regime'] = regime.value

        rsi14 = row.get('rsi14', 50)
        bb_pct = row.get('bb_pct', 0.5)
        range_pos = row.get('range_pos', 0.5)
        ema20_dist = row.get('ema20_dist', 0)
        ema200_dist = row.get('ema200_dist', 0)
        vol_ratio = row.get('vol_ratio', 1)
        consec_up = row.get('consec_up', 0)
        consec_down = row.get('consec_down', 0)

        if pd.isna(rsi14):
            continue

        # TREND_UP: Pullbacks
        if regime == Regime.TREND_UP:
            if rsi14 < 40 and bb_pct < 0.3 and ema200_dist > 0:
                setups.loc[idx, 'has_setup'] = True
                setups.loc[idx, 'strategy'] = Strategy.TREND_FOLLOW_LONG.value
                setups.loc[idx, 'setup_type'] = 'PULLBACK_IN_UPTREND'
            elif rsi14 < 30 and ema20_dist < -2:
                setups.loc[idx, 'has_setup'] = True
                setups.loc[idx, 'strategy'] = Strategy.TREND_FOLLOW_LONG.value
                setups.loc[idx, 'setup_type'] = 'OVERSOLD_IN_UPTREND'

        # TREND_DOWN: Rallies
        elif regime == Regime.TREND_DOWN:
            if rsi14 > 60 and bb_pct > 0.7 and ema200_dist < 0:
                setups.loc[idx, 'has_setup'] = True
                setups.loc[idx, 'strategy'] = Strategy.TREND_FOLLOW_SHORT.value
                setups.loc[idx, 'setup_type'] = 'RALLY_IN_DOWNTREND'
            elif rsi14 > 70 and ema20_dist > 2:
                setups.loc[idx, 'has_setup'] = True
                setups.loc[idx, 'strategy'] = Strategy.TREND_FOLLOW_SHORT.value
                setups.loc[idx, 'setup_type'] = 'OVERBOUGHT_IN_DOWNTREND'

        # RANGE: Mean reversion
        elif regime == Regime.RANGE:
            if range_pos < 0.2 and rsi14 < 35:
                setups.loc[idx, 'has_setup'] = True
                setups.loc[idx, 'strategy'] = Strategy.MEAN_REVERSION_LONG.value
                setups.loc[idx, 'setup_type'] = 'SUPPORT_BOUNCE'
            elif range_pos > 0.8 and rsi14 > 65:
                setups.loc[idx, 'has_setup'] = True
                setups.loc[idx, 'strategy'] = Strategy.MEAN_REVERSION_SHORT.value
                setups.loc[idx, 'setup_type'] = 'RESISTANCE_REJECTION'

        # VOLATILE: Breakouts
        elif regime == Regime.VOLATILE:
            if bb_pct > 1.0 and vol_ratio > 1.5 and consec_up >= 3:
                setups.loc[idx, 'has_setup'] = True
                setups.loc[idx, 'strategy'] = Strategy.BREAKOUT_LONG.value
                setups.loc[idx, 'setup_type'] = 'BREAKOUT_UP'
            elif bb_pct < 0 and vol_ratio > 1.5 and consec_down >= 3:
                setups.loc[idx, 'has_setup'] = True
                setups.loc[idx, 'strategy'] = Strategy.BREAKOUT_SHORT.value
                setups.loc[idx, 'setup_type'] = 'BREAKOUT_DOWN'

    return setups


def get_outcomes(df, setups, direction):
    """Obtiene resultados de trades para entrenar"""
    results = []
    setup_mask = setups['has_setup']

    if direction == 'long':
        valid = [Strategy.TREND_FOLLOW_LONG.value, Strategy.MEAN_REVERSION_LONG.value, Strategy.BREAKOUT_LONG.value]
    else:
        valid = [Strategy.TREND_FOLLOW_SHORT.value, Strategy.MEAN_REVERSION_SHORT.value, Strategy.BREAKOUT_SHORT.value]

    for idx in setups[setup_mask].index:
        if setups.loc[idx, 'strategy'] not in valid:
            continue

        strategy = setups.loc[idx, 'strategy']
        params = STRATEGY_PARAMS.get(Strategy(strategy), {'tp': 0.03, 'sl': 0.015})

        entry_price = df.loc[idx, 'close']
        future = df.loc[idx:].head(50)

        outcome = None
        for future_idx in future.index[1:]:
            future_price = df.loc[future_idx, 'close']

            if direction == 'long':
                pnl = (future_price - entry_price) / entry_price
            else:
                pnl = (entry_price - future_price) / entry_price

            if pnl >= params['tp']:
                outcome = 1
                break
            elif pnl <= -params['sl']:
                outcome = 0
                break

        if outcome is not None:
            results.append({'idx': idx, 'outcome': outcome})

    return pd.DataFrame(results).set_index('idx') if results else None


def train_and_save_models(feat, outcomes_long, outcomes_short):
    """Entrena y guarda los modelos del ensemble"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    models_saved = {'long': [], 'short': []}

    for direction, outcomes in [('long', outcomes_long), ('short', outcomes_short)]:
        if outcomes is None or len(outcomes) < 30:
            print(f"  {direction}: insuficientes datos")
            continue

        # Modelo CONTEXT
        ctx_cols = [c for c in CONTEXT_FEATURES if c in feat.columns]
        X = feat.loc[outcomes.index, ctx_cols].dropna()
        y = outcomes.loc[X.index, 'outcome']

        if len(y) >= 20:
            model = GradientBoostingClassifier(n_estimators=30, max_depth=2, random_state=42)
            model.fit(X, y)
            joblib.dump({'model': model, 'features': ctx_cols}, MODEL_DIR / f'context_{direction}.pkl')
            models_saved[direction].append('context')

        # Modelo MOMENTUM
        mom_cols = [c for c in MOMENTUM_FEATURES if c in feat.columns]
        X = feat.loc[outcomes.index, mom_cols].dropna()
        y = outcomes.loc[X.index, 'outcome']

        if len(y) >= 20:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
            model.fit(X_scaled, y)
            joblib.dump({'model': model, 'features': mom_cols, 'scaler': scaler}, MODEL_DIR / f'momentum_{direction}.pkl')
            models_saved[direction].append('momentum')

        # Modelo VOLUME
        vol_cols = [c for c in VOLUME_FEATURES if c in feat.columns]
        X = feat.loc[outcomes.index, vol_cols].dropna()
        y = outcomes.loc[X.index, 'outcome']

        if len(y) >= 20:
            model = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
            model.fit(X, y)
            joblib.dump({'model': model, 'features': vol_cols}, MODEL_DIR / f'volume_{direction}.pkl')
            models_saved[direction].append('volume')

        print(f"  {direction.upper()}: {models_saved[direction]}")

    # Metadata
    meta = {
        'models_long': models_saved['long'],
        'models_short': models_saved['short'],
        'context_features': CONTEXT_FEATURES,
        'momentum_features': MOMENTUM_FEATURES,
        'volume_features': VOLUME_FEATURES,
    }
    joblib.dump(meta, MODEL_DIR / 'metadata.pkl')

    return models_saved


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("BTC V14 - Entrenamiento de Modelos")
    print("=" * 70)

    print("\n[1/5] Cargando datos...")
    df = load_data()
    print(f"  {len(df):,} candles")

    print("\n[2/5] Calculando features...")
    feat = compute_features(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    print("\n[3/5] Detectando setups...")
    setups = detect_setups(df, feat)
    n_setups = setups['has_setup'].sum()
    print(f"  {n_setups} setups detectados")

    # Distribución de régimenes
    regime_counts = setups['regime'].value_counts()
    for regime, count in regime_counts.head(4).items():
        print(f"    {regime}: {count}")

    print("\n[4/5] Obteniendo outcomes...")
    outcomes_long = get_outcomes(df, setups, 'long')
    outcomes_short = get_outcomes(df, setups, 'short')
    print(f"  LONG: {len(outcomes_long) if outcomes_long is not None else 0}")
    print(f"  SHORT: {len(outcomes_short) if outcomes_short is not None else 0}")

    print("\n[5/5] Entrenando y guardando modelos...")
    models = train_and_save_models(feat, outcomes_long, outcomes_short)

    print("\n" + "=" * 70)
    print("MODELOS GUARDADOS EN:", MODEL_DIR)
    print("=" * 70)

    # Listar archivos
    for f in MODEL_DIR.glob('*.pkl'):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
