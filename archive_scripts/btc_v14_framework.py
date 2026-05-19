"""
BTC V14 Framework - Multi-Estrategia con Confianza
===================================================
Sistema completo:
1. Detector de regimen (TREND_UP/TREND_DOWN/RANGE/VOLATILE)
2. Estrategia por regimen
3. Ensemble ML (3 modelos)
4. Position sizing Half-Kelly
5. Anti-Martingala

Objetivo: +6% mensual = +100% anual (compound)
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')


# =============================================================================
# CONFIGURACION
# =============================================================================

class Regime(Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    VOLATILE = "VOLATILE"


class Strategy(Enum):
    TREND_FOLLOW_LONG = "TREND_FOLLOW_LONG"      # Comprar pullbacks en tendencia alcista
    TREND_FOLLOW_SHORT = "TREND_FOLLOW_SHORT"   # Vender rallies en tendencia bajista
    MEAN_REVERSION_LONG = "MEAN_REVERSION_LONG"  # Comprar en soporte
    MEAN_REVERSION_SHORT = "MEAN_REVERSION_SHORT"  # Vender en resistencia
    BREAKOUT_LONG = "BREAKOUT_LONG"              # Comprar ruptura alcista
    BREAKOUT_SHORT = "BREAKOUT_SHORT"            # Vender ruptura bajista
    NO_TRADE = "NO_TRADE"


# Risk management
BASE_RISK_PCT = 0.02  # 2% base risk per trade
MAX_RISK_PCT = 0.04   # 4% max risk
MIN_RISK_PCT = 0.005  # 0.5% min risk
MAX_PORTFOLIO_HEAT = 0.06  # 6% max total exposure

# Trade parameters por estrategia
STRATEGY_PARAMS = {
    Strategy.TREND_FOLLOW_LONG: {'tp': 0.04, 'sl': 0.015, 'rr': 2.67},
    Strategy.TREND_FOLLOW_SHORT: {'tp': 0.04, 'sl': 0.015, 'rr': 2.67},
    Strategy.MEAN_REVERSION_LONG: {'tp': 0.025, 'sl': 0.012, 'rr': 2.08},
    Strategy.MEAN_REVERSION_SHORT: {'tp': 0.025, 'sl': 0.012, 'rr': 2.08},
    Strategy.BREAKOUT_LONG: {'tp': 0.05, 'sl': 0.02, 'rr': 2.5},
    Strategy.BREAKOUT_SHORT: {'tp': 0.05, 'sl': 0.02, 'rr': 2.5},
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    df = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_full.parquet')
    return df.sort_index()


# =============================================================================
# FEATURES
# =============================================================================

def compute_features(df):
    """Calcula todas las features necesarias."""
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    feat = pd.DataFrame(index=df.index)

    # === TREND INDICATORS ===
    adx_df = ta.adx(h, l, c, length=14)
    if adx_df is not None:
        feat['adx'] = adx_df.iloc[:, 0]
        feat['di_plus'] = adx_df.iloc[:, 1]
        feat['di_minus'] = adx_df.iloc[:, 2]
        feat['di_diff'] = feat['di_plus'] - feat['di_minus']

    # Choppiness Index (alto = rango, bajo = tendencia)
    chop = ta.chop(h, l, c, length=14)
    feat['chop'] = chop if chop is not None else 50

    # EMAs para direccion
    feat['ema20'] = ta.ema(c, length=20)
    feat['ema50'] = ta.ema(c, length=50)
    feat['ema200'] = ta.ema(c, length=200)
    feat['ema20_dist'] = (c - feat['ema20']) / feat['ema20'] * 100
    feat['ema50_dist'] = (c - feat['ema50']) / feat['ema50'] * 100
    feat['ema200_dist'] = (c - feat['ema200']) / feat['ema200'] * 100

    # Pendiente de EMA (direccion)
    feat['ema20_slope'] = feat['ema20'].pct_change(5) * 100
    feat['ema50_slope'] = feat['ema50'].pct_change(10) * 100

    # === VOLATILITY ===
    feat['atr'] = ta.atr(h, l, c, length=14)
    feat['atr_pct'] = feat['atr'] / c * 100

    bb = ta.bbands(c, length=20)
    if bb is not None:
        feat['bb_upper'] = bb.iloc[:, 2]
        feat['bb_lower'] = bb.iloc[:, 0]
        feat['bb_mid'] = bb.iloc[:, 1]
        feat['bb_width'] = (feat['bb_upper'] - feat['bb_lower']) / feat['bb_mid'] * 100
        feat['bb_pct'] = (c - feat['bb_lower']) / (feat['bb_upper'] - feat['bb_lower'])

    # === MOMENTUM ===
    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)
    feat['rsi_diff'] = feat['rsi14'] - feat['rsi7']

    stoch = ta.stoch(h, l, c, k=14, d=3)
    if stoch is not None:
        feat['stoch_k'] = stoch.iloc[:, 0]
        feat['stoch_d'] = stoch.iloc[:, 1]

    feat['ret_1'] = c.pct_change(1) * 100
    feat['ret_5'] = c.pct_change(5) * 100
    feat['ret_20'] = c.pct_change(20) * 100

    # === VOLUME ===
    feat['vol_ratio'] = v / v.rolling(20).mean()
    feat['vol_trend'] = v.rolling(5).mean() / v.rolling(20).mean()

    # OBV trend
    obv = (np.sign(c.diff()) * v).cumsum()
    feat['obv_slope'] = obv.pct_change(10) * 100

    # === STRUCTURE ===
    # Highs and lows recientes
    feat['high_20'] = h.rolling(20).max()
    feat['low_20'] = l.rolling(20).min()
    feat['range_pos'] = (c - feat['low_20']) / (feat['high_20'] - feat['low_20'])

    # Consecutive candles
    feat['consec_up'] = (c > c.shift(1)).rolling(10).sum()
    feat['consec_down'] = (c < c.shift(1)).rolling(10).sum()

    # === TIME ===
    feat['hour'] = df.index.hour
    feat['day_of_week'] = df.index.dayofweek

    return feat


# =============================================================================
# REGIME DETECTOR
# =============================================================================

def detect_regime(feat_row):
    """
    Detecta el regimen actual del mercado.

    TREND_UP: ADX > 25, DI+ > DI-, CHOP < 50, EMA slope positivo
    TREND_DOWN: ADX > 25, DI- > DI+, CHOP < 50, EMA slope negativo
    RANGE: CHOP > 60, ADX < 20
    VOLATILE: ATR_pct alto, BB_width alto
    """
    adx = feat_row.get('adx', 20)
    di_diff = feat_row.get('di_diff', 0)
    chop = feat_row.get('chop', 50)
    atr_pct = feat_row.get('atr_pct', 2)
    bb_width = feat_row.get('bb_width', 5)
    ema20_slope = feat_row.get('ema20_slope', 0)
    ema50_slope = feat_row.get('ema50_slope', 0)

    # Check for NaN
    if pd.isna(adx) or pd.isna(chop):
        return Regime.RANGE

    # High volatility (breakout conditions)
    if atr_pct > 4 and bb_width > 8:
        return Regime.VOLATILE

    # Strong trend
    if adx > 25 and chop < 50:
        if di_diff > 5 and ema20_slope > 0:
            return Regime.TREND_UP
        elif di_diff < -5 and ema20_slope < 0:
            return Regime.TREND_DOWN

    # Range/Choppy
    if chop > 55 or adx < 20:
        return Regime.RANGE

    # Default based on trend direction
    if ema50_slope > 0.5:
        return Regime.TREND_UP
    elif ema50_slope < -0.5:
        return Regime.TREND_DOWN

    return Regime.RANGE


# =============================================================================
# SETUP DETECTION PER STRATEGY
# =============================================================================

def detect_setups(df, feat):
    """
    Detecta setups para cada estrategia basado en el regimen.
    Retorna DataFrame con setup_type y estrategia sugerida.
    """
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
        rsi7 = row.get('rsi7', 50)
        bb_pct = row.get('bb_pct', 0.5)
        stoch_k = row.get('stoch_k', 50)
        consec_down = row.get('consec_down', 0)
        consec_up = row.get('consec_up', 0)
        vol_ratio = row.get('vol_ratio', 1)
        range_pos = row.get('range_pos', 0.5)
        ema20_dist = row.get('ema20_dist', 0)
        ema200_dist = row.get('ema200_dist', 0)
        adx = row.get('adx', 20)

        if pd.isna(rsi14):
            continue

        # === TREND UP: Buscar pullbacks para comprar ===
        if regime == Regime.TREND_UP:
            # Pullback en tendencia alcista
            if rsi14 < 40 and bb_pct < 0.3 and ema200_dist > 0:
                setups.loc[idx, 'has_setup'] = True
                setups.loc[idx, 'strategy'] = Strategy.TREND_FOLLOW_LONG.value
                setups.loc[idx, 'setup_type'] = 'PULLBACK_IN_UPTREND'
            # RSI oversold en tendencia alcista
            elif rsi14 < 30 and ema20_dist < -2:
                setups.loc[idx, 'has_setup'] = True
                setups.loc[idx, 'strategy'] = Strategy.TREND_FOLLOW_LONG.value
                setups.loc[idx, 'setup_type'] = 'OVERSOLD_IN_UPTREND'

        # === TREND DOWN: Buscar rallies para vender ===
        elif regime == Regime.TREND_DOWN:
            # Rally en tendencia bajista (short)
            if rsi14 > 60 and bb_pct > 0.7 and ema200_dist < 0:
                setups.loc[idx, 'has_setup'] = True
                setups.loc[idx, 'strategy'] = Strategy.TREND_FOLLOW_SHORT.value
                setups.loc[idx, 'setup_type'] = 'RALLY_IN_DOWNTREND'
            # RSI overbought en tendencia bajista
            elif rsi14 > 70 and ema20_dist > 2:
                setups.loc[idx, 'has_setup'] = True
                setups.loc[idx, 'strategy'] = Strategy.TREND_FOLLOW_SHORT.value
                setups.loc[idx, 'setup_type'] = 'OVERBOUGHT_IN_DOWNTREND'

        # === RANGE: Mean reversion en soportes y resistencias ===
        elif regime == Regime.RANGE:
            # Cerca de soporte (comprar)
            if range_pos < 0.2 and rsi14 < 35:
                setups.loc[idx, 'has_setup'] = True
                setups.loc[idx, 'strategy'] = Strategy.MEAN_REVERSION_LONG.value
                setups.loc[idx, 'setup_type'] = 'SUPPORT_BOUNCE'
            # Cerca de resistencia (vender)
            elif range_pos > 0.8 and rsi14 > 65:
                setups.loc[idx, 'has_setup'] = True
                setups.loc[idx, 'strategy'] = Strategy.MEAN_REVERSION_SHORT.value
                setups.loc[idx, 'setup_type'] = 'RESISTANCE_REJECTION'

        # === VOLATILE: Breakouts ===
        elif regime == Regime.VOLATILE:
            # Breakout alcista
            if bb_pct > 1.0 and vol_ratio > 1.5 and consec_up >= 3:
                setups.loc[idx, 'has_setup'] = True
                setups.loc[idx, 'strategy'] = Strategy.BREAKOUT_LONG.value
                setups.loc[idx, 'setup_type'] = 'BREAKOUT_UP'
            # Breakout bajista
            elif bb_pct < 0 and vol_ratio > 1.5 and consec_down >= 3:
                setups.loc[idx, 'has_setup'] = True
                setups.loc[idx, 'strategy'] = Strategy.BREAKOUT_SHORT.value
                setups.loc[idx, 'setup_type'] = 'BREAKOUT_DOWN'

    return setups


# =============================================================================
# ENSEMBLE ML
# =============================================================================

CONTEXT_FEATURES = ['adx', 'di_diff', 'chop', 'atr_pct', 'bb_width']
MOMENTUM_FEATURES = ['rsi14', 'rsi7', 'stoch_k', 'ret_5', 'ret_20']
VOLUME_FEATURES = ['vol_ratio', 'vol_trend', 'obv_slope']


def get_setup_outcomes(df, setups, direction='long'):
    """Obtiene resultados de setups para entrenar."""
    results = []
    setup_mask = setups['has_setup']

    if direction == 'long':
        valid_strategies = [Strategy.TREND_FOLLOW_LONG.value,
                           Strategy.MEAN_REVERSION_LONG.value,
                           Strategy.BREAKOUT_LONG.value]
    else:
        valid_strategies = [Strategy.TREND_FOLLOW_SHORT.value,
                           Strategy.MEAN_REVERSION_SHORT.value,
                           Strategy.BREAKOUT_SHORT.value]

    for idx in setups[setup_mask].index:
        if setups.loc[idx, 'strategy'] not in valid_strategies:
            continue
        if idx not in df.index:
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
            results.append({
                'idx': idx,
                'outcome': outcome,
                'strategy': strategy,
                'direction': direction
            })

    return pd.DataFrame(results).set_index('idx') if results else None


def train_ensemble(feat, outcomes_long, outcomes_short, train_mask):
    """Entrena ensemble de 3 modelos para LONG y SHORT."""
    models = {'long': {}, 'short': {}}

    for direction, outcomes in [('long', outcomes_long), ('short', outcomes_short)]:
        if outcomes is None or len(outcomes) < 30:
            continue

        train_outcomes = outcomes[outcomes.index.isin(feat[train_mask].index)]
        if len(train_outcomes) < 30:
            continue

        # Modelo Contexto
        ctx_cols = [c for c in CONTEXT_FEATURES if c in feat.columns]
        X = feat.loc[train_outcomes.index, ctx_cols].dropna()
        y = train_outcomes.loc[X.index, 'outcome']

        if len(y) >= 20:
            models[direction]['context'] = {
                'model': GradientBoostingClassifier(n_estimators=30, max_depth=2, random_state=42),
                'features': ctx_cols
            }
            models[direction]['context']['model'].fit(X, y)

        # Modelo Momentum
        mom_cols = [c for c in MOMENTUM_FEATURES if c in feat.columns]
        X = feat.loc[train_outcomes.index, mom_cols].dropna()
        y = train_outcomes.loc[X.index, 'outcome']

        if len(y) >= 20:
            models[direction]['momentum'] = {
                'model': LogisticRegression(C=0.1, max_iter=1000, random_state=42),
                'features': mom_cols
            }
            # Normalize for logistic regression
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            models[direction]['momentum']['model'].fit(X_scaled, y)
            models[direction]['momentum']['scaler'] = scaler

        # Modelo Volume
        vol_cols = [c for c in VOLUME_FEATURES if c in feat.columns]
        X = feat.loc[train_outcomes.index, vol_cols].dropna()
        y = train_outcomes.loc[X.index, 'outcome']

        if len(y) >= 20:
            models[direction]['volume'] = {
                'model': RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42),
                'features': vol_cols
            }
            models[direction]['volume']['model'].fit(X, y)

    return models


def get_ensemble_confidence(feat_row, models, direction='long'):
    """Obtiene confianza combinada del ensemble."""
    if direction not in models or not models[direction]:
        return 0.5, {}

    probs = []
    votes = {}

    for name, model_dict in models[direction].items():
        model = model_dict['model']
        features = model_dict['features']

        X = feat_row[features].values.reshape(1, -1)
        if np.isnan(X).any():
            continue

        # Apply scaler if exists (for logistic regression)
        if 'scaler' in model_dict:
            X = model_dict['scaler'].transform(X)

        prob = model.predict_proba(X)[0, 1]
        probs.append(prob)
        votes[name] = prob

    if not probs:
        return 0.5, {}

    confidence = np.mean(probs)
    return confidence, votes


# =============================================================================
# POSITION SIZING (Half-Kelly + Anti-Martingale)
# =============================================================================

def calculate_kelly_fraction(win_rate, avg_win, avg_loss):
    """Calcula Kelly fraction."""
    if avg_loss == 0:
        return 0

    rr_ratio = avg_win / avg_loss
    kelly = (win_rate * rr_ratio - (1 - win_rate)) / rr_ratio

    # Half Kelly for safety
    return max(0, kelly * 0.5)


def get_position_size(confidence, streak, base_risk=BASE_RISK_PCT):
    """
    Calcula tamaño de posicion basado en confianza y racha.

    confidence: 0-1 del ensemble
    streak: positivo = rachas ganadoras, negativo = perdedoras
    """
    # Base risk adjusted by confidence (umbrales mas bajos para mas trades)
    if confidence < 0.35:
        return 0, 'skip'
    elif confidence < 0.45:
        risk_mult = 0.25
        label = 'very_low'
    elif confidence < 0.55:
        risk_mult = 0.5
        label = 'low'
    elif confidence < 0.65:
        risk_mult = 1.0
        label = 'medium'
    elif confidence < 0.75:
        risk_mult = 1.5
        label = 'high'
    else:
        risk_mult = 2.0
        label = 'very_high'

    # Anti-Martingale adjustment
    if streak <= -3:
        # Losing streak: reduce by 50%
        risk_mult *= 0.5
        label += '_reduced'
    elif streak >= 3:
        # Winning streak: increase by 25% (capped)
        risk_mult *= 1.25
        label += '_boosted'

    final_risk = base_risk * risk_mult
    final_risk = max(MIN_RISK_PCT, min(MAX_RISK_PCT, final_risk))

    return final_risk, label


# =============================================================================
# BACKTESTING
# =============================================================================

def backtest_v14(df, feat, setups, models, test_mask):
    """Backtest completo del sistema V14."""
    trades = []
    position = None
    streak = 0  # Positive = winning, negative = losing

    for idx in df[test_mask].index:
        price = df.loc[idx, 'close']

        # === Check open position ===
        if position is not None:
            if position['direction'] == 'long':
                pnl_pct = (price - position['entry']) / position['entry']
            else:
                pnl_pct = (position['entry'] - price) / position['entry']

            tp = position['tp']
            sl = position['sl']

            if pnl_pct >= tp:
                # Win
                adjusted_pnl = pnl_pct * position['size_mult']
                streak = max(1, streak + 1)

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': idx,
                    'direction': position['direction'],
                    'strategy': position['strategy'],
                    'regime': position['regime'],
                    'entry_price': position['entry'],
                    'exit_price': price,
                    'pnl_raw': pnl_pct,
                    'pnl_adjusted': adjusted_pnl,
                    'risk_pct': position['risk_pct'],
                    'size_label': position['size_label'],
                    'confidence': position['confidence'],
                    'result': 'WIN'
                })
                position = None

            elif pnl_pct <= -sl:
                # Loss
                adjusted_pnl = pnl_pct * position['size_mult']
                streak = min(-1, streak - 1)

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': idx,
                    'direction': position['direction'],
                    'strategy': position['strategy'],
                    'regime': position['regime'],
                    'entry_price': position['entry'],
                    'exit_price': price,
                    'pnl_raw': pnl_pct,
                    'pnl_adjusted': adjusted_pnl,
                    'risk_pct': position['risk_pct'],
                    'size_label': position['size_label'],
                    'confidence': position['confidence'],
                    'result': 'LOSS'
                })
                position = None

        # === Check for new setup ===
        if position is None and idx in setups.index and setups.loc[idx, 'has_setup']:
            strategy_str = setups.loc[idx, 'strategy']
            regime = setups.loc[idx, 'regime']

            if strategy_str is None:
                continue

            # Determine direction
            if 'LONG' in strategy_str:
                direction = 'long'
            elif 'SHORT' in strategy_str:
                direction = 'short'
            else:
                continue

            # Get confidence
            feat_row = feat.loc[idx]
            confidence, votes = get_ensemble_confidence(feat_row, models, direction)

            # Get position size
            risk_pct, size_label = get_position_size(confidence, streak)

            if risk_pct == 0:
                continue

            # Get TP/SL for this strategy
            strategy_enum = Strategy(strategy_str)
            params = STRATEGY_PARAMS.get(strategy_enum, {'tp': 0.03, 'sl': 0.015})

            # Size multiplier (for PnL calculation)
            size_mult = risk_pct / BASE_RISK_PCT

            position = {
                'entry': price,
                'entry_time': idx,
                'direction': direction,
                'strategy': strategy_str,
                'regime': regime,
                'tp': params['tp'],
                'sl': params['sl'],
                'confidence': confidence,
                'risk_pct': risk_pct,
                'size_mult': size_mult,
                'size_label': size_label
            }

    return pd.DataFrame(trades) if trades else None


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("BTC V14 FRAMEWORK - Multi-Estrategia con Confianza")
    print("=" * 70)

    # Load data
    print("\n[1/6] Cargando datos...")
    df = load_data()
    print(f"  {len(df):,} candles ({df.index.min().date()} a {df.index.max().date()})")

    # Features
    print("\n[2/6] Calculando features...")
    feat = compute_features(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # Detect setups
    print("\n[3/6] Detectando setups por regimen...")
    setups = detect_setups(df, feat)

    n_setups = setups['has_setup'].sum()
    print(f"  Total setups: {n_setups}")

    # Regime distribution
    print(f"\n  Distribucion de regimenes:")
    regime_counts = setups['regime'].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(setups) * 100
        print(f"    {regime}: {count} ({pct:.1f}%)")

    # Strategy distribution
    print(f"\n  Distribucion de estrategias (en setups):")
    strategy_counts = setups[setups['has_setup']]['strategy'].value_counts()
    for strategy, count in strategy_counts.items():
        print(f"    {strategy}: {count}")

    # Split
    train_end = '2024-06-30'
    test_start = '2024-07-01'
    train_mask = df.index <= train_end
    test_mask = df.index >= test_start

    print(f"\n[4/6] Preparando datos...")
    print(f"  Train: hasta {train_end}")
    print(f"  Test: desde {test_start}")

    # Get outcomes
    outcomes_long = get_setup_outcomes(df, setups, 'long')
    outcomes_short = get_setup_outcomes(df, setups, 'short')

    print(f"  Outcomes LONG: {len(outcomes_long) if outcomes_long is not None else 0}")
    print(f"  Outcomes SHORT: {len(outcomes_short) if outcomes_short is not None else 0}")

    # Train ensemble
    print("\n[5/6] Entrenando ensemble...")
    models = train_ensemble(feat, outcomes_long, outcomes_short, train_mask)

    for direction in ['long', 'short']:
        if direction in models and models[direction]:
            print(f"  {direction.upper()}: {list(models[direction].keys())}")

    # Backtest
    print("\n[6/6] Backtesting...")
    trades = backtest_v14(df, feat, setups, models, test_mask)

    # Results
    print("\n" + "=" * 70)
    print("RESULTADOS V14")
    print("=" * 70)

    if trades is not None and len(trades) > 0:
        n = len(trades)
        wins = (trades['result'] == 'WIN').sum()

        pnl_raw = trades['pnl_raw'].sum() * 100
        pnl_adjusted = trades['pnl_adjusted'].sum() * 100

        print(f"\n[GENERAL]")
        print(f"  Trades: {n}")
        print(f"  Win Rate: {wins/n*100:.1f}%")
        print(f"  PnL (raw): {pnl_raw:+.1f}%")
        print(f"  PnL (sized): {pnl_adjusted:+.1f}%")

        # Por direccion
        print(f"\n[POR DIRECCION]")
        for direction in ['long', 'short']:
            subset = trades[trades['direction'] == direction]
            if len(subset) > 0:
                sub_wins = (subset['result'] == 'WIN').sum()
                sub_pnl = subset['pnl_adjusted'].sum() * 100
                print(f"  {direction.upper()}: {len(subset)} trades, WR {sub_wins/len(subset)*100:.1f}%, PnL {sub_pnl:+.1f}%")

        # Por estrategia
        print(f"\n[POR ESTRATEGIA]")
        for strategy in trades['strategy'].unique():
            subset = trades[trades['strategy'] == strategy]
            if len(subset) > 0:
                sub_wins = (subset['result'] == 'WIN').sum()
                sub_pnl = subset['pnl_adjusted'].sum() * 100
                print(f"  {strategy}: {len(subset)} trades, WR {sub_wins/len(subset)*100:.1f}%, PnL {sub_pnl:+.1f}%")

        # Por regimen
        print(f"\n[POR REGIMEN]")
        for regime in trades['regime'].unique():
            subset = trades[trades['regime'] == regime]
            if len(subset) > 0:
                sub_wins = (subset['result'] == 'WIN').sum()
                sub_pnl = subset['pnl_adjusted'].sum() * 100
                print(f"  {regime}: {len(subset)} trades, WR {sub_wins/len(subset)*100:.1f}%, PnL {sub_pnl:+.1f}%")

        # Por size label
        print(f"\n[POR CONFIANZA]")
        for label in trades['size_label'].unique():
            subset = trades[trades['size_label'] == label]
            if len(subset) > 0:
                sub_wins = (subset['result'] == 'WIN').sum()
                sub_pnl = subset['pnl_adjusted'].sum() * 100
                avg_conf = subset['confidence'].mean()
                print(f"  {label}: {len(subset)} trades, WR {sub_wins/len(subset)*100:.1f}%, PnL {sub_pnl:+.1f}%, Conf {avg_conf:.2f}")

        # Monthly breakdown
        print(f"\n[POR MES]")
        trades['month'] = trades['entry_time'].dt.to_period('M')
        for month in sorted(trades['month'].unique()):
            subset = trades[trades['month'] == month]
            sub_pnl = subset['pnl_adjusted'].sum() * 100
            sub_wins = (subset['result'] == 'WIN').sum()
            print(f"  {month}: {len(subset)} trades, WR {sub_wins/len(subset)*100:.1f}%, PnL {sub_pnl:+.1f}%")

        # Losses analysis
        print(f"\n[ANALISIS DE PERDIDAS]")
        losses = trades[trades['result'] == 'LOSS']
        if len(losses) > 0:
            total_loss = losses['pnl_adjusted'].sum() * 100
            max_loss = losses['pnl_adjusted'].min() * 100
            print(f"  Total perdidas: {total_loss:.1f}%")
            print(f"  Peor trade: {max_loss:.1f}%")

        # Ganancias analysis
        wins_df = trades[trades['result'] == 'WIN']
        if len(wins_df) > 0:
            total_win = wins_df['pnl_adjusted'].sum() * 100
            best_win = wins_df['pnl_adjusted'].max() * 100
            print(f"  Total ganancias: {total_win:.1f}%")
            print(f"  Mejor trade: {best_win:.1f}%")

    else:
        print("No hay trades")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
