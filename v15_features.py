"""
V15 Features - Comité de Expertos: Calculo de features.
=========================================================

Los 4 expertos:
  1. Macro Context (reglas, 1d + Fear&Greed)  -> MACRO_BULL / MACRO_BEAR / MACRO_RANGE
  2. Price Action Setup (ML, 4h)              -> prob 0.0-1.0
  3. Funding Rate Sentiment (reglas, 8h)      -> BULLISH_BIAS / BEARISH_BIAS / NEUTRAL
  4. Volume & Order Flow (ML, 4h)             -> VOLUME_CONFIRM / VOLUME_WARN / NEUTRAL

Funcion principal:
  build_feature_matrix(df_4h, df_1d, funding_df, fng_df) -> DataFrame completo
"""

import warnings
import numpy as np
import pandas as pd
import pandas_ta as pta

warnings.filterwarnings('ignore')

from v15_market_structure import add_structure_features, STRUCTURE_FEATURES

# Features de precio/momentum — calculadas por compute_setup_features()
PRICE_FEATURES = [
    'body_ratio', 'upper_wick', 'lower_wick',
    'inside_bar', 'prev_body_ratio',
    'dist_swing_high', 'dist_swing_low', 'range_pos',
    'bb_pct', 'bb_width',
    'rsi14', 'rsi7', 'rsi_slope',
    'stoch_k', 'macd_hist_norm',
    'ret_1', 'ret_3', 'ret_5', 'atr_pct',
    'consec_direction',
]

# Features que usa el modelo ML:
#   precio/momentum + contexto macro y sentimiento numerico
# Las structure features (FVG, PDH/PDL, etc.) actuan como FILTRO en el bot
# en vivo via build_daily_plan + get_nearest_zone — no como input del modelo.
BASE_SETUP_FEATURES = PRICE_FEATURES + [
    'fng_value', 'btc_ret_7d_1d',
    'funding_rate', 'funding_zscore',
]

SETUP_FEATURES = BASE_SETUP_FEATURES

VOLUME_FEATURES = [
    'taker_buy_ratio', 'taker_buy_ratio_ma5',
    'vol_ratio', 'vol_acceleration',
    'obv_slope', 'large_candle_low_vol',
]

MACRO_COLS = ['macro_regime', 'fng_value', 'fng_7d_avg', 'btc_ret_7d_1d']

SENTIMENT_COLS = [
    'funding_rate', 'funding_7d_avg', 'funding_zscore', 'sentiment_regime',
]


# =============================================================================
# EXPERTO 1: MACRO CONTEXT (reglas, daily)
# =============================================================================

def compute_macro_features(df_1d: pd.DataFrame, fng_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features macro diarias y clasifica el regimen.

    Args:
        df_1d: OHLCV diario con index UTC
        fng_df: Fear & Greed diario (columna 'fng_value')

    Returns:
        DataFrame diario con columnas macro_regime, fng_value, fng_7d_avg, btc_ret_7d_1d
        IMPORTANTE: shift(1) aplicado para evitar look-ahead.
    """
    c = df_1d['close'].copy()
    macro = pd.DataFrame(index=df_1d.index)

    # EMA diarias
    macro['ema20_1d'] = pta.ema(c, length=20)
    macro['ema50_1d'] = pta.ema(c, length=50)
    macro['ema200_1d'] = pta.ema(c, length=200)

    # Retorno 7d
    macro['btc_ret_7d_1d'] = c.pct_change(7) * 100

    # Fear & Greed - merge por fecha (solo fecha, ignorar hora)
    if fng_df is not None:
        # Normalizar indices a solo fecha
        fng_daily = fng_df.copy()
        fng_daily.index = fng_daily.index.normalize()

        macro_dates = macro.index.normalize()
        # Para cada dia de macro, tomar el FNG del dia (o del dia anterior si no hay)
        macro['fng_value'] = np.nan
        for i, dt in enumerate(macro.index):
            dt_norm = dt.normalize() if hasattr(dt, 'normalize') else pd.Timestamp(dt).normalize()
            # Buscar FNG <= dt_norm (sin look-ahead)
            mask = fng_daily.index <= dt_norm
            if mask.any():
                macro.iloc[i, macro.columns.get_loc('fng_value')] = fng_daily.loc[mask, 'fng_value'].iloc[-1]

        macro['fng_7d_avg'] = macro['fng_value'].rolling(7).mean()
    else:
        macro['fng_value'] = 50
        macro['fng_7d_avg'] = 50

    # Clasificar regimen (reglas)
    def classify_macro(row):
        ema20 = row.get('ema20_1d', np.nan)
        ema50 = row.get('ema50_1d', np.nan)
        ema200 = row.get('ema200_1d', np.nan)
        fng = row.get('fng_value', 50)
        ret7d = row.get('btc_ret_7d_1d', 0)

        if pd.isna(ema200):
            return 'MACRO_RANGE'

        ema_bull = ema20 > ema50 > ema200
        ema_bear = ema20 < ema50 < ema200

        if ema_bull:
            # Euforia extrema con sobrecompra fuerte = posible techo, no bull puro
            if fng > 85 and ret7d > 15:
                return 'MACRO_RANGE'
            return 'MACRO_BULL'
        elif ema_bear:
            # Miedo extremo con caida fuerte = posible suelo, no bear puro
            if fng < 15 and ret7d < -20:
                return 'MACRO_RANGE'
            return 'MACRO_BEAR'
        else:
            return 'MACRO_RANGE'

    macro['macro_regime'] = macro.apply(classify_macro, axis=1)

    # CRITICO: shift(1) - usar informacion del DIA ANTERIOR para evitar look-ahead
    # (el regimen del dia D se conoce al final del dia D, no al comienzo)
    macro = macro.shift(1)

    return macro[['macro_regime', 'fng_value', 'fng_7d_avg', 'btc_ret_7d_1d']]


def align_macro_to_4h(macro_1d: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Alinea features macro diarias al timeframe 4h.
    Usa forward-fill para propagar el valor del dia a todas las velas 4h del dia.
    """
    # Reindexar macro al indice 4h usando ffill
    macro_4h = macro_1d.reindex(df_4h.index, method='ffill')
    return macro_4h


# =============================================================================
# EXPERTO 2: PRICE ACTION SETUP (features para ML)
# =============================================================================

def compute_setup_features(df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features de price action para el Setup Expert.

    Incluye: patrones de vela, niveles S/R, momentum, ATR.
    """
    o, h, l, c = df_4h['open'], df_4h['high'], df_4h['low'], df_4h['close']
    feat = pd.DataFrame(index=df_4h.index)

    # --- Patrones de vela ---
    candle_range = (h - l).clip(lower=1e-10)
    body = (c - o).abs()
    feat['body_ratio'] = body / candle_range
    feat['upper_wick'] = (h - pd.concat([o, c], axis=1).max(axis=1)) / candle_range
    feat['lower_wick'] = (pd.concat([o, c], axis=1).min(axis=1) - l) / candle_range

    # Inside bar: rango actual dentro del rango anterior
    feat['inside_bar'] = ((h < h.shift(1)) & (l > l.shift(1))).astype(int)

    # Tamano del cuerpo de vela anterior (contexto)
    feat['prev_body_ratio'] = feat['body_ratio'].shift(1)

    # --- Niveles S/R ---
    swing_high = h.rolling(20).max()
    swing_low = l.rolling(20).min()
    swing_range = (swing_high - swing_low).clip(lower=1e-10)

    feat['dist_swing_high'] = (swing_high - c) / c * 100  # % por debajo del high
    feat['dist_swing_low'] = (c - swing_low) / c * 100    # % por encima del low
    feat['range_pos'] = (c - swing_low) / swing_range     # 0=minimo, 1=maximo

    # Bollinger Bands
    bb = pta.bbands(c, length=20)
    if bb is not None:
        bb_lower = bb.iloc[:, 0]
        bb_mid = bb.iloc[:, 1]
        bb_upper = bb.iloc[:, 2]
        bb_range = (bb_upper - bb_lower).clip(lower=1e-10)
        feat['bb_pct'] = (c - bb_lower) / bb_range
        feat['bb_width'] = (bb_upper - bb_lower) / bb_mid * 100
    else:
        feat['bb_pct'] = 0.5
        feat['bb_width'] = 5.0

    # --- Momentum ---
    feat['rsi14'] = pta.rsi(c, length=14)
    feat['rsi7'] = pta.rsi(c, length=7)
    feat['rsi_slope'] = feat['rsi14'].diff(3)

    stoch = pta.stoch(h, l, c, k=14, d=3)
    feat['stoch_k'] = stoch.iloc[:, 0] if stoch is not None else 50

    macd = pta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        hist = macd.iloc[:, 2]  # Histograma
        atr = pta.atr(h, l, c, length=14)
        atr_safe = atr.clip(lower=1e-10) if atr is not None else pd.Series(1.0, index=c.index)
        feat['macd_hist_norm'] = hist / atr_safe
    else:
        feat['macd_hist_norm'] = 0.0

    feat['ret_1'] = c.pct_change(1) * 100
    feat['ret_3'] = c.pct_change(3) * 100
    feat['ret_5'] = c.pct_change(5) * 100

    # ATR %
    atr = pta.atr(h, l, c, length=14)
    feat['atr_pct'] = (atr / c * 100) if atr is not None else 2.0

    # Direccion consecutiva (positivo = bullish, negativo = bearish)
    direction = np.sign(c.diff())
    feat['consec_direction'] = direction.rolling(5).sum()

    return feat[PRICE_FEATURES].replace([np.inf, -np.inf], np.nan)


# =============================================================================
# EXPERTO 3: FUNDING RATE SENTIMENT (reglas)
# =============================================================================

def compute_sentiment_features(funding_df: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features de sentimiento basadas en funding rate.

    Logica contraria:
      funding muy positivo -> mercado overlong -> BEARISH_BIAS (contrarian)
      funding muy negativo -> mercado overshort -> BULLISH_BIAS (contrarian)
      neutral -> NEUTRAL

    Args:
        funding_df: DataFrame con index UTC y columna 'funding_rate' (cada 8h)
        df_4h: OHLCV 4h (para alinear)
    """
    sent = pd.DataFrame(index=df_4h.index)

    if funding_df is None or len(funding_df) == 0:
        sent['funding_rate'] = 0.0
        sent['funding_7d_avg'] = 0.0
        sent['funding_zscore'] = 0.0
        sent['sentiment_regime'] = 'NEUTRAL'
        return sent

    # Resamplear funding (8h) al indice 4h con forward fill
    # CRITICO: shift(1) en funding para evitar look-ahead
    funding_clean = funding_df[['funding_rate']].copy()
    funding_clean = funding_clean[~funding_clean.index.duplicated(keep='first')]
    funding_clean = funding_clean.shift(1)  # usar funding del periodo anterior

    # Reindexar al 4h
    funding_4h = funding_clean.reindex(df_4h.index, method='ffill')
    sent['funding_rate'] = funding_4h['funding_rate'].fillna(0.0)

    # Rolling stats sobre el funding rate (ventana 90 dias = ~270 registros 8h)
    # Calculamos sobre la serie original (8h) y luego proyectamos
    fr = funding_4h['funding_rate']
    rolling_window = 270  # ~90 dias en velas 4h (270 * 4h / 24 = 45 dias)
    fr_mean = fr.rolling(rolling_window, min_periods=30).mean()
    fr_std = fr.rolling(rolling_window, min_periods=30).std()

    sent['funding_7d_avg'] = fr.rolling(42, min_periods=7).mean()  # 42*4h = 7 dias
    sent['funding_zscore'] = ((fr - fr_mean) / fr_std.clip(lower=1e-8)).clip(-5, 5)

    # Clasificar sentimiento (logica contraria)
    def classify_sentiment(row):
        z = row['funding_zscore']
        rate = row['funding_rate']

        # Overlong extremo (longs pagando mucho a shorts)
        if z > 2.0 or rate > 0.0005:
            return 'BEARISH_BIAS'   # Contrarian: no ir LONG
        # Overshort extremo
        elif z < -1.5 or rate < -0.0003:
            return 'BULLISH_BIAS'  # Contrarian: LONG favorecido
        else:
            return 'NEUTRAL'

    sent['sentiment_regime'] = sent[['funding_zscore', 'funding_rate']].apply(
        classify_sentiment, axis=1
    )

    return sent[SENTIMENT_COLS]


# =============================================================================
# EXPERTO 4: VOLUME & ORDER FLOW (features para ML)
# =============================================================================

def compute_volume_features(df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features de volumen para el Volume Expert.

    Si taker_buy_vol esta disponible, usa ratios exactos.
    Si no, estima via proxy basado en price action.
    """
    c = df_4h['close']
    o = df_4h['open']
    h = df_4h['high']
    l = df_4h['low']
    v = df_4h['volume']
    vol_feat = pd.DataFrame(index=df_4h.index)

    # --- Taker Buy Ratio ---
    if 'taker_buy_vol' in df_4h.columns:
        tbv = pd.to_numeric(df_4h['taker_buy_vol'], errors='coerce').fillna(v * 0.5)
        vol_feat['taker_buy_ratio'] = (tbv / v.clip(lower=1e-10)).clip(0, 1)
    else:
        # Proxy: velas alcistas tienen mas taker_buy
        candle_range = (h - l).clip(lower=1e-10)
        body_direction = (c - o) / candle_range  # -1 a +1
        vol_feat['taker_buy_ratio'] = (0.5 + body_direction * 0.4).clip(0.1, 0.9)

    # Media movil de taker_buy_ratio (tendencia de 5 velas)
    vol_feat['taker_buy_ratio_ma5'] = vol_feat['taker_buy_ratio'].rolling(5).mean()

    # --- Volume Ratio ---
    vol_ma20 = v.rolling(20).mean()
    vol_feat['vol_ratio'] = (v / vol_ma20.clip(lower=1e-10)).clip(0, 10)

    # Volume acceleration: vol actual vs media 5 velas
    vol_ma5 = v.rolling(5).mean()
    vol_feat['vol_acceleration'] = (v / vol_ma5.clip(lower=1e-10)).clip(0, 10)

    # --- OBV Slope ---
    obv = (np.sign(c.diff()) * v).cumsum()
    obv_ma10 = obv.rolling(10).mean()
    # Normalizado: cuanto se desvio el OBV reciente de su media
    obv_std = obv.rolling(20).std().clip(lower=1e-10)
    vol_feat['obv_slope'] = ((obv - obv_ma10) / obv_std).clip(-5, 5)

    # --- Trampa: vela grande con volumen bajo ---
    atr = pta.atr(h, l, c, length=14)
    atr_safe = atr.clip(lower=1e-10) if atr is not None else pd.Series(1.0, index=c.index)
    candle_size = (c - o).abs() / atr_safe
    vol_feat['large_candle_low_vol'] = (
        (candle_size > 1.5) & (vol_feat['vol_ratio'] < 0.7)
    ).astype(int)

    return vol_feat[VOLUME_FEATURES].replace([np.inf, -np.inf], np.nan)


# =============================================================================
# PIPELINE COMPLETO: UNIR TODOS LOS EXPERTOS
# =============================================================================

def build_feature_matrix(
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    funding_df: pd.DataFrame,
    fng_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye la matriz de features completa para training/inference.

    Returns:
        DataFrame con todas las features + columna 'macro_regime' + 'sentiment_regime'
        Index: timestamp UTC 4h
    """
    # Experto 2: Price Action
    setup_feat = compute_setup_features(df_4h)

    # Experto 4: Volume
    vol_feat = compute_volume_features(df_4h)

    # Experto 1: Macro (diario -> 4h)
    macro_1d = compute_macro_features(df_1d, fng_df)
    macro_4h = align_macro_to_4h(macro_1d, df_4h)

    # Experto 3: Funding Sentiment
    sent_feat = compute_sentiment_features(funding_df, df_4h)

    # Capa de Estructura de Mercado (FVG, PDH/PDL, PWH/PWL)
    struct_feat = add_structure_features(df_4h)

    # Unir todo
    feat = pd.concat([setup_feat, vol_feat, macro_4h, sent_feat, struct_feat], axis=1)

    # Eliminar filas con demasiados NaN (primeras ~200 filas por warmup de indicadores)
    min_valid = len(PRICE_FEATURES) // 2
    feat = feat.dropna(subset=PRICE_FEATURES[:min_valid], how='any')

    return feat


# =============================================================================
# LABEL: GANARA EL TRADE? (TP antes de SL en N velas)
# =============================================================================

def create_label(
    df_4h: pd.DataFrame,
    direction: str = 'long',
    tp_pct: float = 0.03,
    sl_pct: float = 0.015,
    max_candles: int = 12,
) -> pd.Series:
    """
    Label binario: 1 si el precio sube TP% antes de bajar SL% (para LONG).

    Args:
        df_4h: OHLCV 4h
        direction: 'long' o 'short'
        tp_pct: take profit %
        sl_pct: stop loss %
        max_candles: velas maximas para buscar resultado

    Returns:
        Series binaria (1=ganador, 0=perdedor), sin NaN (solo filas con resultado conocido)
    """
    closes = df_4h['close'].values
    n = len(closes)
    labels = np.full(n, np.nan)

    for i in range(n - 1):
        entry = closes[i]
        for j in range(i + 1, min(i + max_candles + 1, n)):
            fut = closes[j]
            if direction == 'long':
                pnl = (fut - entry) / entry
                if pnl >= tp_pct:
                    labels[i] = 1
                    break
                elif pnl <= -sl_pct:
                    labels[i] = 0
                    break
            else:  # short
                pnl = (entry - fut) / entry
                if pnl >= tp_pct:
                    labels[i] = 1
                    break
                elif pnl <= -sl_pct:
                    labels[i] = 0
                    break

    result = pd.Series(labels, index=df_4h.index, name='label')
    return result.dropna()
