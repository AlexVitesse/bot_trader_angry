"""
Detector de Regimen de Mercado V2 (Menos Restrictivo)
======================================================
V10.1 - Agrega WEAK_TREND para mercados ambiguos y ajusta thresholds.

Cambios vs V1:
- Nuevo regimen WEAK_TREND: Opera con posicion reducida
- chop_lateral_threshold: 55 -> 62 (solo mercados muy choppy son lateral)
- adx_weak_threshold: 20 -> 15 (permite tendencias debiles)
- adx_trend_threshold: 25 -> 20 (baja el requisito para tendencia fuerte)

Regimenes:
- BULL_TREND: Tendencia alcista fuerte
- BEAR_TREND: Tendencia bajista fuerte
- WEAK_TREND: Tendencia debil (operar con posicion reducida) - NUEVO
- LATERAL: Mercado MUY choppy (NO OPERAR)
- HIGH_VOL: Alta volatilidad (reducir posicion)
- LOW_VOL: Baja volatilidad (TP/SL ajustados)
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Tuple


class MarketRegime(Enum):
    """Tipos de regimen de mercado."""
    BULL_TREND = "bull_trend"       # Tendencia alcista fuerte
    BEAR_TREND = "bear_trend"       # Tendencia bajista fuerte
    WEAK_TREND = "weak_trend"       # Tendencia debil pero operable
    LATERAL = "lateral"             # Muy choppy (NO OPERAR)
    HIGH_VOL = "high_vol"           # Alta volatilidad
    LOW_VOL = "low_vol"             # Baja volatilidad
    UNKNOWN = "unknown"             # No clasificado


@dataclass
class RegimeStrategy:
    """Estrategia especifica para un regimen."""
    regime: MarketRegime
    should_trade: bool
    allowed_directions: list
    tp_multiplier: float
    sl_multiplier: float
    use_atr: bool
    position_size_mult: float
    use_trailing: bool
    max_hold_candles: int
    min_conviction: float
    description: str


# Estrategias por regimen V2
REGIME_STRATEGIES = {
    MarketRegime.BULL_TREND: RegimeStrategy(
        regime=MarketRegime.BULL_TREND,
        should_trade=True,
        allowed_directions=[1],
        tp_multiplier=3.0,
        sl_multiplier=1.0,
        use_atr=True,
        position_size_mult=1.0,
        use_trailing=True,
        max_hold_candles=40,
        min_conviction=1.5,
        description="Bull trending: Longs con trailing stop"
    ),

    MarketRegime.BEAR_TREND: RegimeStrategy(
        regime=MarketRegime.BEAR_TREND,
        should_trade=True,
        allowed_directions=[-1],
        tp_multiplier=2.0,
        sl_multiplier=1.0,
        use_atr=True,
        position_size_mult=0.8,
        use_trailing=False,
        max_hold_candles=20,
        min_conviction=1.8,
        description="Bear trending: Shorts con TP rapido"
    ),

    # NUEVO: Tendencia debil pero operable
    MarketRegime.WEAK_TREND: RegimeStrategy(
        regime=MarketRegime.WEAK_TREND,
        should_trade=True,
        allowed_directions=[1, -1],     # Ambas direcciones
        tp_multiplier=2.5,              # TP igual que V9.5
        sl_multiplier=1.0,
        use_atr=True,
        position_size_mult=0.85,        # Posicion casi normal (85%)
        use_trailing=False,
        max_hold_candles=25,
        min_conviction=1.8,             # Conviction como V9.5
        description="Weak trend: Ambas direcciones, posicion 85%"
    ),

    MarketRegime.LATERAL: RegimeStrategy(
        regime=MarketRegime.LATERAL,
        should_trade=False,             # NO OPERAR en muy choppy
        allowed_directions=[],
        tp_multiplier=0,
        sl_multiplier=0,
        use_atr=False,
        position_size_mult=0,
        use_trailing=False,
        max_hold_candles=0,
        min_conviction=999,
        description="Lateral/Choppy extremo: NO OPERAR"
    ),

    MarketRegime.HIGH_VOL: RegimeStrategy(
        regime=MarketRegime.HIGH_VOL,
        should_trade=True,
        allowed_directions=[1, -1],
        tp_multiplier=2.5,
        sl_multiplier=1.5,
        use_atr=True,
        position_size_mult=0.5,
        use_trailing=True,
        max_hold_candles=15,
        min_conviction=2.0,
        description="Alta volatilidad: Posicion reducida, ATR amplio"
    ),

    MarketRegime.LOW_VOL: RegimeStrategy(
        regime=MarketRegime.LOW_VOL,
        should_trade=True,
        allowed_directions=[1, -1],
        tp_multiplier=0.015,
        sl_multiplier=0.01,
        use_atr=False,
        position_size_mult=1.2,
        use_trailing=False,
        max_hold_candles=30,
        min_conviction=1.5,
        description="Baja volatilidad: TP/SL fijos pequenos"
    ),

    MarketRegime.UNKNOWN: RegimeStrategy(
        regime=MarketRegime.UNKNOWN,
        should_trade=False,
        allowed_directions=[],
        tp_multiplier=0,
        sl_multiplier=0,
        use_atr=False,
        position_size_mult=0,
        use_trailing=False,
        max_hold_candles=0,
        min_conviction=999,
        description="Regimen desconocido: NO OPERAR"
    ),
}


class RegimeDetector:
    """
    Detecta el regimen de mercado V2 (menos restrictivo).

    Cambios vs V1:
    - chop_lateral_threshold: 55 -> 62
    - adx_weak_threshold: 20 -> 15
    - adx_trend_threshold: 25 -> 20
    - Nuevo regimen WEAK_TREND para mercados ambiguos
    """

    def __init__(
        self,
        ema_short: int = 8,
        ema_mid: int = 21,
        ema_long: int = 55,
        ema_trend: int = 200,
        adx_period: int = 14,
        adx_trend_threshold: float = 20,    # Bajado de 25
        adx_weak_threshold: float = 15,     # Bajado de 20
        chop_period: int = 14,
        chop_lateral_threshold: float = 62, # Subido de 55
        atr_period: int = 14,
        atr_avg_period: int = 50,
        atr_high_mult: float = 1.5,
        atr_low_mult: float = 0.7,
    ):
        self.ema_short = ema_short
        self.ema_mid = ema_mid
        self.ema_long = ema_long
        self.ema_trend = ema_trend
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_weak_threshold = adx_weak_threshold
        self.chop_period = chop_period
        self.chop_lateral_threshold = chop_lateral_threshold
        self.atr_period = atr_period
        self.atr_avg_period = atr_avg_period
        self.atr_high_mult = atr_high_mult
        self.atr_low_mult = atr_low_mult

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula todos los indicadores necesarios."""
        indicators = pd.DataFrame(index=df.index)

        c, h, l = df['close'], df['high'], df['low']

        # EMAs
        indicators['ema8'] = ta.ema(c, length=self.ema_short)
        indicators['ema21'] = ta.ema(c, length=self.ema_mid)
        indicators['ema55'] = ta.ema(c, length=self.ema_long)
        indicators['ema200'] = ta.ema(c, length=self.ema_trend)

        # ADX
        adx_data = ta.adx(h, l, c, length=self.adx_period)
        if adx_data is not None:
            indicators['adx'] = adx_data.iloc[:, 0]
            indicators['dmi_plus'] = adx_data.iloc[:, 1]
            indicators['dmi_minus'] = adx_data.iloc[:, 2]

        # Choppiness Index
        atr_1 = ta.atr(h, l, c, length=1)
        atr_sum = atr_1.rolling(self.chop_period).sum()
        high_max = h.rolling(self.chop_period).max()
        low_min = l.rolling(self.chop_period).min()
        indicators['chop'] = 100 * np.log10(atr_sum / (high_max - low_min + 1e-10)) / np.log10(self.chop_period)

        # ATR y ratio
        indicators['atr'] = ta.atr(h, l, c, length=self.atr_period)
        indicators['atr_avg'] = indicators['atr'].rolling(self.atr_avg_period).mean()
        indicators['atr_ratio'] = indicators['atr'] / indicators['atr_avg']

        # RSI
        indicators['rsi'] = ta.rsi(c, length=14)

        # Precio relativo a EMAs
        indicators['price_vs_ema200'] = (c - indicators['ema200']) / indicators['ema200'] * 100

        return indicators

    def detect_regime(self, indicators: pd.Series) -> Tuple[MarketRegime, Dict]:
        """
        Detecta el regimen de mercado V2.
        """
        # Extraer valores
        ema8 = indicators.get('ema8', np.nan)
        ema21 = indicators.get('ema21', np.nan)
        ema55 = indicators.get('ema55', np.nan)
        ema200 = indicators.get('ema200', np.nan)
        adx = indicators.get('adx', np.nan)
        dmi_plus = indicators.get('dmi_plus', np.nan)
        dmi_minus = indicators.get('dmi_minus', np.nan)
        chop = indicators.get('chop', np.nan)
        atr_ratio = indicators.get('atr_ratio', np.nan)
        rsi = indicators.get('rsi', np.nan)

        # Verificar datos validos
        if pd.isna(adx) or pd.isna(chop) or pd.isna(atr_ratio):
            return MarketRegime.UNKNOWN, {'reason': 'insufficient_data'}

        metadata = {
            'adx': adx,
            'chop': chop,
            'atr_ratio': atr_ratio,
            'rsi': rsi,
        }

        # REGLA 1: LATERAL solo si MUY choppy (chop > 62 y ADX < 15)
        if chop > self.chop_lateral_threshold and adx < self.adx_weak_threshold:
            return MarketRegime.LATERAL, {**metadata, 'reason': 'extreme_chop'}

        # REGLA 2: ALTA VOLATILIDAD
        if atr_ratio > self.atr_high_mult:
            if adx > self.adx_trend_threshold:
                if dmi_plus > dmi_minus and ema8 > ema21 > ema55:
                    return MarketRegime.HIGH_VOL, {**metadata, 'reason': 'high_vol_bullish', 'bias': 'bull'}
                elif dmi_minus > dmi_plus and ema8 < ema21 < ema55:
                    return MarketRegime.HIGH_VOL, {**metadata, 'reason': 'high_vol_bearish', 'bias': 'bear'}
            return MarketRegime.HIGH_VOL, {**metadata, 'reason': 'high_vol_neutral', 'bias': 'neutral'}

        # REGLA 3: BAJA VOLATILIDAD
        if atr_ratio < self.atr_low_mult:
            return MarketRegime.LOW_VOL, {**metadata, 'reason': 'low_volatility'}

        # REGLA 4: BULL TREND fuerte
        if (adx > self.adx_trend_threshold and
            dmi_plus > dmi_minus and
            ema8 > ema21 and ema21 > ema55):
            return MarketRegime.BULL_TREND, {**metadata, 'reason': 'strong_uptrend'}

        # REGLA 5: BEAR TREND fuerte
        if (adx > self.adx_trend_threshold and
            dmi_minus > dmi_plus and
            ema8 < ema21 and ema21 < ema55):
            return MarketRegime.BEAR_TREND, {**metadata, 'reason': 'strong_downtrend'}

        # REGLA 6: EMAs alineadas pero ADX debil = WEAK_TREND
        if ema8 > ema21 > ema55:
            return MarketRegime.WEAK_TREND, {**metadata, 'reason': 'weak_uptrend'}
        elif ema8 < ema21 < ema55:
            return MarketRegime.WEAK_TREND, {**metadata, 'reason': 'weak_downtrend'}

        # REGLA 7: ADX muy bajo = WEAK_TREND (no lateral)
        if adx < self.adx_weak_threshold:
            return MarketRegime.WEAK_TREND, {**metadata, 'reason': 'very_weak_trend'}

        # REGLA 8: Chop moderado = WEAK_TREND
        if chop > 55:  # Entre 55 y 62
            return MarketRegime.WEAK_TREND, {**metadata, 'reason': 'moderate_chop'}

        # Default: WEAK_TREND (antes era LATERAL)
        return MarketRegime.WEAK_TREND, {**metadata, 'reason': 'unclear_regime'}

    def detect_regime_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta el regimen para todo el DataFrame."""
        indicators = self.compute_indicators(df)

        regimes = []
        reasons = []

        for idx in indicators.index:
            regime, meta = self.detect_regime(indicators.loc[idx])
            regimes.append(regime.value)
            reasons.append(meta.get('reason', ''))

        result = pd.DataFrame({
            'regime': regimes,
            'regime_reason': reasons,
        }, index=indicators.index)

        result['adx'] = indicators['adx']
        result['chop'] = indicators['chop']
        result['atr_ratio'] = indicators['atr_ratio']
        result['atr'] = indicators['atr']

        return result


def get_strategy_for_regime(regime: MarketRegime) -> RegimeStrategy:
    """Obtiene la estrategia para un regimen dado."""
    return REGIME_STRATEGIES.get(regime, REGIME_STRATEGIES[MarketRegime.UNKNOWN])


def should_take_trade(
    regime: MarketRegime,
    signal_direction: int,
    conviction: float,
) -> Tuple[bool, str]:
    """Decide si tomar un trade basado en el regimen."""
    strategy = get_strategy_for_regime(regime)

    if not strategy.should_trade:
        return False, f"Regimen {regime.value}: NO OPERAR"

    if signal_direction not in strategy.allowed_directions:
        return False, f"Direccion {signal_direction} no permitida en {regime.value}"

    if conviction < strategy.min_conviction:
        return False, f"Conviction {conviction:.2f} < {strategy.min_conviction} requerido"

    return True, f"Trade permitido en {regime.value}"


# Test rapido
if __name__ == '__main__':
    print("="*60)
    print("DETECTOR DE REGIMEN V2 (MENOS RESTRICTIVO)")
    print("="*60)

    print("\nESTRATEGIAS POR REGIMEN:")
    print("-"*60)

    for regime, strategy in REGIME_STRATEGIES.items():
        print(f"\n{regime.value.upper()}:")
        print(f"  Operar: {'SI' if strategy.should_trade else 'NO'}")
        if strategy.should_trade:
            dirs = ['Long' if d == 1 else 'Short' for d in strategy.allowed_directions]
            print(f"  Direcciones: {', '.join(dirs)}")
            print(f"  TP mult: {strategy.tp_multiplier}, SL mult: {strategy.sl_multiplier}")
            print(f"  ATR dinamico: {'SI' if strategy.use_atr else 'NO'}")
            print(f"  Trailing: {'SI' if strategy.use_trailing else 'NO'}")
            print(f"  Position size: {strategy.position_size_mult}x")
            print(f"  Min conviction: {strategy.min_conviction}")
        print(f"  >> {strategy.description}")
