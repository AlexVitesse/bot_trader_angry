"""
Detector de Régimen de Mercado + Estrategias Adaptativas
=========================================================
Clasifica el mercado en regímenes y aplica estrategias específicas.

Regímenes:
- BULL_TREND: Tendencia alcista fuerte
- BEAR_TREND: Tendencia bajista fuerte
- LATERAL: Mercado sin dirección clara (NO OPERAR)
- HIGH_VOL: Alta volatilidad (reducir posición)
- LOW_VOL: Baja volatilidad (TP/SL ajustados)

Uso: from regime_detector import RegimeDetector, get_strategy_for_regime
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Tuple


class MarketRegime(Enum):
    """Tipos de régimen de mercado."""
    BULL_TREND = "bull_trend"      # Tendencia alcista
    BEAR_TREND = "bear_trend"      # Tendencia bajista
    LATERAL = "lateral"            # Sin dirección (NO OPERAR)
    HIGH_VOL = "high_vol"          # Alta volatilidad
    LOW_VOL = "low_vol"            # Baja volatilidad
    UNKNOWN = "unknown"            # No clasificado


@dataclass
class RegimeStrategy:
    """Estrategia específica para un régimen."""
    regime: MarketRegime
    should_trade: bool            # ¿Operar en este régimen?
    allowed_directions: list      # [1] solo longs, [-1] solo shorts, [1, -1] ambos
    tp_multiplier: float          # Multiplicador de TP (base o ATR)
    sl_multiplier: float          # Multiplicador de SL
    use_atr: bool                 # Usar ATR dinámico o fijo
    position_size_mult: float     # Multiplicador de tamaño de posición (1.0 = normal)
    use_trailing: bool            # Usar trailing stop
    max_hold_candles: int         # Máximo de velas a mantener posición
    min_conviction: float         # Conviction mínimo para entrar
    description: str              # Descripción de la estrategia


# Estrategias por régimen
REGIME_STRATEGIES = {
    MarketRegime.BULL_TREND: RegimeStrategy(
        regime=MarketRegime.BULL_TREND,
        should_trade=True,
        allowed_directions=[1],        # Solo longs en bull
        tp_multiplier=3.0,             # TP amplio para capturar tendencia
        sl_multiplier=1.0,
        use_atr=True,
        position_size_mult=1.0,
        use_trailing=True,             # Trailing para maximizar ganancias
        max_hold_candles=40,           # Hold más largo en tendencia
        min_conviction=1.5,
        description="Bull trending: Longs con trailing stop, hold largo"
    ),

    MarketRegime.BEAR_TREND: RegimeStrategy(
        regime=MarketRegime.BEAR_TREND,
        should_trade=True,
        allowed_directions=[-1],       # Solo shorts en bear
        tp_multiplier=2.0,             # TP más corto (bear es más rápido)
        sl_multiplier=1.0,
        use_atr=True,
        position_size_mult=0.8,        # Posición menor en bear (más riesgoso)
        use_trailing=False,            # No trailing, tomar profits rápido
        max_hold_candles=20,           # Hold más corto
        min_conviction=2.0,            # Más selectivo en bear
        description="Bear trending: Shorts con TP rápido"
    ),

    MarketRegime.LATERAL: RegimeStrategy(
        regime=MarketRegime.LATERAL,
        should_trade=False,            # NO OPERAR en lateral
        allowed_directions=[],
        tp_multiplier=0,
        sl_multiplier=0,
        use_atr=False,
        position_size_mult=0,
        use_trailing=False,
        max_hold_candles=0,
        min_conviction=999,            # Nunca entra
        description="Lateral/Choppy: NO OPERAR - esperar tendencia"
    ),

    MarketRegime.HIGH_VOL: RegimeStrategy(
        regime=MarketRegime.HIGH_VOL,
        should_trade=True,
        allowed_directions=[1, -1],    # Ambas direcciones
        tp_multiplier=2.5,             # ATR amplio por volatilidad
        sl_multiplier=1.5,             # SL amplio para no ser sacado por ruido
        use_atr=True,
        position_size_mult=0.5,        # Posición reducida por riesgo
        use_trailing=True,
        max_hold_candles=15,           # Hold corto, volatilidad cambia rápido
        min_conviction=2.0,
        description="Alta volatilidad: Posición reducida, ATR amplio"
    ),

    MarketRegime.LOW_VOL: RegimeStrategy(
        regime=MarketRegime.LOW_VOL,
        should_trade=True,
        allowed_directions=[1, -1],
        tp_multiplier=0.015,           # TP fijo 1.5%
        sl_multiplier=0.01,            # SL fijo 1%
        use_atr=False,                 # NO usar ATR, usar fijos
        position_size_mult=1.2,        # Posición mayor (menos riesgo)
        use_trailing=False,
        max_hold_candles=30,
        min_conviction=1.5,
        description="Baja volatilidad: TP/SL fijos pequeños"
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
        description="Régimen desconocido: NO OPERAR"
    ),
}


class RegimeDetector:
    """
    Detecta el régimen de mercado basado en indicadores técnicos.

    Indicadores usados:
    - EMAs (8, 21, 55, 200) para tendencia
    - ADX para fuerza de tendencia
    - Choppiness Index para lateralidad
    - ATR ratio para volatilidad
    - RSI para sobrecompra/sobreventa
    """

    def __init__(
        self,
        ema_short: int = 8,
        ema_mid: int = 21,
        ema_long: int = 55,
        ema_trend: int = 200,
        adx_period: int = 14,
        adx_trend_threshold: float = 25,
        adx_weak_threshold: float = 20,
        chop_period: int = 14,
        chop_lateral_threshold: float = 55,
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
        Detecta el régimen de mercado para una fila de indicadores.

        Returns:
            Tuple[MarketRegime, Dict]: Régimen detectado y metadata
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

        # Verificar datos válidos
        if pd.isna(adx) or pd.isna(chop) or pd.isna(atr_ratio):
            return MarketRegime.UNKNOWN, {'reason': 'insufficient_data'}

        metadata = {
            'adx': adx,
            'chop': chop,
            'atr_ratio': atr_ratio,
            'rsi': rsi,
        }

        # REGLA 1: Detectar LATERAL primero (prioridad alta para evitar pérdidas)
        if chop > self.chop_lateral_threshold and adx < self.adx_weak_threshold:
            return MarketRegime.LATERAL, {**metadata, 'reason': 'high_chop_low_adx'}

        # REGLA 2: Detectar ALTA VOLATILIDAD
        if atr_ratio > self.atr_high_mult:
            # En alta volatilidad, aún clasificar dirección
            if adx > self.adx_trend_threshold:
                if dmi_plus > dmi_minus and ema8 > ema21 > ema55:
                    return MarketRegime.HIGH_VOL, {**metadata, 'reason': 'high_vol_bullish', 'bias': 'bull'}
                elif dmi_minus > dmi_plus and ema8 < ema21 < ema55:
                    return MarketRegime.HIGH_VOL, {**metadata, 'reason': 'high_vol_bearish', 'bias': 'bear'}
            return MarketRegime.HIGH_VOL, {**metadata, 'reason': 'high_vol_neutral', 'bias': 'neutral'}

        # REGLA 3: Detectar BAJA VOLATILIDAD
        if atr_ratio < self.atr_low_mult:
            return MarketRegime.LOW_VOL, {**metadata, 'reason': 'low_volatility'}

        # REGLA 4: Detectar BULL TREND
        if (adx > self.adx_trend_threshold and
            dmi_plus > dmi_minus and
            ema8 > ema21 and ema21 > ema55):
            return MarketRegime.BULL_TREND, {**metadata, 'reason': 'strong_uptrend'}

        # REGLA 5: Detectar BEAR TREND
        if (adx > self.adx_trend_threshold and
            dmi_minus > dmi_plus and
            ema8 < ema21 and ema21 < ema55):
            return MarketRegime.BEAR_TREND, {**metadata, 'reason': 'strong_downtrend'}

        # REGLA 6: Tendencia débil = LATERAL
        if adx < self.adx_weak_threshold:
            return MarketRegime.LATERAL, {**metadata, 'reason': 'weak_trend'}

        # REGLA 7: Chop alto = LATERAL
        if chop > self.chop_lateral_threshold:
            return MarketRegime.LATERAL, {**metadata, 'reason': 'choppy_market'}

        # Default: mirar EMAs para tendencia suave
        if ema8 > ema21 > ema55:
            return MarketRegime.BULL_TREND, {**metadata, 'reason': 'ema_aligned_bull'}
        elif ema8 < ema21 < ema55:
            return MarketRegime.BEAR_TREND, {**metadata, 'reason': 'ema_aligned_bear'}

        # No clasificado claramente
        return MarketRegime.LATERAL, {**metadata, 'reason': 'unclear_regime'}

    def detect_regime_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta el régimen para todo el DataFrame.

        Returns:
            DataFrame con columnas: regime, regime_reason, y metadata
        """
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

        # Agregar indicadores útiles
        result['adx'] = indicators['adx']
        result['chop'] = indicators['chop']
        result['atr_ratio'] = indicators['atr_ratio']
        result['atr'] = indicators['atr']

        return result


def get_strategy_for_regime(regime: MarketRegime) -> RegimeStrategy:
    """Obtiene la estrategia para un régimen dado."""
    return REGIME_STRATEGIES.get(regime, REGIME_STRATEGIES[MarketRegime.UNKNOWN])


def should_take_trade(
    regime: MarketRegime,
    signal_direction: int,
    conviction: float,
) -> Tuple[bool, str]:
    """
    Decide si tomar un trade basado en el régimen.

    Returns:
        Tuple[bool, str]: (should_trade, reason)
    """
    strategy = get_strategy_for_regime(regime)

    # No operar en este régimen
    if not strategy.should_trade:
        return False, f"Régimen {regime.value}: NO OPERAR"

    # Dirección no permitida
    if signal_direction not in strategy.allowed_directions:
        return False, f"Dirección {signal_direction} no permitida en {regime.value}"

    # Conviction insuficiente
    if conviction < strategy.min_conviction:
        return False, f"Conviction {conviction:.2f} < {strategy.min_conviction} requerido"

    return True, f"Trade permitido en {regime.value}"


# Test rápido
if __name__ == '__main__':
    print("="*60)
    print("DETECTOR DE RÉGIMEN DE MERCADO")
    print("="*60)

    print("\nESTRATEGIAS POR RÉGIMEN:")
    print("-"*60)

    for regime, strategy in REGIME_STRATEGIES.items():
        print(f"\n{regime.value.upper()}:")
        print(f"  Operar: {'SI' if strategy.should_trade else 'NO'}")
        if strategy.should_trade:
            dirs = ['Long' if d == 1 else 'Short' for d in strategy.allowed_directions]
            print(f"  Direcciones: {', '.join(dirs)}")
            print(f"  TP mult: {strategy.tp_multiplier}, SL mult: {strategy.sl_multiplier}")
            print(f"  ATR dinámico: {'SI' if strategy.use_atr else 'NO'}")
            print(f"  Trailing: {'SI' if strategy.use_trailing else 'NO'}")
            print(f"  Position size: {strategy.position_size_mult}x")
            print(f"  Min conviction: {strategy.min_conviction}")
        print(f"  >> {strategy.description}")
