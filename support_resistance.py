"""
Support & Resistance Module
============================
Detecta niveles de soporte y resistencia para mejorar timing de entrada.

Tipos de S/R implementados:
1. Swing Highs/Lows - Máximos/mínimos locales
2. Pivot Points - High/Low/Close del período anterior
3. ATR Bands - Niveles dinámicos basados en volatilidad
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class SRLevel:
    """Un nivel de soporte o resistencia."""
    price: float
    type: str  # 'support' o 'resistance'
    strength: int  # Cuántas veces fue testeado
    age: int  # Velas desde que se formó


@dataclass
class SRAnalysis:
    """Análisis de S/R para una vela específica."""
    nearest_support: Optional[float]
    nearest_resistance: Optional[float]
    dist_to_support_pct: float  # % de distancia al soporte
    dist_to_resistance_pct: float  # % de distancia a resistencia
    at_support: bool  # Está cerca de soporte (< threshold)
    at_resistance: bool  # Está cerca de resistencia
    support_strength: int  # Fuerza del soporte cercano
    resistance_strength: int  # Fuerza de la resistencia cercana
    rr_ratio: float  # Risk/Reward ratio basado en S/R


class SupportResistance:
    """
    Detector de Soportes y Resistencias.

    Uso:
        sr = SupportResistance(lookback=20, threshold_pct=0.02)
        analysis = sr.analyze(df, idx)

        if analysis.at_support and signal == 1:  # LONG cerca de soporte
            take_trade = True
    """

    def __init__(
        self,
        lookback: int = 20,  # Velas para buscar swing points
        threshold_pct: float = 0.02,  # 2% = "cerca" de S/R
        min_touches: int = 2,  # Mínimo toques para ser S/R válido
        pivot_period: str = 'daily',  # 'daily', 'weekly'
    ):
        self.lookback = lookback
        self.threshold_pct = threshold_pct
        self.min_touches = min_touches
        self.pivot_period = pivot_period

    def find_swing_highs(self, highs: pd.Series, window: int = 5) -> pd.Series:
        """
        Encuentra swing highs (máximos locales).
        Un swing high es un high mayor que los 'window' highs anteriores y posteriores.
        """
        swing_highs = pd.Series(index=highs.index, dtype=float)
        swing_highs[:] = np.nan

        for i in range(window, len(highs) - window):
            current = highs.iloc[i]
            left = highs.iloc[i-window:i].max()
            right = highs.iloc[i+1:i+window+1].max()

            if current >= left and current >= right:
                swing_highs.iloc[i] = current

        return swing_highs

    def find_swing_lows(self, lows: pd.Series, window: int = 5) -> pd.Series:
        """
        Encuentra swing lows (mínimos locales).
        """
        swing_lows = pd.Series(index=lows.index, dtype=float)
        swing_lows[:] = np.nan

        for i in range(window, len(lows) - window):
            current = lows.iloc[i]
            left = lows.iloc[i-window:i].min()
            right = lows.iloc[i+1:i+window+1].min()

            if current <= left and current <= right:
                swing_lows.iloc[i] = current

        return swing_lows

    def get_pivot_points(self, df: pd.DataFrame, idx: int) -> Tuple[float, float, float]:
        """
        Calcula pivot points del período anterior.
        Returns: (pivot, support1, resistance1)
        """
        # Usar últimas 6 velas (1 día en 4h)
        period = 6 if self.pivot_period == 'daily' else 42  # 42 = 1 semana en 4h

        if idx < period:
            return None, None, None

        period_data = df.iloc[idx-period:idx]
        high = period_data['high'].max()
        low = period_data['low'].min()
        close = period_data['close'].iloc[-1]

        pivot = (high + low + close) / 3
        support1 = 2 * pivot - high
        resistance1 = 2 * pivot - low

        return pivot, support1, resistance1

    def get_recent_levels(
        self,
        df: pd.DataFrame,
        idx: int
    ) -> Tuple[List[float], List[float]]:
        """
        Obtiene niveles de S/R recientes basados en swing points.
        Returns: (supports, resistances)
        """
        if idx < self.lookback + 10:
            return [], []

        # Datos históricos (sin incluir vela actual - evitar look-ahead)
        hist = df.iloc[:idx]

        # Encontrar swings
        swing_highs = self.find_swing_highs(hist['high'].tail(self.lookback * 3), window=5)
        swing_lows = self.find_swing_lows(hist['low'].tail(self.lookback * 3), window=5)

        # Filtrar NaN y obtener valores únicos
        resistances = swing_highs.dropna().values.tolist()
        supports = swing_lows.dropna().values.tolist()

        # Agregar pivot points
        pivot, s1, r1 = self.get_pivot_points(df, idx)
        if pivot is not None:
            supports.append(s1)
            resistances.append(r1)

        # Consolidar niveles cercanos (agrupar si están dentro del 1%)
        supports = self._consolidate_levels(supports)
        resistances = self._consolidate_levels(resistances)

        return supports, resistances

    def _consolidate_levels(self, levels: List[float], tolerance: float = 0.01) -> List[float]:
        """Agrupa niveles que están muy cerca entre sí."""
        if not levels:
            return []

        levels = sorted(levels)
        consolidated = [levels[0]]

        for level in levels[1:]:
            if abs(level - consolidated[-1]) / consolidated[-1] > tolerance:
                consolidated.append(level)
            else:
                # Promediar niveles cercanos
                consolidated[-1] = (consolidated[-1] + level) / 2

        return consolidated

    def analyze(self, df: pd.DataFrame, idx: int) -> SRAnalysis:
        """
        Analiza S/R para una vela específica.

        Args:
            df: DataFrame con OHLCV
            idx: Índice de la vela a analizar

        Returns:
            SRAnalysis con toda la información de S/R
        """
        current_price = df.iloc[idx]['close']

        supports, resistances = self.get_recent_levels(df, idx)

        # Encontrar soporte más cercano (debajo del precio)
        valid_supports = [s for s in supports if s < current_price]
        nearest_support = max(valid_supports) if valid_supports else None

        # Encontrar resistencia más cercana (arriba del precio)
        valid_resistances = [r for r in resistances if r > current_price]
        nearest_resistance = min(valid_resistances) if valid_resistances else None

        # Calcular distancias
        dist_to_support = (current_price - nearest_support) / current_price if nearest_support else 1.0
        dist_to_resistance = (nearest_resistance - current_price) / current_price if nearest_resistance else 1.0

        # Determinar si estamos "cerca" de S/R
        at_support = dist_to_support < self.threshold_pct
        at_resistance = dist_to_resistance < self.threshold_pct

        # Calcular R:R ratio
        if nearest_support and nearest_resistance:
            potential_profit = nearest_resistance - current_price
            potential_loss = current_price - nearest_support
            rr_ratio = potential_profit / potential_loss if potential_loss > 0 else 0
        else:
            rr_ratio = 1.0

        # Calcular fuerza (cuántas veces el precio tocó ese nivel)
        support_strength = self._calculate_level_strength(df, idx, nearest_support, 'support') if nearest_support else 0
        resistance_strength = self._calculate_level_strength(df, idx, nearest_resistance, 'resistance') if nearest_resistance else 0

        return SRAnalysis(
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            dist_to_support_pct=dist_to_support * 100,
            dist_to_resistance_pct=dist_to_resistance * 100,
            at_support=at_support,
            at_resistance=at_resistance,
            support_strength=support_strength,
            resistance_strength=resistance_strength,
            rr_ratio=rr_ratio,
        )

    def _calculate_level_strength(
        self,
        df: pd.DataFrame,
        idx: int,
        level: float,
        level_type: str
    ) -> int:
        """Cuenta cuántas veces el precio tocó un nivel."""
        if level is None:
            return 0

        lookback_data = df.iloc[max(0, idx-self.lookback*3):idx]
        tolerance = level * 0.005  # 0.5% de tolerancia

        touches = 0
        for _, row in lookback_data.iterrows():
            if level_type == 'support':
                if abs(row['low'] - level) < tolerance:
                    touches += 1
            else:
                if abs(row['high'] - level) < tolerance:
                    touches += 1

        return touches

    def compute_sr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula features de S/R para todo el DataFrame.
        Útil para backtest.
        """
        features = pd.DataFrame(index=df.index)

        features['sr_dist_support'] = np.nan
        features['sr_dist_resistance'] = np.nan
        features['sr_at_support'] = False
        features['sr_at_resistance'] = False
        features['sr_rr_ratio'] = 1.0
        features['sr_support_strength'] = 0
        features['sr_resistance_strength'] = 0

        for i in range(self.lookback + 15, len(df)):
            analysis = self.analyze(df, i)
            features.iloc[i, features.columns.get_loc('sr_dist_support')] = analysis.dist_to_support_pct
            features.iloc[i, features.columns.get_loc('sr_dist_resistance')] = analysis.dist_to_resistance_pct
            features.iloc[i, features.columns.get_loc('sr_at_support')] = analysis.at_support
            features.iloc[i, features.columns.get_loc('sr_at_resistance')] = analysis.at_resistance
            features.iloc[i, features.columns.get_loc('sr_rr_ratio')] = analysis.rr_ratio
            features.iloc[i, features.columns.get_loc('sr_support_strength')] = analysis.support_strength
            features.iloc[i, features.columns.get_loc('sr_resistance_strength')] = analysis.resistance_strength

        return features


def should_take_long(analysis: SRAnalysis, min_rr: float = 1.5) -> Tuple[bool, str]:
    """
    Decide si tomar un LONG basado en S/R.

    Returns:
        (should_trade, reason)
    """
    # Rechazar si estamos muy cerca de resistencia
    if analysis.at_resistance:
        return False, "Cerca de resistencia - riesgo de rechazo"

    # Preferir entrar cerca de soporte
    if analysis.at_support:
        return True, f"Cerca de soporte - buen R:R ({analysis.rr_ratio:.1f})"

    # Verificar R:R ratio
    if analysis.rr_ratio < min_rr:
        return False, f"R:R bajo ({analysis.rr_ratio:.1f} < {min_rr})"

    # OK si no hay contraindicaciones
    return True, "S/R neutral"


def should_take_short(analysis: SRAnalysis, min_rr: float = 1.5) -> Tuple[bool, str]:
    """
    Decide si tomar un SHORT basado en S/R.
    """
    # Rechazar si estamos muy cerca de soporte
    if analysis.at_support:
        return False, "Cerca de soporte - riesgo de rebote"

    # Preferir entrar cerca de resistencia
    if analysis.at_resistance:
        return True, f"Cerca de resistencia - buen R:R"

    # Verificar R:R ratio (invertido para shorts)
    if analysis.rr_ratio > 0 and (1/analysis.rr_ratio) < min_rr:
        return False, f"R:R bajo para short"

    return True, "S/R neutral"


# Test rápido
if __name__ == '__main__':
    from pathlib import Path
    import pandas as pd

    # Cargar datos de ejemplo
    DATA_DIR = Path('data')
    df = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_history.parquet')

    print("="*60)
    print("TEST: Support & Resistance Module")
    print("="*60)

    sr = SupportResistance(lookback=20, threshold_pct=0.02)

    # Analizar última vela
    idx = len(df) - 1
    analysis = sr.analyze(df, idx)

    print(f"\nPrecio actual: ${df.iloc[idx]['close']:,.0f}")
    print(f"\nSoporte más cercano: ${analysis.nearest_support:,.0f}" if analysis.nearest_support else "No hay soporte")
    print(f"Resistencia más cercana: ${analysis.nearest_resistance:,.0f}" if analysis.nearest_resistance else "No hay resistencia")
    print(f"\nDistancia a soporte: {analysis.dist_to_support_pct:.1f}%")
    print(f"Distancia a resistencia: {analysis.dist_to_resistance_pct:.1f}%")
    print(f"\nCerca de soporte: {'SÍ' if analysis.at_support else 'NO'}")
    print(f"Cerca de resistencia: {'SÍ' if analysis.at_resistance else 'NO'}")
    print(f"\nR:R Ratio: {analysis.rr_ratio:.2f}")
    print(f"Fuerza soporte: {analysis.support_strength} toques")
    print(f"Fuerza resistencia: {analysis.resistance_strength} toques")

    # Test decisión
    should_long, reason = should_take_long(analysis)
    print(f"\n¿Tomar LONG?: {'SÍ' if should_long else 'NO'} - {reason}")
