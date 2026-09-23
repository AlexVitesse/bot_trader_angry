"""
V15 Market Structure - Capa de planificacion del trader.
=========================================================

Un trader real define DONDE quiere operar ANTES de que aparezca el setup.
Este modulo implementa ese proceso automaticamente:

  CAPA 0 - ESTRUCTURA SEMANAL
    Identifica si el mercado hace Higher Highs/Lows (alcista)
    o Lower Highs/Lows (bajista). Determina el bias de fondo.

  CAPA 1 - ZONAS DE INTERES (plan del dia/semana)
    Fair Value Gaps: zonas donde el precio se movio tan rapido
      que dejo un "vacio" — tiende a volver a llenarlos.
    PDH/PDL: Previous Day High/Low — los niveles mas vigilados
      por traders institucionales.
    PWH/PWL: Previous Week High/Low — niveles semanales clave.

  CAPA 2 - PROXIMIDAD (filtro de ejecucion)
    Solo cuando el precio llega a una zona pre-identificada
    tiene sentido correr el analisis tecnico para entrar.

Integracion:
  - add_structure_features(df_4h): para training (batch, vectorizado)
  - build_daily_plan(df_4h):       para el bot en vivo (ultima vela)
  - get_nearest_zone(price, zones, atr): checar si precio esta en zona
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Tolerancia para "estar en zona": precio a menos de ATR * ZONE_TOLERANCE
ZONE_TOLERANCE = 0.6  # veces el ATR


@dataclass
class Zone:
    """Una zona de interes para operar."""
    level: float          # precio central de la zona
    top: float            # extremo superior
    bottom: float         # extremo inferior
    direction: str        # 'LONG' o 'SHORT'
    zone_type: str        # 'FVG_BULL', 'FVG_BEAR', 'PDH', 'PDL', 'PWH', 'PWL'
    strength: int         # 1 = basico, 2 = confirmado, 3 = muy fuerte
    age_candles: int      # cuantas velas tiene
    description: str = ''


@dataclass
class DailyPlan:
    """Plan del trader para el periodo actual."""
    structure: str            # 'BULL' / 'BEAR' / 'NEUTRAL'
    bias: str                 # 'BULL_BIAS' / 'BEAR_BIAS' / 'NEUTRAL'
    zones: List[Zone] = field(default_factory=list)
    pdh: float = 0.0          # Previous Day High
    pdl: float = 0.0          # Previous Day Low
    pwh: float = 0.0          # Previous Week High
    pwl: float = 0.0          # Previous Week Low
    current_price: float = 0.0
    atr: float = 0.0


# =============================================================================
# FAIR VALUE GAPS (FVG)
# =============================================================================

def compute_fvg_features(df: pd.DataFrame, max_age: int = 40) -> pd.DataFrame:
    """
    Detecta Fair Value Gaps y determina si el precio actual esta dentro de uno.

    Un FVG es una zona donde el precio se movio tan rapido que dejo un "vacio"
    de liquidez — el mercado tiende a volver a llenarlos.

    Bullish FVG (zona de LONG):
      Se forma cuando: high[i-2] < low[i] (candle i es alcista fuerte)
      El "gap" es la zona entre high[i-2] y low[i]
      Cuando el precio vuelve a esta zona -> potencial entrada LONG

    Bearish FVG (zona de SHORT):
      Se forma cuando: low[i-2] > high[i] (candle i es bajista fuerte)
      El "gap" es la zona entre high[i] y low[i-2]
      Cuando el precio vuelve a esta zona -> potencial entrada SHORT

    Returns:
        DataFrame con columnas: at_fvg_bull, at_fvg_bear,
                                 fvg_bull_top, fvg_bull_bot,
                                 fvg_bear_top, fvg_bear_bot
    """
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    n = len(df)

    at_bull = np.zeros(n, dtype=int)
    at_bear = np.zeros(n, dtype=int)
    bull_top = np.full(n, np.nan)
    bull_bot = np.full(n, np.nan)
    bear_top = np.full(n, np.nan)
    bear_bot = np.full(n, np.nan)

    # Lista de FVGs activos: [bottom, top, formed_at]
    active_bull = []
    active_bear = []

    for i in range(2, n):
        # Detectar nuevo FVG en candle i (requiere i-2 y i ya cerrados)
        # Bullish FVG: gap entre high[i-2] y low[i]
        if h[i-2] < l[i]:
            gap_bot = h[i-2]
            gap_top = l[i]
            if gap_top > gap_bot:  # gap real, no ruido
                active_bull.append({'bot': gap_bot, 'top': gap_top, 'formed': i})

        # Bearish FVG: gap entre low[i-2] y high[i]
        if l[i-2] > h[i]:
            gap_bot = h[i]
            gap_top = l[i-2]
            if gap_top > gap_bot:
                active_bear.append({'bot': gap_bot, 'top': gap_top, 'formed': i})

        cur = c[i]

        # Verificar si precio esta dentro de FVG bullish activo
        for fvg in active_bull:
            age = i - fvg['formed']
            if age > max_age:
                continue
            if fvg['bot'] <= cur <= fvg['top']:
                at_bull[i] = 1
                bull_bot[i] = fvg['bot']
                bull_top[i] = fvg['top']
                break  # usar el mas reciente que califica

        # Verificar si precio esta dentro de FVG bearish activo
        for fvg in active_bear:
            age = i - fvg['formed']
            if age > max_age:
                continue
            if fvg['bot'] <= cur <= fvg['top']:
                at_bear[i] = 1
                bear_bot[i] = fvg['bot']
                bear_top[i] = fvg['top']
                break

        # Limpiar FVGs viejos cada 100 candles
        if i % 100 == 0:
            active_bull = [f for f in active_bull if i - f['formed'] <= max_age]
            active_bear = [f for f in active_bear if i - f['formed'] <= max_age]

    result = pd.DataFrame({
        'at_fvg_bull': at_bull,
        'at_fvg_bear': at_bear,
        'fvg_bull_top': bull_top,
        'fvg_bull_bot': bull_bot,
        'fvg_bear_top': bear_top,
        'fvg_bear_bot': bear_bot,
    }, index=df.index)

    return result


def get_active_fvgs_live(df: pd.DataFrame, max_age: int = 40) -> List[Zone]:
    """
    Retorna lista de FVGs activos (sin llenar) al final del DataFrame.
    Para uso en el bot en tiempo real.
    """
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    n = len(df)
    zones = []

    # Buscar FVGs en las ultimas max_age*2 velas
    start = max(0, n - max_age * 2)
    cur_price = c[-1]

    for i in range(start + 2, n - 1):  # -1 para no usar vela actual incompleta
        # Bullish FVG
        if h[i-2] < l[i]:
            gap_bot = h[i-2]
            gap_top = l[i]
            age = (n - 1) - i
            if age <= max_age and gap_top > gap_bot:
                # Verificar que no este "llenado" (precio nunca cerro por debajo del bot)
                min_close_after = min(c[i:]) if i < n else cur_price
                if min_close_after >= gap_bot * 0.99:  # no completamente llenado
                    zones.append(Zone(
                        level=(gap_top + gap_bot) / 2,
                        top=gap_top,
                        bottom=gap_bot,
                        direction='LONG',
                        zone_type='FVG_BULL',
                        strength=2 if age < 10 else 1,
                        age_candles=age,
                        description=f'Bullish FVG {gap_bot:.0f}-{gap_top:.0f} ({age}v)',
                    ))

        # Bearish FVG
        if i >= 2 and l[i-2] > h[i]:
            gap_bot = h[i]
            gap_top = l[i-2]
            age = (n - 1) - i
            if age <= max_age and gap_top > gap_bot:
                max_close_after = max(c[i:]) if i < n else cur_price
                if max_close_after <= gap_top * 1.01:
                    zones.append(Zone(
                        level=(gap_top + gap_bot) / 2,
                        top=gap_top,
                        bottom=gap_bot,
                        direction='SHORT',
                        zone_type='FVG_BEAR',
                        strength=2 if age < 10 else 1,
                        age_candles=age,
                        description=f'Bearish FVG {gap_bot:.0f}-{gap_top:.0f} ({age}v)',
                    ))

    return zones


# =============================================================================
# PREVIOUS DAY / WEEK HIGH-LOW
# =============================================================================

def compute_prev_day_levels(df_4h: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    PDH/PDL (Previous Day High/Low) para cada vela 4h.

    Son los niveles mas vigilados por traders institucionales.
    Una ruptura de PDH = sesgo alcista. Un rechazo en PDH = sesgo bajista.

    IMPORTANTE: shift(1) asegura que usamos el dia ANTERIOR, sin look-ahead.
    """
    daily_high = df_4h['high'].resample('1D').max().shift(1)
    daily_low = df_4h['low'].resample('1D').min().shift(1)
    pdh = daily_high.reindex(df_4h.index, method='ffill')
    pdl = daily_low.reindex(df_4h.index, method='ffill')
    return pdh, pdl


def compute_prev_week_levels(df_4h: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    PWH/PWL (Previous Week High/Low) para cada vela 4h.
    """
    weekly_high = df_4h['high'].resample('W').max().shift(1)
    weekly_low = df_4h['low'].resample('W').min().shift(1)
    pwh = weekly_high.reindex(df_4h.index, method='ffill')
    pwl = weekly_low.reindex(df_4h.index, method='ffill')
    return pwh, pwl


def get_prev_day_levels_live(df_4h: pd.DataFrame) -> Tuple[float, float]:
    """PDH/PDL al momento actual (ultima vela completa)."""
    today = df_4h.index[-1].normalize()
    prev_day = df_4h[df_4h.index.normalize() < today]
    if len(prev_day) == 0:
        return float(df_4h['high'].max()), float(df_4h['low'].min())
    prev_day_data = prev_day.tail(6)  # ultimo dia completo (~6 velas 4h)
    return float(prev_day_data['high'].max()), float(prev_day_data['low'].min())


def get_prev_week_levels_live(df_4h: pd.DataFrame) -> Tuple[float, float]:
    """PWH/PWL al momento actual."""
    cur_week_start = df_4h.index[-1].to_period('W').start_time.tz_localize('UTC') \
        if df_4h.index[-1].tzinfo else df_4h.index[-1].to_period('W').start_time
    prev_week = df_4h[df_4h.index < cur_week_start]
    if len(prev_week) < 6:
        return float(df_4h['high'].max()), float(df_4h['low'].min())
    prev_week_data = prev_week.tail(42)  # ultima semana completa (~42 velas 4h)
    return float(prev_week_data['high'].max()), float(prev_week_data['low'].min())


# =============================================================================
# ESTRUCTURA DE MERCADO (Higher Highs/Lows)
# =============================================================================

def compute_market_structure(df_4h: pd.DataFrame, swing_window: int = 5) -> pd.Series:
    """
    Clasifica la estructura de mercado para cada vela:
      +1 = BULL (Higher Highs + Higher Lows)
      -1 = BEAR (Lower Highs + Lower Lows)
       0 = NEUTRAL / transicion

    Esta es la "fotografia" de donde esta el mercado en cada momento.
    Un trader siempre sabe si esta operando a favor o en contra de la estructura.
    """
    h = df_4h['high']
    l = df_4h['low']
    n = len(df_4h)
    structure = np.zeros(n)

    # Encontrar swing highs y lows
    swing_h = np.full(n, np.nan)
    swing_l = np.full(n, np.nan)

    for i in range(swing_window, n - swing_window):
        window_h = h.iloc[i-swing_window:i+swing_window+1]
        if h.iloc[i] == window_h.max():
            swing_h[i] = h.iloc[i]

        window_l = l.iloc[i-swing_window:i+swing_window+1]
        if l.iloc[i] == window_l.min():
            swing_l[i] = l.iloc[i]

    # Clasificar estructura usando los ultimos swings
    last_struct = 0
    sh_vals = []  # swing highs recientes (valores)
    sl_vals = []  # swing lows recientes (valores)

    for i in range(n):
        if not np.isnan(swing_h[i]):
            sh_vals.append(swing_h[i])
            if len(sh_vals) > 3:
                sh_vals.pop(0)
        if not np.isnan(swing_l[i]):
            sl_vals.append(swing_l[i])
            if len(sl_vals) > 3:
                sl_vals.pop(0)

        if len(sh_vals) >= 2 and len(sl_vals) >= 2:
            hh = sh_vals[-1] > sh_vals[-2]  # Higher High
            hl = sl_vals[-1] > sl_vals[-2]  # Higher Low
            lh = sh_vals[-1] < sh_vals[-2]  # Lower High
            ll = sl_vals[-1] < sl_vals[-2]  # Lower Low

            if hh and hl:
                last_struct = 1   # BULL
            elif lh and ll:
                last_struct = -1  # BEAR
            # Si mixto, mantener estructura anterior (mercado en transicion)

        structure[i] = last_struct

    return pd.Series(structure, index=df_4h.index, dtype=float)


def get_structure_live(df_4h: pd.DataFrame, swing_window: int = 5) -> str:
    """Retorna estructura actual como string para el bot."""
    s = compute_market_structure(df_4h, swing_window)
    val = s.iloc[-1]
    if val > 0:
        return 'BULL'
    elif val < 0:
        return 'BEAR'
    return 'NEUTRAL'


# =============================================================================
# FEATURES PARA TRAINING (vectorizado)
# =============================================================================

def add_structure_features(df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todas las features de estructura para el training pipeline.

    Estas features dan contexto al modelo de DONDE esta el precio
    en relacion a los niveles clave — exactamente lo que revisa un trader
    antes de decidir entrar.

    Returns:
        DataFrame con columnas de estructura (mismo index que df_4h)
    """
    c = df_4h['close']

    # ATR para normalizar distancias
    from pandas_ta import atr as pta_atr
    _atr = pta_atr(df_4h['high'], df_4h['low'], df_4h['close'], length=14)
    atr = _atr.clip(lower=1e-10) if _atr is not None else pd.Series(c * 0.02, index=c.index)

    feat = pd.DataFrame(index=df_4h.index)

    # --- FVG features ---
    fvg = compute_fvg_features(df_4h)
    feat['at_fvg_bull'] = fvg['at_fvg_bull'].astype(float)
    feat['at_fvg_bear'] = fvg['at_fvg_bear'].astype(float)

    # --- Previous Day High/Low ---
    pdh, pdl = compute_prev_day_levels(df_4h)
    feat['dist_pdh_pct'] = (c - pdh) / c * 100   # negativo = debajo del PDH
    feat['dist_pdl_pct'] = (c - pdl) / c * 100   # positivo = encima del PDL

    # Que tan "cerca" esta del PDH/PDL (en multiples de ATR)
    feat['near_pdh'] = ((pdh - c).abs() / atr < ZONE_TOLERANCE).astype(float)
    feat['near_pdl'] = ((pdl - c).abs() / atr < ZONE_TOLERANCE).astype(float)

    # --- Previous Week High/Low ---
    pwh, pwl = compute_prev_week_levels(df_4h)
    feat['dist_pwh_pct'] = (c - pwh) / c * 100
    feat['dist_pwl_pct'] = (c - pwl) / c * 100
    feat['near_pwh'] = ((pwh - c).abs() / atr < ZONE_TOLERANCE * 1.5).astype(float)
    feat['near_pwl'] = ((pwl - c).abs() / atr < ZONE_TOLERANCE * 1.5).astype(float)

    # --- Estructura de mercado ---
    feat['weekly_struct'] = compute_market_structure(df_4h)  # -1/0/+1

    # --- En algun nivel clave ---
    feat['at_key_level'] = (
        (feat['near_pdh'] > 0) | (feat['near_pdl'] > 0) |
        (feat['near_pwh'] > 0) | (feat['near_pwl'] > 0) |
        (feat['at_fvg_bull'] > 0) | (feat['at_fvg_bear'] > 0)
    ).astype(float)

    return feat.replace([np.inf, -np.inf], np.nan)


# Features de estructura a incluir en el modelo
STRUCTURE_FEATURES = [
    'at_fvg_bull', 'at_fvg_bear',
    'dist_pdh_pct', 'dist_pdl_pct',
    'dist_pwh_pct', 'dist_pwl_pct',
    'near_pdh', 'near_pdl',
    'weekly_struct', 'at_key_level',
]


# =============================================================================
# PLAN DIARIO (para el bot en vivo)
# =============================================================================

def build_daily_plan(df_4h: pd.DataFrame) -> DailyPlan:
    """
    Construye el plan del trader para el momento actual.

    Identifica:
    1. Estructura macro (bull/bear/neutral)
    2. Todos los niveles relevantes (FVG, PDH/PDL, PWH/PWL)
    3. Zonas de interes ordenadas por distancia al precio actual

    El bot solo ejecuta cuando el precio llega a una de estas zonas.
    """
    import pandas_ta as pta
    cur_price = float(df_4h['close'].iloc[-1])
    atr_series = pta.atr(df_4h['high'], df_4h['low'], df_4h['close'], length=14)
    cur_atr = float(atr_series.iloc[-1]) if atr_series is not None else cur_price * 0.02

    # Estructura
    structure = get_structure_live(df_4h)

    # Bias basado en estructura
    if structure == 'BULL':
        bias = 'BULL_BIAS'
    elif structure == 'BEAR':
        bias = 'BEAR_BIAS'
    else:
        bias = 'NEUTRAL'

    # Zonas
    zones = []

    # 1. FVGs activos
    fvg_zones = get_active_fvgs_live(df_4h)
    zones.extend(fvg_zones)

    # 2. PDH/PDL
    pdh, pdl = get_prev_day_levels_live(df_4h)
    if pdh > 0:
        zones.append(Zone(
            level=pdh, top=pdh * 1.001, bottom=pdh * 0.999,
            direction='SHORT', zone_type='PDH', strength=2, age_candles=1,
            description=f'PDH {pdh:.0f}',
        ))
    if pdl > 0:
        zones.append(Zone(
            level=pdl, top=pdl * 1.001, bottom=pdl * 0.999,
            direction='LONG', zone_type='PDL', strength=2, age_candles=1,
            description=f'PDL {pdl:.0f}',
        ))

    # 3. PWH/PWL
    pwh, pwl = get_prev_week_levels_live(df_4h)
    if pwh > 0 and abs(pwh - pdh) / cur_price > 0.005:  # no duplicar si estan muy cerca
        zones.append(Zone(
            level=pwh, top=pwh * 1.002, bottom=pwh * 0.998,
            direction='SHORT', zone_type='PWH', strength=3, age_candles=1,
            description=f'PWH {pwh:.0f}',
        ))
    if pwl > 0 and abs(pwl - pdl) / cur_price > 0.005:
        zones.append(Zone(
            level=pwl, top=pwl * 1.002, bottom=pwl * 0.998,
            direction='LONG', zone_type='PWL', strength=3, age_candles=1,
            description=f'PWL {pwl:.0f}',
        ))

    # Aumentar strength si la zona coincide con la estructura
    for z in zones:
        if structure == 'BULL' and z.direction == 'LONG':
            z.strength = min(3, z.strength + 1)
        elif structure == 'BEAR' and z.direction == 'SHORT':
            z.strength = min(3, z.strength + 1)

    # Ordenar por distancia al precio actual
    zones.sort(key=lambda z: abs(z.level - cur_price))

    return DailyPlan(
        structure=structure,
        bias=bias,
        zones=zones,
        pdh=pdh,
        pdl=pdl,
        pwh=pwh,
        pwl=pwl,
        current_price=cur_price,
        atr=cur_atr,
    )


def get_nearest_zone(
    price: float,
    zones: List[Zone],
    atr: float,
    tolerance: float = ZONE_TOLERANCE,
    direction_filter: str = None,
) -> Optional[Zone]:
    """
    Retorna la zona mas cercana al precio actual si esta dentro de la tolerancia.

    Args:
        price: precio actual
        zones: lista de zonas del DailyPlan
        atr: ATR actual en precio absoluto
        tolerance: cuantos ATRs de distancia para "estar en zona"
        direction_filter: si se especifica ('LONG'/'SHORT'), solo zonas de esa direccion

    Returns:
        Zone o None si no hay ninguna zona cerca
    """
    best = None
    best_dist = float('inf')

    for z in zones:
        if direction_filter and z.direction != direction_filter:
            continue
        # Precio dentro del rango de la zona
        if z.bottom <= price <= z.top:
            dist = 0.0
        else:
            dist = min(abs(price - z.top), abs(price - z.bottom))

        # Distancia en multiples de ATR
        dist_atr = dist / max(atr, 1e-10)
        if dist_atr <= tolerance and dist_atr < best_dist:
            best_dist = dist_atr
            best = z

    return best


def describe_plan(plan: DailyPlan) -> str:
    """Genera un resumen legible del plan del dia."""
    lines = [
        f'Estructura: {plan.structure} | Bias: {plan.bias}',
        f'PDH: {plan.pdh:.0f} | PDL: {plan.pdl:.0f}',
        f'PWH: {plan.pwh:.0f} | PWL: {plan.pwl:.0f}',
        f'Zonas activas ({len(plan.zones)}):',
    ]
    for z in plan.zones[:6]:  # mostrar las 6 mas cercanas
        dist_pct = abs(z.level - plan.current_price) / plan.current_price * 100
        lines.append(
            f'  [{z.direction}] {z.zone_type} @ {z.level:.0f} '
            f'(dist: {dist_pct:.1f}%) strength={z.strength} | {z.description}'
        )
    return '\n'.join(lines)


# =============================================================================
# ADAPTACION: MERCADO CAOTICO, REPLAN FORZADO, ZONAS ROTAS
# =============================================================================

def is_market_chaotic(df_4h: pd.DataFrame, atr_multiplier: float = 1.8) -> tuple:
    """
    Detecta si el mercado esta en modo caotico (volatilidad anormal).

    Criterios:
      1. ATR actual > atr_multiplier x ATR_20 promedio
      2. Cuerpo de la ultima vela > 2 x ATR (vela de impulso extremo)

    Returns:
        (is_chaotic: bool, reason: str)

    Cuando es caotico el bot debe esperar — no hay zonas fiables
    porque el precio se mueve demasiado rapido para respetar niveles.
    """
    import pandas_ta as pta

    atr = pta.atr(df_4h['high'], df_4h['low'], df_4h['close'], length=14)
    if atr is None or len(atr.dropna()) < 20:
        return False, ''

    cur_atr = float(atr.iloc[-1])
    avg_atr = float(atr.rolling(20).mean().iloc[-1])

    if avg_atr < 1e-10:
        return False, ''

    atr_ratio = cur_atr / avg_atr
    if atr_ratio >= atr_multiplier:
        return True, f'ATR expandido {atr_ratio:.1f}x el promedio (umbral={atr_multiplier}x)'

    # Vela con cuerpo anormalmente grande (impulso sin control)
    last_body = abs(float(df_4h['close'].iloc[-1]) - float(df_4h['open'].iloc[-1]))
    if last_body > 2.0 * cur_atr:
        return True, f'Vela de impulso extremo: cuerpo={last_body:.0f} > 2x ATR={cur_atr:.0f}'

    return False, ''


def should_force_replan(df_4h: pd.DataFrame, plan: DailyPlan) -> tuple:
    """
    Decide si hay que invalidar el plan cacheado y reconstruirlo ya.

    Casos que fuerzan replan:
      1. Mercado caotico (ATR expandido o vela extrema)
      2. Precio rompio PWH o PWL del plan (estructura semanal invalidada)
      3. Precio se movio mas de 3x ATR desde que se construyo el plan

    Returns:
        (force: bool, reason: str)
    """
    chaotic, reason = is_market_chaotic(df_4h)
    if chaotic:
        return True, f'Mercado caotico: {reason}'

    cur_price = float(df_4h['close'].iloc[-1])

    # Precio rompio los niveles semanales del plan (la semana cambio)
    if plan.pwh > 0 and cur_price > plan.pwh * 1.005:
        return True, f'Precio {cur_price:,.0f} rompio PWH={plan.pwh:,.0f} del plan'
    if plan.pwl > 0 and cur_price < plan.pwl * 0.995:
        return True, f'Precio {cur_price:,.0f} rompio PWL={plan.pwl:,.0f} del plan'

    # Precio se alejo demasiado del precio cuando se construyo el plan
    if plan.current_price > 0 and plan.atr > 0:
        drift = abs(cur_price - plan.current_price) / plan.atr
        if drift > 3.0:
            return True, f'Precio se movio {drift:.1f}x ATR desde el ultimo plan'

    return False, ''


def clean_broken_zones(price: float, zones: List[Zone], atr: float) -> List[Zone]:
    """
    Elimina zonas que el precio ya atraveso claramente.

    Logica:
      - Zona LONG (soporte): invalida si precio bajo mas de 0.5 ATR por debajo del fondo
      - Zona SHORT (resistencia): invalida si precio subio mas de 0.5 ATR por encima del techo
      - Zona NEUTRAL: invalida si precio paso mas de 1 ATR mas alla del nivel central

    Un trader real descarta las zonas rotas — si el precio las atraveso sin rebotar,
    ya no son zonas validas, son simplemente niveles superados.
    """
    valid = []
    margin = atr * 0.5

    for z in zones:
        if z.direction == 'LONG':
            if price < z.bottom - margin:
                continue  # Soporte roto: precio debajo del fondo de la zona
        elif z.direction == 'SHORT':
            if price > z.top + margin:
                continue  # Resistencia rota: precio encima del techo de la zona
        else:  # NEUTRAL (niveles PDH/PDL usados en ambas direcciones)
            if price > z.top + margin or price < z.bottom - margin:
                continue  # Nivel claramente superado
        valid.append(z)

    return valid
