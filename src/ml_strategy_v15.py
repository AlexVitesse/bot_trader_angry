"""
Estrategia en vivo: motor V2 (src/v2_engine.py) por par.

El motor de cada par lo fija `ML_V15_ENGINE` en config/settings.py. Hoy solo
existe 'v2' (A: Donchian-55 + filtro EMA diario + trailing ATR, LONG-only;
F: ruptura tras compresion de volatilidad). La logica V15 (GBM SHORT, reglas
ETH/alts) se borro en la limpieza de 2026-09: estaba muerta desde que BTC
enruta a V2 y sus metas declaraban metricas de un simulador con look-ahead
(AUDITORIA_2026-09 §4-5). Sigue en el historial de git.

`update_regime` (EMA20/50 diario) queda solo para el log y el leverage por
regimen; el filtro de regimen que decide las entradas vive dentro de V2.
"""

import logging
import time
import numpy as np
import pandas as pd
import requests
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src import v2_engine as _v2_engine

logger = logging.getLogger(__name__)

FAPI_BASE = 'https://fapi.binance.com'
# V2 engine necesita >= min_warmup_bars(220) + 2 velas DESPUES del dropna de
# build_features (que recorta ~55 filas por la Donchian-55). Con 250 quedaban
# 195 < 222 y get_live_signal devolvia None SIEMPRE. 420 -> ~365 utiles.
V2_LOOKBACK = 420


def _ohlcv(exchange, pair: str, timeframe: str, limit: int,
           tries: int = 3, delay: int = 5):
    """`fetch_ohlcv` con reintentos.

    La red del VPS parpadea de madrugada: en el log del 19-22 ago, 6 de 21
    velas 4h quedaron ciegas y el bot registro "Sin senales en este ciclo".
    Un fallo de fetch NO es "mercado quieto" — es el motor sin datos, y una
    senal perdida ahi no se recupera. ccxt tarda ~20s en agotar su timeout,
    asi que el peor caso son ~70s por llamada: aceptable en un ciclo de 4h.
    """
    for intento in range(tries):
        try:
            return exchange.fetch_ohlcv(pair, timeframe, limit=limit)
        except Exception as e:
            if intento == tries - 1:
                raise
            logger.warning(f'[NET] {pair} {timeframe} intento '
                           f'{intento + 1}/{tries} fallo ({e}); '
                           f'reintento en {delay}s')
            time.sleep(delay)


def _funding_df(pair: str) -> Optional[pd.DataFrame]:
    """Historial de funding para el veto de V2 (el backtest lo aplicaba; en
    vivo iba None -> funding_z=0). La API devuelve hasta 500 registros
    (~166 dias), de sobra para el z-score de 168 velas 4h (28 dias). Sin red
    -> None, igual que antes."""
    try:
        resp = requests.get(f'{FAPI_BASE}/fapi/v1/fundingRate',
                            params={'symbol': pair.replace('/', ''),
                                    'limit': 1000}, timeout=10)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json())
        df.index = pd.to_datetime(df['fundingTime'].astype('int64'), unit='ms',
                                  utc=True)
        return df[['fundingRate']].astype(float).rename(
            columns={'fundingRate': 'funding_rate'}).sort_index()
    except Exception as e:
        logger.warning(f'[V2] {pair}: funding no disponible ({e}), veto apagado')
        return None


@dataclass
class PairState:
    """Per-pair cached state updated by update_regime()."""
    regime: str = 'RANGE'
    regime_updated: Optional[datetime] = None
    funding_zscore: float = 0.0
    daily_ema20: Optional[float] = None
    daily_ema50: Optional[float] = None
    daily_ema200: Optional[float] = None


class MLStrategyV15:
    """Genera senales V2 por par y mantiene el regimen diario para el log."""

    def __init__(self):
        from config.settings import ML_V15_PAIRS, ML_V15_SIZING, ML_V15_ENGINE
        self.pairs = list(ML_V15_PAIRS)
        self._sizing = dict(ML_V15_SIZING)
        self._engine = dict(ML_V15_ENGINE)
        self._pair_state = {pair: PairState() for pair in self.pairs}

    # Backward compat: self.regime = BTC regime
    @property
    def regime(self):
        return self._pair_state.get('BTC/USDT', PairState()).regime

    @regime.setter
    def regime(self, value):
        if 'BTC/USDT' in self._pair_state:
            self._pair_state['BTC/USDT'].regime = value

    @property
    def regime_updated(self):
        return self._pair_state.get('BTC/USDT', PairState()).regime_updated

    def get_regime(self, pair: str) -> str:
        """Get regime for a specific pair."""
        return self._pair_state.get(pair, PairState()).regime

    def get_regimes_str(self) -> str:
        """Format all pair regimes for display."""
        parts = []
        for pair in self.pairs:
            coin = pair.split('/')[0]
            regime = self._pair_state[pair].regime
            parts.append(f"{coin}:{regime}")
        return ' | '.join(parts)

    # =================================================================
    # ENGINE ROUTING
    # =================================================================
    def load_models(self) -> int:
        """Pares con motor conocido. Un par sin motor en ML_V15_ENGINE no
        opera (antes caia en silencio a la rama ML si faltaba un JSON)."""
        loaded = 0
        for pair in self.pairs:
            engine = self._engine.get(pair)
            if engine == 'v2':
                logger.info(f'[V15] {pair}: motor V2 (reglas congeladas), sin ML')
                loaded += 1
            else:
                logger.error(f'[V15] {pair}: motor desconocido {engine!r} en '
                             f'ML_V15_ENGINE, el par NO opera')
        return loaded

    # =================================================================
    # REGIME DETECTION (called daily by bot)
    # =================================================================
    def update_regime(self, exchange) -> bool:
        """Update macro regime for all pairs + funding rates.

        Devuelve False si el regimen de BTC no se pudo refrescar. El caller NO
        debe marcar el dia como actualizado en ese caso: si lo hace, un fallo
        de red deja el regimen congelado 24h y el motor decide con datos
        viejos (paso justo eso durante el apagon de julio).
        """
        antes = self._pair_state.get('BTC/USDT', PairState()).regime_updated
        # Always fetch BTC daily (needed for BTC regime + ETH follower)
        btc_state = self._update_pair_regime(exchange, 'BTC/USDT')

        # Update other pairs
        for pair in self.pairs:
            if pair == 'BTC/USDT':
                continue
            self._update_pair_regime(exchange, pair)

        # Log all regimes
        logger.info(f'[V15] Regimes: {self.get_regimes_str()}')
        ok = btc_state.regime_updated is not None and btc_state.regime_updated != antes
        if not ok:
            logger.error('[V15] Regimen de BTC NO actualizado — se reintentara '
                         'en la siguiente vela, no se marca el dia como hecho')
        return ok

    def _update_pair_regime(self, exchange, pair: str) -> PairState:
        """Update regime for a single pair."""
        state = self._pair_state.get(pair)
        if state is None:
            state = PairState()
            self._pair_state[pair] = state

        try:
            # Fetch DAILY candles — need 250 days for EMA200
            ohlcv_1d = _ohlcv(exchange, pair, '1d', 250)
            if not ohlcv_1d or len(ohlcv_1d) < 55:
                logger.warning(f'[V15] {pair}: insufficient daily data')
                return state

            df_1d = pd.DataFrame(ohlcv_1d, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            df_1d['timestamp'] = pd.to_datetime(df_1d['ts'], unit='ms', utc=True)
            df_1d = df_1d.set_index('timestamp').sort_index()
            daily_close = df_1d['close'].iloc[:-1]  # exclude today

            ema20 = daily_close.ewm(span=20, adjust=False).mean()
            ema50 = daily_close.ewm(span=50, adjust=False).mean()
            ema200 = daily_close.ewm(span=200, adjust=False).mean() if len(daily_close) >= 200 else None

            state.daily_ema20 = float(ema20.iloc[-1])
            state.daily_ema50 = float(ema50.iloc[-1])
            state.daily_ema200 = float(ema200.iloc[-1]) if ema200 is not None else None

            # Current price from latest 4h candle
            ohlcv_4h = _ohlcv(exchange, pair, '4h', 3)
            cur_close = float(ohlcv_4h[-2][4]) if ohlcv_4h and len(ohlcv_4h) >= 2 else float(daily_close.iloc[-1])

            state.regime = self._classify_regime(
                state.daily_ema20, state.daily_ema50, state.daily_ema200,
                cur_close
            )
            state.regime_updated = datetime.now(timezone.utc)

            # Fetch funding rate z-score
            symbol = pair.replace('/', '').replace('USDT', 'USDT')
            state.funding_zscore = self._fetch_funding_zscore(symbol)

            coin = pair.split('/')[0]
            logger.info(
                f'[V15] {coin}: {state.regime} | '
                f'EMA20={state.daily_ema20:,.0f} EMA50={state.daily_ema50:,.0f} | '
                f'funding_z={state.funding_zscore:.2f}'
            )
        except Exception as e:
            logger.error(f'[V15] Error updating regime for {pair}: {e}')

        return state

    def _classify_regime(self, ema20, ema50, ema200, close, dead_zone=0.02):
        """Classify regime: BULL / BEAR / RANGE. Identical to backtest."""
        dist = (ema20 - ema50) / ema50

        if dist > dead_zone:
            return 'BULL'
        elif dist < -dead_zone:
            if ema200 is not None and close > ema200:
                return 'RANGE'
            if close > ema50:
                return 'RANGE'
            return 'BEAR'
        return 'RANGE'

    def _fetch_funding_zscore(self, symbol: str = 'BTCUSDT') -> float:
        """Fetch funding rate and compute 90-day z-score."""
        try:
            resp = requests.get(
                f'{FAPI_BASE}/fapi/v1/fundingRate',
                params={'symbol': symbol, 'limit': 100},
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            if not data or len(data) < 10:
                return 0.0
            rates = [float(d['fundingRate']) for d in data]
            current = rates[-1]
            mean = np.mean(rates)
            std = np.std(rates)
            if std < 1e-8:
                return 0.0
            return (current - mean) / std
        except Exception as e:
            logger.debug(f'[V15] Funding fetch error ({symbol}): {e}')
            return 0.0

    # =================================================================
    # SIGNAL GENERATION (called every 4h candle)
    # =================================================================
    def generate_signals(self, exchange, open_pairs=None) -> list:
        """Senales de todos los pares con motor V2 (formato de open_position)."""
        open_pairs = open_pairs or set()
        all_signals = []
        for pair in self.pairs:
            if pair in open_pairs:
                logger.info(f'[V15] {pair.split("/")[0]}: already open, skipping')
                continue
            if self._engine.get(pair) == 'v2':
                all_signals.extend(self._generate_v2_signal(pair, exchange))
        return all_signals

    def _generate_v2_signal(self, pair: str, exchange) -> list:
        """Senal V2 para un par: una posicion a la vez, trailing sin
        look-ahead intrabar."""
        try:
            # 4h para las senales; 1d aparte porque derivar el diario desde
            # 420 velas 4h no da historia para la EMA200 del filtro de regimen.
            ohlcv_4h = _ohlcv(exchange, pair, '4h', V2_LOOKBACK)
            if not ohlcv_4h or len(ohlcv_4h) < 100:
                logger.warning(f'[V2] {pair}: insufficient 4h data')
                return []
            df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high',
                                                    'low', 'close', 'volume'])
            df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], unit='ms', utc=True)
            df_4h = df_4h.set_index('timestamp').sort_index()

            # 1000 velas diarias: EMA50/200 con adjust=False arrastran el valor
            # inicial; con 300 velas bull_1d difería del backtest 29 dias en
            # 2019-2026 (justo en los cruces), con 1000 en 0. AUDITORIA_2026-09 §2.2
            df_1d = None
            try:
                ohlcv_1d = _ohlcv(exchange, pair, '1d', 1000)
                if ohlcv_1d and len(ohlcv_1d) >= 600:
                    df_1d = pd.DataFrame(ohlcv_1d, columns=['timestamp', 'open',
                                                            'high', 'low',
                                                            'close', 'volume'])
                    df_1d['timestamp'] = pd.to_datetime(df_1d['timestamp'],
                                                        unit='ms', utc=True)
                    df_1d = df_1d.set_index('timestamp').sort_index()
                else:
                    logger.warning(f'[V2] {pair}: daily data insufficient '
                                   f'({len(ohlcv_1d) if ohlcv_1d else 0} bars), '
                                   f'fallback a derivacion desde 4h')
            except Exception as e:
                logger.warning(f'[V2] {pair}: fetch daily fallo ({e}), '
                               f'fallback a derivacion desde 4h')
            # Llamar al engine V2: devuelve None o dict con side, trail_dist, etc.
            sig = _v2_engine.get_live_signal(df_4h, df_1d=df_1d,
                                             df_funding=_funding_df(pair))
            if sig is None:
                logger.info(f'[V2] {pair}: no signal')
                return []
            sizing_mult = self._sizing.get(pair, 0.3)
            # Convertir trail_dist a tp/sl pct para V14-compat:
            # Como es trailing, usamos sl = trail_dist y tp = trail_dist*2 (heuristic)
            # El portfolio_manager con trail_mode='tight' usa trail_dist directamente.
            # OJO: el contrato que consume ml_bot._execute_v14_signal es
            # direction INT (1/-1) + 'price' — igual que _build_signal(). Con
            # direction='LONG' (str) `direction == 1` era False y un LONG se
            # habria abierto como SHORT; sin 'price' reventaba con KeyError.
            signal_payload = {
                'pair': pair,
                'direction': 1 if sig['side'] == 'LONG' else -1,
                'side': sig['side'],
                'price': sig['entry_price'],
                'tp_pct': sig['trail_dist'] * 2.0,
                'sl_pct': sig['trail_dist'],
                'setup': f"v2_{sig['sig_type']}",
                'confidence': 1.0,
                'sizing_mult': sizing_mult,
                'trail_mode': 'tight',
                'trail_fixed_dist': sig['trail_dist'],
                'max_bars': sig['max_bars'],
                'regime': sig.get('regime', 'UNK'),
                'engine': 'v2_honest',
            }
            logger.info(f"[V2] {pair} {sig['sig_type']} {sig['side']} "
                        f"trail={sig['trail_dist']:.3f} max_bars={sig['max_bars']}")
            return [signal_payload]
        except Exception as e:
            logger.error(f'[V2] {pair} error: {e}')
            return []
