"""
ML Strategy V15 - Expert Committee Multi-Pair
==============================================
BTC/USDT: WF 8/12, OOS PF=1.35, CAGR ~37%.
  BULL  -> Breakout B LONG + Pullback EMA20 LONG (reglas, ATR TP/SL)
  BEAR  -> SHORT ML (GBM threshold=0.60, entrenado en BEAR)
  RANGE -> Breakout B LONG only

ETH/USDT: WF 8/12, PF=1.28, OOS 2026 +16.2% (100% rule-based).
  BULL/RANGE -> BTC-follower LONG (corr>=0.5) + Breakout ETH standalone
  BEAR       -> SHORT Multi-conf (RSI>60+BB>0.75+bear+vol) + BB upper

Gates:
  - Funding veto: z-score > 2 bloquea LONG, < -1.5 bloquea SHORT
  - Regime: EMA20/50 diario per-pair, 2% dead zone + recovery filter

Senales compatibles con V14 (pair/direction int/tp_pct/sl_pct/setup/confidence).
"""

import json
import logging
import numpy as np
import pandas as pd
import pandas_ta as pta
import joblib
import requests
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# V2 paper-trade engine (5 pares con model_type='v2_honest') — ver
# docs/PLAN_PAPER_3MESES.md. Importacion opcional: si v2_engine no existe,
# el bot sigue funcionando con la logica legacy V15.
try:
    from src import v2_engine as _v2_engine
    V2_AVAILABLE = True
except Exception:
    try:
        import v2_engine as _v2_engine
        V2_AVAILABLE = True
    except Exception:
        _v2_engine = None
        V2_AVAILABLE = False

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
FAPI_BASE = 'https://fapi.binance.com'
LOOKBACK = 250  # candles to fetch (V14-compat path)
# V2 engine necesita >= min_warmup_bars(220) + 2 velas DESPUES del dropna de
# build_features (que recorta ~55 filas por la Donchian-55). Con 250 quedaban
# 195 < 222 y get_live_signal devolvia None SIEMPRE. 420 -> ~365 utiles.
V2_LOOKBACK = 420


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
    """V15 Expert Committee — multi-pair, 3 regimenes per pair."""

    def __init__(self):
        # Import pairs from settings (with fallback)
        try:
            from config.settings import ML_V15_PAIRS, ML_V15_SIZING
            self.pairs = list(ML_V15_PAIRS)
            self._sizing = dict(ML_V15_SIZING)
        except ImportError:
            self.pairs = ['BTC/USDT']
            self._sizing = {'BTC/USDT': 1.0}

        # Per-pair state
        self._pair_state = {pair: PairState() for pair in self.pairs}

        # BTC ML SHORT model (only BTC uses ML)
        self.short_model = None
        self.short_scaler = None

        # Per-pair meta config
        self._meta = {}  # pair -> dict

        # Load models at init
        self.load_models()

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

    @property
    def meta(self):
        """Backward compat: returns BTC meta."""
        return self._meta.get('BTC/USDT', {})

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
    # MODEL LOADING
    # =================================================================
    def load_models(self) -> int:
        """Load models and meta configs per pair. Returns count of loaded pairs."""
        loaded = 0

        for pair in self.pairs:
            coin = pair.split('/')[0].lower()
            model_dir = PROJECT_ROOT / 'strategies' / f'{coin}_v15' / 'models'

            if not model_dir.exists():
                logger.warning(f'[V15] Model dir not found for {pair}: {model_dir}')
                continue

            # Load meta config
            meta_path = model_dir / 'meta_v15.json'
            if meta_path.exists():
                with open(meta_path) as f:
                    self._meta[pair] = json.load(f)
            else:
                self._meta[pair] = {}

            # Load ML models (only BTC has ML)
            has_ml = self._meta[pair].get('has_ml', True)  # default True for BTC compat
            if has_ml and pair == 'BTC/USDT':
                try:
                    self.short_model = joblib.load(model_dir / 'short_gbm.pkl')
                    self.short_scaler = joblib.load(model_dir / 'short_scaler.pkl')
                    logger.info(
                        f'[V15] BTC: SHORT GBM loaded | threshold='
                        f'{self._meta[pair].get("short_threshold", 0.60)}'
                    )
                except Exception as e:
                    logger.error(f'[V15] Error loading BTC ML models: {e}')
                    continue
            else:
                logger.info(f'[V15] {pair}: rule-based (no ML)')

            loaded += 1

        if loaded > 0:
            logger.info(f'[V15] {loaded} pair(s) configured: {", ".join(self.pairs[:loaded])}')
        return loaded

    # =================================================================
    # REGIME DETECTION (called daily by bot)
    # =================================================================
    def update_regime(self, exchange):
        """Update macro regime for all pairs + funding rates."""
        # Always fetch BTC daily (needed for BTC regime + ETH follower)
        btc_state = self._update_pair_regime(exchange, 'BTC/USDT')

        # Update other pairs
        for pair in self.pairs:
            if pair == 'BTC/USDT':
                continue
            self._update_pair_regime(exchange, pair)

        # Log all regimes
        logger.info(f'[V15] Regimes: {self.get_regimes_str()}')

    def _update_pair_regime(self, exchange, pair: str) -> PairState:
        """Update regime for a single pair."""
        state = self._pair_state.get(pair)
        if state is None:
            state = PairState()
            self._pair_state[pair] = state

        meta = self._meta.get(pair, {})

        try:
            # Fetch DAILY candles — need 250 days for EMA200
            ohlcv_1d = exchange.fetch_ohlcv(pair, '1d', limit=250)
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
            ohlcv_4h = exchange.fetch_ohlcv(pair, '4h', limit=3)
            cur_close = float(ohlcv_4h[-2][4]) if ohlcv_4h and len(ohlcv_4h) >= 2 else float(daily_close.iloc[-1])

            state.regime = self._classify_regime(
                state.daily_ema20, state.daily_ema50, state.daily_ema200,
                cur_close, meta.get('regime_dead_zone', 0.02)
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
        """Generate signals for all configured pairs. V14-compatible format."""
        if open_pairs is None:
            open_pairs = set()

        all_signals = []

        # Always fetch BTC 4H data (needed for BTC signals + ETH follower)
        df_btc = self._fetch_and_compute(exchange, 'BTC/USDT')

        for pair in self.pairs:
            if pair in open_pairs:
                coin = pair.split('/')[0]
                logger.info(f'[V15] {coin}: already open, skipping')
                continue

            try:
                # ============================================================
                # V2 paper-trade routing — si el par tiene meta_v2_paper.json,
                # usar v2_engine (motor honesto unificado A+F). Esto cubre los
                # 5 pares de paper trading 3 meses (BTC, BNB, DOGE, ETH, OP).
                # ============================================================
                v2_meta_path = (PROJECT_ROOT / 'strategies' /
                                f'{pair.split("/")[0].lower()}_v15' /
                                'models' / 'meta_v2_paper.json')
                if V2_AVAILABLE and v2_meta_path.exists():
                    signals = self._generate_v2_signal(pair, exchange, df_btc)
                    all_signals.extend(signals)
                    continue

                if pair == 'BTC/USDT':
                    signals = self._generate_btc_signals(df_btc)
                elif pair == 'ETH/USDT':
                    df_eth = self._fetch_and_compute(exchange, 'ETH/USDT')
                    if df_eth is not None:
                        signals = self._generate_eth_signals(df_eth, df_btc)
                    else:
                        signals = []
                elif pair in ('ADA/USDT', 'SOL/USDT', 'DOGE/USDT',
                                  'LINK/USDT', 'AVAX/USDT', 'DOT/USDT',
                                  'NEAR/USDT', 'XRP/USDT', 'ATOM/USDT',
                                  'INJ/USDT', 'ALGO/USDT', 'FIL/USDT',
                                  '1000SHIB/USDT', 'BNB/USDT',
                                  'LTC/USDT', 'ETC/USDT', 'BCH/USDT',
                                  'UNI/USDT', 'AAVE/USDT', 'OP/USDT'):
                    df_alt = self._fetch_and_compute(exchange, pair)
                    if df_alt is not None:
                        signals = self._generate_alt_trailing_signals(pair, df_alt, df_btc)
                    else:
                        signals = []
                else:
                    signals = []

                all_signals.extend(signals)
            except Exception as e:
                logger.error(f'[V15] Error generating signals for {pair}: {e}')

        return all_signals

    def _generate_v2_signal(self, pair: str, exchange, df_btc) -> list:
        """
        V2 paper-trade signal generation usando v2_engine (motor honesto).
        Aplica a los 5 pares con meta_v2_paper.json (BTC, BNB, DOGE, ETH, OP).
        Una posicion a la vez por par. TP/SL trailing sin look-ahead intrabar.
        """
        if not V2_AVAILABLE:
            return []
        try:
            # Fetch OHLCV 4h (250 velas = ~42 dias) y 1d (300 velas = ~10 meses).
            # El daily es CRITICO porque v2_engine necesita EMA200 daily para el
            # filtro de regime — derivar daily de 250 velas 4h da solo 42 dias,
            # insuficiente para EMA200 confiable. Binance provee daily historico
            # directamente sin esperar.
            ohlcv_4h = exchange.fetch_ohlcv(pair, '4h', limit=V2_LOOKBACK)
            if not ohlcv_4h or len(ohlcv_4h) < 100:
                logger.warning(f'[V2] {pair}: insufficient 4h data')
                return []
            df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high',
                                                    'low', 'close', 'volume'])
            df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], unit='ms', utc=True)
            df_4h = df_4h.set_index('timestamp').sort_index()

            # Fetch daily 300 velas (~10 meses) — suficiente para EMA200 daily
            df_1d = None
            try:
                ohlcv_1d = exchange.fetch_ohlcv(pair, '1d', limit=300)
                if ohlcv_1d and len(ohlcv_1d) >= 200:
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
            sig = _v2_engine.get_live_signal(df_4h, df_1d=df_1d, df_funding=None)
            if sig is None:
                logger.info(f'[V2] {pair}: no signal')
                return []
            sizing_mult = self._sizing.get(pair, 0.3)
            # Convertir trail_dist a tp/sl pct para V14-compat:
            # Como es trailing, usamos sl = trail_dist y tp = trail_dist*2 (heuristic)
            # El portfolio_manager con trail_mode='tight' usa trail_dist directamente.
            signal_payload = {
                'pair': pair,
                'direction': 'LONG' if sig['side'] == 'LONG' else 'SHORT',
                'side': sig['side'],
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

    def _fetch_and_compute(self, exchange, pair: str) -> Optional[pd.DataFrame]:
        """Fetch 4H OHLCV and compute features for a pair."""
        try:
            ohlcv = exchange.fetch_ohlcv(pair, '4h', limit=LOOKBACK)
            if not ohlcv or len(ohlcv) < 50:
                logger.warning(f'[V15] {pair}: insufficient OHLCV data')
                return None
            df = self._ohlcv_to_df(ohlcv)
            df = self._compute_features(df)
            if df is None or len(df) < 30:
                return None
            return df
        except Exception as e:
            logger.error(f'[V15] Error fetching {pair}: {e}')
            return None

    # =================================================================
    # BTC SIGNAL GENERATION (unchanged logic)
    # =================================================================
    def _generate_btc_signals(self, df_btc) -> list:
        """Generate BTC signals. Same logic as before."""
        if df_btc is None:
            return []

        if self.short_model is None:
            logger.warning('[V15] BTC: ML models not loaded')
            return []

        state = self._pair_state.get('BTC/USDT', PairState())
        meta = self._meta.get('BTC/USDT', {})

        # Add daily macro feature for SHORT ML
        if state.daily_ema20 and state.daily_ema50:
            df_btc['bull_1d'] = int(state.daily_ema20 > state.daily_ema50)
        else:
            df_btc['bull_1d'] = 0

        i = len(df_btc) - 1
        row = df_btc.iloc[i]
        regime = state.regime
        funding_z = state.funding_zscore

        veto_long = meta.get('funding_veto_long', 2.0)
        veto_short = meta.get('funding_veto_short', -1.5)

        trade = None

        if regime == 'BULL':
            if funding_z > veto_long:
                logger.info(f'[V15] BTC BULL: Funding veto (z={funding_z:.2f} > {veto_long})')
                return []
            trade = self._detect_breakout(df_btc, i, meta, regime=regime)
            if trade is None:
                trade = self._detect_pullback(df_btc, i, meta)

        elif regime == 'BEAR':
            if funding_z < veto_short:
                logger.info(f'[V15] BTC BEAR: Funding veto (z={funding_z:.2f} < {veto_short})')
                return []
            trade = self._detect_short_ml(df_btc, i, meta)

        elif regime == 'RANGE':
            if funding_z > veto_long:
                logger.info(f'[V15] BTC RANGE: Funding veto (z={funding_z:.2f} > {veto_long})')
                return []
            trade = self._detect_breakout(df_btc, i, meta, regime=regime)

        if trade is None:
            rsi = float(row.get('rsi14', 0))
            bb = float(row.get('bb_pct', 0))
            logger.info(
                f'[V15] BTC: {regime} | No setup '
                f'(rsi={rsi:.1f} bb={bb:.2f} funding_z={funding_z:.2f})'
            )
            return []

        return [self._build_signal('BTC/USDT', trade, regime, funding_z)]

    # =================================================================
    # ETH SIGNAL GENERATION (rule-based committee)
    # =================================================================
    def _generate_eth_signals(self, df_eth, df_btc) -> list:
        """Generate ETH signals: BTC-follower + Breakout + SHORT multi-conf/BB."""
        state = self._pair_state.get('ETH/USDT', PairState())
        meta = self._meta.get('ETH/USDT', {})
        btc_state = self._pair_state.get('BTC/USDT', PairState())

        regime = state.regime
        funding_z = state.funding_zscore

        veto_long = meta.get('funding_veto_long', 2.0)
        veto_short = meta.get('funding_veto_short', -1.5)

        i = len(df_eth) - 1
        row = df_eth.iloc[i]
        trade = None

        if regime in ('BULL', 'RANGE'):
            if funding_z > veto_long:
                logger.info(f'[V15] ETH {regime}: Funding veto (z={funding_z:.2f})')
                return []

            # 1. Check if BTC has a signal -> ETH follows if correlated
            if df_btc is not None and len(df_btc) > 30:
                btc_meta = self._meta.get('BTC/USDT', {})
                btc_i = len(df_btc) - 1
                btc_regime = btc_state.regime

                btc_signal = None
                if btc_regime in ('BULL', 'RANGE'):
                    btc_signal = self._detect_breakout(df_btc, btc_i, btc_meta)
                    if btc_signal is None and btc_regime == 'BULL':
                        btc_signal = self._detect_pullback(df_btc, btc_i, btc_meta)

                if btc_signal is not None:
                    corr = self._compute_pair_btc_corr(df_eth, df_btc)
                    corr_min = meta.get('follower_corr_min', 0.5)
                    if corr >= corr_min:
                        entry = float(row['close'])
                        atr_pct = float(row.get('atr_pct', 2.5))
                        sl_mult = meta.get('tp_sl_atr_sl_mult', 1.5)
                        tp_mult = meta.get('tp_sl_atr_tp_mult', 2.5)
                        sl_pct = max(min(atr_pct / 100 * sl_mult,
                                        meta.get('tp_sl_max_sl', 0.05)),
                                    meta.get('tp_sl_min', 0.015))
                        tp_pct = max(min(atr_pct / 100 * tp_mult,
                                        meta.get('tp_sl_max_tp', 0.08)),
                                    meta.get('tp_sl_min_tp', 0.025))
                        trade = {
                            'direction': 'LONG',
                            'setup': f"FOLLOW_{btc_signal['setup']}",
                            'entry': entry, 'tp_pct': tp_pct, 'sl_pct': sl_pct,
                            'confidence': 0.60,
                        }

            # 2. If no follower signal, try standalone ETH breakout
            if trade is None:
                trade = self._detect_eth_breakout(df_eth, i, meta)

        elif regime == 'BEAR':
            if funding_z < veto_short:
                logger.info(f'[V15] ETH BEAR: Funding veto (z={funding_z:.2f})')
                return []

            # SHORT: multi-conf first, then BB upper
            trade = self._detect_eth_short_multi_conf(df_eth, i, meta)
            if trade is None:
                trade = self._detect_eth_short_bb_upper(df_eth, i, meta)

        if trade is None:
            rsi = float(row.get('rsi14', 0))
            bb = float(row.get('bb_pct', 0))
            logger.info(
                f'[V15] ETH: {regime} | No setup '
                f'(rsi={rsi:.1f} bb={bb:.2f} funding_z={funding_z:.2f})'
            )
            return []

        return [self._build_signal('ETH/USDT', trade, regime, funding_z)]

    # =================================================================
    # ETH DETECTORS
    # =================================================================
    def _compute_pair_btc_corr(self, df_eth, df_btc) -> float:
        """20-bar rolling correlation of returns between ETH and BTC."""
        try:
            eth_ret = df_eth['close'].pct_change()
            btc_close = df_btc['close'].reindex(df_eth.index, method='ffill')
            btc_ret = btc_close.pct_change()
            corr = eth_ret.rolling(20).corr(btc_ret)
            val = corr.iloc[-1]
            return float(val) if not pd.isna(val) else 0.0
        except Exception:
            return 0.0

    def _detect_eth_breakout(self, df, i, meta):
        """ETH standalone breakout — adapted params (vol>=1.3, bb<5.5)."""
        if i < 25:
            return None
        row = df.iloc[i]

        high20 = float(df['high'].iloc[i-20:i].max())
        if float(row['close']) <= high20:
            return None

        vol_min = meta.get('breakout_vol_min', 1.3)
        if float(row.get('vol_ratio', 1)) < vol_min:
            return None

        bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
        if bar_move > meta.get('breakout_bar_move_max', 3.5):
            return None

        bb_max = meta.get('breakout_bb_max', 5.5)
        recent_bb = df['bb_width'].iloc[i-5:i]
        if (recent_bb < bb_max).sum() < 2:  # 2/5 for ETH (more volatile)
            return None

        adx_max = meta.get('breakout_adx_max', 32)
        if df['adx14'].iloc[i-3:i].mean() > adx_max:
            return None

        entry = float(row['close'])
        sl_raw = float(df['low'].iloc[i-5:i].min()) * 0.995
        sl_pct = (entry - sl_raw) / entry
        sl_min = meta.get('breakout_sl_min', 0.005)
        sl_max = meta.get('breakout_sl_max', 0.06)
        if sl_pct < sl_min or sl_pct > sl_max:
            return None

        # ATR-based TP/SL (validated for ETH)
        atr_pct = float(row.get('atr_pct', 2.5))
        sl_mult = meta.get('tp_sl_atr_sl_mult', 1.5)
        tp_mult = meta.get('tp_sl_atr_tp_mult', 2.5)
        sl_pct = max(min(atr_pct / 100 * sl_mult, meta.get('tp_sl_max_sl', 0.05)),
                     meta.get('tp_sl_min', 0.015))
        tp_pct = max(min(atr_pct / 100 * tp_mult, meta.get('tp_sl_max_tp', 0.08)),
                     meta.get('tp_sl_min_tp', 0.025))

        return {
            'direction': 'LONG', 'setup': 'BREAKOUT_ETH',
            'entry': entry, 'tp_pct': tp_pct, 'sl_pct': sl_pct,
            'confidence': 0.60,
        }

    def _detect_eth_short_multi_conf(self, df, i, meta):
        """ETH SHORT: RSI>60 + BB>0.75 + bearish candle + vol>1.0."""
        if i < 25:
            return None
        row = df.iloc[i]
        c_val = float(row['close'])
        o_val = float(row['open'])

        # Must be bearish candle
        if c_val >= o_val:
            return None

        rsi_min = meta.get('short_multi_rsi_min', 60)
        if float(row.get('rsi14', 50)) < rsi_min:
            return None

        bb_min = meta.get('short_multi_bb_pct_min', 0.75)
        if float(row.get('bb_pct', 0.5)) < bb_min:
            return None

        vol_min = meta.get('short_multi_vol_ratio_min', 1.0)
        if float(row.get('vol_ratio', 1)) < vol_min:
            return None

        entry = c_val
        atr_pct = float(row.get('atr_pct', 2.5))
        sl_mult = meta.get('tp_sl_atr_sl_mult', 1.5)
        tp_mult = meta.get('tp_sl_atr_tp_mult', 2.5)
        tp = max(min(atr_pct / 100 * tp_mult, meta.get('tp_sl_max_tp', 0.08)),
                 meta.get('tp_sl_min_tp', 0.025))
        sl = max(min(atr_pct / 100 * sl_mult, meta.get('tp_sl_max_sl', 0.05)),
                 meta.get('tp_sl_min', 0.015))

        return {
            'direction': 'SHORT', 'setup': 'MULTI_CONF',
            'entry': entry, 'tp_pct': tp, 'sl_pct': sl,
            'confidence': 0.65,
        }

    def _detect_eth_short_bb_upper(self, df, i, meta):
        """ETH SHORT: BB>0.90 + bearish candle."""
        if i < 25:
            return None
        row = df.iloc[i]
        c_val = float(row['close'])
        o_val = float(row['open'])

        if c_val >= o_val:
            return None

        bb_min = meta.get('short_bb_upper_bb_pct_min', 0.90)
        if float(row.get('bb_pct', 0.5)) < bb_min:
            return None

        entry = c_val
        atr_pct = float(row.get('atr_pct', 2.5))
        sl_mult = meta.get('tp_sl_atr_sl_mult', 1.5)
        tp_mult = meta.get('tp_sl_atr_tp_mult', 2.5)
        tp = max(min(atr_pct / 100 * tp_mult, meta.get('tp_sl_max_tp', 0.08)),
                 meta.get('tp_sl_min_tp', 0.025))
        sl = max(min(atr_pct / 100 * sl_mult, meta.get('tp_sl_max_sl', 0.05)),
                 meta.get('tp_sl_min', 0.015))

        return {
            'direction': 'SHORT', 'setup': 'BB_UPPER',
            'entry': entry, 'tp_pct': tp, 'sl_pct': sl,
            'confidence': 0.55,
        }

    # =================================================================
    # ADAPTIVE METHODS (behind adaptive_enabled flag in meta)
    # =================================================================
    def _adaptive_vol_threshold(self, df, i, meta):
        """Vol threshold relative to recent BB_width percentile (50 bars)."""
        window = meta.get('adaptive_bb_window', 50)
        start = max(0, i - window)
        bb_series = df['bb_width'].iloc[start:i]
        if len(bb_series) < 10:
            return meta.get('breakout_vol_min', 1.8)

        pctile = bb_series.rank(pct=True).iloc[-1] if len(bb_series) > 0 else 0.5

        vol_low = meta.get('adaptive_vol_min_low', 1.3)
        vol_high = meta.get('adaptive_vol_min_high', 2.2)

        if pctile < 0.30:
            return vol_low
        elif pctile > 0.70:
            return vol_high
        else:
            t = (pctile - 0.30) / 0.40
            return vol_low + t * (vol_high - vol_low)

    def _adaptive_bb_compression(self, df, i, meta):
        """BB compression relative to median of last 50 bars."""
        window = meta.get('adaptive_bb_window', 50)
        start = max(0, i - window)
        bb_series = df['bb_width'].iloc[start:i]
        if len(bb_series) < 10:
            return True

        median_bb = bb_series.median()
        recent_bb = df['bb_width'].iloc[max(0, i - 5):i]
        bb_count_min = 3 if meta.get('asset', 'BTC') == 'BTC' else 2
        return (recent_bb < median_bb).sum() >= bb_count_min

    def _adaptive_lookback(self, df, i, meta):
        """Lookback for high_N proportional to BB_width/median."""
        window = meta.get('adaptive_bb_window', 50)
        start = max(0, i - window)
        bb_series = df['bb_width'].iloc[start:i]
        if len(bb_series) < 10:
            return 20

        median_bb = bb_series.median()
        current_bb = float(df['bb_width'].iloc[i]) if i < len(df) else median_bb
        if median_bb <= 0:
            return 20

        ratio = current_bb / median_bb
        lb_min = meta.get('adaptive_lookback_min', 12)
        lb_max = meta.get('adaptive_lookback_max', 30)

        if ratio < 0.6:
            return lb_min
        elif ratio > 1.4:
            return lb_max
        else:
            t = (ratio - 0.6) / 0.8
            return int(lb_min + t * (lb_max - lb_min))

    def _compute_signal_quality(self, df, i, regime, meta):
        """Score breakout quality 0-100 with confluences."""
        row = df.iloc[i]
        score = 0

        # 1. BB compression (25 pts)
        window = meta.get('adaptive_bb_window', 50)
        start = max(0, i - window)
        bb_series = df['bb_width'].iloc[start:i]
        if len(bb_series) >= 10:
            median_bb = bb_series.median()
            current_bb = float(df['bb_width'].iloc[max(0, i - 1)])
            if median_bb > 0:
                ratio = current_bb / median_bb
                if ratio < 0.5:
                    score += 25
                elif ratio < 0.7:
                    score += 18
                elif ratio < 1.0:
                    score += 10

        # 2. Vol spike strength (20 pts)
        vol_ratio = float(row.get('vol_ratio', 1.0))
        if vol_ratio >= 3.0:
            score += 20
        elif vol_ratio >= 2.5:
            score += 16
        elif vol_ratio >= 2.0:
            score += 12
        elif vol_ratio >= 1.5:
            score += 6

        # 3. DI+ crossover (15 pts)
        di_diff = float(row.get('di_diff', 0))
        if i >= 1:
            prev_di_diff = float(df.iloc[i - 1].get('di_diff', 0))
            if di_diff > 0 and prev_di_diff <= 0:
                score += 15
            elif di_diff > 5:
                score += 8
            elif di_diff > 0:
                score += 4

        # 4. Regime alignment (20 pts)
        if regime == 'BULL':
            score += 20
        elif regime == 'RANGE':
            score += 10

        # 5. EMA stack (10 pts)
        ema20 = float(row.get('ema20', 0))
        ema50 = float(row.get('ema50', 0))
        close = float(row['close'])
        if ema20 > 0 and ema50 > 0:
            if close > ema20 > ema50:
                score += 10
            elif close > ema50:
                score += 5

        # 6. RSI zone (10 pts)
        rsi = float(row.get('rsi14', 50))
        if 45 <= rsi <= 65:
            score += 10
        elif 35 <= rsi <= 75:
            score += 5

        return min(score, 100)

    # =================================================================
    # BTC DETECTORS
    # =================================================================
    def _detect_breakout(self, df, i, meta=None, regime=None):
        """Breakout from consolidation. Supports adaptive mode via meta flag."""
        if meta is None:
            meta = self._meta.get('BTC/USDT', {})
        if i < 25:
            return None
        row = df.iloc[i]

        adaptive = meta.get('adaptive_enabled', False)

        # Lookback: dynamic or fixed 20
        if adaptive:
            lookback = self._adaptive_lookback(df, i, meta)
            if i < lookback + 5:
                return None
        else:
            lookback = 20

        high_N = float(df['high'].iloc[i - lookback:i].max())
        if float(row['close']) <= high_N:
            return None

        # Vol threshold: adaptive or static
        if adaptive:
            vol_min = self._adaptive_vol_threshold(df, i, meta)
        else:
            vol_min = meta.get('breakout_vol_min', 1.8)
        if float(row.get('vol_ratio', 1)) < vol_min:
            return None

        bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
        if bar_move > meta.get('breakout_bar_move_max', 2.5):
            return None

        # BB compression: fixed threshold (adaptive BB tested, no improvement)
        bb_max = meta.get('breakout_bb_max', 4.0)
        recent_bb = df['bb_width'].iloc[i - 5:i]
        bb_count_min = 3 if meta.get('asset', 'BTC') == 'BTC' else 2
        if (recent_bb < bb_max).sum() < bb_count_min:
            return None

        adx_max = meta.get('breakout_adx_max', 28)
        if df['adx14'].iloc[i - 3:i].mean() > adx_max:
            return None

        entry = float(row['close'])
        sl_raw = float(df['low'].iloc[max(0, i - 5):i].min()) * 0.997
        sl_pct = (entry - sl_raw) / entry
        sl_min = meta.get('breakout_sl_min', 0.005)
        sl_max = meta.get('breakout_sl_max', 0.04)
        if sl_pct < sl_min or sl_pct > sl_max:
            return None

        # RR: quality-based or fixed
        if adaptive:
            if regime is None:
                regime = self.get_regime('BTC/USDT')
            quality = self._compute_signal_quality(df, i, regime, meta)
            quality_min = meta.get('adaptive_quality_min', 30)
            if quality < quality_min:
                return None

            if quality >= 70:
                rr = meta.get('adaptive_rr_high', 2.0)
            elif quality >= 50:
                rr = meta.get('adaptive_rr_mid', 1.5)
            else:
                rr = meta.get('adaptive_rr_low', 1.2)
        else:
            rr = meta.get('breakout_rr', 1.5)

        tp_pct = sl_pct * rr

        return {
            'direction': 'LONG', 'setup': 'BREAKOUT_B',
            'entry': entry, 'tp_pct': tp_pct, 'sl_pct': sl_pct,
            'confidence': 0.65,
        }

    def _detect_pullback(self, df, i, meta=None):
        """Pullback to EMA20 in uptrend. ATR-based TP/SL."""
        if meta is None:
            meta = self._meta.get('BTC/USDT', {})
        if i < 25:
            return None
        row = df.iloc[i]
        prev = df.iloc[i-1]
        c = float(row['close'])
        o = float(row['open'])

        ema20 = float(row.get('ema20', 0))
        ema50 = float(row.get('ema50', 0))
        if ema20 <= 0 or ema50 <= 0:
            return None

        if c < ema50:
            return None

        dist_min = meta.get('pullback_dist_min', -0.005)
        dist_max = meta.get('pullback_dist_max', 0.015)
        dist = (c - ema20) / ema20
        if dist < dist_min or dist > dist_max:
            return None

        adx = float(row.get('adx14', 0))
        if adx < meta.get('pullback_adx_min', 15):
            return None

        rsi = float(row.get('rsi14', 50))
        rsi_min = meta.get('pullback_rsi_min', 33)
        rsi_max = meta.get('pullback_rsi_max', 58)
        if rsi < rsi_min or rsi > rsi_max:
            return None

        if c <= o:
            return None
        if float(prev['close']) >= float(prev['open']):
            return None

        vol_max = meta.get('pullback_vol_max', 2.0)
        if float(row.get('vol_ratio', 1)) > vol_max:
            return None

        atr_pct = float(row.get('atr_pct', 2.0))
        atr_mult = meta.get('pullback_atr_sl_mult', 1.0)
        sl_min = meta.get('pullback_atr_sl_min', 0.01)
        sl_max = meta.get('pullback_atr_sl_max', 0.03)
        sl_pct = max(min(atr_pct / 100 * atr_mult, sl_max), sl_min)
        rr = meta.get('pullback_rr', 1.67)
        tp_pct = sl_pct * rr

        return {
            'direction': 'LONG', 'setup': 'PULLBACK_EMA20',
            'entry': c, 'tp_pct': tp_pct, 'sl_pct': sl_pct,
            'confidence': 0.55,
        }

    def _detect_short_ml(self, df, i, meta=None):
        """SHORT signal from GBM model (BTC only)."""
        if meta is None:
            meta = self._meta.get('BTC/USDT', {})
        if i < 30 or self.short_model is None:
            return None

        row = df.iloc[i]
        features = meta.get('short_features', [])
        if not features:
            return None

        x_vals = [float(row.get(f, 0)) for f in features]
        x = np.array(x_vals).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0)

        x_scaled = self.short_scaler.transform(x)
        prob = float(self.short_model.predict_proba(x_scaled)[0][1])

        threshold = meta.get('short_threshold', 0.60)
        if prob < threshold:
            return None

        # EMA crossover filter: only short when EMA20 < EMA50 (bearish trend)
        if meta.get('short_ema_filter_enabled', False):
            ema20 = float(row.get('ema20', 0))
            ema50 = float(row.get('ema50', 0))
            if ema20 > 0 and ema50 > 0 and ema20 >= ema50:
                return None

        entry = float(row['close'])
        sl_raw = float(df['high'].iloc[max(0, i-3):i+1].max()) * 1.003
        sl_pct = (sl_raw - entry) / entry
        sl_pct = min(max(sl_pct, 0.015), 0.04)
        tp_pct = sl_pct * 1.67

        return {
            'direction': 'SHORT', 'setup': 'ML_SHORT',
            'entry': entry, 'tp_pct': tp_pct, 'sl_pct': sl_pct,
            'confidence': min(prob, 0.90),
        }

    # =================================================================
    # SIGNAL BUILDER
    # =================================================================
    def _build_signal(self, pair, trade, regime, funding_z):
        """Build V14-compatible signal dict."""
        direction_int = 1 if trade['direction'] == 'LONG' else -1
        confidence = trade.get('confidence', 0.60)
        coin = pair.split('/')[0]

        signal = {
            'pair': pair,
            'direction': direction_int,
            'confidence': confidence,
            'setup': f"V15:{regime}:{trade['setup']}",
            'price': trade['entry'],
            'tp_pct': trade['tp_pct'],
            'sl_pct': trade['sl_pct'],
        }

        # Pass-through trailing config for tight mode (ADA/SOL)
        if trade.get('trail_mode'):
            signal['trail_mode'] = trade['trail_mode']
            signal['trail_fixed_dist'] = trade.get('trail_fixed_dist', 0.008)

        side = trade['direction']
        logger.info(
            f'[V15] {coin}: {regime} | {trade["setup"]} | {side} | '
            f'entry=${trade["entry"]:,.2f} | '
            f'TP={trade["tp_pct"]*100:.1f}%/SL={trade["sl_pct"]*100:.1f}% | '
            f'funding_z={funding_z:.2f}'
        )
        return signal

    # =================================================================
    # ADA/SOL SIGNAL GENERATION (tight trailing, rule-based)
    # =================================================================
    def _generate_alt_trailing_signals(self, pair, df_alt, df_btc) -> list:
        """Generate ADA/SOL signals: BTC-follower + breakout LONG, BTC-breakdown SHORT.
        All trades use tight trailing (0.8% fixed distance, immediate activation)."""
        state = self._pair_state.get(pair, PairState())
        meta = self._meta.get(pair, {})
        btc_state = self._pair_state.get('BTC/USDT', PairState())

        regime = state.regime
        funding_z = state.funding_zscore
        coin = pair.split('/')[0]

        veto_long = meta.get('funding_veto_long', 2.0)
        veto_short = meta.get('funding_veto_short', -1.5)

        i = len(df_alt) - 1
        row = df_alt.iloc[i]
        trade = None

        # ATR-adaptive trailing: max(floor, atr_pct * factor)
        atr_pct = float(row.get('atr_pct', 3.0)) / 100.0  # convert % to decimal
        trail_factor = meta.get('trail_atr_factor', 0.0)
        trail_floor = meta.get('trail_floor', 0.0)
        if trail_factor > 0 and trail_floor > 0:
            trail_dist = max(trail_floor, atr_pct * trail_factor)
        else:
            trail_dist = meta.get('trail_fixed_dist', 0.008)

        if regime in ('BULL', 'RANGE'):
            if funding_z > veto_long:
                logger.info(f'[V15] {coin} {regime}: Funding veto (z={funding_z:.2f})')
                return []

            # 1. BTC-follower: BTC breakout -> alt follows if correlated
            if df_btc is not None and len(df_btc) > 30:
                btc_breakout = self._detect_btc_breakout_for_follower(df_btc, meta)
                if btc_breakout:
                    corr = self._compute_pair_btc_corr(df_alt, df_btc)
                    corr_min = meta.get('follower_corr_min', 0.4)
                    if corr >= corr_min:
                        entry = float(row['close'])
                        trade = {
                            'direction': 'LONG',
                            'setup': 'FOLLOW_BTC_BREAKOUT',
                            'entry': entry,
                            'tp_pct': 0.0,  # not used, trailing handles exit
                            'sl_pct': trail_dist,  # initial SL = trail distance
                            'confidence': 0.60,
                            'trail_mode': 'tight',
                            'trail_fixed_dist': trail_dist,
                        }

            # 2. Standalone alt breakout
            if trade is None:
                trade = self._detect_alt_breakout(df_alt, i, meta)

        elif regime == 'BEAR':
            if funding_z < veto_short:
                logger.info(f'[V15] {coin} BEAR: Funding veto (z={funding_z:.2f})')
                return []

            # BTC-breakdown follower SHORT
            if df_btc is not None and len(df_btc) > 30:
                btc_breakdown = self._detect_btc_breakdown_for_follower(df_btc, meta)
                if btc_breakdown:
                    corr = self._compute_pair_btc_corr(df_alt, df_btc)
                    corr_min = meta.get('follower_corr_min', 0.4)
                    if corr >= corr_min:
                        entry = float(row['close'])
                        trade = {
                            'direction': 'SHORT',
                            'setup': 'FOLLOW_BTC_BREAKDOWN',
                            'entry': entry,
                            'tp_pct': 0.0,
                            'sl_pct': trail_dist,
                            'confidence': 0.60,
                            'trail_mode': 'tight',
                            'trail_fixed_dist': trail_dist,
                        }

        if trade is None:
            rsi = float(row.get('rsi14', 0))
            bb = float(row.get('bb_pct', 0))
            logger.info(
                f'[V15] {coin}: {regime} | No setup '
                f'(rsi={rsi:.1f} bb={bb:.2f} funding_z={funding_z:.2f})'
            )
            return []

        logger.info(f'[V15] {coin}: trail_dist={trail_dist:.4f} '
                     f'(ATR={atr_pct:.4f} factor={trail_factor} floor={trail_floor})')
        return [self._build_signal(pair, trade, regime, funding_z)]

    # =================================================================
    # ADA/SOL DETECTORS
    # =================================================================
    def _detect_btc_breakout_for_follower(self, df_btc, meta):
        """BTC breaks above 20-bar high with volume -> follower LONG trigger."""
        i = len(df_btc) - 1
        if i < 25:
            return False
        row = df_btc.iloc[i]
        lookback = meta.get('breakdown_lookback', 20)
        high20 = float(df_btc['high'].iloc[i-lookback:i].max())
        if float(row['close']) <= high20:
            return False
        vol_min = meta.get('breakdown_vol_min', 1.0)
        if float(row.get('vol_ratio', 1)) < vol_min:
            return False
        return True

    def _detect_btc_breakdown_for_follower(self, df_btc, meta):
        """BTC breaks below 20-bar low with volume -> follower SHORT trigger."""
        i = len(df_btc) - 1
        if i < 25:
            return False
        row = df_btc.iloc[i]
        lookback = meta.get('breakdown_lookback', 20)
        low20 = float(df_btc['low'].iloc[i-lookback:i].min())
        if float(row['close']) >= low20:
            return False
        vol_min = meta.get('breakdown_vol_min', 1.0)
        if float(row.get('vol_ratio', 1)) < vol_min:
            return False
        return True

    def _detect_alt_breakout(self, df, i, meta):
        """Standalone alt breakout: close > high20 + vol + bb filters.
        Returns trade dict with tight trailing config."""
        if i < 25:
            return None
        row = df.iloc[i]

        high20 = float(df['high'].iloc[i-20:i].max())
        if float(row['close']) <= high20:
            return None

        vol_min = meta.get('breakout_vol_min', 1.2)
        if float(row.get('vol_ratio', 1)) < vol_min:
            return None

        bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
        if bar_move > meta.get('breakout_bar_move_max', 3.5):
            return None

        bb_max = meta.get('breakout_bb_max', 6.0)
        recent_bb = df['bb_width'].iloc[i-5:i]
        if (recent_bb < bb_max).sum() < 2:
            return None

        adx_max = meta.get('breakout_adx_max', 32)
        if df['adx14'].iloc[i-3:i].mean() > adx_max:
            return None

        entry = float(row['close'])
        # ATR-adaptive trailing
        atr_pct = float(row.get('atr_pct', 3.0)) / 100.0
        trail_factor = meta.get('trail_atr_factor', 0.0)
        trail_floor = meta.get('trail_floor', 0.0)
        if trail_factor > 0 and trail_floor > 0:
            trail_dist = max(trail_floor, atr_pct * trail_factor)
        else:
            trail_dist = meta.get('trail_fixed_dist', 0.008)

        return {
            'direction': 'LONG', 'setup': 'BREAKOUT_ALT',
            'entry': entry,
            'tp_pct': 0.0,  # not used, trailing handles exit
            'sl_pct': trail_dist,
            'confidence': 0.55,
            'trail_mode': 'tight',
            'trail_fixed_dist': trail_dist,
        }

    # =================================================================
    # HELPERS
    # =================================================================
    def _ohlcv_to_df(self, ohlcv) -> pd.DataFrame:
        """Convert ccxt OHLCV to DataFrame."""
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        df = df.set_index('timestamp').sort_index()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        # Exclude last (incomplete) candle
        return df.iloc[:-1]

    def _compute_features(self, df) -> pd.DataFrame:
        """Compute all technical features from 4h OHLCV. Identical to backtest."""
        h, l, c, v = df['high'], df['low'], df['close'], df['volume']

        # EMAs
        for n in [20, 50, 200]:
            df[f'ema{n}'] = pta.ema(c, length=n)
        df['ema20_slope'] = df['ema20'].pct_change(5) * 100
        df['ema50_slope'] = df['ema50'].pct_change(10) * 100
        df['ema200_dist'] = (c - df['ema200']) / df['ema200'] * 100

        # RSI
        df['rsi14'] = pta.rsi(c, length=14)

        # ATR
        atr = pta.atr(h, l, c, length=14)
        df['atr14'] = atr
        df['atr_pct'] = atr / c * 100

        # Bollinger Bands
        bb = pta.bbands(c, length=20)
        if bb is not None:
            bb_low, bb_mid, bb_up = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
            df['bb_pct'] = (c - bb_low) / (bb_up - bb_low).replace(0, np.nan)
            df['bb_width'] = (bb_up - bb_low) / bb_mid * 100
        else:
            df['bb_pct'] = 0.5
            df['bb_width'] = 5.0

        # ADX
        adx_df = pta.adx(h, l, c, length=14)
        if adx_df is not None:
            df['adx14'] = adx_df.iloc[:, 0]
            df['di_plus'] = adx_df.iloc[:, 1]
            df['di_minus'] = adx_df.iloc[:, 2]
            df['di_diff'] = df['di_plus'] - df['di_minus']
        else:
            df['adx14'] = 20.0
            df['di_diff'] = 0.0

        # Volume ratio
        vol_ma = v.rolling(20).mean()
        df['vol_ratio'] = v / vol_ma.replace(0, np.nan)

        # Rolling high/low (20 bars)
        df['high20'] = h.rolling(20).max().shift(1)
        df['low20'] = l.rolling(20).min().shift(1)
        df['range_pos'] = (c - df['low20']) / (df['high20'] - df['low20']).replace(0, np.nan)

        # Returns
        df['ret_1'] = c.pct_change(1) * 100
        df['ret_5'] = c.pct_change(5) * 100

        # Extra features for SHORT ML
        df['rsi_slope'] = df['rsi14'].diff(3)
        vol_ma5 = v.rolling(5).mean()
        vol_ma20 = v.rolling(20).mean()
        df['vol_slope'] = (vol_ma5 / vol_ma20.replace(0, np.nan) - 1) * 100
        df['ret_10'] = c.pct_change(10) * 100
        up = (c > c.shift(1)).astype(int)
        df['consec_up'] = up.rolling(8).sum()

        # Drop rows with NaN in critical features
        required = ['ema20', 'ema50', 'rsi14', 'atr14', 'adx14']
        df = df.dropna(subset=required)
        return df
