"""
ML Strategy V15 - Expert Committee BTC/USDT
============================================
Sistema validado: WF 8/12, OOS PF=1.35, CAGR ~37%.

  BULL  -> Breakout B LONG + Pullback EMA20 LONG (reglas, ATR TP/SL)
  BEAR  -> SHORT ML (GBM threshold=0.60, entrenado en BEAR)
  RANGE -> Breakout B LONG only

Gates:
  - Funding veto: z-score > 2 bloquea LONG, < -1.5 bloquea SHORT
  - Regime: EMA20/50 diario distance con 2% dead zone + recovery filter

Senales compatibles con V14 (pair/direction int/tp_pct/sl_pct/setup/confidence).
"""

import json
import logging
import numpy as np
import pandas as pd
import pandas_ta as pta
import joblib
import requests
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / 'strategies' / 'btc_v15' / 'models'
FAPI_BASE = 'https://fapi.binance.com'
LOOKBACK = 250  # candles to fetch


class MLStrategyV15:
    """V15 Expert Committee — BTC only, 3 regimenes."""

    def __init__(self):
        self.pairs = ['BTC/USDT']
        self.regime = 'RANGE'
        self.regime_updated = None

        # ML SHORT model
        self.short_model = None
        self.short_scaler = None
        self.meta = {}

        # Cached state (updated by update_regime)
        self._funding_zscore = 0.0
        self._daily_ema20 = None
        self._daily_ema50 = None
        self._daily_ema200 = None

        # Load models at init
        self.load_models()

    # =================================================================
    # MODEL LOADING
    # =================================================================
    def load_models(self) -> int:
        """Load SHORT GBM model + meta. Returns 1 if OK, 0 if failed."""
        if not MODEL_DIR.exists():
            logger.error(f'[V15] Model dir not found: {MODEL_DIR}')
            return 0
        try:
            self.short_model = joblib.load(MODEL_DIR / 'short_gbm.pkl')
            self.short_scaler = joblib.load(MODEL_DIR / 'short_scaler.pkl')
            meta_path = MODEL_DIR / 'meta_v15.json'
            if meta_path.exists():
                with open(meta_path) as f:
                    self.meta = json.load(f)
            logger.info(
                f'[V15] Modelos cargados | SHORT GBM threshold='
                f'{self.meta.get("short_threshold", 0.60)}'
            )
            return 1
        except Exception as e:
            logger.error(f'[V15] Error cargando modelos: {e}')
            return 0

    # =================================================================
    # REGIME DETECTION (called daily by bot)
    # =================================================================
    def update_regime(self, exchange):
        """Update macro regime from BTC daily candles + funding rate."""
        try:
            # Fetch DAILY candles (not 4h) — need 250 days for EMA200
            ohlcv_1d = exchange.fetch_ohlcv('BTC/USDT', '1d', limit=250)
            if not ohlcv_1d or len(ohlcv_1d) < 55:
                logger.warning('[V15] Insufficient daily data for regime update')
                return

            df_1d = pd.DataFrame(ohlcv_1d, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            df_1d['timestamp'] = pd.to_datetime(df_1d['ts'], unit='ms', utc=True)
            df_1d = df_1d.set_index('timestamp').sort_index()
            # Exclude today (incomplete)
            daily_close = df_1d['close'].iloc[:-1]

            ema20 = daily_close.ewm(span=20, adjust=False).mean()
            ema50 = daily_close.ewm(span=50, adjust=False).mean()
            ema200 = daily_close.ewm(span=200, adjust=False).mean() if len(daily_close) >= 200 else None

            # Use yesterday's EMAs (shift 1 day to avoid look-ahead)
            self._daily_ema20 = float(ema20.iloc[-1])
            self._daily_ema50 = float(ema50.iloc[-1])
            self._daily_ema200 = float(ema200.iloc[-1]) if ema200 is not None else None

            # Current price from latest 4h candle
            ohlcv_4h = exchange.fetch_ohlcv('BTC/USDT', '4h', limit=3)
            cur_close = float(ohlcv_4h[-2][4]) if ohlcv_4h and len(ohlcv_4h) >= 2 else float(daily_close.iloc[-1])

            self.regime = self._classify_regime(
                self._daily_ema20, self._daily_ema50, self._daily_ema200, cur_close
            )
            self.regime_updated = datetime.now(timezone.utc)

            # Fetch funding rate z-score
            self._funding_zscore = self._fetch_funding_zscore()

            logger.info(
                f'[V15] Regime: {self.regime} | '
                f'EMA20={self._daily_ema20:,.0f} EMA50={self._daily_ema50:,.0f} | '
                f'funding_z={self._funding_zscore:.2f}'
            )
        except Exception as e:
            logger.error(f'[V15] Error updating regime: {e}')

    def _classify_regime(self, ema20, ema50, ema200, close):
        """Classify regime: BULL / BEAR / RANGE. Identical to backtest."""
        dead_zone = self.meta.get('regime_dead_zone', 0.02)
        dist = (ema20 - ema50) / ema50

        if dist > dead_zone:
            return 'BULL'
        elif dist < -dead_zone:
            # Recovery filter: price > EMA200 = not real bear
            if ema200 is not None and close > ema200:
                return 'RANGE'
            # Price must be BELOW EMA50 (confirmed downtrend)
            if close > ema50:
                return 'RANGE'
            return 'BEAR'
        return 'RANGE'

    def _fetch_funding_zscore(self) -> float:
        """Fetch BTC funding rate and compute 90-day z-score."""
        try:
            resp = requests.get(
                f'{FAPI_BASE}/fapi/v1/fundingRate',
                params={'symbol': 'BTCUSDT', 'limit': 100},
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
            logger.debug(f'[V15] Funding fetch error: {e}')
            return 0.0

    # =================================================================
    # SIGNAL GENERATION (called every 4h candle)
    # =================================================================
    def generate_signals(self, exchange, open_pairs=None) -> list:
        """Generate signals for BTC. V14-compatible format."""
        if open_pairs is None:
            open_pairs = set()

        if 'BTC/USDT' in open_pairs:
            logger.info('[V15] BTC/USDT already open, skipping')
            return []

        if self.short_model is None:
            logger.warning('[V15] Models not loaded')
            return []

        try:
            ohlcv = exchange.fetch_ohlcv('BTC/USDT', '4h', limit=LOOKBACK)
            if not ohlcv or len(ohlcv) < 50:
                logger.warning('[V15] Insufficient OHLCV data')
                return []

            df = self._ohlcv_to_df(ohlcv)
            df = self._compute_features(df)
            if df is None or len(df) < 30:
                return []

            # Add daily macro feature for SHORT ML
            if self._daily_ema20 and self._daily_ema50:
                df['bull_1d'] = int(self._daily_ema20 > self._daily_ema50)
            else:
                df['bull_1d'] = 0

            i = len(df) - 1  # last bar
            row = df.iloc[i]
            regime = self.regime
            funding_z = self._funding_zscore

            # Funding veto thresholds
            veto_long = self.meta.get('funding_veto_long', 2.0)
            veto_short = self.meta.get('funding_veto_short', -1.5)

            trade = None

            if regime == 'BULL':
                if funding_z > veto_long:
                    logger.info(f'[V15] BULL: Funding veto (z={funding_z:.2f} > {veto_long})')
                    return []
                trade = self._detect_breakout(df, i)
                if trade is None:
                    trade = self._detect_pullback(df, i)

            elif regime == 'BEAR':
                if funding_z < veto_short:
                    logger.info(f'[V15] BEAR: Funding veto (z={funding_z:.2f} < {veto_short})')
                    return []
                trade = self._detect_short_ml(df, i)

            elif regime == 'RANGE':
                if funding_z > veto_long:
                    logger.info(f'[V15] RANGE: Funding veto (z={funding_z:.2f} > {veto_long})')
                    return []
                trade = self._detect_breakout(df, i)

            if trade is None:
                rsi = float(row.get('rsi14', 0))
                bb = float(row.get('bb_pct', 0))
                logger.info(
                    f'[V15] BTC: {regime} | No setup '
                    f'(rsi={rsi:.1f} bb={bb:.2f} funding_z={funding_z:.2f})'
                )
                return []

            # Build V14-compatible signal
            direction_int = 1 if trade['direction'] == 'LONG' else -1
            confidence = trade.get('confidence', 0.60)

            signal = {
                'pair': 'BTC/USDT',
                'direction': direction_int,
                'confidence': confidence,
                'setup': f"V15:{regime}:{trade['setup']}",
                'price': trade['entry'],
                'tp_pct': trade['tp_pct'],
                'sl_pct': trade['sl_pct'],
            }

            side = trade['direction']
            logger.info(
                f'[V15] BTC: {regime} | {trade["setup"]} | {side} | '
                f'entry=${trade["entry"]:,.0f} | '
                f'TP={trade["tp_pct"]*100:.1f}%/SL={trade["sl_pct"]*100:.1f}% | '
                f'funding_z={funding_z:.2f}'
            )
            return [signal]

        except Exception as e:
            logger.error(f'[V15] Error generating signals: {e}')
            return []

    # =================================================================
    # SETUP: BREAKOUT B (rule-based)
    # =================================================================
    def _detect_breakout(self, df, i):
        """Breakout from consolidation. Identical to backtest."""
        if i < 25:
            return None
        row = df.iloc[i]

        # Close must break 20-bar high
        high20 = float(df['high'].iloc[i-20:i].max())
        if float(row['close']) <= high20:
            return None

        # Volume confirms (>= 1.8x average)
        vol_min = self.meta.get('breakout_vol_min', 1.8)
        if float(row.get('vol_ratio', 1)) < vol_min:
            return None

        # Bar move not too large
        bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
        if bar_move > self.meta.get('breakout_bar_move_max', 2.5):
            return None

        # Narrow BB in recent bars (consolidation)
        bb_max = self.meta.get('breakout_bb_max', 4.0)
        recent_bb = df['bb_width'].iloc[i-5:i]
        if (recent_bb < bb_max).sum() < 3:
            return None

        # Low ADX (not trending yet)
        adx_max = self.meta.get('breakout_adx_max', 28)
        if df['adx14'].iloc[i-3:i].mean() > adx_max:
            return None

        # TP/SL based on consolidation range
        entry = float(row['close'])
        sl_raw = float(df['low'].iloc[i-5:i].min()) * 0.997
        sl_pct = (entry - sl_raw) / entry
        sl_min = self.meta.get('breakout_sl_min', 0.005)
        sl_max = self.meta.get('breakout_sl_max', 0.04)
        if sl_pct < sl_min or sl_pct > sl_max:
            return None
        rr = self.meta.get('breakout_rr', 1.5)
        tp_pct = sl_pct * rr

        return {
            'direction': 'LONG', 'setup': 'BREAKOUT_B',
            'entry': entry, 'tp_pct': tp_pct, 'sl_pct': sl_pct,
            'confidence': 0.65,
        }

    # =================================================================
    # SETUP: PULLBACK EMA20 (rule-based, ATR TP/SL)
    # =================================================================
    def _detect_pullback(self, df, i):
        """Pullback to EMA20 in uptrend. ATR-based TP/SL. Identical to backtest."""
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

        # Price above EMA50
        if c < ema50:
            return None

        # Price near EMA20
        dist_min = self.meta.get('pullback_dist_min', -0.005)
        dist_max = self.meta.get('pullback_dist_max', 0.015)
        dist = (c - ema20) / ema20
        if dist < dist_min or dist > dist_max:
            return None

        # ADX minimum
        adx = float(row.get('adx14', 0))
        if adx < self.meta.get('pullback_adx_min', 15):
            return None

        # RSI in pullback zone
        rsi = float(row.get('rsi14', 50))
        rsi_min = self.meta.get('pullback_rsi_min', 33)
        rsi_max = self.meta.get('pullback_rsi_max', 58)
        if rsi < rsi_min or rsi > rsi_max:
            return None

        # Current candle bullish
        if c <= o:
            return None

        # Previous candle bearish (confirms pullback)
        if float(prev['close']) >= float(prev['open']):
            return None

        # Volume not excessive
        vol_max = self.meta.get('pullback_vol_max', 2.0)
        if float(row.get('vol_ratio', 1)) > vol_max:
            return None

        # ATR-based TP/SL
        atr_pct = float(row.get('atr_pct', 2.0))
        atr_mult = self.meta.get('pullback_atr_sl_mult', 1.0)
        sl_min = self.meta.get('pullback_atr_sl_min', 0.01)
        sl_max = self.meta.get('pullback_atr_sl_max', 0.03)
        sl_pct = max(min(atr_pct / 100 * atr_mult, sl_max), sl_min)
        rr = self.meta.get('pullback_rr', 1.67)
        tp_pct = sl_pct * rr

        return {
            'direction': 'LONG', 'setup': 'PULLBACK_EMA20',
            'entry': c, 'tp_pct': tp_pct, 'sl_pct': sl_pct,
            'confidence': 0.55,
        }

    # =================================================================
    # SETUP: SHORT ML (GradientBoosting)
    # =================================================================
    def _detect_short_ml(self, df, i):
        """SHORT signal from GBM model. Identical to backtest."""
        if i < 30 or self.short_model is None:
            return None

        row = df.iloc[i]
        features = self.meta.get('short_features', [])
        if not features:
            return None

        # Extract features
        x_vals = [float(row.get(f, 0)) for f in features]
        x = np.array(x_vals).reshape(1, -1)
        # Handle NaN
        x = np.nan_to_num(x, nan=0.0)

        x_scaled = self.short_scaler.transform(x)
        prob = float(self.short_model.predict_proba(x_scaled)[0][1])

        threshold = self.meta.get('short_threshold', 0.60)
        if prob < threshold:
            return None

        entry = float(row['close'])
        # Dynamic SL: max of last 3 bars
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
