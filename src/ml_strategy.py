"""
ML Strategy - Motor de Senales V9.5
====================================
V7 base: LightGBM per-pair models, regime detection, trailing stops.
V8.4 adds: Macro intelligence (adaptive threshold + ML sizing + soft risk-off).
V8.5 adds: ConvictionScorer (per-trade PnL prediction -> sizing + filtering).
V9 adds: LossDetector (binary classifier to skip predicted losing trades).
V9.5 adds: Per-pair LossDetector (11 individual models, optimized thresholds).

Pipeline:
  1. MacroScorer (daily): macro_score [0,1] -> adaptive threshold + ML sizing
  2. Soft Risk-Off (daily): reduce sizing on extreme macro days
  3. V7 Signal Models (per 4h candle): confidence + prediction per pair
  4. ConvictionScorer (per trade): predicts PnL -> skip bad trades + adjust sizing
  5. LossDetector V9.5 (per trade): Per-pair P(loss) > threshold -> skip trade
"""

import json
import logging
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
import ccxt
from datetime import datetime, timezone
from scipy.special import expit  # sigmoid

from config.settings import (
    MODELS_DIR, ML_PAIRS, ML_SIGNAL_THRESHOLD, ML_TIMEFRAME,
    ML_LEVERAGE, ML_RISK_PER_TRADE,
    ML_V84_ENABLED, ML_ADAPTIVE_THRESH_MIN, ML_ADAPTIVE_THRESH_MAX,
    ML_SIZING_MIN, ML_SIZING_MAX, ML_RISKOFF_ENABLED,
    ML_V85_ENABLED, ML_CONVICTION_SKIP_MULT,
    ML_CONVICTION_SIZING_MIN, ML_CONVICTION_SIZING_MAX,
    ML_V9_ENABLED, ML_LOSS_THRESHOLD, ML_SHADOW_ENABLED,
    ML_BTC_CONFIG,
)

logger = logging.getLogger(__name__)


class MLStrategy:
    """Genera senales de trading usando modelos ML + macro intelligence."""

    def __init__(self):
        self.models = {}        # {pair: lgb model}
        self.pred_stds = {}     # {pair: float} para confidence
        self.feature_cols = []  # columnas de features V7
        self.regime = 'RANGE'
        self.regime_updated = None
        self.pairs = []         # pares con modelo cargado

        # V8.4 Macro
        self.macro_scorer = None
        self.macro_fcols = []
        self.macro_score = 0.5      # neutral default
        self.risk_off_mult = 1.0    # no reduction default
        self.macro_updated = None
        self.v84_enabled = ML_V84_ENABLED

        # V8.5 ConvictionScorer
        self.conviction_scorer = None
        self.conviction_fcols = []
        self.conviction_pred_std = 1.0
        self.v85_enabled = ML_V85_ENABLED

        # V9 LossDetector (generic - deprecated, kept for fallback)
        self.loss_detector = None
        self.loss_fcols = []
        self.loss_threshold = ML_LOSS_THRESHOLD
        self.v9_enabled = ML_V9_ENABLED

        # V9.5 Per-pair LossDetector
        self.v95_enabled = False  # Auto-enabled if v95 models found
        self.v95_loss_detectors = {}  # {pair: model}
        self.v95_thresholds = {}      # {pair: threshold}
        self.v95_fcols = []           # Feature columns (same for all pairs)

        # V13.01: BTC V2 model (specialized GradientBoosting)
        self.btc_v2_model = None
        self.btc_v2_scaler = None
        self.btc_v2_fcols = []
        self.btc_v2_pred_std = 0.01
        self.btc_v2_enabled = False

    def load_models(self) -> int:
        """Carga modelos desde disco. Retorna cantidad cargada."""
        # V7 metadata
        meta_path = MODELS_DIR / 'v7_meta.json'
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.feature_cols = meta.get('feature_cols', [])
            self.pred_stds = meta.get('pred_stds', {})
            logger.info(f"[ML] V7 metadata: {len(self.feature_cols)} features")

        # V7 per-pair models
        count = 0
        for pair in ML_PAIRS:
            safe = pair.replace('/', '_')
            model_path = MODELS_DIR / f'v7_{safe}.pkl'
            if model_path.exists():
                self.models[pair] = joblib.load(model_path)
                count += 1
                logger.info(f"[ML] Modelo cargado: {pair}")
            else:
                logger.warning(f"[ML] Sin modelo para {pair}")

        self.pairs = list(self.models.keys())

        # V8.4 MacroScorer
        if self.v84_enabled:
            self._load_macro_model()

        # V8.5 ConvictionScorer
        if self.v85_enabled:
            self._load_conviction_model()

        # V9.5 Per-pair LossDetector (preferred over V9)
        self._load_loss_detector_v95()

        # V9 LossDetector (fallback if V9.5 not available)
        if not self.v95_enabled and self.v9_enabled:
            self._load_loss_detector()

        # V13.01: BTC V2 specialized model
        if 'BTC/USDT' in ML_PAIRS and not ML_BTC_CONFIG.get('use_v7_model', True):
            self._load_btc_v2_model()

        logger.info(f"[ML] {count} modelos cargados")
        return count

    def _load_macro_model(self):
        """Carga MacroScorer model y metadata."""
        scorer_path = MODELS_DIR / 'v84_macro_scorer.pkl'
        meta_path = MODELS_DIR / 'v84_meta.json'

        if not scorer_path.exists():
            logger.warning("[ML] V8.4 MacroScorer no encontrado - usando V7 puro")
            self.v84_enabled = False
            return

        try:
            self.macro_scorer = joblib.load(scorer_path)
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                self.macro_fcols = meta.get('macro_feature_cols', [])
            logger.info(f"[ML] V8.4 MacroScorer cargado ({len(self.macro_fcols)} features)")
        except Exception as e:
            logger.error(f"[ML] Error cargando MacroScorer: {e}")
            self.v84_enabled = False

    def _load_conviction_model(self):
        """Carga ConvictionScorer model y metadata."""
        scorer_path = MODELS_DIR / 'v85_conviction_scorer.pkl'
        meta_path = MODELS_DIR / 'v85_meta.json'

        if not scorer_path.exists():
            logger.warning("[ML] V8.5 ConvictionScorer no encontrado - usando V8.4 puro")
            self.v85_enabled = False
            return

        try:
            self.conviction_scorer = joblib.load(scorer_path)
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                self.conviction_fcols = meta.get('conviction_feature_cols', [])
                self.conviction_pred_std = meta.get('conviction_pred_std', 1.0)
            logger.info(f"[ML] V8.5 ConvictionScorer cargado "
                        f"({len(self.conviction_fcols)} features, "
                        f"pred_std={self.conviction_pred_std:.4f})")
        except Exception as e:
            logger.error(f"[ML] Error cargando ConvictionScorer: {e}")
            self.v85_enabled = False

    def _load_loss_detector(self):
        """Carga V9 LossDetector model y metadata (generic fallback)."""
        scorer_path = MODELS_DIR / 'v9_loss_detector.pkl'
        meta_path = MODELS_DIR / 'v9_meta.json'

        if not scorer_path.exists():
            logger.warning("[ML] V9 LossDetector no encontrado - usando V8.5 puro")
            self.v9_enabled = False
            return

        try:
            self.loss_detector = joblib.load(scorer_path)
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                self.loss_fcols = meta.get('loss_feature_cols', [])
                self.loss_threshold = meta.get('loss_threshold', ML_LOSS_THRESHOLD)
            logger.info(f"[ML] V9 LossDetector cargado "
                        f"({len(self.loss_fcols)} features, "
                        f"threshold={self.loss_threshold})")
        except Exception as e:
            logger.error(f"[ML] Error cargando LossDetector: {e}")
            self.v9_enabled = False

    def _load_loss_detector_v95(self):
        """Carga V9.5 Per-pair LossDetector models y metadata."""
        meta_path = MODELS_DIR / 'v95_meta.json'

        if not meta_path.exists():
            logger.info("[ML] V9.5 meta no encontrado - usando V9 generico si disponible")
            return

        try:
            with open(meta_path) as f:
                meta = json.load(f)

            self.v95_fcols = meta.get('feature_cols', [])
            pairs_meta = meta.get('pairs', {})

            count = 0
            for pair, info in pairs_meta.items():
                safe = pair.replace('/', '')
                model_path = MODELS_DIR / f'v95_ld_{safe}.pkl'

                if model_path.exists():
                    self.v95_loss_detectors[pair] = joblib.load(model_path)
                    self.v95_thresholds[pair] = info.get('threshold', 0.55)
                    count += 1

            if count > 0:
                self.v95_enabled = True
                logger.info(f"[ML] V9.5 LossDetector cargado: {count} pares "
                            f"({len(self.v95_fcols)} features)")
                for pair, thresh in self.v95_thresholds.items():
                    auc = pairs_meta.get(pair, {}).get('auc', 0)
                    logger.info(f"  {pair}: thresh={thresh:.2f}, AUC={auc:.3f}")
            else:
                logger.warning("[ML] V9.5 meta existe pero no hay modelos cargados")

        except Exception as e:
            logger.error(f"[ML] Error cargando V9.5 LossDetector: {e}")
            self.v95_enabled = False

    def _load_btc_v2_model(self):
        """Carga BTC V2 model especializado (GradientBoosting con TP/SL optimizado)."""
        model_file = ML_BTC_CONFIG.get('model_file', 'btc_v2_gradientboosting.pkl')
        model_path = MODELS_DIR / model_file

        if not model_path.exists():
            logger.warning(f"[ML] BTC V2 model no encontrado: {model_path}")
            return

        try:
            data = joblib.load(model_path)
            self.btc_v2_model = data.get('model')
            self.btc_v2_scaler = data.get('scaler')
            self.btc_v2_fcols = data.get('feature_cols', [])
            self.btc_v2_pred_std = data.get('pred_std', 0.01)
            self.btc_v2_enabled = True

            tp_pct = ML_BTC_CONFIG.get('tp_pct', 0.04)
            sl_pct = ML_BTC_CONFIG.get('sl_pct', 0.02)
            logger.info(f"[ML] BTC V2 model cargado ({len(self.btc_v2_fcols)} features, "
                        f"TP={tp_pct*100}%, SL={sl_pct*100}%)")
        except Exception as e:
            logger.error(f"[ML] Error cargando BTC V2 model: {e}")
            self.btc_v2_enabled = False

    # =========================================================================
    # V9: LOSS DETECTOR + PAIR TA + BTC CONTEXT
    # =========================================================================
    def compute_pair_ta_live(self, df: pd.DataFrame) -> dict:
        """Compute pair TA features for LossDetector at current bar."""
        c = df['close']

        rsi14 = ta.rsi(c, length=14)
        rsi_val = float(rsi14.iloc[-1]) if rsi14 is not None and len(rsi14) > 0 else 50.0

        bb = ta.bbands(c, length=20, std=2.0)
        if bb is not None:
            bbu, bbl = bb.iloc[:, 0], bb.iloc[:, 2]
            bb_range = bbu - bbl
            bb_pct_s = np.where(bb_range > 0, (c - bbl) / bb_range, 0.5)
            bb_val = float(bb_pct_s[-1]) if len(bb_pct_s) > 0 else 0.5
        else:
            bb_val = 0.5

        vol_ma = df['volume'].rolling(20).mean()
        vr = df['volume'] / vol_ma
        vr_val = float(vr.iloc[-1]) if not np.isnan(vr.iloc[-1]) else 1.0

        ret5 = c.pct_change(5)
        ret20 = c.pct_change(20)

        return {
            'ld_pair_rsi14': rsi_val if not np.isnan(rsi_val) else 50.0,
            'ld_pair_bb_pct': bb_val if not np.isnan(bb_val) else 0.5,
            'ld_pair_vol_ratio': vr_val,
            'ld_pair_ret_5': float(ret5.iloc[-1]) if not np.isnan(ret5.iloc[-1]) else 0.0,
            'ld_pair_ret_20': float(ret20.iloc[-1]) if not np.isnan(ret20.iloc[-1]) else 0.0,
        }

    def compute_btc_context_live(self, btc_df: pd.DataFrame) -> dict:
        """Compute BTC context features for LossDetector."""
        if btc_df is None or len(btc_df) < 25:
            return {'ld_btc_ret_5': 0.0, 'ld_btc_rsi14': 50.0, 'ld_btc_vol20': 0.02}

        c = btc_df['close']
        rsi = ta.rsi(c, length=14)
        vol = c.pct_change().rolling(20).std()
        ret5 = c.pct_change(5)

        return {
            'ld_btc_ret_5': float(ret5.iloc[-1]) if not np.isnan(ret5.iloc[-1]) else 0.0,
            'ld_btc_rsi14': float(rsi.iloc[-1]) if rsi is not None and not np.isnan(rsi.iloc[-1]) else 50.0,
            'ld_btc_vol20': float(vol.iloc[-1]) if not np.isnan(vol.iloc[-1]) else 0.02,
        }

    def score_loss_detector(self, signal, pair_ta, btc_ctx, conviction_pred):
        """Score a trade with LossDetector. Returns True if trade should be SKIPPED."""
        if not self.v9_enabled or self.loss_detector is None:
            return False, 0.0

        features = {
            'cs_conf': signal['confidence'],
            'cs_pred_mag': abs(signal['prediction']),
            'cs_macro_score': self.macro_score,
            'cs_risk_off': self.risk_off_mult,
            'cs_regime_bull': 1.0 if self.regime == 'BULL' else 0.0,
            'cs_regime_bear': 1.0 if self.regime == 'BEAR' else 0.0,
            'cs_regime_range': 1.0 if self.regime == 'RANGE' else 0.0,
            'cs_atr_pct': signal['atr_pct'],
            'cs_n_open': signal.get('n_open', 0),
            'cs_pred_sign': float(signal['direction']),
            'ld_conviction_pred': conviction_pred,
            **pair_ta,
            **btc_ctx,
            'ld_hour': float(datetime.now(timezone.utc).hour),
            'ld_tp_sl_ratio': 0.03 / 0.015,
        }

        df_feat = pd.DataFrame([features])
        cols = [c for c in self.loss_fcols if c in df_feat.columns]
        if not cols:
            return False, 0.0

        p_loss = float(self.loss_detector.predict_proba(df_feat[cols])[0][1])

        if p_loss > self.loss_threshold:
            return True, p_loss

        return False, p_loss

    def score_loss_detector_v95(self, pair, signal, pair_ta, btc_ctx, conviction_pred):
        """Score a trade with V9.5 per-pair LossDetector.

        Returns (skip: bool, p_loss: float).
        Uses pair-specific model and threshold. Falls back to V9 generic if no model.
        """
        # Check if V9.5 model exists for this pair
        if not self.v95_enabled or pair not in self.v95_loss_detectors:
            # Fallback to V9 generic
            return self.score_loss_detector(signal, pair_ta, btc_ctx, conviction_pred)

        model = self.v95_loss_detectors[pair]
        threshold = self.v95_thresholds.get(pair, 0.55)

        features = {
            'cs_conf': signal['confidence'],
            'cs_pred_mag': abs(signal['prediction']),
            'cs_macro_score': self.macro_score,
            'cs_risk_off': self.risk_off_mult,
            'cs_regime_bull': 1.0 if self.regime == 'BULL' else 0.0,
            'cs_regime_bear': 1.0 if self.regime == 'BEAR' else 0.0,
            'cs_regime_range': 1.0 if self.regime == 'RANGE' else 0.0,
            'cs_atr_pct': signal['atr_pct'],
            'cs_n_open': signal.get('n_open', 0),
            'cs_pred_sign': float(signal['direction']),
            'ld_conviction_pred': conviction_pred,
            **pair_ta,
            **btc_ctx,
            'ld_hour': float(datetime.now(timezone.utc).hour) / 24.0,
            'ld_tp_sl_ratio': 0.03 / 0.015,
        }

        df_feat = pd.DataFrame([features])
        cols = [c for c in self.v95_fcols if c in df_feat.columns]
        if not cols:
            return False, 0.0

        p_loss = float(model.predict_proba(df_feat[cols])[0][1])

        if p_loss > threshold:
            return True, p_loss

        return False, p_loss

    def update_regime(self, exchange: ccxt.Exchange):
        """Detecta regimen de mercado desde BTC daily."""
        try:
            ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1d', limit=250)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            c = df['close']
            ema20 = ta.ema(c, length=20)
            ema50 = ta.ema(c, length=50)
            ret_20 = c.pct_change(20)

            last_c = c.iloc[-1]
            last_ema20 = ema20.iloc[-1]
            last_ema50 = ema50.iloc[-1]
            last_ret20 = ret_20.iloc[-1]

            if last_c > last_ema50 and last_ema20 > last_ema50 and last_ret20 > 0.05:
                self.regime = 'BULL'
            elif last_c < last_ema50 and last_ema20 < last_ema50 and last_ret20 < -0.05:
                self.regime = 'BEAR'
            else:
                self.regime = 'RANGE'

            self.regime_updated = datetime.now(timezone.utc)
            logger.info(f"[ML] Regime: {self.regime} (BTC ${last_c:,.0f}, "
                        f"EMA20 ${last_ema20:,.0f}, EMA50 ${last_ema50:,.0f}, "
                        f"ret20d {last_ret20:+.1%})")
        except Exception as e:
            logger.error(f"[ML] Error actualizando regime: {e}")

    # =========================================================================
    # V8.4: MACRO INTELLIGENCE
    # =========================================================================
    def update_macro(self):
        """Downloads macro data and computes macro score + risk-off for today.

        Called once per day. Downloads DXY, Gold, SPY, TNX, ETH/BTC via
        yfinance/ccxt, computes macro features, and scores with MacroScorer.
        """
        if not self.v84_enabled or self.macro_scorer is None:
            self.macro_score = 0.5
            self.risk_off_mult = 1.0
            return

        try:
            from macro_data import download_all_macro, compute_macro_features

            macro = download_all_macro()
            macro_feat = compute_macro_features(
                macro.get('dxy'), macro.get('gold'), macro.get('spy'),
                macro.get('tnx'), macro.get('ethbtc'),
            )

            if macro_feat is None or len(macro_feat) == 0:
                logger.warning("[ML] No macro features - using defaults")
                self.macro_score = 0.5
                self.risk_off_mult = 1.0
                return

            # Get today's macro features
            today_feat = macro_feat.iloc[-1:]
            cols = [c for c in self.macro_fcols if c in today_feat.columns]
            if not cols:
                logger.warning("[ML] Macro feature columns mismatch")
                self.macro_score = 0.5
                self.risk_off_mult = 1.0
                return

            X = today_feat[cols].fillna(0)
            self.macro_score = float(self.macro_scorer.predict_proba(X)[:, 1][0])

            # Compute risk-off multiplier for today
            if ML_RISKOFF_ENABLED:
                self.risk_off_mult = self._compute_risk_off(macro_feat.iloc[-1])
            else:
                self.risk_off_mult = 1.0

            self.macro_updated = datetime.now(timezone.utc)
            logger.info(f"[ML] Macro: score={self.macro_score:.3f} | "
                        f"risk_off_mult={self.risk_off_mult:.2f}")

        except Exception as e:
            logger.error(f"[ML] Error updating macro: {e}")
            self.macro_score = 0.5
            self.risk_off_mult = 1.0

    def _compute_risk_off(self, row) -> float:
        """Compute risk-off sizing multiplier for a single day.

        Stricter thresholds than V8.3 (~10% of days vs 21%).
        Returns multiplier 0.3-1.0 (1.0 = normal).
        """
        mult = 1.0
        dxy5 = row.get('dxy_ret_5d', 0) or 0
        spy5 = row.get('spy_ret_5d', 0) or 0
        gsr = row.get('gold_spy_ratio', 0) or 0
        dxy20 = row.get('dxy_ret_20d', 0) or 0
        spy20 = row.get('spy_ret_20d', 0) or 0

        # SEVERE: DXY up >2% AND SPY down >2% in 5d
        if dxy5 > 0.02 and spy5 < -0.02:
            mult = min(mult, 0.3)
        elif dxy5 > 0.015 and spy5 < -0.015:
            mult = min(mult, 0.5)

        # SEVERE: Massive flight to safety
        if gsr > 0.04:
            mult = min(mult, 0.3)
        elif gsr > 0.03:
            mult = min(mult, 0.5)

        # SEVERE: Dollar surging >3% in 5d
        if dxy5 > 0.03:
            mult = min(mult, 0.3)

        # STRUCTURAL: DXY up >4% in 20d AND SPY down >4% in 20d
        if dxy20 > 0.04 and spy20 < -0.04:
            mult = min(mult, 0.4)

        return mult

    def get_adaptive_threshold(self) -> float:
        """Get confidence threshold adjusted by macro score."""
        if not self.v84_enabled:
            return ML_SIGNAL_THRESHOLD

        # Linear interpolation: score=0 -> max, score=1 -> min
        thresh = ML_ADAPTIVE_THRESH_MAX - (ML_ADAPTIVE_THRESH_MAX - ML_ADAPTIVE_THRESH_MIN) * self.macro_score
        return max(ML_ADAPTIVE_THRESH_MIN, min(ML_ADAPTIVE_THRESH_MAX, thresh))

    def get_sizing_multiplier(self) -> float:
        """Get combined sizing multiplier (ML + risk-off)."""
        if not self.v84_enabled:
            return 1.0

        # ML sizing: score=0 -> min, score=1 -> max
        ml_mult = ML_SIZING_MIN + (ML_SIZING_MAX - ML_SIZING_MIN) * self.macro_score

        # Combined with risk-off
        combined = ml_mult * self.risk_off_mult
        return max(0.2, min(2.0, combined))

    # =========================================================================
    # V8.5: CONVICTION SCORING
    # =========================================================================
    def score_conviction(self, conf, pred_mag, atr_pct, n_open, direction):
        """Score a trade candidate with ConvictionScorer.

        Returns (skip: bool, conviction_mult: float).
        skip=True means the trade should be skipped (predicted negative PnL).
        conviction_mult is the sizing multiplier [0.3, 1.8].
        """
        if not self.v85_enabled or self.conviction_scorer is None:
            return False, 1.0

        features = pd.DataFrame([{
            'cs_conf': conf,
            'cs_pred_mag': pred_mag,
            'cs_macro_score': self.macro_score,
            'cs_risk_off': self.risk_off_mult,
            'cs_regime_bull': 1.0 if self.regime == 'BULL' else 0.0,
            'cs_regime_bear': 1.0 if self.regime == 'BEAR' else 0.0,
            'cs_regime_range': 1.0 if self.regime == 'RANGE' else 0.0,
            'cs_atr_pct': atr_pct,
            'cs_n_open': n_open,
            'cs_pred_sign': float(direction),
        }])

        cols = [c for c in self.conviction_fcols if c in features.columns]
        if not cols:
            return False, 1.0

        pred_pnl = self.conviction_scorer.predict(features[cols])[0]

        # Skip trades with clearly negative conviction
        if pred_pnl < -ML_CONVICTION_SKIP_MULT * self.conviction_pred_std:
            return True, 0.0

        # Map to sizing multiplier via sigmoid
        z = pred_pnl / self.conviction_pred_std if self.conviction_pred_std > 1e-8 else 0
        s = expit(z)  # [0, 1]
        sizing_range = ML_CONVICTION_SIZING_MAX - ML_CONVICTION_SIZING_MIN
        conv_mult = ML_CONVICTION_SIZING_MIN + sizing_range * s

        return False, conv_mult

    # =========================================================================
    # V13.01: BTC V2 FEATURES (54 features)
    # =========================================================================
    def compute_features_btc_v2(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computa features para BTC V2 - 54 features especializadas."""
        feat = pd.DataFrame(index=df.index)
        c, h, l, o, v = df['close'], df['high'], df['low'], df['open'], df['volume']

        # Returns (7)
        for p in [1, 2, 3, 5, 10, 20, 50]:
            feat[f'ret_{p}'] = c.pct_change(p)

        # ATR (2)
        feat['atr14'] = ta.atr(h, l, c, length=14)
        feat['atr_r'] = feat['atr14'] / feat['atr14'].rolling(50).mean()

        # Volatility (3)
        feat['vol5'] = c.pct_change().rolling(5).std()
        feat['vol20'] = c.pct_change().rolling(20).std()
        feat['vol_ratio'] = feat['vol5'] / (feat['vol20'] + 1e-10)

        # RSI (3)
        feat['rsi14'] = ta.rsi(c, length=14)
        feat['rsi7'] = ta.rsi(c, length=7)
        feat['rsi21'] = ta.rsi(c, length=21)

        # StochRSI (2)
        sr = ta.stochrsi(c, length=14, rsi_length=14, k=3, d=3)
        if sr is not None:
            feat['srsi_k'] = sr.iloc[:, 0]
            feat['srsi_d'] = sr.iloc[:, 1]

        # MACD (3)
        macd = ta.macd(c, fast=12, slow=26, signal=9)
        if macd is not None:
            feat['macd'] = macd.iloc[:, 0]
            feat['macd_h'] = macd.iloc[:, 1]
            feat['macd_s'] = macd.iloc[:, 2]

        # ROC (3)
        feat['roc5'] = ta.roc(c, length=5)
        feat['roc10'] = ta.roc(c, length=10)
        feat['roc20'] = ta.roc(c, length=20)

        # EMA distance (5)
        for el in [8, 21, 55, 100, 200]:
            e = ta.ema(c, length=el)
            feat[f'ema{el}_d'] = (c - e) / e * 100

        # EMA slopes (3)
        feat['ema8_sl'] = ta.ema(c, length=8).pct_change(3) * 100
        feat['ema21_sl'] = ta.ema(c, length=21).pct_change(5) * 100
        feat['ema55_sl'] = ta.ema(c, length=55).pct_change(5) * 100

        # Bollinger Bands (2)
        bb = ta.bbands(c, length=20, std=2.0)
        if bb is not None:
            bw = bb.iloc[:, 2] - bb.iloc[:, 0]
            feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw
            feat['bb_w'] = bw / bb.iloc[:, 1] * 100

        # Volume (2)
        feat['vr'] = v / v.rolling(20).mean()
        feat['vr5'] = v.rolling(5).mean() / v.rolling(20).mean()

        # Candle patterns (4)
        feat['spr'] = (h - l) / c * 100
        feat['body'] = abs(c - o) / (h - l + 1e-10)
        feat['upper_wick'] = (h - np.maximum(c, o)) / (h - l + 1e-10)
        feat['lower_wick'] = (np.minimum(c, o) - l) / (h - l + 1e-10)

        # ADX (4)
        ax = ta.adx(h, l, c, length=14)
        if ax is not None:
            feat['adx'] = ax.iloc[:, 0]
            feat['dip'] = ax.iloc[:, 1]
            feat['dim'] = ax.iloc[:, 2]
            feat['di_diff'] = feat['dip'] - feat['dim']

        # Choppiness
        chop = ta.chop(h, l, c, length=14)
        if chop is not None:
            feat['chop'] = chop

        # Time features (4)
        hr = df.index.hour
        dw = df.index.dayofweek
        feat['h_s'] = np.sin(2 * np.pi * hr / 24)
        feat['h_c'] = np.cos(2 * np.pi * hr / 24)
        feat['d_s'] = np.sin(2 * np.pi * dw / 7)
        feat['d_c'] = np.cos(2 * np.pi * dw / 7)

        # Lag features (6)
        feat['ret1_lag1'] = feat['ret_1'].shift(1)
        feat['rsi14_lag1'] = feat['rsi14'].shift(1)
        feat['ret1_lag2'] = feat['ret_1'].shift(2)
        feat['rsi14_lag2'] = feat['rsi14'].shift(2)
        feat['ret1_lag3'] = feat['ret_1'].shift(3)
        feat['rsi14_lag3'] = feat['rsi14'].shift(3)

        return feat

    # =========================================================================
    # V7: FEATURES + SIGNALS
    # =========================================================================
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computa features - IDENTICO a V7 training."""
        feat = pd.DataFrame(index=df.index)
        c, h, l, v = df['close'], df['high'], df['low'], df['volume']

        for p in [1, 3, 5, 10, 20]:
            feat[f'ret_{p}'] = c.pct_change(p)

        feat['atr14'] = ta.atr(h, l, c, length=14)
        feat['atr_r'] = feat['atr14'] / feat['atr14'].rolling(50).mean()
        feat['vol5'] = c.pct_change().rolling(5).std()
        feat['vol20'] = c.pct_change().rolling(20).std()

        feat['rsi14'] = ta.rsi(c, length=14)
        feat['rsi7'] = ta.rsi(c, length=7)
        sr = ta.stochrsi(c, length=14, rsi_length=14, k=3, d=3)
        if sr is not None:
            feat['srsi_k'] = sr.iloc[:, 0]
        macd = ta.macd(c, fast=12, slow=26, signal=9)
        if macd is not None:
            feat['macd_h'] = macd.iloc[:, 1]
        feat['roc5'] = ta.roc(c, length=5)
        feat['roc20'] = ta.roc(c, length=20)

        for el in [8, 21, 55, 100, 200]:
            e = ta.ema(c, length=el)
            feat[f'ema{el}_d'] = (c - e) / e * 100
        feat['ema8_sl'] = ta.ema(c, length=8).pct_change(3) * 100
        feat['ema55_sl'] = ta.ema(c, length=55).pct_change(5) * 100

        bb = ta.bbands(c, length=20, std=2.0)
        if bb is not None:
            bw = bb.iloc[:, 2] - bb.iloc[:, 0]
            feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw
            feat['bb_w'] = bw / bb.iloc[:, 1] * 100

        feat['vr'] = v / v.rolling(20).mean()
        feat['spr'] = (h - l) / c * 100
        feat['body'] = abs(c - df['open']) / (h - l + 1e-10)

        ax = ta.adx(h, l, c, length=14)
        if ax is not None:
            feat['adx'] = ax.iloc[:, 0]
            feat['dip'] = ax.iloc[:, 1]
            feat['dim'] = ax.iloc[:, 2]

        hr = df.index.hour
        dw = df.index.dayofweek
        feat['h_s'] = np.sin(2 * np.pi * hr / 24)
        feat['h_c'] = np.cos(2 * np.pi * hr / 24)
        feat['d_s'] = np.sin(2 * np.pi * dw / 7)
        feat['d_c'] = np.cos(2 * np.pi * dw / 7)

        return feat

    def get_atr_pct(self, df: pd.DataFrame) -> float:
        """Calcula ATR% actual (para trailing stop distance)."""
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        if atr is not None and len(atr) > 0 and not np.isnan(atr.iloc[-1]):
            return atr.iloc[-1] / df['close'].iloc[-1]
        return 0.02  # default 2%

    def generate_signals(self, exchange: ccxt.Exchange, open_pairs: set = None) -> list:
        """Genera senales para todos los pares.

        V8.4: Uses adaptive threshold from macro score.
        V8.5: ConvictionScorer filters and adjusts per-trade sizing.
        Returns sizing_mult in each signal for portfolio manager.
        """
        if open_pairs is None:
            open_pairs = set()

        # V8.4: adaptive threshold
        thresh = self.get_adaptive_threshold()
        macro_sizing = self.get_sizing_multiplier()

        n_open = len(open_pairs)
        signals = []
        for pair in self.pairs:
            if pair in open_pairs:
                continue
            try:
                ohlcv = exchange.fetch_ohlcv(pair, ML_TIMEFRAME, limit=500)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df[~df.index.duplicated(keep='first')].sort_index()

                if len(df) < 250:
                    continue

                # V13.01: Use BTC V2 model for BTC/USDT if enabled
                if pair == 'BTC/USDT' and self.btc_v2_enabled:
                    feat = self.compute_features_btc_v2(df)
                    feat = feat.replace([np.inf, -np.inf], np.nan)
                    cols = [c for c in self.btc_v2_fcols if c in feat.columns]
                    last_feat = feat[cols].iloc[-1:].fillna(0)

                    # Scale features if scaler exists
                    if self.btc_v2_scaler is not None:
                        last_feat_scaled = pd.DataFrame(
                            self.btc_v2_scaler.transform(last_feat),
                            columns=cols
                        )
                    else:
                        last_feat_scaled = last_feat

                    pred = self.btc_v2_model.predict(last_feat_scaled)[0]
                    ps = self.btc_v2_pred_std
                else:
                    # Standard V7 model
                    feat = self.compute_features(df)
                    feat = feat.replace([np.inf, -np.inf], np.nan)

                    if self.feature_cols:
                        cols = [c for c in self.feature_cols if c in feat.columns]
                    else:
                        cols = list(feat.columns)

                    last_feat = feat[cols].iloc[-1:].fillna(0)
                    model = self.models[pair]
                    pred = model.predict(last_feat)[0]

                    ps = self.pred_stds.get(pair, None)
                    if ps is None or ps < 1e-8:
                        recent = feat[cols].tail(100).fillna(0)
                        recent_preds = model.predict(recent)
                        ps = np.std(recent_preds)
                        if ps < 1e-8:
                            ps = 0.01
                        self.pred_stds[pair] = ps

                conf = abs(pred) / ps
                if conf < thresh:
                    continue

                direction = 1 if pred > 0 else -1

                # Regime filter (V7 rules, NO override from V8.4)
                if self.regime == 'BULL' and direction == -1:
                    continue
                if self.regime == 'BEAR' and direction == 1:
                    continue

                atr_pct = self.get_atr_pct(df)

                # V8.5: ConvictionScorer per-trade filter + sizing
                skip, conv_mult = self.score_conviction(
                    conf, abs(pred), atr_pct, n_open, direction,
                )
                if skip:
                    logger.info(f"[ML] {pair} skipped by ConvictionScorer")
                    continue

                # Combined sizing: macro * conviction
                total_sizing = macro_sizing * conv_mult
                total_sizing = max(0.2, min(2.5, total_sizing))

                signals.append({
                    'pair': pair,
                    'direction': direction,
                    'confidence': conf,
                    'prediction': pred,
                    'price': float(df['close'].iloc[-1]),
                    'atr_pct': atr_pct,
                    'sizing_mult': total_sizing,
                    'conviction_mult': conv_mult,
                })

            except Exception as e:
                logger.warning(f"[ML] Error generando senal {pair}: {e}")

        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals

    def generate_dual_signals(self, exchange: ccxt.Exchange,
                              open_pairs_v9: set = None,
                              open_pairs_shadow: set = None,
                              btc_df: pd.DataFrame = None) -> tuple:
        """Generate both V9 and V8.5 shadow signals in a single pass.

        Fetches data once per pair, applies V8.5 pipeline, then V9 LossDetector.
        Returns (v9_signals, v85_shadow_signals).
        """
        if open_pairs_v9 is None:
            open_pairs_v9 = set()
        if open_pairs_shadow is None:
            open_pairs_shadow = set()

        thresh = self.get_adaptive_threshold()
        macro_sizing = self.get_sizing_multiplier()

        # BTC context (computed once)
        btc_ctx = self.compute_btc_context_live(btc_df)

        n_open_v9 = len(open_pairs_v9)
        n_open_shadow = len(open_pairs_shadow)

        v9_signals = []
        v85_signals = []

        for pair in self.pairs:
            try:
                ohlcv = exchange.fetch_ohlcv(pair, ML_TIMEFRAME, limit=500)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df[~df.index.duplicated(keep='first')].sort_index()

                if len(df) < 250:
                    continue

                # V13.01: Use BTC V2 model for BTC/USDT if enabled
                if pair == 'BTC/USDT' and self.btc_v2_enabled:
                    feat = self.compute_features_btc_v2(df)
                    feat = feat.replace([np.inf, -np.inf], np.nan)
                    cols = [c for c in self.btc_v2_fcols if c in feat.columns]
                    last_feat = feat[cols].iloc[-1:].fillna(0)

                    if self.btc_v2_scaler is not None:
                        last_feat_scaled = pd.DataFrame(
                            self.btc_v2_scaler.transform(last_feat),
                            columns=cols
                        )
                    else:
                        last_feat_scaled = last_feat

                    pred = self.btc_v2_model.predict(last_feat_scaled)[0]
                    ps = self.btc_v2_pred_std
                else:
                    feat = self.compute_features(df)
                    feat = feat.replace([np.inf, -np.inf], np.nan)

                    if self.feature_cols:
                        cols = [c for c in self.feature_cols if c in feat.columns]
                    else:
                        cols = list(feat.columns)

                    last_feat = feat[cols].iloc[-1:].fillna(0)
                    model = self.models[pair]
                    pred = model.predict(last_feat)[0]

                    ps = self.pred_stds.get(pair, None)
                    if ps is None or ps < 1e-8:
                        recent = feat[cols].tail(100).fillna(0)
                        recent_preds = model.predict(recent)
                        ps = np.std(recent_preds)
                        if ps < 1e-8:
                            ps = 0.01
                        self.pred_stds[pair] = ps

                conf = abs(pred) / ps
                if conf < thresh:
                    continue

                direction = 1 if pred > 0 else -1

                if self.regime == 'BULL' and direction == -1:
                    continue
                if self.regime == 'BEAR' and direction == 1:
                    continue

                atr_pct = self.get_atr_pct(df)
                price = float(df['close'].iloc[-1])

                # V8.5: ConvictionScorer
                skip, conv_mult = self.score_conviction(
                    conf, abs(pred), atr_pct, n_open_v9, direction,
                )
                if skip:
                    logger.info(f"[ML] {pair} skipped by ConvictionScorer")
                    continue

                total_sizing = macro_sizing * conv_mult
                total_sizing = max(0.2, min(2.5, total_sizing))

                # Get conviction raw prediction for LossDetector
                conviction_pred = 0.0
                if self.conviction_scorer is not None:
                    cs_feat = pd.DataFrame([{
                        'cs_conf': conf, 'cs_pred_mag': abs(pred),
                        'cs_macro_score': self.macro_score,
                        'cs_risk_off': self.risk_off_mult,
                        'cs_regime_bull': 1.0 if self.regime == 'BULL' else 0.0,
                        'cs_regime_bear': 1.0 if self.regime == 'BEAR' else 0.0,
                        'cs_regime_range': 1.0 if self.regime == 'RANGE' else 0.0,
                        'cs_atr_pct': atr_pct, 'cs_n_open': n_open_v9,
                        'cs_pred_sign': float(direction),
                    }])
                    cs_cols = [c for c in self.conviction_fcols if c in cs_feat.columns]
                    if cs_cols:
                        conviction_pred = float(self.conviction_scorer.predict(cs_feat[cs_cols])[0])

                base_signal = {
                    'pair': pair, 'direction': direction,
                    'confidence': conf, 'prediction': pred,
                    'price': price, 'atr_pct': atr_pct,
                    'sizing_mult': total_sizing,
                    'conviction_mult': conv_mult,
                }

                # V8.5 shadow signal (passes ConvictionScorer, no LossDetector)
                if pair not in open_pairs_shadow:
                    v85_signals.append(dict(base_signal))

                # V9/V9.5: Apply LossDetector filter
                if pair not in open_pairs_v9:
                    pair_ta = self.compute_pair_ta_live(df)
                    # Use V9.5 per-pair if available, else V9 generic
                    skip_ld, p_loss = self.score_loss_detector_v95(
                        pair,
                        {**base_signal, 'n_open': n_open_v9},
                        pair_ta, btc_ctx, conviction_pred,
                    )
                    version = "V9.5" if self.v95_enabled and pair in self.v95_loss_detectors else "V9"
                    thresh = self.v95_thresholds.get(pair, self.loss_threshold) if self.v95_enabled else self.loss_threshold
                    if skip_ld:
                        logger.info(f"[ML] {pair} skipped by LossDetector {version} "
                                    f"(P={p_loss:.3f} > {thresh:.2f})")
                        continue
                    v9_signals.append(dict(base_signal))

            except Exception as e:
                logger.warning(f"[ML] Error generando senal {pair}: {e}")

        v9_signals.sort(key=lambda x: x['confidence'], reverse=True)
        v85_signals.sort(key=lambda x: x['confidence'], reverse=True)
        return v9_signals, v85_signals
