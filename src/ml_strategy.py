"""
ML Strategy - Motor de Senales V8.4
====================================
V7 base: LightGBM per-pair models, regime detection, trailing stops.
V8.4 adds: Macro intelligence (adaptive threshold + ML sizing + soft risk-off).

Macro layer:
  1. MacroScorer (LightGBM): predicts daily "macro quality for crypto" [0,1]
  2. Adaptive Threshold: adjusts confidence threshold by macro score
  3. ML Sizing: scales position size by macro score [0.3x to 1.8x]
  4. Soft Risk-Off: reduces sizing on extreme macro days (not regime override)
"""

import json
import logging
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
import ccxt
from datetime import datetime, timezone

from config.settings import (
    MODELS_DIR, ML_PAIRS, ML_SIGNAL_THRESHOLD, ML_TIMEFRAME,
    ML_LEVERAGE, ML_RISK_PER_TRADE,
    ML_V84_ENABLED, ML_ADAPTIVE_THRESH_MIN, ML_ADAPTIVE_THRESH_MAX,
    ML_SIZING_MIN, ML_SIZING_MAX, ML_RISKOFF_ENABLED,
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
        Returns sizing_mult in each signal for portfolio manager.
        """
        if open_pairs is None:
            open_pairs = set()

        # V8.4: adaptive threshold
        thresh = self.get_adaptive_threshold()
        sizing_mult = self.get_sizing_multiplier()

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

                signals.append({
                    'pair': pair,
                    'direction': direction,
                    'confidence': conf,
                    'prediction': pred,
                    'price': float(df['close'].iloc[-1]),
                    'atr_pct': atr_pct,
                    'sizing_mult': sizing_mult,
                })

            except Exception as e:
                logger.warning(f"[ML] Error generando senal {pair}: {e}")

        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals
