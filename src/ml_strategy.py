"""
ML Strategy - Motor de Senales
==============================
Carga modelos LightGBM entrenados, computa features de velas 4h,
detecta regimen de mercado (BULL/BEAR/RANGE), y genera senales de trading.
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
)

logger = logging.getLogger(__name__)


class MLStrategy:
    """Genera senales de trading usando modelos ML."""

    def __init__(self):
        self.models = {}        # {pair: lgb model}
        self.pred_stds = {}     # {pair: float} para confidence
        self.feature_cols = []  # columnas de features
        self.regime = 'RANGE'
        self.regime_updated = None
        self.pairs = []         # pares con modelo cargado

    def load_models(self) -> int:
        """Carga modelos desde disco. Retorna cantidad cargada."""
        # Intentar cargar metadata
        meta_path = MODELS_DIR / 'v7_meta.json'
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.feature_cols = meta.get('feature_cols', [])
            self.pred_stds = meta.get('pred_stds', {})
            logger.info(f"[ML] Metadata cargada: {len(self.feature_cols)} features")

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
        logger.info(f"[ML] {count} modelos cargados")
        return count

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

        Args:
            exchange: ccxt exchange para fetch de candles
            open_pairs: set de pares ya abiertos (para no duplicar)

        Returns:
            Lista de dicts: [{'pair', 'direction', 'confidence', 'price', 'atr_pct'}, ...]
        """
        if open_pairs is None:
            open_pairs = set()

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

                # Usar columnas del modelo
                if self.feature_cols:
                    cols = [c for c in self.feature_cols if c in feat.columns]
                else:
                    cols = list(feat.columns)

                last_feat = feat[cols].iloc[-1:].fillna(0)

                model = self.models[pair]
                pred = model.predict(last_feat)[0]

                # Confidence = abs(pred) / pred_std
                ps = self.pred_stds.get(pair, None)
                if ps is None or ps < 1e-8:
                    # Estimar pred_std de las ultimas 100 predicciones
                    recent = feat[cols].tail(100).fillna(0)
                    recent_preds = model.predict(recent)
                    ps = np.std(recent_preds)
                    if ps < 1e-8:
                        ps = 0.01
                    self.pred_stds[pair] = ps

                conf = abs(pred) / ps
                if conf < ML_SIGNAL_THRESHOLD:
                    continue

                direction = 1 if pred > 0 else -1

                # Filtro de regimen
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
                })

            except Exception as e:
                logger.warning(f"[ML] Error generando senal {pair}: {e}")

        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals
