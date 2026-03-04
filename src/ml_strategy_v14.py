"""
ML Strategy V14 - Motor de Senales Ensemble
============================================
V14 combina multiples expertos:
- BTC: Regimen + Setups (sin ML filter - mejor rendimiento)
- ETH: Setups simples (RSI, volumen)
- DOGE/ADA/DOT/SOL: Ensemble voting (RF + GB + LR)
- Cross-pairs: Usan modelo base correspondiente

Validado con datos sinteticos y walk-forward validation.
"""

import logging
import numpy as np
import pandas as pd
import pandas_ta as pta
import ta
import joblib
import ccxt
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum

from config.settings import (
    ML_V14_EXPERTS, ML_V14_MODEL_FILTERS, ML_V14_FEATURES,
    ML_TIMEFRAME, PROJECT_ROOT,
)

logger = logging.getLogger(__name__)


# =============================================================================
# BTC V14 - Enums
# =============================================================================
class Regime(Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    VOLATILE = "VOLATILE"


class Strategy(Enum):
    TREND_FOLLOW_LONG = "TREND_FOLLOW_LONG"
    TREND_FOLLOW_SHORT = "TREND_FOLLOW_SHORT"
    MEAN_REVERSION_LONG = "MEAN_REVERSION_LONG"
    MEAN_REVERSION_SHORT = "MEAN_REVERSION_SHORT"
    BREAKOUT_LONG = "BREAKOUT_LONG"
    BREAKOUT_SHORT = "BREAKOUT_SHORT"


# BTC: TP/SL fijos (validado: 37% WR, +3490% PnL)
BTC_STRATEGY_PARAMS = {
    Strategy.TREND_FOLLOW_LONG: {'tp': 0.03, 'sl': 0.015},
    Strategy.TREND_FOLLOW_SHORT: {'tp': 0.03, 'sl': 0.015},
    Strategy.MEAN_REVERSION_LONG: {'tp': 0.03, 'sl': 0.015},
    Strategy.MEAN_REVERSION_SHORT: {'tp': 0.03, 'sl': 0.015},
    Strategy.BREAKOUT_LONG: {'tp': 0.03, 'sl': 0.015},
    Strategy.BREAKOUT_SHORT: {'tp': 0.03, 'sl': 0.015},
}


class MLStrategyV14:
    """Estrategia V14 con multiples expertos."""

    def __init__(self):
        self.experts = ML_V14_EXPERTS
        self.model_filters = ML_V14_MODEL_FILTERS
        self.features = ML_V14_FEATURES
        self.ensemble_models = {}  # {asset: {rf, gb, lr, scaler}}
        self.btc_models = {}       # {direction: {context, momentum, volume}}
        self.eth_models = {}       # {regime_detector, context_long, ..., volume_short}
        self.pairs = []
        self.regime = 'RANGE'
        self.regime_updated = None

    def load_models(self) -> int:
        """Carga modelos ensemble desde disco."""
        count = 0
        strategies_dir = PROJECT_ROOT / 'strategies'

        for asset, config in self.experts.items():
            if config['type'] != 'ensemble':
                continue

            # Determinar modelo base (puede ser diferente al asset)
            model_name = config.get('model', asset).lower()
            model_dir = strategies_dir / f'{model_name}_v14' / 'models'

            if not model_dir.exists():
                logger.warning(f"[V14] Modelo no encontrado: {model_dir}")
                continue

            try:
                models = {
                    'scaler': joblib.load(model_dir / 'scaler.pkl'),
                    'rf': joblib.load(model_dir / 'random_forest.pkl'),
                    'gb': joblib.load(model_dir / 'gradient_boosting.pkl'),
                }
                # LR es opcional (DOT no lo tiene)
                lr_path = model_dir / 'logistic_regression.pkl'
                if lr_path.exists():
                    models['lr'] = joblib.load(lr_path)

                # Modelos SHORT (opcionales - entrenados con train_ensemble_short.py)
                short_scaler = model_dir / 'scaler_short.pkl'
                short_rf = model_dir / 'random_forest_short.pkl'
                short_gb = model_dir / 'gradient_boosting_short.pkl'
                short_lr = model_dir / 'logistic_regression_short.pkl'
                if short_scaler.exists() and short_rf.exists() and short_gb.exists():
                    models['scaler_short'] = joblib.load(short_scaler)
                    models['rf_short'] = joblib.load(short_rf)
                    models['gb_short'] = joblib.load(short_gb)
                    if short_lr.exists():
                        models['lr_short'] = joblib.load(short_lr)

                self.ensemble_models[asset] = models
                count += 1

                has_short = 'rf_short' in models
                source = f" (using {model_name.upper()} model)" if model_name != asset.lower() else ""
                dirs = "LONG+SHORT" if has_short else "LONG only"
                logger.info(f"[V14] Loaded {asset} ensemble{source} [{dirs}]")

            except Exception as e:
                logger.error(f"[V14] Error cargando {asset}: {e}")

        # Cargar modelos BTC (context/momentum/volume por direccion)
        btc_model_dir = strategies_dir / 'btc_v14' / 'models'
        for direction in ['long', 'short']:
            dir_models = {}
            for mtype in ['context', 'momentum', 'volume']:
                pkl = btc_model_dir / f'{mtype}_{direction}.pkl'
                if pkl.exists():
                    dir_models[mtype] = joblib.load(pkl)
            if dir_models:
                self.btc_models[direction] = dir_models
        if self.btc_models:
            logger.info(f"[V14] BTC ensemble cargado: {list(self.btc_models.keys())}")
        else:
            logger.warning("[V14] BTC ensemble NO encontrado - usando solo reglas")

        # Cargar modelos ETH (regime_detector + context/momentum/volume por direccion)
        eth_model_dir = strategies_dir / 'ethusdt_v14' / 'models'
        if eth_model_dir.exists():
            eth_models = {}
            regime_pkl = eth_model_dir / 'regime_detector.pkl'
            if regime_pkl.exists():
                eth_models['regime_detector'] = joblib.load(regime_pkl)
            for direction in ['long', 'short']:
                for mtype in ['context', 'momentum', 'volume']:
                    pkl = eth_model_dir / f'ensemble_{mtype}_{direction}.pkl'
                    if pkl.exists():
                        eth_models[f'{mtype}_{direction}'] = joblib.load(pkl)
            if 'regime_detector' in eth_models:
                self.eth_models = eth_models
                logger.info(f"[V14] ETH models cargados: {list(eth_models.keys())}")
            else:
                logger.warning("[V14] ETH regime_detector no encontrado")

        self.pairs = list(self.experts.keys())
        logger.info(f"[V14] {count} ensemble models cargados, {len(self.pairs)} pares activos")
        return count

    # =========================================================================
    # BTC V14 FEATURES
    # =========================================================================
    def compute_btc_features(self, df: pd.DataFrame) -> dict:
        """Computa features para BTC V14."""
        c, h, l, v = df['close'], df['high'], df['low'], df['volume']
        feat = {}

        # ADX/DI
        adx_df = pta.adx(h, l, c, length=14)
        if adx_df is not None:
            feat['adx'] = float(adx_df.iloc[-1, 0])
            feat['di_plus'] = float(adx_df.iloc[-1, 1])
            feat['di_minus'] = float(adx_df.iloc[-1, 2])
            feat['di_diff'] = feat['di_plus'] - feat['di_minus']

        chop = pta.chop(h, l, c, length=14)
        feat['chop'] = float(chop.iloc[-1]) if chop is not None else 50

        ema20 = pta.ema(c, length=20)
        ema50 = pta.ema(c, length=50)
        ema200 = pta.ema(c, length=200)
        feat['ema20'] = float(ema20.iloc[-1]) if ema20 is not None else float(c.iloc[-1])
        feat['ema50'] = float(ema50.iloc[-1]) if ema50 is not None else float(c.iloc[-1])
        feat['ema200'] = float(ema200.iloc[-1]) if ema200 is not None else float(c.iloc[-1])
        feat['ema20_dist'] = (float(c.iloc[-1]) - feat['ema20']) / feat['ema20'] * 100
        feat['ema200_dist'] = (float(c.iloc[-1]) - feat['ema200']) / feat['ema200'] * 100
        feat['ema20_slope'] = float(ema20.pct_change(5).iloc[-1] * 100) if ema20 is not None else 0
        feat['ema50_slope'] = float(ema50.pct_change(10).iloc[-1] * 100) if ema50 is not None else 0

        atr = pta.atr(h, l, c, length=14)
        feat['atr'] = float(atr.iloc[-1]) if atr is not None else 0
        feat['atr_pct'] = feat['atr'] / float(c.iloc[-1]) * 100

        bb = pta.bbands(c, length=20)
        if bb is not None:
            feat['bb_width'] = float((bb.iloc[-1, 2] - bb.iloc[-1, 0]) / bb.iloc[-1, 1] * 100)
            bb_range = bb.iloc[:, 2] - bb.iloc[:, 0]
            feat['bb_pct'] = float((c.iloc[-1] - bb.iloc[-1, 0]) / bb_range.iloc[-1]) if bb_range.iloc[-1] > 0 else 0.5

        rsi14 = pta.rsi(c, length=14)
        rsi7 = pta.rsi(c, length=7)
        feat['rsi14'] = float(rsi14.iloc[-1]) if rsi14 is not None else 50
        feat['rsi7'] = float(rsi7.iloc[-1]) if rsi7 is not None else 50

        stoch = pta.stoch(h, l, c, k=14, d=3)
        feat['stoch_k'] = float(stoch.iloc[-1, 0]) if stoch is not None else 50

        feat['ret_5'] = float(c.pct_change(5).iloc[-1] * 100)
        feat['ret_20'] = float(c.pct_change(20).iloc[-1] * 100)

        vol_ma = v.rolling(20).mean()
        feat['vol_ratio'] = float(v.iloc[-1] / vol_ma.iloc[-1]) if vol_ma.iloc[-1] > 0 else 1
        vol_trend = v.rolling(5).mean() / vol_ma
        feat['vol_trend'] = float(vol_trend.iloc[-1]) if not np.isnan(vol_trend.iloc[-1]) else 1
        obv = (np.sign(c.diff()) * v).cumsum()
        obv_slope = obv.pct_change(10) * 100
        feat['obv_slope'] = float(obv_slope.iloc[-1]) if not np.isnan(obv_slope.iloc[-1]) else 0

        high_20 = h.rolling(20).max()
        low_20 = l.rolling(20).min()
        range_size = high_20 - low_20
        feat['range_pos'] = float((c.iloc[-1] - low_20.iloc[-1]) / range_size.iloc[-1]) if range_size.iloc[-1] > 0 else 0.5

        consec_up = (c > c.shift(1)).rolling(10).sum()
        consec_down = (c < c.shift(1)).rolling(10).sum()
        feat['consec_up'] = float(consec_up.iloc[-1])
        feat['consec_down'] = float(consec_down.iloc[-1])

        return feat

    def predict_btc_confidence(self, feat: dict, direction: str) -> float:
        """Calcula confianza ensemble BTC para 'long' o 'short'. Retorna 0.5 si no hay modelos."""
        if direction not in self.btc_models:
            return 0.5
        weights = {'context': 0.4, 'momentum': 0.35, 'volume': 0.25}
        weighted_prob = 0.0
        total_weight = 0.0
        for mtype, weight in weights.items():
            mdata = self.btc_models[direction].get(mtype)
            if mdata is None:
                continue
            model = mdata['model']
            features = mdata['features']
            X = np.array([[feat.get(f, 0) for f in features]])
            if 'scaler' in mdata:
                X = mdata['scaler'].transform(X)
            prob = model.predict_proba(X)[0, 1]
            weighted_prob += prob * weight
            total_weight += weight
        return weighted_prob / total_weight if total_weight > 0 else 0.5

    # =========================================================================
    # ETH V14 FEATURES + PREDICCION
    # =========================================================================
    def compute_eth_features(self, df: pd.DataFrame) -> dict:
        """Computa features para ETH V14 (regime + ensemble)."""
        c, h, l, v = df['close'], df['high'], df['low'], df['volume']
        feat = {}

        # --- Regime features ---
        feat['adx'] = float(ta.trend.adx(h, l, c, window=14).iloc[-1])
        feat['di_plus'] = float(ta.trend.adx_pos(h, l, c, window=14).iloc[-1])
        feat['di_minus'] = float(ta.trend.adx_neg(h, l, c, window=14).iloc[-1])
        feat['di_diff'] = feat['di_plus'] - feat['di_minus']
        atr = ta.volatility.average_true_range(h, l, c, window=14)
        high_14 = h.rolling(14).max()
        low_14 = l.rolling(14).min()
        chop_raw = 100 * np.log10(atr.rolling(14).sum() / (high_14 - low_14 + 1e-10)) / np.log10(14)
        feat['chop'] = float(chop_raw.iloc[-1]) if not np.isnan(chop_raw.iloc[-1]) else 50.0
        feat['volatility'] = float(c.pct_change().rolling(20).std().iloc[-1])
        sma50 = c.rolling(50).mean()
        sma200 = c.rolling(200).mean()
        feat['trend'] = int(sma50.iloc[-1] > sma200.iloc[-1])

        # --- Context features ---
        feat['rsi'] = float(ta.momentum.rsi(c, window=14).iloc[-1])
        bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
        bb_range = bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1]
        feat['bb_pct'] = float((c.iloc[-1] - bb.bollinger_lband().iloc[-1]) / (bb_range + 1e-10))
        feat['trend_strength'] = float((c.iloc[-1] - sma50.iloc[-1]) / sma50.iloc[-1])

        # --- Momentum features ---
        feat['roc_5'] = float(c.pct_change(5).iloc[-1])
        feat['roc_10'] = float(c.pct_change(10).iloc[-1])
        feat['roc_20'] = float(c.pct_change(20).iloc[-1])
        macd = ta.trend.MACD(c)
        feat['macd_hist'] = float(macd.macd_diff().iloc[-1])
        feat['momentum'] = float(ta.momentum.roc(c, window=10).iloc[-1])

        # --- Volume features ---
        vol_sma = v.rolling(20).mean()
        feat['volume_ratio'] = float(v.iloc[-1] / vol_sma.iloc[-1]) if vol_sma.iloc[-1] > 0 else 1.0
        obv = ta.volume.on_balance_volume(c, v)
        obv_sma = obv.rolling(20).mean()
        feat['obv_trend'] = int(obv.iloc[-1] > obv_sma.iloc[-1])
        feat['mfi'] = float(ta.volume.money_flow_index(h, l, c, v, window=14).iloc[-1])

        return feat

    def predict_eth_signal(self, feat: dict) -> tuple:
        """Predice senal ETH usando regime detector + ensemble voting.
        Requiere 3/3 modelos de acuerdo (umbral estricto - modelos con AUC ~0.5).
        Returns: (direction, confidence) donde direction es 1 (LONG), -1 (SHORT), 0 (sin senal).
        """
        if 'regime_detector' not in self.eth_models:
            return 0, 0.0

        # Paso 1: detectar regimen
        regime_cols = ['adx', 'di_plus', 'di_minus', 'di_diff', 'chop', 'volatility', 'trend']
        X_regime = np.array([[feat.get(f, 0) for f in regime_cols]])
        regime = self.eth_models['regime_detector'].predict(X_regime)[0]

        context_cols = ['rsi', 'bb_pct', 'trend_strength']
        momentum_cols = ['roc_5', 'roc_10', 'roc_20', 'macd_hist', 'momentum']
        volume_cols = ['volume_ratio', 'obv_trend', 'mfi']

        # Paso 2: votar por cada direccion (umbral estricto: 3/3)
        ETH_MIN_PROB = 0.60
        for direction_str, direction_int in [('long', 1), ('short', -1)]:
            keys_cols = [
                (f'context_{direction_str}', context_cols),
                (f'momentum_{direction_str}', momentum_cols),
                (f'volume_{direction_str}', volume_cols),
            ]
            if not all(k in self.eth_models for k, _ in keys_cols):
                continue

            probs = []
            for key, cols in keys_cols:
                X = np.array([[feat.get(f, 0) for f in cols]])
                prob = self.eth_models[key].predict_proba(X)[0, 1]
                probs.append(prob)

            votes = sum(1 for p in probs if p > 0.5)
            avg_prob = sum(probs) / len(probs)

            # 3/3 agree AND avg prob above threshold
            if votes == 3 and avg_prob >= ETH_MIN_PROB:
                logger.info(f"[V14] ETH: {regime} | {'LONG' if direction_int == 1 else 'SHORT'} | prob={avg_prob:.2%} | 3/3 votos")
                return direction_int, avg_prob

        return 0, 0.0

    def detect_btc_regime(self, row: dict) -> Regime:
        """Detecta regimen BTC."""
        adx = row.get('adx', 20)
        di_diff = row.get('di_diff', 0)
        chop = row.get('chop', 50)
        atr_pct = row.get('atr_pct', 2)
        bb_width = row.get('bb_width', 5)
        ema20_slope = row.get('ema20_slope', 0)
        ema50_slope = row.get('ema50_slope', 0)

        if pd.isna(adx) or pd.isna(chop):
            return Regime.RANGE

        if atr_pct > 4 and bb_width > 8:
            return Regime.VOLATILE

        if adx > 25 and chop < 50:
            if di_diff > 5 and ema20_slope > 0:
                return Regime.TREND_UP
            elif di_diff < -5 and ema20_slope < 0:
                return Regime.TREND_DOWN

        if chop > 55 or adx < 20:
            return Regime.RANGE

        if ema50_slope > 0.5:
            return Regime.TREND_UP
        elif ema50_slope < -0.5:
            return Regime.TREND_DOWN

        return Regime.RANGE

    def detect_btc_setup(self, row: dict, regime: Regime):
        """Detecta setup BTC segun regimen."""
        rsi14 = row.get('rsi14', 50)
        bb_pct = row.get('bb_pct', 0.5)
        range_pos = row.get('range_pos', 0.5)
        ema20_dist = row.get('ema20_dist', 0)
        ema200_dist = row.get('ema200_dist', 0)
        vol_ratio = row.get('vol_ratio', 1)
        consec_up = row.get('consec_up', 0)
        consec_down = row.get('consec_down', 0)

        if pd.isna(rsi14):
            return None, None

        # TREND_UP: Pullbacks
        if regime == Regime.TREND_UP:
            if rsi14 < 40 and bb_pct < 0.3 and ema200_dist > 0:
                return Strategy.TREND_FOLLOW_LONG, 'PULLBACK_UPTREND'
            elif rsi14 < 30 and ema20_dist < -2:
                return Strategy.TREND_FOLLOW_LONG, 'OVERSOLD_UPTREND'

        # TREND_DOWN: Rallies
        elif regime == Regime.TREND_DOWN:
            if rsi14 > 60 and bb_pct > 0.7 and ema200_dist < 0:
                return Strategy.TREND_FOLLOW_SHORT, 'RALLY_DOWNTREND'
            elif rsi14 > 70 and ema20_dist > 2:
                return Strategy.TREND_FOLLOW_SHORT, 'OVERBOUGHT_DOWNTREND'

        # RANGE: Mean reversion
        elif regime == Regime.RANGE:
            if range_pos < 0.2 and rsi14 < 35:
                return Strategy.MEAN_REVERSION_LONG, 'SUPPORT_BOUNCE'
            elif range_pos > 0.8 and rsi14 > 65:
                return Strategy.MEAN_REVERSION_SHORT, 'RESISTANCE_REJECT'

        # VOLATILE: Breakouts
        elif regime == Regime.VOLATILE:
            if bb_pct > 1.0 and vol_ratio > 1.5 and consec_up >= 3:
                return Strategy.BREAKOUT_LONG, 'BREAKOUT_UP'
            elif bb_pct < 0 and vol_ratio > 1.5 and consec_down >= 3:
                return Strategy.BREAKOUT_SHORT, 'BREAKOUT_DOWN'

        return None, None

    # =========================================================================
    # ENSEMBLE FEATURES
    # =========================================================================
    def compute_ensemble_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computa features para ensemble simple."""
        df = df.copy()
        df['rsi'] = ta.momentum.rsi(df['close'], window=14) / 100
        macd = ta.trend.MACD(df['close'])
        df['macd_norm'] = macd.macd() / df['close']
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14) / 100
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        bb_range = bb.bollinger_hband() - bb.bollinger_lband()
        df['bb_pct'] = (df['close'] - bb.bollinger_lband()) / (bb_range + 1e-10)
        df['atr_pct'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14) / df['close']
        df['ret_3'] = df['close'].pct_change(3)
        df['ret_5'] = df['close'].pct_change(5)
        df['ret_10'] = df['close'].pct_change(10)
        df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['trend'] = (df['close'] > df['close'].rolling(50).mean()).astype(float)
        return df.dropna()

    def predict_ensemble(self, models: dict, features: np.ndarray) -> tuple:
        """Prediccion ensemble voting LONG.
        Returns: (should_trade, avg_prob)
        """
        X = models['scaler'].transform(features.reshape(1, -1))
        prob_rf = models['rf'].predict_proba(X)[0, 1]
        prob_gb = models['gb'].predict_proba(X)[0, 1]

        if 'lr' in models:
            prob_lr = models['lr'].predict_proba(X)[0, 1]
            votes = int(prob_rf > 0.5) + int(prob_gb > 0.5) + int(prob_lr > 0.5)
            avg_prob = (prob_rf + prob_gb + prob_lr) / 3
            return votes >= 2, avg_prob
        else:
            # Solo 2 modelos (DOT)
            votes = int(prob_rf > 0.5) + int(prob_gb > 0.5)
            return votes >= 2, (prob_rf + prob_gb) / 2

    def predict_ensemble_short(self, models: dict, features: np.ndarray) -> tuple:
        """Prediccion ensemble voting SHORT.
        Returns: (should_trade, avg_prob) o (False, 0.0) si no hay modelos SHORT.
        """
        if 'rf_short' not in models:
            return False, 0.0

        X = models['scaler_short'].transform(features.reshape(1, -1))
        prob_rf = models['rf_short'].predict_proba(X)[0, 1]
        prob_gb = models['gb_short'].predict_proba(X)[0, 1]

        if 'lr_short' in models:
            prob_lr = models['lr_short'].predict_proba(X)[0, 1]
            votes = int(prob_rf > 0.5) + int(prob_gb > 0.5) + int(prob_lr > 0.5)
            avg_prob = (prob_rf + prob_gb + prob_lr) / 3
            return votes >= 2, avg_prob
        else:
            votes = int(prob_rf > 0.5) + int(prob_gb > 0.5)
            return votes >= 2, (prob_rf + prob_gb) / 2

    def check_model_filter(self, model_name: str, features: dict) -> tuple:
        """Verifica si el trade pasa el filtro del modelo."""
        filter_config = self.model_filters.get(model_name)
        if filter_config is None:
            return True, ""

        feat_name = filter_config['filter']
        op = filter_config['op']
        threshold = filter_config['value']
        value = features.get(feat_name, 0)

        if op == '>':
            passed = value > threshold
        elif op == '<':
            passed = value < threshold
        elif op == '>=':
            passed = value >= threshold
        elif op == '<=':
            passed = value <= threshold
        else:
            passed = True

        reason = f"{feat_name}={value:.3f} {'>' if op == '>' else '<'} {threshold}" if not passed else ""
        return passed, reason

    # =========================================================================
    # ETH SETUPS
    # =========================================================================
    def check_eth_setups(self, df: pd.DataFrame) -> list:
        """Detecta setups para ETH."""
        signals = []
        rsi = ta.momentum.rsi(df['close'], window=14).iloc[-1]
        vol_ma = df['volume'].rolling(20).mean().iloc[-1]
        vol_ratio = df['volume'].iloc[-1] / vol_ma if vol_ma > 0 else 1
        price_change = df['close'].iloc[-1] > df['close'].iloc[-2]

        if rsi < 30:
            signals.append({'setup': 'RSI_OVERSOLD', 'direction': 'SHORT', 'prob': 0.65})
        if vol_ratio > 2 and price_change:
            signals.append({'setup': 'VOL_SPIKE_UP', 'direction': 'LONG', 'prob': 0.60})
        if vol_ratio > 2 and not price_change:
            signals.append({'setup': 'VOL_SPIKE_DOWN', 'direction': 'SHORT', 'prob': 0.60})

        return signals

    # =========================================================================
    # REGIME UPDATE
    # =========================================================================
    def update_regime(self, exchange: ccxt.Exchange):
        """Actualiza regimen de mercado desde BTC."""
        try:
            ohlcv = exchange.fetch_ohlcv('BTC/USDT', ML_TIMEFRAME, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            feat = self.compute_btc_features(df)
            btc_regime = self.detect_btc_regime(feat)
            self.regime = btc_regime.value
            self.regime_updated = datetime.now(timezone.utc)
            logger.info(f"[V14] Regime: {self.regime}")
        except Exception as e:
            logger.error(f"[V14] Error updating regime: {e}")

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    def generate_signals(self, exchange: ccxt.Exchange, open_pairs: set = None) -> list:
        """Genera senales para todos los pares V14."""
        if open_pairs is None:
            open_pairs = set()

        signals = []

        for asset, config in self.experts.items():
            symbol = config['symbol']
            if symbol in open_pairs:
                continue

            try:
                ohlcv = exchange.fetch_ohlcv(symbol, ML_TIMEFRAME, limit=100)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                asset_signals = []

                # BTC V14: Regimen + Setups + Ensemble ML confianza
                if config['type'] == 'btc_v14':
                    feat = self.compute_btc_features(df)
                    regime = self.detect_btc_regime(feat)
                    strategy, setup_name = self.detect_btc_setup(feat, regime)

                    if strategy:
                        direction = 1 if 'LONG' in strategy.value else -1
                        direction_str = 'long' if direction == 1 else 'short'
                        params = BTC_STRATEGY_PARAMS.get(strategy, {'tp': 0.03, 'sl': 0.015})

                        # Ensemble ML: calcular confianza (paso 3 arquitectura)
                        confidence = self.predict_btc_confidence(feat, direction_str)
                        SKIP_THRESHOLD = 0.35

                        if confidence < SKIP_THRESHOLD:
                            logger.info(f"[V14] BTC: {setup_name} RECHAZADO por ensemble (conf={confidence:.2%})")
                        else:
                            asset_signals.append({
                                'pair': symbol,
                                'direction': direction,
                                'confidence': confidence,
                                'setup': f"{regime.value}:{setup_name}",
                                'price': float(df['close'].iloc[-1]),
                                'tp_pct': params['tp'],
                                'sl_pct': params['sl'],
                            })
                            logger.info(f"[V14] BTC: {regime.value} | {setup_name} | {'LONG' if direction == 1 else 'SHORT'} | conf={confidence:.2%}")
                    else:
                        logger.info(f"[V14] {asset}: {regime.value} | No setup (rsi={feat.get('rsi14',0):.1f} bb_pct={feat.get('bb_pct',0):.2f})")

                # ETH V14: Regime detector + Ensemble ML
                elif config['type'] == 'eth_v14':
                    if self.eth_models:
                        feat = self.compute_eth_features(df)
                        direction, confidence = self.predict_eth_signal(feat)
                        if direction != 0:
                            direction_label = 'LONG' if direction == 1 else 'SHORT'
                            asset_signals.append({
                                'pair': symbol,
                                'direction': direction,
                                'confidence': confidence,
                                'setup': f'ETH_ML_{direction_label}',
                                'price': float(df['close'].iloc[-1]),
                                'tp_pct': config['tp'],
                                'sl_pct': config['sl'],
                            })
                        else:
                            rsi_eth = feat.get('rsi', 0)
                            vol_r = feat.get('volume_ratio', 1)
                            logger.info(f"[V14] ETH: No signal ML (rsi={rsi_eth:.1f} vol_ratio={vol_r:.2f})")
                    else:
                        logger.warning("[V14] ETH: modelos no cargados - sin senal")

                # Ensemble (DOGE/ADA/DOT/SOL + cross-pairs)
                elif config['type'] == 'ensemble' and asset in self.ensemble_models:
                    df_feat = self.compute_ensemble_features(df)
                    features = df_feat[self.features].iloc[-1].values
                    models = self.ensemble_models[asset]
                    model_base = config.get('model', asset).upper()
                    feat_dict = {col: df_feat[col].iloc[-1] for col in self.features}

                    # Probar LONG
                    long_trade, long_prob = self.predict_ensemble(models, features)
                    # Probar SHORT (si hay modelos entrenados)
                    short_trade, short_prob = self.predict_ensemble_short(models, features)

                    # Seleccionar la mejor senal (mayor prob gana si ambas activas)
                    if long_trade and short_trade:
                        # Ambas activas: tomar la de mayor confianza
                        if long_prob >= short_prob:
                            short_trade = False
                        else:
                            long_trade = False

                    if long_trade:
                        filter_passed, filter_reason = self.check_model_filter(model_base, feat_dict)
                        if not filter_passed:
                            logger.info(f"[V14] {asset}: LONG FILTERED by {model_base} ({filter_reason})")
                        else:
                            asset_signals.append({
                                'pair': symbol,
                                'direction': 1,
                                'confidence': long_prob,
                                'setup': 'ENSEMBLE_LONG',
                                'price': float(df['close'].iloc[-1]),
                                'tp_pct': config['tp'],
                                'sl_pct': config['sl'],
                            })
                    elif short_trade:
                        filter_passed, filter_reason = self.check_model_filter(model_base, feat_dict)
                        if not filter_passed:
                            logger.info(f"[V14] {asset}: SHORT FILTERED by {model_base} ({filter_reason})")
                        else:
                            asset_signals.append({
                                'pair': symbol,
                                'direction': -1,
                                'confidence': short_prob,
                                'setup': 'ENSEMBLE_SHORT',
                                'price': float(df['close'].iloc[-1]),
                                'tp_pct': config['tp'],
                                'sl_pct': config['sl'],
                            })
                    else:
                        has_short = 'rf_short' in models
                        short_info = f"{short_prob:.1%}" if has_short else "N/A"
                        logger.info(f"[V14] {asset}: No signal (long={long_prob:.1%} short={short_info}, votes insuf.)")

                signals.extend(asset_signals)

            except Exception as e:
                logger.warning(f"[V14] Error {asset}: {e}")

        # Ordenar por confianza
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals

    def get_atr_pct(self, df: pd.DataFrame) -> float:
        """Calcula ATR% actual."""
        atr = pta.atr(df['high'], df['low'], df['close'], length=14)
        if atr is not None and len(atr) > 0 and not np.isnan(atr.iloc[-1]):
            return atr.iloc[-1] / df['close'].iloc[-1]
        return 0.02
