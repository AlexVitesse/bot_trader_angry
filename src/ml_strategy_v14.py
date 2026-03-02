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

                self.ensemble_models[asset] = models
                count += 1

                source = f" (using {model_name.upper()} model)" if model_name != asset.lower() else ""
                logger.info(f"[V14] Loaded {asset} ensemble{source}")

            except Exception as e:
                logger.error(f"[V14] Error cargando {asset}: {e}")

        # Verificar modelos BTC
        btc_models_path = strategies_dir / 'btc_v14' / 'models' / 'context_long.pkl'
        btc_ok = btc_models_path.exists()
        logger.info(f"[V14] BTC models: {'OK' if btc_ok else 'MISSING'}")

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

        high_20 = h.rolling(20).max()
        low_20 = l.rolling(20).min()
        range_size = high_20 - low_20
        feat['range_pos'] = float((c.iloc[-1] - low_20.iloc[-1]) / range_size.iloc[-1]) if range_size.iloc[-1] > 0 else 0.5

        consec_up = (c > c.shift(1)).rolling(10).sum()
        consec_down = (c < c.shift(1)).rolling(10).sum()
        feat['consec_up'] = float(consec_up.iloc[-1])
        feat['consec_down'] = float(consec_down.iloc[-1])

        return feat

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
        """Prediccion ensemble voting."""
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

                # BTC V14: Regimen + Setups (SIN filtro ML)
                if config['type'] == 'btc_v14':
                    feat = self.compute_btc_features(df)
                    regime = self.detect_btc_regime(feat)
                    strategy, setup_name = self.detect_btc_setup(feat, regime)

                    if strategy:
                        direction = 1 if 'LONG' in strategy.value else -1
                        params = BTC_STRATEGY_PARAMS.get(strategy, {'tp': 0.03, 'sl': 0.015})

                        asset_signals.append({
                            'pair': symbol,
                            'direction': direction,
                            'confidence': 1.0,
                            'setup': f"{regime.value}:{setup_name}",
                            'price': float(df['close'].iloc[-1]),
                            'tp_pct': params['tp'],
                            'sl_pct': params['sl'],
                        })
                        logger.info(f"[V14] {asset}: {regime.value} | {setup_name} | {'LONG' if direction == 1 else 'SHORT'}")
                    else:
                        logger.debug(f"[V14] {asset}: {regime.value} | No setup")

                # ETH: Setups simples
                elif config['type'] == 'setups' and asset == 'ETH':
                    eth_signals = self.check_eth_setups(df)
                    for sig in eth_signals:
                        direction = 1 if sig['direction'] == 'LONG' else -1
                        asset_signals.append({
                            'pair': symbol,
                            'direction': direction,
                            'confidence': sig['prob'],
                            'setup': sig['setup'],
                            'price': float(df['close'].iloc[-1]),
                            'tp_pct': config['tp'],
                            'sl_pct': config['sl'],
                        })
                    if not eth_signals:
                        logger.debug(f"[V14] ETH: No setup")

                # Ensemble (DOGE/ADA/DOT/SOL + cross-pairs)
                elif config['type'] == 'ensemble' and asset in self.ensemble_models:
                    df_feat = self.compute_ensemble_features(df)
                    features = df_feat[self.features].iloc[-1].values
                    should_trade, prob = self.predict_ensemble(self.ensemble_models[asset], features)

                    if should_trade:
                        # Filtro especifico por modelo
                        model_base = config.get('model', asset).upper()
                        feat_dict = {col: df_feat[col].iloc[-1] for col in self.features}
                        filter_passed, filter_reason = self.check_model_filter(model_base, feat_dict)

                        if not filter_passed:
                            logger.info(f"[V14] {asset}: FILTERED by {model_base} filter ({filter_reason})")
                        else:
                            asset_signals.append({
                                'pair': symbol,
                                'direction': 1,  # Ensemble = LONG only
                                'confidence': prob,
                                'setup': 'ENSEMBLE_VOTE',
                                'price': float(df['close'].iloc[-1]),
                                'tp_pct': config['tp'],
                                'sl_pct': config['sl'],
                            })
                    else:
                        logger.debug(f"[V14] {asset}: No signal (prob={prob:.1%})")

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
