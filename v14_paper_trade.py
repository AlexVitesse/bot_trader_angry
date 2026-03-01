"""
V14 Paper Trading Script - TODOS LOS EXPERTOS
- BTC: Régimen + Setups + Ensemble ML (tal cual backtest)
- ETH: Setups simples
- DOGE/ADA/DOT: Ensemble voting

Uso: python v14_paper_trade.py
"""

import time
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum

import pandas as pd
import numpy as np
import ccxt
import ta
import pandas_ta as pta
import joblib

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACION
# =============================================================================
TIMEFRAME = '4h'
CHECK_INTERVAL = 60
LOG_FILE = 'logs/v14_paper_trade.log'
SIGNALS_FILE = 'data/v14_paper_signals.json'
INITIAL_CAPITAL = 100

# Expertos V14.1 (16 pares - solo aprobados)
EXPERTS = {
    # === ORIGINALES (5) ===
    'BTC': {'symbol': 'BTC/USDT', 'type': 'btc_v14', 'tp': 0.04, 'sl': 0.015},
    'ETH': {'symbol': 'ETH/USDT', 'type': 'setups', 'tp': 0.04, 'sl': 0.02},
    'DOGE': {'symbol': 'DOGE/USDT', 'type': 'ensemble', 'tp': 0.06, 'sl': 0.04},
    'ADA': {'symbol': 'ADA/USDT', 'type': 'ensemble', 'tp': 0.06, 'sl': 0.04},
    'DOT': {'symbol': 'DOT/USDT', 'type': 'ensemble', 'tp': 0.05, 'sl': 0.03},
    # === MODELO ADA - Smart Contracts (4) ===
    'SOL': {'symbol': 'SOL/USDT', 'type': 'ensemble', 'tp': 0.06, 'sl': 0.04, 'model': 'ada'},
    'ATOM': {'symbol': 'ATOM/USDT', 'type': 'ensemble', 'tp': 0.06, 'sl': 0.04, 'model': 'ada'},
    'AVAX': {'symbol': 'AVAX/USDT', 'type': 'ensemble', 'tp': 0.06, 'sl': 0.04, 'model': 'ada'},
    'POL': {'symbol': 'POL/USDT', 'type': 'ensemble', 'tp': 0.06, 'sl': 0.04, 'model': 'ada'},  # ex-MATIC
    # === MODELO DOGE - Memecoins (3) ===
    'SHIB': {'symbol': '1000SHIB/USDT', 'type': 'ensemble', 'tp': 0.06, 'sl': 0.04, 'model': 'doge'},
    'PEPE': {'symbol': '1000PEPE/USDT', 'type': 'ensemble', 'tp': 0.06, 'sl': 0.04, 'model': 'doge'},
    'FLOKI': {'symbol': '1000FLOKI/USDT', 'type': 'ensemble', 'tp': 0.06, 'sl': 0.04, 'model': 'doge'},
    # === MODELO DOT - Infraestructura (4) ===
    'LINK': {'symbol': 'LINK/USDT', 'type': 'ensemble', 'tp': 0.05, 'sl': 0.03, 'model': 'dot'},
    'ALGO': {'symbol': 'ALGO/USDT', 'type': 'ensemble', 'tp': 0.05, 'sl': 0.03, 'model': 'dot'},
    'FIL': {'symbol': 'FIL/USDT', 'type': 'ensemble', 'tp': 0.05, 'sl': 0.03, 'model': 'dot'},
    'NEAR': {'symbol': 'NEAR/USDT', 'type': 'ensemble', 'tp': 0.05, 'sl': 0.03, 'model': 'dot'},
}

# Features para ensemble simple
ENSEMBLE_FEATURES = ['rsi', 'macd_norm', 'adx', 'bb_pct', 'atr_pct',
                     'ret_3', 'ret_5', 'ret_10', 'vol_ratio', 'trend']

# =============================================================================
# BTC V14 - Clases y funciones
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

BTC_STRATEGY_PARAMS = {
    Strategy.TREND_FOLLOW_LONG: {'tp': 0.04, 'sl': 0.015},
    Strategy.TREND_FOLLOW_SHORT: {'tp': 0.04, 'sl': 0.015},
    Strategy.MEAN_REVERSION_LONG: {'tp': 0.025, 'sl': 0.012},
    Strategy.MEAN_REVERSION_SHORT: {'tp': 0.025, 'sl': 0.012},
    Strategy.BREAKOUT_LONG: {'tp': 0.05, 'sl': 0.02},
    Strategy.BREAKOUT_SHORT: {'tp': 0.05, 'sl': 0.02},
}

# =============================================================================
# LOGGING
# =============================================================================

def log(msg: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line)
    Path('logs').mkdir(exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_ohlcv(exchange, symbol: str, limit: int = 100) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    return df

# =============================================================================
# BTC V14 FEATURES (pandas_ta)
# =============================================================================

def compute_btc_features(df):
    """Features completas para BTC V14"""
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    feat = pd.DataFrame(index=df.index)

    # ADX/DI
    adx_df = pta.adx(h, l, c, length=14)
    if adx_df is not None:
        feat['adx'] = adx_df.iloc[:, 0]
        feat['di_plus'] = adx_df.iloc[:, 1]
        feat['di_minus'] = adx_df.iloc[:, 2]
        feat['di_diff'] = feat['di_plus'] - feat['di_minus']

    chop = pta.chop(h, l, c, length=14)
    feat['chop'] = chop if chop is not None else 50

    feat['ema20'] = pta.ema(c, length=20)
    feat['ema50'] = pta.ema(c, length=50)
    feat['ema200'] = pta.ema(c, length=200)
    feat['ema20_dist'] = (c - feat['ema20']) / feat['ema20'] * 100
    feat['ema200_dist'] = (c - feat['ema200']) / feat['ema200'] * 100
    feat['ema20_slope'] = feat['ema20'].pct_change(5) * 100
    feat['ema50_slope'] = feat['ema50'].pct_change(10) * 100

    feat['atr'] = pta.atr(h, l, c, length=14)
    feat['atr_pct'] = feat['atr'] / c * 100

    bb = pta.bbands(c, length=20)
    if bb is not None:
        feat['bb_width'] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / bb.iloc[:, 1] * 100
        feat['bb_pct'] = (c - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])

    feat['rsi14'] = pta.rsi(c, length=14)
    feat['rsi7'] = pta.rsi(c, length=7)

    stoch = pta.stoch(h, l, c, k=14, d=3)
    if stoch is not None:
        feat['stoch_k'] = stoch.iloc[:, 0]

    feat['ret_5'] = c.pct_change(5) * 100
    feat['ret_20'] = c.pct_change(20) * 100

    feat['vol_ratio'] = v / v.rolling(20).mean()
    feat['vol_trend'] = v.rolling(5).mean() / v.rolling(20).mean()
    obv = (np.sign(c.diff()) * v).cumsum()
    feat['obv_slope'] = obv.pct_change(10) * 100

    feat['high_20'] = h.rolling(20).max()
    feat['low_20'] = l.rolling(20).min()
    feat['range_pos'] = (c - feat['low_20']) / (feat['high_20'] - feat['low_20'])
    feat['consec_up'] = (c > c.shift(1)).rolling(10).sum()
    feat['consec_down'] = (c < c.shift(1)).rolling(10).sum()

    return feat.iloc[-1]  # Solo última fila


def detect_btc_regime(row):
    """Detecta régimen BTC"""
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


def detect_btc_setup(row, regime):
    """Detecta setup BTC según régimen"""
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


def get_btc_ensemble_confidence(row, models, direction):
    """Obtiene confianza del ensemble BTC"""
    probs = []

    for name in ['context', 'momentum', 'volume']:
        model_path = Path(f'strategies/btc_v14/models/{name}_{direction}.pkl')
        if not model_path.exists():
            continue

        try:
            model_data = joblib.load(model_path)
            model = model_data['model']
            features = model_data['features']

            X = np.array([row.get(f, 0) for f in features]).reshape(1, -1)
            if np.isnan(X).any():
                continue

            if 'scaler' in model_data:
                X = model_data['scaler'].transform(X)

            prob = model.predict_proba(X)[0, 1]
            probs.append(prob)
        except:
            continue

    return np.mean(probs) if probs else 0.5


# =============================================================================
# ENSEMBLE SIMPLE (DOGE/ADA/DOT)
# =============================================================================

def compute_ensemble_features(df):
    """Features para ensemble simple"""
    df = df.copy()
    df['rsi'] = ta.momentum.rsi(df['close'], window=14) / 100
    macd = ta.trend.MACD(df['close'])
    df['macd_norm'] = macd.macd() / df['close']
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14) / 100
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_pct'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-10)
    df['atr_pct'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14) / df['close']
    df['ret_3'] = df['close'].pct_change(3)
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_10'] = df['close'].pct_change(10)
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['trend'] = (df['close'] > df['close'].rolling(50).mean()).astype(float)
    return df.dropna()


def load_ensemble_models(asset, model_name=None):
    """Carga modelos ensemble. Si model_name especificado, usa ese modelo base."""
    model_asset = model_name if model_name else asset
    model_dir = Path(f'strategies/{model_asset.lower()}_v14/models')
    if not model_dir.exists():
        return None

    try:
        models = {'scaler': joblib.load(model_dir / 'scaler.pkl'),
                  'rf': joblib.load(model_dir / 'random_forest.pkl'),
                  'gb': joblib.load(model_dir / 'gradient_boosting.pkl')}
        lr_path = model_dir / 'logistic_regression.pkl'
        if lr_path.exists():
            models['lr'] = joblib.load(lr_path)
        return models
    except:
        return None


def predict_ensemble(models, features):
    X = models['scaler'].transform(features.reshape(1, -1))
    prob_rf = models['rf'].predict_proba(X)[0, 1]
    prob_gb = models['gb'].predict_proba(X)[0, 1]

    if 'lr' in models:
        prob_lr = models['lr'].predict_proba(X)[0, 1]
        votes = int(prob_rf > 0.5) + int(prob_gb > 0.5) + int(prob_lr > 0.5)
        avg_prob = (prob_rf + prob_gb + prob_lr) / 3
        return votes >= 2, avg_prob
    else:
        votes = int(prob_rf > 0.5) + int(prob_gb > 0.5)
        return votes >= 2, (prob_rf + prob_gb) / 2


# =============================================================================
# ETH SETUPS
# =============================================================================

def check_eth_setups(df):
    signals = []
    rsi = ta.momentum.rsi(df['close'], window=14).iloc[-1]
    vol_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
    price_change = df['close'].iloc[-1] > df['close'].iloc[-2]

    if rsi < 30:
        signals.append({'setup': 'RSI_OVERSOLD', 'direction': 'SHORT', 'prob': 0.65})
    if vol_ratio > 2 and price_change:
        signals.append({'setup': 'VOL_SPIKE_UP', 'direction': 'LONG', 'prob': 0.60})
    if vol_ratio > 2 and not price_change:
        signals.append({'setup': 'VOL_SPIKE_DOWN', 'direction': 'SHORT', 'prob': 0.60})

    return signals


# =============================================================================
# PAPER PORTFOLIO
# =============================================================================

class PaperPortfolio:
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.closed_trades = []
        self.load_state()

    def load_state(self):
        if Path(SIGNALS_FILE).exists():
            try:
                with open(SIGNALS_FILE, 'r') as f:
                    data = json.load(f)
                    self.capital = data.get('capital', self.initial_capital)
                    self.positions = data.get('positions', {})
                    self.closed_trades = data.get('closed_trades', [])
            except:
                pass

    def save_state(self):
        Path('data').mkdir(exist_ok=True)
        with open(SIGNALS_FILE, 'w') as f:
            json.dump({
                'capital': self.capital,
                'positions': self.positions,
                'closed_trades': self.closed_trades,
                'last_update': datetime.now().isoformat()
            }, f, indent=2, default=str)

    def open_position(self, symbol, direction, entry_price, tp_pct, sl_pct, prob, setup=''):
        if symbol in self.positions:
            return False

        risk_amount = self.capital * 0.015
        position_size = risk_amount / sl_pct

        self.positions[symbol] = {
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': datetime.now(timezone.utc).isoformat(),
            'tp_pct': tp_pct,
            'sl_pct': sl_pct,
            'size': position_size,
            'prob': prob,
            'setup': setup,
        }

        log(f"📈 OPEN {direction} {symbol} @ {entry_price:.4f} [{setup}] (conf: {prob:.1%})")
        self.save_state()
        return True

    def check_exits(self, prices):
        to_close = []

        for symbol, pos in self.positions.items():
            if symbol not in prices:
                continue

            current_price = prices[symbol]
            entry = pos['entry_price']
            direction = pos['direction']

            if direction == 'LONG':
                pnl_pct = (current_price - entry) / entry
            else:
                pnl_pct = (entry - current_price) / entry

            hit_tp = pnl_pct >= pos['tp_pct']
            hit_sl = pnl_pct <= -pos['sl_pct']

            if hit_tp or hit_sl:
                reason = 'TP' if hit_tp else 'SL'
                pnl = pos['size'] * pnl_pct
                self.capital += pnl

                self.closed_trades.append({
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': entry,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'reason': reason,
                    'setup': pos.get('setup', ''),
                    'exit_time': datetime.now(timezone.utc).isoformat(),
                })
                to_close.append(symbol)

                emoji = '✅' if pnl > 0 else '❌'
                log(f"{emoji} CLOSE {symbol} | {reason} | PnL: ${pnl:.2f} ({pnl_pct:.1%})")

        for symbol in to_close:
            del self.positions[symbol]

        if to_close:
            self.save_state()

    def get_summary(self):
        total_pnl = sum(t['pnl'] for t in self.closed_trades)
        wins = len([t for t in self.closed_trades if t['pnl'] > 0])
        total = len(self.closed_trades)
        wr = wins / total if total > 0 else 0
        return f"Capital: ${self.capital:.2f} | PnL: ${total_pnl:+.2f} | Trades: {total} | WR: {wr:.0%} | Open: {len(self.positions)}"


# =============================================================================
# MAIN
# =============================================================================

def main():
    log("=" * 70)
    log("V14 PAPER TRADING - 5 EXPERTOS")
    log("=" * 70)

    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    portfolio = PaperPortfolio(INITIAL_CAPITAL)

    log(f"Capital: ${INITIAL_CAPITAL}")
    log(f"Expertos: {list(EXPERTS.keys())}")

    # Cargar modelos
    ensemble_models = {}
    for asset, config in EXPERTS.items():
        if config['type'] == 'ensemble':
            model_name = config.get('model')  # None si no especificado
            m = load_ensemble_models(asset, model_name)
            if m:
                ensemble_models[asset] = m
                source = f" (using {model_name.upper()} model)" if model_name else ""
                log(f"Loaded {asset} ensemble{source}")

    # Verificar modelos BTC
    btc_models_exist = Path('strategies/btc_v14/models/context_long.pkl').exists()
    log(f"BTC V14 models: {'OK' if btc_models_exist else 'MISSING'}")

    last_candle_time = None

    log(f"\nEsperando nueva vela 4h...")
    log(f"Próximas: 04:00, 08:00, 12:00, 16:00, 20:00, 00:00 UTC")

    try:
        while True:
            now = datetime.now(timezone.utc)
            current_candle = now.replace(minute=0, second=0, microsecond=0)
            current_candle = current_candle.replace(hour=(now.hour // 4) * 4)

            if last_candle_time != current_candle:
                last_candle_time = current_candle
                log(f"\n{'='*50}")
                log(f"NUEVA VELA 4H: {current_candle}")
                log(f"{'='*50}")

                for asset, config in EXPERTS.items():
                    try:
                        symbol = config['symbol']
                        df = fetch_ohlcv(exchange, symbol, limit=100)
                        signals = []

                        # BTC V14: Régimen + Setups + Ensemble
                        if config['type'] == 'btc_v14' and btc_models_exist:
                            feat = compute_btc_features(df)
                            regime = detect_btc_regime(feat)
                            strategy, setup_name = detect_btc_setup(feat, regime)

                            if strategy:
                                direction = 'LONG' if 'LONG' in strategy.value else 'SHORT'
                                confidence = get_btc_ensemble_confidence(feat, None, direction.lower())

                                if confidence >= 0.35:  # Umbral mínimo
                                    params = BTC_STRATEGY_PARAMS.get(strategy, {'tp': 0.04, 'sl': 0.015})
                                    signals.append({
                                        'setup': f"{regime.value}:{setup_name}",
                                        'direction': direction,
                                        'prob': confidence,
                                        'tp': params['tp'],
                                        'sl': params['sl'],
                                    })
                                    log(f"BTC: {regime.value} | {setup_name} | conf={confidence:.1%}")
                                else:
                                    log(f"BTC: {regime.value} | {setup_name} | SKIP (conf={confidence:.1%})")
                            else:
                                log(f"BTC: {regime.value} | No setup")

                        # Ensemble simple (DOGE/ADA/DOT)
                        elif config['type'] == 'ensemble' and asset in ensemble_models:
                            df_feat = compute_ensemble_features(df)
                            features = df_feat[ENSEMBLE_FEATURES].iloc[-1].values
                            should_trade, prob = predict_ensemble(ensemble_models[asset], features)

                            if should_trade:
                                signals.append({
                                    'setup': 'ENSEMBLE_VOTE',
                                    'direction': 'LONG',
                                    'prob': prob,
                                    'tp': config['tp'],
                                    'sl': config['sl'],
                                })
                            else:
                                log(f"{asset}: No signal (prob={prob:.1%})")

                        # ETH setups
                        elif config['type'] == 'setups' and asset == 'ETH':
                            eth_signals = check_eth_setups(df)
                            for sig in eth_signals:
                                sig['tp'] = config['tp']
                                sig['sl'] = config['sl']
                            signals.extend(eth_signals)
                            if not signals:
                                log(f"ETH: No setup")

                        # Abrir posiciones
                        for sig in signals:
                            portfolio.open_position(
                                symbol=symbol,
                                direction=sig['direction'],
                                entry_price=df['close'].iloc[-1],
                                tp_pct=sig.get('tp', config['tp']),
                                sl_pct=sig.get('sl', config['sl']),
                                prob=sig['prob'],
                                setup=sig['setup']
                            )

                    except Exception as e:
                        log(f"{asset}: Error - {e}")

                log(f"\n{portfolio.get_summary()}")

            # Check exits
            try:
                prices = {}
                for symbol in portfolio.positions.keys():
                    ticker = exchange.fetch_ticker(symbol)
                    prices[symbol] = ticker['last']
                if prices:
                    portfolio.check_exits(prices)
            except:
                pass

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        log("\nDeteniendo...")
        log(f"Final: {portfolio.get_summary()}")
        portfolio.save_state()


if __name__ == '__main__':
    main()
