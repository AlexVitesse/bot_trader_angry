"""
Backtest Comparativo V9.5 vs V9 vs V8.5
=======================================
Compara performance de las tres estrategias usando walk-forward.

Uso: poetry run python backtest_v95.py
"""

import json
import time
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
import ccxt
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACION
# ============================================================================
DATA_DIR = Path('data')
MODELS_DIR = Path('models')

PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT',
    'NEAR/USDT',
]

# Trading params
TP_PCT = 0.03
SL_PCT = 0.015
MAX_HOLD = 30
INITIAL_CAPITAL = 500.0
RISK_PER_TRADE = 0.02  # 2% per trade
MAX_POSITIONS = 3

LOSS_FEATURES = [
    'cs_conf', 'cs_pred_mag', 'cs_macro_score', 'cs_risk_off',
    'cs_regime_bull', 'cs_regime_bear', 'cs_regime_range',
    'cs_atr_pct', 'cs_n_open', 'cs_pred_sign',
    'ld_conviction_pred',
    'ld_pair_rsi14', 'ld_pair_bb_pct', 'ld_pair_vol_ratio',
    'ld_pair_ret_5', 'ld_pair_ret_20',
    'ld_btc_ret_5', 'ld_btc_rsi14', 'ld_btc_vol20',
    'ld_hour', 'ld_tp_sl_ratio',
]


def load_data(pair, timeframe='4h'):
    """Carga datos desde cache."""
    safe = pair.replace('/', '_')
    # Try V9.5 cache first, then V7
    for suffix in ['_v95.parquet', '_4h.parquet', '.parquet']:
        cache = DATA_DIR / f'{safe}_{timeframe}{suffix}'
        if not cache.exists():
            cache = DATA_DIR / f'{safe}{suffix}'
        if cache.exists():
            return pd.read_parquet(cache)
    return None


def compute_features(df):
    """Features V7."""
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


def detect_regime(btc_df):
    """Detecta regimen de mercado."""
    c = btc_df['close']
    ema20 = ta.ema(c, length=20)
    ema50 = ta.ema(c, length=50)
    ret20 = c.pct_change(20)

    regime = pd.Series('RANGE', index=btc_df.index)
    regime[(c > ema20) & (ema20 > ema50) & (ret20 > 0.05)] = 'BULL'
    regime[(c < ema20) & (ema20 < ema50) & (ret20 < -0.05)] = 'BEAR'
    return regime


def compute_loss_features(df, ts, price, direction, conf, pred, regime, btc_df):
    """Compute features for LossDetector."""
    idx_pos = df.index.get_loc(ts)

    # ATR %
    atr = ta.atr(df['high'], df['low'], df['close'], length=14)
    atr_pct = atr.iloc[idx_pos] / price if idx_pos < len(atr) else 0.02

    # RSI
    rsi = ta.rsi(df['close'], length=14)
    rsi_val = rsi.iloc[idx_pos] if idx_pos < len(rsi) else 50

    # BB position
    bb = ta.bbands(df['close'], length=20, std=2.0)
    if bb is not None and idx_pos < len(bb):
        bb_lower = bb.iloc[idx_pos, 0]
        bb_upper = bb.iloc[idx_pos, 2]
        bb_pct = (price - bb_lower) / (bb_upper - bb_lower + 1e-10)
    else:
        bb_pct = 0.5

    # Volume ratio
    vol = df['volume']
    vol_ma = vol.rolling(20).mean()
    vol_ratio = vol.iloc[idx_pos] / vol_ma.iloc[idx_pos] if idx_pos < len(vol_ma) else 1.0

    # Returns
    ret_5 = df['close'].pct_change(5).iloc[idx_pos] if idx_pos >= 5 else 0
    ret_20 = df['close'].pct_change(20).iloc[idx_pos] if idx_pos >= 20 else 0

    # BTC context
    btc_idx = btc_df.index.get_indexer([ts], method='nearest')[0]
    btc_ret_5 = btc_df['close'].pct_change(5).iloc[btc_idx] if btc_idx >= 5 else 0
    btc_rsi = ta.rsi(btc_df['close'], length=14)
    btc_rsi_val = btc_rsi.iloc[btc_idx] if btc_idx < len(btc_rsi) else 50
    btc_vol = btc_df['close'].pct_change().rolling(20).std()
    btc_vol_val = btc_vol.iloc[btc_idx] if btc_idx < len(btc_vol) else 0.02

    return {
        'cs_conf': conf,
        'cs_pred_mag': abs(pred),
        'cs_macro_score': 0.5,
        'cs_risk_off': 1.0,
        'cs_regime_bull': 1.0 if regime == 'BULL' else 0.0,
        'cs_regime_bear': 1.0 if regime == 'BEAR' else 0.0,
        'cs_regime_range': 1.0 if regime == 'RANGE' else 0.0,
        'cs_atr_pct': atr_pct,
        'cs_n_open': 0,
        'cs_pred_sign': float(direction),
        'ld_conviction_pred': pred,
        'ld_pair_rsi14': rsi_val / 100.0,
        'ld_pair_bb_pct': bb_pct,
        'ld_pair_vol_ratio': vol_ratio,
        'ld_pair_ret_5': ret_5,
        'ld_pair_ret_20': ret_20,
        'ld_btc_ret_5': btc_ret_5,
        'ld_btc_rsi14': btc_rsi_val / 100.0,
        'ld_btc_vol20': btc_vol_val,
        'ld_hour': ts.hour / 24.0,
        'ld_tp_sl_ratio': TP_PCT / SL_PCT,
    }


class BacktestEngine:
    """Motor de backtest con soporte para V8.5, V9, y V9.5."""

    def __init__(self, strategy_name, loss_detector=None, loss_threshold=0.55,
                 loss_detectors_per_pair=None, thresholds_per_pair=None):
        self.strategy_name = strategy_name
        self.loss_detector = loss_detector  # V9 generic
        self.loss_threshold = loss_threshold  # V9 generic threshold
        self.loss_detectors_per_pair = loss_detectors_per_pair or {}  # V9.5
        self.thresholds_per_pair = thresholds_per_pair or {}  # V9.5

        self.capital = INITIAL_CAPITAL
        self.peak = INITIAL_CAPITAL
        self.positions = {}
        self.trades = []
        self.equity_curve = []

    def should_skip_trade(self, pair, loss_features):
        """Check if trade should be skipped by LossDetector."""
        # V9.5: per-pair model
        if pair in self.loss_detectors_per_pair:
            model = self.loss_detectors_per_pair[pair]
            threshold = self.thresholds_per_pair.get(pair, 0.55)
            df_feat = pd.DataFrame([loss_features])
            cols = [c for c in LOSS_FEATURES if c in df_feat.columns]
            if cols:
                p_loss = float(model.predict_proba(df_feat[cols])[0][1])
                return p_loss > threshold

        # V9: generic model
        if self.loss_detector is not None:
            df_feat = pd.DataFrame([loss_features])
            cols = [c for c in LOSS_FEATURES if c in df_feat.columns]
            if cols:
                p_loss = float(self.loss_detector.predict_proba(df_feat[cols])[0][1])
                return p_loss > self.loss_threshold

        # V8.5: no LossDetector
        return False

    def run(self, all_data, v7_models, regime_series, btc_df, test_start=None):
        """Run backtest on test period."""
        self.capital = INITIAL_CAPITAL
        self.peak = INITIAL_CAPITAL
        self.positions = {}
        self.trades = []
        self.equity_curve = []

        # Build timeline
        all_times = set()
        for df in all_data.values():
            if test_start:
                df = df[df.index >= test_start]
            all_times.update(df.index.tolist())
        timeline = sorted(all_times)

        for ts in timeline:
            # Update positions
            self._update_positions(ts, all_data)

            # Record equity
            equity = self._compute_equity(ts, all_data)
            self.equity_curve.append({'timestamp': ts, 'equity': equity})
            self.peak = max(self.peak, equity)

            # Generate signals
            if len(self.positions) >= MAX_POSITIONS:
                continue

            for pair, df in all_data.items():
                if pair in self.positions:
                    continue
                if pair not in v7_models:
                    continue
                if ts not in df.index:
                    continue
                if test_start and ts < test_start:
                    continue

                model_info = v7_models[pair]
                model = model_info['model']
                fcols = model_info['feature_cols']
                pred_std = model_info['pred_std']

                # Get prediction
                feat = compute_features(df.loc[:ts])
                feat = feat.replace([np.inf, -np.inf], np.nan)
                if ts not in feat.index:
                    continue

                X = feat.loc[[ts], fcols].fillna(0)
                pred = model.predict(X)[0]
                conf = abs(pred) / pred_std if pred_std > 1e-8 else 0

                if conf < 0.7:
                    continue

                direction = 1 if pred > 0 else -1

                # Regime filter
                regime = 'RANGE'
                if ts in regime_series.index:
                    regime = regime_series.loc[ts]

                if regime == 'BULL' and direction == -1:
                    continue
                if regime == 'BEAR' and direction == 1:
                    continue

                price = df.loc[ts, 'close']

                # LossDetector filter
                loss_features = compute_loss_features(
                    df.loc[:ts], ts, price, direction, conf, pred, regime, btc_df
                )
                if self.should_skip_trade(pair, loss_features):
                    continue

                # Open position
                position_size = self.capital * RISK_PER_TRADE / SL_PCT
                position_size = min(position_size, self.capital * 0.3)

                self.positions[pair] = {
                    'entry_time': ts,
                    'entry_price': price,
                    'direction': direction,
                    'size': position_size,
                    'conf': conf,
                    'hold_bars': 0,
                }

        # Close remaining positions
        for pair in list(self.positions.keys()):
            if pair in all_data:
                last_ts = all_data[pair].index[-1]
                last_price = all_data[pair]['close'].iloc[-1]
                self._close_position(pair, last_ts, last_price, 'END')

        return self._compute_metrics()

    def _update_positions(self, ts, all_data):
        """Update open positions, check TP/SL/Timeout."""
        for pair in list(self.positions.keys()):
            if pair not in all_data:
                continue
            df = all_data[pair]
            if ts not in df.index:
                continue

            pos = self.positions[pair]
            pos['hold_bars'] += 1

            price = df.loc[ts, 'close']
            entry = pos['entry_price']
            d = pos['direction']

            if d == 1:
                pnl_pct = (price - entry) / entry
            else:
                pnl_pct = (entry - price) / entry

            if pnl_pct >= TP_PCT:
                self._close_position(pair, ts, price, 'TP')
            elif pnl_pct <= -SL_PCT:
                self._close_position(pair, ts, price, 'SL')
            elif pos['hold_bars'] >= MAX_HOLD:
                self._close_position(pair, ts, price, 'TIME')

    def _close_position(self, pair, ts, price, reason):
        """Close a position."""
        pos = self.positions.pop(pair)
        entry = pos['entry_price']
        d = pos['direction']

        if d == 1:
            pnl_pct = (price - entry) / entry
        else:
            pnl_pct = (entry - price) / entry

        pnl_usd = pos['size'] * pnl_pct
        self.capital += pnl_usd

        self.trades.append({
            'pair': pair,
            'entry_time': pos['entry_time'],
            'exit_time': ts,
            'direction': d,
            'pnl_pct': pnl_pct,
            'pnl': pnl_usd,
            'reason': reason,
        })

    def _compute_equity(self, ts, all_data):
        """Compute current equity."""
        equity = self.capital
        for pair, pos in self.positions.items():
            if pair not in all_data:
                continue
            df = all_data[pair]
            if ts not in df.index:
                continue
            price = df.loc[ts, 'close']
            entry = pos['entry_price']
            d = pos['direction']
            if d == 1:
                unrealized = pos['size'] * (price - entry) / entry
            else:
                unrealized = pos['size'] * (entry - price) / entry
            equity += unrealized
        return equity

    def _compute_metrics(self):
        """Compute performance metrics."""
        if not self.trades:
            return {
                'strategy': self.strategy_name,
                'trades': 0, 'wins': 0, 'wr': 0.0,
                'pnl': 0.0, 'pf': 0.0, 'dd': 0.0, 'return_pct': 0.0,
            }

        df = pd.DataFrame(self.trades)
        wins = (df['pnl'] > 0).sum()
        losses = (df['pnl'] <= 0).sum()
        total_pnl = df['pnl'].sum()

        gross_profit = df[df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df[df['pnl'] <= 0]['pnl'].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Drawdown from equity curve
        eq = pd.DataFrame(self.equity_curve)
        eq['peak'] = eq['equity'].cummax()
        eq['dd'] = (eq['peak'] - eq['equity']) / eq['peak'] * 100
        max_dd = eq['dd'].max()

        return_pct = (self.capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

        return {
            'strategy': self.strategy_name,
            'trades': len(df),
            'wins': wins,
            'wr': wins / len(df) * 100 if len(df) > 0 else 0,
            'pnl': total_pnl,
            'pf': pf,
            'dd': max_dd,
            'return_pct': return_pct,
            'final_capital': self.capital,
        }


def main():
    print('=' * 70)
    print('BACKTEST COMPARATIVO: V9.5 vs V9 vs V8.5')
    print('=' * 70)

    # Load data
    print('\n[1] Cargando datos...')
    all_data = {}
    for pair in PAIRS:
        df = load_data(pair)
        if df is not None:
            all_data[pair] = df
            print(f'  {pair}: {len(df)} velas')

    if not all_data:
        print('ERROR: No hay datos. Ejecuta ml_train_v95.py primero.')
        return

    # Load BTC for regime
    btc_df = all_data.get('BTC/USDT')
    if btc_df is None:
        print('ERROR: BTC data required')
        return

    # Detect regime
    regime_series = detect_regime(btc_df)

    # Load V7 models
    print('\n[2] Cargando modelos V7...')
    v7_models = {}
    for pair in PAIRS:
        safe = pair.replace('/', '_')
        # Try V9.5 first, then standard
        for prefix in ['v95_v7_', 'v7_']:
            model_path = MODELS_DIR / f'{prefix}{safe}.pkl'
            if model_path.exists():
                try:
                    model = joblib.load(model_path)
                    # Load metadata
                    meta_path = MODELS_DIR / 'v7_meta.json'
                    with open(meta_path) as f:
                        meta = json.load(f)
                    v7_models[pair] = {
                        'model': model,
                        'feature_cols': meta.get('feature_cols', []),
                        'pred_std': meta.get('pred_stds', {}).get(pair, 0.01),
                    }
                    print(f'  {pair}: loaded ({prefix})')
                    break
                except Exception as e:
                    print(f'  {pair}: error - {e}')

    if not v7_models:
        print('ERROR: No V7 models found')
        return

    # Load V9 generic LossDetector
    print('\n[3] Cargando LossDetectors...')
    v9_loss_detector = None
    v9_threshold = 0.55
    v9_meta_path = MODELS_DIR / 'v9_meta.json'
    v9_model_path = MODELS_DIR / 'v9_loss_detector.pkl'

    if v9_model_path.exists():
        v9_loss_detector = joblib.load(v9_model_path)
        if v9_meta_path.exists():
            with open(v9_meta_path) as f:
                meta = json.load(f)
            v9_threshold = meta.get('loss_threshold', 0.55)
        print(f'  V9 generic: loaded (threshold={v9_threshold})')

    # Load V9.5 per-pair LossDetectors
    v95_detectors = {}
    v95_thresholds = {}
    v95_meta_path = MODELS_DIR / 'v95_meta.json'

    if v95_meta_path.exists():
        with open(v95_meta_path) as f:
            v95_meta = json.load(f)

        for pair, info in v95_meta.get('pairs', {}).items():
            safe = pair.replace('/', '')
            model_path = MODELS_DIR / f'v95_ld_{safe}.pkl'
            if model_path.exists():
                v95_detectors[pair] = joblib.load(model_path)
                v95_thresholds[pair] = info.get('threshold', 0.55)
                print(f'  V9.5 {pair}: loaded (threshold={v95_thresholds[pair]})')

    # Define test period (last 20% of data)
    min_date = min(df.index.min() for df in all_data.values())
    max_date = max(df.index.max() for df in all_data.values())
    test_start = min_date + (max_date - min_date) * 0.8
    print(f'\n  Test period: {test_start.date()} to {max_date.date()}')

    # Run backtests
    print('\n[4] Ejecutando backtests...')
    results = []

    # V8.5 (no LossDetector)
    print('  Running V8.5...')
    engine_v85 = BacktestEngine('V8.5')
    r = engine_v85.run(all_data, v7_models, regime_series, btc_df, test_start)
    results.append(r)
    print(f'    {r["trades"]} trades, WR={r["wr"]:.1f}%, PnL=${r["pnl"]:.2f}')

    # V9 (generic LossDetector)
    if v9_loss_detector is not None:
        print('  Running V9...')
        engine_v9 = BacktestEngine('V9', loss_detector=v9_loss_detector,
                                    loss_threshold=v9_threshold)
        r = engine_v9.run(all_data, v7_models, regime_series, btc_df, test_start)
        results.append(r)
        print(f'    {r["trades"]} trades, WR={r["wr"]:.1f}%, PnL=${r["pnl"]:.2f}')

    # V9.5 (per-pair LossDetector)
    if v95_detectors:
        print('  Running V9.5...')
        engine_v95 = BacktestEngine('V9.5',
                                     loss_detectors_per_pair=v95_detectors,
                                     thresholds_per_pair=v95_thresholds)
        r = engine_v95.run(all_data, v7_models, regime_series, btc_df, test_start)
        results.append(r)
        print(f'    {r["trades"]} trades, WR={r["wr"]:.1f}%, PnL=${r["pnl"]:.2f}')

    # Print results
    print('\n' + '=' * 70)
    print('RESULTADOS COMPARATIVOS')
    print('=' * 70)
    print(f'{"Strategy":<10} {"Trades":>8} {"Wins":>6} {"WR%":>8} '
          f'{"PnL":>10} {"PF":>6} {"DD%":>8} {"Ret%":>10}')
    print('-' * 70)

    for r in results:
        print(f'{r["strategy"]:<10} {r["trades"]:>8} {r["wins"]:>6} '
              f'{r["wr"]:>7.1f}% {r["pnl"]:>10.2f} {r["pf"]:>6.2f} '
              f'{r["dd"]:>7.1f}% {r["return_pct"]:>9.1f}%')

    # Winner
    if results:
        best = max(results, key=lambda x: x['pnl'])
        print('-' * 70)
        print(f'GANADOR: {best["strategy"]} con ${best["pnl"]:.2f} '
              f'({best["return_pct"]:.1f}% return)')

    # Save results
    results_path = MODELS_DIR / 'backtest_comparison.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nResultados guardados en: {results_path}')


if __name__ == '__main__':
    main()
