"""
Optimizador de Parametros por Regimen
=====================================
Entrena/optimiza parametros especificos para cada tipo de regimen
para maximizar WR mientras mantiene PnL positivo.
"""

import json
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from regime_detector_v2 import RegimeDetector, MarketRegime

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
INITIAL_CAPITAL = 500.0

def load_data(pair):
    safe = pair.replace('/', '_')
    for suffix in ['_4h_v95.parquet', '_4h_history.parquet']:
        cache = DATA_DIR / f'{safe}{suffix}'
        if cache.exists():
            df = pd.read_parquet(cache)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            return df
    return None

def compute_features(df):
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

    atr_1 = ta.atr(h, l, c, length=1)
    atr_sum = atr_1.rolling(14).sum()
    high_max = h.rolling(14).max()
    low_min = l.rolling(14).min()
    feat['chop'] = 100 * np.log10(atr_sum / (high_max - low_min + 1e-10)) / np.log10(14)

    return feat


def simulate_trades_for_regime(signals_data, params):
    """Simula trades con parametros dados."""
    min_rsi = params.get('min_rsi', 30)
    max_rsi = params.get('max_rsi', 70)
    min_bb = params.get('min_bb', 0.2)
    max_bb = params.get('max_bb', 0.8)
    min_adx = params.get('min_adx', 15)
    max_adx = params.get('max_adx', 45)
    max_chop = params.get('max_chop', 55)
    min_conv = params.get('min_conv', 1.5)
    tp_mult = params.get('tp_mult', 2.0)
    sl_mult = params.get('sl_mult', 1.0)

    wins = 0
    losses = 0
    total_pnl = 0

    for sig in signals_data:
        # Aplicar filtros
        if not (min_rsi <= sig['rsi'] <= max_rsi):
            continue
        if not (min_bb <= sig['bb_pos'] <= max_bb):
            continue
        if not (min_adx <= sig['adx'] <= max_adx):
            continue
        if sig['chop'] > max_chop:
            continue
        if sig['conviction'] < min_conv:
            continue

        # Simular resultado
        # Usar ATR para TP/SL
        tp_pct = sig['atr_pct'] * tp_mult
        sl_pct = sig['atr_pct'] * sl_mult

        # El resultado ya esta calculado en signals_data
        if sig['hit_tp']:
            wins += 1
            total_pnl += tp_pct * 100  # % gain
        elif sig['hit_sl']:
            losses += 1
            total_pnl -= sl_pct * 100  # % loss
        else:
            # Timeout
            if sig['timeout_pnl'] > 0:
                wins += 1
            else:
                losses += 1
            total_pnl += sig['timeout_pnl'] * 100

    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0
    return {'wr': wr, 'trades': total, 'pnl': total_pnl}


def collect_signals_by_regime(pair_data, models, start_date, end_date, regime_detector):
    """Recolecta todas las senales categorizadas por regimen."""
    signals_by_regime = {r.value: [] for r in MarketRegime}
    MAX_HOLD = 20

    for pair, df in pair_data.items():
        df = df[(df.index >= start_date) & (df.index < end_date)]
        if len(df) < 250:
            continue

        safe = pair.replace('/', '')
        model = models.get(safe)
        if model is None:
            continue

        feat = compute_features(df)
        regimes = regime_detector.detect_regime_series(df)
        fcols = [c for c in model.feature_name_ if c in feat.columns]

        for i in range(250, len(df) - MAX_HOLD - 1):
            ts = df.index[i]
            row = df.iloc[i]
            price = row['close']

            X = feat.loc[ts:ts][fcols]
            if X.isna().any().any():
                continue

            pred = model.predict(X)[0]
            sig = 1 if pred > 0.5 else -1
            conviction = abs(pred - 0.5) * 10

            regime_str = regimes.loc[ts, 'regime']
            rsi = feat.loc[ts, 'rsi14']
            bb_pos = feat.loc[ts, 'bb_pos'] if 'bb_pos' in feat.columns else 0.5
            adx = feat.loc[ts, 'adx'] if 'adx' in feat.columns else 25
            chop = feat.loc[ts, 'chop'] if 'chop' in feat.columns else 50
            atr = feat.loc[ts, 'atr14']
            atr_pct = atr / price

            # Simular TP/SL hit (usando 2.0x/1.0x como base)
            hit_tp = False
            hit_sl = False
            timeout_pnl = 0

            for j in range(1, MAX_HOLD + 1):
                future_row = df.iloc[i + j]
                if sig == 1:
                    if future_row['high'] >= price + atr * 2.0:
                        hit_tp = True
                        break
                    if future_row['low'] <= price - atr * 1.0:
                        hit_sl = True
                        break
                else:
                    if future_row['low'] <= price - atr * 2.0:
                        hit_tp = True
                        break
                    if future_row['high'] >= price + atr * 1.0:
                        hit_sl = True
                        break

            if not hit_tp and not hit_sl:
                exit_price = df.iloc[i + MAX_HOLD]['close']
                timeout_pnl = (exit_price - price) / price * sig

            signal_data = {
                'pair': pair,
                'ts': ts,
                'dir': sig,
                'conviction': conviction,
                'rsi': rsi,
                'bb_pos': bb_pos,
                'adx': adx,
                'chop': chop,
                'atr_pct': atr_pct,
                'hit_tp': hit_tp,
                'hit_sl': hit_sl,
                'timeout_pnl': timeout_pnl,
            }

            signals_by_regime[regime_str].append(signal_data)

    return signals_by_regime


print("="*70)
print("OPTIMIZADOR DE PARAMETROS POR REGIMEN")
print("="*70)

# Cargar modelos
models = {}
for pair in PAIRS:
    safe = pair.replace('/', '')
    try:
        models[safe] = joblib.load(MODELS_DIR / f'v95_v7_{safe}.pkl')
    except:
        pass

# Cargar datos
pair_data = {}
for pair in PAIRS:
    df = load_data(pair)
    if df is not None:
        pair_data[pair] = df

detector = RegimeDetector()

print("\n[1] Recolectando senales por regimen...")
signals = collect_signals_by_regime(
    pair_data, models,
    '2022-01-01', '2026-02-24',  # Todo el periodo
    detector
)

for regime, sigs in signals.items():
    print(f"  {regime}: {len(sigs)} senales")

# Optimizar parametros para cada regimen
print("\n[2] Optimizando parametros por regimen...")
print("="*70)

# Grid de parametros a probar
param_grid = {
    'min_rsi': [30, 35, 40],
    'max_rsi': [65, 70, 75],
    'min_bb': [0.15, 0.25, 0.35],
    'max_bb': [0.65, 0.75, 0.85],
    'min_adx': [10, 15, 20],
    'max_adx': [35, 45, 55],
    'max_chop': [45, 50, 55],
    'min_conv': [1.5, 2.0, 2.5],
    'tp_mult': [1.5, 2.0, 2.5],
    'sl_mult': [0.75, 1.0, 1.25],
}

best_params = {}

for regime in ['weak_trend', 'bull_trend', 'bear_trend', 'high_vol', 'low_vol']:
    sigs = signals.get(regime, [])
    if len(sigs) < 100:
        print(f"\n{regime.upper()}: Insuficientes senales ({len(sigs)})")
        continue

    print(f"\n{regime.upper()}: {len(sigs)} senales")
    print("-"*50)

    best_wr = 0
    best_config = None

    # Probar combinaciones (simplificado para velocidad)
    for min_rsi in param_grid['min_rsi']:
        for max_rsi in param_grid['max_rsi']:
            for min_conv in param_grid['min_conv']:
                for max_chop in param_grid['max_chop']:
                    for tp_mult in param_grid['tp_mult']:
                        params = {
                            'min_rsi': min_rsi,
                            'max_rsi': max_rsi,
                            'min_bb': 0.2,
                            'max_bb': 0.8,
                            'min_adx': 15,
                            'max_adx': 45,
                            'max_chop': max_chop,
                            'min_conv': min_conv,
                            'tp_mult': tp_mult,
                            'sl_mult': 1.0,
                        }

                        result = simulate_trades_for_regime(sigs, params)

                        # Filtrar: al menos 50 trades y PnL positivo
                        if result['trades'] >= 50 and result['pnl'] > 0:
                            if result['wr'] > best_wr:
                                best_wr = result['wr']
                                best_config = {**params, **result}

    if best_config:
        print(f"  MEJOR WR: {best_config['wr']:.1f}%")
        print(f"  Trades: {best_config['trades']}, PnL: {best_config['pnl']:.1f}%")
        print(f"  Params: RSI {best_config['min_rsi']}-{best_config['max_rsi']}, "
              f"Conv >= {best_config['min_conv']}, Chop < {best_config['max_chop']}, "
              f"TP {best_config['tp_mult']}x")
        best_params[regime] = best_config
    else:
        print(f"  No se encontro configuracion con WR alto y PnL positivo")

# Guardar resultados
print("\n" + "="*70)
print("MEJORES PARAMETROS POR REGIMEN")
print("="*70)

for regime, cfg in best_params.items():
    print(f"\n{regime.upper()}:")
    print(f"  WR: {cfg['wr']:.1f}%, Trades: {cfg['trades']}, PnL: {cfg['pnl']:.1f}%")
    print(f"  RSI: {cfg['min_rsi']}-{cfg['max_rsi']}")
    print(f"  Conviction: >= {cfg['min_conv']}")
    print(f"  Chop: < {cfg['max_chop']}")
    print(f"  TP: {cfg['tp_mult']}x ATR")

# Guardar configuracion
output = {regime: {k: v for k, v in cfg.items() if k not in ['wr', 'trades', 'pnl']}
          for regime, cfg in best_params.items()}
with open(MODELS_DIR / 'v12_regime_params.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nParametros guardados en models/v12_regime_params.json")
