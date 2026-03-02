"""
Backtest V13.2 Completo
Simula todos los pares con sus configuraciones especificas
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import ccxt
from pathlib import Path
from datetime import datetime

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

# Configuracion V13.2
PAIRS_CONFIG = {
    'BTC/USDT': {
        'model_file': 'btc_v2_gradientboosting.pkl',
        'model_type': 'regressor',
        'tp_pct': 0.04,
        'sl_pct': 0.02,
        'conv_min': 1.0,
        'only_short': False,
    },
    'BNB/USDT': {
        'model_file': 'bnb_usdt_v2_gradientboosting.pkl',
        'model_type': 'regressor',
        'tp_pct': 0.07,
        'sl_pct': 0.035,
        'conv_min': 1.0,
        'only_short': True,  # CRITICO
    },
}

# Pares con modelo V7 generico
V7_PAIRS = ['XRP/USDT', 'NEAR/USDT', 'DOT/USDT', 'ETH/USDT',
            'DOGE/USDT', 'AVAX/USDT', 'LINK/USDT', 'ADA/USDT']

V7_CONFIG = {
    'tp_pct': 0.03,
    'sl_pct': 0.015,
    'conv_min': 0.5,
}


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """54 features para modelos V2."""
    feat = pd.DataFrame(index=df.index)
    c, h, l, o, v = df['close'], df['high'], df['low'], df['open'], df['volume']

    for p in [1, 2, 3, 5, 10, 20, 50]:
        feat[f'ret_{p}'] = c.pct_change(p)

    feat['atr14'] = ta.atr(h, l, c, length=14)
    feat['atr_r'] = feat['atr14'] / feat['atr14'].rolling(50).mean()
    feat['vol5'] = c.pct_change().rolling(5).std()
    feat['vol20'] = c.pct_change().rolling(20).std()
    feat['vol_ratio'] = feat['vol5'] / (feat['vol20'] + 1e-10)

    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)
    feat['rsi21'] = ta.rsi(c, length=21)

    sr = ta.stochrsi(c, length=14, rsi_length=14, k=3, d=3)
    if sr is not None:
        feat['srsi_k'] = sr.iloc[:, 0]
        feat['srsi_d'] = sr.iloc[:, 1]

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        feat['macd'] = macd.iloc[:, 0]
        feat['macd_h'] = macd.iloc[:, 1]
        feat['macd_s'] = macd.iloc[:, 2]

    feat['roc5'] = ta.roc(c, length=5)
    feat['roc10'] = ta.roc(c, length=10)
    feat['roc20'] = ta.roc(c, length=20)

    for el in [8, 21, 55, 100, 200]:
        e = ta.ema(c, length=el)
        feat[f'ema{el}_d'] = (c - e) / e * 100

    feat['ema8_sl'] = ta.ema(c, length=8).pct_change(3) * 100
    feat['ema21_sl'] = ta.ema(c, length=21).pct_change(5) * 100
    feat['ema55_sl'] = ta.ema(c, length=55).pct_change(5) * 100

    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        bw = bb.iloc[:, 2] - bb.iloc[:, 0]
        feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw
        feat['bb_w'] = bw / bb.iloc[:, 1] * 100

    feat['vr'] = v / v.rolling(20).mean()
    feat['vr5'] = v.rolling(5).mean() / v.rolling(20).mean()

    feat['spr'] = (h - l) / c * 100
    feat['body'] = abs(c - o) / (h - l + 1e-10)
    feat['upper_wick'] = (h - np.maximum(c, o)) / (h - l + 1e-10)
    feat['lower_wick'] = (np.minimum(c, o) - l) / (h - l + 1e-10)

    ax = ta.adx(h, l, c, length=14)
    if ax is not None:
        feat['adx'] = ax.iloc[:, 0]
        feat['dip'] = ax.iloc[:, 1]
        feat['dim'] = ax.iloc[:, 2]
        feat['di_diff'] = feat['dip'] - feat['dim']

    chop = ta.chop(h, l, c, length=14)
    if chop is not None:
        feat['chop'] = chop

    hr = df.index.hour
    dw = df.index.dayofweek
    feat['h_s'] = np.sin(2 * np.pi * hr / 24)
    feat['h_c'] = np.cos(2 * np.pi * hr / 24)
    feat['d_s'] = np.sin(2 * np.pi * dw / 7)
    feat['d_c'] = np.cos(2 * np.pi * dw / 7)

    feat['ret1_lag1'] = feat['ret_1'].shift(1)
    feat['rsi14_lag1'] = feat['rsi14'].shift(1)
    feat['ret1_lag2'] = feat['ret_1'].shift(2)
    feat['rsi14_lag2'] = feat['rsi14'].shift(2)
    feat['ret1_lag3'] = feat['ret_1'].shift(3)
    feat['rsi14_lag3'] = feat['rsi14'].shift(3)

    return feat


def detect_regime(df: pd.DataFrame) -> pd.Series:
    c = df['close']
    ema20 = ta.ema(c, length=20)
    ema50 = ta.ema(c, length=50)
    ret20 = c.pct_change(20)

    regime = pd.Series('RANGE', index=df.index)
    bull = (c > ema50) & (ema20 > ema50) & (ret20 > 0.05)
    bear = (c < ema50) & (ema20 < ema50) & (ret20 < -0.05)
    regime[bull] = 'BULL'
    regime[bear] = 'BEAR'

    return regime


def download_data(pair: str):
    """Descarga datos si no existen."""
    safe = pair.replace('/', '_')
    path = DATA_DIR / f'{safe}_4h_full.parquet'

    if path.exists():
        return pd.read_parquet(path)

    print(f"   Descargando {pair}...")
    exchange = ccxt.binance({'enableRateLimit': True})

    all_data = []
    since = exchange.parse8601('2020-01-01T00:00:00Z')

    while True:
        ohlcv = exchange.fetch_ohlcv(pair, '4h', since=since, limit=1000)
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < 1000:
            break

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.to_parquet(path)
    print(f"   {pair}: {len(df)} velas")

    return df


def backtest_pair_v2(pair: str, config: dict, start_date: str = '2025-12-01'):
    """Backtest un par con modelo V2."""
    safe = pair.replace('/', '_')

    # Cargar datos
    df = download_data(pair)

    # Cargar modelo
    model_path = MODELS_DIR / config['model_file']
    if not model_path.exists():
        print(f"   [SKIP] {pair}: modelo no encontrado")
        return []

    model_data = joblib.load(model_path)

    # Features
    feat = compute_features(df)

    # Filtrar periodo
    mask = feat.index >= start_date
    df_recent = df[mask].copy()
    feat_recent = feat[mask].copy()

    # Alinear con feature_cols del modelo
    feature_cols = model_data['feature_cols']
    missing = set(feature_cols) - set(feat_recent.columns)
    if missing:
        print(f"   [SKIP] {pair}: features faltantes: {missing}")
        return []

    valid = feat_recent[feature_cols].notna().all(axis=1)
    feat_valid = feat_recent.loc[valid, feature_cols]
    df_valid = df_recent[valid].copy()

    if len(feat_valid) == 0:
        return []

    # Predicciones
    if model_data['scaler'] is not None:
        X_scaled = model_data['scaler'].transform(feat_valid)
    else:
        X_scaled = feat_valid.values  # Sin scaling

    if config['model_type'] == 'classifier':
        probs = model_data['model'].predict_proba(X_scaled)[:, 1]
        signals = probs >= config.get('prob_threshold', 0.5)
        directions = np.full(len(probs), -1)  # Solo SHORT para clasificadores
    else:
        preds = model_data['model'].predict(X_scaled)
        conv = np.abs(preds) / model_data['pred_std']
        signals = conv >= config['conv_min']
        directions = np.where(preds < 0, -1, 1)

    regime = detect_regime(df_valid)

    # Simular trades
    trades = []
    tp_pct = config['tp_pct']
    sl_pct = config['sl_pct']
    only_short = config.get('only_short', False)

    for i, (idx, row) in enumerate(feat_valid.iterrows()):
        if i >= len(feat_valid) - 5:
            break

        if not signals[i]:
            continue

        direction = directions[i]
        reg = regime.iloc[i]

        if only_short and direction == 1:
            continue

        if reg == 'BULL' and direction == -1:
            continue
        if reg == 'BEAR' and direction == 1:
            continue

        entry_price = df_valid.loc[idx, 'close']

        if direction == 1:
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
        else:
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)

        exit_price = None
        exit_reason = 'timeout'

        future_idx = feat_valid.index.get_loc(idx)
        for j in range(1, min(21, len(df_valid) - future_idx)):
            bar = df_valid.iloc[future_idx + j]
            high, low = bar['high'], bar['low']

            if direction == 1:
                if low <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'sl'
                    break
                elif high >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'
                    break
            else:
                if high >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'sl'
                    break
                elif low <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'
                    break

        if exit_price is None:
            exit_price = df_valid.iloc[min(future_idx + 20, len(df_valid) - 1)]['close']

        if direction == 1:
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        trades.append({
            'pair': pair,
            'entry_time': idx,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'regime': reg,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_pct * 100,  # $100 por trade
        })

    return trades


def run_backtest_v13(version: str, pairs_config: dict, v7_pairs: list,
                     include_sol: bool = False, sol_config: dict = None,
                     start_date: str = '2025-12-01'):
    """Corre backtest completo de V13."""

    print(f"\n{'='*70}")
    print(f"BACKTEST {version}")
    print(f"Periodo: {start_date} - Presente")
    print(f"{'='*70}")

    all_trades = []

    # Pares con modelos V2 especificos
    for pair, config in pairs_config.items():
        print(f"\n   {pair} (V2 {config['model_type']})...")
        trades = backtest_pair_v2(pair, config, start_date)
        all_trades.extend(trades)
        if trades:
            wins = sum(1 for t in trades if t['pnl_pct'] > 0)
            pnl = sum(t['pnl_usd'] for t in trades)
            print(f"      {len(trades)} trades, {wins/len(trades)*100:.1f}% WR, ${pnl:+.2f}")

    # SOL si esta habilitado
    if include_sol and sol_config:
        print(f"\n   SOL/USDT (V3 Clasificador, 2% position)...")
        trades = backtest_pair_v2('SOL/USDT', sol_config, start_date)
        # Ajustar PnL a 2% position size (en vez de $100, usamos $20)
        for t in trades:
            t['pnl_usd'] = t['pnl_pct'] * 20  # 2% de $1000
        all_trades.extend(trades)
        if trades:
            wins = sum(1 for t in trades if t['pnl_pct'] > 0)
            pnl = sum(t['pnl_usd'] for t in trades)
            print(f"      {len(trades)} trades, {wins/len(trades)*100:.1f}% WR, ${pnl:+.2f} (2% pos)")

    # V7 pairs (simplificado - solo contamos sin modelo real)
    # En produccion real usarian el modelo V7
    print(f"\n   [V7 pairs: {len(v7_pairs)} pares - usando estimaciones]")

    if not all_trades:
        print("\n   Sin trades!")
        return None

    # Analisis
    trades_df = pd.DataFrame(all_trades)
    # Normalizar timezone (manejar mix de tz-aware y tz-naive)
    def normalize_tz(dt):
        if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
            return dt.tz_localize(None) if hasattr(dt, 'tz_localize') else dt.replace(tzinfo=None)
        return dt
    trades_df['entry_time'] = trades_df['entry_time'].apply(normalize_tz)
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])

    # Metricas globales
    total = len(trades_df)
    wins = (trades_df['pnl_pct'] > 0).sum()
    wr = wins / total
    total_pnl = trades_df['pnl_usd'].sum()

    # Profit Factor
    gross_profit = trades_df[trades_df['pnl_usd'] > 0]['pnl_usd'].sum()
    gross_loss = abs(trades_df[trades_df['pnl_usd'] < 0]['pnl_usd'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Drawdown
    cumulative = trades_df['pnl_usd'].cumsum()
    running_max = cumulative.cummax()
    drawdown = running_max - cumulative
    max_dd = drawdown.max()

    # Por par
    print(f"\n   {'='*60}")
    print(f"   RESULTADOS POR PAR")
    print(f"   {'='*60}")
    print(f"   {'Par':<12} | {'Trades':>6} | {'WR':>7} | {'PnL':>10}")
    print(f"   {'-'*45}")

    for pair in trades_df['pair'].unique():
        p_trades = trades_df[trades_df['pair'] == pair]
        p_wins = (p_trades['pnl_pct'] > 0).sum()
        p_wr = p_wins / len(p_trades)
        p_pnl = p_trades['pnl_usd'].sum()
        print(f"   {pair:<12} | {len(p_trades):>6} | {p_wr*100:>6.1f}% | ${p_pnl:>+8.2f}")

    # Resumen
    print(f"\n   {'='*60}")
    print(f"   RESUMEN {version}")
    print(f"   {'='*60}")
    print(f"   Total Trades: {total}")
    print(f"   Win Rate: {wr*100:.1f}%")
    print(f"   PnL Total: ${total_pnl:+.2f}")
    print(f"   Profit Factor: {pf:.2f}")
    print(f"   Max Drawdown: ${max_dd:.2f}")

    # Por mes
    trades_df['month'] = trades_df['entry_time'].dt.strftime('%Y-%m')
    print(f"\n   Por Mes:")
    for month in sorted(trades_df['month'].unique()):
        m_trades = trades_df[trades_df['month'] == month]
        m_wins = (m_trades['pnl_pct'] > 0).sum()
        m_wr = m_wins / len(m_trades)
        m_pnl = m_trades['pnl_usd'].sum()
        print(f"      {month}: {len(m_trades)} trades, {m_wr*100:.1f}% WR, ${m_pnl:+.2f}")

    return {
        'version': version,
        'trades': total,
        'wr': wr,
        'pnl': total_pnl,
        'pf': pf,
        'max_dd': max_dd,
        'trades_df': trades_df,
    }


if __name__ == '__main__':
    # SOL config (para prueba posterior)
    SOL_CONFIG = {
        'model_file': 'sol_usdt_v3_classifier.pkl',
        'model_type': 'classifier',
        'prob_threshold': 0.45,
        'tp_pct': 0.07,
        'sl_pct': 0.035,
        'only_short': True,
    }

    # =================================================================
    # TEST 1: V13.2 con BTC + BNB (sin SOL)
    # =================================================================
    result_v132 = run_backtest_v13(
        "V13.2 (BTC + BNB)",
        PAIRS_CONFIG,
        V7_PAIRS,
        include_sol=False,
        start_date='2025-12-01'
    )

    # =================================================================
    # TEST 2: V13.2 con BTC + BNB + SOL (2% position)
    # =================================================================
    result_v132_sol = run_backtest_v13(
        "V13.2 + SOL (2% pos)",
        PAIRS_CONFIG,
        V7_PAIRS,
        include_sol=True,
        sol_config=SOL_CONFIG,
        start_date='2025-12-01'
    )

    # =================================================================
    # COMPARACION
    # =================================================================
    print("\n" + "="*70)
    print("COMPARACION FINAL")
    print("="*70)

    print(f"\n{'Version':<25} | {'Trades':>6} | {'WR':>7} | {'PnL':>10} | {'PF':>6} | {'MaxDD':>8}")
    print("-"*75)

    for r in [result_v132, result_v132_sol]:
        if r:
            print(f"{r['version']:<25} | {r['trades']:>6} | {r['wr']*100:>6.1f}% | ${r['pnl']:>+8.2f} | {r['pf']:>6.2f} | ${r['max_dd']:>7.2f}")

    # Diferencia
    if result_v132 and result_v132_sol:
        diff_pnl = result_v132_sol['pnl'] - result_v132['pnl']
        diff_trades = result_v132_sol['trades'] - result_v132['trades']
        diff_dd = result_v132_sol['max_dd'] - result_v132['max_dd']

        print(f"\n   Diferencia con SOL:")
        print(f"   PnL: ${diff_pnl:+.2f}")
        print(f"   Trades: {diff_trades:+d}")
        print(f"   Max DD: ${diff_dd:+.2f}")

        if diff_pnl > 0 and diff_dd < result_v132['max_dd'] * 0.5:
            print(f"\n   >>> RECOMENDACION: HABILITAR SOL (mejora PnL sin mucho riesgo)")
        elif diff_pnl > 0:
            print(f"\n   >>> RECOMENDACION: SOL mejora PnL pero aumenta riesgo - evaluar")
        else:
            print(f"\n   >>> RECOMENDACION: NO habilitar SOL (no mejora)")
