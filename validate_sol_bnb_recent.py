"""
Validar SOL y BNB en ultimos 3 meses (Dic 2025 - Feb 2026)
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
from pathlib import Path
from datetime import datetime

DATA_DIR = Path('data')
MODELS_DIR = Path('models')


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """54 features igual que V2."""
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


def backtest_recent(pair: str, tp_pct: float, sl_pct: float,
                    start_date: str = '2025-12-01', conv_min: float = 1.0):
    """Backtest en periodo reciente."""

    # Cargar datos
    safe_pair = pair.replace('/', '_')
    df = pd.read_parquet(DATA_DIR / f'{safe_pair}_4h_full.parquet')

    # Cargar modelo
    safe_lower = pair.replace('/', '_').lower()
    model_data = joblib.load(MODELS_DIR / f'{safe_lower}_v2_gradientboosting.pkl')
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['feature_cols']
    pred_std = model_data['pred_std']

    # Features
    feat = compute_features(df)

    # Filtrar periodo
    mask = feat.index >= start_date
    df_recent = df[mask].copy()
    feat_recent = feat[mask].copy()

    # Predicciones
    valid = feat_recent.notna().all(axis=1)
    feat_valid = feat_recent[valid]
    df_valid = df_recent[valid]

    X = feat_valid[feature_cols]
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    conv = np.abs(preds) / pred_std
    regime = detect_regime(df_valid)

    # Simular trades
    trades = []
    for i, (idx, row) in enumerate(X.iterrows()):
        if i >= len(X) - 5:
            break

        pred = preds[i]
        c = conv[i]
        reg = regime.iloc[i]

        if c < conv_min:
            continue

        direction = 1 if pred > 0 else -1

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

        future_idx = X.index.get_loc(idx)
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
            'entry_time': idx,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'conviction': c,
            'regime': reg,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_pct * 100,  # Asumiendo $100 por trade
        })

    return pd.DataFrame(trades) if trades else pd.DataFrame()


def analyze_pair(pair: str, tp_pct: float, sl_pct: float):
    """Analiza un par en detalle."""
    print(f"\n{'='*60}")
    print(f"{pair} - Analisis Ultimos 3 Meses")
    print(f"TP={tp_pct*100:.1f}%, SL={sl_pct*100:.1f}%")
    print('='*60)

    trades_df = backtest_recent(pair, tp_pct, sl_pct, start_date='2025-12-01')

    if len(trades_df) == 0:
        print("Sin trades en el periodo")
        return None

    # Metricas
    total = len(trades_df)
    wins = (trades_df['pnl_pct'] > 0).sum()
    wr = wins / total
    total_pnl = trades_df['pnl_pct'].sum() * 100
    total_usd = trades_df['pnl_usd'].sum()

    print(f"\nRESUMEN:")
    print(f"  Trades: {total}")
    print(f"  Wins: {wins} ({wr*100:.1f}%)")
    print(f"  PnL: {total_pnl:+.1f}% (${total_usd:+.2f})")

    # Por mes
    print(f"\nPOR MES:")
    trades_df['month'] = trades_df['entry_time'].dt.strftime('%Y-%m')
    for month in sorted(trades_df['month'].unique()):
        m_trades = trades_df[trades_df['month'] == month]
        m_wins = (m_trades['pnl_pct'] > 0).sum()
        m_wr = m_wins / len(m_trades) if len(m_trades) > 0 else 0
        m_pnl = m_trades['pnl_usd'].sum()
        print(f"  {month}: {len(m_trades)} trades, {m_wr*100:.1f}% WR, ${m_pnl:+.2f}")

    # Por regimen
    print(f"\nPOR REGIMEN:")
    for reg in ['BULL', 'BEAR', 'RANGE']:
        r_trades = trades_df[trades_df['regime'] == reg]
        if len(r_trades) == 0:
            continue
        r_wins = (r_trades['pnl_pct'] > 0).sum()
        r_wr = r_wins / len(r_trades)
        r_pnl = r_trades['pnl_usd'].sum()
        print(f"  {reg}: {len(r_trades)} trades, {r_wr*100:.1f}% WR, ${r_pnl:+.2f}")

    # Por direccion
    print(f"\nPOR DIRECCION:")
    for dir in ['LONG', 'SHORT']:
        d_trades = trades_df[trades_df['direction'] == dir]
        if len(d_trades) == 0:
            continue
        d_wins = (d_trades['pnl_pct'] > 0).sum()
        d_wr = d_wins / len(d_trades)
        d_pnl = d_trades['pnl_usd'].sum()
        print(f"  {dir}: {len(d_trades)} trades, {d_wr*100:.1f}% WR, ${d_pnl:+.2f}")

    # Por exit reason
    print(f"\nPOR EXIT REASON:")
    for reason in ['tp', 'sl', 'timeout']:
        r_trades = trades_df[trades_df['exit_reason'] == reason]
        if len(r_trades) == 0:
            continue
        r_pnl = r_trades['pnl_usd'].sum()
        print(f"  {reason.upper()}: {len(r_trades)} trades, ${r_pnl:+.2f}")

    # Ultimos 10 trades
    print(f"\nULTIMOS 10 TRADES:")
    print(f"  {'Fecha':<16} | {'Dir':<5} | {'Conv':>5} | {'Reg':>5} | {'Exit':>7} | {'PnL':>8}")
    print(f"  " + "-" * 60)
    for _, t in trades_df.tail(10).iterrows():
        print(f"  {t['entry_time'].strftime('%Y-%m-%d %H:%M'):<16} | "
              f"{t['direction']:<5} | {t['conviction']:>5.2f} | {t['regime']:>5} | "
              f"{t['exit_reason']:>7} | ${t['pnl_usd']:>+7.2f}")

    return {
        'pair': pair,
        'trades': total,
        'wr': wr,
        'pnl_pct': total_pnl,
        'pnl_usd': total_usd,
        'trades_df': trades_df,
    }


# Probar diferentes configuraciones de conviction
def test_conviction_thresholds(pair: str, tp_pct: float, sl_pct: float):
    """Prueba diferentes umbrales de conviction."""
    print(f"\n{pair} - Test de Conviction Thresholds")
    print("-" * 50)

    results = []
    for conv in [0.5, 0.7, 1.0, 1.2, 1.5, 2.0]:
        trades_df = backtest_recent(pair, tp_pct, sl_pct,
                                    start_date='2025-12-01', conv_min=conv)
        if len(trades_df) == 0:
            continue
        wins = (trades_df['pnl_pct'] > 0).sum()
        wr = wins / len(trades_df)
        pnl = trades_df['pnl_usd'].sum()
        results.append({
            'conv_min': conv,
            'trades': len(trades_df),
            'wr': wr,
            'pnl': pnl,
        })

    print(f"{'Conv':>6} | {'Trades':>6} | {'WR':>6} | {'PnL':>10}")
    print("-" * 35)
    for r in results:
        print(f"{r['conv_min']:>6.1f} | {r['trades']:>6} | {r['wr']*100:>5.1f}% | ${r['pnl']:>+9.2f}")

    return results


if __name__ == '__main__':
    # SOL
    print("\n" + "="*70)
    print("SOL/USDT - Validacion")
    print("="*70)

    sol_result = analyze_pair('SOL/USDT', tp_pct=0.05, sl_pct=0.025)
    sol_conv_test = test_conviction_thresholds('SOL/USDT', tp_pct=0.05, sl_pct=0.025)

    # BNB
    print("\n" + "="*70)
    print("BNB/USDT - Validacion")
    print("="*70)

    bnb_result = analyze_pair('BNB/USDT', tp_pct=0.05, sl_pct=0.025)
    bnb_conv_test = test_conviction_thresholds('BNB/USDT', tp_pct=0.05, sl_pct=0.025)

    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL - Ultimos 3 Meses (Dic 2025 - Feb 2026)")
    print("="*70)

    print(f"\n{'Par':<12} | {'Trades':>6} | {'WR':>8} | {'PnL USD':>10} | {'Recomendacion':>15}")
    print("-" * 60)

    for res in [sol_result, bnb_result]:
        if res is None:
            continue
        if res['pnl_usd'] > 10 and res['wr'] > 0.5:
            rec = "HABILITAR"
        elif res['pnl_usd'] > 0:
            rec = "PROBAR"
        else:
            rec = "NO HABILITAR"
        print(f"{res['pair']:<12} | {res['trades']:>6} | {res['wr']*100:>7.1f}% | ${res['pnl_usd']:>+9.2f} | {rec:>15}")
