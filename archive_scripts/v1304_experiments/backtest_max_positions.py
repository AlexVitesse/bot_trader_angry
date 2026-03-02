"""
Backtest: Comparar MAX_POSITIONS 3 vs 4
========================================
Simula V13 con limite de posiciones concurrentes real.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

# V13 Pairs (sin SOL/BTC/BNB)
PAIRS = [
    'XRP/USDT', 'ETH/USDT', 'DOGE/USDT', 'ADA/USDT',
    'AVAX/USDT', 'LINK/USDT', 'DOT/USDT', 'NEAR/USDT',
]

# Config comun
INITIAL_CAPITAL = 500.0
RISK_PER_TRADE = 0.02
MAX_HOLD = 20
TP_PCT = 0.03
SL_PCT = 0.015
CONVICTION_MIN = 1.0
CHOP_MAX = 60

# Periodo de test
START_DATE = '2026-02-01'
END_DATE = '2026-02-25'


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

    # Choppiness
    atr_1 = ta.atr(h, l, c, length=1)
    atr_sum = atr_1.rolling(14).sum()
    high_max = h.rolling(14).max()
    low_min = l.rolling(14).min()
    feat['chop'] = 100 * np.log10(atr_sum / (high_max - low_min + 1e-10)) / np.log10(14)

    return feat


def run_backtest(max_positions, all_data, all_features, all_models):
    """
    Backtest con limite de posiciones concurrentes.
    Recorre cronologicamente todas las velas 4h y decide entrada/salida.
    """
    # Obtener todas las timestamps unicas ordenadas
    all_timestamps = set()
    for pair, df in all_data.items():
        mask = (df.index >= START_DATE) & (df.index < END_DATE)
        all_timestamps.update(df[mask].index.tolist())
    all_timestamps = sorted(all_timestamps)

    positions = {}  # {pair: {'entry': price, 'dir': 1/-1, 'entry_ts': ts, 'bar': 0}}
    trades = []
    balance = INITIAL_CAPITAL

    for ts in all_timestamps:
        # 1. Actualizar posiciones existentes (check TP/SL/timeout)
        closed = []
        for pair, pos in positions.items():
            if pair not in all_data:
                continue
            df = all_data[pair]
            if ts not in df.index:
                continue

            candle = df.loc[ts]
            entry = pos['entry']
            direction = pos['dir']

            # TP/SL prices
            tp_price = entry * (1 + TP_PCT) if direction == 1 else entry * (1 - TP_PCT)
            sl_price = entry * (1 - SL_PCT) if direction == 1 else entry * (1 + SL_PCT)

            pos['bar'] += 1
            pnl_pct = None
            exit_reason = None

            if direction == 1:
                if candle['low'] <= sl_price:
                    pnl_pct = -SL_PCT
                    exit_reason = 'SL'
                elif candle['high'] >= tp_price:
                    pnl_pct = TP_PCT
                    exit_reason = 'TP'
            else:
                if candle['high'] >= sl_price:
                    pnl_pct = -SL_PCT
                    exit_reason = 'SL'
                elif candle['low'] <= tp_price:
                    pnl_pct = TP_PCT
                    exit_reason = 'TP'

            # Timeout
            if pnl_pct is None and pos['bar'] >= MAX_HOLD:
                exit_price = candle['close']
                pnl_pct = (exit_price - entry) / entry * direction
                exit_reason = 'TIME'

            if pnl_pct is not None:
                pnl = balance * RISK_PER_TRADE / SL_PCT * pnl_pct
                balance += pnl
                trades.append({
                    'pair': pair,
                    'entry_ts': pos['entry_ts'],
                    'exit_ts': ts,
                    'direction': direction,
                    'pnl_pct': pnl_pct,
                    'pnl': pnl,
                    'win': pnl > 0,
                    'exit_reason': exit_reason
                })
                closed.append(pair)

        for pair in closed:
            del positions[pair]

        # 2. Buscar nuevas senales (si hay espacio)
        if len(positions) >= max_positions:
            continue

        signals = []
        for pair in PAIRS:
            if pair in positions:
                continue
            if pair not in all_data or pair not in all_features or pair not in all_models:
                continue

            df = all_data[pair]
            feat = all_features[pair]
            model = all_models[pair]

            if ts not in df.index or ts not in feat.index:
                continue

            row = feat.loc[ts]
            fcols = [c for c in model.feature_name_ if c in feat.columns]

            if row[fcols].isna().any():
                continue

            # Chop filter
            chop = row.get('chop', 50)
            if pd.isna(chop) or chop > CHOP_MAX:
                continue

            # Predict
            try:
                pred = model.predict(row[fcols].values.reshape(1, -1))[0]
            except:
                continue

            direction = 1 if pred > 0 else -1
            conviction = abs(pred) / 0.005

            if conviction < CONVICTION_MIN:
                continue

            signals.append({
                'pair': pair,
                'direction': direction,
                'conviction': conviction,
                'price': df.loc[ts, 'close']
            })

        # Ordenar por conviction y tomar las mejores
        signals.sort(key=lambda x: x['conviction'], reverse=True)
        slots = max_positions - len(positions)

        for sig in signals[:slots]:
            positions[sig['pair']] = {
                'entry': sig['price'],
                'dir': sig['direction'],
                'entry_ts': ts,
                'bar': 0
            }

    # Cerrar posiciones abiertas al final
    final_ts = all_timestamps[-1] if all_timestamps else None
    for pair, pos in positions.items():
        if pair in all_data and final_ts in all_data[pair].index:
            exit_price = all_data[pair].loc[final_ts, 'close']
            pnl_pct = (exit_price - pos['entry']) / pos['entry'] * pos['dir']
            pnl = balance * RISK_PER_TRADE / SL_PCT * pnl_pct
            trades.append({
                'pair': pair,
                'entry_ts': pos['entry_ts'],
                'exit_ts': final_ts,
                'direction': pos['dir'],
                'pnl_pct': pnl_pct,
                'pnl': pnl,
                'win': pnl > 0,
                'exit_reason': 'OPEN'
            })

    return trades


def calculate_metrics(trades):
    if not trades:
        return {'trades': 0, 'wins': 0, 'wr': 0, 'pnl': 0, 'pf': 0, 'max_dd': 0}

    n_trades = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n_trades * 100

    pnl = sum(t['pnl'] for t in trades)
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999

    # Max drawdown
    equity = [INITIAL_CAPITAL]
    for t in sorted(trades, key=lambda x: x['entry_ts']):
        equity.append(equity[-1] + t['pnl'])

    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        'trades': n_trades,
        'wins': wins,
        'wr': wr,
        'pnl': pnl,
        'pf': pf,
        'max_dd': max_dd
    }


def main():
    print('=' * 70)
    print('BACKTEST: MAX_POSITIONS 3 vs 4')
    print('=' * 70)
    print(f'Periodo: {START_DATE} a {END_DATE}')
    print(f'Pares V13: {len(PAIRS)}')
    print(f'Capital inicial: ${INITIAL_CAPITAL}')
    print()

    # Cargar datos
    print('[1] Cargando datos...')
    all_data = {}
    for pair in PAIRS:
        df = load_data(pair)
        if df is not None:
            all_data[pair] = df
            print(f'  {pair}: OK')

    # Cargar modelos
    print('\n[2] Cargando modelos V7...')
    all_models = {}
    for pair in PAIRS:
        safe = pair.replace('/', '').replace('_', '')
        try:
            model = joblib.load(MODELS_DIR / f'v95_v7_{safe}.pkl')
            all_models[pair] = model
            print(f'  {pair}: OK')
        except:
            print(f'  {pair}: NO ENCONTRADO')

    # Calcular features
    print('\n[3] Calculando features...')
    all_features = {}
    for pair in PAIRS:
        if pair in all_data:
            all_features[pair] = compute_features(all_data[pair])
            print(f'  {pair}: OK')

    # Backtest con 3 posiciones
    print('\n[4] Backtest MAX_POSITIONS = 3...')
    trades_3 = run_backtest(3, all_data, all_features, all_models)
    metrics_3 = calculate_metrics(trades_3)

    # Backtest con 4 posiciones
    print('\n[5] Backtest MAX_POSITIONS = 4...')
    trades_4 = run_backtest(4, all_data, all_features, all_models)
    metrics_4 = calculate_metrics(trades_4)

    # Resultados
    print('\n' + '=' * 70)
    print('RESULTADOS')
    print('=' * 70)

    print(f'\n{"Metrica":<20} {"3 Pos":>15} {"4 Pos":>15} {"Diferencia":>15}')
    print('-' * 65)

    metrics = [
        ('Trades', 'trades', '', 0),
        ('Wins', 'wins', '', 0),
        ('Win Rate', 'wr', '%', 1),
        ('PnL', 'pnl', '$', 2),
        ('Profit Factor', 'pf', '', 2),
        ('Max Drawdown', 'max_dd', '%', 1),
    ]

    for name, key, suffix, decimals in metrics:
        v3 = metrics_3[key]
        v4 = metrics_4[key]
        diff = v4 - v3

        if suffix == '$':
            v3_str = f'${v3:,.2f}'
            v4_str = f'${v4:,.2f}'
            diff_str = f'{diff:+,.2f}'
        elif suffix == '%':
            v3_str = f'{v3:.{decimals}f}%'
            v4_str = f'{v4:.{decimals}f}%'
            diff_str = f'{diff:+.{decimals}f}%'
        else:
            v3_str = f'{v3:.{decimals}f}'
            v4_str = f'{v4:.{decimals}f}'
            diff_str = f'{diff:+.{decimals}f}'

        if key in ['wr', 'pnl', 'pf']:
            indicator = '++' if diff > 0 else '--' if diff < 0 else '=='
        elif key == 'max_dd':
            indicator = '++' if diff < 0 else '--' if diff > 0 else '=='
        else:
            indicator = ''

        print(f'{name:<20} {v3_str:>15} {v4_str:>15} {diff_str:>12} {indicator}')

    # Recomendacion
    print('\n' + '-' * 65)

    # Criterio: mejor PnL con DD controlado
    better_pnl = metrics_4['pnl'] > metrics_3['pnl']
    similar_dd = abs(metrics_4['max_dd'] - metrics_3['max_dd']) < 3  # <3% diferencia DD
    better_wr = metrics_4['wr'] >= metrics_3['wr'] - 5  # no perder mas de 5% WR

    if better_pnl and similar_dd and better_wr:
        print('\nRECOMENDACION: Usar MAX_POSITIONS = 4')
        print('  + Mayor PnL')
        print('  + Drawdown similar')
        winner = 4
    elif metrics_3['pnl'] > metrics_4['pnl'] or metrics_4['max_dd'] > metrics_3['max_dd'] + 5:
        print('\nRECOMENDACION: Mantener MAX_POSITIONS = 3')
        print('  + Mas conservador')
        print('  + Validado en backtest original')
        winner = 3
    else:
        print('\nRECOMENDACION: Mantener MAX_POSITIONS = 3 (diferencia marginal)')
        winner = 3

    print(f'\n[Backtest completado]')

    return winner, metrics_3, metrics_4


if __name__ == '__main__':
    main()
