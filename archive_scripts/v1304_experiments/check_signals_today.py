"""
Revisar senales de hoy - debug
"""
import pandas as pd
import pandas_ta as ta
import joblib
import numpy as np
from pathlib import Path

DATA_DIR = Path('data')
MODELS_DIR = Path('models')

PAIRS = ['XRP/USDT', 'ETH/USDT', 'DOGE/USDT', 'ADA/USDT',
         'AVAX/USDT', 'LINK/USDT', 'DOT/USDT', 'NEAR/USDT']

CONVICTION_MIN = 1.0
CHOP_MAX = 60

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

if __name__ == '__main__':
    print('='*60)
    print('ANALISIS SENALES - Ultimas 48h')
    print('='*60)

    # Revisar ultimas velas
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=48)

    all_signals = []

    for pair in PAIRS:
        df = load_data(pair)
        if df is None:
            print(f'{pair}: NO DATA')
            continue

        # Cargar modelo
        safe = pair.replace('/', '').replace('_', '')
        try:
            model = joblib.load(MODELS_DIR / f'v95_v7_{safe}.pkl')
        except Exception as e:
            print(f'{pair}: NO MODEL - {e}')
            continue

        feat = compute_features(df)

        # Ultimas velas
        mask = df.index >= cutoff
        recent = df[mask]

        if len(recent) == 0:
            # Mostrar ultimo dato disponible
            print(f'{pair}: ultimo dato = {df.index[-1]}')
            continue

        print(f'\n{pair} (ultimo: {df.index[-1]})')
        print('-'*50)

        fcols = [c for c in model.feature_name_ if c in feat.columns]

        for ts in recent.index:
            if ts not in feat.index:
                continue
            row = feat.loc[ts]

            if row[fcols].isna().any():
                continue

            chop = row.get('chop', 50)
            if pd.isna(chop):
                chop = 50

            try:
                pred = model.predict(row[fcols].values.reshape(1, -1))[0]
            except:
                continue

            direction = 'LONG' if pred > 0 else 'SHORT'
            conviction = abs(pred) / 0.005

            # Filtros
            chop_ok = chop <= CHOP_MAX
            conv_ok = conviction >= CONVICTION_MIN

            if chop_ok and conv_ok:
                status = 'SIGNAL'
                all_signals.append({
                    'ts': ts,
                    'pair': pair,
                    'dir': direction,
                    'conv': conviction,
                    'chop': chop
                })
            else:
                status = 'FILTERED'

            filter_reason = []
            if not chop_ok:
                filter_reason.append(f'chop={chop:.1f}')
            if not conv_ok:
                filter_reason.append(f'conv={conviction:.2f}')

            print(f'{ts.strftime("%m-%d %H:%M")} | {direction:5} | conv={conviction:.2f} | chop={chop:.1f} | {status} {" ".join(filter_reason)}')

    print('\n' + '='*60)
    print(f'RESUMEN: {len(all_signals)} senales validas en ultimas 48h')
    print('='*60)

    if all_signals:
        print('\nSenales que debieron generar trades:')
        for s in all_signals:
            print(f"  {s['ts']} | {s['pair']:12} | {s['dir']:5} | conv={s['conv']:.2f}")
    else:
        print('\nNo hubo senales validas - el bot actuó correctamente.')
