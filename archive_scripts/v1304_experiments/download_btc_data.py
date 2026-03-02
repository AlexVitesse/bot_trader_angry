"""
Descargar datos de BTC/USDT para experimentos
"""
import ccxt
import pandas as pd
from pathlib import Path

print('='*50)
print('Descargando BTC/USDT 4h')
print('='*50)

exchange = ccxt.binance({'enableRateLimit': True})
DATA_DIR = Path('data')

# Desde Feb 2025
since = exchange.parse8601('2025-02-01T00:00:00Z')
all_ohlcv = []

while True:
    try:
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '4h', since=since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        print(f'  Descargadas {len(all_ohlcv)} velas...', flush=True)
        if len(ohlcv) < 1000:
            break
    except Exception as e:
        print(f'Error: {e}')
        break

if all_ohlcv:
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    outpath = DATA_DIR / 'BTC_USDT_4h_history.parquet'
    df.to_parquet(outpath)
    print(f'\nGuardado: {outpath}')
    print(f'Total: {len(df)} velas')
    print(f'Rango: {df.index[0]} a {df.index[-1]}')
else:
    print('No se descargaron datos')
