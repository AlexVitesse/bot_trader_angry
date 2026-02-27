"""
Descargar historial completo BTC 2019-2026
"""
import ccxt
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

print('='*60)
print('DESCARGA BTC/USDT 4h - 2019 a 2026')
print('='*60)

exchange = ccxt.binance({'enableRateLimit': True})
DATA_DIR = Path('data')

# Desde enero 2019
since = exchange.parse8601('2019-01-01T00:00:00Z')
all_ohlcv = []

print('Descargando...', flush=True)
while True:
    try:
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '4h', since=since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1

        # Progress
        last_date = datetime.utcfromtimestamp(ohlcv[-1][0] / 1000)
        print(f'  {len(all_ohlcv)} velas... (hasta {last_date.strftime("%Y-%m-%d")})', flush=True)

        if len(ohlcv) < 1000:
            break

        time.sleep(0.1)  # Rate limit
    except Exception as e:
        print(f'Error: {e}')
        time.sleep(5)
        continue

if all_ohlcv:
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    outpath = DATA_DIR / 'BTC_USDT_4h_full.parquet'
    df.to_parquet(outpath)

    print(f'\n{"="*60}')
    print(f'COMPLETADO')
    print(f'{"="*60}')
    print(f'Archivo: {outpath}')
    print(f'Total velas: {len(df)}')
    print(f'Rango: {df.index[0]} a {df.index[-1]}')
    print(f'Años de datos: {(df.index[-1] - df.index[0]).days / 365:.1f}')
else:
    print('No se descargaron datos')
