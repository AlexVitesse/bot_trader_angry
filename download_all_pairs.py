"""
Descargar datos historicos de todos los pares para backtest V13.02
"""
import ccxt
import pandas as pd
from pathlib import Path
import time

print('='*60)
print('DESCARGA DE DATOS HISTORICOS - V13.02')
print('='*60)

exchange = ccxt.binance({'enableRateLimit': True})
DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

# Pares a descargar (los que faltan + actualizar existentes)
PAIRS = [
    'XRP/USDT',
    'NEAR/USDT',
    'DOT/USDT',
    'DOGE/USDT',
    'AVAX/USDT',
    'LINK/USDT',
    'ADA/USDT',
    'ETH/USDT',  # Actualizar
]

# Desde 2020 para tener datos de anos buenos/malos
START_DATE = '2020-01-01T00:00:00Z'

for pair in PAIRS:
    safe = pair.replace('/', '_')
    outpath = DATA_DIR / f'{safe}_4h_full.parquet'

    print(f'\n{pair}:')

    since = exchange.parse8601(START_DATE)
    all_ohlcv = []

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(pair, '4h', since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            print(f'  {len(all_ohlcv)} velas...', end='\r', flush=True)
            if len(ohlcv) < 1000:
                break
            time.sleep(0.1)  # Rate limit
        except Exception as e:
            print(f'  Error: {e}')
            break

    if all_ohlcv:
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df.to_parquet(outpath)
        print(f'  Guardado: {len(df)} velas ({df.index[0].year}-{df.index[-1].year})')
    else:
        print(f'  Sin datos')

print('\n' + '='*60)
print('Descarga completada!')
print('='*60)
