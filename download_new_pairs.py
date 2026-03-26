"""Download 4H + 1D klines for new pairs from Binance Futures."""
import sys, time, requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)
FAPI_BASE = 'https://fapi.binance.com'
SINCE = '2019-01-01'

PAIRS = ['LTC', 'ETC', 'BCH', 'TRX', 'UNI', 'AAVE', 'ARB', 'OP', 'FET', 'SUI']


def download_klines(symbol, interval, since=SINCE):
    out_name = f'{symbol}_{interval}_v15.parquet'
    out_path = DATA_DIR / out_name

    since_ts = int(datetime.strptime(since, '%Y-%m-%d').replace(
        tzinfo=timezone.utc).timestamp() * 1000)
    rows = []
    print(f'  Downloading {symbol} {interval} from {since}...')

    while True:
        try:
            resp = requests.get(
                f'{FAPI_BASE}/fapi/v1/klines',
                params={'symbol': symbol, 'interval': interval,
                        'startTime': since_ts, 'limit': 1500},
                timeout=20,
            )
            resp.raise_for_status()
            page = resp.json()
        except Exception as e:
            print(f'    Error: {e}, retrying...')
            time.sleep(5)
            continue

        if not page:
            break
        rows.extend(page)
        since_ts = page[-1][0] + 1
        if len(page) < 1500:
            break
        time.sleep(0.2)

    if not rows:
        print(f'    NO DATA for {symbol}')
        return None

    df = pd.DataFrame(rows, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_vol', 'trades', 'taker_buy_vol',
        'taker_buy_quote', 'ignore',
    ])
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    for c in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_vol']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['taker_sell_vol'] = df['volume'] - df['taker_buy_vol']
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume',
             'taker_buy_vol', 'taker_sell_vol']].copy()
    df = df.set_index('timestamp').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    df.dropna(subset=['close'], inplace=True)

    df.to_parquet(out_path)
    print(f'    {len(df):,} bars -> {out_path}')
    return df


for pair in PAIRS:
    symbol = f'{pair}USDT'
    # 4H data
    df4 = download_klines(symbol, '4h')
    if df4 is not None:
        # Also save as {PAIR}_USDT_4h_full.parquet (format load_pair_4h expects)
        full_path = DATA_DIR / f'{pair}_USDT_4h_full.parquet'
        df4.to_parquet(full_path)
        print(f'    Also saved: {full_path}')

    # 1D data
    df1d = download_klines(symbol, '1d')
    if df1d is not None:
        full_path = DATA_DIR / f'{pair}_USDT_1d_full.parquet'
        df1d.to_parquet(full_path)
        print(f'    Also saved: {full_path}')

    print()

print('DONE')
