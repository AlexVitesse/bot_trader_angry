"""
V15 Data Pipeline - Descarga y prepara datos para el sistema Expert Committee.
==============================================================================

Descarga y guarda:
  1. BTC/USDT klines 4h (con taker_buy_vol) -> data/btc_v15_4h.parquet
  2. BTC/USDT klines 1d (macro)             -> data/btc_v15_1d.parquet
  3. Funding rates BTC                       -> data/btc_v15_funding.parquet
  4. Fear & Greed Index                      -> data/fear_greed_history.parquet (cache)

Ejecutar con el venv de produccion:
  C:\\Users\\pcdec\\AppData\\Local\\pypoetry\\Cache\\virtualenvs\\binance-scalper-bot-ofXWUGOe-py3.12\\Scripts\\python.exe v15_data_pipeline.py
"""

import requests
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timezone

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

FAPI_BASE = 'https://fapi.binance.com'
SYMBOL = 'BTCUSDT'
SINCE = '2019-01-01'


# =============================================================================
# KLINES CON TAKER BUY/SELL VOLUME
# =============================================================================

def _klines_page(symbol: str, interval: str, start_ts: int, limit: int = 1500) -> list:
    """Descarga una pagina de klines desde Binance Futures API."""
    resp = requests.get(
        f'{FAPI_BASE}/fapi/v1/klines',
        params={'symbol': symbol, 'interval': interval,
                'startTime': start_ts, 'limit': limit},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def download_klines(symbol: str, interval: str, since: str = SINCE) -> pd.DataFrame:
    """
    Descarga klines completos desde Binance Futures incluyendo taker_buy_vol.

    Columnas retornadas:
        open, high, low, close, volume, taker_buy_vol, taker_sell_vol
    """
    cache_name = f'{symbol.lower()}_{interval}_v15.parquet'
    cache = DATA_DIR / cache_name

    since_ts = int(datetime.strptime(since, '%Y-%m-%d').replace(
        tzinfo=timezone.utc).timestamp() * 1000)
    rows = []

    print(f'  Descargando {symbol} {interval} desde {since}...')
    while True:
        try:
            page = _klines_page(symbol, interval, since_ts)
        except Exception as e:
            print(f'    Error: {e}, reintentando...')
            time.sleep(5)
            continue

        if not page:
            break

        rows.extend(page)
        last_open_time = page[-1][0]
        since_ts = last_open_time + 1

        if len(page) < 1500:
            break

        time.sleep(0.2)

    print(f'    {len(rows):,} velas descargadas')

    # Parsear: Binance kline format
    # [open_time, open, high, low, close, volume, close_time, quote_vol,
    #  n_trades, taker_buy_base_vol, taker_buy_quote_vol, ignore]
    df = pd.DataFrame(rows, columns=[
        'ts', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_vol', 'n_trades',
        'taker_buy_vol', 'taker_buy_quote_vol', 'ignore',
    ])
    df['timestamp'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df = df.set_index('timestamp').sort_index()

    for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_vol']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['taker_sell_vol'] = df['volume'] - df['taker_buy_vol']
    df = df[['open', 'high', 'low', 'close', 'volume', 'taker_buy_vol', 'taker_sell_vol']]

    # Quitar duplicados y ultima vela incompleta
    df = df[~df.index.duplicated(keep='last')]
    df.dropna(subset=['close'], inplace=True)

    df.to_parquet(cache)
    print(f'    Guardado: {cache}')
    return df


# =============================================================================
# FUNDING RATES
# =============================================================================

def download_funding_rates(symbol: str = SYMBOL, since: str = '2020-01-01') -> pd.DataFrame:
    """
    Descarga historial de funding rates desde Binance Futures.
    Cada 8h desde 2020. Reusa cache si tiene menos de 6h.
    """
    cache = DATA_DIR / 'btc_v15_funding.parquet'

    if cache.exists() and (time.time() - cache.stat().st_mtime) / 3600 < 6:
        df = pd.read_parquet(cache)
        print(f'  Funding: cache OK ({len(df):,} registros)')
        return df

    since_ts = int(datetime.strptime(since, '%Y-%m-%d').replace(
        tzinfo=timezone.utc).timestamp() * 1000)
    rows = []

    print(f'  Descargando funding rates {symbol} desde {since}...')
    while True:
        try:
            resp = requests.get(
                f'{FAPI_BASE}/fapi/v1/fundingRate',
                params={'symbol': symbol, 'startTime': since_ts, 'limit': 1000},
                timeout=15,
            )
            resp.raise_for_status()
            page = resp.json()
        except Exception as e:
            print(f'    Error: {e}, reintentando...')
            time.sleep(5)
            continue

        if not page:
            break

        rows.extend(page)
        since_ts = page[-1]['fundingTime'] + 1

        if len(page) < 1000:
            break
        time.sleep(0.3)

    print(f'    {len(rows):,} registros de funding')

    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms', utc=True)
    df['funding_rate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
    df = df[['timestamp', 'funding_rate']].dropna()
    df = df.set_index('timestamp').sort_index()
    df = df[~df.index.duplicated(keep='first')]

    df.to_parquet(cache)
    print(f'    Guardado: {cache}')
    return df


# =============================================================================
# FEAR & GREED INDEX
# =============================================================================

def download_fear_greed() -> pd.DataFrame:
    """
    Descarga Fear & Greed Index desde alternative.me.
    Reusa cache existente si tiene menos de 24h.
    """
    cache = DATA_DIR / 'fear_greed_history.parquet'

    if cache.exists() and (time.time() - cache.stat().st_mtime) / 3600 < 24:
        df = pd.read_parquet(cache)
        print(f'  Fear & Greed: cache OK ({len(df):,} dias)')
        return df

    print('  Descargando Fear & Greed Index...')
    try:
        resp = requests.get(
            'https://api.alternative.me/fng/',
            params={'limit': 0, 'format': 'json'},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json().get('data', [])
    except Exception as e:
        print(f'    Error: {e}')
        if cache.exists():
            print('    Usando cache antigua')
            return pd.read_parquet(cache)
        return None

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s', utc=True)
    df['fng_value'] = df['value'].astype(int)
    df = df[['timestamp', 'fng_value']].set_index('timestamp').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df.to_parquet(cache)
    print(f'    {len(df):,} dias | {df.index[0].date()} a {df.index[-1].date()}')
    return df


# =============================================================================
# MAIN
# =============================================================================

def run_pipeline(force_download: bool = False) -> dict:
    """Ejecuta el pipeline completo y retorna paths de los archivos."""
    print('=' * 60)
    print('V15 Data Pipeline')
    print('=' * 60)

    # 1. 4h klines (con taker_buy_vol)
    cache_4h = DATA_DIR / 'btcusdt_4h_v15.parquet'
    if force_download or not cache_4h.exists():
        df_4h = download_klines(SYMBOL, '4h', SINCE)
    else:
        df_4h = pd.read_parquet(cache_4h)
        print(f'  4h klines: cache OK ({len(df_4h):,} velas, '
              f'{df_4h.index[0].date()} a {df_4h.index[-1].date()})')

    # 2. 1d klines (para macro)
    cache_1d = DATA_DIR / 'btcusdt_1d_v15.parquet'
    if force_download or not cache_1d.exists():
        df_1d = download_klines(SYMBOL, '1d', SINCE)
        df_1d.to_parquet(cache_1d)
        print(f'  1d klines guardados: {cache_1d}')
    else:
        df_1d = pd.read_parquet(cache_1d)
        print(f'  1d klines: cache OK ({len(df_1d):,} velas)')

    # 3. Funding rates
    df_funding = download_funding_rates(SYMBOL)

    # 4. Fear & Greed
    df_fng = download_fear_greed()

    # Resumen
    print('\nResumen:')
    print(f'  4h: {len(df_4h):,} velas | {df_4h.index[0].date()} a {df_4h.index[-1].date()}')
    print(f'  1d: {len(df_1d):,} dias  | {df_1d.index[0].date()} a {df_1d.index[-1].date()}')
    print(f'  Funding: {len(df_funding):,} registros')
    if df_fng is not None:
        print(f'  FNG: {len(df_fng):,} dias | {df_fng.index[0].date()} a {df_fng.index[-1].date()}')

    print('\nDatos listos para v15_features.py')
    return {
        '4h': df_4h,
        '1d': df_1d,
        'funding': df_funding,
        'fng': df_fng,
    }


if __name__ == '__main__':
    import sys
    force = '--force' in sys.argv
    run_pipeline(force_download=force)
