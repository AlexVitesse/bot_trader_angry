"""
Alternative Data Fetcher - Datos para V8
=========================================
Descarga datos historicos de:
  1. Funding Rate (Binance Futures API) - desde 2019
  2. Fear & Greed Index (alternative.me) - desde 2018

Ejecutar: poetry run python alt_data_fetcher.py
"""

import requests
import pandas as pd
import time
from pathlib import Path
from datetime import datetime

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

BINANCE_FAPI = 'https://fapi.binance.com'


# ============================================================
# 1. FUNDING RATE
# ============================================================
def download_funding_rates(symbol, since='2020-01-01'):
    """Descarga historial completo de funding rate para un par.

    Binance API: GET /fapi/v1/fundingRate
    - limit: max 1000 por request
    - Funding cada 8h -> 1000 registros = ~333 dias
    - 2020 a 2026 = ~2200 dias = ~6600 registros = ~7 requests por par

    Returns: DataFrame con index=timestamp, columnas=[funding_rate, mark_price]
    """
    safe = symbol.replace('/', '_')
    cache = DATA_DIR / f'{safe}_funding_history.parquet'

    # Cache 24h
    if cache.exists() and (time.time() - cache.stat().st_mtime) / 3600 < 24:
        return pd.read_parquet(cache)

    api_symbol = symbol.replace('/', '')  # BTC/USDT -> BTCUSDT
    since_ts = int(datetime.strptime(since, '%Y-%m-%d').timestamp() * 1000)
    rows = []

    while True:
        try:
            resp = requests.get(
                f'{BINANCE_FAPI}/fapi/v1/fundingRate',
                params={
                    'symbol': api_symbol,
                    'startTime': since_ts,
                    'limit': 1000,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f'    Error funding {symbol}: {e}')
            time.sleep(5)
            continue

        if not data:
            break

        rows.extend(data)
        since_ts = data[-1]['fundingTime'] + 1

        if len(data) < 1000:
            break
        time.sleep(0.3)  # Rate limit conservador

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
    df['funding_rate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
    df['mark_price'] = pd.to_numeric(df.get('markPrice', 0), errors='coerce')
    df = df[['timestamp', 'funding_rate', 'mark_price']]
    df.dropna(subset=['funding_rate'], inplace=True)
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    df.to_parquet(cache)
    return df


# ============================================================
# 2. FEAR & GREED INDEX
# ============================================================
def download_fear_greed():
    """Descarga historial completo del Fear & Greed Index.

    API: https://api.alternative.me/fng/?limit=0&format=json
    - Una sola llamada retorna TODO el historial desde 2018-02
    - Valores: 0 (miedo extremo) a 100 (codicia extrema)

    Returns: DataFrame con index=timestamp, columna=[fng_value]
    """
    cache = DATA_DIR / 'fear_greed_history.parquet'

    # Cache 24h
    if cache.exists() and (time.time() - cache.stat().st_mtime) / 3600 < 24:
        return pd.read_parquet(cache)

    try:
        resp = requests.get(
            'https://api.alternative.me/fng/',
            params={'limit': 0, 'format': 'json'},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json().get('data', [])
    except Exception as e:
        print(f'    Error Fear & Greed: {e}')
        return None

    if not data:
        return None

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
    df['fng_value'] = df['value'].astype(int)
    df = df[['timestamp', 'fng_value']]
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    df.to_parquet(cache)
    return df


# ============================================================
# 3. DOWNLOAD ALL
# ============================================================
def download_all_alt_data(pairs):
    """Descarga todos los datos alternativos para una lista de pares.

    Returns: (funding_data: dict, fng_df: DataFrame)
    """
    print('\n  Descargando datos alternativos...')

    # Funding rates por par
    funding_data = {}
    for pair in pairs:
        fr = download_funding_rates(pair)
        if fr is not None and len(fr) > 0:
            funding_data[pair] = fr
            print(f'    {pair:<12} Funding: {len(fr):>6,} registros '
                  f'| {fr.index[0].date()} a {fr.index[-1].date()}')
        else:
            print(f'    {pair:<12} Funding: NO DISPONIBLE')

    # Fear & Greed (global, no por par)
    fng = download_fear_greed()
    if fng is not None:
        print(f'    Fear&Greed  {len(fng):>6,} dias '
              f'| {fng.index[0].date()} a {fng.index[-1].date()}')
    else:
        print('    Fear&Greed  NO DISPONIBLE')

    return funding_data, fng


# ============================================================
# TEST
# ============================================================
if __name__ == '__main__':
    pairs = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
        'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT',
        'NEAR/USDT', 'APT/USDT', 'ARB/USDT', 'OP/USDT', 'SUI/USDT',
    ]
    funding_data, fng = download_all_alt_data(pairs)
    print(f'\n  Resumen: {len(funding_data)} pares con funding, '
          f'F&G {"OK" if fng is not None else "FALLO"}')
