"""
Alternative Data Collector - Recolector diario para V8.1
========================================================
Recolecta datos de Open Interest y Long/Short Ratio de Binance.
Estos endpoints solo tienen 30 dias de historia, asi que hay que
recolectarlos diariamente para acumular datos.

Correr como cron diario:
  0 1 * * * cd /path/to/bot && poetry run python alt_data_collector.py

Despues de 3-6 meses se pueden usar como features en V8.1.
"""

import requests
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

BINANCE_FAPI = 'https://fapi.binance.com'

PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT',
    'NEAR/USDT', 'APT/USDT', 'ARB/USDT', 'OP/USDT', 'SUI/USDT',
]


def collect_open_interest(symbol, period='4h', limit=500):
    """Descarga Open Interest historico (max 30 dias)."""
    api_symbol = symbol.replace('/', '')
    try:
        resp = requests.get(
            f'{BINANCE_FAPI}/futures/data/openInterestHist',
            params={'symbol': api_symbol, 'period': period, 'limit': limit},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f'  Error OI {symbol}: {e}')
        return None

    if not data:
        return None

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['oi'] = df['sumOpenInterest'].astype(float)
    df['oi_value'] = df['sumOpenInterestValue'].astype(float)
    df = df[['timestamp', 'oi', 'oi_value']].set_index('timestamp')
    return df


def collect_ls_ratio(symbol, period='4h', limit=500):
    """Descarga Long/Short Ratio de top traders (max 30 dias)."""
    api_symbol = symbol.replace('/', '')
    try:
        resp = requests.get(
            f'{BINANCE_FAPI}/futures/data/topLongShortAccountRatio',
            params={'symbol': api_symbol, 'period': period, 'limit': limit},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f'  Error L/S {symbol}: {e}')
        return None

    if not data:
        return None

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['long_ratio'] = df['longAccount'].astype(float)
    df['short_ratio'] = df['shortAccount'].astype(float)
    df['ls_ratio'] = df['longShortRatio'].astype(float)
    df = df[['timestamp', 'long_ratio', 'short_ratio', 'ls_ratio']].set_index('timestamp')
    return df


def append_to_parquet(new_df, path):
    """Append new data to existing parquet, avoiding duplicates."""
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    else:
        combined = new_df
    combined.to_parquet(path)
    return len(combined)


def collect_all():
    """Recolecta OI y L/S ratio para todos los pares."""
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M')
    print(f'[{now}] Recolectando datos alternativos...')

    for pair in PAIRS:
        safe = pair.replace('/', '_')

        # Open Interest
        oi = collect_open_interest(pair)
        if oi is not None and len(oi) > 0:
            oi_path = DATA_DIR / f'{safe}_oi_history.parquet'
            total = append_to_parquet(oi, oi_path)
            print(f'  {pair:<12} OI: +{len(oi)} registros (total: {total})')

        # Long/Short Ratio
        ls = collect_ls_ratio(pair)
        if ls is not None and len(ls) > 0:
            ls_path = DATA_DIR / f'{safe}_ls_history.parquet'
            total = append_to_parquet(ls, ls_path)
            print(f'  {pair:<12} L/S: +{len(ls)} registros (total: {total})')

        time.sleep(0.3)  # Rate limit

    print('Recoleccion completada.')


if __name__ == '__main__':
    collect_all()
