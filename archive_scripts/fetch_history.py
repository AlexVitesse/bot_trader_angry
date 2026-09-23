import pandas as pd
import time
import requests
from config.settings import SYMBOL

# Binance produccion (datos publicos, no requiere API key)
BINANCE_PUBLIC_URL = "https://fapi.binance.com"


def fetch_massive_history(symbol, interval='1m', days=30, use_public=False):
    """Descarga historial de velas de Binance.

    Args:
        symbol: Par de trading (ej: 'BTC/USDT')
        interval: Intervalo de velas (ej: '1m', '5m', '1h')
        days: Dias de historial a descargar
        use_public: Si True, usa API publica de Binance produccion (sin API key)
    """
    print(f"Descargando historial de {symbol} ({days} dias)...")

    clean_symbol = symbol.replace('/', '')
    all_klines = []
    start_time = int((time.time() - (days * 24 * 3600)) * 1000)
    total_expected = days * 24 * 60  # velas de 1m esperadas

    while True:
        current_date = pd.to_datetime(start_time, unit='ms')
        progress = len(all_klines) / total_expected * 100 if total_expected > 0 else 0
        print(f"  [{progress:5.1f}%] Obteniendo desde: {current_date}")

        params = {
            'symbol': clean_symbol,
            'interval': interval,
            'limit': 1500,
            'startTime': start_time
        }

        if use_public:
            # API publica de Binance produccion (datos reales, sin auth)
            resp = requests.get(f"{BINANCE_PUBLIC_URL}/fapi/v1/klines", params=params, timeout=15)
            resp.raise_for_status()
            klines = resp.json()
        else:
            # API del cliente configurado (testnet o live segun settings)
            from src.exchange import client
            klines = client._request('GET', '/fapi/v1/klines', params)

        if not klines or len(klines) <= 1:
            break

        all_klines.extend(klines)
        start_time = klines[-1][6] + 1

        # Sin limite artificial - descargar todo lo solicitado
        time.sleep(0.3)  # Rate limit: ~3 req/s es seguro

    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'tbb', 'tbq', 'ignore'
    ])

    # Eliminar duplicados por timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    filename = f"data/{symbol.replace('/', '_')}_{interval}_history.parquet"
    df.to_parquet(filename)

    date_from = df['timestamp'].iloc[0].strftime('%Y-%m-%d')
    date_to = df['timestamp'].iloc[-1].strftime('%Y-%m-%d')
    print(f"Historial guardado: {filename}")
    print(f"  Velas: {len(df):,} | Desde: {date_from} | Hasta: {date_to}")
    return filename


if __name__ == "__main__":
    import sys

    days = 365  # 1 ano por defecto
    if len(sys.argv) > 1:
        days = int(sys.argv[1])

    # Usar API publica de Binance produccion para datos historicos reales
    fetch_massive_history(SYMBOL, days=days, use_public=True)
