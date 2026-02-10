import pandas as pd
import time
from src.exchange import client
from config.settings import SYMBOL

def fetch_massive_history(symbol, interval='1m', days=30):
    print(f"📥 Descargando historial masivo de {symbol} ({days} dias)...")
    
    # Binance permite 1500 velas por peticion
    # 30 dias * 24h * 60m = 43,200 velas
    # Necesitamos ~29 peticiones
    
    all_klines = []
    # Usar timestamp de hace N dias
    start_time = int((time.time() - (days * 24 * 3600)) * 1000)
    
    while True:
        print(f"  --> Obteniendo desde: {pd.to_datetime(start_time, unit='ms')}")
        params = {
            'symbol': symbol.replace('/', ''),
            'interval': interval,
            'limit': 1500,
            'startTime': start_time
        }
        
        # Usar la request base del cliente para peticiones personalizadas
        klines = client._request('GET', '/fapi/v1/klines', params)
        
        if not klines or len(klines) <= 1:
            break
            
        all_klines.extend(klines)
        
        # El proximo start_time es el final del ultimo kline + 1ms
        start_time = klines[-1][6] + 1
        
        if len(all_klines) > 100000: # Limite de seguridad
            break
            
        time.sleep(0.5) # Evitar ban de IP

    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'tbb', 'tbq', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
    
    filename = f"data/{symbol.replace('/', '_')}_{interval}_history.parquet"
    df.to_parquet(filename)
    print(f"✅ Historial guardado: {filename} ({len(df)} velas)")
    return filename

if __name__ == "__main__":
    fetch_massive_history(SYMBOL, days=15) # Empecemos con 15 dias para no saturar
