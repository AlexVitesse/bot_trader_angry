"""
Macro Data Fetcher - Datos macro para V8.2 Pipeline
====================================================
Descarga datos historicos de:
  1. DXY (Dollar Index) - yfinance
  2. Gold (GC=F) - yfinance
  3. S&P 500 (SPY) - yfinance
  4. US Treasury 10Y (^TNX) - yfinance
  5. ETH/BTC - Binance via ccxt

Cache en parquet con 24h de vigencia.
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

CACHE_HOURS = 24


def _cache_valid(path):
    """Verifica si el cache existe y tiene menos de CACHE_HOURS."""
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < CACHE_HOURS


# ============================================================
# 1. YFINANCE DOWNLOADS
# ============================================================

def _download_yfinance(ticker, name, since='2019-01-01'):
    """Descarga OHLCV diario de yfinance con cache."""
    cache = DATA_DIR / f'{name}_daily_history.parquet'
    if _cache_valid(cache):
        return pd.read_parquet(cache)

    try:
        import yfinance as yf
        df = yf.download(ticker, start=since, auto_adjust=True, progress=False)
    except Exception as e:
        print(f'    Error descargando {name} ({ticker}): {e}')
        # Try to return stale cache if available
        if cache.exists():
            return pd.read_parquet(cache)
        return None

    if df is None or len(df) == 0:
        print(f'    {name}: sin datos')
        if cache.exists():
            return pd.read_parquet(cache)
        return None

    # Flatten MultiIndex columns if present (yfinance sometimes returns multi-level)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standardize column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    df.index.name = 'timestamp'
    df = df[~df.index.duplicated(keep='first')].sort_index()
    df.to_parquet(cache)
    return df


def download_dxy(since='2019-01-01'):
    """Descarga Dollar Index (DXY) diario."""
    return _download_yfinance('DX-Y.NYB', 'dxy', since)


def download_gold(since='2019-01-01'):
    """Descarga Gold Futures diario."""
    return _download_yfinance('GC=F', 'gold', since)


def download_spy(since='2019-01-01'):
    """Descarga S&P 500 ETF (SPY) diario."""
    return _download_yfinance('SPY', 'spy', since)


def download_tnx(since='2019-01-01'):
    """Descarga US Treasury 10Y Yield diario."""
    return _download_yfinance('^TNX', 'tnx', since)


# ============================================================
# 2. ETH/BTC FROM BINANCE
# ============================================================

def download_ethbtc(since='2019-01-01'):
    """Descarga ETH/BTC daily desde Binance."""
    cache = DATA_DIR / 'ethbtc_daily_history.parquet'
    if _cache_valid(cache):
        return pd.read_parquet(cache)

    try:
        import ccxt
        exchange = ccxt.binance({'enableRateLimit': True})
        since_ts = int(datetime.strptime(since, '%Y-%m-%d').timestamp() * 1000)
        rows = []

        while True:
            try:
                candles = exchange.fetch_ohlcv('ETH/BTC', '1d', since=since_ts, limit=1000)
            except Exception as e:
                print(f'    Error ETH/BTC: {e}')
                time.sleep(5)
                continue

            if not candles:
                break
            rows.extend(candles)
            since_ts = candles[-1][0] + 1
            if len(candles) < 1000:
                break
            time.sleep(0.15)

        if not rows:
            if cache.exists():
                return pd.read_parquet(cache)
            return None

        df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')].sort_index()
        df.to_parquet(cache)
        return df

    except Exception as e:
        print(f'    Error ETH/BTC: {e}')
        if cache.exists():
            return pd.read_parquet(cache)
        return None


# ============================================================
# 3. COMPUTE MACRO FEATURES
# ============================================================

def compute_macro_features(dxy, gold, spy, tnx, ethbtc):
    """Calcula ~25 features macro desde datos diarios.

    Todos los features usan shift(1) para evitar look-ahead.
    Returns: DataFrame con index=fecha, columnas=macro features.
    """
    # Use DXY index as base (has most trading days)
    if dxy is None or len(dxy) == 0:
        return None

    feat = pd.DataFrame(index=dxy.index)

    assets = {
        'dxy': dxy,
        'gold': gold,
        'spy': spy,
        'tnx': tnx,
        'ethbtc': ethbtc,
    }

    for name, df in assets.items():
        if df is None or len(df) == 0:
            continue

        c = df['close'].reindex(feat.index, method='ffill')

        # SHIFT(1): usar datos del dia anterior
        c_prev = c.shift(1)

        # Retornos a distintos horizontes
        feat[f'{name}_ret_5d'] = c_prev.pct_change(5)
        feat[f'{name}_ret_20d'] = c_prev.pct_change(20)

        # Trend: distancia al EMA50
        ema50 = c_prev.ewm(span=50, adjust=False).mean()
        feat[f'{name}_vs_ema50'] = (c_prev - ema50) / (ema50 + 1e-10)

        # Volatilidad 20 dias
        feat[f'{name}_vol_20d'] = c_prev.pct_change().rolling(20).std()

    # Cross-market signals
    if 'dxy_ret_5d' in feat.columns and 'spy_ret_5d' in feat.columns:
        # DXY sube + SPY baja = risk-off fuerte
        feat['dxy_spy_diverge'] = feat['dxy_ret_5d'] + feat['spy_ret_5d']

    if 'gold_ret_5d' in feat.columns and 'spy_ret_5d' in feat.columns:
        # Gold sube mas que SPY = flight to safety
        feat['gold_spy_ratio'] = feat['gold_ret_5d'] - feat['spy_ret_5d']

    if 'dxy_ret_20d' in feat.columns and 'gold_ret_20d' in feat.columns:
        # Dollar y oro inverso = confirmacion de tendencia macro
        feat['dxy_gold_diverge'] = feat['dxy_ret_20d'] + feat['gold_ret_20d']

    return feat


# ============================================================
# 4. DOWNLOAD ALL
# ============================================================

def download_all_macro(since='2019-01-01'):
    """Descarga todos los datos macro.

    Returns: dict con keys ['dxy', 'gold', 'spy', 'tnx', 'ethbtc']
    """
    print('\n  Descargando datos macro...')

    data = {}
    for name, func in [
        ('dxy', download_dxy),
        ('gold', download_gold),
        ('spy', download_spy),
        ('tnx', download_tnx),
        ('ethbtc', download_ethbtc),
    ]:
        df = func(since)
        if df is not None and len(df) > 0:
            data[name] = df
            print(f'    {name:<10} {len(df):>6,} dias | {df.index[0].date()} a {df.index[-1].date()}')
        else:
            print(f'    {name:<10} NO DISPONIBLE')

    return data


# ============================================================
# TEST
# ============================================================

if __name__ == '__main__':
    macro = download_all_macro()
    if macro:
        feat = compute_macro_features(
            macro.get('dxy'), macro.get('gold'), macro.get('spy'),
            macro.get('tnx'), macro.get('ethbtc'),
        )
        if feat is not None:
            print(f'\n  Macro features: {len(feat.columns)} columnas, {len(feat)} filas')
            print(f'  Periodo: {feat.index[0].date()} a {feat.index[-1].date()}')
            print(f'  NaN por columna (primeras 50 filas excluidas):')
            valid = feat.iloc[50:]
            for col in feat.columns:
                nan_pct = valid[col].isna().mean() * 100
                if nan_pct > 0:
                    print(f'    {col}: {nan_pct:.1f}% NaN')
            if valid.isna().sum().sum() == 0:
                print(f'    Ninguno (perfecto)')
