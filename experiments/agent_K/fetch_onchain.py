"""
fetch_onchain.py -- Download BTC on-chain metrics from Coin Metrics community API.

Community-tier metrics available for BTC (free, no auth):
  - AdrActCnt:       Active addresses (daily)
  - CapMVRVCur:      Market-Value-to-Realized-Value (MVRV)
  - CapMrktCurUSD:   Current market cap USD
  - CapMrktEstUSD:   Estimated market cap USD (= realized cap proxy)
  - FlowInExNtv:     Exchange inflow (native units = BTC)
  - FlowOutExNtv:    Exchange outflow (BTC)
  - FlowInExUSD:     Exchange inflow USD
  - FlowOutExUSD:    Exchange outflow USD
  - HashRate:        Network hash rate (TH/s)
  - TxCnt:           Transaction count
  - TxTfrCnt:        Transfer count
  - SplyCur:         Current supply (BTC)
  - SplyExNtv:       Supply on exchanges (BTC)

Saves as parquet for fast loading. Frequency: daily.

NOTE on look-ahead:
  Coin Metrics publishes on-chain metrics with up to 1 day delay typically.
  We will apply .shift(1) (or shift(2) for safety) in features.py before
  reindexing to 4h. Cutoff <= 2025-12-31 applied in feature build, not here
  (we download everything so we can verify the OOS too).
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
from coinmetrics.api_client import CoinMetricsClient

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / 'onchain_data.parquet'

# Metrics we want (all community-tier, no auth required)
METRICS = [
    'AdrActCnt',         # active addresses
    'CapMVRVCur',        # MVRV
    'CapMrktCurUSD',     # market cap
    'CapMrktEstUSD',     # realized cap proxy
    'FlowInExNtv',       # exchange inflow BTC
    'FlowOutExNtv',      # exchange outflow BTC
    'FlowInExUSD',       # exchange inflow USD
    'FlowOutExUSD',      # exchange outflow USD
    'HashRate',          # hash rate
    'TxCnt',             # tx count
    'TxTfrCnt',          # transfer count
    'SplyCur',           # current supply
    'SplyExNtv',         # supply on exchanges
]

START = '2019-06-01'   # plenty of warmup before BTC 4h data starts (2020-01)
END = '2026-04-30'     # well past cutoff so 2026 OOS works too


def main():
    c = CoinMetricsClient()
    print(f'Fetching {len(METRICS)} metrics for BTC from {START} to {END} ...')
    df = c.get_asset_metrics(
        assets=['btc'],
        metrics=METRICS,
        frequency='1d',
        start_time=START,
        end_time=END,
        page_size=10000,
    ).to_dataframe()
    print(f'Raw rows: {len(df)}')
    print(f'Columns: {df.columns.tolist()}')
    # Set time as index, drop asset, ensure UTC
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('time').sort_index()
    if 'asset' in df.columns:
        df = df.drop(columns=['asset'])
    # Convert numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f'Final rows: {len(df)}, date range: {df.index[0]} -> {df.index[-1]}')
    print('Sample:')
    print(df.head(3))
    print(df.tail(3))
    df.to_parquet(OUT)
    print(f'Saved to: {OUT}')


if __name__ == '__main__':
    main()
