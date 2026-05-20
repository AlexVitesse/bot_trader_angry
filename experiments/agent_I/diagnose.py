"""
Quick diagnostic to understand RANGE frequency in ETH 4h.
Single throwaway script, not part of the validated pipeline.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import strategy as S

CUTOFF = pd.Timestamp('2025-12-31 23:59:59', tz='UTC')
df_raw = pd.read_parquet(ROOT / 'data' / 'ETH_USDT_4h_full.parquet')
if df_raw.index.tz is None:
    df_raw.index = df_raw.index.tz_localize('UTC')
df_raw = df_raw.sort_index()
df_raw = df_raw[df_raw.index <= CUTOFF]
df = S.prepare(df_raw)
mask = df.index >= pd.Timestamp('2020-01-01', tz='UTC')
df_in = df[mask]

print(f'Total bars in-sample: {len(df_in)}')

# Each individual condition
adx_ok = df_in['adx14'] < 20
bbw_ok = df_in['bb_width_pctile'] < 30
sep = (df_in['ema20_1d'] - df_in['ema50_1d']).abs() / df_in['ema50_1d']
macro_ok = sep <= 0.02

print(f'\nIndividual condition rates (in-sample):')
print(f'  ADX<20            : {adx_ok.mean():.1%}')
print(f'  BB-width pctile<30: {bbw_ok.mean():.1%}')
print(f'  |EMA20d-EMA50d|<=2%: {macro_ok.mean():.1%}')

print(f'\nPairwise AND:')
print(f'  ADX AND BBW       : {(adx_ok & bbw_ok).mean():.1%}')
print(f'  ADX AND MACRO     : {(adx_ok & macro_ok).mean():.1%}')
print(f'  BBW AND MACRO     : {(bbw_ok & macro_ok).mean():.1%}')

print(f'\nTriple AND (current PARAMS): {(adx_ok & bbw_ok & macro_ok).mean():.1%}')

# What about looser thresholds?
print(f'\n--- Relaxed candidates ---')
for adx_th, bbw_th, sep_th in [
    (25, 40, 0.03), (25, 50, 0.03), (30, 50, 0.04),
    (22, 40, 0.025), (25, 100, 0.04), (100, 40, 0.03),
    (100, 50, 0.04), (100, 100, 0.04),  # macro-only variants
    (25, 100, 100), (100, 40, 100),
]:
    adx_c = df_in['adx14'] < adx_th
    bbw_c = df_in['bb_width_pctile'] < bbw_th
    mac_c = sep <= sep_th
    pct = (adx_c & bbw_c & mac_c).mean()
    print(f'  adx<{adx_th}, bbw<{bbw_th}, |sep|<={sep_th}: {pct:.1%}')

# Frequency of OR-extremes inside RANGE-with-current-PARAMS
range_mask = adx_ok & bbw_ok & macro_ok
extreme_long = (df_in['rsi14'] <= 30) | (df_in['bb_pct'] <= 0.10)
extreme_short = (df_in['rsi14'] >= 70) | (df_in['bb_pct'] >= 0.90)

print(f'\nExtreme bar frequency (across ALL in-sample):')
print(f'  Oversold (RSI<=30 OR BB<=0.10) : {extreme_long.mean():.1%}')
print(f'  Overbought (RSI>=70 OR BB>=0.90): {extreme_short.mean():.1%}')

print(f'\nExtreme bar frequency WITHIN current RANGE ({range_mask.mean():.1%}):')
if range_mask.sum() > 0:
    print(f'  Oversold in RANGE : {(extreme_long & range_mask).sum()} / {range_mask.sum()} = '
          f'{(extreme_long & range_mask).sum() / range_mask.sum():.1%}')
    print(f'  Overbought in RANGE: {(extreme_short & range_mask).sum()} / {range_mask.sum()} = '
          f'{(extreme_short & range_mask).sum() / range_mask.sum():.1%}')

# Looser RANGE definitions: how many extreme bars do they give us?
print(f'\n--- Extreme-bars-in-RANGE for different RANGE defs ---')
for label, rng in [
    ('triple AND (current)', adx_ok & bbw_ok & macro_ok),
    ('adx<25 AND bbw<40 AND |sep|<3%', (df_in['adx14']<25) & (df_in['bb_width_pctile']<40) & (sep<=0.03)),
    ('adx<25 AND bbw<50', (df_in['adx14']<25) & (df_in['bb_width_pctile']<50)),
    ('adx<25 AND |sep|<3%', (df_in['adx14']<25) & (sep<=0.03)),
    ('bbw<40 AND |sep|<3%', (df_in['bb_width_pctile']<40) & (sep<=0.03)),
    ('adx<25 only', (df_in['adx14']<25)),
    ('bbw<40 only', (df_in['bb_width_pctile']<40)),
    ('macro<3% only', sep<=0.03),
    ('macro<5% only', sep<=0.05),
]:
    n = (extreme_long & rng).sum()
    pct = rng.mean()
    print(f'  {label:<40} RANGE={pct:5.1%}  oversold-in-RANGE bars={n:>4}')
