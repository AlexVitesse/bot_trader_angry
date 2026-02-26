"""
Analiza la distribucion de regimenes detectados en el backtest V10.
Objetivo: entender por que V10 es demasiado conservador.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from regime_detector import RegimeDetector, MarketRegime

DATA_DIR = Path('data')

# Cargar datos
print("="*60)
print("ANALISIS DE DISTRIBUCION DE REGIMENES")
print("="*60)

# Cargar pares
pairs = [
    "BTC_USDT", "ETH_USDT", "BNB_USDT", "SOL_USDT", "XRP_USDT",
    "ADA_USDT", "AVAX_USDT", "DOGE_USDT", "LINK_USDT", "DOT_USDT"
]

detector = RegimeDetector()

def load_data(pair):
    """Carga datos OHLCV 4h."""
    for suffix in ['_4h_v95.parquet', '_4h_history.parquet']:
        cache = DATA_DIR / f'{pair}{suffix}'
        if cache.exists():
            df = pd.read_parquet(cache)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            return df
    return None

# Analizar ultimo ano
print("\n[1] ULTIMO ANO (Feb 2025 - Feb 2026)")
print("-"*60)

regime_counts = {r.value: 0 for r in MarketRegime}
total_candles = 0

for pair in pairs:
    df = load_data(pair)
    if df is None:
        continue

    # Filtrar ultimo ano
    df = df[(df.index >= '2025-02-01') & (df.index < '2026-02-24')]

    if len(df) < 200:
        continue

    # Detectar regimenes
    regimes = detector.detect_regime_series(df)

    for regime in MarketRegime:
        count = (regimes['regime'] == regime.value).sum()
        regime_counts[regime.value] += count

    total_candles += len(df)

print(f"\nTotal velas analizadas: {total_candles:,}")
print(f"\nDistribucion de regimenes:")
for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
    pct = count / total_candles * 100 if total_candles > 0 else 0
    bar = "=" * int(pct / 2)
    print(f"  {regime:<12}: {count:>6} ({pct:>5.1f}%) {bar}")

# Calcular porcentaje de tiempo en LATERAL (no operable)
lateral_pct = regime_counts.get('lateral', 0) / total_candles * 100 if total_candles > 0 else 0
operable_pct = 100 - lateral_pct - (regime_counts.get('unknown', 0) / total_candles * 100 if total_candles > 0 else 0)

print(f"\n>> Tiempo OPERABLE: {operable_pct:.1f}%")
print(f">> Tiempo NO OPERABLE (lateral): {lateral_pct:.1f}%")


# Analizar bear market 2022
print("\n" + "="*60)
print("[2] BEAR MARKET 2022")
print("-"*60)

regime_counts_bear = {r.value: 0 for r in MarketRegime}
total_candles_bear = 0

for pair in pairs:
    df = load_data(pair)
    if df is None:
        continue

    df = df[(df.index >= '2022-01-01') & (df.index < '2023-01-01')]

    if len(df) < 200:
        continue

    regimes = detector.detect_regime_series(df)

    for regime in MarketRegime:
        count = (regimes['regime'] == regime.value).sum()
        regime_counts_bear[regime.value] += count

    total_candles_bear += len(df)

print(f"\nTotal velas analizadas: {total_candles_bear:,}")
print(f"\nDistribucion de regimenes:")
for regime, count in sorted(regime_counts_bear.items(), key=lambda x: -x[1]):
    pct = count / total_candles_bear * 100 if total_candles_bear > 0 else 0
    bar = "=" * int(pct / 2)
    print(f"  {regime:<12}: {count:>6} ({pct:>5.1f}%) {bar}")

lateral_pct_bear = regime_counts_bear.get('lateral', 0) / total_candles_bear * 100 if total_candles_bear > 0 else 0
operable_pct_bear = 100 - lateral_pct_bear - (regime_counts_bear.get('unknown', 0) / total_candles_bear * 100 if total_candles_bear > 0 else 0)

print(f"\n>> Tiempo OPERABLE: {operable_pct_bear:.1f}%")
print(f">> Tiempo NO OPERABLE (lateral): {lateral_pct_bear:.1f}%")


# Comparacion
print("\n" + "="*60)
print("COMPARACION DE PERIODOS")
print("="*60)

print(f"\n{'Metrica':<25} {'Ultimo Ano':>15} {'Bear 2022':>15}")
print("-"*60)
print(f"{'Tiempo LATERAL':<25} {lateral_pct:>14.1f}% {lateral_pct_bear:>14.1f}%")
print(f"{'Tiempo OPERABLE':<25} {operable_pct:>14.1f}% {operable_pct_bear:>14.1f}%")

# Sugerir ajustes
print("\n" + "="*60)
print("DIAGNOSTICO")
print("="*60)

if lateral_pct > 40:
    print(f"""
PROBLEMA: El ultimo ano tiene {lateral_pct:.0f}% de tiempo en LATERAL.
Esto explica por que V10 filtra tantos trades.

POSIBLES AJUSTES:
1. Reducir chop_lateral_threshold de 55 a 50 o 45
   >> Menos tiempo clasificado como lateral

2. Reducir adx_weak_threshold de 20 a 15
   >> Permite operar en tendencias mas debiles

3. Crear regimen "WEAK_TREND" intermedio
   >> En vez de NO operar, operar con posicion reducida

4. Permitir operar en LATERAL con scalping
   >> TP/SL muy pequenos (0.5%/0.25%)
""")
else:
    print(f"Tiempo lateral ({lateral_pct:.0f}%) parece razonable.")
