"""
V14 Cross Validation - Prueba en ETH + Datos Sinteticos
========================================================
1. Probar reglas de BTC en ETH sin modificar
2. Generar datos sinteticos (bull, bear, mixed)
3. Probar sistema en datos nunca vistos
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')

# =============================================================================
# SETUP DEFINITIONS (Las mismas de BTC - NO MODIFICAR)
# =============================================================================

def compute_features(df):
    """Features - igual que BTC."""
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    feat = pd.DataFrame(index=df.index)

    adx_df = ta.adx(h, l, c, length=14)
    if adx_df is not None:
        feat['adx'] = adx_df.iloc[:, 0]
        feat['di_plus'] = adx_df.iloc[:, 1]
        feat['di_minus'] = adx_df.iloc[:, 2]
        feat['di_diff'] = feat['di_plus'] - feat['di_minus']

    chop = ta.chop(h, l, c, length=14)
    feat['chop'] = chop if chop is not None else 50

    feat['ema20'] = ta.ema(c, length=20)
    feat['ema50'] = ta.ema(c, length=50)
    feat['ema200'] = ta.ema(c, length=200)
    feat['ema20_dist'] = (c - feat['ema20']) / feat['ema20'] * 100
    feat['ema200_dist'] = (c - feat['ema200']) / feat['ema200'] * 100
    feat['ema20_slope'] = feat['ema20'].pct_change(5) * 100

    feat['atr_pct'] = ta.atr(h, l, c, length=14) / c * 100
    bb = ta.bbands(c, length=20)
    if bb is not None:
        feat['bb_upper'] = bb.iloc[:, 2]
        feat['bb_lower'] = bb.iloc[:, 0]
        feat['bb_mid'] = bb.iloc[:, 1]
        feat['bb_width'] = (feat['bb_upper'] - feat['bb_lower']) / feat['bb_mid'] * 100
        feat['bb_pct'] = (c - feat['bb_lower']) / (feat['bb_upper'] - feat['bb_lower'])

    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)
    stoch = ta.stoch(h, l, c, k=14, d=3)
    if stoch is not None:
        feat['stoch_k'] = stoch.iloc[:, 0]

    feat['ret_5'] = c.pct_change(5) * 100
    feat['ret_20'] = c.pct_change(20) * 100

    feat['vol_ratio'] = v / v.rolling(20).mean()
    feat['high_20'] = h.rolling(20).max()
    feat['low_20'] = l.rolling(20).min()
    feat['range_pos'] = (c - feat['low_20']) / (feat['high_20'] - feat['low_20'])
    feat['consec_down'] = (c < c.shift(1)).rolling(10).sum()
    feat['consec_up'] = (c > c.shift(1)).rolling(10).sum()

    return feat


SETUP_DEFINITIONS = {
    'PULLBACK_UPTREND': {
        'direction': 'long',
        'conditions': lambda r: r.get('rsi14', 50) < 40 and r.get('bb_pct', 0.5) < 0.3 and r.get('ema200_dist', 0) > 0,
    },
    'OVERSOLD_EXTREME': {
        'direction': 'long',
        'conditions': lambda r: r.get('rsi14', 50) < 25 and r.get('bb_pct', 0.5) < 0.2,
    },
    'SUPPORT_BOUNCE': {
        'direction': 'long',
        'conditions': lambda r: r.get('range_pos', 0.5) < 0.15 and r.get('rsi14', 50) < 35,
    },
    'CAPITULATION': {
        'direction': 'long',
        'conditions': lambda r: r.get('consec_down', 0) >= 4 and r.get('rsi14', 50) < 30 and r.get('vol_ratio', 1) > 1.5,
    },
    'RALLY_DOWNTREND': {
        'direction': 'short',
        'conditions': lambda r: r.get('rsi14', 50) > 60 and r.get('bb_pct', 0.5) > 0.7 and r.get('ema200_dist', 0) < 0,
    },
    'OVERBOUGHT_EXTREME': {
        'direction': 'short',
        'conditions': lambda r: r.get('rsi14', 50) > 75 and r.get('bb_pct', 0.5) > 0.8,
    },
    'RESISTANCE_REJECTION': {
        'direction': 'short',
        'conditions': lambda r: r.get('range_pos', 0.5) > 0.85 and r.get('rsi14', 50) > 65,
    },
    'EXHAUSTION': {
        'direction': 'short',
        'conditions': lambda r: r.get('consec_up', 0) >= 4 and r.get('rsi14', 50) > 70 and r.get('vol_ratio', 1) > 1.5,
    },
}


def detect_setups(df, feat):
    """Detecta setups."""
    setups = []
    for idx in feat.index:
        row = feat.loc[idx].to_dict()
        if pd.isna(row.get('rsi14')) or pd.isna(row.get('bb_pct')):
            continue

        for setup_name, setup_def in SETUP_DEFINITIONS.items():
            try:
                if setup_def['conditions'](row):
                    setups.append({
                        'idx': idx,
                        'setup': setup_name,
                        'direction': setup_def['direction']
                    })
            except:
                continue

    return pd.DataFrame(setups) if setups else None


def get_setup_outcome(df, idx, direction, tp=0.03, sl=0.015):
    """Obtiene resultado de un setup."""
    if idx not in df.index:
        return None

    entry_price = df.loc[idx, 'close']
    future = df.loc[idx:].head(50)

    for future_idx in future.index[1:]:
        future_price = df.loc[future_idx, 'close']

        if direction == 'long':
            pnl = (future_price - entry_price) / entry_price
        else:
            pnl = (entry_price - future_price) / entry_price

        if pnl >= tp:
            return {'outcome': 1, 'pnl': pnl}
        elif pnl <= -sl:
            return {'outcome': 0, 'pnl': -sl}

    return None


def test_on_data(df, name):
    """Prueba el sistema en un dataset."""
    print(f"\n{'='*60}")
    print(f"PRUEBA EN: {name}")
    print(f"{'='*60}")

    print(f"Datos: {len(df):,} candles")
    if hasattr(df.index, 'min'):
        print(f"Periodo: {df.index.min()} a {df.index.max()}")

    # Features
    feat = compute_features(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # Setups
    setups_df = detect_setups(df, feat)

    if setups_df is None or len(setups_df) == 0:
        print("No se detectaron setups")
        return None

    print(f"Setups detectados: {len(setups_df)}")

    # Outcomes
    results = []
    for _, row in setups_df.iterrows():
        result = get_setup_outcome(df, row['idx'], row['direction'])
        if result:
            result['setup'] = row['setup']
            result['direction'] = row['direction']
            results.append(result)

    if not results:
        print("No hay resultados")
        return None

    results_df = pd.DataFrame(results)
    n = len(results_df)
    wins = results_df['outcome'].sum()
    wr = wins / n * 100
    total_pnl = results_df['pnl'].sum() * 100

    print(f"\nRESULTADOS GENERALES:")
    print(f"  Trades: {n}")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  PnL Total: {total_pnl:+.1f}%")

    # Por setup
    print(f"\nPOR SETUP:")
    for setup in results_df['setup'].unique():
        subset = results_df[results_df['setup'] == setup]
        if len(subset) >= 3:
            sub_wr = subset['outcome'].mean() * 100
            sub_pnl = subset['pnl'].sum() * 100
            status = "[OK]" if sub_pnl > 0 else "[BAD]"
            print(f"  {setup:<25} {len(subset):>4} trades, WR {sub_wr:>5.1f}%, PnL {sub_pnl:>+7.1f}% {status}")

    # Por direccion
    print(f"\nPOR DIRECCION:")
    for direction in ['long', 'short']:
        subset = results_df[results_df['direction'] == direction]
        if len(subset) > 0:
            sub_wr = subset['outcome'].mean() * 100
            sub_pnl = subset['pnl'].sum() * 100
            print(f"  {direction.upper():<10} {len(subset):>4} trades, WR {sub_wr:>5.1f}%, PnL {sub_pnl:>+7.1f}%")

    return {
        'name': name,
        'trades': n,
        'wr': wr,
        'pnl': total_pnl
    }


# =============================================================================
# GENERADOR DE DATOS SINTETICOS
# =============================================================================

def generate_synthetic_ohlcv(n_candles, market_type='mixed', start_price=100, seed=None):
    """
    Genera datos OHLCV sinteticos con Geometric Brownian Motion.

    market_type:
    - 'bull': drift positivo, volatilidad moderada
    - 'bear': drift negativo, volatilidad alta
    - 'mixed': periodos alternados
    - 'range': sin drift, volatilidad baja
    """
    if seed:
        np.random.seed(seed)

    # Parametros por tipo de mercado (por vela de 4h)
    params = {
        'bull': {'drift': 0.0015, 'vol': 0.02},      # +0.15% por vela, 2% vol
        'bear': {'drift': -0.0012, 'vol': 0.025},    # -0.12% por vela, 2.5% vol
        'range': {'drift': 0.0001, 'vol': 0.015},    # casi flat, 1.5% vol
        'volatile': {'drift': 0.0, 'vol': 0.04},     # sin tendencia, 4% vol
    }

    dates = pd.date_range(start='2025-01-01', periods=n_candles, freq='4h')

    closes = [start_price]

    if market_type == 'mixed':
        # Alternar entre bull, bear, range cada ~500 velas (~3 meses)
        segment_size = n_candles // 4
        segments = ['bull', 'bear', 'range', 'bull']

        for i in range(1, n_candles):
            segment_idx = min(i // segment_size, len(segments) - 1)
            current_type = segments[segment_idx]
            p = params[current_type]

            returns = np.random.normal(p['drift'], p['vol'])
            new_price = closes[-1] * (1 + returns)
            closes.append(max(new_price, 0.01))
    else:
        p = params.get(market_type, params['range'])
        for i in range(1, n_candles):
            returns = np.random.normal(p['drift'], p['vol'])
            new_price = closes[-1] * (1 + returns)
            closes.append(max(new_price, 0.01))

    closes = np.array(closes)

    # Generar OHLV a partir de close
    highs = closes * (1 + np.abs(np.random.normal(0.005, 0.003, n_candles)))
    lows = closes * (1 - np.abs(np.random.normal(0.005, 0.003, n_candles)))
    opens = np.roll(closes, 1)
    opens[0] = start_price

    # Volumen: base + ruido + correlacion con volatilidad
    base_volume = 1000000
    vol_noise = np.random.lognormal(0, 0.5, n_candles)
    price_changes = np.abs(np.diff(closes, prepend=closes[0])) / closes
    volume = base_volume * vol_noise * (1 + price_changes * 10)

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volume
    }, index=dates)

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("V14 CROSS VALIDATION")
    print("Probando reglas de BTC en datos nunca vistos")
    print("=" * 70)

    all_results = []

    # ===========================================
    # PARTE 1: Probar en ETH real
    # ===========================================
    print("\n" + "#" * 70)
    print("# PARTE 1: PRUEBA EN ETH (datos reales, reglas de BTC)")
    print("#" * 70)

    try:
        eth_df = pd.read_parquet(DATA_DIR / 'ETH_USDT_4h_full.parquet')
        result = test_on_data(eth_df, "ETH (datos reales)")
        if result:
            all_results.append(result)
    except Exception as e:
        print(f"Error cargando ETH: {e}")

    # ===========================================
    # PARTE 2: Datos sinteticos
    # ===========================================
    print("\n" + "#" * 70)
    print("# PARTE 2: DATOS SINTETICOS (nunca vistos)")
    print("#" * 70)

    # 1 año = 365 * 6 = 2190 velas de 4h
    candles_per_year = 2190

    # Bull market puro
    print("\nGenerando mercado BULL puro (1 año)...")
    bull_df = generate_synthetic_ohlcv(candles_per_year, 'bull', start_price=50000, seed=42)
    result = test_on_data(bull_df, "SINTETICO BULL (1 año)")
    if result:
        all_results.append(result)

    # Bear market puro
    print("\nGenerando mercado BEAR puro (1 año)...")
    bear_df = generate_synthetic_ohlcv(candles_per_year, 'bear', start_price=50000, seed=43)
    result = test_on_data(bear_df, "SINTETICO BEAR (1 año)")
    if result:
        all_results.append(result)

    # Mercado lateral/rango
    print("\nGenerando mercado RANGE puro (1 año)...")
    range_df = generate_synthetic_ohlcv(candles_per_year, 'range', start_price=50000, seed=44)
    result = test_on_data(range_df, "SINTETICO RANGE (1 año)")
    if result:
        all_results.append(result)

    # Mercado mixto (mas realista)
    print("\nGenerando mercado MIXED (1 año)...")
    mixed_df = generate_synthetic_ohlcv(candles_per_year, 'mixed', start_price=50000, seed=45)
    result = test_on_data(mixed_df, "SINTETICO MIXED (1 año)")
    if result:
        all_results.append(result)

    # Mercado volatil
    print("\nGenerando mercado VOLATIL (1 año)...")
    volatile_df = generate_synthetic_ohlcv(candles_per_year, 'volatile', start_price=50000, seed=46)
    result = test_on_data(volatile_df, "SINTETICO VOLATIL (1 año)")
    if result:
        all_results.append(result)

    # ===========================================
    # RESUMEN FINAL
    # ===========================================
    print("\n" + "=" * 70)
    print("RESUMEN FINAL - CROSS VALIDATION")
    print("=" * 70)

    if all_results:
        print(f"\n{'Dataset':<30} {'Trades':>8} {'WR':>8} {'PnL':>10} {'Status':>10}")
        print("-" * 70)

        for r in all_results:
            status = "[OK]" if r['pnl'] > 0 else "[FAIL]"
            print(f"{r['name']:<30} {r['trades']:>8} {r['wr']:>7.1f}% {r['pnl']:>+9.1f}% {status:>10}")

        # Estadisticas
        positive = sum(1 for r in all_results if r['pnl'] > 0)
        total = len(all_results)
        avg_pnl = np.mean([r['pnl'] for r in all_results])
        avg_wr = np.mean([r['wr'] for r in all_results])

        print(f"\n{'='*70}")
        print(f"Datasets positivos: {positive}/{total} ({positive/total*100:.0f}%)")
        print(f"WR promedio: {avg_wr:.1f}%")
        print(f"PnL promedio: {avg_pnl:+.1f}%")

        if positive >= total * 0.7:
            print(f"\n[APROBADO] Sistema robusto - funciona en {positive}/{total} escenarios")
        elif positive >= total * 0.5:
            print(f"\n[MARGINAL] Sistema funciona en algunos escenarios")
        else:
            print(f"\n[RECHAZADO] Sistema no generaliza bien")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
