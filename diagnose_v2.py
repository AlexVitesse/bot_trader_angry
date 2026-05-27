"""
diagnose_v2.py
==============
Diagnostica POR QUE v2_engine no esta generando senales en vivo.

Para cada par de ML_V15_PAIRS:
  1. Fetch live OHLCV 4h
  2. Build features con v2_engine
  3. Inspeccionar la ultima vela cerrada: que filtros pasan? cuales bloquean?
  4. Reportar si A o F deberian haber firado

Uso:
  poetry run python diagnose_v2.py
"""
import os
import sys
import ccxt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, '.')

from config.settings import ML_V15_PAIRS
from src import v2_engine as v2


def fetch_4h(exchange, pair, n=500):
    """Fetch 4h candles. n=500 para tener mas warmup que LOOKBACK=250 del bot."""
    ohlcv = exchange.fetch_ohlcv(pair, '4h', limit=n)
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df.set_index('ts').sort_index()


def fetch_1d(exchange, pair, n=300):
    """Fetch daily candles para tener EMA200 confiable."""
    ohlcv = exchange.fetch_ohlcv(pair, '1d', limit=n)
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df.set_index('ts').sort_index()


def diagnose(pair, df_4h, df_1d):
    """Diagnostica una par con ambos: con y sin df_1d explicito."""
    print(f"\n{'=' * 70}")
    print(f"  {pair}")
    print(f"{'=' * 70}")
    print(f"  4h bars: {len(df_4h)} ({df_4h.index[0].date()} -> {df_4h.index[-1].date()})")
    print(f"  1d bars: {len(df_1d)} ({df_1d.index[0].date()} -> {df_1d.index[-1].date()})")

    for label, df_1d_param in [("(SIN df_1d - como hace el bot HOY)", None),
                                ("(CON df_1d explicito - el fix)", df_1d)]:
        print(f"\n  {label}")
        feats = v2.build_features(df_4h, df_1d=df_1d_param)
        if len(feats) < 2:
            print(f"    Insuficientes features: {len(feats)} bars")
            continue
        # Ultima vela cerrada (idx = -2 segun get_live_signal)
        idx = len(feats) - 2
        row = feats.iloc[idx]
        ts_close = feats.index[idx]
        print(f"    Ultima vela cerrada: {ts_close}")
        print(f"    close={row['close']:.4f}, atr_pct={row['atr_pct']:.4f}")
        # Daily regime
        bull = row.get('bull_1d', float('nan'))
        print(f"    bull_1d = {bull}  (1=BULL, 0=BEAR/RANGE)")
        # A filters
        print(f"    --- Filtros A (Donchian LONG) ---")
        donch_h = row.get('donchian_high', float('nan'))
        print(f"    donchian_high(55) = {donch_h:.4f}")
        print(f"    close > donchian_high? {row['close'] > donch_h}  "
              f"(diff: {(row['close'] - donch_h) / donch_h * 100:+.2f}%)")
        vr = row.get('vol_ratio', float('nan'))
        print(f"    vol_ratio = {vr:.2f}  (>= 1.2?: {vr >= 1.2})")
        adx = row.get('adx', float('nan'))
        print(f"    adx = {adx:.1f}  (>= 18?: {adx >= 18})")
        fz = row.get('funding_z', 0)
        print(f"    funding_z = {fz:.2f}  (<= 2.5?: {fz <= 2.5})")
        # F filters
        print(f"    --- Filtros F (Vol compression breakout) ---")
        comp = row.get('compression_sustained', 0)
        print(f"    compression_sustained = {comp}  (==1?: {comp == 1})")
        hi_n = row.get('hi_n', float('nan'))
        lo_n = row.get('lo_n', float('nan'))
        print(f"    hi_n(12) = {hi_n:.4f}  | lo_n(12) = {lo_n:.4f}")
        print(f"    close > hi_n? {row['close'] > hi_n} | close < lo_n? {row['close'] < lo_n}")
        # Resultado
        sig = v2.detect_signal(feats, idx)
        print(f"    --> SIGNAL: {sig if sig else 'None'}")


def main():
    print("Conectando a Binance testnet...")
    ex = ccxt.binanceusdm({
        'apiKey': os.getenv('BINANCE_TESTNET_API_KEY', ''),
        'secret': os.getenv('BINANCE_TESTNET_API_SECRET', ''),
        'enableRateLimit': True,
    })
    ex.set_sandbox_mode(True)

    for pair in ML_V15_PAIRS:
        try:
            df_4h = fetch_4h(ex, pair, n=500)
            df_1d = fetch_1d(ex, pair, n=300)
            diagnose(pair, df_4h, df_1d)
        except Exception as e:
            print(f"\nERROR en {pair}: {e}")

    print(f"\n{'=' * 70}")
    print("HIPOTESIS:")
    print("  Si SIN df_1d todos dan bull_1d=0 pero CON df_1d dan bull_1d=1,")
    print("  el bug es que el bot pasa df_1d=None y v2_engine no puede computar")
    print("  EMA200 daily con solo 250 velas 4h (= 42 dias < 200 dias).")
    print("  Fix: modificar _generate_v2_signal para fetch df_1d explicito.")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
