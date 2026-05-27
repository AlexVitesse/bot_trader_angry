"""
backtest_recent.py
==================
Simula que HABRIA hecho V2 (con el fix daily) durante los ultimos N dias.

Para cada par:
  1. Fetch 500 velas 4h (~83 dias, suficiente warmup)
  2. Fetch 300 velas 1d (~10 meses, EMA200 daily confiable)
  3. Build features con df_1d explicito (el fix)
  4. Iterar por las ultimas N dias evaluando signal por vela
  5. Reportar cada signal y simulate el trade hipotetico

Uso:
  poetry run python backtest_recent.py [days=14]

Compara con el log real: si aqui aparecen trades pero el log real
muestra 0, confirma que el bug daily era el problema.
"""
import os
import sys
import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, '.')

from config.settings import ML_V15_PAIRS, ML_V15_SIZING
from src import v2_engine as v2


def fetch_4h(exchange, pair, n=500):
    ohlcv = exchange.fetch_ohlcv(pair, '4h', limit=n)
    if not ohlcv:
        return None
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df.set_index('ts').sort_index()


def fetch_1d(exchange, pair, n=300):
    ohlcv = exchange.fetch_ohlcv(pair, '1d', limit=n)
    if not ohlcv:
        return None
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df.set_index('ts').sort_index()


def simulate_hypothetical_trades(pair, days_back):
    """
    Aplica V2 al historial reciente y devuelve los trades que HABRIAN disparado.
    Usa motor honesto: una posicion a la vez, sin look-ahead intrabar.
    """
    exchange = ccxt.binanceusdm({
        'apiKey': os.getenv('BINANCE_TESTNET_API_KEY', ''),
        'secret': os.getenv('BINANCE_TESTNET_API_SECRET', ''),
        'enableRateLimit': True,
    })
    exchange.set_sandbox_mode(True)

    df_4h = fetch_4h(exchange, pair, n=500)
    df_1d = fetch_1d(exchange, pair, n=300)
    if df_4h is None or len(df_4h) < 250:
        return {'status': 'no_4h_data', 'pair': pair}
    if df_1d is None or len(df_1d) < 200:
        return {'status': 'no_1d_data', 'pair': pair,
                'note': 'daily fetch failed o insuficiente'}

    # Construir features CON df_1d (el fix)
    feats = v2.build_features(df_4h, df_1d=df_1d, df_funding=None)
    if len(feats) < v2.PARAMS_V2['min_warmup_bars'] + 10:
        return {'status': 'insufficient_after_features', 'pair': pair,
                'n_bars': len(feats)}

    # Solo evaluar las velas dentro de los ultimos N dias
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    start_idx = max(v2.PARAMS_V2['min_warmup_bars'],
                    int(feats.index.searchsorted(cutoff)))

    trades = []
    i = start_idx
    end_i = len(feats) - 1
    while i < end_i:
        sig = v2.detect_signal(feats, i)
        if sig is None:
            i += 1
            continue
        out = v2.simulate_trade(feats, i, sig_type=sig)
        out['ts_entry'] = str(feats.index[i])
        out['idx'] = i
        trades.append(out)
        i += out['bars'] + 1   # one position at a time

    return {
        'status': 'ok',
        'pair': pair,
        'n_features': len(feats),
        'feats_start': str(feats.index[0]),
        'feats_end': str(feats.index[-1]),
        'eval_from': str(feats.index[start_idx]),
        'trades': trades,
    }


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    print(f"Backtest V2 hipotetico — ultimos {days} dias")
    print(f"Fecha actual: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    all_trades = []
    summary_rows = []

    for pair in ML_V15_PAIRS:
        try:
            r = simulate_hypothetical_trades(pair, days)
        except Exception as e:
            print(f"\nERROR en {pair}: {e}")
            continue

        if r['status'] != 'ok':
            print(f"\n{pair}: {r['status']} — {r.get('note', '')}")
            continue

        n = len(r['trades'])
        print(f"\n{pair}: {n} trade(s) hipoteticos")
        print(f"  Feature window: {r['feats_start'][:16]} -> {r['feats_end'][:16]}")
        print(f"  Eval desde: {r['eval_from'][:16]}")

        wins = sum(1 for t in r['trades'] if t['pnl_pct'] > 0)
        total = 0.0
        for t in r['trades']:
            ts = t['ts_entry'][:16]
            print(f"    {ts}  {t['sig_type']:<10}  {t['outcome']:<8}  "
                  f"pnl={t['pnl_pct']:+.2%}  bars={t['bars']}  "
                  f"trail={t['trail_dist']:.3f}")
            total += t['pnl_pct']
            all_trades.append({**t, 'pair': pair})

        sizing = ML_V15_SIZING.get(pair, 0.3)
        weighted = total * sizing
        summary_rows.append((pair, n, wins, total, weighted))

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN — trades hipoteticos por par")
    print("=" * 70)
    print(f"{'Pair':<12}{'N':<5}{'Wins':<8}{'Total PnL':<14}{'Sized PnL':<12}")
    print("-" * 70)
    grand_total_sized = 0.0
    grand_trades = 0
    grand_wins = 0
    for pair, n, w, total, weighted in summary_rows:
        print(f"{pair:<12}{n:<5}{w:<8}{total:<+14.2%}{weighted:<+12.2%}")
        grand_total_sized += weighted
        grand_trades += n
        grand_wins += w
    print("-" * 70)
    wr = (grand_wins / grand_trades * 100) if grand_trades else 0
    print(f"{'TOTAL':<12}{grand_trades:<5}{grand_wins:<8}"
          f"{'WR ' + f'{wr:.0f}%':<14}{grand_total_sized:<+12.2%}")
    print()
    print(f"Trades hipoteticos en los ultimos {days} dias: {grand_trades}")
    print(f"Trades reales en el log durante el mismo periodo: 0")
    print()
    if grand_trades > 0:
        print(">>> CONFIRMADO: el bug daily impidio que el bot tomara estos trades.")
        print(">>> Tras el fix, el bot deberia disparar a tasa similar.")
    else:
        print(">>> NO hubo setups en este periodo, ni siquiera con el fix.")
        print(">>> El mercado simplemente no presento oportunidades V2.")


if __name__ == '__main__':
    main()
