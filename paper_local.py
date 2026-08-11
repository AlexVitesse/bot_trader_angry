"""
paper_local.py — Paper trade V2 en tu PC, aislado de produccion.
================================================================
Corre el MISMO motor de senales que el bot (MLStrategyV15.generate_signals),
pero:
  - Sin claves API: solo datos publicos de Binance futures.
  - Sin ordenes, sin Telegram, sin yield manager, sin la DB de prod.
  - Trailing por CIERRE DE VELA 4h -> replica exacta del backtest validado
    (prod usa ticks de 30s, que es justo la divergencia a medir).

Uso:
  # que habria hecho en los ultimos 90 dias + arranca a vigilar en vivo
  poetry run python paper_local.py

  # solo el replay historico, sin quedarse escuchando
  poetry run python paper_local.py --replay-only --days 180

Estado en data/paper_local.json (posiciones) y logs/paper_local.jsonl (trades).
Se puede matar y relanzar: retoma donde iba.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config.settings import ML_V15_PAIRS, ML_V15_SIZING, ML_RISK_PER_TRADE
from src import v2_engine as v2
from src.ml_strategy_v15 import MLStrategyV15, V2_LOOKBACK

CANDLE_HOURS = (0, 4, 8, 12, 16, 20)

# Perfil activo. Cada perfil escribe sus propios ficheros, asi que se pueden
# correr dos instancias en paralelo sin pisarse.
CFG = {'name': 'conservador', 'leverage': 1.0, 'no_short': False}


def state_file() -> Path:
    return ROOT / 'data' / f"paper_{CFG['name']}.json"


def trades_file() -> Path:
    return ROOT / 'logs' / f"paper_{CFG['name']}.jsonl"


# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} | {msg}", flush=True)


def public_exchange() -> ccxt.Exchange:
    """Mismo cliente publico que ml_bot._init_exchange_public (sin auth)."""
    return ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
    })


def load_state() -> dict:
    if state_file().exists():
        return json.loads(state_file().read_text(encoding='utf-8'))
    return {'positions': {}, 'closed': 0, 'pnl_pct_sum': 0.0}


def save_state(state: dict) -> None:
    state_file().parent.mkdir(exist_ok=True)
    state_file().write_text(json.dumps(state, indent=2), encoding='utf-8')


def record_trade(trade: dict) -> None:
    trades_file().parent.mkdir(exist_ok=True)
    with trades_file().open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(trade) + '\n')


def check_contract(sig: dict) -> None:
    """El payload debe cumplir lo que exige portfolio_manager.open_position.
    Si esto revienta, el bot de prod tampoco podria ejecutar la senal."""
    assert isinstance(sig.get('direction'), int), \
        f"direction debe ser int (1/-1), llego {sig.get('direction')!r}"
    assert sig['direction'] in (1, -1), f"direction invalido: {sig['direction']}"
    assert isinstance(sig.get('price'), (int, float)) and sig['price'] > 0, \
        f"'price' ausente o invalido: {sig.get('price')!r}"
    assert sig.get('sl_pct', 0) > 0, "sl_pct debe ser > 0"


def klines(ex, pair: str, timeframe: str, limit: int) -> pd.DataFrame:
    raw = ex.fetch_ohlcv(pair, timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df.set_index('ts').sort_index()


# ---------------------------------------------------------------------------
# 1) replay historico — feedback inmediato, sin esperar semanas
# ---------------------------------------------------------------------------
def replay(ex, days: int) -> None:
    since = pd.Timestamp.now(tz='UTC') - timedelta(days=days)
    # el replay debe usar el MISMO perfil que el loop en vivo, si no enseña
    # trades que el bot no habria tomado
    params = {**v2.PARAMS_V2, 'f_enable_short': not CFG['no_short']}
    lev = CFG['leverage']
    log(f"Replay de los ultimos {days} dias sobre {len(ML_V15_PAIRS)} pares "
        f"[{CFG['name']}, {lev}x]\n")

    total, wins, pnl_sum = 0, 0, 0.0
    for pair in ML_V15_PAIRS:
        try:
            feats = v2.build_features(klines(ex, pair, '4h', 1500),
                                      df_1d=klines(ex, pair, '1d', 500))
        except Exception as e:
            log(f"  {pair}: error bajando datos: {e}")
            continue

        idxs = [i for i, t in enumerate(feats.index) if t >= since]
        i, end = (idxs[0] if idxs else len(feats)), len(feats) - 1
        trades = []
        while i < end:                      # una posicion a la vez por par
            sig = v2.detect_signal(feats, i, params)
            if sig is None:
                i += 1
                continue
            out = v2.simulate_trade(feats, i, params, sig_type=sig)
            trades.append((str(feats.index[i])[:16], sig, out['outcome'],
                           out['pnl_pct'] * lev * 100))
            i += out['bars'] + 1

        total += len(trades)
        wins += sum(1 for t in trades if t[3] > 0)
        pnl_sum += sum(t[3] for t in trades)
        mult = ML_V15_SIZING.get(pair, 0.3)
        print(f"  {pair:<11} {len(trades):>2} senales  (sizing {mult}x)")
        for ts, sig, outcome, pnl in trades:
            print(f"      {ts}  {sig:<8} {outcome:<8} {pnl:+6.2f}%")

    wr = (wins / total * 100) if total else 0.0
    print(f"\n  TOTAL: {total} senales | WR {wr:.0f}% | suma {pnl_sum:+.2f}%")
    print("  (suma de % por trade, sin ponderar por sizing ni capital)\n")


# ---------------------------------------------------------------------------
# 2) gestion de posiciones abiertas — trailing por cierre de vela 4h
# ---------------------------------------------------------------------------
def update_positions(ex, state: dict) -> None:
    for pair, pos in list(state['positions'].items()):
        try:
            df = klines(ex, pair, '4h', 5)
        except Exception as e:
            log(f"[{pair}] no pude actualizar posicion: {e}")
            continue

        bar = df.iloc[-2]                    # ultima vela CERRADA
        if str(df.index[-2]) == pos.get('last_bar'):
            continue                          # esta vela ya se proceso
        pos['last_bar'] = str(df.index[-2])
        pos['bars'] += 1
        hi, lo, close = float(bar['high']), float(bar['low']), float(bar['close'])
        trail, exit_price, reason = pos['trail_dist'], None, None

        if pos['direction'] == 1:
            if lo <= pos['sl']:               # 1) salida con el SL PREVIO
                exit_price, reason = pos['sl'], 'SL'
            else:                             # 2) subir el SL para la siguiente
                pos['peak'] = max(pos['peak'], hi)
                pos['sl'] = max(pos['sl'], pos['peak'] * (1 - trail))
        else:
            if hi >= pos['sl']:
                exit_price, reason = pos['sl'], 'SL'
            else:
                pos['peak'] = min(pos['peak'], lo)
                pos['sl'] = min(pos['sl'], pos['peak'] * (1 + trail))

        if exit_price is None and pos['bars'] >= pos['max_bars']:
            exit_price, reason = close, 'TIMEOUT'
        if exit_price is None:
            continue

        gross = (exit_price - pos['entry']) / pos['entry'] * pos['direction']
        # Retorno SOBRE LA CUENTA, no sobre el notional: el bot dimensiona por
        # riesgo (notional = equity*risk/SL), asi que el impacto en la cuenta es
        # el movimiento del precio escalado por risk/SL. Sin esto los numeros del
        # paper no serian comparables con los de produccion.
        escala = ML_RISK_PER_TRADE / pos['trail_dist']
        pnl = (gross - 2 * v2.COMMISSION) * escala * CFG['leverage'] * 100
        trade = {**pos, 'pair': pair, 'exit_price': exit_price,
                 'exit_reason': reason, 'pnl_pct': round(pnl, 3),
                 'closed_at': datetime.now(timezone.utc).isoformat()}
        record_trade(trade)
        state['closed'] += 1
        state['pnl_pct_sum'] = round(state['pnl_pct_sum'] + pnl, 3)
        del state['positions'][pair]
        log(f"CIERRE {pair} {reason} @ {exit_price:.4f} -> {pnl:+.2f}% "
            f"({pos['bars']} velas) | acumulado {state['pnl_pct_sum']:+.2f}%")


# ---------------------------------------------------------------------------
# 3) señales en vivo — el MISMO metodo que llama el bot de prod
# ---------------------------------------------------------------------------
def check_signals(ex, strategy, state: dict) -> None:
    open_pairs = set(state['positions'].keys())
    signals = strategy.generate_signals(ex, open_pairs)

    if not signals:
        log(f"Sin senales | abiertas: {len(open_pairs)}")
        return

    for sig in signals:
        check_contract(sig)                   # falla ruidosamente, no en silencio
        pair, d = sig['pair'], sig['direction']
        if CFG['no_short'] and d == -1:
            log(f"omitida {pair} SHORT ({sig.get('setup')}) — perfil sin F_SHORT")
            continue
        state['positions'][pair] = {
            'direction': d,
            'side': 'LONG' if d == 1 else 'SHORT',
            'setup': sig.get('setup', ''),
            'entry': sig['price'],
            'trail_dist': sig.get('trail_fixed_dist') or sig['sl_pct'],
            'max_bars': sig.get('max_bars', 40),
            'sizing_mult': sig.get('sizing_mult', 1.0),
            'sl': sig['price'] * (1 - sig['sl_pct'] * d),
            'peak': sig['price'],
            'bars': 0,
            'last_bar': None,
            'opened_at': datetime.now(timezone.utc).isoformat(),
        }
        log(f"APERTURA {pair} {sig.get('setup')} {state['positions'][pair]['side']} "
            f"@ {sig['price']:.4f} trail={state['positions'][pair]['trail_dist']:.3f}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=90, help='dias de replay inicial')
    ap.add_argument('--replay-only', action='store_true')
    ap.add_argument('--no-replay', action='store_true')
    ap.add_argument('--perfil', choices=['conservador', 'agresivo'],
                    default='conservador',
                    help='conservador: 1x sin F_SHORT | agresivo: 2x sin F_SHORT')
    ap.add_argument('--leverage', type=float,
                    help='sobreescribe el leverage del perfil')
    ap.add_argument('--con-short', action='store_true',
                    help='incluir F_SHORT (PF 0.88, p=0.639 — se espera que pierda, '
                         'pero es lo unico que opera en bear)')
    args = ap.parse_args()

    CFG['name'] = args.perfil + ('_short' if args.con_short else '')
    CFG['no_short'] = not args.con_short      # F_SHORT: PF 0.88, p=0.639 -> fuera
    CFG['leverage'] = args.leverage if args.leverage else (
        2.0 if args.perfil == 'agresivo' else 1.0)

    ex = public_exchange()
    log(f"paper_local [{CFG['name']}] leverage {CFG['leverage']}x | "
        f"pares: {', '.join(ML_V15_PAIRS)}")
    log("SIN claves API, SIN ordenes, SIN Telegram — no toca produccion\n")

    if not args.no_replay:
        replay(ex, args.days)
    if args.replay_only:
        return

    strategy = MLStrategyV15()
    state = load_state()
    log(f"Vigilando velas 4h (Ctrl-C para salir) | "
        f"abiertas={len(state['positions'])} cerradas={state['closed']} "
        f"acumulado={state['pnl_pct_sum']:+.2f}%")

    last_candle = None
    ultimo_latido = 0.0
    while True:
        try:
            now = datetime.now(timezone.utc)
            key = now.strftime('%Y-%m-%d-%H')
            if now.hour in CANDLE_HOURS and now.minute >= 2 and key != last_candle:
                last_candle = key
                log(f"--- vela 4h {key} ---")
                update_positions(ex, state)   # primero cerrar, luego abrir
                check_signals(ex, strategy, state)
                save_state(state)
            # latido cada 15 min: entre velas 4h no pasa nada y sin esto no se
            # distingue "esperando" de "colgado" — el fallo que costo 2 meses
            if time.time() - ultimo_latido > 900:
                ultimo_latido = time.time()
                prox = next((h for h in CANDLE_HOURS if h > now.hour), CANDLE_HOURS[0])
                log(f"vivo | proxima vela {prox:02d}:02 UTC | "
                    f"abiertas={len(state['positions'])} cerradas={state['closed']} "
                    f"acumulado={state['pnl_pct_sum']:+.2f}%")
            time.sleep(30)
        except KeyboardInterrupt:
            save_state(state)
            log(f"Detenido | cerradas={state['closed']} "
                f"acumulado={state['pnl_pct_sum']:+.2f}%")
            return
        except Exception as e:
            log(f"ERROR en loop: {type(e).__name__}: {e}")
            time.sleep(60)


if __name__ == '__main__':
    main()
