"""
Portfolio Manager - Gestion de Posiciones Multi-Par
====================================================
Maneja hasta 3 posiciones simultaneas con trailing stops,
risk management profesional, y kill switches.
"""

import sqlite3
import logging
import time
import ccxt
import pandas as pd
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Optional, Dict, List
from pathlib import Path

from config.settings import (
    ML_DB_FILE, ML_MAX_CONCURRENT, ML_MAX_DD_PCT, ML_MAX_DAILY_LOSS_PCT,
    ML_RISK_PER_TRADE, ML_MAX_NOTIONAL_PCT, ML_LEVERAGE, ML_TP_PCT, ML_SL_PCT,
    ML_TRAILING_ACTIVATION, ML_TRAILING_LOCK, ML_MAX_HOLD,
    COMMISSION_RATE, SLIPPAGE_PCT, INITIAL_CAPITAL, ML_ENTRY_LIMIT_TIMEOUT_S,
)

from src.v2_engine import COMMISSION as SIM_COMMISSION
from src.v2_engine import _sim_long_trailing, _sim_short_trailing

logger = logging.getLogger(__name__)

BAR_MS = 4 * 3600 * 1000
# Columnas de ml_trades para medir vivo vs simulado (plan 2.1) y el PnL real
# de Binance (plan 5.2). Se rellenan despues del cierre en
# reconcile_closed_trades: el income tarda en aparecer y la salida simulada
# puede necesitar velas posteriores a la salida real.
TRADE_EXTRA_COLS = {
    'trail_dist': 'REAL', 'max_hold': 'INTEGER',
    'signal_close': 'REAL', 'exit_sim_price': 'REAL',
    'exit_sim_reason': 'TEXT', 'pnl_sim_pct': 'REAL',
    'pnl_real': 'REAL', 'commission_real': 'REAL', 'funding_real': 'REAL',
}


@dataclass
class Position:
    pair: str
    side: str               # "long" or "short"
    direction: int           # 1 = long, -1 = short
    entry_price: float
    quantity: float
    notional: float
    leverage: int
    tp_price: float
    sl_price: float
    tp_pct: float
    sl_pct: float
    atr_pct: float
    trail_active: bool = False
    trail_sl: Optional[float] = None
    peak_price: Optional[float] = None
    regime: str = 'RANGE'
    confidence: float = 0.0
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    bars: int = 0
    max_hold: int = 30
    sl_order_id: Optional[str] = None
    trail_mode: str = 'default'       # 'default' or 'tight' (ADA/SOL)
    trail_fixed_dist: float = 0.0     # 0.008 = 0.8% for tight trailing
    # 'pending' = guardada ANTES de mandar la orden; si el proceso muere entre
    # la orden y el fill, el arranque la adopta con estos parametros del motor
    # en vez de con el TP/SL generico ML_TP_PCT/ML_SL_PCT (AUDITORIA_2026-09 §2.4).
    status: str = 'open'


class PortfolioManager:
    """Gestiona multiples posiciones con risk management profesional."""

    def __init__(self, exchange: ccxt.Exchange, db_path: Path = ML_DB_FILE):
        self.exchange = exchange
        self.db_path = db_path
        self.positions: Dict[str, Position] = {}
        self.balance = INITIAL_CAPITAL
        self.peak_balance = INITIAL_CAPITAL
        self.daily_pnl = 0.0
        self.daily_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        self.paused = False
        self.killed = False
        self.trade_log: List[dict] = []
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """Crea tablas para ML bot."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ml_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    notional REAL NOT NULL,
                    leverage INTEGER NOT NULL,
                    pnl REAL DEFAULT 0.0,
                    exit_reason TEXT DEFAULT '',
                    regime TEXT DEFAULT '',
                    confidence REAL DEFAULT 0.0,
                    commission REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS ml_positions (
                    symbol TEXT PRIMARY KEY,
                    entry_time TEXT NOT NULL,
                    side TEXT NOT NULL,
                    direction INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    notional REAL NOT NULL,
                    leverage INTEGER NOT NULL,
                    tp_price REAL NOT NULL,
                    sl_price REAL NOT NULL,
                    tp_pct REAL NOT NULL,
                    sl_pct REAL NOT NULL,
                    atr_pct REAL NOT NULL,
                    trail_active INTEGER DEFAULT 0,
                    trail_sl REAL,
                    peak_price REAL,
                    regime TEXT DEFAULT '',
                    confidence REAL DEFAULT 0.0,
                    bars INTEGER DEFAULT 0,
                    max_hold INTEGER DEFAULT 30,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS ml_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                -- Foto del mercado en cada senal y fill (docs/GRABACION_DATOS_VIVO.md,
                -- capa A). Se une con ml_trades por (symbol, entry_time).
                CREATE TABLE IF NOT EXISTS ml_exec_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    event TEXT NOT NULL,
                    entry_time TEXT,
                    best_bid REAL, best_ask REAL, spread_bps REAL,
                    depth_bid_1pct REAL, depth_ask_1pct REAL,
                    mark REAL, index_price REAL, premium REAL,
                    funding_rate REAL, next_funding_ts INTEGER,
                    open_interest REAL, taker_ratio_5m REAL,
                    order_type TEXT, filled_qty REAL, avg_price REAL,
                    ref_price REAL, latency_ms REAL
                );
            """)
            conn.commit()
            # Migration: add sl_order_id column if missing
            try:
                conn.execute("ALTER TABLE ml_positions ADD COLUMN sl_order_id TEXT")
                conn.commit()
            except Exception:
                pass  # Column already exists
            # Migration: add strategy column to ml_trades
            try:
                conn.execute("ALTER TABLE ml_trades ADD COLUMN strategy TEXT DEFAULT 'v9'")
                conn.commit()
            except Exception:
                pass  # Column already exists
            # Migration: add trail_mode and trail_fixed_dist for tight trailing (ADA/SOL)
            try:
                conn.execute("ALTER TABLE ml_positions ADD COLUMN trail_mode TEXT DEFAULT 'default'")
                conn.commit()
            except Exception:
                pass
            try:
                conn.execute("ALTER TABLE ml_positions ADD COLUMN trail_fixed_dist REAL DEFAULT 0.0")
                conn.commit()
            except Exception:
                pass
            try:
                conn.execute("ALTER TABLE ml_positions ADD COLUMN status TEXT DEFAULT 'open'")
                conn.commit()
            except Exception:
                pass
            have = {r['name'] for r in conn.execute("PRAGMA table_info(ml_trades)")}
            for col, typ in TRADE_EXTRA_COLS.items():
                if col not in have:
                    conn.execute(f"ALTER TABLE ml_trades ADD COLUMN {col} {typ}")
            conn.commit()
        finally:
            conn.close()

    def _save_position(self, pos: Position):
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO ml_positions
                    (symbol, entry_time, side, direction, entry_price, quantity,
                     notional, leverage, tp_price, sl_price, tp_pct, sl_pct, atr_pct,
                     trail_active, trail_sl, peak_price, regime, confidence, bars,
                     max_hold, sl_order_id, trail_mode, trail_fixed_dist, status,
                     updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pos.pair, pos.entry_time.isoformat(), pos.side, pos.direction,
                pos.entry_price, pos.quantity, pos.notional, pos.leverage,
                pos.tp_price, pos.sl_price, pos.tp_pct, pos.sl_pct, pos.atr_pct,
                1 if pos.trail_active else 0, pos.trail_sl, pos.peak_price,
                pos.regime, pos.confidence, pos.bars, pos.max_hold,
                pos.sl_order_id, pos.trail_mode, pos.trail_fixed_dist, pos.status,
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()
        finally:
            conn.close()

    def _delete_position(self, pair: str):
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM ml_positions WHERE symbol = ?", (pair,))
            conn.commit()
        finally:
            conn.close()

    def _save_trade(self, trade: dict):
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT INTO ml_trades
                    (symbol, entry_time, exit_time, side, entry_price, exit_price,
                     quantity, notional, leverage, pnl, exit_reason, regime,
                     confidence, commission, strategy, trail_dist, max_hold)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade['symbol'], trade['entry_time'], trade['exit_time'],
                trade['side'], trade['entry_price'], trade['exit_price'],
                trade['quantity'], trade['notional'], trade['leverage'],
                trade['pnl'], trade['exit_reason'], trade['regime'],
                trade['confidence'], trade['commission'],
                trade.get('strategy', 'v85_prod'),
                trade.get('trail_dist'), trade.get('max_hold'),
            ))
            conn.commit()
        finally:
            conn.close()

    def _save_state(self, key: str, value: str):
        conn = self._get_conn()
        try:
            conn.execute("INSERT OR REPLACE INTO ml_state (key, value) VALUES (?, ?)",
                         (key, value))
            conn.commit()
        finally:
            conn.close()

    def _snapshot(self, event: str, pair: str, entry_time: datetime, **fill):
        """Graba libro, mark/index/funding, OI y ratio taker en ml_exec_snapshots.
        Nunca bloquea una orden: cada campo que falle queda NULL y cualquier
        otro error solo se loguea. `fill`: order_type, filled_qty, avg_price,
        ref_price, latency_ms."""
        def _try(fn):
            try:
                return fn()
            except Exception:
                return None
        try:
            sym = pair.replace('/', '')
            row = {'ts': datetime.now(timezone.utc).isoformat(), 'symbol': pair,
                   'event': event, 'entry_time': entry_time.isoformat(), **fill}
            ob = _try(lambda: self.exchange.fetch_order_book(pair, 100))
            if ob and ob.get('bids') and ob.get('asks'):
                bid, ask = ob['bids'][0][0], ob['asks'][0][0]
                mid = (bid + ask) / 2
                # ponytail: solo los 100 niveles pedidos; si no llegan al 1% la
                # profundidad sale truncada (sigue sirviendo frente a nuestro notional)
                row.update(
                    best_bid=bid, best_ask=ask, spread_bps=(ask - bid) / mid * 1e4,
                    depth_bid_1pct=sum(p * q for p, q, *_ in ob['bids'] if p >= mid * 0.99),
                    depth_ask_1pct=sum(p * q for p, q, *_ in ob['asks'] if p <= mid * 1.01))
            pi = _try(lambda: self.exchange.fapiPublicGetPremiumIndex({'symbol': sym}))
            if pi:
                mark, idx = float(pi['markPrice']), float(pi['indexPrice'])
                row.update(mark=mark, index_price=idx, premium=mark / idx - 1,
                           funding_rate=float(pi['lastFundingRate']),
                           next_funding_ts=int(pi['nextFundingTime']))
            row['open_interest'] = _try(lambda: float(
                self.exchange.fapiPublicGetOpenInterest({'symbol': sym})['openInterest']))
            row['taker_ratio_5m'] = _try(lambda: float(
                self.exchange.fapiDataGetTakerlongshortRatio(
                    {'symbol': sym, 'period': '5m', 'limit': 1})[-1]['buySellRatio']))
            conn = self._get_conn()
            try:
                conn.execute(f"INSERT INTO ml_exec_snapshots ({', '.join(row)}) "
                             f"VALUES ({', '.join('?' * len(row))})", tuple(row.values()))
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"[PM] snapshot {event} {pair} no grabado: {e}")

    def _get_state(self, key: str, default: str = '') -> str:
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT value FROM ml_state WHERE key = ?", (key,)).fetchone()
            return row['value'] if row else default
        finally:
            conn.close()

    # =========================================================================
    # SYNC & BALANCE
    # =========================================================================
    def sync_positions(self):
        """Recupera posiciones desde DB y reconcilia con exchange."""
        # --- Paso 1: Leer posiciones guardadas en DB local ---
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT * FROM ml_positions").fetchall()
            for row in rows:
                r = dict(row)
                pos = Position(
                    pair=r['symbol'], side=r['side'], direction=r['direction'],
                    entry_price=r['entry_price'], quantity=r['quantity'],
                    notional=r['notional'], leverage=r['leverage'],
                    tp_price=r['tp_price'], sl_price=r['sl_price'],
                    tp_pct=r['tp_pct'], sl_pct=r['sl_pct'], atr_pct=r['atr_pct'],
                    trail_active=bool(r['trail_active']), trail_sl=r['trail_sl'],
                    peak_price=r['peak_price'], regime=r['regime'],
                    confidence=r['confidence'],
                    entry_time=datetime.fromisoformat(r['entry_time']),
                    bars=r['bars'], max_hold=r['max_hold'],
                    sl_order_id=r.get('sl_order_id'),
                    trail_mode=r.get('trail_mode', 'default') or 'default',
                    trail_fixed_dist=float(r.get('trail_fixed_dist', 0) or 0),
                    status=r.get('status') or 'open',
                )
                self.positions[pos.pair] = pos
                logger.info(f"[PM] Posicion recuperada (DB): {pos.pair} {pos.side} "
                            f"@ ${pos.entry_price:,.2f}")
        finally:
            conn.close()

        # --- Paso 2: Reconciliar con posiciones reales en exchange ---
        self._reconcile_with_exchange()

        # --- Paso 2.5: Colocar SL orders en exchange para todas las posiciones ---
        for pair, pos in self.positions.items():
            if pos.status == 'pending':
                continue  # sin red: se resuelve en update_positions
            # Cancel any stale SL order from previous session
            if pos.sl_order_id:
                self._cancel_exchange_sl(pair, pos.sl_order_id)
                pos.sl_order_id = None
            # Check current price before placing SL (avoid "would immediately trigger")
            effective_sl = pos.trail_sl if (pos.trail_active and pos.trail_sl) else pos.sl_price
            try:
                ticker = self.exchange.fetch_ticker(pair)
                current_price = ticker.get('last', 0)
                if current_price and effective_sl:
                    # SL would trigger immediately - let monitoring loop handle it
                    if pos.direction == 1 and effective_sl >= current_price:
                        logger.warning(f"[PM] SL {pair} ya superado ({effective_sl:.2f} >= {current_price:.2f}), "
                                       f"se cerrara en monitoring loop")
                        continue
                    if pos.direction == -1 and effective_sl <= current_price:
                        logger.warning(f"[PM] SL {pair} ya superado ({effective_sl:.2f} <= {current_price:.2f}), "
                                       f"se cerrara en monitoring loop")
                        continue
            except Exception:
                pass  # If price check fails, try placing SL anyway
            # Place fresh SL order
            sl_id = self._place_exchange_sl(pair, pos.side, pos.quantity, effective_sl)
            if sl_id:
                pos.sl_order_id = sl_id
                self._save_position(pos)

        # --- Paso 3: Recuperar state ---
        bal = self._get_state('balance')
        if bal:
            self.balance = float(bal)
        peak = self._get_state('peak_balance')
        if peak:
            self.peak_balance = float(peak)
        # El kill switch sobrevive al reinicio: antes vivia en memoria y un
        # crash (run_bot.sh relanza en 30 s) lo borraba. AUDITORIA_2026-09 §2.6
        self.killed = self._get_state('killed') == '1'

        # Restaurar daily_pnl desde DB (sobrevive reinicios)
        today_trades = self.get_today_trades_from_db()
        if today_trades:
            self.daily_pnl = sum(t['pnl'] for t in today_trades)
            self.daily_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            logger.info(f"[PM] Daily PnL restaurado: ${self.daily_pnl:+.2f} "
                        f"({len(today_trades)} trades hoy)")

        logger.info(f"[PM] {len(self.positions)} posiciones activas, "
                    f"balance=${self.balance:.2f}")

    def _reconcile_with_exchange(self):
        """Reconcilia posiciones entre DB local y exchange real.
        - Adopta posiciones en exchange que no estan en DB (migracion)
        - Elimina posiciones en DB que ya no existen en exchange (cierre manual)"""
        try:
            exchange_positions = self.exchange.fetch_positions()
        except Exception as e:
            logger.warning(f"[PM] Error obteniendo posiciones de exchange: {e}")
            return

        # Construir set de pares con posicion abierta en exchange
        exchange_pairs = set()
        for ep in exchange_positions:
            contracts = float(ep.get('contracts', 0) or 0)
            if contracts == 0:
                continue

            # Normalizar symbol: "SOL/USDT:USDT" -> "SOL/USDT"
            symbol = ep.get('symbol', '')
            pair = symbol.split(':')[0] if ':' in symbol else symbol
            exchange_pairs.add(pair)

            if pair in self.positions:
                pos = self.positions[pair]
                ex_entry = float(ep.get('entryPrice', 0) or 0)
                mismatch = (ex_entry > 0 and
                            abs(pos.entry_price - ex_entry) / ex_entry > 0.01)
                if pos.status == 'pending' or mismatch:
                    # Entry/qty del exchange (fuente de verdad); salida del motor.
                    logger.warning(
                        f"[PM] {pair}: adoptando desde exchange "
                        f"({'pendiente' if pos.status == 'pending' else 'entry mismatch'}) "
                        f"DB=${pos.entry_price:,.2f} vs Exchange=${ex_entry:,.2f}")
                    adopted = self._adopt(pos, ep)
                    self.positions[pair] = adopted
                    self._save_position(adopted)
                continue

            # Posicion en exchange SIN registro en DB -> adoptarla
            side_str = ep.get('side', 'long')  # "long" o "short"
            direction = 1 if side_str == 'long' else -1
            entry_price = float(ep.get('entryPrice', 0) or 0)
            if entry_price <= 0:
                logger.warning(f"[PM] {pair}: entry_price=0, ignorando posicion")
                continue
            leverage = int(ep.get('leverage', 3) or 3)
            notional = contracts * entry_price

            # Calcular TP/SL con valores por defecto (V13.01: per-pair)
            pair_tp, pair_sl = ML_TP_PCT, ML_SL_PCT
            if direction == 1:
                tp_price = entry_price * (1 + pair_tp)
                sl_price = entry_price * (1 - pair_sl)
            else:
                tp_price = entry_price * (1 - pair_tp)
                sl_price = entry_price * (1 + pair_sl)

            pos = Position(
                pair=pair, side=side_str, direction=direction,
                entry_price=entry_price, quantity=contracts,
                notional=notional, leverage=leverage,
                tp_price=tp_price, sl_price=sl_price,
                tp_pct=pair_tp, sl_pct=pair_sl,
                atr_pct=0.02,  # Default conservador
                regime='RANGE', confidence=0.0,
                peak_price=entry_price, max_hold=30,
            )

            self.positions[pair] = pos
            self._save_position(pos)
            logger.info(f"[PM] Posicion ADOPTADA de exchange: {pair} {side_str.upper()} "
                        f"@ ${entry_price:,.2f} | Qty={contracts} | Lev={leverage}x")

        # Posiciones en DB que ya no existen en exchange (cerradas por SL/manual/externo)
        stale = [p for p in self.positions if p not in exchange_pairs]
        for pair in stale:
            if self.positions[pair].status == 'pending':
                # La orden nunca entro: no hay trade que registrar.
                logger.warning(f"[PM] {pair}: posicion pendiente sin fill en "
                               f"exchange, descartada")
                del self.positions[pair]
                self._delete_position(pair)
            else:
                self._handle_stale_position(pair)

    def _set_stops(self, pos: Position, entry: float):
        """Fija entry y stops iniciales a partir de los parametros del motor."""
        d = pos.direction
        pos.entry_price = entry
        pos.peak_price = entry
        pos.tp_price = entry * (1 + d * pos.tp_pct)
        pos.sl_price = entry * (1 - d * pos.sl_pct)
        pos.trail_active, pos.trail_sl = False, None
        if pos.trail_mode == 'tight' and pos.trail_fixed_dist > 0:
            pos.trail_active = True
            pos.trail_sl = entry * (1 - d * pos.trail_fixed_dist)
            pos.tp_price = entry * (2.0 if d == 1 else 0.5)  # red lejana

    def _adopt(self, tpl: Position, ep: dict) -> Position:
        """Posicion real del exchange con los parametros de salida de `tpl`."""
        contracts = float(ep.get('contracts', 0) or 0)
        entry = float(ep.get('entryPrice', 0) or 0) or tpl.entry_price
        pos = replace(tpl, quantity=contracts, notional=contracts * entry,
                      leverage=int(ep.get('leverage') or tpl.leverage),
                      status='open')
        self._set_stops(pos, entry)
        return pos

    def _effective_sl(self, pos: Position) -> float:
        return pos.trail_sl if (pos.trail_active and pos.trail_sl) else pos.sl_price

    def _move_exchange_sl(self, pos: Position, new_sl: float):
        """Coloca el stop nuevo ANTES de cancelar el viejo. Si falla, el viejo
        sigue vigente y `sl_order_id` sigue apuntando a una orden viva; dos stops
        reduceOnly durante un instante es inocuo. AUDITORIA_2026-09 §2.5"""
        sl_id = self._place_exchange_sl(pos.pair, pos.side, pos.quantity, new_sl)
        if not sl_id:
            logger.warning(f"[PM] {pos.pair}: stop nuevo no colocado, se mantiene "
                           f"el anterior (id={pos.sl_order_id})")
            return
        self._cancel_exchange_sl(pos.pair, pos.sl_order_id)
        pos.sl_order_id = sl_id

    # =========================================================================
    # EXCHANGE SL ORDERS
    # =========================================================================
    def _place_exchange_sl(self, pair: str, pos_side: str, quantity: float,
                           stop_price: float) -> Optional[str]:
        """Place a STOP_MARKET order on exchange as safety net SL."""
        try:
            close_side = 'sell' if pos_side == 'long' else 'buy'
            stop_price = float(self.exchange.price_to_precision(pair, stop_price))
            order = self.exchange.create_order(
                symbol=pair,
                type='STOP_MARKET',
                side=close_side,
                amount=quantity,
                params={
                    'stopPrice': stop_price,
                    'reduceOnly': True,
                }
            )
            order_id = order.get('id')
            logger.info(f"[PM] SL order en exchange: {pair} @ ${stop_price:,.2f} "
                        f"(id={order_id})")
            return order_id
        except Exception as e:
            logger.warning(f"[PM] Error colocando SL en exchange {pair}: {e}")
            return None

    def _cancel_exchange_sl(self, pair: str, order_id: str):
        """Cancel existing SL order on exchange."""
        if not order_id:
            return
        try:
            self.exchange.cancel_order(order_id, pair)
            logger.debug(f"[PM] SL order cancelada: {pair} (id={order_id})")
        except Exception:
            pass  # Order might already be filled or cancelled

    def _enter(self, pair: str, side: str, qty: float, ref_price: float) -> tuple:
        """Entrada maker con fallback a market. Devuelve (precio medio, qty).

        1. Limit post-only (GTX) al mejor bid (compra) / ask (venta). Si cruzara,
           Binance la rechaza sin registrarla (-5022) y se va directo a market.
        2. Espera hasta ML_ENTRY_LIMIT_TIMEOUT_S; si no se lleno entera, se
           cancela y el resto va a market.
        Un fallo a mitad deja la posicion 'pending': update_positions/sync la
        adoptan con lo que haya en el exchange."""
        filled, cost, oid = 0.0, 0.0, None
        if ML_ENTRY_LIMIT_TIMEOUT_S > 0:
            try:
                ob = self.exchange.fetch_order_book(pair, 5)
                px = ob['bids'][0][0] if side == 'buy' else ob['asks'][0][0]
                o = self.exchange.create_order(pair, 'limit', side, qty, px,
                                               {'timeInForce': 'GTX'})
                oid = o.get('id')
                deadline = time.time() + ML_ENTRY_LIMIT_TIMEOUT_S
                while o.get('status') == 'open' and time.time() < deadline:
                    time.sleep(2)
                    o = self.exchange.fetch_order(oid, pair)
                if o.get('status') == 'open':
                    self._cancel_exchange_sl(pair, oid)       # cancela sin ruido
                    o = self.exchange.fetch_order(oid, pair)  # filled definitivo
                filled = float(o.get('filled') or 0)
                cost = filled * float(o.get('average') or px)
                logger.info(f"[PM] Entrada maker {pair}: {filled}/{qty} @ "
                            f"{o.get('average') or px}")
            except Exception as e:
                logger.info(f"[PM] Entrada maker {pair} no aplicada ({e}); a market")
                if oid:
                    # La limit existio y pudo llenarse en parte: sin saber cuanto,
                    # mandar la qty entera a market duplicaria la posicion. Si no
                    # se puede leer, se propaga y la 'pending' se adopta del exchange.
                    self._cancel_exchange_sl(pair, oid)
                    o = self.exchange.fetch_order(oid, pair)
                    filled = float(o.get('filled') or 0)
                    cost = filled * float(o.get('average') or ref_price)
        rest = qty - filled
        min_qty = ((getattr(self.exchange, 'markets', None) or {}).get(pair, {})
                   .get('limits', {}).get('amount', {}).get('min') or 0)
        maker_q = filled
        if rest > 1e-12 and rest >= min_qty:
            rest = float(self.exchange.amount_to_precision(pair, rest))
            o = self.exchange.create_order(pair, 'market', side, rest)
            mq = float(o.get('filled') or rest)
            cost += mq * self._order_fill_price(o, pair, ref_price)
            filled += mq
        if filled <= 0:
            raise RuntimeError(f"entrada {pair} sin fill")
        otype = 'maker' if maker_q >= filled else ('market' if maker_q <= 0 else 'mixed')
        return cost / filled, filled, otype

    def _order_fill_price(self, order: dict, pair: str, fallback: float) -> float:
        """Precio medio real de una orden market.
        demo-fapi devuelve la orden con average=None (la clave existe, asi que
        order.get('average', x) no cae al default y float(None) revienta DESPUES
        de ejecutar la orden). Eso dejaba posiciones abiertas sin SL y cierres
        registrados al precio del SL: 2 trades ganadores (+$437 reales) quedaron
        como perdidas en la DB (sep-2026)."""
        avg = order.get('average')
        if not avg and order.get('id'):
            try:
                avg = self.exchange.fetch_order(order['id'], pair).get('average')
            except Exception as e:
                logger.warning(f"[PM] No se pudo leer fill de {pair}: {e}")
        return float(avg or fallback)

    def _get_exchange_sl_fill(self, pair: str, pos) -> Optional[float]:
        """Check if position was closed by exchange SL. Returns fill price or None."""
        try:
            exchange_positions = self.exchange.fetch_positions([pair])
            for ep in exchange_positions:
                contracts = float(ep.get('contracts', 0) or 0)
                if contracts > 0:
                    symbol = ep.get('symbol', '')
                    ep_pair = symbol.split(':')[0] if ':' in symbol else symbol
                    if ep_pair == pair:
                        return None  # Position still open - real error
        except Exception:
            return None

        # Position gone from exchange - try to get SL fill price
        if pos.sl_order_id:
            try:
                order = self.exchange.fetch_order(pos.sl_order_id, pair)
                if order.get('status') == 'closed':
                    avg = float(order.get('average', 0) or 0)
                    if avg > 0:
                        return avg
            except Exception:
                pass

        # Fallback: use effective SL price
        return pos.trail_sl if (pos.trail_active and pos.trail_sl) else pos.sl_price

    def _handle_stale_position(self, pair: str):
        """Record and clean up a position closed externally (e.g., by exchange SL)."""
        pos = self.positions[pair]
        fill_price = None
        reason = 'EXTERNAL'

        # Try to get fill price from our SL order
        if pos.sl_order_id:
            try:
                order = self.exchange.fetch_order(pos.sl_order_id, pair)
                if order.get('status') == 'closed':
                    fill_price = float(order.get('average', 0) or 0)
                    if fill_price > 0:
                        reason = 'SL'
            except Exception:
                pass

        # Fallback: use effective SL price
        if not fill_price or fill_price <= 0:
            effective_sl = pos.trail_sl if (pos.trail_active and pos.trail_sl) else pos.sl_price
            fill_price = effective_sl
            reason = 'SL'
        # El libro ya es posterior al fill (se detecta en el siguiente tick);
        # lo que vale aqui es avg_price frente al stop.
        self._snapshot('exit_fill', pair, pos.entry_time, order_type='EXCHANGE_SL',
                       filled_qty=pos.quantity, avg_price=fill_price,
                       ref_price=self._effective_sl(pos))

        # Calculate PnL
        if pos.direction == 1:
            gross_pnl_pct = (fill_price - pos.entry_price) / pos.entry_price
        else:
            gross_pnl_pct = (pos.entry_price - fill_price) / pos.entry_price

        commission = pos.notional * (COMMISSION_RATE + SLIPPAGE_PCT) * 2
        pnl = pos.notional * gross_pnl_pct - commission

        # Record trade
        trade = {
            'symbol': pair,
            'entry_time': pos.entry_time.isoformat(),
            'exit_time': datetime.now(timezone.utc).isoformat(),
            'side': pos.side,
            'entry_price': pos.entry_price,
            'exit_price': fill_price,
            'quantity': pos.quantity,
            'notional': pos.notional,
            'leverage': pos.leverage,
            'pnl': pnl,
            'exit_reason': reason,
            'regime': pos.regime,
            'confidence': pos.confidence,
            'commission': commission,
            'trail_dist': pos.trail_fixed_dist if pos.trail_mode == 'tight' else None,
            'max_hold': pos.max_hold,
        }
        self._save_trade(trade)
        self.trade_log.append(trade)

        # Clean up
        del self.positions[pair]
        self._delete_position(pair)

        emoji = '+' if pnl > 0 else ''
        logger.info(f"[PM] CERRADO (exchange) {pair} {pos.side.upper()} | "
                    f"${pos.entry_price:,.2f} -> ${fill_price:,.2f} | "
                    f"PnL: ${pnl:{emoji}.2f} | Razon: {reason}")

    def refresh_balance(self) -> bool:
        """Actualiza balance desde exchange. Devuelve False si no hubo red
        (lo usa el watchdog de ml_bot para detectar un proceso ciego)."""
        try:
            bal = self.exchange.fetch_balance()
            # 'total' = free + margen en uso. Con 'free' el balance BAJA al abrir
            # una posicion (el margen pasa a 'used') sin que haya perdida: eso
            # encogia el sizing del siguiente trade y simulaba drawdown falso
            # en el kill switch. Con 'total' el equity es estable.
            # OJO: el capital en Simple Earn vive en otro wallet y NO entra aqui,
            # asi que con el yield activo esto sigue subestimando el equity real.
            u = bal.get('USDT', {})
            usdt = u.get('total') or u.get('free', 0)
            if usdt > 0:
                self.balance = float(usdt)
                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance
                self._save_state('balance', str(self.balance))
                self._save_state('peak_balance', str(self.peak_balance))
            return True
        except Exception as e:
            logger.warning(f"[PM] Error obteniendo balance: {e}")
            return False

    # =========================================================================
    # OPEN POSITION
    # =========================================================================
    def can_open(self, pair: str, direction: int) -> bool:
        """Verifica si se puede abrir una nueva posicion."""
        if self.killed or self.paused:
            return False
        if len(self.positions) >= ML_MAX_CONCURRENT:
            return False
        if pair in self.positions:
            return False
        # Filtro de correlacion: max 2 misma direccion
        same_dir = sum(1 for p in self.positions.values() if p.direction == direction)
        if same_dir >= 2:
            return False
        return True

    def open_position(self, pair: str, direction: int, confidence: float,
                      regime: str, price: float, atr_pct: float,
                      sizing_mult: float = 1.0,
                      tp_pct_override: float = None,
                      sl_pct_override: float = None,
                      trail_mode: str = 'default',
                      trail_fixed_dist: float = 0.0,
                      max_hold_override: int = None) -> bool:
        """Abre una nueva posicion. sizing_mult from V8.4 macro intelligence.
        tp_pct_override/sl_pct_override permiten valores personalizados (V14).
        trail_mode='tight' activates immediate trailing with fixed distance (ADA/SOL).
        max_hold_override: velas de hold maximo del MOTOR que genero la senal.
          Sin esto se usaba ML_MAX_HOLD (15 en RANGE, 2.5 dias) mientras el
          motor V2 asume 60 velas (10 dias) — y ese recorte cuesta: medido
          sobre 6,5 anos, 15/15 da PF 1.68 / +16.8% / DD 17.0% frente a
          60/40 con PF 1.83 / +19.1% / DD 14.7%. Ver experiments/max_bars/.
        """
        if not self.can_open(pair, direction):
            return False

        lev = ML_LEVERAGE.get(regime, 3)
        side = 'long' if direction == 1 else 'short'

        # Sizing: risk-based with V8.4 macro multiplier
        risk_pct = ML_RISK_PER_TRADE
        if confidence > 2.0:
            risk_pct = 0.03
        elif confidence > 1.5:
            risk_pct = 0.025
        risk_pct *= sizing_mult

        # Sizing sobre el balance REAL, no sobre INITIAL_CAPITAL. Estaba fijo en
        # $100 con la cuenta en $4.437: cada trade arriesgaba $2 (0,045% del
        # capital) y el motor no podia mover la cuenta hiciera lo que hiciera.
        risk_amt = self.balance * risk_pct
        # V14: override TP/SL si se especifican, sino usar por-par
        if tp_pct_override is not None and sl_pct_override is not None:
            pair_tp, pair_sl = tp_pct_override, sl_pct_override
        else:
            pair_tp, pair_sl = ML_TP_PCT, ML_SL_PCT
        notional = risk_amt / pair_sl if pair_sl > 0 else risk_amt
        # Tope relativo al balance, no absoluto: un tope en dolares se queda
        # obsoleto en cuanto cambia el capital (que es lo que paso con los $300).
        notional = min(notional, self.balance * ML_MAX_NOTIONAL_PCT)

        max_hold = (max_hold_override if max_hold_override
                    else ML_MAX_HOLD.get(regime, 15))

        # Cantidad en base currency
        quantity = notional / price

        # Plantilla con los parametros de salida del motor: se usa para el
        # registro 'pending', para adoptar un duplicado y tras el fill.
        pos = Position(
            pair=pair, side=side, direction=direction,
            entry_price=price, quantity=quantity, notional=notional,
            leverage=lev, tp_price=0.0, sl_price=0.0,
            tp_pct=pair_tp, sl_pct=pair_sl,
            atr_pct=atr_pct, regime=regime, confidence=confidence,
            max_hold=max_hold, trail_mode=trail_mode,
            trail_fixed_dist=trail_fixed_dist, status='pending',
        )
        self._set_stops(pos, price)

        try:
            # Safety check: verify no open position on exchange before placing order
            # Prevents duplicates if local state is out of sync
            try:
                exchange_positions = self.exchange.fetch_positions([pair])
                for ep in exchange_positions:
                    contracts = float(ep.get('contracts', 0) or 0)
                    if contracts > 0:
                        symbol = ep.get('symbol', '')
                        ep_pair = symbol.split(':')[0] if ':' in symbol else symbol
                        if ep_pair == pair:
                            logger.warning(
                                f"[PM] DUPLICADO EVITADO: {pair} ya tiene posicion "
                                f"abierta en exchange ({contracts} contracts) - "
                                f"adoptando en vez de abrir nueva"
                            )
                            adopted = self._adopt(pos, ep)
                            self.positions[pair] = adopted
                            self._save_position(adopted)
                            return False  # Don't send "trade opened" alert
            except Exception as e:
                logger.warning(f"[PM] Error en safety check de {pair}: {e} - continuando")

            # Set leverage
            symbol_ccxt = pair
            self.exchange.set_leverage(lev, symbol_ccxt)

            # Precision
            quantity = float(self.exchange.amount_to_precision(symbol_ccxt, quantity))
            if quantity <= 0:
                logger.warning(f"[PM] Cantidad invalida para {pair}: {quantity}")
                return False
            pos.quantity = quantity

            # Registro ANTES de la orden. Si algo revienta de aqui en adelante la
            # posicion queda 'pending' y update_positions/sync la resuelven contra
            # el exchange (adoptar con estos parametros o descartar).
            self.positions[pair] = pos
            self._save_position(pos)

            # Colocar orden
            order_side = 'buy' if direction == 1 else 'sell'
            self._snapshot('signal', pair, pos.entry_time, ref_price=price)
            t0 = time.time()
            fill_price, pos.quantity, otype = self._enter(symbol_ccxt, order_side,
                                                          quantity, price)
            self._snapshot('entry_fill', pair, pos.entry_time, order_type=otype,
                           filled_qty=pos.quantity, avg_price=fill_price,
                           ref_price=price, latency_ms=(time.time() - t0) * 1000)
            pos.notional = pos.quantity * fill_price
            self._set_stops(pos, fill_price)
            pos.status = 'open'
            self._save_position(pos)

            # Place SL order on exchange as safety net
            sl_id = self._place_exchange_sl(pair, side, pos.quantity,
                                            self._effective_sl(pos))
            if sl_id:
                pos.sl_order_id = sl_id
                self._save_position(pos)

            margin = pos.notional / lev
            logger.info(f"[PM] ABIERTO {pair} {side.upper()} @ ${fill_price:,.2f} | "
                        f"Qty={pos.quantity} | Notional=${pos.notional:.0f} | "
                        f"Margin=${margin:.1f} | Lev={lev}x | Conf={confidence:.2f}")
            return True

        except Exception as e:
            logger.error(f"[PM] Error abriendo {pair}: {e}")
            return False

    # =========================================================================
    # UPDATE / MONITOR POSITIONS
    # =========================================================================
    def update_positions(self) -> List[dict]:
        """Chequea todas las posiciones. Retorna trades cerrados."""
        if not self.positions:
            return []

        # Una orden quedo a medias (crash de red entre la orden y el fill):
        # resolverla contra el exchange antes de gestionar nada.
        if any(p.status == 'pending' for p in self.positions.values()):
            self._reconcile_with_exchange()

        closed_trades = []

        # Fetch precios actuales
        try:
            pairs = [p for p, pos in self.positions.items() if pos.status == 'open']
            tickers = {}
            for pair in pairs:
                try:
                    t = self.exchange.fetch_ticker(pair)
                    tickers[pair] = float(t['last'])
                except Exception as e:
                    logger.warning(f"[PM] Error precio {pair}: {e}")
        except Exception as e:
            logger.error(f"[PM] Error fetching tickers: {e}")
            return []

        to_close = []
        for pair, pos in self.positions.items():
            price = tickers.get(pair)
            if price is None:
                continue

            exit_price = None
            exit_reason = None

            # 1. Check TP (skip for tight mode — trail handles exits)
            if pos.trail_mode != 'tight':
                if pos.direction == 1 and price >= pos.tp_price:
                    exit_price, exit_reason = pos.tp_price, 'TP'
                elif pos.direction == -1 and price <= pos.tp_price:
                    exit_price, exit_reason = pos.tp_price, 'TP'

            # 2. Check Trailing Stop
            if exit_reason is None and pos.trail_active and pos.trail_sl is not None:
                if pos.direction == 1 and price <= pos.trail_sl:
                    exit_price, exit_reason = pos.trail_sl, 'TRAIL'
                elif pos.direction == -1 and price >= pos.trail_sl:
                    exit_price, exit_reason = pos.trail_sl, 'TRAIL'

            # 3. Check SL
            if exit_reason is None:
                if pos.direction == 1 and price <= pos.sl_price:
                    exit_price, exit_reason = pos.sl_price, 'SL'
                elif pos.direction == -1 and price >= pos.sl_price:
                    exit_price, exit_reason = pos.sl_price, 'SL'

            # 4. Timeout (basado en tiempo, no bars en live)
            if exit_reason is None:
                hours_open = (datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 3600
                max_hours = pos.max_hold * 4  # max_hold es en velas de 4h
                if hours_open >= max_hours:
                    exit_price, exit_reason = price, 'TIMEOUT'

            if exit_price and exit_reason:
                to_close.append((pair, exit_price, exit_reason))
            else:
                # Posicion sin stop en exchange (adopcion, fallo al colocarlo):
                # reintentar cada tick hasta que haya red de seguridad.
                if not pos.sl_order_id:
                    pos.sl_order_id = self._place_exchange_sl(
                        pair, pos.side, pos.quantity, self._effective_sl(pos))
                # V2 (tight): el trail solo se mueve con velas 4h cerradas, igual
                # que el backtest -> update_trail_on_closed_bars.
                if pos.trail_mode != 'tight':
                    old_eff_sl = self._effective_sl(pos)
                    self._update_trailing(pos, price)
                    new_eff_sl = self._effective_sl(pos)
                    if old_eff_sl and old_eff_sl > 0 and abs(new_eff_sl - old_eff_sl) / old_eff_sl > 0.001:
                        self._move_exchange_sl(pos, new_eff_sl)
                self._save_position(pos)

        # Cerrar posiciones
        for pair, exit_price, reason in to_close:
            trade = self._close_position(pair, exit_price, reason)
            if trade:
                closed_trades.append(trade)

        return closed_trades

    def update_trail_on_closed_bars(self):
        """Trail V2 (tight) por vela 4h CERRADA, como `_sim_long_trailing`.

        El backtest sube el peak con el high de cada vela ya cerrada; entre
        velas solo se chequea el stop. Antes el bot lo subia con el ticker cada
        30 s: stop mas ceñido que el simulado (re-simulado a 1h: PF 2,06 ->
        1,93, 13/82 trades cambian). AUDITORIA_2026-09 §2.1, plan 2.3 (a).

        Idempotente: recalcula desde la vela de entrada, asi que tras un
        reinicio o una vela perdida se pone al dia solo.
        """
        bar_ms = 4 * 3600 * 1000
        now_ms = time.time() * 1000
        for pair, pos in list(self.positions.items()):
            if pos.status != 'open' or pos.trail_mode != 'tight':
                continue
            since = int(pos.entry_time.timestamp() * 1000) // bar_ms * bar_ms
            try:
                bars = self.exchange.fetch_ohlcv(pair, '4h', since=since,
                                                 limit=pos.max_hold + 5)
            except Exception as e:
                logger.warning(f"[PM] {pair}: sin velas para el trail ({e})")
                continue
            closed = [b for b in bars if b[0] >= since and b[0] + bar_ms <= now_ms]
            if not closed:
                continue
            dist, old_sl = pos.trail_fixed_dist, pos.trail_sl
            if pos.direction == 1:
                pos.peak_price = max(pos.peak_price or pos.entry_price,
                                     max(b[2] for b in closed))
                pos.trail_sl = max(old_sl or 0.0, pos.peak_price * (1 - dist))
            else:
                pos.peak_price = min(pos.peak_price or pos.entry_price,
                                     min(b[3] for b in closed))
                pos.trail_sl = min(old_sl or float('inf'),
                                   pos.peak_price * (1 + dist))
            if pos.trail_sl != old_sl:
                logger.info(f"[PM] Trail {pair}: peak=${pos.peak_price:,.2f} "
                            f"stop=${pos.trail_sl:,.2f}")
                self._move_exchange_sl(pos, pos.trail_sl)
            self._save_position(pos)

    def _update_trailing(self, pos: Position, price: float):
        """Trailing por tick del modo 'default' (legacy, no lo usa V2)."""
        if pos.trail_active:
            # Actualizar peak y trail_sl
            if pos.direction == 1:
                if price > (pos.peak_price or 0):
                    pos.peak_price = price
                trail_dist = ML_TRAILING_LOCK * pos.atr_pct
                new_sl = pos.peak_price * (1 - trail_dist)
                if pos.trail_sl is None or new_sl > pos.trail_sl:
                    pos.trail_sl = new_sl
            else:
                if pos.peak_price is None or price < pos.peak_price:
                    pos.peak_price = price
                trail_dist = ML_TRAILING_LOCK * pos.atr_pct
                new_sl = pos.peak_price * (1 + trail_dist)
                if pos.trail_sl is None or new_sl < pos.trail_sl:
                    pos.trail_sl = new_sl
        else:
            # Verificar activacion
            if pos.direction == 1:
                profit_pct = (price - pos.entry_price) / pos.entry_price
            else:
                profit_pct = (pos.entry_price - price) / pos.entry_price

            if profit_pct >= pos.tp_pct * ML_TRAILING_ACTIVATION:
                pos.trail_active = True
                pos.peak_price = price
                # Lock 30% del profit actual
                if pos.direction == 1:
                    pos.trail_sl = pos.entry_price * (1 + profit_pct * 0.3)
                else:
                    pos.trail_sl = pos.entry_price * (1 - profit_pct * 0.3)
                logger.info(f"[PM] Trailing ACTIVADO {pos.pair}: "
                            f"profit={profit_pct:.2%}, trail_sl=${pos.trail_sl:,.2f}")

    def _close_position(self, pair: str, exit_price: float, reason: str) -> Optional[dict]:
        """Cierra una posicion y registra el trade."""
        pos = self.positions.get(pair)
        if pos is None:
            return None

        # Cancel exchange SL order first
        if pos.sl_order_id:
            self._cancel_exchange_sl(pair, pos.sl_order_id)

        try:
            # Orden de cierre
            close_side = 'sell' if pos.direction == 1 else 'buy'
            order = self.exchange.create_order(
                symbol=pair,
                type='market',
                side=close_side,
                amount=pos.quantity,
                params={'reduceOnly': True},
            )

            fill_price = self._order_fill_price(order, pair, exit_price)
        except Exception as e:
            # Check if exchange already closed it via SL
            fill_price = self._get_exchange_sl_fill(pair, pos)
            if fill_price is None:
                logger.error(f"[PM] Error cerrando {pair}: {e}")
                return None
            reason = 'SL'
        self._snapshot('exit_fill', pair, pos.entry_time, order_type=reason,
                       filled_qty=pos.quantity, avg_price=fill_price,
                       ref_price=exit_price)

        # Calcular PnL
        if pos.direction == 1:
            gross_pnl_pct = (fill_price - pos.entry_price) / pos.entry_price
        else:
            gross_pnl_pct = (pos.entry_price - fill_price) / pos.entry_price

        commission = pos.notional * (COMMISSION_RATE + SLIPPAGE_PCT) * 2
        pnl = pos.notional * gross_pnl_pct - commission

        # Update balance
        self.balance += pnl
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        self._save_state('balance', str(self.balance))
        self._save_state('peak_balance', str(self.peak_balance))

        # Daily PnL tracking
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if today != self.daily_date:
            self.daily_pnl = 0.0
            self.daily_date = today
        self.daily_pnl += pnl

        # Registrar trade
        trade = {
            'symbol': pair,
            'entry_time': pos.entry_time.isoformat(),
            'exit_time': datetime.now(timezone.utc).isoformat(),
            'side': pos.side,
            'entry_price': pos.entry_price,
            'exit_price': fill_price,
            'quantity': pos.quantity,
            'notional': pos.notional,
            'leverage': pos.leverage,
            'pnl': pnl,
            'exit_reason': reason,
            'regime': pos.regime,
            'confidence': pos.confidence,
            'commission': commission,
            'trail_dist': pos.trail_fixed_dist if pos.trail_mode == 'tight' else None,
            'max_hold': pos.max_hold,
        }
        self._save_trade(trade)

        # Limpiar
        del self.positions[pair]
        self._delete_position(pair)

        emoji = '+' if pnl > 0 else ''
        logger.info(f"[PM] CERRADO {pair} {pos.side.upper()} | "
                    f"${pos.entry_price:,.2f} -> ${fill_price:,.2f} | "
                    f"PnL: ${pnl:{emoji}.2f} | Razon: {reason} | "
                    f"Balance: ${self.balance:.2f}")

        # Sin pausa por racha de perdidas: no existe en el simulador y contaba
        # por etiqueta ('SL' incluia stops ganadores). Quedan el limite diario
        # y el kill switch. AUDITORIA_2026-09 §2.3, plan 1.1 opcion (a).

        self.trade_log.append(trade)
        return trade

    # =========================================================================
    # RISK CHECKS
    # =========================================================================
    def check_risk(self) -> bool:
        """Verifica DD y daily loss. Retorna True si OK, False si hay problema."""
        # Portfolio DD
        if self.peak_balance > 0:
            dd = (self.peak_balance - self.balance) / self.peak_balance
            if dd >= ML_MAX_DD_PCT:
                self.killed = True
                self._save_state('killed', '1')
                logger.critical(f"[PM] KILL SWITCH: DD {dd:.1%} >= {ML_MAX_DD_PCT:.0%} | "
                                f"Peak=${self.peak_balance:.2f} Balance=${self.balance:.2f}")
                return False

        # Daily loss. El reset de la pausa va AQUI, junto al cambio de dia: el
        # bloque que habia al final comparaba today != self.daily_date despues
        # de haber asignado daily_date = today, asi que era codigo muerto y una
        # pausa diaria no se levantaba nunca sin /resume manual.
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if today != self.daily_date:
            self.daily_pnl = 0.0
            self.daily_date = today
            if self.paused:
                self.paused = False
                logger.info("[PM] Nuevo dia: pausa por daily loss levantada")

        daily_limit = self.balance * ML_MAX_DAILY_LOSS_PCT  # antes: INITIAL_CAPITAL
        if self.daily_pnl < -daily_limit:
            if not self.paused:
                logger.warning(f"[PM] PAUSA: daily loss ${self.daily_pnl:.2f} >= "
                               f"{ML_MAX_DAILY_LOSS_PCT:.0%} de capital (${daily_limit:.2f})")
            self.paused = True
            return False

        return True

    def get_today_trades_from_db(self, strategy: str = None) -> list:
        """Obtiene trades de hoy desde la DB (sobrevive reinicios).
        Si strategy es None, devuelve solo V9 (no shadow).
        Si strategy='all', devuelve todos.
        """
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        conn = self._get_conn()
        try:
            if strategy == 'all':
                rows = conn.execute(
                    "SELECT * FROM ml_trades WHERE exit_time LIKE ?",
                    (f"{today}%",)
                ).fetchall()
            else:
                rows = conn.execute(
                    # Los trades reales se guardan como 'v85_prod'; filtrar por
                    # 'v9' dejaba el resumen diario siempre en 0 trades.
                    "SELECT * FROM ml_trades WHERE exit_time LIKE ? AND "
                    "(strategy IS NULL OR strategy NOT LIKE '%shadow%')",
                    (f"{today}%",)
                ).fetchall()
            out = [dict(r) for r in rows]
            for t in out:   # el PnL de Binance manda sobre el estimado
                if t.get('pnl_real') is not None:
                    t['pnl'] = t['pnl_real']
            return out
        finally:
            conn.close()

    # =========================================================================
    # VIVO vs SIMULADO / PnL REAL (plan 2.1 y 5.2)
    # =========================================================================
    def reconcile_closed_trades(self, limit: int = 20):
        """Completa los trades cerrados con el PnL real de Binance y la salida
        que habria dado el simulador (`_sim_*_trailing`) con la misma senal.
        Idempotente: solo toca columnas NULL. Llamar periodicamente."""
        now_ms = time.time() * 1000
        conn = self._get_conn()
        try:
            rows = [dict(r) for r in conn.execute(
                "SELECT * FROM ml_trades WHERE exit_time >= ? AND "
                "(pnl_real IS NULL OR (exit_sim_reason IS NULL AND trail_dist > 0)) "
                "ORDER BY id DESC LIMIT ?",
                (datetime.fromtimestamp(now_ms / 1000 - 90 * 86400,
                                        timezone.utc).isoformat(), limit)).fetchall()]
        finally:
            conn.close()
        for t in rows:
            upd = {}
            entry_ms = datetime.fromisoformat(t['entry_time']).timestamp() * 1000
            exit_ms = datetime.fromisoformat(t['exit_time']).timestamp() * 1000
            if t['pnl_real'] is None and now_ms - exit_ms > 5 * 60 * 1000:
                upd.update(self._income_between(t['symbol'], entry_ms - 60_000,
                                                exit_ms + 5 * 60_000))
            if t['exit_sim_reason'] is None and (t['trail_dist'] or 0) > 0:
                upd.update(self._sim_exit(t, entry_ms, now_ms))
            if upd:
                conn = self._get_conn()
                try:
                    sets = ', '.join(f"{k} = ?" for k in upd)
                    conn.execute(f"UPDATE ml_trades SET {sets} WHERE id = ?",
                                 (*upd.values(), t['id']))
                    conn.commit()
                finally:
                    conn.close()

    def _income_between(self, pair: str, start_ms: float, end_ms: float) -> dict:
        """REALIZED_PNL + COMMISSION + FUNDING_FEE de Binance en la ventana del
        trade. Con una sola posicion a la vez por par la ventana no se mezcla
        con otro trade. Sin REALIZED_PNL -> {} (se reintenta despues)."""
        try:
            rows = self.exchange.fapiPrivateGetIncome({
                'symbol': pair.replace('/', ''), 'startTime': int(start_ms),
                'endTime': int(end_ms), 'limit': 1000})
        except Exception as e:
            logger.warning(f"[PM] income {pair} no disponible: {e}")
            return {}
        tot = {'REALIZED_PNL': 0.0, 'COMMISSION': 0.0, 'FUNDING_FEE': 0.0}
        for r in rows:
            if r.get('incomeType') in tot:
                tot[r['incomeType']] += float(r['income'])
        if not any(r.get('incomeType') == 'REALIZED_PNL' for r in rows):
            return {}
        return {'pnl_real': sum(tot.values()),
                'commission_real': tot['COMMISSION'],
                'funding_real': tot['FUNDING_FEE']}

    def _sim_exit(self, t: dict, entry_ms: float, now_ms: float) -> dict:
        """Salida del simulador para esta senal: entrada al close de la vela de
        senal (la anterior a la de entrada), mismo trail_dist y max_hold."""
        signal_bar = int(entry_ms) // BAR_MS * BAR_MS - BAR_MS
        try:
            bars = self.exchange.fetch_ohlcv(t['symbol'], '4h', since=signal_bar,
                                             limit=int(t['max_hold']) + 3)
        except Exception as e:
            logger.warning(f"[PM] velas para sim de trade {t['id']}: {e}")
            return {}
        bars = [b for b in bars if b[0] >= signal_bar and b[0] + BAR_MS <= now_ms]
        if not bars or bars[0][0] != signal_bar:
            return {}
        df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        sim = _sim_long_trailing if t['side'] == 'long' else _sim_short_trailing
        reason, exit_p, pnl_pct, _ = sim(df, 0, float(df['close'].iloc[0]),
                                         float(t['trail_dist']), int(t['max_hold']),
                                         SIM_COMMISSION)
        if reason == 'NO_RESUELTO':
            return {'signal_close': float(df['close'].iloc[0])}
        return {'signal_close': float(df['close'].iloc[0]),
                'exit_sim_price': exit_p, 'exit_sim_reason': reason,
                'pnl_sim_pct': pnl_pct}

    def get_status(self) -> dict:
        """Retorna estado actual del portfolio."""
        dd = 0
        if self.peak_balance > 0:
            dd = (self.peak_balance - self.balance) / self.peak_balance

        unrealized = 0.0
        pos_details = []
        for pair, pos in self.positions.items():
            pos_details.append({
                'pair': pair,
                'side': pos.side,
                'entry': pos.entry_price,
                'trail': 'ON' if pos.trail_active else 'OFF',
                'confidence': pos.confidence,
            })

        return {
            'balance': self.balance,
            'peak': self.peak_balance,
            'dd': dd,
            'daily_pnl': self.daily_pnl,
            'positions': len(self.positions),
            'position_details': pos_details,
            'paused': self.paused,
            'killed': self.killed,
            'total_trades': len(self.trade_log),
        }
