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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, List
from pathlib import Path

from config.settings import (
    ML_DB_FILE, ML_MAX_CONCURRENT, ML_MAX_DD_PCT, ML_MAX_DAILY_LOSS_PCT,
    ML_RISK_PER_TRADE, ML_MAX_NOTIONAL, ML_LEVERAGE, ML_TP_PCT, ML_SL_PCT,
    ML_TRAILING_ACTIVATION, ML_TRAILING_LOCK, ML_MAX_HOLD,
    COMMISSION_RATE, SLIPPAGE_PCT, INITIAL_CAPITAL,
)

logger = logging.getLogger(__name__)


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
            """)
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
                     max_hold, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pos.pair, pos.entry_time.isoformat(), pos.side, pos.direction,
                pos.entry_price, pos.quantity, pos.notional, pos.leverage,
                pos.tp_price, pos.sl_price, pos.tp_pct, pos.sl_pct, pos.atr_pct,
                1 if pos.trail_active else 0, pos.trail_sl, pos.peak_price,
                pos.regime, pos.confidence, pos.bars, pos.max_hold,
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
                     confidence, commission)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade['symbol'], trade['entry_time'], trade['exit_time'],
                trade['side'], trade['entry_price'], trade['exit_price'],
                trade['quantity'], trade['notional'], trade['leverage'],
                trade['pnl'], trade['exit_reason'], trade['regime'],
                trade['confidence'], trade['commission'],
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
                )
                self.positions[pos.pair] = pos
                logger.info(f"[PM] Posicion recuperada (DB): {pos.pair} {pos.side} "
                            f"@ ${pos.entry_price:,.2f}")
        finally:
            conn.close()

        # --- Paso 2: Reconciliar con posiciones reales en exchange ---
        self._reconcile_with_exchange()

        # --- Paso 3: Recuperar state ---
        bal = self._get_state('balance')
        if bal:
            self.balance = float(bal)
        peak = self._get_state('peak_balance')
        if peak:
            self.peak_balance = float(peak)

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
                # Ya la tenemos en DB, todo bien
                continue

            # Posicion en exchange SIN registro en DB -> adoptarla
            side_str = ep.get('side', 'long')  # "long" o "short"
            direction = 1 if side_str == 'long' else -1
            entry_price = float(ep.get('entryPrice', 0) or 0)
            leverage = int(ep.get('leverage', 3) or 3)
            notional = contracts * entry_price

            # Calcular TP/SL con valores por defecto
            if direction == 1:
                tp_price = entry_price * (1 + ML_TP_PCT)
                sl_price = entry_price * (1 - ML_SL_PCT)
            else:
                tp_price = entry_price * (1 - ML_TP_PCT)
                sl_price = entry_price * (1 + ML_SL_PCT)

            pos = Position(
                pair=pair, side=side_str, direction=direction,
                entry_price=entry_price, quantity=contracts,
                notional=notional, leverage=leverage,
                tp_price=tp_price, sl_price=sl_price,
                tp_pct=ML_TP_PCT, sl_pct=ML_SL_PCT,
                atr_pct=0.02,  # Default conservador
                regime='RANGE', confidence=0.0,
                peak_price=entry_price, max_hold=30,
            )

            self.positions[pair] = pos
            self._save_position(pos)
            logger.info(f"[PM] Posicion ADOPTADA de exchange: {pair} {side_str.upper()} "
                        f"@ ${entry_price:,.2f} | Qty={contracts} | Lev={leverage}x")

        # Eliminar posiciones en DB que ya no existen en exchange (cierre manual/externo)
        stale = [p for p in self.positions if p not in exchange_pairs]
        for pair in stale:
            logger.info(f"[PM] Posicion {pair} cerrada externamente - eliminando de DB")
            del self.positions[pair]
            self._delete_position(pair)

    def refresh_balance(self):
        """Actualiza balance desde exchange."""
        try:
            bal = self.exchange.fetch_balance()
            usdt = bal.get('USDT', {}).get('free', 0)
            if usdt > 0:
                self.balance = float(usdt)
                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance
                self._save_state('balance', str(self.balance))
                self._save_state('peak_balance', str(self.peak_balance))
        except Exception as e:
            logger.warning(f"[PM] Error obteniendo balance: {e}")

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
                      regime: str, price: float, atr_pct: float) -> bool:
        """Abre una nueva posicion."""
        if not self.can_open(pair, direction):
            return False

        lev = ML_LEVERAGE.get(regime, 3)
        side = 'long' if direction == 1 else 'short'

        # Sizing: risk-based
        risk_pct = ML_RISK_PER_TRADE
        if confidence > 2.0:
            risk_pct = 0.03
        elif confidence > 1.5:
            risk_pct = 0.025

        risk_amt = INITIAL_CAPITAL * risk_pct  # FLAT sizing
        notional = risk_amt / ML_SL_PCT if ML_SL_PCT > 0 else risk_amt
        notional = min(notional, ML_MAX_NOTIONAL)

        # TP/SL precios
        if direction == 1:
            tp_price = price * (1 + ML_TP_PCT)
            sl_price = price * (1 - ML_SL_PCT)
        else:
            tp_price = price * (1 - ML_TP_PCT)
            sl_price = price * (1 + ML_SL_PCT)

        max_hold = ML_MAX_HOLD.get(regime, 15)

        # Cantidad en base currency
        quantity = notional / price

        try:
            # Set leverage
            symbol_ccxt = pair
            self.exchange.set_leverage(lev, symbol_ccxt)

            # Precision
            quantity = float(self.exchange.amount_to_precision(symbol_ccxt, quantity))
            if quantity <= 0:
                logger.warning(f"[PM] Cantidad invalida para {pair}: {quantity}")
                return False

            # Colocar orden
            order_side = 'buy' if direction == 1 else 'sell'
            order = self.exchange.create_order(
                symbol=symbol_ccxt,
                type='market',
                side=order_side,
                amount=quantity,
            )

            fill_price = float(order.get('average', price))
            filled_qty = float(order.get('filled', quantity))

            # Recalcular TP/SL con precio real de fill
            if direction == 1:
                tp_price = fill_price * (1 + ML_TP_PCT)
                sl_price = fill_price * (1 - ML_SL_PCT)
            else:
                tp_price = fill_price * (1 - ML_TP_PCT)
                sl_price = fill_price * (1 + ML_SL_PCT)

            actual_notional = filled_qty * fill_price

            pos = Position(
                pair=pair, side=side, direction=direction,
                entry_price=fill_price, quantity=filled_qty,
                notional=actual_notional, leverage=lev,
                tp_price=tp_price, sl_price=sl_price,
                tp_pct=ML_TP_PCT, sl_pct=ML_SL_PCT,
                atr_pct=atr_pct, regime=regime, confidence=confidence,
                peak_price=fill_price, max_hold=max_hold,
            )

            self.positions[pair] = pos
            self._save_position(pos)

            margin = actual_notional / lev
            logger.info(f"[PM] ABIERTO {pair} {side.upper()} @ ${fill_price:,.2f} | "
                        f"Qty={filled_qty} | Notional=${actual_notional:.0f} | "
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

        closed_trades = []

        # Fetch precios actuales
        try:
            pairs = list(self.positions.keys())
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

            # 1. Check TP
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
                # Update trailing stop
                self._update_trailing(pos, price)
                self._save_position(pos)

        # Cerrar posiciones
        for pair, exit_price, reason in to_close:
            trade = self._close_position(pair, exit_price, reason)
            if trade:
                closed_trades.append(trade)

        return closed_trades

    def _update_trailing(self, pos: Position, price: float):
        """Actualiza trailing stop para una posicion."""
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

            fill_price = float(order.get('average', exit_price))
        except Exception as e:
            logger.error(f"[PM] Error cerrando {pair}: {e}")
            # No borrar la posicion - reintentar en el proximo ciclo
            return None

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
                logger.critical(f"[PM] KILL SWITCH: DD {dd:.1%} >= {ML_MAX_DD_PCT:.0%} | "
                                f"Peak=${self.peak_balance:.2f} Balance=${self.balance:.2f}")
                return False

        # Daily loss
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if today != self.daily_date:
            self.daily_pnl = 0.0
            self.daily_date = today

        if self.daily_pnl < -(INITIAL_CAPITAL * ML_MAX_DAILY_LOSS_PCT):
            self.paused = True
            logger.warning(f"[PM] PAUSA: daily loss ${self.daily_pnl:.2f} >= "
                           f"{ML_MAX_DAILY_LOSS_PCT:.0%} de capital")
            return False

        # Reset pause al dia siguiente
        if self.paused and today != self.daily_date:
            self.paused = False
            self.daily_pnl = 0.0

        return True

    def get_today_trades_from_db(self) -> list:
        """Obtiene trades de hoy desde la DB (sobrevive reinicios)."""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM ml_trades WHERE exit_time LIKE ?",
                (f"{today}%",)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

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
