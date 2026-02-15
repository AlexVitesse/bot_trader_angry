"""
Shadow Portfolio Manager - Paper Trading para Comparacion de Estrategias
=========================================================================
Replica la logica de PortfolioManager (TP/SL/trailing/timeout) sin
interactuar con el exchange. Registra trades en SQLite con columna
strategy para comparacion A/B con la estrategia real.
"""

import sqlite3
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, List
from pathlib import Path

from config.settings import (
    ML_DB_FILE, ML_MAX_CONCURRENT, ML_RISK_PER_TRADE, ML_MAX_NOTIONAL,
    ML_LEVERAGE, ML_TP_PCT, ML_SL_PCT, ML_TRAILING_ACTIVATION,
    ML_TRAILING_LOCK, ML_MAX_HOLD, COMMISSION_RATE, SLIPPAGE_PCT,
    INITIAL_CAPITAL,
)

logger = logging.getLogger(__name__)


@dataclass
class ShadowPosition:
    pair: str
    side: str
    direction: int
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
    max_hold: int = 30


class ShadowPortfolioManager:
    """Paper trading portfolio manager for shadow strategy comparison."""

    def __init__(self, db_path: Path = ML_DB_FILE, strategy: str = 'v85_shadow'):
        self.db_path = db_path
        self.strategy = strategy
        self.positions: Dict[str, ShadowPosition] = {}
        self.balance = INITIAL_CAPITAL
        self.peak_balance = INITIAL_CAPITAL
        self.trade_log: List[dict] = []
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(str(self.db_path))

    def _init_db(self):
        conn = self._get_conn()
        try:
            # Ensure ml_trades table exists (may be first to init on fresh DB)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, entry_time TEXT, exit_time TEXT,
                    side TEXT, entry_price REAL, exit_price REAL,
                    quantity REAL, notional REAL, leverage INTEGER,
                    pnl REAL, exit_reason TEXT, regime TEXT,
                    confidence REAL, commission REAL,
                    strategy TEXT DEFAULT 'v9'
                )
            """)
            conn.commit()

            # Add strategy column to ml_trades if missing (legacy DBs)
            try:
                conn.execute("ALTER TABLE ml_trades ADD COLUMN strategy TEXT DEFAULT 'v9'")
                conn.commit()
            except Exception:
                pass  # Column already exists

            # Shadow positions table (separate from real positions)
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ml_shadow_positions (
                    symbol TEXT PRIMARY KEY,
                    strategy TEXT NOT NULL,
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
                    max_hold INTEGER DEFAULT 30,
                    updated_at TEXT
                );
            """)
            conn.commit()
        finally:
            conn.close()

    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================
    def can_open(self, pair: str, direction: int) -> bool:
        if len(self.positions) >= ML_MAX_CONCURRENT:
            return False
        if pair in self.positions:
            return False
        same_dir = sum(1 for p in self.positions.values() if p.direction == direction)
        if same_dir >= 2:
            return False
        return True

    def open_position(self, pair, direction, confidence, regime, price,
                      atr_pct, sizing_mult=1.0) -> bool:
        if not self.can_open(pair, direction):
            return False

        lev = ML_LEVERAGE.get(regime, 3)
        side = 'long' if direction == 1 else 'short'

        # Sizing identico al PortfolioManager real
        risk_pct = ML_RISK_PER_TRADE
        if confidence > 2.0:
            risk_pct = 0.03
        elif confidence > 1.5:
            risk_pct = 0.025
        risk_pct *= sizing_mult

        risk_amt = INITIAL_CAPITAL * risk_pct
        notional = risk_amt / ML_SL_PCT if ML_SL_PCT > 0 else risk_amt
        notional = min(notional, ML_MAX_NOTIONAL)
        quantity = notional / price

        # TP/SL
        if direction == 1:
            tp_price = price * (1 + ML_TP_PCT)
            sl_price = price * (1 - ML_SL_PCT)
        else:
            tp_price = price * (1 - ML_TP_PCT)
            sl_price = price * (1 + ML_SL_PCT)

        max_hold = ML_MAX_HOLD.get(regime, 15)

        pos = ShadowPosition(
            pair=pair, side=side, direction=direction,
            entry_price=price, quantity=quantity,
            notional=notional, leverage=lev,
            tp_price=tp_price, sl_price=sl_price,
            tp_pct=ML_TP_PCT, sl_pct=ML_SL_PCT,
            atr_pct=atr_pct, regime=regime,
            confidence=confidence, peak_price=price,
            max_hold=max_hold,
        )
        self.positions[pair] = pos
        self._save_shadow_position(pos)

        logger.info(f"[SHADOW] ABIERTO {pair} {side.upper()} @ ${price:,.2f} | "
                    f"Not=${notional:.0f} | Conf={confidence:.2f} | Reg={regime}")
        return True

    def update_positions(self, tickers: dict) -> List[dict]:
        """Check shadow positions against current prices. Returns closed trades."""
        if not self.positions:
            return []

        closed_trades = []
        to_close = []

        for pair, pos in self.positions.items():
            price = tickers.get(pair)
            if price is None:
                continue

            exit_price = None
            exit_reason = None

            # 1. TP check
            if pos.direction == 1 and price >= pos.tp_price:
                exit_price, exit_reason = pos.tp_price, 'TP'
            elif pos.direction == -1 and price <= pos.tp_price:
                exit_price, exit_reason = pos.tp_price, 'TP'

            # 2. Trailing stop check
            if exit_reason is None and pos.trail_active and pos.trail_sl is not None:
                if pos.direction == 1 and price <= pos.trail_sl:
                    exit_price, exit_reason = pos.trail_sl, 'TRAIL'
                elif pos.direction == -1 and price >= pos.trail_sl:
                    exit_price, exit_reason = pos.trail_sl, 'TRAIL'

            # 3. SL check
            if exit_reason is None:
                if pos.direction == 1 and price <= pos.sl_price:
                    exit_price, exit_reason = pos.sl_price, 'SL'
                elif pos.direction == -1 and price >= pos.sl_price:
                    exit_price, exit_reason = pos.sl_price, 'SL'

            # 4. Timeout (based on time, not bars)
            if exit_reason is None:
                hours_open = (datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 3600
                max_hours = pos.max_hold * 4
                if hours_open >= max_hours:
                    exit_price, exit_reason = price, 'TIMEOUT'

            if exit_price and exit_reason:
                to_close.append((pair, exit_price, exit_reason))
            else:
                self._update_trailing(pos, price)
                self._save_shadow_position(pos)

        for pair, exit_price, reason in to_close:
            trade = self._close_position(pair, exit_price, reason)
            if trade:
                closed_trades.append(trade)

        return closed_trades

    def _update_trailing(self, pos: ShadowPosition, price: float):
        """Actualiza trailing stop - logica identica a PortfolioManager."""
        if pos.trail_active:
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
            if pos.direction == 1:
                profit_pct = (price - pos.entry_price) / pos.entry_price
            else:
                profit_pct = (pos.entry_price - price) / pos.entry_price

            if profit_pct >= pos.tp_pct * ML_TRAILING_ACTIVATION:
                pos.trail_active = True
                pos.peak_price = price
                if pos.direction == 1:
                    pos.trail_sl = pos.entry_price * (1 + profit_pct * 0.3)
                else:
                    pos.trail_sl = pos.entry_price * (1 - profit_pct * 0.3)
                logger.info(f"[SHADOW] Trailing ACTIVADO {pos.pair}: "
                            f"profit={profit_pct:.2%}, trail_sl=${pos.trail_sl:,.2f}")

    def _close_position(self, pair, exit_price, reason):
        pos = self.positions.get(pair)
        if pos is None:
            return None

        # PnL identico a PortfolioManager
        if pos.direction == 1:
            gross_pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            gross_pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        commission = pos.notional * (COMMISSION_RATE + SLIPPAGE_PCT) * 2
        pnl = pos.notional * gross_pnl_pct - commission

        self.balance += pnl
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        trade = {
            'symbol': pair,
            'entry_time': pos.entry_time.isoformat(),
            'exit_time': datetime.now(timezone.utc).isoformat(),
            'side': pos.side,
            'entry_price': pos.entry_price,
            'exit_price': exit_price,
            'quantity': pos.quantity,
            'notional': pos.notional,
            'leverage': pos.leverage,
            'pnl': pnl,
            'exit_reason': reason,
            'regime': pos.regime,
            'confidence': pos.confidence,
            'commission': commission,
            'strategy': self.strategy,
        }
        self._save_trade(trade)

        del self.positions[pair]
        self._delete_shadow_position(pair)

        sign = '+' if pnl > 0 else ''
        logger.info(f"[SHADOW] CERRADO {pair} {pos.side.upper()} | "
                    f"${pos.entry_price:,.2f} -> ${exit_price:,.2f} | "
                    f"PnL: ${pnl:{sign}.2f} | Razon: {reason}")

        self.trade_log.append(trade)
        return trade

    # =========================================================================
    # DATABASE
    # =========================================================================
    def _save_trade(self, trade: dict):
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT INTO ml_trades
                    (symbol, entry_time, exit_time, side, entry_price, exit_price,
                     quantity, notional, leverage, pnl, exit_reason, regime,
                     confidence, commission, strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade['symbol'], trade['entry_time'], trade['exit_time'],
                trade['side'], trade['entry_price'], trade['exit_price'],
                trade['quantity'], trade['notional'], trade['leverage'],
                trade['pnl'], trade['exit_reason'], trade['regime'],
                trade['confidence'], trade['commission'], trade['strategy'],
            ))
            conn.commit()
        finally:
            conn.close()

    def _save_shadow_position(self, pos: ShadowPosition):
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO ml_shadow_positions
                    (symbol, strategy, entry_time, side, direction, entry_price,
                     quantity, notional, leverage, tp_price, sl_price, tp_pct,
                     sl_pct, atr_pct, trail_active, trail_sl, peak_price,
                     regime, confidence, max_hold, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pos.pair, self.strategy, pos.entry_time.isoformat(),
                pos.side, pos.direction, pos.entry_price,
                pos.quantity, pos.notional, pos.leverage,
                pos.tp_price, pos.sl_price, pos.tp_pct, pos.sl_pct,
                pos.atr_pct, int(pos.trail_active),
                pos.trail_sl, pos.peak_price,
                pos.regime, pos.confidence, pos.max_hold,
                datetime.now(timezone.utc).isoformat(),
            ))
            conn.commit()
        finally:
            conn.close()

    def _delete_shadow_position(self, pair: str):
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM ml_shadow_positions WHERE symbol = ?", (pair,))
            conn.commit()
        finally:
            conn.close()

    def sync_positions(self):
        """Load shadow positions from DB (for restart recovery)."""
        conn = self._get_conn()
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM ml_shadow_positions WHERE strategy = ?",
                (self.strategy,)
            ).fetchall()

            for row in rows:
                r = dict(row)
                pos = ShadowPosition(
                    pair=r['symbol'], side=r['side'],
                    direction=r['direction'], entry_price=r['entry_price'],
                    quantity=r['quantity'], notional=r['notional'],
                    leverage=r['leverage'], tp_price=r['tp_price'],
                    sl_price=r['sl_price'], tp_pct=r['tp_pct'],
                    sl_pct=r['sl_pct'], atr_pct=r['atr_pct'],
                    trail_active=bool(r['trail_active']),
                    trail_sl=r['trail_sl'], peak_price=r['peak_price'],
                    regime=r['regime'], confidence=r['confidence'],
                    max_hold=r['max_hold'],
                )
                try:
                    pos.entry_time = datetime.fromisoformat(r['entry_time'])
                except Exception:
                    pos.entry_time = datetime.now(timezone.utc)
                self.positions[r['symbol']] = pos

            if self.positions:
                logger.info(f"[SHADOW] Recuperadas {len(self.positions)} posiciones shadow")
        except Exception as e:
            logger.warning(f"[SHADOW] Error cargando posiciones: {e}")
        finally:
            conn.close()

    def get_today_trades_from_db(self) -> list:
        """Obtiene trades shadow de hoy desde la DB (sobrevive reinicios)."""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM ml_trades WHERE exit_time LIKE ? AND strategy = ?",
                (f"{today}%", self.strategy)
            ).fetchall()
            return [dict(zip([d[0] for d in conn.execute("SELECT * FROM ml_trades LIMIT 0").description], r)) for r in rows]
        except Exception:
            return []
        finally:
            conn.close()

    def get_summary(self) -> dict:
        """Returns summary stats for heartbeat/status (reads from DB, survives restarts)."""
        today_trades = self.get_today_trades_from_db()
        total_pnl = sum(t.get('pnl', 0) for t in today_trades)
        n_trades = len(today_trades)
        n_wins = sum(1 for t in today_trades if t.get('pnl', 0) > 0)
        wr = n_wins / n_trades * 100 if n_trades > 0 else 0
        return {
            'strategy': self.strategy,
            'n_open': len(self.positions),
            'n_trades': n_trades,
            'total_pnl': total_pnl,
            'win_rate': wr,
            'balance': self.balance,
        }
