"""
Database - Persistencia SQLite
===============================
Almacena trades, features, candles y posicion activa.
Pre-requisito para ML (Fase 4A).
"""

import sys
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DB_FILE

logger = logging.getLogger(__name__)


class Database:
    """Gestor de base de datos SQLite para el bot."""

    def __init__(self, db_path: Path = DB_FILE):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """Crea las tablas si no existen."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    avg_price REAL NOT NULL,
                    total_quantity REAL NOT NULL,
                    safety_orders_count INTEGER DEFAULT 0,
                    pnl REAL DEFAULT 0.0,
                    pnl_pct REAL DEFAULT 0.0,
                    exit_reason TEXT DEFAULT '',
                    commission REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id INTEGER,
                    timestamp TEXT NOT NULL,
                    price REAL NOT NULL,
                    ema_200 REAL,
                    ema_dist_pct REAL,
                    bb_lower REAL,
                    bb_upper REAL,
                    bb_position REAL,
                    stoch_k REAL,
                    atr REAL,
                    atr_sma REAL,
                    atr_ratio REAL,
                    volume REAL,
                    volume_sma_20 REAL,
                    volume_relative REAL,
                    spread REAL,
                    spread_relative REAL,
                    hour_utc INTEGER,
                    day_of_week INTEGER,
                    return_5 REAL,
                    return_15 REAL,
                    return_60 REAL,
                    consecutive_wins INTEGER DEFAULT 0,
                    consecutive_losses INTEGER DEFAULT 0,
                    funding_rate REAL,
                    taker_buy_ratio REAL,
                    long_short_ratio REAL,
                    open_interest REAL,
                    outcome TEXT DEFAULT '',
                    FOREIGN KEY (trade_id) REFERENCES trades(id)
                );

                CREATE TABLE IF NOT EXISTS candles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL UNIQUE,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    taker_buy_volume REAL,
                    quote_volume REAL
                );

                CREATE TABLE IF NOT EXISTS active_position (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    entry_time TEXT NOT NULL,
                    side TEXT NOT NULL,
                    avg_price REAL NOT NULL,
                    total_quantity REAL NOT NULL,
                    safety_orders_count INTEGER DEFAULT 0,
                    commission REAL DEFAULT 0.0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
                CREATE INDEX IF NOT EXISTS idx_features_trade_id ON features(trade_id);
                CREATE INDEX IF NOT EXISTS idx_candles_timestamp ON candles(timestamp);
            """)
            conn.commit()

            # Migraciones: agregar columnas nuevas a tablas existentes
            self._migrate(conn)

            logger.info(f"[DB] Base de datos inicializada: {self.db_path}")
        finally:
            conn.close()

    def _migrate(self, conn: sqlite3.Connection):
        """Agrega columnas nuevas si no existen (safe para DB existentes)."""
        migrations = [
            ("candles", "taker_buy_volume", "REAL"),
            ("candles", "quote_volume", "REAL"),
            ("features", "funding_rate", "REAL"),
            ("features", "taker_buy_ratio", "REAL"),
            ("features", "long_short_ratio", "REAL"),
            ("features", "open_interest", "REAL"),
        ]
        for table, column, col_type in migrations:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                conn.commit()
                logger.info(f"[DB] Migración: {table}.{column} agregada")
            except sqlite3.OperationalError:
                pass  # Columna ya existe

    # =========================================================================
    # TRADES
    # =========================================================================
    def save_trade(self, trade_data: Dict) -> int:
        """Guarda un trade completado. Retorna el ID del trade."""
        conn = self._get_conn()
        try:
            cursor = conn.execute("""
                INSERT INTO trades (entry_time, exit_time, side, entry_price, exit_price,
                    avg_price, total_quantity, safety_orders_count, pnl, pnl_pct,
                    exit_reason, commission)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data['entry_time'], trade_data.get('exit_time'),
                trade_data['side'], trade_data['entry_price'],
                trade_data.get('exit_price'), trade_data['avg_price'],
                trade_data['total_quantity'], trade_data.get('safety_orders_count', 0),
                trade_data.get('pnl', 0.0), trade_data.get('pnl_pct', 0.0),
                trade_data.get('exit_reason', ''), trade_data.get('commission', 0.0)
            ))
            conn.commit()
            trade_id = cursor.lastrowid
            logger.info(f"[DB] Trade #{trade_id} guardado: {trade_data['side']} PnL=${trade_data.get('pnl', 0):.2f}")
            return trade_id
        finally:
            conn.close()

    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        """Obtiene los ultimos N trades."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in reversed(rows)]
        finally:
            conn.close()

    def get_trade_count(self) -> int:
        """Retorna el total de trades en la DB."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM trades").fetchone()
            return row['cnt']
        finally:
            conn.close()

    # =========================================================================
    # FEATURES (para ML)
    # =========================================================================
    def save_features(self, features: Dict, trade_id: Optional[int] = None) -> int:
        """Guarda un snapshot de features al momento de la entrada."""
        conn = self._get_conn()
        try:
            cursor = conn.execute("""
                INSERT INTO features (trade_id, timestamp, price, ema_200, ema_dist_pct,
                    bb_lower, bb_upper, bb_position, stoch_k, atr, atr_sma, atr_ratio,
                    volume, volume_sma_20, volume_relative, spread, spread_relative,
                    hour_utc, day_of_week, return_5, return_15, return_60,
                    consecutive_wins, consecutive_losses, funding_rate, taker_buy_ratio,
                    long_short_ratio, open_interest, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id, features['timestamp'], features['price'],
                features.get('ema_200'), features.get('ema_dist_pct'),
                features.get('bb_lower'), features.get('bb_upper'),
                features.get('bb_position'), features.get('stoch_k'),
                features.get('atr'), features.get('atr_sma'),
                features.get('atr_ratio'), features.get('volume'),
                features.get('volume_sma_20'), features.get('volume_relative'),
                features.get('spread'), features.get('spread_relative'),
                features.get('hour_utc'), features.get('day_of_week'),
                features.get('return_5'), features.get('return_15'),
                features.get('return_60'), features.get('consecutive_wins', 0),
                features.get('consecutive_losses', 0),
                features.get('funding_rate'), features.get('taker_buy_ratio'),
                features.get('long_short_ratio'), features.get('open_interest'),
                features.get('outcome', '')
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def update_feature_outcome(self, trade_id: int, outcome: str):
        """Actualiza el outcome (WIN/LOSS) de un feature record tras cerrar trade."""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE features SET outcome = ? WHERE trade_id = ?",
                (outcome, trade_id)
            )
            conn.commit()
        finally:
            conn.close()

    def get_training_data(self) -> List[Dict]:
        """Obtiene features + outcome para entrenamiento ML."""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT f.*, t.pnl, t.safety_orders_count as trade_dca_count
                FROM features f
                JOIN trades t ON f.trade_id = t.id
                WHERE f.outcome != ''
                ORDER BY f.id
            """).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # =========================================================================
    # CANDLES
    # =========================================================================
    def save_candle(self, candle: Dict):
        """Guarda una vela 1m. Ignora duplicados."""
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR IGNORE INTO candles (timestamp, open, high, low, close, volume,
                    taker_buy_volume, quote_volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                candle['timestamp'], candle['open'], candle['high'],
                candle['low'], candle['close'], candle['volume'],
                candle.get('taker_buy_volume'), candle.get('quote_volume')
            ))
            conn.commit()
        finally:
            conn.close()

    def get_candle_count(self) -> int:
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM candles").fetchone()
            return row['cnt']
        finally:
            conn.close()

    # =========================================================================
    # ACTIVE POSITION (para recovery tras reinicio)
    # =========================================================================
    def save_active_position(self, position: Dict):
        """Guarda o actualiza la posicion activa (solo 1 fila, id=1)."""
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO active_position
                    (id, entry_time, side, avg_price, total_quantity,
                     safety_orders_count, commission, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position['entry_time'], position['side'],
                position['avg_price'], position['total_quantity'],
                position.get('safety_orders_count', 0),
                position.get('commission', 0.0),
                datetime.now().isoformat()
            ))
            conn.commit()
            logger.info(f"[DB] Posicion activa guardada: {position['side']} @ ${position['avg_price']:.2f}")
        finally:
            conn.close()

    def get_active_position(self) -> Optional[Dict]:
        """Recupera la posicion activa (si existe)."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM active_position WHERE id = 1").fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()

    def clear_active_position(self):
        """Elimina la posicion activa (cuando se cierra el trade)."""
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM active_position WHERE id = 1")
            conn.commit()
            logger.info("[DB] Posicion activa limpiada")
        finally:
            conn.close()

    # =========================================================================
    # STATS
    # =========================================================================
    def get_daily_stats(self, date: Optional[str] = None) -> Dict:
        """Obtiene estadisticas del dia."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM trades WHERE date(entry_time) = ?", (date,)
            ).fetchall()
            trades = [dict(r) for r in rows]
            wins = [t for t in trades if t['pnl'] > 0]
            losses = [t for t in trades if t['pnl'] <= 0]
            total_pnl = sum(t['pnl'] for t in trades)
            return {
                'date': date,
                'total_trades': len(trades),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': len(wins) / len(trades) * 100 if trades else 0,
                'total_pnl': total_pnl
            }
        finally:
            conn.close()

    def get_all_stats(self) -> Dict:
        """Obtiene estadisticas globales."""
        conn = self._get_conn()
        try:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                    SUM(pnl) as total_pnl,
                    AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                    AVG(CASE WHEN pnl <= 0 THEN pnl END) as avg_loss,
                    MIN(pnl) as worst_trade,
                    MAX(pnl) as best_trade
                FROM trades
            """).fetchone()
            stats = dict(row)
            total = stats['total_trades']
            stats['win_rate'] = (stats['wins'] / total * 100) if total > 0 else 0
            return stats
        finally:
            conn.close()


# Instancia global
db = Database()
