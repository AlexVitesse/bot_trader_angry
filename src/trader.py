"""
Trader - Motor de Ejecucion
============================
Gestiona la ejecucion de trades con soporte para DCA (Grinder).
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, field

# Agregar raiz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exchange import client
from src.strategy import Signal
from src.database import db
from config.settings import (
    SYMBOL,
    LEVERAGE,
    INITIAL_CAPITAL,
    MAX_POSITION_SIZE_PCT,
    COMMISSION_RATE,
    POSITION_SIZE_PCT,
    MIN_ORDER_MARGIN,
    MAX_DAILY_LOSS_PCT,
    MAX_CONSECUTIVE_LOSSES,
    MAX_TRADES_PER_DAY,
    ROLLING_WR_WINDOW,
    ROLLING_WR_MIN,
    ROLLING_WR_RESUME
)

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Registro de un trade ejecutado."""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    side: str = ""
    avg_price: float = 0.0
    total_quantity: float = 0.0
    safety_orders_count: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    commission: float = 0.0


@dataclass
class DailyStats:
    """Estadisticas diarias."""
    date: str = ""
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0


class Trader:
    """Motor de ejecucion de trades con soporte para DCA (Grinder)."""

    def __init__(self):
        self.symbol = SYMBOL.replace('/', '')
        self.leverage = LEVERAGE
        self.initial_capital = INITIAL_CAPITAL

        # Estado
        self.current_trade: Optional[TradeRecord] = None
        self.trade_history: List[TradeRecord] = []
        self.daily_stats = DailyStats(date=datetime.now().strftime('%Y-%m-%d'))

        # Importar settings de Grinder
        from config.settings import (
            BASE_ORDER_MARGIN, DCA_STEP_PCT, MAX_SAFETY_ORDERS, 
            MARTINGALE_MULTIPLIER, TAKE_PROFIT_PCT, STOP_LOSS_CATASTROPHIC
        )
        self.grinder_settings = {
            'base_margin': BASE_ORDER_MARGIN,
            'dca_step': DCA_STEP_PCT,
            'max_so': MAX_SAFETY_ORDERS,
            'martingale': MARTINGALE_MULTIPLIER,
            'tp_pct': TAKE_PROFIT_PCT,
            'sl_pct': STOP_LOSS_CATASTROPHIC
        }

        # Kill switch
        self.is_paused = False
        self.pause_reason = ""

        # Cargar historial de trades desde SQLite
        self._load_trade_history()

        # Inicializar
        self._setup()

    def _load_trade_history(self):
        """Carga historial reciente de trades desde SQLite."""
        try:
            recent = db.get_recent_trades(limit=ROLLING_WR_WINDOW)
            for t in recent:
                record = TradeRecord(
                    entry_time=datetime.fromisoformat(t['entry_time']),
                    exit_time=datetime.fromisoformat(t['exit_time']) if t.get('exit_time') else None,
                    side=t['side'],
                    avg_price=t['avg_price'],
                    total_quantity=t['total_quantity'],
                    safety_orders_count=t.get('safety_orders_count', 0),
                    pnl=t.get('pnl', 0.0),
                    pnl_pct=t.get('pnl_pct', 0.0),
                    exit_reason=t.get('exit_reason', ''),
                    commission=t.get('commission', 0.0)
                )
                self.trade_history.append(record)
            total = db.get_trade_count()
            if total > 0:
                logger.info(f"[TRADER] Historial cargado desde DB: {len(self.trade_history)} recientes / {total} total")
        except Exception as e:
            logger.warning(f"[WARN] No se pudo cargar historial de DB: {e}")

    def _extract_fill_price(self, order: Dict, fallback_price: float) -> float:
        """Extrae el precio de llenado de la respuesta de Binance.

        Cadena de intentos:
        1. avgPrice del response (normal en live)
        2. Calcular desde array 'fills'
        3. Consultar posicion al exchange (entryPrice)
        4. Fallback al precio de la senal (ultimo recurso)
        """
        # 1. avgPrice directo
        avg = float(order.get('avgPrice', 0))
        if avg > 0:
            return avg

        # 2. Calcular desde fills array
        fills = order.get('fills', [])
        if fills:
            total_qty = sum(float(f['qty']) for f in fills)
            total_cost = sum(float(f['qty']) * float(f['price']) for f in fills)
            if total_qty > 0:
                return total_cost / total_qty

        # 3. Consultar posicion al exchange (tiene el entryPrice real)
        try:
            position = client.get_position(self.symbol)
            if position and position['entry_price'] > 0:
                logger.info(f"[TRADER] Fill price obtenido via consulta de posicion: ${position['entry_price']:,.2f}")
                return position['entry_price']
        except Exception:
            pass

        # 4. Fallback (no ideal, pero evita crash)
        logger.warning(f"[WARN] No se pudo obtener fill price real, usando fallback: ${fallback_price:,.2f}")
        return fallback_price

    def _setup(self):
        """Configura el trader."""
        try:
            # Configurar leverage
            client.set_leverage(self.leverage, self.symbol)
            logger.info(f"[TRADER] Leverage configurado: {self.leverage}x")
        except Exception as e:
            logger.warning(f"[WARN] No se pudo configurar leverage: {e}")

    def _get_dynamic_margin(self) -> float:
        """Calcula margen dinamico: POSITION_SIZE_PCT del balance actual.

        Retorna el margen en USDT, con minimo MIN_ORDER_MARGIN.
        Si no puede obtener balance, usa BASE_ORDER_MARGIN como fallback.
        """
        try:
            balance = client.get_usdt_balance()
            if balance > 0:
                margin = balance * POSITION_SIZE_PCT
                margin = max(margin, MIN_ORDER_MARGIN)
                return margin
        except Exception as e:
            logger.warning(f"[WARN] No se pudo obtener balance para sizing dinamico: {e}")
        return self.grinder_settings['base_margin']

    def calculate_base_quantity(self, entry_price: float) -> float:
        """Calcula cantidad inicial basada en % del balance (sizing dinamico)."""
        if entry_price <= 0:
            logger.error(f"[ERROR] entry_price invalido: {entry_price}")
            return 0.0
        margin = self._get_dynamic_margin()
        notional = margin * self.leverage
        quantity = notional / entry_price
        logger.info(f"[SIZING] Margen: ${margin:.2f} ({POSITION_SIZE_PCT*100}% del balance) | Notional: ${notional:.2f}")
        return round(quantity, 3)

    def open_position(self, signal: Signal, current_price: float, atr: float) -> bool:
        """Abre la posicion inicial (Base Order)."""
        if self.is_paused or self.current_trade:
            return False

        quantity = self.calculate_base_quantity(current_price)
        if quantity < 0.001: return False

        try:
            side = 'long' if signal == Signal.LONG else 'short'
            order = client.market_buy(quantity, self.symbol) if signal == Signal.LONG else client.market_sell(quantity, self.symbol)
            fill_price = self._extract_fill_price(order, current_price)

            self.current_trade = TradeRecord(
                entry_time=datetime.now(),
                side=side,
                avg_price=fill_price,
                total_quantity=quantity,
                safety_orders_count=0,
                commission=quantity * fill_price * COMMISSION_RATE
            )

            logger.info(f"[GRINDER] OPEN {side.upper()} | Precio: ${fill_price:,.2f} | Qty: {quantity}")
            self.daily_stats.trades += 1

            # Persistir posicion activa en DB
            try:
                db.save_active_position({
                    'entry_time': self.current_trade.entry_time.isoformat(),
                    'side': side,
                    'avg_price': fill_price,
                    'total_quantity': quantity,
                    'safety_orders_count': 0,
                    'commission': self.current_trade.commission
                })
            except Exception as e:
                logger.warning(f"[WARN] No se pudo guardar posicion en DB: {e}")

            return True
        except Exception as e:
            logger.error(f"[ERROR] open_position: {e}")
            return False

    def execute_dca(self, current_price: float) -> bool:
        """Ejecuta una Safety Order para promediar el precio."""
        trade = self.current_trade
        so_num = trade.safety_orders_count + 1

        # Cantidad de la recompra (Martingala sobre margen dinamico)
        if current_price <= 0:
            logger.error(f"[ERROR] DCA: current_price invalido: {current_price}")
            return False
        base_margin = self._get_dynamic_margin()
        so_margin = base_margin * (self.grinder_settings['martingale'] ** so_num)
        so_quantity = round((so_margin * self.leverage) / current_price, 3)

        # Verificar balance disponible antes de ejecutar
        try:
            available_balance = client.get_usdt_balance()
            if available_balance < so_margin:
                logger.warning(f"[DCA] Balance insuficiente: ${available_balance:.2f} < ${so_margin:.2f} requerido")
                return False
        except Exception as e:
            logger.warning(f"[DCA] No se pudo verificar balance: {e}")
            return False

        try:
            logger.info(f"[DCA] Ejecutando Recompra #{so_num}...")
            order = client.market_buy(so_quantity, self.symbol) if trade.side == 'long' else client.market_sell(so_quantity, self.symbol)
            fill_price = self._extract_fill_price(order, current_price)

            # Recalcular Promedio
            new_total_qty = trade.total_quantity + so_quantity
            trade.avg_price = ((trade.avg_price * trade.total_quantity) + (fill_price * so_quantity)) / new_total_qty
            trade.total_quantity = new_total_qty
            trade.safety_orders_count = so_num
            trade.commission += so_quantity * fill_price * COMMISSION_RATE

            logger.info(f"[DCA] OK | Nuevo Promedio: ${trade.avg_price:,.2f} | Total Qty: {trade.total_quantity}")

            # Actualizar posicion activa en DB
            try:
                db.save_active_position({
                    'entry_time': trade.entry_time.isoformat(),
                    'side': trade.side,
                    'avg_price': trade.avg_price,
                    'total_quantity': trade.total_quantity,
                    'safety_orders_count': trade.safety_orders_count,
                    'commission': trade.commission
                })
            except Exception:
                pass

            return True
        except Exception as e:
            logger.error(f"[ERROR] execute_dca: {e}")
            return False

    def update_position(self, high: float, low: float, close: float) -> Optional[Signal]:
        """Actualiza y gestiona TP, SL y DCA (en ese orden de prioridad)."""
        if self.current_trade is None: return None

        trade = self.current_trade
        settings = self.grinder_settings

        # 1. Verificar Take Profit (prioridad maxima)
        if trade.side == 'long':
            tp_price = trade.avg_price * (1 + settings['tp_pct'])
            if high >= tp_price:
                self.close_position(tp_price, "TAKE_PROFIT_GRINDER")
                return Signal.CLOSE
        else:
            tp_price = trade.avg_price * (1 - settings['tp_pct'])
            if low <= tp_price:
                self.close_position(tp_price, "TAKE_PROFIT_GRINDER")
                return Signal.CLOSE

        # 2. Verificar Stop Loss Catastrofico ANTES de DCA
        #    (no abrir mas posicion si ya estamos en zona de SL)
        if trade.side == 'long':
            sl_trigger = trade.avg_price * (1 - settings['sl_pct'])
            if low <= sl_trigger:
                self.close_position(close, "STOP_LOSS_CATASTROPHIC")
                return Signal.CLOSE
        else:
            sl_trigger = trade.avg_price * (1 + settings['sl_pct'])
            if high >= sl_trigger:
                self.close_position(close, "STOP_LOSS_CATASTROPHIC")
                return Signal.CLOSE

        # 3. Verificar DCA (Safety Orders) - solo si no tocamos SL
        if trade.safety_orders_count < settings['max_so']:
            if trade.side == 'long':
                dca_trigger = trade.avg_price * (1 - settings['dca_step'])
                if low <= dca_trigger:
                    self.execute_dca(close)
            else:
                dca_trigger = trade.avg_price * (1 + settings['dca_step'])
                if high >= dca_trigger:
                    self.execute_dca(close)

        return None

    def close_position(self, current_price: float, reason: str = "MANUAL") -> bool:
        """
        Cierra la posicion actual.
        """
        if self.current_trade is None: return False

        try:
            order = client.close_position(self.symbol)
            if order:
                current_price = self._extract_fill_price(order, current_price)

            trade = self.current_trade
            trade.exit_time = datetime.now()
            
            # PnL Neto (proteccion contra avg_price=0)
            if trade.avg_price <= 0:
                logger.error(f"[ERROR] avg_price invalido ({trade.avg_price}), usando close price para evitar crash")
                trade.avg_price = current_price
            pnl_pct = ((current_price - trade.avg_price) / trade.avg_price) if trade.side == 'long' else ((trade.avg_price - current_price) / trade.avg_price)
            pnl_pct = pnl_pct * self.leverage
            
            trade.commission += trade.total_quantity * current_price * COMMISSION_RATE
            trade.pnl = (trade.total_quantity * trade.avg_price * pnl_pct) - trade.commission
            trade.pnl_pct = pnl_pct

            logger.info(f"[GRINDER] CLOSE {trade.side.upper()} | PnL: ${trade.pnl:,.2f} ({pnl_pct*100:+.2f}%) | Razon: {reason}")

            self._update_stats(trade)
            self.trade_history.append(trade)

            # Persistir trade en SQLite
            try:
                trade_id = db.save_trade({
                    'entry_time': trade.entry_time.isoformat(),
                    'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                    'side': trade.side,
                    'entry_price': trade.avg_price,
                    'exit_price': current_price,
                    'avg_price': trade.avg_price,
                    'total_quantity': trade.total_quantity,
                    'safety_orders_count': trade.safety_orders_count,
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'exit_reason': reason,
                    'commission': trade.commission
                })
                # Actualizar outcome en features
                outcome = 'WIN' if trade.pnl > 0 else 'LOSS'
                db.update_feature_outcome(trade_id, outcome)
                db.clear_active_position()
            except Exception as e:
                logger.warning(f"[WARN] No se pudo guardar trade en DB: {e}")

            self.current_trade = None
            return True
        except Exception as e:
            logger.error(f"[ERROR] close_position: {e}")
            return False

    def _update_stats(self, trade: TradeRecord):
        """Actualiza estadisticas despues de un trade."""
        today = datetime.now().strftime('%Y-%m-%d')
        if self.daily_stats.date != today:
            self.daily_stats = DailyStats(date=today)

        self.daily_stats.total_pnl += trade.pnl
        if trade.pnl > 0:
            self.daily_stats.wins += 1
            self.daily_stats.consecutive_losses = 0
        else:
            self.daily_stats.losses += 1
            self.daily_stats.consecutive_losses += 1
            self.daily_stats.max_consecutive_losses = max(self.daily_stats.max_consecutive_losses, self.daily_stats.consecutive_losses)

        self._check_kill_switch()

    def _get_rolling_win_rate(self) -> float:
        """Calcula el Win Rate de los ultimos N trades."""
        if len(self.trade_history) < ROLLING_WR_WINDOW:
            return 1.0  # No tenemos suficientes datos, asumir OK
        recent = self.trade_history[-ROLLING_WR_WINDOW:]
        wins = sum(1 for t in recent if t.pnl > 0)
        return wins / len(recent)

    def _check_kill_switch(self):
        """Verifica si debe pausar el bot por seguridad."""
        # 1. Perdidas consecutivas
        if self.daily_stats.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            self._pause(f"{MAX_CONSECUTIVE_LOSSES} perdidas consecutivas")
            return

        # 2. Perdida diaria maxima
        try:
            balance = client.get_usdt_balance()
            if balance > 0:
                daily_loss_pct = abs(self.daily_stats.total_pnl) / balance
                if self.daily_stats.total_pnl < 0 and daily_loss_pct >= MAX_DAILY_LOSS_PCT:
                    self._pause(f"Perdida diaria >= {MAX_DAILY_LOSS_PCT*100}%")
                    return
        except:
            pass

        # 3. Rolling Win Rate - detecta degradacion de la estrategia
        rolling_wr = self._get_rolling_win_rate()
        if rolling_wr < ROLLING_WR_MIN:
            self._pause(f"Rolling WR ({rolling_wr*100:.1f}%) < minimo ({ROLLING_WR_MIN*100}%)")
            return

    def _pause(self, reason: str):
        self.is_paused = True
        self.pause_reason = reason
        logger.warning(f"[KILL SWITCH] Bot pausado: {reason}")

    def resume(self, force: bool = False):
        """Reanuda el bot. Si fue pausado por WR, solo reanuda si WR se recupero."""
        if not force and "Rolling WR" in self.pause_reason:
            rolling_wr = self._get_rolling_win_rate()
            if rolling_wr < ROLLING_WR_RESUME:
                logger.warning(f"[KILL SWITCH] No se puede reanudar: WR rolling ({rolling_wr*100:.1f}%) < {ROLLING_WR_RESUME*100}%")
                return
        self.is_paused = False
        self.pause_reason = ""
        logger.info("[TRADER] Bot reanudado")

    def has_open_position(self) -> bool:
        return self.current_trade is not None

    def get_position_info(self) -> Optional[Dict]:
        if self.current_trade is None: return None
        return {
            'side': self.current_trade.side,
            'avg_price': self.current_trade.avg_price,
            'total_quantity': self.current_trade.total_quantity,
            'so_count': self.current_trade.safety_orders_count
        }

    def sync_with_exchange(self):
        """Sincroniza el estado con exchange. Prioriza datos de DB sobre exchange."""
        try:
            position = client.get_position()

            if position and self.current_trade is None:
                # Intentar recuperar de DB primero (tiene datos mas completos)
                db_pos = db.get_active_position()
                if db_pos and db_pos.get('avg_price', 0) > 0:
                    self.current_trade = TradeRecord(
                        entry_time=datetime.fromisoformat(db_pos['entry_time']),
                        side=db_pos['side'],
                        avg_price=db_pos['avg_price'],
                        total_quantity=db_pos['total_quantity'],
                        safety_orders_count=db_pos.get('safety_orders_count', 0),
                        commission=db_pos.get('commission', 0.0)
                    )
                    logger.info(f"[TRADER] Posicion recuperada desde DB: {db_pos['side']} @ ${db_pos['avg_price']:.2f} (DCA: {db_pos.get('safety_orders_count', 0)})")
                else:
                    if db_pos:
                        logger.warning(f"[WARN] DB tiene avg_price invalido ({db_pos.get('avg_price')}), usando exchange")
                        db.clear_active_position()
                    # Fallback: reconstruir desde exchange (sin datos de DCA)
                    self.current_trade = TradeRecord(
                        entry_time=datetime.now(),
                        side=position['side'],
                        avg_price=position['entry_price'],
                        total_quantity=position['size'],
                        safety_orders_count=0
                    )
                    logger.info(f"[TRADER] Posicion recuperada desde Exchange: {position['side']} @ ${position['entry_price']:.2f}")

            elif not position and self.current_trade is not None:
                self.current_trade = None
                db.clear_active_position()

            elif not position:
                # Limpiar posicion huerfana en DB si existe
                db_pos = db.get_active_position()
                if db_pos:
                    db.clear_active_position()
                    logger.info("[TRADER] Posicion huerfana en DB limpiada")

        except Exception as e:
            logger.warning(f"[WARN] sync_with_exchange: {e}")

    def get_stats_summary(self) -> str:
        s = self.daily_stats
        total = s.wins + s.losses
        win_rate = (s.wins / total * 100) if total > 0 else 0
        return f"Trades: {total} | Wins: {s.wins} | WR: {win_rate:.1f}% | PnL: ${s.total_pnl:+.2f}"

trader = Trader()
