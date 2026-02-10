"""
Trader - Motor de Ejecucion
============================
Gestiona la ejecucion de trades con soporte para DCA (Grinder).
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, field

# Agregar raiz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exchange import client
from src.strategy import strategy, Signal
from config.settings import (
    SYMBOL,
    LEVERAGE,
    INITIAL_CAPITAL,
    MAX_POSITION_SIZE_PCT,
    COMMISSION_RATE,
    MAX_DAILY_LOSS_PCT,
    MAX_CONSECUTIVE_LOSSES,
    MAX_TRADES_PER_DAY
)


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

        # Inicializar
        self._setup()

    def _setup(self):
        """Configura el trader."""
        try:
            # Configurar leverage
            client.set_leverage(self.leverage, self.symbol)
            print(f"[TRADER] Leverage configurado: {self.leverage}x")
        except Exception as e:
            print(f"[WARN] No se pudo configurar leverage: {e}")

    def calculate_base_quantity(self, entry_price: float) -> float:
        """Calcula cantidad inicial basada en BASE_ORDER_MARGIN."""
        notional = self.grinder_settings['base_margin'] * self.leverage
        quantity = notional / entry_price
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
            fill_price = float(order.get('avgPrice', current_price))

            # Sincronizar con Estrategia
            strategy.open_trade(side, fill_price, quantity)

            self.current_trade = TradeRecord(
                entry_time=datetime.now(),
                side=side,
                avg_price=fill_price,
                total_quantity=quantity,
                safety_orders_count=0,
                commission=quantity * fill_price * COMMISSION_RATE
            )

            print(f"[GRINDER] OPEN {side.upper()} | Precio: ${fill_price:,.2f} | Qty: {quantity}")
            self.daily_stats.trades += 1
            return True
        except Exception as e:
            print(f"[ERROR] open_position: {e}")
            return False

    def execute_dca(self, current_price: float):
        """Ejecuta una Safety Order para promediar el precio."""
        trade = self.current_trade
        so_num = trade.safety_orders_count + 1
        
        # Cantidad de la recompra (Martingala)
        so_margin = self.grinder_settings['base_margin'] * (self.grinder_settings['martingale'] ** so_num)
        so_quantity = round((so_margin * self.leverage) / current_price, 3)

        try:
            print(f"[DCA] Ejecutando Recompra #{so_num}...")
            order = client.market_buy(so_quantity, self.symbol) if trade.side == 'long' else client.market_sell(so_quantity, self.symbol)
            fill_price = float(order.get('avgPrice', current_price))

            # Recalcular Promedio
            new_total_qty = trade.total_quantity + so_quantity
            trade.avg_price = ((trade.avg_price * trade.total_quantity) + (fill_price * so_quantity)) / new_total_qty
            trade.total_quantity = new_total_qty
            trade.safety_orders_count = so_num
            trade.commission += so_quantity * fill_price * COMMISSION_RATE

            print(f"[DCA] OK | Nuevo Promedio: ${trade.avg_price:,.2f} | Total Qty: {trade.total_quantity}")
        except Exception as e:
            print(f"[ERROR] execute_dca: {e}")

    def update_position(self, high: float, low: float, close: float) -> Optional[Signal]:
        """Actualiza y gestiona TP, DCA y SL."""
        if self.current_trade is None: return None

        trade = self.current_trade
        settings = self.grinder_settings
        
        # 1. Verificar Take Profit
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

        # 2. Verificar DCA (Safety Orders)
        if trade.safety_orders_count < settings['max_so']:
            if trade.side == 'long':
                dca_trigger = trade.avg_price * (1 - settings['dca_step'])
                if low <= dca_trigger:
                    self.execute_dca(close)
            else:
                dca_trigger = trade.avg_price * (1 + settings['dca_step'])
                if high >= dca_trigger:
                    self.execute_dca(close)

        # 3. Verificar Stop Loss Catastrofico
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

        return None

    def close_position(self, current_price: float, reason: str = "MANUAL") -> bool:
        """
        Cierra la posicion actual.
        """
        if self.current_trade is None: return False

        try:
            order = client.close_position(self.symbol)
            if order:
                current_price = float(order.get('avgPrice', current_price))

            trade = self.current_trade
            trade.exit_time = datetime.now()
            
            # PnL Neto
            pnl_pct = ((current_price - trade.avg_price) / trade.avg_price) if trade.side == 'long' else ((trade.avg_price - current_price) / trade.avg_price)
            pnl_pct = pnl_pct * self.leverage
            
            trade.commission += trade.total_quantity * current_price * COMMISSION_RATE
            trade.pnl = (trade.total_quantity * trade.avg_price * pnl_pct) - trade.commission
            trade.pnl_pct = pnl_pct

            print(f"[GRINDER] CLOSE {trade.side.upper()} | PnL: ${trade.pnl:,.2f} ({pnl_pct*100:+.2f}%) | Razon: {reason}")

            self._update_stats(trade)
            self.trade_history.append(trade)
            self.current_trade = None
            strategy.close_trade()
            return True
        except Exception as e:
            print(f"[ERROR] close_position: {e}")
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

    def _check_kill_switch(self):
        """Verifica si debe pausar el bot por seguridad."""
        if self.daily_stats.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            self._pause(f"{MAX_CONSECUTIVE_LOSSES} perdidas consecutivas")
            return

        try:
            balance = client.get_usdt_balance()
            if balance > 0:
                daily_loss_pct = abs(self.daily_stats.total_pnl) / balance
                if self.daily_stats.total_pnl < 0 and daily_loss_pct >= MAX_DAILY_LOSS_PCT:
                    self._pause(f"Perdida diaria >= {MAX_DAILY_LOSS_PCT*100}%")
        except: pass

    def _pause(self, reason: str):
        self.is_paused = True
        self.pause_reason = reason
        print(f"[KILL SWITCH] Bot pausado: {reason}")

    def resume(self):
        self.is_paused = False
        self.pause_reason = ""

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
        """Sincroniza el estado con exchange."""
        try:
            position = client.get_position()
            if position and self.current_trade is None:
                self.current_trade = TradeRecord(
                    entry_time=datetime.now(),
                    side=position['side'],
                    avg_price=position['entry_price'],
                    total_quantity=position['size'],
                    safety_orders_count=0
                )
            elif not position and self.current_trade is not None:
                self.current_trade = None
        except: pass

    def get_stats_summary(self) -> str:
        s = self.daily_stats
        total = s.wins + s.losses
        win_rate = (s.wins / total * 100) if total > 0 else 0
        return f"Trades: {total} | Wins: {s.wins} | WR: {win_rate:.1f}% | PnL: ${s.total_pnl:+.2f}"

trader = Trader()
