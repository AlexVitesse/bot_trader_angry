"""
yield_manager.py — Mercado-Pago-style yield para el bot V2.
==========================================================
Hace trabajar el capital ocioso (~95% del tiempo segun V2's selectividad).

Arquitectura:
  - Trading buffer: % del capital se mantiene en futures wallet (listo para trades)
  - Yield pool: el resto en Binance Simple Earn Flexible USDT (~2-5% APY)
  - Rebalanceo periodico: sweep si futures > max, redeem si futures < min
  - NO interfiere con la logica de trading — solo opera entre ciclos de senal

Modos:
  - TESTNET: simulate_mode=True. Trackea yield virtualmente (testnet no tiene Earn).
  - MAINNET: opera con la API real de Binance Simple Earn.

Comando Telegram: /yield muestra balance + interes acumulado.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default config (override via config/settings.py)
DEFAULT_CONFIG = {
    'enabled': True,
    'simulate_mode': None,            # auto-detect: True si testnet
    'buffer_target_pct': 0.20,        # 20% en futures wallet (ready for ~3 trades)
    'buffer_max_pct': 0.30,           # arriba de 30% -> sweep
    'buffer_min_pct': 0.15,           # debajo de 15% -> redeem
    'rebalance_interval_s': 600,      # cada 10 min
    'simulate_apy': 0.03,             # 3% APY default (Binance Earn flexible USDT)
    'min_sweep_amount': 50.0,         # no hacer sweeps de <$50 (overhead)
    'state_file': 'data/yield_state.json',
    'earn_asset': 'USDT',
    'earn_product_id': 'USDT001',     # Binance Simple Earn flexible USDT product
}


@dataclass
class YieldState:
    """Estado persistente del yield manager."""
    earn_balance: float = 0.0              # balance en yield pool (USDT)
    accumulated_interest: float = 0.0      # interes total acumulado (USDT)
    last_accrual_ts: float = 0.0           # timestamp ultima acumulacion
    last_rebalance_ts: float = 0.0         # timestamp ultimo rebalance
    total_swept: float = 0.0               # total movido a Earn (lifetime)
    total_redeemed: float = 0.0            # total movido a futures (lifetime)
    n_sweeps: int = 0
    n_redeems: int = 0
    started_at: str = ''                   # ISO timestamp del primer arranque

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'YieldState':
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(**{k: data.get(k, getattr(cls(), k)) for k in cls.__annotations__})
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()


class YieldManager:
    """Gestor de yield estilo Mercado Pago para el bot V2."""

    def __init__(self, exchange_futures, config: Optional[dict] = None,
                 testnet: bool = False):
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        # Auto-detect simulate mode
        if cfg['simulate_mode'] is None:
            cfg['simulate_mode'] = testnet
        self.cfg = cfg
        self.exchange = exchange_futures   # ccxt binanceusdm
        self.testnet = testnet
        self.simulate = cfg['simulate_mode']
        self.state = YieldState.load(cfg['state_file'])
        if not self.state.started_at:
            self.state.started_at = datetime.now(timezone.utc).isoformat()
            self.state.save(cfg['state_file'])
        mode = 'SIMULATE' if self.simulate else 'LIVE'
        logger.info(f"[YIELD] inicializado | modo={mode} | "
                    f"buffer target={cfg['buffer_target_pct']:.0%} | "
                    f"APY sim={cfg['simulate_apy']:.1%}")
        if self.state.earn_balance > 0:
            logger.info(f"[YIELD] balance previo cargado: "
                        f"${self.state.earn_balance:.2f} en pool, "
                        f"${self.state.accumulated_interest:.2f} interes")

    # ===================================================================
    # API publica
    # ===================================================================
    def check_and_rebalance(self) -> dict:
        """
        Llamar periodicamente (cada N min). Acumula intereses, decide si
        sweep/redeem segun los thresholds. Devuelve dict con balances.
        """
        if not self.cfg['enabled']:
            return {'enabled': False}

        now = time.time()
        if now - self.state.last_rebalance_ts < self.cfg['rebalance_interval_s']:
            # No es tiempo de rebalance todavia, solo acumular interes
            self._accrue_interest(now)
            return self.get_status()

        try:
            self._accrue_interest(now)
            real_futures = self._get_futures_balance()
            # En SIMULATE no movemos capital real -> el "futures virtual" es el
            # balance real menos lo que VIRTUALMENTE ya esta en la pool.
            # En LIVE el real_futures ya refleja los sweeps reales.
            if self.simulate:
                futures_balance = max(0, real_futures - self.state.earn_balance)
                total_capital = real_futures  # el real es la unica fuente
            else:
                futures_balance = real_futures
                total_capital = real_futures + self.state.earn_balance

            if total_capital < 10:
                logger.warning(f"[YIELD] capital total ${total_capital:.2f} "
                               f"demasiado bajo, omitiendo rebalance")
                return self.get_status()

            target = total_capital * self.cfg['buffer_target_pct']
            buf_max = total_capital * self.cfg['buffer_max_pct']
            buf_min = total_capital * self.cfg['buffer_min_pct']

            excess = futures_balance - buf_max
            deficit = buf_min - futures_balance

            if excess >= self.cfg['min_sweep_amount']:
                # Tenemos exceso en futures, mover a Earn
                amount_to_sweep = futures_balance - target  # llevar a target
                self._do_sweep(amount_to_sweep)
            elif deficit > 0:
                # Necesitamos mas en futures, redimir de Earn
                amount_to_redeem = target - futures_balance  # llevar a target
                self._do_redeem(amount_to_redeem)

            self.state.last_rebalance_ts = now
            self.state.save(self.cfg['state_file'])
        except Exception as e:
            logger.error(f"[YIELD] error en rebalance: {e}")

        return self.get_status()

    def get_status(self) -> dict:
        """Estado actual para logs / Telegram /yield."""
        try:
            real_futures = self._get_futures_balance()
        except Exception:
            real_futures = 0.0
        # En SIMULATE, el "futures virtual" es real menos lo virtualmente movido.
        if self.simulate:
            futures = max(0, real_futures - self.state.earn_balance)
            total = real_futures
        else:
            futures = real_futures
            total = real_futures + self.state.earn_balance
        days_running = 0.0
        if self.state.started_at:
            try:
                start = datetime.fromisoformat(self.state.started_at)
                days_running = (datetime.now(timezone.utc) - start).total_seconds() / 86400
            except Exception:
                pass
        annualized_yield_pct = 0.0
        if self.state.earn_balance > 0 and days_running > 1:
            # Estimacion grosera: interest / earn_balance promedio anualizado
            annualized_yield_pct = (
                self.state.accumulated_interest /
                max(self.state.earn_balance, 1) *
                (365 / max(days_running, 1)) * 100
            )
        return {
            'enabled': self.cfg['enabled'],
            'mode': 'SIMULATE' if self.simulate else 'LIVE',
            'futures_balance': round(futures, 2),
            'earn_balance': round(self.state.earn_balance, 2),
            'accumulated_interest': round(self.state.accumulated_interest, 4),
            'total_capital': round(total, 2),
            'buffer_target_pct': self.cfg['buffer_target_pct'],
            'apy_simulated': self.cfg['simulate_apy'] if self.simulate else None,
            'n_sweeps': self.state.n_sweeps,
            'n_redeems': self.state.n_redeems,
            'total_swept': round(self.state.total_swept, 2),
            'total_redeemed': round(self.state.total_redeemed, 2),
            'days_running': round(days_running, 1),
            'effective_annualized_pct': round(annualized_yield_pct, 2),
        }

    def telegram_summary(self) -> str:
        """Resumen formato Telegram /yield."""
        s = self.get_status()
        if not s.get('enabled'):
            return "Yield manager: DISABLED"
        apy = f"{s['apy_simulated']:.1%}" if s.get('apy_simulated') else 'live'
        return (
            f"💰 *Yield Manager* ({s['mode']})\n"
            f"Futures: `${s['futures_balance']:.2f}`\n"
            f"Earn pool: `${s['earn_balance']:.2f}` (APY {apy})\n"
            f"Interes acum: `${s['accumulated_interest']:.4f}`\n"
            f"Total capital: `${s['total_capital']:.2f}`\n"
            f"Sweeps/Redeems: {s['n_sweeps']}/{s['n_redeems']}\n"
            f"Dias: {s['days_running']:.1f}\n"
            f"APY efectiva anualizada: {s['effective_annualized_pct']:.2f}%"
        )

    # ===================================================================
    # Internos
    # ===================================================================
    def _get_futures_balance(self) -> float:
        """USDT free en futures wallet."""
        bal = self.exchange.fetch_balance()
        usdt = bal.get('USDT', {})
        # En ccxt, 'free' es el disponible para tradeo
        free = float(usdt.get('free', 0) or 0)
        return free

    def _accrue_interest(self, now: float):
        """Acumula interes simulado por el tiempo transcurrido."""
        if not self.simulate:
            # En live, el interes se acumula del lado de Binance automaticamente.
            # Solo refrescamos earn_balance llamando a la API (no implementado aqui).
            return
        if self.state.last_accrual_ts == 0:
            self.state.last_accrual_ts = now
            return
        dt = now - self.state.last_accrual_ts
        if dt <= 0 or self.state.earn_balance <= 0:
            self.state.last_accrual_ts = now
            return
        # interest = balance * APY * dt / (365*86400)
        apy = self.cfg['simulate_apy']
        interest = self.state.earn_balance * apy * dt / (365 * 86400)
        self.state.earn_balance += interest
        self.state.accumulated_interest += interest
        self.state.last_accrual_ts = now

    def _do_sweep(self, amount_usdt: float):
        """Mover USDT de futures wallet a Earn pool."""
        if amount_usdt < self.cfg['min_sweep_amount']:
            return
        if self.simulate:
            # Solo tracking — no movemos capital real
            self.state.earn_balance += amount_usdt
            self.state.total_swept += amount_usdt
            self.state.n_sweeps += 1
            logger.info(f"[YIELD] SIM sweep ${amount_usdt:.2f} -> Earn "
                        f"(pool: ${self.state.earn_balance:.2f})")
        else:
            # MAINNET: transferir futures -> spot, luego subscribe Earn
            ok = self._real_sweep(amount_usdt)
            if ok:
                self.state.earn_balance += amount_usdt
                self.state.total_swept += amount_usdt
                self.state.n_sweeps += 1
                logger.info(f"[YIELD] LIVE sweep ${amount_usdt:.2f} -> Earn OK")
            else:
                logger.error(f"[YIELD] sweep ${amount_usdt:.2f} fallo")

    def _do_redeem(self, amount_usdt: float):
        """Mover USDT de Earn pool a futures wallet."""
        amount_usdt = min(amount_usdt, self.state.earn_balance)
        if amount_usdt <= 0:
            return
        if self.simulate:
            self.state.earn_balance -= amount_usdt
            self.state.total_redeemed += amount_usdt
            self.state.n_redeems += 1
            logger.info(f"[YIELD] SIM redeem ${amount_usdt:.2f} -> futures "
                        f"(pool: ${self.state.earn_balance:.2f})")
        else:
            ok = self._real_redeem(amount_usdt)
            if ok:
                self.state.earn_balance -= amount_usdt
                self.state.total_redeemed += amount_usdt
                self.state.n_redeems += 1
                logger.info(f"[YIELD] LIVE redeem ${amount_usdt:.2f} -> futures OK")
            else:
                logger.error(f"[YIELD] redeem ${amount_usdt:.2f} fallo")

    # ===================================================================
    # Integracion con Binance API real (solo mainnet)
    # ===================================================================
    def _real_sweep(self, amount_usdt: float) -> bool:
        """
        MAINNET: transferir USDT de futures wallet -> spot, luego subscribe
        a Binance Simple Earn Flexible USDT.
        """
        try:
            # 1) Transfer UMFUTURE -> MAIN (futures -> spot)
            self.exchange.sapiPostAssetTransfer({
                'type': 'UMFUTURE_MAIN',
                'asset': 'USDT',
                'amount': str(amount_usdt),
            })
            time.sleep(1)
            # 2) Subscribe a Earn flexible
            self.exchange.sapiPostSimpleEarnFlexibleSubscribe({
                'productId': self.cfg['earn_product_id'],
                'amount': str(amount_usdt),
            })
            return True
        except Exception as e:
            logger.error(f"[YIELD] _real_sweep error: {e}")
            return False

    def _real_redeem(self, amount_usdt: float) -> bool:
        """
        MAINNET: redeem de Earn flexible -> spot, luego transferir
        spot -> futures wallet.
        """
        try:
            # 1) Redeem flexible
            self.exchange.sapiPostSimpleEarnFlexibleRedeem({
                'productId': self.cfg['earn_product_id'],
                'amount': str(amount_usdt),
                'destAccount': 'SPOT',
            })
            time.sleep(2)
            # 2) Transfer MAIN -> UMFUTURE
            self.exchange.sapiPostAssetTransfer({
                'type': 'MAIN_UMFUTURE',
                'asset': 'USDT',
                'amount': str(amount_usdt),
            })
            return True
        except Exception as e:
            logger.error(f"[YIELD] _real_redeem error: {e}")
            return False

    def refresh_live_earn_balance(self):
        """
        MAINNET: refresca earn_balance llamando a la API de Binance para
        ver el balance real en Simple Earn. Llamar periodicamente.
        """
        if self.simulate:
            return
        try:
            r = self.exchange.sapiGetSimpleEarnFlexiblePosition({
                'asset': self.cfg['earn_asset'],
            })
            rows = r.get('rows', [])
            total = 0.0
            for row in rows:
                total += float(row.get('totalAmount', 0))
            self.state.earn_balance = total
        except Exception as e:
            logger.warning(f"[YIELD] refresh_live_earn_balance error: {e}")
