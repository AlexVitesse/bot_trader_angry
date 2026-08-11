"""
portfolio_sim.py — Simulador de CARTERA para V2
===============================================
Todos los experimentos anteriores simulaban un BTC secuencial idealizado. El
bot real es otra cosa, y las diferencias multiplican el riesgo:

  - varios pares a la vez (ML_V15_PAIRS)
  - hasta ML_MAX_CONCURRENT posiciones simultaneas
  - tope de 2 posiciones en la misma direccion (can_open)
  - equity COMPARTIDO: las posiciones concurrentes compiten por el capital
  - margen finito: notional/leverage tiene que caber
  - correlacion real entre pares (implicita al usar las series simultaneas)

Ademas corrige tres divergencias vivo-vs-backtest que la revision externa
identifico:

  1. FILL POSTERIOR AL CIERRE. La senal se detecta con el close de la vela t y
     la entrada se ejecuta al OPEN de t+1. El backtest viejo entraba al mismo
     close que generaba la senal, un fill imposible.
  2. max_bars DEL MOTOR (A=60, F=40), no ML_MAX_HOLD (15/30).
  3. DD sobre la curva de equity CON el capital inicial como primer pico.

Sin look-ahead intrabar: el stop se comprueba con el nivel fijado en la vela
anterior y solo despues se actualiza para la siguiente.

Uso:
    from portfolio_sim import PortfolioSim, cargar_pares
    datos = cargar_pares(['BTC/USDT', 'ETH/USDT'])
    res = PortfolioSim(datos, risk_pct=0.02).run()
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src import v2_engine as v2

# Costes (config/settings.py: 0.04% comision + 0.01% slippage por lado)
COMISION = 0.0005
FUNDING_8H = 0.00013          # mediana BTC perp (agent_D). Aproximacion.

PARQUETS = {
    'BTC/USDT': 'BTC_USDT_4h_full.parquet',
    'BNB/USDT': 'BNB_USDT_4h_full.parquet',
    'DOGE/USDT': 'DOGE_USDT_4h_full.parquet',
    'ETH/USDT': 'ETH_USDT_4h_full.parquet',
    'OP/USDT': 'OPUSDT_4h_v15.parquet',
}


def cargar_pares(pares: list[str], params: dict = None) -> dict:
    """Carga 4h + daily derivado y construye features por par."""
    params = params or v2.PARAMS_V2
    out = {}
    for p in pares:
        f = ROOT / 'data' / PARQUETS[p]
        if not f.exists():
            print(f"  aviso: sin datos para {p}")
            continue
        df = pd.read_parquet(f)[['open', 'high', 'low', 'close', 'volume']].astype(float)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df = df[~df.index.duplicated(keep='last')].sort_index()
        d1 = df.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                    'close': 'last', 'volume': 'sum'}).dropna()
        out[p] = v2.build_features(df, d1, None, params)
    return out


@dataclass
class Pos:
    par: str
    direccion: int          # 1 long, -1 short
    sig: str
    entrada: float
    notional: float
    trail: float
    max_bars: int
    stop: float
    extremo: float          # peak (long) o trough (short)
    barras: int = 0
    funding_pagado: float = 0.0
    equity_entrada: float = 0.0   # denominador correcto para el retorno del trade


@dataclass
class Resultado:
    equity: pd.Series
    trades: list = field(default_factory=list)
    rechazos: dict = field(default_factory=dict)

    @property
    def metricas(self) -> dict:
        eq = self.equity
        if len(eq) < 2 or not self.trades:
            return {'n': 0}
        años = (eq.index[-1] - eq.index[0]).days / 365.25
        # el primer valor de eq ES el capital inicial -> cuenta como primer pico
        dd = float((1 - eq / eq.cummax()).max() * 100)
        r = np.array([t['r'] for t in self.trades])     # retorno sobre equity de entrada
        w, l = r[r > 0].sum(), abs(r[r <= 0].sum())
        rng = np.random.default_rng(42)
        mu = rng.choice(r, size=(3000, len(r)), replace=True).mean(axis=1)
        return {
            'n': len(r), 'por_año': len(r) / años,
            'wr': float((r > 0).mean() * 100),
            'pf': float(w / l) if l else float('inf'),
            'cagr': float(((eq.iloc[-1] / eq.iloc[0]) ** (1 / años) - 1) * 100),
            'dd': dd, 'p': float((mu <= 0).mean()),
            'final': float(eq.iloc[-1] / eq.iloc[0]),
        }


class PortfolioSim:
    def __init__(self, datos: dict, params: dict = None, risk_pct: float = 0.02,
                 max_concurrent: int = 3, max_misma_dir: int = 2,
                 leverage: int = 4, max_notional_pct: float = 2.5,
                 capital: float = 10_000.0, aplicar_funding: bool = True):
        self.datos = datos
        self.params = params or v2.PARAMS_V2
        self.risk_pct = risk_pct
        self.max_concurrent = max_concurrent
        self.max_misma_dir = max_misma_dir
        self.leverage = leverage
        self.max_notional_pct = max_notional_pct
        self.capital0 = capital
        self.aplicar_funding = aplicar_funding

    # -- gates de can_open, replicando portfolio_manager.can_open -------------
    def _puede_abrir(self, par, direccion, abiertas, notional, equity):
        if len(abiertas) >= self.max_concurrent:
            return 'max_concurrent'
        if par in abiertas:
            return 'par_ya_abierto'
        if sum(1 for p in abiertas.values() if p.direccion == direccion) >= self.max_misma_dir:
            return 'max_misma_direccion'
        margen_usado = sum(p.notional for p in abiertas.values()) / self.leverage
        if notional / self.leverage > equity - margen_usado:
            return 'sin_margen'
        return None

    def run(self, desde=None, hasta=None) -> Resultado:
        # indice maestro: union de todas las velas de todos los pares
        idxs = [d.index for d in self.datos.values()]
        maestro = idxs[0]
        for i in idxs[1:]:
            maestro = maestro.union(i)
        if desde is not None:
            maestro = maestro[maestro >= desde]
        if hasta is not None:
            maestro = maestro[maestro < hasta]

        # mapa timestamp -> posicion en el df de cada par
        pos_en = {p: {ts: i for i, ts in enumerate(d.index)}
                  for p, d in self.datos.items()}

        cash = self.capital0
        abiertas: dict[str, Pos] = {}
        pendientes: list[tuple] = []       # (par, sig) a ejecutar en la sig. vela
        trades, rechazos = [], {}
        curva_ts, curva_eq = [], []

        for ts in maestro:
            # ---------- 1. marcar a mercado ----------
            no_realizado = 0.0
            for par, p in abiertas.items():
                i = pos_en[par].get(ts)
                if i is None:
                    continue
                px = float(self.datos[par]['close'].iloc[i])
                no_realizado += (px - p.entrada) / p.entrada * p.direccion * p.notional
            equity = cash + no_realizado
            curva_ts.append(ts)
            curva_eq.append(equity)
            if equity <= 0:
                break                                    # cuenta liquidada

            # ---------- 2. salidas (stop fijado en la vela anterior) ----------
            for par in list(abiertas):
                i = pos_en[par].get(ts)
                if i is None:
                    continue
                d = self.datos[par]
                hi, lo = float(d['high'].iloc[i]), float(d['low'].iloc[i])
                cl = float(d['close'].iloc[i])
                p = abiertas[par]
                p.barras += 1
                if self.aplicar_funding:                 # 4h = medio periodo de 8h
                    p.funding_pagado += p.notional * FUNDING_8H * 0.5 * p.direccion

                salida, motivo = None, None
                if p.direccion == 1 and lo <= p.stop:
                    salida, motivo = p.stop, 'SL'
                elif p.direccion == -1 and hi >= p.stop:
                    salida, motivo = p.stop, 'SL'
                elif p.barras >= p.max_bars:
                    salida, motivo = cl, 'TIMEOUT'

                if salida is None:
                    # actualizar trailing para la SIGUIENTE vela
                    if p.direccion == 1:
                        p.extremo = max(p.extremo, hi)
                        p.stop = max(p.stop, p.extremo * (1 - p.trail))
                    else:
                        p.extremo = min(p.extremo, lo)
                        p.stop = min(p.stop, p.extremo * (1 + p.trail))
                    continue

                bruto = (salida - p.entrada) / p.entrada * p.direccion
                pnl = (bruto - 2 * COMISION) * p.notional - p.funding_pagado
                cash += pnl
                trades.append({
                    'par': par, 'sig': p.sig, 'dir': p.direccion,
                    'ts_salida': ts, 'motivo': motivo, 'barras': p.barras,
                    'notional': p.notional, 'pnl': pnl,
                    'r': pnl / p.equity_entrada,   # equity de ENTRADA, no de salida
                })
                del abiertas[par]

            # ---------- 3. ejecutar pendientes al OPEN de esta vela ----------
            for par, sig in pendientes:
                i = pos_en[par].get(ts)
                if i is None or par in abiertas:
                    continue
                d = self.datos[par]
                entrada = float(d['open'].iloc[i])       # fill posterior al cierre
                atr = float(d['atr_pct'].iloc[i - 1]) if i > 0 else float(d['atr_pct'].iloc[i])
                if sig == 'A_LONG':
                    mult, fl, ce = 2.5, self.params['a_trail_floor_pct'], self.params['a_trail_ceiling_pct']
                    mb, direccion = self.params['a_max_bars'], 1
                else:
                    mult, fl, ce = 2.0, self.params['f_trail_floor_pct'], self.params['f_trail_ceiling_pct']
                    mb = self.params['f_max_bars']
                    direccion = 1 if sig == 'F_LONG' else -1
                trail = min(max(atr * mult, fl), ce)

                notional = min((equity * self.risk_pct) / trail,
                               equity * self.max_notional_pct)
                motivo = self._puede_abrir(par, direccion, abiertas, notional, equity)
                if motivo:
                    rechazos[motivo] = rechazos.get(motivo, 0) + 1
                    continue
                abiertas[par] = Pos(
                    par=par, direccion=direccion, sig=sig, entrada=entrada,
                    notional=notional, trail=trail, max_bars=mb,
                    stop=entrada * (1 - trail * direccion), extremo=entrada,
                    equity_entrada=equity,
                )
            pendientes = []

            # ---------- 4. detectar senales en la vela CERRADA ----------
            for par, d in self.datos.items():
                if par in abiertas:
                    continue
                i = pos_en[par].get(ts)
                if i is None:
                    continue
                sig = v2.detect_signal(d, i, self.params)
                if sig is not None:
                    pendientes.append((par, sig))

        eq = pd.Series(curva_eq, index=pd.DatetimeIndex(curva_ts))
        return Resultado(equity=eq, trades=trades, rechazos=rechazos)
