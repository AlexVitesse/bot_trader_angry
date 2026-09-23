"""KPI de parada: PnL real vs simulado, trade a trade.

Lee ml_trades (columnas que rellena PortfolioManager.reconcile_closed_trades):
  real   = pnl_real / notional          (Binance: realizado + comision + funding)
  sim    = pnl_sim_pct                  (_sim_*_trailing con la misma senal)
  slip   = entry_price / signal_close - 1  (fill real vs close de la vela de senal)

CLAUDE.md: divergencia > 25% a 50 trades -> STOP.

Si existe ml_exec_snapshots (docs/GRABACION_DATOS_VIVO.md, capa A) reporta
ademas el slippage medio por evento y tipo de orden: avg_price vs ref_price
(entrada: close de la vela de senal; salida: precio del stop/trail). Positivo
= coste.

Uso: python experiments/ejecucion_vivo/compare_live_vs_sim.py [ruta_db]
"""
import sqlite3
import sys
from pathlib import Path

DB = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path(__file__).resolve().parents[2] / 'data' / 'ml_bot.db'

with sqlite3.connect(DB) as c:
    c.row_factory = sqlite3.Row
    rows = [dict(r) for r in c.execute(
        "SELECT id, entry_time, side, entry_price, notional, exit_reason, "
        "exit_sim_reason, pnl_real, pnl_sim_pct, signal_close FROM ml_trades "
        "WHERE pnl_real IS NOT NULL AND pnl_sim_pct IS NOT NULL ORDER BY id")]


def slippage_snapshots():
    with sqlite3.connect(DB) as c:
        if not c.execute("SELECT 1 FROM sqlite_master WHERE name='ml_exec_snapshots'").fetchone():
            return
        q = c.execute(
            "SELECT s.event, s.order_type, s.avg_price, s.ref_price, s.spread_bps, "
            "COALESCE(t.side, 'long') FROM ml_exec_snapshots s LEFT JOIN ml_trades t "
            "ON t.symbol = s.symbol AND t.entry_time = s.entry_time "
            "WHERE s.event != 'signal' AND s.avg_price > 0 AND s.ref_price > 0").fetchall()
    grupos = {}
    for ev, ot, avg, ref, spr, side in q:
        d = 1 if side == 'long' else -1
        # entrada: pagar mas que ref es coste; salida: cobrar menos que ref es coste
        slip = (avg / ref - 1) * d * (1 if ev == 'entry_fill' else -1)
        grupos.setdefault((ev, ot), []).append((slip, spr))
    print('\nSlippage real (ml_exec_snapshots) vs 0,02%/lado asumido:')
    for (ev, ot), v in sorted(grupos.items()):
        sl = [x for x, _ in v]
        sp = [y for _, y in v if y is not None]
        print(f"  {ev:10} {ot or '?':12} n={len(v):3d} slip medio {sum(sl)/len(sl):+.3%}"
              + (f" | spread medio {sum(sp)/len(sp):.2f} bps" if sp else ''))


if not rows:
    slippage_snapshots()
    sys.exit('Sin trades con PnL real y simulado todavia.')

print(f"{'id':>4} {'entrada':16} {'real%':>7} {'sim%':>7} {'slip%':>6}  salida real/sim")
sum_real = sum_sim = 0.0
for r in rows:
    real = r['pnl_real'] / r['notional']
    sim = r['pnl_sim_pct']
    slip = (r['entry_price'] / r['signal_close'] - 1) * (1 if r['side'] == 'long' else -1)
    sum_real += real
    sum_sim += sim
    print(f"{r['id']:>4} {r['entry_time'][:16]:16} {real:+7.2%} {sim:+7.2%} "
          f"{slip:+6.2%}  {r['exit_reason']}/{r['exit_sim_reason']}")

n = len(rows)
div = (sum_real - sum_sim) / abs(sum_sim) if sum_sim else float('nan')
print(f"\nn={n} | suma real {sum_real:+.2%} | suma sim {sum_sim:+.2%} | "
      f"divergencia {div:+.1%}")
if n >= 50 and abs(div) > 0.25:
    print('KPI: divergencia > 25% con >= 50 trades -> STOP (CLAUDE.md)')
slippage_snapshots()
