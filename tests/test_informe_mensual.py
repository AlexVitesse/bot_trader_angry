"""El informe mensual cuenta bien los trades del mes y cruza la alerta."""
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import informe_mensual as im  # noqa: E402


def _db(f, filas, estado=None):
    con = sqlite3.connect(f)
    con.execute("CREATE TABLE ml_trades (entry_time TEXT, exit_time TEXT, side TEXT, pnl REAL, "
                "pnl_real REAL, pnl_sim_pct REAL, notional REAL)")
    con.executemany("INSERT INTO ml_trades VALUES (?,?,?,?,?,?,?)", filas)
    con.execute("CREATE TABLE ml_state (key TEXT, value TEXT)")
    con.executemany("INSERT INTO ml_state VALUES (?,?)", (estado or {}).items())
    con.commit()
    con.close()


def test_informe(tmp_path):
    _db(tmp_path / 'ml_bot.db', [
        ('2026-02-13T04:00:00+00:00', '2026-02-13T08:00:00+00:00', 'long', -3, None, None, 100),  # V15: fuera
        ('2026-10-21T04:00:00+00:00', '2026-10-22T04:00:00+00:00', 'long', 100, 98, 0.0105, 10000),
        ('2026-10-25T04:00:00+00:00', '2026-10-26T04:00:00+00:00', 'long', -50, None, None, 10000),
        ('2026-11-02T04:00:00+00:00', '2026-11-03T04:00:00+00:00', 'long', 70, 70, None, 10000),
    ], {'balance': '4800', 'peak_balance': '5000'})
    _db(tmp_path / 'ml_agresivo.db', [
        ('2026-10-18T20:00:00+00:00', '2026-10-19T00:00:00+00:00', 'short', -4.6, -4.6, None, 184)])
    (tmp_path / 'alerta_caidas.jsonl').write_text(
        json.dumps({'dia': '2026-10-24', 'p': .3, 'umbral': .27, 'activa': True}) + '\n'
        + json.dumps({'dia': '2026-10-20', 'p': .1, 'umbral': .27, 'activa': False}) + '\n')

    ini, fin, tit = im.mes_anterior(pd.Timestamp('2026-11-01 01:00', tz='UTC'))
    assert (ini, fin) == (pd.Timestamp('2026-10-01', tz='UTC'), pd.Timestamp('2026-11-01', tz='UTC'))
    txt = im.generar(tmp_path, ini, fin, tit)
    assert "Mes: 2 trades | WR 50% | PF 1.96 | PnL $+48.00" in txt          # usa pnl_real si existe
    assert "Desde 2026-08-11: 3 trades" in txt                              # el de febrero (V15) no cuenta
    assert "DD 4.0%" in txt
    assert "en día de alerta: 1 trades" in txt and "PnL $-50.00" in txt     # entrada 25-oct, alerta del 24
    assert "short 1t $-4.60" in txt
