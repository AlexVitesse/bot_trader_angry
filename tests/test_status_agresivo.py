import json
import sqlite3

import config.settings as settings
import src.ml_bot as ml_bot


class Ticker:
    def fetch_ticker(self, sym):
        return {'last': 110.0}


def test_status_agresivo_lee_el_otro_bot(tmp_path, monkeypatch):
    (tmp_path / 'paper_agresivo_state.json').write_text(json.dumps({
        'balance': 104.0,
        'positions': {'SOL/USDT': {'side': 'short', 'qty': 0.5, 'entry': 100.0}}}))
    con = sqlite3.connect(tmp_path / 'ml_agresivo.db')
    con.execute("CREATE TABLE ml_trades (pnl REAL)")
    con.execute("CREATE TABLE ml_state (key TEXT, value TEXT)")
    con.executemany("INSERT INTO ml_trades VALUES (?)", [(6.0,), (-2.0,)])
    con.execute("INSERT INTO ml_state VALUES ('peak_balance', '106.0')")
    con.commit()
    con.close()
    monkeypatch.setattr(settings, 'DATA_DIR', tmp_path)
    enviados = []
    monkeypatch.setattr(ml_bot, 'send_alert', lambda t, tag=None: enviados.append((tag, t)))

    bot = ml_bot.MLBot.__new__(ml_bot.MLBot)
    bot.exchange_public = Ticker()
    bot._cmd_status('agresivo')

    tag, txt = enviados[0]
    assert tag == 'AGRESIVO'
    assert 'Balance: $104.00' in txt and 'abierto $-5.00' in txt   # short 0,5 de 100 a 110
    assert 'Trades: 2 | WR 50% | PnL $+4.00' in txt
    assert 'DD: 1.9%' in txt and 'SOL SHORT' in txt
