"""Experto de noticias — SOLO GRABA, no opera (experiments/sentimiento/README.md).

Cada CICLO_S lee titulares por RSS, guarda los nuevos y los clasifica con el
LLM local de Ollama (sin API keys). Un LLM no se puede validar con noticias
pasadas: sabe lo que paso despues. Por eso se graba hacia adelante y se evalua
dentro de 6-12 meses contra los trades y el precio.

Sin look-ahead en la evaluacion: la hora que cuenta es `fetched_at` (cuando el
bot pudo verlo), no `published`.

Proceso aparte del bot, como record_stream.py. Uso:
    python scripts/record_news.py            # bucle
    python scripts/record_news.py --once     # un ciclo y sale
"""
import hashlib
import html
import json
import logging
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import requests

DB = Path(__file__).resolve().parents[1] / 'data_live' / 'news.db'
OLLAMA = 'http://localhost:11434/api/generate'
MODELO = 'qwen3:14b'
PROMPT_V = 1                 # subir si cambia el prompt: las filas viejas quedan marcadas
CICLO_S = 1800
MAX_CLASIF_CICLO = 150       # ~7 s/titular en CPU
UA = {'User-Agent': 'Mozilla/5.0 (bot_trader research)'}
FEEDS = {
    'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
    'cointelegraph': 'https://cointelegraph.com/rss',
    'decrypt': 'https://decrypt.co/feed',
    'bitcoinmagazine': 'https://bitcoinmagazine.com/feed',
    'theblock': 'https://www.theblock.co/rss.xml',
    'fed': 'https://www.federalreserve.gov/feeds/press_all.xml',
    'sec': 'https://www.sec.gov/news/pressreleases.rss',
    'gnews_btc': 'https://news.google.com/rss/search?q=bitcoin+when:1d&hl=en-US&gl=US&ceid=US:en',
}
TEMAS = ('regulacion', 'macro', 'hack', 'institucional', 'mercado', 'tecnologia', 'otro')
PROMPT = """Eres un analista de riesgo de un bot que opera futuros de BTC en velas de 4h.
Titular ({fuente}): {titulo}
Resumen: {resumen}

Evalua el efecto probable en el precio de BTC en las proximas 24 horas.
Responde SOLO JSON:
{{"relevante": true|false, "sesgo": -2..2, "riesgo": 0..3, "tema": "{temas}"}}
sesgo: -2 muy bajista, 0 neutro, 2 muy alcista.
riesgo: probabilidad de un movimiento brusco (0 ninguna, 3 evento mayor: hack grande, decision de la Fed, prohibicion).
Si la noticia no afecta a BTC, relevante=false, sesgo=0, riesgo=0."""

log = logging.getLogger('record_news')


def ahora():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def limpiar(s, n=400):
    s = html.unescape(re.sub(r'<[^>]+>', ' ', s or ''))
    return re.sub(r'\s+', ' ', s).strip()[:n]


def parsear(xml_bytes):
    """RSS 2.0 y Atom -> [(titulo, link, publicado, resumen)]."""
    root = ET.fromstring(xml_bytes)
    out = []
    for it in root.iter():
        tag = it.tag.rsplit('}', 1)[-1]
        if tag not in ('item', 'entry'):
            continue
        campos = {c.tag.rsplit('}', 1)[-1]: c for c in it}
        link = campos.get('link')
        href = (link.get('href') or link.text) if link is not None else ''
        txt = lambda k: (campos[k].text or '') if k in campos else ''
        titulo = limpiar(txt('title'), 300)
        if titulo:
            out.append((titulo, (href or '').strip(),
                        txt('pubDate') or txt('published') or txt('updated'),
                        limpiar(txt('description') or txt('summary'))))
    return out


def validar(raw):
    """JSON del LLM -> dict con rangos forzados, o None si no se puede leer."""
    try:
        d = json.loads(raw)
        tema = d.get('tema') if d.get('tema') in TEMAS else 'otro'
        return {'relevante': bool(d.get('relevante', False)),
                'sesgo': max(-2, min(2, int(round(float(d['sesgo']))))),
                'riesgo': max(0, min(3, int(round(float(d['riesgo']))))),
                'tema': tema}
    except (ValueError, KeyError, TypeError):
        return None


def db():
    DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS news (
        id TEXT PRIMARY KEY, fuente TEXT, titulo TEXT, resumen TEXT, link TEXT,
        published TEXT, fetched_at TEXT,
        relevante INTEGER, sesgo INTEGER, riesgo INTEGER, tema TEXT,
        modelo TEXT, prompt_v INTEGER, classified_at TEXT, raw TEXT)""")
    return con


def recoger(con):
    nuevos = 0
    for fuente, url in FEEDS.items():
        try:
            r = requests.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            items = parsear(r.content)
        except Exception as e:
            log.warning(f"{fuente}: {e}")
            continue
        for titulo, link, pub, resumen in items:
            # id por titulo normalizado: Google News repite la misma noticia con otro link
            id_ = hashlib.sha1(titulo.lower().encode()).hexdigest()
            cur = con.execute("INSERT OR IGNORE INTO news (id, fuente, titulo, resumen, link,"
                              " published, fetched_at) VALUES (?,?,?,?,?,?,?)",
                              (id_, fuente, titulo, resumen, link, pub, ahora()))
            nuevos += cur.rowcount
    con.commit()
    return nuevos


def clasificar(con):
    filas = con.execute("SELECT id, fuente, titulo, resumen FROM news WHERE classified_at IS NULL"
                        " ORDER BY fetched_at DESC LIMIT ?", (MAX_CLASIF_CICLO,)).fetchall()
    for id_, fuente, titulo, resumen in filas:
        try:
            r = requests.post(OLLAMA, timeout=300, json={
                'model': MODELO, 'stream': False, 'think': False, 'format': 'json',
                'options': {'temperature': 0}, 'keep_alive': '10m',
                'prompt': PROMPT.format(fuente=fuente, titulo=titulo, resumen=resumen or '-',
                                        temas='|'.join(TEMAS))})
            r.raise_for_status()
            raw = r.json().get('response', '')
        except Exception as e:
            log.warning(f"ollama: {e}")
            return                                   # reintenta el ciclo siguiente
        v = validar(raw) or {'relevante': None, 'sesgo': None, 'riesgo': None, 'tema': None}
        con.execute("UPDATE news SET relevante=?, sesgo=?, riesgo=?, tema=?, modelo=?,"
                    " prompt_v=?, classified_at=?, raw=? WHERE id=?",
                    (v['relevante'], v['sesgo'], v['riesgo'], v['tema'], MODELO,
                     PROMPT_V, ahora(), raw[:500], id_))
        con.commit()
    return len(filas)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    con = db()
    while True:
        t0 = time.time()
        n = recoger(con)
        c = clasificar(con)
        log.info(f"ciclo: {n} titulares nuevos, {c} clasificados ({time.time() - t0:.0f}s)")
        if '--once' in sys.argv:
            return
        time.sleep(max(60, CICLO_S - (time.time() - t0)))


if __name__ == '__main__':
    main()
