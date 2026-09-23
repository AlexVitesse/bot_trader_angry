"""Grabador de streams publicos de Binance Futures (capa B de
docs/GRABACION_DATOS_VIVO.md). Proceso aparte del bot: si muere, el bot no se
entera, y al reves.

Escribe CSV gzip diario (UTC) en data_live/<stream>/<YYYY-MM-DD>.csv.gz,
volcando cada FLUSH_S segundos (un crash pierde como mucho eso).

Uso: python scripts/record_stream.py
"""
import asyncio
import csv
import gzip
import io
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import websockets

OUT = Path(__file__).resolve().parents[1] / 'data_live'
FLUSH_S = 60
# stream -> segundos entre muestras (0 = todos los mensajes). depth20@500ms y
# markPrice@1s (cada 10 s) se anaden cuando se haya visto una semana de disco.
STREAMS = {
    'btcusdt@forceOrder': 0,
    'btcusdt@bookTicker': 1,
}
URL = 'wss://fstream.binance.com/stream?streams=' + '/'.join(STREAMS)

log = logging.getLogger('record_stream')


def fila(stream: str, d: dict) -> dict:
    d = d.get('o', d)                       # forceOrder trae la orden en 'o'
    return {'recv_ms': int(time.time() * 1000),
            **{k: v for k, v in d.items() if not isinstance(v, (dict, list))}}


class Grabador:
    def __init__(self):
        self.buf = {s: [] for s in STREAMS}
        self.last = {}                      # stream muestreado -> ultima fila

    def recibir(self, stream: str, data: dict):
        r = fila(stream, data)
        if STREAMS[stream]:
            self.last[stream] = r
        else:
            self.buf[stream].append(r)

    async def muestrear(self, stream: str, cada: float):
        while True:
            await asyncio.sleep(cada)
            r = self.last.pop(stream, None)
            if r:
                self.buf[stream].append(r)

    def volcar(self):
        for stream, rows in self.buf.items():
            if not rows:
                continue
            self.buf[stream] = []
            for dia, grupo in _por_dia(rows).items():
                path = OUT / stream.replace('@', '_') / f'{dia}.csv.gz'
                path.parent.mkdir(parents=True, exist_ok=True)
                nuevo = not path.exists()
                cols = list(grupo[0])
                s = io.StringIO()
                w = csv.DictWriter(s, cols, extrasaction='ignore')
                if nuevo:
                    w.writeheader()
                w.writerows(grupo)
                with gzip.open(path, 'at', newline='') as f:   # miembro gzip nuevo
                    f.write(s.getvalue())

    async def volcar_periodico(self):
        while True:
            await asyncio.sleep(FLUSH_S)
            try:
                self.volcar()
            except Exception as e:
                log.error(f'volcado fallido: {e}')


def _por_dia(rows):
    out = {}
    for r in rows:
        dia = datetime.fromtimestamp(r['recv_ms'] / 1000, timezone.utc).strftime('%Y-%m-%d')
        out.setdefault(dia, []).append(r)
    return out


async def main():
    g = Grabador()
    tareas = [asyncio.create_task(g.volcar_periodico())]
    tareas += [asyncio.create_task(g.muestrear(s, c)) for s, c in STREAMS.items() if c]
    espera = 5
    while True:
        try:
            # Binance cierra el socket cada 24 h; se reabre y se sigue.
            async with websockets.connect(URL, ping_interval=60, max_size=2**22) as ws:
                log.info(f'conectado: {", ".join(STREAMS)}')
                espera = 5
                async for msg in ws:
                    m = json.loads(msg)
                    g.recibir(m['stream'], m['data'])
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning(f'socket caido ({e}); reintento en {espera}s')
            await asyncio.sleep(espera)
            espera = min(espera * 2, 60)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
