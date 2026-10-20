"""
Pata BTC spot en PAPER: la mezcla de experiments/mixta_v2_btc/ (E6) seguida en
vivo, sin mandar ninguna orden.

Mezcla = 50% V2 (el balance real de la cuenta demo) + 50% en BTC spot con
exposición fija 0,176, rebalanceada a diario con coste 0,06% sobre el volumen.
Las dos mitades se igualan el primer día de cada mes. NO validada (H6 no pasó,
p=0,18): esto mide en vivo lo que el backtest predice, nada más.

Un paso por día (resumen diario de V2). Estado en data/pata_btc_state.json.
ponytail: la mitad V2 usa el balance realizado, no el no realizado; con trades
de ~2 días la diferencia se cierra sola.
"""
import json
from pathlib import Path

W_BTC = 0.176
COSTE = 0.0006


def paso(s: dict | None, v2_eq: float, px: float, fecha: str) -> dict:
    """fecha 'YYYY-MM-DD'. Devuelve el estado nuevo (s no se modifica)."""
    if s is None:
        b = v2_eq / 2
        return {'inicio': fecha, 'fecha': fecha, 'eq0': v2_eq, 'v2': v2_eq, 'px': px,
                'a': v2_eq / 2, 'b': b - W_BTC * b * COSTE, 'pico': v2_eq}
    if fecha == s['fecha']:
        return s
    s = dict(s)
    if fecha[:7] != s['fecha'][:7]:                     # primer día del mes
        s['a'] = s['b'] = (s['a'] + s['b']) / 2
    r_v2 = v2_eq / s['v2'] - 1 if s['v2'] > 0 else 0.0
    r = px / s['px'] - 1
    s['a'] *= 1 + r_v2
    b = s['b'] * (1 + W_BTC * r)
    w_deriva = W_BTC * (1 + r) / (1 + W_BTC * r)
    s['b'] = b - abs(w_deriva - W_BTC) * b * COSTE      # vuelta a 0,176
    s.update(fecha=fecha, v2=v2_eq, px=px)
    s['pico'] = max(s['pico'], s['a'] + s['b'])
    return s


def resumen(s: dict) -> str:
    mix = s['a'] + s['b']
    return (f"🧪 Mezcla V2+BTC (paper, desde {s['inicio']}): ${mix:,.2f} "
            f"({mix / s['eq0'] - 1:+.1%}) | V2 solo {s['v2'] / s['eq0'] - 1:+.1%} | "
            f"DD {1 - mix / s['pico']:.1%}")


def cargar(f: Path) -> dict | None:
    try:
        return json.loads(Path(f).read_text())
    except (OSError, ValueError):
        return None


def guardar(f: Path, s: dict):
    Path(f).write_text(json.dumps(s))
