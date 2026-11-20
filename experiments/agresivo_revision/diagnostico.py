"""Diagnostico: el agresivo pierde por falta de edge o por costes? (descriptivo)

Candidatos del motor (ag.eventos_panel) desde 2021, sin modelo. r_R neto es la
etiqueta; r_R bruto le devuelve comision+slippage (2*COM) y funding.
Uso: python experiments/agresivo_revision/diagnostico.py
"""
from datos import cargar
from backtest import ag

datos = cargar()
feats = {p: ag.features(d, datos['BTC/USDT']) for p, d in datos.items()}
ev = ag.eventos_panel(feats)
ev = ev[ev.t >= '2021-01-01'].copy()
barras = ((ev.fin - ev.ini) / ag.BAR).astype(int)
ev['coste_R'] = 2 * ag.COM / ev.trail
ev['fund_R'] = ev['dir'] * ag.FUND_8H * 0.5 * barras / ev.trail
ev['bruto_R'] = ev.r_R + ev.coste_R + ev.fund_R
ev['lado'] = ev['dir'].map({1: 'LONG', -1: 'SHORT'})
print(f'candidatos {len(ev)} | trail medio {ev.trail.mean():.2%} | duracion media {barras.mean():.1f} velas')
cols = ['r_R', 'bruto_R', 'coste_R', 'fund_R']
print(ev.groupby('lado')[cols].mean().round(4).to_string())
print('todos:', ev[cols].mean().round(4).to_dict())
print('\npor tipo (bruto / neto):')
print(ev.groupby(['lado', 'tipo'])[['bruto_R', 'r_R']].mean().round(3).to_string())
# arrastre de costes de la cartera tal como opera (salida.txt: 381 trades/ano)
for riesgo in (0.01, 0.035):
    print(f'riesgo {riesgo:.1%}: notional medio {riesgo / ev.trail.mean():.2f}x equity; '
          f'coste por trade {riesgo * ev.coste_R.mean():.3%} del equity; '
          f'x381 trades/ano = {381 * riesgo * ev.coste_R.mean():.1%}/ano')
# cuenta de 100 USD: notional por operacion contra minimos de Binance (2026-11-20)
MIN = {'BTC/USDT': 62.0, 'ETH/USDT': 20.0, 'LINK/USDT': 20.0}      # 0,001 BTC = 62 USD
u = ev[ev.t >= '2025-01-01']
for riesgo in (0.01, 0.02, 0.035, 0.045):
    notional = 100 * riesgo / u.trail
    bajo = notional < u.par.map(MIN).fillna(5.0)
    print(f'100 USD, riesgo {riesgo:.1%}: notional mediano {notional.median():.0f} USD; bajo el minimo: '
          f'{bajo.mean():.0%} de candidatos (BTC {bajo[u.par == "BTC/USDT"].mean():.0%})')
