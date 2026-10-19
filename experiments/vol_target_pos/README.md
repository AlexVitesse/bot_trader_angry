# E2 — Volatilidad objetivo con la posición abierta

> **Pre-registro escrito el 2026-10-19, ANTES de correr nada.** Viene de
> `docs/PLAN_GITHUB_2026-10.md` (E2), con las precisiones de implementación.
> Familia de 5 tests → **α = 0,01**. Las entradas y las salidas son las de V2:
> solo cambia el tamaño durante el trade.

> **Aviso previo (E1, `benchmark_bh/`):** en BTC, la volatilidad objetivo
> empeoró al B&H (CAGR/DD 0,28 contra 0,39), porque la vol alta coincide con
> los rallies. El test se corre igual, porque está en el plan.

## Regla (única)

En cada vela 4h con la posición abierta (después del stop y del trailing):

`objetivo = notional_entrada × min(1, σ_entrada / σ̂_t)`

- **σ̂**: suma diaria de la varianza Garman-Klass de las velas 4h, media de 20
  días, anualizada. Se usa el valor del **día anterior** (sin look-ahead).
- **σ_entrada** = σ̂ en la vela de entrada.
- **Solo reduce, nunca añade.** Se ejecuta si `objetivo < notional_actual × 0,8`
  (cambio > 20%).
- La reducción se llena al **close** de la vela, cobrando comisión de entrada
  y de salida sobre la parte cerrada (0,06% × 2, como el resto del simulador).
  La decisión usa información del día anterior, así que llenar al close solo
  retrasa la ejecución: no hay look-ahead.
- Implementada como hook opcional `ajuste` en `PortfolioSim`. Sin el hook, el
  simulador tiene que dar exactamente las cifras de V2 (n=165, CAGR +23,5%,
  DD 42,4% con `max_notional_pct=2,0`). El script lo comprueba con `assert`.

## Comparación justa

**V2-m**: V2 a riesgo fijo con `risk_pct` escalado para igualar la
**exposición media** (notional/equity promediado en todas las velas, con 0
fuera de posición) de E2. Se ajusta una vez por proporción, se itera una
segunda vez y se reporta la exposición alcanzada.

## Hipótesis y test

**H2:** CAGR/DD de E2 > CAGR/DD de V2-m.
Bootstrap pareado por bloques circulares de calendario (30 días) sobre los
retornos diarios conjuntos de los dos brazos, 20.000 réplicas.
`p = P(Δ ≤ 0)`, con `Δ = CAGR/DD(E2) − CAGR/DD(V2-m)`. **Pasa si p < 0,01.**
Los trades son los mismos, así que no hace falta el null sintético (como en
`vol_sizing/`).

---

## Resultados (2026-10-19) — `test_vol_target_pos.py`, salida en `salida.txt`

**La regla no se activa nunca: 0 reducciones en 165 trades.** E2 es idéntico a
V2 (CAGR +23,5%, DD 42,4%). **H2 no pasa** (Δ = 0, p = 1).

No es un bug, lo comprobé aparte. Dentro de cada trade, el cociente
`σ̂_max / σ_entrada` tiene mediana **1,00**, p90 **1,04** y máximo **1,24**. Para
recortar hace falta llegar a 1,25 (umbral del 20%). Una vol de 20 días con un
día de retraso apenas se mueve durante un trade de V2, que dura unos 2 días de
mediana (9 velas).

### Lectura

- **La idea no aplica a V2.** Funciona en libros de tendencia que mantienen
  posiciones semanas o meses (OctopusTakopi). V2 sale antes de que la
  volatilidad lenta cambie. El sizing por ATR de entrada ya hace todo lo que
  esta regla podría hacer.
- **No se ajustan los parámetros** (ventana corta, umbral menor): eso sería una
  búsqueda nueva. Además, E1 mostró que en BTC recortar con vol alta resta
  retorno, porque la vol alta coincide con los rallies.
- El hook `ajuste` queda en `PortfolioSim`. Sin hook, las cifras de V2 son
  idénticas (lo comprueba el `assert` del script).
