# E12: la alerta de caídas de E7 como veto de entradas de V2

> **Pre-registro escrito el 2026-10-20, ANTES de correr nada.** Viene de E7
> (`experiments/especialistas/`), que pasó. Es el uso natural de una alerta de
> caídas: no abrir largos cuando la alerta está alta. **α = 0,05.** Cambia
> qué trades existen, así que se exige null sintético (`CLAUDE.md`).

## Alerta

- **Modelo:** el de E7 que pasó, `base + macro`, con las mismas features,
  etiqueta y walk-forward anual. Da una p̂ de caída por día UTC, fuera de
  muestra desde 2021.
- **Umbral:** el cuantil 0,80 de las p̂ **del propio train** de ese año, es
  decir, el 20% de días más alarmantes según el pasado.
- **Alerta del día `d`:** `p̂_d ≥ umbral`.

## Veto en V2

- V2 de referencia: `portfolio_sim`, BTC, 4,5%, `max_notional_pct=2,0`,
  costes nuevos.
- **Ventana:** 2021-01-01 → fin del parquet, porque la alerta solo existe fuera
  de muestra desde 2021.
- Una señal que se llena al open de la vela `ts` usa la alerta del último día
  UTC completo antes de `ts`, es decir, `ts.normalize() − 1 día`.
- **Si la alerta está activa, la señal se descarta.** No ocupa hueco y la
  siguiente señal puede entrar.
  - Implementación: el hook `sizing` de `PortfolioSim` devuelve 0, y el
    simulador lo trata como rechazo `'veto'`.
  - Se añaden 2 líneas al simulador. Sin veto, V2 da exactamente sus cifras
    (se comprueba con un `assert`).

## Hipótesis H12 (deben cumplirse las dos)

1. **CAGR/DD(V2 + veto) > CAGR/DD(V2 a igual exposición media)**. El riesgo de
   V2 se escala por el ratio de exposiciones y se itera 2 veces. Vetar baja la
   exposición, y eso sola no cuenta como mejora.
2. **Null sintético:** 200 series `sintetica()` de BTC con bloques de 180 velas.
   El macro se queda en sus fechas reales y se recalcula todo:
   - las features de E7 y su etiqueta;
   - el walk-forward y el umbral;
   - V2 con y sin veto, al 4,5% los dos.

   Estadístico: `Δ = CAGR/DD(veto) − CAGR/DD(sin veto)`.
   `p = P(Δ_sint ≥ Δ_obs)`. **p < 0,05.**

**Descriptivo:**
- fracción de días con alerta y número de trades vetados;
- media de r, WR y PF de los trades vetados (corriendo V2 sin veto y marcando
  los que habrían caído en alerta);
- bootstrap por bloques de 10 trades sobre `mean(r) < 0` de los vetados.

## Riesgos conocidos antes de empezar

- V2 tiene unos 115 trades en 2021-2026 y el veto tocará unos 20. La potencia
  es baja, así que el resultado más probable es que no haya conclusión.
- E7 no ayudó en 2021-2022, cuando estuvieron las caídas grandes.
- V2 ya está fuera en los bajistas (filtro EMA50/200 diario): parte de lo que
  avisa la alerta, V2 ya lo esquiva.

---

## Resultados (2026-10-20): `test_veto.py`, salida en `salida.txt`

La alerta está activa el **18,6%** de los días desde 2021. De los 115 trades de V2
en 2021-2026, **21** habrían entrado en día de alerta. Con veto quedan 95 trades;
se rechazaron 37 señales, porque al vetar una pueden aparecer otras que también
caen en alerta.

| trades de V2 sin veto | n | media r | WR | PF |
|---|--:|--:|--:|--:|
| **en día de alerta** | 21 | **−0,84%** | 28,6% | 0,57 |
| resto | 94 | +0,74% | 43,6% | |

| brazo (2021-01 → 2026-02) | CAGR | DD | CAGR/DD | exposición |
|---|--:|--:|--:|--:|
| V2 al 4,5% | +7,2% | 39,9% | 0,18 | 0,177 |
| **V2 + veto al 4,5%** | **+11,7%** | **28,4%** | **0,41** | 0,161 |
| V2 al 4,02% (igual exposición) | +7,0% | 37,2% | 0,19 | 0,162 |

- **(1) Cumple:** 0,41 contra 0,19 a igual exposición.
- **(2) NO cumple:** en las 200 series sintéticas el Δ CAGR/DD tiene mediana
  −0,022 y **p90 +0,239**. El observado (+0,229) queda en **p = 0,105**.
- El "p = 0,000" de la media de los vetados no sirve: con 21 trades y bloques de
  10 solo hay 3 bloques, así que el bootstrap no tiene resolución. Es
  descriptivo.

**H12: NO pasa.**

### Lectura

1. **La mejora en la historia es grande y tiene lógica.** Los trades en día de
   alerta pierden (WR 29%, PF 0,57), y quitarlos sube el CAGR/DD de 0,18 a 0,41.
   Recordatorio de E11: la alerta es en realidad "rebote dentro de una bajada"
   del propio BTC, no el oro.
2. **No se distingue del azar.** Vetar el 19% de los días mueve tanto el
   resultado que, en series de BTC con otro orden de regímenes, un veto
   construido igual da +0,24 o más en 1 de cada 10 series. Con 21 trades
   vetados no hay potencia, como ya avisaba el pre-registro.
3. **No se conecta al bot.** Si se quisiera seguir esta pista, la vía honesta es
   **grabarla en vivo**: registrar la alerta en cada señal de V2 sin vetar nada
   y comparar dentro de 1-2 años. Eso no cambia el sistema.
