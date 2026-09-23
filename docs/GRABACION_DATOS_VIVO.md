# Grabación de datos propios en vivo

> 2026-09-23. Origen: `docs/FUENTES_DE_DATOS_2026-09.md` §6.
>
> **Estado (2026-09-23): implementadas las tres fases del orden de abajo.**
> - Capa A: `PortfolioManager._snapshot` + tabla `ml_exec_snapshots`
>   (eventos `signal`, `entry_fill`, `exit_fill`; en las salidas `order_type`
>   guarda el motivo: `TRAIL`, `SL`, `TIMEOUT`, `EXCHANGE_SL`). Se une con
>   `ml_trades` por `(symbol, entry_time)` en vez de `trade_id`, porque el
>   trade no existe en `ml_trades` hasta que cierra. El libro se pide con 100
>   niveles, no 20: con 20 no se llega ni al 0,05% en BTC. Test:
>   `test_exec_snapshots_never_block_orders`.
> - Capa B: `scripts/record_stream.py` con `forceOrder` y `bookTicker`
>   (1/s). Arranca desde el cron `@reboot` de `deploy/setup_server.sh`
>   dentro de un bucle que lo relanza. Escribe en `data_live/` (ignorado por
>   git).
> - `compare_live_vs_sim.py` reporta slippage por evento y tipo de orden.

## Por qué

- Lo único con margen real que queda en el proyecto es la **ejecución**
  (slippage, fills, coste del maker frente al market) y hoy no se mide nada de
  eso: `compare_live_vs_sim.py` compara precio de salida simulado contra PnL
  real, sin saber cómo era el libro cuando se entró.
- Las fuentes de microestructura que Binance publica (`bookDepth`,
  `bookTicker`) empiezan en 2023 y las liquidaciones no las publica. Grabarlas
  nosotros cuesta cero y en 12 meses hay un dataset que nadie más tiene para
  este sistema.
- Coincide con los 6-12 meses de paper trade ya planificados. Si no se graba
  ahora, en 2027 se estará en el mismo punto.

## Qué se graba

Dos capas independientes. La **A** es la que importa y es la más pequeña; la
**B** es opcional y se puede apagar sin tocar el bot.

### Capa A: foto en cada señal y cada fill (dentro del bot)

Una fila por evento en una tabla nueva `ml_exec_snapshots` de la misma
SQLite del bot, tomada por REST justo antes de mandar la orden y justo
después de cada fill (entrada, stop, trail, cierre manual).

| campo | fuente REST (fapi) | para qué |
|---|---|---|
| `ts`, `symbol`, `event` (`signal`, `entry_fill`, `exit_fill`), `trade_id` | bot | unir con `ml_trades` |
| `best_bid`, `best_ask`, `spread_bps` | `fetch_order_book(pair, 20)` (ya se llama en `_enter`) | coste real del cruce |
| `depth_bid_1pct`, `depth_ask_1pct` (notional acumulado a ±1%) | mismo libro, 20 niveles | si nuestro tamaño mueve el precio |
| `mark`, `index`, `premium` | `fapiPublicGetPremiumIndex` | base en el instante |
| `funding_rate`, `next_funding_ts` | mismo endpoint | funding que va a pagar el trade |
| `open_interest` | `fapiPublicGetOpenInterest` | continuidad con `derivados/` |
| `taker_ratio_5m` | `fapiDataGetTakerlongshortRatio` (period 5m, limit 1) | continuidad con `derivados_ratios/` |
| `order_type` (`maker`, `market`, `mixed`), `filled_qty`, `avg_price`, `ref_price` | respuesta de la orden (ya está en `_enter`) | slippage = `avg_price / ref_price − 1` |
| `latency_ms` | reloj del bot entre señal y ack | diagnóstico |

Coste: 3 llamadas REST más por evento, ~25 eventos al año. Cero riesgo para
el peso de la API. Si una llamada falla, el campo queda NULL y la orden sigue:
**la grabación nunca bloquea una entrada ni una salida.**

Punto de enganche: `PortfolioManager._enter` (ya tiene el libro y la orden) y
`close_position` / `reconcile_closed_trades` para los fills de salida. Una
función `_snapshot(event, pair, trade_id, order=None)` de ~40 líneas y la
tabla. Test: uno en `tests/test_pm_money_path.py` con `FakeExchange` que
compruebe que una excepción en el snapshot no impide el fill.

### Capa B: flujo continuo (proceso aparte en el VPS)

`scripts/record_stream.py`, proceso independiente lanzado por el mismo cron
`@reboot`, con `websockets` (ya está en `requirements-vps.txt`). Escribe CSV
gzip diario en `data_live/<stream>/<YYYY-MM-DD>.csv.gz`. Si muere, el bot no
se entera; si el bot muere, el recorder sigue.

| stream (BTCUSDT) | muestreo | filas/día | tamaño/año (gz) |
|---|---|--:|--:|
| `btcusdt@forceOrder` (liquidaciones) | todas | 200-5.000 | < 30 MB |
| `btcusdt@bookTicker` | 1 por segundo (última) | 86.400 | ~180 MB |
| `btcusdt@depth20@500ms` | 1 cada 10 s (última) | 8.640 | ~250 MB |
| `btcusdt@markPrice@1s` (mark, index, funding) | 1 cada 10 s | 8.640 | ~40 MB |

Total ~500 MB al año. Comprobar `df -h` en el VPS antes de arrancar y rotar
con `find data_live -mtime +400 -delete` si hace falta. Sin sftp en el VPS:
se copia con `tar` por ssh, igual que los logs.

Reconexión: bucle `while True` con backoff 5-60 s. Binance cierra el socket
cada 24 h; se reabre y se sigue. Sin estado en memoria más allá del último
mensaje.

## Qué pregunta responde cada cosa (y cuándo)

| pregunta | datos | cuándo hay respuesta |
|---|---|---|
| ¿Cuánto cuesta de verdad entrar y salir? (slippage real vs 0,02% asumido) | Capa A | a 30 trades reales (~1 año) |
| ¿El maker post-only con 60 s de timeout llena, o casi siempre acaba en market? | Capa A (`order_type`) | a 20 entradas |
| ¿El stop con gap del simulador se parece al fill real del stop? | Capa A `exit_fill` vs `ml_trades.exit_sim` | a 30 salidas |
| ¿Nuestro tamaño al 4,5% de riesgo mueve el libro? | Capa A `depth_*_1pct` vs notional | inmediato, por trade |
| ¿Las cascadas de liquidación coinciden con nuestros stops? | Capa B `forceOrder` + `ml_trades` | 12 meses, y solo descriptivo |
| ¿Hay señal en el libro a 4h? | Capa B `depth20` | **no antes de 2-3 años**; no es el objetivo |

La última fila es honesta a propósito: la capa B no va a producir una señal
de trading en el horizonte del proyecto. Su valor es tener el dato cuando
alguien pregunte, en vez de tener que comprarlo.

## Lo que NO se hace

- No se graba nada de los 21 pares que no se operan. Solo BTCUSDT.
- No se mete el recorder dentro del proceso del bot. Un socket colgado no
  puede tumbar el trading.
- No se construye base de datos de series temporales, ni dashboard, ni API.
  CSV gzip por día y SQLite son suficientes para el volumen.
- No se usa nada de esto como feature de V2 hasta que pase el mismo protocolo
  de `derivados/` (etapa 1 a nivel de mercado, etapa 2 sobre trades).

## Orden de implementación propuesto

1. **Capa A** (una tarde, un commit, un test). Es la que alimenta el KPI de
   parada del paper trade (real vs simulado por trade).
2. **Capa B** (una tarde). Solo `forceOrder` y `bookTicker` al principio;
   `depth20` y `markPrice` cuando se haya visto el consumo de disco real de
   una semana.
3. Ampliar `experiments/ejecucion_vivo/compare_live_vs_sim.py` para leer
   `ml_exec_snapshots` y reportar slippage medio por lado y tipo de orden.
