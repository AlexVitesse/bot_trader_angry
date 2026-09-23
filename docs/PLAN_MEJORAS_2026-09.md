# Plan de mejoras 2026-09 — derivado de `docs/AUDITORIA_2026-09.md`

> Regla del plan: **ningún punto busca más retorno**. Todos cierran la
> distancia entre lo que se simuló y lo que corre, o quitan riesgo operativo.
> Cada tarea lleva archivo, cambio, criterio de aceptación y prueba mínima.
> Referencias `§n.m` apuntan a la auditoría.

**Fuera de alcance por decisión del usuario (2026-09-22):** §3.1 autorización
de `chat_id` en Telegram y §3.2 rotación del token. Quedan registrados en la
auditoría; no se planifican aquí.

## Estado (2026-09-23) — desplegado en el VPS

> Registro completo de la ejecución: `docs/SESION_2026-09-23.md`.

| Fase | Estado | Commit |
|---|---|---|
| 0 | Hecha | `7c60de8` |
| 1.1–1.6 | Hechas. 1.1 opción **(a)**: pausa por racha eliminada | `50bbcb4` |
| 2.1 + 5.2 | Hechas: `reconcile_closed_trades` rellena `pnl_real` (income) y `exit_sim_*` por trade; `experiments/ejecucion_vivo/compare_live_vs_sim.py` | `de60d90` |
| 2.2 | Hecha: `experiments/ejecucion_vivo/` | `af547eb` |
| 2.3 | Hecha, opción **(a)**: trail por vela 4h cerrada | `50bbcb4` |
| 3.1–3.4 | Hechas (`CLAUDE.md`, nota en SESION_2026-08-22, METODOLOGIA_TESTING archivada, portfolio_sim README) | ver log |
| 4 | Hecha: `experiments/bootstrap_bloques/`. **p por bloques 0,004–0,033; contra null sintético 0,13–0,20 (0,33–0,68 con selección)** | `e64b837` |
| 5.1 | Hecha **con cambio**: el VPS no tiene sudo ni systemd de usuario → cron `@reboot` + `run_bot.sh` (`deploy/setup_server.sh`) | `0762409` |
| 5.3 | Hecha **con cambio**: `requirements-vps.txt` (pip freeze de prod; el VPS no usa poetry). Log en UTC. pytest en dev | `432ae0c`, `3bd8f4d` |
| 5.4 | No hecha: el yield queda **apagado** (decisión del usuario: aún no se puede validar) | — |
| 6.1–6.4 | Hechas | `877a7c2` |
| 6.5 | Parcial: borrado `ml_trades.db` (0 B). **No** se borraron duplicados (ambos nombres los usan ~35 scripts de `experiments/`) ni `bot_trades.db`, ni se refrescaron los parquets (cambiaría las cifras de todos los README; `oos_2026H1/` ya midió que no aportan información) | — |
| 7 | Hecha: 17 tests (`tests/`), pasan en local y en el VPS | varios |

**5.2, aceptación:** en #84-86 el REALIZED_PNL coincide exacto con la tabla de
`SESION_2026-09-22.md` (149,88 / 324,41 / 113,02). El `pnl_real` total sale
$4,93 más alto que los $563,11 de la sesión: faltan comisión de entrada y
funding porque la DB tiene `entry_time` posteriores a la apertura real en esos
trades (herencia del bug de fills). Trades nuevos no tienen ese problema.

**Despliegue 2026-09-23 06:29 UTC:** `git checkout -- src && git pull`
(los cambios locales del VPS eran idénticos a `b23ad0d`), backup
`data/ml_bot.db.bak-20260923`, bot relanzado sin posición abierta. Verificado:
un solo `run_bot.sh` + un solo python, `[BOT] Listo`, `Pos=0/1`, sin línea de
yield, motor V2 sin ML, log en UTC.

---

## Fase 0 — Un flag y un número (mismo día)

### 0.1 Desactivar el yield manager hasta validarlo aparte — §3.3

- **Archivo:** `config/settings.py:454`
- **Cambio:** `YIELD_MANAGER_ENABLED = False`. Dejar el bloque `YIELD_CONFIG`
  con un comentario: "reactivar solo tras la Fase 5".
- **Aceptación:** arranque del bot sin la línea `[BOT] Yield manager
  activado`; `/yield` responde "no esta activo".
- **Efecto colateral a vigilar:** en testnet el yield era simulado, así que el
  balance de futures no cambia. En mainnet evita transfers reales no probados.

### 0.2 Warm-up de la EMA200 diaria — §2.2

- **Archivo:** `src/ml_strategy_v15.py:410`
- **Cambio:** `_ohlcv(exchange, pair, '1d', 300)` → `1000`. Ajustar el umbral
  `len(ohlcv_1d) >= 200` a `>= 600` para que el fallback a derivación desde 4h
  no se active con datos a medias.
- **Aceptación:** el script de la tarea 2.2 da **0 desacuerdos** de `bull_1d`
  entre vivo y backtest sobre 2019-2026 (medido: 29 con 300 velas, 0 con 1000).
- **Nota:** Binance permite `limit` hasta 1500 en `klines`; una llamada.

### 0.3 Mensaje del kill switch — §5

- **Archivo:** `src/ml_bot.py:1371`
- **Cambio:** `>= 20%` → `>= {ML_MAX_DD_PCT:.0%}` (importar la constante).

---

## Fase 1 — Que vivo y simulado sean el mismo sistema (1-2 días)

### 1.1 Racha de pérdidas por signo de PnL, no por etiqueta — §2.3

- **Archivo:** `src/portfolio_manager.py:999-1005`
- **Cambio:** `if reason == 'SL'` → `if pnl < 0`. Aplicar la misma regla en
  `_handle_stale_position` (`:531-549`), que hoy no toca la racha.
- **Decisión pendiente (usuario):** ¿debe existir la pausa por 3 pérdidas?
  No está en el simulador. Con WR 43-45%, tres pérdidas seguidas ocurren en
  ~18% de las ventanas de tres trades. Opciones: (a) quitarla y dejar solo el
  límite diario y el kill switch; (b) mantenerla y **añadirla al simulador**
  (`portfolio_sim.py`) para que la expectativa la incluya. Recomendación: (a).
- **Test:** `tests/test_pm_streak.py` con un `PortfolioManager` sobre un
  exchange falso: tres cierres con `reason='SL'` y `pnl>0` **no** pausan; tres
  con `reason='TRAIL'` y `pnl<0` **sí** (si se elige (b)).

### 1.2 Persistir el estado de riesgo — §2.6

- **Archivo:** `src/portfolio_manager.py`
- **Cambio:** guardar `killed` y `consecutive_losses` en `ml_state` con
  `_save_state` cada vez que cambian (`:1000-1005`, `:1019`); leerlos en
  `sync_positions` (`:301-307`). Si `killed=='1'` al arrancar: log crítico,
  alerta y `sys.exit(0)` para que `run_bot.sh` no relance.
- **Aceptación:** matar el proceso con `killed=True` guardado → al relanzar
  sale sin operar. `/resume` sigue reseteando `consecutive_losses` y lo
  persiste.

### 1.3 Posición `pending` antes de la orden; adopción con parámetros del motor — §2.4

- **Archivo:** `src/portfolio_manager.py:660-770`
- **Cambio:**
  1. Antes de `create_order` (`:717`), `_save_position` con el `Position`
     completo (trail_mode, trail_fixed_dist, max_hold, tp/sl del motor) y un
     campo nuevo `status='pending'` (migración `ALTER TABLE ml_positions ADD
     COLUMN status TEXT DEFAULT 'open'`).
  2. Tras el fill, actualizar `entry_price`, `quantity`, `notional`,
     `status='open'`.
  3. En `sync_positions`, una fila `pending` con posición en exchange → adoptar
     **con los parámetros guardados**, no con `get_pair_tp_sl`. Una fila
     `pending` sin posición en exchange → borrar (la orden no entró).
  4. En `_reconcile_with_exchange:365-374` (mismatch >1%) conservar
     `trail_mode`, `trail_fixed_dist`, `max_hold` del `old_pos`.
  5. En el "DUPLICADO EVITADO" (`:689-698`) usar `trail_mode`,
     `trail_fixed_dist` y `max_hold` de la señal actual (ya llegan como
     argumentos) en vez de `get_pair_tp_sl`.
- **Aceptación:** test que simula excepción entre la orden y el save; al
  reconstruir, la posición tiene `trail_mode='tight'` y el `trail_fixed_dist`
  de la señal.

### 1.4 Reemplazo del stop con rollback — §2.5

- **Archivo:** `src/portfolio_manager.py:848-852`
- **Cambio:** colocar el stop nuevo **antes** de cancelar el viejo (dos stops
  `reduceOnly` durante un instante es inocuo); si la colocación falla, no
  cancelar y loguear. Si se elige mantener el orden actual, al fallar poner
  `pos.sl_order_id=None` y reintentar en el siguiente tick.
- **Aceptación:** test con `create_order` que lanza → `sl_order_id` sigue
  siendo el del stop vigente en exchange.

### 1.5 Funding en vivo — §2.7

- **Archivo:** `src/ml_strategy_v15.py:426`
- **Cambio:** pasar `df_funding` desde `/fapi/v1/fundingRate?limit=1000`
  (ya hay un fetch parecido en `_fetch_funding_zscore`). Construir DataFrame
  con índice UTC y columna `funding_rate`.
- **Aceptación:** log `[V2]` muestra `funding_z` distinto de 0.
- **Prioridad baja:** 6 trades en 6,5 años. Hacerlo por coherencia, no por
  retorno.

### 1.6 `/resetdb` — §2.9

- **Archivo:** `src/ml_bot.py` `_cmd_resetdb`
- **Cambio:** `UPDATE ml_state SET peak = balance WHERE id = 1` → dos
  `INSERT OR REPLACE INTO ml_state (key, value)` para `peak_balance` con el
  balance actual. O eliminar el comando.

---

## Fase 2 — Medir la ejecución en vivo (arranca ya, corre 6-12 meses)

Es el único sitio donde `CLAUDE.md` dice que queda margen, y la auditoría
muestra que la divergencia del trailing es **negativa**.

### 2.1 Registrar por trade el stop simulado vs el stop real — §2.1

- **Archivos:** `src/portfolio_manager.py`, tabla `ml_trades`
- **Cambio:** al cerrar un trade, guardar además: `exit_sim_price` y
  `exit_sim_reason` calculados con `_sim_long_trailing` sobre las velas 4h
  cerradas desde la entrada (fetch de `bars+2` velas); `exit_real_price` del
  `income`/`userTrades` de Binance (no del ticker); `fill_slippage` = fill
  real − close de la vela de señal.
- **Script:** `experiments/ejecucion_vivo/compare_live_vs_sim.py` que lee
  `ml_trades` y saca: PnL sim vs real acumulado, divergencia %, n trades.
- **KPI:** el que ya está en `CLAUDE.md` (divergencia >25% a 50 trades →
  STOP), pero medido con el sim **por trade**, no con la curva in-sample.

### 2.2 Dejar reproducibles las dos mediciones de la auditoría

- **Archivos nuevos:** `experiments/ejecucion_vivo/trail_granularity.py`
  (re-simula los trades V2 con trail por vela 1h vs 4h; medido: PF 2,06 →
  1,93, 13/82 trades cambian) y `experiments/ejecucion_vivo/regime_warmup.py`
  (compara `bull_1d` con 300/1000 velas vs historia completa; medido: 29 → 0
  desacuerdos). README con las tablas de la auditoría §2.1 y §2.2.

### 2.3 Decidir el modo del trailing en vivo — §2.1

Dos opciones, elegir **una** y dejarla documentada:

- **(a) Trailing por vela cerrada**, igual que el sim: actualizar `peak` y
  `trail_sl` solo cuando cierra la vela 4h (en `_on_new_candle`), dejando el
  stop del exchange como red de seguridad al nivel del trail. Ventaja: vivo ==
  sim por construcción. Coste: dentro de la vela el stop no se aprieta.
- **(b) Mantener el tick de 30 s** y **re-simular el backtest con esa regla**
  (trail sobre highs 1h/1m) para que la expectativa publicada refleje lo que
  corre. La medición de la auditoría dice que la expectativa baja.

Recomendación: **(a)**. Es un cambio pequeño en `_update_trailing` (guardar
`last_bar_ts` en `Position` y solo actualizar si cambió) y elimina la mayor
fuente conocida de divergencia.

---

## Fase 3 — Corregir la documentación de validación (medio día)

### 3.1 `CLAUDE.md`

- Retirar "V2 da p=0,004 estable en todos los niveles de riesgo" (§1.1).
  Sustituir por: "p≈0,004 con bootstrap i.i.d. sobre 165 trades in-sample;
  invariante al riesgo por construcción; no corregido por selección de
  variante ni por clustering de régimen".
- En "Requisitos de Validación" añadir: **bootstrap por bloques** (bloques de
  ≥30 días o por episodio de régimen) y **corrección por comparaciones
  múltiples** declarando cuántas variantes se probaron.
- Aclarar que los "2/6 folds" y los "WF 7/12, 8/12" históricos son
  particiones in-sample con parámetros fijos (§1.4).
- Fijar **una** definición de V2 de referencia: la de `portfolio_sim/` (fill
  al open, sin apalancamiento en F, 165 trades). Reescribir la tabla del
  "Candidato real" con esas cifras y borrar las demás (§1.6).

### 3.2 `docs/SESION_2026-08-22.md`

- Nota al pie en las líneas 72-74 y 85-88 remitiendo a `AUDITORIA_2026-09.md`
  §1.1 y §1.5.

### 3.3 `METODOLOGIA_TESTING.md`

- Mover a `docs/archive/` con una cabecera "V14, obsoleto: describe filtros
  post-hoc por modelo, prohibidos por CLAUDE.md". Enlazar desde `CLAUDE.md` la
  sección de validación como la metodología vigente.

### 3.4 `experiments/portfolio_sim/README.md`

- Corregir la tabla de p por nivel de riesgo: una sola fila. Añadir la
  aclaración de §1.1.

---

## Fase 4 — Bootstrap honesto (1-2 días de cómputo y escritura)

Objetivo: saber si el edge sobrevive a un test que respete la dependencia
temporal y la selección. **No** cambia parámetros.

### 4.1 Bootstrap por bloques — §1.2

- **Archivo nuevo:** `experiments/bootstrap_bloques/test_bloques.py`
- Sobre los trades de `portfolio_sim` (open-fill): stationary/circular block
  bootstrap con longitud de bloque = mediana de días entre trades × 5 y,
  alternativamente, bloques = episodios de régimen (`bull_1d` contiguo).
  Reportar p para cada longitud de bloque.
- **Null con clustering:** generar 1.000 series sintéticas **sin edge** con la
  misma autocorrelación de régimen (block-shuffle de las velas 4h) y correr el
  V2 completo sobre cada una. La fracción con `mean(r)>observado` es el p
  honesto. Esto cierra el hueco de §1.5.

### 4.2 Corrección por selección — §1.3

- Documentar en el mismo README cuántas variantes/parámetros se compararon
  antes de fijar `PARAMS_V2` (6 variantes en `combined_AF`, barrido de
  `agent_F/explore_params.py`, ablación F_SHORT). Aplicar Bonferroni-Holm o,
  mejor, incluir la selección **dentro** del null de 4.1 (elegir la mejor de 6
  variantes en cada serie sintética).

### 4.3 Costes realistas en el simulador — §1.7

- `portfolio_sim.py:200`: fill del stop = `min(stop, open_siguiente)` si hay
  gap por debajo del stop.
- `:47,196`: funding histórico real desde `data/btc_v15_funding.parquet`
  cuando cubra el periodo; constante solo como fallback.
- Slippage 0,02%/lado en vez de 0,01% hasta que 2.1 dé el número real.

**Resultado esperado:** un solo número p honesto y una sola tabla de
expectativa (CAGR, DD, PF a 2% y 4,5% de riesgo). Si p > 0,05, decirlo en
`CLAUDE.md` y seguir en paper trade igualmente: el paper trade es la evidencia
que falta, no el backtest.

---

## Fase 5 — Operación (1 día)

### 5.1 Supervisor real en el VPS — §3.4

- **Archivos:** `deploy/bot-trader.service`, `deploy/setup_server.sh`,
  `deploy/update.sh`
- **Cambio:** `ExecStart=/home/space-user2/envs/deepseek/bin/python -u -m
  src.ml_bot`, `WorkingDirectory=~/bot_trader_angry`, `Restart=on-failure`,
  `RestartSec=30`, **`SuccessExitStatus=0`** para que el kill switch (exit 0)
  no relance y el watchdog (exit 1) y `/restart` (exit 43) sí. Unidad de
  usuario con `loginctl enable-linger`. Borrar `logrotate-bot` (el bot ya
  rota) o cambiarlo a `ml_bot.log` sin `copytruncate`.
- **Aceptación:** `sudo reboot` del VPS → bot arriba en <2 min con `Pos=0/1`.
- **Riesgo conocido:** con dos supervisores (systemd + `run_bot.sh` a mano)
  vuelven las dos instancias de agosto. Antes de habilitar la unidad, `pkill
  -f run_bot.sh`.

### 5.2 PnL real desde Binance — §2.8

- **Archivo:** `src/portfolio_manager.py` `_close_position`,
  `_handle_stale_position`
- **Cambio:** tras cerrar, leer `fapiPrivateGetIncome` (REALIZED_PNL,
  COMMISSION, FUNDING_FEE) filtrado por símbolo y ventana del trade; guardar
  `pnl_real` y `commission_real` en columnas nuevas. `pnl` estimado se
  conserva para comparar.
- **Aceptación:** para los trades #84-86 de septiembre, `pnl_real` coincide
  con la tabla de `SESION_2026-09-22.md` Parte 1.

### 5.3 Reproducibilidad

- Quitar `poetry.lock` de `.gitignore` y commitearlo desde el venv de
  producción (sklearn 1.8.0).
- `logging.Formatter.converter = time.gmtime` en `setup_logging`
  (`ml_bot.py:1390`) para que el log vaya en UTC como el resto.
- Añadir `pytest` a `[tool.poetry.group.dev.dependencies]`.

### 5.4 Yield manager: validar o eliminar — §3.3

Solo si se quiere reactivar (0.1). Requisitos antes de `simulate_mode=False`:

1. `open_position` redime de Earn el margen que falte **antes** de la orden
   (Fase 4.1 del plan de junio).
2. `refresh_balance` suma `earn_balance` al equity para sizing y kill switch.
3. Transfer y subscribe en una sola transacción lógica con rollback (si
   subscribe falla, transfer de vuelta).
4. `refresh_live_earn_balance` llamado en cada rebalance.
5. Probado en mainnet con capital mínimo ($50) durante 2 semanas antes de
   conectarlo al bot.

Si no se va a hacer todo eso, **borrar** `src/yield_manager.py` y el bloque
de settings. Por 1,8%/año no compensa.

---

## Fase 6 — Que el repo diga la verdad (1 día, sin riesgo)

### 6.1 Enrutado a V2 por flag, no por archivo — §5

- **Archivos:** `config/settings.py`, `src/ml_strategy_v15.py:177,347-353`
- **Cambio:** `ML_V15_ENGINE = {'BTC/USDT': 'v2'}` en settings;
  `generate_signals` y `load_models` leen ese dict. Eliminar la dependencia
  de `meta_v2_paper.json`. Si el flag es `'v2'`, **no** cargar ningún `.pkl`.
- **Test:** `tests/test_engine_routing.py`: con el flag en `'v2'`, borrar el
  JSON no cambia el comportamiento; `short_model is None`.

### 6.2 Limpiar `settings.py`

- Borrar: bloque v6.7 (`SYMBOL`, `TIMEFRAME="1m"`, `LEVERAGE=10`, DCA…),
  `ML_PAIRS`, `ML_PAIR_CONFIGS`, `ML_BTC_CONFIG`, V8.4, V8.5, V9, V13.03/04,
  `ML_V14_EXPERTS`, `ML_V14_MODEL_FILTERS`. Conservar `COMMISSION_RATE`,
  `SLIPPAGE_PCT`, `INITIAL_CAPITAL` (solo para el log), riesgo, leverage,
  Telegram, DB, V15/V2, yield (apagado).
- `get_pair_tp_sl` (`portfolio_manager.py:27-35`) desaparece con 1.3.
- `validate_config` y `print_config` reescritos para lo que queda.

### 6.3 Código muerto

- Borrar `src/ml_strategy.py`, `src/ml_strategy_v14.py`,
  `src/shadow_portfolio_manager.py`, y en `ml_bot.py` los caminos
  `_on_new_candle_dual`, `_on_new_candle_single`, `_execute_v9_signal`,
  `_execute_shadow_signal` y los imports asociados.
- En `ml_strategy_v15.py` borrar `_generate_btc_signals`,
  `_generate_eth_signals`, `_generate_alt_trailing_signals` y helpers
  (`_short_prob`, breakout/follower). Quedan `update_regime` (solo para el log
  y leverage) y `_generate_v2_signal`.
- `/retrain` en Telegram: eliminar (entrena un GBM que nadie carga).
- `telegram_alerts.py:80-268`: borrar las `alert_*` con `SYMBOL` y
  `run_git_pull/run_export_models/run_pull_and_export`.
- Mover a `archive_scripts/`: `v15_features.py`, `v15_data_pipeline.py`,
  `v15_market_structure.py`, `v15_framework.py`, `train_v15_*.py`,
  `evaluate_new_pairs_v15.py`, `revalidate_v15.py`, `ml_export_v14.py`,
  `ml_train_v7.py`, `ml_train_v85.py`, `ml_export_models.py`.

### 6.4 Metas y modelos

- `strategies/`: dejar solo `btc_v15/` con `meta_v2_paper.json` (o el flag de
  6.1). Los 21 directorios restantes y los `.pkl` de BTC → `backup_models/`
  con un README: "métricas producidas por un simulador con look-ahead
  intrabar, ver AUDITORIA_2026-09 §4.2; no usar".
- `strategies/TRAINING_PLAN.md` → `docs/archive/`.

### 6.5 Datos

- Borrar duplicados byte-idénticos en `data/` (`X_4h_v15.parquet` ==
  `X_USDT_4h_full.parquet`), `ml_trades.db` (0 B), `bot_trades.db`.
- Refrescar `btcusdt_4h_v15.parquet`, `btcusdt_1d_v15.parquet`,
  `btcusdt_1h.parquet` hasta hoy con `download_new_pairs.py` en el venv de
  producción (higiene, `CLAUDE.md` punto 4 ya dice que no aporta información).

---

## Fase 7 — Tests del camino del dinero (1 día)

Sin framework pesado: `pytest` con un `FakeExchange` en `tests/conftest.py`
que registre `create_order`, `cancel_order`, `fetch_positions`,
`fetch_order`, `fetch_ticker`, `fetch_balance` y permita inyectar fallos.

| test | cubre |
|---|---|
| `test_pm_open_close.py` | open → fill → SL en exchange → cierre por trail; PnL con comisión; DB consistente. |
| `test_pm_pending_recovery.py` | 1.3: excepción tras la orden → arranque → adopción con parámetros del motor. |
| `test_pm_streak.py` | 1.1: racha por signo de PnL. |
| `test_pm_state_persist.py` | 1.2: `killed` sobrevive al reinicio. |
| `test_pm_sl_replace.py` | 1.4: fallo al recolocar stop no deja `sl_order_id` huérfano. |
| `test_engine_routing.py` | 6.1. |
| `test_regime_warmup.py` | 0.2: `bull_1d` con 1000 velas == historia completa en los 29 días conflictivos. |

---

## Orden y dependencias

```
Fase 0 (hoy) ──► Fase 1 ──► Fase 7 (tests de lo que cambió en 1)
                    │
                    └──► Fase 2.1/2.3 (arranca ya, mide 6-12 meses)
Fase 3 (docs) — independiente, hacerla esta semana
Fase 4 (bootstrap) — independiente, cuando haya un día de cómputo
Fase 5 (operación) — tras Fase 1; 5.1 antes del próximo reboot del VPS
Fase 6 (limpieza) — cuando el bot lleve ≥1 trade correcto con las Fases 0-1
```

Despliegue: cada fase es un commit propio y se sube al VPS con `git checkout
-- src && git pull` (el VPS tiene cambios sin commit sobre `8048163`, ver
`SESION_2026-09-22.md` Parte 5), reinicio con `kill <pid python>` y
verificación de `[BOT] Listo` + `Pos=0/1` + una sola instancia (`ps -eo
pid,lstart,cmd | grep ml_bot`).

## Lo que este plan NO hace, a propósito

- No toca `PARAMS_V2`. Siete búsquedas murieron en validación.
- No añade pares, direcciones ni modelos. Cerrado en `CLAUDE.md` punto 0.
- No promete acercarse al 30% anual. La expectativa honesta es la de
  `portfolio_sim` y, tras la Fase 4, probablemente algo menor. La decisión de
  seguir o parar es del usuario con esa cifra delante.
