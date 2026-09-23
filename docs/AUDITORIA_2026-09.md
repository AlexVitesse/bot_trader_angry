# Auditoría 2026-09-22 — Diseño, validación, entrenamiento y ejecución

> Alcance: revisión completa del proyecto en `main` (HEAD `b23ad0d`), desde el
> diseño estadístico del motor V2 hasta el path de ejecución en el VPS.
> Método: lectura del path en vivo (`ml_bot.py` → `ml_strategy_v15.py` →
> `v2_engine.py` → `portfolio_manager.py`), auditoría de `experiments/` y
> scripts de entrenamiento, y dos mediciones nuevas con datos reales
> (sección 2.1 y 2.2). Plan de acción: `docs/PLAN_MEJORAS_2026-09.md`.

---

## Veredicto

El motor V2 está bien construido y la documentación es honesta. Pero:

1. El sistema desplegado **no cumple el objetivo del proyecto** (30% anual):
   con fills honestos da +6,2%/año a 2% de riesgo, menos que BTC comprar y
   mantener (+9,7%). El perfil desplegado (4,5% riesgo, kill al 45% DD) es el
   mismo edge apalancado.
2. La **evidencia estadística es más débil** de lo que afirman `CLAUDE.md` y
   `SESION_2026-08-22.md` (sección 1).
3. El path en vivo tiene **dos agujeros de seguridad** y **varias divergencias
   sim/real**, dos de ellas ahora medidas y ambas negativas (secciones 2 y 3).

Nada de esto justifica capital real todavía.

---

## Lo que está bien

| Área | Evidencia |
|---|---|
| Motor V2 sin look-ahead | `src/v2_engine.py:146,154-156,162-163,172-179,190`: Donchian, hi/lo, cuantil BB, régimen diario y funding con `shift(1)`. Trailing chequea stop previo antes de subir peak (`:299-320`). Ningún estadístico full-sample. |
| Simulador de cartera honesto en ejecución | `experiments/portfolio_sim/portfolio_sim.py:233` fill al open siguiente; `:199-213` stop con nivel previo; `:137-147` margen y equity compartido; `:106` DD con capital inicial como pico. |
| Experimentos negativos documentados | Siete búsquedas de parámetros rechazadas, once familias cerradas. Cada una con README. |
| Secretos | Todo por `os.getenv` (`config/settings.py:31-40,99-100`). `.env` ignorado y nunca commiteado. |
| Tests de regresión | `tests/` 4 archivos, 6/6 pasan, cada uno ligado a un incidente real. |
| Verdad de trades | Desde sep-2026 se cruza con `income`/`userTrades` de Binance, no solo con `ml_trades`. |

---

## 1. Diseño y validación estadística

### 1.1 [ALTA] "p=0,004 estable en todos los niveles de riesgo" es una tautología

`portfolio_sim.py:244` fija `notional = equity·risk/trail` y `:223` define
`r = pnl/equity_entrada`, así que `r ∝ risk_pct`. El bootstrap (`:109-116`)
testea `mean(r) ≤ 0`. Escalar `r` por una constante nunca cambia el signo de
la media: los cinco p iguales del README (`:29-35`) son **el mismo número**.
El 0,005 a 6% aparece solo porque muerde el tope 2,5× (`:245`).

Afecta a: `CLAUDE.md` (sección "Requisitos de Validación"),
`docs/SESION_2026-08-22.md:72-74,85-88`. La frase debe retirarse como
argumento.

### 1.2 [ALTA] Bootstrap i.i.d. sobre trades agrupados en dos regímenes

`portfolio_sim.py:110` remuestrea trades con reemplazo, sin bloques. Los ~165
trades se concentran en dos episodios alcistas (folds 1 y 4 del README
`:136-141`). Es el mismo error de independencia que la sesión 08-22 denuncia
en `criterio_validacion/`, `estacionalidad/` y `presupuesto_informacion/`.
Nunca se midió la tasa de falsos positivos del bootstrap bajo clustering de
régimen.

### 1.3 [ALTA] El bootstrap no es independiente de las decisiones de diseño

- V2 es la mejor de **6 variantes** (`experiments/combined_AF/README.md:19-25`)
  y se declara significativa con p=0,031 sin corrección. Con 6 pruebas no
  sobrevive Bonferroni.
- `f_enable_short=False` se decidió sobre la historia completa
  (`v2_engine.py:64-70`) y se aplica en todos los folds del "walk-forward".
- `agent_F/explore_params.py:75-113` **sí barrió** percentil, breakout_n,
  min_bars, trail y vol_ratio sobre 2020-2025. Que "conservara el baseline" se
  afirma, no se puede verificar. V2 además cambia F respecto a agent_F
  (`f_max_bars` 40 vs 48, warmup 220 vs 250, sin vol-targeting) sin documentar.
- `v2_all_coins`: 21 monedas a α=0,05 → ~1 falso positivo esperado; DOGE
  p=0,001 se presenta sin corrección (`README.md:9-12`). "BNB Tier 1 capital
  real" con 4 trades OOS (`README.md:148,157-159`) contradice los requisitos.

### 1.4 [ALTA] El walk-forward con parámetros congelados no es walk-forward

`run_walkforward.py`: solo la **selección de pares** es out-of-sample.
`PARAMS_V2` se aplica fijo en todos los folds. Para BTC-only, "2/6 folds
positivos" es la misma curva in-sample partida en seis. Sin purga ni embargo
entre train y test (`:87-92`). `pf_en_train` usa `run_v2_backtest` con fill al
close (`v2_engine.py:361`), no el honesto.

Lo mismo aplica a los "WF 8/12" de V15: `v15_framework.py:315-357` y
`revalidate_v15.py:339-355` recorren 12 semestres con parámetros elegidos
mirando la historia completa (`evaluate_new_pairs_v15.py:31-33`).

### 1.5 [MEDIA] `criterio_validacion/` es circular

`test_criterio.py:53-55` remuestrea trades in-sample de V2 (fill al close,
n=131) y concluye que "V2 barajado pasa 91%". Asume el edge que pretende
validar. Muestra que el fold-count tiene poca potencia contra un i.i.d. de EV
positiva; **no demuestra que el bootstrap discrimine**, porque nunca simula un
sistema sin edge con clustering de régimen. Folds por número de trades
(`:32`), no por calendario.

### 1.6 [MEDIA] Circulan cuatro "V2" distintos

| fuente | trades | fill | apalancamiento F |
|---|--:|---|---|
| `criterio_validacion/` | 131 | close | no |
| `combined_AF/` | 163 | close | sí (vol-targeting hasta 3×), mezclado con A sin apalancar en un solo bootstrap (`test_combined.py:160`) |
| `portfolio_sim/` | 165 | open t+1 | no |
| `f_short_ablation/` | 168/139 | close | ? |

`CLAUDE.md` mezcla cifras de varios. `portfolio_sim/README.md:125` dice
+6,6%/año; `CLAUDE.md` dice +6,2%. El 13,2% CAGR full-history vs 6,6% WF no se
reconcilia (los folds no componen y excluyen 2019-20).

### 1.7 [MEDIA] Costes no modelados en el simulador

- Fill del stop al nivel exacto (`portfolio_sim.py:200`), sin gap. Con
  notional 1,8-2,4× equity es optimista.
- Slippage 0,01%/lado (`:46`).
- Funding constante (`:47,196`). El funding es alto precisamente en régimen
  alcista, donde opera A_LONG → coste subestimado para una long-only filtrada
  por bull.

### 1.8 [INFO] `oos_2026H1/` correcto pero débil

Parámetros congelados, datos posteriores: bien. Pero el "filtro acertó" se
apoya en 21 trades contrafactuales con fill al close (`test_oos.py:84`) y una
ventana de 170 días. Evidencia direccional, no validación.

---

## 2. Ejecución en vivo — divergencias sim/real

### 2.1 [ALTA] Trailing por tick vs por vela — MEDIDO, negativo

Sim: `v2_engine.py:299-320` actualiza el peak **una vez por vela 4h**, con el
`high`, después de chequear el stop previo.
Vivo: `portfolio_manager.py:863-880` (`_update_trailing`, modo tight) sube el
peak con `ticker['last']` **cada 30 s** y aprieta el SL en el mismo tick. El
stop vivo es más ceñido y se mueve intrabar.

Medición (2026-09-22): los 82 trades de V2 desde 2022-01 (donde hay datos 1h,
`data/btcusdt_1h.parquet`) re-simulados con el trail actualizado por vela de
1h en vez de 4h, misma entrada, mismo `trail_dist`, mismo `max_bars`:

| granularidad del trail | n | suma PnL | WR | PF | media/trade |
|---|--:|--:|--:|--:|--:|
| vela 4h (backtest) | 82 | +87,0% | 49% | 2,06 | +1,06% |
| vela 1h (aprox. vivo) | 82 | +78,7% | 49% | 1,93 | +0,96% |

13 de 82 trades cambian de resultado. A 30 s la pérdida será mayor que a 1h.
Esto solo puede disparar el KPI "real diverge >25% del simulado".

### 2.2 [ALTA] Régimen diario con warm-up insuficiente — MEDIDO

`ml_strategy_v15.py:410` baja **300 velas diarias**. `v2_engine.py:170-172`
calcula EMA50/EMA200 con `adjust=False` desde la primera vela: tras 300 velas
el valor inicial aún pesa ≈5%. El backtest la calcula sobre toda la historia.

Medición sobre `btcusdt_1d_v15.parquet` (2019-09 → 2026-03, 1.770 días
evaluados):

| velas diarias en vivo | días en que `bull_1d` difiere del backtest |
|--:|--:|
| 300 (actual) | 29 (1,6%), concentrados en los cruces: 2022-01, 2023-02, 2023-09, 2024-09, 2025-11 |
| 1000 | 0 |

Los cruces son justo cuando hay señal. Fix: un número.

### 2.3 [MEDIA] La pausa por 3 SL cuenta por etiqueta, no por PnL

`portfolio_manager.py:999`: `if reason == 'SL': consecutive_losses += 1`.
- Si el **stop del exchange** cierra un trade ganador (trail por encima de la
  entrada), `_close_position` cae en el `except` → `_get_exchange_sl_fill` →
  `reason='SL'` (`:944`) → suma a la racha. El trade #84 de sep-2026 (+$142)
  contó como pérdida.
- Si el **bot** cierra un perdedor por trail, `reason='TRAIL'` (`:822-825`) →
  resetea la racha.

Además, la pausa por racha, el límite diario y el kill switch **no existen en
el simulador**: vivo y sim ya no son el mismo sistema.

### 2.4 [MEDIA] Adopción tras crash pierde los parámetros de V2

`portfolio_manager.py:717` manda la orden; `_save_position` es en `:750`; el
SL de exchange en `:754`. Si el proceso muere entre `:717` y `:750`, o si el
entry difiere >1% del exchange (`:346`), `_reconcile_with_exchange`
(`:365-374`) o el "DUPLICADO EVITADO" (`:689-698`) reconstruyen la posición
con `get_pair_tp_sl` → **TP 4% / SL 2% de la config V13** (`ML_BTC_CONFIG`,
`settings.py:205-212`), `max_hold=30`, `trail_mode='default'`. La posición
queda gestionada por una lógica de salida distinta a la simulada.
`docs/SESION_2026-09-22.md` Parte 6.2 ya lo señala como "adopción débil".

### 2.5 [MEDIA] Reemplazo del stop en exchange sin rollback

`portfolio_manager.py:849-852`: cancela el stop viejo y coloca uno nuevo. Si
la colocación falla, `pos.sl_order_id` sigue apuntando a la orden **ya
cancelada**. La posición queda sin stop real y el bot cree que lo tiene.
`_get_exchange_sl_fill` leería luego una orden `canceled` y caería al precio
teórico.

### 2.6 [MEDIA] Estado de riesgo no persistido

`killed` (`:1019`) y `consecutive_losses` (`:998-1002`) viven en memoria. Un
`/restart` (exit 43) o crash (reinicio en 30 s por `run_bot.sh`) los borra.
`daily_pnl` sí se restaura (`:308-313`). Con systemd `Restart=always` el kill
switch se evaporaría.

### 2.7 [BAJA] Funding veto apagado en vivo

`ml_strategy_v15.py:426` pasa `df_funding=None` → `funding_z=0`
(`v2_engine.py:192-193`). El backtest sí lo aplicaba. Impacto pequeño (6
trades en 6,5 años según `experiments/funding_veto/`).

### 2.8 [BAJA] PnL de la DB es estimado

`_close_position` (`:949-953`) y `_handle_stale_position` (`:517-529`) usan
`COMMISSION_RATE + SLIPPAGE_PCT` fijos y el precio teórico del SL si no hay
fill. Ya causó el bug de sep-2026. Sigue siendo la fuente de `/status`,
resumen diario y KPI de parada.

### 2.9 [BAJA] `/resetdb` está roto

`ml_bot.py` `_cmd_resetdb`: `UPDATE ml_state SET peak = balance WHERE id = 1`
contra una tabla `(key, value)` (`portfolio_manager.py:136-139`) →
`OperationalError` antes del `commit` → rollback. Nunca borra nada, siempre
responde error.

### 2.10 [INFO] Fills y comisiones

Sim entra al close de la vela de señal (`v2_engine.py:361,447`); el bot espera
minuto ≥2 (`ml_bot.py:872-873`) y manda market → fill ≈ open t+1 + slippage.
`portfolio_sim` sí modela el open t+1. Comisión: sim 0,05%/lado; real demo
0,04% (medido en sep-2026: $18,18 sobre $45k notional). Conservador, OK.

---

## 3. Seguridad y operación

### 3.1 [ALTA] Cualquiera puede mandar comandos al bot — FUERA DE PLAN por decisión del usuario 2026-09-22

`src/telegram_alerts.py:302-349` (`_poll_loop`) ejecuta cualquier `/comando`
de `getUpdates` sin comparar `msg['chat']['id']` con `TELEGRAM_CHAT_ID`.
Callbacks expuestos (`ml_bot.py:303-321`): `/resume` (levanta la pausa de
riesgo), `/restart`, `/restart_clean`, `/pull` (git stash+pull), `/install`,
`/resetdb`, `/clearlog`, `/yield_reset`. Verificado en código.

### 3.2 [ALTA] Token de Telegram en el log — FUERA DE PLAN por decisión del usuario 2026-09-22

`telegram_alerts.py:39,65,351` loguean `{e}` de `requests`; el mensaje incluye
`url: /bot<TOKEN>/...`. Confirmado: **11 ocurrencias** en `logs/ml_bot.log`
local. `/log` reenvía ese archivo y `CLAUDE.md` indica copiar logs con `tar`.
`docs/SESION_2026-09-22.md` Parte 6.3 ya pedía rotar el token.

### 3.3 [ALTA] El yield manager moverá capital real al pasar a mainnet

`config/settings.py:454-457`: `YIELD_MANAGER_ENABLED=True`,
`simulate_mode=None` → autodetecta LIVE en mainnet. `src/yield_manager.py:299-344`
hace `sapiPostAssetTransfer UMFUTURE_MAIN` + `SimpleEarnFlexibleSubscribe`
cada 10 min sobre el 50-60% del margen. **Nunca probado** (testnet no tiene
Earn). Problemas:

- `open_position` no redime antes de ordenar (`settings.py:463-465`, "Fase
  4.1 sin hacer") → fallos por margen.
- Si el transfer sale bien y el subscribe falla, el USDT queda huérfano en spot.
- `refresh_live_earn_balance` (`:346`) **nunca se llama**.
- `refresh_balance` (`portfolio_manager.py:571-572`) no ve el capital en Earn:
  en mainnet el riesgo del 4,5% se aplicaría sobre el **40%** del equity
  (≈1,8% real) y el kill switch mediría una cartera parcial que cambia con
  cada sweep.
- Las llamadas `sapi` van a `api.binance.com`, no redirigidas
  (`ml_bot.py:120-129`), y exigen permiso de transferencia universal.

Por ~1,8%/año de yield (3% APY × 60% ocioso) no compensa la complejidad ni el
riesgo. El riesgo y el yield usan **dos nociones distintas de equity**.

### 3.4 [MEDIA] `deploy/` apunta a un binario inexistente

`deploy/bot-trader.service:11` y `setup_server.sh:75` ejecutan `src/bot.py`
(scalper v6.7, no existe). `update.sh:21` reinicia esa unidad. `logrotate-bot`
rota `bot.log`; el bot usa `RotatingFileHandler` sobre `logs/ml_bot.log`
(`ml_bot.py:1397`) → con `copytruncate` chocarían. Consecuencia: el VPS corre
`run_bot.sh` a mano, **sin supervisor que sobreviva a un reboot**. El
docstring de `tests/test_network_watchdog.py:1-2` asume un systemd que no
existe.

### 3.5 [BAJA] Varios

- `poetry.lock` en `.gitignore` → VPS no reproducible; explica los parches
  `acd291c`/`f291f87` a `/update`. `pyproject.toml` arrastra fastapi, uvicorn,
  nbformat, yfinance, optuna que el bot no usa.
- `asctime` del log en hora local (VPS UTC-6) mientras toda la lógica es UTC.
- `/pull` hace `git stash` sobre el código en ejecución → el proceso corre
  código distinto al disco hasta `/restart`.
- Poller de Telegram ejecuta `/backup`, `/resetdb`, `/resume` desde otro hilo
  sin lock sobre SQLite (tolerable con WAL y un proceso).
- `run_bot.ps1:19` usa `python` a secas (entorno incorrecto en Windows).
- Sin CI. `pytest` no está en `pyproject.toml`. Cero tests sobre
  `portfolio_manager` (el camino del dinero), `yield_manager` ni Telegram.

---

## 4. Entrenamiento ML — defectos reales, todo muerto

Ruta viva real (grep de imports): `ml_bot.py` → `ml_strategy_v15.py` →
`v2_engine.py` → `portfolio_manager.py`. Nada en `src/` importa
`v15_framework`, `v15_features`, `v15_data_pipeline` ni `v15_market_structure`.

### 4.1 El GBM SHORT nunca se carga

`strategies/btc_v15/models/meta_v2_paper.json` existe → `routes_to_v2=True`
(`ml_strategy_v15.py:177-184`). `_generate_btc_signals` retorna al instante;
`_short_prob` (`:1029-1042`) es inalcanzable. Igual de muertos:
`setup_model_long/short.pkl`, `volume_model.pkl`, `short_model.pkl`,
`scaler.pkl`. `meta_v15.json` de BTC sigue declarando
`model_type: GradientBoostingClassifier`, `wf 8/12`, `cagr 37.5`.

### 4.2 Defectos en los scripts (histórico, trampas para quien los reactive)

| dónde | defecto |
|---|---|
| `train_v15_short_btc.py:148-149` | WF sin purga: `train_mask = index < start_s` con label a 16 velas → las últimas 16 etiquetas del train se resuelven con precios del test. `train_v15_btc.py:101` repite el patrón. |
| `train_v15_short_btc.py:122-135` vs `:204-208` | Label con TP 2,5%/SL 1,5% fijos; simulación con SL dinámico hasta 4% y RR 1,67. El modelo aprende un objetivo distinto del que se opera. Threshold 0,55 research vs 0,60 prod (`train_v15_prod.py:99`) sin documentar. |
| `evaluate_new_pairs_v15.py:82-85` | Simulador sube el peak con el high **antes** de chequear el stop, y un trade por cada vela con señal (solape). Es lo que produce PF 12-15 y DD 1-2% en `ada/sol/doge/dot_v15/meta_v15.json` (`short_pf` 13,5/15,0/12,7/13,9). Documentado en PASO0 pero los metas nunca se corrigieron. Criterios de fold: SHORT ok con `n>=1 and pf>0.8` (`:254`). |
| `v15_framework.py:236-238` | `sim_trade_fixed`: TP y SL en la misma vela → cuenta TP si el close supera el punto medio. Timeouts etiquetados TP/SL por signo (`:231,245`) → infla WR; `metrics()` (`:269-275`) puede meter pnl negativo en `gross_win`. |
| `v15_market_structure.py:297-304` | Look-ahead latente: swing en `i` requiere `i+1..i+5`; `weekly_struct` en `STRUCTURE_FEATURES` (`:408-414`). No alimentó modelos (`SETUP_FEATURES`, `v15_features.py:40-45`), pero está armado. |
| `v15_features.create_label:425-448` | Label solo con closes; el sim usa high/low. |
| `ml_export_v14.create_labels:179-201` | TP antes que SL en la misma vela (optimista). |
| `METODOLOGIA_TESTING.md:32-38,91-96` | Describe cómo añadir filtros post-hoc por modelo ("filtros que retengan 20% de trades"): la receta de overfitting que `CLAUDE.md` prohíbe. Obsoleto y contradictorio. |

### 4.3 Scripts muertos o superados

| script | estado |
|---|---|
| `v15_features.py`, `v15_data_pipeline.py`, `v15_market_structure.py`, `train_v15_btc.py` | Solo se importan entre sí; sus `.pkl` no se cargan. |
| `train_v15_short_btc.py` | Superado por `train_v15_prod.py`; ambos producen un GBM que el bot no usa. `/retrain` en Telegram lo ejecuta igualmente. |
| `evaluate_new_pairs_v15.py` | Superado por `revalidate_v15.py`; ambos irrelevantes con BTC-only. |
| `strategies/TRAINING_PLAN.md` | Describe el pipeline V14. Obsoleto. |
| `ml_export_v14.py`, `ml_strategy_v14.py`, `ml_strategy.py` (1.100 líneas), `shadow_portfolio_manager.py` (437) | V14/V9/shadow desactivados. Inalcanzables: `_on_new_candle_dual`, `_on_new_candle_single`, `_execute_v9_signal`. |

---

## 5. Higiene — el repo miente sobre lo que corre

De ~7.700 líneas en `src/`, el path vivo son ~600.

- `config/settings.py` tiene activos a la vez `ML_V84_ENABLED`,
  `ML_V85_ENABLED`, `ML_V1304_ENABLED`, `ML_V15_ENABLED`; `TIMEFRAME="1m"`,
  `LEVERAGE=10`, `INITIAL_CAPITAL=100`; diez pares V13 con TP/SL
  (`ML_PAIR_CONFIGS`) de los que `get_pair_tp_sl` sigue leyendo la config BTC
  para las adopciones (2.4).
- **Que BTC corra V2 depende de que exista un JSON**, no de un flag
  (`ml_strategy_v15.py:347-353`). Borrar `meta_v2_paper.json` resucita el
  GBM SHORT en silencio (`has_ml` por defecto `True`, `:181`).
- 22 directorios `strategies/*_v15` con metas que declaran PF 13-20.
- `data/` ~60 MB con duplicados byte-idénticos (`X_4h_v15.parquet` ==
  `X_USDT_4h_full.parquet`); DBs huérfanas `ml_trades.db` (0 B),
  `bot_trades.db`.
- Mensaje del kill switch hardcodea "≥ 20%" (`ml_bot.py:1371`) con
  `ML_MAX_DD_PCT=0.45`.

---

## Reproducir las mediciones

Ambas con `C:\Python\python.exe` desde la raíz del repo, sobre
`data/btcusdt_4h_v15.parquet`, `data/btcusdt_1d_v15.parquet` y
`data/btcusdt_1h.parquet`. Los scripts están en el plan
(`docs/PLAN_MEJORAS_2026-09.md`, tareas 2.1 y 2.2) para dejarlos en
`experiments/ejecucion_vivo/`.
