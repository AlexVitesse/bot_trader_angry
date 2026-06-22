# Plan de Mejoras — Junio 2026

> Origen: tras arreglar el bug del engine V2 que dejó al bot 34 días sin operar
> (commit `0d0419d`), se hizo una auditoría en 3 dimensiones (fidelidad
> vivo-vs-backtest, observabilidad, yield/economía). Este documento es el plan
> de implementación de los hallazgos.

## Principios

- **Cada fix = un commit pequeño y revertible** en `v15/multi-pair`.
- **Los `PARAMS_V2` frozen NO se tocan.** Ningún cambio debe alterar el path de
  backtest (`live=False` por defecto). Test obligatorio: el nº de trades del
  backtest histórico V2 no cambia tras cada fix.
- **No re-tunear estrategia** (disciplina anti-overfitting). Solo correctness,
  fidelidad y observabilidad.
- **Objetivo guía**: que los datos de paper trade sean fieles a la estrategia
  validada, para que la decisión de capital real se base en datos limpios.

---

## FASE 0 — Verificación previa (sin cambios de código)

Confirmar los hallazgos marcados "a verificar" antes de escribir fixes. Cada uno
es una lectura dirigida del código + (si aplica) un test de reproducción.

| # | Qué verificar | Cómo | Salida |
|---|---------------|------|--------|
| V1 | ¿La validación in-sample/OOS de V2 usó funding? | Revisar `experiments/combined_AF/`, `agent_A/`, `agent_F/` y notebooks de validación | Decide si el veto de funding se CONECTA (si se validó con él) o se ELIMINA como código muerto (si no) |
| V2 | SL ficticios en cierres externos | Leer `portfolio_manager.py:490-510` (`_close_position`, `_handle_stale_position`) | Confirmar que `reason='SL'` se asigna sin verificar fill real |
| V3 | Tope cross-pair "2 misma dirección" | Leer `portfolio_manager.py:560-575` (`can_open`) + cómo valida el backtest V2 el portfolio | Confirmar si el backtest asume multi-LONG simultáneo sin tope |
| V4 | Trailing por tick vs por vela | Leer `portfolio_manager.py:755-820` (`update_positions`) | Confirmar que el SL se mueve cada 30s con `ticker['last']` |

**Entregable Fase 0**: nota corta (1 párrafo por item) confirmando o descartando,
para ajustar el alcance de Fases 1 y 3.

---

## FASE 1 — Fidelidad vivo-vs-backtest (confirmados) 🔴

### 1.1 — Propagar `max_bars` al timeout de la posición
- **Problema**: el engine V2 devuelve `max_bars` (A=60, F=40) pero `open_position`
  usa `ML_MAX_HOLD` (30 BEAR / 15 RANGE). Los trades A se cortan a la mitad/cuarto.
- **Archivos**: `src/portfolio_manager.py` (`open_position`, ~línea 600-660),
  `src/ml_bot.py` (`_execute_v14_signal`, líneas 1011-1023).
- **Pasos**:
  1. Añadir parámetro `max_hold_override: int | None = None` a `open_position`.
  2. En la creación de la posición: `max_hold = max_hold_override or ML_MAX_HOLD.get(regime, 15)`.
  3. En `_execute_v14_signal` pasar `max_hold_override=signal.get('max_bars')`.
  4. Verificar que el payload V2 ya incluye `max_bars` (`ml_strategy_v15.py:398`) — sí.
- **Validación**: abrir una posición V2 simulada y confirmar en la DB que
  `max_hold` = 60/40 (no 30/15). Confirmar que V14 (sin `max_bars`) sigue usando
  `ML_MAX_HOLD`.
- **Riesgo**: bajo. Aditivo, default preserva comportamiento V14.
- **Esfuerzo**: ~30 min.

### 1.2 — Veto de funding en vivo (o eliminar código muerto)
- **Problema**: `get_live_signal(..., df_funding=None)` → `funding_z=0` → filtros de
  funding nunca bloquean en vivo.
- **Depende de**: V1 (Fase 0).
- **Camino A (si V2 se validó CON funding)**:
  1. Reusar/crear fetch de funding por par (verificar `_fetch_funding_zscore` en
     `ml_strategy_v15.py`; el régimen V15 ya usa funding z-score).
  2. Construir `df_funding` (resample a 4h, columna `funding_rate`) y pasarlo a
     `get_live_signal` en `ml_strategy_v15.py:383`.
  3. Validar que `funding_z` no es 0 constante en vivo (log de diagnóstico).
- **Camino B (si NO se validó con funding)**:
  1. Documentar en `v2_engine.py` que el veto de funding no forma parte de la
     estrategia validada y dejar `funding_z=0` explícito, o quitar los 3 filtros.
- **Validación**: que el live ejecute exactamente la estrategia validada (con o
  sin funding, pero coherente).
- **Riesgo**: medio (camino A toca fetch de datos). 
- **Esfuerzo**: ~1-2 h (A) / 15 min (B).

**Commit sugerido Fase 1**: `fix(v2): fidelidad vivo-vs-backtest (max_bars + funding)`

---

## FASE 2 — Observabilidad 🟠

Objetivo: que un fallo como el de los 34 días se detecte en <24h. La causa raíz
es que "0 trades" es el estado normal → hay que distinguir "motor muerto" de
"mercado quieto".

### 2.1 — Loguear la RAZÓN del "no signal" *(la más importante)*
- **Problema**: `[V2] {pair}: no signal` no dice qué filtro bloqueó; un motor roto
  es indistinguible de un mercado quieto.
- **Archivos**: `src/v2_engine.py` (`get_live_signal` → devolver `reason`),
  `src/ml_strategy_v15.py` (`_generate_v2_signal`, log).
- **Pasos**:
  1. Hacer que `detect_signal`/`get_live_signal` devuelvan un motivo cuando no hay
     señal (`'warmup'`, `'regime'`, `'no_breakout'`, `'no_compression'`, `'vol_low'`,
     `'funding_veto'`, `'error'`). Sin tocar la lógica de decisión.
  2. Loguear `[V2] {pair}: no signal (reason)`.
  3. Resumen agregado por vela: `[V15] vela: 5 pares, 0 señales (3 regime, 2 no_breakout), 0 errores`.
- **Validación**: forzar cada rama con datos sintéticos y ver el `reason` correcto.
- **Riesgo**: bajo (solo añade información; no cambia decisiones). Cuidar no alterar
  el backtest → el `reason` solo se computa/retorna en `get_live_signal`.
- **Esfuerzo**: ~1-2 h.

### 2.2 — Contador "N días sin señal" + alerta proactiva
- **Archivos**: `src/ml_bot.py` (`__init__`, `_on_new_candle_v14`, `_periodic_tasks`).
- **Pasos**:
  1. `__init__`: `self.candles_since_signal = 0`, `self.last_signal_ts`.
  2. En `_on_new_candle_v14`: reset si hubo señales, `+=1` si no.
  3. En `_periodic_tasks`: si `candles_since_signal >= 18` (~3 días) y no se alertó
     hoy → `send_alert("⚠️ ENGINE SIN SEÑALES: 0 en {d} días sobre {n} pares...")`.
- **Validación**: simular 18 ciclos sin señal y confirmar 1 alerta (no spam).
- **Riesgo**: bajo.
- **Esfuerzo**: ~45 min.

### 2.3 — "Días desde último trade" en heartbeat y `/status` (desde DB)
- **Archivos**: `src/portfolio_manager.py` (nuevo `get_last_trade_time()` →
  `SELECT MAX(exit_time) FROM ml_trades`), `src/ml_bot.py` (`_send_heartbeat`,
  `_cmd_status`).
- **Pasos**:
  1. Método que lee el último `exit_time` de la DB (sobrevive reinicios).
  2. Línea `📅 Último trade: hace {X}d` en heartbeat y `/status`; `⚠️` si `X>7`.
- **Validación**: comparar contra la DB real.
- **Riesgo**: bajo (solo lectura).
- **Esfuerzo**: ~45 min.

### 2.4 — Robustez de monitoreo (clase de bugs hermana)
- **2.4a Propagar errores del engine a `recent_errors`**: el `try/except` de
  `generate_signals` (`ml_strategy_v15.py:336`) loguea pero no alerta. Contar
  excepciones y exponerlas al heartbeat. Añadir `consecutive_fetch_fails` con
  alerta si N velas seguidas fallan todos los fetch.
- **2.4b Watchdog de "vela procesada"**: en `_periodic_tasks`, si
  `now - last_candle_processed > 5h` → alerta (loop colgado).
- **2.4c Heartbeat de salud del ENGINE**: sustituir `🟢 V15 OK` fijo por estado
  condicionado (engine evaluó sin errores + último trade < umbral).
- **Archivos**: `src/ml_bot.py`, `src/ml_strategy_v15.py`.
- **Esfuerzo**: ~2 h combinado.

**Commits sugeridos Fase 2**: uno por subfase (2.1, 2.2+2.3, 2.4).

---

## FASE 3 — Fidelidad vivo-vs-backtest (tras verificación) 🟡

Solo proceder con los confirmados en Fase 0.

### 3.1 — No etiquetar cierres externos como `SL` *(si V2 confirma)*
- **Problema**: cierres por liquidación/manual/TP-exchange se loguean `reason='SL'`
  al precio de SL → falsean WR/PnL del bootstrap.
- **Archivos**: `src/portfolio_manager.py:490-510`.
- **Pasos**: cuando no se pueda confirmar el fill real, `reason='EXTERNAL'` y usar
  último precio de mercado, no asumir SL.
- **Validación**: simular desaparición de posición sin fill y ver `reason='EXTERNAL'`.
- **Esfuerzo**: ~45 min.

### 3.2 — Tope cross-pair "2 misma dirección" *(si V3 confirma mismatch)*
- **Archivos**: `src/portfolio_manager.py:560-575`.
- **Pasos**: para `engine='v2_honest'`, subir/eliminar el tope o registrar
  "señal perdida por límite" para cuantificar el impacto. Documentar la diferencia
  con el backtest si se deja.
- **Esfuerzo**: ~30 min + decisión.

### 3.3 — Modelo de trailing tick vs vela *(si V4 confirma)*
- **Problema**: trailing por tick de 30s vs por vela 4h en backtest → divergencia.
- **Opciones**:
  - (a) Documentar la divergencia esperada y trackearla contra el KPI "real diverge
    <25% del simulado" del CLAUDE.md (menos invasivo).
  - (b) En vivo, actualizar el trailing solo en cierre de vela 4h (replica exacta
    del backtest) — más fiel pero cambia comportamiento de salida.
- **Recomendación**: (a) primero (medir), (b) solo si la divergencia supera el KPI.
- **Esfuerzo**: (a) ~30 min / (b) ~2 h.

### 3.4 — Saneamiento de errores silenciosos
- Convertir los `except: pass` / `return []` silenciosos
  (`ml_strategy_v15.py:336,409,425,584`; `portfolio_manager.py:293,453,467,478`)
  en contadores que alimenten `recent_errors`. (Solapa con 2.4a.)
- **Esfuerzo**: ~1 h.

---

## FASE 4 — Yield / preparación mainnet 🟢

Solo relevante al acercarse a capital real (en testnet el yield se simula).

### 4.1 — Redeem síncrono antes de abrir trade (mainnet)
- **Problema**: `open_position` no redime de Earn; un trade puede fallar por margen.
- **Archivos**: `src/portfolio_manager.py` (`open_position`), `src/yield_manager.py`.
- **Pasos**: antes de `create_order` en live, asegurar
  `futures_free >= margen_requerido + colchón`; si no, redeem síncrono.
- **Riesgo**: alto (toca flujo de capital real) → probar en testnet con simulación
  de déficit.
- **Esfuerzo**: ~2-3 h.

### 4.2 — Atomicidad/reintento/alerta en redeem fallido
- `src/yield_manager.py:286-344`. Separar redeem y transfer en pasos idempotentes,
  reintentar el transfer, alertar por Telegram, reconciliar con
  `refresh_live_earn_balance()`.
- **Esfuerzo**: ~2 h.

### 4.3 — Llamar `refresh_live_earn_balance` en el loop (mainnet)
- Nunca se invoca → en mainnet el interés real no se reconcilia.
  En `check_and_rebalance`, si `not simulate`, llamarla antes de calcular targets.
- **Esfuerzo**: ~20 min.

### 4.4 — Validar `productId` de Binance Earn
- `settings.py:444` (`'USDT001'`) hardcodeado. Validar vía API
  (`sapiGetSimpleEarnFlexibleList`) antes del primer sweep.
- **Esfuerzo**: ~30 min.

### 4.5 — Buffer dinámico (optimización de drag)
- Depende de 4.1 (redeem on-demand seguro). `target=0.05` si no hay posiciones ni
  señal reciente; `0.20` si hay. Recupera ~$20/año.
- **Esfuerzo**: ~1 h.

### 4.6 — Reportar dos APYs
- `get_status` (`yield_manager.py:179`): "APY del pool" y "APY efectiva sobre
  capital total" (`interest/total_capital` anualizado). Aclara el 2.90% vs ~2.4% real.
- **Esfuerzo**: ~20 min.

---

## Orden de ejecución recomendado

1. **Fase 0** (verificación) — barato, desbloquea el resto.
2. **Fase 1** (max_bars + funding) — confirmados, baratos, críticos para datos limpios.
3. **Fase 2.1–2.3** (razón de no-señal, alerta sin-señal, días-desde-trade) — anti-ceguera.
4. **Fase 2.4** (robustez monitoreo).
5. **Fase 3** (lo que Fase 0 confirme).
6. **Fase 4** — solo al planear mainnet.

## KPIs de "no rompí nada"
- El backtest histórico V2 da el **mismo nº de trades y PnL** tras cada commit.
- Tras Fase 1, una posición V2 simulada muestra `max_hold` correcto en DB.
- Tras Fase 2, una simulación de "motor muerto" dispara alerta en <3 días.
