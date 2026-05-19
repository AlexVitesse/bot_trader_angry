# Paso 0 — Auditoría de look-ahead del pipeline

> Antes de re-validar nada: ¿los features usan datos del futuro? ¿El backtest
> mide lo que el bot realmente haría?
> Fecha: 2026-05-19 · Archivos auditados: `v15_features.py`, `v15_framework.py`,
> `v15_market_structure.py` (parcial), `evaluate_new_pairs_v15.py`,
> `src/ml_strategy_v15.py` (rutas de detección).

---

## Resumen ejecutivo

| Área | Veredicto |
|------|-----------|
| Cálculo de features (indicadores, macro, funding) | ✅ **LIMPIO** — sin look-ahead |
| Timing de entrada/salida en el simulador | ✅ **LIMPIO** — entra en vela cerrada |
| **Motor de backtest (conteo de trades)** | ❌ **FALLO GRAVE** — no modela "una posición a la vez" |
| Umbrales de aprobación de folds | ❌ **DEMASIADO LAXOS** — un fold pasa con 1 trade |

**Conclusión:** no hay look-ahead clásico (eso es buena noticia: los features son
sanos). Pero el motor de backtest tiene un fallo mecánico que **infla todas las
métricas declaradas en `meta_v15.json`**. Explica directamente los PF de 8–20 y
los DD de 1–4%. No es overfitting de ML — es un bug de simulación.

---

## 1. Lo que está BIEN (sin look-ahead)

- **Indicadores** (`v15_features.py`, `v15_framework.py`): EMA, RSI, ATR, BB, ADX,
  MACD, stoch, OBV, vol_ratio — todos sobre ventanas rolling/cumulativas de velas
  pasadas. El valor en la vela `i` solo usa velas `≤ i`.
- **Features macro 1D→4h**: `compute_macro_features` y `compute_macro_daily`
  aplican `.shift(1)` explícito (el régimen del día D se usa el día D+1). Comentado
  en el código. ✓
- **Funding rate**: `compute_sentiment_features` aplica `.shift(1)` antes del
  `reindex/ffill` al 4h. ✓
- **high20 / low20**: `h.rolling(20).max().shift(1)` — shift explícito. ✓
- **Funciones `detect_*`** (`evaluate_new_pairs_v15.py`): usan
  `df['high'].iloc[idx-20:idx]` — rango pasado, exclusivo de la vela actual; la
  decisión se toma con `close[idx]` de la vela **cerrada**. ✓
- **Simuladores** (`sim_trade_fixed`, `sim_long/short_trailing`): el bucle empieza
  en `entry_bar + 1`; la entrada es `close[idx]` y la evaluación de TP/SL/trailing
  ocurre solo en velas posteriores. ✓
- **`create_label`**: mira hacia adelante (velas `i+1..i+max`), que es lo
  **correcto** para una etiqueta — es el objetivo a predecir, no un feature.

> El bug histórico de look-ahead en MTF (mencionado en `MEMORY`) está corregido en
> este pipeline.

---

## 2. El FALLO GRAVE: el backtest no modela "una posición a la vez"

### 2.1 Qué hace el código

En `evaluate_new_pairs_v15.py::evaluate_pair` (líneas 188-212, 235-250) y en el OOS
(275+), el bucle es:

```python
for idx in fold_indices:        # CADA vela del fold
    ...
    if detect_breakout(df, idx, ...):
        out = sim_long_trailing(df, idx, entry, ...)
        fold_trades.append(...)          # abre un trade
    if not triggered and detect_btc_breakout(...):
        out = sim_long_trailing(df, idx, entry, ...)
        fold_trades.append(...)          # abre OTRO trade
```

No hay ninguna variable de estado "estoy en posición". Se abre un trade
**independiente en cada vela** donde la señal es verdadera.

`v15_framework.py::walk_forward` tiene una variable `in_trade` (líneas 332-334)
pero **nunca se pone a `True`** — el guard está muerto. Misma consecuencia.

### 2.2 Por qué esto infla las métricas brutalmente

Las condiciones de señal **persisten durante muchas velas seguidas**:
- Un breakout: `close > high20` se mantiene cierto mientras el precio siga arriba.
- Un BTC-breakdown: `close < low20` se mantiene cierto durante toda una caída.

En una tendencia de 30 velas, la señal puede dispararse en 15-20 velas
consecutivas → **15-20 trades solapados**, todos entrando casi al mismo precio,
todos siguiendo el mismo trailing, todos cerrando en "TP" juntos.

Efectos directos:
- **PF se dispara a 8–20**: decenas de trades correlacionados que ganan en bloque.
- **DD cae a 1–4%**: `equity_stats` encadena trades solapados como si fueran
  secuenciales; los ganadores en bloque suavizan la curva. El DD real de **una**
  posición sería mucho mayor.
- **WF ~100%**: con tantos trades clonados, cualquier fold en tendencia pasa.

### 2.3 Por qué el bot en vivo NO puede replicarlo

El bot real (`portfolio_manager.py`, `ML_MAX_CONCURRENT=3`, una posición por par)
abre **una** posición y la mantiene hasta que cierra. Jamás abriría 15 trades
solapados del mismo par. Por tanto **`meta_v15.json` mide algo que el bot nunca
hará**. Las columnas `long_pf`, `short_pf`, `oos_2026_combined` no son alcanzables.

---

## 3. Fallos secundarios

| # | Fallo | Ubicación | Efecto |
|---|-------|-----------|--------|
| 3.1 | Fold SHORT pasa con `n>=1 and pf>0.8` | `evaluate_new_pairs_v15.py:254` | 1 trade ganador = fold aprobado → "10/10" trivial |
| 3.2 | Fold LONG pasa con `n>=3` | línea 215 | Con trades solapados, n>=3 es trivial |
| 3.3 | `pf = inf` si 0 pérdidas | `v15_framework.py:275` | Folds con racha corta cuentan como ok |
| 3.4 | `annual_pct` con proxy de leverage x100 | `v15_framework.py:289` | "% anual" declarado no tiene sentido |
| 3.5 | TP+SL en la misma vela: desempate optimista | `v15_framework.py:236` | Sesgo leve a favor (debería asumir SL) |

### 3.6 BUG ADICIONAL descubierto al ejecutar el motor corregido — **look-ahead intrabar en el trailing**

`sim_long_trailing` / `sim_short_trailing` (`evaluate_new_pairs_v15.py:71-113`):

```python
for i in range(1, max_bars+1):
    b = entry_bar + i
    hi = high[b]; lo = low[b]
    if hi > peak: peak = hi                       # <-- usa el HIGH de la vela
    sl_price = max(sl_price, peak*(1-trail_dist)) # <-- sube el stop con ese high
    if lo <= sl_price: exit at sl_price           # <-- y SALE en la misma vela
```

Dentro de **una sola vela** el sim: (1) sube el peak con el HIGH, (2) traila el
stop al nuevo peak menos 0.8%, (3) comprueba si el LOW de la **misma** vela
toca ese stop trailed. Resultado: cada vela volátil con `high-low > trail_dist`
se cierra en `high - 0.8%` — un "vendí en el techo de cada vela menos 0.8%"
que ningún broker puede ejecutar (no conoces el high antes de que ocurra, y la
secuencia interna H↔L dentro de la vela es desconocida).

**Es la causa principal de los PF altos.** Más que el solape de trades.

**Fix (aplicado en `revalidate_v15.py`):** invertir el orden — comprobar la
salida contra el stop que ya estaba antes de la vela; recién después actualizar
peak/stop para la **siguiente** vela.

**Impacto medido (ADA, 12 semestres):**

| | Declarado | Motor solo sin solape | Motor sin solape + sin look-ahead intrabar |
|--|-----------|------------------------|--------------------------------------------|
| WF | 10/12 | 12/12 | **5/12** |
| PF medio | 2.86 LONG, 13.51 SHORT | 4-25 | **0.7-1.9** |
| WR medio | — | 60-83% | **11-38%** |
| Bootstrap | — | p≈0 (falsamente) | **p=0.326 (NO significativo)** |

Con los dos bugs corregidos, ADA — supuesto par "validado" — no sobrevive el
test honesto. Era el bug, no el edge.

---

## 4. Implicación para el plan de re-validación

El forward-OOS (Capa A) **también saldría inflado** si se corre con este motor.
Por tanto, antes de re-validar nada:

> **Paso 0-bis (nuevo, bloqueante): arreglar el motor de backtest.**
> El script `revalidate_v15.py` debe incorporar un guard de posición:
> una vela solo puede abrir un trade nuevo si `idx > bar_en_que_cerró_el_anterior`.
> Es decir: detectar señal → abrir 1 trade → simular hasta su cierra en la vela
> `idx + bars` → reanudar la búsqueda **después** de esa vela. Nunca solapar.

Con ese arreglo:
- Cada par tendrá muchísimos menos trades (los reales, no los clonados).
- Los PF caerán hacia 1.2–2.5 (lo creíble) o por debajo de 1 (rechazo).
- Los DD subirán a valores realistas.
- Será la primera vez que el backtest mida lo que el bot haría.

Además, endurecer umbrales de fold: mínimo de trades por fold mayor (p. ej. ≥5),
y manejar `pf=inf` (cap o exigir ≥1 pérdida para puntuar).

---

## 5. Veredicto del Paso 0

- ✅ **No hay look-ahead** — los features son sólidos, no hay que reconstruirlos.
- ❌ **El motor de backtest está roto** — cuenta trades solapados que el bot no
  puede tomar. Es la causa mecánica de los PF 8–20 y DD 1–4% de `meta_v15.json`.
- 🔧 **Acción**: `revalidate_v15.py` debe simular con guard de una-posición-a-la-vez
  y umbrales de fold realistas. Sin eso, cualquier re-validación repetiría el error.

Buena noticia: el problema es un bug acotado y reparable, no un defecto de fondo
de la estrategia. Recién con el motor corregido sabremos si las reglas tienen
edge real.
