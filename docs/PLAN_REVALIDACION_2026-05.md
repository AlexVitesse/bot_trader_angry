# Plan de Re-Validación V15 — Par por Par

> Objetivo: re-evaluar honestamente los 22 pares de `ML_V15_PAIRS` y decidir cuáles
> merecen seguir, cuáles quedan solo en paper trading, y cuáles se rechazan.
> Fecha: 2026-05-19 · Contexto: `docs/AUDITORIA_2026-05.md`
> Sin capital real — todo paper trading; esto es validación previa, sin prisa.

---

## 1. La ventaja: datos posteriores al cutoff = OOS verdadero

Los modelos/reglas se congelaron en marzo 2026. Todo lo que pasó **después** del
cutoff es información que el backtest **nunca vio** — es out-of-sample real, no
simulado. Ese es el test más honesto posible.

| Grupo | Cutoff / fecha de congelado | Ventana OOS verdadera (hasta hoy) |
|-------|------------------------------|-----------------------------------|
| BTC | 2026-03-01 (entrenado 03-07) | ~11 semanas |
| ETH | 2026-03-23 | ~8 semanas |
| ADA, SOL | 2026-03-24 | ~8 semanas |
| 18 pares ALT | 2026-03-25 | ~8 semanas |

8 semanas en 4h ≈ 330 velas. No es enorme, pero es **suficiente para detectar
overfitting**: si un par declaró PF 18 y en datos nuevos da PF 1.0 o negativo, el
backtest era humo. Cuanto más pase el tiempo, más fuerte el test — se puede repetir
cada mes.

---

## 2. Criterios de aprobación (por par)

Un par se considera **VALIDADO** solo si cumple **todo**:

| Test | Umbral |
|------|--------|
| Forward-OOS (post-cutoff), parámetros congelados | PF ≥ 1.2 **y** retorno positivo |
| Walk-forward **purgado** (gap train/test) | ≥ 7/12 folds positivos |
| Bootstrap de significancia | p < 0.05 (el retorno no es azar) |
| Win Rate | > break-even = SL / (TP+SL) |
| Profit Factor coherente | PF ≤ 4 (PF > 4 = sospecha de overfitting, revisar) |

Resultado por par → una de tres etiquetas:
- **KEEP** — cumple todo. Apto para capital real cuando lo haya.
- **PAPER-ONLY** — cumple WF pero forward-OOS débil o pocos trades. Sigue en
  testnet recogiendo datos, no recibe capital.
- **REJECT** — falla forward-OOS o bootstrap. Se quita de `ML_V15_PAIRS`.

---

## 2-bis. Salvaguardas: que el propio test no nos engañe

El forward-OOS es a prueba de overfitting **para lo que mide** (no se puede
sobreajustar a datos que no existían al congelar los parámetros). Pero quedan 4
trampas; cada una tiene su mitigación obligatoria:

| Trampa | Por qué engaña | Mitigación obligatoria |
|--------|----------------|------------------------|
| **Muestra pequeña** | 8 sem ≈ 330 velas, 5–20 trades; pasar/fallar por suerte | Mínimo **≥ 10 trades** en la ventana OOS; con menos → PAPER-ONLY automático. El bootstrap (Capa C) mide la suerte. |
| **Sesgo de selección de cartera** | Con 22 pares, ~1 pasa por azar a p<0.05; "quedarse con los ganadores" reintroduce el sesgo | Exigir las **3 capas a la vez**. Criterios fijados **antes** de ver resultados — no mover la portería. |
| **Dependencia de régimen** | La ventana OOS fue bajista; un par con sesgo SHORT pasa "gratis" | Anotar el régimen de la ventana. "Validado" no es permanente: **re-test mensual** en ventanas de distinto régimen. |
| **Look-ahead en el código** | Si un feature usa datos del futuro, hasta el forward-OOS sale inflado | **Paso 0** (abajo): auditar el pipeline de features antes de confiar en ningún número. Correr la re-validación por el **mismo código del bot en vivo**. |

Regla de oro: **si el resultado vuelve a salir brillante (PF > 4), desconfiar más,
no menos.** El éxito creíble se parece a BTC (PF 1.35), no a PF 18.

---

## 3. Metodología — 3 capas

### Capa A — Forward-OOS con parámetros congelados (la prueba clave)
1. Descargar datos frescos hasta hoy (`download_new_pairs.py` / `fetch_history.py`).
2. Cortar la serie en la fecha de cutoff del par.
3. Correr la estrategia sobre el tramo **post-cutoff** usando **exactamente** los
   parámetros de `meta_v15.json` — **prohibido re-tunear nada**.
4. Medir: nº trades, WR, PF, retorno, DD.
5. Comparar con el `backtest` declarado en `meta_v15.json`. La degradación esperada
   en un par sano es moderada; un colapso (PF 15 → <1.5) confirma overfitting.

### Capa B — Walk-forward purgado
- Re-correr el WF sobre todo el histórico, pero insertando un **gap de purga**
  (p. ej. 1–2 semanas) entre la ventana de train y la de test, para eliminar fuga
  de información por features con look-ahead.
- Si un par que declaraba 12/12 cae a 5/12 con purga → el WF original estaba sesgado.

### Capa C — Bootstrap de significancia
- Tomar la lista de retornos por trade del par.
- Re-muestrear / barajar 1000+ veces y construir la distribución de PF/retorno.
- Si el resultado real está dentro del ruido (p ≥ 0.05) → el "edge" es suerte.
- Opcional: Monte Carlo barajando el orden de trades para estimar el DD peor caso
  real (los DD de 1–4% declarados son sospechosamente bajos).

---

## 4. Herramienta a construir

Un único script `revalidate_v15.py` que, dado un par:
- Reutiliza la lógica de detección de `src/ml_strategy_v15.py` (no reimplementar
  señales).
- **Corrige el motor de backtest** — ver Paso 0: el motor actual abre trades
  solapados (`evaluate_new_pairs_v15.py` y `v15_framework.py` no modelan "una
  posición a la vez"). El nuevo simulador DEBE: detectar señal → abrir 1 trade →
  simular hasta su cierre → reanudar la búsqueda **después** de la vela de cierre.
  Sin solapar nunca. Es la causa mecánica de los PF 8–20 de `meta_v15.json`.
- Endurece umbrales de fold: mínimo ≥5 trades por fold; manejar `pf=inf`.
- Ejecuta las 3 capas (A, B, C).
- Imprime una ficha por par y la guarda en `docs/revalidation/{PAR}.md`.
- Se ejecuta con el Python de producción (sklearn 1.8.0) — ver `CLAUDE.md`.

Modo de uso previsto: `python revalidate_v15.py --pair BTC` y `--all`.

> El resultado del Paso 0 está en `docs/revalidation/PASO0_lookahead.md`: los
> **features están limpios** (sin look-ahead), pero el **motor de backtest está
> roto**. Corregirlo es el primer trabajo de código, antes de re-validar pares.

---

## 5. Orden de ejecución (par por par)

**Paso 0 — Auditar el pipeline de features por look-ahead (ANTES de todo)**
Si un feature usa datos del futuro, todos los números de abajo son humo. Revisar
`v15_features.py`, `v15_data_pipeline.py`, `v15_framework.py` y la generación de
señales en `src/ml_strategy_v15.py`:
- Indicadores (EMA, RSI, ATR, BB, ADX) calculados solo con velas pasadas/cerradas.
- Features multi-timeframe (1D sobre 4h): `.shift(1)` **antes** del `reindex/ffill`
  (bug histórico del proyecto — ver `MEMORY`).
- Labels/targets que no filtren al conjunto de features.
- El split de WF: el test nunca toca velas usadas en train (de ahí la purga).
- La señal en backtest se decide con la vela **cerrada**, no la actual en curso.
Resultado del Paso 0 → `docs/revalidation/PASO0_lookahead.md`. Si aparece un bug,
se corrige antes de re-validar nada.

**Fase 1 — Calibrar la metodología con los pares de confianza**
BTC → ETH → ADA → SOL. Sabemos que tienen métricas modestas y creíbles. Si el
script les da forward-OOS razonable (PF 1.2–2.0), la metodología es correcta y
podemos confiar en ella para el resto.

**Fase 2 — Los 18 pares sin documento** (en bloques de ~4-5):
- Bloque 1: DOGE, DOT, LINK, XRP, ATOM
- Bloque 2: AVAX, NEAR, INJ, ALGO, FIL
- Bloque 3: 1000SHIB, BNB, LTC, ETC, BCH
- Bloque 4: UNI, AAVE, OP
Para cada uno: ficha completa + etiqueta KEEP / PAPER-ONLY / REJECT.

**Fase 3 — Cruzar con paper trading real**
El bot lleva semanas en testnet. Extraer sus trades reales (DB / logs) y
compararlos contra lo que el backtest predecía para esas mismas velas. Es una
cuarta fuente de verdad, independiente del backtest.

**Fase 4 — Decisión y limpieza**
- Actualizar `ML_V15_PAIRS` dejando solo KEEP + PAPER-ONLY.
- Reescribir cada `meta_v15.json` con el bloque `backtest` honesto (forward-OOS).
- Documento resumen `docs/REVALIDACION_RESULTADOS.md` con la tabla final.
- Actualizar `CLAUDE.md` con la lista depurada.

---

## 6. Entregable por par (ficha tipo)

```
PAR: XRP/USDT
Cutoff: 2026-03-25 | Ventana OOS: 2026-03-25 → 2026-05-19 (8 sem)

A. Forward-OOS (params congelados)
   Trades: N | WR: % | PF: x.xx | Retorno: % | DD: %
   Backtest declaró: PF 4.36 LONG / 18.80 SHORT  → degradación: ___

B. Walk-forward purgado:  __/12 folds   (declarado: 12/12)
C. Bootstrap:             p = ___       (significativo: sí/no)

VEREDICTO: KEEP / PAPER-ONLY / REJECT
Notas: ___
```

---

## 7. Qué NO hacer (errores a evitar)

- ❌ Re-tunear parámetros para que el forward-OOS "pase" — eso reintroduce el
  overfitting. Los parámetros están congelados; solo se mide.
- ❌ Elegir la ventana OOS para que favorezca. Es fija: cutoff → hoy.
- ❌ Aprobar un par por WF solo. WF sin purga ya falló (auditoría sección B).
- ❌ Rescatar un par REJECT con "un umbral más estricto" — regla de
  `METODOLOGIA_TESTING.md`: se rechaza o se reentrena, no se parchea.

---

## 8. Resumen

El plan convierte una debilidad (18 pares con backtest sospechoso) en una prueba
limpia: **dejar que el tiempo valide**. Los datos posteriores al cutoff son un
juez imparcial. Sin capital en riesgo y sin prisa, es el momento ideal para
hacerlo bien — par por par, con criterios fijos y sin auto-engaño.
