# Auditoría del Proyecto — Mayo 2026

> Revisión a fondo del estado real del bot V15, qué hacemos bien, qué hacemos mal,
> y qué aprender de proyectos referentes de GitHub.
> Fecha: 2026-05-19 · Rama auditada: `v15/multi-pair`

---

## A. Estado real vs. documentado

### A.1 Lo que dice cada fuente

| Fuente | Afirma | Realidad |
|--------|--------|----------|
| `README.md` | "Estrategia Actual: V13 CORE", 8 pares, backtest 64% WR / PF 3.41 | **Obsoleto.** V13 ya no corre. |
| `CLAUDE.md` (previo) | "V15 opera **solo BTC/USDT**", rama `v15/momentum-breakout` | **Desactualizado.** |
| `config/settings.py` | `ML_V15_PAIRS` = **22 pares**, `ML_V15_ENABLED=True` | ✅ Estado real. |
| `MEMORY.md` | "BTC + ETH deployed, ADA + SOL aprobado" | Parcial: faltan 16 pares más. |
| Rama git | `v15/multi-pair`, commit `f4cea1e` "deploy 22-pair multi-asset" | ✅ Estado real. |

**Conclusión:** el bot opera **22 pares** en paper trading, no 1 ni 4. La
documentación de cabecera estaba 2-3 iteraciones por detrás del código.

### A.2 Arquitectura real V15 (22 pares)

| Tipo | Pares | Lógica | model_type |
|------|-------|--------|------------|
| **ML** | BTC (1) | GBM SHORT (threshold 0.60) + Breakout B / Pullback EMA20 LONG | `GradientBoostingClassifier` |
| **Reglas ETH** | ETH (1) | BTC-follower LONG + Breakout + SHORT multi-conf/BB | `rule_based` |
| **Reglas trailing** | ADA, SOL, DOGE, LINK, AVAX, DOT, NEAR, XRP, ATOM, INJ, ALGO, FIL, 1000SHIB, BNB, LTC, ETC, BCH, UNI, AAVE, OP (20) | BTC-follower LONG + BTC-breakdown SHORT, ambos con trailing stop tight | `rule_based_trailing` |

Solo BTC usa ML. Los otros 21 pares son 100% reglas.

### A.3 Backtest declarado por par (`meta_v15.json`)

| Par | LONG WF | LONG PF | SHORT WF | SHORT PF | DD | OOS 2026 comb. | Doc dedicado |
|-----|---------|---------|----------|----------|-----|----------------|--------------|
| BTC | 8/12 | 1.35 (OOS) | — | — | 35% | — | ✅ V15_COMMITTEE |
| ETH | 8/12 | 1.28 (OOS) | — | — | 43% | +16.2% | ✅ V15_ETH_eval |
| ADA | 10/12 | 2.86 | 10/10 | 13.51 | 7% | +32.3% | ✅ V15_ADA_eval |
| SOL | 8/10 | 2.56 | 7/8 | 15.04 | 10% | +21.1% | ✅ V15_SOL_eval |
| DOGE | 12/12 | 7.51 | 11/11 | 12.73 | 4% | +38.0% | ❌ |
| DOT | 11/11 | 7.62 | 10/10 | 13.90 | 3% | +46.6% | ❌ |
| LINK | 12/12 | 7.40 | 10/11 | 15.02 | 3% | +26.4% | ❌ |
| XRP | 12/12 | 4.36 | 9/11 | 18.80 | 4% | +36.2% | ❌ |
| AVAX | 10/11 | 8.33 | 11/11 | 15.43 | 3% | +36.4% | ❌ |
| NEAR | 11/11 | 11.05 | 10/10 | 16.80 | 3% | +49.2% | ❌ |
| ATOM | 12/12 | 7.85 | 11/11 | 14.81 | 3% | +19.3% | ❌ |
| INJ | 10/10 | 8.89 | 8/9 | 18.96 | 3% | +47.1% | ❌ |
| ALGO | 12/12 | 6.62 | 12/12 | 15.36 | 3% | +39.8% | ❌ |
| FIL | 9/10 | 9.88 | 10/11 | 18.43 | 3% | +39.8% | ❌ |
| 1000SHIB | 9/9 | 11.68 | 8/10 | 17.08 | 2% | +42.0% | ❌ |
| BNB | 11/12 | 2.85 | 8/10 | 8.64 | 5% | +19.5% | ❌ |
| LTC | 12/12 | 7.10 | 11/12 | 18.21 | 3% | +35.1% | ❌ |
| ETC | 12/12 | 8.27 | 11/11 | 17.12 | 3% | +36.6% | ❌ |
| BCH | 12/12 | 7.86 | 11/11 | 20.13 | 4% | +19.5% | ❌ |
| UNI | 11/11 | 8.34 | 10/11 | 20.17 | 2% | +40.8% | ❌ |
| AAVE | 11/11 | 6.99 | 10/10 | 19.76 | 4% | +39.7% | ❌ |
| OP | 7/7 | 14.91 | 7/7 | 13.13 | 2% | +46.7% | ❌ |

---

## B. Hallazgo crítico: el patrón de overfitting se está repitiendo

CLAUDE.md documenta una historia explícita: **cada versión con backtest brillante
falló en producción** (V7, V9, BTC V2, SOL V2, V13.03). La regla del proyecto es:

> "Si el backtest muestra 70%+ WR → sospechar overfitting." · "PF realista 1.2-1.5."
> "Un modelo con métricas malas NO se arregla con umbral más estricto."

**Los 18 pares añadidos sin documento muestran exactamente la firma de overfitting
que el proyecto juró evitar:**

1. **PF imposibles.** SHORT PF de 13–20 y LONG PF de 7–15. El propio BTC validado
   tiene OOS PF=1.35. Una estrategia real de crypto no produce PF 20. V13.03 fue
   rechazado por overfitting con PF 3.98 — estos pares declaran 4–6× peor señal.
2. **Drawdown irreal.** DD de 1–4% en estrategias direccionales de crypto en 4h.
   BTC validado tiene DD 35%. Un DD de 2% no es robustez, es ausencia de muestra
   adversa — el backtest nunca vio una racha mala.
3. **Walk-forward ~100%.** 12/12, 11/11, 10/10, 7/7. El umbral del proyecto es
   ≥7/12. Pasar 12/12 con PF 20 no es señal de calidad: es señal de que el WF no
   está separando train de test (sin gap de purga / posible look-ahead).
4. **Sesgo de selección.** Existe `evaluate_new_pairs_v15.py`. El patrón típico:
   se corre la MISMA plantilla de reglas sobre decenas de pares y se conservan los
   que "pasaron". Eso es minería de datos — los 16 ganadores se eligieron *después*
   de ver sus resultados.
5. **Plantilla idéntica para 20 pares.** OP, BNB, INJ, ALGO, NEAR, ETC, UNI,
   1000SHIB... tienen `meta_v15.json` **byte-idénticos** salvo `asset` y `backtest`.
   Ningún par fue ajustado individualmente. CLAUDE.md exige: "Adaptar
   features/thresholds por par, no reutilizar modelos BTC directamente."

**Contraste revelador:** ADA y SOL — los únicos ALT con evaluación documentada y
escéptica — muestran LONG PF 2.5–2.9 y OOS LONG **negativo** (-1% a -2%). Los 16
pares sin documento muestran LONG PF 7–15 y OOS positivo. Mismo motor de reglas,
resultados 3× mejores. La diferencia no es el mercado: es que ADA/SOL fueron
medidos con escepticismo y los demás no.

**El edge declarado vive 100% en el SHORT.** Las columnas OOS LONG son +1% a +5%;
las OOS SHORT son +18% a +49%. Es decir: todo el rendimiento multi-par depende de
trades SHORT con PF 13–20 — y CLAUDE.md dice literalmente *"SHORT Direction: No
Aprobado para Altcoins"*, *"SHORT trades = 0% WR en períodos alcistas"*. El
"funciona" actual viene de que la ventana OOS (Ene-Mar 2026) fue un mercado
bajista. No hay evidencia de que el SHORT sobreviva un tramo alcista.

> **Veredicto:** los 18 pares sin doc (DOGE, DOT, LINK, XRP, AVAX, NEAR, ATOM,
> INJ, ALGO, FIL, 1000SHIB, BNB, LTC, ETC, BCH, UNI, AAVE, OP) **no cumplen los
> requisitos de validación del propio proyecto.** No deben considerarse validados
> para capital real. Operarlos en paper trading testnet es aceptable y útil
> (recolecta datos), pero el backtest que los justificó no es creíble.

---

## C. Qué hacemos BIEN

1. **Autoconciencia del overfitting.** `ANALISIS_CRITICO_OVERFITTING.md` y los
   post-mortems (`POST_MORTEM_V14.1_BIDIRECTIONAL.md`) son honestos y valiosos.
   Pocos proyectos documentan sus fracasos así.
2. **Disciplina de walk-forward como concepto.** El umbral ≥7/12 y la exigencia de
   cross-asset están bien definidos en `METODOLOGIA_TESTING.md`. El problema es la
   *aplicación*, no la idea.
3. **Filosofía correcta.** "El edge está en la SALIDA (trailing stop), no en la
   entrada" — coincide con el único éxito real histórico (V7: trailing stop,
   322% anual). Apostar por trailing stops sobre predicción ML pura es sensato.
4. **Detección de régimen.** Clasificar BULL/BEAR/RANGE por EMAs diarias y cambiar
   de setup por régimen es lo que hacen los traders reales.
5. **Gestión de riesgo en el bot.** `ML_MAX_CONCURRENT=3`, límites de drawdown y
   pérdida diaria, funding veto, kill switches, alertas Telegram — el lado de
   *ejecución* está maduro y bien construido.
6. **BTC y ETH están bien validados.** WF 8/12, PF 1.28–1.35, DD 35–43%. Métricas
   creíbles y modestas — justo lo que el proyecto dice buscar.

## D. Qué hacemos MAL

1. **Se rompió la propia disciplina anti-overfitting** (sección B). Es el error más
   grave: el proyecto sabe identificar overfitting y aun así desplegó 18 pares con
   su firma clara.
2. **Documentación como única fuente de verdad, rota.** README→V13, CLAUDE.md→
   "BTC only", settings→22 pares, MEMORY→4 pares. Cuatro versiones de la "verdad".
3. **Métricas de backtest sobrevendidas.** README declara PF 3.41 en un backtest de
   14 días/161 trades — muestra demasiado corta para significar nada.
4. **WF sin purga.** Pasar 12/12 con PF 20 sugiere que train y test comparten
   información (sin gap entre ventanas, o features con look-ahead). Un WF que nunca
   falla no está midiendo generalización.
5. **Sesgo de selección no controlado.** Elegir pares *después* de ver su backtest
   es look-ahead a nivel de cartera.
6. **Caos de archivos.** ~92 scripts en la raíz, ~35 sin commitear, basura (`nul`,
   `ml_bot.log` 1 MB, `__pycache__`, `catboost_info`). Imposible saber qué está vivo.
7. **Insistencia con ETH y SHORT** pese a fracasos repetidos — aunque esta vez ETH
   sí logró una validación creíble (WF 8/12).
8. **Sin test de significancia estadística.** Nada distingue "edge real" de "racha
   afortunada en la ventana OOS".

---

## E. Inspiración: proyectos referentes de GitHub

Tres proyectos open-source maduros resuelven exactamente los problemas de la
sección D. No se trata de copiar código, sino de adoptar su **metodología**.

### E.1 Freqtrade + FreqAI (~25k ⭐)
El bot de crypto más usado. Su módulo ML, **FreqAI**, aporta:
- **Retraining adaptativo continuo**: el modelo se reentrena sobre una ventana
  móvil en vivo, en lugar de un único entrenamiento estático que memoriza
  2020-2024 (la causa raíz citada en `ANALISIS_CRITICO_OVERFITTING.md`).
- **Normalización estadísticamente segura** y **PCA** para reducir dimensionalidad
  — combate el overfitting por exceso de features.
- **Purga de modelos viejos** y backtesting integrado.
- **Idea a adoptar:** *purged walk-forward* — insertar un gap temporal entre la
  ventana de train y la de test para eliminar fugas de datos. Si nuestro WF
  pasara a fallar con esto, confirmaría el diagnóstico de la sección B.

### E.2 Jesse (jesse-ai/jesse)
Framework Python de research/backtest enfocado en **rigor**:
- **Backtester sin look-ahead bias por diseño** — el grid search histórico de
  TP/SL que el proyecto ya reconoce como look-ahead bias clásico desaparecería.
- **Bootstrap resampling**: determina si el retorno histórico de una regla pudo
  aparecer por azar.
- **Trade-order shuffling / Monte Carlo**: estresa la estrategia barajando el
  orden de los trades para distinguir habilidad de suerte.
- **Optimización con algoritmos genéticos** en vez de grid search exhaustivo
  (menos sobreajuste a la rejilla).
- **Idea a adoptar:** antes de aprobar cualquier par, exigir un **test de
  significancia** (bootstrap): si el PF 20 de UNI no sobrevive al resampling, se
  rechaza. Esto habría bloqueado los 18 pares dudosos.

### E.3 OctoBot (Drakkar-Software/OctoBot)
Bot con 20k+ usuarios, fuerte en estrategias **simples y robustas**:
- DCA y Grid trading — estrategias sin ML, difíciles de sobreajustar.
- Backtesting con portafolio simulado sobre periodos largos.
- **Idea a adoptar:** mantener un *baseline* simple (DCA/Grid o buy&hold) contra
  el cual comparar. Si la estrategia ML/reglas no lo supera con holgura tras
  costes, no justifica su complejidad ni el riesgo.

### E.4 Recomendación concreta
El proyecto tiene un motor de **ejecución** sólido (sección C.5) pero un motor de
**validación** que produce overfitting. La opción de mayor impacto:

> **Migrar la capa de backtest/validación a Jesse** (o adoptar su metodología:
> sin look-ahead, bootstrap, Monte Carlo). Mantener `ml_bot.py` /
> `portfolio_manager.py` como capa de ejecución en vivo. Así cada par nuevo se
> aprueba solo si supera un test estadístico, no un grid search.

---

## F. Acciones recomendadas (priorizadas)

1. **No mover capital real** hasta re-validar. Paper trading testnet de los 22
   pares es correcto y útil (acumula datos reales).
2. **Re-validar los 18 pares sin doc** con WF purgado (gap train/test) + bootstrap.
   Esperar que la mayoría caiga a PF 1.2–2.0 o sea rechazada. Esto NO es opcional:
   es aplicar la regla que el proyecto ya tiene escrita.
3. **Tratar el SHORT de altcoins como no probado** hasta verlo sobrevivir un tramo
   alcista. El OOS positivo actual coincide con un mercado bajista.
4. **Una sola fuente de verdad.** CLAUDE.md actualizado (hecho en esta auditoría) +
   este documento. README reescrito.
5. **Documentar cada par** o reducir `ML_V15_PAIRS` a los que tengan doc creíble
   (BTC, ETH, ADA, SOL).
6. **Adoptar bootstrap/Monte Carlo** (estilo Jesse) como gate obligatorio antes de
   aprobar cualquier par — además del WF.
7. **Limpieza de repositorio** (ejecutada en esta auditoría: ver sección Estructura
   en CLAUDE.md).

---

## Fuentes

- [Freqtrade](https://github.com/freqtrade/freqtrade) ·
  [FreqAI docs](https://www.freqtrade.io/en/stable/freqai/)
- [Jesse](https://github.com/jesse-ai/jesse) · [jesse.trade](https://jesse.trade/)
- [OctoBot](https://github.com/Drakkar-Software/OctoBot)
- Walk-forward / overfitting:
  [QuantInsti](https://blog.quantinsti.com/walk-forward-optimization-introduction/),
  [Blockchain Council](https://www.blockchain-council.org/cryptocurrency/backtesting-ai-crypto-trading-strategies-avoiding-overfitting-lookahead-bias-data-leakage/)
- Documentos internos: `ANALISIS_CRITICO_OVERFITTING.md`, `METODOLOGIA_TESTING.md`,
  `POST_MORTEM_V14.1_BIDIRECTIONAL.md`, `LOW_OVERFIT_MODEL_RESULTS.md`,
  `meta_v15.json` de los 22 pares.
