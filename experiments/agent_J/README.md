# Agent J — ETH/USDT 1D Trend-Following (Rescaled-A)

> Long-only. Cutoff inviolable 2025-12-31. **Cero re-tuneo** de params para
> ETH: las ventanas se ESCALARON proporcionalmente del 4h al 1D, definidas
> *a priori* antes de ver resultados. Funding desactivado (no hay
> `ethusdt_funding.parquet` en `data/`).

---

## TL;DR — VEREDICTO: REJECT (1/4 criterios)

| Criterio | Umbral | ETH-J 1D | Pasa? |
|----------|--------|---------:|:-----:|
| Bootstrap p<0.05 (real ETH) | < 0.05 | **0.383** | NO |
| Mediana sintético > 0 | > 0 | +0.9% | SÍ |
| ≥14/20 sintéticas positivas | ≥ 14 | 11/20 | NO |
| Edge vs null > 5% | > 5% | **−6.4%** | NO |

**Hipótesis falsada.** ETH 1D **NO tiene ciclos más limpios que 4h** para
la mecánica de A. De hecho, el resultado es peor en todas las métricas
comparado con ETH-A 4h (anual +2.5% vs +11.3%; bootstrap p 0.383 vs 0.103).
Además, el "edge vs null" es **negativo** (−6.4%): la null hypothesis
(shuffle aleatorio) produjo mejores resultados medianos (+7.2%) que las
sintéticas con estructura real (+0.9%). Eso es la firma de un mecanismo
que **no aporta valor sobre azar** — peor aún, sufre por la estructura real.

---

## Resultado in-sample 2020-2025

| Métrica | ETH-J 1D | ETH-A 4h (ref) | BTC-A 4h (ref) |
|---|---:|---:|---:|
| Trades | **31** | 87 | 74 |
| Win rate | **38.7%** | 46.0% | 44.1% |
| Profit factor | **1.27** | 1.53 | 1.66 |
| Retorno total | **+13.0%** | +75% est. | +75% est. |
| Anual | **+2.5%** | +11.3% | +11.1% |
| Drawdown máximo | **20.3%** | 18.2% | 16.8% |
| Bootstrap p (3000 iter) | **0.383** | 0.103 | 0.088 |

**31 trades en 6 años = ~5 trades/año.** Frecuencia bajísima — el
Donchian-10 en 1D casi nunca encuentra rupturas que pasen también los
filtros de volumen + ADX + bull macro.

---

## Walk-forward (12 semestres, gap purga 14 d)

| Semestre | Trades | WR | PF | Total | Mensual | DD | OK |
|---|---:|---:|---:|---:|---:|---:|---|
| 2020-01 | 0 | — | — | 0 | 0 | 0 | *no-signal* |
| 2020-07 | 6 | 33.3 % | 0.35 | −8.2 % | −3.70 % | 10.5 % | ❌ |
| 2021-01 | 2 | 50.0 % | 0.64 | −2.4 % | −2.45 % | 6.1 % | *small-n* |
| 2021-07 | 5 | 40.0 % | 1.57 | +7.2 % | +1.63 % | 11.8 % | ✅ |
| 2022-01 | 0 | — | — | 0 | 0 | 0 | *no-signal* |
| 2022-07 | 0 | — | — | 0 | 0 | 0 | *no-signal* |
| 2023-01 | 3 | 33.3 % | 0.46 | −2.1 % | −0.60 % | 3.8 % | ❌ |
| 2023-07 | 4 | 25.0 % | 0.04 | −13.6 % | −3.54 % | 13.6 % | ❌ |
| 2024-01 | 4 | 50.0 % | 2.79 | +12.0 % | +3.41 % | 6.8 % | ✅ |
| 2024-07 | 1 | 100 % | inf | +0.0 % | +0.02 % | 0.0 % | *small-n* |
| 2025-01 | 0 | — | — | 0 | 0 | 0 | *no-signal* |
| 2025-07 | 4 | 25.0 % | 2.34 | +8.0 % | +2.79 % | 6.5 % | ✅ |

- **3/6 folds evaluados OK** (debajo del umbral del proyecto 7/12 o 6/10).
- 4 folds *no-signal* (filtro daily filtrando bear correctamente — es FEATURE).
- 2 folds *small-n* (1-2 trades, indeterminado).
- **PF mediano de folds evaluados: 1.02** (esencialmente break-even).
- Pésimo 2023-07: 4 trades, 25% WR, PF 0.04 — un solo ganador minúsculo
  contra 3 perdedores grandes. ETH cayó en consolidación choppy.

---

## Sintéticas (20 series, block bootstrap bloques 30 d)

| Estadístico | Valor |
|---|---:|
| Mediana annual | +0.9% |
| Media annual | +3.3% |
| p25–p75 | [−4.3%, +9.2%] |
| p5–p95 | [−10.5%, +22.0%] |
| # series con annual > 0 | **11/20 (55%)** |
| # series con annual > 10% | 4/20 (20%) |
| Real ETH (+2.5%) en p5–p95 | Sí (centrado) |

Solo 11/20 positivas — borderline coin flip. ETH-A 4h dio 15/20 (75%),
lo cual es mejor (aunque tampoco pasó significancia).

## Null hypothesis (shuffle log-returns)

| Estadístico | Valor |
|---|---:|
| Mediana annual (null) | **+7.2%** |
| Media annual (null) | +4.8% |
| Mediana sintético − null | **−6.4%** |

**El null hypothesis dio MEJOR resultado mediano (+7.2%) que el sintético
estructurado (+0.9%).** Eso es lo opuesto al patrón esperado para un
mecanismo con edge: si la estructura temporal real fuera útil, debería
producir mejores resultados que la versión "ruido puro". Aquí no:
el shuffle aleatorio supera al sintético. La interpretación honesta es
que **Donchian-10 + EMA50/200 en 1D no capturan ningún edge sobre ruido
en ETH**.

(Diagnóstico secundario: el null bootstrap produce más trades por serie
~40-50 vs sintético ~25-45, lo cual sugiere que los breakouts "limpios"
de Donchian son más frecuentes en datos shuffleados que en datos reales
con momentum/anti-momentum reales).

---

## Comparativa con ETH-A 4h (la hipótesis a falsar)

| Métrica | ETH-J 1D | ETH-A 4h | Veredicto |
|---|---:|---:|---|
| N trades | 31 | 87 | 1D es muchísimo más selectivo (−64%) |
| WR | 38.7% | 46.0% | 1D peor (−7.3pp) |
| PF | 1.27 | 1.53 | 1D peor |
| Annual | +2.5% | +11.3% | 1D mucho peor (−8.8pp) |
| Max DD | 20.3% | 18.2% | 1D ligeramente peor |
| Bootstrap p | 0.383 | 0.103 | 1D mucho peor |
| Sintético >0 | 11/20 | 15/20 | 1D peor |

**Conclusión empírica:** la hipótesis "1D tiene ciclos más limpios que
4h en ETH" **es falsa** para esta mecánica trend-follower. En 1D pasa
lo opuesto: la menor frecuencia de muestreo destruye el muestreo
estadístico (31 trades vs 87) sin mejorar la calidad por trade. Cada
breakout 1D que sí cumple los filtros es **menos rentable en promedio**
que un breakout 4h equivalente.

---

## Por qué falla 1D — hipótesis con datos

1. **Demasiado pocos trades**. 31 trades en 6 años no permiten
   significancia estadística (bootstrap p=0.38). Para subir significancia,
   habría que bajar `donchian_n` a 5-6 — pero eso ya es overfitting.

2. **ATR% en 1D es 3-5%**, así que `trail_atr_mult = 2.5` × ATR%
   produce trail_dist típica de 8-12%, que CHOCA con el ceiling de 6%.
   El ceiling bind hace que los stops en 1D sean efectivamente fijos al
   6%, perdiendo la adaptabilidad ATR. No re-tuneé el ceiling porque eso
   sería data-aware.

3. **Filtro EMA50/200 + ADX 18 en 1D es demasiado restrictivo**. Aplicado
   sobre la propia 1D (no MTF), el conjunto de bars que pasan TODOS los
   filtros (bull + Donchian + vol≥1.2 + ADX≥18) es minúsculo. En 4h hay
   muchas más oportunidades de breakouts intra-tendencia daily.

4. **Choppy de 2023 demuele 1D peor que 4h**. En 2023-07, ETH lateralizó.
   Un Donchian-10 en 1D dispara 4 trades, 3 de ellos cierran en SL casi
   inmediato (PF 0.04). El 4h tendría más oportunidades de revertir y
   compensar.

5. **El null hypothesis lo hace mejor**: indicio claro de que el
   mecanismo capta MENOS información en 1D que el ruido shuffleado.
   Posible explicación: el shuffle elimina la auto-correlación negativa
   intraciclo de ETH (mean-reversion natural post-breakout daily), mientras
   que la serie real preserva ese ruido adverso al breakout.

---

## SELF-AUDIT — auditoría honesta de decisiones

### Decisiones tomadas SIN mirar resultados (defensa principal)

1. **Rescaling de windows DEFINIDO A PRIORI**: ver tabla en `strategy.py`.
   Donchian-10 = mismo horizonte temporal que Donchian-55 en 4h (~10 días).
   max_bars=10 = mismo techo temporal. Resto de parámetros idénticos.
   **Cero parámetros tuneados a ETH 1D**.
2. **Trailing floor/ceiling igual al de A**: 2.5%/6%. Sabía que en 1D
   el ATR% sería mayor y el ceiling podría bind. Mantenido por honestidad
   (no tunear para que el resultado sea "mejor").
3. **Funding desactivado**: no hay parquet de ETH funding, re-usar el de
   BTC sería un hack. La alternativa "honesta" es operar sin ese veto.
4. **Mismo simulador que A**: copia exacta, mismos bug-guards (una
   posición a la vez, sin look-ahead intrabar). Validado contra el
   código de A.
5. **Min trades por fold (n≥3) y umbral PF≥1.2 idénticos a A**.

### Decisiones que sí miran datos (riesgos identificados)

- **Long-only**: viene del histórico del proyecto (SHORT en BTC fracaso).
  ETH SHORT no probado en este test — y no lo añado para evitar selection
  bias.
- **Bull filter EMA50/200 daily**: clásico, justificado a priori.
- **Cutoff 2025-12-31**: dictado por el experimento.

### Lo que NO hice (defensa contra overfitting)

- **No probé múltiples valores de `donchian_n`** (8, 10, 12, 15, 20)
  para reportar el mejor. Un solo conjunto de parámetros, una sola
  corrida, resultado reportado tal cual.
- **No ajusté trail_floor/ceiling** aunque sabía que el ceiling 6% bind.
- **No deshabilité ADX** aunque solo 31 trades sugiere que es demasiado
  restrictivo en 1D.
- **No probé donchian sin el filtro bull**, ni viceversa.

### Bugs evitados (mismos que A)

1. ✅ Una posición a la vez: `i += max(1, bars) + 1` tras cada trade.
2. ✅ Sin look-ahead intrabar en trailing: SL comprobado contra peak
   heredado antes de actualizar peak con `high[b]`.
3. ✅ MTF features con shift(1): `bull_1d`, `donchian_high` con `.shift(1)`.
4. ✅ Sin tuneo: parámetros frozen antes de ver resultados.
5. ✅ Cutoff respetado: `df = df[df.index <= '2025-12-31']` con assert.

---

## Implicación para el proyecto

ETH 1D **se suma a los 6 enfoques ETH ya rechazados** (`VERDICTO_ETH.md`).
La conclusión global del proyecto se refuerza:

> **ETH no es operable con metodología honesta en ningún timeframe probado
> (4h ni 1D) con ningún mecanismo convencional (trend, mean-rev, ML, ratio,
> vol-breakout).**

Total acumulado de enfoques ETH rechazados: **7** (ETH-A 4h, ETH-F 4h,
ETH-V2 4h, ETH-G ML, ETH-H ratio, ETH-I MR-RANGE, ETH-J 1D).

**Recomendación**: cerrar el bucle ETH definitivamente. Cualquier nuevo
intento de ETH necesita un **mecanismo genuinamente nuevo** (no
trend-follower clásico) y respetando el protocolo completo. Las opciones
restantes según `VERDICTO_ETH.md` son:
- ETH/BTC ratio en daily/weekly (no testeado a fondo)
- Macro/news-driven (no backtest-able honestamente)
- Productos no-direccionales (spot DCA, staking) — fuera del scope del bot

---

## Cómo ejecutar

```bash
C:/Python/python.exe experiments/agent_J/train.py
```

Salida: console + `experiments/agent_J/results.json`.

---

## Datos usados

- `data/ETH_USDT_1d_history.parquet` (2191 bars 1D, cortado a ≤ 2025-12-31)
- Sin MTF (ya es 1D)
- Sin funding (no hay ETH funding parquet)

NO se consultaron datos posteriores a 2025-12-31.

---

## Archivos

```
experiments/agent_J/
├── strategy.py    # PARAMS frozen, signal(), simulate(), run_backtest()
├── train.py       # 4 capas: WF + bootstrap + sintético + null + comp 4h
├── results.json   # output completo de train.py
└── README.md      # este archivo
```
