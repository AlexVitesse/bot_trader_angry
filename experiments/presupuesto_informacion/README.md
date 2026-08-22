# Presupuesto de información — por qué ningún modelo más complejo puede funcionar

> Fecha: 2026-08-22 · `test_presupuesto.py` · BTC 2019-09 → 2026-08 (datos vivos
> pegados al parquet de ajuste)

Los siete negativos del proyecto explican **qué** no funciona. Éste explica
**por qué no puede funcionar**, que es lo que evita el octavo intento.

---

## A) Las 15.187 velas son una ilusión

Un modelo ML "ve" 15.187 velas de 4h. Pero las features están casi congeladas,
así que el número de observaciones **independientes** es otro:

| feature | ρ (autocorr lag-1) | N efectivo |
|---|--:|--:|
| `bull_1d` | 0,999 | **11** |
| `adx` | 0,995 | **35** |
| `atr_pct` | 0,993 | **50** |
| `bb_width` | 0,988 | **90** |
| `vol_ratio` | 0,386 | 6.731 |

`N_eff = N·(1−ρ)/(1+ρ)`. Un ADX con autocorrelación 0,995 aporta **35**
observaciones, no 15.187. El régimen aporta **11**.

Y lo que de verdad quieres predecir no son velas, son **trades**:

```
episodios de régimen distintos en 6,5 años:   11
TRADES:                                      131   (59 ganan, 72 pierden)
```

### El contraste que lo decide

| | parámetros | eventos/parámetro |
|---|--:|--:|
| **V2 (`PARAMS_V2`)** | **10** | **13,1** |
| LightGBM 100×31 | 6.200 | 0,0211 |
| LightGBM 500×31 | 31.000 | 0,0042 |
| LightGBM 1000×63 | 126.000 | 0,0010 |

Un LightGBM modesto está **3.000 veces sobresuscrito**. Le estás pidiendo que
estime 31.000 números con 131 observaciones. Con esa proporción no hay
metodología que salve nada: el `train AUC 0,805 → test AUC 0,513` de
`agent_G/` no es un fallo de validación, es el resultado aritméticamente
esperado.

**El cuello de botella no es la capacidad del modelo, es la información.**
Añadir capacidad sobre un presupuesto fijo de información solo compra más
overfitting — que es literalmente la tabla de fracasos de `CLAUDE.md`.

### Lo que ya se probó, para no repetirlo

GradientBoosting · RandomForest · Ridge · LightGBM · LogisticRegression ·
XGBoost · CatBoost · ExtraTrees · ElasticNet · StackingClassifier · **LSTM**
(`docs/archive/EXPERIMENTO_BTC_V2.md`: 60 trades, 16,7% WR, **−44,65%**).

Con validación honesta: `agent_B/` (BTC, purged CV) → test AUC 0,520,
bootstrap **p=0,607**, PnL agregado **−16,0%**. `agent_G/` (ETH, purged CV,
con ratio ETH/BTC) → train AUC 0,805, **test AUC 0,513**, 2/11 folds.

---

## B) Los 5 meses que faltaban: sí son distintos, y eso lo empeora

Objeción legítima: el parquet de ajuste acaba el 2026-03-04 y los 170 días
siguientes no se parecen a lo anterior. **Es cierto y está medido** — los
cuatro features se desplazan de forma significativa:

| feature | mediana antes | mediana ahora | KS p |
|---|--:|--:|--:|
| `atr_pct` | 0,0160 | 0,0118 | 1,5e−96 |
| `bb_width` | 0,0628 | 0,0466 | 5,5e−47 |
| `adx` | 25,83 | 25,32 | 1,3e−03 |
| `vol_ratio` | 0,828 | 0,818 | 5,7e−03 |

**La volatilidad anualizada cayó del 60,8% al 38,8%** — un tercio menos. BTC se
ha vuelto un activo notablemente más tranquilo. El mercado sí cambió de
carácter, no solo de dirección.

Pero la información **nueva** que aportan esos 5 meses es:

```
episodios de régimen nuevos:  1
trades nuevos de V2:          0
```

Y `N_eff` de `adx` pasa de 32 a **35**; el de `bull_1d` se queda en **11**.

> Las dos cosas son ciertas a la vez, y su combinación es el peor caso posible
> para un modelo complejo: **la distribución se ha movido Y no hay muestras
> nuevas con las que aprender la distribución nueva.** Eso es no-estacionariedad
> con datos insuficientes — exactamente el escenario donde el ML es menos
> aplicable, no más.

---

## C) ¿Siguen calibrados los parámetros congelados? Sí, con una deriva

Si la volatilidad cayó un tercio, la pregunta obligada es si `PARAMS_V2` sigue
en su sitio. Se comprobó umbral por umbral:

| umbral | antes | ahora | veredicto |
|---|--:|--:|---|
| `a_adx_min = 18` | pasa 80,2% | pasa 85,1% | sano |
| `a_vol_ratio_min = 1,2` | pasa 27,4% | pasa 28,2% | sano |
| `compression_sustained` | activa 18,7% | activa 19,2% | se auto-adapta (cuantil móvil) |
| ruptura Donchian-55 | 3,0% | 3,1% | sano |

**La deriva está en el trailing** — `min(max(atr_pct·2,5, 2,5%), 6,0%)`:

| periodo | `atr_pct` mediano | trail bruto | % en SUELO | % en TECHO |
|---|--:|--:|--:|--:|
| entrenamiento | 1,60% | 4,00% | 12,1% | 18,2% |
| **los 5 meses nuevos** | **1,18%** | **2,94%** | **27,9%** | **0,8%** |

El techo del 6% ha dejado de existir en la práctica (18,2% → 0,8%) y el suelo
del 2,5% muerde **2,3 veces más** (12,1% → 27,9%).

**La deriva va en la dirección conservadora.** Cuando el suelo muerde, el stop
queda más ancho de lo que pediría el ATR; como el sizing es `riesgo/SL`, un
stop más ancho da una posición **más pequeña**. El sistema se está
autolimitando, no sobreexponiendo.

No se toca nada: el `trail_ceiling/` de agosto ya falsó mover ambos límites, y
el mecanismo ATR está haciendo justo lo que debe. Queda anotado para revisarlo
si la volatilidad siguiera cayendo y el suelo pasara a morder la mayoría del
tiempo.

---

## Conclusión

1. **No a modelos más complejos.** 11 episodios de régimen y 131 trades no
   alimentan 31.000 parámetros. V2 gana por ser pequeño: 13,1 eventos por
   parámetro es la única proporción sana del proyecto.
2. **Refrescar datos no arregla esto.** Los últimos 5 meses aportaron 0 trades
   y 1 episodio. La curva de información es plana.
3. **Lo único que movería la aguja es información ortogonal**, no capacidad:
   order book, open interest, liquidaciones, on-chain. Todo lo que el bot mira
   hoy deriva de OHLCV.
4. **Y lo más barato de esa lista ya está en disco**: `btc_v15_funding.parquet`
   tiene 6 años de funding rate (2020-01 → 2026-03) y en vivo la llamada es
   `get_live_signal(..., df_funding=None)`. El veto de funding está muerto —
   punto 3 de "Parte 7" de `docs/SESION_2026-08-09.md`. Es la única fuente
   no-OHLCV ya descargada, con el mecanismo ya escrito en el motor.
