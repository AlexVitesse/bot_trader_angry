# Plan: especialistas (macro y volumen) + meta-labeling como decisor (2026-10-20)

> Origen: propuesta del usuario. Un modelo que mire la paridad con el oro y el
> macro y avise de caídas; otro que mire el volumen; y uno final que, con esos
> resultados más las reglas, decida. Probarlo en BTC dentro del bot agresivo.
>
> **Este plan no toca el bot en vivo hasta E10.** Todo corre en `experiments/`.

## Estado

| # | qué | estado | resultado |
|---|---|---|---|
| E7 | Especialista macro: oro, DXY, S&P 500 → ¿avisa caídas de BTC? | hecho (2026-10-20) | **PASA** (p=0,016; rotación 0,025). Todo viene de la paridad BTC/oro; no ayuda en 2021-22. `experiments/especialistas/` |
| E8 | Especialista volumen → ¿avisa caídas de BTC? | hecho (2026-10-20) | **No pasa** (p=0,089). No entra en E9 |
| E9 | Meta-labeling como decisor del agresivo en BTC | hecho (2026-10-20) | **No pasa** etapa 1: AUC 0,475 (p=0,87); sin macro 0,493. El meta-labeling no se traslada a eventos con reversión y cortos. `experiments/meta_agresivo/` |
| E10 | Conectar al agresivo en vivo (paper) | descartado | E9 no pasó |
| E11 | Otras bolsas: Hong Kong (`^HSI`) y Reino Unido (`^FTSE`) | pendiente, idea del usuario (2026-10-20) | |

**E11 (por pre-registrar).** No entra en E7: su pre-registro ya está cerrado y
añadir mercados después sería probar variantes. Se hace aparte, con su propio
README, el mismo test incremental y **solo si E7 muestra que el macro aporta
algo**. Si el oro, el dólar y el S&P no aportan, es poco probable que dos bolsas
más sí lo hagan, y solo sumarían comparaciones múltiples. Lo interesante de Hong
Kong es el **horario**: cierra hacia las 08:00 UTC, antes de que abra EE. UU.
Podría avisar dentro del mismo día de BTC, algo que el S&P no puede hacer.

## Por qué este orden

El decisor final solo puede combinar lo que sepan los especialistas. Si cada uno
está en AUC ≈ 0,5, juntarlos da ≈ 0,5 con más parámetros para memorizar (es lo
que le pasa hoy al agresivo: HGB con 11 features, AUC 0,497). Por eso:

1. **Primero** se demuestra que cada especialista ve algo él solo (E7, E8).
2. **Después** se construye el decisor (E9) con las reglas del agresivo, los
   features que ya probaron servir en `meta_labeling/` (AUC 0,61 fuera de
   muestra, el único positivo del proyecto) y **solo** los especialistas que
   pasen.
3. **Al final**, si E9 pasa, se conecta (E10).

## Reglas comunes (pre-registro)

1. **Una sola configuración por experimento**, fijada aquí. Si falla, se acepta
   el no. Sin "probemos otro umbral".
2. **E7 y E8 forman una familia de 2 tests → α = 0,025 cada uno.** E9 tiene su
   propio α = 0,05 por etapa.
3. **Modelo fijo: regresión logística** de sklearn (`C=1,0`, features
   estandarizadas con media y desviación del train). Nada de HGB ni redes: con
   ~11 episodios de régimen no caben (`presupuesto_informacion/`).
4. **Walk-forward anual expansivo:** para cada año de test Y ∈ {2021, …, 2026}
   se entrena con lo anterior a `1-ene-Y` menos una purga igual al horizonte
   de la etiqueta.
5. **Nulls que conservan la estructura temporal**, nunca un t-test (las caídas
   vienen en racimos; `CLAUDE.md`, sección de autocorrelación).
6. Un README por experimento: el pre-registro primero y los resultados debajo,
   sin tocarlo.

---

## E7 — Especialista macro (oro, dólar, bolsa)

**Pregunta:** ¿el oro, el dólar y la bolsa avisan de una caída de BTC **más allá
de lo que ya avisa la propia volatilidad de BTC**? Esa es la trampa a evitar:
las caídas vienen tras volatilidad alta, y cualquier cosa correlacionada con
volatilidad "predice" caídas sin aportar nada nuevo.

**Datos.** Diarios de yfinance desde 2018: oro `GC=F`, dólar `DX-Y.NYB`,
S&P 500 `^GSPC`. BTC diario a partir de `data/BTC_USDT_4h_full.parquet`
(desde 2019). Los mercados tradicionales no abren el fin de semana: valor del
último día hábil, **desplazado 1 día** (sin look-ahead).

**Etiqueta (compartida con E8).** Día `d` (cierre UTC):
`y = 1` si `min(low de BTC en d+1 … d+10) ≤ close_d × 0,90`.
Es decir, **una caída de al menos 10% en los próximos 10 días**. Se informa la
tasa base.

**Features (3, fijas):**
- `oro_z`: z-score de 60 días de `log(BTC / oro)` → la paridad con el oro.
- `dxy_20`: retorno de 20 días del dólar.
- `spx_dd`: distancia del S&P 500 a su máximo de 60 días.

**Base (lo que BTC ya sabe de sí mismo):** `vol20` (desviación de retornos
diarios de 20 días), `dist_ema200` y `ret20` de BTC.

**Test (H7).** Dos modelos con el mismo walk-forward: **base** y **base + macro**.
- Estadístico: `ΔAUC = AUC(base+macro) − AUC(base)` fuera de muestra
  (2021-2026, predicciones de todos los años juntas).
- p: bootstrap **pareado** por bloques de 30 días de calendario, 10.000 réplicas,
  `p = P(ΔAUC* ≤ 0)`. **Pasa si p < 0,025.**
- Descriptivo: AUC de cada modelo por año, AUC de "solo macro", coeficientes, y
  un null por rotación circular de las features macro contra la etiqueta
  (desplazamiento ≥ 90 días).

## E8 — Especialista volumen

Misma etiqueta, misma base, mismo test (H8). Solo cambian las features.

**Features (3, fijas),** calculadas con velas 4h y tomadas al cierre del día:
- `vol_z`: log del volumen de las últimas 6 velas (1 día) contra la media y la
  desviación de las 120 anteriores (20 días).
- `vol_sube`: volumen de las velas alcistas / volumen total, últimas 60 velas
  (10 días). Si las subidas van con poco volumen y las bajadas con mucho, es
  presión vendedora.
- `pct_sube`: fracción de velas alcistas en las últimas 60.

No se usa el ratio taker (rechazado en `derivados_ratios/`).

**Pasa si p < 0,025** con el mismo bootstrap pareado.

---

## E9 — Meta-labeling como decisor del agresivo, solo BTC

**Idea.** Sustituir el cerebro del agresivo (HGB, AUC 0,497) por el meta-labeling
de `meta_labeling/`, que sí discriminó (AUC 0,61), y darle además las features
de los especialistas que hayan pasado.

**Selección declarada.** El resultado de `meta_labeling/` ya se vio. Por eso
se reutilizan **exactamente** sus tres features, sin elegir otras.

**Eventos primarios:** los candidatos del agresivo en BTC 4h, tal como los genera
`src/agresivo_engine.py` (`comp`, `don55`, `don20`, `mr`; long y short), con su
etiqueta (`y = 1` si el trade con trailing gana neto de costes).

**Modelo:**
- Logística con `atr_pct`, `bb_width` y `dir × dist_ema_d`, donde el signo se
  invierte en los cortos.
- Se añaden las 3 features de E7 y/o las de E8 **solo si ese experimento pasó**.
  Si no pasó ninguno, van solo las tres.
- Pesos por unicidad y purga de 60 velas, como el motor actual.

**Regla de decisión:** la misma del agresivo.
- Se entra si `p̂ > 1/(1+b)`, es decir, si Kelly es mayor que 0.
- Riesgo = Kelly/2, entre 1% y 10%, con el freno por DD actual.

**Etapa 1:**
- Estadístico: AUC fuera de muestra (2021-2026).
- Test: bootstrap por bloques de 30 días.
- **Pasa si p < 0,05.**
- Descriptivo: el AUC del HGB actual sobre los mismos eventos.

**Etapa 2** (con `experiments/agresivo/backtest.py` limitado a BTC):

| brazo | qué es |
|---|---|
| A | agresivo-BTC con HGB (el actual) |
| B | agresivo-BTC con la logística |
| C | V2 al 4,5% (referencia) |

**H9 pasa si se cumplen las tres:**
1. CAGR/DD de B mayor que el de A.
2. CAGR/DD de B mayor que el de C, a igual exposición media.
3. Null sintético: 200 series `sintetica()` con bloques de 180 velas, corriendo
   todo el pipeline (también el reentreno) en cada una. p del CAGR de B **< 0,05**.

**Expectativa honesta:** lo que aprendió `meta_labeling/` fue "largo por encima
de la EMA200 con volatilidad baja", que es casi el filtro de V2. Lo probable es
que B acabe pareciéndose a V2 con más trades. Si es mejor, saldrá aquí.

## E10 — Conexión en vivo (paper)

- **Si E9 pasa:** el agresivo usa la logística en BTC (`src/agresivo_engine.py`),
  sigue en paper con sus $100 y se compara en vivo contra su predicción del
  backtest.
- **Si E9 no pasa:** no se conecta. Correrlo igualmente para medirlo sería
  decisión del usuario, como pasó con el agresivo actual.

## Coste estimado

- E7 y E8: un script cada uno, minutos de cómputo. Los datos de yfinance son
  gratis.
- E9: lo caro es el null sintético con reentreno (200 series × 6 años); con la
  logística son minutos, no horas.
