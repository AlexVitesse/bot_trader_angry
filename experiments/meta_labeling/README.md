# Meta-labeling (López de Prado) para el sizing de V2, con muestra de eventos ampliada

> **Pre-registro escrito el 2026-09-23, ANTES de correr nada.** Los resultados
> se añaden debajo sin tocar esta sección.
>
> Origen: idea 3 de la revisión de GitHub del 2026-09-23. El giro frente a los
> ML ya rechazados (`agent_B/G/O`, V7-V14) es triple:
> - el modelo **no predice dirección**, puntúa señales de una estrategia primaria;
> - solo cambia el **tamaño** de la posición, nunca si se entra;
> - se entrena sobre **miles de eventos relajados**, no sobre los 165 trades.

## Eventos primarios (relajados, solo LONG, BTC 4h desde 2019)

Una vela `t` es un evento si se cumple **al menos uno** de estos disparadores.
Cada vela cuenta una sola vez aunque cumpla varios.

- **Ruptura Donchian:** `close_t > max(high de las N velas previas)`, con N ∈ {20, 40, 55, 100}.
- **Ruptura tras compresión:** `bb_width` de la vela anterior por debajo de su
  cuantil q ∈ {0,10; 0,20; 0,30} (la misma ventana que usa V2), **y**
  `close_t > max(high de las 20 velas previas)`.

No se aplica filtro de régimen: si el régimen importa, el modelo lo aprende de
su feature.

## Etiqueta

- **El trade del evento:** entrada al open de `t+1`, trailing de V2 tipo A
  (`trail = clip(atr_pct·2,5; 2,5%; 6%)`), `max_bars = 60`, stop con gap
  (`min(stop, open)`) y costes nuevos.
- **Etiqueta:** `y = 1` si el PnL neto es > 0.
- **Span del evento:** de `t+1` a la vela de salida. Se usa para la purga y
  los pesos.

## Modelo (fijo, sin búsqueda de hiperparámetros)

- **Features (3):**
  - `atr_pct`;
  - `dist_ema200` = `close / EMA200 diaria − 1` (EMA del cierre diario, con
    shift de 1 día);
  - `bb_width`.

  Las tres se estandarizan con media y desviación del conjunto de entrenamiento.
- **Modelo:** regresión logística de sklearn con `C=1,0` (su default) y pesos de
  muestra iguales a la **unicidad media** de cada evento (López de Prado cap. 4).
  Eventos solapados pesan menos.
- **Walk-forward anual expansivo:** para cada año de test Y ∈ {2021, …, 2026},
  se entrena con los eventos cuya **salida** es anterior al 1-ene-Y menos una
  purga de 60 velas. Se predice p̂ para los eventos del año Y.

## Etapa 1: ¿el modelo discrimina fuera de muestra?

- **Métrica:** AUC de p̂ sobre todos los eventos fuera de muestra (2021-2026),
  con las predicciones de todos los años juntas.
- **Test:** bootstrap por bloques de **30 días de calendario** sobre los
  eventos fuera de muestra, 10.000 réplicas, H0: AUC ≤ 0,5. p = fracción de
  réplicas con `AUC* − AUC ≥ AUC − 0,5`.
- **Informado:** N de eventos, N_eff por unicidad (Σ pesos) y AUC por año.
- **Regla de parada:** si p ≥ 0,05, el experimento termina.

## Etapa 2: ¿el sizing mejora V2?

- **Qué trades:** los de V2 de `portfolio_sim` (BTC, costes nuevos, riesgo 2%)
  con entrada en 2021-2026. A cada uno se le asigna el p̂ del modelo de su año,
  calculado sobre su vela de señal.
- **Tamaño:** multiplicador `m = 0,5 + ECDF_train(p̂)`, que va de 0,5 a 1,5 y
  vale 1 de media. El `notional` de V2 se multiplica por `m`, con el mismo tope
  `max_notional_pct = 2,5`.
- **Métrica primaria:** Sharpe por trade, `mean(r)/std(r)`, meta-sized frente a
  V2, sobre los mismos trades.
- **Test:** bootstrap pareado por bloques circulares de 10 trades, 20.000
  réplicas. p = P(ΔSharpe ≤ 0). **α = 0,05** (una sola comparación).
- **Adopción:** p < 0,05 **y** que el DD máximo del meta-sized no sea peor que
  el de V2 en el mismo tramo.
- **Descriptivas:** CAGR, DD y PF de los dos brazos, y la correlación entre p̂ y
  `r` en los trades de V2.

## Riesgos conocidos antes de empezar

- Los eventos relajados son en buena parte los mismos episodios de mercado
  contados muchas veces. El N_eff real es pequeño.
- `predictibilidad/` midió un R² in-sample de 0,068% con features técnicas
  parecidas. Lo esperable es AUC ≈ 0,5.
- El repo de referencia (`hudson-and-thames/meta-labeling`) solo tiene
  evidencia con datos sintéticos.

**Adenda de implementación (2026-09-23, escrita ANTES de correr).** Precisa
puntos que el pre-registro dejaba abiertos:

1. **Base de velas.** Se usa `cargar_pares(['BTC/USDT'])`: velas 4h tras el
   `dropna` de `build_features`, desde 2019-01-10.
2. **Features de un evento en `t`.** `atr_pct_t` y `bb_width_t` se conocen al
   cierre de `t`. `dist_ema200` es `close_t / EMA200 − 1`, con la EMA200 del
   cierre diario (resample 1D) desplazada 1 día y rellenada hacia delante a 4h.
3. **Compresión.** `bb_width_{t-1} < quantile_q(bb_width, ventana 100)_{t-1}`,
   la misma forma que V2 pero sin exigir que se sostenga varias velas.
4. **Costes de la etiqueta.**
   - Comisión 0,0006 por lado y stop con gap.
   - Funding como en `portfolio_sim`: el histórico de
     `data/btc_v15_funding.parquet`, 0,5 × tasa por vela en posición, con
     0,00013 como fallback fuera de cobertura.
   - `y = 1` si `bruto − 2·comisión − Σfunding > 0`.
5. **Eventos sin resolver.** Si no hay 60 velas después del evento y no salió
   por stop, no hay etiqueta y el evento se descarta.
6. **Unicidad media.** Se calcula **solo dentro del conjunto de entrenamiento
   de cada año** (concurrencia de spans por vela entre los eventos de train),
   para no usar información del periodo de test.
7. **Qué eventos entran en cada año.**
   - **Test del año Y:** eventos con vela de señal en el año Y.
   - **Train:** eventos cuya vela de salida es anterior a `1-ene-Y − 60 velas`
     (240 h).
8. **`ECDF_train(p̂)`.** ECDF de las predicciones del modelo del año sobre sus
   propios eventos de entrenamiento.
9. **Bootstrap de AUC.** Los eventos fuera de muestra se agrupan en bloques de
   30 días de calendario contados desde el primero. Se remuestrean bloques con
   reemplazo hasta el mismo número de bloques. El AUC es sin pesos.
10. **Etapa 2.**
    - El p̂ de cada trade sale de su vela de señal (`ts_entrada − 4h`), con el
      modelo del año de esa vela.
    - Solo se remuestrean los trades con señal en 2021-2026. Los anteriores
      usan el sizing de V2 en los dos brazos, así que el conjunto de trades es
      idéntico.

---

## Resultados

> Corrida el 2026-09-23 · `test_meta_labeling.py` · salida completa en `salida.txt`.

### Etapa 1: el meta-modelo SÍ discrimina fuera de muestra

- 823 eventos con etiqueta; el 43,9% tiene `y = 1`.
- **N_eff por unicidad = 355**: los eventos solapados cuentan menos de la mitad.

| año de test | eventos train | N_eff train | eventos test | AUC |
|--:|--:|--:|--:|--:|
| 2021 | 252 | 109,7 | 126 | 0,589 |
| 2022 | 384 | 161,4 | 77 | 0,539 |
| 2023 | 463 | 201,5 | 102 | 0,593 |
| 2024 | 564 | 252,2 | 125 | 0,625 |
| 2025 | 691 | 301,0 | 117 | 0,572 |
| 2026 | 806 | 345,7 | 15 | 0,841 |
| **pooled 2021-2026** | | | **562** | **0,606** |

- **p = 0,0005** con bootstrap por bloques de 30 días (63 bloques): la etapa 1
  pasa.
- AUC > 0,5 en los 6 años de test. El de 2026 se apoya en solo 15 eventos.
- **Coeficientes estables todos los años:**
  - `atr_pct` negativo (−0,21 a −0,47);
  - `dist_ema200` positivo (+0,13 a +0,27);
  - `bb_width` negativo pequeño.
- Lectura: una ruptura tiene más probabilidad de acabar en ganancia **con
  volatilidad baja y el precio por encima de la EMA200 diaria**.

### Etapa 2: el sizing NO mejora V2 de forma significativa

Mismos trades en los dos brazos (verificado): 115 trades de V2 con señal en
2021-2026, de 165 en total.

| brazo | Sharpe por trade | PF | CAGR 2021-26 | DD 2021-26 |
|---|--:|--:|--:|--:|
| V2 | 0,0746 | 1,22 | +4,0% | **21,5%** |
| meta-sized | 0,0955 | 1,32 | +6,3% | 24,2% |

- Diferencia de Sharpe: **+0,021, p = 0,139**. No llega a α = 0,05.
- `corr(m, r_V2) = +0,119`.
- El multiplicador medio es **1,165** (entre 0,53 y 1,50), no 1. Las p̂ fuera
  de muestra salen más altas que en el entrenamiento, así que el brazo meta
  también arriesga más en promedio. Eso explica parte del CAGR y del DD
  mayores. El Sharpe por trade no depende de esa escala media.

### Lectura

- **El ML sí encuentra algo sobre el conjunto amplio de rupturas.** Es la
  primera vez en el proyecto que un modelo supera el azar fuera de muestra con
  un test que respeta la autocorrelación. Lo que aprende es conocido: tendencia
  por encima de la EMA200 más volatilidad contenida.
- **Dentro de los trades de V2 esa información está casi agotada.** V2 ya filtra
  por régimen diario (EMA50/200) y dimensiona por ATR, justo las dos cosas que
  el modelo aprendió. Por eso la correlación entre multiplicador y resultado
  cae a +0,12, y la mejora de Sharpe (+28% relativo) no se distingue del ruido
  con 115 trades.
- El DD empeora 2,7 puntos, en parte por la exposición media más alta.

### Veredicto

**RECHAZADO según el criterio pre-registrado**: p = 0,139 ≥ 0,05, y además
el DD del brazo meta es peor que el de V2. No se conecta al bot.

Queda como único positivo de etapa 1 del proyecto (AUC 0,61 fuera de
muestra). Si se retoma, tiene que ser con un pre-registro nuevo y sin reutilizar
estos resultados para elegir parámetros.

