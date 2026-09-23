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

---

## Resultados

*(se completan después de correr `test_meta_labeling.py`)*
