# Revisión de GitHub: proyectos de ML para trading de cripto (2026-09-23)

> Pedido del usuario: *"el chiste de esto es usar ML, revisa en GitHub si hay
> proyectos similares que puedan nutrir el proyecto"*.
>
> Método: se revisaron 25 repositorios. Estrellas, fecha del último push y
> licencia salen de la API de GitHub (consulta del 2026-09-23). Los resultados
> citados vienen de los README o artículos leídos; lo no verificado se indica.
> Ya revisados en mayo 2026: Freqtrade/FreqAI, Jesse y OctoBot
> (`docs/AUDITORIA_2026-05.md` §E).
>
> Filtro aplicado: el proyecto ya había probado y rechazado el ML para
> predecir dirección (clasificadores con AUC 0,51-0,52, LSTM −44%, on-chain,
> multi-par, mean-reversion, SHORT). Se buscaron **usos de ML distintos a
> predecir el precio**.

**Conclusión:** ningún repo trae una fuente de edge probada para BTC 4h
long-only con unos 165 trades. Lo útil es de dos tipos:
- **Herramientas de validación**, que son casi seguras.
- **Usos de ML que no predicen dirección**, con probabilidad baja o media.

Las cuatro ideas del ranking se probaron después con pre-registro; los
resultados están al final de este documento.

---

## 1. Herramientas de validación

| repo | ⭐ / actividad / licencia | qué aporta |
|---|---|---|
| [bashtage/arch](https://github.com/bashtage/arch) | 1.574 / push 2026-09 / NCSA-BSD | Bootstrap estacionario, circular y por bloques móviles. SPA (Reality Check de Hansen/White), StepM y Model Confidence Set: la corrección formal por data-snooping. Además GARCH y HAR. **Ya instalado y usado en `vol_sizing/`.** |
| [skfolio/skfolio](https://github.com/skfolio/skfolio) | 2.431 / activo / BSD-3 | `CombinatorialPurgedCV` (CPCV) y `WalkForward`. Con solo 11 episodios de régimen, las trayectorias de CPCV salen muy correlacionadas. |
| [esvhd/pypbo](https://github.com/esvhd/pypbo) | 140 / 2026-07 / **AGPL-3.0** | PBO (probabilidad de sobreajuste) por CSCV, PSR, DSR y MinTRL. Por la licencia, usarlo solo como herramienta de análisis. |
| sam31415/timeseriescv | 290 / 2022 / MIT | Purged K-fold. Superado por skfolio. |

## 2. Meta-labeling y triple barrera

| repo | estado | valoración |
|---|---|---|
| hudson-and-thames/mlfinlab | 4.929 ⭐, 2023, **licencia comercial** | No se puede tomar código. |
| [baobach/mlfinpy](https://github.com/baobach/mlfinpy) | 84 ⭐, 2025-01, MIT | Clon abierto: triple barrera, pesos por unicidad, bet sizing. Útil como referencia. |
| [hudson-and-thames/meta-labeling](https://github.com/hudson-and-thames/meta-labeling) | 103 ⭐, 2023, sin licencia | Código de los 4 papers del JFDS: calibración y 6 algoritmos de sizing. **Solo datos sintéticos** (AR(3) con cambio de régimen), sin evidencia en mercado real. |

## 3. Pronóstico de volatilidad

- **[tonykark1/btc-realized-volatility](https://github.com/tonykark1/btc-realized-volatility)**
  (0 ⭐, sin licencia). Tiene la metodología más seria: velas de 5 minutos,
  608 orígenes rolling, QLIKE y bootstrap por bloques. Conclusión literal:
  *"Better realized-volatility forecasts do not automatically produce better
  spot-BTC trading strategies"*. Con 25 pb de costes, la tendencia sola gana
  al volatility targeting.
- V2 ya dimensiona por el inverso de la volatilidad (riesgo / trail ATR), así
  que un mejor pronóstico solo mejoraría un insumo que ya usa.

## 4. Régimen con HMM

- **hmmlearn** (3.427 ⭐, BSD-3) es la librería estándar.
- **[MaverickThompson/stock-agent](https://github.com/MaverickThompson/stock-agent)**
  es un negativo honesto. HMM causal con walk-forward en SPY 2010-2026:
  CAGR 8,77% contra 15,16% del comprar y mantener, DD −14% contra −34%. El HMM
  detecta regímenes de volatilidad, no de retorno. Coincide con lo que midió
  este proyecto sobre el timing.
- **Humo:** MarketRegimeTrader (HMM + TDA), HMM+LSTM para predecir el precio
  y material de acompañamiento de vídeos.

## 5. ML cross-sectional (panel de monedas)

- **microsoft/qlib** (48.775 ⭐, MIT) es la mejor infraestructura para este
  enfoque. El único ejemplo en cripto que se encontró (15 monedas, Alpha158)
  da **IC −0,052**.
- **Encaje malo:** la correlación entre pares es de 0,69, y el ranking exige
  rotación o largo-corto.

## 6. RL de cartera y ejecución

- **FinRL** (16.381 ⭐) y **TensorTrade** (7.145 ⭐): sin evidencia fuera de
  muestra creíble en cripto. **Humo.**
- **FinRL_Crypto** (201 ⭐): trae CPCV y PBO. Su prueba OOS dura **8 semanas
  en pleno crash** de 2022, y en un crash basta con estar fuera del mercado
  para ganar. Humo como evidencia.
- **hummingbot** (20.173 ⭐, Apache-2.0): útil para la idea de órdenes
  post-only. A 25 trades al año, RL para ejecución es sobreingeniería.
  **Se implementó sin ML** (entrada maker, ver abajo).

## 7. Datos de derivados

- **Binance public data** (verificado): hay open interest, ratios
  largo/corto y ratio taker cada 5 minutos desde 2020-09-01. **No hay
  dataset de liquidaciones.**
- **[pavel-shkliar/Trading-research](https://github.com/pavel-shkliar/Trading-research)**:
  el más honesto de este apartado. Rechazó el OI y los ratios largo/corto y
  declaró una única señal, "DVOL en el 5% inferior → BTC rinde menos a 60
  días". Esa señal se apoyaba en un **t-test sobre ventanas solapadas**, que
  es inválido.

## 8. FreqAI (detalle no citado en mayo)

El **Dissimilarity Index** (`DI_threshold`) no opera cuando las features
están fuera de la distribución de entrenamiento. No hay evidencia de que
funcione. Prioridad baja.

## No recomendado

FinRL, FinRL_Crypto, TensorTrade, HMM+LSTM, ranking con qlib, RL de
ejecución y "AdaptiveTrend" (arXiv 2602.11708: Sharpe 2,41 y DD −12,7%,
cifras típicas de sobreajuste).

---

## Qué pasó con cada idea (probadas el 2026-09-23, todas pre-registradas)

| idea | experimento | resultado |
|---|---|---|
| Pronóstico de volatilidad (HAR-RV) para el sizing | `experiments/vol_sizing/` | **Rechazado.** HAR no pronostica mejor que el ATR que ya usa V2 (QLIKE −1,6%, p=0,84) |
| Meta-labeling con muestra ampliada | `experiments/meta_labeling/` | El modelo **sí discrimina** (AUC OOS 0,606, p=0,0005), pero el sizing no mejora V2 (p=0,139, DD peor): lo que aprende ya lo hace V2 |
| OI y DVOL como filtro | `experiments/derivados/` | **Rechazado** en la etapa 1: OI p=0,993 (signo invertido); DVOL p=0,41 (el hallazgo de pavel-shkliar no sobrevive a un null correcto) |
| Otras fuentes y horizontes | `experiments/predictibilidad_fuentes/` | 10 celdas con R² OOS < 0 en todas; ninguna señal cubre costes |
| Bot agresivo ML multi-régimen (pedido del usuario) | `experiments/agresivo/` | **Rechazado.** AUC 0,497, CAGR −12,6%, DD 50,5% |
| Entrada maker (idea de hummingbot, sin ML) | `src/portfolio_manager.py` | **Desplegada.** Comisión de entrada 0,04% → 0,02% |
