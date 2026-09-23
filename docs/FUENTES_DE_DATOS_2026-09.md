# Hallazgos: por qué pierde el bot agresivo y qué fuentes de datos quedan

> 2026-09-23. Revisión de `experiments/agresivo/`, `docs/SESION_2026-09-23.md`
> Parte 12, la rama `feature/bot-agresivo` y el inventario de fuentes de datos
> ya medidas. Documentos derivados:
> - plan de prueba de las fuentes que quedan: `experiments/derivados_ratios/README.md`
> - grabación de datos propios en vivo: `docs/GRABACION_DATOS_VIVO.md`

---

## 1. Qué se construyó

Bot ML multi-régimen en 9 pares (BTC, ETH, SOL, BNB, XRP, DOGE, ADA, LINK,
AVAX), velas 4h desde 2020:

- Candidatos long y short en todos los regímenes: Donchian 20/55, ruptura tras
  compresión de BB-width y reversión a la media en Bollinger.
- `HistGradientBoosting` sobre 25.657 eventos etiquetados con la salida de V2
  (trailing ATR), pesos por unicidad, calibración Platt, reentreno mensual.
- Riesgo por trade decidido por el modelo: medio Kelly entre 1% y 10%, con
  freno `× max(0, 1 − DD/50%)`.
- Diseño fijado antes de correr. Una sola variante, sin retoques después.

Metodológicamente es correcto. El código vive en `src/agresivo_engine.py`
(rama `feature/bot-agresivo`, un commit `5ae31e9` sobre `main`).

## 2. Resultado

| sistema (2021-01 → 2026-02) | CAGR | DD máx | PF | trades/año |
|---|--:|--:|--:|--:|
| **Agresivo con ML** | **−12,6%** | **50,5%** | 0,32 | 381 |
| V2 al 4,5% (desplegado) | +7,5% | 40,9% | 1,26 | 22 |
| Comprar y mantener BTC | +16,8% | 77,0% | — | — |

p (bootstrap por bloques, media de r > 0) = 0,98.

## 3. Por qué pierde: tres causas medidas

1. **El modelo no discrimina.** AUC fuera de muestra 0,497 sobre 21.208
   candidatos; por año entre 0,478 y 0,527. Además es sobreconfiado: donde
   pronostica 40-50% de aciertos la tasa real es 37%.
2. **Los candidatos que añade tienen expectativa negativa después de costes.**

   | tipo | n | WR | suma r |
   |---|--:|--:|--:|
   | mr (reversión a la media) | 1.519 | 37,4% | −0,66 |
   | comp | 170 | 35,9% | −0,00 |
   | don20 | 124 | 34,7% | −0,01 |
   | don55 (la entrada de V2) | 150 | 49,3% | +0,00 |

   Por dirección: shorts 1.136 trades y −0,70 R; longs 827 trades y +0,03 R.
   Es la repetición, con 9 pares y 25.000 eventos, de lo que ya dieron
   `agent_I` (mean-reversion, PF 0,66), el SHORT de V2 (PF 0,88) y los
   clasificadores ML (AUC 0,51-0,52).
3. **Kelly amplifica el ruido.** Con una `p` sin valor abre riesgo en 8.142
   candidatos cuyo r medio es −0,045 R. Pierde el 50% en 2021 con 664 trades
   y el freno por drawdown congela la cuenta el resto de la historia. Nota:
   por eso los años 2022-2026 salen a ~0%; no es que mejore, es que no opera.

## 4. ¿Se arregla con ML?

No con los datos que hay. Cinco mediciones independientes de este mes lo
cierran:

| experimento | qué preguntaba | resultado |
|---|---|---|
| `meta_labeling/` | ¿un meta-modelo mejora el sizing de V2? | discrimina (AUC 0,61) pero aprende lo que V2 ya tiene; sizing no mejora (p=0,14) |
| `vol_sizing/` | ¿HAR predice volatilidad mejor que el ATR? | no (p=0,84) |
| `derivados/` | ¿OI o DVOL filtran trades? | no (p=0,99 y 0,41) |
| `predictibilidad_fuentes/` | ¿BTC 1h/1d, panel altcoins, derivados tienen señal? | R² OOS < 0 en las 10 celdas |
| `presupuesto_informacion/` | ¿caben los parámetros? | 131 trades y 11 episodios contra miles de parámetros |

La variante que quedó pendiente en la sesión, "rupturas long del motor
agresivo en más pares", ya está medida en `experiments/v2_all_coins/`: de 22
monedas solo BTC pasa los tres criterios y DOGE es marginal por su deriva
propia.

**Veredicto:** dejar el motor agresivo aparcado en su rama. No volver a
gastar sesiones en ML de señales con las fuentes actuales.

## 5. Inventario de fuentes de datos

### Ya medidas (no repetir)

| fuente | dónde se midió | veredicto |
|---|---|---|
| Velas 4h/1h/1d, 22 monedas | `predictibilidad/`, `predictibilidad_fuentes/`, `v2_all_coins/` | sin señal que cubra costes; solo BTC pasa |
| Funding | `funding_veto/`, `carry_funding/` | veto: 6 trades en 6,5 años; carry rechazado |
| Open interest (Binance, 5 min) | `derivados/` | p=0,993, signo invertido |
| DVOL (Deribit, diario) | `derivados/` | p=0,41 |
| On-chain (Coin Metrics: MVRV, direcciones, flujos, hashrate) | `agent_K/` | 35% de shuffles igualan al real; AUC empeora |

### Obtenibles gratis y sin medir

Verificado el 2026-09-23 contra `data.binance.vision` (prefijo
`data/futures/um/`):

| fuente | ruta / API | historia | prior |
|---|---|---|---|
| Ratio taker compra/venta, ratio largo/corto (cuentas y top traders) | `metrics/BTCUSDT`, diario, 5 min. **Ya descargado**; `derivados/` solo guardó la columna OI | 2020-09-01 | bajo. `pavel-shkliar/Trading-research` rechazó los ratios largo/corto |
| Premium index (mark vs índice, base del perpetuo) | `premiumIndexKlines/BTCUSDT/4h`, mensual | 2020-01 | bajo-medio. Es el insumo del funding con más resolución |
| Profundidad del libro (±1% a ±5%, cada minuto) | `bookDepth/BTCUSDT`, diario | 2023-01-01 | medio para ejecución, bajo para señal. 2,7 años, un régimen |
| Mejor bid/ask | `bookTicker/BTCUSDT`, diario | 2023-05-16 | solo ejecución |
| aggTrades (cada operación con lado agresor) | `aggTrades/BTCUSDT` | 2019-12-31 | bajo como señal a 4h; útil para medir slippage |
| Coinbase premium (spot USA vs Binance) | API Coinbase Exchange, velas 1h | 2019 | bajo-medio. Señal conocida de 2020-21, sin evidencia después |
| Macro: DXY, S&P 500, tipos, M2 | FRED, yfinance, diario | décadas | bajo. N_eff diario minúsculo; la correlación cripto-bolsa cambia de régimen |
| Fear & Greed | alternative.me, diario | 2018 | muy bajo. Se construye con precio y volatilidad: redundante |

No existe `liquidationSnapshot` para BTCUSDT en Binance public data (prefijo
vacío al listarlo). Confirma lo que ya decía
`docs/INVESTIGACION_GITHUB_ML_2026-09.md` §7.

### De pago

| fuente | proveedor | historia | comentario |
|---|---|---|---|
| Liquidaciones históricas | Coinglass, Tardis | 2021 | la fuente con más folclore de trader; a 4h y con 0,12% de coste, prior bajo |
| Libro completo tick a tick | Tardis | 2019 | cientos de dólares al mes; solo para microestructura, que no es este bot |

## 6. Recomendación

1. **Probar una sola cosa gratis, con pre-registro y regla de parada:** ratio
   taker y premium index sobre rupturas Donchian-55 crudas, mismo protocolo de
   etapa 1 que `derivados/`. Un día de trabajo. Plan en
   `experiments/derivados_ratios/README.md`.
2. **Grabar datos propios desde ya en el VPS:** liquidaciones en vivo, libro y
   bookTicker en cada señal y cada fill. Coste cero, coincide con los 6-12
   meses de paper trade y produce el dataset de ejecución que hoy no existe,
   que es la parte con margen real. Especificación en
   `docs/GRABACION_DATOS_VIVO.md`.
3. **No pagar por datos** hasta que el punto 1 haya pasado o fallado. Si dos
   fuentes gratuitas de derivados no aportan nada a nivel de mercado, una de
   pago de la misma familia no va a cambiar la conclusión.

El cuello de botella sigue siendo el mismo: 131 trades de V2 y 11 episodios
de régimen. Cualquier fuente nueva tiene que demostrar señal primero sobre
miles de eventos de mercado, nunca sobre los trades de V2 directamente.
