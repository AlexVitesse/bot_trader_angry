# Sentimiento y noticias como "expertos"

> Fecha: 2026-10-17. Origen: la idea del usuario de tener expertos distintos,
> incluido uno de noticias con IA.
>
> - **Parte 1, Fear & Greed: RECHAZADO** (etapa 1, p = 0,64 y 0,43).
> - **Parte 2, noticias con LLM: GRABANDO** en el VPS desde 2026-10-17. Se
>   evalúa con el protocolo de abajo, fijado hoy.

## Por qué no hay backtest de noticias con LLM

Un LLM conoce lo que pasó después de cada noticia que entra en sus datos de
entrenamiento. Si se le pasan titulares de 2021 y "predice" el precio, está
recordando, no prediciendo. El backtest saldría brillante y sería falso, que es
el patrón de los cinco fracasos de `CLAUDE.md`. La única evaluación limpia es
**hacia adelante**.

## Parte 1 — Fear & Greed (alternative.me, diario desde 2018)

`test_fear_greed.py`, mismo protocolo que `derivados_ratios/`: dos hipótesis
fijadas de antemano, Bonferroni α = 0,025 y null por rotación circular de la
serie (2.555 rotaciones). Sin look-ahead: el valor del día D se usa para velas
que abren el día D.

| hipótesis | grupo | n | diferencia | p |
|---|---|--:|--:|--:|
| S1-FEAR: con miedo extremo (FG ≤ 25), BTC rinde más a 7 días | días | 495 vs 2.113 | **−0,30%** (signo contrario) | 0,64 |
| S1-GREED: las rupturas con euforia (FG ≥ 75) rinden menos | rupturas Donchian-55 4h | 130 vs 387 | −0,27% | 0,43 |

Ninguna pasa la etapa 1, así que no se prueba como filtro de V2. El "compra
cuando hay miedo" no aparece en BTC 2019-2026 a 7 días. Es la quinta fuente de
datos alternativa rechazada en la etapa 1, tras OI, DVOL, taker y premium.

## Parte 2 — Experto de noticias con LLM local (grabando)

`scripts/record_news.py` en el VPS, proceso aparte del bot y arrancado por
cron `@reboot` (`deploy/setup_server.sh`). **No opera.**

- **Fuentes RSS:** CoinDesk, Cointelegraph, Decrypt, Bitcoin Magazine, The
  Block, Fed, SEC y Google News "bitcoin". Unos 250 titulares al día.
- **Modelo:** `qwen3:14b` en el Ollama del VPS (`localhost:11434`, sin API
  keys). Corre en CPU porque el driver de la GPU no carga: ~7 s por titular.
- **Salida por titular:** `relevante`, `sesgo` (−2..2), `riesgo` (0..3) y
  `tema`. Si se cambia el prompt se sube `PROMPT_V`.
- **Datos:** `data_live/news.db`, tabla `news`. La hora válida es
  `fetched_at` (cuando el bot pudo verlo), nunca `published`.

### Protocolo de evaluación (fijado 2026-10-17, antes de tener datos)

Se evalúa cuando haya **≥ 6 meses** de grabación o **≥ 15 trades de V2**, lo
que llegue después. Features por vela 4h, usando solo titulares con
`fetched_at` ≤ cierre de la vela y `relevante = 1` de las 24h previas:

- `sesgo_24h` = media de `sesgo`;
- `riesgo_max_24h` = máximo de `riesgo`.

Tres hipótesis, y solo tres (Bonferroni α = 0,017):

1. **H-SESGO (etapa 1, mercado):** `sesgo_24h` correlaciona con el retorno
   de BTC de las 6 velas siguientes. Null: rotación circular de la feature.
2. **H-RIESGO (etapa 1, volatilidad):** las velas con `riesgo_max_24h = 3`
   tienen un rango absoluto en las 6 velas siguientes mayor que el resto.
   Null: rotación. Es la más plausible: un LLM distingue "hackeo de un
   exchange" de "opinión de un analista" mejor que predice la dirección.
3. **H-VETO (etapa 2, solo si pasa la 2):** reducir a la mitad el riesgo de
   los trades de V2 que abren con `riesgo_max_24h = 3` mejora el DD sin bajar
   el CAGR más que el DD. Con pocos trades se reporta como descriptivo.

Si ninguna pasa, el experto se apaga y se documenta el negativo. **No se
re-prueba con otro prompt u otro modelo sobre los mismos meses**: sería
selección. Un prompt nuevo empieza su propia ventana de grabación.
