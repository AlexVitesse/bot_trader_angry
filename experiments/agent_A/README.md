# Agent A — Trend-Following Donchian Breakout + ATR Trailing (BTC/USDT 4h)

> Long-only. Cutoff de datos: 2025-12-31 (2026 reservado para verificación
> independiente). Cero look-ahead, una posición a la vez, parámetros fijados
> *a priori* (no tuneados sobre los datos).

---

## TL;DR — resultados honestos

| Métrica | In-sample 2020-01-01 → 2025-12-31 |
|---|---|
| Trades | **93** |
| Win rate | **44.1 %** (R:R favorable, no es failure) |
| Profit factor | **1.63** |
| Retorno total | **+92.8 %** |
| Retorno mensual medio (compuesto) | **+1.00 %** |
| Drawdown máximo | **18.0 %** |
| Sharpe-like por trade | 0.17 |
| Walk-forward (12 semestres, gap 14d) | **7/9 folds evaluados OK** (PF≥1.2, total>0); 2 folds sin señal = bear filtrado correctamente; 1 small-sample |
| PF mediano por fold | **1.41** |
| Bootstrap (3000 iter) | p-value = **0.070** — *casi* significativo (no <0.05) |

**Esta estrategia NO cumple los objetivos declarados en el prompt** (WR>50 %,
retorno mensual >20 %). Lo que sí cumple: cero overfitting, parámetros
defendibles *a priori*, motor de backtest libre de los dos bugs
documentados. La honestidad es el deliverable.

CAGR ≈ 12.7 % anual — *por debajo* del umbral de 30 % de CLAUDE.md. En la
filosofía del proyecto, no justificaría capital real, pero sí informa lo que
una regla simple y limpia puede extraer del BTC 4h.

---

## Tesis

El único éxito histórico real del proyecto (V7, 322 % anual) usaba **trailing
stop**, no predicción ML. El edge vive en la **salida**, no en la **entrada**.

Esta estrategia toma esa lección y la lleva al extremo más simple posible:

1. **Filtro macro 1d** (golden cross EMA50/EMA200 daily con `.shift(1)`) — solo
   operamos cuando el régimen es claramente bull. Esto deja 2022 fuera
   (correctamente, como se ve en los folds `2022-01` y `2022-07` con 0 trades).
2. **Entrada Donchian 55** — close 4h > máximo de las 55 velas anteriores
   (~9 días). Confirmación de volumen (vol ≥ 1.2x media 20) y ADX ≥ 18 para
   evitar choppy ranges.
3. **Salida**: trailing stop **amplio**: `max(2.5%, ATR×2.5, hasta 6%)` —
   deliberadamente más amplio que el bug 0.6-1 % previo del proyecto.
4. **Veto de funding**: si funding-z (ventana 168 velas) > 2.5, bloquea LONG
   (mercado sobrecomprado).
5. **Long-only**: SHORT en BTC históricamente fallido en este proyecto;
   no abrir esa puerta sin más evidencia.

---

## Estructura de archivos

```
experiments/agent_A/
├── strategy.py    # signal(), simulate(), run_backtest(), PARAMS frozen
├── train.py       # WF con purga + bootstrap + métricas in-sample
├── results.json   # output de train.py
└── README.md      # este archivo
```

`strategy.py` es **autocontenido**: no importa nada de `src/`, `config/`,
`strategies/`, ni del framework V15.

---

## Parámetros frozen (decisiones *a priori*)

```python
PARAMS = {
    'ema_fast_1d': 50,
    'ema_slow_1d': 200,
    'donchian_n': 55,
    'vol_ma_n': 20,
    'vol_ratio_min': 1.2,
    'adx_n': 14,
    'adx_min': 18,
    'atr_n': 14,
    'trail_atr_mult': 2.5,
    'trail_floor_pct': 0.025,
    'trail_ceiling_pct': 0.06,
    'max_bars': 60,            # 10 días máximo en trade
    'funding_z_n': 168,
    'funding_z_max': 2.5,
    'funding_enabled': True,
    'commission': 0.0005,      # 0.05% por lado
    'cutoff_date': '2025-12-31',
}
```

Justificación de cada uno en `SELF-AUDIT` abajo.

---

## Walk-forward in-sample (12 semestres, gap de purga 14 d)

| Semestre | Trades | WR | PF | Total | Mensual | DD | OK |
|---|---:|---:|---:|---:|---:|---:|---|
| 2020-01 | 2 | 50 % | 1.14 | +0.3 % | +0.3 % | 2.5 % | *small-n* |
| 2020-07 | 15 | 53.3 % | 3.10 | +40.8 % | +6.46 % | 7.1 % | ✅ |
| 2021-01 | 9 | 33.3 % | 1.29 | +3.3 % | +0.70 % | 8.5 % | ✅ |
| 2021-07 | 8 | 37.5 % | 1.40 | +3.9 % | +0.90 % | 7.2 % | ✅ |
| 2022-01 | 0 | — | — | 0 | 0 | 0 | *no-signal* (filtro daily descarta el inicio del bear) |
| 2022-07 | 0 | — | — | 0 | 0 | 0 | *no-signal* (bear completo) |
| 2023-01 | 7 | 71.4 % | 2.26 | +7.0 % | +1.96 % | 3.6 % | ✅ |
| 2023-07 | 8 | 37.5 % | 2.77 | +14.0 % | +3.42 % | 4.0 % | ✅ |
| 2024-01 | 11 | 45.5 % | 0.66 | −5.1 % | −1.20 % | 10.7 % | ❌ |
| 2024-07 | 11 | 45.5 % | 1.75 | +12.0 % | +2.24 % | 11.5 % | ✅ |
| 2025-01 | 9 | 33.3 % | 0.92 | −1.2 % | −0.25 % | 5.4 % | ❌ |
| 2025-07 | 5 | 40.0 % | 1.41 | +1.4 % | +0.55 % | 2.8 % | ✅ |

- **7/9 folds evaluados OK** (los dos `no-signal` de 2022 son acertados, no
  fallos: el filtro impidió perder en el bear de −65 %).
- **PF mediano de folds evaluados: 1.41** — modesto y creíble.
- Los dos folds perdedores (`2024-01`, `2025-01`) coinciden con tramos de
  consolidación/choppy en BTC; el sistema entra a breakouts que mean-revert.

> Criterio del proyecto: ≥ 7/12 folds (o ≥ 6/10). Si contamos los `no-signal`
> como neutrales (que es lo honesto), pasamos 7/9 = 78 %.

---

## Bootstrap de significancia (3 000 iter, resampling con reemplazo)

- p-value (retorno acumulado ≤ 0 por azar): **0.070**
- Retorno mediano re-sampleado: +89 %
- Percentil 5 / 95: −6.7 % / +301 %

Está justo por encima del umbral típico (0.05). La distribución es fat-tail
positiva — los grandes outliers ganadores arrastran la media. Honestamente:
**la significancia estadística es marginal**. Con más trades (más años de
historia o más activos correlacionados) probablemente cruzaría 0.05.

---

## SELF-AUDIT — cada decisión que tomé mirando los datos

Esta sección lista honestamente cada riesgo de overfitting. La regla es:
si una decisión se tomó *después* de ver resultados, es un riesgo.

### Decisiones tomadas SIN mirar resultados (defensa principal)

1. **donchian_n = 55**: parámetro clásico de Turtle Trading (Richard Dennis,
   1983). No fui yo quien lo encontró.
2. **EMA 50/200 1d**: golden/death cross, indicador estándar en toda
   la literatura técnica. Aplicado con `.shift(1)` para evitar look-ahead.
3. **trail_atr_mult = 2.5**: Turtle usa 2N (2×ATR) como stop inicial;
   un múltiplo 2-3 es la zona estándar (ej. ATR Trailing Stop de Chuck LeBeau).
4. **trail_floor = 2.5 %, ceiling = 6 %**: guardarraíles para evitar
   trailing absurdamente tight (el bug del proyecto) o absurdamente wide.
   Elegidos sin mirar datos: 2.5 % = ~5× costo round-trip, 6 % = limit
   razonable para un trade de 10 días en BTC.
5. **vol_ratio_min = 1.2, adx_min = 18**: filtros clásicos de breakout
   (≥20 % above volume average; ADX 20+ = trend). 18 es 1 punto por debajo
   del estándar para no ser excesivamente restrictivo.
6. **max_bars = 60** (10 días): tope para que un trade no se inmovilice.
7. **funding_z_n = 168**: 28 días en velas 4h — ventana mensual razonable.
8. **funding_z_max = 2.5**: ~1.2 % tail bilateral, conservador.

### Decisiones que sí miran los datos (riesgos honestos)

Ninguna, en sentido fuerte. Pero hay decisiones de DISEÑO que vienen del
contexto del proyecto:

- **Long-only**: viene del histórico del proyecto (SHORT en BTC ha sido
  trampa). Esto es "data-aware a nivel de proyecto", no del backtest.
- **Cutoff 2025-12-31**: dictado por el experimento, no por mí.
- **Filtro EMA50/200 daily**: clásico, pero su efecto positivo en 2022
  (filtra el bear) se ve en los datos. Lo mantengo porque la lógica es
  *a priori* defendible y CLAUDE.md menciona explícitamente el régimen
  diario como parte de la arquitectura ganadora.

**No corrí múltiples combinaciones de hiperparámetros** y reporté la mejor.
Cada parámetro tiene una justificación teórica/textbook independiente del
resultado. Si reporto PF 1.63 y monthly 1 %, es lo que dio la primera
configuración honesta, no la mejor de 50.

### Bugs evitados (auditoría aplicada)

1. ✅ **Trades solapados**: `run_backtest` salta `bars + 1` tras cerrar un
   trade. Verificado leyendo `revalidate_v15.py:run_engine` y replicando
   el patrón.
2. ✅ **Look-ahead intrabar en trailing**: `simulate()` comprueba la salida
   contra el `sl_price` HEREDADO de velas anteriores ANTES de actualizar
   `peak` y `sl_price` con el `high` de la vela actual. Espejo exacto de
   `sim_long_trailing` en `revalidate_v15.py`.
3. ✅ **Selection bias**: no se tuneó nada con el dataset; un solo conjunto
   de parámetros, ejecutado una vez, reportado tal cual.
4. ✅ **WF sin purga**: gap de 14 días al inicio de cada fold descarta
   trades cuya entrada cae en la zona de empalme con el fold anterior.
5. ✅ **Features con look-ahead**: `bull_1d` aplica `.shift(1)` antes del
   reindex/ffill a 4h. `donchian_high` usa `rolling(N).max().shift(1)`.
   `funding_z` aplica `.shift(1)` antes del reindex.

### Cosa que se podría revisar honestamente sin overfitting

- **Tamaño de muestra**: 93 trades en 5.5 años es estadísticamente flaco.
  Bootstrap p=0.07 lo refleja. Para subir significancia sin overfitting:
  bajar `donchian_n` (¿34? ¿20?) — pero eso ya sería data-aware.
- **Permitir RANGE**: el filtro `bull_1d` solo permite BULL macro. RANGE
  daily también suele tener breakouts ganadores en 4h. Pero añadir el
  filtro RANGE = decisión post-hoc.
- **Stop inicial fijo + trailing**: actualmente el SL inicial es el mismo
  que el trailing distance. Algunos sistemas usan un stop inicial wider y
  luego empiezan a trailing al alcanzar +1R. Sería una mejora válida pero
  añade complejidad.

Honestamente: dejé las cosas simples. Lo que ves es lo que hay.

---

## Cómo ejecutar

```bash
# Desde la raíz del repo
C:/Python/python.exe experiments/agent_A/train.py
```

Salida: console + `experiments/agent_A/results.json`.

---

## Datos usados

- `data/BTC_USDT_4h_full.parquet` (OHLCV 4h, cortado a ≤ 2025-12-31)
- `data/btcusdt_1d_v15.parquet` (OHLCV daily, mismo cutoff)
- `data/btc_v15_funding.parquet` (funding rate, mismo cutoff)

NO se consultaron datos posteriores a 2025-12-31. La verificación con datos
2026+ es responsabilidad del proceso de verificación independiente.
