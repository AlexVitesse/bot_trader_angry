# Carry de funding — y el mapa de qué familias quedan sin probar

> Fecha: 2026-08-22 · `test_carry.py` · funding BTC 2020-01 → 2026-03 (6.765 pagos)

## Por qué este experimento

Pregunta legítima: *"hay distintas estrategias de trading, ¿las hemos probado
todas?"*. Casi. Este documento cierra el mapa y mide la única familia
estructuralmente distinta que faltaba.

### Lo que el proyecto YA ha probado

| familia | dónde | veredicto |
|---|---|---|
| Trend-following (breakout + trailing) | `agent_A`, `agent_D`, `agent_J` | **sobrevive** → es la `A` de V2 |
| Vol-compression breakout | `agent_F`, `agent_N` | **sobrevive** → es la `F` de V2 |
| Mean-reversion en rango | `agent_I` | REJECT (PF 0,66, 3,2 trades/año) |
| Mean-reversion por funding extremo | `agent_E` | evaluado |
| Régimen adaptativo | `agent_C`, `experiments/regime/` | no mejora al EMA50/200 |
| ML clasificador | `agent_B`, `agent_G`, `agent_O` | REJECT (test AUC 0,51-0,52) |
| **Features on-chain** | **`agent_K`** | **REJECT** — 35% de los shuffles aleatorios igualan al real |
| Rotación cross-asset (ratio ETH/BTC) | `agent_H` | REJECT (percentil 33 del control aleatorio) |
| Beta apalancada (SOL sobre señal BTC) | `agent_M` | evaluado |
| Deep learning (LSTM) | `docs/archive/EXPERIMENTO_BTC_V2.md` | REJECT (16,7% WR, −44,65%) |

> ⚠️ **Corrección**: `presupuesto_informacion/README.md` listaba "on-chain"
> entre las fuentes sin explorar. Es un error mío — `agent_K/` ya lo probó con
> Coin Metrics (MVRV, exchange flows, active addresses, hashrate) y lo rechazó.
> Corregido allí.

### Lo que quedaba fuera del mapa

1. **Carry de funding** (delta-neutral) — este documento.
2. **Market making / captura de spread** — requiere latencia baja e infra
   distinta. Fuera de alcance, no se mide.
3. **Arbitraje estadístico / cointegración** — pares market-neutral entre
   cripto. Sin probar; hereda el problema de correlación 0,69.
4. **Estacionalidad / efectos de calendario** — sin probar.

`agent_E` usó el funding como **señal direccional**. Esto es distinto:
**cobrarlo**, estando corto en el perp y largo en spot. No predice nada, así
que esquiva el hallazgo de `predictibilidad/` (R² = 0,068%), y en teoría
debería pagar cuando V2 está parado.

---

## 1) ¿Cuánto paga?

```
medio: 0,01144% por pago  ->  +12,52% anual bruto
pagos positivos (cobras): 87,0%
```

Suena bien. Pero por año:

| año | anual bruto | % positivos |
|---|--:|--:|
| 2020 | +17,19% | 85,7% |
| **2021** | **+30,61%** | 92,7% |
| 2022 | +4,16% | 77,9% |
| 2023 | +7,87% | 89,9% |
| 2024 | +11,92% | 91,6% |
| 2025 | +5,13% | 87,1% |
| **2026** | **+2,33%** | 70,9% |

**El carry se ha comprimido de +30,6% a +2,3%.** Es la firma exacta de un trade
arbitrado: en 2021 pagaba una prima de frenesí minorista, y el capital
institucional la ha ido cerrando. El +12,52% medio de 6 años es un número
histórico que ya no existe — y nótese que **el año que lo sostiene es 2021**,
el mismo fold que hacía parecer buena la vía multi-par en `portfolio_sim/`. El
mismo sesgo, otra vez.

## 2) ¿Diversifica? No — y esto lo mata

La razón para querer el carry era tener ingresos mientras V2 está parado.
Medido contra el propio filtro de régimen de V2:

| régimen | carry anual | n pagos |
|---|--:|--:|
| BULL (V2 opera) | **+15,56%** | 4.722 |
| BEAR/RANGE (V2 parado) | **+5,50%** | 2.043 |

**El carry paga 2,8× más cuando V2 ya está operando.** Está correlacionado con
V2, no lo diversifica: los dos viven del mismo apetito alcista. El funding es
positivo porque hay longs apalancados pagando; cuando no los hay, ni V2 entra
ni el carry cobra.

En el bear actual (desde 2025-11-17): **+3,31% anual**.

## 3) Contra qué compite

```
Earn de stablecoins (YA desplegado en el bot):  ~4-8% anual
carry hoy:                                       +3,31% bruto
                                                 − comisiones de 2 patas
                                                 − rebalanceo del delta
                                                 − riesgo de base y liquidación
```

**El carry rinde hoy menos que el Earn que el bot ya tiene**, con dos patas que
mantener, delta que rebalancear, y riesgo de liquidación en el perp que el Earn
no tiene. Y el bot es futures-only en testnet: la pata spot no existe en la
arquitectura actual.

---

## Veredicto: NO

No por dogma — porque paga +3,31% donde ya se cobra 4-8% sin complejidad, y
porque no diversifica, que era la única razón para quererlo.

Queda anotado que **si el funding volviera a niveles de 2021** (+30%) la
conclusión cambiaría. La forma barata de vigilarlo es mirar el `funding_z` que
el bot ya calcula cada vela para la línea de régimen: no hace falta construir
nada para saber cuándo merecería la pena reabrir esta pregunta.

## Lo que este experimento no cerraba — ya cerrado

Cuando se escribió esto, arbitraje estadístico y estacionalidad seguían sin
medir. Ya no:

- **Arbitraje estadístico** → `stat_arb/`: **REJECT**. En 210 pares cointegran
  3 cuando el azar solo da 10,5; en el universo de histórico largo, 4 de 91
  contra 4,5 esperados. Ninguno persiste en test (0%). El backtest da WR 65-70%
  con PF 0,80-0,88 — el perfil exacto que `criterio_validacion/` enseña a
  desconfiar.
- **Estacionalidad** → `estacionalidad/`: **REJECT**. El efecto miércoles
  sobrevive a persistencia y a outliers, pero el t-test (p=0,005) estaba mal
  calibrado; con el null correcto por rotación de calendario da **p=0,14**. Y
  operarlo rinde +15,1% anual contra +35,2% de comprar y mantener.
- **Market making** → apéndice de `stat_arb/`: **no aplicable**. Spread de
  0,015% contra comisión taker de 0,100%: cada ronda pierde 0,085%. Requiere
  fee tier VIP y latencia de ms; el bot corre a 30s sobre velas de 4h.

**Con eso no quedan familias de estrategia sin medir** dentro de lo que permite
la arquitectura. Lo que sigue sin explorar son **fuentes de datos** (order
book, open interest, liquidaciones), no estrategias.
