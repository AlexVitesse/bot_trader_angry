# Agent H — ETH/BTC Ratio Rotation (ETH/USDT 4h, long-only)

> Ronda 3, ETH-específica. Hipótesis: el ratio ETH/BTC es un indicador
> subestimado de rotación de capital cripto. Cuando el capital prefiere ETH
> (ratio sube o ratio extremadamente bajo en bull global) — LONG ETH.
>
> Cutoff inviolable: 2025-12-31. Sin re-tuning. Sin look-ahead.

---

## Veredicto rápido

**REJECT** — el "edge del ratio" como mecanismo de entrada no se distingue
estadísticamente de azar:

| | REAL ETH | RANDOM ratio (30 seeds) |
|--|---------:|------------------------:|
| Annual return | **-5.52%** | mean +0.3%, median -1.2%, p5-p95 = -16.8% … +18.7% |
| Posición del REAL en la distribución del control | **33° percentil** | (mediana) |
| **CONTROL VERDICT** | **FALLA** — el ratio no aporta edge sobre azar | |

| Métrica | Valor |
|---------|------:|
| WF (PF≥1.2, total>0, n≥3) | **4/10 evaluados** (2 folds sin señal por filtro BTC bear) |
| Trades 2020-2025 | 272 |
| WR | 31.2% |
| PF | 1.04 |
| Total return | -28.3% |
| Annual return | -5.5% |
| Max DD | 77.6% |
| Bootstrap p-value | 0.624 (no significativo) |
| Sharpe-like | 0.01 |

La estrategia, tal como fue diseñada a priori (combinando ambas señales),
**pierde dinero en sample y es indistinguible de un ratio aleatorio.**
La hipótesis "el ratio ETH/BTC es subestimado pero poderoso" no se sostiene
con este diseño y estos params frozen.

---

## Diseño (a priori, sin tuning con datos)

### Mecanismo
- Calcular `ratio = ETH_close / BTC_close` (daily, shift(1) al usar en 4h).
- Features del ratio:
  - `ratio_ema_fast` (20), `ratio_ema_slow` (50)
  - `ratio_slope` (cambio promedio en 10 días)
  - `ratio_z_score` (90 días)
  - `ratio_high20` (alto rolling 20d, con shift(1))

### Señales (dos subhipótesis, ambas committed)

**Primary — momentum del ratio:**
- ratio > ratio_ema_slow (uptrend), Y
- ratio_ema_fast > ratio_ema_slow (golden cross del ratio), Y
- ratio_slope > 0 (acelerando)

**Secondary — mean-reversion del ratio (oversold extremo):**
- ratio_z_score < -1.5 (muy barato vs su media 90d)

**Filtros macros aplicados a ambas:**
- BTC daily NO en bear (EMA50_1d > EMA200_1d, shift(1))
- Funding BTC z-score ≤ 2.5 (anti-euforia; proxy para ETH funding)
- Volumen ETH > 1.1× promedio 20 (confirmación suave)

### Exit
- Trailing stop ATR-based AMPLIO (2.5× ATR, piso 2.5%, techo 7%).
- Max 60 velas (10 días). LONG-only.

### Anti-bugs (mismas protecciones que Agent A)
- **Una posición a la vez**: tras un trade, el motor salta a `entry_bar + bars + 1`.
- **Sin look-ahead intrabar**: el trailing comprueba el SL HEREDADO antes de
  actualizar peak/SL con el HIGH actual. Espejo del fix en `revalidate_v15.py`.
- **MTF shift(1)**: TODO feature daily lleva shift(1) explícito.
- **Cutoff 2025-12-31**: aplicado en `prepare_data()` con assertion.

---

## Resultados completos (real ETH 2020-2025)

### Walk-forward 12 semestres (purga 14 días)

| period | n | WR | PF | total | monthly | DD | status |
|---|---:|---:|---:|---:|---:|---:|---|
| 2020-01 | 11 | 27.3% | 0.54 | -13.8% | -3.08% | 22.7% | [-] |
| 2020-07 | 31 | 35.5% | 1.32 | +19.3% | +3.26% | 36.5% | [+] |
| 2021-01 | 35 | 37.1% | 1.57 | +44.9% | +8.16% | 39.1% | [+] |
| 2021-07 | 25 | 28.0% | 0.60 | -28.6% | -7.26% | 32.1% | [-] |
| 2022-01 |  0 | — | — | — | — | — | no-signal (BTC bear) |
| 2022-07 |  0 | — | — | — | — | — | no-signal (BTC bear) |
| 2023-01 | 23 | 39.1% | 0.93 | -4.2% | -1.00% | 14.3% | [-] |
| 2023-07 | 12 | 25.0% | 2.03 | +12.5% | +4.09% | 7.7% | [+] |
| 2024-01 | 26 | 23.1% | 0.76 | -16.4% | -3.15% | 22.9% | [-] |
| 2024-07 | 23 | 30.4% | 1.56 | +18.5% | +3.53% | 15.6% | [+] |
| 2025-01 | 42 | 33.3% | 0.50 | -47.7% | -11.85% | 47.7% | [-] |
| 2025-07 | 24 | 20.8% | 0.79 | -17.9% | -4.56% | 40.1% | [-] |

- 4/10 folds evaluados OK, 6/10 negativos.
- 2 folds 2022 stayed-out (BTC daily bear). Comportamiento correcto del filtro
  macro, pero esos folds NO suman a la evidencia de edge positivo.
- 2025 colapsa completo (semestre 1: -47.7% / DD 47.7%; semestre 2: -17.9%).

### Backtest global 2020-01-01 → 2025-12-31

- N=272, WR=31.2%, PF=1.04, total=-28.3%, annual=-5.5%, DD=77.6%
- Bootstrap p=0.624 — totalmente no significativo
- Sharpe-like 0.01

### Sanity checks
- PF 1.04 → no inflado.
- WR 31% → bajo, pero PF positivo apenas: el sistema sobrevive con winners grandes
  raros (avg ganador 5%, avg perdedor -2.2%, max 46.8% del trade del ciclo 2021).
- DD 77.6% → catastrófico. Inoperable.
- N=272 → muestra suficiente, no es ruido de tamaño.

---

## Análisis por componente (transparencia)

Para entender qué subcomponente aporta qué, corro el mismo backtest con
cada subsignal aislada (sin tuning de params, solo filtrado interno):

| Modo | N | WR | PF | total | annual | DD | bootstrap p | WF |
|------|--:|---:|---:|------:|-------:|---:|------------:|---:|
| **BOTH** (oficial) | 272 | 31.2% | 1.04 | -28.3% | -5.5% | 77.6% | 0.624 | 4/10 |
| PRIMARY only | 160 | 28.7% | 0.87 | -58.1% | -14.1% | 82.1% | 0.849 | 3/9 |
| SECONDARY only | 112 | 34.8% | 1.36 | +71.2% | +11.0% | 50.3% | 0.228 | 4/7 |

### Lectura
- **PRIMARY (momentum del ratio) DESTRUYE valor**: -58% total, DD 82%.
  Comprar ETH cuando el ratio ya está en uptrend confirmado es **late entry**:
  la rotación ya ocurrió y entras justo cuando los whales empiezan a tomar
  profit. La literatura de "breakout del ratio" no se sostiene en cripto 4h.
- **SECONDARY (oversold mean-rev) SÍ tiene retorno positivo en sample**:
  +11% anual, PF 1.36, WR 35%. Pero:
  - DD 50% sigue siendo muy alto.
  - Bootstrap p=0.23 → **NO significativo** estadísticamente.
  - WF 4/7 folds positivos (margen estrecho).

### Control test sobre SECONDARY-only
Repito el experimento de "reemplazar el ratio real por random walk con misma
estadística" sobre la subsignal SECONDARY:

| | SECONDARY-only REAL | RANDOM ratio (30 seeds) |
|--|--------------------:|------------------------:|
| Annual | **+11.0%** | mean -0.3%, median -0.8%, p25-p75 = -9.5% … +7.0% |
| Posición del REAL | **83° percentil** | |
| VERDICT | **MARGINAL** — real > p75 random pero no > p95 | |

Es decir: si tomas un ratio aleatorio con misma distribución de retornos,
en ~17% de los casos obtienes annual ≥ 11% **por puro azar**. La SECONDARY
está marginalmente arriba del azar pero no significativa.

---

## Por qué fallé honestamente

1. **La hipótesis del ratio como "rotación predictiva" está sobrevendida en la
   literatura cripto.** El ratio ETH/BTC es _descriptivo_ (medida ex-post de qué
   activo lo está haciendo mejor), no _predictivo_ con leverage estadístico en
   timeframes 4h.

2. **El ratio se mueve LENTO comparado con el precio 4h.** Por construcción es
   más estable que cualquier precio individual (cancela el componente
   sistemático). Las señales de breakout/uptrend del ratio se cumplen *después*
   de que el movimiento ya ocurrió en el precio absoluto.

3. **SECONDARY (oversold) tiene un grano de verdad**: "comprar ETH cuando está
   históricamente barato vs BTC, si BTC global está bull" es una idea de value
   defendible. Pero con DD 50% y bootstrap p=0.23, **no pasa el protocolo**.

4. **Los 2022 folds (no-signal) son honestos** — el filtro BTC bear sí
   protegió capital, pero también demostró que la estrategia NO opera en bear,
   por lo que su edge global queda restringido a periodos bull.

5. **2025 fue catastrófico** (-58% combinado entre H1 y H2). El régimen
   "BTC bull pero ETH underperformando crónico" pilló a la estrategia: el
   filtro BTC permitió entrar, el ratio estaba muchas veces oversold, pero
   el rebote esperado nunca llegó porque la rotación estructural era de ETH
   hacia BTC, no al revés.

---

## SELF-AUDIT honesto

| Pregunta | Respuesta |
|----------|-----------|
| ¿Usé el ratio real con shift(1) y reindex/ffill correcto al 4h? | Sí — `prepare_data` aplica shift(1) ANTES del reindex |
| ¿Filtro daily BTC con shift(1)? | Sí |
| ¿Trailing intrabar sin look-ahead? | Sí — espejo de Agent A / revalidate_v15 |
| ¿Una posición a la vez? | Sí — bucle avanza `i += bars + 1` tras cada trade |
| ¿Cutoff 2025-12-31 inviolable? | Sí — `assert df.index.max() <= cut` |
| ¿Re-tuneé params después de ver resultados? | **No** — PARAMS frozen en `strategy.py` desde el primer `train.py run`. Reporto la versión combinada como oficial. |
| ¿Inflé n_trades con señales solapadas? | No — 272 trades en 6 años = ~3.8/mes promedio. Honesto. |
| ¿Bootstrap con suficientes iter? | 3000 iter sobre 272 trades. |
| ¿Control de "edge vs azar"? | Sí — 30 seeds random ratio. Real cayó en 33° percentil = sin edge. |
| ¿Reporté la subsignal que SÍ funcionó? | Sí, en sección "Análisis por componente". Con su propio control: marginal, no aprobado. |
| ¿Probé SHORT? | No — el proyecto desincentiva SHORT en altcoins; no aporta. |
| ¿Resultados consistentes con la historia ETH del proyecto? | Sí — ETH ha fallado en cada versión (V7, V9, V13.03, V14, ETH-A, ETH-V2). H confirma la tendencia. |

---

## Implicaciones

### Para el portfolio
- **No agregar Agent H al bot en ninguna forma.** Combined: edge negativo + DD
  77%. Secondary-only: marginal + DD 50% + bootstrap no significativo + control
  marginal.
- ETH sigue fuera del portfolio (consistente con conclusión de ETH-A y ETH-V2).

### Para la hipótesis del ratio
- **La hipótesis "el ratio ETH/BTC contiene info que ni A ni F capturan" no se
  sostiene** con esta evidencia.
- El ratio probablemente sí tiene información, pero ya está **arbitrada** en
  cualquier breakout/momentum visible. Edge ex-ante en 4h: indistinguible de azar.
- Si alguien quisiera insistir: tendría que probar timeframes mayores (1D, 1W),
  exit más sofisticado, o usar el ratio como filtro auxiliar (no como entry
  primario). Pero el experimento natural ya fue hecho.

### Para el método del proyecto
- El **control test "random ratio vs real ratio"** demostró ser la prueba
  crítica: sin él, podríamos haber argumentado que "+11% annual de secondary
  con bootstrap p=0.23 es marginal aprobable". Con el control viendo el real
  en el 83° percentil (~p75) del random, queda claro que **el edge no es del
  ratio**, es de cómo el ETH 4h se comporta en bulls de BTC en general.
- Este control debería incorporarse al protocolo general del proyecto: cuando
  el feature principal es uno (no un ensemble), comparar contra random walk
  de mismas propiedades estadísticas es barato y revelador.

---

## Archivos entregables

- `strategy.py` — código auto-contenido (PARAMS frozen, `prepare_data`, `signal`,
  `simulate`, `run_backtest`, `metrics`)
- `train.py` — runner del experimento (cutoff, WF, bootstrap, control random)
- `analyze_components.py` — diagnóstico por subsignal + control sobre secondary
- `results.json` — métricas oficiales (combined)
- `components.json` — métricas por subsignal + control secondary
- `README.md` — este documento

---

## Comando para reproducir

```
C:/Python/python.exe experiments/agent_H/train.py
C:/Python/python.exe experiments/agent_H/analyze_components.py
```

Toma ~3 minutos. Determinístico (seeds fijas en bootstrap y control).
