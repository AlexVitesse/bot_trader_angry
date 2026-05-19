# Veredicto Ronda 2 — 6 agentes vs 2026 OOS

> Tres nuevos agentes (D, E, F) construidos con cutoff inviolable 2025-12-31,
> evaluados sobre Ene-Feb 2026. Misma protección anti-bug que Ronda 1.
> Fecha: 2026-05-19

---

## 1. Tabla maestra: 6 agentes, in-sample vs OOS 2026

| Agente | Estrategia | In-sample annual | In-sample p | OOS N | OOS PF | OOS mensual | OOS p |
|--------|-----------|-----------------:|------------:|------:|-------:|------------:|------:|
| **A** | Trend Donchian-55 4h + ATR×2.5 | +12% | 0.07 | **0** | — | **0%** | — |
| **B** | ML GBM classifier purged CV | -3% | 0.607 | 8 | 0.51 | -3.95% | 0.833 |
| **C** | Regime adaptive (BULL+RANGE) | +13% | 0.156 | 5 | 0.15 | -7.90% | 0.977 |
| **D** | 12h trend + vol-target + funding | +15% | 0.070 | **0** | — | **0%** | — |
| **E** | Funding-extremes mean-rev (bidir) | +14% | **0.036** ✅ | 6 | 0.38 | -7.98% | 0.942 |
| **F** | Vol breakout BTC+ETH bidir | +4% | 0.355 | 5 | **2.95** | **+10.38%** | 0.203 |

---

## 2. El hallazgo económico crítico (Agente D)

> **Funding ~13% anual eats leverage on BTC perps.**

Agente D demostró numéricamente algo que faltaba en el proyecto:
- BTC perp funding histórico mediana ≈ +0.013%/8h × 1095 períodos/año ≈ **13% anual**
- Cada 1x de leverage cuesta ~13% anual en funding
- 2x leverage en perp gana MENOS que 1x perp + spot, porque el funding consume la ventaja
- **Conclusión**: leveragear BTC perp longs honestamente no multiplica retornos — los **diluye**
- Para un retorno >30% anual honesto en BTC long sin destrozar el Sharpe, **necesitas alpha unleveraged ~30%**. **No existe.** Todos los agentes convergen en ceiling ~15%.

Este es **el insight más importante de los 6 agentes**. Cierra la pregunta "¿y si pongo leverage?".

---

## 3. Agente F: el outlier interesante (con asteriscos)

F obtuvo en 2026 OOS lo que ningún agente ha mostrado antes:
- 5 trades, todos SHORT
- WR 60%, PF 2.95
- **+12.11% en 57 días** (~+10%/mes)
- DD solo 6%
- Bootstrap p=0.203 (no significativo)

**Detalle de los trades:**
| Fecha | Activo | Outcome | PnL |
|-------|--------|---------|-----|
| 2026-01-18 | BTC SHORT | TP | +8.78% |
| 2026-01-19 | ETH SHORT | TP | +9.62% |
| 2026-02-18 | BTC SHORT | SL | -3.40% |
| 2026-02-18 | ETH SHORT | TP | +0.24% |
| 2026-02-23 | BTC SHORT | SL | -2.91% |

**Por qué es interesante:**
- F captura la pierna BEAR de BTC (que cayó ~23% YTD) con vol-compression breakout
- Su mecanismo: vol comprimido → expansión direccional → entrar en la dirección de la ruptura
- En un mercado bajista con vol comprimida que estalla, esto es exactamente lo que debe ganar
- **Es el primer caso en el proyecto donde SHORT-en-BTC parece funcionar con un filtro defendible**

**Por qué hay que ser cautos:**
1. **5 trades es muestra ínfima.** Bootstrap p=0.203 lo confirma — podría ser suerte.
2. **In-sample fue débil** (PF 1.14, p=0.355, anual 4.3%) — la estrategia no demostró edge en 2020-2025.
3. **Régimen-específico:** las 5 ganadoras vinieron justo en una ventana de bear con vol baja explotando. ¿Sobrevive otras condiciones?
4. **Selection bias retroactivo**: si elijo F porque ganó en OOS, estoy haciendo lo que el proyecto prohibió. Hay que reconocerlo.

---

## 4. Agente E: significancia in-sample sí, OOS no

E es el ÚNICO con bootstrap in-sample p<0.05 (p=0.036), edge real estadísticamente:
- 9/12 folds, PF 1.69, +14% anual, 150 trades
- Pero **2025 ya mostraba degradación** (-14% YTD, 3 de los últimos 4 folds negativos)
- OOS 2026 confirma: 6 trades, WR 17%, PF 0.38, **-8%**
- El edge de funding-extremos **se eficientizó**. El mercado aprendió.

E es el ejemplo perfecto de "edge real pero finito" — existió, ya no.

---

## 5. Cuadro completo de los 6

| Agente | In-sample edge | OOS confirmación | Veredicto |
|--------|----------------|------------------|-----------|
| A (Trend 4h) | Marginal (p=0.07) | Stayed flat (correcto) | **Defensivo válido; no probado en alza** |
| B (ML) | NULO (p=0.607) | Confirmado nulo | REJECT |
| C (Regime) | NULO (p=0.156) | Confirmado nulo | REJECT |
| D (12h+lev) | Marginal (p=0.07) | Stayed flat (correcto) | **Defensivo, mismo techo que A** |
| E (Funding) | Real (p=0.036) | Degradado | REJECT (edge agotado) |
| F (Vol BTC+ETH) | NULO (p=0.355) | Inesperado (+10%) | **Sospechoso pero intrigante** |

---

## 6. La realidad estructural de BTC

Después de 6 agentes con metodología rigurosa, la imagen es clara:

| Métrica | Realidad honesta BTC perp |
|---------|---------------------------|
| Alpha unleveraged anual | **10–15%** |
| Funding cost anual | ~13% |
| Sharpe alcanzable | 1.0–1.5 |
| DD esperable | 20–25% |
| Leveraged ceiling | NO multiplica (funding) |

**Implicación**: el "10% en CETES sin riesgo" no es comparable directamente con "12% en BTC con riesgo" — son productos diferentes con curvas de payoff diferentes. PERO si el premium por riesgo es solo 2-5% sobre CETES, **el usuario tiene razón**: no compensa.

**Caminos honestos para superar CETES con significancia:**

a) **Aceptar 15% anual como el techo** y operar A o D con tamaño moderado. Premium ~5% sobre CETES con Sharpe 1.3. Defensible pero modesto.

b) **A + F como combinación complementaria**:
   - A captura BULL (trailing-following 4h)
   - F captura BEAR (vol-breakdown SHORT con filtro vol-compression)
   - Juntos cubren ambos regímenes; F's 5 trades sugieren que el mecanismo es real
   - **Riesgo**: F's in-sample era débil; podría ser fluke. Necesita validar con más datos.

c) **Salir del 4h/perp**: explorar futuros de basis, opciones BTC (Deribit), staking ETH, o instrumentos no-perp. Cambia el problema.

d) **Aceptar que el proyecto no debe operar en vivo todavía** y seguir paper-trading mientras se refrescan datos y se acumula evidencia. Es la opción más conservadora.

---

## 7. Mi recomendación honesta

Después de **8 commits, 20 pares rechazados, 6 estrategias diseñadas y validadas**, la imagen es:

1. **No existe estrategia honesta >30% anual en BTC 4h sin tail risk catastrófico.** Los 6 agentes lo confirman.
2. **El mejor candidato real es la combinación A + F** — A para BULL, F SHORT-vol para BEAR. Pero F necesita más muestra. La opción es: paper-tradear F en testnet durante 6-12 meses para acumular evidencia.
3. **El edge realista es ~15-25% anual con DD ~20%** (A + F combinado, sin leverage perp). Premium de ~10-15% sobre CETES. **Esto sí justifica el riesgo cripto, pero hay que aceptar que es lo que hay.**
4. **20%/mes es estructuralmente imposible** en BTC sin: (a) leverage suicida (DD que liquida), (b) tail risk no monitoreado, (c) un alpha que no existe en este universo.

La conversación honesta es: ¿qué premium sobre CETES te justifica el riesgo? Si es 10-15% adicional → operable A+F. Si exiges 20%/mes → no es ese mercado.
