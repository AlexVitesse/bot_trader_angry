# E1 — V2 frente a comprar BTC con la misma exposición (+ E0: MinTRL)

> **Pre-registro escrito el 2026-10-19, ANTES de correr nada.** Copia de
> `docs/PLAN_GITHUB_2026-10.md` (E0 y E1), con las precisiones de implementación
> que el plan dejaba abiertas. Los resultados van debajo, sin tocar esta sección.
> Familia de 5 tests del plan → **α = 0,01**.

## E1: brazos

Periodo: el de la curva de equity de V2 (`data/BTC_USDT_4h_full.parquet`,
2019-01 → 2026-02). Todo en retornos diarios (cierre UTC).

| brazo | definición |
|---|---|
| **V2** | `PortfolioSim`, BTC, `risk_pct=0,045`, `max_concurrent=1`, `max_notional_pct=2,0` (config viva), costes nuevos |
| **B&H-w** | BTC long siempre, exposición constante `w` = exposición media de V2, rebalanceo diario |
| **B&H-vt** | BTC long siempre, `w_t = k · min(1, σ*/σ̂)` |
| **Mixta** | 50% B&H-w + 50% V2, rebalanceo el primer día de cada mes (descriptiva) |

Precisiones:
- **Exposición de V2** en cada vela 4h con posición abierta:
  `notional · (close / entrada) / equity`. Fuera de posición vale 0. Se promedia
  sobre todas las velas.
- **σ̂**: suma diaria de la varianza Garman-Klass de las velas 4h, media móvil
  de 20 días, anualizada. **σ\***: mediana expandente de σ̂. Se usan los valores
  conocidos al cierre del día t−1 para el peso del día t (sin look-ahead).
- **k**: escalar único para que la exposición media de B&H-vt iguale a la de
  V2. Usa la media de toda la historia, un look-ahead de un solo número que
  solo sirve para igualar exposiciones.
- **Rebalanceo de B&H-vt** solo si `|Δw| > 0,1` medido en la escala sin `k`
  (`min(1, σ*/σ̂)`); con `k` la regla sería casi nunca operar.
- **Costes de los B&H** (mismo instrumento que V2, perpetuo):
  - 0,06% sobre el volumen negociado en cada rebalanceo.
  - Funding histórico sobre la exposición. Antes de 2020 se usa la constante
    `FUNDING_8H`, igual que el simulador.
  - Se reporta también la versión spot (sin funding) como información.

## E1: hipótesis y test

**H1:** CAGR/DD de V2 > CAGR/DD de B&H-vt.

**Test primario** (el del plan): diferencia de retornos log diarios
`d_t = r_V2 − r_BHvt · (σ_V2/σ_BHvt)`, es decir, B&H-vt reescalado a la misma
volatilidad. Estadístico: `mean(d)`. Null: `d` centrado, bootstrap circular por
bloques de 30 días, 20.000 réplicas. p = fracción de réplicas con media ≥ la
observada. **Pasa si p < 0,01.**

**Secundario** (descriptivo): bootstrap pareado por bloques de 30 días de los
retornos diarios conjuntos. Se reporta P(CAGR/DD de V2 ≤ CAGR/DD de B&H-vt).

**Lectura pre-acordada:**
- Si V2 no gana a B&H-vt: lo decide el usuario. Se documenta y no se cambia
  nada en vivo.
- Si gana: es evidencia de que el timing aporta algo en relación con el
  riesgo, no en retorno.

## E0: Minimum Track Record Length (sin test)

Trades de V2 de la misma corrida. Con `SR` el Sharpe por trade, `γ3` la
asimetría y `γ4` la curtosis (no exceso):

`MinTRL = 1 + [1 − γ3·SR + (γ4−1)/4·SR²] · (z_0,95 / SR)²` trades,

contra Sharpe 0 con confianza del 95%. Se convierte a años con la frecuencia
histórica de trades. Es un **suelo**: asume trades independientes, y los de V2
se agrupan por régimen.

---

## Resultados (2026-10-19) — `test_benchmark.py`, salida en `salida.txt`

### V2 está fuera del mercado el 87% del tiempo

Tiempo en mercado: **13,1%**. Exposición media: **0,176**. Cuando está dentro
va a **1,34×** de media. Igualar exposiciones significa comparar contra BTC al
17,6%, no al 100%.

### E1 — tabla (periodo pre-registrado, 2019-01 → 2026-02, curvas diarias)

| brazo | exposición | CAGR | DD | CAGR/DD |
|---|--:|--:|--:|--:|
| **V2** (4,5%) | 0,176 | +23,5% | 40,6% | **0,58** |
| B&H-w, perp | 0,176 | +8,1% | 20,8% | 0,39 |
| B&H-vt, perp | 0,176 | +6,5% | 22,8% | 0,28 |
| B&H-w, spot | 0,176 | +10,5% | 20,0% | 0,53 |
| B&H-vt, spot | 0,176 | +8,8% | 22,0% | 0,40 |
| **Mixta 50/50** (V2 + B&H-w perp) | — | +16,4% | 22,7% | **0,72** |
| B&H 100% spot (referencia) | 1,0 | +50,2% | 76,6% | 0,66 |

(El DD de V2 sale 40,6% y no 42,4% porque aquí se muestrea al cierre diario.)

**H1: NO pasa.** Diferencia de retornos a igual volatilidad: +4,5%/año a favor
de V2, **p = 0,333** (α 0,01). Bootstrap pareado: en el **19,7%** de las
réplicas V2 tiene peor CAGR/DD que B&H-vt.

La sensibilidad desde la primera fecha operable (2019-03-16) no cambia nada:
V2 0,59, B&H-w spot 0,52, Mixta 0,73. No se encontró una columna EMA200 en
las features y se usó la fecha del primer trade; el cambio es de 2 meses.

### Lectura

1. **V2 tiene mejor CAGR/DD que BTC a igual exposición, pero la diferencia no
   es distinguible del azar.** Coincide con `bootstrap_bloques/`: gana dinero,
   y lo que gana sobre la deriva no es significativo. Frente a B&H spot (sin
   funding) la ventaja es mínima (0,58 contra 0,53).
2. **Comprar y mantener BTC al 100% tiene mejor CAGR/DD que V2** (0,66 contra
   0,58), con un DD del 77% que nadie tolera. El valor práctico de V2 es
   **poder elegir el nivel de DD**, no un edge de timing.
3. **La volatilidad objetivo empeora a B&H** (0,28 contra 0,39 en perp). En
   BTC la volatilidad alta coincide con los rallies, y recortar ahí cuesta
   retorno. **Aviso previo para E2**: la evidencia externa (DD 43% → 30% en
   OctopusTakopi) venía de un libro multi-activo largo-corto. Aquí, en BTC solo
   y long-only, el mismo mecanismo resta.
4. **El funding le cuesta a un B&H en perpetuo ~2,4 puntos/año a exposición
   0,18** (8,1% contra 10,5%). Para la pata de "mantener" lo correcto es spot.
5. **Hallazgo no testeado (era descriptivo en el pre-registro):** la Mixta
   50/50 tiene el mejor CAGR/DD de la tabla (**0,72**). A un DD parecido al de
   V2 al 2% (21,5%), da +16,4% contra +11,1%. Tiene sentido: V2 está fuera el
   87% del tiempo y la pata de BTC cobra la deriva mientras V2 espera.
   **No es un resultado validado:** no se pre-registró como hipótesis, y la
   ventaja viene de la deriva de BTC 2019-2026 (CAGR 50%), que no tiene por qué
   repetirse. Si se quiere usar, hay que pre-registrarlo como test propio con
   el null sintético (la deriva es exactamente lo que el null preserva).

### E0 — MinTRL

SR por trade 0,180, asimetría +1,45, curtosis 4,80.
**MinTRL (95% contra Sharpe 0) = 65 trades ≈ 2,8 años** a 23 trades/año.

Es un **suelo**: asume trades independientes, y los de V2 se agrupan por
régimen. El "≥30 trades" de CLAUDE.md es la **mitad** de lo necesario para
distinguir V2 de cero con un 95% de confianza. Seis o doce meses de paper
(unos 12-25 trades) sirven para medir la ejecución (real contra simulado),
**no para validar el edge**.

### Por subperiodo (descriptivo) — `por_periodo.py`, salida en `salida_periodos.txt`

CAGR anualizado / DD dentro de cada tramo. **BH18** = BTC al 17,6% de
exposición. Ojo: 2019 empieza el 10 de enero y 2026 son solo 2 meses
anualizados, así que el −27% de 2026 equivale a un −5% real.

| tramo | trades | BTC | V2 | BH18 spot | Mixta |
|---|--:|--:|--:|--:|--:|
| 2019-01 → 2020-10 | 42 | +262% | **+56% / 21%** | +18% / 13% | +36% / 12% |
| 2020-10 → 2022-08 | 36 | +75% | +6% / 30% | **+10% / 18%** | +7% / 19% |
| 2022-08 → 2024-05 | 42 | +180% | +19% / 27% | +13% / 7% | +16% / 15% |
| 2024-05 → 2026-02 | 45 | −3% | **+16% / 27%** | +1% / 11% | +8% / 14% |

Por año: V2 gana a BH18 en 2019, 2020, 2024 y 2025. Pierde en **2021**
(−24% contra +14%) y **2023** (+10% contra +20%). En **2022 y 2026 hace 0
trades**: el filtro de régimen lo deja fuera mientras BTC cae un 65% y un 26%.

Lectura:
- Medido por CAGR/DD, V2 gana a BTC en 2 de los 4 bloques. Es la misma
  concentración temporal que ya mostraban los "2/6 folds" de CLAUDE.md.
- Lo que más aporta V2 es **quedarse fuera en los mercados bajistas**.
- Su DD es siempre mayor (21-30% contra 7-18%) porque, cuando está dentro, va
  a 1,34×.
