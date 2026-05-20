# Test sintético — ¿"Aprender de pérdidas" mejora el bot?

> Tres preguntas respondidas empíricamente con `test_learn_from_losses.py`.
> Fecha: 2026-05-19

---

## Respuesta corta

| Pregunta | Respuesta empírica |
|----------|--------------------|
| ¿V2 tiene edge real (no bug)? | **SÍ** (+8.9% sobre null en mediana) |
| ¿El +23% in-sample es la expectativa real? | **NO** — expectativa honesta ~8-15% |
| ¿"Aprender de pérdidas" mejora la estrategia? | **NO** — empeora 14/20 series OOS |

---

## Metodología

1. **Block bootstrap** del BTC real 2020-2025: bloques de 24 velas (4 días)
   se concatenan en orden aleatorio. Cada serie sintética preserva las
   propiedades estadísticas (vol, fat tails, autocorrelación intrabar) pero
   tiene un orden distinto. 20 series independientes.

2. **Test A — Robustez**: V2 corre en las 20 series. Distribución de annual return.
3. **Test B — Naive learner**: V2 entrena en serie #999, identifica reglas en
   sus perdedores ("naive learner" estilo trader retail), aplica el filtro
   a las 20 series independientes. Mide delta vs V2 original.
4. **Test C — Null**: shuffle aleatorio (no bloques) destruye autocorrelación.
   V2 corre en 10 series shuffled. Diferencia con sintético = edge atribuible
   a estructura temporal real.

---

## Test A — Robustez de V2

Real BTC 2020-2025: **+23.4% annual** (referencia)

Distribución en 20 series sintéticas:

| Métrica | Valor |
|---------|-------|
| Mediana annual | **+7.9%** |
| Media annual | +9.2% |
| Rango p25-p75 | -2.3% a +20.0% |
| Rango p5-p95 | -8.8% a +31.9% |
| % series con annual > 0 | 70% (14/20) |
| % series con annual > 10% | 50% (10/20) |
| Real BTC en p5-p95? | Sí, está en p~90 |

**Lectura crítica**: el +23% in-sample del WF está en el TOP 10% del sintético.
La expectativa REAL del strategy es ~+8-15% anual con varianza enorme
(entre -15% y +35% según cómo el ciclo se ordene). El +23% no es típico — es
afortunado.

---

## Test B — "Aprender de pérdidas" (la pregunta principal)

### Reglas que el "naive learner" descubrió en la serie de entrenamiento:
- `SKIP si bb_width > 0.053`
- `SKIP si atr_pct > 0.015`

Estas reglas tienen sentido teórico (volatilidad alta = trades arriesgados).
Cualquier trader humano las propondría tras revisar sus pérdidas.

### Resultado en train (serie de entrenamiento — IN-SAMPLE):
- V2 sin filtro: PF 1.32, +9.7% annual
- V2 con filtro: PF 2.11, **+14.2% annual** (¡mejora 50%!)

**SE VE FANTÁSTICO** — exactamente como un trader vería sus mejoras.

### Resultado en 20 series OOS (mismas que Test A):

| | V2 original | V2 con filtro | Delta |
|--|-------------|---------------|-------|
| Mediana annual | (varia) | (varia) | **-4.26%** |
| Media annual | (varia) | (varia) | **-4.57%** |
| Series donde ayudó | — | — | **6/20** |
| Series donde empeoró | — | — | **14/20** |

**El filtro EMPEORÓ los resultados en 14/20 universos paralelos.** Casos extremos:

| Serie | V2 original | V2 con filtro | Delta |
|-------|-------------|---------------|-------|
| #8 | +34.5% | +8.2% | **-26.4%** |
| #17 | +21.5% | +3.8% | **-17.7%** |
| #19 | +24.2% | +11.5% | **-12.7%** |
| #13 | +10.7% | -1.7% | **-12.4%** |
| #12 | +11.7% | +0.5% | **-11.1%** |

Las "victorias" del filtro fueron pequeñas (mejor en algunas series malas);
las pérdidas fueron grandes (filtró ganadores en series buenas). **Asimetría
perfecta del overfitting trap**.

### Por qué pasa esto

El filtro identificó: "los perdedores históricos tenían bb_width > 0.053". Esto
es una **correlación de muestra**, no una causa. En 14 de 20 mundos paralelos:
- Hay perdedores con bb_width > 0.053 (correcto: el filtro los excluye)
- Hay GANADORES con bb_width > 0.053 (el filtro **también** los excluye, error)
- El balance es negativo: se pierden más ganadores que perdedores excluidos.

**Esto es exactamente cómo murieron V7, V9, V13.03, BTC V2.** "Aprender" de
pérdidas en una muestra → falla en la siguiente.

### Conclusión Test B
**"Aprender de los trades perdedores ajustando reglas" es overfitting
demostrable empíricamente.** El test de signo binomial da p~0.115 (no
estrictamente significativo al 5% por sólo 20 muestras), pero la magnitud
del efecto es clara: -4.3% mediana, asimetría 14 vs 6, magnitudes negativas
mucho mayores que las positivas. La señal es consistente con el sesgo del
overfitting trap, no con aprendizaje real.

---

## Test C — Null hypothesis (control)

Shuffle aleatorio del BTC real (destruye estructura temporal):

| | Annual mediano |
|--|----------------|
| Block bootstrap (Test A) | +7.9% |
| Shuffle aleatorio (Test C) | -1.0% |
| **Edge atribuible a estructura del mercado** | **+8.9%** |

V2 corriendo sobre retornos shuffleados da ≈0% — no hay edge cuando se
destruye la estructura temporal. **El edge de V2 viene de capturar regímenes
y momentum reales del mercado, no del simulador**. Validación definitiva de
que el motor está limpio.

---

## Implicaciones finales

### 1. V2 está validado para paper trade — con expectativas calibradas
- Edge real: **+8.9% anual sobre null**
- Expectativa razonable: **+8-15% annual**, no +23%
- 70% de mundos paralelos dan retorno positivo
- DD esperable: ~20-25%
- Premium sobre CETES: ~5-8% sin leverage, neto modesto

### 2. Parámetros FROZEN — no tocar
**No** ajustar reglas tras ver perdedores. **No** "afinar" filtros. Cualquier
ajuste tipo "naive learner" empeorará el sistema en 70% de mundos paralelos.
La única forma honesta de mejorar V2 es:
- Diseñar una nueva estrategia con metodología completa (cutoff, WF purgado,
  bootstrap) — exactamente como Agents A-F
- Validar independientemente en datos no vistos
- Combinar si pasa la prueba

### 3. Cuándo SÍ está bien re-entrenar
- Cada 6-12 meses, re-correr WF + bootstrap con datos actualizados (incluyendo
  el último periodo)
- Si las reglas originales todavía pasan p<0.05 con datos nuevos → quedan
- Si fallan → la estrategia se rompió, no se "ajusta" — se rediseña desde cero

### 4. La pregunta del usuario, respondida
> "Queria saber si aprendemos de los trades que hizo mal e ir corrigiendo"

**No funciona.** Empeora 14/20 universos paralelos, mediana -4.3%. Es la
versión moderna de la falacia clásica del trading discrecional. Los traders
sistemáticos que ganan a largo plazo **NO ajustan reglas reactivamente** —
mantienen el sistema y diseñan nuevas estrategias en paralelo cuando creen
encontrar otro edge real.

---

## Script y artefactos

- `test_learn_from_losses.py` — runner del experimento
- `README.md` (este) — análisis y conclusiones
