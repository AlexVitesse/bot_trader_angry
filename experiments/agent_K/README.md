# Agent K — V2 (A + F_BTC) + on-chain features

## TL;DR — Honest verdict

**NEGATIVO.** Anadir features on-chain de Coin Metrics community (MVRV,
exchange flows, active addresses, hashrate) a la estrategia V2 NO produce un
edge estadisticamente distinguible del azar.

| Test | Resultado |
|------|-----------|
| In-sample baseline V2 | n=163, PF=1.59, annual=+21.1%, DD=23.9%, **p=0.030** |
| In-sample V2 + on-chain filter (Modo A) | n=150, PF=1.67, annual=+22.0%, DD=23.9%, **p=0.025** |
| Delta bootstrap p | **+0.005** (mejora insignificante) |
| Cross-check 1: ON-CHAIN BLOCK-SHUFFLED (20 seeds) | median p=0.038, **35% de seeds aleatorios igualan o superan al real** |
| Cross-check 2: random DROP de 25 trades (50 seeds) | median p=0.055, **16% de drops aleatorios igualan o superan al real** |
| ML hybrid (Mode B) test AUC con on-chain | **0.451** vs tech-only 0.479 (on-chain *empeora* generalizacion) |
| OOS 2026 (Ene-Abr) | 2 vetoes correctas (F_SHORT en capitulacion), pero n=2 |

El "umbral" estadistico que se suele exigir es: cross-check con random debe
verse **claramente peor** (p random > p real + 0.05 con baja varianza). Aqui
35% de los seeds shuffleados producen un p igual o mejor que el real
(0.025). Conclusion: el filtro on-chain "funciona" en in-sample por
selection bias, no por contenido informacional real de las metricas.

---

## Que se hizo

### Fuente de datos on-chain

**Coin Metrics community API** (gratis, sin auth, instalable con
`pip install coinmetrics-api-client`). Endpoint base
`https://community-api.coinmetrics.io/v4/`.

Metricas obtenidas (diarias, 2019-06-01 -> 2026-04-30, 2526 dias):

| Metrica | Significado |
|---------|-------------|
| `CapMVRVCur` | Market-Value-to-Realized-Value (clasico para market cycles) |
| `AdrActCnt` | Active addresses (proxy adopcion) |
| `FlowInExNtv` / `FlowOutExNtv` | Exchange flows en BTC nativo |
| `HashRate` | Hash rate de la red (proxy confianza minera) |
| `TxCnt`, `TxTfrCnt` | Tx count y transfer count |
| `SplyCur`, `SplyExNtv` | Supply total y supply en exchanges |
| `CapMrktCurUSD`, `CapMrktEstUSD` | Market cap y realized cap proxy |

**SOPR** NO esta disponible en la tier community (es de pago en
Glassnode/CryptoQuant). Se uso `CapMVRVCur` como proxy de "value ratio
fundamental".

### Anti-look-ahead crítico

Los datos on-chain publicos tienen un delay de publicacion de T+1 dia
tipico (Coin Metrics community). Para eliminar look-ahead se aplica:

```python
df_onchain_features.shift(2)   # 2 dias de margen, conservador
                .reindex(df_4h_index, method='ffill')
```

Asi, en la vela 4h del dia D, las features on-chain representan datos
de **dia D-2** ya disponibles publicamente. Si se shifteara solo 1 dia,
algunos timestamps 4h podrian ver datos no publicados aun.

### Disenño del filtro (Modo A — filtro a-priori)

Sobre V2 baseline (A primero, F_BTC segundo, una posicion a la vez),
**sin re-tunear A ni F**, se aplica:

- **MVRV z-score > 2.5** → bloquea LONG (sobrecomprado historico)
  - z-score rolling 90 dias
  - Threshold a-priori: 2.5 ≈ percentil 99 (1% de extremo superior)
- **Exchange netflow z-score > 2.0** → bloquea LONG (presion vendedora)
  - z-score rolling 90 dias del net flow (FlowIn - FlowOut)
- **MVRV z-score < -1.5** → bloquea SHORT (capitulacion / fondo probable)

Todos los thresholds elegidos A-PRIORI basados en literatura on-chain
(Glassnode, Coin Metrics research). NO se barre el rango buscando mejor
p (eso seria look-ahead).

### Sensitividad (diagnostico solo, NO usado para tunear)

```
MVRV LONG (manteniendo otros)    n    PF    annual    p
       1.50                    124   1.63   +16.0%   0.052
       2.00                    140   1.71   +21.6%   0.023
       2.50  <-- a-priori      150   1.67   +22.0%   0.025
       3.00                    153   1.68   +22.6%   0.022

MVRV SHORT                       n    PF    annual    p
      -2.50                    156   1.65   +22.1%   0.024
      -1.50  <-- a-priori      150   1.67   +22.0%   0.025
      -0.50                    138   1.71   +22.7%   0.023

Exchange flow LONG               n    PF    annual    p
       1.00                    143   1.60   +18.6%   0.041
       2.00  <-- a-priori      150   1.67   +22.0%   0.025
       3.00                    151   1.69   +22.9%   0.023
```

**Lectura:** todos los thresholds razonables dan p ∈ [0.02, 0.05]. Es decir
NO hay sweet spot — y eso refuerza la conclusion: la "mejora" es ruido,
no senal.

---

## Cross-checks de robustez

### Cross-check 1: on-chain block-shuffled (20 seeds)

Permuto en bloques de 30 dias TODAS las features on-chain (preservando
autocorrelacion local pero rompiendo correlacion con precio). Aplico el
mismo filtro.

```
seed=1: vetoed=18, p=0.050     seed=11: vetoed=28, p=0.062
seed=2: vetoed=14, p=0.025     seed=12: vetoed=13, p=0.016  <- mejor que real
seed=3: vetoed=17, p=0.048     seed=13: vetoed= 5, p=0.022  <- mejor que real
seed=4: vetoed=27, p=0.078     seed=14: vetoed= 7, p=0.019  <- mejor que real
seed=5: vetoed=11, p=0.036     seed=15: vetoed=14, p=0.047
seed=6: vetoed=13, p=0.021     seed=16: vetoed=11, p=0.052
seed=7: vetoed=20, p=0.040     seed=17: vetoed=19, p=0.065
seed=8: vetoed=10, p=0.017     seed=18: vetoed= 9, p=0.032
seed=9: vetoed=24, p=0.032     seed=19: vetoed=18, p=0.055
seed=10: vetoed=22, p=0.049    seed=20: vetoed=10, p=0.024

Real on-chain p:        0.025
Median random p:        0.038
% random p <= real p:   35%
```

**7 de 20 seeds aleatorios producen p igual o mejor que el real.** Si las
features on-chain reales tuvieran edge, esperariamos < 5% de los seeds
random igualando o superando. **35% = sin senal real.**

### Cross-check 2: drop aleatorio de trades (mismo N)

Como control adicional, drop-eo aleatoriamente 25 trades del baseline V2
(sin importar features). Esto mide el efecto puro de subsampling:

```
Real on-chain p:        0.025
Median random-drop p:   0.055
% random-drop p <= real p:  16%
```

El filtro real es marginalmente mejor que un drop aleatorio (8 de 50 drops
aleatorios alcanzan el p del real). Esta es la "ventaja" honesta: ~5 puntos
porcentuales menos de seeds random igualan o superan al real, comparado con
el placebo de pure shuffling. Aun asi 16% > 10% = umbral de "indistinguible".

### Mode B: ML hybrid (chequeo cualitativo)

Entreno `GradientBoostingClassifier` (max_depth=2, n=50, lr=0.05) sobre los
163 trades V2 con walk-forward de 4 folds. Features: 7 tecnicas + 5 on-chain.

```
            Train AUC   Test AUC   Cumulative kept
tech_only      0.927       0.479         +137.5%
tech + on-chain 0.963       0.451         +72.1%   <- on-chain EMPEORA test AUC
oc_only        0.885       0.470         +118.4%
```

**Train AUC ~0.95 vs Test AUC ~0.45-0.48 = overfitting masivo.** Anadir
on-chain a los features tecnicos REDUCE el AUC de test de 0.479 a 0.451.

Esto es coherente con lo que Agent B encontro para BTC y Agent G para ETH:
ML clasifico sobre 4h BTC con pocos features no generaliza. Anadir mas
features (on-chain) empeora la generalizacion, no la mejora.

---

## OOS 2026 (Ene-Abr) — anecdotico, n=3

Es la unica ventana realmente fuera de muestra. Trades:

| Fecha | Setup | Outcome | PnL | Vetoed? |
|-------|-------|---------|-----|---------|
| 2026-01-18 | F_SHORT | TP | +8.78% | NO (mvrv_z neutro) |
| 2026-02-18 | F_SHORT | SL | -3.40% | **SI** (mvrv_z=-2.11 < -1.5: capitulacion) |
| 2026-02-23 | F_SHORT | SL | -2.91% | **SI** (mvrv_z=-1.83 < -1.5: capitulacion) |

El filtro SHORT-block-on-capitulacion **se llevo los 2 unicos losers** OOS y
dejo pasar el winner. Total V2 baseline = +2.0%; con on-chain filter = +8.8%.

**Pero n=2 vetoes en OOS es anecdotico** — no permite distinguir senal de
azar. Hace falta el primer bear sostenido tras la regla para verificar si
el filtro MVRV<-1.5 captura efectivamente capitulaciones genuinas.

---

## Interpretacion

### Por que no funciona?

1. **Los precios YA reflejan el on-chain.** Las metricas como MVRV son
   derivadas del precio (cap. mercado / cap. realizado). En 4h, el mercado
   ya proceso la info en cuestion de horas, no dias. Cuando los datos se
   publican con T+1 delay, el move ya ocurrio.

2. **El proyecto YA usa proxies on-chain via funding.** El funding
   z-score (que ya esta en A y F) captura "presion direccional / euforia"
   que tradicionalmente requeria on-chain. Las exchange flows aportan
   informacion solapada — no marginal.

3. **El sample es chico.** 163 trades en 6 anos no permite distinguir
   bootstrap p=0.025 de p=0.030. La diferencia esta en el ruido.

4. **MVRV es feature LENTA.** El paper TFT cita +6% edge en 2 semanas con
   SOPR+TVL+AA en ETH-USDT, pero en muestra de solo ~50 trades. A 4h en BTC
   con 163 trades, ese efecto se diluye.

### Que SI mostro contenido

El veto de SHORT en capitulacion (MVRV z < -1.5) interceptaria 4 trades
in-sample con avg PnL -3.4% (analizando el subset). En OOS 2026 intercepto
2 de 3 losers. Es la regla mas plausible — captura la intuicion clasica
"don't short at the bottom". Pero el n es insuficiente para validarla
estadisticamente y el cross-check no la separa del azar.

### Que NO mostro contenido

- **Active addresses, hash rate, supply on exchanges**: distribuciones
  amplias, sin patron discriminante.
- **MVRV > 2.5 como filtro LONG**: 23 trades vetoed, avg PnL -0.64%,
  WR 35%. Suena util pero el cross-check shuffled lo replica.
- **Exchange net inflow > 2σ**: solo 2 trades vetoed in-sample. Muestra
  irrelevante.

---

## Comparacion con literatura

| Source | Claim | Tested here? | Result |
|--------|-------|--------------|--------|
| Paper TFT junio 2025 | SOPR+TVL+AA → +6% ETH 2 sem | NO (SOPR es Glassnode pago) | n/a |
| Glassnode Studio | MVRV > 3.7 = topes ciclo | SI (con z-score equivalente) | NO replica en 4h |
| Coin Metrics research | Active addresses = adopcion | SI | sin edge marginal |
| "Untapped alpha" claims | exchange flows predicen moves | SI | sin edge marginal |

La diferencia: estas referencias trabajan en timeframes 1D-1W con muestras
de cientos de trades en multiples activos. En BTC 4h con un sistema YA
filtrado por trend (A) y compression (F), no queda informacion marginal
detectable.

---

## Validacion del protocolo

- ✅ Cutoff <= 2025-12-31 respetado (in-sample). OOS 2026 evaluado pero NO
  usado para tunear.
- ✅ Una posicion a la vez (motor de PASS 1 / PASS 2).
- ✅ Sin look-ahead intrabar (heredado de A y F que ya lo aplican).
- ✅ On-chain shift(2 dias) ANTES de reindex 4h → publicacion delay
  considerado.
- ✅ Thresholds elegidos a-priori, sensibilidad solo diagnostica.
- ✅ Cross-check vs random ON-CHAIN block-shuffled (20 seeds).
- ✅ Cross-check vs random DROP (50 seeds).
- ✅ Mode B (ML hybrid) sanity check.
- ✅ Documentar resultados negativos honestamente.

---

## Implicacion para el proyecto

Cierra la hipotesis "on-chain mejora V2 BTC". Esto es informacion
valiosa: si el unico recurso genuinamente nuevo (datos on-chain
gratis-tier) tampoco aporta edge, entonces:

1. **BTC 4h con precio + volumen + funding es el limite practico** del
   approach actual. No hay un upgrade "facil" pendiente.

2. **Camino restante**:
   - Validar V2 en paper trading 6-12 meses (ya recomendado en VERDICTO).
   - Si la realidad confirma p<0.05 con trades reales: V2 es la
     estrategia a desplegar tal cual.
   - Si quieres mas retorno: aumentar capital sobre V2 (no buscar nueva
     estrategia).

3. **Si en el futuro hay acceso a metricas on-chain PREMIUM**
   (Glassnode pro: SOPR, NUPL, MVRV-Z, AVIV — todas refinadas), valdria
   reintentar. Pero con community-tier solamente, esta hipotesis queda
   refutada.

---

## Files

| File | Purpose |
|------|---------|
| `fetch_onchain.py` | Bajar datos on-chain de Coin Metrics community |
| `onchain_data.parquet` | Datos descargados (2526 dias, publicos) |
| `strategy.py` | Auto-contenido: PARAMS, features, veto, motor V2+on-chain |
| `train.py` | Runner principal — produce all results |
| `sensitivity.py` | Diagnostico de thresholds (no se usa para tunear) |
| `ml_hybrid_check.py` | Sanity check Mode B (ML hybrid) |
| `results.json` | Resultados completos en JSON |

---

## Como reproducir

```bash
cd "C:/Users/pcdec/OneDrive/Documentos/MIS EMPRENDIMIENTOS/BOTDETRADINGAGRESIVO"
C:/Python/python.exe -u experiments/agent_K/fetch_onchain.py   # 60s — baja datos
C:/Python/python.exe -u experiments/agent_K/train.py            # 90s — main run
C:/Python/python.exe -u experiments/agent_K/sensitivity.py      # 30s — diag thresholds
C:/Python/python.exe -u experiments/agent_K/ml_hybrid_check.py  # 10s — ML mode B
```

---

## Conclusion

Las features on-chain de Coin Metrics community tier, aplicadas como
filtro a-priori sobre V2 (A + F_BTC), **mejoran el bootstrap p marginalmente
(0.030 -> 0.025) pero el efecto es indistinguible de filtrado aleatorio**:

- 35% de seeds con on-chain shuffled producen p igual o mejor.
- 16% de drops aleatorios producen p igual o mejor.
- ML hybrid: anadir on-chain EMPEORA el test AUC.

Veredicto: **NEGATIVO con alto nivel de confianza**. La hipotesis del paper
TFT (on-chain como fuente de alpha "no tocada") no se sostiene en BTC 4h
con las metricas tier-community disponibles gratis.

El proyecto, despues de A,B,C,D,E,F,G,H,I,J,K probados, tiene **una unica
estrategia con edge bootstrap-significativo: V2 = A+F_BTC**. Esa es la
realidad. El siguiente paso productivo es validarla en paper trade
acumulando trades reales, no buscar mas variantes.
