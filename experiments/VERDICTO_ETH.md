# Veredicto Consolidado — ETH no es operable con metodología honesta (4h)

> Tras 6 enfoques distintos probados a fondo, ETH/USDT 4h no genera edge
> estadísticamente significativo con ningún mecanismo convencional.
> Fecha: 2026-05-19

---

## Tabla maestra — los 6 experimentos en ETH

| # | Enfoque | Annual | Bootstrap p | Veredicto |
|---|---------|-------:|------------:|-----------|
| 1 | **ETH-A** (Donchian trend, params BTC) | +11.3% | 0.103 | REJECT marginal |
| 2 | **ETH-F** (vol-compression breakout) | -4.7% | 0.738 | REJECT (lastre) |
| 3 | **ETH-V2** (A + F combinado) | +0.2% | 0.484 | REJECT (sample desafortunado) |
| 4 | **ETH-G** (ML LightGBM 16 features incl. ratio) | -20.9% | 0.808 | REJECT |
| 5 | **ETH-H** (ETH/BTC ratio rotation) | -5.5% | 0.624 | REJECT |
| 6 | **ETH-I** (mean-reversion en RANGE) | -5.3% | 0.760 | REJECT (3 trades/año) |

**Ninguno alcanza bootstrap p<0.05.** El "mejor" (ETH-A) llega a 0.103 — fuera
del umbral estándar. El sintético sugiere que ETH-V2 podría tener edge esperado
~+12% (Test paradox documentado), pero no demostrable en el sample real.

---

## Por qué ETH es más difícil que BTC (hipótesis con datos)

### 1. Menos estructura de tendencia
ETH es más beta a BTC. Cuando BTC tiende, ETH tiende ~igual pero con más ruido.
Agent A (trend) en ETH dio +11.3% vs BTC +12% — similar nivel base, pero
bootstrap p subió de 0.07 (BTC) a 0.103 (ETH) por mayor varianza.

### 2. Menos tiempo en RANGE
Agent I midió empíricamente: ETH pasa **9.7% de bars en RANGE** vs BTC **~29%**.
Mean-reversion en RANGE es muchísimo menos operable en ETH simplemente porque
ETH casi nunca está en rango limpio — siempre está tendendo o transicionando.

### 3. El ratio ETH/BTC es signal lenta
Agent H midió: el ratio es feature importante (top rank) pero **late entry**.
Cuando el ratio rompe al alza en daily, el move ya pasó en 4h. La literatura
está sesgada hacia frames más largos (1D-1W) donde sí funciona.

### 4. ML overfittea peor en ETH
Agent G: train AUC 0.805 → test AUC 0.513 (**el gap más grande del proyecto**).
ETH tiene más "noise relative to signal" que BTC, lo que hace que cualquier
ML capture relaciones aparentes que no generalizan.

### 5. F (vol-breakout) es lastre genuino
Agent F (Round 2) y validación en V2 confirman: F_ETH PF 0.89-1.02, p≈0.7+.
La mecánica de vol-compression breakout funciona en BTC pero no en ETH —
posiblemente porque ETH tiene microestructura distinta (más participación
DeFi / retail) que destruye las compresiones limpias.

---

## Lo que NO se ha probado (y posibles caminos futuros)

Estas son hipótesis NO testeadas en esta ronda. No están descartadas, solo
no probadas. **Cualquier nuevo intento debe seguir el mismo protocolo riguroso**:

1. **Timeframes mayores** (1D, weekly): ETH tiene ciclos macro limpios; los
   trends de meses pueden ser explotables sin el ruido de 4h.

2. **Pares spot/staking en lugar de algo perp**: stETH/ETH arb, Lido vs
   Curve, etc. No es trading direccional, es yield/arb — distinto problema.

3. **Estrategias macro/news-driven**: upgrades de ETH (Merge, Pectra, etc.),
   ETF news, regulatory events. Requiere LLM o feed estructurado. NO se puede
   backtestear honestamente (look-ahead del LLM).

4. **ETH/BTC ratio en daily** (no 4h): la literatura funciona en 1D porque
   el ratio tiene momentum a más largo plazo.

5. **DeFi yield**: staking ETH (~3-4% APR) + restaking (~3-7% extra) + lending
   premium. Sin riesgo direccional. Premio modesto pero seguro.

---

## Resultado para el bot

**ETH se queda DEFINITIVAMENTE fuera del bot V2 4h.**

- ML_V15_PAIRS = `['BTC/USDT']` único
- ETH no recibe ni paper trading testnet con estrategia direccional 4h
- Si se quiere exposición a ETH: spot DCA o staking (productos distintos)
- Cualquier futuro intento de ETH 4h algo trading necesita demostrar:
  - Mecanismo NUEVO (no los 6 ya probados)
  - Mismo protocolo (cutoff, bootstrap, sintético, 2026 OOS)
  - Bootstrap p<0.05 reproducible

---

## Lección general del proyecto

Después de:
- **22 pares originales** rechazados (revalidación)
- **6 agentes BTC** probados (A,B,C,D,E,F) → 1 candidato real (V2)
- **6 enfoques ETH** probados (A,F,V2,G,H,I) → 0 candidatos
- **2 tests sintéticos** (learn-from-losses, extra-features)
- **1 test combinado** (BTC V2 validado)

La conclusión empírica del proyecto es:

> **Solo BTC 4h con la combinación A+F (trend + vol-breakout) tiene edge
> estadísticamente significativo bajo medición honesta.** Esa es la única
> estrategia con bootstrap p<0.05 en TODO lo testeado.

Todo lo demás:
- O bug del simulador (los 20 alts originales)
- O sin edge real (ML, ratio rotation, mean-reversion)
- O edge marginal indistinguible de azar (ETH-A, ETH-V2)

**No es porque "lo hicimos mal" — es la realidad estructural del mercado.**
Crypto 4h es eficiente o cuasi-eficiente. El edge real disponible es modesto
(BTC V2 ~8-15% anual esperado) y NO transfiere a otros pares sin pérdida
significativa de calidad.

---

## Recomendación final del proyecto

Camino honesto:

1. **Bot opera SOLO BTC V2** (mismas conclusiones de VERDICTO_RONDA2)
2. **Paper trade 6-12 meses** para acumular evidencia
3. **Capital real solo si**: real consistente con sintético + ventana alcista valida A_LONG
4. **No expandir a más pares** salvo se descubra mecanismo genuinamente nuevo
5. **Si quieres más retorno**: aumentar capital en BTC (manteniendo % riesgo)
   más leverage prudente, NO buscar más estrategias

El proyecto **ha agotado las opciones razonables** en BTC y ETH 4h. Cualquier
ronda adicional tiene probabilidad alta de retorno marginal y riesgo de
selection bias por exceso de tests.

---

## Archivos

- `experiments/agent_G/` — ML LightGBM
- `experiments/agent_H/` — ETH/BTC ratio
- `experiments/agent_I/` — Mean-reversion RANGE
- `experiments/synthetic/README_ETH_A.md` — ETH-A solo
- `experiments/synthetic/README_ETH_V2.md` — ETH-V2 combinado
- `experiments/VERDICTO_RONDA2.md` — visión BTC + ETH-F
