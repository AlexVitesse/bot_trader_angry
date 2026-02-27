# Analisis Critico: Overfitting en Backtests

## Fecha: 27 Feb 2026

---

## El Patron Repetido

Cada vez que creamos un "mejor modelo" en backtest, falla en produccion:

| Version | Backtest | Produccion Real | Problema |
|---------|----------|-----------------|----------|
| V7 Original | "Bueno" | 33-42% WR en SOL/BTC | Fallo |
| V9 LossDetector | 68% WR | 41.4% WR | Fallo |
| BTC V2 | 65.7% WR hist | 43.8% WR en 2026 | Degradacion |
| SOL V2 | 63.6% WR hist | 12.5% WR validacion | Catastrofe |
| V13.03 (hoy) | 67.3% WR | ??? | Sin validar |

---

## Evidencia de Overfitting en V13.03

### 1. Correlacion Train vs Test (Alarmante)

| Par | Train Corr | Test Corr | Drop |
|-----|------------|-----------|------|
| XRP | 0.7366 | 0.0802 | -89% |
| ETH | 0.6543 | 0.0628 | -90% |
| AVAX | 0.7165 | 0.0810 | -89% |
| ADA | 0.6788 | 0.1127 | -83% |
| LINK | 0.6602 | 0.0752 | -89% |
| DOGE | 0.9143 | 0.0723 | -92% |
| NEAR | 0.6858 | -0.0672 | -110% (negativo!) |
| DOT | 0.6863 | 0.0383 | -94% |

**Conclusion:** Los modelos memorizan el pasado, NO predicen el futuro.

### 2. Optimizacion en Datos de Test

Hoy optimizamos TP/SL usando el periodo de test:
- Grid search en datos historicos completos
- Elegimos parametros que FUNCIONARON en el pasado
- No hay garantia de que funcionen en el futuro

Esto es **look-ahead bias** clasico.

### 3. Numeros que No Cuadran

Documentacion anterior V13.02:
- "63 trades, 76.2% WR, $+214"

Backtest V13.02 hoy:
- "166 trades, 54.2% WR, $22.59"

**Misma configuracion, resultados completamente diferentes.**

---

## Historial de "Mejores Modelos"

### SOL - La Peor Historia

1. **V7 Original**: En produccion 42% WR, -$31 (excluido)
2. **V2 Regresor**: Backtest +2265%, validacion -$15 (excluido)
3. **V3 Clasificador**: Backtest +$394, MaxDD $302 (60% del capital!)
4. **Conclusion**: SOL simplemente NO funciona con ML

### BTC - Degradacion Clara

| Ano | WR en Backtest |
|-----|----------------|
| 2022 | 73.9% |
| 2023 | 81.0% |
| 2024 | 72.1% |
| 2025 | 55.7% |
| 2026 | 43.8% |

**El modelo se degrada con el tiempo.** El mercado cambia, el modelo no.

### V9 LossDetector

- Backtest: 68% WR, +914.9%
- Despues: 41.4% WR, PF 1.08

**Fallo del 40% en WR.**

---

## Por Que Siempre Falla

### 1. El Mercado Cambia
- Patrones de 2020-2022 no aplican a 2025-2026
- Mas bots ML = mercado se adapta
- Condiciones macro diferentes

### 2. Grid Search es Trampa
- Probamos 1000+ combinaciones TP/SL
- Elegimos la que MEJOR funciono en el pasado
- Probabilidad de que sea la mejor en el futuro: muy baja

### 3. No Hay Out-of-Sample Real
- Siempre usamos datos que ya vimos de alguna forma
- "Test set" ya fue analizado visualmente
- No es realmente "unseen data"

### 4. Demasiados Parametros
Por cada par optimizamos:
- Modelo (5 opciones)
- TP% (7 opciones)
- SL% (7 opciones)
- Conviction (4 opciones)
- Direccion (3 opciones)

Total: 5 x 7 x 7 x 4 x 3 = **2940 combinaciones**

Encontrar "la mejor" en datos historicos es facil.
Que funcione en el futuro es otra cosa.

---

## Que Hacer Diferente

### 1. Walk-Forward Validation Real
- Train: Enero 2020 - Dic 2024
- Validation: Enero 2025 - Sep 2025
- Test: Oct 2025 - Feb 2026 (SIN TOCAR hasta el final)
- Optimizar SOLO en Train, validar en Validation
- Test es sagrado, solo se usa UNA vez

### 2. Parametros Robustos, No Optimos
En vez de optimizar TP/SL por par:
- Usar TP=4%, SL=2% para TODOS
- Si no funciona con parametros genericos, no funciona

### 3. Menos Pairs, Mejor Calidad
- Solo operar pares donde el modelo realmente generalice (test corr > 0.2)
- Actualmente TODOS los pares tienen test corr < 0.15

### 4. Tiempo Real es la Unica Validacion
- 2 semanas en paper trading ANTES de confiar
- Si el paper trading no replica backtest, el backtest esta mal

### 5. Expectativas Realistas
- Un bot rentable tiene 55-60% WR, no 70%+
- Si el backtest muestra 70%+ WR, esta overfiteado
- Profit Factor 1.2-1.5 es realista, no 4.0

---

## Estado Real del Proyecto

| Aspecto | Lo Que Creemos | Realidad Probable |
|---------|----------------|-------------------|
| WR esperado | 67% | 45-50% |
| PnL anual | $400 | $50-100 |
| MaxDD | 10% | 20-30% |
| Pares rentables | 10 | 2-3 |

---

## Recomendacion

### Opcion A: Simplificar
1. Solo BTC + BNB con parametros conservadores
2. TP=3%, SL=1.5%, Conv=1.5
3. Expectativa: 50% WR, $20-50/ano con $100

### Opcion B: Validacion Rigurosa
1. Crear test set "sagrado" (Oct 2025 - Feb 2026)
2. Re-optimizar TODO sin tocar ese periodo
3. Solo cuando este listo, probar en test set UNA vez
4. Si falla, volver a empezar

### Opcion C: Paper Trading Primero
1. Tomar V13.03 actual
2. Correr 2-4 semanas en paper
3. Comparar con backtest
4. Si diverge >10%, el backtest esta mal

---

## Conclusion

**V13.03 probablemente NO funcionara como el backtest sugiere.**

Los numeros son demasiado buenos (67% WR, PF 4.0) y la evidencia historica muestra que cada "mejor modelo" falla en produccion.

La unica forma de saber es probarlo en tiempo real con dinero que se puede perder.

---

*Documento creado: 27 Feb 2026*
*Proposito: Mantener expectativas realistas*
