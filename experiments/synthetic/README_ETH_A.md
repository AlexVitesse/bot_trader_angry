# Test — ¿A transfiere a ETH? (siguiente par)

> Aplicación honesta de Agente A (Donchian-55 + EMA daily + ATR×2.5 trailing)
> TAL CUAL sobre ETH 4h. Sin re-tuning de parámetros. Si el mecanismo es
> universal, ETH debe pasar el mismo protocolo que BTC.
> Fecha: 2026-05-19 · `test_eth_A.py`

---

## Resultado

### Real ETH 2020-2025 (in-sample)
| Métrica | ETH-A | BTC-A (referencia) |
|---------|------:|-------------------:|
| N trades | 87 | 74 |
| WR | 46.0% | 44.1% |
| PF | 1.53 | 1.66 |
| Annual | +11.3% | +11.1% |
| Max DD | 18.2% | 16.8% |
| **Bootstrap p** | **0.103** | 0.088 |

ETH-A da números **muy similares a BTC-A**: PF ~1.5, annual ~11%, DD ~18%.
El mecanismo trend-follower opera con la misma intensidad en ETH que en BTC.

### Sintético (20 series block bootstrap del ETH real)
| Métrica | Valor |
|---------|------:|
| Mediana annual | +5.8% |
| Media annual | +6.8% |
| p25-p75 | +0.3% a +13.8% |
| p5-p95 | -6.1% a +18.9% |
| Series positivas | **15/20 (75%)** |
| Series con annual > 10% | 7/20 (35%) |
| Real ETH en p5-p95? | Sí (centrado) |

### Null hypothesis (shuffle aleatorio)
| | Annual mediano |
|---|------:|
| Sintético (estructura preservada) | +5.8% |
| Null (sin estructura temporal) | +2.2% |
| **Edge atribuible a estructura** | **+3.5%** |

---

## Criterios de aprobación y veredicto

| Criterio | Umbral | ETH-A | Pasa? |
|----------|--------|------:|:-----:|
| Bootstrap p<0.05 (real ETH) | < 0.05 | 0.103 | **NO** |
| Mediana sintético > 0 | > 0 | +5.8% | SÍ |
| ≥14/20 sintéticas positivas | ≥ 14 | 15/20 | SÍ |
| Edge vs null > 5% | > 5% | +3.5% | **NO** |

**Veredicto: REJECT (marginal, no desastre).**

2 de 4 criterios pasan, 2 fallan. El mecanismo TIENE algo (15/20 positivas en
sintético, edge real +3.5% sobre null) pero **no alcanza significancia
estadística** estándar (p<0.05) ni el umbral de edge claro vs null (>5%).

---

## Contraste con los 20 pares rechazados previamente

| | ETH-A (esta prueba) | 20 alt pairs (V15 original) |
|--|---------------------|------------------------------|
| PF declarado | 1.53 | 7-20 (inflado por bugs) |
| PF medido honestamente | 1.53 | 0.75-1.72 |
| Sintéticas positivas | 15/20 (75%) | (no testeado entonces) |
| Edge vs null | +3.5% | ~0% |
| Categoría | **Marginal genuino** | Ruido del simulador |

**ETH-A es genuinamente marginal** — diferente cualitativamente de los 20 alts
que tenían bug-inflated backtests. El mecanismo de A se está midiendo
correctamente y entrega ~6% anual real, pero no suficiente para certeza
estadística.

---

## Implicaciones

### Si quisiéramos forzar ETH al portfolio
Necesitaría:
1. Combinar ETH-A con otro mecanismo (al estilo V2 = A+F en BTC) → quizás
   ETH-A + algo nuevo lleva el p combinado a <0.05
2. Pero antes había que verificar que el "otro mecanismo" funciona en ETH
3. Riesgo de selection bias por probar muchas combinaciones

### Si lo dejamos como está (recomendado)
- ETH-A REJECT bajo el protocolo
- V2 sigue siendo BTC-only
- Bot opera 1 par
- No mover capital real a ETH

### Lo que nos dice esto sobre la transferibilidad del mecanismo
A's params **están sesgados a BTC** sin re-tuning porque:
- ATR%, vol patterns, funding dynamics son diferentes en ETH
- BTC tiene cycles más limpios; ETH tiene más beta
- Re-tunear A para ETH **sería overfitting** — exactamente lo que el proyecto
  prohíbe

**Conclusión honesta**: el mecanismo trend-follower de A funciona ~igual en
ETH que en BTC, pero el ruido es mayor → la significancia no llega.

---

## Siguiente decisión

Opciones discutibles:
1. **Aceptar ETH-A REJECT y operar solo BTC** (más simple, defendible)
2. **Construir ETH-V2 propio** (A-style + F-style para ETH, validado por separado) — bigger work, decisión del usuario
3. **Probar SOL u otro** par siguiente con el mismo protocolo
4. **Pausar para implementar V2 BTC en el bot** (lo más práctico — ya tenemos
   candidato validado)

---

## Script
- `test_eth_A.py` — runner del experimento
