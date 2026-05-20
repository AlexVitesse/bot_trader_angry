# Veredicto Final del Proyecto — Mayo 2026

> Tras 4 rondas de agentes (11 agentes A-K), múltiples tests sintéticos,
> revalidación honesta de 22 pares y exploración exhaustiva de mecanismos,
> el proyecto llegó a una conclusión empírica robusta.
> Fecha: 2026-05-19 (commit f5d8e4c y siguientes)

---

## Tabla maestra — todo lo probado honestamente

### Pares revalidados con motor honesto (Round 0)
| Pares | Veredicto |
|-------|-----------|
| 22 originales V15 (ADA, SOL, DOGE...) | 17 REJECT, 3 marginales (ETC, BCH, UNI), 0 KEEP |

### Agentes BTC (Rounds 1-2)
| Agente | Enfoque | Bootstrap p | Veredicto |
|--------|---------|------------:|-----------|
| **A** | Trend Donchian-55 + EMA daily + ATR×2.5 trail | 0.088 (marginal) | (componente) |
| B | ML GBM classifier purged CV | 0.607 | REJECT |
| C | Regime adaptive (BULL+RANGE) | 0.156 | REJECT |
| D | 12h trend + vol-targeting + funding | 0.07 (marginal) | REJECT (funding dilutes) |
| E | Funding extremes mean-reversion | 0.036 | REJECT (degradado 2025) |
| **F** | Vol-compression breakout BTC+ETH | 0.355 (BTC alone) | (componente) |
| **V2 = A + F_BTC** | **Combinación** | **0.031 ✅** | **KEEP** |

### Agentes ETH (Round 3 + 4)
| Agente | Enfoque | Bootstrap p | Veredicto |
|--------|---------|------------:|-----------|
| ETH-A | A's params en ETH 4h | 0.103 | REJECT (marginal) |
| ETH-F | F's vol-breakout en ETH | 0.738 | REJECT (lastre) |
| ETH-V2 | A+F combinado en ETH | 0.484 | REJECT (sample desafortunado, synth +12%) |
| G | ML LightGBM + ETH/BTC ratio | 0.808 | REJECT (worst train→test gap del proyecto) |
| H | ETH/BTC ratio rotation | 0.624 | REJECT (control random p=33%) |
| I | Mean-reversion en RANGE | 0.760 | REJECT (3 trades/año) |
| **J** | **ETH 1D timeframe** | **0.383** | **REJECT (edge vs null NEGATIVO -6%)** |

### Ronda 4 — Recursos nuevos (on-chain)
| Agente | Enfoque | Resultado |
|--------|---------|-----------|
| **K** | **V2 + on-chain (Coin Metrics free)** | **REJECT** — mejora p de 0.030→0.025 PERO 35% de seeds shuffled logran lo mismo. Indistinguible de filtro aleatorio. |

### Tests sintéticos / validación de método
| Test | Conclusión |
|------|-----------|
| Naive learner (aprender de pérdidas) | Empeoró 14/20 universos paralelos |
| Filtros extras (vol_z, body, candle, engulfing) | Ninguno mejora V2; control random valida protocolo |
| Combinado A+F (BTC) variantes | V2 = A+F_BTC sin ETH óptimo |

---

## Insights centrales

### 1. El único edge real demostrable: BTC V2

```
V2 = A (Donchian breakout LONG, BULL filter) + F_BTC (vol-compression breakout bi-direccional)

In-sample 2020-2025:
  N = 163 trades, PF 1.59, WR 43%, annual +22.6%, DD 23.9%
  Bootstrap p = 0.031 ✅ (único con p<0.05 en todo el proyecto)

Sintético (20 series block bootstrap):
  Mediana annual +7.9%, 70% series positivas
  Real BTC en p~90 (afortunado) -> expectativa honesta ~8-15% annual

OOS 2026 (Ene-Feb, BTC -23%):
  Solo F_SHORT firmó (3 trades, +2%, DD 6%) — A correctamente staid flat en bear
```

### 2. Información valiosa que SE CONFIRMÓ por exclusión

- **ML clasificador en crypto 4h no funciona** (B en BTC + G en ETH, ambos train AUC >0.75, test AUC ~0.5)
- **ETH es estructuralmente más difícil que BTC** (7 enfoques distintos, ninguno significativo)
- **"Aprender de pérdidas" empeora el sistema** (test sintético 14/20)
- **On-chain free-tier no añade alpha a 4h** (delay de publicación dilye señal)
- **ETH/BTC ratio es señal lenta** — funciona en 1D-W, no en 4h
- **Padrones de velas (engulfing, etc.) son peores que random** — confirma Bulkowski formal

### 3. La realidad estructural del mercado

Crypto 4h es **cuasi-eficiente**. El edge real disponible con métodos sistemáticos
honestos en universo BTC/ETH es modesto (~10-15% anual no leveraged). El
mercado es lo suficientemente líquido para que los mecanismos obvios (trend,
vol-breakout, mean-rev, ML) ya estén priced in.

El edge que queda es:
- Pequeño (V2 expected ~10% annual)
- Difícil de transferir entre activos
- No se mejora con más features
- No se mejora con más complejidad de modelo
- Solo se preserva con disciplina anti-overfitting

---

## Lo que NO se ha probado (caminos abiertos, todos requieren inversión nueva)

1. **Glassnode/CryptoQuant pro** (~$39-99/mes): SOPR refinado, NUPL, MVRV-Z, HODL
   waves. Posiblemente añadan alpha real que el free-tier no captura.
2. **Timeframe mayor** (1W ETH): demasiado pocos puntos para WF significativo.
3. **Estrategias no direccionales**:
   - Basis trading (spot vs perp arb) — yield 5-15% baja varianza
   - Funding rate carry — yield modesto si funding consistente
   - Lending APR (Aave, Compound) — 2-5% APY ETH/USDC
4. **DeFi yield + staking**:
   - ETH staking ~3-4% APR
   - Restaking (EigenLayer) ~3-7% extra
   - Combinado spot ETH + staking ~5-10% APR sin riesgo direccional
5. **Multi-strategy ensemble**: combinar V2 BTC con basis BTC y staking ETH
   en un solo bot. Cada componente con edge demostrable; el portfolio diversifica.

---

## Recomendación final del proyecto

**Acción inmediata**: deploy V2 BTC en paper trade testnet, acumular ≥30 trades reales en 6-12 meses.

**Después de paper trade exitoso**:
- Capital pequeño (10-20% del planeado), escalado gradual
- O combinar con estrategias no direccionales (basis, staking) para reducir varianza

**No hacer**:
- ❌ Más backtests buscando "el santo grial" — agotado
- ❌ Re-tunear V2 — test sintético ya demostró empeora
- ❌ Pagar Glassnode pro sin tener V2 corriendo primero
- ❌ Forzar ETH al portfolio — 7 enfoques distintos demostraron que no es operable a 4h

**Honesty check final**:
Premium realista sobre CETES (10% TIIE 2026):
- V2 BTC unleveraged spot: ~5-8% premium → marginalmente justifica riesgo
- V2 BTC perp con funding -13% anual: ~-2% premium → NO justifica
- V2 BTC perp 1x lev neta: ~0-3% premium → marginal
- Pequeña asignación BTC + DeFi yield: posiblemente la mejor combinación honesta

Si esos números no compensan el riesgo psicológico/operacional del cripto,
**la conclusión válida es: este mercado no es para tu perfil**. El proyecto
cumplió su misión al darte claridad sobre qué es real y qué es ruido.

---

## Archivos clave

- `experiments/agent_A/`, `agent_F/` — componentes de V2
- `experiments/combined_AF/README.md` — V2 validación
- `experiments/synthetic/` — tests de validación de método
- `experiments/VERDICTO_RONDA2.md`, `VERDICTO_ETH.md` — verdictos parciales
- Este documento: el veredicto final consolidado

**Total de commits desde la auditoría inicial: 13+**
**Total de pares/configs evaluados honestamente: ~40+**
**Únicos con bootstrap p<0.05: 1 (V2)**
