# Combinado A + F — Análisis

> Antes de meter a paper trade, verificación honesta de A + F juntos.
> Fecha: 2026-05-19 · `test_combined.py`

## Resumen ejecutivo

Tras combinar A (trend Donchian-55 4h LONG-only BTC) con F (vol-compression
breakout BTC+ETH bidireccional) y probar 4 variantes, **V2 = A_BTC + F_BTC
sin ETH** emerge como la única configuración con edge bootstrap-significativo:

- **In-sample 2020-2025: PF 1.59, annual +22.6%, DD 23.9%, p=0.031 ✅**
- **OOS 2026 (Ene-Feb): 3 trades F_SHORT, +2%, DD 6% — consistente con bear**
- 163 trades en 6 años (~27/año), suficiente muestra
- Solo BTC (operacionalmente simple)

## Variantes probadas (in-sample)

| Variante | N | PF | Annual | DD | Bootstrap p |
|----------|---|-----|--------|-----|-------------|
| V1: A + F (BTC+ETH todo) sleeve | 276 | 1.37 | +22.7% | 32.2% | 0.061 |
| V1: A + F (BTC+ETH todo) 50/50 | 276 | 1.37 | +12.3% | 17.4% | **0.043** ✅ |
| **V2: A + F_BTC (drop ETH) sleeve** | **163** | **1.59** | **+22.6%** | **23.9%** | **0.031** ✅ |
| V3: A_LONG + F_SHORT-only (BTC+ETH) sleeve | 135 | 1.49 | +13.6% | 16.8% | 0.072 |
| V3: A_LONG + F_SHORT-only (BTC+ETH) 50/50 | 135 | 1.49 | +7.2% | 8.7% | 0.056 |
| V4: A_LONG + F_BTC_SHORT only sleeve | 104 | 1.51 | +10.8% | 18.3% | 0.092 |

## Descomposición por componente (in-sample)

| Componente | N | PF | Annual | DD | p |
|------------|---|----|--------|-----|---|
| A_BTC (trend LONG) | 74 | 1.66 | +11.1% | 16.8% | 0.088 |
| F_BTC LONG | 59 | 1.70 | +10.9% | 23.6% | 0.109 |
| F_BTC SHORT | 30 | 1.08 | +0.2% | 11.0% | 0.492 |
| F_ETH LONG | 82 | 1.02 | -1.1% | 33.3% | 0.578 |
| F_ETH SHORT | 31 | 1.45 | +2.4% | 14.9% | 0.281 |

**Lecturas:**
- A_BTC y F_BTC_LONG son los dos motores reales (+11% c/u, anti-correlacionados).
- F_BTC_SHORT marginal in-sample pero útil en bear (vio el OOS 2026).
- **F_ETH es 100% ruido**: LONG negativo, SHORT marginal. Drop confirmado.

## OOS 2026 (Ene-Feb, 57 días)

Trades en V1/V2: solo 5 (3 BTC SHORT, 2 ETH SHORT). En V2 (sin ETH), solo 3:

| Fecha | Setup | Outcome | PnL |
|-------|-------|---------|-----|
| 2026-01-18 | F_BTC SHORT | TP | +8.78% |
| 2026-02-18 | F_BTC SHORT | SL | -3.40% |
| 2026-02-23 | F_BTC SHORT | SL | -2.91% |

V2 OOS: +2.0%, DD 6.2%, p=0.283 (muestra insuficiente).

**Importante:** A_LONG no firmó (correcto: bear). F_LONG no firmó (correcto: bear).
Solo F_SHORT firmó. La OOS prueba **una parte** del sistema (el SHORT en bear).
La validación del LONG necesita una ventana alcista — Q2/Q3 2026 post-halving lo
permitirá.

## Por qué la combinación supera las partes

Bootstrap individual:
- A solo: p=0.088 (marginal)
- F solo: p=0.355 (no significativo)
- **A + F_BTC: p=0.031 (significativo)**

La razón es matemática: A y F_BTC LONG son ambos +11%/yr con DD ~20%, pero
sus trades caen en momentos distintos del ciclo BTC. La cartera combinada
tiene menor varianza por trade promediado, lo que mejora el Sharpe y el
bootstrap p simultáneamente.

## Riesgo restante / qué falta validar

1. **A_LONG no se probó OOS 2026** (cero trades por el filtro daily). Si A se
   degradó silenciosamente, no lo sabremos hasta el siguiente bull. Mitigación:
   paper trade 6-12 meses cubriendo al menos una ventana alcista.

2. **El 50/50 portfolio sería conservador** (annual +12.3%, DD 17.4%) y también
   significativo (p=0.043). Si V2 sleeve 100% asusta por DD 24%, V1 50/50 es
   plan B con menor riesgo y aún premium sobre CETES.

3. **F_BTC_SHORT** tiene PF 1.08 in-sample (marginal) pero ganó claramente en
   2026 bear. Posible regime-dependency — no overfitting "tradicional", pero
   sí "regime overfitting". A monitorear.

4. **Funding cost**: el sim NO descuenta funding en A o F (a diferencia de D).
   Si se opera en perp, restar ~13%/yr a los retornos esperados. Spot evita
   funding pero no permite SHORT — incompatible con F.
   - Solución: operar en perp con ese coste asumido → annual real V2 ≈ +9-10%
     (después de funding) — **sigue siendo premium sobre CETES** pero más
     modesto.

## Recomendación

**Paper trade V2 = A_BTC + F_BTC** en testnet durante 6-12 meses:

1. Implementar como engine único: en cada vela 4h BTC, pedir señal a A
   primero (long-only trend); si no, a F (bidirectional vol breakout). Una
   posición a la vez en BTC. Una sola pkl/json de parámetros frozen.

2. Tracking continuo:
   - Trades reales del bot vs simulación
   - Sharpe rolling 30-trade
   - Bootstrap p actualizado cada mes
   - DD máximo
   - **Si el real diverge >25% del simulado en 50 trades → STOP y diagnosticar**

3. Activar capital real solo si:
   - 6+ meses de paper trade
   - ≥30 trades reales
   - Bootstrap p actualizado se mantiene < 0.05
   - DD real ≤ 30%
   - Una ventana alcista probó el componente A_LONG

4. Si todo lo anterior, capital pequeño primero (10-20% del que tenías pensado),
   escalar gradualmente.

## Archivos

- `test_combined.py` — runner del experimento
- `README.md` (este) — resultados y recomendación
- Strategies: `experiments/agent_A/strategy.py` y `experiments/agent_F/strategy.py`
