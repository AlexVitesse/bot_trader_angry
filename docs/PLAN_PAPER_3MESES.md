# Plan Paper Trading 3 Meses — 5 Monedas V2

> Plan operativo y de tracking para paper trading testnet de V2 sobre 5 monedas.
> Fecha de inicio: 2026-05-19 · Fin previsto: 2026-08-19 (90 días)

---

## Universo de monedas — Tier 1 + Tier 2

| Tier | Coin | Tamaño | Origen del veredicto |
|------|------|-------:|---------------------|
| **1** | **BTC/USDT** | 1.0x | KEEP 3/3 in-sample + PASS OOS sólido |
| **1** | **BNB/USDT** | 0.7x | MARGINAL 2/3 + PASS OOS consistente (synth 9/10) |
| **2** | DOGE/USDT | 0.5x | p=0.001 in-sample (mejor del proyecto), 1 trade OOS indeterminado |
| **2** | ETH/USDT | 0.5x | 2 trades OOS ambos +9.88%, n<3 indeterminado |
| **2** | OP/USDT | 0.4x | 3 wins OOS perfectos, in-sample limitado 1.8 años |

**Otros pares**: TODOS los demás (SOL, ADA, XRP, LINK, AVAX, DOT, NEAR, etc.)
**fuera** del bot. Esto representa una reducción de `ML_V15_PAIRS` de 22 → 5.

---

## Estrategia: V2 = A + F (motor honesto)

Aplicado uniformemente a las 5 monedas:

### Componente A (trend follower LONG-only)
- Filtro daily: `EMA50_1d > EMA200_1d` (con shift(1))
- Donchian breakout 4h: `close > rolling_55_high.shift(1)`
- vol_ratio >= 1.2
- ADX >= 18
- ATR×2.5 trailing (sin look-ahead intrabar)
- Funding veto: z-score > 2.5 bloquea
- max_bars = 60

### Componente F (vol-compression breakout, LONG y SHORT)
- BB width en cuantil bajo histórico (compresión sostenida 3+ velas)
- Breakout direccional: close > hi_n (LONG) o < lo_n (SHORT)
- vol_ratio >= 1.2 confirmación
- Trail ATR (con SL previo, peak update después)
- Funding veto bidireccional

### Engine
- **Una posición a la vez por par**
- A primero (más conservador); si A no fira, F segundo
- Tras cierre del trade, avanzar índice +bars+1 (no solape)

---

## KPIs a trackear semanalmente

Cada semana, registrar para CADA par:

### Métricas operacionales
| KPI | Cómo medir | Umbral alarma |
|-----|------------|---------------|
| Trades reales acumulados | conteo simple | esperado 3-8/mes según par |
| Win rate | wins / total | < 30% activar revisión |
| PF rolling | gross_win / gross_loss | < 0.8 activar parada |
| DD desde inicio | (peak - current) / peak | > 30% **STOP TRADING** |
| Mensual return | trades del mes | meta: -3% a +5% por mes en testnet |

### Métricas de validación
| KPI | Cómo medir | Esperado |
|-----|------------|----------|
| Real vs simulado | comparar trades reales vs backtest del mismo período | divergencia < 25% en 50 trades |
| Sharpe rolling 30 trades | mean(returns) / std(returns) × sqrt(N) | > 0.5 |
| Slippage real | precio ejecutado vs señal | < 0.1% promedio |
| Latencia ejecución | timestamp señal vs orden | < 5s |

### Métricas de comparación entre pares
- **Correlación de retornos** entre los 5 pares (queremos < 0.5 idealmente)
- **Distribución de trades**: ¿1 par domina? ¿hay equilibrio?
- **Contribución al PnL**: % de cada par al PnL total

---

## Reglas de parada (kill switches)

Estas son AUTOMÁTICAS — si se disparan, el bot pausa ese par:

### Por par individual
1. **DD por par > 30%** → pausar par, revisar
2. **5 SLs consecutivos** en mismo par → pausar, revisar régimen
3. **Real diverge > 30%** del simulado en 30 trades → bug investigation
4. **Outage/error** > 2h → pausar, alertar

### Global del bot
5. **DD global del paper portfolio > 25%** → pausar TODO
6. **Pérdida diaria > 10%** → pausar el día
7. **3 días seguidos negativos** → revisar régimen del mercado

---

## Hitos del plan 3 meses

### Mes 1 (días 1-30): Calibración
**Objetivo**: confirmar que el bot ejecuta V2 correctamente.

Métricas mínimas a fin de mes 1:
- ≥ 10 trades reales totales (entre los 5 pares)
- Real vs simulado: divergencia < 30%
- Slippage promedio < 0.2%
- Zero bugs críticos detectados

**Decisión fin de mes 1**:
- Si todos los KPIs OK → continuar mes 2
- Si divergencia > 30% → STOP, debug, ajustar

### Mes 2 (días 31-60): Acumulación
**Objetivo**: acumular muestra estadística por par.

Métricas mínimas a fin de mes 2:
- ≥ 25 trades reales totales
- ≥ 5 trades por par activo (los que dispararon)
- Sharpe rolling > 0
- DD global < 20%

**Decisión fin de mes 2**:
- Si trends consistentes → continuar mes 3
- Si DD > 20% → reducir sizing 50%, continuar más cautelosos

### Mes 3 (días 61-90): Validación final
**Objetivo**: tener datos suficientes para decidir Tier 1 vs Tier 2 vs OUT.

Métricas finales:
- ≥ 40 trades reales totales
- ≥ 10 trades por par activo
- Bootstrap p calculado sobre los trades reales por par
- Comparación final real vs sintético

---

## Criterios de promoción/democión tras 3 meses

Al día 90, cada par se evalúa con estos criterios:

### Para PROMOVER de Tier 2 a Tier 1 (capital real candidato)
**Todos deben pasar**:
- ≥ 10 trades reales acumulados
- PF real ≥ 1.2
- Retorno total ≥ 0
- Bootstrap p sobre trades reales < 0.10 (menos estricto que in-sample por muestra pequeña)
- DD ≤ 25%

### Para MANTENER en Tier 2 (continuar paper)
- ≥ 5 trades pero < 10 (más datos necesarios)
- PF entre 0.8 y 1.2 (zona ambigua)
- Sin DD catastrófico

### Para DEMOTAR a Tier 3 (OUT del bot)
**Cualquiera de**:
- DD > 30% en cualquier momento
- 10+ trades con PF < 0.8
- Divergencia > 30% real vs simulado consistente

---

## Estructura de datos para tracking

Cada trade se registra con:

```
{
  "ts_signal": "2026-05-19T08:00:00Z",
  "ts_entry": "2026-05-19T08:00:15Z",
  "pair": "BTC/USDT",
  "strat": "A|F",
  "side": "LONG|SHORT",
  "entry_price": 67234.5,
  "exit_price": ...,
  "exit_reason": "TP|SL|TIMEOUT",
  "pnl_pct": ...,
  "bars": 12,
  "regime_at_entry": "BULL",
  "vol_ratio": 1.43,
  "atr_pct": 2.1
}
```

Guardado en `paper_trade/v2_3months/trades.jsonl` con append cada trade.

---

## Comparación real vs simulado

**Cada par tiene su simulador de respaldo**. Cada noche el sistema:
1. Toma los trades reales del día
2. Re-corre `experiments/v2_all_coins/test_v2_all.py` sobre las mismas velas
3. Compara: trades reales coincidían con señal simulada?
4. Genera `paper_trade/v2_3months/divergence_report.csv`

**Umbral alarma**: si en 20 trades consecutivos divergen > 30% → STOP y diagnose
(slippage, ejecución, datos en tiempo real, etc).

---

## Métricas de éxito al día 90

El plan se considera EXITOSO si:

1. **≥1 par promovido a Tier 1** (BTC esperado seguro; BNB también)
2. **Total PnL paper > 0** (premium sobre buy-and-hold testnet)
3. **DD global < 25%**
4. **≥ 40 trades reales** acumulados (sample base sólida)
5. **Sin divergencias críticas** real vs simulado

El plan se considera FALLIDO si:
1. DD > 30% en cualquier punto
2. < 20 trades en 3 meses (señales no firando)
3. Divergencia real vs simulado > 30% consistente
4. BTC pierde dinero en paper (era el más sólido)

---

## Riesgos identificados

| Riesgo | Mitigación |
|--------|-----------|
| Bear market sostenido bloquea A → poca actividad | F SHORT debería compensar parcialmente |
| Bull market sin pullbacks → A fira pero entra tarde | trailing stops capturan parte |
| Sample 3 meses sigue siendo pequeño | criterios menos estrictos (p<0.10 vs 0.05) |
| Mercado choppy → muchos SLs en A | DD limit pausa par |
| API testnet downtime | logging continuo + retry logic |
| Cambio régimen brusco | regime detection del par lo maneja |

---

## Acciones inmediatas para arrancar

1. ✅ **Actualizar `config/settings.py`**:
   - `ML_V15_PAIRS = ['BTC/USDT', 'BNB/USDT', 'DOGE/USDT', 'ETH/USDT', 'OP/USDT']`
   - `ML_V15_SIZING` con valores Tier 1/2

2. ✅ **Crear meta files V2** para cada par:
   - `strategies/{coin}_v15/models/meta_v15.json` con `model_type = "v2_honest"`

3. ✅ **Módulo V2 engine** (`src/v2_engine.py`):
   - Implementación honest A + F
   - Función `generate_v2_signal(pair, df_4h)` que el bot puede llamar

4. ⏳ **Integración mínima en `ml_strategy_v15.py`**:
   - Para pares con `model_type == "v2_honest"`, llamar a v2_engine
   - Resto del código v15 puede coexistir (legacy)

5. ⏳ **Setup tracking**:
   - Carpeta `paper_trade/v2_3months/`
   - Logging hooks en `portfolio_manager.py`
   - Script de reporte semanal

---

## Después del día 90

Tres escenarios posibles:

### Escenario A — Éxito esperado (BTC + BNB Tier 1 confirmados)
- BTC y BNB acumularon evidencia OOS positiva
- DOGE/ETH/OP necesitan más tiempo o pasan también
- **Acción**: capital real pequeño (10-20% del planeado) en BTC + BNB. Continuar paper en Tier 2.

### Escenario B — Mixto (BTC Tier 1, otros indeterminados)
- Solo BTC pasa criterios
- BNB y otros marginales
- **Acción**: capital real solo en BTC. Continuar paper 3 meses más en otros.

### Escenario C — Fracaso (todos fallaron en paper real)
- DDs altos, divergencias, KPIs malos
- **Acción**: diagnóstico profundo. Posiblemente el motor honesto NO transfiere
  bien a live (slippage real, microestructura). Pausar bot, replantear.

---

## Archivos relacionados

- `src/v2_engine.py` (nuevo) — motor V2 honest
- `config/settings.py` — actualizado con 5 pares
- `strategies/{coin}_v15/models/meta_v15.json` — 5 meta files V2
- `experiments/v2_all_coins/` — backtest reference (origen de la decisión)
- `experiments/combined_AF/` — V2 BTC validation
- `paper_trade/v2_3months/` (a crear durante el run) — tracking data
