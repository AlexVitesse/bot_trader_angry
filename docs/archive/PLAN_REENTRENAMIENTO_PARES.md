# Plan de Re-Entrenamiento: Pares Excluidos
## V13 CORE -> V14+ Expansion

---

## Estado Actual (Feb 2026)

### Pares EXCLUIDOS de V13 (por bajo rendimiento)
| Par | WR Produccion | PnL 14d | Problema |
|-----|---------------|---------|----------|
| SOL/USDT | 42% | -$31.29 | 24 trades, EL PEOR |
| BTC/USDT | 33% | -$12.21 | 9 trades, todas perdidas |
| BNB/USDT | ~45% | marginal | Volatil, poco volumen |

### Pares ACTIVOS en V13
| Par | Tier | WR Backtest | Status |
|-----|------|-------------|--------|
| XRP/USDT | 1 | 86% | Excelente |
| NEAR/USDT | 1 | 67-73% | Excelente |
| DOT/USDT | 2 | 67% | Bueno |
| ETH/USDT | 2 | bueno en BULL | Bueno |
| DOGE/USDT | 2 | 71% | Bueno |
| AVAX/USDT | 2 | 100% ult 14d | Bueno |
| LINK/USDT | 2 | 64% | Bueno |
| ADA/USDT | 3 | bajo volumen | Marginal |

---

## Objetivo de Re-Entrenamiento

Lograr que SOL, BTC y BNB alcancen:
- **WR >= 55%** en backtest reciente (30 dias)
- **PF >= 1.3** (Profit Factor)
- **Consistencia** en distintos regimenes (BULL, BEAR, RANGE)

---

## Estrategia de Re-Entrenamiento

### Fase 1: Diagnostico Profundo
1. **Analizar por que fallan**
   - Revisar trades perdedores: timing, direccion, volatilidad
   - Comparar features que funcionan en NEAR/XRP vs SOL/BTC
   - Identificar patrones de mercado donde fallan

2. **Hipotesis iniciales**
   - SOL: Muy volatil, ATR dinamico puede ayudar
   - BTC: Dominancia afecta otros pares, puede necesitar features macro
   - BNB: Correlacionado con noticias de Binance, dificil de modelar

### Fase 2: Features Especializados
1. **Para SOL/USDT**
   - ATR multi-timeframe (1h, 4h, 1d)
   - Ratio SOL/BTC para detectar rotacion
   - Funding rate de futuros
   - Volumen relativo a 7 dias (no 20)

2. **Para BTC/USDT**
   - BTC.D (dominancia) como feature principal
   - Correlacion con SPY/Gold (macro)
   - Open Interest de futuros
   - Features de opciones (si disponible)

3. **Para BNB/USDT**
   - Volumen de Binance (exchange flows)
   - Correlacion con BTC (evitar trades contra tendencia)
   - Features de temporada (quemas de BNB)

### Fase 3: Arquitectura de Modelos
1. **Ensemble diferente**
   - Probar XGBoost ademas de LightGBM
   - Agregar capa de meta-learner
   - Experimentar con redes neuronales simples

2. **Targets alternativos**
   - Clasificacion binaria (win/loss) vs regresion
   - Target con horizonte variable (3-7 velas)
   - Target ajustado por volatilidad

### Fase 4: Validacion Rigurosa
1. **Walk-forward de 6 meses**
   - Train: 4 meses, Test: 2 meses
   - Minimo 50 trades por fold
   - WR >= 55% en CADA fold

2. **Test en condiciones adversas**
   - Backtest en crash de mayo 2025
   - Backtest en rally de enero 2026
   - Backtest en rango lateral

---

## Criterios de Graduacion

Un par excluido puede volver a produccion cuando:

| Metrica | Minimo | Ideal |
|---------|--------|-------|
| WR backtest 30d | 55% | 60%+ |
| WR walk-forward | 52% | 55%+ |
| Profit Factor | 1.3 | 1.5+ |
| Max Drawdown | <15% | <10% |
| Trades/mes | 8+ | 15+ |
| Consistencia (folds) | 3/4 | 4/4 |

---

## Linea de Tiempo Sugerida

### Semana 1-2: Diagnostico
- Ejecutar analisis de trades perdedores
- Identificar features faltantes
- Documentar hallazgos

### Semana 3-4: Desarrollo SOL
- Implementar features especializados
- Entrenar modelos v2
- Walk-forward validation

### Semana 5-6: Desarrollo BTC
- Features macro + dominancia
- Entrenar modelos v2
- Walk-forward validation

### Semana 7-8: Desarrollo BNB
- Features de exchange
- Entrenar modelos v2
- Walk-forward validation

### Semana 9+: Integracion
- Agregar pares graduados a V14
- Shadow testing 1-2 semanas
- Produccion

---

## Notas Importantes

1. **No apurar** - Mejor esperar que forzar un par malo
2. **Un par a la vez** - Enfoque en calidad, no cantidad
3. **Documentar todo** - Cada experimento debe tener resultados
4. **V13 es estable** - Los 8 pares actuales funcionan bien

---

## Archivos Relacionados

- `config/settings.py` - ML_PAIRS actual
- `ml_export_models.py` - Re-entrenar modelos
- `backtest_last_14_days.py` - Validar cambios
- `analyze_market_conditions.py` - Ver por regimen

---

*Documento creado: 25 Feb 2026*
*Version: V13 CORE*
