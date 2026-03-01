# V14.1 Branch Documentation

## Branch: `feature/v14.1-more-pairs`

**Fecha:** 2026-03-01
**Objetivo:** Expandir V14 de 5 a 16 pares validados para paper trading

---

## Resumen de Cambios

### Pares Finales (16 aprobados)

| Modelo Base | Pares | TP/SL |
|-------------|-------|-------|
| BTC (propio) | BTC | 4%/1.5% |
| ETH (setups) | ETH | 4%/2% |
| ADA ensemble | ADA, SOL, ATOM, AVAX, POL | 6%/4% |
| DOGE ensemble | DOGE, 1000SHIB, 1000PEPE, 1000FLOKI | 6%/4% |
| DOT ensemble | DOT, LINK, ALGO, FIL, NEAR | 5%/3% |

### Pares Rechazados (no pasaron validacion)

| Par | Modelo | WF Folds | Razon |
|-----|--------|----------|-------|
| INJ | DOT | 5/10 | < 60% folds positivos |
| BONK | DOGE | 5/10 | < 60% folds positivos |
| WIF | DOGE | 5/10 | < 60% folds positivos |

---

## Validacion de Nuevos Pares

### Metodologia
1. **Walk-Forward Validation** (10 folds)
   - Criterio: >= 60% folds con PnL positivo
2. **Datos Sinteticos** (5 escenarios)
   - BULL, BEAR, RANGE, VOLATILE, MIXED
   - Criterio: >= 3/5 escenarios positivos

### Resultados por Modelo

#### Modelo ADA (Smart Contracts)
```
SOL/USDT:   9/10 folds, +250% PnL, 5/5 synth [APROBADO]
ATOM/USDT:  8/10 folds, +304% PnL, 4/5 synth [APROBADO]
AVAX/USDT:  8/10 folds, +150% PnL, 4/5 synth [APROBADO]
MATIC/USDT: 8/10 folds, +310% PnL, 4/5 synth [APROBADO]
```

#### Modelo DOGE (Memecoins)
```
SHIB/USDT:  6/10 folds, +126% PnL, 5/5 synth [APROBADO]
PEPE/USDT:  6/10 folds, +184% PnL, 4/5 synth [APROBADO]
FLOKI/USDT: 6/10 folds, +144% PnL, 4/5 synth [APROBADO]
BONK/USDT:  5/10 folds, +38% PnL,  4/5 synth [RECHAZADO]
WIF/USDT:   5/10 folds, +58% PnL,  4/5 synth [RECHAZADO]
```

#### Modelo DOT (Infraestructura)
```
LINK/USDT: 7/10 folds, +116% PnL, 5/5 synth [APROBADO]
ALGO/USDT: 7/10 folds, +68% PnL,  5/5 synth [APROBADO]
FIL/USDT:  6/10 folds, +74% PnL,  5/5 synth [APROBADO]
NEAR/USDT: 6/10 folds, +98% PnL,  5/5 synth [APROBADO]
INJ/USDT:  5/10 folds, +86% PnL,  4/5 synth [RECHAZADO]
```

---

## Arquitectura V14

### Tipos de Estrategia

#### 1. BTC V14 (Regimen + Setups + Ensemble)
- **Regimen Detection:** TREND_UP, TREND_DOWN, RANGE, VOLATILE
- **Setups por regimen:**
  - TREND_UP: PULLBACK_IN_UPTREND, OVERSOLD_IN_UPTREND
  - TREND_DOWN: RALLY_IN_DOWNTREND, OVERBOUGHT_IN_DOWNTREND
  - RANGE: SUPPORT_BOUNCE, RESISTANCE_REJECTION
  - VOLATILE: BREAKOUT_UP, BREAKOUT_DOWN
- **Ensemble ML:** context + momentum + volume models
- **Modelos:** `strategies/btc_v14/models/`

#### 2. ETH (Setups Simples)
- RSI oversold + volume spike
- Sin ML, solo reglas tecnicas

#### 3. Ensemble Voting (ADA, DOGE, DOT y derivados)
- **Modelos:** RandomForest + GradientBoosting + LogisticRegression
- **Voting:** >= 2/3 modelos deben predecir positivo
- **Threshold:** probabilidad promedio > 50%
- **Features:** rsi, macd_norm, adx, bb_pct, atr_pct, ret_3, ret_5, ret_10, vol_ratio, trend

---

## Archivos Creados/Modificados

### Scripts
```
v14_paper_trade.py      # Paper trading con 16 pares
validate_new_pairs.py   # Validacion WF + sinteticos
train_btc_v14.py        # Entrenamiento modelos BTC
train_ensemble_voting.py # Entrenamiento ensembles
```

### Estrategias
```
strategies/btc_v14/     # BTC regime + setups + ensemble
strategies/ada_v14/     # ADA ensemble (usado por SOL, ATOM, AVAX, POL)
strategies/doge_v14/    # DOGE ensemble (usado por SHIB, PEPE, FLOKI)
strategies/dot_v14/     # DOT ensemble (usado por LINK, ALGO, FIL, NEAR)
strategies/correlation_risk.py  # Gestion correlacion BTC-ETH
```

### Modelos Guardados
```
strategies/*/models/
  - scaler.pkl
  - random_forest.pkl
  - gradient_boosting.pkl
  - logistic_regression.pkl (algunos)
  - metadata.pkl
```

---

## Simbolos Binance

Algunos pares usan nombres diferentes en Binance:

| Interno | Binance Symbol |
|---------|----------------|
| MATIC | POL/USDT (rebranded) |
| SHIB | 1000SHIB/USDT |
| PEPE | 1000PEPE/USDT |
| FLOKI | 1000FLOKI/USDT |

---

## Estimaciones

### Frecuencia de Trades
- **Por vela 4h:** 0-1 trades promedio
- **Por dia (6 velas):** 1-2 trades promedio
- **Variabilidad:** 0 en dias tranquilos, 3-5 en dias volatiles

### Selectividad del Modelo
- Threshold: 50% probabilidad
- Solo entra cuando ensemble vota positivo
- Evita trades en condiciones desfavorables

---

## Proximos Pasos

1. [x] Paper trading 12 horas
2. [ ] Analizar resultados paper trading
3. [ ] Si OK: Deploy V14 a produccion
4. [ ] Poner V13 en shadow mode para comparacion

---

## Commits en esta Rama

```
cd48785 feat(v14.1): add 4 new pairs (SOL, ATOM, SHIB, PEPE)
af08624 feat(v14.1): expand to 19 pairs for paper trading
7fa0e34 fix(v14.1): keep only 16 approved pairs
```

---

## Notas Tecnicas

### Por que "No setup" en BTC
El sistema BTC detecta primero el regimen (ej: TREND_DOWN) y luego busca setups especificos dentro de ese regimen. "No setup" significa que aunque el regimen esta identificado, no hay patron de entrada valido en este momento.

### Por que probabilidades bajas (20-40%)
Las probabilidades reflejan la confianza del ensemble. Valores bajos indican que el mercado no presenta las condiciones que el modelo asocia con trades ganadores. Esto es esperado y es lo que hace al modelo rentable - no fuerza trades.

### Correlacion BTC-ETH
Se implemento `correlation_risk.py` para manejar la alta correlacion (83%) entre BTC y ETH. Limita exposicion combinada a MAX_PORTFOLIO_HEAT=6%.
