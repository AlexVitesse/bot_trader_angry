# Arquitectura V14 - Documentacion Completa

## Resumen de Arquitecturas

### BTC V14 (y ETH similar)
```
[Datos] → [Detectar Regimen] → [Detectar Setup] → [Ensemble ML Confianza] → [Position Sizing]
```

**Paso 1: Detectar Regimen**
- TREND_UP: ADX > 25, DI+ > DI-, CHOP < 50
- TREND_DOWN: ADX > 25, DI- > DI+, CHOP < 50
- RANGE: CHOP > 55 o ADX < 20
- VOLATILE: ATR% > 4, BB_width > 8

**Paso 2: Detectar Setup (reglas tecnicas)**
```python
# Ejemplo: Pullback en tendencia alcista
if regime == TREND_UP:
    if rsi14 < 40 and bb_pct < 0.3 and ema200_dist > 0:
        setup = "PULLBACK_IN_UPTREND"
        strategy = TREND_FOLLOW_LONG
```

**Paso 3: Ensemble ML (3 modelos votan)**
- Modelo Contexto: ADX, DI_diff, CHOP, ATR%, BB_width
- Modelo Momentum: RSI14, RSI7, Stoch_K, ret_5, ret_20
- Modelo Volume: vol_ratio, vol_trend, obv_slope

```python
confidence = mean([prob_context, prob_momentum, prob_volume])
# Si confidence < 0.35 → skip
# Si confidence 0.35-0.45 → 25% del riesgo base
# Si confidence 0.55-0.65 → 100% del riesgo base
# etc.
```

**Paso 4: Position Sizing**
- Base risk: 2%
- Ajustado por confianza (0.25x a 2x)
- Anti-Martingale: reducir despues de 3 perdidas, aumentar despues de 3 ganancias

---

### DOGE V14 (arquitectura diferente)
```
[Datos] → [Features] → [3 Modelos ML Votan] → [Tradear si 2/3 coinciden]
```

**No usa setups tecnicos** - los memecoins no siguen patrones tradicionales

**3 Modelos:**
- RandomForest
- GradientBoosting
- LogisticRegression

**Regla:** Solo tradear si al menos 2 de 3 modelos predicen prob > 0.5

---

## El Problema del 80/20

### Preocupacion del Usuario
Si dividimos los datos cronologicamente:
- Train: 2018-2024 (80%)
- Test: 2024-2026 (20%)

El modelo aprende patrones de mercados pasados que pueden no existir en 2026+.

### Solucion: Walk-Forward Validation

En vez de un solo 80/20, hacemos MULTIPLES divisiones:

```
Fold 1:  Train [2018----2020] Test [2020-2021]
Fold 2:  Train [2019----2021] Test [2021-2022]
Fold 3:  Train [2020----2022] Test [2022-2023]
...
Fold 12: Train [2022----2024] Test [2024-2025]
```

**Ventajas:**
1. El modelo se prueba en DIFERENTES epocas del mercado
2. Si funciona en 10/12 folds, es robusto a cambios de mercado
3. Los folds recientes (entrenados con data reciente) tambien deben funcionar

### Verificacion Pendiente
Debemos verificar que los **ultimos folds** (entrenados con data 2022-2024) funcionan bien, no solo el promedio.

---

## Validacion Contra Overfitting

| Experto | Walk-Forward | Cross-Asset | Sinteticos | Status |
|---------|--------------|-------------|------------|--------|
| BTC V14 | 12/12 folds+ | ETH +2829% | 5/5 | APROBADO |
| ETH V14 | 32/36 folds+ | BTC +730% | 4/5 | APROBADO |
| DOGE V14 | 7/9 folds+ | SHIB +194%, PEPE +192% | 4/5 | APROBADO |
| ADA V14 | 11/12 folds+ | DOT +220%, SOL +322%, ATOM +326% | N/A | APROBADO |
| DOT V14 | 6/8 folds+ | (menos historia desde 2020) | N/A | APROBADO |

---

## Diferencias Clave

| Aspecto | BTC/ETH | DOGE/ADA/DOT |
|---------|---------|--------------|
| Deteccion de senal | Reglas tecnicas | ML Ensemble Voting |
| Rol del ML | Calcular confianza | Decidir si tradear |
| Setups | 8 (BTC), 3 (ETH) | N/A (ML puro) |
| Modelos | Context+Momentum+Volume | RF+GB+LR (3 votan) |
| Cross-validation | Assets correlacionados | Similar coins |
| TP/SL | 6%/3% (BTC), 4%/2% (ETH) | 6%/4% (DOGE/ADA), 5%/3% (DOT) |

### Resumen de Expertos V14

| Expert | Symbol | Arquitectura | PnL Backtest | Walk-Forward |
|--------|--------|--------------|--------------|--------------|
| BTC V14 | BTC/USDT | Setups + ML | +1126% | 6/6 folds+ |
| ETH V14 | ETH/USDT | Setups + ML | +1230% | 32/36 folds+ |
| DOGE V14 | DOGE/USDT | ML Ensemble | +414% | 7/9 folds+ |
| ADA V14 | ADA/USDT | ML Ensemble | +458% | 11/12 folds+ |
| DOT V14 | DOT/USDT | ML Ensemble | +77% | 6/8 folds+ |

---

## Archivos Relevantes

### Expertos Aprobados
- `strategies/btc_v14/` - Experto BTC (Setups + ML)
- `strategies/eth_v14/` - Experto ETH (Setups + ML)
- `strategies/doge_v14/` - Experto DOGE (ML Ensemble)
- `strategies/ada_v14/` - Experto ADA (ML Ensemble)
- `strategies/dot_v14/` - Experto DOT (ML Ensemble)

### Gestion de Riesgo
- `strategies/correlation_risk.py` - Manejo de correlacion BTC/ETH
- `strategies/__init__.py` - Registro central de expertos

### Scripts de Entrenamiento
- `discover_setups.py` - Descubrir setups tecnicos para un asset
- `train_ensemble_voting.py` - Entrenar modelo ML ensemble
- `v14_cross_validation.py` - Validacion cruzada

---

## Gestion de Riesgo por Correlacion

### Problema Identificado
- Correlacion BTC-ETH: **83%**
- Cuando BTC cae >2%, ETH cae **98%** de las veces
- Durante crashes: mas senales LONG que SHORT
- Max Drawdown sin proteccion: **-99.7%**

### Soluciones Implementadas (`strategies/correlation_risk.py`)

**1. Limites de Exposicion**
```python
MAX_PORTFOLIO_HEAT = 0.06      # 6% max total
MAX_CORRELATED_HEAT = 0.04     # 4% max para activos correlacionados
```

**2. Reduccion por Correlacion**
```python
# Si ya tenemos BTC y queremos abrir ETH:
if correlation > 0.7:
    size_multiplier = 0.5  # 50% del tamano normal
```

**3. Crash Mode**
```python
# Si BTC cae >2% en 3 velas:
if is_crash_mode(btc_returns):
    allow_long = False   # Solo SHORT permitido
```

**4. Deteccion de Regimen de Mercado**
```python
# NORMAL: trading normal
# CAUTION: 75% size, max 2 posiciones
# CRASH: 50% size, solo SHORT, max 1 posicion
```

### Resultado Esperado
| Metrica | Sin Proteccion | Con Proteccion |
|---------|----------------|----------------|
| Max Drawdown | -99.7% | ~-25% |
| PnL Total | +50% | +40% |
| Ratio Sharpe | Bajo | Alto |

---

## Proximos Pasos

1. ~~Entrenar ADA y DOT~~ ✓ COMPLETADO
2. Paper trading para validacion final (Binance testnet)
3. Considerar reentrenamiento periodico (cada mes)
4. Monitorear correlaciones en tiempo real
5. Implementar alertas de drawdown excesivo
