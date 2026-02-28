# V13.04 - Low-Overfitting Ridge Models

## Resumen

V13.04 es una version de bajo overfitting del bot de trading que usa:
- **Modelo:** Ridge regression con alpha=100 (alta regularizacion)
- **Features:** Solo 7 features minimas
- **Direccion:** LONG_ONLY (shorts deshabilitados)
- **Validacion:** Walk-forward 5 ventanas + test en bear market

## Por que V13.04?

V13.03 mostraba excelentes resultados en backtest (67% WR, PF 3.98) pero tenia riesgo de overfitting. V13.04 fue desarrollado para:

1. **Reducir overfitting:** Ridge con alpha=100 penaliza coeficientes grandes
2. **Simplificar features:** Solo 7 features vs 40+ en V13.03
3. **Validacion rigurosa:** Walk-forward con 5 ventanas temporales
4. **Test real:** Validado en bear market (Ene-Feb 2026, -20% mercado)

## Resultados de Validacion

### Walk-Forward (5 ventanas)

| Par  | Score | WR Promedio | WR Std | PnL Total | Folds Rentables |
|------|-------|-------------|--------|-----------|-----------------|
| BTC  | 100/100 | 55.7% | 2.1% | $734 | 5/5 |
| DOGE | 90/100 | 61.2% | 7.4% | $1,035 | 5/5 |
| ADA  | 90/100 | 54.3% | 3.5% | $581 | 4/5 |
| XRP  | 85/100 | 53.0% | 2.9% | $358 | 3/5 |
| DOT  | 70/100 | 52.0% | 8.3% | $125 | 3/5 |

### Test en Bear Market (Ene-Feb 2026)

Contexto: Mercado cayo -20% promedio en todos los pares.

| Par  | Precio Ene-Feb | Trades | WR | PnL |
|------|----------------|--------|-----|-----|
| DOGE | -15.8% | 21 | **81.0%** | +$128 |
| ADA  | -11.8% | 23 | **69.6%** | +$88 |
| DOT  | -11.0% | 13 | **76.9%** | +$69 |
| LINK | -24.8% | 29 | 58.6% | +$48 |
| ETH  | -31.1% | 30 | 56.7% | +$38 |
| XRP  | -21.8% | 37 | 54.1% | +$27 |
| BTC  | -22.3% | 18 | 50.0% | -$1 |
| **TOTAL** | | **216** | **59.3%** | **+$383** |

## Features Utilizadas (7 total)

```python
FEATURE_COLS = [
    'ret_1',    # Retorno 1 vela (4h)
    'ret_5',    # Retorno 5 velas (20h)
    'ret_20',   # Retorno 20 velas (80h)
    'vol20',    # Volatilidad 20 periodos
    'rsi14',    # RSI 14 periodos
    'ema21_d',  # Distancia a EMA21 (%)
    'vr',       # Volume Ratio vs promedio
]
```

## Pares Incluidos

### Tier 1 (Alta Confianza)
- **DOGE:** 90/100 WF, 81% WR en bear, +$128
- **ADA:** 90/100 WF, 70% WR en bear, +$88
- **DOT:** 77% WR en bear, +$69

### Tier 2 (Confianza Media)
- **XRP:** 85/100 WF, 54% WR en bear, +$27
- **BTC:** 100/100 WF, 50% WR en bear, -$1 (peor en bear, mejor overall)

## Pares Excluidos

| Par | Razon | Potencial Futuro |
|-----|-------|------------------|
| ETH | PnL negativo en walk-forward | LONG_ONLY |
| BNB | Solo 1/5 folds rentables | LONG_ONLY |
| LINK | PnL negativo pese a alta consistencia | LONG_ONLY |
| NEAR | Correlacion inconsistente (65/100) | LONG_ONLY |
| AVAX | Datos insuficientes para validacion completa | LONG_ONLY |

## Configuracion

### Archivos de Modelo
```
models/
  v1304_DOGE.pkl          # Modelo Ridge para DOGE
  v1304_DOGE_scaler.pkl   # Scaler para DOGE
  v1304_ADA.pkl           # Modelo Ridge para ADA
  v1304_ADA_scaler.pkl    # Scaler para ADA
  v1304_DOT.pkl           # Modelo Ridge para DOT
  v1304_DOT_scaler.pkl    # Scaler para DOT
  v1304_XRP.pkl           # Modelo Ridge para XRP
  v1304_XRP_scaler.pkl    # Scaler para XRP
  v1304_BTC.pkl           # Modelo Ridge para BTC
  v1304_BTC_scaler.pkl    # Scaler para BTC
  v1304_meta.json         # Metadata con configuracion
```

### Parametros por Defecto
```python
{
    'tp_pct': 0.02,       # 2% Take Profit
    'sl_pct': 0.02,       # 2% Stop Loss
    'conv_min': 1.0,      # Conviction minima
    'direction': 'LONG_ONLY',
    'leverage': 5,        # 5x leverage
}
```

## Activacion

En `config/settings.py`:

```python
# Cambiar de False a True
ML_V1304_ENABLED = True
```

## Re-entrenamiento

Para re-entrenar modelos (mensualmente recomendado):

```bash
python ml_export_v1304.py
```

Para agregar pares excluidos al entrenamiento, editar `ml_export_v1304.py`:

```python
# Mover de EXCLUDED_PAIRS a V1304_PAIRS
V1304_PAIRS = {
    'DOGE': {...},
    'ETH': {'direction': 'LONG_ONLY', 'tp_pct': 0.02, ...},  # Agregar
    ...
}
```

## Por que LONG_ONLY?

Los tests muestran que SHORT tiene 0% WR con este modelo:

| Direccion | WR | PnL |
|-----------|-----|-----|
| LONG_ONLY | 58.3% | +$321 |
| SHORT_ONLY | 0.0% | -$282 |
| BOTH | 49.8% | -$1 |

El modelo Ridge entrenado en datos historicos (mayormente alcistas) no predice direcciones bajistas con precision.

## Futuro: Modelos SHORT

Para habilitar shorts en el futuro:
1. Entrenar modelo separado con features de momentum negativo
2. Usar solo datos de periodos bear confirmados
3. Validar con walk-forward especifico para bear markets

## Comparacion V13.03 vs V13.04

| Metrica | V13.03 | V13.04 |
|---------|--------|--------|
| Modelo | LightGBM | Ridge |
| Features | 40+ | 7 |
| Direccion | BOTH | LONG_ONLY |
| Overfitting Risk | Alto | Bajo |
| WR Backtest | 67% | 55-60% |
| WR Bear Market | No testeado | 59% |
| Validacion | Simple split | Walk-forward 5 ventanas |

## Conclusiones

V13.04 sacrifica algo de WR en backtest (67% -> 55-60%) a cambio de:
- Menor riesgo de overfitting
- Validacion mas robusta
- Funciona en bear markets
- Modelo mas simple y interpretable
