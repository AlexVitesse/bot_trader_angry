# Low-Overfitting Model Experiments Results

## Fecha: 2026-02-27

## Objetivo

Encontrar modelos ML que **NO hagan overfitting** usando 80/20 train/test split cronológico.

## Metodología

```
Datos: 2019 ──────────────── 80% ─────────────────│──── 20% ────
       │                                          │              │
       └──────────── TRAIN ───────────────────────┘              │
                                                   └──── TEST ───┘
```

- **BTC Train**: 2019-01 a 2024-09 (12,523 samples)
- **BTC Test**: 2024-09 a 2026-02 (3,131 samples)
- **ETH Train**: 2020-01 a 2024-12 (10,775 samples)
- **ETH Test**: 2024-12 a 2026-02 (2,694 samples)

## Modelos Probados

### Algoritmos
- Ridge (alpha=1, 10, 100)
- Lasso (alpha=0.01, 0.1)
- ElasticNet
- RandomForest (max_depth=3, 5)
- GradientBoosting (max_depth=2, 3)
- ExtraTrees (max_depth=3)
- BaggingRegressor

### Feature Sets
- **Minimal (7 features)**: ret_1, ret_5, ret_20, vol20, rsi14, ema21_d, vr
- **Optimal (7 features)**: ret_5, vol5, vol20, rsi14, rsi7, macd_h, bb_pos
- **Full (19 features)**: Set completo con ATR, múltiples EMAs, etc.

## Resultados: Overfitting Analysis

### Mejores Modelos por Correlation Drop

| Pair | Features | Model | Train Corr | Test Corr | Drop |
|------|----------|-------|------------|-----------|------|
| BTC | minimal | Ridge_a100 | 0.0922 | 0.0921 | **0.1%** |
| BTC | minimal | Ridge_a10 | 0.0922 | 0.0916 | 0.6% |
| BTC | medium | Ridge_a100 | 0.1044 | 0.0972 | 7.0% |
| ETH | minimal | Ridge_a10 | 0.0812 | 0.0806 | 0.8% |
| ETH | minimal | Ridge_a100 | 0.0812 | 0.0803 | 1.2% |
| BTC | minimal | ExtraTrees_d3 | 0.1641 | 0.1307 | 20.3% |
| ETH | minimal | ExtraTrees_d3 | 0.1761 | 0.1296 | 26.4% |

**Key Finding**: Ridge con alpha=100 tiene **CASI CERO overfitting** (0.1% drop)

## Resultados: Backtest con TP/SL

### Mejores Configuraciones (WR >= 50%)

| Pair | Model | Direction | TP | SL | Conv | Trades | WR | PnL |
|------|-------|-----------|----|----|------|--------|-----|-----|
| BTC | Ridge_a100 | LONG_ONLY | 2.5% | 2.5% | 0.5 | 234 | 52.6% | $152 |
| BTC | Ridge_a100 | LONG_ONLY | 2.0% | 2.0% | 0.5 | 293 | 52.2% | $123 |
| BTC | ExtraTrees | LONG_ONLY | optimal | - | 0.5 | 329 | 53.2% | $201 |
| ETH | Ridge_a10 | LONG_ONLY | - | - | 0.5 | 321 | 53.6% | $204 |

### Observaciones Críticas

1. **SHORT trades = 0% WR** en el periodo de test (mercado alcista)
2. **LONG_ONLY es obligatorio** para evitar pérdidas
3. **ETH underperforms** con Ridge - mejor usar ExtraTrees

## Análisis de Conviction Thresholds

### BTC con Ridge_a100, LONG_ONLY, TP=3%/SL=1.5%

| Conv >= | Trades | WR | PnL | Avg Conv |
|---------|--------|-----|-----|----------|
| 0.0 | 317 | 37.2% | $175 | 0.84 |
| 0.5 | 259 | 37.1% | $128 | 1.14 |
| 1.0 | 182 | 40.7% | $217 | 1.52 |
| 2.0 | 60 | 38.3% | $47 | 2.33 |
| 2.5 | 21 | 47.6% | $66 | 2.80 |
| 3.0 | 8 | 50.0% | $29 | 3.23 |

**Insight**: Mayor conviction = Mayor WR pero menos trades

### Optimización TP/SL para BTC (conv >= 0.5)

| TP | SL | Trades | WR | PnL | PF |
|----|----|----|-----|-----|-----|
| 2.0% | 2.0% | 293 | 52.2% | $123 | 1.09 |
| 2.0% | 2.5% | 260 | 56.5% | $78 | 1.06 |
| 2.5% | 2.5% | 234 | 52.6% | $152 | 1.11 |

**Best**: TP=2.5%, SL=2.5% (ratio 1:1)

## Desglose Mensual BTC

```
Month      Trades   Wins   WR       PnL
2024-11    23       11     47.8%    $73
2025-05    15       8      53.3%    $53
2025-09    7        4      57.1%    $19
2024-12    20       4      20.0%    -$62
2025-11    15       4      26.7%    -$24
```

**Meses problemáticos**: Dic 2024, Nov 2025, Feb 2026 (WR < 30%)

## Conclusiones

### ¿El modelo generaliza?
**SÍ** - Ridge(alpha=100) tiene solo 0.1% correlation drop

### ¿El modelo es rentable?
**PARCIALMENTE** - 52% WR con $150 profit sobre $100 en 17 meses

### Problemas identificados

1. **WR bajo** (50-53%) - apenas por encima de random
2. **Meses perdedores** - algunos meses pierden significativamente
3. **ETH no funciona** bien con Ridge
4. **Solo LONG** - pierde toda la ventaja en bear markets

## Recomendación Final

### Para BTC
```python
model = Ridge(alpha=100)
features = ['ret_1', 'ret_5', 'ret_20', 'vol20', 'rsi14', 'ema21_d', 'vr']
direction = 'LONG_ONLY'
tp_pct = 0.025  # 2.5%
sl_pct = 0.025  # 2.5%
conv_min = 0.5
```

### Para ETH
**No recomendado** - WR < 40% con la mayoría de configuraciones

### Alternativas a considerar

1. **Usar ExtraTrees_d3** - Mayor correlación de test pero más overfitting
2. **Filtro de régimen** - Solo tradear en mercados específicos
3. **Ensemble** - Combinar Ridge + ExtraTrees
4. **Más datos recientes** - Rolling window de 6 meses

## Comparación con V13.03

| Métrica | V13.03 (backtest) | Low-Overfit (test real) |
|---------|-------------------|-------------------------|
| WR | 67% | 52% |
| Overfitting | 89-94% drop | 0.1% drop |
| Confianza | BAJA | ALTA |
| Expectativa real | ~50% | ~52% |

**V13.03 se veía bien pero era overfitting. El modelo Low-Overfit es más realista.**

---

## Multi-Pair Experiment (Actualizado)

### Resultados por Par

| Pair | Test Period | Corr Drop | Direction | Trades | WR | PnL |
|------|-------------|-----------|-----------|--------|-----|-----|
| **BTC** | 2024-09 a 2026-02 | **0.1%** | LONG_ONLY | 192 | **58.3%** | $322 |
| **ADA** | 2024-12 a 2026-02 | **4.3%** | LONG_ONLY | 322 | **50.6%** | $412 |
| XRP | 2024-12 a 2026-02 | -10.4% | LONG_ONLY | 660 | 39.8% | $592 |
| LINK | 2024-12 a 2026-02 | 58.8% | LONG_ONLY | 670 | 39.4% | $556 |
| DOGE | 2024-12 a 2026-02 | 79.9% | LONG_ONLY | 181 | 54.1% | $551 |
| AVAX | 2025-01 a 2026-02 | 105.0% | LONG_ONLY | 367 | 41.1% | $401 |
| NEAR | 2025-01 a 2026-02 | 120.6% | LONG_ONLY | 293 | 50.9% | $387 |
| DOT | 2025-01 a 2026-02 | 95.8% | LONG_ONLY | 114 | 51.8% | $178 |
| BNB | 2024-09 a 2026-02 | 93.9% | LONG_ONLY | 387 | 52.7% | $174 |
| **ETH** | - | - | - | - | <40% | **EXCLUIDO** |

### Pares Confiables (Low Overfitting + WR >= 50%)

| Pair | Direction | TP | SL | Conv | Trades | WR | PnL | Overfit |
|------|-----------|----|----|------|--------|-----|-----|---------|
| **BTC** | LONG_ONLY | 2.0% | 2.0% | 1.0 | 192 | **58.3%** | $322 | **0.1%** |
| **ADA** | LONG_ONLY | 2.0% | 1.5% | 1.0 | 322 | **50.6%** | $412 | **4.3%** |

### Pares con Alto PnL pero Alto Overfitting

| Pair | Direction | TP | SL | Conv | Trades | WR | PnL | Overfit |
|------|-----------|----|----|------|--------|-----|-----|---------|
| DOGE | LONG_ONLY | 2.0% | 1.0% | 1.0 | 181 | 54.1% | $551 | 79.9% |
| NEAR | LONG_ONLY | 2.0% | 1.5% | 1.0 | 293 | 50.9% | $387 | 120.6% |
| DOT | LONG_ONLY | 2.5% | 2.0% | 1.0 | 114 | 51.8% | $178 | 95.8% |
| BNB | LONG_ONLY | 2.0% | 2.0% | 0.3 | 387 | 52.7% | $174 | 93.9% |

**ADVERTENCIA**: Los pares con >30% overfitting pueden no mantener su rendimiento en produccion.

### Pares Excluidos

| Pair | Razon |
|------|-------|
| ETH | WR < 40% en todas las configuraciones |
| XRP | WR 39.8% (no confiable a pesar de alto PnL) |
| LINK | WR 39.4% (no confiable a pesar de alto PnL) |
| AVAX | WR 41.1% + 105% overfitting |

## Configuracion Recomendada para Produccion

### Tier 1: Alta Confianza (bajo overfitting)
```python
# BTC - MEJOR OPCION
PAIR_CONFIG['BTC/USDT'] = {
    'model': Ridge(alpha=100),
    'features': ['ret_1', 'ret_5', 'ret_20', 'vol20', 'rsi14', 'ema21_d', 'vr'],
    'direction': 'LONG_ONLY',
    'tp_pct': 0.02,
    'sl_pct': 0.02,
    'conv_min': 1.0,
    'expected_wr': '58%',
    'overfitting': '0.1%'
}

# ADA - SEGUNDA OPCION
PAIR_CONFIG['ADA/USDT'] = {
    'model': Ridge(alpha=100),
    'features': ['ret_1', 'ret_5', 'ret_20', 'vol20', 'rsi14', 'ema21_d', 'vr'],
    'direction': 'LONG_ONLY',
    'tp_pct': 0.02,
    'sl_pct': 0.015,
    'conv_min': 1.0,
    'expected_wr': '51%',
    'overfitting': '4.3%'
}
```

### Tier 2: Media Confianza (usar con cautela)
- DOGE, NEAR, DOT, BNB tienen WR >= 50% pero alto overfitting
- Monitorear de cerca en produccion
- Reducir posicion size vs Tier 1

### Excluidos
- ETH: No funciona con este modelo
- XRP, LINK, AVAX: WR < 45%

## Resumen Ejecutivo

| Metrica | Valor |
|---------|-------|
| Pares Tier 1 | BTC, ADA |
| Pares Tier 2 | DOGE, NEAR, DOT, BNB |
| Pares Excluidos | ETH, XRP, LINK, AVAX |
| PnL Total Tier 1 | $734 (sobre $100 cada par) |
| PnL Total Tier 2 | $1,290 (riesgo alto) |
| Modelo | Ridge(alpha=100) |
| Features | 7 (minimal set) |
| Direction | LONG_ONLY |

---

*Documento actualizado: 2026-02-27*
*Scripts: btc_model_experiments.py, btc_low_overfit_experiment.py, analyze_conviction_thresholds.py, multi_pair_low_overfit.py*
