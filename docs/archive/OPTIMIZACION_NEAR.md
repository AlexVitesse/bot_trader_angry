# Optimizacion NEAR/USDT - V13.03

## Fecha: 2026-02-27

## Resumen
- **Par**: NEAR/USDT
- **Modelo**: GradientBoosting V2 (54 features)
- **Correlacion Test**: -0.0672

## Configuracion Optima

| Parametro | Valor |
|-----------|-------|
| TP | 5.0% |
| SL | 1.5% |
| Conviction Min | 0.5 |
| Direccion | SHORT_ONLY |

## Analisis de Direccion (Periodo Test)

| Direccion | Trades | WR | PnL |
|-----------|--------|-----|-----|
| AMBOS | 1169 | 25.6% | $18.74 |
| LONG ONLY | 662 | 22.2% | $-3.75 |
| SHORT ONLY | 507 | 30.0% | $22.49 |

**Decision**: SHORT_ONLY

## Validacion Multi-Periodo

| Periodo | Trades | WR | PnL | PF |
|---------|--------|-----|-----|-----|
| 2020 | 7 | 28.6% | $0.25 | 1.33 |
| 2021 | 198 | 39.9% | $21.65 | 2.21 |
| 2022 | 359 | 44.3% | $49.50 | 2.65 |
| 2023 | 11 | 63.6% | $2.90 | 5.83 |
| 2024 | 131 | 53.4% | $25.85 | 3.83 |
| 2025 | 86 | 50.0% | $15.17 | 3.40 |
| Ultimo Ano | 76 | 50.0% | $13.42 | 3.40 |

## Totales

- **Total Trades**: 868
- **PnL Total**: $128.74
- **WR Promedio**: 47.1%

## Comparacion V7 vs V2

| Metrica | V7 (Anterior) | V2 (Nuevo) | Mejora |
|---------|---------------|------------|--------|
| WR Ultimo Ano | 36.1% | 50.0% | +13.9% |
| PnL Ultimo Ano | $19.87 | $13.42 | $-6.45 |
| Trades Ultimo Ano | 1685 | 76 | -1609 |

## Archivo de Modelo
- `models/near_usdt_v2_gradientboosting.pkl`

## Codigo para Configuracion
```python
ML_NEAR_CONFIG = {
    'model_file': 'near_usdt_v2_gradientboosting.pkl',
    'tp_pct': 0.05,
    'sl_pct': 0.015,
    'conv_min': 0.5,
    'only_long': False,
    'only_short': True,
}
```
