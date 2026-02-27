# Optimizacion ADA/USDT - V13.03

## Fecha: 2026-02-27

## Resumen
- **Par**: ADA/USDT
- **Modelo**: GradientBoosting V2 (54 features)
- **Correlacion Test**: 0.1127

## Configuracion Optima

| Parametro | Valor |
|-----------|-------|
| TP | 5.0% |
| SL | 4.0% |
| Conviction Min | 0.5 |
| Direccion | BOTH |

## Analisis de Direccion (Periodo Test)

| Direccion | Trades | WR | PnL |
|-----------|--------|-----|-----|
| AMBOS | 785 | 51.7% | $49.56 |
| LONG ONLY | 485 | 49.1% | $18.38 |
| SHORT ONLY | 300 | 56.0% | $31.18 |

**Decision**: BOTH

## Validacion Multi-Periodo

| Periodo | Trades | WR | PnL | PF |
|---------|--------|-----|-----|-----|
| 2020 | 476 | 72.1% | $119.17 | 3.30 |
| 2021 | 721 | 71.0% | $166.36 | 3.01 |
| 2022 | 251 | 71.3% | $58.14 | 3.02 |
| 2023 | 196 | 63.8% | $34.24 | 2.27 |
| 2024 | 247 | 65.2% | $46.10 | 2.35 |
| 2025 | 267 | 69.3% | $58.17 | 2.81 |
| Ultimo Ano | 267 | 70.8% | $62.58 | 3.05 |

## Totales

- **Total Trades**: 2425
- **PnL Total**: $544.76
- **WR Promedio**: 69.1%

## Comparacion V7 vs V2

| Metrica | V7 (Anterior) | V2 (Nuevo) | Mejora |
|---------|---------------|------------|--------|
| WR Ultimo Ano | 36.1% | 70.8% | +34.7% |
| PnL Ultimo Ano | $19.87 | $62.58 | $+42.71 |
| Trades Ultimo Ano | 1685 | 267 | -1418 |

## Archivo de Modelo
- `models/ada_usdt_v2_gradientboosting.pkl`

## Codigo para Configuracion
```python
ML_ADA_CONFIG = {
    'model_file': 'ada_usdt_v2_gradientboosting.pkl',
    'tp_pct': 0.05,
    'sl_pct': 0.04,
    'conv_min': 0.5,
    'only_long': False,
    'only_short': False,
}
```
