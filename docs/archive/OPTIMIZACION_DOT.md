# Optimizacion DOT/USDT - V13.03

## Fecha: 2026-02-27

## Resumen
- **Par**: DOT/USDT
- **Modelo**: GradientBoosting V2 (54 features)
- **Correlacion Test**: 0.0383

## Configuracion Optima

| Parametro | Valor |
|-----------|-------|
| TP | 6.0% |
| SL | 2.5% |
| Conviction Min | 0.5 |
| Direccion | SHORT_ONLY |

## Analisis de Direccion (Periodo Test)

| Direccion | Trades | WR | PnL |
|-----------|--------|-----|-----|
| AMBOS | 680 | 32.8% | $13.96 |
| LONG ONLY | 409 | 26.9% | $-9.30 |
| SHORT ONLY | 271 | 41.7% | $23.26 |

**Decision**: SHORT_ONLY

## Validacion Multi-Periodo

| Periodo | Trades | WR | PnL | PF |
|---------|--------|-----|-----|-----|
| 2020 | 3 | 66.7% | $0.95 | 4.80 |
| 2021 | 252 | 60.7% | $67.05 | 3.71 |
| 2022 | 141 | 63.8% | $40.90 | 4.21 |
| 2023 | 1 | 100.0% | $0.60 | 60.00 |
| 2024 | 65 | 75.4% | $25.24 | 7.31 |
| 2025 | 35 | 85.7% | $16.75 | 14.40 |
| Ultimo Ano | 49 | 83.7% | $22.60 | 12.30 |

## Totales

- **Total Trades**: 546
- **PnL Total**: $174.09
- **WR Promedio**: 76.6%

## Comparacion V7 vs V2

| Metrica | V7 (Anterior) | V2 (Nuevo) | Mejora |
|---------|---------------|------------|--------|
| WR Ultimo Ano | 36.1% | 83.7% | +47.6% |
| PnL Ultimo Ano | $19.87 | $22.60 | $+2.73 |
| Trades Ultimo Ano | 1685 | 49 | -1636 |

## Archivo de Modelo
- `models/dot_usdt_v2_gradientboosting.pkl`

## Codigo para Configuracion
```python
ML_DOT_CONFIG = {
    'model_file': 'dot_usdt_v2_gradientboosting.pkl',
    'tp_pct': 0.06,
    'sl_pct': 0.025,
    'conv_min': 0.5,
    'only_long': False,
    'only_short': True,
}
```
