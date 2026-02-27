# Optimizacion DOGE/USDT - V13.03

## Fecha: 2026-02-27

## Resumen
- **Par**: DOGE/USDT
- **Modelo**: GradientBoosting V2 (54 features)
- **Correlacion Test**: 0.0723

## Configuracion Optima

| Parametro | Valor |
|-----------|-------|
| TP | 5.0% |
| SL | 2.5% |
| Conviction Min | 0.5 |
| Direccion | BOTH |

## Analisis de Direccion (Periodo Test)

| Direccion | Trades | WR | PnL |
|-----------|--------|-----|-----|
| AMBOS | 330 | 39.1% | $13.81 |
| LONG ONLY | 221 | 38.9% | $9.25 |
| SHORT ONLY | 109 | 39.4% | $4.56 |

**Decision**: BOTH

## Validacion Multi-Periodo

| Periodo | Trades | WR | PnL | PF |
|---------|--------|-----|-----|-----|
| 2020 | 40 | 42.5% | $2.75 | 1.48 |
| 2021 | 205 | 45.9% | $19.25 | 1.69 |
| 2022 | 26 | 57.7% | $4.75 | 2.73 |
| 2023 | 4 | 25.0% | $-0.74 | 0.01 |
| 2024 | 44 | 50.0% | $5.50 | 2.00 |
| 2025 | 3 | 66.7% | $0.75 | 4.00 |
| Ultimo Ano | 5 | 40.0% | $0.25 | 1.33 |

## Totales

- **Total Trades**: 327
- **PnL Total**: $32.51
- **WR Promedio**: 46.8%

## Comparacion V7 vs V2

| Metrica | V7 (Anterior) | V2 (Nuevo) | Mejora |
|---------|---------------|------------|--------|
| WR Ultimo Ano | 36.1% | 40.0% | +3.9% |
| PnL Ultimo Ano | $19.87 | $0.25 | $-19.62 |
| Trades Ultimo Ano | 1685 | 5 | -1680 |

## Archivo de Modelo
- `models/doge_usdt_v2_gradientboosting.pkl`

## Codigo para Configuracion
```python
ML_DOGE_CONFIG = {
    'model_file': 'doge_usdt_v2_gradientboosting.pkl',
    'tp_pct': 0.05,
    'sl_pct': 0.025,
    'conv_min': 0.5,
    'only_long': False,
    'only_short': False,
}
```
