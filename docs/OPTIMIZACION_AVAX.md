# Optimizacion AVAX/USDT - V13.03

## Fecha: 2026-02-27

## Resumen
- **Par**: AVAX/USDT
- **Modelo**: GradientBoosting V2 (54 features)
- **Correlacion Test**: 0.0810

## Configuracion Optima

| Parametro | Valor |
|-----------|-------|
| TP | 7.0% |
| SL | 2.0% |
| Conviction Min | 0.5 |
| Direccion | BOTH |

## Analisis de Direccion (Periodo Test)

| Direccion | Trades | WR | PnL |
|-----------|--------|-----|-----|
| AMBOS | 958 | 25.1% | $12.42 |
| LONG ONLY | 487 | 24.8% | $6.75 |
| SHORT ONLY | 471 | 25.3% | $5.67 |

**Decision**: BOTH

## Validacion Multi-Periodo

| Periodo | Trades | WR | PnL | PF |
|---------|--------|-----|-----|-----|
| 2020 | 32 | 65.6% | $12.50 | 6.68 |
| 2021 | 615 | 35.1% | $71.40 | 1.89 |
| 2022 | 322 | 51.2% | $83.18 | 3.65 |
| 2023 | 216 | 50.5% | $53.49 | 3.50 |
| 2024 | 161 | 49.7% | $39.26 | 3.42 |
| 2025 | 91 | 65.9% | $35.80 | 6.77 |
| Ultimo Ano | 94 | 61.7% | $33.40 | 5.64 |

## Totales

- **Total Trades**: 1531
- **PnL Total**: $329.04
- **WR Promedio**: 54.3%

## Comparacion V7 vs V2

| Metrica | V7 (Anterior) | V2 (Nuevo) | Mejora |
|---------|---------------|------------|--------|
| WR Ultimo Ano | 36.1% | 61.7% | +25.6% |
| PnL Ultimo Ano | $19.87 | $33.40 | $+13.53 |
| Trades Ultimo Ano | 1685 | 94 | -1591 |

## Archivo de Modelo
- `models/avax_usdt_v2_gradientboosting.pkl`

## Codigo para Configuracion
```python
ML_AVAX_CONFIG = {
    'model_file': 'avax_usdt_v2_gradientboosting.pkl',
    'tp_pct': 0.07,
    'sl_pct': 0.02,
    'conv_min': 0.5,
    'only_long': False,
    'only_short': False,
}
```
