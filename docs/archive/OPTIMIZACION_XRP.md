# Optimizacion XRP/USDT - V13.03

## Fecha: 2026-02-27

## Resumen
- **Par**: XRP/USDT
- **Modelo**: GradientBoosting V2 (54 features)
- **Correlacion Test**: 0.0802

## Configuracion Optima

| Parametro | Valor |
|-----------|-------|
| TP | 8.0% |
| SL | 4.0% |
| Conviction Min | 0.5 |
| Direccion | BOTH |

## Analisis de Direccion (Periodo Test)

| Direccion | Trades | WR | PnL |
|-----------|--------|-----|-----|
| AMBOS | 540 | 51.1% | $71.76 |
| LONG ONLY | 236 | 53.0% | $46.35 |
| SHORT ONLY | 304 | 49.7% | $25.42 |

**Decision**: BOTH

## Validacion Multi-Periodo

| Periodo | Trades | WR | PnL | PF |
|---------|--------|-----|-----|-----|
| 2020 | 172 | 59.9% | $52.48 | 2.93 |
| 2021 | 360 | 58.3% | $107.80 | 2.80 |
| 2022 | 104 | 56.7% | $29.20 | 2.62 |
| 2023 | 75 | 66.7% | $27.32 | 3.83 |
| 2024 | 173 | 64.2% | $60.83 | 3.47 |
| 2025 | 112 | 62.5% | $35.65 | 3.23 |
| Ultimo Ano | 122 | 68.0% | $46.27 | 4.14 |

## Totales

- **Total Trades**: 1118
- **PnL Total**: $359.54
- **WR Promedio**: 62.3%

## Comparacion V7 vs V2

| Metrica | V7 (Anterior) | V2 (Nuevo) | Mejora |
|---------|---------------|------------|--------|
| WR Ultimo Ano | 36.1% | 68.0% | +31.9% |
| PnL Ultimo Ano | $19.87 | $46.27 | $+26.40 |
| Trades Ultimo Ano | 1685 | 122 | -1563 |

## Archivo de Modelo
- `models/xrp_usdt_v2_gradientboosting.pkl`

## Codigo para Configuracion
```python
ML_XRP_CONFIG = {
    'model_file': 'xrp_usdt_v2_gradientboosting.pkl',
    'tp_pct': 0.08,
    'sl_pct': 0.04,
    'conv_min': 0.5,
    'only_long': False,
    'only_short': False,
}
```
