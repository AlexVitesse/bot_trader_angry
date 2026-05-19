# Optimizacion LINK/USDT - V13.03

## Fecha: 2026-02-27

## Resumen
- **Par**: LINK/USDT
- **Modelo**: GradientBoosting V2 (54 features)
- **Correlacion Test**: 0.0752

## Configuracion Optima

| Parametro | Valor |
|-----------|-------|
| TP | 7.0% |
| SL | 4.0% |
| Conviction Min | 0.5 |
| Direccion | BOTH |

## Analisis de Direccion (Periodo Test)

| Direccion | Trades | WR | PnL |
|-----------|--------|-----|-----|
| AMBOS | 1186 | 41.0% | $43.84 |
| LONG ONLY | 654 | 39.3% | $18.65 |
| SHORT ONLY | 532 | 43.0% | $25.19 |

**Decision**: BOTH

## Validacion Multi-Periodo

| Periodo | Trades | WR | PnL | PF |
|---------|--------|-----|-----|-----|
| 2020 | 513 | 69.8% | $184.50 | 4.06 |
| 2021 | 523 | 69.0% | $187.23 | 3.90 |
| 2022 | 209 | 79.9% | $99.86 | 7.16 |
| 2023 | 161 | 66.5% | $47.78 | 3.51 |
| 2024 | 167 | 75.4% | $69.75 | 5.76 |
| 2025 | 182 | 79.7% | $84.90 | 6.88 |
| Ultimo Ano | 185 | 79.5% | $84.52 | 6.82 |

## Totales

- **Total Trades**: 1940
- **PnL Total**: $758.53
- **WR Promedio**: 74.3%

## Comparacion V7 vs V2

| Metrica | V7 (Anterior) | V2 (Nuevo) | Mejora |
|---------|---------------|------------|--------|
| WR Ultimo Ano | 36.1% | 79.5% | +43.4% |
| PnL Ultimo Ano | $19.87 | $84.52 | $+64.65 |
| Trades Ultimo Ano | 1685 | 185 | -1500 |

## Archivo de Modelo
- `models/link_usdt_v2_gradientboosting.pkl`

## Codigo para Configuracion
```python
ML_LINK_CONFIG = {
    'model_file': 'link_usdt_v2_gradientboosting.pkl',
    'tp_pct': 0.07,
    'sl_pct': 0.04,
    'conv_min': 0.5,
    'only_long': False,
    'only_short': False,
}
```
