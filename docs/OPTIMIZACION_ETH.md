# Optimizacion ETH/USDT - V13.03

## Fecha: 2026-02-27

## Resumen
- **Par**: ETH/USDT
- **Modelo**: GradientBoosting V2 (54 features)
- **Correlacion Test**: 0.0628

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
| AMBOS | 1060 | 45.4% | $68.40 |
| LONG ONLY | 516 | 44.6% | $33.11 |
| SHORT ONLY | 544 | 46.1% | $35.29 |

**Decision**: BOTH

## Validacion Multi-Periodo

| Periodo | Trades | WR | PnL | PF |
|---------|--------|-----|-----|-----|
| 2020 | 575 | 69.9% | $206.41 | 4.33 |
| 2021 | 521 | 72.7% | $236.76 | 5.22 |
| 2022 | 385 | 65.2% | $134.06 | 3.77 |
| 2023 | 147 | 66.7% | $28.62 | 2.93 |
| 2024 | 184 | 69.0% | $62.43 | 4.19 |
| 2025 | 293 | 67.2% | $95.91 | 3.76 |
| Ultimo Ano | 331 | 68.0% | $117.45 | 4.06 |

## Totales

- **Total Trades**: 2436
- **PnL Total**: $881.64
- **WR Promedio**: 68.4%

## Comparacion V7 vs V2

| Metrica | V7 (Anterior) | V2 (Nuevo) | Mejora |
|---------|---------------|------------|--------|
| WR Ultimo Ano | 36.1% | 68.0% | +31.9% |
| PnL Ultimo Ano | $19.87 | $117.45 | $+97.58 |
| Trades Ultimo Ano | 1685 | 331 | -1354 |

## Archivo de Modelo
- `models/eth_usdt_v2_gradientboosting.pkl`

## Codigo para Configuracion
```python
ML_ETH_CONFIG = {
    'model_file': 'eth_usdt_v2_gradientboosting.pkl',
    'tp_pct': 0.08,
    'sl_pct': 0.04,
    'conv_min': 0.5,
    'only_long': False,
    'only_short': False,
}
```
