# V13.03 Release - Todos los Pares Optimizados

## Fecha: 2026-02-27

## Resumen Ejecutivo

V13.03 representa una mejora MASIVA sobre V13.02:
- **Win Rate**: 35.4% -> 67.3% (+31.9%)
- **PnL**: $79.96 -> $402.74 (+404%)
- **Max DD**: 16.1% -> 10.1% (menor riesgo)
- **Profit Factor**: 1.10 -> 3.98

## Cambios Principales

### Todos los pares ahora usan modelos V2 GradientBoosting

Antes (V13.02):
- Solo BTC y BNB tenian modelos V2 optimizados
- Los otros 8 pares usaban modelo V7 generico
- V7 generaba demasiados trades de baja calidad

Ahora (V13.03):
- 10 pares con modelos V2 individuales
- TP/SL optimizado por par
- Filtros de direccion donde aplica (SHORT ONLY para BNB, NEAR, DOT)

## Configuracion por Par

| Par | TP | SL | Conv | Direccion |
|-----|-----|-----|------|-----------|
| BTC/USDT | 4% | 2% | 1.0 | BOTH |
| BNB/USDT | 7% | 3.5% | 1.0 | SHORT ONLY |
| XRP/USDT | 8% | 4% | 0.5 | BOTH |
| ETH/USDT | 8% | 4% | 0.5 | BOTH |
| AVAX/USDT | 7% | 2% | 0.5 | BOTH |
| ADA/USDT | 5% | 4% | 0.5 | BOTH |
| LINK/USDT | 7% | 4% | 0.5 | BOTH |
| DOGE/USDT | 5% | 2.5% | 0.5 | BOTH |
| NEAR/USDT | 5% | 1.5% | 0.5 | SHORT ONLY |
| DOT/USDT | 6% | 2.5% | 0.5 | SHORT ONLY |

## Resultados Backtest

### Por Periodo (Capital $100)

| Periodo | Trades | WR | PnL | MaxDD% | PF |
|---------|--------|-----|-----|--------|-----|
| Ultimo Ano | 1,299 | 67.3% | $402.74 | 10.1% | 3.98 |
| Mejor (2022) | 1,957 | 62.3% | $563.03 | 6.7% | 3.89 |
| Peor (2025) | 1,182 | 66.0% | $344.86 | 10.8% | 3.65 |
| Sintetico | 1,870 | 59.6% | $490.71 | 6.7% | 3.49 |

### Por Par (Ultimo Ano)

| Par | Trades | WR | PnL | Notas |
|-----|--------|-----|-----|-------|
| ETH/USDT | 335 | 67.5% | $117.11 | Top performer |
| LINK/USDT | 185 | 79.5% | $84.52 | Mejor WR |
| ADA/USDT | 267 | 70.8% | $62.58 | Consistente |
| XRP/USDT | 122 | 68.0% | $46.27 | Bueno |
| AVAX/USDT | 94 | 61.7% | $33.40 | Bueno |
| DOT/USDT | 49 | 83.7% | $22.60 | WR excepcional |
| BNB/USDT | 99 | 53.5% | $14.25 | Solo SHORT |
| NEAR/USDT | 76 | 50.0% | $13.42 | Solo SHORT |
| BTC/USDT | 67 | 55.2% | $8.34 | Conservador |
| DOGE/USDT | 5 | 40.0% | $0.25 | Pocos trades |

## Comparacion V13.02 vs V13.03

| Metrica | V13.02 | V13.03 | Mejora |
|---------|--------|--------|--------|
| Trades | 8,142 | 1,299 | -84% |
| Win Rate | 35.4% | 67.3% | +31.9% |
| PnL | $79.96 | $402.74 | +404% |
| Max DD | 16.1% | 10.1% | -37% |
| Profit Factor | 1.10 | 3.98 | +262% |

## Archivos de Modelo

```
models/
  btc_v2_gradientboosting.pkl
  bnb_usdt_v2_gradientboosting.pkl
  xrp_usdt_v2_gradientboosting.pkl
  eth_usdt_v2_gradientboosting.pkl
  avax_usdt_v2_gradientboosting.pkl
  ada_usdt_v2_gradientboosting.pkl
  link_usdt_v2_gradientboosting.pkl
  doge_usdt_v2_gradientboosting.pkl
  near_usdt_v2_gradientboosting.pkl
  dot_usdt_v2_gradientboosting.pkl
```

## Notas de Implementacion

### DOGE/USDT
- Solo 5 trades en el ultimo ano
- Considerar desactivar o aumentar conviction
- Mantener en observacion

### SHORT ONLY
- BNB, NEAR, DOT funcionan mejor solo con SHORTs
- LONGs en estos pares tienen WR muy bajo

## Proximos Pasos

1. Actualizar `config/settings.py` con nueva configuracion
2. Actualizar `src/ml_strategy.py` para cargar modelos V2 por par
3. Testing en testnet
4. Deploy a produccion

## Conclusion

V13.03 es un bot ganador con:
- 67% Win Rate (vs 35% anterior)
- $402 PnL anual con $100 capital (402% retorno)
- 10% Max Drawdown (aceptable)
- Profit Factor 3.98 (excelente)
