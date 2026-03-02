# Experimento: Reglas Tecnicas Simples

## Fecha: 2026-02-28

## Objetivo
Probar si reglas tecnicas simples (sin ML) pueden generar edge sobre random.

## Estrategias Probadas

1. **RSI < 25**: Entrada cuando RSI muy oversold
2. **RSI + BB**: RSI < 30 + precio bajo en Bollinger Bands
3. **RSI + BB + Vol**: Agregar filtro de volumen alto
4. **Capitulacion**: 3+ velas rojas + RSI extremo + volumen
5. **Mean Reversion**: 4+ velas rojas + RSI < 25
6. **BB Squeeze**: BB estrecho + breakout con volumen
7. **Multi Confirm**: EMA200 alcista + pullback + soporte BB

## Resultados en TEST (Sep 2025 - Feb 2026)

| Estrategia | Trades | WR | PnL | PF |
|------------|--------|-----|-----|-----|
| Random | 48 | 33.3% | -22.6% | 0.72 |
| RSI < 25 | 20 | 45.0% | +4.3% | 1.15 |
| RSI + BB | 37 | 35.1% | -18.1% | 0.72 |
| RSI + BB + Vol | 30 | 33.3% | -19.5% | 0.63 |
| Capitulacion | 30 | 36.7% | -16.1% | 0.69 |
| Mean Reversion | 18 | 38.9% | -4.7% | 0.84 |
| BB Squeeze | 5 | 60.0% | +7.7% | 3.03 |
| Multi Confirm | 3 | 0.0% | -6.5% | 0.00 |

## Analisis

### Lo que funciono (marginalmente):
- **RSI < 25**: Edge positivo (+4.3% vs -22.6% random)
- **BB Squeeze**: Muy buen PF pero solo 5 trades

### Por que no es suficiente:
1. **Pocos trades**: RSI < 25 solo 20 trades en 6 meses
2. **No adaptativo**: La misma regla para todas las condiciones
3. **Mercado cambiante**: Lo que funciono en train no funciona igual en test
4. **Sin contexto**: Ignora el estado general del mercado

## Conclusion

Las reglas simples pueden dar un edge marginal, pero:
- Son muy rigidas
- No se adaptan al mercado
- El mercado cambia y las invalida

**Se requiere un sistema adaptativo** que:
1. Detecte el tipo de mercado actual
2. Seleccione la estrategia apropiada
3. Sepa cuando NO tradear

Ver: `PLAN_V1305_ADAPTIVE_EXPERT.md`

---

*Documento creado: 2026-02-28*
