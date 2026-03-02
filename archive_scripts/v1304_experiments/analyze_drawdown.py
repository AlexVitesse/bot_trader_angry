"""
Analiza el drawdown del ultimo ano para V9.5+ATR
"""
import json
import pandas as pd
import numpy as np

# Cargar resultados del backtest
print("="*60)
print("ANALISIS DE DRAWDOWN")
print("="*60)

# Los datos del backtest muestran:
# - Ultimo Ano V9.5+ATR: max_dd = 51.9%
# - Bear Market 2022 V9.5+ATR: max_dd = 13.2%

# Comparacion con V8.5:
# - Ultimo Ano V8.5: max_dd = 65.3%
# - Bear Market 2022 V8.5: max_dd = 6.3%

print("\nCOMPARACION DRAWDOWN:")
print("-"*60)
print(f"{'Periodo':<25} {'V8.5':>12} {'V9.5+ATR':>12} {'Mejor':>10}")
print("-"*60)
print(f"{'Ultimo Ano 2025-2026':<25} {'65.3%':>12} {'51.9%':>12} {'V9.5':>10}")
print(f"{'Bear Market 2022':<25} {'6.3%':>12} {'13.2%':>12} {'V8.5':>10}")

print("\n" + "="*60)
print("PROBLEMA IDENTIFICADO:")
print("="*60)
print("""
El drawdown del 52% en el ultimo ano es causado por:

1. ATR DINAMICO EN MERCADO LATERAL:
   - ATR 2.5x TP / 1.0x SL significa ratio 2.5:1
   - En mercado lateral, el precio no alcanza TP
   - Pero SI alcanza SL frecuentemente
   - Resultado: muchas perdidas consecutivas

2. ULTIMO ANO FUE POCO VOLATIL:
   - WR bajo (34.1%) porque mercado lateral
   - ATR grande = SL grande en % del precio
   - Perdidas de 2-3% por trade acumulan rapido

3. BEAR MARKET 2022 FUE DIFERENTE:
   - Alta volatilidad = precio alcanza TP
   - WR alto (67.9%)
   - DD bajo (13.2%) porque gana mas de lo que pierde
""")

print("="*60)
print("SOLUCIONES PROPUESTAS:")
print("="*60)
print("""
OPCION A: Reducir ATR multiplier
   - Cambiar de 2.5x/1.0x a 2.0x/1.0x o 1.5x/0.75x
   - Menor TP = mas trades ganadores
   - Menor DD pero tambien menor profit en bull runs

OPCION B: Agregar filtro de volatilidad
   - Solo operar cuando ATR > promedio
   - Evitar mercados laterales de baja volatilidad
   - Menos trades pero mejor calidad

OPCION C: Reducir tamano de posicion
   - En vez de 2% riesgo, usar 1% o 1.5%
   - DD se reduce proporcionalmente
   - Ganancias tambien se reducen

OPCION D: Usar TP/SL fijos en mercado lateral
   - Detectar mercado lateral (chop alto)
   - Usar TP/SL fijos 3%/1.5% en lateral
   - Usar ATR dinamico solo en mercado trending
""")

# Calcular impacto de reducir riesgo
print("="*60)
print("SIMULACION: Reducir riesgo de 2% a 1%")
print("="*60)

# Con 2% riesgo
dd_2pct = 51.9
pnl_2pct = 4503

# Con 1% riesgo (DD y PnL se reducen ~50%)
dd_1pct = dd_2pct * 0.5
pnl_1pct = pnl_2pct * 0.5

print(f"\nCon 2% riesgo por trade:")
print(f"  Max DD: {dd_2pct:.1f}%")
print(f"  PnL: ${pnl_2pct:,.0f}")

print(f"\nCon 1% riesgo por trade:")
print(f"  Max DD: {dd_1pct:.1f}%")
print(f"  PnL: ${pnl_1pct:,.0f}")

print(f"\nCon 1.5% riesgo por trade:")
print(f"  Max DD: {dd_2pct * 0.75:.1f}%")
print(f"  PnL: ${pnl_2pct * 0.75:,.0f}")

print("\n" + "="*60)
print("RECOMENDACION:")
print("="*60)
print("""
Para produccion, usar configuracion CONSERVADORA:

1. Riesgo por trade: 1% (no 2%)
2. Max posiciones: 2 (no 3)
3. ATR: 2.0x/1.0x (no 2.5x/1.0x)

Resultado esperado:
- Max DD: ~25-30% (aceptable)
- PnL mensual: ~$150-250/mes (conservador)
- Protege capital en mercados dificiles
""")
