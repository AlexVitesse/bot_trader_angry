import pandas as pd
import pandas_ta as ta
import numpy as np
from config.settings import SYMBOL

# Configuración v6.7 "Metralladora Inteligente"
INITIAL_CAPITAL = 100.0
BASE_ORDER_MARGIN = 5.0     # $5 inicial
LEVERAGE = 10               # Volvemos a 10x
DCA_STEP_PCT = 0.008        # Distancia mas larga (0.8%) para mayor seguridad
TAKE_PROFIT_PCT = 0.006     # 0.6% TP (6% ROE con 10x) - Cubre comisiones y deja profit
MAX_SAFETY_ORDERS = 2       # Solo 2 recompras para no sobre-apalancarse
MARTINGALE_MULTIPLIER = 2.0 
STOP_LOSS_CATASTROPHIC = 0.015 # 1.5% SL desde el promedio (Mas ajustado)

def run_massive_backtest():
    filename = f"data/{SYMBOL.replace('/', '_')}_1m_history.parquet"
    df = pd.read_parquet(filename)
    
    # Indicadores
    df['ema_trend'] = ta.ema(df['close'], length=200) # FILTRO DE TENDENCIA
    bbands = ta.bbands(df['close'], length=20, std=2.0)
    df['bb_lower'] = bbands.iloc[:, 0]
    df['bb_upper'] = bbands.iloc[:, 2]
    
    stoch = ta.stochrsi(df['close'], length=14, rsi_length=14, k=3, d=3)
    df['stoch_k'] = stoch.iloc[:, 0]
    
    df.dropna(inplace=True)

    balance = INITIAL_CAPITAL
    in_position = False
    position_type = 0 
    avg_price = 0.0
    total_size_notional = 0.0
    so_count = 0
    trades = []
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['close']
        
        if in_position:
            # Gestion de Salida
            closed = False
            if position_type == 1:
                # TP
                if price >= avg_price * (1 + TAKE_PROFIT_PCT):
                    pnl = (total_size_notional * TAKE_PROFIT_PCT) - (total_size_notional * 0.0008)
                    balance += pnl
                    trades.append({'res': 'WIN', 'pnl': pnl})
                    in_position = False
                    continue
                # DCA
                if so_count < MAX_SAFETY_ORDERS and price <= avg_price * (1 - DCA_STEP_PCT):
                    so_count += 1
                    so_size = (BASE_ORDER_MARGIN * LEVERAGE) * (MARTINGALE_MULTIPLIER ** so_count)
                    new_size = total_size_notional + so_size
                    avg_price = ((avg_price * total_size_notional) + (price * so_size)) / new_size
                    total_size_notional = new_size
                # SL
                if price <= avg_price * (1 - STOP_LOSS_CATASTROPHIC):
                    pnl = -(total_size_notional * STOP_LOSS_CATASTROPHIC) - (total_size_notional * 0.0008)
                    balance += pnl
                    trades.append({'res': 'LOSS', 'pnl': pnl})
                    in_position = False

            elif position_type == -1:
                if price <= avg_price * (1 - TAKE_PROFIT_PCT):
                    pnl = (total_size_notional * TAKE_PROFIT_PCT) - (total_size_notional * 0.0008)
                    balance += pnl
                    trades.append({'res': 'WIN', 'pnl': pnl})
                    in_position = False
                    continue
                if so_count < MAX_SAFETY_ORDERS and price >= avg_price * (1 + DCA_STEP_PCT):
                    so_count += 1
                    so_size = (BASE_ORDER_MARGIN * LEVERAGE) * (MARTINGALE_MULTIPLIER ** so_count)
                    new_size = total_size_notional + so_size
                    avg_price = ((avg_price * total_size_notional) + (price * so_size)) / new_size
                    total_size_notional = new_size
                if price >= avg_price * (1 + STOP_LOSS_CATASTROPHIC):
                    pnl = -(total_size_notional * STOP_LOSS_CATASTROPHIC) - (total_size_notional * 0.0008)
                    balance += pnl
                    trades.append({'res': 'LOSS', 'pnl': pnl})
                    in_position = False

        if not in_position:
            # TENDENCIA ALCISTA (Price > EMA 200) -> Solo Longs
            if price > row['ema_trend']:
                if prev['close'] <= prev['bb_lower'] and row['close'] > row['bb_lower'] and row['stoch_k'] < 20:
                    in_position = True
                    position_type = 1
                    avg_price = price
                    total_size_notional = BASE_ORDER_MARGIN * LEVERAGE
                    so_count = 0
            
            # TENDENCIA BAJISTA (Price < EMA 200) -> Solo Shorts
            elif price < row['ema_trend']:
                if prev['close'] >= prev['bb_upper'] and row['close'] < row['bb_upper'] and row['stoch_k'] > 80:
                    in_position = True
                    position_type = -1
                    avg_price = price
                    total_size_notional = BASE_ORDER_MARGIN * LEVERAGE
                    so_count = 0

    print("\n" + "="*60)
    print(f"🏆 RESULTADOS METRALLADORA v6.7 (CON FILTRO TENDENCIA)")
    print("="*60)
    print(f"Balance Inicial: ${INITIAL_CAPITAL:.2f}")
    print(f"Balance Final:   ${balance:.2f} ({((balance/INITIAL_CAPITAL)-1)*100:+.2f}%)")
    print(f"Total Trades:    {len(trades)}")
    if trades:
        wins = [t for t in trades if t['res'] == 'WIN']
        print(f"Win Rate:        {(len(wins)/len(trades))*100:.2f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_massive_backtest()
