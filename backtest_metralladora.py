"""
Backtest v6.7 - Smart Metralladora (con Protecciones)
======================================================
Incluye:
  - Filtro de regimen de volatilidad (ATR)
  - Rolling WR kill switch (pausa si WR < 78%)
  - Comparacion con/sin protecciones
"""

import pandas as pd
import pandas_ta as ta
import numpy as np

from config.settings import (
    SYMBOL, BB_LENGTH, BB_STD, EMA_TREND_LENGTH,
    INITIAL_CAPITAL, LEVERAGE, COMMISSION_RATE,
    BASE_ORDER_MARGIN, DCA_STEP_PCT, MAX_SAFETY_ORDERS,
    MARTINGALE_MULTIPLIER, TAKE_PROFIT_PCT, STOP_LOSS_CATASTROPHIC,
    ATR_REGIME_LENGTH, ATR_REGIME_MULT_HIGH, ATR_REGIME_MULT_LOW,
    ROLLING_WR_WINDOW, ROLLING_WR_MIN, MAX_CONSECUTIVE_LOSSES
)


def run_backtest(use_protections: bool = True):
    """Ejecuta backtest. use_protections=True activa filtro ATR + rolling WR."""
    filename = f"data/{SYMBOL.replace('/', '_')}_1m_history.parquet"
    df = pd.read_parquet(filename)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Indicadores
    df['ema_trend'] = ta.ema(df['close'], length=EMA_TREND_LENGTH)
    bbands = ta.bbands(df['close'], length=BB_LENGTH, std=BB_STD)
    df['bb_lower'] = bbands.iloc[:, 0]
    df['bb_upper'] = bbands.iloc[:, 2]
    stoch = ta.stochrsi(df['close'], length=14, rsi_length=14, k=3, d=3)
    df['stoch_k'] = stoch.iloc[:, 0]

    # ATR para filtro de regimen
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=ATR_REGIME_LENGTH)
    df['atr_sma'] = df['atr'].rolling(window=100).mean()

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Estado
    balance = INITIAL_CAPITAL
    peak_balance = INITIAL_CAPITAL
    max_drawdown = 0.0
    in_position = False
    position_type = 0
    avg_price = 0.0
    total_size_notional = 0.0
    so_count = 0
    entry_commission = 0.0
    trades = []
    balance_curve = [INITIAL_CAPITAL]

    # Protecciones
    is_paused = False
    consecutive_losses = 0
    signals_filtered = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        price = row['close']
        high = row['high']
        low = row['low']

        if in_position:
            if position_type == 1:  # LONG
                tp_target = avg_price * (1 + TAKE_PROFIT_PCT)
                sl_target = avg_price * (1 - STOP_LOSS_CATASTROPHIC)

                if high >= tp_target:
                    gross_pnl = total_size_notional * TAKE_PROFIT_PCT
                    commission = entry_commission + (total_size_notional * COMMISSION_RATE)
                    net_pnl = gross_pnl - commission
                    balance += net_pnl
                    trades.append({'res': 'WIN', 'pnl': net_pnl, 'gross': gross_pnl, 'comm': commission, 'so': so_count})
                    in_position = False
                    consecutive_losses = 0

                elif low <= sl_target:
                    gross_pnl = -(total_size_notional * STOP_LOSS_CATASTROPHIC)
                    commission = entry_commission + (total_size_notional * COMMISSION_RATE)
                    net_pnl = gross_pnl - commission
                    balance += net_pnl
                    trades.append({'res': 'LOSS', 'pnl': net_pnl, 'gross': gross_pnl, 'comm': commission, 'so': so_count})
                    in_position = False
                    consecutive_losses += 1

                elif so_count < MAX_SAFETY_ORDERS:
                    dca_target = avg_price * (1 - DCA_STEP_PCT)
                    if low <= dca_target:
                        so_count += 1
                        so_size = (BASE_ORDER_MARGIN * LEVERAGE) * (MARTINGALE_MULTIPLIER ** so_count)
                        new_size = total_size_notional + so_size
                        avg_price = ((avg_price * total_size_notional) + (price * so_size)) / new_size
                        total_size_notional = new_size
                        entry_commission += so_size * COMMISSION_RATE

            elif position_type == -1:  # SHORT
                tp_target = avg_price * (1 - TAKE_PROFIT_PCT)
                sl_target = avg_price * (1 + STOP_LOSS_CATASTROPHIC)

                if low <= tp_target:
                    gross_pnl = total_size_notional * TAKE_PROFIT_PCT
                    commission = entry_commission + (total_size_notional * COMMISSION_RATE)
                    net_pnl = gross_pnl - commission
                    balance += net_pnl
                    trades.append({'res': 'WIN', 'pnl': net_pnl, 'gross': gross_pnl, 'comm': commission, 'so': so_count})
                    in_position = False
                    consecutive_losses = 0

                elif high >= sl_target:
                    gross_pnl = -(total_size_notional * STOP_LOSS_CATASTROPHIC)
                    commission = entry_commission + (total_size_notional * COMMISSION_RATE)
                    net_pnl = gross_pnl - commission
                    balance += net_pnl
                    trades.append({'res': 'LOSS', 'pnl': net_pnl, 'gross': gross_pnl, 'comm': commission, 'so': so_count})
                    in_position = False
                    consecutive_losses += 1

                elif so_count < MAX_SAFETY_ORDERS:
                    dca_target = avg_price * (1 + DCA_STEP_PCT)
                    if high >= dca_target:
                        so_count += 1
                        so_size = (BASE_ORDER_MARGIN * LEVERAGE) * (MARTINGALE_MULTIPLIER ** so_count)
                        new_size = total_size_notional + so_size
                        avg_price = ((avg_price * total_size_notional) + (price * so_size)) / new_size
                        total_size_notional = new_size
                        entry_commission += so_size * COMMISSION_RATE

        # ENTRADAS
        if not in_position:
            # === PROTECCIONES (solo si activadas) ===
            skip_entry = False
            if use_protections:
                # 1. Consecutive losses kill switch
                if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                    skip_entry = True

                # 2. Rolling WR check
                if len(trades) >= ROLLING_WR_WINDOW:
                    recent = trades[-ROLLING_WR_WINDOW:]
                    rolling_wr = sum(1 for t in recent if t['res'] == 'WIN') / len(recent)
                    if rolling_wr < ROLLING_WR_MIN:
                        skip_entry = True

                # 3. ATR regime filter
                if not pd.isna(row['atr']) and not pd.isna(row['atr_sma']) and row['atr_sma'] > 0:
                    atr_ratio = row['atr'] / row['atr_sma']
                    if atr_ratio > ATR_REGIME_MULT_HIGH or atr_ratio < ATR_REGIME_MULT_LOW:
                        skip_entry = True

            if skip_entry:
                signals_filtered += 1
                # Reset consecutive losses after cooling off (simula esperar)
                # En live el bot se pausa y el usuario reanuda
                if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                    consecutive_losses = 0
            else:
                # TENDENCIA ALCISTA -> Solo Longs
                if price > row['ema_trend']:
                    if prev['close'] <= prev['bb_lower'] and price > row['bb_lower'] and row['stoch_k'] < 20:
                        in_position = True
                        position_type = 1
                        avg_price = price
                        total_size_notional = BASE_ORDER_MARGIN * LEVERAGE
                        so_count = 0
                        entry_commission = total_size_notional * COMMISSION_RATE

                # TENDENCIA BAJISTA -> Solo Shorts
                elif price < row['ema_trend']:
                    if prev['close'] >= prev['bb_upper'] and price < row['bb_upper'] and row['stoch_k'] > 80:
                        in_position = True
                        position_type = -1
                        avg_price = price
                        total_size_notional = BASE_ORDER_MARGIN * LEVERAGE
                        so_count = 0
                        entry_commission = total_size_notional * COMMISSION_RATE

        balance_curve.append(balance)
        if balance > peak_balance:
            peak_balance = balance
        current_dd = (peak_balance - balance) / peak_balance
        if current_dd > max_drawdown:
            max_drawdown = current_dd

    return trades, balance_curve, max_drawdown, signals_filtered


def print_results(label, trades, max_drawdown, signals_filtered, balance):
    """Imprime resultados de un backtest."""
    total_return = ((balance / INITIAL_CAPITAL) - 1) * 100

    print(f"\n--- {label} ---")
    print(f"  Balance Final:      ${balance:.2f} ({total_return:+.2f}%)")
    print(f"  Max Drawdown:       {max_drawdown * 100:.2f}%")
    print(f"  Total Trades:       {len(trades)}")

    if signals_filtered > 0:
        print(f"  Senales filtradas:  {signals_filtered}")

    if trades:
        wins = [t for t in trades if t['res'] == 'WIN']
        losses = [t for t in trades if t['res'] == 'LOSS']
        win_rate = (len(wins) / len(trades)) * 100

        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
        total_gross_win = sum(t['pnl'] for t in wins) if wins else 0
        total_gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0
        profit_factor = total_gross_win / total_gross_loss if total_gross_loss > 0 else float('inf')
        total_commission = sum(t['comm'] for t in trades)
        dca_trades = [t for t in trades if t['so'] > 0]

        pnl_series = np.array([t['pnl'] for t in trades])
        sharpe = (np.mean(pnl_series) / np.std(pnl_series)) * np.sqrt(len(trades)) if np.std(pnl_series) > 0 else 0

        max_consecutive_losses = 0
        current_streak = 0
        for t in trades:
            if t['res'] == 'LOSS':
                current_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_streak)
            else:
                current_streak = 0

        print(f"  Win Rate:           {win_rate:.1f}%")
        print(f"  Profit Factor:      {profit_factor:.2f}")
        print(f"  Sharpe Ratio:       {sharpe:.2f}")
        print(f"  Wins:  {len(wins)} | Avg Win:  ${avg_win:+.4f}")
        print(f"  Losses: {len(losses)} | Avg Loss: ${avg_loss:+.4f}")
        if avg_loss != 0:
            print(f"  Risk/Reward:        1:{abs(avg_win / avg_loss):.2f}")
        print(f"  Total Comisiones:   ${total_commission:.4f}")
        print(f"  Trades con DCA:     {len(dca_trades)} ({len(dca_trades) / len(trades) * 100:.1f}%)")
        print(f"  Max Perdidas Seg.:  {max_consecutive_losses}")


def main():
    print("=" * 65)
    print("  BACKTEST v6.7 SMART METRALLADORA - COMPARACION")
    print("=" * 65)
    print(f"  TP: {TAKE_PROFIT_PCT * 100:.1f}% | SL: {STOP_LOSS_CATASTROPHIC * 100:.1f}% | Leverage: {LEVERAGE}x")
    print(f"  Protecciones: ATR regime [{ATR_REGIME_MULT_LOW}x - {ATR_REGIME_MULT_HIGH}x]")
    print(f"                Rolling WR min: {ROLLING_WR_MIN * 100}% (ventana: {ROLLING_WR_WINDOW})")
    print(f"                Max consecutive losses: {MAX_CONSECUTIVE_LOSSES}")

    # Sin protecciones
    trades_off, curve_off, dd_off, _ = run_backtest(use_protections=False)
    balance_off = curve_off[-1]
    print_results("SIN PROTECCIONES", trades_off, dd_off, 0, balance_off)

    # Con protecciones
    trades_on, curve_on, dd_on, filtered = run_backtest(use_protections=True)
    balance_on = curve_on[-1]
    print_results("CON PROTECCIONES", trades_on, dd_on, filtered, balance_on)

    # Comparacion
    print(f"\n--- Comparacion ---")
    dd_improvement = ((dd_off - dd_on) / dd_off * 100) if dd_off > 0 else 0
    print(f"  Drawdown reducido:  {dd_improvement:+.1f}%")
    print(f"  Trades evitados:    {len(trades_off) - len(trades_on)}")
    print(f"  Senales filtradas:  {filtered}")
    print("=" * 65 + "\n")

    return trades_on, curve_on


if __name__ == "__main__":
    main()
