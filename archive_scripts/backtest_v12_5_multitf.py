"""
Backtest V12.5 - Multi-Timeframe (4h + 1D)
==========================================
Concepto: Solo operar cuando 4h y 1D estan alineados.

Logica:
- 1D bullish (close > EMA20): solo permitir LONGs en 4h
- 1D bearish (close < EMA20): solo permitir SHORTs en 4h
- Neutro/conflicto: no operar

IMPORTANTE: Usar .shift(1) en datos 1D para evitar look-ahead bias
(solo podemos conocer el cierre del DIA ANTERIOR, no el actual)
"""

import json
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'
MODELS_DIR = Path(__file__).parent / 'models'

INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.02

# Filtros V12 base (probados y estables)
MIN_CONVICTION = 2.0
RSI_MIN, RSI_MAX = 38, 72
BB_MIN, BB_MAX = 0.2, 0.8
MAX_CHOP = 52

ACTIVE_PAIRS = ['ADA/USDT', 'XRP/USDT', 'DOT/USDT', 'DOGE/USDT',
                'ETH/USDT', 'AVAX/USDT', 'NEAR/USDT', 'LINK/USDT']


def load_data_4h(pair):
    """Cargar datos 4h."""
    safe = pair.replace('/', '_')
    for suffix in ['_4h_v95.parquet', '_4h_history.parquet']:
        cache = DATA_DIR / f'{safe}{suffix}'
        if cache.exists():
            df = pd.read_parquet(cache)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            return df
    return None


def load_data_1d(pair):
    """Cargar datos 1D."""
    safe = pair.replace('/', '_')
    cache = DATA_DIR / f'{safe}_1d_history.parquet'
    if cache.exists():
        df = pd.read_parquet(cache)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        return df
    return None


def compute_daily_trend(df_1d):
    """
    Calcular tendencia diaria basada en EMA20.

    Retorna DataFrame con:
    - daily_trend: 1 (bullish), -1 (bearish), 0 (neutral)
    - ema20_1d: valor de EMA20

    IMPORTANTE: Usar .shift(1) para evitar look-ahead bias
    """
    c = df_1d['close']
    ema20 = ta.ema(c, length=20)
    ema50 = ta.ema(c, length=50)

    # Tendencia: close vs EMA20
    # Bullish: close > EMA20 AND EMA20 > EMA50
    # Bearish: close < EMA20 AND EMA20 < EMA50
    # Neutral: conflicto o sin datos

    trend = pd.Series(0, index=df_1d.index)

    bullish = (c > ema20) & (ema20 > ema50)
    bearish = (c < ema20) & (ema20 < ema50)

    trend[bullish] = 1
    trend[bearish] = -1

    # CRUCIAL: shift(1) para usar el cierre del dia ANTERIOR
    # Esto evita look-ahead bias
    trend_shifted = trend.shift(1)
    ema20_shifted = ema20.shift(1)

    result = pd.DataFrame({
        'daily_trend': trend_shifted,
        'ema20_1d': ema20_shifted,
    }, index=df_1d.index)

    return result


def merge_4h_with_daily(df_4h, daily_trend):
    """
    Merge datos 4h con tendencia diaria.
    Cada vela 4h recibe la tendencia del dia ANTERIOR.
    """
    # Crear columna de fecha sin hora
    df_4h = df_4h.copy()
    df_4h['date'] = df_4h.index.normalize()

    daily_trend = daily_trend.copy()
    daily_trend['date'] = daily_trend.index.normalize()

    # Merge por fecha
    merged = df_4h.merge(
        daily_trend[['date', 'daily_trend', 'ema20_1d']],
        on='date',
        how='left'
    )
    merged.index = df_4h.index

    # Llenar NaN con 0 (neutral)
    merged['daily_trend'] = merged['daily_trend'].fillna(0)

    return merged


def compute_features(df):
    """Calcular features para el modelo ML."""
    feat = pd.DataFrame(index=df.index)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    for p in [1, 3, 5, 10, 20]:
        feat[f'ret_{p}'] = c.pct_change(p)

    feat['atr14'] = ta.atr(h, l, c, length=14)
    feat['atr_r'] = feat['atr14'] / feat['atr14'].rolling(50).mean()
    feat['vol5'] = c.pct_change().rolling(5).std()
    feat['vol20'] = c.pct_change().rolling(20).std()
    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)

    sr = ta.stochrsi(c, length=14, rsi_length=14, k=3, d=3)
    if sr is not None:
        feat['srsi_k'] = sr.iloc[:, 0]
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        feat['macd_h'] = macd.iloc[:, 1]

    feat['roc5'] = ta.roc(c, length=5)
    feat['roc20'] = ta.roc(c, length=20)

    for el in [8, 21, 55, 100, 200]:
        e = ta.ema(c, length=el)
        feat[f'ema{el}_d'] = (c - e) / e * 100
    feat['ema8_sl'] = ta.ema(c, length=8).pct_change(3) * 100
    feat['ema55_sl'] = ta.ema(c, length=55).pct_change(5) * 100

    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        bw = bb.iloc[:, 2] - bb.iloc[:, 0]
        feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw
        feat['bb_w'] = bw / bb.iloc[:, 1] * 100

    feat['vr'] = v / v.rolling(20).mean()
    feat['spr'] = (h - l) / c * 100
    feat['body'] = abs(c - df['open']) / (h - l + 1e-10)

    ax = ta.adx(h, l, c, length=14)
    if ax is not None:
        feat['adx'] = ax.iloc[:, 0]
        feat['dip'] = ax.iloc[:, 1]
        feat['dim'] = ax.iloc[:, 2]

    hr = df.index.hour
    dw = df.index.dayofweek
    feat['h_s'] = np.sin(2 * np.pi * hr / 24)
    feat['h_c'] = np.cos(2 * np.pi * hr / 24)
    feat['d_s'] = np.sin(2 * np.pi * dw / 7)
    feat['d_c'] = np.cos(2 * np.pi * dw / 7)

    atr_1 = ta.atr(h, l, c, length=1)
    atr_sum = atr_1.rolling(14).sum()
    high_max = h.rolling(14).max()
    low_min = l.rolling(14).min()
    feat['chop'] = 100 * np.log10(atr_sum / (high_max - low_min + 1e-10)) / np.log10(14)

    return feat


def run_backtest_v12_5(pair_data_4h, pair_data_1d, models, start_date, end_date):
    """Backtest V12.5 con filtro Multi-Timeframe."""
    trades = []
    mtf_stats = {
        'filtered_by_mtf': 0,
        'passed_mtf': 0,
        'bullish_days': 0,
        'bearish_days': 0,
        'neutral_days': 0,
    }

    for pair in ACTIVE_PAIRS:
        if pair not in pair_data_4h or pair not in pair_data_1d:
            continue

        df_4h = pair_data_4h[pair]
        df_1d = pair_data_1d[pair]

        # Calcular tendencia diaria
        daily_trend = compute_daily_trend(df_1d)

        # Merge 4h con tendencia diaria
        df = merge_4h_with_daily(df_4h, daily_trend)

        df_period = df[(df.index >= start_date) & (df.index < end_date)]
        if len(df_period) < 250:
            continue

        safe = pair.replace('/', '')
        model = models.get(safe)
        if model is None:
            continue

        feat = compute_features(df_4h)
        feat = feat[(feat.index >= start_date) & (feat.index < end_date)]
        fcols = [c for c in model.feature_name_ if c in feat.columns]

        balance = INITIAL_CAPITAL
        pos = None

        for i in range(250, len(df_period)):
            ts = df_period.index[i]
            if ts not in feat.index:
                continue
            price = df_period.iloc[i]['close']

            # Cerrar posicion existente
            if pos is not None:
                pnl_pct = (price - pos['entry']) / pos['entry'] * pos['dir']
                hit_tp = pnl_pct >= pos['tp_pct']
                hit_sl = pnl_pct <= -pos['sl_pct']
                timeout = (i - pos['bar']) >= 20

                if hit_tp or hit_sl or timeout:
                    pnl = pos['size'] * pos['dir'] * (price - pos['entry'])
                    balance += pnl
                    trades.append({
                        'pair': pair,
                        'pnl': pnl,
                        'daily_trend': pos.get('daily_trend', 0),
                        'signal_dir': pos['dir']
                    })
                    pos = None

            # Abrir nueva posicion
            if pos is None:
                # Obtener tendencia diaria (ya shifteada = dia anterior)
                daily_t = df_period.iloc[i].get('daily_trend', 0)

                # Contar estadisticas
                if daily_t == 1:
                    mtf_stats['bullish_days'] += 1
                elif daily_t == -1:
                    mtf_stats['bearish_days'] += 1
                else:
                    mtf_stats['neutral_days'] += 1

                # Obtener features
                rsi = feat.loc[ts, 'rsi14'] if 'rsi14' in feat.columns else 50
                bb_pos_val = feat.loc[ts, 'bb_pos'] if 'bb_pos' in feat.columns else 0.5
                chop = feat.loc[ts, 'chop'] if 'chop' in feat.columns else 50
                atr = feat.loc[ts, 'atr14'] if 'atr14' in feat.columns else price * 0.02

                X = feat.loc[ts:ts][fcols]
                if X.isna().any().any():
                    continue

                pred = model.predict(X)[0]
                sig = 1 if pred > 0.5 else -1
                conviction = abs(pred - 0.5) * 10

                # Filtros V12 base
                if conviction < MIN_CONVICTION:
                    continue
                if not (RSI_MIN <= rsi <= RSI_MAX):
                    continue
                if not (BB_MIN <= bb_pos_val <= BB_MAX):
                    continue
                if chop > MAX_CHOP:
                    continue

                # === FILTRO MULTI-TIMEFRAME V12.5 ===
                # Solo LONG si 1D bullish, solo SHORT si 1D bearish
                mtf_pass = False

                if daily_t == 1 and sig == 1:
                    # 1D bullish + 4h LONG = OK
                    mtf_pass = True
                elif daily_t == -1 and sig == -1:
                    # 1D bearish + 4h SHORT = OK
                    mtf_pass = True
                elif daily_t == 0:
                    # Neutral: permitir cualquier direccion (no penalizar)
                    mtf_pass = True

                if not mtf_pass:
                    mtf_stats['filtered_by_mtf'] += 1
                    continue

                mtf_stats['passed_mtf'] += 1

                tp_pct = atr / price * 2.0
                sl_pct = atr / price * 1.0
                risk_amt = balance * RISK_PER_TRADE
                size = risk_amt / (sl_pct * price) if sl_pct > 0 else 0

                pos = {
                    'entry': price, 'dir': sig, 'size': size,
                    'tp_pct': tp_pct, 'sl_pct': sl_pct, 'bar': i,
                    'daily_trend': daily_t
                }

        # Cerrar posicion abierta al final
        if pos:
            pnl = pos['size'] * pos['dir'] * (df_period.iloc[-1]['close'] - pos['entry'])
            trades.append({
                'pair': pair,
                'pnl': pnl,
                'daily_trend': pos.get('daily_trend', 0),
                'signal_dir': pos['dir']
            })

    return trades, mtf_stats


def run_backtest_v12_base(pair_data_4h, models, start_date, end_date):
    """Backtest V12 base sin filtro MTF (para comparacion)."""
    trades = []

    for pair in ACTIVE_PAIRS:
        if pair not in pair_data_4h:
            continue

        df = pair_data_4h[pair]
        df_period = df[(df.index >= start_date) & (df.index < end_date)]
        if len(df_period) < 250:
            continue

        safe = pair.replace('/', '')
        model = models.get(safe)
        if model is None:
            continue

        feat = compute_features(df)
        feat = feat[(feat.index >= start_date) & (feat.index < end_date)]
        fcols = [c for c in model.feature_name_ if c in feat.columns]

        balance = INITIAL_CAPITAL
        pos = None

        for i in range(250, len(df_period)):
            ts = df_period.index[i]
            if ts not in feat.index:
                continue
            price = df_period.iloc[i]['close']

            if pos is not None:
                pnl_pct = (price - pos['entry']) / pos['entry'] * pos['dir']
                hit_tp = pnl_pct >= pos['tp_pct']
                hit_sl = pnl_pct <= -pos['sl_pct']
                timeout = (i - pos['bar']) >= 20

                if hit_tp or hit_sl or timeout:
                    pnl = pos['size'] * pos['dir'] * (price - pos['entry'])
                    balance += pnl
                    trades.append({'pair': pair, 'pnl': pnl})
                    pos = None

            if pos is None:
                rsi = feat.loc[ts, 'rsi14'] if 'rsi14' in feat.columns else 50
                bb_pos_val = feat.loc[ts, 'bb_pos'] if 'bb_pos' in feat.columns else 0.5
                chop = feat.loc[ts, 'chop'] if 'chop' in feat.columns else 50
                atr = feat.loc[ts, 'atr14'] if 'atr14' in feat.columns else price * 0.02

                X = feat.loc[ts:ts][fcols]
                if X.isna().any().any():
                    continue

                pred = model.predict(X)[0]
                sig = 1 if pred > 0.5 else -1
                conviction = abs(pred - 0.5) * 10

                if conviction < MIN_CONVICTION:
                    continue
                if not (RSI_MIN <= rsi <= RSI_MAX):
                    continue
                if not (BB_MIN <= bb_pos_val <= BB_MAX):
                    continue
                if chop > MAX_CHOP:
                    continue

                tp_pct = atr / price * 2.0
                sl_pct = atr / price * 1.0
                risk_amt = balance * RISK_PER_TRADE
                size = risk_amt / (sl_pct * price) if sl_pct > 0 else 0

                pos = {'entry': price, 'dir': sig, 'size': size,
                       'tp_pct': tp_pct, 'sl_pct': sl_pct, 'bar': i}

        if pos:
            pnl = pos['size'] * pos['dir'] * (df_period.iloc[-1]['close'] - pos['entry'])
            trades.append({'pair': pair, 'pnl': pnl})

    return trades


def compute_metrics(trades):
    if not trades:
        return {'trades': 0, 'wins': 0, 'wr': 0, 'pnl': 0, 'pf': 0}
    wins = sum(1 for t in trades if t['pnl'] > 0)
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99
    return {
        'trades': len(trades),
        'wins': wins,
        'wr': wins / len(trades) * 100 if trades else 0,
        'pnl': sum(t['pnl'] for t in trades),
        'pf': pf,
    }


def main():
    print("="*70)
    print("BACKTEST V12.5 - MULTI-TIMEFRAME (4h + 1D)")
    print("="*70)

    print("\nConcepto:")
    print("- Usar tendencia 1D como filtro direccional para senales 4h")
    print("- 1D bullish (close > EMA20 > EMA50): solo permitir LONGs")
    print("- 1D bearish (close < EMA20 < EMA50): solo permitir SHORTs")
    print("- 1D neutral: permitir ambas direcciones")
    print("\nIMPORTANTE: Datos 1D shifteados 1 dia para evitar look-ahead bias")

    print("\nFiltros V12 base (sin cambios):")
    print(f"- Conviction >= {MIN_CONVICTION}")
    print(f"- RSI: {RSI_MIN}-{RSI_MAX}")
    print(f"- BB pos: {BB_MIN}-{BB_MAX}")
    print(f"- Chop < {MAX_CHOP}")

    # Cargar modelos
    models = {}
    pair_data_4h = {}
    pair_data_1d = {}

    for pair in ACTIVE_PAIRS:
        safe = pair.replace('/', '')
        try:
            models[safe] = joblib.load(MODELS_DIR / f'v95_v7_{safe}USDT.pkl')
        except:
            try:
                models[safe] = joblib.load(MODELS_DIR / f'v7_{pair.replace("/", "_")}.pkl')
            except:
                continue

        df_4h = load_data_4h(pair)
        df_1d = load_data_1d(pair)

        if df_4h is not None:
            pair_data_4h[pair] = df_4h
        if df_1d is not None:
            pair_data_1d[pair] = df_1d

    print(f"\nModelos cargados: {len(models)}")
    print(f"Pares con datos 4h: {len(pair_data_4h)}")
    print(f"Pares con datos 1D: {len(pair_data_1d)}")

    periods = [
        ('Ultimo Ano', '2024-02-01', '2025-02-24'),
        ('Bear Market 2022', '2022-01-01', '2023-01-01'),
    ]

    all_v12 = []
    all_v12_5 = []

    for period_name, start, end in periods:
        print(f"\n{'='*70}")
        print(f"PERIODO: {period_name}")
        print("="*70)

        # V12 Base
        trades_v12 = run_backtest_v12_base(pair_data_4h, models, start, end)
        m_v12 = compute_metrics(trades_v12)
        all_v12.append(m_v12)

        # V12.5 Multi-Timeframe
        trades_v12_5, mtf_stats = run_backtest_v12_5(
            pair_data_4h, pair_data_1d, models, start, end
        )
        m_v12_5 = compute_metrics(trades_v12_5)
        all_v12_5.append(m_v12_5)

        wr_diff = m_v12_5['wr'] - m_v12['wr']
        wr_mark = ">> MEJOR" if wr_diff > 1 else ("<< PEOR" if wr_diff < -1 else "~igual")

        print(f"\n{'Metrica':<12} {'V12 Base':<15} {'V12.5 MTF':<15} {'Diferencia':<12}")
        print("-"*55)
        print(f"{'Trades':<12} {m_v12['trades']:<15} {m_v12_5['trades']:<15} {m_v12_5['trades'] - m_v12['trades']:+d}")
        print(f"{'Win Rate':<12} {m_v12['wr']:<14.1f}% {m_v12_5['wr']:<14.1f}% {wr_diff:+.1f}% {wr_mark}")
        print(f"{'PnL':<12} ${m_v12['pnl']:<13,.0f} ${m_v12_5['pnl']:<13,.0f} ${m_v12_5['pnl'] - m_v12['pnl']:+,.0f}")
        print(f"{'P. Factor':<12} {m_v12['pf']:<15.2f} {m_v12_5['pf']:<15.2f} {m_v12_5['pf'] - m_v12['pf']:+.2f}")

        # Estadisticas MTF
        total_signals = mtf_stats['passed_mtf'] + mtf_stats['filtered_by_mtf']
        if total_signals > 0:
            filter_rate = mtf_stats['filtered_by_mtf'] / total_signals * 100
            print(f"\nEstadisticas MTF:")
            print(f"  Senales filtradas por MTF: {mtf_stats['filtered_by_mtf']} ({filter_rate:.1f}%)")
            print(f"  Senales que pasaron MTF: {mtf_stats['passed_mtf']}")

        # Desglose por alineacion
        if trades_v12_5:
            aligned_trades = [t for t in trades_v12_5 if t['daily_trend'] != 0]
            neutral_trades = [t for t in trades_v12_5 if t['daily_trend'] == 0]

            if aligned_trades:
                wins_aligned = sum(1 for t in aligned_trades if t['pnl'] > 0)
                wr_aligned = wins_aligned / len(aligned_trades) * 100
                pnl_aligned = sum(t['pnl'] for t in aligned_trades)
                print(f"\n  Trades ALINEADOS (4h=1D): {len(aligned_trades)}, WR {wr_aligned:.1f}%, PnL ${pnl_aligned:,.0f}")

            if neutral_trades:
                wins_neutral = sum(1 for t in neutral_trades if t['pnl'] > 0)
                wr_neutral = wins_neutral / len(neutral_trades) * 100
                pnl_neutral = sum(t['pnl'] for t in neutral_trades)
                print(f"  Trades NEUTROS (1D=0): {len(neutral_trades)}, WR {wr_neutral:.1f}%, PnL ${pnl_neutral:,.0f}")

    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)

    avg_wr_v12 = np.mean([r['wr'] for r in all_v12])
    avg_wr_v12_5 = np.mean([r['wr'] for r in all_v12_5])
    total_pnl_v12 = sum(r['pnl'] for r in all_v12)
    total_pnl_v12_5 = sum(r['pnl'] for r in all_v12_5)
    total_trades_v12 = sum(r['trades'] for r in all_v12)
    total_trades_v12_5 = sum(r['trades'] for r in all_v12_5)

    print(f"\n{'Version':<20} {'Trades':<10} {'WR':<12} {'PnL':<12}")
    print("-"*55)
    print(f"{'V12 Base':<20} {total_trades_v12:<10} {avg_wr_v12:<11.1f}% ${total_pnl_v12:<10,.0f}")
    print(f"{'V12.5 MTF':<20} {total_trades_v12_5:<10} {avg_wr_v12_5:<11.1f}% ${total_pnl_v12_5:<10,.0f}")
    print("-"*55)
    print(f"{'Diferencia':<20} {total_trades_v12_5 - total_trades_v12:+d}        {avg_wr_v12_5 - avg_wr_v12:+.1f}%       ${total_pnl_v12_5 - total_pnl_v12:+,.0f}")

    # Veredicto
    if avg_wr_v12_5 > avg_wr_v12 and total_pnl_v12_5 > total_pnl_v12:
        print("\n*** V12.5 MEJORA AMBAS METRICAS ***")
        result = "EXITO"
    elif avg_wr_v12_5 > avg_wr_v12:
        print("\n** V12.5 mejora WR pero reduce PnL **")
        result = "PARCIAL_WR"
    elif total_pnl_v12_5 > total_pnl_v12:
        print("\n** V12.5 mejora PnL pero reduce WR **")
        result = "PARCIAL_PNL"
    else:
        print("\n!! V12.5 no mejora !!")
        result = "FALLO"

    # Guardar config
    config = {
        'version': 'V12.5',
        'description': 'Multi-Timeframe (4h + 1D alignment)',
        'result': result,
        'mtf_logic': {
            '1D_bullish': 'close > EMA20 > EMA50 -> only LONGs',
            '1D_bearish': 'close < EMA20 < EMA50 -> only SHORTs',
            '1D_neutral': 'allow both directions',
            'lookahead_prevention': 'shift(1) on daily data'
        },
        'filters': {
            'min_conviction': MIN_CONVICTION,
            'rsi_range': [RSI_MIN, RSI_MAX],
            'bb_range': [BB_MIN, BB_MAX],
            'max_chop': MAX_CHOP,
        },
        'metrics': {
            'base_wr': avg_wr_v12,
            'mtf_wr': avg_wr_v12_5,
            'base_pnl': total_pnl_v12,
            'mtf_pnl': total_pnl_v12_5,
        },
    }

    with open(MODELS_DIR / 'v12_5_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("\nConfiguracion guardada en v12_5_config.json")


if __name__ == '__main__':
    main()
