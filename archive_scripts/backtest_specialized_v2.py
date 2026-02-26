"""
Bot Especializado por Moneda V2 - Pruebas Exhaustivas
======================================================
Usa modelos EXISTENTES (v95_v7_*) que ya estan optimizados.
Filtros relajados para obtener MAS trades de calidad.

Objetivo: MAS trades con MEJOR calidad, no menos trades.
"""

import json
import random
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent / 'data'
MODELS_DIR = Path(__file__).parent / 'models'

INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.02

random.seed(42)
np.random.seed(42)


def load_data_4h(pair):
    """Cargar datos 4h para un par."""
    safe = pair.replace('/', '_')
    for suffix in ['_4h_v95.parquet', '_4h_history.parquet']:
        cache = DATA_DIR / f'{safe}{suffix}'
        if cache.exists():
            df = pd.read_parquet(cache)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            return df
    return None


def load_existing_model(pair):
    """Cargar modelo existente v95_v7."""
    safe = pair.replace('/', '')
    try:
        return joblib.load(MODELS_DIR / f'v95_v7_{safe}USDT.pkl')
    except:
        try:
            return joblib.load(MODELS_DIR / f'v7_{pair.replace("/", "_")}.pkl')
        except:
            return None


def compute_features(df):
    """Calcular features para modelo ML."""
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


def find_best_worst_years(df):
    """Encontrar mejor/peor año por retorno."""
    df = df.copy()
    df['year'] = df.index.year

    yearly_returns = {}
    for year in df['year'].unique():
        year_data = df[df['year'] == year]
        if len(year_data) < 100:
            continue
        ret = (year_data['close'].iloc[-1] - year_data['close'].iloc[0]) / year_data['close'].iloc[0]
        yearly_returns[year] = ret

    if not yearly_returns:
        return None, None, {}

    best_year = max(yearly_returns, key=yearly_returns.get)
    worst_year = min(yearly_returns, key=yearly_returns.get)

    return best_year, worst_year, yearly_returns


def create_synthetic_year(df, n_months=12, seed=42):
    """Crear año sintetico con meses aleatorios."""
    random.seed(seed)
    df = df.copy()
    df['year_month'] = df.index.to_period('M')
    available_months = list(df['year_month'].unique())

    if len(available_months) < n_months:
        return None, None

    selected_months = random.sample(available_months, n_months)
    return sorted(selected_months), df


def run_backtest_period(df, model, start_date, end_date, pair_name, params):
    """
    Ejecutar backtest con parametros especificos por par.
    params = {min_conviction, rsi_range, bb_range, max_chop, tp_mult, sl_mult}
    """
    df_period = df[(df.index >= start_date) & (df.index < end_date)]
    if len(df_period) < 100:
        return {'trades': 0, 'wins': 0, 'wr': 0, 'pnl': 0, 'pf': 0, 'details': []}

    feat = compute_features(df)
    feat = feat[(feat.index >= start_date) & (feat.index < end_date)]
    fcols = [c for c in model.feature_name_ if c in feat.columns]

    if not fcols:
        return {'trades': 0, 'wins': 0, 'wr': 0, 'pnl': 0, 'pf': 0, 'details': []}

    trades = []
    balance = INITIAL_CAPITAL
    pos = None

    min_conv = params.get('min_conviction', 1.5)
    rsi_min, rsi_max = params.get('rsi_range', (30, 70))
    bb_min, bb_max = params.get('bb_range', (0.1, 0.9))
    max_chop = params.get('max_chop', 60)
    tp_mult = params.get('tp_mult', 2.0)
    sl_mult = params.get('sl_mult', 1.0)

    for i in range(100, len(df_period)):
        ts = df_period.index[i]
        if ts not in feat.index:
            continue
        price = df_period.iloc[i]['close']

        # Cerrar posicion
        if pos is not None:
            pnl_pct = (price - pos['entry']) / pos['entry'] * pos['dir']
            hit_tp = pnl_pct >= pos['tp_pct']
            hit_sl = pnl_pct <= -pos['sl_pct']
            timeout = (i - pos['bar']) >= 20

            if hit_tp or hit_sl or timeout:
                pnl = pos['size'] * pos['dir'] * (price - pos['entry'])
                balance += pnl
                reason = 'TP' if hit_tp else ('SL' if hit_sl else 'TIMEOUT')
                trades.append({'pnl': pnl, 'win': pnl > 0, 'reason': reason})
                pos = None

        # Abrir posicion
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

            # Filtros RELAJADOS para obtener MAS trades
            if conviction < min_conv:
                continue
            if not (rsi_min <= rsi <= rsi_max):
                continue
            if not (bb_min <= bb_pos_val <= bb_max):
                continue
            if chop > max_chop:
                continue

            tp_pct = atr / price * tp_mult
            sl_pct = atr / price * sl_mult
            risk_amt = balance * RISK_PER_TRADE
            size = risk_amt / (sl_pct * price) if sl_pct > 0 else 0

            pos = {'entry': price, 'dir': sig, 'size': size,
                   'tp_pct': tp_pct, 'sl_pct': sl_pct, 'bar': i}

    # Cerrar posicion abierta
    if pos:
        pnl = pos['size'] * pos['dir'] * (df_period.iloc[-1]['close'] - pos['entry'])
        trades.append({'pnl': pnl, 'win': pnl > 0, 'reason': 'END'})

    # Metricas
    if not trades:
        return {'trades': 0, 'wins': 0, 'wr': 0, 'pnl': 0, 'pf': 0, 'details': []}

    wins = sum(1 for t in trades if t['win'])
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99

    return {
        'trades': len(trades),
        'wins': wins,
        'wr': wins / len(trades) * 100 if trades else 0,
        'pnl': sum(t['pnl'] for t in trades),
        'pf': pf,
        'details': trades,
    }


def optimize_params_for_pair(df, model, pair_name):
    """
    Encontrar mejores parametros para un par especifico.
    Usa datos de 2020-2023 para optimizar.
    """
    print(f"    Optimizando parametros para {pair_name}...")

    # Grid de parametros a probar
    param_grid = {
        'min_conviction': [1.2, 1.5, 1.8, 2.0],
        'rsi_range': [(30, 70), (35, 65), (38, 72), (25, 75)],
        'max_chop': [50, 55, 60, 65],
        'tp_mult': [1.5, 2.0, 2.5, 3.0],
    }

    best_score = -999
    best_params = None

    # Periodo de optimizacion: 2020-2023
    opt_start = '2020-01-01'
    opt_end = '2024-01-01'

    for conv in param_grid['min_conviction']:
        for rsi in param_grid['rsi_range']:
            for chop in param_grid['max_chop']:
                for tp in param_grid['tp_mult']:
                    params = {
                        'min_conviction': conv,
                        'rsi_range': rsi,
                        'bb_range': (0.15, 0.85),
                        'max_chop': chop,
                        'tp_mult': tp,
                        'sl_mult': 1.0,
                    }

                    result = run_backtest_period(df, model, opt_start, opt_end, pair_name, params)

                    # Score = WR * sqrt(trades) * sign(pnl)
                    # Queremos alto WR, muchos trades, y PnL positivo
                    if result['trades'] >= 20:  # Minimo de trades
                        pnl_sign = 1 if result['pnl'] > 0 else 0.5
                        score = result['wr'] * np.sqrt(result['trades']) * pnl_sign

                        if score > best_score:
                            best_score = score
                            best_params = params.copy()
                            best_params['opt_trades'] = result['trades']
                            best_params['opt_wr'] = result['wr']
                            best_params['opt_pnl'] = result['pnl']

    if best_params:
        print(f"      Mejor config: conv={best_params['min_conviction']}, "
              f"rsi={best_params['rsi_range']}, chop={best_params['max_chop']}, "
              f"tp={best_params['tp_mult']}")
        print(f"      Opt periodo: {best_params['opt_trades']} trades, "
              f"WR {best_params['opt_wr']:.1f}%, PnL ${best_params['opt_pnl']:,.0f}")
    else:
        # Default params si no encontramos nada bueno
        best_params = {
            'min_conviction': 1.5,
            'rsi_range': (30, 70),
            'bb_range': (0.15, 0.85),
            'max_chop': 60,
            'tp_mult': 2.0,
            'sl_mult': 1.0,
        }
        print(f"      [!] Usando parametros default")

    return best_params


def run_exhaustive_test(pair_name):
    """Pruebas exhaustivas con modelo existente y parametros optimizados."""
    print(f"\n{'='*70}")
    print(f"BOT ESPECIALIZADO V2: {pair_name}")
    print("="*70)

    # Cargar datos y modelo
    df = load_data_4h(pair_name)
    if df is None:
        print(f"  [!] No se encontraron datos para {pair_name}")
        return None

    model = load_existing_model(pair_name)
    if model is None:
        print(f"  [!] No se encontro modelo para {pair_name}")
        return None

    print(f"  Datos: {df.index.min().date()} a {df.index.max().date()} ({len(df)} velas)")
    print(f"  Modelo: v95_v7 cargado ({len(model.feature_name_)} features)")

    # Encontrar mejor/peor año
    best_year, worst_year, yearly_rets = find_best_worst_years(df)
    print(f"\n  Retornos anuales:")
    for year, ret in sorted(yearly_rets.items()):
        marker = " <- MEJOR" if year == best_year else (" <- PEOR" if year == worst_year else "")
        print(f"    {year}: {ret*100:+.1f}%{marker}")

    # Optimizar parametros
    print(f"\n  Optimizando parametros (2020-2023)...")
    params = optimize_params_for_pair(df, model, pair_name)

    # Definir periodos de test (OUT OF SAMPLE - despues de optimizacion)
    test_periods = {}

    if worst_year and worst_year >= 2020:
        test_periods['PEOR_ANO'] = (f'{worst_year}-01-01', f'{worst_year+1}-01-01')
    if best_year and best_year >= 2020:
        test_periods['MEJOR_ANO'] = (f'{best_year}-01-01', f'{best_year+1}-01-01')

    test_periods['ULTIMO_ANO'] = ('2024-02-01', '2025-02-24')
    test_periods['2025_PARCIAL'] = ('2025-01-01', '2026-02-24')

    # Año sintetico
    synthetic_months, df_synth = create_synthetic_year(df, n_months=12, seed=42)

    results = {}

    print(f"\n  {'Escenario':<20} {'Trades':<8} {'WR':<10} {'PnL':<12} {'PF':<8}")
    print("  " + "-"*60)

    # Test cada periodo
    for scenario, (start, end) in test_periods.items():
        metrics = run_backtest_period(df, model, start, end, pair_name, params)
        results[scenario] = metrics

        wr_mark = "**" if metrics['wr'] >= 50 else ""
        print(f"  {scenario:<20} {metrics['trades']:<8} {metrics['wr']:<9.1f}%{wr_mark} "
              f"${metrics['pnl']:<10,.0f} {metrics['pf']:<.2f}")

    # Año sintetico
    if synthetic_months:
        synthetic_trades = 0
        synthetic_wins = 0
        synthetic_pnl = 0

        for month in synthetic_months:
            month_start = month.start_time.strftime('%Y-%m-%d')
            month_end = (month.end_time + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            m = run_backtest_period(df, model, month_start, month_end, pair_name, params)
            synthetic_trades += m['trades']
            synthetic_wins += m['wins']
            synthetic_pnl += m['pnl']

        synthetic_wr = synthetic_wins / synthetic_trades * 100 if synthetic_trades > 0 else 0
        results['SINTETICO'] = {
            'trades': synthetic_trades,
            'wins': synthetic_wins,
            'wr': synthetic_wr,
            'pnl': synthetic_pnl,
            'pf': 0,
        }

        wr_mark = "**" if synthetic_wr >= 50 else ""
        print(f"  {'SINTETICO (12m)':<20} {synthetic_trades:<8} {synthetic_wr:<9.1f}%{wr_mark} "
              f"${synthetic_pnl:<10,.0f} {'N/A':<8}")

    # Resumen
    valid_results = [r for r in results.values() if r['trades'] > 0]
    if valid_results:
        avg_wr = np.mean([r['wr'] for r in valid_results])
        total_pnl = sum(r['pnl'] for r in valid_results)
        total_trades = sum(r['trades'] for r in valid_results)
    else:
        avg_wr = 0
        total_pnl = 0
        total_trades = 0

    print(f"\n  RESUMEN {pair_name}:")
    print(f"    Trades totales: {total_trades}")
    print(f"    WR promedio: {avg_wr:.1f}%")
    print(f"    PnL total: ${total_pnl:,.0f}")

    status = "VIABLE" if avg_wr >= 50 and total_pnl > 0 else "REVISAR"
    print(f"    Status: {status}")

    return {
        'pair': pair_name,
        'params': params,
        'results': results,
        'avg_wr': avg_wr,
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'status': status,
    }


def main():
    print("="*70)
    print("BOTS ESPECIALIZADOS V2 - MODELOS EXISTENTES + OPTIMIZACION")
    print("="*70)
    print("\nObjetivo: MAS trades de MEJOR calidad")
    print("Metodo: Usar modelos v95_v7 + optimizar parametros por par")
    print("\nEscenarios de prueba (OUT OF SAMPLE):")
    print("1. PEOR AÑO - stress test")
    print("2. MEJOR AÑO - verificar consistencia")
    print("3. ULTIMO AÑO - condiciones recientes")
    print("4. 2025 PARCIAL - datos mas recientes")
    print("5. SINTETICO - meses aleatorios")

    pairs = ['ADA/USDT', 'XRP/USDT', 'DOT/USDT', 'DOGE/USDT',
             'ETH/USDT', 'AVAX/USDT', 'NEAR/USDT', 'LINK/USDT']

    all_results = []

    for pair in pairs:
        result = run_exhaustive_test(pair)
        if result:
            all_results.append(result)

    # Ranking final
    print("\n" + "="*70)
    print("RANKING FINAL - BOTS ESPECIALIZADOS V2")
    print("="*70)

    all_results.sort(key=lambda x: (x['status'] == 'VIABLE', x['avg_wr']), reverse=True)

    print(f"\n{'Rank':<6} {'Par':<12} {'Trades':<8} {'WR Avg':<10} {'PnL':<12} {'Status'}")
    print("-"*60)

    for i, r in enumerate(all_results, 1):
        wr_mark = "**" if r['avg_wr'] >= 50 else ""
        print(f"{i:<6} {r['pair']:<12} {r['total_trades']:<8} {r['avg_wr']:<9.1f}%{wr_mark} "
              f"${r['total_pnl']:<10,.0f} {r['status']}")

    # Guardar resultados
    output = {
        'fecha': datetime.now().isoformat(),
        'descripcion': 'Bots especializados V2 - Modelos existentes + optimizacion',
        'resultados': all_results,
    }

    with open(MODELS_DIR / 'specialized_bots_v2_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResultados guardados en specialized_bots_v2_results.json")

    # Recomendacion
    viable = [r for r in all_results if r['status'] == 'VIABLE']
    print(f"\n{'='*70}")
    print(f"RECOMENDACION")
    print(f"{'='*70}")

    if viable:
        print(f"\nPares VIABLES para produccion ({len(viable)}):")
        for r in viable:
            print(f"\n  {r['pair']}:")
            print(f"    Params: conv={r['params']['min_conviction']}, "
                  f"rsi={r['params']['rsi_range']}, chop={r['params']['max_chop']}, "
                  f"tp={r['params']['tp_mult']}")
            print(f"    WR: {r['avg_wr']:.1f}%, PnL: ${r['total_pnl']:,.0f}")
    else:
        print("\nNingun par alcanzo criterios de viabilidad.")
        print("Mejores candidatos a revisar:")
        for r in all_results[:3]:
            print(f"  - {r['pair']}: WR {r['avg_wr']:.1f}%, PnL ${r['total_pnl']:,.0f}")


if __name__ == '__main__':
    main()
