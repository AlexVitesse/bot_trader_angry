"""
Bot Especializado por Moneda - Pruebas Exhaustivas
===================================================
Cada moneda tiene su propio modelo optimizado.

Escenarios de prueba:
1. PEOR AÑO - stress test maximo
2. MEJOR AÑO - verificar que no sobre-optimizamos
3. AÑO SINTETICO - meses aleatorios (simula variabilidad)
4. ULTIMO AÑO - condiciones recientes

Objetivo: Mas trades de MEJOR calidad, no menos trades.
"""

import json
import random
import numpy as np
import pandas as pd
import pandas_ta as ta
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit

DATA_DIR = Path(__file__).parent / 'data'
MODELS_DIR = Path(__file__).parent / 'models'

INITIAL_CAPITAL = 500
RISK_PER_TRADE = 0.02

# Semilla para reproducibilidad en año sintetico
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


def compute_features(df):
    """Calcular features para modelo ML."""
    feat = pd.DataFrame(index=df.index)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    # Returns
    for p in [1, 3, 5, 10, 20]:
        feat[f'ret_{p}'] = c.pct_change(p)

    # Volatilidad
    feat['atr14'] = ta.atr(h, l, c, length=14)
    feat['atr_r'] = feat['atr14'] / feat['atr14'].rolling(50).mean()
    feat['vol5'] = c.pct_change().rolling(5).std()
    feat['vol20'] = c.pct_change().rolling(20).std()

    # RSI
    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)

    # Stoch RSI
    sr = ta.stochrsi(c, length=14, rsi_length=14, k=3, d=3)
    if sr is not None:
        feat['srsi_k'] = sr.iloc[:, 0]

    # MACD
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        feat['macd_h'] = macd.iloc[:, 1]

    # ROC
    feat['roc5'] = ta.roc(c, length=5)
    feat['roc20'] = ta.roc(c, length=20)

    # EMAs
    for el in [8, 21, 55, 100, 200]:
        e = ta.ema(c, length=el)
        feat[f'ema{el}_d'] = (c - e) / e * 100
    feat['ema8_sl'] = ta.ema(c, length=8).pct_change(3) * 100
    feat['ema55_sl'] = ta.ema(c, length=55).pct_change(5) * 100

    # Bollinger
    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        bw = bb.iloc[:, 2] - bb.iloc[:, 0]
        feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw
        feat['bb_w'] = bw / bb.iloc[:, 1] * 100

    # Volume y Price Action
    feat['vr'] = v / v.rolling(20).mean()
    feat['spr'] = (h - l) / c * 100
    feat['body'] = abs(c - df['open']) / (h - l + 1e-10)

    # ADX
    ax = ta.adx(h, l, c, length=14)
    if ax is not None:
        feat['adx'] = ax.iloc[:, 0]
        feat['dip'] = ax.iloc[:, 1]
        feat['dim'] = ax.iloc[:, 2]

    # Hora y dia
    hr = df.index.hour
    dw = df.index.dayofweek
    feat['h_s'] = np.sin(2 * np.pi * hr / 24)
    feat['h_c'] = np.cos(2 * np.pi * hr / 24)
    feat['d_s'] = np.sin(2 * np.pi * dw / 7)
    feat['d_c'] = np.cos(2 * np.pi * dw / 7)

    # Choppiness
    atr_1 = ta.atr(h, l, c, length=1)
    atr_sum = atr_1.rolling(14).sum()
    high_max = h.rolling(14).max()
    low_min = l.rolling(14).min()
    feat['chop'] = 100 * np.log10(atr_sum / (high_max - low_min + 1e-10)) / np.log10(14)

    return feat


def create_labels(df, lookahead=20, tp_mult=2.0, sl_mult=1.0):
    """
    Crear labels basados en si el trade hubiera sido ganador.
    1 = trade ganador (hit TP antes que SL)
    0 = trade perdedor
    """
    c = df['close']
    atr = ta.atr(df['high'], df['low'], c, length=14)

    labels = []
    for i in range(len(df)):
        if i + lookahead >= len(df) or pd.isna(atr.iloc[i]):
            labels.append(np.nan)
            continue

        entry = c.iloc[i]
        atr_val = atr.iloc[i]
        tp_dist = atr_val * tp_mult
        sl_dist = atr_val * sl_mult

        # Simular LONG
        long_win = 0
        for j in range(i+1, min(i+lookahead+1, len(df))):
            high_j = df['high'].iloc[j]
            low_j = df['low'].iloc[j]

            # Check TP (high alcanza entry + tp_dist)
            if high_j >= entry + tp_dist:
                long_win = 1
                break
            # Check SL (low alcanza entry - sl_dist)
            if low_j <= entry - sl_dist:
                long_win = 0
                break

        # Simular SHORT
        short_win = 0
        for j in range(i+1, min(i+lookahead+1, len(df))):
            high_j = df['high'].iloc[j]
            low_j = df['low'].iloc[j]

            # Check TP (low alcanza entry - tp_dist)
            if low_j <= entry - tp_dist:
                short_win = 1
                break
            # Check SL (high alcanza entry + sl_dist)
            if high_j >= entry + sl_dist:
                short_win = 0
                break

        # Label = mejor de los dos
        labels.append(max(long_win, short_win))

    return pd.Series(labels, index=df.index)


def train_specialized_model(df, pair_name):
    """
    Entrenar modelo especializado para UN par.
    Usa walk-forward validation.
    """
    feat = compute_features(df)
    labels = create_labels(df)

    # Alinear y limpiar
    data = feat.copy()
    data['label'] = labels
    data = data.dropna()

    if len(data) < 1000:
        print(f"  [!] Datos insuficientes para {pair_name}: {len(data)}")
        return None, None

    X = data.drop('label', axis=1)
    y = data['label']

    # Walk-forward: entrenar en 70%, validar en 30%
    split_idx = int(len(X) * 0.7)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    # Entrenar LightGBM
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 200,
        'early_stopping_rounds': 20,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    # Evaluar
    val_pred = model.predict_proba(X_val)[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_val, val_pred)

    return model, auc


def find_best_worst_years(df):
    """
    Encontrar el mejor y peor año en los datos basado en retorno.
    """
    df = df.copy()
    df['year'] = df.index.year

    yearly_returns = {}
    for year in df['year'].unique():
        year_data = df[df['year'] == year]
        if len(year_data) < 100:  # Minimo de datos
            continue
        ret = (year_data['close'].iloc[-1] - year_data['close'].iloc[0]) / year_data['close'].iloc[0]
        yearly_returns[year] = ret

    if not yearly_returns:
        return None, None

    best_year = max(yearly_returns, key=yearly_returns.get)
    worst_year = min(yearly_returns, key=yearly_returns.get)

    return best_year, worst_year, yearly_returns


def create_synthetic_year(df, n_months=12, seed=42):
    """
    Crear un año sintetico mezclando meses aleatorios de diferentes años.
    Simula variabilidad del mercado.
    """
    random.seed(seed)

    df = df.copy()
    df['year_month'] = df.index.to_period('M')

    # Obtener todos los meses disponibles
    available_months = df['year_month'].unique()

    if len(available_months) < n_months:
        return None

    # Seleccionar meses aleatorios
    selected_months = random.sample(list(available_months), n_months)
    selected_months = sorted(selected_months)  # Ordenar cronologicamente

    # Concatenar datos de esos meses
    synthetic_data = []
    for month in selected_months:
        month_data = df[df['year_month'] == month].copy()
        synthetic_data.append(month_data)

    synthetic_df = pd.concat(synthetic_data)

    # Reindexar para que sea continuo
    synthetic_df = synthetic_df.drop('year_month', axis=1)

    return synthetic_df, selected_months


def run_backtest_period(df, model, start_date, end_date, pair_name,
                        min_conviction=2.0, rsi_range=(38, 72),
                        bb_range=(0.2, 0.8), max_chop=52):
    """
    Ejecutar backtest en un periodo especifico.
    """
    df_period = df[(df.index >= start_date) & (df.index < end_date)]
    if len(df_period) < 250:
        return {'trades': 0, 'wins': 0, 'wr': 0, 'pnl': 0, 'pf': 0}

    feat = compute_features(df)
    feat = feat[(feat.index >= start_date) & (feat.index < end_date)]
    fcols = [c for c in model.feature_name_ if c in feat.columns]

    trades = []
    balance = INITIAL_CAPITAL
    pos = None

    for i in range(250, len(df_period)):
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
                trades.append({'pnl': pnl, 'win': pnl > 0})
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
            prob = model.predict_proba(X)[0]
            confidence = max(prob)
            sig = 1 if pred == 1 else -1
            conviction = abs(prob[1] - 0.5) * 10

            # Filtros
            if conviction < min_conviction:
                continue
            if not (rsi_range[0] <= rsi <= rsi_range[1]):
                continue
            if not (bb_range[0] <= bb_pos_val <= bb_range[1]):
                continue
            if chop > max_chop:
                continue

            tp_pct = atr / price * 2.0
            sl_pct = atr / price * 1.0
            risk_amt = balance * RISK_PER_TRADE
            size = risk_amt / (sl_pct * price) if sl_pct > 0 else 0

            pos = {'entry': price, 'dir': sig, 'size': size,
                   'tp_pct': tp_pct, 'sl_pct': sl_pct, 'bar': i}

    # Cerrar posicion abierta
    if pos:
        pnl = pos['size'] * pos['dir'] * (df_period.iloc[-1]['close'] - pos['entry'])
        trades.append({'pnl': pnl, 'win': pnl > 0})

    # Calcular metricas
    if not trades:
        return {'trades': 0, 'wins': 0, 'wr': 0, 'pnl': 0, 'pf': 0}

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
    }


def run_exhaustive_test(pair_name):
    """
    Ejecutar pruebas exhaustivas para un par:
    1. Peor año
    2. Mejor año
    3. Año sintetico (meses aleatorios)
    4. Ultimo año
    """
    print(f"\n{'='*70}")
    print(f"BOT ESPECIALIZADO: {pair_name}")
    print("="*70)

    # Cargar datos
    df = load_data_4h(pair_name)
    if df is None:
        print(f"  [!] No se encontraron datos para {pair_name}")
        return None

    print(f"  Datos: {df.index.min().date()} a {df.index.max().date()} ({len(df)} velas)")

    # Encontrar mejor/peor año
    best_year, worst_year, yearly_rets = find_best_worst_years(df)
    print(f"\n  Retornos anuales del par:")
    for year, ret in sorted(yearly_rets.items()):
        marker = " <- MEJOR" if year == best_year else (" <- PEOR" if year == worst_year else "")
        print(f"    {year}: {ret*100:+.1f}%{marker}")

    # Entrenar modelo especializado (usando datos hasta 2023 para entrenar)
    print(f"\n  Entrenando modelo especializado...")
    train_end = '2024-01-01'
    df_train = df[df.index < train_end]
    model, auc = train_specialized_model(df_train, pair_name)

    if model is None:
        print(f"  [!] No se pudo entrenar modelo para {pair_name}")
        return None

    print(f"  Modelo entrenado - AUC validacion: {auc:.3f}")

    # Definir periodos de prueba
    test_periods = {
        'PEOR_ANO': (f'{worst_year}-01-01', f'{worst_year+1}-01-01'),
        'MEJOR_ANO': (f'{best_year}-01-01', f'{best_year+1}-01-01'),
        'ULTIMO_ANO': ('2024-02-01', '2025-02-24'),
    }

    # Crear año sintetico
    synthetic_df, selected_months = create_synthetic_year(df, n_months=12, seed=42)

    results = {}

    print(f"\n  {'Escenario':<20} {'Trades':<8} {'WR':<10} {'PnL':<12} {'PF':<8}")
    print("  " + "-"*60)

    # Probar cada periodo
    for scenario, (start, end) in test_periods.items():
        metrics = run_backtest_period(df, model, start, end, pair_name)
        results[scenario] = metrics

        wr_color = "**" if metrics['wr'] >= 50 else ""
        print(f"  {scenario:<20} {metrics['trades']:<8} {metrics['wr']:<9.1f}%{wr_color} ${metrics['pnl']:<10,.0f} {metrics['pf']:<.2f}")

    # Año sintetico (usar df completo pero con filtro de fechas del sintetico)
    if synthetic_df is not None:
        # Para el año sintetico, hacemos backtest en cada mes y sumamos
        synthetic_trades = 0
        synthetic_wins = 0
        synthetic_pnl = 0

        for month in selected_months:
            month_start = month.start_time.strftime('%Y-%m-%d')
            month_end = (month.end_time + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            m = run_backtest_period(df, model, month_start, month_end, pair_name)
            synthetic_trades += m['trades']
            synthetic_wins += m['wins']
            synthetic_pnl += m['pnl']

        synthetic_wr = synthetic_wins / synthetic_trades * 100 if synthetic_trades > 0 else 0
        results['SINTETICO'] = {
            'trades': synthetic_trades,
            'wins': synthetic_wins,
            'wr': synthetic_wr,
            'pnl': synthetic_pnl,
            'pf': 0,  # No calculamos PF para sintetico
            'months': [str(m) for m in selected_months],
        }

        wr_color = "**" if synthetic_wr >= 50 else ""
        print(f"  {'SINTETICO (12m)':<20} {synthetic_trades:<8} {synthetic_wr:<9.1f}%{wr_color} ${synthetic_pnl:<10,.0f} {'N/A':<8}")
        print(f"    Meses: {', '.join([str(m) for m in selected_months[:6]])}...")

    # Resumen
    print(f"\n  RESUMEN {pair_name}:")
    avg_wr = np.mean([r['wr'] for r in results.values() if r['trades'] > 0])
    total_pnl = sum(r['pnl'] for r in results.values())
    print(f"    WR promedio: {avg_wr:.1f}%")
    print(f"    PnL total (todos escenarios): ${total_pnl:,.0f}")

    if avg_wr >= 50:
        print(f"    >>> {pair_name} es VIABLE para bot especializado <<<")
    else:
        print(f"    [!] {pair_name} necesita mas optimizacion")

    return {
        'pair': pair_name,
        'model_auc': auc,
        'results': results,
        'avg_wr': avg_wr,
        'total_pnl': total_pnl,
    }


def main():
    print("="*70)
    print("BOTS ESPECIALIZADOS POR MONEDA - PRUEBAS EXHAUSTIVAS")
    print("="*70)
    print("\nEscenarios de prueba:")
    print("1. PEOR AÑO - stress test maximo")
    print("2. MEJOR AÑO - verificar no sobre-optimizacion")
    print("3. AÑO SINTETICO - meses aleatorios mezclados")
    print("4. ULTIMO AÑO - condiciones recientes")

    # Probar todos los pares
    pairs = ['ADA/USDT', 'XRP/USDT', 'DOT/USDT', 'DOGE/USDT',
             'ETH/USDT', 'AVAX/USDT', 'NEAR/USDT', 'LINK/USDT']

    all_results = []

    for pair in pairs:
        result = run_exhaustive_test(pair)
        if result:
            all_results.append(result)

    # Ranking final
    print("\n" + "="*70)
    print("RANKING FINAL - BOTS ESPECIALIZADOS")
    print("="*70)

    # Ordenar por WR promedio
    all_results.sort(key=lambda x: x['avg_wr'], reverse=True)

    print(f"\n{'Rank':<6} {'Par':<12} {'AUC':<8} {'WR Avg':<10} {'PnL Total':<12} {'Status'}")
    print("-"*60)

    for i, r in enumerate(all_results, 1):
        status = "VIABLE" if r['avg_wr'] >= 50 else "REVISAR"
        wr_mark = "**" if r['avg_wr'] >= 50 else ""
        print(f"{i:<6} {r['pair']:<12} {r['model_auc']:<8.3f} {r['avg_wr']:<9.1f}%{wr_mark} ${r['total_pnl']:<10,.0f} {status}")

    # Guardar resultados
    output = {
        'fecha': datetime.now().isoformat(),
        'descripcion': 'Bots especializados por moneda - Pruebas exhaustivas',
        'escenarios': ['PEOR_ANO', 'MEJOR_ANO', 'SINTETICO', 'ULTIMO_ANO'],
        'resultados': all_results,
    }

    with open(MODELS_DIR / 'specialized_bots_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResultados guardados en specialized_bots_results.json")

    # Recomendacion
    viable = [r for r in all_results if r['avg_wr'] >= 50]
    print(f"\n{'='*70}")
    print(f"RECOMENDACION")
    print(f"{'='*70}")
    if viable:
        print(f"Pares VIABLES para bots especializados ({len(viable)}):")
        for r in viable:
            print(f"  - {r['pair']}: WR {r['avg_wr']:.1f}%, PnL ${r['total_pnl']:,.0f}")
    else:
        print("Ningun par alcanzo WR >= 50% en promedio.")
        print("Considerar optimizacion adicional de filtros por par.")


if __name__ == '__main__':
    main()
