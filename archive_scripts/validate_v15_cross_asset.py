"""
V15 Cross-Asset Validation - Prueba modelos BTC en ETH sin reentrenar.
=======================================================================

Por que es la validacion correcta:
  - El walk-forward (train_v15_btc.py) ya valida que no hay look-ahead bias
  - Este script prueba que el modelo NO esta memorizando patrones especificos de BTC
  - Si los modelos entrenados en BTC funcionan en ETH -> el modelo aprendio algo real
  - Si funcionan en BTC pero fallan en ETH -> overfitting a BTC, descartarlo

Logica:
  1. Carga modelos entrenados en BTC (setup_model_long/short, volume_model)
  2. Descarga datos de ETH (nunca vistos durante el entrenamiento)
  3. Calcula las mismas features en ETH
  4. Aplica los modelos BTC en ETH SIN reentrenar
  5. Reporta WR, PF, trades -> compara con BTC walk-forward

Criterio de aprobacion:
  - ETH debe mostrar WR > break-even (33%)
  - Si ETH WR < 40% pero BTC WR > 55% -> sospecha de overfitting

Ejecutar despues de train_v15_btc.py:
  python validate_v15_cross_asset.py
"""

import warnings
import json
import joblib
import sys
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

MODEL_DIR = Path('strategies/btc_v15/models')
DATA_DIR = Path('data')

TP_PCT = 0.03
SL_PCT = 0.015
MAX_CANDLES = 12
BREAK_EVEN_WR = SL_PCT / (TP_PCT + SL_PCT)

CROSS_PAIRS = ['ETH/USDT']  # Ampliar despues: SOL, BNB
FAPI_BASE = 'https://fapi.binance.com'


# =============================================================================
# DESCARGA DE DATOS CROSS-ASSET
# =============================================================================

def download_klines_for_validation(symbol: str, interval: str = '4h') -> pd.DataFrame:
    """Descarga klines del par de validacion (mismo formato que BTC)."""
    import requests, time
    from datetime import datetime, timezone

    api_symbol = symbol.replace('/', '')
    cache = DATA_DIR / f'{api_symbol.lower()}_{interval}_v15_cross.parquet'

    if cache.exists() and (time.time() - cache.stat().st_mtime) / 3600 < 12:
        df = pd.read_parquet(cache)
        print(f'  {symbol} {interval}: cache OK ({len(df):,} velas)')
        return df

    since_ts = int(datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    rows = []
    print(f'  Descargando {symbol} {interval}...')
    while True:
        try:
            resp = requests.get(
                f'{FAPI_BASE}/fapi/v1/klines',
                params={'symbol': api_symbol, 'interval': interval,
                        'startTime': since_ts, 'limit': 1500},
                timeout=20,
            )
            resp.raise_for_status()
            page = resp.json()
        except Exception as e:
            print(f'  Error: {e}, reintentando...')
            time.sleep(5)
            continue

        if not page:
            break
        rows.extend(page)
        since_ts = page[-1][0] + 1
        if len(page) < 1500:
            break
        time.sleep(0.2)

    df = pd.DataFrame(rows, columns=[
        'ts', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_vol', 'n_trades',
        'taker_buy_vol', 'taker_buy_quote_vol', 'ignore',
    ])
    df['timestamp'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df = df.set_index('timestamp').sort_index()
    for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_vol']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['taker_sell_vol'] = df['volume'] - df['taker_buy_vol']
    df = df[['open', 'high', 'low', 'close', 'volume', 'taker_buy_vol', 'taker_sell_vol']]
    df = df[~df.index.duplicated(keep='last')].dropna(subset=['close'])
    df.to_parquet(cache)
    print(f'  {symbol} {interval}: {len(df):,} velas guardadas')
    return df


# =============================================================================
# SIMULACION CROSS-ASSET
# =============================================================================

def simulate_cross_asset(
    df_4h: pd.DataFrame,
    feat: pd.DataFrame,
    setup_model,
    volume_model,
    scaler,
    setup_features: list,
    volume_features: list,
    direction: str,
    macro_series: pd.Series,
    sentiment_series: pd.Series,
    setup_threshold: float = 0.52,
) -> dict:
    """
    Aplica modelos entrenados en BTC sobre datos de otro par.
    NO reentrenar. Medir si funciona.
    """
    trades = []

    for ts in feat.index:
        if ts not in df_4h.index:
            continue

        row = feat.loc[ts]
        macro = macro_series.get(ts, 'MACRO_RANGE')
        sentiment = sentiment_series.get(ts, 'NEUTRAL')

        if direction == 'long':
            if macro == 'MACRO_BEAR':
                continue
            if sentiment == 'BEARISH_BIAS':
                continue
        else:
            if macro == 'MACRO_BULL':
                continue
            if sentiment == 'BULLISH_BIAS':
                continue

        # Setup score con modelo BTC
        setup_cols = [c for c in setup_features if c in feat.columns]
        setup_vals = row[setup_cols].values.astype(float)
        if np.any(np.isnan(setup_vals)):
            continue
        setup_prob = setup_model.predict_proba(setup_vals.reshape(1, -1))[0, 1]
        if setup_prob < setup_threshold:
            continue

        # Volume score
        vol_cols = [c for c in volume_features if c in feat.columns]
        vol_vals = row[vol_cols].values
        if not np.any(np.isnan(vol_vals)):
            vol_scaled = scaler.transform(vol_vals.reshape(1, -1))

        # Resultado en este par
        entry_price = df_4h.loc[ts, 'close']
        outcome = None
        for _, frow in df_4h.loc[ts:].iloc[1:MAX_CANDLES + 1].iterrows():
            fut = frow['close']
            pnl = (fut - entry_price) / entry_price if direction == 'long' \
                else (entry_price - fut) / entry_price
            if pnl >= TP_PCT:
                outcome = 1
                break
            elif pnl <= -SL_PCT:
                outcome = 0
                break

        if outcome is not None:
            trades.append({'outcome': outcome, 'setup_prob': setup_prob})

    if not trades:
        return {'n_trades': 0, 'wr': 0, 'pf': 0}

    df_t = pd.DataFrame(trades)
    n = len(df_t)
    wins = df_t['outcome'].sum()
    losses = n - wins
    wr = wins / n
    pf = (wins * TP_PCT) / (losses * SL_PCT + 1e-10)
    return {
        'n_trades': n,
        'wr': round(wr, 4),
        'pf': round(pf, 3),
        'wins': int(wins),
        'losses': int(losses),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('=' * 70)
    print('V15 - Cross-Asset Validation (BTC modelos -> ETH data)')
    print('=' * 70)
    print()
    print('Logica: si el modelo BTC funciona en ETH sin reentrenar,')
    print('aprendio patrones reales de mercado, no memorizo BTC.')
    print()

    # Cargar modelos BTC
    if not (MODEL_DIR / 'setup_model_long.pkl').exists():
        print('ERROR: No se encontraron modelos. Ejecutar train_v15_btc.py primero.')
        sys.exit(1)

    setup_long = joblib.load(MODEL_DIR / 'setup_model_long.pkl')
    setup_short = joblib.load(MODEL_DIR / 'setup_model_short.pkl') \
        if (MODEL_DIR / 'setup_model_short.pkl').exists() else None
    vol_model = joblib.load(MODEL_DIR / 'volume_model.pkl')
    scaler = joblib.load(MODEL_DIR / 'scaler.pkl')

    with open(MODEL_DIR / 'meta.json') as f:
        meta = json.load(f)

    setup_features = meta['setup_features']
    volume_features = meta['volume_features']
    threshold_long = meta.get('setup_threshold_long', 0.52)
    threshold_short = meta.get('setup_threshold_short', 0.52)

    # Mostrar resultados BTC walk-forward como referencia
    print('REFERENCIA - BTC Walk-Forward:')
    btc_long = meta.get('long', {})
    btc_short = meta.get('short', {})
    print(f'  BTC LONG:  {btc_long.get("folds_positive","?")} folds | '
          f'WR {btc_long.get("avg_wr", 0):.1%} | PF {btc_long.get("avg_pf", 0):.2f} | '
          f'{"APROBADO" if btc_long.get("approved") else "NO APROBADO"}')
    print(f'  BTC SHORT: {btc_short.get("folds_positive","?")} folds | '
          f'WR {btc_short.get("avg_wr", 0):.1%} | PF {btc_short.get("avg_pf", 0):.2f} | '
          f'{"APROBADO" if btc_short.get("approved") else "NO APROBADO"}')
    print()

    # Lazy import de features para no contaminar
    from v15_features import (
        build_feature_matrix,
        compute_macro_features,
        align_macro_to_4h,
        compute_sentiment_features,
    )
    from v15_data_pipeline import download_funding_rates, download_fear_greed

    results_all = {}

    for symbol in CROSS_PAIRS:
        print(f'VALIDANDO: {symbol}')
        print('-' * 50)

        # Datos del par de validacion
        df_4h = download_klines_for_validation(symbol, '4h')
        df_1d = download_klines_for_validation(symbol, '1d')
        funding_df = download_funding_rates(symbol.replace('/', ''))
        fng_df = download_fear_greed()

        # Features (mismas funciones, diferente par)
        feat = build_feature_matrix(df_4h, df_1d, funding_df, fng_df)
        macro_series = feat['macro_regime']
        sentiment_series = feat['sentiment_regime']

        # Solo datos desde 2022 (out-of-sample respecto a BTC training)
        oos_start = pd.Timestamp('2022-01-01', tz='UTC')
        feat_oos = feat[feat.index >= oos_start]
        print(f'  Features OOS (desde 2022): {len(feat_oos):,} velas')

        # LONG con modelos BTC
        print('  Probando LONG...')
        long_result = simulate_cross_asset(
            df_4h, feat_oos, setup_long, vol_model, scaler,
            setup_features, volume_features, 'long',
            macro_series, sentiment_series, threshold_long,
        )

        # SHORT con modelos BTC
        short_result = {'n_trades': 0, 'wr': 0, 'pf': 0}
        if setup_short is not None:
            print('  Probando SHORT...')
            short_result = simulate_cross_asset(
                df_4h, feat_oos, setup_short, vol_model, scaler,
                setup_features, volume_features, 'short',
                macro_series, sentiment_series, threshold_short,
            )

        # Reporte
        long_ok = long_result['wr'] > BREAK_EVEN_WR and long_result['n_trades'] > 20
        short_ok = short_result['wr'] > BREAK_EVEN_WR and short_result['n_trades'] > 20

        print(f'\n  {symbol} LONG:  {long_result["n_trades"]:3d} trades | '
              f'WR {long_result["wr"]:.1%} | PF {long_result["pf"]:.2f} | '
              f'{"OK" if long_ok else "FAIL"}')
        print(f'  {symbol} SHORT: {short_result["n_trades"]:3d} trades | '
              f'WR {short_result["wr"]:.1%} | PF {short_result["pf"]:.2f} | '
              f'{"OK" if short_ok else "FAIL"}')

        # Comparar con BTC para detectar overfitting
        btc_wr_long = btc_long.get('avg_wr', 0)
        btc_wr_short = btc_short.get('avg_wr', 0)
        long_drop = btc_wr_long - long_result['wr']
        short_drop = btc_wr_short - short_result['wr']

        print()
        print(f'  Degradacion LONG  (BTC WR - {symbol} WR): {long_drop:+.1%}')
        print(f'  Degradacion SHORT (BTC WR - {symbol} WR): {short_drop:+.1%}')

        if long_drop > 0.15:
            print(f'  ALERTA: Degradacion LONG > 15pp -> posible overfitting a BTC')
        if short_drop > 0.15:
            print(f'  ALERTA: Degradacion SHORT > 15pp -> posible overfitting a BTC')
        print()

        results_all[symbol] = {
            'long': long_result,
            'short': short_result,
            'long_ok': long_ok,
            'short_ok': short_ok,
            'long_degradation': round(long_drop, 4),
            'short_degradation': round(short_drop, 4),
        }

    # Resumen final
    print('=' * 50)
    print('RESUMEN CROSS-ASSET VALIDATION:')
    print()
    print(f'  BTC LONG  (walk-forward):  '
          f'WR {btc_long.get("avg_wr", 0):.1%} | '
          f'{"APROBADO" if btc_long.get("approved") else "NO APROBADO"}')
    for sym, res in results_all.items():
        print(f'  {sym} LONG  (cross):        '
              f'WR {res["long"]["wr"]:.1%} | '
              f'{"OK" if res["long_ok"] else "FAIL"}')
    print()
    print(f'  BTC SHORT (walk-forward):  '
          f'WR {btc_short.get("avg_wr", 0):.1%} | '
          f'{"APROBADO" if btc_short.get("approved") else "NO APROBADO"}')
    for sym, res in results_all.items():
        print(f'  {sym} SHORT (cross):        '
              f'WR {res["short"]["wr"]:.1%} | '
              f'{"OK" if res["short_ok"] else "FAIL"}')

    # Veredicto
    print()
    print('VEREDICTO:')
    long_cross_ok = all(r['long_ok'] for r in results_all.values())
    short_cross_ok = all(r['short_ok'] for r in results_all.values())

    if btc_long.get('approved') and long_cross_ok:
        print('  LONG: APROBADO para integracion al bot')
    elif btc_long.get('approved') and not long_cross_ok:
        print('  LONG: Walk-forward OK pero cross-asset FALLA -> overfitting. NO conectar.')
    else:
        print('  LONG: No pasa walk-forward. Reentrenar con mejor metodologia.')

    if btc_short.get('approved') and short_cross_ok:
        print('  SHORT: APROBADO para integracion al bot')
    elif btc_short.get('approved') and not short_cross_ok:
        print('  SHORT: Walk-forward OK pero cross-asset FALLA -> overfitting. NO conectar.')
    else:
        print('  SHORT: No pasa walk-forward. Reentrenar o excluir SHORT.')

    # Guardar resultados
    cross_results = {
        'validated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        'btc_reference': {'long': btc_long, 'short': btc_short},
        'cross_asset': results_all,
    }
    out = Path('strategies/btc_v15/cross_asset_results.json')
    with open(out, 'w') as f:
        json.dump(cross_results, f, indent=2, default=str)
    print(f'\nResultados guardados: {out}')


if __name__ == '__main__':
    main()
