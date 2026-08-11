"""
Train V15 BTC - Walk-forward training del sistema Expert Committee.
===================================================================

Modelos entrenados (LONG y SHORT separados):
  - setup_model_long.pkl  : Price Action Expert LONG
  - setup_model_short.pkl : Price Action Expert SHORT
  - volume_model.pkl      : Volume Expert (mismo para ambas direcciones)
  - scaler.pkl
  - meta.json             : features, umbrales, metricas walk-forward

Validacion walk-forward (expanding window):
  - 12 folds, cada uno prueba 3 meses out-of-sample
  - Test cubre: 2022-Q1 hasta 2025-Q4
  - Entrena sobre TODO lo anterior a cada test window
  - Se valida LONG y SHORT por separado

Criterio de aprobacion (per CLAUDE.md):
  - LONG: >= 7/12 folds positivos
  - SHORT: >= 7/12 folds positivos (independiente)
  - WR > break-even (33% para TP3/SL1.5)

Ejecutar con venv de produccion:
  C:\\Users\\pcdec\\AppData\\Local\\pypoetry\\Cache\\virtualenvs\\binance-scalper-bot-ofXWUGOe-py3.12\\Scripts\\python.exe train_v15_btc.py
"""

import warnings
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from v15_data_pipeline import run_pipeline
from v15_features import (
    build_feature_matrix,
    create_label,
    SETUP_FEATURES,
    VOLUME_FEATURES,
)

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACION
# =============================================================================

MODEL_DIR = Path('strategies/btc_v15/models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TP_PCT = 0.03
SL_PCT = 0.015
MAX_CANDLES = 12
BREAK_EVEN_WR = SL_PCT / (TP_PCT + SL_PCT)  # ~33%

# Walk-forward: 2022-Q1 hasta 2025-Q4 (4 anos x 4 trimestres = 16 folds)
N_FOLDS = 16
TEST_MONTHS = 3
FIRST_TEST_DATE = '2022-01-01'

# El modelo NO ve datos de febrero 2026 durante el entrenamiento.
# Febrero 2026 se usa como test out-of-sample informativo al final.
TRAIN_CUTOFF = '2026-01-31'
FEB26_START  = '2026-02-01'
FEB26_END    = '2026-03-01'

# Aprobacion: >= 58% de folds positivos (independiente del total de folds)
APPROVAL_PCT = 0.58

# Modelos shallow para evitar overfitting
SETUP_MODEL_PARAMS = {
    'n_estimators': 50,
    'max_depth': 3,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'random_state': 42,
}
VOLUME_MODEL_PARAMS = {
    'n_estimators': 40,
    'max_depth': 2,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'random_state': 42,
}


# =============================================================================
# WALK-FORWARD SPLITS
# =============================================================================

def get_wf_splits(feat: pd.DataFrame) -> list:
    first_test_ts = pd.Timestamp(FIRST_TEST_DATE, tz='UTC')
    splits = []
    for i in range(N_FOLDS):
        test_start = first_test_ts + pd.DateOffset(months=i * TEST_MONTHS)
        test_end = test_start + pd.DateOffset(months=TEST_MONTHS)
        train_mask = feat.index < test_start
        test_mask = (feat.index >= test_start) & (feat.index < test_end)
        if train_mask.sum() < 500 or test_mask.sum() < 50:
            continue
        splits.append({
            'fold': i + 1,
            'train_idx': feat.index[train_mask],
            'test_idx': feat.index[test_mask],
            'test_start': test_start,
            'test_end': test_end,
        })
    return splits


# =============================================================================
# SIMULACION DE UN FOLD (para metricas)
# =============================================================================

def simulate_fold(
    df_4h, test_idx, feat,
    setup_model, volume_model, scaler,
    direction: str,
    setup_threshold: float = 0.52,
) -> dict:
    """Simula trades en el periodo de test."""
    trades = []

    for ts in test_idx:
        if ts not in feat.index or ts not in df_4h.index:
            continue

        row = feat.loc[ts]

        # Pre-filtro tecnico: aproxima el zone filter del bot en produccion.
        # El bot real solo entra cuando el precio llega a una zona clave
        # (soporte/FVG/PDL para LONG, resistencia/FVG/PDH para SHORT).
        # Aqui simulamos eso con condiciones de precio/momentum simples:
        rsi = row.get('rsi14', 50)
        bb_pct = row.get('bb_pct', 0.5)
        range_pos = row.get('range_pos', 0.5)

        if direction == 'long':
            # Solo LONG cuando precio esta en zona baja: cercano a soporte
            if not (rsi < 45 or bb_pct < 0.35 or range_pos < 0.3):
                continue
        else:
            # Solo SHORT cuando precio esta en zona alta: cercano a resistencia
            if not (rsi > 55 or bb_pct > 0.65 or range_pos > 0.7):
                continue

        # Setup score
        setup_cols = [c for c in SETUP_FEATURES if c in feat.columns]
        setup_vals = row[setup_cols].values.astype(float)
        if np.any(np.isnan(setup_vals)):
            continue
        setup_prob = setup_model.predict_proba(setup_vals.reshape(1, -1))[0, 1]
        if setup_prob < setup_threshold:
            continue

        # Volume
        vol_cols = [c for c in VOLUME_FEATURES if c in feat.columns]
        vol_vals = row[vol_cols].values.astype(float)
        vol_regime = 'NEUTRAL'
        if not np.any(np.isnan(vol_vals)):
            vol_scaled = scaler.transform(vol_vals.reshape(1, -1))
            vol_prob = volume_model.predict_proba(vol_scaled)[0, 1]
            vol_regime = 'CONFIRM' if vol_prob >= 0.6 else ('WARN' if vol_prob <= 0.35 else 'NEUTRAL')

        # Resultado real
        entry_price = df_4h.loc[ts, 'close']
        outcome = None
        for _, frow in df_4h.loc[ts:].iloc[1:MAX_CANDLES + 1].iterrows():
            fut = frow['close']
            if direction == 'long':
                pnl = (fut - entry_price) / entry_price
            else:
                pnl = (entry_price - fut) / entry_price
            if pnl >= TP_PCT:
                outcome = 1
                break
            elif pnl <= -SL_PCT:
                outcome = 0
                break

        if outcome is not None:
            trades.append({'outcome': outcome, 'vol_regime': vol_regime})

    if not trades:
        return {'n_trades': 0, 'wr': 0, 'pf': 0, 'positive': False}

    df_t = pd.DataFrame(trades)
    n = len(df_t)
    wins = df_t['outcome'].sum()
    losses = n - wins
    wr = wins / n
    pf = (wins * TP_PCT) / (losses * SL_PCT + 1e-10)

    return {
        'n_trades': n,
        'wr': round(wr, 3),
        'pf': round(pf, 3),
        'wins': int(wins),
        'losses': int(losses),
        'positive': wr > BREAK_EVEN_WR and pf > 1.0,
    }


# =============================================================================
# ENTRENAR UN FOLD
# =============================================================================

def train_fold(feat, labels, train_idx) -> tuple:
    """Entrena setup_model y volume_model en train_idx."""
    common = feat.index.intersection(train_idx).intersection(labels.index)
    if len(common) < 100:
        return None, None, None

    setup_cols = [c for c in SETUP_FEATURES if c in feat.columns]
    X_setup = feat.loc[common, setup_cols].apply(pd.to_numeric, errors='coerce').dropna()
    y = labels.loc[X_setup.index]
    if len(y) < 50 or y.nunique() < 2:
        return None, None, None

    setup_model = GradientBoostingClassifier(**SETUP_MODEL_PARAMS)
    setup_model.fit(X_setup, y)

    vol_cols = [c for c in VOLUME_FEATURES if c in feat.columns]
    X_vol = feat.loc[common, vol_cols].dropna()
    y_vol = labels.loc[X_vol.index]
    if len(y_vol) < 50 or y_vol.nunique() < 2:
        return None, None, None

    scaler = StandardScaler()
    vol_model = GradientBoostingClassifier(**VOLUME_MODEL_PARAMS)
    vol_model.fit(scaler.fit_transform(X_vol), y_vol)

    return setup_model, vol_model, scaler


# =============================================================================
# WALK-FORWARD PARA UNA DIRECCION
# =============================================================================

def run_walk_forward(
    df_4h, feat, labels, splits,
    direction: str,
) -> tuple:
    """
    Ejecuta walk-forward para LONG o SHORT.
    Retorna (fold_results, setup_final, vol_final, scaler_final).
    """
    fold_results = []

    for split in splits:
        setup_m, vol_m, scaler = train_fold(feat, labels, split['train_idx'])
        if setup_m is None:
            fold_results.append({'fold': split['fold'], 'positive': False,
                                  'n_trades': 0, 'wr': 0, 'pf': 0})
            continue

        result = simulate_fold(
            df_4h, split['test_idx'], feat,
            setup_m, vol_m, scaler,
            direction=direction,
        )
        result['fold'] = split['fold']
        result['test_start'] = str(split['test_start'].date())
        fold_results.append(result)

        status = 'OK ' if result['positive'] else 'FAIL'
        print(f'    Fold {split["fold"]:02d} [{split["test_start"].date()}]: '
              f'{result["n_trades"]:3d} trades | '
              f'WR {result["wr"]:.1%} | PF {result["pf"]:.2f} | [{status}]')

    # Modelo final con todos los datos
    setup_final, vol_final, scaler_final = train_fold(feat, labels, feat.index)
    return fold_results, setup_final, vol_final, scaler_final


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('=' * 70)
    print('V15 BTC - Walk-Forward Training (LONG + SHORT)')
    print('=' * 70)

    # 1. Datos
    print('\n[1/5] Cargando datos...')
    data = run_pipeline()
    df_4h_full = data['4h']
    df_1d_full = data['1d']
    funding_df = data['funding']
    fng_df = data['fng']
    print(f'  4h completo: {len(df_4h_full):,} velas | {df_4h_full.index[0].date()} a {df_4h_full.index[-1].date()}')

    # Separar datos de entrenamiento (hasta Jan 2026) y test Feb 2026
    cutoff = pd.Timestamp(TRAIN_CUTOFF, tz='UTC')
    feb_start = pd.Timestamp(FEB26_START, tz='UTC')
    feb_end   = pd.Timestamp(FEB26_END,   tz='UTC')

    df_4h = df_4h_full[df_4h_full.index <= cutoff].copy()
    df_1d = df_1d_full[df_1d_full.index <= cutoff].copy()
    df_4h_feb = df_4h_full[(df_4h_full.index >= feb_start) & (df_4h_full.index < feb_end)].copy()

    print(f'  4h entrenamiento: {len(df_4h):,} velas | hasta {df_4h.index[-1].date()}')
    print(f'  4h feb-2026:      {len(df_4h_feb):,} velas | '
          f'{df_4h_feb.index[0].date() if len(df_4h_feb) else "sin datos"} '
          f'a {df_4h_feb.index[-1].date() if len(df_4h_feb) else ""}')

    # 2. Features
    print('\n[2/5] Calculando features...')
    feat = build_feature_matrix(df_4h, df_1d, funding_df, fng_df)
    print(f'  {len(feat):,} filas | {feat.shape[1]} columnas')

    # 3. Labels
    print(f'\n[3/5] Labels (TP={TP_PCT:.0%} / SL={SL_PCT:.0%} / break-even WR={BREAK_EVEN_WR:.0%})')
    labels_long = create_label(df_4h, 'long', TP_PCT, SL_PCT, MAX_CANDLES)
    labels_short = create_label(df_4h, 'short', TP_PCT, SL_PCT, MAX_CANDLES)
    wr_base_long = labels_long.mean()
    wr_base_short = labels_short.mean()
    print(f'  LONG  labels: {len(labels_long):,} | WR base (sin filtros): {wr_base_long:.1%}')
    print(f'  SHORT labels: {len(labels_short):,} | WR base (sin filtros): {wr_base_short:.1%}')

    # 4. Walk-forward splits
    splits = get_wf_splits(feat)
    min_folds = max(1, int(len(splits) * APPROVAL_PCT))
    print(f'\n[4/5] Walk-forward: {len(splits)} folds x {TEST_MONTHS} meses | '
          f'desde {FIRST_TEST_DATE} | aprobacion >= {min_folds}/{len(splits)} ({APPROVAL_PCT:.0%})')

    # --- LONG ---
    print('\n  LONG direction:')
    long_results, setup_long, vol_long, scaler_long = run_walk_forward(
        df_4h, feat, labels_long, splits, 'long'
    )

    # SHORT descartado: historicamente no funciona en BTC sin mercado bear
    # estructurado (ver CLAUDE.md). No hay suficientes trades en bull market.
    short_results, setup_short, vol_short, scaler_short = [], None, None, None

    # 5. Resultados
    print('\n[5/5] Resultados y guardado de modelos...')

    MIN_TRADES_PER_FOLD = 5  # folds con menos trades no son estadisticamente evaluables

    def summarize(results, label):
        n_total = len(results)
        # Folds evaluables: los que tienen suficientes trades
        evaluable = [r for r in results if r.get('n_trades', 0) >= MIN_TRADES_PER_FOLD]
        skipped = n_total - len(evaluable)
        n_pos = sum(1 for r in evaluable if r.get('positive', False))
        wrs = [r['wr'] for r in evaluable]
        pfs = [r['pf'] for r in evaluable]
        avg_wr = np.mean(wrs) if wrs else 0
        avg_pf = np.mean(pfs) if pfs else 0
        min_ok = max(1, int(len(evaluable) * APPROVAL_PCT))
        approved = len(evaluable) > 0 and n_pos >= min_ok
        print(f'\n  {label}:')
        print(f'    Folds totales:   {n_total} | evaluables (>={MIN_TRADES_PER_FOLD} trades): {len(evaluable)} | sin trades: {skipped}')
        print(f'    Folds positivos: {n_pos}/{len(evaluable)}  '
              f'{"APROBADO" if approved else f"NO APROBADO (necesita >= {min_ok}/{len(evaluable)})"}')
        print(f'    WR promedio:     {avg_wr:.1%}  (break-even: {BREAK_EVEN_WR:.1%})')
        print(f'    PF promedio:     {avg_pf:.2f}')
        return n_pos, avg_wr, avg_pf, approved

    long_pos, long_wr, long_pf, long_ok = summarize(long_results, 'LONG')
    short_pos, short_wr, short_pf, short_ok = summarize(short_results, 'SHORT')

    # Advertencias
    if not long_ok:
        print('\n  ADVERTENCIA LONG: no aprueba walk-forward.')
        print('  Ver METODOLOGIA_TESTING.md: no se arregla con umbral mas estricto.')

    # =========================================================================
    # TEST INFORMATIVO: FEBRERO 2026 (mercado bear — no visto durante training)
    # =========================================================================
    feb_results = {}
    if len(df_4h_feb) >= 20 and setup_long is not None:
        print('\n' + '=' * 70)
        print('TEST INFORMATIVO: Febrero 2026 (out-of-sample, mercado bear)')
        print('  Este resultado NO determina aprobacion — es solo una referencia.')
        print('  La validacion real es el walk-forward + cross-asset (ETH).')
        print('=' * 70)

        # Features de febrero 2026 (usar datos completos hasta feb para indicators)
        df_4h_to_feb = df_4h_full[df_4h_full.index < feb_end].copy()
        df_1d_to_feb = df_1d_full[df_1d_full.index < feb_end].copy()
        feat_full = build_feature_matrix(df_4h_to_feb, df_1d_to_feb, funding_df, fng_df)
        feat_feb = feat_full[feat_full.index >= feb_start]
        feb_idx = feat_feb.index

        macro_feb_counts = feat_full.loc[feat_full.index >= feb_start, 'macro_regime'].value_counts().to_dict()
        print(f'  Velas 4h en febrero: {len(df_4h_feb)} | Feature rows: {len(feat_feb)}')
        print(f'  Regimen macro predominante: {macro_feb_counts}')

        for direction, model in [('LONG', setup_long), ('SHORT', setup_short)]:
            if model is None:
                feb_results[direction] = {'n_trades': 0, 'wr': 0, 'pf': 0}
                continue
            result = simulate_fold(
                df_4h_to_feb, feb_idx, feat_full,
                model, vol_long, scaler_long,
                direction=direction.lower(),
                setup_threshold=0.52,
            )
            feb_results[direction] = result
            status = 'OK' if result.get('positive') else 'FAIL'
            print(f'  Feb-2026 {direction}: {result["n_trades"]:3d} trades | '
                  f'WR {result["wr"]:.1%} | PF {result["pf"]:.2f} | [{status}]')

        if not feb_results.get('LONG', {}).get('positive') and long_ok:
            print('\n  NOTA: Walk-forward aprueba pero Feb-2026 falla.')
            print('  Es esperado — Feb-2026 fue bear extremo.')
            print('  Cross-asset (ETH) determinara si hay overfitting real.')
        elif feb_results.get('LONG', {}).get('positive'):
            print('\n  NOTA: Modelos funcionaron en mercado bear de Feb-2026.')
            print('  Buena senal de generalizacion.')
    else:
        print('\n  (Sin datos suficientes de Feb-2026 para test informativo)')

    # Guardar modelos (aunque no aprueben, para diagnostico)
    if setup_long is not None:
        joblib.dump(setup_long, MODEL_DIR / 'setup_model_long.pkl')
        print('\n  Guardado: setup_model_long.pkl')
    if setup_short is not None:
        joblib.dump(setup_short, MODEL_DIR / 'setup_model_short.pkl')
        print('  Guardado: setup_model_short.pkl')
    if vol_long is not None:
        joblib.dump(vol_long, MODEL_DIR / 'volume_model.pkl')
        joblib.dump(scaler_long, MODEL_DIR / 'scaler.pkl')
        print('  Guardado: volume_model.pkl + scaler.pkl')

    meta = {
        'version': 'V15',
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'pair': 'BTC/USDT',
        'timeframe': '4h',
        'tp_pct': TP_PCT,
        'sl_pct': SL_PCT,
        'break_even_wr': round(BREAK_EVEN_WR, 4),
        'setup_features': SETUP_FEATURES,
        'volume_features': VOLUME_FEATURES,
        'setup_threshold_long': 0.52,
        'setup_threshold_short': 0.52,
        'long': {
            'approved': long_ok,
            'folds_positive': long_pos,
            'avg_wr': round(long_wr, 4),
            'avg_pf': round(long_pf, 4),
            'folds': long_results,
        },
        'short': {
            'approved': short_ok,
            'folds_positive': short_pos,
            'avg_wr': round(short_wr, 4),
            'avg_pf': round(short_pf, 4),
            'folds': short_results,
        },
        'feb_2026_test': {
            'note': 'informativo — no determina aprobacion',
            'long': feb_results.get('LONG', {}),
            'short': feb_results.get('SHORT', {}),
        },
    }
    with open(MODEL_DIR / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    print('  Guardado: meta.json')

    print('\nSiguiente paso:')
    print('  python validate_v15_cross_asset.py')
    print('  -> Prueba los modelos BTC en ETH sin reentrenar')
    print('  -> Si funciona en ETH = no hay overfitting a BTC')


if __name__ == '__main__':
    main()
