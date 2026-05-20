"""
V14 BTC - Walk-Forward con Reentrenamiento ML por Fold
=======================================================
Expanding window: para cada fold de test se reentrena el ML con
TODOS los datos anteriores al fold. Esto imita el reentrenamiento
mensual que usa el bot en produccion.

Flujo por fold:
  TRAIN: 2018-01 -> fold_start  (todo el historico previo)
  TEST:  fold_start -> fold_end  (3 meses)

  1. Detectar setups en TRAIN (reglas V14)
  2. Calcular outcomes de esos setups
  3. Entrenar 3 modelos ML (context / momentum / volume)
  4. Detectar setups en TEST
  5. Aplicar ML -> filtrar por voto (>=2/3 modelos > 0.5)
  6. Comparar: reglas puras vs reglas+ML

Metricas reportadas por fold y resumen global.

Ejecutar:
  poetry run python train_validate_btc_v14_ml.py
"""
import warnings
import numpy as np
import pandas as pd
import pandas_ta as pta
from pathlib import Path
from enum import Enum
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

DATA_DIR     = Path('data')
MODEL_DIR    = Path('strategies/btc_v14/models')
TRAIN_CUTOFF = pd.Timestamp('2026-01-31', tz='UTC')
FEB26_START  = pd.Timestamp('2026-02-01', tz='UTC')
FEB26_END    = pd.Timestamp('2026-03-01', tz='UTC')
WF_START     = pd.Timestamp('2022-01-01', tz='UTC')
TRAIN_START  = pd.Timestamp('2018-01-01', tz='UTC')
N_FOLDS      = 12
FOLD_MONTHS  = 3
MIN_TRADES   = 4
MIN_TRAIN_SAMPLES = 30   # minimo de setups para entrenar ML

# TP/SL por tipo de setup
SETUP_PARAMS = {
    'PULLBACK_IN_UPTREND':  {'tp': 0.04,  'sl': 0.015},
    'OVERSOLD_IN_UPTREND':  {'tp': 0.04,  'sl': 0.015},
    'SUPPORT_BOUNCE':       {'tp': 0.025, 'sl': 0.012},
    'BREAKOUT_UP':          {'tp': 0.05,  'sl': 0.02},
}
MAX_CANDLES  = 50

# Features de cada experto ML (igual que train_btc_v14.py)
CONTEXT_FEATURES  = ['adx', 'di_diff', 'chop', 'atr_pct', 'bb_width']
MOMENTUM_FEATURES = ['rsi14', 'rsi7', 'stoch_k', 'ret_5', 'ret_20']
VOLUME_FEATURES   = ['vol_ratio', 'vol_trend', 'obv_slope']
# Umbrales a probar: skip si avg_prob < umbral (umbral bajo = menos restrictivo)
# 0.30 = filtrar solo los muy malos  |  0.38 = filtrar ~40-50% de setups
SKIP_THRESHOLDS = [None, 0.30, 0.33, 0.36, 0.38]  # None = sin filtro (raw)


# =============================================================================
# FEATURES V14
# =============================================================================

class Regime(Enum):
    TREND_UP   = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE      = "RANGE"
    VOLATILE   = "VOLATILE"


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    feat = pd.DataFrame(index=df.index)

    adx_df = pta.adx(h, l, c, length=14)
    if adx_df is not None:
        feat['adx']     = adx_df.iloc[:, 0]
        feat['di_plus'] = adx_df.iloc[:, 1]
        feat['di_minus']= adx_df.iloc[:, 2]
        feat['di_diff'] = feat['di_plus'] - feat['di_minus']

    chop = pta.chop(h, l, c, length=14)
    feat['chop'] = chop if chop is not None else 50

    feat['ema20']       = pta.ema(c, length=20)
    feat['ema50']       = pta.ema(c, length=50)
    feat['ema200']      = pta.ema(c, length=200)
    feat['ema20_dist']  = (c - feat['ema20'])  / feat['ema20']  * 100
    feat['ema200_dist'] = (c - feat['ema200']) / feat['ema200'] * 100
    feat['ema20_slope'] = feat['ema20'].pct_change(5) * 100
    feat['ema50_slope'] = feat['ema50'].pct_change(10) * 100

    feat['atr']     = pta.atr(h, l, c, length=14)
    feat['atr_pct'] = feat['atr'] / c * 100

    bb = pta.bbands(c, length=20)
    if bb is not None:
        feat['bb_upper'] = bb.iloc[:, 2]
        feat['bb_lower'] = bb.iloc[:, 0]
        feat['bb_mid']   = bb.iloc[:, 1]
        feat['bb_width'] = (feat['bb_upper'] - feat['bb_lower']) / feat['bb_mid'] * 100
        feat['bb_pct']   = (c - feat['bb_lower']) / (feat['bb_upper'] - feat['bb_lower'])

    feat['rsi14'] = pta.rsi(c, length=14)
    feat['rsi7']  = pta.rsi(c, length=7)

    stoch = pta.stoch(h, l, c, k=14, d=3)
    if stoch is not None:
        feat['stoch_k'] = stoch.iloc[:, 0]

    feat['ret_1']  = c.pct_change(1) * 100
    feat['ret_5']  = c.pct_change(5) * 100
    feat['ret_20'] = c.pct_change(20) * 100

    feat['vol_ratio'] = v / v.rolling(20).mean()
    feat['vol_trend'] = v.rolling(5).mean() / v.rolling(20).mean()
    obv = (np.sign(c.diff()) * v).cumsum()
    feat['obv_slope'] = obv.pct_change(10) * 100

    feat['high_20']   = h.rolling(20).max()
    feat['low_20']    = l.rolling(20).min()
    feat['range_pos'] = (c - feat['low_20']) / (feat['high_20'] - feat['low_20'])
    feat['consec_up'] = (c > c.shift(1)).rolling(10).sum()

    return feat.replace([np.inf, -np.inf], np.nan)


def detect_regime(row: pd.Series) -> Regime:
    adx      = row.get('adx', 20)
    di_diff  = row.get('di_diff', 0)
    chop     = row.get('chop', 50)
    atr_pct  = row.get('atr_pct', 2)
    bb_width = row.get('bb_width', 5)
    ema20_sl = row.get('ema20_slope', 0)
    ema50_sl = row.get('ema50_slope', 0)

    if pd.isna(adx) or pd.isna(chop):
        return Regime.RANGE
    if atr_pct > 4 and bb_width > 8:
        return Regime.VOLATILE
    if adx > 25 and chop < 50:
        if di_diff > 5 and ema20_sl > 0:
            return Regime.TREND_UP
        elif di_diff < -5 and ema20_sl < 0:
            return Regime.TREND_DOWN
    if chop > 55 or adx < 20:
        return Regime.RANGE
    if ema50_sl > 0.5:
        return Regime.TREND_UP
    elif ema50_sl < -0.5:
        return Regime.TREND_DOWN
    return Regime.RANGE


def detect_setups_long(feat: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, row in feat.iterrows():
        regime    = detect_regime(row)
        rsi14     = row.get('rsi14', 50)
        bb_pct    = row.get('bb_pct', 0.5)
        range_pos = row.get('range_pos', 0.5)
        ema20_d   = row.get('ema20_dist', 0)
        ema200_d  = row.get('ema200_dist', 0)
        vol_ratio = row.get('vol_ratio', 1)
        consec_up = row.get('consec_up', 0)

        if pd.isna(rsi14):
            continue

        if regime == Regime.TREND_UP:
            if rsi14 < 40 and bb_pct < 0.3 and ema200_d > 0:
                records.append({'ts': idx, 'setup_type': 'PULLBACK_IN_UPTREND'})
            elif rsi14 < 30 and ema20_d < -2:
                records.append({'ts': idx, 'setup_type': 'OVERSOLD_IN_UPTREND'})
        elif regime == Regime.RANGE:
            if range_pos < 0.2 and rsi14 < 35:
                records.append({'ts': idx, 'setup_type': 'SUPPORT_BOUNCE'})
        elif regime == Regime.VOLATILE:
            if bb_pct > 1.0 and vol_ratio > 1.5 and consec_up >= 3:
                records.append({'ts': idx, 'setup_type': 'BREAKOUT_UP'})

    return pd.DataFrame(records).set_index('ts') if records else pd.DataFrame(columns=['setup_type'])


# =============================================================================
# OUTCOMES
# =============================================================================

def compute_outcomes(df: pd.DataFrame, setups: pd.DataFrame) -> pd.Series:
    """Calcula si cada setup resulto en TP (1) o SL (0)."""
    outcomes = {}
    for ts, srow in setups.iterrows():
        if ts not in df.index:
            continue
        params = SETUP_PARAMS.get(srow['setup_type'], {'tp': 0.03, 'sl': 0.015})
        tp, sl = params['tp'], params['sl']
        entry  = df.loc[ts, 'close']
        future = df.loc[ts:].iloc[1:MAX_CANDLES + 1]

        for _, frow in future.iterrows():
            pnl = (frow['close'] - entry) / entry
            if pnl >= tp:
                outcomes[ts] = 1
                break
            elif pnl <= -sl:
                outcomes[ts] = 0
                break

    return pd.Series(outcomes, name='outcome')


# =============================================================================
# ENTRENAMIENTO ML (3 expertos)
# =============================================================================

def train_experts(feat: pd.DataFrame, setups: pd.DataFrame,
                  outcomes: pd.Series) -> dict:
    """
    Entrena los 3 modelos experto en los setups del periodo de training.
    Retorna dict con modelos listos o None si hay pocos datos.
    """
    common = setups.index.intersection(outcomes.index)
    if len(common) < MIN_TRAIN_SAMPLES:
        return None

    X_all = feat.loc[common].copy()
    y     = outcomes.loc[common]

    models = {}

    # --- Context expert ---
    ctx_cols = [c for c in CONTEXT_FEATURES if c in X_all.columns]
    X_ctx = X_all[ctx_cols].dropna()
    y_ctx = y.loc[X_ctx.index]
    if len(y_ctx) >= 20 and y_ctx.nunique() == 2:
        m = GradientBoostingClassifier(n_estimators=30, max_depth=2, random_state=42)
        m.fit(X_ctx, y_ctx)
        models['context'] = {'model': m, 'features': ctx_cols, 'scaler': None}

    # --- Momentum expert ---
    mom_cols = [c for c in MOMENTUM_FEATURES if c in X_all.columns]
    X_mom = X_all[mom_cols].dropna()
    y_mom = y.loc[X_mom.index]
    if len(y_mom) >= 20 and y_mom.nunique() == 2:
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_mom)
        m = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        m.fit(X_sc, y_mom)
        models['momentum'] = {'model': m, 'features': mom_cols, 'scaler': scaler}

    # --- Volume expert ---
    vol_cols = [c for c in VOLUME_FEATURES if c in X_all.columns]
    X_vol = X_all[vol_cols].dropna()
    y_vol = y.loc[X_vol.index]
    if len(y_vol) >= 20 and y_vol.nunique() == 2:
        m = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
        m.fit(X_vol, y_vol)
        models['volume'] = {'model': m, 'features': vol_cols, 'scaler': None}

    return models if models else None


# =============================================================================
# PREDICCION Y SIMULACION
# =============================================================================

def predict_avg_prob(feat_row: pd.Series, models: dict) -> float:
    """
    Aplica los 3 expertos y retorna probabilidad media.
    Si algun modelo no puede predecir (NaN en features), se omite.
    """
    probs = []
    for name, exp in models.items():
        cols = exp['features']
        vals = feat_row[cols].values.astype(float)
        if np.any(np.isnan(vals)):
            continue
        X = vals.reshape(1, -1)
        if exp['scaler'] is not None:
            X = exp['scaler'].transform(X)
        prob = exp['model'].predict_proba(X)[0, 1]
        probs.append(prob)
    return np.mean(probs) if probs else np.nan


def simulate_trades(df: pd.DataFrame, setups: pd.DataFrame,
                    feat: pd.DataFrame, models: dict = None,
                    skip_threshold: float = None) -> list:
    """
    Simula trades.
    Si models != None y skip_threshold != None:
      skip si avg_prob < skip_threshold  (filtrar los peores setups)
    """
    trades = []
    for ts, srow in setups.iterrows():
        if ts not in df.index:
            continue

        # Filtro ML: skip si avg_prob < umbral
        if models is not None and skip_threshold is not None and ts in feat.index:
            avg_prob = predict_avg_prob(feat.loc[ts], models)
            if np.isnan(avg_prob) or avg_prob < skip_threshold:
                continue

        params = SETUP_PARAMS.get(srow['setup_type'], {'tp': 0.03, 'sl': 0.015})
        tp, sl = params['tp'], params['sl']
        entry  = df.loc[ts, 'close']
        future = df.loc[ts:].iloc[1:MAX_CANDLES + 1]

        outcome = None
        for _, frow in future.iterrows():
            pnl = (frow['close'] - entry) / entry
            if pnl >= tp:
                outcome = 1
                break
            elif pnl <= -sl:
                outcome = 0
                break

        if outcome is not None:
            trades.append({'ts': ts, 'setup_type': srow['setup_type'],
                           'outcome': outcome, 'tp': tp, 'sl': sl})
    return trades


def fold_metrics(trades: list) -> dict:
    if not trades:
        return {'n_trades': 0, 'wr': 0.0, 'pf': 0.0}
    wins      = sum(t['outcome'] for t in trades)
    losses    = len(trades) - wins
    gross_win = sum(t['tp'] for t in trades if t['outcome'] == 1)
    gross_los = sum(t['sl'] for t in trades if t['outcome'] == 0)
    return {
        'n_trades': len(trades),
        'wr':       round(wins / len(trades), 4),
        'pf':       round(gross_win / (gross_los + 1e-10), 3),
    }


def avg_be_wr(trades: list) -> float:
    if not trades:
        return 0.333
    return np.mean([t['sl'] for t in trades]) / (
        np.mean([t['tp'] for t in trades]) + np.mean([t['sl'] for t in trades]))


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('=' * 78)
    print('V14 BTC - Walk-Forward con ML por Fold (expanding window)')
    print('=' * 78)
    n_feat = len(CONTEXT_FEATURES) + len(MOMENTUM_FEATURES) + len(VOLUME_FEATURES)
    print(f'  ML: {n_feat} features | umbrales: {SKIP_THRESHOLDS}')

    # ----- Datos -----
    print('\n[1/3] Cargando datos y calculando features...')
    df = pd.read_csv(DATA_DIR / 'BTCUSDT_4h.csv', parse_dates=['timestamp'])
    df = df.set_index('timestamp').sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    feat = compute_features(df)
    feat = feat[feat.index <= TRAIN_CUTOFF]
    df_tr = df[df.index <= TRAIN_CUTOFF]
    print(f'  {len(df_tr):,} velas 4h  [{df_tr.index[0].date()} a {df_tr.index[-1].date()}]')

    # Calcular TODOS los setups y outcomes de antemano (eficiente)
    print('\n[2/3] Calculando todos los setups y outcomes...')
    all_setups = detect_setups_long(feat)
    all_outcomes = compute_outcomes(df_tr, all_setups)
    print(f'  Setups totales: {len(all_setups)}  |  Resueltos: {len(all_outcomes)}')
    if len(all_outcomes):
        print(f'  WR global (todo el periodo): {all_outcomes.mean():.1%}')

    # ----- Walk-Forward -----
    print(f'\n[3/3] Walk-Forward: {N_FOLDS} folds x {FOLD_MONTHS} meses')
    print(f'      Test: {WF_START.date()} -> {TRAIN_CUTOFF.date()}')
    print()

    # Pre-calcular setups y modelos por fold (eficiente)
    fold_data = []
    print('  Entrenando ML por fold...')
    for i in range(N_FOLDS):
        fold_start = WF_START + pd.DateOffset(months=i * FOLD_MONTHS)
        fold_end   = fold_start + pd.DateOffset(months=FOLD_MONTHS) - pd.Timedelta(seconds=1)
        if fold_start > TRAIN_CUTOFF:
            break

        mask_test  = (feat.index >= fold_start) & (feat.index <= fold_end)
        feat_test  = feat[mask_test]
        if len(feat_test) < 30:
            continue

        setups_test = all_setups[all_setups.index.isin(feat_test.index)]

        mask_train    = (feat.index >= TRAIN_START) & (feat.index < fold_start)
        feat_train    = feat[mask_train]
        setups_train  = all_setups[all_setups.index.isin(feat_train.index)]
        outcomes_tr   = all_outcomes[all_outcomes.index.isin(setups_train.index)]
        models        = train_experts(feat_train, setups_train, outcomes_tr)

        fold_data.append({
            'i': i, 'fold_start': fold_start, 'fold_end': fold_end,
            'feat_test': feat_test, 'setups_test': setups_test,
            'models': models, 'n_train': len(outcomes_tr),
        })
        print(f'    Fold {i+1:2d}: {len(setups_test):3d} setups test | '
              f'{len(outcomes_tr):3d} setups train', end='')
        if models:
            print(f' | modelos: {list(models.keys())}')
        else:
            print(' | SIN MODELOS (pocos datos)')

    # Tabla por umbral
    all_thresh_results = {}

    for thr in SKIP_THRESHOLDS:
        label = 'Raw' if thr is None else f'skip<{thr}'
        results = []
        for fd in fold_data:
            tT = simulate_trades(df_tr, fd['setups_test'], fd['feat_test'],
                                 models=fd['models'] if thr is not None else None,
                                 skip_threshold=thr)
            mT = fold_metrics(tT)
            be = avg_be_wr(tT)
            ok = mT['n_trades'] >= MIN_TRADES and mT['wr'] > be
            results.append({'n': mT['n_trades'], 'wr': mT['wr'],
                            'pf': mT['pf'], 'pos': ok})
        all_thresh_results[label] = results

    # Mostrar tabla comparativa
    print()
    labels = list(all_thresh_results.keys())
    hdr = f"{'Fold':>4}  {'Periodo':>13}"
    for lb in labels:
        hdr += f"  {lb:>12}"
    print(hdr)
    print('-' * (20 + 14 * len(labels)))

    for idx, fd in enumerate(fold_data):
        period = f"{fd['fold_start'].strftime('%Y-%m')}/{fd['fold_end'].strftime('%m')}"
        row = f"  {fd['i']+1:2d}   {period:>13}"
        for lb in labels:
            r = all_thresh_results[lb][idx]
            flag = 'OK' if r['pos'] else ('--' if r['n'] < MIN_TRADES else 'FL')
            row += f"  {r['n']:3d} {r['wr']:5.1%}{flag}"
        print(row)

    # Resumen por umbral
    print()
    print('=' * 78)
    print('RESUMEN POR UMBRAL:')
    min_ok = max(1, int(np.ceil(12 * 0.58)))

    best_label = None
    best_score = (-1, -1)
    summaries = {}
    for lb, results in all_thresh_results.items():
        ev   = [r for r in results if r['n'] >= MIN_TRADES]
        npos = sum(r['pos'] for r in ev)
        awr  = np.mean([r['wr'] for r in ev]) if ev else 0
        apf  = np.mean([r['pf'] for r in ev]) if ev else 0
        an   = np.mean([r['n']  for r in ev]) if ev else 0
        ok   = 'APROBADO' if npos >= min_ok else 'NO APROBADO'
        summaries[lb] = {'pos': npos, 'wr': awr, 'pf': apf, 'n': an, 'ev': len(ev)}
        print(f'  {lb:12}: {ok:12} {npos}/12 folds  '
              f'WR {awr:.1%}  PF {apf:.2f}  {an:.1f} trades/fold')
        score = (npos, awr)
        if score > best_score:
            best_score = score
            best_label = lb

    print()
    print(f'  Mejor umbral: {best_label}  '
          f'({summaries[best_label]["pos"]}/12 folds, '
          f'WR {summaries[best_label]["wr"]:.1%}, '
          f'PF {summaries[best_label]["pf"]:.2f})')

    sR = summaries['Raw']
    sM = summaries.get(best_label, sR)
    wr_gain = sM['wr'] - sR['wr']
    print(f'  Impacto vs raw: WR {wr_gain:+.1%} | '
          f'Trades: {sR["n"]:.1f} -> {sM["n"]:.1f}/fold')

    # ---- Guardar modelos finales -----
    import joblib, json
    print()
    print('-' * 78)
    print('GUARDANDO MODELOS FINALES (entrenados en 2018-2026-01)...')
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    final_setups   = all_setups[all_setups.index <= TRAIN_CUTOFF]
    final_outcomes = all_outcomes[all_outcomes.index <= TRAIN_CUTOFF]
    final_models   = train_experts(feat, final_setups, final_outcomes)

    # Umbral optimo
    best_threshold = None if best_label == 'Raw' else float(best_label.split('<')[1])

    if final_models:
        for name, exp in final_models.items():
            joblib.dump({'model': exp['model'], 'features': exp['features'],
                         'scaler': exp['scaler']},
                        MODEL_DIR / f'{name}_long.pkl')
            print(f'  Guardado: {name}_long.pkl')

        meta = {
            'trained_up_to':     str(TRAIN_CUTOFF.date()),
            'n_setups_train':    int(len(final_setups)),
            'n_outcomes_train':  int(len(final_outcomes)),
            'context_features':  CONTEXT_FEATURES,
            'momentum_features': MOMENTUM_FEATURES,
            'volume_features':   VOLUME_FEATURES,
            'skip_threshold':    float(best_threshold) if best_threshold else None,
            'wf_long': {
                'folds_ok':    int(summaries[best_label]['pos']),
                'folds_total': 12,
                'avg_wr':      round(float(summaries[best_label]['wr']), 4),
                'avg_pf':      round(float(summaries[best_label]['pf']), 3),
                'approved':    bool(summaries[best_label]['pos'] >= min_ok),
            },
        }
        with open(MODEL_DIR / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'  Guardado: meta.json  (skip_threshold={best_threshold})')
    else:
        print('  WARN: No fue posible entrenar modelos finales')

    # ---- Test Feb 2026 -----
    print()
    print('-' * 78)
    print('TEST INFORMATIVO: Febrero 2026')

    feat_all = compute_features(df)
    mask_feb = (feat_all.index >= FEB26_START) & (feat_all.index < FEB26_END)
    feat_feb  = feat_all[mask_feb]
    df_feb    = df[df.index >= FEB26_START]
    setups_feb = detect_setups_long(feat_feb)

    tR_f = simulate_trades(df_feb, setups_feb, feat_feb, models=None)
    tM_f = simulate_trades(df_feb, setups_feb, feat_feb,
                           models=final_models, skip_threshold=best_threshold)
    mR_f = fold_metrics(tR_f)
    mM_f = fold_metrics(tM_f)

    print(f'  Setups detectados: {len(setups_feb)}')
    print(f'  Reglas puras:  {mR_f["n_trades"]} trades  WR {mR_f["wr"]:.1%}  PF {mR_f["pf"]:.2f}')
    print(f'  Mejor umbral ({best_label}): {mM_f["n_trades"]} trades  WR {mM_f["wr"]:.1%}  PF {mM_f["pf"]:.2f}')

    print()
    print('=' * 78)


if __name__ == '__main__':
    main()
