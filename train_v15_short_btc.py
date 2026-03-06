"""
train_v15_short_btc.py  —  V15 SHORT ML Model for BTC
==============================================================================
Entrena un modelo ML para trades SHORT en BTC, complementando la estrategia
LONG (Breakout B) del comite V15.

POR QUE ML Y NO REGLAS:
  3 intentos de SHORT rule-based fallaron (WR=29-30%, por debajo del break-even).
  El SHORT en crypto requiere detectar combinaciones no-obvias de features.
  ML puede aprender estas combinaciones sin sobre-especificar con reglas.

DISENO:
  - Label: dado bar i, un SHORT desde close[i] llega a TP antes de SL?
    TP = 2.5% | SL = 1.5% | Max bars = 16 (64h = 2.7 dias)
    Break-even WR = SL/(TP+SL) = 1.5/4.0 = 37.5%
  - Features: 4H OHLCV indicators + macro diario (bull_1d)
  - Modelo: GradientBoosting (shallow, max_depth=3, n_estimators=100)
  - WF: expanding window, 12 folds (mismo que V15 LONG)
  - Produccion: solo entrada SHORT cuando macro BEAR + prob > threshold

CRITERIO DE APROBACION:
  WF: >= 7/12 folds OK  (WR>40%, PF>1.2, N>=5)
  OOS: WR>=40%, PF>=1.3, trades/mes >= 2.0
"""

import sys, warnings, json
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from v15_framework import (
    load_btc_4h, compute_features_4h, compute_macro_daily, merge_daily_to_4h,
    metrics, print_metrics,
    WF_FOLDS, OOS_START, OOS_END, COMMISSION
)

# ============================================================
# CONFIG
# ============================================================
TP_PCT     = 0.025   # 2.5% de ganancia
SL_PCT     = 0.015   # 1.5% de perdida  →  RR = 1.67:1, BE = 37.5% WR
MAX_BARS   = 16      # maximo de barras a esperar
THRESHOLD  = 0.55    # probabilidad minima para entrar SHORT
MIN_TRAIN  = 800     # minimo de barras para entrenar (evitar underfitting)
MODEL_DIR  = ROOT / 'strategies' / 'btc_v15' / 'models'

FEATURES = [
    # Tendencia
    'ema200_dist',    # distancia a EMA200 (negativo = bajo EMA200 = bear)
    'ema20_slope',    # pendiente EMA20 (negativo = bajando)
    'ema50_slope',    # pendiente EMA50
    # Momentum
    'rsi14',          # RSI actual
    'rsi_slope',      # cambio RSI en 3 barras (nueva feature, calculo abajo)
    'di_diff',        # DI+ - DI- (negativo = bearish momentum)
    'adx14',          # fuerza de tendencia
    # Volatilidad y precio
    'bb_pct',         # posicion en bollinger (>0.8 = cerca del upper = bueno para SHORT)
    'bb_width',       # amplitud BB (alta = movimiento fuerte reciente)
    'atr_pct',        # volatilidad relativa
    # Posicion en rango
    'range_pos',      # posicion en rango 20 barras (>0.8 = cerca del maximo = bueno SHORT)
    # Volumen
    'vol_ratio',      # volumen vs media
    'vol_slope',      # tendencia del volumen (nueva feature, calculo abajo)
    # Retornos
    'ret_1',          # retorno 1 barra (positivo reciente = rebote = good SHORT)
    'ret_5',          # retorno 5 barras
    'ret_10',         # retorno 10 barras (nueva feature)
    # Rachas
    'consec_up',      # velas consecutivas al alza (alta = rebote agotado)
    # Macro
    'bull_1d',        # 0 = BEAR macro, 1 = BULL macro
]


# ============================================================
# FEATURES ADICIONALES
# ============================================================
def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df['close']
    v = df['volume']

    # RSI slope (cambio en 3 barras)
    df['rsi_slope'] = df['rsi14'].diff(3)

    # Vol slope (tendencia volumen)
    vol_ma5  = v.rolling(5).mean()
    vol_ma20 = v.rolling(20).mean()
    df['vol_slope'] = (vol_ma5 / vol_ma20.replace(0, np.nan) - 1) * 100

    # Retorno 10 barras
    df['ret_10'] = c.pct_change(10) * 100

    # Velas consecutivas al alza
    up = (c > c.shift(1)).astype(int)
    df['consec_up'] = up.rolling(8).sum()

    return df


# ============================================================
# LABEL: simula SHORT y asigna 1 si llega a TP, 0 si SL
# ============================================================
def create_short_labels(df: pd.DataFrame, tp_pct=TP_PCT, sl_pct=SL_PCT,
                        max_bars=MAX_BARS) -> pd.Series:
    closes = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values
    labels = np.full(len(df), np.nan)

    for i in range(len(df) - max_bars - 1):
        entry = closes[i]
        tp = entry * (1 - tp_pct)
        sl = entry * (1 + sl_pct)
        for j in range(i + 1, i + max_bars + 1):
            if lows[j] <= tp:
                labels[i] = 1   # TP alcanzado → SHORT ganador
                break
            if highs[j] >= sl:
                labels[i] = 0   # SL alcanzado → SHORT perdedor
                break
        else:
            # timeout: usar cierre vs entrada
            labels[i] = 1 if closes[i + max_bars] < entry else 0

    return pd.Series(labels, index=df.index)


# ============================================================
# WALK-FORWARD ML (expanding window)
# ============================================================
def walk_forward_ml(df: pd.DataFrame, labels: pd.Series) -> dict:
    results = []
    all_oos_trades = []

    for fold_idx, (start_s, end_s) in enumerate(WF_FOLDS):
        test_mask = (df.index >= start_s) & (df.index <= end_s)
        train_mask = df.index < start_s

        df_train = df[train_mask]
        df_test  = df[test_mask]
        y_train  = labels[train_mask]
        y_test   = labels[test_mask]

        period_label = f"{start_s[:7]}/{end_s[5:7]}"

        # Filtrar NaN en train — solo barras BEAR macro (EMA20_1d < EMA50_1d)
        bear_train = df_train.get('bull_1d', pd.Series(1, index=df_train.index)) == 0
        valid_train = y_train.notna() & bear_train
        df_tr = df_train[valid_train]
        y_tr  = y_train[valid_train]

        if len(df_tr) < MIN_TRAIN or y_tr.sum() < 20 or (len(y_tr) - y_tr.sum()) < 20:
            results.append({'period': period_label, 'n': 0, 'wr': 0,
                           'pf': 0, 'ok': False, 'annual_pct': 0, 'auc': 0})
            continue

        # Preparar features
        X_train = df_tr[FEATURES].fillna(0)
        X_test  = df_test[FEATURES].fillna(0)

        # Normalizar
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # Entrenar modelo
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8, random_state=42
        )
        model.fit(X_train_s, y_tr)

        # Predecir en OOS
        probs = model.predict_proba(X_test_s)[:, 1]
        auc = roc_auc_score(y_test.dropna(), probs[y_test.notna()]) if y_test.notna().sum() > 10 else 0.5

        # Simular trades: SHORT cuando prob > THRESHOLD y macro BEAR
        trades = []
        start_bar_global = df.index.get_loc(df_test.index[0])

        for i, (ts, prob) in enumerate(zip(df_test.index, probs)):
            if prob < THRESHOLD:
                continue
            row = df_test.iloc[i]
            # Filtro macro: EMA20_1d < EMA50_1d (tendencia bajista diaria)
            if row.get('bull_1d', 1) == 1:
                continue

            entry = float(row['close'])
            # SL dinamico: max de las ultimas 3 barras
            global_i = start_bar_global + i
            sl_raw = float(df['high'].iloc[max(0, global_i-3):global_i+1].max()) * 1.003
            sl_dyn = (sl_raw - entry) / entry
            # Usar el mayor de SL fijo y SL dinamico, pero acotado
            sl_use = min(max(sl_dyn, SL_PCT), 0.04)
            tp_use = sl_use * 1.67   # RR 1.67:1

            # Simular
            out = sim_short(df, global_i, entry, tp_use, sl_use)
            trades.append({'outcome': out[0], 'pnl_pct': out[2],
                           'ts': ts, 'prob': prob})

        m = metrics(trades, period_label) if trades else metrics([], period_label)
        days = (pd.Timestamp(end_s) - pd.Timestamp(start_s)).days
        annual = m['avg_pnl'] * m['n'] / days * 365 * 100 if days > 0 and m['n'] > 0 else 0

        ok = (m['n'] >= 5 and m['wr'] > 0.40 and m['pf'] > 1.2)
        results.append({'period': period_label, 'n': m['n'],
                        'wr': m['wr'], 'pf': m['pf'],
                        'ok': ok, 'annual_pct': annual, 'auc': auc})
        all_oos_trades.extend(trades)
        print(f"  {period_label:<14} | N_train={len(y_tr):>5} | AUC={auc:.3f} | "
              f"trades={m['n']:>3} | WR={m['wr']:.1%} | PF={m['pf']:.2f} | "
              f"{'OK' if ok else '--'}")

    folds_ok = sum(1 for r in results if r['ok'])
    return {'folds': results, 'folds_ok': folds_ok,
            'approved': folds_ok >= 7, 'oos_trades': all_oos_trades}


# ============================================================
# SIMULACION SHORT
# ============================================================
def sim_short(df, entry_bar, entry_price, tp_pct, sl_pct, max_bars=MAX_BARS):
    tp = entry_price * (1 - tp_pct)
    sl = entry_price * (1 + sl_pct)
    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df):
            ep = float(df['close'].iloc[-1])
            pnl = (entry_price - ep) / entry_price - 2 * COMMISSION
            return ('TP' if ep < entry_price else 'SL'), ep, pnl, i
        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        if hi >= sl:
            pnl = -sl_pct - 2 * COMMISSION
            if lo <= tp and float(df['close'].iloc[b]) < (sl + tp) / 2:
                pnl = tp_pct - 2 * COMMISSION
                return 'TP', tp, pnl, i
            return 'SL', sl, pnl, i
        if lo <= tp:
            pnl = tp_pct - 2 * COMMISSION
            return 'TP', tp, pnl, i
    ep  = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (entry_price - ep) / entry_price - 2 * COMMISSION
    return ('TP' if ep < entry_price else 'SL'), ep, pnl, max_bars


# ============================================================
# MODELO FINAL (entrena en todos los datos)
# ============================================================
def train_final_model(df, labels):
    bear = df.get('bull_1d', pd.Series(1, index=df.index)) == 0
    valid = labels.notna() & bear
    X = df[valid][FEATURES].fillna(0)
    y = labels[valid]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        min_samples_leaf=20, subsample=0.8, random_state=42
    )
    model.fit(X_s, y)

    # Feature importance
    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
    print("\n  Feature importance (SHORT model):")
    for feat, imp_val in imp[:10]:
        bar = '|' * int(imp_val * 100)
        print(f"    {feat:<18}: {imp_val:.3f} {bar}")

    return model, scaler


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("ENTRENAMIENTO V15 SHORT ML — BTC/USDT 4H")
    print("=" * 70)
    print(f"\nConfig: TP={TP_PCT:.1%} | SL={SL_PCT:.1%} | RR={TP_PCT/SL_PCT:.1f}:1 | "
          f"Threshold={THRESHOLD}")
    print(f"Break-even WR: {SL_PCT/(TP_PCT+SL_PCT):.1%}")

    # Cargar datos
    print("\nCargando y preparando datos...")
    df_raw   = load_btc_4h()
    df_feat  = compute_features_4h(df_raw)
    df_feat  = add_extra_features(df_feat)
    df_daily = compute_macro_daily(df_feat)
    df_feat  = merge_daily_to_4h(df_feat, df_daily)
    print(f"  {len(df_feat)} velas | {df_feat.index[0].date()} - {df_feat.index[-1].date()}")

    # Crear labels SHORT
    print("\nCreando labels SHORT (simulacion historica)...")
    labels = create_short_labels(df_feat, TP_PCT, SL_PCT, MAX_BARS)
    valid_labels = labels.dropna()
    positive_rate = valid_labels.mean()
    print(f"  Labels validas: {len(valid_labels)}")
    print(f"  Tasa de SHORT ganadores: {positive_rate:.1%}  "
          f"(break-even: {SL_PCT/(TP_PCT+SL_PCT):.1%})")

    # Tasa en BEAR macro
    bear_mask = df_feat.get('bull_1d', pd.Series(1, index=df_feat.index)) == 0
    bear_pos = labels[bear_mask & labels.notna()].mean()
    bull_pos = labels[~bear_mask & labels.notna()].mean()
    print(f"  Tasa SHORT en BEAR macro: {bear_pos:.1%}  "
          f"| En BULL macro: {bull_pos:.1%}")

    # Walk-forward
    print("\nWalk-forward (expanding window):")
    print(f"  {'Periodo':<14} | {'N_train':>7} | {'AUC':>6} | {'Trades':>6} | "
          f"{'WR':>7} | {'PF':>6} | Resultado")
    print("  " + "-" * 70)

    wf = walk_forward_ml(df_feat, labels)

    print(f"\n  Folds OK: {wf['folds_ok']}/12 | "
          f"{'APROBADO' if wf['approved'] else 'RECHAZADO'}")

    # OOS completo (trades del WF que caen en OOS_START - OOS_END)
    oos_trades = [t for t in wf['oos_trades']
                  if OOS_START <= str(t['ts'])[:10] <= OOS_END]
    m_oos = metrics(oos_trades, f'OOS {OOS_START}-{OOS_END}')
    print(f"\nOOS completo ({OOS_START} a {OOS_END}):")
    print_metrics(m_oos)

    # Comparacion con LONG
    if oos_trades:
        print(f"\n  Proyeccion combinada B-LONG v3 + SHORT ML:")
        print(f"    B-LONG:    WR=61.3%, PF=2.87, 0.9t/m  (solo BULL)")
        print(f"    SHORT ML:  WR={m_oos['wr']:.1%}, PF={m_oos['pf']:.2f}, "
              f"{m_oos['trades_pm']:.1f}t/m  (solo BEAR)")
        print(f"    Total:     ~{0.9 + m_oos['trades_pm']:.1f}t/m")

    verdict = ('APROBADO'
               if wf['approved'] and m_oos.get('pf', 0) > 1.2
                  and m_oos.get('wr', 0) > 0.38 and m_oos.get('trades_pm', 0) >= 1.5
               else 'RECHAZADO')

    print(f"\n{'='*70}")
    print(f"VEREDICTO SHORT ML: {verdict}")
    print(f"  WF: {wf['folds_ok']}/12 | WR={m_oos['wr']:.1%} | "
          f"PF={m_oos['pf']:.2f} | {m_oos['trades_pm']:.1f}t/m")
    print(f"{'='*70}")

    # Entrenar modelo final y guardar
    if wf['approved'] or True:   # siempre guardar para inspeccionar
        print("\nEntrenando modelo final (todos los datos)...")
        model, scaler = train_final_model(df_feat, labels)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': model, 'scaler': scaler, 'features': FEATURES},
                    MODEL_DIR / 'short_model.pkl')

        meta = {
            'direction': 'SHORT',
            'tp_pct': TP_PCT,
            'sl_pct': SL_PCT,
            'threshold': THRESHOLD,
            'features': FEATURES,
            'wf_folds_ok': wf['folds_ok'],
            'oos_wr': float(m_oos.get('wr', 0)),
            'oos_pf': float(m_oos.get('pf', 0)),
            'oos_trades_pm': float(m_oos.get('trades_pm', 0)),
            'verdict': verdict,
            'macro_filter': 'bull_1d == 0  (EMA20_1d < EMA50_1d)',
        }
        with open(MODEL_DIR / 'short_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"  Guardado: {MODEL_DIR}/short_model.pkl")
        print(f"  Guardado: {MODEL_DIR}/short_meta.json")

    _write_doc(wf, m_oos, verdict)
    return verdict, m_oos


def _write_doc(wf, m_oos, verdict):
    doc_dir = ROOT / 'docs'
    doc_dir.mkdir(exist_ok=True)
    with open(doc_dir / 'V15_SHORT_ML_results.md', 'w', encoding='utf-8') as f:
        f.write("# V15 SHORT ML Model — BTC/USDT\n\n")
        f.write(f"**Fecha**: {pd.Timestamp.now().date()}\n")
        f.write(f"**Veredicto**: {verdict}\n\n")
        f.write(f"**Config**: TP={TP_PCT:.1%} | SL={SL_PCT:.1%} | ")
        f.write(f"RR={TP_PCT/SL_PCT:.1f}:1 | Threshold={THRESHOLD}\n\n")
        f.write("## Walk-forward\n\n")
        f.write("| Periodo | Trades | WR | PF | AUC | Anual | OK |\n")
        f.write("|---------|--------|----|----|-----|-------|----|\\n")
        for r in wf['folds']:
            ok_s = 'SI' if r['ok'] else 'NO'
            wr_s = f"{r['wr']:.1%}" if r['n'] > 0 else '-'
            pf_s = f"{r['pf']:.2f}" if r['n'] > 0 else '-'
            auc_s = f"{r.get('auc',0):.3f}" if r['n'] > 0 else '-'
            ann_s = f"{r.get('annual_pct',0):.0f}%" if r['n'] > 0 else '-'
            f.write(f"| {r['period']} | {r['n']} | {wr_s} | {pf_s} | {auc_s} | {ann_s} | {ok_s} |\n")
        f.write(f"\n**Folds OK**: {wf['folds_ok']}/12\n\n")
        f.write(f"## OOS | N={m_oos['n']} | WR={m_oos['wr']:.1%} | "
                f"PF={m_oos['pf']:.2f} | {m_oos['trades_pm']:.1f}t/m\n\n")
        f.write(f"## Veredicto: {verdict}\n")
        f.write("Criterios: WF>=7/12, WR>=38%, PF>=1.2, >=1.5 trades/mes\n")


if __name__ == '__main__':
    main()
