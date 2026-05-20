"""
validate_1h_confirmation.py
============================
Valida la mejora F: Confirmacion 1H antes de entrar.

Logica: un setup LONG en 4H se confirma si en 1H:
  - precio > EMA20_1H  (tendencia alcista corta)
  - RSI_1H > 40        (no sobrevendido bajista)
  - (opcional) vela 1H actual es bullish

Si no se cumple, se salta el trade.

OOS: 2022-01-01 a 2026-01-31 (mismo que validate_trader_improvements.py)
Criterio: mejora WR o PF sin reducir trades > 30%
"""

import numpy as np
import pandas as pd
import pandas_ta as pta
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / 'data'
MODELS_DIR   = PROJECT_ROOT / 'strategies' / 'btc_v14' / 'models'

OOS_START      = '2022-01-01'
OOS_END        = '2026-01-31'
MAX_BARS       = 12
SKIP_THRESHOLD = 0.30

SETUP_PARAMS = {
    'SUPPORT_BOUNCE':      {'tp': 0.025, 'sl': 0.012},
    'PULLBACK_IN_UPTREND': {'tp': 0.040, 'sl': 0.015},
    'OVERSOLD_IN_UPTREND': {'tp': 0.030, 'sl': 0.015},
    'BREAKOUT_UP':         {'tp': 0.050, 'sl': 0.020},
}
DEFAULT_PARAMS = {'tp': 0.030, 'sl': 0.015}

# ============================================================
# CARGA DE DATOS
# ============================================================
def load_btc_4h() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / 'BTCUSDT_4h.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    df.set_index('timestamp', inplace=True)
    return df.sort_index()


def load_btc_1h() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / 'btcusdt_1h.parquet')
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df.sort_index()


def load_models() -> dict:
    models = {}
    for mtype in ['context', 'momentum', 'volume']:
        pkl = MODELS_DIR / f'{mtype}_long.pkl'
        if pkl.exists():
            models[mtype] = joblib.load(pkl)
    return models


# ============================================================
# FEATURES 4H
# ============================================================
def compute_features_4h(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    h, l, c, v = df['high'], df['low'], df['close'], df['volume']

    adx_df = pta.adx(h, l, c, length=14)
    if adx_df is not None:
        df['adx']      = adx_df.iloc[:, 0]
        df['di_plus']  = adx_df.iloc[:, 1]
        df['di_minus'] = adx_df.iloc[:, 2]
        df['di_diff']  = df['di_plus'] - df['di_minus']
    else:
        df['adx'] = df['di_diff'] = 20.0

    chop = pta.chop(h, l, c, length=14)
    df['chop'] = chop if chop is not None else 50.0

    for n in [20, 50, 200]:
        df[f'ema{n}'] = pta.ema(c, length=n)
    df['ema20_slope']  = df['ema20'].pct_change(5) * 100
    df['ema50_slope']  = df['ema50'].pct_change(10) * 100
    df['ema200_dist']  = (c - df['ema200']) / df['ema200'] * 100

    atr = pta.atr(h, l, c, length=14)
    df['atr_pct'] = atr / c * 100 if atr is not None else 2.0

    bb = pta.bbands(c, length=20)
    if bb is not None:
        bb_low, bb_mid, bb_up = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
        bb_range = bb_up - bb_low
        df['bb_pct']   = (c - bb_low) / bb_range.replace(0, np.nan)
        df['bb_width'] = (bb_range / bb_mid * 100)
    else:
        df['bb_pct']   = 0.5
        df['bb_width'] = 5.0

    df['rsi14']  = pta.rsi(c, length=14)
    df['rsi7']   = pta.rsi(c, length=7)
    stoch = pta.stoch(h, l, c, k=14, d=3)
    df['stoch_k'] = stoch.iloc[:, 0] if stoch is not None else 50.0

    for n in [5, 20]:
        df[f'ret_{n}'] = c.pct_change(n) * 100

    vol_sma = v.rolling(20).mean()
    df['vol_ratio'] = v / vol_sma.replace(0, np.nan)
    df['vol_trend'] = vol_sma.pct_change(5) * 100
    obv = (np.sign(c.diff()) * v).cumsum()
    df['obv_slope'] = obv.diff(5)

    df.dropna(inplace=True)
    return df


# ============================================================
# FEATURES 1H
# ============================================================
def compute_features_1h(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df['close']
    df['ema20_1h']  = pta.ema(c, length=20)
    df['rsi14_1h']  = pta.rsi(c, length=14)
    df['bullish_1h'] = (c > df['open']).astype(int)
    df.dropna(inplace=True)
    return df


# ============================================================
# DETECCION SETUPS 4H
# ============================================================
def detect_regime(row) -> str:
    if row['ema20'] > row['ema50'] > row['ema200'] and row['ema20_slope'] > 0:
        return 'TREND_UP'
    if row['ema20'] < row['ema50'] and row['ema50_slope'] < 0:
        return 'TREND_DOWN'
    return 'RANGE'


def detect_setups(df: pd.DataFrame) -> list:
    setups = []
    for i, (ts, row) in enumerate(df.iterrows()):
        r = detect_regime(row)
        if r == 'TREND_DOWN':
            continue
        s = None
        if r == 'TREND_UP':
            if row['rsi14'] < 45 and row['bb_pct'] < 0.5 and row['ema200_dist'] > 0:
                s = 'PULLBACK_IN_UPTREND'
            elif row['rsi14'] < 35 and row['bb_pct'] < 0.25:
                s = 'OVERSOLD_IN_UPTREND'
        if s is None:
            if row['rsi14'] < 40 and row['bb_pct'] < 0.3 and row['ema200_dist'] > 0:
                s = 'SUPPORT_BOUNCE'
        if s is None:
            continue
        params = SETUP_PARAMS.get(s, DEFAULT_PARAMS)
        setups.append({
            'bar': i,
            'ts': ts,
            'setup': s,
            'regime': r,
            'close': float(row['close']),
            'tp_pct': params['tp'],
            'sl_pct': params['sl'],
            'rsi14': float(row.get('rsi14', 50)),
            'bb_pct': float(row.get('bb_pct', 0.5)),
        })
    return setups


# ============================================================
# CONFIANZA ML
# ============================================================
CONTEXT_FEATURES  = ['adx', 'di_diff', 'chop', 'atr_pct', 'bb_width']
MOMENTUM_FEATURES = ['rsi14', 'rsi7', 'stoch_k', 'ret_5', 'ret_20']
VOLUME_FEATURES   = ['vol_ratio', 'vol_trend', 'obv_slope']
FEATURE_MAP = {'context': CONTEXT_FEATURES, 'momentum': MOMENTUM_FEATURES,
               'volume': VOLUME_FEATURES}

def get_confidence(row, models: dict) -> float:
    if not models:
        return 0.5
    probs = []
    for mtype, mdata in models.items():
        feats = FEATURE_MAP.get(mtype, [])
        x = np.array([[row.get(f, 0) for f in feats]])
        model = mdata['model']
        if mdata.get('scaler') is not None:
            x = mdata['scaler'].transform(x)
        p = model.predict_proba(x)[0, 1]
        probs.append(p)
    return float(np.mean(probs)) if probs else 0.5


# ============================================================
# SIMULACION DE TRADE
# ============================================================
def sim_trade(df: pd.DataFrame, entry_bar: int, entry_price: float,
              tp_pct: float, sl_pct: float) -> tuple:
    tp_price = entry_price * (1 + tp_pct)
    sl_price = entry_price * (1 - sl_pct)
    for i in range(1, MAX_BARS + 1):
        bar = entry_bar + i
        if bar >= len(df):
            return 'TIMEOUT', float(df.iloc[-1]['close']), i
        brow = df.iloc[bar]
        if brow['high'] >= tp_price:
            return 'TP', tp_price, i
        if brow['low'] <= sl_price:
            return 'SL', sl_price, i
    return 'TIMEOUT', float(df.iloc[entry_bar + MAX_BARS]['close']), MAX_BARS


# ============================================================
# METRICAS
# ============================================================
def metrics(trades: list, label: str) -> dict:
    if not trades:
        return {'label': label, 'n': 0, 'wr': 0, 'pf': 0, 'trades_pm': 0, 'annual': 0}
    tp_n  = sum(1 for t in trades if t['outcome'] == 'TP')
    sl_n  = sum(1 for t in trades if t['outcome'] == 'SL')
    wr    = tp_n / len(trades) if trades else 0
    gross_wins  = sum(t['tp_pct'] for t in trades if t['outcome'] == 'TP')
    gross_losses= sum(t['sl_pct'] for t in trades if t['outcome'] == 'SL')
    pf    = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    months = 48  # 2022-01-01 to 2026-01-01
    trades_pm = len(trades) / months
    avg_pnl = sum((t['tp_pct'] if t['outcome'] == 'TP' else -t['sl_pct']) for t in trades) / len(trades)
    annual  = avg_pnl * trades_pm * 12 * 100
    return {'label': label, 'n': len(trades), 'wr': wr, 'pf': pf,
            'trades_pm': trades_pm, 'annual': annual, 'avg_pnl': avg_pnl}


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 65)
    print("VALIDACION F: CONFIRMACION 1H - BTC V14")
    print(f"OOS: {OOS_START} a {OOS_END}")
    print("=" * 65)

    # Cargar datos
    print("\nCargando datos...")
    df_4h_raw = load_btc_4h()
    df_1h_raw = load_btc_1h()
    models    = load_models()
    print(f"  4H: {len(df_4h_raw)} velas | {df_4h_raw.index[0].date()} - {df_4h_raw.index[-1].date()}")
    print(f"  1H: {len(df_1h_raw)} velas | {df_1h_raw.index[0].date()} - {df_1h_raw.index[-1].date()}")
    print(f"  Modelos ML: {list(models.keys())}")

    # Features
    print("\nCalculando features...")
    df_4h = compute_features_4h(df_4h_raw)
    df_1h = compute_features_1h(df_1h_raw)

    oos_mask = (df_4h.index >= OOS_START) & (df_4h.index <= OOS_END)
    df_oos   = df_4h[oos_mask]
    print(f"  OOS 4H: {len(df_oos)} velas ({df_oos.index[0].date()} - {df_oos.index[-1].date()})")

    # Setups
    print("\nDetectando setups...")
    all_setups = detect_setups(df_oos)
    print(f"  Setups detectados: {len(all_setups)}")
    for s, n in pd.Series([x['setup'] for x in all_setups]).value_counts().items():
        print(f"    {s}: {n}")

    # Aplicar ML confidence filter al baseline
    print("\nAplicando filtro ML (skip_threshold=0.30)...")
    df_4h_indexed = df_4h.reset_index()
    oos_start_bar = df_4h_indexed[df_4h_indexed['timestamp'] >= OOS_START].index[0]

    for s in all_setups:
        global_bar = oos_start_bar + s['bar']
        row_data   = df_oos.iloc[s['bar']].to_dict()
        s['conf']  = get_confidence(row_data, models)

    # Baseline: setups con conf >= threshold
    base_setups = [s for s in all_setups if s['conf'] >= SKIP_THRESHOLD]
    print(f"  Post-ML: {len(base_setups)} setups (filtrados {len(all_setups) - len(base_setups)})")

    # Simular baseline
    baseline_trades = []
    for s in base_setups:
        global_bar = oos_start_bar + s['bar']
        outcome, exit_p, bars = sim_trade(df_4h, global_bar, s['close'], s['tp_pct'], s['sl_pct'])
        baseline_trades.append({
            'setup': s['setup'], 'ts': s['ts'], 'conf': s['conf'],
            'outcome': outcome, 'exit_price': exit_p, 'bars': bars,
            'tp_pct': s['tp_pct'], 'sl_pct': s['sl_pct'],
        })

    m_base = metrics(baseline_trades, 'BASELINE')

    # ============================
    # TEST F: CONFIRMACION 1H
    # Variantes a probar:
    #   F1: precio > EMA20_1H
    #   F2: RSI_1H > 40
    #   F3: F1 AND F2 (ambas condiciones)
    #   F4: F1 OR RSI_1H > 40 (mas permisivo)
    # ============================
    print("\n" + "=" * 65)
    print("TESTS DE CONFIRMACION 1H")
    print("=" * 65)

    def get_1h_state(ts_4h):
        """
        Para un timestamp de vela 4H, busca la ultima vela 1H completa
        anterior (shift de 1H para evitar look-ahead).
        """
        # La vela 4H cierra en ts_4h -> buscamos 1H cerrada justo antes
        ts_1h_prev = ts_4h - pd.Timedelta(hours=1)
        idx = df_1h.index.asof(ts_1h_prev)
        if pd.isnull(idx):
            return None
        row = df_1h.loc[idx]
        return {
            'ema20_1h':   float(row.get('ema20_1h', np.nan)),
            'rsi14_1h':   float(row.get('rsi14_1h', np.nan)),
            'bullish_1h': int(row.get('bullish_1h', 1)),
            'close_1h':   float(row['close']),
        }

    def run_variant(name, condition_fn, label):
        """Simula trades del baseline con filtro 1H adicional."""
        filtered_in  = []
        filtered_out = []
        no_data      = 0

        for s in base_setups:
            state = get_1h_state(s['ts'])
            if state is None or np.isnan(state['ema20_1h']) or np.isnan(state['rsi14_1h']):
                no_data += 1
                filtered_in.append(s)  # Sin datos 1H: no bloquear (safe default)
                continue

            if condition_fn(state):
                filtered_in.append(s)
            else:
                filtered_out.append(s)

        # Simular trades filtrados
        trades_in  = []
        trades_out = []
        for s in filtered_in:
            global_bar = oos_start_bar + s['bar']
            outcome, exit_p, bars = sim_trade(df_4h, global_bar, s['close'], s['tp_pct'], s['sl_pct'])
            trades_in.append({
                'setup': s['setup'], 'ts': s['ts'], 'conf': s['conf'],
                'outcome': outcome, 'exit_price': exit_p, 'bars': bars,
                'tp_pct': s['tp_pct'], 'sl_pct': s['sl_pct'],
            })
        for s in filtered_out:
            global_bar = oos_start_bar + s['bar']
            outcome, exit_p, bars = sim_trade(df_4h, global_bar, s['close'], s['tp_pct'], s['sl_pct'])
            trades_out.append({
                'setup': s['setup'], 'ts': s['ts'], 'conf': s['conf'],
                'outcome': outcome, 'exit_price': exit_p, 'bars': bars,
                'tp_pct': s['tp_pct'], 'sl_pct': s['sl_pct'],
            })

        m_in  = metrics(trades_in,  f'{name} (aceptados)')
        m_out = metrics(trades_out, f'{name} (bloqueados)')

        tp_out = sum(1 for t in trades_out if t['outcome'] == 'TP')
        sl_out = sum(1 for t in trades_out if t['outcome'] == 'SL')
        wr_out = tp_out / len(trades_out) if trades_out else 0

        print(f"\n{label}")
        print(f"  Filtrados: {len(filtered_in)} entran / {len(filtered_out)} bloqueados (sin datos: {no_data})")
        print(f"  Aceptados: N={m_in['n']} | WR={m_in['wr']:.1%} ({m_in['wr']-m_base['wr']:+.1%}pp) | "
              f"PF={m_in['pf']:.2f} ({m_in['pf']-m_base['pf']:+.2f}) | "
              f"anual={m_in['annual']:+.1f}%")
        if trades_out:
            print(f"  Bloqueados: N={len(trades_out)} | WR bloqueados={wr_out:.1%} "
                  f"(si baja < baseline -> filtro BUENO)")

        verdict = 'NEUTRO'
        wr_ok = m_in['wr'] >= m_base['wr'] - 0.01  # tolerancia 1pp
        pf_ok = m_in['pf'] >= m_base['pf'] - 0.05
        n_ok  = m_in['n'] >= m_base['n'] * 0.60    # no pierde mas del 40% de trades
        if (m_in['wr'] > m_base['wr'] + 0.005 or m_in['pf'] > m_base['pf'] + 0.05) and n_ok:
            verdict = 'APROBADO'
        elif not n_ok or (m_in['wr'] < m_base['wr'] - 0.02 and m_in['pf'] < m_base['pf'] - 0.10):
            verdict = 'RECHAZADO'
        print(f"  Veredicto: {verdict}")
        return m_in, verdict

    f1_m, f1_v = run_variant(
        'F1', lambda s: s['close_1h'] > s['ema20_1h'],
        'F1: precio_1H > EMA20_1H  (tendencia alcista en 1H)'
    )
    f2_m, f2_v = run_variant(
        'F2', lambda s: s['rsi14_1h'] > 40,
        'F2: RSI_1H > 40  (no sobrevendido)'
    )
    f3_m, f3_v = run_variant(
        'F3', lambda s: s['close_1h'] > s['ema20_1h'] and s['rsi14_1h'] > 40,
        'F3: F1 AND F2  (mas estricto)'
    )
    f4_m, f4_v = run_variant(
        'F4', lambda s: s['close_1h'] > s['ema20_1h'] or s['rsi14_1h'] > 40,
        'F4: F1 OR F2  (mas permisivo)'
    )
    f5_m, f5_v = run_variant(
        'F5', lambda s: s['bullish_1h'] == 1 and s['rsi14_1h'] > 35,
        'F5: vela_1H bullish AND RSI_1H > 35  (momentum reciente)'
    )

    # Resumen
    print("\n" + "=" * 65)
    print("RESUMEN COMPARATIVO")
    print("=" * 65)
    print(f"{'Variante':<35} | {'N':>5} | {'WR':>7} | {'PF':>6} | {'Veredicto'}")
    print("-" * 70)
    b = m_base
    print(f"{'BASELINE':<35} | {b['n']:>5} | {b['wr']:>6.1%} | {b['pf']:>6.2f} | -")
    for label, m, v in [('F1: precio > EMA20_1H', f1_m, f1_v),
                         ('F2: RSI_1H > 40', f2_m, f2_v),
                         ('F3: F1 AND F2 (estricto)', f3_m, f3_v),
                         ('F4: F1 OR F2 (permisivo)', f4_m, f4_v),
                         ('F5: bullish_1H AND RSI>35', f5_m, f5_v)]:
        wr_d  = m['wr']  - b['wr']
        pf_d  = m['pf']  - b['pf']
        print(f"{label:<35} | {m['n']:>5} | {m['wr']:>6.1%} ({wr_d:+.1%}) | "
              f"{m['pf']:>6.2f} ({pf_d:+.2f}) | {v}")

    print("\n" + "=" * 65)
    print("FIN")
    print("=" * 65)


if __name__ == '__main__':
    main()
