"""
validate_btc_v2_improvements.py
================================
Valida 3 mejoras estructurales al bot BTC V14:

  G. Filtro macro diario      (skip LONG si EMA20_1d < EMA50_1d)
  H. Condiciones setup mas laxas (mas trades, mismo ML filter)
  I. Trailing stop            (SL sigue al precio, no TP fijo)

Baseline: V14 reglas + ML skip<0.30
OOS: 2022-01-01 a 2026-01-31
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pandas_ta as pta
import joblib
from pathlib import Path

PROJECT_ROOT   = Path(__file__).parent
DATA_DIR       = PROJECT_ROOT / 'data'
MODELS_DIR     = PROJECT_ROOT / 'strategies' / 'btc_v14' / 'models'

OOS_START      = '2022-01-01'
OOS_END        = '2026-01-31'
SKIP_THRESHOLD = 0.30
MAX_BARS_BASE  = 12   # baseline: max velas 4H para resolver trade
MAX_BARS_TRAIL = 30   # trailing stop puede durar mas (7.5 dias)

# TP/SL baseline (validados WF)
SETUP_PARAMS = {
    'PULLBACK_IN_UPTREND': {'tp': 0.040, 'sl': 0.015},
    'OVERSOLD_IN_UPTREND': {'tp': 0.030, 'sl': 0.015},
    'SUPPORT_BOUNCE':      {'tp': 0.025, 'sl': 0.012},
    'BREAKOUT_UP':         {'tp': 0.050, 'sl': 0.020},
}
DEFAULT_PARAMS = {'tp': 0.030, 'sl': 0.015}


# ============================================================
# CARGA
# ============================================================
def load_btc_4h() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / 'BTCUSDT_4h.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    df.set_index('timestamp', inplace=True)
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
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
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
    df['ema20_slope'] = df['ema20'].pct_change(5) * 100
    df['ema50_slope'] = df['ema50'].pct_change(10) * 100
    df['ema200_dist'] = (c - df['ema200']) / df['ema200'] * 100

    atr = pta.atr(h, l, c, length=14)
    df['atr_pct'] = atr / c * 100 if atr is not None else 2.0

    bb = pta.bbands(c, length=20)
    if bb is not None:
        bb_low, bb_mid, bb_up = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
        bb_range = bb_up - bb_low
        df['bb_pct']   = (c - bb_low) / bb_range.replace(0, np.nan)
        df['bb_width'] = bb_range / bb_mid * 100
    else:
        df['bb_pct'] = 0.5; df['bb_width'] = 5.0

    df['rsi14'] = pta.rsi(c, length=14)
    df['rsi7']  = pta.rsi(c, length=7)
    stoch = pta.stoch(h, l, c, k=14, d=3)
    df['stoch_k'] = stoch.iloc[:, 0] if stoch is not None else 50.0

    df['ret_5']  = c.pct_change(5) * 100
    df['ret_20'] = c.pct_change(20) * 100

    vol_ma = v.rolling(20).mean()
    df['vol_ratio'] = v / vol_ma.replace(0, np.nan)
    df['vol_trend'] = v.rolling(5).mean() / vol_ma.replace(0, np.nan)
    obv = (np.sign(c.diff()) * v).cumsum()
    df['obv_slope'] = obv.pct_change(10) * 100

    high20 = h.rolling(20).max()
    low20  = l.rolling(20).min()
    rng    = (high20 - low20).replace(0, np.nan)
    df['range_pos'] = (c - low20) / rng

    df['consec_up']   = (c > c.shift(1)).rolling(10).sum()
    df['consec_down'] = (c < c.shift(1)).rolling(10).sum()

    return df.dropna(subset=['adx', 'rsi14', 'bb_pct', 'range_pos'])


# ============================================================
# MACRO DIARIO (resample 4H -> 1D)
# ============================================================
def compute_macro_daily(df_4h: pd.DataFrame) -> pd.Series:
    """
    Retorna Series con fecha -> 'BULL' | 'BEAR' | 'RANGE'
    Shift 1 dia para evitar look-ahead.
    """
    daily = df_4h['close'].resample('1D').last().dropna()
    ema20d  = daily.ewm(span=20, adjust=False).mean()
    ema50d  = daily.ewm(span=50, adjust=False).mean()
    ema200d = daily.ewm(span=200, adjust=False).mean()

    regime = pd.Series('RANGE', index=daily.index)
    regime[ema20d > ema50d]  = 'BULL'
    regime[ema20d < ema50d]  = 'BEAR'

    # Shift 1 dia: hoy usamos el regimen de AYER (no look-ahead)
    regime = regime.shift(1).dropna()
    return regime


# ============================================================
# REGIMEN Y SETUPS 4H
# ============================================================
def detect_regime_4h(row) -> str:
    adx      = row.get('adx', 20)
    di_diff  = row.get('di_diff', 0)
    chop     = row.get('chop', 50)
    atr_pct  = row.get('atr_pct', 2)
    bb_width = row.get('bb_width', 5)
    e20s     = row.get('ema20_slope', 0)
    e50s     = row.get('ema50_slope', 0)

    if atr_pct > 4 and bb_width > 8:
        return 'VOLATILE'
    if adx > 25 and chop < 50:
        if di_diff > 5 and e20s > 0:
            return 'TREND_UP'
        if di_diff < -5 and e20s < 0:
            return 'TREND_DOWN'
    if chop > 55 or adx < 20:
        return 'RANGE'
    if e50s > 0.5:
        return 'TREND_UP'
    if e50s < -0.5:
        return 'TREND_DOWN'
    return 'RANGE'


def detect_setup_strict(row, regime: str):
    """Condiciones originales V14 (baseline)."""
    rsi    = row.get('rsi14', 50)
    bb_pct = row.get('bb_pct', 0.5)
    rp     = row.get('range_pos', 0.5)
    e200d  = row.get('ema200_dist', 0)
    e20_val= row.get('ema20', row.get('close', 1))
    e20d   = (row.get('close', 0) - e20_val) / max(e20_val, 1) * 100
    vr     = row.get('vol_ratio', 1)
    cup    = row.get('consec_up', 0)

    if regime == 'TREND_UP':
        if rsi < 40 and bb_pct < 0.3 and e200d > 0:
            return 'PULLBACK_IN_UPTREND'
        if rsi < 30 and e20d < -2:
            return 'OVERSOLD_IN_UPTREND'
    elif regime == 'RANGE':
        if rp < 0.2 and rsi < 35:
            return 'SUPPORT_BOUNCE'
    elif regime == 'VOLATILE':
        if bb_pct > 1.0 and vr > 1.5 and cup >= 3:
            return 'BREAKOUT_UP'
    return None


def detect_setup_relaxed(row, regime: str):
    """
    Condiciones mas laxas (H):
    - PULLBACK: RSI<45 (era <40), bb_pct<0.4 (era <0.3)
    - SUPPORT:  range_pos<0.30 (era <0.20), RSI<42 (era <35)
    - OVERSOLD: sin cambio (ya es estricto)
    """
    rsi    = row.get('rsi14', 50)
    bb_pct = row.get('bb_pct', 0.5)
    rp     = row.get('range_pos', 0.5)
    e200d  = row.get('ema200_dist', 0)
    e20_val= row.get('ema20', row.get('close', 1))
    e20d   = (row.get('close', 0) - e20_val) / max(e20_val, 1) * 100
    vr     = row.get('vol_ratio', 1)
    cup    = row.get('consec_up', 0)

    if regime == 'TREND_UP':
        if rsi < 45 and bb_pct < 0.4 and e200d > 0:
            return 'PULLBACK_IN_UPTREND'
        if rsi < 30 and e20d < -2:
            return 'OVERSOLD_IN_UPTREND'
    elif regime == 'RANGE':
        if rp < 0.30 and rsi < 42:
            return 'SUPPORT_BOUNCE'
    elif regime == 'VOLATILE':
        if bb_pct > 1.0 and vr > 1.5 and cup >= 3:
            return 'BREAKOUT_UP'
    return None


def build_setups(df_oos, detect_fn) -> list:
    setups = []
    for i in range(len(df_oos)):
        row = df_oos.iloc[i]
        r   = detect_regime_4h(row)
        s   = detect_fn(row, r)
        if s:
            params = SETUP_PARAMS.get(s, DEFAULT_PARAMS)
            setups.append({
                'bar':    i,
                'ts':     df_oos.index[i],
                'setup':  s,
                'regime': r,
                'close':  float(row['close']),
                'tp_pct': params['tp'],
                'sl_pct': params['sl'],
            })
    return setups


# ============================================================
# CONFIANZA ML
# ============================================================
CONTEXT_FEATURES  = ['adx', 'di_diff', 'chop', 'atr_pct', 'bb_width']
MOMENTUM_FEATURES = ['rsi14', 'rsi7', 'stoch_k', 'ret_5', 'ret_20']
VOLUME_FEATURES   = ['vol_ratio', 'vol_trend', 'obv_slope']
FEATURE_MAP = {
    'context':  CONTEXT_FEATURES,
    'momentum': MOMENTUM_FEATURES,
    'volume':   VOLUME_FEATURES,
}

def get_confidence(row_dict, models: dict) -> float:
    if not models:
        return 0.5
    probs = []
    for mtype, mdata in models.items():
        feats = FEATURE_MAP[mtype]
        x     = np.array([[row_dict.get(f, 0) for f in feats]])
        if mdata.get('scaler') is not None:
            x = mdata['scaler'].transform(x)
        probs.append(mdata['model'].predict_proba(x)[0, 1])
    return float(np.mean(probs)) if probs else 0.5


# ============================================================
# SIMULACION
# ============================================================
def sim_fixed_tp(df_4h, entry_bar, entry_price, tp_pct, sl_pct,
                 max_bars=MAX_BARS_BASE):
    """Trade con TP y SL fijos."""
    tp = entry_price * (1 + tp_pct)
    sl = entry_price * (1 - sl_pct)
    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df_4h):
            return 'TIMEOUT', float(df_4h['close'].iloc[-1]), i
        hi = float(df_4h['high'].iloc[b])
        lo = float(df_4h['low'].iloc[b])
        if lo <= sl:
            if hi >= tp and df_4h['close'].iloc[b] > (sl + tp) / 2:
                return 'TP', tp, i
            return 'SL', sl, i
        if hi >= tp:
            return 'TP', tp, i
    return 'TIMEOUT', float(df_4h['close'].iloc[entry_bar + max_bars]), max_bars


def sim_trailing(df_4h, entry_bar, entry_price, sl_pct,
                 trail_trigger_pct=None, max_bars=MAX_BARS_TRAIL):
    """
    Trailing stop sin TP fijo.
    - trail_trigger: precio sube X% desde entry -> activar trailing
    - trailing SL = peak_price * (1 - sl_pct)
    Si no se especifica trail_trigger, el trailing es inmediato desde bar 1.
    """
    sl_price = entry_price * (1 - sl_pct)
    peak     = entry_price
    trailing_active = (trail_trigger_pct is None)

    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df_4h):
            exit_p = float(df_4h['close'].iloc[-1])
            outcome = 'TP' if exit_p > entry_price else 'SL'
            return outcome, exit_p, i
        hi = float(df_4h['high'].iloc[b])
        lo = float(df_4h['low'].iloc[b])
        cl = float(df_4h['close'].iloc[b])

        # Actualizar peak
        if hi > peak:
            peak = hi

        # Activar trailing si precio sube suficiente
        if not trailing_active and trail_trigger_pct is not None:
            if hi >= entry_price * (1 + trail_trigger_pct):
                trailing_active = True

        # Calcular SL efectivo
        if trailing_active:
            sl_price = max(sl_price, peak * (1 - sl_pct))

        if lo <= sl_price:
            outcome = 'TP' if sl_price > entry_price else 'SL'
            return outcome, sl_price, i

    # Timeout: cerrar al cierre
    exit_p  = float(df_4h['close'].iloc[entry_bar + max_bars])
    outcome = 'TP' if exit_p > entry_price else 'SL'
    return outcome, exit_p, max_bars


# ============================================================
# METRICAS
# ============================================================
def metrics(trades, label):
    if not trades:
        return {'label': label, 'n': 0, 'wr': 0, 'pf': 0,
                'avg_r': 0, 'trades_pm': 0, 'annual_pct': 0}
    n   = len(trades)
    wins = [t for t in trades if t['outcome'] == 'TP']
    loss = [t for t in trades if t['outcome'] == 'SL']
    wr   = len(wins) / n

    # R-multiples (ganancia / riesgo en %)
    r_wins  = sum(t['pnl_pct'] for t in wins)
    r_loss  = sum(abs(t['pnl_pct']) for t in loss)
    pf      = r_wins / r_loss if r_loss > 0 else float('inf')

    avg_r   = sum(t['pnl_pct'] for t in trades) / n
    months  = 48  # OOS 2022-2026
    tpm     = n / months
    # Retorno anual estimado (riesgo fijo 2% por trade)
    annual  = avg_r / 0.02 * 2 * tpm * 12   # avg_r como % del precio
    return {
        'label': label, 'n': n, 'wr': wr, 'pf': pf,
        'avg_r': avg_r, 'trades_pm': tpm, 'annual_pct': annual,
    }


def print_row(m, baseline=None):
    b = baseline
    wr_d  = f"({m['wr']-b['wr']:+.1%})"  if b else ''
    pf_d  = f"({m['pf']-b['pf']:+.2f})"  if b else ''
    ann_d = f"({m['annual_pct']-b['annual_pct']:+.0f}%)" if b else ''
    print(f"  {m['label']:<40} | N={m['n']:>4} | "
          f"WR={m['wr']:.1%}{wr_d:<9} | "
          f"PF={m['pf']:.2f}{pf_d:<8} | "
          f"~{m['annual_pct']:.0f}%/yr {ann_d}")


def verdict(m, b):
    n_ok  = m['n'] >= b['n'] * 0.60
    wr_up = m['wr']  > b['wr']  + 0.005
    pf_up = m['pf']  > b['pf']  + 0.05
    wr_ok = m['wr']  >= b['wr'] - 0.01
    pf_ok = m['pf']  >= b['pf'] - 0.05
    if n_ok and (wr_up or pf_up):
        return 'APROBADO'
    if not n_ok or (not wr_ok and not pf_ok):
        return 'RECHAZADO'
    return 'NEUTRO'


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("VALIDACION V2: MACRO FILTER + SETUPS LAXOS + TRAILING STOP")
    print(f"OOS: {OOS_START} a {OOS_END}")
    print("=" * 70)

    # ----- Cargar -----
    print("\nCargando datos y calculando features...")
    df_raw  = load_btc_4h()
    models  = load_models()
    df_feat = compute_features(df_raw)
    macro_d = compute_macro_daily(df_raw)

    oos_mask = (df_feat.index >= OOS_START) & (df_feat.index <= OOS_END)
    df_oos   = df_feat[oos_mask].copy()

    # Bar offset para sim_trade (necesita idx en df_feat completo)
    oos_start_bar = df_feat.index.get_loc(df_oos.index[0])
    print(f"  OOS: {len(df_oos)} velas | modelos: {list(models.keys())}")

    # Macro diario -> mapear a cada vela 4H (asof lookup)
    macro_tz = macro_d.copy()
    if macro_tz.index.tz is None:
        macro_tz.index = macro_tz.index.tz_localize('UTC')

    def get_macro(ts):
        idx = macro_tz.index.asof(ts)
        if pd.isnull(idx):
            return 'RANGE'
        return macro_tz.loc[idx]

    # ----- Construir setups -----
    print("\nDetectando setups (strict y relaxed)...")
    setups_strict  = build_setups(df_oos, detect_setup_strict)
    setups_relaxed = build_setups(df_oos, detect_setup_relaxed)
    print(f"  Strict : {len(setups_strict)} setups")
    print(f"  Relaxed: {len(setups_relaxed)} setups "
          f"(+{len(setups_relaxed)-len(setups_strict)} adicionales)")

    # Aplicar ML confidence a todos los setups
    for s in setups_strict + setups_relaxed:
        row_d    = df_oos.iloc[s['bar']].to_dict()
        s['conf'] = get_confidence(row_d, models)
        s['macro'] = get_macro(s['ts'])

    # Filtrar por ML
    base_setups = [s for s in setups_strict  if s['conf'] >= SKIP_THRESHOLD]
    rela_setups = [s for s in setups_relaxed if s['conf'] >= SKIP_THRESHOLD]
    print(f"  Post-ML strict : {len(base_setups)} "
          f"(filtrados {len(setups_strict)-len(base_setups)})")
    print(f"  Post-ML relaxed: {len(rela_setups)} "
          f"(filtrados {len(setups_relaxed)-len(rela_setups)})")

    # ----- Helper: simular lista de setups -----
    def run_fixed(setups, label, extra_filter=None):
        trades = []
        for s in setups:
            if extra_filter and not extra_filter(s):
                continue
            gb = oos_start_bar + s['bar']
            outcome, exit_p, bars = sim_fixed_tp(
                df_feat, gb, s['close'], s['tp_pct'], s['sl_pct'])
            pnl = s['tp_pct'] if outcome == 'TP' else (
                  -s['sl_pct'] if outcome == 'SL' else
                  (exit_p - s['close']) / s['close'])
            trades.append({'outcome': outcome, 'pnl_pct': pnl,
                           'setup': s['setup'], 'ts': s['ts']})
        return metrics(trades, label)

    def run_trail(setups, label, trigger_pct, extra_filter=None):
        trades = []
        for s in setups:
            if extra_filter and not extra_filter(s):
                continue
            gb = oos_start_bar + s['bar']
            outcome, exit_p, bars = sim_trailing(
                df_feat, gb, s['close'], s['sl_pct'],
                trail_trigger_pct=trigger_pct)
            pnl = (exit_p - s['close']) / s['close']
            if outcome == 'SL' and pnl > 0:
                outcome = 'TP'  # trailing salio en ganancia
            trades.append({'outcome': outcome, 'pnl_pct': pnl,
                           'setup': s['setup'], 'ts': s['ts']})
        return metrics(trades, label)

    # Filtros macro
    no_bear   = lambda s: s['macro'] != 'BEAR'
    only_bull = lambda s: s['macro'] == 'BULL'

    # =========================================================
    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print(f"\n{'Escenario':<42} | {'N':>4} | {'WR':>16} | {'PF':>12} | Anual est.")
    print("-" * 80)

    # --- BASELINE ---
    m_base = run_fixed(base_setups, 'BASELINE (strict + ML)')
    print_row(m_base)
    print()

    # --- G: FILTRO MACRO ---
    print("G. FILTRO MACRO DIARIO (EMA20d vs EMA50d)")
    m_g1 = run_fixed(base_setups, 'G1: skip BEAR',    extra_filter=no_bear)
    m_g2 = run_fixed(base_setups, 'G2: solo BULL',    extra_filter=only_bull)
    print_row(m_g1, m_base)
    print_row(m_g2, m_base)
    v_g1, v_g2 = verdict(m_g1, m_base), verdict(m_g2, m_base)
    print(f"    -> G1 skip BEAR: {v_g1} | G2 solo BULL: {v_g2}")

    # Analisis: cuantos setups son BEAR, BULL, RANGE
    macro_dist = pd.Series([s['macro'] for s in base_setups]).value_counts()
    print(f"    Distribucion macro en setups: {macro_dist.to_dict()}")
    bear_wr = sum(1 for s in base_setups if s['macro']=='BEAR' and
                  run_fixed([s], '')['wr']==1) / max(1, sum(1 for s in base_setups if s['macro']=='BEAR'))
    # Simple: WR de los que SERIAN bloqueados por G1
    bear_set = [s for s in base_setups if s['macro'] == 'BEAR']
    if bear_set:
        m_bear = run_fixed(bear_set, 'solo BEAR')
        print(f"    WR de setups BEAR (bloqueados por G1): {m_bear['wr']:.1%} "
              f"(si < baseline -> filtro util)")
    print()

    # --- H: SETUPS LAXOS ---
    print("H. CONDICIONES DE SETUP LAXAS")
    m_h1 = run_fixed(rela_setups, 'H1: relaxed (sin macro)')
    m_h2 = run_fixed(rela_setups, 'H2: relaxed + skip BEAR', extra_filter=no_bear)
    print_row(m_h1, m_base)
    print_row(m_h2, m_base)
    v_h1, v_h2 = verdict(m_h1, m_base), verdict(m_h2, m_base)
    print(f"    -> H1 relaxed: {v_h1} | H2 relaxed+macro: {v_h2}")

    # Setups nuevos (en relaxed pero no en strict)
    strict_ts = {s['ts'] for s in base_setups}
    new_setups = [s for s in rela_setups if s['ts'] not in strict_ts]
    if new_setups:
        m_new = run_fixed(new_setups, 'solo setups nuevos (H)')
        print(f"    WR de setups NUEVOS (adicionales): {m_new['wr']:.1%} | "
              f"N={m_new['n']} -> "
              f"{'buenos' if m_new['wr'] >= m_base['wr']-0.03 else 'peores que baseline'}")
    print()

    # --- I: TRAILING STOP ---
    print("I. TRAILING STOP (sin TP fijo, SL sigue al precio)")
    # Variantes: trigger 0% (inmediato), 1%, 1.5%, 2%
    for trigger in [None, 0.010, 0.015, 0.020]:
        lbl = f"I-trail (trigger={'inmediato' if trigger is None else f'{trigger:.0%}'})"
        m_t = run_trail(base_setups, lbl, trigger_pct=trigger)
        print_row(m_t, m_base)
    best_trail_trigger = 0.015  # candidato inicial
    v_i = verdict(run_trail(base_setups, '', trigger_pct=best_trail_trigger), m_base)
    print(f"    -> Trailing (trigger=1.5%): {v_i}")
    print()

    # --- COMBINACIONES ---
    print("MEJORES COMBINACIONES")
    combos = []

    # Combo 1: mejor macro + baseline
    best_macro_filter = no_bear if v_g1 != 'RECHAZADO' else only_bull
    m_c1 = run_fixed(base_setups, 'C1: strict + skip BEAR',
                     extra_filter=no_bear)
    combos.append(('C1: strict + skip BEAR', m_c1))

    # Combo 2: relaxed + macro
    m_c2 = run_fixed(rela_setups, 'C2: relaxed + skip BEAR',
                     extra_filter=no_bear)
    combos.append(('C2: relaxed + skip BEAR', m_c2))

    # Combo 3: strict + macro + trailing
    m_c3 = run_trail(base_setups, 'C3: strict + macro + trail 1.5%',
                     trigger_pct=0.015, extra_filter=no_bear)
    combos.append(('C3: strict + macro + trail 1.5%', m_c3))

    # Combo 4: relaxed + macro + trailing
    m_c4 = run_trail(rela_setups, 'C4: relaxed + macro + trail 1.5%',
                     trigger_pct=0.015, extra_filter=no_bear)
    combos.append(('C4: relaxed + macro + trail 1.5%', m_c4))

    for name, m in combos:
        print_row(m, m_base)
    print()

    # --- RESUMEN VEREDICTOS ---
    print("=" * 70)
    print("VEREDICTOS FINALES")
    print("=" * 70)
    all_results = [
        ('G1: Macro filter (skip BEAR)',         m_g1, v_g1),
        ('G2: Macro filter (solo BULL)',          m_g2, v_g2),
        ('H1: Setups laxos',                      m_h1, v_h1),
        ('H2: Setups laxos + skip BEAR',          m_h2, v_h2),
        ('I:  Trailing stop (1.5% trigger)',
         run_trail(base_setups,'',trigger_pct=0.015), v_i),
        ('C2: Relaxed + skip BEAR (combo)',       m_c2, verdict(m_c2, m_base)),
        ('C3: Strict + macro + trail (combo)',    m_c3, verdict(m_c3, m_base)),
        ('C4: Relaxed + macro + trail (combo)',   m_c4, verdict(m_c4, m_base)),
    ]
    for name, m, v_ in all_results:
        mark = {'APROBADO': '+', 'RECHAZADO': 'x', 'NEUTRO': '~'}[v_]
        print(f"  [{mark}] {name:<45} WR={m['wr']:.1%} PF={m['pf']:.2f} "
              f"~{m['annual_pct']:.0f}%/yr -> {v_}")

    print("\n" + "=" * 70)
    print("FIN")
    print("=" * 70)


if __name__ == '__main__':
    main()
