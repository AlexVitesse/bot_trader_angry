"""
validate_trader_improvements.py
================================
Valida sobre datos historicos BTC (OOS 2022-2026) los 6 cambios "trader real":

  A. Filtro funding rate  (skip LONG si funding > +0.04%)
  B. Break-even stop      (SL -> entry cuando profit >= 50% de SL_dist)
  C. Racha de perdidas    (pausa tras 3 SL consecutivos)
  D. Sizing por confianza (1.0x / 1.2x / 1.4x segun conf ML)
  E. Limit order entrada  (SUPPORT_BOUNCE entra 0.2% bajo cierre)
  F. Confirmacion 1H      (requiere datos 1H - reporta fill rate aproximada)

Baseline: V14 BTC reglas + ML skip<0.30 (7/12 folds, WR 38%, PF 1.85)
Criterio aprobacion: mejora WR o PF sin reducir trades > 30%
"""

import sys
import numpy as np
import pandas as pd
import pandas_ta as pta
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / 'data'
MODELS_DIR   = PROJECT_ROOT / 'strategies' / 'btc_v14' / 'models'

OOS_START   = '2022-01-01'
OOS_END     = '2026-01-31'
MAX_BARS    = 12            # Max velas 4H para resolver trade
SKIP_THRESHOLD = 0.30       # ML confidence minima (validada)

# TP/SL por setup (validados walk-forward)
SETUP_PARAMS = {
    'SUPPORT_BOUNCE':      {'tp': 0.025, 'sl': 0.012},
    'PULLBACK_IN_UPTREND': {'tp': 0.040, 'sl': 0.015},
    'OVERSOLD_IN_UPTREND': {'tp': 0.030, 'sl': 0.015},
    'BREAKOUT_UP':         {'tp': 0.050, 'sl': 0.020},
}
DEFAULT_PARAMS = {'tp': 0.030, 'sl': 0.015}

# Umbrales de los filtros a validar
FUNDING_LONG_MAX  = 0.0004   # +0.04%
MAX_CONSEC_LOSSES = 3
PAUSE_BARS        = 42       # ~1 semana en 4H despues de pausa

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


def load_funding_4h() -> pd.Series:
    """Resamplea funding rate 8H a 4H con forward-fill."""
    f = pd.read_parquet(DATA_DIR / 'btc_v15_funding.parquet')
    return f['funding_rate'].resample('4h').ffill()


def load_models() -> dict:
    models = {}
    for mtype in ['context', 'momentum', 'volume']:
        pkl = MODELS_DIR / f'{mtype}_long.pkl'
        if pkl.exists():
            models[mtype] = joblib.load(pkl)
    return models


# ============================================================
# FEATURES (identico a validate_btc_v14_walkforward.py)
# ============================================================
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    h, l, c, v = df['high'], df['low'], df['close'], df['volume']

    adx_df = pta.adx(h, l, c, length=14)
    if adx_df is not None:
        df['adx']     = adx_df.iloc[:, 0]
        df['di_plus'] = adx_df.iloc[:, 1]
        df['di_minus']= adx_df.iloc[:, 2]
        df['di_diff'] = df['di_plus'] - df['di_minus']
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
        df['bb_width'] = (bb_range / bb_mid * 100)
    else:
        df['bb_pct'] = 0.5
        df['bb_width'] = 5.0

    df['rsi14']  = pta.rsi(c, length=14)
    df['rsi7']   = pta.rsi(c, length=7)
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
# REGIMEN Y SETUPS (identico al walk-forward)
# ============================================================
def detect_regime(row) -> str:
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


def detect_setup(row, regime: str):
    rsi      = row.get('rsi14', 50)
    bb_pct   = row.get('bb_pct', 0.5)
    rp       = row.get('range_pos', 0.5)
    e200d    = row.get('ema200_dist', 0)
    e20d     = (row.get('close', 0) - row.get('ema20', row.get('close', 0))) / max(row.get('ema20', 1), 1) * 100
    vr       = row.get('vol_ratio', 1)
    cup      = row.get('consec_up', 0)
    cdn      = row.get('consec_down', 0)

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


def detect_all_setups(df: pd.DataFrame) -> list:
    setups = []
    for i in range(len(df)):
        row = df.iloc[i]
        r = detect_regime(row)
        s = detect_setup(row, r)
        if s:
            setups.append({
                'bar': i,
                'ts': df.index[i],
                'setup': s,
                'regime': r,
                'close': float(row['close']),
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
# SIMULACION DE UN TRADE (bar por bar, con break-even)
# ============================================================
def sim_trade(df: pd.DataFrame, entry_bar: int, entry_price: float,
              tp_pct: float, sl_pct: float,
              use_breakeven: bool = False) -> tuple:
    """
    Retorna (outcome, exit_price, bars_held)
    outcome: 'TP' | 'SL' | 'BE' | 'TIMEOUT'
    BE = break-even (salio en entrada, no es perdida)
    """
    tp_price  = entry_price * (1 + tp_pct)
    sl_price  = entry_price * (1 - sl_pct)
    cur_sl    = sl_price
    be_active = False

    for i in range(1, MAX_BARS + 1):
        bar = entry_bar + i
        if bar >= len(df):
            return 'TIMEOUT', float(df['close'].iloc[-1]), i

        high  = float(df['high'].iloc[bar])
        low   = float(df['low'].iloc[bar])
        close = float(df['close'].iloc[bar])

        # Break-even: cuando profit >= 50% de SL_dist, mover SL a entry
        if use_breakeven and not be_active:
            be_trigger = entry_price + (entry_price - sl_price) * 0.5
            if high >= be_trigger:
                cur_sl    = entry_price
                be_active = True

        # SL check (primero, conservador)
        if low <= cur_sl:
            if high >= tp_price and close > (cur_sl + tp_price) / 2:
                return 'TP', tp_price, i   # Ambiguidad -> TP si close bullish
            label = 'BE' if be_active else 'SL'
            return label, cur_sl, i

        # TP check
        if high >= tp_price:
            return 'TP', tp_price, i

    return 'TIMEOUT', float(df['close'].iloc[entry_bar + MAX_BARS]), MAX_BARS


# ============================================================
# SIMULACION COMPLETA
# ============================================================
def simulate(df: pd.DataFrame, setups: list, models: dict,
             funding: pd.Series = None,
             use_funding:    bool = False,
             use_breakeven:  bool = False,
             use_consec:     bool = False,
             use_sizing:     bool = False,
             use_limit:      bool = False) -> dict:

    trades = []
    consec_losses = 0
    paused_until  = -1   # bar index hasta el cual estamos pausados

    for s in setups:
        bar         = s['bar']
        entry_price = s['close']
        setup_name  = s['setup']
        row         = df.iloc[bar]

        # Confianza ML (pre-computada o en vivo)
        conf = get_confidence(row.to_dict(), models)
        if conf < SKIP_THRESHOLD:
            trades.append({**s, 'outcome': 'SKIP_ML', 'conf': conf, 'pnl': 0.0})
            continue

        # C. Pausa por racha de perdidas
        if use_consec and bar <= paused_until:
            params = SETUP_PARAMS.get(setup_name, DEFAULT_PARAMS)
            out, ep, bh = sim_trade(df, bar, entry_price, params['tp'], params['sl'])
            trades.append({**s, 'outcome': 'PAUSED_' + out, 'conf': conf, 'pnl': 0.0})
            continue

        # A. Filtro funding rate (solo LONG, solo SUPPORT_BOUNCE y similares)
        if use_funding and funding is not None:
            ts = df.index[bar]
            try:
                rate = funding.asof(ts)
                if pd.notna(rate) and rate > FUNDING_LONG_MAX:
                    trades.append({**s, 'outcome': 'SKIP_FUNDING', 'conf': conf,
                                   'pnl': 0.0, 'funding': float(rate)})
                    continue
            except Exception:
                pass

        params   = SETUP_PARAMS.get(setup_name, DEFAULT_PARAMS)
        tp_pct   = params['tp']
        sl_pct   = params['sl']

        # E. Limit order: SUPPORT_BOUNCE entra 0.2% bajo cierre actual
        effective_entry = entry_price
        if use_limit and setup_name == 'SUPPORT_BOUNCE':
            limit_price = entry_price * 0.998
            next_bar    = bar + 1
            if next_bar >= len(df):
                trades.append({**s, 'outcome': 'LIMIT_NOFILL', 'conf': conf, 'pnl': 0.0})
                continue
            if float(df['low'].iloc[next_bar]) <= limit_price:
                effective_entry = limit_price   # Fill!
            else:
                trades.append({**s, 'outcome': 'LIMIT_NOFILL', 'conf': conf, 'pnl': 0.0})
                continue

        # D. Sizing por confianza
        if use_sizing:
            sz = 1.4 if conf >= 0.45 else (1.2 if conf >= 0.40 else 1.0)
        else:
            sz = 1.0

        # Simular trade bar por bar
        outcome, exit_price, bars_held = sim_trade(
            df, bar, effective_entry, tp_pct, sl_pct,
            use_breakeven=use_breakeven
        )

        # PnL en unidades de "1 riesgo"
        if outcome == 'TP':
            pnl = (tp_pct / sl_pct) * sz
        elif outcome == 'SL':
            pnl = -1.0 * sz
        elif outcome == 'BE':
            pnl = 0.0
        else:  # TIMEOUT
            raw_ret = (exit_price - effective_entry) / effective_entry
            pnl = (raw_ret / sl_pct) * sz

        # Racha de perdidas
        if use_consec:
            if pnl < 0:
                consec_losses += 1
                if consec_losses >= MAX_CONSEC_LOSSES:
                    paused_until  = bar + PAUSE_BARS
                    consec_losses = 0
            else:
                consec_losses = 0

        trades.append({
            **s,
            'outcome':    outcome,
            'exit_price': exit_price,
            'bars_held':  bars_held,
            'conf':       conf,
            'sizing':     sz,
            'pnl':        pnl,
        })

    return trades


# ============================================================
# METRICAS
# ============================================================
def metrics(trades: list, label: str) -> dict:
    skip_outs = {'SKIP_ML', 'SKIP_FUNDING', 'LIMIT_NOFILL'} | \
                {t['outcome'] for t in trades if t['outcome'].startswith('PAUSED')}

    executed  = [t for t in trades if t['outcome'] not in skip_outs]
    filtered  = [t for t in trades if t['outcome'] in skip_outs]
    skip_fund = [t for t in trades if t['outcome'] == 'SKIP_FUNDING']
    nofill    = [t for t in trades if t['outcome'] == 'LIMIT_NOFILL']
    paused_t  = [t for t in trades if t['outcome'].startswith('PAUSED')]

    if not executed:
        return {'label': label, 'n': 0, 'wr': 0, 'pf': 0, 'avg_pnl': 0}

    wins   = [t for t in executed if t['pnl'] > 0]
    losses = [t for t in executed if t['pnl'] < 0]
    be_    = [t for t in executed if t['pnl'] == 0 and t['outcome'] == 'BE']

    wr     = len(wins) / len(executed)
    gw     = sum(t['pnl'] for t in wins)
    gl     = abs(sum(t['pnl'] for t in losses))
    pf     = gw / gl if gl > 0 else float('inf')
    avg    = sum(t['pnl'] for t in executed) / len(executed)

    # Anual estimado (36 trades/año baseline)
    n_months = (pd.Timestamp(OOS_END) - pd.Timestamp(OOS_START)).days / 30.44
    tpm      = len(executed) / n_months
    annual   = avg * tpm * 12  # en unidades de riesgo/año

    return {
        'label':       label,
        'n':           len(executed),
        'n_filtered':  len(filtered),
        'n_skip_fund': len(skip_fund),
        'n_nofill':    len(nofill),
        'n_paused':    len(paused_t),
        'n_be':        len(be_),
        'wr':          wr,
        'pf':          pf,
        'avg_pnl':     avg,
        'tpm':         tpm,
        'annual_units': annual,
    }


def print_row(m: dict, baseline: dict = None):
    b = baseline or m
    dwr  = (m['wr']  - b['wr'])  * 100
    dpf  = m['pf']  - b['pf']
    dn   = m['n']   - b['n']
    extra = ''
    if m['n_be'] > 0:
        extra += f"  BE={m['n_be']}"
    if m['n_skip_fund'] > 0:
        extra += f"  SkipFund={m['n_skip_fund']}"
    if m['n_nofill'] > 0:
        extra += f"  NoFill={m['n_nofill']}"
    if m['n_paused'] > 0:
        extra += f"  Paused={m['n_paused']}"
    print(
        f"  {m['label']:<30} | "
        f"N={m['n']:>4} ({dn:>+4}) | "
        f"WR={m['wr']:.1%} ({dwr:>+5.1f}pp) | "
        f"PF={m['pf']:.2f} ({dpf:>+5.2f}) | "
        f"avg={m['avg_pnl']:>+.3f}"
        + extra
    )


def verdict(m: dict, baseline: dict) -> str:
    wr_ok  = m['wr'] >= baseline['wr'] - 0.01      # no baja WR mas de 1pp
    pf_ok  = m['pf'] >= baseline['pf'] - 0.05      # no baja PF mas de 0.05
    n_ok   = m['n']  >= baseline['n'] * 0.70        # no filtra mas del 30%
    better = m['wr'] > baseline['wr'] + 0.005 or m['pf'] > baseline['pf'] + 0.05
    if better and wr_ok and pf_ok and n_ok:
        return 'APROBADO'
    if wr_ok and pf_ok and n_ok:
        return 'NEUTRO (no mejora, no empeora)'
    return 'RECHAZADO'


# ============================================================
# ANALISIS DE CALIDAD POR SETUP FILTRADO
# ============================================================
def analyze_filtered(trades_base: list, trades_filt: list, filt_key: str):
    """Muestra WR de los trades que el filtro elimino."""
    base_map = {t['bar']: t for t in trades_base if t['outcome'] not in ('SKIP_ML',)}
    skipped  = [t for t in trades_filt if t['outcome'] == filt_key]
    if not skipped:
        return
    wins_sk  = sum(1 for t in skipped if base_map.get(t['bar'], {}).get('pnl', 0) > 0)
    total_sk = len(skipped)
    wr_sk    = wins_sk / total_sk if total_sk else 0
    print(f"    Trades filtrados por {filt_key}: {total_sk} | "
          f"WR si hubieran entrado: {wr_sk:.1%}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 65)
    print("VALIDACION DE MEJORAS 'TRADER REAL' - BTC V14")
    print(f"Periodo OOS: {OOS_START} a {OOS_END}")
    print("=" * 65)

    # --- Cargar datos ---
    print("\nCargando datos...")
    df_raw   = load_btc_4h()
    funding  = load_funding_4h()
    models   = load_models()
    print(f"  BTC 4H: {len(df_raw)} velas | {df_raw.index.min().date()} - {df_raw.index.max().date()}")
    print(f"  Funding: {len(funding)} registros 4H")
    print(f"  Modelos ML: {list(models.keys())}")

    # --- Calcular features ---
    print("\nCalculando features...")
    df = compute_features(df_raw)

    # --- Filtrar OOS ---
    df_oos = df[OOS_START:OOS_END].copy()
    df_oos = df_oos.reset_index(drop=False)  # bar index numerico
    print(f"  OOS: {len(df_oos)} velas ({OOS_START} - {OOS_END})")

    # Necesitamos el df_oos con indices numericos pero el timestamp para el funding
    timestamps = df_oos['timestamp'].copy() if 'timestamp' in df_oos.columns else df_oos.index

    # Convertir timestamps del funding al mismo TZ
    funding.index = funding.index.tz_localize(None) if funding.index.tzinfo else funding.index
    # Crear lookup rapido: para cada bar del OOS, el funding rate correspondiente
    df_oos_ts = df_oos.set_index('timestamp') if 'timestamp' in df_oos.columns else df_oos
    df_oos_for_sim = df_oos  # ya tiene indice numerico

    # --- Detectar setups ---
    print("\nDetectando setups...")
    df_feat = df_oos.copy()
    # Agregar close al dict para detect_setup
    setups_all = []
    for i in range(len(df_feat)):
        row_dict = df_feat.iloc[i].to_dict()
        r = detect_regime(row_dict)
        s = detect_setup(row_dict, r)
        if s:
            setups_all.append({
                'bar':    i,
                'ts':     df_feat.index[i] if isinstance(df_feat.index, pd.DatetimeIndex)
                          else (df_feat['timestamp'].iloc[i] if 'timestamp' in df_feat.columns else i),
                'setup':  s,
                'regime': r,
                'close':  float(df_feat['close'].iloc[i]),
            })
            # Copiar features al setup para get_confidence
            for col in CONTEXT_FEATURES + MOMENTUM_FEATURES + VOLUME_FEATURES:
                setups_all[-1][col] = float(df_feat[col].iloc[i]) if col in df_feat.columns else 0.0

    print(f"  Setups detectados: {len(setups_all)}")
    from collections import Counter
    cnt = Counter(s['setup'] for s in setups_all)
    for k, v in cnt.most_common():
        print(f"    {k}: {v}")

    # Para usar get_confidence desde el setup (no desde df), creamos wrapper
    def sim_with_models(setups, **kwargs):
        """Wrapper que saca las features del setup dict en vez del df."""
        # Enriquecemos el df con info de modelos precomputada
        for s in setups:
            row_data = {k: s.get(k, 0) for k in CONTEXT_FEATURES + MOMENTUM_FEATURES + VOLUME_FEATURES}
            s['_conf'] = get_confidence(row_data, models)
        return simulate(df_oos_for_sim, setups, models={}, funding=funding, **kwargs)

    # Parchar get_confidence para usar _conf pre-calculado
    original_gc = globals()['get_confidence']
    def patched_gc(row, models_arg):
        if '_conf' in row:
            return row['_conf']
        return original_gc(row, models_arg)

    import builtins
    # Pre-calcular confianza para todos los setups
    for s in setups_all:
        row_data = {k: s.get(k, 0) for k in CONTEXT_FEATURES + MOMENTUM_FEATURES + VOLUME_FEATURES}
        s['_conf_val'] = get_confidence(row_data, models)

    def simulate_patched(df_sim, setups, models_arg=None, funding_ser=None, **kwargs):
        """Usa confianza pre-calculada en _conf_val."""
        trades = []
        consec_losses = 0
        paused_until  = -1

        use_funding   = kwargs.get('use_funding', False)
        use_breakeven = kwargs.get('use_breakeven', False)
        use_consec    = kwargs.get('use_consec', False)
        use_sizing    = kwargs.get('use_sizing', False)
        use_limit     = kwargs.get('use_limit', False)

        for s in setups:
            bar        = s['bar']
            entry_p    = s['close']
            setup_name = s['setup']
            conf       = s.get('_conf_val', 0.35)

            if conf < SKIP_THRESHOLD:
                trades.append({**s, 'outcome': 'SKIP_ML', 'conf': conf, 'pnl': 0.0})
                continue

            if use_consec and bar <= paused_until:
                params = SETUP_PARAMS.get(setup_name, DEFAULT_PARAMS)
                out, ep, bh = sim_trade(df_sim, bar, entry_p, params['tp'], params['sl'])
                trades.append({**s, 'outcome': 'PAUSED_' + out, 'conf': conf, 'pnl': 0.0})
                continue

            if use_funding and funding_ser is not None:
                ts = s.get('ts')
                if ts is not None:
                    try:
                        ts_naive = ts.tz_localize(None) if hasattr(ts, 'tz_localize') and ts.tzinfo else ts
                        rate = funding_ser.asof(ts_naive)
                        if pd.notna(rate) and rate > FUNDING_LONG_MAX:
                            trades.append({**s, 'outcome': 'SKIP_FUNDING', 'conf': conf,
                                           'pnl': 0.0, 'funding': float(rate)})
                            continue
                    except Exception:
                        pass

            params = SETUP_PARAMS.get(setup_name, DEFAULT_PARAMS)
            tp_pct, sl_pct = params['tp'], params['sl']

            effective_entry = entry_p
            if use_limit and setup_name == 'SUPPORT_BOUNCE':
                limit_p  = entry_p * 0.998
                next_bar = bar + 1
                if next_bar >= len(df_sim):
                    trades.append({**s, 'outcome': 'LIMIT_NOFILL', 'conf': conf, 'pnl': 0.0})
                    continue
                if float(df_sim['low'].iloc[next_bar]) <= limit_p:
                    effective_entry = limit_p
                else:
                    trades.append({**s, 'outcome': 'LIMIT_NOFILL', 'conf': conf, 'pnl': 0.0})
                    continue

            sz = (1.4 if conf >= 0.45 else 1.2 if conf >= 0.40 else 1.0) if use_sizing else 1.0

            outcome, exit_price, bars_held = sim_trade(
                df_sim, bar, effective_entry, tp_pct, sl_pct,
                use_breakeven=use_breakeven
            )

            if outcome == 'TP':
                pnl = (tp_pct / sl_pct) * sz
            elif outcome == 'SL':
                pnl = -1.0 * sz
            elif outcome == 'BE':
                pnl = 0.0
            else:
                raw = (exit_price - effective_entry) / effective_entry
                pnl = (raw / sl_pct) * sz

            if use_consec:
                if pnl < 0:
                    consec_losses += 1
                    if consec_losses >= MAX_CONSEC_LOSSES:
                        paused_until  = bar + PAUSE_BARS
                        consec_losses = 0
                else:
                    consec_losses = 0

            trades.append({
                **s, 'outcome': outcome, 'exit_price': exit_price,
                'bars_held': bars_held, 'conf': conf, 'sizing': sz, 'pnl': pnl,
            })

        return trades

    # Funding con tz naive para asof lookup
    funding_naive = funding.copy()
    funding_naive.index = funding_naive.index.tz_localize(None) if funding_naive.index.tzinfo else funding_naive.index

    # Hacer timestamps naive tambien en los setups
    for s in setups_all:
        ts = s.get('ts')
        if ts is not None and hasattr(ts, 'tz_localize') and ts.tzinfo:
            s['ts'] = ts.tz_localize(None)
        elif ts is not None and hasattr(ts, 'tz_convert'):
            s['ts'] = ts.tz_convert(None)

    print("\n" + "=" * 65)
    print("RESULTADOS")
    print("=" * 65)

    # === BASELINE ===
    t_base = simulate_patched(df_oos_for_sim, setups_all)
    m_base = metrics(t_base, 'BASELINE (reglas + ML skip<0.30)')
    print(f"\n{'Setup':<30} | {'N':>5} | {'WR':>6} | {'PF':>5} | {'avg':>6}")
    print("-" * 65)
    print_row(m_base)

    # === A: FUNDING RATE ===
    t_fund = simulate_patched(df_oos_for_sim, setups_all,
                              funding_ser=funding_naive, use_funding=True)
    m_fund = metrics(t_fund, 'A. Funding rate filter')

    # === B: BREAK-EVEN ===
    t_be = simulate_patched(df_oos_for_sim, setups_all, use_breakeven=True)
    m_be = metrics(t_be, 'B. Break-even stop')

    # === C: RACHA DE PERDIDAS ===
    t_cons = simulate_patched(df_oos_for_sim, setups_all, use_consec=True)
    m_cons = metrics(t_cons, 'C. Racha perdidas (3 SL)')

    # === D: SIZING POR CONFIANZA ===
    t_sz = simulate_patched(df_oos_for_sim, setups_all, use_sizing=True)
    m_sz = metrics(t_sz, 'D. Sizing por confianza')

    # === E: LIMIT ORDERS ===
    t_lim = simulate_patched(df_oos_for_sim, setups_all, use_limit=True)
    m_lim = metrics(t_lim, 'E. Limit orders (SUPPORT)')

    # === TODAS COMBINADAS ===
    t_all = simulate_patched(df_oos_for_sim, setups_all,
                             funding_ser=funding_naive,
                             use_funding=True, use_breakeven=True,
                             use_consec=True, use_sizing=True, use_limit=True)
    m_all = metrics(t_all, 'TODAS COMBINADAS')

    print("\nComparacion vs Baseline:")
    print("-" * 65)
    print_row(m_base, m_base)
    print_row(m_fund, m_base)
    print_row(m_be,   m_base)
    print_row(m_cons, m_base)
    print_row(m_sz,   m_base)
    print_row(m_lim,  m_base)
    print("-" * 65)
    print_row(m_all,  m_base)

    # === DETALLE POR MEJORA ===
    print("\n" + "=" * 65)
    print("DETALLE POR MEJORA")
    print("=" * 65)

    print("\nA. FILTRO FUNDING RATE:")
    skipped_f = [t for t in t_fund if t['outcome'] == 'SKIP_FUNDING']
    if skipped_f:
        # WR de los que hubieran entrado sin filtro
        bars_f = {t['bar'] for t in skipped_f}
        equiv  = [t for t in t_base if t['bar'] in bars_f and t['outcome'] not in ('SKIP_ML',)]
        if equiv:
            wr_f = sum(1 for t in equiv if t['pnl'] > 0) / len(equiv)
            avg_rate = np.mean([t.get('funding', 0) for t in skipped_f])
            print(f"  Trades filtrados: {len(skipped_f)} | "
                  f"WR si hubieran entrado: {wr_f:.1%} | "
                  f"Funding promedio: {avg_rate:.4%}")
            if wr_f < m_base['wr']:
                print("  -> Trades eliminados eran PEORES que el promedio (FILTRO UTIL)")
            else:
                print("  -> Trades eliminados eran IGUALES o MEJORES (filtro dudoso)")
    else:
        print("  Sin setups con funding > umbral en el periodo OOS")

    print("\nB. BREAK-EVEN STOP:")
    be_trades = [t for t in t_be if t['outcome'] == 'BE']
    print(f"  Trades salvados (SL -> BE): {len(be_trades)}")
    if be_trades:
        # Que habria pasado sin BE
        bars_be = {t['bar'] for t in be_trades}
        sin_be  = [t for t in t_base if t['bar'] in bars_be and t['outcome'] == 'SL']
        print(f"  De estos, {len(sin_be)} eran SL sin el BE -> ahorros confirmados")

    print("\nC. RACHA DE PERDIDAS:")
    paused_t = [t for t in t_cons if t['outcome'].startswith('PAUSED')]
    if paused_t:
        wins_p = sum(1 for t in paused_t if '_TP' in t['outcome'])
        wr_p   = wins_p / len(paused_t)
        print(f"  Periodos de pausa activados: {sum(1 for t in t_cons if t['outcome'] == 'PAUSED_SL')}")
        print(f"  Trades que hubieramos perdido evitados: {len(paused_t)}")
        print(f"  WR de lo que saltamos: {wr_p:.1%}")
        if wr_p < m_base['wr']:
            print("  -> Saltamos trades malos (REGLA UTIL)")
        else:
            print("  -> Saltamos trades normales (regla dudosa)")
    else:
        print("  Sin pausas activadas en el periodo OOS")

    print("\nD. SIZING POR CONFIANZA:")
    # Distribucion de confianza y WR por tier
    tiers = {
        'conf < 0.40 (sz=1.0x)':  [t for t in t_sz if t.get('conf', 0) < 0.40 and t['outcome'] not in ('SKIP_ML',)],
        'conf 0.40-0.44 (sz=1.2x)': [t for t in t_sz if 0.40 <= t.get('conf', 0) < 0.45 and t['outcome'] not in ('SKIP_ML',)],
        'conf >= 0.45 (sz=1.4x)': [t for t in t_sz if t.get('conf', 0) >= 0.45 and t['outcome'] not in ('SKIP_ML',)],
    }
    for tier_name, tier_trades in tiers.items():
        if tier_trades:
            wr_t = sum(1 for t in tier_trades if t['pnl'] > 0) / len(tier_trades)
            print(f"  {tier_name}: N={len(tier_trades)}, WR={wr_t:.1%}")
    print("  -> Si WR crece con confianza, el sizing diferencial es util")

    print("\nE. LIMIT ORDERS (SUPPORT_BOUNCE):")
    sb_base  = [t for t in t_base  if t['setup'] == 'SUPPORT_BOUNCE' and t['outcome'] not in ('SKIP_ML',)]
    sb_lim_f = [t for t in t_lim   if t['outcome'] == 'LIMIT_NOFILL']
    sb_lim_e = [t for t in t_lim   if t['setup'] == 'SUPPORT_BOUNCE' and t['outcome'] not in ('SKIP_ML', 'LIMIT_NOFILL')]
    fill_rate = len(sb_lim_e) / (len(sb_lim_e) + len(sb_lim_f)) if (sb_lim_e or sb_lim_f) else 0
    print(f"  SUPPORT_BOUNCE setups base: {len(sb_base)}")
    print(f"  Fill rate con limit 0.2%: {fill_rate:.1%}")
    print(f"  Sin fill (perdemos): {len(sb_lim_f)}")
    if sb_base and sb_lim_e:
        wr_base_sb = sum(1 for t in sb_base if t['pnl'] > 0) / len(sb_base)
        wr_lim_sb  = sum(1 for t in sb_lim_e if t['pnl'] > 0) / len(sb_lim_e)
        print(f"  WR market vs limit: {wr_base_sb:.1%} -> {wr_lim_sb:.1%}")
    print(f"  -> Si fill_rate > 60% y WR mejora -> UTIL; si fill_rate < 40% -> pierde trades buenos")

    print("\nF. CONFIRMACION 1H:")
    print("  No se puede validar: no hay datos 1H de BTC disponibles.")
    print("  Para validar: descargar BTC/USDT 1H (2022-2026) y correr test separado.")
    print("  Estimacion: filtro 1H reduciria ~20-40% setups segun mercado.")

    # === VEREDICTOS ===
    print("\n" + "=" * 65)
    print("VEREDICTOS")
    print("=" * 65)
    for m, label in [
        (m_fund, 'A. Funding rate'),
        (m_be,   'B. Break-even'),
        (m_cons, 'C. Racha perdidas'),
        (m_sz,   'D. Sizing confianza'),
        (m_lim,  'E. Limit orders'),
    ]:
        v = verdict(m, m_base)
        print(f"  {label:<25}: {v}")
    print(f"  {'F. Confirmacion 1H':<25}: PENDIENTE (sin datos 1H)")

    print("\n" + "=" * 65)
    print("FIN")
    print("=" * 65)


if __name__ == '__main__':
    main()
