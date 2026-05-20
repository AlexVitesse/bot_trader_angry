"""
V14 BTC Walk-Forward Validation (puro, sin ML)
===============================================
Valida si las REGLAS de deteccion de setups de V14 siguen siendo rentables
en datos hasta Jan 2026. El ML es solo un filtro de calidad; si las reglas
no funcionan, el ML no salva nada.

Metodologia:
  - 12 folds de 3 meses (test) sobre periodo 2022-01 a 2026-01
  - Entrenamiento NO necesario (las reglas son deterministicas)
  - LONG only: pullbacks en uptrend, support bounces en range
  - TP/SL por tipo de setup (mismo que STRATEGY_PARAMS en train_btc_v14.py)
  - Feb 2026 como test informativo

Ejecutar:
  poetry run python validate_btc_v14_walkforward.py
"""
import warnings
import numpy as np
import pandas as pd
import pandas_ta as pta
from pathlib import Path
from enum import Enum

warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
TRAIN_CUTOFF = pd.Timestamp('2026-01-31', tz='UTC')
FEB26_START  = pd.Timestamp('2026-02-01', tz='UTC')
FEB26_END    = pd.Timestamp('2026-03-01', tz='UTC')
WF_START     = pd.Timestamp('2022-01-01', tz='UTC')
N_FOLDS      = 12
FOLD_MONTHS  = 3
APPROVAL_PCT = 0.58        # 7/12 folds positivos
MIN_TRADES   = 4           # minimo para evaluar un fold

# TP/SL por tipo de setup (tomado de train_btc_v14.py STRATEGY_PARAMS)
SETUP_PARAMS = {
    'PULLBACK_IN_UPTREND':     {'tp': 0.04, 'sl': 0.015},
    'OVERSOLD_IN_UPTREND':     {'tp': 0.04, 'sl': 0.015},
    'SUPPORT_BOUNCE':          {'tp': 0.025, 'sl': 0.012},
    'RESISTANCE_REJECTION':    {'tp': 0.025, 'sl': 0.012},
    'BREAKOUT_UP':             {'tp': 0.05,  'sl': 0.02},
}
MAX_CANDLES = 50   # horizonte maximo para resolver el trade


# =============================================================================
# FEATURES (identico a train_btc_v14.py)
# =============================================================================

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    feat = pd.DataFrame(index=df.index)

    adx_df = pta.adx(h, l, c, length=14)
    if adx_df is not None:
        feat['adx'] = adx_df.iloc[:, 0]
        feat['di_plus'] = adx_df.iloc[:, 1]
        feat['di_minus'] = adx_df.iloc[:, 2]
        feat['di_diff'] = feat['di_plus'] - feat['di_minus']

    chop = pta.chop(h, l, c, length=14)
    feat['chop'] = chop if chop is not None else 50

    feat['ema20']      = pta.ema(c, length=20)
    feat['ema50']      = pta.ema(c, length=50)
    feat['ema200']     = pta.ema(c, length=200)
    feat['ema20_dist'] = (c - feat['ema20'])  / feat['ema20']  * 100
    feat['ema50_dist'] = (c - feat['ema50'])  / feat['ema50']  * 100
    feat['ema200_dist']= (c - feat['ema200']) / feat['ema200'] * 100
    feat['ema20_slope']= feat['ema20'].pct_change(5) * 100
    feat['ema50_slope']= feat['ema50'].pct_change(10) * 100

    feat['atr']    = pta.atr(h, l, c, length=14)
    feat['atr_pct']= feat['atr'] / c * 100

    bb = pta.bbands(c, length=20)
    if bb is not None:
        feat['bb_upper'] = bb.iloc[:, 2]
        feat['bb_lower'] = bb.iloc[:, 0]
        feat['bb_mid']   = bb.iloc[:, 1]
        feat['bb_width'] = (feat['bb_upper'] - feat['bb_lower']) / feat['bb_mid'] * 100
        feat['bb_pct']   = (c - feat['bb_lower']) / (feat['bb_upper'] - feat['bb_lower'])

    feat['rsi14']   = pta.rsi(c, length=14)
    feat['rsi7']    = pta.rsi(c, length=7)

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

    feat['high_20']  = h.rolling(20).max()
    feat['low_20']   = l.rolling(20).min()
    feat['range_pos']= (c - feat['low_20']) / (feat['high_20'] - feat['low_20'])
    feat['consec_up']  = (c > c.shift(1)).rolling(10).sum()
    feat['consec_down']= (c < c.shift(1)).rolling(10).sum()

    return feat.replace([np.inf, -np.inf], np.nan)


# =============================================================================
# DETECCION DE REGIMEN (identico a train_btc_v14.py)
# =============================================================================

class Regime(Enum):
    TREND_UP   = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE      = "RANGE"
    VOLATILE   = "VOLATILE"


def detect_regime(row: pd.Series) -> Regime:
    adx       = row.get('adx', 20)
    di_diff   = row.get('di_diff', 0)
    chop      = row.get('chop', 50)
    atr_pct   = row.get('atr_pct', 2)
    bb_width  = row.get('bb_width', 5)
    ema20_sl  = row.get('ema20_slope', 0)
    ema50_sl  = row.get('ema50_slope', 0)

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


# =============================================================================
# DETECCION DE SETUPS LONG (identico a detect_setups en train_btc_v14.py)
# =============================================================================

def detect_setups_long(feat: pd.DataFrame) -> pd.DataFrame:
    """Retorna DataFrame con columnas [setup_type, params] solo para LONG."""
    records = []
    for idx, row in feat.iterrows():
        regime = detect_regime(row)
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
# SIMULACION DE TRADES
# =============================================================================

def simulate_trades(df: pd.DataFrame, setups: pd.DataFrame) -> list:
    """Simula trades basados en setups detectados en el periodo dado."""
    trades = []
    for ts, srow in setups.iterrows():
        if ts not in df.index:
            continue
        setup_type = srow['setup_type']
        params = SETUP_PARAMS.get(setup_type, {'tp': 0.03, 'sl': 0.015})
        tp, sl = params['tp'], params['sl']

        entry = df.loc[ts, 'close']
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
            trades.append({
                'ts':         ts,
                'setup_type': setup_type,
                'outcome':    outcome,
                'tp':         tp,
                'sl':         sl,
            })
    return trades


def fold_metrics(trades: list) -> dict:
    if not trades:
        return {'n_trades': 0, 'wr': 0.0, 'pf': 0.0}
    wins   = sum(t['outcome'] for t in trades)
    losses = len(trades) - wins
    gross_win  = sum(t['tp'] for t in trades if t['outcome'] == 1)
    gross_loss = sum(t['sl'] for t in trades if t['outcome'] == 0)
    pf = gross_win / (gross_loss + 1e-10)
    wr = wins / len(trades)
    return {
        'n_trades': len(trades),
        'wins':     wins,
        'losses':   losses,
        'wr':       round(wr, 4),
        'pf':       round(pf, 3),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('=' * 70)
    print('V14 BTC - Walk-Forward Validation (reglas puras, sin ML)')
    print('=' * 70)

    # Cargar datos
    print('\n[1/3] Cargando datos...')
    df = pd.read_csv(DATA_DIR / 'BTCUSDT_4h.csv', parse_dates=['timestamp'])
    df = df.set_index('timestamp').sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    print(f'  {len(df):,} velas  [{df.index[0].date()} a {df.index[-1].date()}]')

    # Calcular features (sobre TODO el dataset para warm-up correcto)
    print('\n[2/3] Calculando features...')
    feat = compute_features(df)
    print(f'  Features: {feat.shape[1]} columnas')

    # Filtrar hasta cutoff de entrenamiento
    df_train   = df[df.index <= TRAIN_CUTOFF]
    feat_train = feat[feat.index <= TRAIN_CUTOFF]

    # Walk-forward: folds de 3 meses en 2022-2026
    print(f'\n[3/3] Walk-Forward: {N_FOLDS} folds x {FOLD_MONTHS} meses')
    print(f'      Periodo: {WF_START.date()} a {TRAIN_CUTOFF.date()}')
    print()

    fold_results = []
    for i in range(N_FOLDS):
        fold_start = WF_START + pd.DateOffset(months=i * FOLD_MONTHS)
        fold_end   = fold_start + pd.DateOffset(months=FOLD_MONTHS) - pd.Timedelta(seconds=1)
        if fold_start > TRAIN_CUTOFF:
            break

        mask = (feat_train.index >= fold_start) & (feat_train.index <= fold_end)
        feat_fold = feat_train[mask]
        df_fold   = df_train[df_train.index <= fold_end]  # necesitamos el futuro para resolver trades

        if len(feat_fold) < 50:
            continue

        setups = detect_setups_long(feat_fold)
        if len(setups) == 0:
            trades = []
        else:
            trades = simulate_trades(df_fold, setups)

        m = fold_metrics(trades)
        # Calcular break-even WR para este fold (promedio de tp/sl ponderado)
        if trades:
            avg_tp = np.mean([t['tp'] for t in trades])
            avg_sl = np.mean([t['sl'] for t in trades])
            be_wr = avg_sl / (avg_tp + avg_sl)
        else:
            be_wr = 0.333

        positive = m['n_trades'] >= MIN_TRADES and m['wr'] > be_wr
        fold_results.append({
            'fold': i + 1,
            'start': fold_start.strftime('%Y-%m'),
            'end':   fold_end.strftime('%Y-%m'),
            'n_trades': m['n_trades'],
            'wr':       m['wr'],
            'pf':       m['pf'],
            'be_wr':    round(be_wr, 3),
            'positive': positive,
        })

        flag = 'OK' if positive else ('skip' if m['n_trades'] < MIN_TRADES else 'FAIL')
        print(f"  Fold {i+1:2d} [{fold_start.strftime('%Y-%m')} a {fold_end.strftime('%Y-%m')}]"
              f"  {m['n_trades']:3d} trades  WR {m['wr']:.1%}  PF {m['pf']:.2f}  BE {be_wr:.1%}  {flag}")

    # ---- Resumen ----
    print()
    evaluable = [r for r in fold_results if r['n_trades'] >= MIN_TRADES]
    n_pos     = sum(1 for r in evaluable if r['positive'])
    n_eval    = len(evaluable)
    min_ok    = max(1, int(np.ceil(n_eval * APPROVAL_PCT)))

    if evaluable:
        avg_wr = np.mean([r['wr'] for r in evaluable])
        avg_pf = np.mean([r['pf'] for r in evaluable])
        avg_trades = np.mean([r['n_trades'] for r in evaluable])
    else:
        avg_wr = avg_pf = avg_trades = 0

    approved = n_pos >= min_ok

    print('=' * 70)
    print('RESUMEN WALK-FORWARD:')
    print(f'  Folds evaluados (>= {MIN_TRADES} trades): {n_eval}/{len(fold_results)}')
    print(f'  Folds positivos:    {n_pos}/{n_eval} (necesita {min_ok})')
    print(f'  WR promedio:        {avg_wr:.1%}')
    print(f'  PF promedio:        {avg_pf:.2f}')
    print(f'  Trades/fold prom:   {avg_trades:.1f}')
    print()

    # Distribución de setups
    all_setup_types = {}
    for i, res in enumerate(fold_results):
        pass  # contamos abajo
    print('DISTRIBUCION POR SETUP TYPE (todo el periodo train):')
    feat_all_wf = feat[(feat.index >= WF_START) & (feat.index <= TRAIN_CUTOFF)]
    all_setups = detect_setups_long(feat_all_wf)
    if len(all_setups):
        st_counts = all_setups['setup_type'].value_counts()
        for st, cnt in st_counts.items():
            print(f'    {st}: {cnt} setups')
    print()

    if approved:
        print(f'VEREDICTO: APROBADO ({n_pos}/{n_eval} folds positivos, WR {avg_wr:.1%})')
    else:
        print(f'VEREDICTO: NO APROBADO ({n_pos}/{n_eval} folds positivos, WR {avg_wr:.1%})')

    # ---- Test Feb 2026 (informativo) ----
    print()
    print('-' * 70)
    print('TEST INFORMATIVO: Febrero 2026 (bear/rango, fuera del training)')
    mask_feb = (feat.index >= FEB26_START) & (feat.index < FEB26_END)
    feat_feb = feat[mask_feb]
    df_feb   = df[df.index >= FEB26_START]
    print(f'  Velas: {len(feat_feb)}')

    if len(feat_feb) > 0:
        setups_feb = detect_setups_long(feat_feb)
        trades_feb = simulate_trades(df_feb, setups_feb)
        m_feb = fold_metrics(trades_feb)
        if trades_feb:
            avg_tp_feb = np.mean([t['tp'] for t in trades_feb])
            avg_sl_feb = np.mean([t['sl'] for t in trades_feb])
            be_wr_feb = avg_sl_feb / (avg_tp_feb + avg_sl_feb)
        else:
            be_wr_feb = 0.333

        print(f'  Setups detectados: {len(setups_feb)}')
        if setups_feb is not None and len(setups_feb):
            print(f'    {setups_feb["setup_type"].value_counts().to_dict()}')
        print(f'  Trades resueltos:  {m_feb["n_trades"]}')
        print(f'  WR: {m_feb["wr"]:.1%}  PF: {m_feb["pf"]:.2f}  BE: {be_wr_feb:.1%}')
        if m_feb['n_trades'] >= MIN_TRADES:
            result_str = 'OK (>BE)' if m_feb['wr'] > be_wr_feb else 'FAIL (<BE)'
            print(f'  Resultado Feb 2026: {result_str}')
        else:
            print(f'  Resultado Feb 2026: sin suficientes trades para evaluar')
    else:
        print('  Sin datos de Feb 2026')

    print()
    print('=' * 70)


if __name__ == '__main__':
    main()
