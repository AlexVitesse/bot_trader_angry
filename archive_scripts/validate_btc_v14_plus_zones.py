"""
V14 BTC + V15 Zone Filter - Comparacion
========================================
Pregunta: el filtro de zonas de V15 (FVG / PDH/PDL / PWH/PWL) mejora V14?

Metodologia:
  - Mismas reglas de V14 (detect_setups_long)
  - Run A: todos los setups V14 (baseline)
  - Run B: solo setups donde hay confluencia de zona V15 (at_fvg_bull OR near_pdl OR near_pwl)
  - Comparar fold a fold: n_trades, WR, PF

Si Run B tiene WR/PF significativamente mejor -> integrar el zone filter.
Si Run B no mejora (o tiene muy pocos trades) -> el filtro no aporta.

Ejecutar:
  poetry run python validate_btc_v14_plus_zones.py
"""
import sys
import warnings
import numpy as np
import pandas as pd
import pandas_ta as pta
from pathlib import Path
from enum import Enum

warnings.filterwarnings('ignore')

# Importar el modulo de estructura de mercado
sys.path.insert(0, str(Path(__file__).parent))
from v15_market_structure import add_structure_features

DATA_DIR    = Path('data')
TRAIN_CUTOFF = pd.Timestamp('2026-01-31', tz='UTC')
FEB26_START  = pd.Timestamp('2026-02-01', tz='UTC')
FEB26_END    = pd.Timestamp('2026-03-01', tz='UTC')
WF_START     = pd.Timestamp('2022-01-01', tz='UTC')
N_FOLDS      = 12
FOLD_MONTHS  = 3
MIN_TRADES   = 4

SETUP_PARAMS = {
    'PULLBACK_IN_UPTREND':  {'tp': 0.04,  'sl': 0.015},
    'OVERSOLD_IN_UPTREND':  {'tp': 0.04,  'sl': 0.015},
    'SUPPORT_BOUNCE':       {'tp': 0.025, 'sl': 0.012},
    'BREAKOUT_UP':          {'tp': 0.05,  'sl': 0.02},
}
MAX_CANDLES = 50


# =============================================================================
# FEATURES V14 (identico a train_btc_v14.py)
# =============================================================================

class Regime(Enum):
    TREND_UP   = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE      = "RANGE"
    VOLATILE   = "VOLATILE"


def compute_features_v14(df: pd.DataFrame) -> pd.DataFrame:
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

    feat['rsi14']   = pta.rsi(c, length=14)
    feat['rsi7']    = pta.rsi(c, length=7)

    stoch = pta.stoch(h, l, c, k=14, d=3)
    if stoch is not None:
        feat['stoch_k'] = stoch.iloc[:, 0]

    feat['vol_ratio']  = v / v.rolling(20).mean()
    obv = (np.sign(c.diff()) * v).cumsum()
    feat['obv_slope']  = obv.pct_change(10) * 100

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
# SIMULACION
# =============================================================================

def simulate_trades(df: pd.DataFrame, setups: pd.DataFrame) -> list:
    trades = []
    for ts, srow in setups.iterrows():
        if ts not in df.index:
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
        'wr': round(wins / len(trades), 4),
        'pf': round(gross_win / (gross_los + 1e-10), 3),
    }


def avg_be_wr(trades: list) -> float:
    if not trades:
        return 0.333
    tps = np.mean([t['tp'] for t in trades])
    sls = np.mean([t['sl'] for t in trades])
    return sls / (tps + sls)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('=' * 72)
    print('V14 BTC vs V14 + V15 Zone Filter - Comparacion Walk-Forward')
    print('=' * 72)

    # ----- Datos -----
    print('\n[1/4] Cargando datos...')
    df = pd.read_csv(DATA_DIR / 'BTCUSDT_4h.csv', parse_dates=['timestamp'])
    df = df.set_index('timestamp').sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    print(f'  {len(df):,} velas  [{df.index[0].date()} a {df.index[-1].date()}]')

    # ----- Features V14 -----
    print('\n[2/4] Calculando features V14...')
    feat14 = compute_features_v14(df)

    # ----- Features V15 (zonas) -----
    print('[3/4] Calculando features de estructura V15 (FVG / PDH / PWH)...')
    struct = add_structure_features(df)
    # Unir en un solo DataFrame
    feat = pd.concat([feat14, struct], axis=1)
    print(f'  Columnas estructura: {list(struct.columns)}')

    # Filtrar hasta cutoff
    feat_tr = feat[feat.index <= TRAIN_CUTOFF]
    df_tr   = df[df.index <= TRAIN_CUTOFF]

    # ----- Walk-Forward -----
    print(f'\n[4/4] Walk-Forward: {N_FOLDS} folds x {FOLD_MONTHS} meses')
    print(f'      Zona filter: at_fvg_bull=1 OR near_pdl=1 OR near_pwl=1')
    print()

    header = f"{'Fold':>4}  {'Periodo':>13}  " \
             f"{'V14 N':>6} {'V14 WR':>7} {'V14 PF':>7}  " \
             f"{'+ZN N':>6} {'+ZN WR':>7} {'+ZN PF':>7}  {'Mejora':>7}"
    print(header)
    print('-' * 74)

    results_raw   = []
    results_filt  = []

    for i in range(N_FOLDS):
        fold_start = WF_START + pd.DateOffset(months=i * FOLD_MONTHS)
        fold_end   = fold_start + pd.DateOffset(months=FOLD_MONTHS) - pd.Timedelta(seconds=1)
        if fold_start > TRAIN_CUTOFF:
            break

        mask  = (feat_tr.index >= fold_start) & (feat_tr.index <= fold_end)
        feat_f = feat_tr[mask]

        if len(feat_f) < 50:
            continue

        # --- Run A: V14 puro ---
        setups_all = detect_setups_long(feat_f)
        trades_all = simulate_trades(df_tr, setups_all)
        m_all      = fold_metrics(trades_all)
        be_all     = avg_be_wr(trades_all)
        pos_all    = m_all['n_trades'] >= MIN_TRADES and m_all['wr'] > be_all

        # --- Run B: V14 + zone filter ---
        # Solo setups donde hay una zona V15 activa (LONG direction)
        if len(setups_all) > 0:
            # Extraer flags de zona para los timestamps de setup
            zone_flags = feat_f.loc[
                feat_f.index.isin(setups_all.index),
                [c for c in ['at_fvg_bull', 'near_pdl', 'near_pwl'] if c in feat_f.columns]
            ]
            # Un setup pasa el filtro si al menos una zona LONG esta activa
            at_zone = zone_flags.max(axis=1) > 0  # True si cualquier columna == 1
            setups_filt = setups_all[setups_all.index.isin(at_zone[at_zone].index)]
        else:
            setups_filt = pd.DataFrame(columns=['setup_type'])

        trades_filt = simulate_trades(df_tr, setups_filt)
        m_filt      = fold_metrics(trades_filt)
        be_filt     = avg_be_wr(trades_filt)
        pos_filt    = m_filt['n_trades'] >= MIN_TRADES and m_filt['wr'] > be_filt

        # Mejora de WR
        wr_delta = m_filt['wr'] - m_all['wr'] if m_filt['n_trades'] >= MIN_TRADES else float('nan')

        results_raw.append({'n': m_all['n_trades'], 'wr': m_all['wr'], 'pf': m_all['pf'], 'pos': pos_all})
        results_filt.append({'n': m_filt['n_trades'], 'wr': m_filt['wr'], 'pf': m_filt['pf'], 'pos': pos_filt})

        flag_raw  = 'OK' if pos_all  else ('--' if m_all['n_trades']  < MIN_TRADES else 'FL')
        flag_filt = 'OK' if pos_filt else ('--' if m_filt['n_trades'] < MIN_TRADES else 'FL')
        delta_str = f'{wr_delta:+.1%}' if not np.isnan(wr_delta) else '  n/a '

        period = f"{fold_start.strftime('%Y-%m')}/{fold_end.strftime('%m')}"
        print(f"  {i+1:2d}   {period:>13}  "
              f"{m_all['n_trades']:5d} {m_all['wr']:6.1%}{flag_raw:>2} {m_all['pf']:6.2f}  "
              f"{m_filt['n_trades']:5d} {m_filt['wr']:6.1%}{flag_filt:>2} {m_filt['pf']:6.2f}  "
              f"{delta_str:>7}")

    # ---- Resumen -----
    print()
    eval_raw   = [r for r in results_raw  if r['n'] >= MIN_TRADES]
    eval_filt  = [r for r in results_filt if r['n'] >= MIN_TRADES]
    npos_raw   = sum(r['pos'] for r in eval_raw)
    npos_filt  = sum(r['pos'] for r in eval_filt)

    avg_wr_raw  = np.mean([r['wr'] for r in eval_raw])  if eval_raw  else 0
    avg_pf_raw  = np.mean([r['pf'] for r in eval_raw])  if eval_raw  else 0
    avg_wr_filt = np.mean([r['wr'] for r in eval_filt]) if eval_filt else 0
    avg_pf_filt = np.mean([r['pf'] for r in eval_filt]) if eval_filt else 0
    avg_n_raw   = np.mean([r['n']  for r in eval_raw])  if eval_raw  else 0
    avg_n_filt  = np.mean([r['n']  for r in eval_filt]) if eval_filt else 0

    print('=' * 72)
    print(f'RESUMEN:')
    print(f'{"":22}  {"V14 puro":>20}  {"V14 + Zonas":>20}')
    print(f'  Folds evaluados:    {len(eval_raw):>20d}  {len(eval_filt):>20d}')
    print(f'  Folds positivos:    {npos_raw:>20d}  {npos_filt:>20d}')
    print(f'  WR promedio:        {avg_wr_raw:>20.1%}  {avg_wr_filt:>20.1%}')
    print(f'  PF promedio:        {avg_pf_raw:>20.2f}  {avg_pf_filt:>20.2f}')
    print(f'  Trades/fold prom:   {avg_n_raw:>20.1f}  {avg_n_filt:>20.1f}')
    print()

    # Analisis de zonas: que % de setups estan en zona
    print('COBERTURA DEL FILTRO (periodo completo 2022-2026):')
    feat_wf = feat[(feat.index >= WF_START) & (feat.index <= TRAIN_CUTOFF)]
    all_setups = detect_setups_long(feat_wf)
    if len(all_setups) > 0:
        zone_cols = [c for c in ['at_fvg_bull', 'near_pdl', 'near_pwl'] if c in feat_wf.columns]
        if zone_cols:
            zone_f = feat_wf.loc[feat_wf.index.isin(all_setups.index), zone_cols]
            at_any = zone_f.max(axis=1) > 0
            pct_covered = at_any.mean()
            print(f'  Setups totales V14:          {len(all_setups)}')
            print(f'  Con confluencia de zona:     {at_any.sum()} ({pct_covered:.1%})')
            print(f'  Sin zona (filtrados afuera): {(~at_any).sum()} ({1-pct_covered:.1%})')
            print()
            # Desglose por zona
            for col in zone_cols:
                cnt = (zone_f[col] > 0).sum()
                print(f'    {col}: {cnt} setups ({cnt/len(all_setups):.1%})')

    # ---- Veredicto -----
    print()
    print('VEREDICTO:')
    min_ok = max(1, int(np.ceil(12 * 0.58)))  # 7/12

    if npos_raw >= min_ok:
        print(f'  V14 puro:    APROBADO ({npos_raw}/12, WR {avg_wr_raw:.1%}, PF {avg_pf_raw:.2f})')
    else:
        print(f'  V14 puro:    NO APROBADO ({npos_raw}/12)')

    if npos_filt >= min_ok:
        print(f'  V14 + zonas: APROBADO ({npos_filt}/12, WR {avg_wr_filt:.1%}, PF {avg_pf_filt:.2f})')
    else:
        print(f'  V14 + zonas: NO APROBADO ({npos_filt}/12, WR {avg_wr_filt:.1%})')

    wr_gain = avg_wr_filt - avg_wr_raw
    if wr_gain > 0.03 and npos_filt >= npos_raw:
        print(f'\n  RECOMENDACION: Usar V14 + Zone Filter')
        print(f'  Mejora WR: {wr_gain:+.1%}  Folds: {npos_filt} vs {npos_raw}')
    elif wr_gain > 0 and len(eval_filt) >= len(eval_raw) * 0.7:
        print(f'\n  RECOMENDACION: Zone filter mejora marginalmente (+{wr_gain:.1%} WR)')
        print(f'  Pero reduce trades ({avg_n_filt:.1f} vs {avg_n_raw:.1f}/fold). Depende del objetivo.')
    else:
        print(f'\n  RECOMENDACION: Zone filter no mejora lo suficiente ({wr_gain:+.1%} WR)')
        print(f'  Usar V14 puro o revisar parametros del zone filter.')

    # ---- Test Feb 2026 -----
    print()
    print('-' * 72)
    print('TEST INFORMATIVO: Febrero 2026')
    mask_feb = (feat.index >= FEB26_START) & (feat.index < FEB26_END)
    feat_feb = feat[mask_feb]
    df_feb   = df[df.index >= FEB26_START]

    if len(feat_feb) > 0:
        setups_feb = detect_setups_long(feat_feb)
        trades_raw_feb  = simulate_trades(df_feb, setups_feb)
        m_raw_feb = fold_metrics(trades_raw_feb)

        if len(setups_feb) > 0:
            zone_cols_f = [c for c in ['at_fvg_bull', 'near_pdl', 'near_pwl'] if c in feat_feb.columns]
            if zone_cols_f:
                zone_f_feb = feat_feb.loc[feat_feb.index.isin(setups_feb.index), zone_cols_f]
                at_zone_feb = zone_f_feb.max(axis=1) > 0
                setups_filt_feb = setups_feb[setups_feb.index.isin(at_zone_feb[at_zone_feb].index)]
            else:
                setups_filt_feb = setups_feb
        else:
            setups_filt_feb = setups_feb

        trades_filt_feb = simulate_trades(df_feb, setups_filt_feb)
        m_filt_feb = fold_metrics(trades_filt_feb)

        print(f'  V14 puro:    {m_raw_feb["n_trades"]} trades  WR {m_raw_feb["wr"]:.1%}  PF {m_raw_feb["pf"]:.2f}')
        print(f'  V14 + zonas: {m_filt_feb["n_trades"]} trades  WR {m_filt_feb["wr"]:.1%}  PF {m_filt_feb["pf"]:.2f}')

    print()
    print('=' * 72)


if __name__ == '__main__':
    main()
