"""
V14 BTC + Filtro Macro Diario - Comparacion
=============================================
Pregunta: anadir "no LONG si EMA diaria en bear" mejora V14?

Variantes:
  A) V14 puro         (baseline)
  B) V14 + MACRO      (skip LONG si EMA20d < EMA50d < EMA200d)
  C) V14 + MACRO_SOFT (skip LONG solo si EMA20d < EMA200d, mas flexible)

Ejecutar:
  poetry run python validate_btc_v14_macro.py
"""
import warnings
import numpy as np
import pandas as pd
import pandas_ta as pta
from pathlib import Path
from enum import Enum

warnings.filterwarnings('ignore')

DATA_DIR     = Path('data')
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
# MACRO REGIME (diario)
# =============================================================================

def compute_macro_regime(df_1d: pd.DataFrame) -> pd.Series:
    """
    Clasifica el regimen macro para cada dia usando EMAs diarias.

    MACRO_BULL: EMA20 > EMA50 > EMA200  (tendencia alcista clara)
    MACRO_BEAR: EMA20 < EMA50 < EMA200  (tendencia bajista clara)
    MACRO_RANGE: cualquier otra configuracion

    Usa shift(1) para evitar look-ahead bias:
    el regimen del dia N se determina con datos hasta el dia N-1.
    """
    c = df_1d['close']
    ema20  = pta.ema(c, length=20).shift(1)
    ema50  = pta.ema(c, length=50).shift(1)
    ema200 = pta.ema(c, length=200).shift(1)

    regime = pd.Series('MACRO_RANGE', index=df_1d.index)
    bull = (ema20 > ema50) & (ema50 > ema200)
    bear = (ema20 < ema50) & (ema50 < ema200)
    regime[bull] = 'MACRO_BULL'
    regime[bear] = 'MACRO_BEAR'

    return regime


def compute_macro_soft(df_1d: pd.DataFrame) -> pd.Series:
    """
    Regimen macro suave: solo EMA20 vs EMA200 (menos restrictivo).

    MACRO_BULL: EMA20 > EMA200
    MACRO_BEAR: EMA20 < EMA200
    """
    c = df_1d['close']
    ema20  = pta.ema(c, length=20).shift(1)
    ema200 = pta.ema(c, length=200).shift(1)

    regime = pd.Series('MACRO_RANGE', index=df_1d.index)
    regime[ema20 > ema200] = 'MACRO_BULL'
    regime[ema20 < ema200] = 'MACRO_BEAR'

    return regime


def align_macro_to_4h(macro_1d: pd.Series, idx_4h: pd.Index) -> pd.Series:
    """
    Lleva el regimen diario al timeframe 4h (forward-fill).
    El regimen del dia D se aplica a TODAS las velas 4h del dia D.
    No hay look-ahead: la vela 4h de las 00:00 ya conoce el cierre del dia anterior.
    """
    # Reindexar al indice 4h con forward-fill
    macro_4h = macro_1d.reindex(idx_4h, method='ffill')
    return macro_4h.fillna('MACRO_RANGE')


# =============================================================================
# FEATURES V14 (identico)
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

def simulate_trades(df: pd.DataFrame, setups: pd.DataFrame,
                    macro_4h: pd.Series = None,
                    allowed_macros: set = None) -> list:
    """
    allowed_macros: si se especifica, solo opera en esos regimenes.
    Ej: {'MACRO_BULL', 'MACRO_RANGE'} = no LONG en MACRO_BEAR
    """
    trades = []
    for ts, srow in setups.iterrows():
        if ts not in df.index:
            continue
        # Filtro macro (si aplica)
        if macro_4h is not None and allowed_macros is not None:
            regime_macro = macro_4h.get(ts, 'MACRO_RANGE')
            if regime_macro not in allowed_macros:
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
    tps = np.mean([t['tp'] for t in trades])
    sls = np.mean([t['sl'] for t in trades])
    return sls / (tps + sls)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('=' * 78)
    print('V14 BTC + Filtro Macro Diario - Comparacion Walk-Forward')
    print('=' * 78)

    # ----- Datos 4h -----
    print('\n[1/4] Cargando datos 4h...')
    df = pd.read_csv(DATA_DIR / 'BTCUSDT_4h.csv', parse_dates=['timestamp'])
    df = df.set_index('timestamp').sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    print(f'  {len(df):,} velas 4h  [{df.index[0].date()} a {df.index[-1].date()}]')

    # ----- Datos 1d para macro -----
    print('[2/4] Cargando datos 1d y calculando regimen macro...')
    df_1d = pd.read_parquet(DATA_DIR / 'btcusdt_1d_v15.parquet')
    if df_1d.index.tz is None:
        df_1d.index = df_1d.index.tz_localize('UTC')
    df_1d = df_1d.sort_index()

    macro_strict = compute_macro_regime(df_1d)      # EMA20 < EMA50 < EMA200
    macro_soft   = compute_macro_soft(df_1d)         # EMA20 < EMA200

    macro_strict_4h = align_macro_to_4h(macro_strict, df.index)
    macro_soft_4h   = align_macro_to_4h(macro_soft,   df.index)

    # Distribucion macro en el periodo de validacion
    mask_wf = (macro_strict_4h.index >= WF_START) & (macro_strict_4h.index <= TRAIN_CUTOFF)
    dist_strict = macro_strict_4h[mask_wf].value_counts(normalize=True)
    dist_soft   = macro_soft_4h[mask_wf].value_counts(normalize=True)
    print('  Distribucion macro STRICT (2022-2026):')
    for k, v in dist_strict.items():
        print(f'    {k}: {v:.1%}')
    print('  Distribucion macro SOFT (2022-2026):')
    for k, v in dist_soft.items():
        print(f'    {k}: {v:.1%}')

    # ----- Features V14 -----
    print('\n[3/4] Calculando features V14...')
    feat14 = compute_features_v14(df)
    feat_tr = feat14[feat14.index <= TRAIN_CUTOFF]
    df_tr   = df[df.index <= TRAIN_CUTOFF]

    # ----- Walk-Forward -----
    print(f'\n[4/4] Walk-Forward: {N_FOLDS} folds x {FOLD_MONTHS} meses')
    print()

    header = (f"{'Fold':>4}  {'Periodo':>13}  "
              f"{'A-Raw':>6} {'WR':>7} {'PF':>5}  "
              f"{'B-Strict':>8} {'WR':>7} {'PF':>5}  "
              f"{'C-Soft':>6} {'WR':>7} {'PF':>5}")
    print(header)
    print('-' * 80)

    res_raw    = []
    res_strict = []
    res_soft   = []

    for i in range(N_FOLDS):
        fold_start = WF_START + pd.DateOffset(months=i * FOLD_MONTHS)
        fold_end   = fold_start + pd.DateOffset(months=FOLD_MONTHS) - pd.Timedelta(seconds=1)
        if fold_start > TRAIN_CUTOFF:
            break

        mask   = (feat_tr.index >= fold_start) & (feat_tr.index <= fold_end)
        feat_f = feat_tr[mask]
        if len(feat_f) < 50:
            continue

        setups = detect_setups_long(feat_f)

        # A: Sin filtro macro
        tA = simulate_trades(df_tr, setups)
        mA = fold_metrics(tA)
        be = avg_be_wr(tA)
        okA = mA['n_trades'] >= MIN_TRADES and mA['wr'] > be

        # B: Macro STRICT (no LONG en MACRO_BEAR: necesita EMA20<EMA50<EMA200)
        tB = simulate_trades(df_tr, setups, macro_strict_4h, {'MACRO_BULL', 'MACRO_RANGE'})
        mB = fold_metrics(tB)
        be_b = avg_be_wr(tB)
        okB = mB['n_trades'] >= MIN_TRADES and mB['wr'] > be_b

        # C: Macro SOFT (no LONG en MACRO_BEAR: necesita EMA20<EMA200)
        tC = simulate_trades(df_tr, setups, macro_soft_4h, {'MACRO_BULL', 'MACRO_RANGE'})
        mC = fold_metrics(tC)
        be_c = avg_be_wr(tC)
        okC = mC['n_trades'] >= MIN_TRADES and mC['wr'] > be_c

        res_raw.append({'n': mA['n_trades'], 'wr': mA['wr'], 'pf': mA['pf'], 'pos': okA})
        res_strict.append({'n': mB['n_trades'], 'wr': mB['wr'], 'pf': mB['pf'], 'pos': okB})
        res_soft.append({'n': mC['n_trades'], 'wr': mC['wr'], 'pf': mC['pf'], 'pos': okC})

        fA = 'OK' if okA else ('--' if mA['n_trades'] < MIN_TRADES else 'FL')
        fB = 'OK' if okB else ('--' if mB['n_trades'] < MIN_TRADES else 'FL')
        fC = 'OK' if okC else ('--' if mC['n_trades'] < MIN_TRADES else 'FL')

        period = f"{fold_start.strftime('%Y-%m')}/{fold_end.strftime('%m')}"
        print(f"  {i+1:2d}   {period:>13}  "
              f"{mA['n_trades']:5d} {mA['wr']:6.1%}{fA} {mA['pf']:5.2f}  "
              f"{mB['n_trades']:7d} {mB['wr']:6.1%}{fB} {mB['pf']:5.2f}  "
              f"{mC['n_trades']:5d} {mC['wr']:6.1%}{fC} {mC['pf']:5.2f}")

    # ---- Resumen -----
    def summarize(results, label, min_trades=MIN_TRADES):
        ev = [r for r in results if r['n'] >= min_trades]
        npos = sum(r['pos'] for r in ev)
        avg_wr = np.mean([r['wr'] for r in ev]) if ev else 0
        avg_pf = np.mean([r['pf'] for r in ev]) if ev else 0
        avg_n  = np.mean([r['n']  for r in ev]) if ev else 0
        return {'label': label, 'eval': len(ev), 'pos': npos,
                'wr': avg_wr, 'pf': avg_pf, 'n': avg_n}

    sA = summarize(res_raw,    'V14 puro (A)')
    sB = summarize(res_strict, 'V14+Macro STRICT (B)')
    sC = summarize(res_soft,   'V14+Macro SOFT (C)')

    min_ok = max(1, int(np.ceil(12 * 0.58)))  # 7/12

    print()
    print('=' * 78)
    print(f'RESUMEN:')
    print(f'{"":28}  {"A-Raw":>10}  {"B-Strict":>12}  {"C-Soft":>10}')
    print(f'  Folds evaluados:          {sA["eval"]:>10d}  {sB["eval"]:>12d}  {sC["eval"]:>10d}')
    print(f'  Folds positivos:          {sA["pos"]:>10d}  {sB["pos"]:>12d}  {sC["pos"]:>10d}')
    print(f'  WR promedio:              {sA["wr"]:>10.1%}  {sB["wr"]:>12.1%}  {sC["wr"]:>10.1%}')
    print(f'  PF promedio:              {sA["pf"]:>10.2f}  {sB["pf"]:>12.2f}  {sC["pf"]:>10.2f}')
    print(f'  Trades/fold prom:         {sA["n"]:>10.1f}  {sB["n"]:>12.1f}  {sC["n"]:>10.1f}')
    print()

    # Cual es la mejor opcion
    opts = [sA, sB, sC]
    best = max(opts, key=lambda s: (s['pos'], s['wr'], s['pf']))

    print('VEREDICTO:')
    for s in opts:
        status = 'APROBADO' if s['pos'] >= min_ok else 'NO APROBADO'
        print(f'  {s["label"]}: {status} ({s["pos"]}/{len(res_raw)} folds, '
              f'WR {s["wr"]:.1%}, PF {s["pf"]:.2f}, {s["n"]:.1f} trades/fold)')

    print()
    if best['label'] != sA['label']:
        wr_gain = best['wr'] - sA['wr']
        pf_gain = best['pf'] - sA['pf']
        print(f'  RECOMENDACION: {best["label"]}')
        print(f'  Mejora vs raw: WR {wr_gain:+.1%}  PF {pf_gain:+.2f}')
        print(f'  Trades reducidos: {sA["n"]:.1f} -> {best["n"]:.1f}/fold '
              f'({(best["n"]/sA["n"]-1):+.0%})')
    else:
        print('  RECOMENDACION: V14 puro es el mejor resultado.')
        print('  El filtro macro no mejora significativamente.')

    # Breakdown de que foldsconvierte el filtro
    print()
    print('ANALISIS FOLD A FOLD (folds convertidos por filtro):')
    for i, (rA, rB, rC) in enumerate(zip(res_raw, res_strict, res_soft)):
        fold_label = f'Fold {i+1}'
        changed_B = rA['pos'] != rB['pos']
        changed_C = rA['pos'] != rC['pos']
        if changed_B or changed_C:
            orig = 'OK' if rA['pos'] else 'FL'
            new_B = 'OK' if rB['pos'] else ('--' if rB['n'] < MIN_TRADES else 'FL')
            new_C = 'OK' if rC['pos'] else ('--' if rC['n'] < MIN_TRADES else 'FL')
            print(f'  {fold_label}: Raw={orig}  Strict={new_B}  Soft={new_C}  '
                  f'(raw {rA["n"]}t WR{rA["wr"]:.0%} | '
                  f'strict {rB["n"]}t WR{rB["wr"]:.0%} | '
                  f'soft {rC["n"]}t WR{rC["wr"]:.0%})')

    # ---- Periodo bear 2022 -----
    print()
    print('DETALLE PERIODO 2022 (el mas critico):')
    feat_2022 = feat14[(feat14.index >= '2022-01-01') & (feat14.index <= '2022-12-31')]
    df_2022   = df[(df.index >= '2022-01-01') & (df.index <= '2023-03-31')]  # incluir futuro para resolver
    setups_2022 = detect_setups_long(feat_2022)
    if len(setups_2022):
        tA_22 = simulate_trades(df_2022, setups_2022)
        tB_22 = simulate_trades(df_2022, setups_2022, macro_strict_4h, {'MACRO_BULL', 'MACRO_RANGE'})
        tC_22 = simulate_trades(df_2022, setups_2022, macro_soft_4h,   {'MACRO_BULL', 'MACRO_RANGE'})
        mA22 = fold_metrics(tA_22)
        mB22 = fold_metrics(tB_22)
        mC22 = fold_metrics(tC_22)
        print(f'  Setups 2022: {len(setups_2022)} detectados')
        print(f'  Raw:    {mA22["n_trades"]} trades  WR {mA22["wr"]:.1%}  PF {mA22["pf"]:.2f}')
        print(f'  Strict: {mB22["n_trades"]} trades  WR {mB22["wr"]:.1%}  PF {mB22["pf"]:.2f}')
        print(f'  Soft:   {mC22["n_trades"]} trades  WR {mC22["wr"]:.1%}  PF {mC22["pf"]:.2f}')

        # Distribucion macro en setups 2022
        macro_at_setups = macro_strict_4h[macro_strict_4h.index.isin(setups_2022.index)]
        print(f'  Regimen macro en setups 2022:')
        for k, v in macro_at_setups.value_counts().items():
            print(f'    {k}: {v} setups')

    # ---- Test Feb 2026 -----
    print()
    print('-' * 78)
    print('TEST INFORMATIVO: Febrero 2026')
    mask_feb = (feat14.index >= FEB26_START) & (feat14.index < FEB26_END)
    feat_feb  = feat14[mask_feb]
    df_feb    = df[df.index >= FEB26_START]

    macro_feb = macro_strict_4h.get(FEB26_START, 'MACRO_RANGE')
    macro_soft_feb = macro_soft_4h.get(FEB26_START, 'MACRO_RANGE')
    print(f'  Macro STRICT Feb 2026: {macro_feb}')
    print(f'  Macro SOFT   Feb 2026: {macro_soft_feb}')

    if len(feat_feb) > 0:
        setups_feb = detect_setups_long(feat_feb)
        tA_f = simulate_trades(df_feb, setups_feb)
        tB_f = simulate_trades(df_feb, setups_feb, macro_strict_4h, {'MACRO_BULL', 'MACRO_RANGE'})
        tC_f = simulate_trades(df_feb, setups_feb, macro_soft_4h,   {'MACRO_BULL', 'MACRO_RANGE'})
        print(f'  Raw:    {fold_metrics(tA_f)["n_trades"]} trades  WR {fold_metrics(tA_f)["wr"]:.1%}')
        print(f'  Strict: {fold_metrics(tB_f)["n_trades"]} trades  WR {fold_metrics(tB_f)["wr"]:.1%}')
        print(f'  Soft:   {fold_metrics(tC_f)["n_trades"]} trades  WR {fold_metrics(tC_f)["wr"]:.1%}')

    print()
    print('=' * 78)


if __name__ == '__main__':
    main()
