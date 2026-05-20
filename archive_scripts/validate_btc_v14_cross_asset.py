"""
V14 BTC - Cross-Asset Validation (BTC modelos -> ETH)
======================================================
Logica:
  Si el modelo entrenado en BTC funciona en ETH SIN reentrenar,
  aprendio patrones reales de mercado, no memorizo BTC especificamente.

  Si funciona en BTC pero falla en ETH -> sospechar overfitting a BTC.

Metodologia:
  - Carga modelos entrenados en BTC (context/momentum/volume _long.pkl)
  - Descarga/carga datos ETH 4h (nunca vistos en training)
  - Aplica las MISMAS REGLAS V14 en ETH (detect_setups_long)
  - Aplica los MISMOS MODELOS ML BTC en ETH (sin reentrenar)
  - Reporta WR, PF, n_trades
  - Compara con BTC walk-forward como referencia

Criterio de aprobacion:
  - WR ETH > break-even (31%)
  - Degradacion WR (BTC-ETH) < 15pp
  - PF ETH > 1.0

Ejecutar:
  poetry run python validate_btc_v14_cross_asset.py
"""
import warnings
import json
import joblib
import numpy as np
import pandas as pd
import pandas_ta as pta
from pathlib import Path
from enum import Enum

warnings.filterwarnings('ignore')

DATA_DIR  = Path('data')
MODEL_DIR = Path('strategies/btc_v14/models')

OOS_START = pd.Timestamp('2022-01-01', tz='UTC')   # mismo periodo que BTC WF
OOS_END   = pd.Timestamp('2026-01-31', tz='UTC')   # mismo cutoff que training BTC
FEB26_START = pd.Timestamp('2026-02-01', tz='UTC')
FEB26_END   = pd.Timestamp('2026-03-01', tz='UTC')

SETUP_PARAMS = {
    'PULLBACK_IN_UPTREND':  {'tp': 0.04,  'sl': 0.015},
    'OVERSOLD_IN_UPTREND':  {'tp': 0.04,  'sl': 0.015},
    'SUPPORT_BOUNCE':       {'tp': 0.025, 'sl': 0.012},
    'BREAKOUT_UP':          {'tp': 0.05,  'sl': 0.02},
}
MAX_CANDLES = 50


# =============================================================================
# FEATURES + REGLAS V14 (identico a train_validate_btc_v14_ml.py)
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
# PREDICCION Y SIMULACION
# =============================================================================

def predict_avg_prob(feat_row: pd.Series, experts: dict) -> float:
    probs = []
    for name, exp in experts.items():
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
                    feat: pd.DataFrame, experts: dict,
                    skip_threshold: float) -> list:
    trades = []
    for ts, srow in setups.iterrows():
        if ts not in df.index:
            continue

        if ts in feat.index:
            avg_prob = predict_avg_prob(feat.loc[ts], experts)
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


def metrics(trades: list) -> dict:
    if not trades:
        return {'n_trades': 0, 'wr': 0.0, 'pf': 0.0, 'wins': 0, 'losses': 0}
    wins      = sum(t['outcome'] for t in trades)
    losses    = len(trades) - wins
    gross_win = sum(t['tp'] for t in trades if t['outcome'] == 1)
    gross_los = sum(t['sl'] for t in trades if t['outcome'] == 0)
    return {
        'n_trades': len(trades),
        'wins':     int(wins),
        'losses':   int(losses),
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
    print('=' * 70)
    print('V14 BTC - Cross-Asset Validation (BTC modelos -> ETH)')
    print('=' * 70)

    # ----- Cargar modelos BTC -----
    print('\n[1/4] Cargando modelos BTC...')
    needed = ['context_long.pkl', 'momentum_long.pkl', 'volume_long.pkl']
    for f in needed:
        if not (MODEL_DIR / f).exists():
            print(f'  ERROR: {f} no existe. Ejecutar train_validate_btc_v14_ml.py primero.')
            return

    experts = {}
    for name_pkl in needed:
        name = name_pkl.replace('_long.pkl', '')
        pkg = joblib.load(MODEL_DIR / name_pkl)
        experts[name] = pkg

    # Leer meta
    with open(MODEL_DIR / 'meta.json') as f:
        meta = json.load(f)

    skip_thr = meta.get('skip_threshold', 0.30)
    btc_wf   = meta.get('wf_long', {})

    print(f'  Modelos: {list(experts.keys())}')
    print(f'  Skip threshold: {skip_thr}')
    print()
    print('  REFERENCIA BTC Walk-Forward:')
    print(f'    Folds OK: {btc_wf.get("folds_ok","?")}/12  '
          f'WR: {btc_wf.get("avg_wr",0):.1%}  '
          f'PF: {btc_wf.get("avg_pf",0):.2f}  '
          f'{"APROBADO" if btc_wf.get("approved") else "NO APROBADO"}')

    # ----- Cargar datos ETH -----
    print('\n[2/4] Cargando datos ETH 4h...')
    df_eth = pd.read_csv(DATA_DIR / 'ETHUSDT_4h.csv', parse_dates=['timestamp'])
    df_eth = df_eth.set_index('timestamp').sort_index()
    if df_eth.index.tz is None:
        df_eth.index = df_eth.index.tz_localize('UTC')
    print(f'  {len(df_eth):,} velas  [{df_eth.index[0].date()} a {df_eth.index[-1].date()}]')

    # ----- Features ETH -----
    print('\n[3/4] Calculando features V14 en ETH...')
    feat_eth = compute_features(df_eth)

    # OOS: mismo periodo que el test walk-forward de BTC
    feat_oos = feat_eth[(feat_eth.index >= OOS_START) & (feat_eth.index <= OOS_END)]
    print(f'  OOS (2022-01 a 2026-01): {len(feat_oos):,} velas')

    # ----- Simulacion ETH -----
    print('\n[4/4] Simulando trades ETH con modelos BTC...')

    setups_oos = detect_setups_long(feat_oos)
    print(f'  Setups detectados (reglas V14 en ETH): {len(setups_oos)}')
    if len(setups_oos) > 0:
        print(f'  Distribucion: {setups_oos["setup_type"].value_counts().to_dict()}')

    # A: Solo reglas (sin ML)
    trades_raw = []
    for ts, srow in setups_oos.iterrows():
        if ts not in df_eth.index:
            continue
        params = SETUP_PARAMS.get(srow['setup_type'], {'tp': 0.03, 'sl': 0.015})
        tp, sl = params['tp'], params['sl']
        entry  = df_eth.loc[ts, 'close']
        future = df_eth.loc[ts:].iloc[1:MAX_CANDLES + 1]
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
            trades_raw.append({'ts': ts, 'setup_type': srow['setup_type'],
                               'outcome': outcome, 'tp': tp, 'sl': sl})

    # B: Reglas + ML BTC
    trades_ml = simulate_trades(df_eth, setups_oos, feat_oos, experts, skip_thr)

    m_raw = metrics(trades_raw)
    m_ml  = metrics(trades_ml)
    be_raw = avg_be_wr(trades_raw)
    be_ml  = avg_be_wr(trades_ml)

    # ----- Comparacion trimestral (misma granularidad que BTC WF) -----
    print()
    print('DETALLE POR TRIMESTRE:')
    print(f"  {'Periodo':>13}  {'Setups':>6}  {'Raw N':>5} {'WR':>6} {'PF':>5}  {'ML N':>5} {'WR':>6} {'PF':>5}")
    print('  ' + '-' * 65)

    quarterly_ml = []
    ts_ref = OOS_START
    while ts_ref <= OOS_END:
        q_end = ts_ref + pd.DateOffset(months=3) - pd.Timedelta(seconds=1)
        mask  = (feat_oos.index >= ts_ref) & (feat_oos.index <= q_end)
        feat_q = feat_oos[mask]
        if len(feat_q) == 0:
            ts_ref += pd.DateOffset(months=3)
            continue

        setups_q = setups_oos[setups_oos.index.isin(feat_q.index)]
        tr_q  = [t for t in trades_raw if t['ts'] >= ts_ref and t['ts'] <= q_end]
        tm_q  = [t for t in trades_ml  if t['ts'] >= ts_ref and t['ts'] <= q_end]
        mr_q  = metrics(tr_q)
        mm_q  = metrics(tm_q)
        be_q  = avg_be_wr(tr_q)
        ok_ml = mm_q['n_trades'] >= 4 and mm_q['wr'] > avg_be_wr(tm_q)
        quarterly_ml.append({'pos': ok_ml, 'n': mm_q['n_trades']})

        fR = 'OK' if (mr_q['n_trades'] >= 4 and mr_q['wr'] > be_q) else \
             ('--' if mr_q['n_trades'] < 4 else 'FL')
        fM = 'OK' if ok_ml else ('--' if mm_q['n_trades'] < 4 else 'FL')
        period = f"{ts_ref.strftime('%Y-%m')}/{q_end.strftime('%m')}"
        print(f"  {period:>13}  {len(setups_q):6d}  "
              f"{mr_q['n_trades']:5d} {mr_q['wr']:5.1%}{fR} {mr_q['pf']:5.2f}  "
              f"{mm_q['n_trades']:5d} {mm_q['wr']:5.1%}{fM} {mm_q['pf']:5.2f}")

        ts_ref += pd.DateOffset(months=3)

    # ----- Resumen -----
    btc_wr = btc_wf.get('avg_wr', 0)
    btc_pf = btc_wf.get('avg_pf', 0)

    print()
    print('=' * 70)
    print('RESUMEN CROSS-ASSET:')
    print()
    print(f'  {"":30}  {"Reglas puras":>14}  {"Reglas+ML BTC":>14}')
    btc_raw_str = f'WR {btc_wr:.1%} | PF {btc_pf:.2f}'
    eth_raw_str = f'WR {m_raw["wr"]:.1%} | {m_raw["n_trades"]}t'
    eth_ml_str  = f'WR {m_ml["wr"]:.1%} | {m_ml["n_trades"]}t'
    print(f'  BTC walk-forward (referencia): {btc_raw_str}')
    print(f'  ETH OOS reglas puras:          {eth_raw_str}')
    print(f'  ETH OOS reglas + ML BTC:       {eth_ml_str}')
    print()

    # Degradacion
    deg_raw = btc_wr - m_raw['wr']
    deg_ml  = btc_wr - m_ml['wr']
    print(f'  Degradacion WR (BTC - ETH):')
    print(f'    Reglas puras:  {deg_raw:+.1%}  {"ALERTA >15pp" if deg_raw > 0.15 else "OK"}')
    print(f'    Reglas + ML:   {deg_ml:+.1%}  {"ALERTA >15pp" if deg_ml > 0.15 else "OK"}')
    print()

    # Criterios de aprobacion
    wr_ok_raw = m_raw['wr'] > be_raw and m_raw['n_trades'] >= 20
    wr_ok_ml  = m_ml['wr']  > be_ml  and m_ml['n_trades']  >= 20
    deg_ok_raw = deg_raw < 0.15
    deg_ok_ml  = deg_ml  < 0.15

    print('CRITERIOS DE APROBACION (ETH):')
    print(f'  WR > break-even ({be_raw:.1%}):')
    print(f'    Reglas puras: {m_raw["wr"]:.1%}  {"OK" if wr_ok_raw else "FAIL"}')
    print(f'    Reglas + ML:  {m_ml["wr"]:.1%}  {"OK" if wr_ok_ml else "FAIL"}')
    print(f'  Degradacion < 15pp:')
    print(f'    Reglas puras: {deg_raw:+.1%}  {"OK" if deg_ok_raw else "ALERTA"}')
    print(f'    Reglas + ML:  {deg_ml:+.1%}  {"OK" if deg_ok_ml else "ALERTA"}')
    print()

    # Veredicto
    print('VEREDICTO CROSS-ASSET:')
    eth_ml_ok = wr_ok_ml and deg_ok_ml
    eth_raw_ok = wr_ok_raw and deg_ok_raw

    if eth_ml_ok:
        print('  Reglas + ML BTC -> ETH: APROBADO')
        print('  El modelo no esta sobreajustado a BTC. Generaliza a ETH.')
    elif eth_raw_ok:
        print('  Reglas puras -> ETH: APROBADO')
        print('  Pero ML BTC no generaliza bien a ETH (WR o degradacion).')
        print('  Usar V14 BTC con reglas puras en ETH (sin ML).')
    else:
        btc_approved = btc_wf.get('approved', False)
        if btc_approved and not eth_raw_ok:
            print('  BTC walk-forward OK pero ETH FALLA -> posible overfitting a BTC.')
            print('  NO conectar ETH. Usar solo BTC.')
        else:
            print('  Resultados insuficientes en ETH.')
            print('  ETH historicamente ha fallado en este proyecto. Usar solo BTC.')

    # ----- Test Feb 2026 en ETH -----
    print()
    print('-' * 70)
    print('TEST INFORMATIVO: Febrero 2026 en ETH')
    mask_feb  = (feat_eth.index >= FEB26_START) & (feat_eth.index < FEB26_END)
    feat_feb  = feat_eth[mask_feb]
    df_feb    = df_eth[df_eth.index >= FEB26_START]
    setups_feb = detect_setups_long(feat_feb)
    trades_feb_ml = simulate_trades(df_feb, setups_feb, feat_feb, experts, skip_thr)
    m_feb = metrics(trades_feb_ml)
    print(f'  Setups ETH Feb 2026: {len(setups_feb)}')
    print(f'  Trades resueltos (ML filter): {m_feb["n_trades"]}  WR {m_feb["wr"]:.1%}')

    print()
    print('=' * 70)


if __name__ == '__main__':
    main()
