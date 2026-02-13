"""
ML V8: Alternative Data Features
=================================
V7 + Funding Rate + Fear & Greed Index = 44 features
Compara V8 vs V7 con mismas condiciones para validar mejora.

Ejecutar: poetry run python -u ml_train_v8.py
"""
import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
import lightgbm as lgb
import optuna
import joblib
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from alt_data_fetcher import download_all_alt_data

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)

PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT',
    'NEAR/USDT', 'APT/USDT', 'ARB/USDT', 'OP/USDT', 'SUI/USDT',
]

COMMISSION = 0.0004
SLIPPAGE = 0.0001
START_CAPITAL = 100.0
MIN_TRAIN_CANDLES = 3000

# === RISK MANAGEMENT ===
MAX_CONCURRENT = 3
MAX_DD_PCT = 0.20
MAX_DAILY_LOSS_PCT = 0.05
RISK_PER_TRADE = 0.02
MAX_NOTIONAL = 500.0
TRAILING_ACTIVATION = 0.5
TRAILING_LOCK = 0.4

LEV = {'BULL': 5, 'BEAR': 4, 'RANGE': 3}
ATR_TP = {'BULL': 3.0, 'BEAR': 2.5, 'RANGE': 1.5}
ATR_SL = {'BULL': 1.5, 'BEAR': 1.5, 'RANGE': 1.0}


# ============================================================
# 1. DOWNLOAD (identico a V7)
# ============================================================
def download_pair(symbol, timeframe='4h', since='2020-01-01'):
    safe = symbol.replace('/', '_')
    cache = DATA_DIR / f'{safe}_{timeframe}_history.parquet'
    if cache.exists() and (time.time() - cache.stat().st_mtime) / 3600 < 24:
        return pd.read_parquet(cache)
    exchange = ccxt.binance({'enableRateLimit': True})
    since_ts = int(datetime.strptime(since, '%Y-%m-%d').timestamp() * 1000)
    rows = []
    while True:
        try:
            c = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=1000)
        except Exception as e:
            print(f'    Error {symbol}: {e}'); time.sleep(5); continue
        if not c: break
        rows.extend(c)
        since_ts = c[-1][0] + 1
        if len(c) < 1000: break
        time.sleep(0.15)
    if not rows: return None
    df = pd.DataFrame(rows, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    df.to_parquet(cache)
    return df


def download_all():
    print('=' * 70)
    print('DATOS')
    print('=' * 70)
    data = {}
    for pair in PAIRS:
        df = download_pair(pair, '4h')
        if df is not None and len(df) > 500:
            data[pair] = df
            print(f'  {pair:<12} {len(df):>6,} velas | {df.index[0].date()} a {df.index[-1].date()}')
    btc_d = download_pair('BTC/USDT', '1d')
    print(f'  BTC daily   {len(btc_d):>6,} velas (regime)')

    # V8: datos alternativos
    funding_data, fng = download_all_alt_data(PAIRS)

    return data, btc_d, funding_data, fng


# ============================================================
# 2. REGIME DETECTION
# ============================================================
def detect_regime(btc_daily):
    c = btc_daily['close'].copy()
    h, l = btc_daily['high'].copy(), btc_daily['low'].copy()
    ema20 = ta.ema(c, length=20)
    ema50 = ta.ema(c, length=50)
    ret_20 = c.pct_change(20)

    regime = pd.Series('RANGE', index=btc_daily.index)
    bull = (c > ema50) & (ema20 > ema50) & (ret_20 > 0.05)
    bear = (c < ema50) & (ema20 < ema50) & (ret_20 < -0.05)
    regime[bull] = 'BULL'
    regime[bear] = 'BEAR'

    # Suavizar: mantener regimen minimo 5 dias
    current = 'RANGE'
    count = 0
    for i in range(len(regime)):
        if regime.iloc[i] != current:
            count += 1
            if count >= 5:
                current = regime.iloc[i]
                count = 0
            else:
                regime.iloc[i] = current
        else:
            count = 0
    return regime


def detect_regime_v8(btc_daily, fng_df=None):
    """V8: Regime base + confirmacion con Fear & Greed."""
    regime = detect_regime(btc_daily)

    if fng_df is not None and len(fng_df) > 0:
        # Shift(1) para evitar look-ahead, luego ffill a daily
        fng_shifted = fng_df['fng_value'].shift(1)
        fng_daily = fng_shifted.reindex(btc_daily.index, method='ffill')

        # Degradar regimes contradictorios a RANGE
        bull_but_fear = (regime == 'BULL') & (fng_daily < 25)
        bear_but_greed = (regime == 'BEAR') & (fng_daily > 75)
        regime[bull_but_fear] = 'RANGE'
        regime[bear_but_greed] = 'RANGE'

    return regime


def map_regime_4h(regime_daily, idx_4h):
    return regime_daily.reindex(idx_4h, method='ffill').fillna('RANGE')


# ============================================================
# 3. FEATURES
# ============================================================
def compute_features(df):
    """V7: 34 features de TA puro (identico a V7)."""
    feat = pd.DataFrame(index=df.index)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    for p in [1, 3, 5, 10, 20]:
        feat[f'ret_{p}'] = c.pct_change(p)

    feat['atr14'] = ta.atr(h, l, c, length=14)
    feat['atr_r'] = feat['atr14'] / feat['atr14'].rolling(50).mean()
    feat['vol5'] = c.pct_change().rolling(5).std()
    feat['vol20'] = c.pct_change().rolling(20).std()

    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)
    sr = ta.stochrsi(c, length=14, rsi_length=14, k=3, d=3)
    if sr is not None: feat['srsi_k'] = sr.iloc[:, 0]
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None: feat['macd_h'] = macd.iloc[:, 1]
    feat['roc5'] = ta.roc(c, length=5)
    feat['roc20'] = ta.roc(c, length=20)

    for el in [8, 21, 55, 100, 200]:
        e = ta.ema(c, length=el)
        feat[f'ema{el}_d'] = (c - e) / e * 100
    feat['ema8_sl'] = ta.ema(c, length=8).pct_change(3) * 100
    feat['ema55_sl'] = ta.ema(c, length=55).pct_change(5) * 100

    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        bw = bb.iloc[:, 2] - bb.iloc[:, 0]
        feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw
        feat['bb_w'] = bw / bb.iloc[:, 1] * 100

    feat['vr'] = v / v.rolling(20).mean()
    feat['spr'] = (h - l) / c * 100
    feat['body'] = abs(c - df['open']) / (h - l + 1e-10)

    ax = ta.adx(h, l, c, length=14)
    if ax is not None:
        feat['adx'] = ax.iloc[:, 0]
        feat['dip'] = ax.iloc[:, 1]
        feat['dim'] = ax.iloc[:, 2]

    hr = df.index.hour; dw = df.index.dayofweek
    feat['h_s'] = np.sin(2*np.pi*hr/24)
    feat['h_c'] = np.cos(2*np.pi*hr/24)
    feat['d_s'] = np.sin(2*np.pi*dw/7)
    feat['d_c'] = np.cos(2*np.pi*dw/7)

    return feat


def compute_features_v8(df, funding_df=None, fng_df=None):
    """V8: 34 features TA + 10 features alt data = 44 total."""
    feat = compute_features(df)
    idx = df.index

    # === FUNDING RATE (5 features) ===
    if funding_df is not None and len(funding_df) > 0:
        # CRITICAL: shift(1) antes de reindex para evitar look-ahead
        fr_shifted = funding_df['funding_rate'].shift(1)
        fr_4h = fr_shifted.reindex(idx, method='ffill')

        feat['fr'] = fr_4h
        feat['fr_ma8'] = fr_4h.rolling(8, min_periods=1).mean()
        fr_ma50 = fr_4h.rolling(50, min_periods=10).mean()
        fr_std50 = fr_4h.rolling(50, min_periods=10).std()
        feat['fr_z'] = (fr_4h - fr_ma50) / (fr_std50 + 1e-10)
        feat['fr_cumsum5'] = fr_4h.rolling(5, min_periods=1).sum()
        feat['fr_extreme'] = (feat['fr_z'].abs() > 2).astype(float)
    else:
        for col in ['fr', 'fr_ma8', 'fr_z', 'fr_cumsum5', 'fr_extreme']:
            feat[col] = np.nan

    # === FEAR & GREED INDEX (5 features) ===
    if fng_df is not None and len(fng_df) > 0:
        # F&G es diario. shift(1) para evitar look-ahead del mismo dia
        fng_shifted = fng_df['fng_value'].shift(1)
        fng_4h = fng_shifted.reindex(idx, method='ffill')

        feat['fng'] = fng_4h
        feat['fng_ma7'] = fng_4h.rolling(42, min_periods=6).mean()  # 7 dias * 6 bars/dia
        feat['fng_chg7'] = fng_4h - fng_4h.shift(42)
        feat['fng_extreme_fear'] = (fng_4h < 20).astype(float)
        feat['fng_extreme_greed'] = (fng_4h > 80).astype(float)
    else:
        for col in ['fng', 'fng_ma7', 'fng_chg7', 'fng_extreme_fear', 'fng_extreme_greed']:
            feat[col] = np.nan

    return feat


# ============================================================
# 4. TRAIN MODELS
# ============================================================
def train_all(all_data, train_end, horizon=5, n_trials=40,
              funding_data=None, fng_df=None, use_v8=True):
    """Entrena modelos. Si use_v8=True usa 44 features, si no 34 (V7)."""
    tag = 'V8' if use_v8 else 'V7'
    nf = 44 if use_v8 else 34
    print(f'\n  Entrenando {tag} ({nf}f, train hasta {train_end}, horizonte {horizon})...')
    models = {}
    for pair, df in all_data.items():
        if use_v8:
            feat = compute_features_v8(df, funding_data.get(pair) if funding_data else None, fng_df)
        else:
            feat = compute_features(df)

        feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
        fcols = list(feat.columns)

        tgt = f'fwd_{horizon}'
        feat[tgt] = (df['close'].shift(-horizon) - df['close']) / df['close']
        ds = feat.dropna()

        tr = ds.loc[:train_end]
        te = ds.loc[train_end:]

        if len(tr) < MIN_TRAIN_CANDLES:
            print(f'    {pair:<12} OMITIDO (solo {len(tr)} velas, min {MIN_TRAIN_CANDLES})')
            continue
        if len(te) < 30:
            continue

        X_tr = tr[fcols].fillna(0)
        y_tr = tr[tgt].clip(tr[tgt].quantile(0.01), tr[tgt].quantile(0.99))

        sp = int(len(X_tr) * 0.85)
        Xt, Xv = X_tr.iloc[:sp], X_tr.iloc[sp:]
        yt, yv = y_tr.iloc[:sp], y_tr.iloc[sp:]

        def obj(trial):
            p = {
                'n_estimators': trial.suggest_int('ne', 100, 500),
                'max_depth': trial.suggest_int('md', 3, 8),
                'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('ss', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('cs', 0.4, 1.0),
                'min_child_samples': trial.suggest_int('mc', 10, 80),
                'reg_alpha': trial.suggest_float('ra', 1e-6, 5.0, log=True),
                'reg_lambda': trial.suggest_float('rl', 1e-6, 5.0, log=True),
                'num_leaves': trial.suggest_int('nl', 15, 80),
                'random_state': 42, 'n_jobs': -1, 'verbose': -1
            }
            m = lgb.LGBMRegressor(**p)
            m.fit(Xt, yt, eval_set=[(Xv, yv)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
            pr = m.predict(Xv)
            c = np.corrcoef(pr, yv.values)[0, 1]
            return c if not np.isnan(c) else 0.0

        study = optuna.create_study(direction='maximize')
        study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

        bp = study.best_params.copy()
        bp.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})
        rename = {'ne':'n_estimators','md':'max_depth','lr':'learning_rate',
                  'ss':'subsample','cs':'colsample_bytree','mc':'min_child_samples',
                  'ra':'reg_alpha','rl':'reg_lambda','nl':'num_leaves'}
        bp2 = {rename.get(k, k): v for k, v in bp.items()}
        mdl = lgb.LGBMRegressor(**bp2)
        mdl.fit(X_tr, y_tr, eval_set=[(Xv, yv)],
                callbacks=[lgb.early_stopping(50, verbose=False)])

        Xte = te[fcols].fillna(0)
        yte = te[tgt]
        preds = mdl.predict(Xte)
        corr = np.corrcoef(preds, yte.values)[0, 1]
        ps = np.std(preds)

        models[pair] = {
            'model': mdl, 'fcols': fcols, 'pred_std': ps,
            'corr': corr, 'preds': preds, 'index': te.index,
        }
        print(f'    {pair:<12} train={len(tr):>5,} | test={len(te):>4,} | '
              f'corr={corr:.4f} | {len(fcols)}f')

    return models


# ============================================================
# 5. BACKTEST (identico a V7)
# ============================================================
def backtest(all_data, regime_daily, models, thresh=0.7,
             use_trailing=True, use_atr=True, use_compound=False,
             max_notional=MAX_NOTIONAL):
    all_t = set()
    for p in models:
        if p in all_data:
            all_t.update(all_data[p].index.tolist())
    timeline = sorted(all_t)
    if not timeline:
        return [], [], 0, False

    regime_4h = map_regime_4h(regime_daily, pd.DatetimeIndex(timeline))

    pred_dict = {}
    atr_dict = {}
    for pair, info in models.items():
        pred_dict[pair] = dict(zip(info['index'], info['preds']))
        df = all_data[pair]
        atr_dict[pair] = ta.atr(df['high'], df['low'], df['close'], length=14)

    balance = START_CAPITAL
    peak = START_CAPITAL
    max_dd = 0.0
    positions = {}
    trades = []
    daily_pnl = defaultdict(float)
    paused_until = None
    killed = False

    for t in timeline:
        if killed: break
        ds = t.strftime('%Y-%m-%d')

        if paused_until and t.date() <= paused_until:
            continue

        reg = regime_4h.loc[t] if t in regime_4h.index else 'RANGE'
        lev = LEV[reg]

        # === UPDATE POSITIONS ===
        to_close = []
        for pair, pos in positions.items():
            df = all_data[pair]
            if t not in df.index: continue

            bh = df.loc[t, 'high']
            bl = df.loc[t, 'low']
            bc = df.loc[t, 'close']
            pos['bars'] += 1

            if use_trailing and pos.get('trail_on'):
                if pos['dir'] == 1:
                    new = bh * (1 - TRAILING_LOCK * pos['atr_pct'])
                    pos['trail_sl'] = max(pos.get('trail_sl', 0), new)
                else:
                    new = bl * (1 + TRAILING_LOCK * pos['atr_pct'])
                    pos['trail_sl'] = min(pos.get('trail_sl', 1e15), new)

            ex_price = ex_reason = None
            if pos['dir'] == 1:
                if bh >= pos['tp']: ex_price, ex_reason = pos['tp'], 'TP'
                elif use_trailing and pos.get('trail_sl') and bl <= pos['trail_sl']:
                    ex_price, ex_reason = pos['trail_sl'], 'TRAIL'
                elif bl <= pos['sl']: ex_price, ex_reason = pos['sl'], 'SL'
                elif pos['bars'] >= pos['mh']: ex_price, ex_reason = bc, 'TIMEOUT'
                elif use_trailing and not pos.get('trail_on'):
                    g = (bh - pos['entry']) / pos['entry']
                    if g >= pos['tp_pct'] * TRAILING_ACTIVATION:
                        pos['trail_on'] = True
                        pos['trail_sl'] = pos['entry'] * (1 + g * 0.3)
            else:
                if bl <= pos['tp']: ex_price, ex_reason = pos['tp'], 'TP'
                elif use_trailing and pos.get('trail_sl') and bh >= pos['trail_sl']:
                    ex_price, ex_reason = pos['trail_sl'], 'TRAIL'
                elif bh >= pos['sl']: ex_price, ex_reason = pos['sl'], 'SL'
                elif pos['bars'] >= pos['mh']: ex_price, ex_reason = bc, 'TIMEOUT'
                elif use_trailing and not pos.get('trail_on'):
                    g = (pos['entry'] - bl) / pos['entry']
                    if g >= pos['tp_pct'] * TRAILING_ACTIVATION:
                        pos['trail_on'] = True
                        pos['trail_sl'] = pos['entry'] * (1 - g * 0.3)

            if ex_price:
                if pos['dir'] == 1:
                    gpnl = (ex_price - pos['entry']) / pos['entry']
                else:
                    gpnl = (pos['entry'] - ex_price) / pos['entry']
                cost = pos['not'] * (COMMISSION + SLIPPAGE) * 2
                pnl = pos['not'] * gpnl - cost
                balance += pnl
                daily_pnl[ds] += pnl

                trades.append({
                    'pair': pair, 'dir': pos['dir'], 'pnl': pnl,
                    'pnl_pct': gpnl*100, 'reason': ex_reason,
                    'bars': pos['bars'], 'time': t, 'regime': pos['reg'],
                    'lev': pos['lev'], 'conf': pos['conf'], 'bal': balance,
                })
                to_close.append(pair)

        for p in to_close:
            del positions[p]

        if balance > peak: peak = balance
        dd = (peak - balance) / peak if peak > 0 else 0
        if dd > max_dd: max_dd = dd
        if dd >= MAX_DD_PCT:
            killed = True; break

        if daily_pnl[ds] < -(START_CAPITAL * MAX_DAILY_LOSS_PCT):
            paused_until = t.date(); continue

        # === OPEN NEW POSITIONS ===
        if len(positions) >= MAX_CONCURRENT: continue

        cands = []
        for pair, info in models.items():
            if pair in positions: continue
            if pair not in all_data: continue
            df = all_data[pair]
            if t not in df.index: continue

            pred = pred_dict[pair].get(t)
            if pred is None: continue
            ps = info['pred_std']
            if ps < 1e-8: continue
            conf = abs(pred) / ps
            if conf < thresh: continue

            d = 1 if pred > 0 else -1
            if reg == 'BULL' and d == -1: continue
            if reg == 'BEAR' and d == 1: continue

            same = sum(1 for p in positions.values() if p['dir'] == d)
            if same >= 2: continue
            cands.append({'pair': pair, 'dir': d, 'conf': conf})

        cands.sort(key=lambda x: x['conf'], reverse=True)

        for cand in cands:
            if len(positions) >= MAX_CONCURRENT: break
            pair = cand['pair']
            d = cand['dir']
            conf = cand['conf']
            df = all_data[pair]
            entry = df.loc[t, 'close']

            atr_val = atr_dict[pair].get(t) if hasattr(atr_dict[pair], 'get') else (
                atr_dict[pair].loc[t] if t in atr_dict[pair].index else None)

            if use_atr and atr_val and not np.isnan(atr_val):
                atr_pct = atr_val / entry
                tp_pct = max(0.005, min(0.08, atr_pct * ATR_TP[reg]))
                sl_pct = max(0.003, min(0.04, atr_pct * ATR_SL[reg]))
            else:
                tp_pct = 0.03; sl_pct = 0.015

            mh = 30 if reg != 'RANGE' else 15

            risk_pct = RISK_PER_TRADE
            if conf > 2.0: risk_pct = 0.03
            elif conf > 1.5: risk_pct = 0.025

            base = balance if use_compound else START_CAPITAL
            risk_amt = base * risk_pct
            margin = risk_amt / (sl_pct * lev) if sl_pct > 0 else risk_amt
            notional = margin * lev
            notional = min(notional, max_notional)
            margin = notional / lev

            if d == 1:
                tp_pr = entry * (1 + tp_pct)
                sl_pr = entry * (1 - sl_pct)
            else:
                tp_pr = entry * (1 - tp_pct)
                sl_pr = entry * (1 + sl_pct)

            positions[pair] = {
                'entry': entry, 'dir': d, 'tp': tp_pr, 'sl': sl_pr,
                'tp_pct': tp_pct, 'sl_pct': sl_pct,
                'atr_pct': atr_val/entry if atr_val else 0.02,
                'not': notional, 'lev': lev, 'mh': mh,
                'bars': 0, 'conf': conf, 'reg': reg,
                'trail_on': False, 'trail_sl': None,
            }

    return trades, [], max_dd, killed


# ============================================================
# 6. METRICS (identico a V7)
# ============================================================
def print_results(trades, max_dd, killed, label):
    if not trades:
        print(f'\n  {label}: Sin trades.'); return

    final = trades[-1]['bal']
    ret = ((final / START_CAPITAL) - 1) * 100
    total = len(trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    wr = wins / total * 100 if total else 0
    gp = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gl = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
    pf = gp / gl if gl > 0 else 0

    days = (trades[-1]['time'] - trades[0]['time']).days
    years = max(days / 365, 0.1)
    if final > 0 and years > 0.1:
        annual = ((final / START_CAPITAL) ** (1 / years) - 1) * 100
    else:
        annual = ret / years if years > 0 else 0

    dr = defaultdict(float)
    for t in trades: dr[t['time'].strftime('%Y-%m-%d')] += t['pnl']
    rv = list(dr.values()) if dr else [0]
    sharpe = (np.mean(rv) / np.std(rv) * np.sqrt(252)) if np.std(rv) > 0 else 0

    mc = cc = 0
    for t in trades:
        if t['pnl'] < 0: cc += 1; mc = max(mc, cc)
        else: cc = 0

    wins_pnl = [t['pnl'] for t in trades if t['pnl'] > 0]
    loss_pnl = [t['pnl'] for t in trades if t['pnl'] <= 0]
    avg_w = np.mean(wins_pnl) if wins_pnl else 0
    avg_l = np.mean(loss_pnl) if loss_pnl else 0

    print(f'\n  {"="*65}')
    print(f'  {label}')
    print(f'  {"="*65}')
    print(f'  Capital:       ${START_CAPITAL:.0f} a ${final:.2f}')
    print(f'  Retorno total: {ret:+.1f}%')
    print(f'  Retorno anual: {annual:+.1f}%')
    print(f'  Trades:        {total} ({total/max(years,0.5):.0f}/ano)')
    print(f'  Win Rate:      {wr:.0f}%')
    print(f'  Profit Factor: {pf:.2f}')
    print(f'  Avg Win/Loss:  ${avg_w:+.2f} / ${avg_l:.2f}')
    print(f'  Max DD:        {max_dd*100:.1f}%')
    print(f'  Sharpe:        {sharpe:.2f}')
    print(f'  Max consec L:  {mc}')
    print(f'  Periodo:       {days} dias ({years:.1f} anos)')
    if killed: print(f'  ** KILL SWITCH **')

    # Por regimen
    rs = {}
    for t in trades:
        r = t.get('regime', 'RANGE')
        if r not in rs: rs[r] = {'w': 0, 'n': 0, 'pnl': 0}
        rs[r]['n'] += 1; rs[r]['pnl'] += t['pnl']
        if t['pnl'] > 0: rs[r]['w'] += 1
    print(f'\n  Regimen:')
    for r in ['BULL', 'BEAR', 'RANGE']:
        if r in rs:
            s = rs[r]
            print(f'    {r:<6} {s["n"]:>4}t | WR {s["w"]/s["n"]*100:.0f}% | PnL ${s["pnl"]:+.2f}')

    # Por razon
    rr = defaultdict(lambda: {'n': 0, 'pnl': 0})
    for t in trades:
        rr[t['reason']]['n'] += 1; rr[t['reason']]['pnl'] += t['pnl']
    print(f'\n  Razon de cierre:')
    for r in ['TP', 'TRAIL', 'SL', 'TIMEOUT']:
        if r in rr: print(f'    {r:<8} {rr[r]["n"]:>4}t | PnL ${rr[r]["pnl"]:+.2f}')

    # Top pares
    pp = defaultdict(lambda: {'n': 0, 'pnl': 0, 'w': 0})
    for t in trades:
        pp[t['pair']]['n'] += 1; pp[t['pair']]['pnl'] += t['pnl']
        if t['pnl'] > 0: pp[t['pair']]['w'] += 1
    print(f'\n  Top 5 pares:')
    for p, s in sorted(pp.items(), key=lambda x: x[1]['pnl'], reverse=True)[:5]:
        print(f'    {p:<12} {s["n"]:>3}t | WR {s["w"]/s["n"]*100:.0f}% | PnL ${s["pnl"]:+.2f}')

    # Monthly
    mo = defaultdict(lambda: {'pnl': 0, 'n': 0})
    for t in trades:
        m = t['time'].strftime('%Y-%m')
        mo[m]['pnl'] += t['pnl']; mo[m]['n'] += 1
    print(f'\n  Equity mensual:')
    for m in sorted(mo.keys()):
        pnl = mo[m]['pnl']; nt = mo[m]['n']
        bl = min(int(abs(pnl) / 1), 25)
        bar = ('+' * bl) if pnl >= 0 else ('-' * bl)
        print(f'    {m}: {nt:>3}t | ${pnl:>+7.2f} | {bar}')

    wm = sum(1 for v in mo.values() if v['pnl'] > 0)
    print(f'  Meses positivos: {wm}/{len(mo)} ({wm/len(mo)*100:.0f}%)')


def print_feature_importance(models, label, top_n=15):
    """Muestra importancia promedio de features."""
    print(f'\n  Feature Importance ({label}, top {top_n}):')
    all_imp = defaultdict(list)
    for pair, info in models.items():
        mdl = info['model']
        fcols = info['fcols']
        imp = mdl.feature_importances_
        for f, i in zip(fcols, imp):
            all_imp[f].append(i)

    avg_imp = {f: np.mean(v) for f, v in all_imp.items()}
    sorted_imp = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)

    for i, (f, v) in enumerate(sorted_imp[:top_n]):
        bar = '#' * min(int(v / 5), 40)
        tag = ' <-- ALT' if f in ['fr','fr_ma8','fr_z','fr_cumsum5','fr_extreme',
                                    'fng','fng_ma7','fng_chg7','fng_extreme_fear',
                                    'fng_extreme_greed'] else ''
        print(f'    {i+1:>2}. {f:<20} {v:>6.1f} {bar}{tag}')

    # Resumen alt features
    alt_feats = ['fr','fr_ma8','fr_z','fr_cumsum5','fr_extreme',
                 'fng','fng_ma7','fng_chg7','fng_extreme_fear','fng_extreme_greed']
    alt_total = sum(avg_imp.get(f, 0) for f in alt_feats)
    all_total = sum(avg_imp.values())
    pct = alt_total / all_total * 100 if all_total > 0 else 0
    print(f'\n  Alt data features: {pct:.1f}% de importancia total')


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    all_data, btc_d, funding_data, fng = download_all()
    if len(all_data) < 3 or btc_d is None:
        print('ERROR: datos insuficientes'); return

    # Regime V7 (base)
    regime_v7 = detect_regime(btc_d)
    # Regime V8 (con F&G confirmation)
    regime_v8 = detect_regime_v8(btc_d, fng)

    for label, reg in [('V7', regime_v7), ('V8', regime_v8)]:
        bull = (reg == 'BULL').sum()
        bear = (reg == 'BEAR').sum()
        rng = (reg == 'RANGE').sum()
        print(f'\n  Regimen {label}: BULL {bull}d ({bull/len(reg)*100:.0f}%) | '
              f'BEAR {bear}d ({bear/len(reg)*100:.0f}%) | '
              f'RANGE {rng}d ({rng/len(reg)*100:.0f}%)')

    # =========================================================
    # WALK-FORWARD COMPARISON
    # =========================================================
    print('\n' + '=' * 70)
    print('WALK-FORWARD: V7 vs V8')
    print('=' * 70)

    folds = [
        ('2023-06', '2023-06', '2024-01', 'H2-2023'),
        ('2024-01', '2024-01', '2024-07', 'H1-2024'),
        ('2024-07', '2024-07', '2025-01', 'H2-2024'),
        ('2025-01', '2025-01', '2026-03', '2025+'),
    ]

    for version, use_v8, regime in [('V7', False, regime_v7), ('V8', True, regime_v8)]:
        print(f'\n  --- {version} Walk-Forward ---')
        fold_res = []
        for tr_e, te_s, te_e, fn in folds:
            mdls = train_all(all_data, tr_e, horizon=5, n_trials=30,
                             funding_data=funding_data, fng_df=fng, use_v8=use_v8)
            mdls_f = {}
            for p, i in mdls.items():
                m = (i['index'] >= te_s) & (i['index'] < te_e)
                if m.sum() > 0:
                    mdls_f[p] = {**i, 'preds': i['preds'][m], 'index': i['index'][m]}
            if not mdls_f: continue

            tr, _, mdd, kil = backtest(all_data, regime, mdls_f,
                                        use_trailing=True, use_atr=True,
                                        use_compound=False, max_notional=300)
            if not tr: continue

            fin = tr[-1]['bal']
            ret = ((fin / START_CAPITAL) - 1) * 100
            tot = len(tr)
            w = sum(1 for t in tr if t['pnl'] > 0)
            wr = w/tot*100 if tot else 0
            gpp = sum(t['pnl'] for t in tr if t['pnl'] > 0)
            gll = abs(sum(t['pnl'] for t in tr if t['pnl'] <= 0))
            pf = gpp/gll if gll > 0 else 0

            mk = ' ***' if ret > 20 else (' **' if ret > 5 else (' *' if ret > 0 else ' --'))
            print(f'    {fn}: {tot:>4}t | WR {wr:.0f}% | PF {pf:.2f} | '
                  f'Ret {ret:+.1f}% | DD {mdd*100:.1f}%{mk}{" KILLED" if kil else ""}')
            fold_res.append({'fn': fn, 'ret': ret, 'pf': pf, 'dd': mdd*100, 'n': tot, 'k': kil})

        if fold_res:
            prof = sum(1 for f in fold_res if f['ret'] > 0)
            print(f'\n    RESUMEN {version}: {prof}/{len(fold_res)} folds rentables | '
                  f'PF avg {np.mean([f["pf"] for f in fold_res]):.2f} | '
                  f'Ret avg {np.mean([f["ret"] for f in fold_res]):+.1f}% | '
                  f'DD avg {np.mean([f["dd"] for f in fold_res]):.1f}%')

    # =========================================================
    # FULL BACKTEST COMPARISON
    # =========================================================
    print('\n' + '=' * 70)
    print('BACKTEST COMPLETO V7 vs V8 (FLAT + ATR + Trailing)')
    print('=' * 70)

    # V7 baseline
    models_v7 = train_all(all_data, '2024-01', horizon=5, n_trials=40,
                          funding_data=funding_data, fng_df=fng, use_v8=False)
    tr_v7, _, mdd_v7, kil_v7 = backtest(all_data, regime_v7, models_v7,
                                          use_trailing=True, use_atr=True,
                                          use_compound=False, max_notional=300)
    print_results(tr_v7, mdd_v7, kil_v7, 'V7 BASELINE (34 features)')
    print_feature_importance(models_v7, 'V7')

    # V8 con alt data
    models_v8 = train_all(all_data, '2024-01', horizon=5, n_trials=40,
                          funding_data=funding_data, fng_df=fng, use_v8=True)
    tr_v8, _, mdd_v8, kil_v8 = backtest(all_data, regime_v8, models_v8,
                                          use_trailing=True, use_atr=True,
                                          use_compound=False, max_notional=300)
    print_results(tr_v8, mdd_v8, kil_v8, 'V8 ALT DATA (44 features)')
    print_feature_importance(models_v8, 'V8', top_n=20)

    # =========================================================
    # COMPARISON SUMMARY
    # =========================================================
    print('\n' + '=' * 70)
    print('COMPARACION FINAL V7 vs V8')
    print('=' * 70)

    def _metrics(trades, max_dd):
        if not trades: return {}
        final = trades[-1]['bal']
        total = len(trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        gp = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gl = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
        days = (trades[-1]['time'] - trades[0]['time']).days
        years = max(days / 365, 0.1)
        dr = defaultdict(float)
        for t in trades: dr[t['time'].strftime('%Y-%m-%d')] += t['pnl']
        rv = list(dr.values())
        return {
            'final': final, 'ret': ((final/START_CAPITAL)-1)*100,
            'annual': ((final/START_CAPITAL)**(1/years)-1)*100 if final > 0 else 0,
            'trades': total, 'wr': wins/total*100, 'pf': gp/gl if gl > 0 else 0,
            'dd': max_dd*100,
            'sharpe': np.mean(rv)/np.std(rv)*np.sqrt(252) if np.std(rv) > 0 else 0,
        }

    m7 = _metrics(tr_v7, mdd_v7)
    m8 = _metrics(tr_v8, mdd_v8)

    if m7 and m8:
        print(f'\n  {"Metrica":<20} {"V7":>10} {"V8":>10} {"Delta":>10}')
        print(f'  {"-"*50}')
        for key, label, fmt, dfmt in [
            ('ret', 'Retorno %', '+.1f', '+.1f'), ('annual', 'Anual %', '+.1f', '+.1f'),
            ('trades', 'Trades', '.0f', '+.0f'), ('wr', 'Win Rate %', '.0f', '+.0f'),
            ('pf', 'Profit Factor', '.2f', '+.2f'), ('dd', 'Max DD %', '.1f', '+.1f'),
            ('sharpe', 'Sharpe', '.2f', '+.2f'),
        ]:
            v7 = m7.get(key, 0); v8 = m8.get(key, 0)
            d = v8 - v7
            better = d > 0 if key != 'dd' else d < 0
            arrow = ' <<<' if better else ''
            d_str = f'{d:{dfmt}}'
            print(f'  {label:<20} {v7:>10{fmt}} {v8:>10{fmt}} {d_str:>10}{arrow}')

        # Veredicto
        print(f'\n  VEREDICTO:')
        wins_v8 = 0
        if m8['pf'] >= m7['pf']: wins_v8 += 1; print(f'    [OK] PF: V8 >= V7')
        else: print(f'    [X]  PF: V8 < V7')
        if m8['dd'] <= m7['dd'] + 5: wins_v8 += 1; print(f'    [OK] DD: V8 aceptable')
        else: print(f'    [X]  DD: V8 peor que V7')
        if m8['sharpe'] >= m7['sharpe']: wins_v8 += 1; print(f'    [OK] Sharpe: V8 >= V7')
        else: print(f'    [X]  Sharpe: V8 < V7')

        if wins_v8 >= 2:
            print(f'\n  >>> V8 APROBADO ({wins_v8}/3 criterios). Listo para deploy en paralelo.')
        else:
            print(f'\n  >>> V8 NO APROBADO ({wins_v8}/3). Necesita mas trabajo.')

    print(f'\nTiempo total: {(time.time()-t0)/60:.1f} minutos')


if __name__ == '__main__':
    main()
