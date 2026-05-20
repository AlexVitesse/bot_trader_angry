"""
DOT Hybrid System - V13.05
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
TP_PCT = 0.03
SL_PCT = 0.015
TRAIN_END = '2024-12-31'
TEST_START = '2025-09-01'


def load_data():
    df = pd.read_parquet(DATA_DIR / 'DOT_USDT_4h_full.parquet')
    return df.sort_index()


def compute_features(df):
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    feat = pd.DataFrame(index=df.index)

    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)

    bb = ta.bbands(c, length=20)
    if bb is not None:
        feat['bb_pct'] = (c - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])
        feat['bb_width'] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / bb.iloc[:, 1] * 100

    ema20 = ta.ema(c, length=20)
    ema50 = ta.ema(c, length=50)
    ema200 = ta.ema(c, length=200)
    feat['ema20_dist'] = (c - ema20) / ema20 * 100
    feat['ema50_dist'] = (c - ema50) / ema50 * 100
    feat['ema200_dist'] = (c - ema200) / ema200 * 100

    adx = ta.adx(h, l, c, length=14)
    if adx is not None:
        feat['adx'] = adx.iloc[:, 0]
        feat['di_diff'] = adx.iloc[:, 1] - adx.iloc[:, 2]

    feat['atr_pct'] = ta.atr(h, l, c, length=14) / c * 100
    feat['vol_ratio'] = v / v.rolling(20).mean()

    chop = ta.chop(h, l, c, length=14)
    if chop is not None:
        feat['chop'] = chop

    feat['ret_1'] = c.pct_change(1) * 100
    feat['ret_5'] = c.pct_change(5) * 100
    feat['ret_20'] = c.pct_change(20) * 100
    feat['consec_down'] = (c < c.shift(1)).rolling(10).sum()
    feat['hour'] = df.index.hour
    feat['hour_sin'] = np.sin(2 * np.pi * feat['hour'] / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * feat['hour'] / 24)

    return feat


def detect_setups(feat):
    setups = pd.DataFrame(index=feat.index)
    setups['has_setup'] = False
    setups['setup_type'] = ''

    for idx in feat.index:
        row = feat.loc[idx]
        rsi14 = row.get('rsi14', 50)
        rsi7 = row.get('rsi7', 50)
        bb_pct = row.get('bb_pct', 0.5)
        consec_down = row.get('consec_down', 0)
        vol_ratio = row.get('vol_ratio', 1)
        ema200_dist = row.get('ema200_dist', 0)
        ema20_dist = row.get('ema20_dist', 0)
        adx = row.get('adx', 20)

        if pd.isna(rsi14) or pd.isna(bb_pct):
            continue

        if rsi14 < 25 and bb_pct < 0.3:
            setups.loc[idx, 'has_setup'] = True
            setups.loc[idx, 'setup_type'] = 'RSI_EXTREME'
        elif consec_down >= 4 and rsi14 < 35 and vol_ratio > 1.3:
            setups.loc[idx, 'has_setup'] = True
            setups.loc[idx, 'setup_type'] = 'CAPITULATION'
        elif ema200_dist > 5 and adx > 25 and rsi14 < 40 and ema20_dist < -1:
            setups.loc[idx, 'has_setup'] = True
            setups.loc[idx, 'setup_type'] = 'TREND_PULLBACK'
        elif rsi7 < 25 and rsi14 > rsi7 + 5 and bb_pct < 0.2:
            setups.loc[idx, 'has_setup'] = True
            setups.loc[idx, 'setup_type'] = 'DIVERGENCE'

    return setups


def backtest_setups(df, feat, setups, use_filter=False, model=None, threshold=0.45):
    trades = []
    position = None
    context_cols = ['adx', 'di_diff', 'chop', 'atr_pct', 'vol_ratio',
                    'bb_width', 'ret_5', 'ret_20', 'hour_sin', 'hour_cos']

    for idx in df.index:
        price = df.loc[idx, 'close']

        if position is not None:
            pnl_pct = (price - position['entry']) / position['entry']
            if pnl_pct >= TP_PCT:
                trades.append({'pnl': pnl_pct, 'result': 'WIN', 'setup': position['setup']})
                position = None
            elif pnl_pct <= -SL_PCT:
                trades.append({'pnl': pnl_pct, 'result': 'LOSS', 'setup': position['setup']})
                position = None

        if position is None and idx in setups.index:
            if setups.loc[idx, 'has_setup']:
                should_enter = True
                if use_filter and model is not None:
                    available_cols = [c for c in context_cols if c in feat.columns]
                    X = feat.loc[[idx], available_cols]
                    if not X.isna().any().any():
                        prob = model.predict_proba(X)[0, 1]
                        should_enter = prob >= threshold
                if should_enter:
                    position = {'entry': price, 'setup': setups.loc[idx, 'setup_type']}

    return pd.DataFrame(trades) if trades else None


def train_ml_filter(feat, setups, df, train_mask):
    context_cols = ['adx', 'di_diff', 'chop', 'atr_pct', 'vol_ratio',
                    'bb_width', 'ret_5', 'ret_20', 'hour_sin', 'hour_cos']
    train_setups = setups[train_mask & setups['has_setup']]

    setup_results = []
    for idx in train_setups.index:
        if idx not in df.index:
            continue
        entry_price = df.loc[idx, 'close']
        future = df.loc[idx:].head(30)
        result = None
        for future_idx in future.index[1:]:
            future_price = df.loc[future_idx, 'close']
            pnl = (future_price - entry_price) / entry_price
            if pnl >= TP_PCT:
                result = 1
                break
            elif pnl <= -SL_PCT:
                result = 0
                break
        if result is not None:
            setup_results.append({'idx': idx, 'result': result})

    if len(setup_results) < 30:
        return None

    results_df = pd.DataFrame(setup_results).set_index('idx')
    available_cols = [c for c in context_cols if c in feat.columns]
    X = feat.loc[results_df.index, available_cols].dropna()
    y = results_df.loc[X.index, 'result']

    model = GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)
    model.fit(X, y)

    train_acc = accuracy_score(y, model.predict(X))
    train_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    print(f"  Setups train: {len(y)} | Base WR: {y.mean()*100:.1f}% | Acc: {train_acc:.1%} | AUC: {train_auc:.3f}")

    return model


def main():
    print("=" * 70)
    print("DOT HYBRID SYSTEM - V13.05")
    print("=" * 70)

    df = load_data()
    print(f"\nTotal: {len(df):,} candles ({df.index.min().date()} a {df.index.max().date()})")

    feat = compute_features(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    setups = detect_setups(feat)
    n_setups = setups['has_setup'].sum()
    print(f"Setups detectados: {n_setups} ({n_setups/len(setups)*100:.1f}% de velas)")

    train_mask = df.index <= TRAIN_END
    test_mask = df.index >= TEST_START

    print(f"\nTrain: hasta {TRAIN_END} | Test: desde {TEST_START}")
    print(f"Setups train: {setups[train_mask & setups['has_setup']].shape[0]} | test: {setups[test_mask & setups['has_setup']].shape[0]}")

    print("\n--- SIN FILTRO ML ---")
    trades_no_filter = backtest_setups(df[test_mask], feat[test_mask], setups[test_mask])

    if trades_no_filter is not None and len(trades_no_filter) > 0:
        n = len(trades_no_filter)
        wins = (trades_no_filter['result'] == 'WIN').sum()
        total_pnl = trades_no_filter['pnl'].sum() * 100
        gp = trades_no_filter[trades_no_filter['pnl'] > 0]['pnl'].sum()
        gl = abs(trades_no_filter[trades_no_filter['pnl'] < 0]['pnl'].sum())
        pf = gp / gl if gl > 0 else 999
        print(f"Trades: {n} | WR: {wins/n*100:.1f}% | PnL: {total_pnl:+.1f}% | PF: {pf:.2f}")
    else:
        print("No hay trades")

    print("\n--- CON FILTRO ML ---")
    model = train_ml_filter(feat, setups, df, train_mask)

    if model:
        print(f"\n{'Threshold':>10} {'Trades':>8} {'WR':>8} {'PnL':>10} {'PF':>8}")
        print("-" * 50)

        results = []
        for th in [0.40, 0.45, 0.50, 0.55]:
            trades_filtered = backtest_setups(df[test_mask], feat[test_mask], setups[test_mask],
                                              use_filter=True, model=model, threshold=th)
            if trades_filtered is not None and len(trades_filtered) > 0:
                n = len(trades_filtered)
                wins = (trades_filtered['result'] == 'WIN').sum()
                total_pnl = trades_filtered['pnl'].sum() * 100
                gp = trades_filtered[trades_filtered['pnl'] > 0]['pnl'].sum()
                gl = abs(trades_filtered[trades_filtered['pnl'] < 0]['pnl'].sum())
                pf = gp / gl if gl > 0 else 999
                results.append({'th': th, 'n': n, 'wr': wins/n*100, 'pnl': total_pnl, 'pf': pf})
                print(f"{th:>10.2f} {n:>8} {wins/n*100:>7.1f}% {total_pnl:>9.1f}% {pf:>7.2f}")

        if results and trades_no_filter is not None:
            best = max(results, key=lambda x: x['pnl'])
            orig_pnl = trades_no_filter['pnl'].sum() * 100
            print(f"\nMejor threshold: {best['th']} | Mejora PnL: {best['pnl']-orig_pnl:+.1f}%")

            if best['pnl'] > 0 and best['wr'] >= 45:
                print("[APROBADO]")
            elif best['pnl'] > orig_pnl:
                print("[MARGINAL]")
            else:
                print("[RECHAZADO]")


if __name__ == '__main__':
    main()
