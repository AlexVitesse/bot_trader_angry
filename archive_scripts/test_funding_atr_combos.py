"""
Test combinaciones de Funding Filter + ATR dinamico
Busca la configuracion con mayor retorno
"""
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path

MODELS_DIR = Path('models')
DATA_DIR = Path('data')
PAIRS = ['BTC/USDT','ETH/USDT','SOL/USDT','BNB/USDT','XRP/USDT','DOGE/USDT','ADA/USDT','AVAX/USDT','LINK/USDT','DOT/USDT','NEAR/USDT']

print('Cargando datos...')

# Load data
all_data = {}
for pair in PAIRS:
    fn = DATA_DIR / f'{pair.replace("/","")}_4h.pkl'
    if fn.exists():
        all_data[pair] = pickle.load(open(fn,'rb'))

# Load predictions
all_preds = {}
for pair in PAIRS:
    fn = MODELS_DIR / f'v7_{pair.replace("/","_")}_predictions.pkl'
    if fn.exists():
        all_preds[pair] = pickle.load(open(fn,'rb'))

# Load loss features
all_loss_features = {}
for pair in PAIRS:
    fn = MODELS_DIR / f'v95_{pair.replace("/","_")}_loss_features.pkl'
    if fn.exists():
        all_loss_features[pair] = pickle.load(open(fn,'rb'))

# Load detectors
v95_detectors = {}
v95_thresholds = {}
meta = json.load(open(MODELS_DIR / 'v95_meta.json'))
for pair in PAIRS:
    key = pair.replace('/','_')
    fn = MODELS_DIR / f'v95_loss_detector_{key}.pkl'
    if fn.exists():
        v95_detectors[pair] = pickle.load(open(fn,'rb'))
        v95_thresholds[pair] = meta['thresholds'].get(key, 0.5)

# Precompute chop, atr, funding
def compute_chop(df, period=14):
    high = df['high'].rolling(period).max()
    low = df['low'].rolling(period).min()
    atr_sum = (df['high'] - df['low']).rolling(period).sum()
    chop = 100 * np.log10(atr_sum / (high - low + 1e-10)) / np.log10(period)
    return chop

def compute_atr(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

print('Precomputando indicadores...')
all_chop = {}
all_atr = {}
all_funding = {}
for pair, df in all_data.items():
    chop = compute_chop(df)
    all_chop[pair] = {ts: v for ts, v in zip(df.index, chop) if pd.notna(v)}
    atr = compute_atr(df)
    all_atr[pair] = {ts: v for ts, v in zip(df.index, atr) if pd.notna(v)}
    # Simulated funding
    ret = df['close'].pct_change()
    funding = ret.rolling(20).mean() * 0.1
    all_funding[pair] = {ts: v for ts, v in zip(df.index, funding) if pd.notna(v)}

test_start = pd.Timestamp('2024-01-01', tz='UTC')

print('Probando combinaciones funding + ATR...\n')

def run_combo(conf_threshold, chop_filter, funding_filter, atr_tp, atr_sl):
    trades = []
    for pair in PAIRS:
        if pair not in all_preds or pair not in v95_detectors:
            continue
        preds = all_preds[pair]
        loss_feats = all_loss_features.get(pair, {})
        model = v95_detectors[pair]
        thresh = v95_thresholds[pair] + 0.15
        df = all_data[pair]

        for ts, pred in preds.items():
            if ts < test_start:
                continue
            direction = pred.get('direction', 0)
            conviction = pred.get('conviction', 0)
            if direction == 0 or conviction < conf_threshold:
                continue

            # Chop filter
            chop_val = all_chop[pair].get(ts)
            if chop_val is None or chop_val >= chop_filter:
                continue

            # Funding filter
            if funding_filter:
                fund_val = all_funding[pair].get(ts, 0)
                if direction == 1 and fund_val > funding_filter:
                    continue
                if direction == -1 and fund_val < -funding_filter:
                    continue

            # LossDetector
            feat_row = loss_feats.get(ts)
            if feat_row is None:
                continue
            try:
                fcols = [c for c in feat_row.index if c != 'is_loss']
                p_loss = float(model.predict_proba(feat_row[fcols].values.reshape(1,-1))[0][1])
                if p_loss > thresh:
                    continue
            except:
                continue

            # Simulate trade with ATR TP/SL
            try:
                idx = df.index.get_loc(ts)
            except:
                continue
            if idx + 1 >= len(df):
                continue

            entry = df.iloc[idx + 1]['open']
            atr_val = all_atr[pair].get(ts)
            if atr_val is None:
                continue

            atr_pct = atr_val / entry
            tp_pct = atr_pct * atr_tp
            sl_pct = atr_pct * atr_sl

            tp_price = entry * (1 + tp_pct) if direction == 1 else entry * (1 - tp_pct)
            sl_price = entry * (1 - sl_pct) if direction == 1 else entry * (1 + sl_pct)

            # Check next 20 candles
            pnl_pct = None
            for j in range(idx + 1, min(idx + 21, len(df))):
                candle = df.iloc[j]
                if direction == 1:
                    if candle['low'] <= sl_price:
                        pnl_pct = -sl_pct
                        break
                    if candle['high'] >= tp_price:
                        pnl_pct = tp_pct
                        break
                else:
                    if candle['high'] >= sl_price:
                        pnl_pct = -sl_pct
                        break
                    if candle['low'] <= tp_price:
                        pnl_pct = tp_pct
                        break

            if pnl_pct is None:
                exit_price = df.iloc[min(idx + 20, len(df)-1)]['close']
                pnl_pct = (exit_price - entry) / entry * direction

            trades.append({'pnl_pct': pnl_pct, 'win': pnl_pct > 0})

    if not trades:
        return None

    wins = sum(1 for t in trades if t['win'])
    wr = wins / len(trades) * 100
    pnl = sum(t['pnl_pct'] * 500 for t in trades)
    gross_profit = sum(t['pnl_pct'] * 500 for t in trades if t['pnl_pct'] > 0)
    gross_loss = abs(sum(t['pnl_pct'] * 500 for t in trades if t['pnl_pct'] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999
    ret = pnl / 500 * 100

    return {'trades': len(trades), 'wr': wr, 'pnl': pnl, 'pf': pf, 'return_pct': ret}

# Test combinations
combos = [
    # Funding + ATR variations
    (1.8, 50, 0.002, 2.5, 1.0, 'fund0.002+ATR2.5'),
    (1.8, 50, 0.0025, 2.5, 1.0, 'fund0.0025+ATR2.5'),
    (1.8, 50, 0.002, 2.0, 1.0, 'fund0.002+ATR2.0'),
    (1.8, 50, 0.0025, 2.0, 1.0, 'fund0.0025+ATR2.0'),
    (1.8, 50, 0.002, 3.0, 1.5, 'fund0.002+ATR3.0'),
    (1.8, 50, 0.0025, 3.0, 1.5, 'fund0.0025+ATR3.0'),
    (1.8, 50, 0.003, 2.5, 1.0, 'fund0.003+ATR2.5'),
    (1.8, 50, 0.0015, 2.5, 1.0, 'fund0.0015+ATR2.5'),
    # Looser chop with funding+ATR
    (1.8, 55, 0.002, 2.5, 1.0, 'chop55+fund+ATR2.5'),
    (1.8, 52, 0.0025, 2.5, 1.0, 'chop52+fund+ATR2.5'),
    # Tighter conf with funding+ATR
    (2.0, 50, 0.002, 2.5, 1.0, 'conf2.0+fund+ATR2.5'),
    (1.9, 50, 0.0025, 2.5, 1.0, 'conf1.9+fund+ATR2.5'),
    # More ATR variations
    (1.8, 50, 0.002, 2.0, 0.8, 'fund+ATR2.0/0.8'),
    (1.8, 50, 0.002, 2.5, 0.8, 'fund+ATR2.5/0.8'),
    (1.8, 50, 0.002, 3.0, 1.0, 'fund+ATR3.0/1.0'),
    # Baselines for comparison
    (1.8, 50, None, 2.5, 1.0, 'ATR2.5_only'),
    (1.8, 50, 0.0025, None, None, 'fund0.0025_only'),
]

results = []
print(f'{"Config":<25} {"Trades":>7} {"WR%":>7} {"PnL":>9} {"PF":>6} {"Ret%":>8}')
print('-' * 65)

for conf, chop, fund, atr_tp, atr_sl, name in combos:
    if atr_tp is None:
        # Use fixed TP/SL for funding only test
        r = run_combo(conf, chop, fund, 1.5, 1.0)  # Default 1.5% TP, 1% SL approx
    else:
        r = run_combo(conf, chop, fund, atr_tp, atr_sl)
    if r:
        r['name'] = name
        results.append(r)
        print(f'{name:<25} {r["trades"]:>7} {r["wr"]:>6.1f}% {r["pnl"]:>9.0f} {r["pf"]:>6.2f} {r["return_pct"]:>7.1f}%')

# Sort by return
results_sorted = sorted(results, key=lambda x: x['return_pct'], reverse=True)
print('\n' + '=' * 65)
print('TOP 5 POR RETORNO:')
for r in results_sorted[:5]:
    print(f'  {r["name"]:<25} Ret={r["return_pct"]:.1f}% | WR={r["wr"]:.1f}% | PnL=${r["pnl"]:.0f}')

print('\nTOP 5 POR WIN RATE (con PnL > $500):')
good = [r for r in results if r['pnl'] > 500]
good_wr = sorted(good, key=lambda x: x['wr'], reverse=True)
for r in good_wr[:5]:
    print(f'  {r["name"]:<25} WR={r["wr"]:.1f}% | Ret={r["return_pct"]:.1f}% | PnL=${r["pnl"]:.0f}')

# Save best
best = results_sorted[0]
print(f'\n{"="*65}')
print(f'*** MEJOR CONFIGURACION: {best["name"]} ***')
print(f'    Trades: {best["trades"]} | WR: {best["wr"]:.1f}% | PnL: ${best["pnl"]:.0f} | PF: {best["pf"]:.2f} | Ret: {best["return_pct"]:.1f}%')

# Save results
with open(MODELS_DIR / 'experiment_results_funding_atr.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResultados guardados en: {MODELS_DIR}/experiment_results_funding_atr.json')
