"""Descriptivo post-hoc (no decide): ¿oro_z aporta por el oro o por el z de BTC solo?"""
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'especialistas'))
from test_especialistas import construir, walk_forward, BASE

df = construir()
import pandas as pd
from test_especialistas import ROOT
v = pd.read_parquet(ROOT / 'data' / 'BTC_USDT_4h_full.parquet')['close'].astype(float)
v.index = v.index.tz_localize('UTC') if v.index.tz is None else v.index
x = np.log(v.resample('1D').last().dropna())
df['btc_z'] = ((x - x.rolling(60).mean()) / x.rolling(60).std()).reindex(df.index)
print('corr btc_z con oro_z:', round(df['btc_z'].corr(df['oro_z']), 3))
pb, _ = walk_forward(df, BASE)
y = df['y'].reindex(pb.index)
for f in (BASE + ['btc_z'], BASE + ['oro_z'], BASE + ['btc_z', 'oro_z']):
    p, c = walk_forward(df, f)
    print(f'{"+".join(f[3:]):<12} AUC {roc_auc_score(y, p):.3f} coef {np.round(c, 2)}')
