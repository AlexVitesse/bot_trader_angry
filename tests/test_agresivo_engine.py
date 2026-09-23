"""Motor agresivo: sin look-ahead, limites de riesgo y contrato de senal.

Uso: python -m pytest tests/test_agresivo_engine.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import agresivo_engine as ag


def _velas(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.01, n))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.01, n))
    idx = pd.date_range('2021-01-01', periods=n, freq='4h', tz='UTC')
    return pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c, 'volume': 1.0}, index=idx)


def test_features_sin_look_ahead():
    df = _velas()
    i = 1200
    f1 = ag.features(df, df)
    fut = df.copy()
    fut.iloc[i + 1:, :4] *= 3.0                        # cambia todo el futuro
    f2 = ag.features(fut, fut)
    cols = ['atr_pct', 'bb_width', 'dist_ema', 'ret20', 'ret60', 'btc_ema',
            'hi20', 'lo55', 'comprimido']
    pd.testing.assert_series_equal(f1[cols].iloc[i], f2[cols].iloc[i])
    assert ag.candidatos(f1, i) == ag.candidatos(f2, i)


def test_riesgo_respeta_limites():
    assert ag.riesgo_de(0.99, 5.0, 0.0) == ag.RIESGO_MAX
    assert ag.riesgo_de(0.30, 1.0, 0.0) == 0.0                 # Kelly negativo
    assert ag.riesgo_de(0.505, 1.0, 0.0) == 0.0                # 0,5% < 1% -> no opera
    assert ag.riesgo_de(0.99, 5.0, 0.25) == ag.RIESGO_MAX * 0.5
    assert ag.riesgo_de(0.99, 5.0, 0.60) == 0.0                 # DD >= 50% -> cero


def test_senal_vivo_contrato():
    df = _velas()
    feats = {'BTC/USDT': ag.features(df, df)}
    ev = ag.eventos_panel(feats)
    bundle = ag.entrenar(ev)
    bundle['b'] = 50.0                                   # fuerza que opere si hay candidato
    dfs = {'BTC/USDT': df}
    f = feats['BTC/USDT']
    # ultima vela con candidato para que haya senal
    i = max(j for j in range(300, len(f)) if ag.candidatos(f, j))
    sigs = ag.señal_vivo(bundle, {'BTC/USDT': df.iloc[:i + 1]}, df.iloc[:i + 1], set(), 0.0)
    assert sigs, 'esperaba una senal'
    s = sigs[0]
    assert isinstance(s['direction'], int) and s['direction'] in (1, -1)
    assert isinstance(s['price'], float) and s['price'] > 0
    assert s['trail_mode'] == 'tight' and ag.TRAIL_MIN <= s['trail_fixed_dist'] <= ag.TRAIL_MAX
    assert 0 < s['risk_pct'] <= ag.RIESGO_MAX and s['engine'] == 'agresivo'
    assert s['max_bars'] == ag.MAX_BARS
    assert ag.señal_vivo(bundle, dfs, df, {'BTC/USDT'}, 0.0) == []   # par abierto


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('OK', name)
