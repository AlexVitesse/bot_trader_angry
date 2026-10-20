"""La alerta en vivo calcula las mismas features que el experimento E7."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'especialistas'))
PARQUET = ROOT / 'data' / 'BTC_USDT_4h_full.parquet'
pytestmark = pytest.mark.skipif(not PARQUET.exists(), reason='sin datos locales')


def test_mismas_features_que_e7(tmp_path):
    from src import alerta_caidas as al
    from test_especialistas import construir, leer_macro

    ref = construir()
    v = pd.read_parquet(PARQUET).astype(float)
    v.index = v.index.tz_localize('UTC') if v.index.tz is None else v.index
    d = v.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min',
                              'close': 'last', 'volume': 'sum'}).dropna()
    t = al.tabla(d, {k: leer_macro(k) for k in al.TICKERS})
    comun = ref.index
    assert np.allclose(t.loc[comun, al.FEATS + ['y']].values, ref[al.FEATS + ['y']].values)

    r = al.evaluar(t)
    assert 0 < r['p'] < 1 and 0 < r['umbral'] < 1 and r['activa'] == (r['p'] >= r['umbral'])
    f = tmp_path / 'a.jsonl'
    al.registrar(f, r)
    al.registrar(f, r)                              # mismo dia: no duplica
    assert len(f.read_text().splitlines()) == 1 and al.ultimo(f)['dia'] == r['dia']
