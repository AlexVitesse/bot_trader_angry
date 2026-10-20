"""La pata BTC en paper reproduce mixta() + bh() de experiments/benchmark_bh/."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'benchmark_bh'))
from src.pata_btc import W_BTC, paso  # noqa: E402


def test_igual_que_el_backtest():
    from test_benchmark import bh, mixta
    rng = np.random.default_rng(0)
    idx = pd.date_range('2026-01-01', periods=120, freq='D')
    px = pd.Series(60_000 * np.exp(np.cumsum(rng.normal(0, 0.03, len(idx)))), index=idx)
    v2 = pd.Series(1_000 * np.exp(np.cumsum(rng.normal(0, 0.02, len(idx)))), index=idx)

    ref = mixta(v2, bh(px, pd.Series(W_BTC, index=idx))) * v2.iloc[0]
    s = None
    for ts in idx:
        s = paso(s, v2[ts], px[ts], ts.strftime('%Y-%m-%d'))
    assert abs((s['a'] + s['b']) / ref.iloc[-1] - 1) < 1e-3


def test_mismo_dia_no_repite():
    s = paso(None, 1000, 50_000, '2026-10-19')
    assert paso(s, 1100, 60_000, '2026-10-19') is s
