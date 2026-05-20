"""
Diagnostic: what happens to trade count and PF if we relax the vol filter?
This is for the README's "what did not work / why we report REJECT" section.
NOT used for the validation pipeline.
"""
import sys, copy, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import strategy as S

CUTOFF = pd.Timestamp('2025-12-31 23:59:59', tz='UTC')
df_raw = pd.read_parquet(ROOT / 'data' / 'ETH_USDT_4h_full.parquet')
if df_raw.index.tz is None:
    df_raw.index = df_raw.index.tz_localize('UTC')
df_raw = df_raw.sort_index()
df_raw = df_raw[df_raw.index <= CUTOFF]
df = S.prepare(df_raw)


def run_and_summarize(params, label):
    trades = S.run_engine(df, params, 0, len(df))
    if not trades:
        print(f'  {label:<40}  N=0')
        return
    pnls = np.array([t['pnl_pct'] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    wr = len(wins) / len(pnls)
    gw = wins.sum() if len(wins) else 0
    gl = abs(losses.sum()) if len(losses) else 1e-9
    pf = gw / gl if gl > 0 else float('inf')
    cum = np.prod(1 + pnls) - 1
    print(f'  {label:<40}  N={len(trades):>3}  WR={wr:>5.1%}  PF={pf:>5.2f}  '
          f'Total={cum:>+6.1%}')


print('Continuous backtest, varying single dials (in-sample):')
base = dict(S.PARAMS)
run_and_summarize(base, 'baseline (current PARAMS)')

p = dict(base); p['long_vol_ratio_min'] = 0.0; p['short_vol_ratio_min'] = 0.0
run_and_summarize(p, 'no vol confirmation')

p = dict(base); p['exit_on_regime_change'] = False
run_and_summarize(p, 'no regime-change exit')

p = dict(base); p['long_require_bullish_candle'] = False; p['short_require_bearish_candle'] = False
run_and_summarize(p, 'no candle confirmation')

p = dict(base); p['regime_adx_max'] = 30; p['regime_bb_width_pct_max'] = 50; p['regime_daily_ema_sep_max'] = 0.04
run_and_summarize(p, 'broader RANGE def (30/50/4%)')

p = dict(base)
p['long_vol_ratio_min'] = 0.0; p['short_vol_ratio_min'] = 0.0
p['long_require_bullish_candle'] = False; p['short_require_bearish_candle'] = False
run_and_summarize(p, 'no vol AND no candle')

# What about looking at just LONG (drop SHORT entirely)?
p = dict(base); p['short_enabled'] = False
run_and_summarize(p, 'LONG-only')

# AND-logic instead of OR-logic for the extreme
print('\nNot relaxations - just sanity:')
# raw count of trades if we don't require the bullish/vol/atr filters at all
# i.e. just RANGE + extreme bar
def signal_just_extreme(df, idx, p):
    reg = S.detect_regime(df, idx, p)
    if reg != 'RANGE': return None, reg
    row = df.iloc[idx]
    if row['rsi14'] <= p['long_rsi_max'] or row.get('bb_pct', 1) <= p['long_bb_pct_max']:
        return 'LONG', reg
    if row['rsi14'] >= p['short_rsi_min'] or row.get('bb_pct', 0) >= p['short_bb_pct_min']:
        return 'SHORT', reg
    return None, reg

# count raw extremes within RANGE (before any candle/vol/atr filter)
n_raw = 0
i = 260
while i < len(df) - 2:
    s, r = signal_just_extreme(df, i, base)
    if s:
        n_raw += 1
        # jump past max_bars to avoid clustering on same setup
        i += base['max_bars'] + 1
    else:
        i += 1
print(f'  Raw RANGE-extreme bars (no candle/vol/atr filter): N={n_raw}')
