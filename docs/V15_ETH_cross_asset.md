# V15 ETH Committee - Cross-Asset Validation

## Date: 2026-03-23

## Objective
Test if the ETH V15 committee logic generalizes to other L1 altcoins (SOL, ADA).

## Methodology
- Same committee logic: BTC-follower LONG + Standalone Breakout + SHORT multi-conf/BB
- Per-pair regime (EMAs diarias del par), BTC-follower (corr >= 0.5)
- Walk-forward: 12 semesters (2020-2025)
- SOL: no daily data available, resampled 4h -> 1d

## Results

| Pair | WF    | N   | WR    | PF   | $1K->  | DD    | Verdict   |
|------|-------|-----|-------|------|--------|-------|-----------|
| ETH  | 8/12  | 467 | 49.0% | 1.28 | $4,820 | 42.7% | APROBADO  |
| SOL  | 4/12  | 300 | 42.0% | 0.97 | $561   | 66.4% | RECHAZADO |
| ADA  | 6/12  | 373 | 37.8% | 0.89 | $284   | 80.9% | RECHAZADO |

## Analysis

### Why SOL/ADA fail:
- **FOLLOW_PB_BTC** (pullback follower) is the dominant setup (~60% of trades) and performs poorly on SOL (37% WR) and ADA (38% WR). ETH has higher correlation with BTC so following BTC pullbacks works; SOL/ADA diverge more.
- **MULTI_CONF SHORT** works for ETH (61% WR) but poorly for ADA (33% WR). SOL marginal (41%).
- SOL and ADA have more extreme volatility, so ATR-based TP/SL calibrated for ETH is suboptimal.

### ETH specifics that make it work:
- Highest BTC correlation among altcoins -> BTC follower is reliable
- Lower volatility than SOL/ADA -> ATR TP/SL thresholds fit better
- SHORT multi-conf (RSI>60+BB>0.75+bear candle) captures mean-reversion in BEAR regimes

## Conclusion
- **ETH committee is validated for ETH only** - does NOT generalize to other L1s
- SOL/ADA would need different parameter tuning or different strategy entirely
- Proceed with ETH deployment in paper trading

## Script
`validate_eth_v15_cross_asset.py`
