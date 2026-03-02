# V14 Validation Results

## Overview
V14 is a regime-based ensemble trading system for BTC that uses:
- **Regime Detection**: TREND_UP, TREND_DOWN, RANGE, VOLATILE
- **8 Setup Types**: Different strategies for different market conditions
- **Ensemble ML**: Context, momentum, volume models voting
- **Anti-Martingale Position Sizing**: Based on confidence + streak

## Cross-Validation Tests

### Test 1: BTC Rules on ETH (Real Data)
Testing if rules designed for BTC work on ETH without modification.

| Metric | Value |
|--------|-------|
| Period | 2020-01 to 2026-02 |
| Candles | 13,499 |
| Trades | 4,231 |
| Win Rate | 38.0% |
| **PnL** | **+2,829.6%** |

**By Direction:**
- LONG: 1,640 trades, WR 42.3%, PnL +1,538.9%
- SHORT: 2,591 trades, WR 35.3%, PnL +1,290.7%

### Test 2: Synthetic Data (5 Market Types)

| Market Type | Trades | Win Rate | PnL | Status |
|-------------|--------|----------|-----|--------|
| BULL (1 year) | 836 | 35.0% | +397.2% | OK |
| BEAR (1 year) | 535 | 47.5% | +731.5% | OK |
| RANGE (1 year) | 658 | 35.9% | +281.0% | OK |
| MIXED (1 year) | 774 | 38.6% | +514.6% | OK |
| VOLATILE (1 year) | 674 | 44.7% | +1,127.0% | OK |

## Setup Performance Analysis

### Strongest Setups (Positive in All Markets)
1. **SUPPORT_BOUNCE** - Works in all market conditions
2. **RESISTANCE_REJECTION** - Consistent performance
3. **RALLY_DOWNTREND** - Counter-trend works well

### Weak Setups (Needs Attention)
1. **EXHAUSTION** - Negative in BULL (-5.5%), RANGE (-2.7%)
2. **PULLBACK_UPTREND** - Negative in BEAR (-3.2%)

## Conclusions

### Evidence Against Overfitting
1. **Cross-Asset Generalization**: BTC rules work on ETH (+2,829.6%)
2. **Synthetic Data Performance**: 5/5 market types profitable
3. **Different Market Conditions**: Works in BULL, BEAR, RANGE, MIXED, VOLATILE

### Recommendation
- **EXHAUSTION** setup should be monitored or filtered more aggressively
- **PULLBACK_UPTREND** should not be used in BEAR regime
- System is ready for paper trading validation

## Next Steps
1. Paper trading on Binance testnet
2. If positive after 2-4 weeks, deploy with small capital
3. Create separate ETH expert (fine-tuned rules)

---
*Generated: 2026-02-28*
*Status: APPROVED FOR PAPER TRADING*
