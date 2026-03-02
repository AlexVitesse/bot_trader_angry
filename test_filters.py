"""
Test filter combinations to improve win rate
Based on analysis findings:
- vol_ratio: winners avg 2.18, losers avg 1.68
- ret_3: negative momentum = higher failure
- bb_pct: winners slightly higher
"""

import json
from pathlib import Path

def load_trades():
    """Load all trades from analysis"""
    with open('analysis/all_trades.json', 'r') as f:
        return json.load(f)


def apply_filter(trades, filter_fn):
    """Apply filter and return stats"""
    filtered = [t for t in trades if filter_fn(t)]
    if not filtered:
        return None

    wins = len([t for t in filtered if t['trade']['exit_reason'] == 'TP'])
    total = len(filtered)
    total_pnl = sum(t['trade']['pnl_pct'] for t in filtered)

    return {
        'trades': total,
        'wins': wins,
        'wr': round(wins / total * 100, 1) if total > 0 else 0,
        'pnl': round(total_pnl, 1),
        'avg_pnl': round(total_pnl / total, 2) if total > 0 else 0
    }


def main():
    trades = load_trades()
    print(f"Total trades loaded: {len(trades)}")

    # Baseline stats
    baseline = apply_filter(trades, lambda t: True)
    print(f"\nBASELINE (no filter):")
    print(f"  Trades: {baseline['trades']}, WR: {baseline['wr']}%, PnL: {baseline['pnl']}%")

    print("\n" + "=" * 70)
    print("TESTING FILTER COMBINATIONS")
    print("=" * 70)

    # Define filters to test
    filters = [
        # vol_ratio filters
        ("vol_ratio > 1.5", lambda t: t['features']['vol_ratio'] > 1.5),
        ("vol_ratio > 1.7", lambda t: t['features']['vol_ratio'] > 1.7),
        ("vol_ratio > 2.0", lambda t: t['features']['vol_ratio'] > 2.0),

        # ret_3 filters (momentum)
        ("ret_3 > -0.02", lambda t: t['features']['ret_3'] > -0.02),
        ("ret_3 > -0.01", lambda t: t['features']['ret_3'] > -0.01),
        ("ret_3 > 0", lambda t: t['features']['ret_3'] > 0),

        # bb_pct filters
        ("bb_pct > 0.3", lambda t: t['features']['bb_pct'] > 0.3),
        ("bb_pct > 0.4", lambda t: t['features']['bb_pct'] > 0.4),
        ("bb_pct < 0.7", lambda t: t['features']['bb_pct'] < 0.7),

        # Combined filters
        ("vol_ratio > 1.5 AND ret_3 > -0.02",
         lambda t: t['features']['vol_ratio'] > 1.5 and t['features']['ret_3'] > -0.02),

        ("vol_ratio > 1.7 AND ret_3 > -0.01",
         lambda t: t['features']['vol_ratio'] > 1.7 and t['features']['ret_3'] > -0.01),

        ("vol_ratio > 1.5 AND ret_3 > 0",
         lambda t: t['features']['vol_ratio'] > 1.5 and t['features']['ret_3'] > 0),

        ("vol_ratio > 2.0 AND ret_3 > 0",
         lambda t: t['features']['vol_ratio'] > 2.0 and t['features']['ret_3'] > 0),

        # Triple combinations
        ("vol_ratio > 1.5 AND ret_3 > -0.02 AND bb_pct > 0.3",
         lambda t: t['features']['vol_ratio'] > 1.5 and t['features']['ret_3'] > -0.02 and t['features']['bb_pct'] > 0.3),

        ("vol_ratio > 1.7 AND ret_3 > 0 AND bb_pct > 0.3",
         lambda t: t['features']['vol_ratio'] > 1.7 and t['features']['ret_3'] > 0 and t['features']['bb_pct'] > 0.3),

        # RSI filters
        ("rsi > 0.4 AND rsi < 0.7",
         lambda t: t['features']['rsi'] > 0.4 and t['features']['rsi'] < 0.7),

        ("rsi > 0.45 AND rsi < 0.65",
         lambda t: t['features']['rsi'] > 0.45 and t['features']['rsi'] < 0.65),

        # Trend filter
        ("trend == 1 (bullish)", lambda t: t['features']['trend'] == 1.0),

        # Combined with trend
        ("trend == 1 AND vol_ratio > 1.5",
         lambda t: t['features']['trend'] == 1.0 and t['features']['vol_ratio'] > 1.5),

        ("trend == 1 AND ret_3 > 0",
         lambda t: t['features']['trend'] == 1.0 and t['features']['ret_3'] > 0),

        # Best combo candidates
        ("vol_ratio > 1.5 AND ret_3 > -0.01 AND trend == 1",
         lambda t: t['features']['vol_ratio'] > 1.5 and t['features']['ret_3'] > -0.01 and t['features']['trend'] == 1.0),

        ("vol_ratio > 1.7 AND ret_3 > -0.02 AND rsi > 0.4",
         lambda t: t['features']['vol_ratio'] > 1.7 and t['features']['ret_3'] > -0.02 and t['features']['rsi'] > 0.4),
    ]

    print(f"\n{'Filter':<55} {'Trades':<8} {'WR%':<8} {'PnL%':<10} {'AvgPnL':<8}")
    print("-" * 90)

    results = []
    for name, filter_fn in filters:
        stats = apply_filter(trades, filter_fn)
        if stats and stats['trades'] >= 20:  # Min 20 trades for significance
            wr_diff = stats['wr'] - baseline['wr']
            pnl_diff = stats['pnl'] - baseline['pnl']
            results.append({
                'name': name,
                'stats': stats,
                'wr_diff': wr_diff,
                'pnl_diff': pnl_diff
            })

            marker = ""
            if wr_diff >= 5 and pnl_diff >= 0:
                marker = " [GOOD]"
            elif wr_diff >= 10:
                marker = " [HIGH WR]"

            print(f"{name:<55} {stats['trades']:<8} {stats['wr']:<8} {stats['pnl']:<10} {stats['avg_pnl']:<8}{marker}")

    # Find best filters
    print("\n" + "=" * 70)
    print("BEST FILTERS (sorted by WR improvement)")
    print("=" * 70)

    results.sort(key=lambda x: x['wr_diff'], reverse=True)

    print(f"\n{'Rank':<6} {'Filter':<50} {'WR+':<8} {'PnL+':<10}")
    print("-" * 76)

    for i, r in enumerate(results[:10], 1):
        print(f"{i:<6} {r['name']:<50} {r['wr_diff']:+.1f}%   {r['pnl_diff']:+.1f}%")

    # Efficiency analysis
    print("\n" + "=" * 70)
    print("EFFICIENCY ANALYSIS (avg PnL per trade)")
    print("=" * 70)

    baseline_eff = baseline['pnl'] / baseline['trades']
    print(f"\nBaseline efficiency: {baseline_eff:.2f}% per trade")

    for r in results:
        r['efficiency'] = r['stats']['avg_pnl']
        r['eff_gain'] = (r['efficiency'] - baseline_eff) / baseline_eff * 100
        r['retention'] = r['stats']['trades'] / baseline['trades'] * 100

    # Sort by efficiency gain
    results.sort(key=lambda x: x['eff_gain'], reverse=True)

    print(f"\n{'Filter':<45} {'WR%':<8} {'Eff%':<8} {'Eff+':<8} {'Retain':<8}")
    print("-" * 80)

    for r in results[:15]:
        print(f"{r['name']:<45} {r['stats']['wr']:<8} {r['efficiency']:<8} {r['eff_gain']:+.1f}%   {r['retention']:.0f}%")

    # Best balanced: good efficiency gain with reasonable retention
    print("\n" + "=" * 70)
    print("BEST BALANCED (Eff gain > 10% AND Retention > 30%)")
    print("=" * 70)

    balanced = [r for r in results if r['eff_gain'] >= 10 and r['retention'] >= 30]
    balanced.sort(key=lambda x: (x['eff_gain'] + x['stats']['wr']), reverse=True)

    for r in balanced[:5]:
        print(f"\n  {r['name']}")
        print(f"    WR: {r['stats']['wr']}%, Efficiency: {r['efficiency']:.2f}% (+{r['eff_gain']:.1f}%)")
        print(f"    Trades: {r['stats']['trades']} ({r['retention']:.0f}% retained), Total PnL: {r['stats']['pnl']}%")

    # Final recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if balanced:
        best = balanced[0]
        print(f"\nBest filter: {best['name']}")
        print(f"  - WR: {baseline['wr']}% -> {best['stats']['wr']}% (+{best['wr_diff']:.1f}%)")
        print(f"  - Efficiency: {baseline_eff:.2f}% -> {best['efficiency']:.2f}% per trade (+{best['eff_gain']:.1f}%)")
        print(f"  - Trades: {baseline['trades']} -> {best['stats']['trades']} ({best['retention']:.0f}% retained)")
        print(f"  - Total PnL: {baseline['pnl']}% -> {best['stats']['pnl']}%")
    else:
        # Find best WR with > 35% retention
        high_wr = [r for r in results if r['retention'] >= 35]
        high_wr.sort(key=lambda x: x['stats']['wr'], reverse=True)
        if high_wr:
            best = high_wr[0]
            print(f"\nBest filter (prioritizing WR with 35%+ retention): {best['name']}")
            print(f"  - WR: {baseline['wr']}% -> {best['stats']['wr']}% (+{best['wr_diff']:.1f}%)")
            print(f"  - Efficiency: {baseline_eff:.2f}% -> {best['efficiency']:.2f}% per trade (+{best['eff_gain']:.1f}%)")
            print(f"  - Trades: {baseline['trades']} -> {best['stats']['trades']} ({best['retention']:.0f}% retained)")


if __name__ == '__main__':
    main()
