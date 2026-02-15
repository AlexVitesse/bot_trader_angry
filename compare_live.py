"""
Compare Live Strategies - V9 vs V8.5 Shadow
=============================================
Reads ml_trades from SQLite grouped by strategy column.
Shows head-to-head comparison table.

Usage:
    python compare_live.py [--days 14]
"""

import sys
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))
from config.settings import ML_DB_FILE, INITIAL_CAPITAL


def get_trades(db_path, strategy=None, since=None):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    query = "SELECT * FROM ml_trades WHERE 1=1"
    params = []
    if strategy:
        query += " AND strategy = ?"
        params.append(strategy)
    if since:
        query += " AND exit_time >= ?"
        params.append(since.isoformat())
    query += " ORDER BY exit_time ASC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def compute_metrics(trades, capital=INITIAL_CAPITAL):
    if not trades:
        return {
            'n_trades': 0, 'wins': 0, 'losses': 0, 'wr': 0,
            'total_pnl': 0, 'avg_pnl': 0, 'pf': 0,
            'max_dd': 0, 'best_trade': 0, 'worst_trade': 0,
            'avg_win': 0, 'avg_loss': 0, 'final_balance': capital,
            'return_pct': 0, 'days_positive': 0, 'days_total': 0,
        }

    n = len(trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    total_pnl = sum(t['pnl'] for t in trades)
    gross_profit = sum(t['pnl'] for t in wins)
    gross_loss = abs(sum(t['pnl'] for t in losses))

    # Max drawdown
    balance = capital
    peak = capital
    max_dd = 0
    for t in trades:
        balance += t['pnl']
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Daily PnL
    daily = {}
    for t in trades:
        try:
            day = t['exit_time'][:10]
        except Exception:
            continue
        daily[day] = daily.get(day, 0) + t['pnl']
    days_positive = sum(1 for v in daily.values() if v > 0)

    return {
        'n_trades': n,
        'wins': len(wins),
        'losses': len(losses),
        'wr': len(wins) / n * 100 if n else 0,
        'total_pnl': total_pnl,
        'avg_pnl': total_pnl / n if n else 0,
        'pf': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        'max_dd': max_dd * 100,
        'best_trade': max(t['pnl'] for t in trades),
        'worst_trade': min(t['pnl'] for t in trades),
        'avg_win': gross_profit / len(wins) if wins else 0,
        'avg_loss': -gross_loss / len(losses) if losses else 0,
        'final_balance': capital + total_pnl,
        'return_pct': total_pnl / capital * 100,
        'days_positive': days_positive,
        'days_total': len(daily),
    }


def print_comparison(v9_m, v85_m):
    w = 18  # column width

    def fmt(label, v9_val, v85_val, fmt_str='.2f', better='higher'):
        v9_s = f"{v9_val:{fmt_str}}"
        v85_s = f"{v85_val:{fmt_str}}"
        if better == 'higher':
            winner = 'V9' if v9_val > v85_val else ('V8.5' if v85_val > v9_val else 'TIE')
        else:
            winner = 'V9' if v9_val < v85_val else ('V8.5' if v85_val < v9_val else 'TIE')
        mark = '<--' if winner == 'V9' else ('  -->' if winner == 'V8.5' else '')
        return f"  {label:<20} {v9_s:>{w}} {v85_s:>{w}}  {mark}"

    print()
    print("=" * 70)
    print("  COMPARACION EN VIVO: V9 (LossDetector) vs V8.5 (Shadow)")
    print("=" * 70)
    print(f"  {'Metrica':<20} {'V9 (real)':>{w}} {'V8.5 (shadow)':>{w}}")
    print("  " + "-" * 60)
    print(fmt("Trades", v9_m['n_trades'], v85_m['n_trades'], 'd'))
    print(fmt("Wins", v9_m['wins'], v85_m['wins'], 'd'))
    print(fmt("Losses", v9_m['losses'], v85_m['losses'], 'd', 'lower'))
    print(fmt("Win Rate %", v9_m['wr'], v85_m['wr'], '.1f'))
    print(fmt("Total PnL $", v9_m['total_pnl'], v85_m['total_pnl'], '+.2f'))
    print(fmt("Avg PnL/trade $", v9_m['avg_pnl'], v85_m['avg_pnl'], '+.2f'))
    print(fmt("Profit Factor", v9_m['pf'], v85_m['pf'], '.2f'))
    print(fmt("Max DD %", v9_m['max_dd'], v85_m['max_dd'], '.1f', 'lower'))
    print(fmt("Best Trade $", v9_m['best_trade'], v85_m['best_trade'], '+.2f'))
    print(fmt("Worst Trade $", v9_m['worst_trade'], v85_m['worst_trade'], '+.2f', 'lower'))
    print(fmt("Avg Win $", v9_m['avg_win'], v85_m['avg_win'], '+.2f'))
    print(fmt("Avg Loss $", v9_m['avg_loss'], v85_m['avg_loss'], '+.2f', 'lower'))
    print(fmt("Return %", v9_m['return_pct'], v85_m['return_pct'], '+.1f'))
    print(fmt("Final Balance $", v9_m['final_balance'], v85_m['final_balance'], ',.2f'))
    print(fmt("Days Positive", v9_m['days_positive'], v85_m['days_positive'], 'd'))
    print(fmt("Days Total", v9_m['days_total'], v85_m['days_total'], 'd'))
    print("  " + "-" * 60)

    # Verdict
    v9_score = 0
    v85_score = 0
    for key, better in [('total_pnl', 'h'), ('wr', 'h'), ('pf', 'h'), ('max_dd', 'l')]:
        if better == 'h':
            if v9_m[key] > v85_m[key]: v9_score += 1
            elif v85_m[key] > v9_m[key]: v85_score += 1
        else:
            if v9_m[key] < v85_m[key]: v9_score += 1
            elif v85_m[key] < v9_m[key]: v85_score += 1

    if v9_score > v85_score:
        print(f"\n  >>> V9 WINS ({v9_score}-{v85_score}) <<<")
    elif v85_score > v9_score:
        print(f"\n  >>> V8.5 WINS ({v85_score}-{v9_score}) <<<")
    else:
        print(f"\n  >>> TIE ({v9_score}-{v85_score}) <<<")
    print()


def main():
    parser = argparse.ArgumentParser(description='Compare V9 vs V8.5 live strategies')
    parser.add_argument('--days', type=int, default=0, help='Only last N days (0=all)')
    args = parser.parse_args()

    if not ML_DB_FILE.exists():
        print(f"ERROR: Database not found: {ML_DB_FILE}")
        sys.exit(1)

    since = None
    if args.days > 0:
        since = datetime.now(timezone.utc) - timedelta(days=args.days)
        print(f"Filtering: last {args.days} days (since {since.strftime('%Y-%m-%d')})")

    v9_trades = get_trades(ML_DB_FILE, strategy='v9', since=since)
    v85_trades = get_trades(ML_DB_FILE, strategy='v85_shadow', since=since)

    print(f"\nLoaded: {len(v9_trades)} V9 trades, {len(v85_trades)} V8.5 shadow trades")

    if not v9_trades and not v85_trades:
        print("No trades found. Bot needs to run first.")
        sys.exit(0)

    v9_m = compute_metrics(v9_trades)
    v85_m = compute_metrics(v85_trades)

    print_comparison(v9_m, v85_m)

    # Exit reason breakdown
    for label, trades in [('V9', v9_trades), ('V8.5 Shadow', v85_trades)]:
        if not trades:
            continue
        reasons = {}
        for t in trades:
            r = t.get('exit_reason', 'UNKNOWN')
            if r not in reasons:
                reasons[r] = {'n': 0, 'pnl': 0}
            reasons[r]['n'] += 1
            reasons[r]['pnl'] += t['pnl']
        print(f"  {label} - Exit Reasons:")
        for r, v in sorted(reasons.items()):
            print(f"    {r:<10} {v['n']:>4} trades  ${v['pnl']:>+8.2f}")
        print()


if __name__ == '__main__':
    main()
