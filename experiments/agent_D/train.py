"""
Agent D - Investigacion / Validacion in-sample (2020-01-01 -> 2025-12-31)
========================================================================

Que hace:
1. Carga datos diarios BTC + funding (cutoff 2025-12-31)
2. Construye features y simula con run_backtest (1 posicion a la vez)
3. Corre MULTIPLES configs documentadas:
   - 1D vol-targeted (primary)
   - 1D fixed 1x (unleveraged baseline)
   - 1D fixed 2x (aggressive futures - shows funding effect)
   - 1D 2x spot (no funding) - "if we used spot margin not perps"
   - 12h vol-targeted
4. Reporta metricas UNLEVERAGED y LEVERAGED separadas
5. Walk-forward 12 semestres con purga 14d (en config primary)
6. Bootstrap de significancia (>=3000 iter)
7. Stress test: DD durante March 2020 (COVID crash) y 2022 bear
8. Self-audit completo

NO mira datos posteriores a 2025-12-31. Verificacion 2026 OOS la hace
verify_2026.py externo.
"""
from __future__ import annotations
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from strategy import (PARAMS, PARAMS_12H, prepare_data, run_backtest, metrics,
                       daily_equity_curve, annualized_sharpe, signal,
                       size_position, simulate)


ROOT = Path("C:/Users/pcdec/OneDrive/Documentos/MIS EMPRENDIMIENTOS/BOTDETRADINGAGRESIVO")
DATA = ROOT / "data"
OUT = ROOT / "experiments" / "agent_D"


def load_data_1d():
    df_1d = pd.read_parquet(DATA / "btcusdt_1d_v15.parquet")
    df_funding = pd.read_parquet(DATA / "btc_v15_funding.parquet")
    return df_1d, df_funding


def load_data_12h():
    df_4h = pd.read_parquet(DATA / "BTC_USDT_4h_full.parquet")
    df_12h = df_4h.resample('12h').agg(
        {'open': 'first', 'high': 'max', 'low': 'min',
         'close': 'last', 'volume': 'sum'}).dropna()
    df_funding = pd.read_parquet(DATA / "btc_v15_funding.parquet")
    return df_12h, df_funding


def walk_forward(df: pd.DataFrame, params: dict,
                 fold_months: int = 6, purge_days: int = 14) -> list[dict]:
    folds = []
    starts = []
    y = 2020
    while True:
        s_jan = pd.Timestamp(f"{y}-01-01", tz='UTC')
        s_jul = pd.Timestamp(f"{y}-07-01", tz='UTC')
        if s_jan > df.index.max():
            break
        starts.append(s_jan)
        if s_jul > df.index.max():
            break
        starts.append(s_jul)
        y += 1

    for start in starts:
        end = start + pd.DateOffset(months=fold_months)
        eff_start = start + pd.Timedelta(days=purge_days)
        mask = (df.index >= eff_start) & (df.index < end)
        if mask.sum() < 30:
            continue
        first_i = df.index.get_indexer([df.index[mask][0]])[0]
        last_i = df.index.get_indexer([df.index[mask][-1]])[0] + 1
        trades = run_backtest(df, params, start_i=first_i, end_i=last_i,
                              use_leverage=True)
        m_lev = metrics(trades, pnl_key='leveraged_pnl_pct')
        m_unl = metrics(trades, pnl_key='net_pnl_pct')
        folds.append({
            'fold': start.strftime("%Y-%m"),
            'n': m_lev['n'],
            'wr': m_lev['wr'],
            'pf_lev': m_lev['pf'],
            'pf_unl': m_unl['pf'],
            'total_lev': m_lev['total_return'],
            'total_unl': m_unl['total_return'],
            'dd_lev': m_lev['max_dd'],
            'monthly_lev': m_lev['monthly_return'],
        })
    return folds


def bootstrap_pvalue(pnls: np.ndarray, n_iter: int = 3000,
                     seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(pnls)
    if n < 5:
        return {'p_value': 1.0, 'n_iter': 0, 'median': 0.0,
                'p5': 0.0, 'p95': 0.0}
    sims = np.empty(n_iter)
    for k in range(n_iter):
        sample = rng.choice(pnls, size=n, replace=True)
        eq = np.prod(1 + sample) - 1
        sims[k] = eq
    return {'p_value': float((sims <= 0).mean()),
            'n_iter': n_iter,
            'median': float(np.median(sims)),
            'p5': float(np.percentile(sims, 5)),
            'p95': float(np.percentile(sims, 95))}


def stress_window(df, trades, start_str, end_str, label,
                  pnl_key='leveraged_pnl_pct'):
    start = pd.Timestamp(start_str, tz='UTC')
    end = pd.Timestamp(end_str, tz='UTC')
    relevant = [t for t in trades
                if (pd.to_datetime(t['entry_ts']) <= end and
                    pd.to_datetime(t['exit_ts']) >= start)]
    if not relevant:
        return {'label': label, 'trades_active': 0, 'window_dd': 0.0,
                'pnl_in_window': 0.0,
                'note': f'no trades in {label} window (regime filter active)'}
    total = sum(t[pnl_key] for t in relevant)
    eq = 1.0
    peak = 1.0
    dd = 0.0
    for t in sorted(relevant, key=lambda x: x['exit_ts']):
        eq *= (1 + t[pnl_key])
        peak = max(peak, eq)
        dd = max(dd, (peak - eq) / max(peak, 1e-9))
    return {'label': label, 'trades_active': len(relevant),
            'pnl_in_window': float(total), 'window_dd': float(dd),
            'leverages': [round(t['leverage_used'], 2) for t in relevant]}


# =============================================================================
# RUN ONE CONFIG
# =============================================================================
def run_config(label: str, params: dict, df: pd.DataFrame,
               do_wf: bool = False, do_bootstrap: bool = False,
               do_stress: bool = False, do_audit: bool = False) -> dict:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    trades = run_backtest(df, params, use_leverage=True)
    if not trades:
        print("  NO TRADES")
        return {}

    m_lev = metrics(trades, pnl_key='leveraged_pnl_pct')
    m_unl = metrics(trades, pnl_key='net_pnl_pct')
    levs = np.array([t['leverage_used'] for t in trades])
    total_funding = sum(t['funding_cost_pct'] for t in trades)

    eq_lev = daily_equity_curve(trades, df, pnl_key='leveraged_pnl_pct')
    eq_unl = daily_equity_curve(trades, df, pnl_key='net_pnl_pct')
    sharpe_lev = annualized_sharpe(eq_lev)
    sharpe_unl = annualized_sharpe(eq_unl)

    print(f"  N trades:     {m_lev['n']}")
    print(f"  Years:        {m_lev['years']:.2f}")
    print(f"  Leverage:     min {levs.min():.2f}x median {np.median(levs):.2f}x "
          f"max {levs.max():.2f}x")
    print(f"  Funding tot:  {total_funding:+.2%}")
    print(f"\n  ---- UNLEVERAGED (1.0x, no funding effect on net) ----")
    print(f"  WR: {m_unl['wr']:.1%}  PF: {m_unl['pf']:.2f}  "
          f"CAGR: {m_unl['annual_return']:+.1%}  "
          f"DD: {m_unl['max_dd']:.1%}  Sharpe: {sharpe_unl:.2f}")
    print(f"\n  ---- LEVERAGED (as configured, with funding cost) ----")
    print(f"  WR: {m_lev['wr']:.1%}  PF: {m_lev['pf']:.2f}  "
          f"CAGR: {m_lev['annual_return']:+.1%}  "
          f"DD: {m_lev['max_dd']:.1%}  Sharpe: {sharpe_lev:.2f}")

    result = {
        'label': label,
        'n_trades': m_lev['n'],
        'years': m_lev['years'],
        'leverage_min': float(levs.min()),
        'leverage_median': float(np.median(levs)),
        'leverage_max': float(levs.max()),
        'leverage_mean': float(levs.mean()),
        'funding_total_cost': float(total_funding),
        'unleveraged': {
            'wr': m_unl['wr'], 'pf': m_unl['pf'],
            'total_return': m_unl['total_return'],
            'annual_return': m_unl['annual_return'],
            'monthly_return': m_unl['monthly_return'],
            'max_dd': m_unl['max_dd'],
            'sharpe_annual': sharpe_unl,
        },
        'leveraged': {
            'wr': m_lev['wr'], 'pf': m_lev['pf'],
            'total_return': m_lev['total_return'],
            'annual_return': m_lev['annual_return'],
            'monthly_return': m_lev['monthly_return'],
            'max_dd': m_lev['max_dd'],
            'sharpe_annual': sharpe_lev,
        },
    }

    if do_wf:
        print(f"\n  WALK-FORWARD (12 semestres, purga 14d):")
        folds = walk_forward(df, params, fold_months=6, purge_days=14)
        n_ok = 0
        n_eval = 0
        pf_list = []
        print(f"  {'Fold':<10} {'N':>4} {'WR':>6} {'PF':>6} {'Total':>8} {'DD':>7}")
        for f in folds:
            ok = (f['n'] >= 3 and f['pf_lev'] >= 1.2 and f['total_lev'] > 0)
            mark = "OK" if ok else ("--" if f['n'] == 0 else "FAIL")
            if f['n'] >= 3:
                n_eval += 1
                pf_list.append(f['pf_lev'])
                if ok:
                    n_ok += 1
            print(f"  {f['fold']:<10} {f['n']:>4} "
                  f"{f['wr']*100:>5.1f}% {f['pf_lev']:>6.2f} "
                  f"{f['total_lev']*100:>+7.1f}% {f['dd_lev']*100:>6.1f}%  {mark}")
        pf_median = float(np.median(pf_list)) if pf_list else 0.0
        print(f"  -> Evaluable (n>=3): {n_eval}/{len(folds)}  OK: {n_ok}/{n_eval}  "
              f"PF median: {pf_median:.2f}")
        result['walk_forward'] = {
            'n_folds': len(folds), 'n_evaluable': n_eval,
            'n_ok': n_ok, 'pf_median': pf_median,
            'folds': folds,
        }

    if do_bootstrap:
        pnls_lev = np.array([t['leveraged_pnl_pct'] for t in trades])
        pnls_unl = np.array([t['net_pnl_pct'] for t in trades])
        b_lev = bootstrap_pvalue(pnls_lev, 3000)
        b_unl = bootstrap_pvalue(pnls_unl, 3000)
        print(f"\n  BOOTSTRAP (3000 iter):")
        print(f"    UNL p-value: {b_unl['p_value']:.4f}  "
              f"median: {b_unl['median']:+.2%}  p5/p95: "
              f"{b_unl['p5']:+.2%}/{b_unl['p95']:+.2%}")
        print(f"    LEV p-value: {b_lev['p_value']:.4f}  "
              f"median: {b_lev['median']:+.2%}  p5/p95: "
              f"{b_lev['p5']:+.2%}/{b_lev['p95']:+.2%}")
        result['bootstrap'] = {'unleveraged': b_unl, 'leveraged': b_lev}

    if do_stress:
        print(f"\n  STRESS TESTS:")
        windows = [
            ('March 2020 COVID', '2020-02-15', '2020-04-15'),
            ('May 2021 crash', '2021-05-01', '2021-07-31'),
            ('2022 bear (full)', '2022-01-01', '2022-12-31'),
            ('LUNA collapse', '2022-05-01', '2022-05-31'),
            ('FTX collapse', '2022-11-01', '2022-12-15'),
        ]
        stresses = []
        for w_label, s, e in windows:
            sr = stress_window(df, trades, s, e, w_label)
            stresses.append(sr)
            print(f"    {sr['label']:25} n={sr['trades_active']:2d}  "
                  f"DD={sr['window_dd']*100:5.1f}%  "
                  f"PnL={sr['pnl_in_window']*100:+6.1f}%")
        result['stress_tests'] = stresses

    if do_audit:
        print(f"\n  SELF-AUDIT:")
        overlap = sum(1 for a, b in zip(trades, trades[1:])
                      if pd.to_datetime(b['entry_ts']) <= pd.to_datetime(a['exit_ts']))
        print(f"    overlap: {overlap}")
        assert overlap == 0
        gross = np.array([t['gross_pnl_pct'] for t in trades])
        print(f"    gross avg/std: {gross.mean():+.2%} / {gross.std():.2%}")
        print(f"    PF lev range: 0.5 < {m_lev['pf']:.2f} < 4 ? "
              f"{'OK' if 0.5 < m_lev['pf'] < 4 else 'WARN'}")
        print(f"    WR range:    20% < {m_lev['wr']:.1%} < 65% ? "
              f"{'OK' if 0.2 < m_lev['wr'] < 0.65 else 'WARN'}")
        print(f"    DD range:    5% < {m_lev['max_dd']:.1%} < 70% ? "
              f"{'OK' if 0.05 < m_lev['max_dd'] < 0.70 else 'WARN'}")
        cutoff = pd.Timestamp(params['cutoff_date'], tz='UTC')
        max_exit = max(pd.to_datetime(t['exit_ts']) for t in trades)
        print(f"    cutoff ok:   {max_exit} <= {cutoff} ? "
              f"{'OK' if max_exit <= cutoff + pd.Timedelta(days=1) else 'FAIL'}")
        assert max_exit <= cutoff + pd.Timedelta(days=1)

    result['trades'] = trades
    return result


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 72)
    print("AGENT D - Trend-Following + Vol-Targeting")
    print("Multi-config validation, cutoff 2025-12-31")
    print("=" * 72)

    df_1d_raw, df_funding = load_data_1d()
    df_12h_raw, _ = load_data_12h()

    df_1d = prepare_data(df_1d_raw, df_funding, PARAMS)
    df_12h = prepare_data(df_12h_raw, df_funding, PARAMS_12H)
    print(f"\n1D features: {len(df_1d)} bars ({df_1d.index.min().date()} -> "
          f"{df_1d.index.max().date()})")
    print(f"12h features: {len(df_12h)} bars ({df_12h.index.min().date()} -> "
          f"{df_12h.index.max().date()})")

    # Verificar cutoff
    cutoff = pd.Timestamp('2025-12-31', tz='UTC')
    assert df_1d.index.max() <= cutoff
    assert df_12h.index.max() <= cutoff
    print(f"Cutoff verificado: <= {cutoff.date()}")

    all_results = {}

    # ============================
    # CONFIG 1: 1D vol-targeted (PRIMARY)
    # ============================
    res = run_config(
        "CFG-1: 1D vol-targeted (target 2.1% daily, cap 2.5x, perp futures + funding)",
        PARAMS, df_1d,
        do_wf=True, do_bootstrap=True, do_stress=True, do_audit=True)
    all_results['1d_voltarget'] = {k: v for k, v in res.items() if k != 'trades'}
    primary_trades = res.get('trades', [])

    # ============================
    # CONFIG 2: 1D fixed 1x (baseline)
    # ============================
    p_1x = {**PARAMS, 'fixed_leverage': 1.0}
    res = run_config("CFG-2: 1D fixed 1.0x (baseline unleveraged perp + funding)",
                     p_1x, df_1d, do_audit=True)
    all_results['1d_fixed1x_perp'] = {k: v for k, v in res.items() if k != 'trades'}

    # ============================
    # CONFIG 3: 1D fixed 2x (aggressive futures)
    # ============================
    p_2x = {**PARAMS, 'fixed_leverage': 2.0}
    res = run_config("CFG-3: 1D fixed 2.0x perp (shows funding penalty)",
                     p_2x, df_1d, do_audit=True)
    all_results['1d_fixed2x_perp'] = {k: v for k, v in res.items() if k != 'trades'}

    # ============================
    # CONFIG 4: 1D fixed 2x SPOT (no funding) - "alternative execution venue"
    # ============================
    p_2x_spot = {**PARAMS, 'fixed_leverage': 2.0, 'funding_enabled': False}
    res = run_config("CFG-4: 1D fixed 2.0x SPOT (no funding - margin loan alternative)",
                     p_2x_spot, df_1d, do_audit=True)
    all_results['1d_fixed2x_spot'] = {k: v for k, v in res.items() if k != 'trades'}

    # ============================
    # CONFIG 5: 12h vol-targeted (more trades)
    # ============================
    res = run_config("CFG-5: 12h vol-targeted (perp + funding)",
                     PARAMS_12H, df_12h,
                     do_wf=True, do_bootstrap=True, do_stress=True, do_audit=True)
    all_results['12h_voltarget'] = {k: v for k, v in res.items() if k != 'trades'}

    # ============================
    # CONFIG 6: 12h fixed 2x SPOT (best case scenario)
    # ============================
    p_12h_spot = {**PARAMS_12H, 'fixed_leverage': 2.0, 'funding_enabled': False}
    res = run_config("CFG-6: 12h fixed 2.0x SPOT (no funding - max defensible scenario)",
                     p_12h_spot, df_12h, do_audit=True)
    all_results['12h_fixed2x_spot'] = {k: v for k, v in res.items() if k != 'trades'}

    # ============================
    # SAVE RESULTS
    # ============================
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        'agent': 'D',
        'cutoff_date': '2025-12-31',
        'strategy': 'TrendFollow_Donchian+EMA_Chandelier_VolTarget',
        'configs': all_results,
    }
    out_file = OUT / "results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n\nResults written: {out_file}")

    # Trades from primary config
    if primary_trades:
        df_trades = pd.DataFrame(primary_trades)
        df_trades.to_csv(OUT / "trades_primary.csv", index=False)

    # ============================
    # FINAL SUMMARY TABLE
    # ============================
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print(f"\n{'Config':<55} {'CAGR':>7} {'DD':>7} {'PF':>5} {'Sharpe':>7}")
    print("-" * 85)
    for cfg, r in all_results.items():
        lev = r['leveraged']
        print(f"{cfg:<55} {lev['annual_return']*100:>+6.1f}% "
              f"{lev['max_dd']*100:>6.1f}% "
              f"{lev['pf']:>5.2f} {lev['sharpe_annual']:>6.2f}")

    # Honesty check
    print("\n" + "=" * 72)
    print("HONESTY CHECK vs OBJECTIVE >30% annual NET")
    print("=" * 72)
    for cfg, r in all_results.items():
        cagr = r['leveraged']['annual_return']
        dd = r['leveraged']['max_dd']
        ok = (cagr > 0.30 and dd < 0.25)
        sub = (cagr > 0.20 and dd < 0.30)
        flag = "[MEETS]" if ok else ("[CLOSE]" if sub else "[BELOW]")
        print(f"  {flag} {cfg}: CAGR={cagr:+.1%} DD={dd:.1%}")


if __name__ == "__main__":
    main()
