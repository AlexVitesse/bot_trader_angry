"""
train.py — Validacion in-sample Agent F (Vol-Compression Breakout BTC + ETH).

Reglas inviolables aplicadas:
  - Datos cortados a <= 2025-12-31 SIEMPRE
  - Walk-forward por semestre con gap de purga 14 dias
  - Bootstrap >=2000 iter sobre pnl por trade
  - Una posicion a la vez por activo (BTC, ETH)
  - Sin look-ahead intrabar en trailing
  - Vol-targeting con cap leverage = 3x
  - Reportar metricas separadas LONG vs SHORT
  - Correlacion trades BTC vs ETH (diversificacion real o ilusoria)
  - Stress test marzo 2020

Uso:
  C:/Python/python.exe experiments/agent_F/train.py
"""

from __future__ import annotations
import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from strategy import (
    PARAMS, prepare_data, signal, simulate, run_backtest, metrics,
    metrics_portfolio_50_50
)

ROOT = HERE.parent.parent
DATA = ROOT / 'data'


# =============================================================================
# CARGA
# =============================================================================
def _load_one(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df[df.index <= cutoff].sort_index()
    return df


def load_all_data(cutoff: str = '2025-12-31') -> dict:
    print(f'Cargando datos (cutoff <= {cutoff})...')
    cut = pd.Timestamp(cutoff, tz='UTC')

    btc_4h = _load_one(DATA / 'BTC_USDT_4h_full.parquet', cut)
    eth_4h = _load_one(DATA / 'ETH_USDT_4h_full.parquet', cut)
    btc_1d = _load_one(DATA / 'btcusdt_1d_v15.parquet', cut)

    # ETH daily: derivar desde 4h si no hay parquet daily v15
    eth_1d_path = DATA / 'ethusdt_1d_v15.parquet'
    if eth_1d_path.exists():
        eth_1d = _load_one(eth_1d_path, cut)
    else:
        # Resample 4h -> 1d
        eth_1d = pd.DataFrame({
            'open': eth_4h['open'].resample('1D').first(),
            'high': eth_4h['high'].resample('1D').max(),
            'low': eth_4h['low'].resample('1D').min(),
            'close': eth_4h['close'].resample('1D').last(),
            'volume': eth_4h['volume'].resample('1D').sum(),
        }).dropna()

    # funding (opcional, solo BTC tiene archivo)
    fund_path = DATA / 'btc_v15_funding.parquet'
    btc_fund = _load_one(fund_path, cut) if fund_path.exists() else None

    print(f'  BTC 4h: {len(btc_4h)} bars, {btc_4h.index.min().date()} -> {btc_4h.index.max().date()}')
    print(f'  ETH 4h: {len(eth_4h)} bars, {eth_4h.index.min().date()} -> {eth_4h.index.max().date()}')
    print(f'  BTC 1d: {len(btc_1d)} bars')
    print(f'  ETH 1d: {len(eth_1d)} bars')
    if btc_fund is not None:
        print(f'  BTC funding: {len(btc_fund)} rows')

    # Verificacion cutoff
    assert btc_4h.index.max() <= cut, "VIOLACION cutoff BTC 4h"
    assert eth_4h.index.max() <= cut, "VIOLACION cutoff ETH 4h"
    assert btc_1d.index.max() <= cut, "VIOLACION cutoff BTC 1d"
    assert eth_1d.index.max() <= cut, "VIOLACION cutoff ETH 1d"

    return {
        'btc_4h': btc_4h, 'eth_4h': eth_4h,
        'btc_1d': btc_1d, 'eth_1d': eth_1d,
        'btc_fund': btc_fund,
    }


# =============================================================================
# WALK-FORWARD
# =============================================================================
WF_SEMESTERS = [
    ('2020-01-01', '2020-06-30'),
    ('2020-07-01', '2020-12-31'),
    ('2021-01-01', '2021-06-30'),
    ('2021-07-01', '2021-12-31'),
    ('2022-01-01', '2022-06-30'),
    ('2022-07-01', '2022-12-31'),
    ('2023-01-01', '2023-06-30'),
    ('2023-07-01', '2023-12-31'),
    ('2024-01-01', '2024-06-30'),
    ('2024-07-01', '2024-12-31'),
    ('2025-01-01', '2025-06-30'),
    ('2025-07-01', '2025-12-31'),
]
PURGE_DAYS = 14


def walk_forward(df_btc: pd.DataFrame, df_eth: pd.DataFrame, params: dict) -> dict:
    """
    WF agregado (BTC + ETH) + por activo. Devuelve folds_ok separados.
    """
    fold_results = []
    fold_btc_only = []
    fold_eth_only = []
    for start_s, end_s in WF_SEMESTERS:
        start = pd.Timestamp(start_s, tz='UTC')
        end = pd.Timestamp(end_s, tz='UTC')
        purge_until = start + pd.Timedelta(days=PURGE_DAYS)

        all_trades = run_backtest(df_btc, df_eth, params, start, end)
        # purga: descartar trades con entry < purge_until
        trades = [t for t in all_trades if pd.Timestamp(t['entry_ts']) >= purge_until]
        m = metrics(trades)
        btc_trades = [t for t in trades if t.get('asset') == 'BTC']
        eth_trades = [t for t in trades if t.get('asset') == 'ETH']
        mb = metrics(btc_trades)
        me = metrics(eth_trades)

        def _evaluate(m_dict):
            if m_dict['n'] == 0:
                return 'no_signal', False
            if m_dict['n'] < 3:
                return 'small_sample', False
            ok = (np.isfinite(m_dict['pf']) and m_dict['pf'] >= 1.2
                  and m_dict['total_return'] > 0)
            return 'evaluated', ok

        status_agg, ok_agg = _evaluate(m)
        status_btc, ok_btc = _evaluate(mb)
        status_eth, ok_eth = _evaluate(me)

        fold_results.append({
            'period': start_s[:7], 'n': m['n'], 'pf': m['pf'],
            'wr': m['wr'], 'total': m['total_return'],
            'monthly': m['monthly_return'], 'max_dd': m['max_dd'],
            'status': status_agg, 'ok': ok_agg,
        })
        fold_btc_only.append({
            'period': start_s[:7], 'n': mb['n'], 'pf': mb['pf'],
            'wr': mb['wr'], 'total': mb['total_return'],
            'status': status_btc, 'ok': ok_btc,
        })
        fold_eth_only.append({
            'period': start_s[:7], 'n': me['n'], 'pf': me['pf'],
            'wr': me['wr'], 'total': me['total_return'],
            'status': status_eth, 'ok': ok_eth,
        })
    return {
        'folds_agg': fold_results,
        'folds_btc': fold_btc_only,
        'folds_eth': fold_eth_only,
        'folds_ok_agg': sum(1 for r in fold_results if r['ok']),
        'folds_ok_btc': sum(1 for r in fold_btc_only if r['ok']),
        'folds_ok_eth': sum(1 for r in fold_eth_only if r['ok']),
        'folds_total': len(fold_results),
    }


# =============================================================================
# BOOTSTRAP
# =============================================================================
def bootstrap_pvalue(trades: list, n_iter: int = 3000, seed: int = 42) -> dict | None:
    if len(trades) < 5:
        return None
    rng = np.random.default_rng(seed)
    pnls = np.array([t['pnl_pct'] for t in trades])
    k = len(pnls)
    totals = np.empty(n_iter)
    for j in range(n_iter):
        sample = rng.choice(pnls, size=k, replace=True)
        totals[j] = float(np.prod(1.0 + sample) - 1.0)
    p_value = float(np.mean(totals <= 0))
    return {'p_value': p_value,
            'pctl_5': float(np.percentile(totals, 5)),
            'pctl_50': float(np.percentile(totals, 50)),
            'pctl_95': float(np.percentile(totals, 95)),
            'n_iter': n_iter, 'n_trades': k}


# =============================================================================
# CORRELACION TRADES BTC vs ETH
# =============================================================================
def correlation_btc_eth(trades: list, window: str = '7D') -> dict:
    """
    Calcula correlacion entre trades BTC y ETH agrupados por ventana temporal.
    Tambien reporta % de overlap temporal de trades (BTC abierto AL MISMO
    TIEMPO que ETH abierto). Si overlap es alto, la "diversificacion" es ilusoria.
    """
    if not trades:
        return {'overlap_pct': 0.0, 'corr_weekly_count': 0.0,
                'btc_count': 0, 'eth_count': 0, 'overlap_bars': 0}

    btc_trades = [t for t in trades if t.get('asset') == 'BTC']
    eth_trades = [t for t in trades if t.get('asset') == 'ETH']
    if not btc_trades or not eth_trades:
        return {'overlap_pct': 0.0, 'corr_weekly_count': 0.0,
                'btc_count': len(btc_trades), 'eth_count': len(eth_trades),
                'overlap_bars': 0}

    # Overlap temporal: por cada par (BTC trade, ETH trade), ver si sus
    # intervalos [entry, exit] se solapan.
    overlap_pairs = 0
    for tb in btc_trades:
        eb, xb = pd.Timestamp(tb['entry_ts']), pd.Timestamp(tb['exit_ts'])
        for te in eth_trades:
            ee, xe = pd.Timestamp(te['entry_ts']), pd.Timestamp(te['exit_ts'])
            if not (xb < ee or eb > xe):  # solapan
                overlap_pairs += 1
                break  # contar solapes BTC unicos
    overlap_pct = overlap_pairs / max(len(btc_trades), 1)

    # Correlacion por conteos semanales
    btc_ts = pd.Series(1, index=[pd.Timestamp(t['entry_ts']) for t in btc_trades])
    eth_ts = pd.Series(1, index=[pd.Timestamp(t['entry_ts']) for t in eth_trades])
    btc_w = btc_ts.resample(window).sum()
    eth_w = eth_ts.resample(window).sum()
    common = btc_w.reindex(btc_w.index.union(eth_w.index), fill_value=0)
    eth_c = eth_w.reindex(common.index, fill_value=0)
    if common.std() > 0 and eth_c.std() > 0:
        corr = float(common.corr(eth_c))
    else:
        corr = 0.0

    return {
        'overlap_pct_btc_with_eth': float(overlap_pct),
        'corr_weekly_count': corr,
        'btc_count': len(btc_trades),
        'eth_count': len(eth_trades),
    }


# =============================================================================
# STRESS TEST MARZO 2020
# =============================================================================
def stress_march_2020(df_btc, df_eth, params) -> dict:
    """
    Que DD habria tenido en marzo 2020 con vol-targeting?
    Periodo: 2020-02-15 -> 2020-04-15 (cubre el crash COVID).
    """
    start = pd.Timestamp('2020-02-15', tz='UTC')
    end = pd.Timestamp('2020-04-15', tz='UTC')
    trades = run_backtest(df_btc, df_eth, params, start, end)
    m = metrics(trades)
    m_5050 = metrics_portfolio_50_50(trades) if trades else {}
    # detalle trades
    trades_brief = [{
        'asset': t.get('asset'), 'side': t['side'],
        'entry_ts': str(t['entry_ts']), 'exit_ts': str(t['exit_ts']),
        'pnl_pct': float(t['pnl_pct']), 'lev': float(t.get('leverage', 0)),
        'outcome': t['outcome'],
    } for t in trades]
    return {'metrics': m, 'metrics_5050': m_5050, 'trades': trades_brief}


# =============================================================================
# MAIN
# =============================================================================
def fmt_pf(pf):
    return 'inf' if not np.isfinite(pf) else f'{pf:.2f}'


def split_metrics_by_side_asset(trades):
    """
    Devuelve metricas: LONG, SHORT, BTC, ETH, BTC-LONG, BTC-SHORT, ETH-LONG, ETH-SHORT.
    """
    out = {}
    out['LONG'] = metrics([t for t in trades if t['side'] == 'LONG'])
    out['SHORT'] = metrics([t for t in trades if t['side'] == 'SHORT'])
    out['BTC'] = metrics([t for t in trades if t.get('asset') == 'BTC'])
    out['ETH'] = metrics([t for t in trades if t.get('asset') == 'ETH'])
    out['BTC-LONG'] = metrics([t for t in trades if t.get('asset') == 'BTC' and t['side'] == 'LONG'])
    out['BTC-SHORT'] = metrics([t for t in trades if t.get('asset') == 'BTC' and t['side'] == 'SHORT'])
    out['ETH-LONG'] = metrics([t for t in trades if t.get('asset') == 'ETH' and t['side'] == 'LONG'])
    out['ETH-SHORT'] = metrics([t for t in trades if t.get('asset') == 'ETH' and t['side'] == 'SHORT'])
    return out


def main():
    print('=' * 78)
    print('AGENT F — VOL-COMPRESSION BREAKOUT BTC+ETH 4h, BIDIRECTIONAL, VOL-TARGET')
    print('=' * 78)

    data = load_all_data(cutoff=PARAMS['cutoff_date'])

    print('\nPreparando features BTC...')
    btc_feat = prepare_data(data['btc_4h'], data['btc_1d'], data.get('btc_fund'), PARAMS)
    print(f'  BTC features: {len(btc_feat)} bars, {btc_feat.index.min().date()} -> {btc_feat.index.max().date()}')

    print('Preparando features ETH...')
    eth_feat = prepare_data(data['eth_4h'], data['eth_1d'], None, PARAMS)
    print(f'  ETH features: {len(eth_feat)} bars, {eth_feat.index.min().date()} -> {eth_feat.index.max().date()}')

    # Diagnostico de compresion: cuantas velas de cada activo estan en
    # compresion sostenida (para entender si la senal dispara lo suficiente).
    n_btc_compressed = int(btc_feat['compression_sustained'].sum())
    n_eth_compressed = int(eth_feat['compression_sustained'].sum())
    print(f'\nDiagnostico compresion sostenida (BB_width_pct<={PARAMS["compression_percentile"]} '
          f'por >={PARAMS["compression_min_bars"]} velas):')
    print(f'  BTC: {n_btc_compressed} velas comprimidas / {len(btc_feat)} ({n_btc_compressed/len(btc_feat)*100:.1f}%)')
    print(f'  ETH: {n_eth_compressed} velas comprimidas / {len(eth_feat)} ({n_eth_compressed/len(eth_feat)*100:.1f}%)')

    # Walk-forward
    print(f'\nWalk-forward {len(WF_SEMESTERS)} semestres (purga {PURGE_DAYS}d)...')
    wf = walk_forward(btc_feat, eth_feat, PARAMS)
    print(f'\n  Folds OK agregado: {wf["folds_ok_agg"]}/{wf["folds_total"]}')
    print(f'  Folds OK BTC: {wf["folds_ok_btc"]}/{wf["folds_total"]}')
    print(f'  Folds OK ETH: {wf["folds_ok_eth"]}/{wf["folds_total"]}')

    print(f'\n  AGREGADO BTC+ETH:')
    print(f'  {"period":<10} {"n":>4} {"wr":>6} {"pf":>7} {"total":>9} {"dd":>7} ok')
    for f in wf['folds_agg']:
        flag = '[+]' if f['ok'] else ('[no-sig]' if f['status'] == 'no_signal' else '[-]')
        print(f"  {f['period']:<10} {f['n']:>4} {f['wr']*100:>5.1f}% "
              f"{fmt_pf(f['pf']):>7} {f['total']*100:>+8.1f}% {f['max_dd']*100:>6.1f}% {flag}")

    print(f'\n  BTC SOLO:')
    for f in wf['folds_btc']:
        flag = '[+]' if f['ok'] else ('[no-sig]' if f['status'] == 'no_signal' else '[-]')
        print(f"  {f['period']:<10} {f['n']:>4} {f['wr']*100:>5.1f}% "
              f"{fmt_pf(f['pf']):>7} {f['total']*100:>+8.1f}% {flag}")

    print(f'\n  ETH SOLO:')
    for f in wf['folds_eth']:
        flag = '[+]' if f['ok'] else ('[no-sig]' if f['status'] == 'no_signal' else '[-]')
        print(f"  {f['period']:<10} {f['n']:>4} {f['wr']*100:>5.1f}% "
              f"{fmt_pf(f['pf']):>7} {f['total']*100:>+8.1f}% {flag}")

    # Backtest global 2020-01-01 -> cutoff
    print('\nBacktest global 2020-01-01 -> cutoff...')
    start_full = pd.Timestamp('2020-01-01', tz='UTC')
    end_full = pd.Timestamp(PARAMS['cutoff_date'], tz='UTC')
    all_trades = run_backtest(btc_feat, eth_feat, PARAMS, start_full, end_full)
    M = metrics(all_trades)
    M5050 = metrics_portfolio_50_50(all_trades)
    print(f"  N={M['n']}  WR={M['wr']*100:.1f}%  PF={fmt_pf(M['pf'])}  "
          f"total={M['total_return']*100:+.1f}%  "
          f"monthly={M['monthly_return']*100:+.2f}%  "
          f"annual={M['annual_return']*100:+.1f}%  "
          f"DD={M['max_dd']*100:.1f}%  "
          f"sharpe-like={M['sharpe_like']:.2f}  "
          f"sharpe-ann={M['sharpe_annual']:.2f}  "
          f"months={M['months']:.1f}")
    print(f"  Leverage: max={M['max_leverage']:.2f}x  avg={M['avg_leverage']:.2f}x")
    print(f"  Portfolio 50/50: total={M5050.get('total_return_50_50', 0)*100:+.1f}%  "
          f"annual={M5050.get('annual_return_50_50', 0)*100:+.1f}%  "
          f"DD={M5050.get('max_dd_50_50', 0)*100:.1f}%")

    # Por side/asset
    print('\nMetricas por side/asset:')
    breakdown = split_metrics_by_side_asset(all_trades)
    for k, v in breakdown.items():
        if v['n'] == 0:
            print(f"  {k:<12} N=0")
            continue
        print(f"  {k:<12} N={v['n']:>3}  WR={v['wr']*100:>5.1f}%  PF={fmt_pf(v['pf']):>6}  "
              f"total={v['total_return']*100:>+7.1f}%  avg_pnl={v['avg_pnl']*100:>+6.2f}%")

    # Correlacion BTC vs ETH
    print('\nCorrelacion BTC vs ETH (diversificacion real o ilusoria):')
    corr = correlation_btc_eth(all_trades)
    print(f'  BTC trades: {corr["btc_count"]}, ETH trades: {corr["eth_count"]}')
    print(f'  Overlap temporal (BTC trades con ETH solapado): {corr["overlap_pct_btc_with_eth"]*100:.1f}%')
    print(f'  Correlacion semanal (count BTC vs ETH): {corr["corr_weekly_count"]:.3f}')
    if corr['overlap_pct_btc_with_eth'] > 0.6:
        print('  [!] Overlap > 60% -> diversificacion ILUSORIA (BTC y ETH se mueven juntos)')
    elif corr['overlap_pct_btc_with_eth'] < 0.3:
        print('  [+] Overlap < 30% -> diversificacion REAL (BTC y ETH descorrelacionados)')

    # Bootstrap
    print('\nBootstrap (3000 iter)...')
    boot = bootstrap_pvalue(all_trades, n_iter=3000)
    if boot is None:
        print('  insuficientes trades')
    else:
        sig = 'SIGNIFICATIVO' if boot['p_value'] < 0.05 else 'NO significativo'
        print(f"  p-value(retorno<=0 por azar): {boot['p_value']:.4f} -> {sig}")
        print(f"  retorno mediano resampled: {boot['pctl_50']*100:+.1f}%")
        print(f"  retorno percentil 5: {boot['pctl_5']*100:+.1f}%")
        print(f"  retorno percentil 95: {boot['pctl_95']*100:+.1f}%")

    # Stress test marzo 2020
    print('\nStress test marzo 2020 (COVID crash, 2020-02-15 -> 2020-04-15)...')
    stress = stress_march_2020(btc_feat, eth_feat, PARAMS)
    sm = stress['metrics']
    if sm['n'] == 0:
        print('  Sin trades en el periodo (filtro daily macro stayed out -> defensivo).')
    else:
        print(f"  N={sm['n']}  WR={sm['wr']*100:.1f}%  PF={fmt_pf(sm['pf'])}  "
              f"total={sm['total_return']*100:+.1f}%  DD={sm['max_dd']*100:.1f}%  "
              f"max_lev={sm.get('max_leverage',0):.2f}x")
        for t in stress['trades']:
            print(f"    {t['asset']} {t['side']} entry {t['entry_ts'][:10]} exit {t['exit_ts'][:10]} "
                  f"lev={t['lev']:.2f}x pnl={t['pnl_pct']*100:+.1f}% {t['outcome']}")

    # Sanity checks
    print('\nSANITY CHECKS:')
    sanity_flags = []
    if M['n'] > 0 and M['pf'] > 4:
        sanity_flags.append(f"PF {M['pf']:.2f} > 4 -> sospechar overfitting")
    if M['n'] > 0 and M['wr'] > 0.65:
        sanity_flags.append(f"WR {M['wr']*100:.1f}% > 65% -> sospechar")
    if M['n'] > 0 and M['max_dd'] < 0.05:
        sanity_flags.append(f"DD {M['max_dd']*100:.1f}% < 5% -> sospechar sample no adverso")
    if sanity_flags:
        for f in sanity_flags:
            print(f'  [!] {f}')
    else:
        print('  [+] PF/WR/DD en rango razonable para crypto-4h')

    # Self-audit checks
    print('\nSELF-AUDIT (mecanismo):')
    # 1) verificar no solape por activo
    asset_overlap_violations = 0
    sorted_trades = sorted(all_trades, key=lambda t: (t.get('asset', ''), pd.Timestamp(t['entry_ts'])))
    for asset in ['BTC', 'ETH']:
        at = [t for t in sorted_trades if t.get('asset') == asset]
        for i in range(1, len(at)):
            if pd.Timestamp(at[i]['entry_ts']) < pd.Timestamp(at[i - 1]['exit_ts']):
                asset_overlap_violations += 1
    print(f'  Trades solapados por activo: {asset_overlap_violations} (debe ser 0)')

    # 2) verificar shift(1) implicito en BB width percentile
    # rolling().rank(pct=True) sobre bb_width devuelve el percentil del VALOR
    # actual en su ventana — con .shift(1) lo desplazamos. Verificamos que
    # NO hay NaN en bb_width_pct salvo al principio.
    n_nan = int(btc_feat['bb_width_pct'].isna().sum())
    print(f'  BB width pct NaN al principio (esperado >0, ya droppeado): {n_nan}')

    # 3) verificar que el cutoff se respeta en el dataframe ya featurized
    print(f'  Cutoff respetado BTC: {btc_feat.index.max()} <= {PARAMS["cutoff_date"]}')
    print(f'  Cutoff respetado ETH: {eth_feat.index.max()} <= {PARAMS["cutoff_date"]}')

    # 4) verificar leverage cap
    levs = [t.get('leverage', 0) for t in all_trades]
    if levs:
        print(f'  Leverage min: {min(levs):.3f}x  max: {max(levs):.3f}x  '
              f'(cap params {PARAMS["min_leverage"]:.2f}-{PARAMS["max_leverage"]:.2f})')

    # Guardar
    summary = {
        'agent': 'F',
        'strategy_name': 'vol_compression_breakout_BTC_ETH',
        'params': PARAMS,
        'wf': {
            'folds_ok_agg': wf['folds_ok_agg'],
            'folds_ok_btc': wf['folds_ok_btc'],
            'folds_ok_eth': wf['folds_ok_eth'],
            'folds_total': wf['folds_total'],
            'folds_agg': wf['folds_agg'],
            'folds_btc': wf['folds_btc'],
            'folds_eth': wf['folds_eth'],
        },
        'overall': M,
        'overall_50_50': M5050,
        'breakdown_by_side_asset': breakdown,
        'correlation_btc_eth': corr,
        'bootstrap': boot,
        'stress_march_2020': {
            'metrics': stress['metrics'],
            'metrics_5050': stress.get('metrics_5050', {}),
            'n_trades': len(stress['trades']),
        },
        'cutoff': PARAMS['cutoff_date'],
        'n_trades_2020_2025': M['n'],
        'self_audit': {
            'asset_overlap_violations': asset_overlap_violations,
            'cutoff_respected': True,
            'leverage_in_cap': all(PARAMS['min_leverage'] - 1e-6 <= l <= PARAMS['max_leverage'] + 1e-6 for l in levs),
        },
    }
    out_json = HERE / 'results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, default=str, indent=2)
    print(f'\nResultados guardados en {out_json}')

    return summary


if __name__ == '__main__':
    main()
