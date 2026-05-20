"""
train.py -- Runner principal Agent K.

Genera:
1) V2 baseline reproducido (sanity check vs combined_AF: p=0.031 esperado)
2) V2 + on-chain filters (in-sample)
3) Cross-check vs features on-chain ALEATORIAS (5 seeds)
4) OOS 2026 (Ene-Abr)
5) Resultados JSON para el reporte final.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Path setup
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import strategy as S  # noqa


def variant_metrics_line(label, trades, weight=1.0):
    m = S.metrics(trades, weight=weight)
    p = S.bootstrap_p(trades, weight=weight) if len(trades) >= 3 else None
    p_s = f'{p:.3f}' if p is not None else '---'
    sig = '[OK]' if (p is not None and p < 0.05) else ('[~]' if (p is not None and p < 0.10) else '[X]')
    print(f"  {sig} {label:<58}  n={m['n']:>3}  WR={m['wr']:>5.1%}  "
          f"PF={m['pf']:>5.2f}  annual={m['annual']:>+6.1%}  DD={m['max_dd']:>5.1%}  p={p_s}")
    return m, p


def evaluate_variant(label, trades):
    m, p = variant_metrics_line(label, trades, weight=1.0)
    return {'label': label, 'n': m['n'], 'wr': m['wr'], 'pf': m['pf'],
            'annual': m['annual'], 'total': m['total'], 'dd': m['max_dd'],
            'bootstrap_p': p}


def make_random_onchain(real_onchain: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Construye un DataFrame on-chain ALEATORIO con las mismas estadisticas
    marginales (media/std) pero permutado en el TIEMPO. Mantiene la estructura
    del shift / reindex / cutoff, pero rompe cualquier relacion temporal.

    Usamos permutacion en BLOQUE para preservar autocorrelacion local
    (bloques de 30 dias) — esto es el placebo mas honesto:
    si nuestro filtro on-chain "funciona" con la version permutada, entonces
    NO esta capturando senal real, sino artefacto estadistico.
    """
    rng = np.random.default_rng(seed)
    out = real_onchain.copy()
    n = len(out)
    block = 30
    n_blocks = n // block
    idx_blocks = np.arange(n_blocks)
    rng.shuffle(idx_blocks)
    # construir indices permutados
    new_idx_pos = []
    for b in idx_blocks:
        new_idx_pos.extend(range(b * block, (b + 1) * block))
    # los sobrantes al final
    tail = list(range(n_blocks * block, n))
    new_idx_pos.extend(tail)
    # Aplicar a CADA columna (mantenemos correlaciones intra-bloque)
    df_shuffled = pd.DataFrame(index=out.index)
    for c in out.columns:
        df_shuffled[c] = out[c].values[new_idx_pos]
    return df_shuffled


def main():
    print('=' * 100)
    print('AGENT K — V2 (A + F_BTC) + ON-CHAIN FILTERS')
    print('=' * 100)
    print()
    print('Configuracion:')
    print(f'  shift on-chain: {S.PARAMS["onchain_shift_days"]} dias (anti-look-ahead)')
    print(f'  z-score lookback: {S.PARAMS["zscore_lookback_days"]} dias')
    print(f'  MVRV z >{S.PARAMS["mvrv_z_block_long"]} -> bloquea LONG')
    print(f'  Exch netflow z >{S.PARAMS["exch_netflow_z_block_long"]} -> bloquea LONG')
    print(f'  MVRV z <{S.PARAMS["mvrv_z_unblock_short"]} -> bloquea SHORT')
    print()

    results = {}

    # =====================================================
    # IN-SAMPLE 2020-2025 — V2 baseline reproducido
    # =====================================================
    print('-' * 100)
    print('IN-SAMPLE 2020-2025  (cutoff 2025-12-31)')
    print('-' * 100)
    is_start = pd.Timestamp('2020-01-01', tz='UTC')
    is_end = pd.Timestamp('2026-01-01', tz='UTC')

    is_result = S.backtest_v2_plus_onchain(is_start, is_end, S.PARAMS)
    v2_baseline = is_result['trades_v2_baseline']
    v2_filtered = is_result['trades_with_filter']
    vetoed = is_result['vetoed']

    print(f'\n[IS] Trades V2 baseline: {len(v2_baseline)}, vetoed by on-chain: {len(vetoed)}')
    print()

    r_v2_baseline_is = evaluate_variant('V2 baseline (A + F_BTC) in-sample', v2_baseline)
    r_v2_filter_is = evaluate_variant('V2 + on-chain FILTER in-sample', v2_filtered)

    # Desglose por lado (filtered)
    fA = [t for t in v2_filtered if t['strat'] == 'A']
    fF = [t for t in v2_filtered if t['strat'] == 'F']
    fF_long = [t for t in fF if 'LONG' in t['side']]
    fF_short = [t for t in fF if 'SHORT' in t['side']]
    print(f'\n  desglose filtered: A={len(fA)}, F_LONG={len(fF_long)}, F_SHORT={len(fF_short)}')

    # Detalle vetoed
    print(f'\nDetalle vetoes (in-sample):')
    veto_by_reason = {}
    for v in vetoed:
        r = v['veto_reason'].split('=')[0]
        veto_by_reason.setdefault(r, []).append(v['pnl_pct'])
    for r, pnls in veto_by_reason.items():
        avg = np.mean(pnls) * 100
        nwin = sum(1 for p in pnls if p > 0)
        print(f'  {r:30s}: {len(pnls):>3} trades, avg PnL={avg:+.2f}%, '
              f'WR={nwin/len(pnls):.0%} (lo que NO habria pasado)')

    results['in_sample'] = {
        'v2_baseline': r_v2_baseline_is,
        'v2_with_onchain_filter': r_v2_filter_is,
        'n_vetoed': len(vetoed),
        'vetoed_breakdown': {r: {'n': len(pnls),
                                  'avg_pnl': float(np.mean(pnls)),
                                  'wr': float(sum(1 for p in pnls if p > 0) / len(pnls))}
                              for r, pnls in veto_by_reason.items()},
    }

    # =====================================================
    # OOS 2026 (Ene-Abr) — no se usa para optimizar nada
    # =====================================================
    print()
    print('-' * 100)
    print('OOS 2026 (Ene-Abr) — datos NO usados al disenar / NO usados al elegir thresholds')
    print('-' * 100)
    oos_start = pd.Timestamp('2026-01-01', tz='UTC')
    oos_end = pd.Timestamp('2026-05-01', tz='UTC')
    oos_result = S.backtest_v2_plus_onchain(oos_start, oos_end, S.PARAMS)
    v2_baseline_oos = oos_result['trades_v2_baseline']
    v2_filter_oos = oos_result['trades_with_filter']
    vetoed_oos = oos_result['vetoed']

    print(f'\n[OOS] Trades V2 baseline: {len(v2_baseline_oos)}, vetoed: {len(vetoed_oos)}')

    r_v2_baseline_oos = evaluate_variant('V2 baseline OOS 2026', v2_baseline_oos)
    r_v2_filter_oos = evaluate_variant('V2 + on-chain OOS 2026', v2_filter_oos)

    print(f'\nTrades OOS 2026 (V2 baseline):')
    for t in v2_baseline_oos:
        ts = pd.to_datetime(t['ts']).strftime('%Y-%m-%d %H:%M')
        print(f'    {ts}  {t["side"]:<10s}  outcome={t["outcome"]:>7s}  pnl={t["pnl_pct"]:+.2%}  ({t["bars"]}b)')
    print(f'\nVetoed OOS 2026:')
    for v in vetoed_oos:
        ts = pd.to_datetime(v['ts']).strftime('%Y-%m-%d %H:%M')
        print(f'    {ts}  {v["side"]:<10s}  veto={v["veto_reason"]:30s}  '
              f'(habria sido pnl={v["pnl_pct"]:+.2%})')

    results['oos_2026'] = {
        'v2_baseline': r_v2_baseline_oos,
        'v2_with_onchain_filter': r_v2_filter_oos,
        'n_vetoed': len(vetoed_oos),
    }

    # =====================================================
    # CROSS-CHECK 1: features on-chain ALEATORIAS (block-shuffled)
    # =====================================================
    print()
    print('-' * 100)
    print('CROSS-CHECK 1: aplicar el mismo filtro pero con ON-CHAIN ALEATORIO (block-shuffle, 20 seeds)')
    print('Si la mejora es REAL, debe desaparecer con datos aleatorios.')
    print('-' * 100)

    df_oc_real = S.load_onchain()
    random_results = []
    for seed in range(1, 21):
        df_oc_random = make_random_onchain(df_oc_real, seed)
        r = S.backtest_v2_plus_onchain(is_start, is_end, S.PARAMS,
                                        df_onchain_override=df_oc_random)
        rb = r['trades_with_filter']
        m = S.metrics(rb, weight=1.0)
        p = S.bootstrap_p(rb, weight=1.0) if len(rb) >= 3 else None
        random_results.append({
            'seed': seed,
            'n': m['n'], 'wr': m['wr'], 'pf': m['pf'],
            'annual': m['annual'], 'dd': m['max_dd'], 'p': p,
            'n_vetoed': len(r['vetoed']),
        })

    for r in random_results:
        p_s = f'{r["p"]:.3f}' if r['p'] is not None else '---'
        sig = '[~]' if (r['p'] is not None and r['p'] < 0.10) else '[X]'
        print(f"  {sig} seed={r['seed']:>2}: n={r['n']:>3}, vetoed={r['n_vetoed']:>3}, "
              f"WR={r['wr']:.1%}, PF={r['pf']:.2f}, ann={r['annual']:+.1%}, p={p_s}")

    median_p_random = float(np.median([r['p'] for r in random_results if r['p'] is not None]))
    median_annual_random = float(np.median([r['annual'] for r in random_results if not np.isnan(r['annual'])]))
    pct_random_under_real = float(np.mean([(r['p'] is not None and r['p'] <= r_v2_filter_is['bootstrap_p']) for r in random_results]))
    results['cross_check_random'] = {
        'per_seed': random_results,
        'median_p': median_p_random,
        'median_annual': median_annual_random,
        'frac_random_p_le_real_p': pct_random_under_real,
    }

    print(f'\n  Real on-chain p: {r_v2_filter_is["bootstrap_p"]:.3f}')
    print(f'  Median p random: {median_p_random:.3f}')
    print(f'  Median annual random: {median_annual_random:+.1%}')
    print(f'  % random seeds with p <= real_p: {pct_random_under_real:.0%}')
    print(f'  (Si >=10% -> on-chain filter es indistinguible de random)')

    # =====================================================
    # CROSS-CHECK 2: RANDOM DROP de trades (mismo count que vetoed)
    # ==========================================================
    print()
    print('-' * 100)
    print('CROSS-CHECK 2: RANDOM DROP de N=vetoed_count trades del baseline (50 seeds)')
    print('Mide el efecto puro de SUBSAMPLING (mismo N final, sin filtro).')
    print('Si on-chain mejora p simplemente porque drop-eo trades, este test deberia')
    print('dar p similar al real.')
    print('-' * 100)

    n_vetoed = len(vetoed)
    random_drop_results = []
    base_pnls = np.array([t['pnl_pct'] for t in v2_baseline])
    for seed in range(1, 51):
        rng = np.random.default_rng(seed * 17)
        keep_idx = sorted(rng.choice(len(base_pnls), len(base_pnls) - n_vetoed, replace=False))
        kept_trades = [v2_baseline[k] for k in keep_idx]
        m = S.metrics(kept_trades)
        p = S.bootstrap_p(kept_trades)
        random_drop_results.append({'seed': seed, 'n': m['n'], 'wr': m['wr'], 'pf': m['pf'],
                                     'annual': m['annual'], 'p': p, 'dd': m['max_dd']})

    median_p_drop = float(np.median([r['p'] for r in random_drop_results if r['p'] is not None]))
    median_annual_drop = float(np.median([r['annual'] for r in random_drop_results if not np.isnan(r['annual'])]))
    pct_drop_under_real = float(np.mean([(r['p'] is not None and r['p'] <= r_v2_filter_is['bootstrap_p']) for r in random_drop_results]))
    results['cross_check_random_drop'] = {
        'n_dropped': n_vetoed,
        'median_p': median_p_drop,
        'median_annual': median_annual_drop,
        'frac_drop_p_le_real_p': pct_drop_under_real,
        'p_per_seed': [r['p'] for r in random_drop_results[:10]],
    }
    print(f'  Real on-chain p:      {r_v2_filter_is["bootstrap_p"]:.3f}')
    print(f'  Median p random-drop: {median_p_drop:.3f}')
    print(f'  Median annual random-drop: {median_annual_drop:+.1%}')
    print(f'  % random-drop seeds with p <= real_p: {pct_drop_under_real:.0%}')
    print(f'  (Si >=10% -> on-chain filter no es mejor que un drop aleatorio)')

    # =====================================================
    # COMPARACION FINAL
    # =====================================================
    print()
    print('=' * 100)
    print('COMPARACION FINAL')
    print('=' * 100)

    p_base = r_v2_baseline_is['bootstrap_p']
    p_oc = r_v2_filter_is['bootstrap_p']
    delta_p = p_base - p_oc

    print(f'\nIn-sample 2020-2025:')
    print(f'  V2 baseline:           p={p_base:.3f}, annual={r_v2_baseline_is["annual"]:+.1%}, '
          f'DD={r_v2_baseline_is["dd"]:.1%}, n={r_v2_baseline_is["n"]}')
    print(f'  V2 + on-chain filter:  p={p_oc:.3f}, annual={r_v2_filter_is["annual"]:+.1%}, '
          f'DD={r_v2_filter_is["dd"]:.1%}, n={r_v2_filter_is["n"]}')
    print(f'  Mejora delta p:        {delta_p:+.3f}  '
          f'({"MEJORA" if delta_p > 0 else "EMPEORA O IGUAL"})')

    print(f'\nCross-check vs random ON-CHAIN (block-shuffle):')
    print(f'  Real on-chain p:       {p_oc:.3f}')
    print(f'  Random median p:       {median_p_random:.3f}')
    print(f'  % random p <= real p:  {pct_random_under_real:.0%}')

    print(f'\nCross-check vs random DROP (subsampling):')
    print(f'  Real on-chain p:       {p_oc:.3f}')
    print(f'  Random-drop median p:  {median_p_drop:.3f}')
    print(f'  % random-drop p <= real p: {pct_drop_under_real:.0%}')

    # Cross-check overall verdict
    diff_oc = median_p_random - p_oc
    diff_drop = median_p_drop - p_oc
    cross_pass = (pct_random_under_real < 0.10) and (pct_drop_under_real < 0.10)
    if cross_pass:
        cross_verdict = ('REAL real mejor que ambos placebos (real p mejor que '
                         '>=90% de seeds random y >=90% de drops aleatorios)')
    elif pct_random_under_real >= 0.10:
        cross_verdict = f'on-chain INDISTINGUIBLE de random ({pct_random_under_real:.0%} de seeds aleatorios igualan o superan al real)'
    else:
        cross_verdict = f'on-chain INDISTINGUIBLE de drop aleatorio ({pct_drop_under_real:.0%} de drops aleatorios igualan o superan al real)'
    print(f'\n  Veredicto cross-check: {cross_verdict}')

    print(f'\nOOS 2026:')
    print(f'  V2 baseline:           p={r_v2_baseline_oos.get("bootstrap_p", "N/A")}, '
          f'n={r_v2_baseline_oos["n"]}, '
          f'total={r_v2_baseline_oos["total"]:+.1%}')
    print(f'  V2 + on-chain filter:  p={r_v2_filter_oos.get("bootstrap_p", "N/A")}, '
          f'n={r_v2_filter_oos["n"]}, '
          f'total={r_v2_filter_oos["total"]:+.1%}')

    # Verdict
    print()
    print('=' * 100)
    if p_oc < p_base and cross_pass:
        verdict = 'POSITIVO: on-chain mejora V2 y supera ambos placebos (random shuffle Y random drop)'
    elif p_oc < p_base and not cross_pass:
        verdict = 'NEGATIVO: on-chain mejora bootstrap p PERO el efecto es indistinguible de filtrado aleatorio (selection bias)'
    elif p_oc >= p_base:
        verdict = 'NEGATIVO: on-chain NO mejora V2 bootstrap p en absoluto'
    else:
        verdict = 'INCONCLUSIVO'
    print(f'VEREDICTO: {verdict}')
    print('=' * 100)

    results['verdict'] = verdict
    results['cross_check_verdict'] = cross_verdict

    # Save JSON
    out_path = HERE / 'results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nResultados guardados en: {out_path}')


if __name__ == '__main__':
    main()
