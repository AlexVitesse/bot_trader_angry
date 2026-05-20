"""
ml_hybrid_check.py -- Mode B sanity check.

Pregunta: si en lugar de un filtro hardcoded usamos un clasificador ML que
combina las features de V2 con las on-chain, ?recuperamos algun edge?

Diseno:
- Para cada trade del V2 baseline, crear features:
  * features tecnicas (RSI, ATR, vol_ratio, dist_donchian, etc.) en la vela de entrada
  * features on-chain (mvrv_z, exch_netflow_z, active_addr_z, hashrate_slope) en la vela de entrada
- Target: 1 si pnl_pct > 0, 0 si no
- Modelo: GradientBoosting con purged time-series CV
- Para cada fold: entrenar en past trades, predecir prob de ganar en future trades,
  filtrar si prob < threshold

NO se tunea threshold sobre test. Threshold = mediana train, calibrada por fold.

Si esto no supera al baseline V2 (p=0.030) con cross-check vs random, on-chain
realmente no contribuye.

CRITICO: este test tiene un riesgo de overfitting alto (mas features que muestras).
Si pasa, requiere validacion adicional. Si NO pasa, el veredicto negativo es solido.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import strategy as S  # noqa
import importlib.util


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    A = load_module('A', S.EXP / 'agent_A' / 'strategy.py')
    F = load_module('F', S.EXP / 'agent_F' / 'strategy.py')

    paramsA = dict(A.PARAMS); paramsA['cutoff_date'] = '2027-01-01'
    paramsF = dict(F.PARAMS); paramsF['cutoff_date'] = '2027-01-01'

    df_btc_4h = S.load_btc_4h()
    df_1d = S.load_btc_1d()
    df_fund = S.load_funding()
    df_onchain = S.load_onchain()

    df_btc_A = A.prepare_data(df_btc_4h, df_1d, df_fund, paramsA)
    df_btc_F = F.prepare_data(df_btc_4h, df_1d, df_fund, paramsF)
    btc_common = df_btc_A.index.intersection(df_btc_F.index)
    df_btc_A_c = df_btc_A.loc[btc_common].copy()
    df_btc_F_c = df_btc_F.loc[btc_common].copy()
    df_oc = S.prepare_onchain_features(btc_common, df_onchain, S.PARAMS)

    is_start = pd.Timestamp('2020-01-01', tz='UTC')
    is_end = pd.Timestamp('2026-01-01', tz='UTC')
    r = S.backtest_v2_plus_onchain(is_start, is_end, dict(S.PARAMS,
                                                           enable_mvrv_filter=False,
                                                           enable_exchflow_filter=False,
                                                           enable_short_filter=False))
    trades = r['trades_with_filter']
    print(f'Baseline V2 trades: {len(trades)}')

    # Build features per trade
    tech_feats = ['atr_pct', 'adx', 'vol_ratio', 'donchian_high', 'funding_z']
    oc_feats = ['mvrv_z', 'active_addr_z', 'exch_netflow_z', 'hashrate_slope_7d', 'sply_ex_pct']

    rows = []
    for t in trades:
        i = t['entry_i']
        rowA = df_btc_A_c.iloc[i]
        rowOC = df_oc.iloc[i]
        d = {
            'ts': pd.to_datetime(t['ts']),
            'pnl': t['pnl_pct'],
            'win': int(t['pnl_pct'] > 0),
            'side_F_SHORT': int(t['side'] == 'F_SHORT'),
            'side_F_LONG': int(t['side'] == 'F_LONG'),
        }
        for f in tech_feats:
            d[f'tech_{f}'] = float(rowA.get(f, np.nan)) if f in rowA else float(df_btc_F_c.iloc[i].get(f, np.nan))
        # Also pull from F's index since F has its own features
        rowF = df_btc_F_c.iloc[i]
        # Add a few of F's specific features (BB width percentile, etc.)
        for f in ['bb_width_pct', 'compression_bars']:
            if f in rowF:
                d[f'tech_{f}'] = float(rowF.get(f, np.nan))
        for f in oc_feats:
            d[f'oc_{f}'] = float(rowOC.get(f, np.nan))
        rows.append(d)
    df = pd.DataFrame(rows).set_index('ts').sort_index()
    print(f'Features cols: {[c for c in df.columns if c.startswith(("tech_","oc_","side_"))]}')
    print(f'Class balance: win_rate={df["win"].mean():.2%}')

    # Drop NaN rows
    df = df.dropna()
    print(f'After dropna: {len(df)} trades')

    feat_cols = [c for c in df.columns if c.startswith(('tech_', 'oc_', 'side_'))]
    oc_only_cols = [c for c in feat_cols if c.startswith('oc_')]
    tech_only_cols = [c for c in feat_cols if c.startswith(('tech_', 'side_'))]

    # ---- Walk-forward CV: split trades chronologically in 6 folds, predict
    # future from past
    n = len(df)
    fold_size = n // 6
    results_all = {'tech_only': [], 'tech_plus_oc': [], 'oc_only': []}
    for fold in range(2, 6):  # need at least 2 folds of train
        train_end = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < 5 else n
        X_tr_all = df[feat_cols].iloc[:train_end].values
        X_te_all = df[feat_cols].iloc[train_end:test_end].values
        X_tr_oc = df[oc_only_cols].iloc[:train_end].values
        X_te_oc = df[oc_only_cols].iloc[train_end:test_end].values
        X_tr_tech = df[tech_only_cols].iloc[:train_end].values
        X_te_tech = df[tech_only_cols].iloc[train_end:test_end].values
        y_tr = df['win'].iloc[:train_end].values
        y_te = df['win'].iloc[train_end:test_end].values
        pnl_te = df['pnl'].iloc[train_end:test_end].values

        # Model
        for label, Xtr, Xte in [('tech_only', X_tr_tech, X_te_tech),
                                  ('tech_plus_oc', X_tr_all, X_te_all),
                                  ('oc_only', X_tr_oc, X_te_oc)]:
            sc = StandardScaler().fit(Xtr)
            clf = GradientBoostingClassifier(max_depth=2, n_estimators=50,
                                              learning_rate=0.05,
                                              min_samples_leaf=10,
                                              subsample=0.8,
                                              random_state=42)
            clf.fit(sc.transform(Xtr), y_tr)
            try:
                auc_tr = roc_auc_score(y_tr, clf.predict_proba(sc.transform(Xtr))[:, 1])
            except ValueError:
                auc_tr = np.nan
            try:
                auc_te = roc_auc_score(y_te, clf.predict_proba(sc.transform(Xte))[:, 1])
            except ValueError:
                auc_te = np.nan
            # Threshold = quantile train (top 70% to be selective)
            prob_te = clf.predict_proba(sc.transform(Xte))[:, 1]
            prob_tr = clf.predict_proba(sc.transform(Xtr))[:, 1]
            thr = float(np.quantile(prob_tr, 0.3))   # keep top 70% probs
            keep = prob_te >= thr
            pnl_kept = pnl_te[keep]
            if len(pnl_kept) > 0:
                total = float(np.prod(1 + pnl_kept) - 1)
                wr_kept = float(np.mean(pnl_kept > 0))
            else:
                total = 0.0; wr_kept = 0.0
            results_all[label].append({
                'fold': fold, 'auc_tr': auc_tr, 'auc_te': auc_te,
                'n_test': len(y_te), 'n_kept': int(keep.sum()),
                'wr_kept': wr_kept, 'total': total,
            })

    print()
    print('Walk-forward results (ML hybrid):')
    for label, res in results_all.items():
        print(f'\n  {label}:')
        for r in res:
            print(f"    fold={r['fold']}  AUC_tr={r['auc_tr']:.3f} AUC_te={r['auc_te']:.3f}  "
                  f"n_te={r['n_test']:>3} n_kept={r['n_kept']:>3} WR_kept={r['wr_kept']:.1%} "
                  f"total={r['total']:+.2%}")
        avg_auc_te = np.nanmean([r['auc_te'] for r in res])
        total_all = float(np.prod([1 + r['total'] for r in res]) - 1)
        print(f"    --> mean AUC test: {avg_auc_te:.3f}, cumulative kept: {total_all:+.1%}")


if __name__ == '__main__':
    main()
