"""E11: Hang Seng y FTSE sobre base + macro E7. Pre-registro en README.md.

Uso: python experiments/otras_bolsas/test_otras_bolsas.py [n_rotaciones]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'especialistas'))
from test_especialistas import (construir, walk_forward, boot_delta, auc_por_año,  # noqa: E402
                                z_relativo, BASE, MACRO, SEED, ROOT)

NUEVAS = ['hsi_z', 'ftse_z']


def main():
    n_rot = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    rng = np.random.default_rng(SEED + 11)
    df = construir()
    v = pd.read_parquet(ROOT / 'data' / 'BTC_USDT_4h_full.parquet')
    c = v['close'].astype(float)
    c.index = c.index.tz_localize('UTC') if c.index.tz is None else c.index
    c = c.resample('1D').last().dropna()
    for k, n in (('hsi_z', 'hsi'), ('ftse_z', 'ftse')):
        df[k] = z_relativo(c, n).reindex(df.index)
    df = df.dropna()
    print(f"Dias: {len(df)}  corr con base/oro_z:")
    print(df[BASE + ['oro_z'] + NUEVAS].corr().loc[NUEVAS, BASE + ['oro_z']].round(2).to_string())

    ref = BASE + MACRO
    p_ref, _ = walk_forward(df, ref)
    y = df['y'].reindex(p_ref.index)
    modelos = {'base+macro (ref)': ref, 'ref+hsi+ftse': ref + NUEVAS,
               'base+hsi_z': BASE + ['hsi_z'], 'base+ftse_z': BASE + ['ftse_z'], 'base': BASE}
    preds = {}
    for nombre, f in modelos.items():
        preds[nombre], coef = walk_forward(df, f)
        pa = auc_por_año(y, preds[nombre])
        print(f"  {nombre:<17} AUC {roc_auc_score(y, preds[nombre]):.3f} | "
              + "  ".join(f"{a}:{v_:.2f}" for a, v_ in pa.items()) + f" | coef {np.round(coef, 2)}")

    d_obs = roc_auc_score(y, preds['ref+hsi+ftse']) - roc_auc_score(y, p_ref)
    bs = boot_delta(y, preds['ref+hsi+ftse'], p_ref, rng)
    p = float((bs <= 0).mean())
    rot = []
    n = len(df)
    for k in rng.integers(90, n - 90, n_rot):
        dr = df.copy()
        dr[NUEVAS] = np.roll(df[NUEVAS].to_numpy(), k, axis=0)
        pr, _ = walk_forward(dr, ref + NUEVAS)
        rot.append(roc_auc_score(y, pr.reindex(y.index)) - roc_auc_score(y, p_ref))
    print(f"\nH11: dAUC {d_obs:+.4f}  IC90 [{np.percentile(bs, 5):+.4f}, {np.percentile(bs, 95):+.4f}]  "
          f"p = {p:.4f} -> {'PASA' if p < 0.05 else 'NO pasa'} (p<0,05)")
    print(f"   null rotacion ({n_rot}): mediana {np.median(rot):+.4f}  p_rot = {np.mean(np.array(rot) >= d_obs):.3f}")


if __name__ == '__main__':
    main()
