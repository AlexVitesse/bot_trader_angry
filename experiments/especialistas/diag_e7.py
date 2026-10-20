"""Descriptivo post-hoc de E7 (no decide): que feature macro lleva la mejora y AUC absoluto."""
import numpy as np
from sklearn.metrics import roc_auc_score
from test_especialistas import construir, walk_forward, BASE, MACRO, BLOQUE

df = construir()
pb, _ = walk_forward(df, BASE)
y = df['y'].reindex(pb.index)
for f in MACRO:
    p, c = walk_forward(df, BASE + [f])
    print(f"base+{f:<7} AUC {roc_auc_score(y, p):.3f}  coef {np.round(c, 2)}")
pm, _ = walk_forward(df, BASE + MACRO)
rng = np.random.default_rng(1)
yy, pp = y.to_numpy(), pm.to_numpy()
blq = np.arange(len(yy)) // BLOQUE; nb = blq.max() + 1
g = [np.flatnonzero(blq == b) for b in range(nb)]
a = []
for _ in range(10000):
    ii = np.concatenate([g[j] for j in rng.integers(0, nb, nb)])
    if yy[ii].min() != yy[ii].max():
        a.append(roc_auc_score(yy[ii], pp[ii]))
a = np.array(a); obs = roc_auc_score(yy, pp)
print(f"AUC base+macro {obs:.3f}  IC90 [{np.percentile(a,5):.3f}, {np.percentile(a,95):.3f}]  p(AUC<=0,5) = {np.mean(a - obs >= obs - 0.5):.3f}")
