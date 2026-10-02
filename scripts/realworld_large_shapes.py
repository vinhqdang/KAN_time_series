"""SPADE sign/shape checks on the hourly datasets (criteria fixed in realworld_large.py's pre-registration).
bike_hourly: workingday->registered positive, workingday->casual negative.
beijing:     Iws->pm decreasing (phi(q90) < phi(q10) of the standardized cause).
Fits SPADE on the six primary resamples (n=2,000) of each dataset; also reports the edge's importance rank."""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
import numpy as np, pandas as pd
from realworld_large import LOADERS, block_bootstrap, N_SUB, B_RES, OUT_DIR
from realworld_bike_shapes import fit, phi

CHECKS = {"bike_hourly": [("workingday", "registered", "pos"), ("workingday", "casual", "neg")],
          "beijing": [("Iws", "pm", "dec")]}
rows = []
for ds, checks in CHECKS.items():
    X0, names, tier, pos = LOADERS[ds](); idx = {n: i for i, n in enumerate(names)}; d = len(names)
    for r in range(B_RES + 1):
        X = block_bootstrap(X0, r, N_SUB); m, Xt, Z = fit(X, r); imp = m.importance(Xt).numpy()
        flat = sorted([(imp[i, j], i, j) for i in range(d) for j in range(d) if i != j], reverse=True)
        rank = {(i, j): k + 1 for k, (_, i, j) in enumerate(flat)}
        for cause, effect, kind in checks:
            c, e = idx[cause], idx[effect]
            if kind in ("pos", "neg"):
                v = np.unique(Z[:, c]); f = phi(m, e, c, v); delta = float(f[-1] - f[0])
            else:
                q = np.quantile(Z[:, c], [0.1, 0.9]); f = phi(m, e, c, q); delta = float(f[1] - f[0])
            ok = delta > 0 if kind == "pos" else delta < 0
            rows.append(dict(dataset=ds, resample=r, cause=cause, effect=effect, expected=kind, delta=delta,
                             sign_ok=bool(ok), importance=float(imp[e, c]), rank=rank[(e, c)], n_edges_total=d * (d - 1)))
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT_DIR, "realworld_large_shapes.csv"), index=False)
print(df.round(3).to_string())
print(df.groupby(["dataset", "cause", "effect"]).agg(ok=("sign_ok", "sum"), n=("sign_ok", "size"), med_rank=("rank", "median")))
