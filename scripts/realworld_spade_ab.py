"""
Post-hoc check (these datasets were already seen in the published analysis): SPADE-AB vs SPADE on the six
common-sample resamples of the hourly bike-sharing and Beijing studies, same BK-AUROC protocol as realworld_large.py.
SPADE-AB was frozen on synthetic validation seeds (X1) before this run; nothing is tuned here.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
import numpy as np, pandas as pd
import realworld_large as R, spade_variants as V

rows = []
for ds in ("bike_hourly", "beijing"):
    X0, names, tier, pos = R.LOADERS[ds]()
    for r in range(R.B_RES + 1):
        X = R.block_bootstrap(X0, r, R.N_SUB)
        for name, key in [("SPADE", "base"), ("SPADE-AB", "adaptive_boot")]:
            met = R.metrics(V.VARIANTS[key](X, r), tier, pos, names)
            rows.append(dict(dataset=ds, resample=r, method=name, **met)); print(ds, r, name, round(met["bk_auroc"], 3), flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(os.path.dirname(__file__), "..", "experimental_results", "realworld_spade_ab.csv"), index=False)
print(df.groupby(["dataset", "method"]).bk_auroc.agg(["mean", "std"]).round(3))
