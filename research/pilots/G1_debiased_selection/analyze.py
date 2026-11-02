import os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
H = os.path.dirname(os.path.abspath(__file__)); out = []
for d in (10, 20):
    f = os.path.join(H, f"raw_test_d{d}.csv")
    if not os.path.exists(f): continue
    df = pd.read_csv(f); n = df.seed.nunique(); out.append(f"\n=== d={d}, n_graphs={n} (test seeds) ===")
    methods = ["base", "adaptive", "stab", "refit_s0.001", "refit_s0.0", "dss_s0.001"]
    mets = ["auroc", "auprc", "curve_r2", "slope"]
    m = df.groupby("method")[mets].mean().reindex(methods); out.append(m.round(3).to_string())
    out.append("paired Wilcoxon vs base (mean diff, p):")
    b = df[df.method == "base"].set_index("seed")
    for meth in methods[1:]:
        o = df[df.method == meth].set_index("seed"); row = []
        for k in mets[:3]:
            if o[k].isna().all(): continue
            diff = (o[k] - b[k].reindex(o.index)).dropna()
            try: p = wilcoxon(diff).pvalue
            except Exception: p = float("nan")
            row.append(f"{k}: {diff.mean():+.3f} (p={p:.3g}, W/L={int((diff>0).sum())}/{int((diff<0).sum())})")
        out.append(f"  {meth:13s} " + " | ".join(row))
print("\n".join(out)); open(os.path.join(H, "results.txt"), "w").write("\n".join(out))
