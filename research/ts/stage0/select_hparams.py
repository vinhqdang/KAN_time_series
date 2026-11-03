import pandas as pd, json
import os
r = pd.concat([pd.read_csv(f) for f in ("results_val.csv", "results_val2.csv") if os.path.exists(f)]); r = r[r.scale == "std"]
sel = {}; lines = ["# Validation (seeds 5000..5004, 20 subgraphs): mean directed AUROC by config and resolution\n",
                   "| method | config | 6h | 12h | 24h |", "|---|---|---|---|---|"]
for rn in ["6h", "12h", "24h"]: sel[rn] = {}
for m, g in r.groupby("method"):
    t = g.groupby(["cfg", "res"]).auroc.mean().unstack()
    for cfg, row in t.iterrows():
        lines.append(f"| {m} | {cfg} | " + " | ".join(f"{row[k]:.3f}" for k in ["6h", "12h", "24h"]) + " |")
    if m in ("pcmci+", "varlingam", "dynotears", "notears_svar2"):
        for rn in ["6h", "12h", "24h"]:
            best = t[rn].idxmax(); c = json.loads(best); c.pop("p", None); sel[rn][m] = c
json.dump(sel, open("selected_hparams.json", "w"), indent=1)
open("validation.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines)); print(json.dumps(sel))
print(r.groupby("method").err.apply(lambda s: (s.fillna("") != "").sum()))
