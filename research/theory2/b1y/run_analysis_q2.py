"""Q2 analysis driver (PLAN.md section 4.2 + Addendum 1): merge test shards -> results/q2.csv.gz; precondition, HY1, HY2, HY3, manipulation check per CO method
(DAGMA, NOTEARS, GOLEM-EV, GOLEM-NV) and family; timeouts; claims for Holm. Writes results/analysis_Q2.txt and results/analysis_Q2.json.
  python run_analysis_q2.py
"""
import glob, json, os
import pandas as pd
import analyse_y as ay

HERE = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(HERE, "results", "shards", "q2", "q2_test_*.csv")))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv(os.path.join(HERE, "results", "q2.csv.gz"), index=False)
lines = [f"Q2 analysis. {len(df)} rows from {len(files)} shards; graphs {df.groupby(['family', 'p', 'seed']).ngroups}.",
         "Status counts: " + json.dumps(df.status.value_counts().to_dict()), ""]
tout = df[df.status != "ok"]
if len(tout):
    lines.append("Fits not finished (status != ok), by method and arm: " + json.dumps(tout.groupby(["method", "arm"]).size().to_dict(), default=str)); lines.append("")
r = ay.q2_extended(df)
tuned = json.load(open(os.path.join(HERE, "results", "q2_tuned.json")))
r["tuned_hyperparameters"] = tuned
lines.append("Tuned hyper-parameters (validation seeds): " + json.dumps(tuned)); lines.append("")
lines.append(json.dumps(r, indent=1, default=float))
# level table: mean SHD_x1n by method x arm x family (descriptive)
d = df[df.status == "ok"]
tab = d.groupby(["family", "method", "arm"]).shd_x1n.mean().unstack("arm").round(3)
lines.append("\nMean SHD_x1n by family, method, arm (graph x T rows pooled)\n" + tab.to_string())
open(os.path.join(HERE, "results", "analysis_Q2.txt"), "w").write("\n".join(lines))
json.dump(r, open(os.path.join(HERE, "results", "analysis_Q2.json"), "w"), indent=1, default=float)
print("\n".join(lines[:60]))
