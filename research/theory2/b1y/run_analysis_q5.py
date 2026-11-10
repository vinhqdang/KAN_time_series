"""Q5 analysis driver (PLAN.md section 7 + Addendum 1 A3): merge shards -> results/q5.csv.gz; HB1 and the full method table (SHD_x1n with graph-cluster CIs),
the G1 -> G2 shift of every method, status counts. Writes results/analysis_Q5.txt and results/analysis_Q5.json.
  python run_analysis_q5.py
"""
import glob, json, os
import pandas as pd
import analyse_y as ay

HERE = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(HERE, "results", "shards", "q5", "q5_test_*.csv")))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv(os.path.join(HERE, "results", "q5.csv.gz"), index=False)
r = ay.q5_hb1(df)
lines = [f"Q5 analysis. {len(df)} rows from {len(files)} shards; graphs {df.groupby(['family', 'p', 'seed']).ngroups}; data sets {df.groupby(['family', 'p', 'seed', 'T', 'arm']).ngroups}.",
         "Status counts: " + json.dumps(df.status.value_counts().to_dict()), ""]
bad = df[df.status != "ok"]
if len(bad):
    lines.append("Fits not finished, by method and arm: " + json.dumps({f"{k[0]}|{k[1]}": int(v) for k, v in bad.groupby(["method", "arm"]).size().items()})); lines.append("")
lines.append("HB1: " + json.dumps(r["HB1"])); lines.append("")
lines.append("Groups (family|arm): best method among the HB1 pool and FLOP's mean SHD_x1n")
for k, v in r["groups"].items():
    lines.append(f"  {k:14s} best {v['best']:13s} FLOP {v['FLOP']:.3f}  " + "  ".join(f"{m} {x:.3f}" for m, x in sorted(v["means"].items(), key=lambda t: t[1])))
lines.append("\nMean SHD_x1n by arm and method (graph-cluster 95% CI; unit = summary graph; all families, p, T pooled)")
for m in sorted({k.split("|")[1] for k in r["table_shd_x1n"]}):
    a = r["table_shd_x1n"].get(f"G1|{m}"); b = r["table_shd_x1n"].get(f"G2|{m}"); s = r["shift_G1_minus_G2"].get(m)
    lines.append(f"  {m:17s} G1 {a['mean']:.3f} [{a['ci'][0]:.3f}, {a['ci'][1]:.3f}]  G2 {b['mean']:.3f} [{b['ci'][0]:.3f}, {b['ci'][1]:.3f}]  G1-G2 {s['G1_minus_G2']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]")
ok = df[df.status == "ok"]
lines.append("\nMean secs per fit by method (CPU, one core)\n" + ok.groupby("method").secs.mean().round(2).to_string())
open(os.path.join(HERE, "results", "analysis_Q5.txt"), "w").write("\n".join(lines))
json.dump(r, open(os.path.join(HERE, "results", "analysis_Q5.json"), "w"), indent=1, default=float)
print("\n".join(lines))
