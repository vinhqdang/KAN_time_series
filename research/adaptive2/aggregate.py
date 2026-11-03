import glob, pandas as pd, numpy as np
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
R = "../replication/results/all.csv"
out = []
g = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/grid_*.csv"))])
g["gamma"] = g.get("gamma"); 
def lab(r):
    return r.method
t = g.groupby("method")[["nrmse", "slope", "auroc", "orient", "t_cost"]].mean().round(4)
nd = g.groupby("method").seed.nunique(); t["n"] = nd
a = pd.read_csv(R); a = a[(a.d == 100) & (a.noise == "unif") & (a.n == 1000) & (a.mech == "m1") & a.seed.between(2000, 2004)]
bw = a[a.method == "BASILWs"].set_index("seed").nrmse; ssr = a[a.method == "SSR_k1.5"].set_index("seed").nrmse
out.append("GRID (d100 unif, seeds 2000-2004), mean over 5 datasets\n" + t.to_string())
out.append("reference (replication all.csv, same seeds): " + str(a.groupby("method")[["nrmse", "slope", "auroc"]].mean().round(4).to_dict("index")))
dfl = g[g.method == "SSA"].set_index("seed").nrmse
rows = []
for m, sub in g[g.method.str.startswith("SSA")].groupby("method"):
    s = sub.set_index("seed").nrmse
    rows.append((m, s.mean(), (s - dfl).mean(), int((s < dfl).sum()), int((s <= bw + 0.01).sum()) if False else int((s.values <= bw.loc[s.index].values + 0.01).sum()), s.min(), s.max()))
d = pd.DataFrame(rows, columns=["setting", "nrmse", "minus_default", "n_better_than_default(of5)", "n_datasets_reaching_BASILW+0.01", "min", "max"]).round(4)
out.append("vs default and BASIL-W (BASIL-W mean %.4f)\n" % bw.mean() + d.to_string(index=False))
out.append("per-dataset NRMSE\n" + g.pivot(index="method", columns="seed", values="nrmse").round(3).to_string())
out.append("per-dataset AUROC\n" + g.pivot(index="method", columns="seed", values="auroc").round(4).to_string())
# part A
fs = sorted(glob.glob("results/d500_*.csv"))
if fs:
    n = pd.concat([pd.read_csv(f) for f in fs]); n["src"] = "rerun"
    old = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("../spire/results/do/d500_*.csv"))]); old["src"] = "old_do"
    st = pd.read_csv("../spire/results/do_stab/d500_unif_m1_n1000_s1001.csv"); st["src"] = "old_stab"; st["method"] = st.method.replace({"BASILW": "BASILWs"})
    allr = pd.concat([n, old, st])
    c = ["seed", "src", "method", "nrmse", "slope", "do_direct", "do_indirect", "auroc", "orient", "t_ss", "t_cost", "time"]
    out.append("PART A raw rows\n" + allr[c].sort_values(["seed", "method", "src"]).round(4).to_string(index=False))
    ra = n.groupby("method")[["nrmse", "slope", "do_direct", "do_indirect", "auroc", "orient", "t_ss", "t_cost"]].mean().round(4)
    out.append("PART A rerun, mean over datasets present\n" + ra.to_string())
open("analysis.txt", "w").write("\n\n".join(out)); print("\n\n".join(out))
