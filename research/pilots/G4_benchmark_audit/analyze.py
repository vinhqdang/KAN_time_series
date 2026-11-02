"""Recompute the random-order control (first version used RandomState(seed) = the generator's own
permutation stream, i.e. it leaked the TRUE order; fixed to seed+1e6), then aggregate."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import g4_audit as G
from src.synthetic import generate_instantaneous_anm
import instantaneous_dag_benchmark as B

OUT = G.OUT
frames = []
for gen in ["A_raw", "B_std"]:
    p = os.path.join(OUT, f"results_{gen}.csv"); df = pd.read_csv(p)
    if "random_leak_fixed" not in df.columns:
        old = df[df.method == "random-sortnregress"].copy()
        df.loc[df.method == "random-sortnregress", "leaked_auroc_old"] = old.auroc
        for i, r in old.iterrows():
            X, A = (generate_instantaneous_anm if gen == "A_raw" else G.gen_std)(1500, int(r.d), seed=int(r.seed))
            S = G.baselines(X, int(r.seed))["random-sortnregress"]
            df.loc[i, ["auroc", "auprc", "f1"]] = B.score(S, A)
        df["random_leak_fixed"] = 1; df.to_csv(p, index=False)
    frames.append(df)
df = pd.concat(frames)
meths = ["var-sortnregress", "R2-sortnregress", "random-sortnregress", "SPADE", "NOTEARS-linear", "DAGMA-linear", "GOLEM", "NOTEARS-MLP"]
ds = df.drop_duplicates(["gen", "d", "seed"])
print("== sortability (mean over seeds)")
print(ds.groupby(["gen", "d"])[["vs_raw", "vs_std", "r2s_raw", "r2s_std", "n_edges"]].mean().round(3))
print("== AUROC mean (sd) n")
g = df.groupby(["gen", "d", "method"]).auroc.agg(["mean", "std", "count"]).round(3)
t = g["mean"].unstack("method")[meths]; print(t)
print(g["std"].unstack("method")[meths])
print("== per-dataset corr of AUROC with sortability (A_raw, all d pooled, Spearman)")
for gen in ["A_raw", "B_std"]:
    sub = df[df.gen == gen]; w = sub.pivot_table(index=["d", "seed"], columns="method", values="auroc")
    ms = ds[ds.gen == gen].set_index(["d", "seed"])[["vs_raw", "r2s_raw"]].loc[w.index]
    for m in meths:
        print(gen, m, "rho(vs_raw)=%.2f rho(r2s_raw)=%.2f" % (w[m].corr(ms.vs_raw, method="spearman"), w[m].corr(ms.r2s_raw, method="spearman")))
    # paired gain SPADE minus var-sortnregress
    for b in ["var-sortnregress", "R2-sortnregress", "NOTEARS-linear"]:
        dd = (w["SPADE"] - w[b]).groupby(level="d").agg(["mean", "std", "count"]); print(gen, "SPADE -", b); print(dd.round(3))
print("== variance floor")
b = pd.read_csv(os.path.join(OUT, "variance_floor_between.csv"))
print(b.groupby("method").auroc.agg(["mean", "std", "count"]).round(3))
w = b.pivot(index="seed", columns="method", values="auroc"); dd = w.SPADE - w["NOTEARS-linear"]
print("paired SPADE-NOTEARSlin over 30 seeds: mean %.3f sd %.3f se %.3f" % (dd.mean(), dd.std(), dd.std() / np.sqrt(len(dd))))
w2 = pd.read_csv(os.path.join(OUT, "variance_floor_within.csv"))
m = w2[w2.arm == "mseed"]
print("method-seed sd within dataset (mean over 5 datasets):", round(m.groupby("seed").auroc.std().mean(), 3),
      "range per dataset", m.groupby("seed").auroc.agg(lambda x: x.max() - x.min()).round(3).tolist())
print(m.groupby("seed").auroc.agg(["mean", "std"]).round(3))
for arm in ["repeat_1thr", "thr2", "float64"]:
    a = w2[w2.arm == arm]; print(arm, "dAUROC:", a.dAUROC.round(3).tolist(), "RMS %.3f" % np.sqrt((a.dAUROC ** 2).mean()), "max|dS|", a.maxdS.round(4).tolist())
