import pickle, json, sys, os, pandas as pd, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
from shape_bench import summarise
D = [pickle.load(open(f, "rb")) for f in ("raw_2000_2005.pkl", "raw_2005_2010.pkl")]
df = pd.concat([d[0] for d in D]); A = {}; S = {}
for d in D: A.update(d[1]); S.update(d[2])
tau = json.load(open("tau_by_method.json"))
s = summarise(df, A, S, tau); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
s.round(3).to_csv("summary_by_rho.csv", index=False)
print(s.round(3).to_string())
p = s.groupby("method").mean(numeric_only=True).drop(columns="rho").round(3); p.to_csv("summary_pooled.csv")
print(p.to_string())
print(p[["AUROC", "AUPRC", "F1", "SF-F1", "SFrecall", "condR2", "uncondR2", "stab", "F1_k", "SF-F1_k"]].rank(ascending=False).to_string())
rows = []
for (rho, n), g in df.groupby(["rho", "method"]):
    for r in g.itertuples():
        Sm = S[(r.seed, rho, n)]
        rows.append(dict(method=n, mech=r.mech, r2=r.r2, found=Sm[r.c, r.e] / (Sm * (1 - np.eye(8))).max() >= tau[n]))
m = pd.DataFrame(rows)
print(m.pivot_table(index="method", columns="mech", values="r2").round(2).to_string())
print(m.pivot_table(index="method", columns="mech", values="found").round(2).to_string())
print(df.groupby("method").time.mean().round(1))
print("n true edges per rho:", df[df.method == "SPADE"].groupby("rho").size().to_dict())
# bootstrap CI for SPADE minus others on SF-F1 (seed-level, pooled rho)
rng = np.random.RandomState(0)
def per_seed(name, key):
    v = []
    for seed in sorted(A):
        for rho in (0.0, 0.6, 0.9):
            g = df[(df.seed == seed) & (df.rho == rho) & (df.method == name)]
            Sm = S[(seed, rho, name)]; sel = (Sm * (1 - np.eye(8))) / (Sm * (1 - np.eye(8))).max() >= tau[name]
            fnd = np.array([sel[r.c, r.e] for r in g.itertuples()]); good = fnd & (g.r2.values >= .8)
            v.append(2 * good.sum() / (sel.sum() + A[seed].sum()) if key == "sf" else g.auroc.iloc[0])
    return np.array(v)
for key in ("sf", "au"):
    sp = per_seed("SPADE", key)
    for o in ("MLP-edge", "NT-MLP", "GAM-ridge", "Linear"):
        dlt = sp - per_seed(o, key); bs = [dlt[rng.randint(0, len(dlt), len(dlt))].mean() for _ in range(2000)]
        print(key, "SPADE -", o, round(dlt.mean(), 3), np.round(np.percentile(bs, [2.5, 97.5]), 3))
