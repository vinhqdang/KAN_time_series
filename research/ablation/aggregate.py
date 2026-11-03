import sys, numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
R = "results/"; KG = np.round(np.arange(1.0, 3.0001, 0.05), 2); out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
# ---------- Part B: validation diagnosis and kappa
ev = pd.read_csv(R + "val/unif_edges.csv"); mv = pd.read_csv(R + "val/unif_main.csv")
P("## Validation (seeds 3000-3004), kappa=1 diagnosis; edges:", len(ev))
P("pointwise coverage mean (per-dataset mean):", round(ev.groupby("seed")["cov_1.0"].mean().mean(), 3), " all-rows containment:", round(ev["all_1.0"].mean(), 3), " frac edges with >=90% rows covered:", round((ev["cov_1.0"] >= 0.9).mean(), 3))
ev["Pbin"] = pd.cut(ev.P, [-.01, .5, .9, .99, 1.0], labels=["<=.5", ".5-.9", ".9-.99", ">.99"])
g = ev.groupby("Pbin", observed=True).agg(n=("P", "size"), cov=("cov_1.0", "mean"), cov_cond=("cov_cond", "mean"), allc=("all_1.0", "mean"), zmean=("zmean", "mean"), zsd=("zsd", "mean"), rmsz=("rmsz", "mean"), approx_rel=("approx_rel", "mean"), post_sd=("post_sd", "mean"), bias_rms=("bias_rms", "mean"), width=("width1", "mean"), incl=("incl_frac", "mean"))
P(g.round(3).to_string())
hi = ev[ev.P > 0.9]; P("P>0.9 edges: n=%d cov=%.3f; P<=0.9: n=%d cov=%.3f; share of the coverage shortfall (0.9-cov) from P<=0.9 edges: %.2f" % (len(hi), hi["cov_1.0"].mean(), (ev.P <= .9).sum(), ev[ev.P <= .9]["cov_1.0"].mean(), (ev.P <= .9).mean() * (0.9 - ev[ev.P <= .9]["cov_1.0"].mean()) / max(1e-9, 0.9 - ev["cov_1.0"].mean())))
P("P>0.9 edges: coverage conditional-on-inclusion band %.3f; fraction of pointwise z-scores: mean z %.3f, sd z %.3f; spline approx error / posterior sd: %.3f; bias_rms/post_sd: %.2f" % (hi.cov_cond.mean(), hi.zmean.mean(), hi.zsd.mean(), (hi.approx_rms / hi.post_sd).mean(), (hi.bias_rms / hi.post_sd).mean()))
nv = pd.read_csv(R + "val/unif_nodes.csv"); rt = nv.s2_model / nv.s2_true
P("sigma^2: model s2 / true residual var over nodes with parents: median %.3f mean %.3f; 90th pct %.3f; frac within 5%%: %.2f" % (rt.median(), rt.mean(), rt.quantile(.9), ((rt - 1).abs() < .05).mean()))
cv = np.array([ev.groupby("seed")[f"cov_{k}"].mean().mean() for k in KG]); wv = np.array([ev[f"w_{k}"].mean() for k in KG])
P("validation coverage vs kappa:", {float(k): round(float(c), 3) for k, c in list(zip(KG, cv))[::4]})
ok = np.nonzero(cv >= 0.90)[0]; kstar = float(KG[ok[0]]) if len(ok) else None
P("KAPPA* (smallest grid value with validation coverage >= 0.90):", kstar, " (max coverage on grid %.3f)" % cv.max())
open(R + "kappa.txt", "w").write(str(kstar))
# ---------- tests
def covrep(df, label, k):
    kk = f"cov_{k}"; ak = f"all_{k}"; per = df.groupby("seed")[kk].mean()
    hi_ = df[df.P > .9]
    P("%-10s k=%.2f n_ds=%d  pointwise cov %.3f (sd over ds %.3f)  [P>.9: %.3f | P<=.9: %.3f]  all-rows %.3f  >=90%%rows %.3f  width %.3f  edges %d" % (label, k, per.size, per.mean(), per.std(), hi_[kk].mean(), df[df.P <= .9][kk].mean(), df[ak].mean(), (df[kk] >= .9).mean(), df[f"w_{k}"].mean(), len(df)))
if kstar:
    P("\n## Band coverage on test (kappa=1 vs kappa*)")
    for nm, lab in (("test/unif", "unif"), ("laplace/laplace", "laplace"), ("hetero/hetero", "hetero")):
        d_ = pd.read_csv(R + nm.split("/")[0] + f"/{nm.split('/')[1]}_edges.csv"); d_ = d_[d_.variant == "base"]
        covrep(d_, lab, 1.0); covrep(d_, lab, kstar)
        if lab == "unif": 
            P("   width ratio kappa*/1: %.2f" % (d_[f"w_{kstar}"].mean() / d_["w_1.0"].mean()))
            # what kappa would be needed on test (descriptive only)
            ct = np.array([d_.groupby("seed")[f"cov_{k}"].mean().mean() for k in KG]); P("   (descriptive, not used) test coverage vs kappa:", {float(k): round(float(c), 3) for k, c in list(zip(KG, ct))[::4]})
# ---------- Part A tables
m = pd.read_csv(R + "test/unif_main.csv"); cols = ["nrmse", "slope", "do_direct", "do_indirect", "auroc", "auprc", "orient", "cover", "ece", "recall", "gap", "time"]
P("\n## Part A: mean over test datasets (n per variant: %s)" % m.groupby("variant").size().to_dict())
order = ["base", "v1_R2cand", "v2_cold", "v3_cold_R2", "v4a_g_n/4", "v4b_g_4n", "v5a_pi_x0.5", "v5b_pi_x2", "v6_kmax2", "v7_4ch_600d"]
T = m.groupby("variant")[cols].mean().reindex(order); P(T.round(3).to_string())
P("t_spade mean: %.1f s" % m.groupby("seed").t_spade.first().mean())
P("\n## paired differences to base (variant - base): mean (sd) [# datasets worse of n]; 'worse' = higher for nrmse/do/gap/time, lower otherwise")
b = m[m.variant == "base"].set_index("seed"); rows = []
for v in order[1:]:
    x = m[m.variant == v].set_index("seed"); idx = b.index.intersection(x.index); r = {"variant": v}
    for c in ["nrmse", "do_indirect", "auroc", "auprc", "orient", "cover", "recall", "gap", "time"]:
        dd = (x.loc[idx, c] - b.loc[idx, c]); worse = (dd > 0).sum() if c in ("nrmse", "do_indirect", "gap", "time") else (dd < 0).sum()
        r[c] = "%+.3f (%.3f) [%d/%d]" % (dd.mean(), dd.std(), worse, len(idx))
    rows.append(r)
P(pd.DataFrame(rows).set_index("variant").to_string())
open(R + "analysis.txt", "w").write("\n".join(out))
