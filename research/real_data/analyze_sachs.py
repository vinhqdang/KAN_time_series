"""Analysis for research/real_data/PLAN.md -> results/analysis.txt, results/*.csv, fig_curves.png"""
import os, sys, itertools, numpy as np, pandas as pd
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results"); ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path[:0] = [os.path.join(ROOT, "scripts")]
import sachs_benchmark as SB
X, A, names = SB.load(); d = 11; nid = {n: i for i, n in enumerate(names)}
GRID = np.linspace(-1.8, 1.8, 50); rng = np.random.RandomState(0)
st = pd.read_csv(os.path.join(RES, "structure.csv")); out = []
P = lambda *a: out.append(" ".join(str(x) for x in a))
def boot(x, B=10000):
    x = np.asarray(x, float); m = rng.choice(len(x), (B, len(x))); b = x[m].mean(1); return np.percentile(b, [2.5, 97.5])
def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); df = a - b; lo, hi = boot(df)
    try: p = wilcoxon(a, b).pvalue
    except Exception: p = np.nan
    return f"diff {df.mean():+.3f} CI[{lo:+.3f},{hi:+.3f}] wilcoxon p={p:.3f}"
mets = ["auroc", "auprc", "skel_auroc", "orient", "hits18"]
P("== STRUCTURE (mean +- sd over 10 subsamples; bootstrap 95% CI of mean AUROC)")
tab = st.groupby("method")[mets + ["time"]].agg(["mean", "std"]).round(3); P(tab.to_string())
for m, g in st.groupby("method"):
    lo, hi = boot(g.auroc); P(f"  {m:28s} n={len(g)} AUROC CI[{lo:.3f},{hi:.3f}]")
piv = {m: st[st.method == m].sort_values("sub").set_index("sub") for m in st.method.unique()}
P("\n== PAIRED (method minus reference)")
for ref in ["SSC", "NOTEARS-linear"]:
    for m in piv:
        if m == ref: continue
        for k in ["auroc", "auprc", "orient"]:
            if len(piv[m]) == 10 and len(piv[ref]) == 10: P(f"{m:26s} - {ref:15s} {k:7s} {paired(piv[m][k], piv[ref][k])}")
st.groupby("method")[mets].mean().round(4).to_csv(os.path.join(RES, "structure_summary.csv"))

# ---- curves
cm = ["SSC", "SSR_k1.5", "BASILW", "BASIL", "GAM-ridge"]
Cg, Am, Sc = {}, {}, {}
for m in cm:
    Cg[m] = np.stack([np.load(os.path.join(RES, "curves", f"{m}_{r}.npz"))["grid"] for r in range(10)]); Am[m] = np.stack([np.load(os.path.join(RES, "curves", f"{m}_{r}.npz"))["amp"] for r in range(10)])
for m in cm + ["NOTEARS-linear"]: Sc[m] = np.stack([np.load(os.path.join(RES, "scores", f"{m}_{r}.npy")) for r in range(10)])
off = ~np.eye(d, dtype=bool)
def top18(S):
    f = np.where(off, np.abs(S), -1).flatten(); idx = np.argsort(-f, kind="stable")[:18]; return {divmod(int(i), d) for i in idx}
def pair_stats(curves):  # curves list of 50-vectors -> (mean R2_aff, mean signed corr)
    r2, cr = [], []
    for a, b in itertools.combinations(curves, 2):
        if a.std() < 1e-12 or b.std() < 1e-12: r2.append(0.0); cr.append(0.0); continue
        c = np.corrcoef(a, b)[0, 1]; r2.append(c * c); cr.append(c)
    return np.mean(r2), np.mean(cr)
P("\n== STABILITY, plan definition: edges in method's own top-18 in >=3 subsamples")
rows = []
for m in cm:
    T = [top18(Sc[m][r]) for r in range(10)]; cnt = {}
    for t in T:
        for e in t: cnt[e] = cnt.get(e, 0) + 1
    E = [e for e, k in cnt.items() if k >= 3]; res = []
    for e in E:
        sub = [r for r in range(10) if e in T[r]]; res.append((e, len(sub), *pair_stats([Cg[m][r][e] for r in sub])))
    true = [x for x in res if A[x[0]] == 1]
    rows.append(dict(method=m, n_edges=len(res), r2aff=np.mean([x[2] for x in res]), corr=np.mean([x[3] for x in res]), n_true=len(true),
                     r2aff_true=np.mean([x[2] for x in true]) if true else np.nan, corr_true=np.mean([x[3] for x in true]) if true else np.nan))
sd_ = pd.DataFrame(rows).round(3); P(sd_.to_string(index=False)); sd_.to_csv(os.path.join(RES, "stability_plan.csv"), index=False)
P("\n== STABILITY, common edge set (SSC top-18 in >=3 subsamples), per-subsample score = mean over edges and other subsamples of R2_aff; paired over subsamples")
T = [top18(Sc["SSC"][r]) for r in range(10)]; cnt = {}
for t in T:
    for e in t: cnt[e] = cnt.get(e, 0) + 1
E = sorted([e for e, k in cnt.items() if k >= 3]); P("edges:", [(names[c], names[e], cnt[(c, e)], int(A[c, e])) for c, e in E])
per = {}
for m in cm:
    v = []
    for r in range(10):
        s = [np.mean([(np.corrcoef(Cg[m][r][e], Cg[m][q][e])[0, 1] ** 2) if Cg[m][r][e].std() > 1e-12 and Cg[m][q][e].std() > 1e-12 else 0.0 for q in range(10) if q != r]) for e in E]
        v.append(np.mean(s))
    per[m] = np.array(v); lo, hi = boot(v); P(f"  {m:12s} mean R2_aff {np.mean(v):.3f} CI[{lo:.3f},{hi:.3f}]")
for m in cm[1:]: P(f"  {m} - SSC : {paired(per[m], per['SSC'])}")
pd.DataFrame(per).round(4).to_csv(os.path.join(RES, "stability_common.csv"), index_label="sub")

P("\n== STABILITY, common edge set, CONDITIONAL on both subsamples having a non-zero curve (edge selected in both); mean R2_aff over edges/pairs; n_pairs = number of such pairs")
for m in cm:
    r2s = []
    for e in E:
        for r, q in itertools.combinations(range(10), 2):
            if Am[m][r][e] > 1e-9 and Am[m][q][e] > 1e-9 and Cg[m][r][e].std() > 1e-12 and Cg[m][q][e].std() > 1e-12: r2s.append(np.corrcoef(Cg[m][r][e], Cg[m][q][e])[0, 1] ** 2)
    P(f"  {m:12s} mean R2_aff {np.mean(r2s) if r2s else np.nan:.3f} over {len(r2s)} of {len(E)*45} edge-pairs; share of zero-curve edge-subsamples {np.mean([Am[m][r][e] <= 1e-9 for e in E for r in range(10)]):.2f}")

P("\n== AMPLITUDE ratio of curve std on SSC top-18 edges (per subsample mean of per-edge ratios; CI over subsamples)")
for m in ["SSR_k1.5", "BASILW", "BASIL", "GAM-ridge"]:
    rr = []
    for r in range(10):
        ed = sorted(top18(Sc["SSC"][r])); rr.append(np.mean([Am[m][r][e] / max(Am["SSC"][r][e], 1e-12) for e in ed]))
    lo, hi = boot(rr); P(f"  {m:10s}/SSC mean {np.mean(rr):.2f} CI[{lo:.2f},{hi:.2f}] (min {min(rr):.2f}, max {max(rr):.2f})")
for m in cm:
    P(f"  mean curve std over SSC top-18 edges, {m}: {np.mean([np.mean([Am[m][r][e] for e in top18(Sc['SSC'][r])]) for r in range(10)]):.3f}")

P("\n== SIGN SANITY: correlation(curve, cause) > 0 counted correct (curve with zero amplitude counted as absent)")
for c, e in [("praf", "pmek"), ("pmek", "p44/42"), ("PIP3", "pakts473")]:
    ci, ei = nid[c], nid[e]; line = []
    for m in cm:
        pos = neg = ab = 0
        for r in range(10):
            y = Cg[m][r][ci, ei]
            if Am[m][r][ci, ei] < 1e-9 or y.std() < 1e-12: ab += 1; continue
            if np.corrcoef(y, GRID)[0, 1] > 0: pos += 1
            else: neg += 1
        line.append(f"{m}: +{pos}/-{neg}/absent{ab}")
    P(f"  {c}->{e} (marginal corr in log data {np.corrcoef(X[:, ci], X[:, ei])[0,1]:+.3f}):", "; ".join(line), "| mean score SSC %.3f BASILW %.3f" % (Sc['SSC'][:, ci, ei].mean(), Sc['BASILW'][:, ci, ei].mean()))

# per-edge mean scores for the report
P("\n== TOP EDGES by mean SSC score (mean over subsamples), with truth flag and rank-score of other methods")
ms = Sc["SSC"].mean(0); idx = np.argsort(-np.where(off, ms, -1).flatten())[:12]
for i in idx:
    c, e = divmod(int(i), d); P(f"  {names[c]}->{names[e]} truth={int(A[c,e])} reverse_truth={int(A[e,c])} SSC={ms[c,e]:.3f} SSR={Sc['SSR_k1.5'].mean(0)[c,e]:.3f} BASILW={Sc['BASILW'].mean(0)[c,e]:.3f} BASIL={Sc['BASIL'].mean(0)[c,e]:.3f}")
open(os.path.join(RES, "analysis.txt"), "w").write("\n".join(out)); print("\n".join(out))

# ---- figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ed6 = [divmod(int(i), d) for i in idx[:6]]
fig, ax = plt.subplots(3, 6, figsize=(20, 8.5), sharex=True)
cols = {"SSC": "#1f77b4", "SSR_k1.5": "#d95f02", "BASILW": "#2ca02c"}
for j, (c, e) in enumerate(ed6):
    for i, m in enumerate(["SSC", "SSR_k1.5", "BASILW"]):
        a = ax[i, j]
        for r in range(10): a.plot(GRID, Cg[m][r][c, e], color=cols[m], alpha=0.45, lw=1)
        a.plot(GRID, Cg[m].mean(0)[c, e], color="k", lw=2); a.axhline(0, color="gray", lw=.5)
        a.set_ylim(-1.2, 1.2)
        if i == 0: a.set_title(f"{names[c]} -> {names[e]}" + (" (consensus)" if A[c, e] else " (reverse of consensus)" if A[e, c] else " (not in consensus)"), fontsize=10)
        if j == 0: a.set_ylabel({"SSC": "SSC", "SSR_k1.5": "SSR (refit k=1.5)", "BASILW": "BASIL-W"}[m] + "\ncentred curve")
        if i == 2: a.set_xlabel(f"{names[c]} (z-score, log scale)")
fig.suptitle("Sachs: edge curves over 10 subsamples (colored), black = mean incl. subsamples where the edge was not selected (flat zero line); top 6 edges by mean SSC score", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_curves.png"), dpi=110)
