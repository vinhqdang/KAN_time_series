import numpy as np, pandas as pd, json
from lib import *

sub = load_sub(); e = edge_table()
out = []
P = lambda s="": out.append(s)

# ---------------- G1
e["g1_ok"] = (e.tau_e > 0) & (e.r_peak > e.r_mirror)
g1 = e.g1_ok.mean()
P("# Stage 0 validity gates (computed before any method comparison)\n")
P("Plan: research/ts/02_plan_stage0.md (commit 3cd4581). Subgraphs: all 30 seeds (5000..5029) per size and region, 120 subgraphs, %d edge occurrences (edges counted once per subgraph in which they occur)." % len(e))
P("Gates use only the sampled graphs, labels and the three scale-only controls. No method was run.\n")
P("## G1  physics oracle consistent with data (threshold: >= 70% of sampled edges)\n")
P("Criterion per edge: peak lag (over 0..14 d, upstream leading, first-differenced 15-min discharge) is > 0 and the correlation at the peak exceeds the correlation at the mirrored negative lag.\n")
P("| group | edges | frac G1-consistent | frac tau_e=0 | median tau_e (15-min steps) |")
P("|---|---|---|---|---|")
def row(name, g):
    P(f"| {name} | {len(g)} | {g.g1_ok.mean():.3f} | {(g.tau_e==0).mean():.3f} | {g.tau_e.median():.0f} |")
row("ALL", e)
for (r, d), g in e.groupby(["region", "d"]): row(f"{r} d={d}", g)
for r, g in e.groupby("region"): row(f"{r} (both sizes)", g)
ue = e.drop_duplicates(["u", "v"]); P(f"\nUnique edges: {len(ue)}; G1 fraction over unique edges: {ue.g1_ok.mean():.3f}")
P(f"\n**G1 = {g1:.3f} -> {'PASS' if g1 >= 0.70 else 'FAIL'}**\n")

# ---------------- G2
P("## G2  enough contemporaneous edges (24 h: >= 30% of edges and >= 100 contemporaneous edges across the d=10 subgraphs of each region)\n")
P("Label: contemporaneous iff tau_e < Delta/2 (Delta/2 = 3 h, 6 h, 12 h = 12, 24, 48 steps of 15 min).\n")
P("| resolution | all edges: frac contemp | ES d=10: n contemp / n | BA d=10: n contemp / n | ES frac | BA frac | verdict (all conditions) |")
P("|---|---|---|---|---|---|---|")
g2res = {}
for rn in RES:
    c = label_edges(e, rn); e["c_" + rn] = c
    f = c.mean()
    s = {}
    for r in REGIONS:
        m = (e.region == r) & (e.d == 10); s[r] = (c[m].sum(), m.sum())
    fr = {r: c[(e.region == r)].mean() for r in REGIONS}
    ok = f >= 0.30 and all(s[r][0] >= 100 for r in REGIONS)
    g2res[rn] = ok
    P(f"| {rn} | {f:.3f} | {s['east_germany'][0]}/{s['east_germany'][1]} | {s['bavaria'][0]}/{s['bavaria'][1]} | {fr['east_germany']:.3f} | {fr['bavaria']:.3f} | {'PASS' if ok else 'FAIL'} |")
P("\nSensitivity of the label rule (fraction contemporaneous, all edges):\n")
P("| resolution | tau<Delta/4 | tau<Delta/2 (primary) | tau<Delta |")
P("|---|---|---|---|")
for rn in RES:
    P(f"| {rn} | {label_edges(e, rn, .25).mean():.3f} | {label_edges(e, rn, .5).mean():.3f} | {label_edges(e, rn, 1.0).mean():.3f} |")
P(f"\n**G2 at 24 h: {'PASS' if g2res['24h'] else 'FAIL'}**; resolutions passing: {[k for k, v in g2res.items() if v]}.\n")

# ---------------- G3
P("## G3  scale leakage: orientation accuracy of scale-only controls on contemporaneous edges\n")
P("Orientation of a true edge u->v by the control's causal order (increasing score = earlier). Ties count 0.5. var-sortnregress: score = marginal variance of the series used (standardised: all variances are 1, so every pair is a tie by construction). R2-sortnregress: score = R^2 of a node regressed on all other nodes at the same time step (OLS, series standardised or raw). Extra row (not in the plan): RP = order by series mean (the CausalRivers 'reverse physical' baseline).\n")
P("Threshold: control orientation accuracy >= 0.8 on standardised data means orientation is leaked by something other than causal mechanism.\n")
rows = []
for sg in sub:
    E = [(u, v) for (u, v) in sg["edges"]]
    for rn in RES:
        for std in (True, False):
            X = get_series(sg, rn, standardize=std)
            sc = {"var": var_order_score(X), "R2": r2_order_score(X), "mean(RP)": X.mean(0)}
            for (u, v) in E:
                iu, iv = sg["nodes"].index(u), sg["nodes"].index(v)
                for k, s in sc.items():
                    rows.append({"region": sg["region"], "d": sg["d"], "seed": sg["seed"], "u": u, "v": v, "res": rn,
                                 "std": std, "ctrl": k, "acc": pair_orient(s, iu, iv)})
G = pd.DataFrame(rows)
m = e[["region", "d", "seed", "u", "v", "c_6h", "c_12h", "c_24h"]]
G = G.merge(m, on=["region", "d", "seed", "u", "v"])
G["contemp"] = [bool(r["c_" + r["res"]]) for _, r in G.iterrows()]
G.to_csv("g3_rows.csv", index=False)
P("| control | data | " + " | ".join(f"{rn} contemp (n)" for rn in RES) + " | " + " | ".join(f"{rn} lagged" for rn in RES) + " |")
P("|---|---|---|---|---|---|---|---|")
flag = False
for ctrl in ["var", "R2", "mean(RP)"]:
    for std in (True, False):
        cells = []; cl = []
        for rn in RES:
            g = G[(G.ctrl == ctrl) & (G["std"] == std) & (G.res == rn)]
            gc = g[g.contemp]; gl = g[~g.contemp]
            cells.append(f"{gc.acc.mean():.3f} ({len(gc)})"); cl.append(f"{gl.acc.mean():.3f}")
            if std and ctrl in ("var", "R2") and gc.acc.mean() >= 0.8: flag = True
        P(f"| {ctrl} | {'standardised' if std else 'raw'} | " + " | ".join(cells) + " | " + " | ".join(cl) + " |")
P(f"\n**G3: {'FLAG - scale-only control reaches >= 0.8 on standardised data' if flag else 'no leakage flag: neither var- nor R2-sortnregress reaches 0.8 on standardised contemporaneous edges at any resolution'}.**")
P("Raw-scale rows show how much of the orientation is available from magnitude alone.\n")

# ---------------- sibling residual diagnostic
P("## Diagnostic: residual correlation among sibling stations\n")
P("VAR(1) (OLS, standardised) residual correlations, averaged over node pairs in the sampled subgraphs. Siblings: two stations that are both upstream parents of the same downstream station. Adjacent: true edge u->v. A high sibling value at coarse resolution points to shared forcing (rain) creating contemporaneous dependence without a direct edge.\n")
rr = []
for sg in sub:
    nodes = sg["nodes"]; d = len(nodes); idx = {n: i for i, n in enumerate(nodes)}
    par = {}
    for u, v in sg["edges"]: par.setdefault(v, []).append(u)
    sib = set()
    for v, ps in par.items():
        for a in ps:
            for b in ps:
                if a < b: sib.add((a, b))
    adj = {frozenset(x) for x in sg["edges"]}
    for rn in RES:
        X = get_series(sg, rn)
        Z = X[1:]; Y = X[:-1]
        B, *_ = np.linalg.lstsq(np.c_[Y, np.ones(len(Y))], Z, rcond=None)
        R = Z - np.c_[Y, np.ones(len(Y))] @ B
        C = np.corrcoef(R.T)
        for i in range(d):
            for j in range(i + 1, d):
                a, b = nodes[i], nodes[j]
                t = "sibling" if (min(a, b), max(a, b)) in sib else ("adjacent" if frozenset((a, b)) in adj else "other")
                rr.append({"res": rn, "type": t, "r": C[i, j]})
R = pd.DataFrame(rr)
P("| resolution | sibling mean r (n) | adjacent mean r (n) | other mean r (n) |")
P("|---|---|---|---|")
for rn in RES:
    g = R[R.res == rn]; s = lambda t: f"{g[g.type == t].r.mean():.3f} ({(g.type == t).sum()})"
    P(f"| {rn} | {s('sibling')} | {s('adjacent')} | {s('other')} |")
open("gates.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
