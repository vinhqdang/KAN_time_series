import numpy as np, pandas as pd, json
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from lib import *

rng = np.random.default_rng(0); NB = 2000
R = pd.read_csv("results_test.csv"); E = pd.read_csv("edgeres_test.csv"); T = pd.read_csv("edges.csv")
E = E.merge(T[["region", "d", "seed", "u", "v", "tau_e"]], on=["region", "d", "seed", "u", "v"], how="left")
for f, nm in [(0.5, "c"), (0.25, "c4"), (1.0, "c1")]:
    E[nm] = [t < f * RES[r] for t, r in zip(E.tau_e, E.res)]
sgk = lambda df: df.region + "|" + df.d.astype(str) + "|" + df.seed.astype(str)
R["sg"] = sgk(R); E["sg"] = sgk(E)
CAP = ["pcmci+", "varlingam", "dynotears"]; REF = ["notears_svar2"]; LAG = ["var", "corr"]
CTRL = ["var_sortnregress", "r2_sortnregress", "random"]
out = []; P = lambda s="": out.append(s)

def boot_mean(v):
    v = np.asarray(v); b = v[rng.integers(0, len(v), (NB, len(v)))].mean(1)
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)

def orient(E_sub, byname=True):
    """pooled orientation accuracy with cluster bootstrap over subgraphs. returns acc, lo, hi, se, n_edges, n_subgraphs"""
    g = E_sub.groupby("sg").orient.agg(["sum", "count"])
    if len(g) == 0: return (np.nan,) * 4 + (0, 0)
    s, c = g["sum"].values, g["count"].values
    idx = rng.integers(0, len(g), (NB, len(g)))
    b = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), np.percentile(b, 2.5), np.percentile(b, 97.5), b.std(), int(c.sum()), len(g)

def sel(method, res, p=1, scale="std"):
    return (R.method == method) & (R.res == res) & (R.scale == scale) & (R.cfg.apply(lambda c: json.loads(c).get("p", 1)) == p)
def esel(method, res, p=1, scale="std"):
    return (E.method == method) & (E.res == res) & (E.scale == scale) & (E.cfg.apply(lambda c: json.loads(c).get("p", 1)) == p)

def auc_by_sg(method, res, p=1, scale="std", col="auroc"):
    return R[sel(method, res, p, scale)].set_index("sg")[col].sort_index()

def by_def(method, e):      # lag-only methods: contemporaneous edges scored 0.5 by definition
    e = e.copy()
    if method in LAG: e["orient"] = 0.5
    return e

res_all = {}
nsg = R[sel("var", "24h")].sg.nunique()
P("# Stage 0 results (test seeds 5005..5029)\n")
P(f"Test subgraphs: {nsg} (25 per size and region). Primary analysis: standardised data, lag order p=1, label rule tau_e < Delta/2. Bootstrap: 2000 resamples of subgraphs, percentile 95% CI. Orientation accuracy is on true contemporaneous edges given the true skeleton (chance 0.5); VAR and correlation are 0.5 by definition there (coefficient-based orientation in the secondary table).\n")

# ---- primary table
ALL = LAG + CAP + REF + CTRL
tab = []
for rn in RES:
    for m in ALL:
        a = auc_by_sg(m, rn); ap = auc_by_sg(m, rn, col="auprc")
        mu, lo, hi = boot_mean(a.values); mp, plo, phi = boot_mean(ap.values)
        e = by_def(m, E[esel(m, rn) & E.c])
        oa = orient(e)
        res_all[(rn, m)] = dict(auc=mu, lo=lo, hi=hi, acc=oa[0], alo=oa[1], ahi=oa[2], ase=oa[3], ne=oa[4], nsg=oa[5])
        tab.append(f"| {rn} | {m} | {mu:.3f} [{lo:.3f}, {hi:.3f}] | {mp:.3f} [{plo:.3f}, {phi:.3f}] | {oa[0]:.3f} [{oa[1]:.3f}, {oa[2]:.3f}] | {oa[3]:.3f} | {oa[4]} / {oa[5]} |")
P("## Primary table\n")
P("| res | method | directed AUROC [95% CI] | AUPRC [95% CI] | orientation acc, contemporaneous [95% CI] | SE | edges / subgraphs |")
P("|---|---|---|---|---|---|---|")
out += tab; P()

# ---- paired comparisons vs VAR
P("## Paired comparison of directed AUROC against lag-only VAR (paired Wilcoxon, two-sided; Holm over all pairs below)\n")
pairs = []
for rn in RES:
    base = auc_by_sg("var", rn)
    for m in CAP + REF:
        a = auc_by_sg(m, rn); dff = (a - base).values
        try: p = wilcoxon(dff).pvalue
        except Exception: p = np.nan
        pairs.append(dict(res=rn, method=m, diff=dff.mean(), p=p))
PD = pd.DataFrame(pairs)
vm = PD.method.isin(CAP)
PD["p_holm"] = np.nan
PD.loc[vm, "p_holm"] = multipletests(PD[vm].p.fillna(1), method="holm")[1]
PD["p_holm_incl_ref"] = multipletests(PD.p.fillna(1), method="holm")[1]
P("| res | method | mean AUROC diff vs VAR | raw p | Holm p (3 capable methods x 3 res = 9) | Holm p (incl. reference, 12) |")
P("|---|---|---|---|---|---|")
for _, r in PD.iterrows():
    P(f"| {r.res} | {r.method} | {r['diff']:+.3f} | {r.p:.2g} | {'' if np.isnan(r.p_holm) else format(r.p_holm, '.2g')} | {r.p_holm_incl_ref:.2g} |")
P()

# ---- verdicts
def verdict(methods, holmcol):
    h1 = {}
    for rn in ("12h", "24h"):
        ok = []
        for m in methods:
            r = res_all[(rn, m)]; pr = PD[(PD.res == rn) & (PD.method == m)].iloc[0]
            c = (r["acc"] >= 0.60 and r["alo"] > 0.5 and pr["diff"] >= 0.03 and pr[holmcol] < 0.05)
            ok.append((m, bool(c)))
        h1[rn] = ok
    H1 = all(any(c for _, c in h1[rn]) for rn in h1)
    r24 = [res_all[("24h", m)] for m in methods]
    n_ok = min(r["nsg"] for r in r24) >= 30
    neg1 = n_ok and all(abs(r["acc"] - 0.5) < 2 * r["ase"] for r in r24)
    neg2 = all(res_all[("24h", "var")]["auc"] >= r["auc"] for r in r24)
    return H1, h1, neg1, neg2, n_ok
for name, methods, hc in [("verdict methods (PCMCI+, VAR-LiNGAM, DYNOTEARS)", CAP, "p_holm"),
                          ("sensitivity: reference NOTEARS-SVAR included", CAP + REF, "p_holm_incl_ref")]:
    H1, h1, n1, n2, n_ok = verdict(methods, hc)
    P(f"## Hypothesis verdicts - {name}\n")
    for rn in h1: P(f"- H1 condition at {rn}: " + ", ".join(f"{m}={'met' if c else 'not met'}" for m, c in h1[rn]))
    P(f"- **H1 (needs 12 h and 24 h): {'HOLDS' if H1 else 'does not hold'}**")
    P(f"- Negative (i), accuracy within 2 SE of 0.5 for every capable method at 24 h (>=30 subgraphs with contemporaneous edges: {n_ok}): **{'MET' if n1 else 'not met'}**")
    P(f"- Negative (ii), VAR directed AUROC >= every capable method at 24 h: **{'MET' if n2 else 'not met'}**")
    for rn in ("12h", "6h"):
        P(f"  - (ii) at {rn}: " + ("met" if all(res_all[(rn, 'var')]['auc'] >= res_all[(rn, m)]['auc'] for m in methods) else "not met"))
    verdict_txt = "H1 HOLDS" if H1 and not (n1 or n2) else ("NEGATIVE" if (n1 or n2) and not H1 else ("AMBIGUOUS (both H1 and a negative rule fire)" if H1 else "AMBIGUOUS"))
    P(f"- **Overall: {verdict_txt}**\n")

# ---- secondary tables
def sec(title, rows):
    P(f"## {title}\n"); out.extend(rows); P()
rows = ["| res | method | accuracy (coefficient/score-based, no by-definition 0.5) [95% CI] | accuracy on lagged edges [95% CI] |", "|---|---|---|---|"]
for rn in RES:
    for m in ALL:
        a = orient(E[esel(m, rn) & E.c]); b = orient(E[esel(m, rn) & ~E.c])
        rows.append(f"| {rn} | {m} | {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] | {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}] |")
sec("Secondary: raw orientation by scores (VAR and correlation not forced to 0.5) and orientation on lagged edges", rows)

rows = ["| res | method | acc Delta/4 rule (n) | acc Delta/2 primary (n) | acc Delta rule (n) |", "|---|---|---|---|---|"]
for rn in RES:
    for m in CAP + REF:
        cells = []
        for c in ("c4", "c", "c1"):
            a = orient(E[esel(m, rn) & E[c]]); cells.append(f"{a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] ({a[4]})")
        rows.append(f"| {rn} | {m} | " + " | ".join(cells) + " |")
sec("Sensitivity to the label rule", rows)

rows = ["| res | method | AUROC p=1 | AUROC p=2 | acc p=1 | acc p=2 |", "|---|---|---|---|---|---|"]
for rn in RES:
    for m in ["var"] + CAP + REF:
        a1 = auc_by_sg(m, rn, 1).mean(); a2 = auc_by_sg(m, rn, 2).mean()
        o1 = orient(by_def(m, E[esel(m, rn, 1) & E.c])); o2 = orient(by_def(m, E[esel(m, rn, 2) & E.c]))
        rows.append(f"| {rn} | {m} | {a1:.3f} | {a2:.3f} | {o1[0]:.3f} [{o1[1]:.3f}, {o1[2]:.3f}] | {o2[0]:.3f} [{o2[1]:.3f}, {o2[2]:.3f}] |")
sec("Sensitivity to lag order (p=2, same hyperparameters)", rows)

rows = ["| res | method | AUROC raw | acc contemp raw [95% CI] | AUROC std | acc contemp std |", "|---|---|---|---|---|---|"]
for rn in RES:
    for m in ["var", "corr"] + CAP + REF + ["var_sortnregress", "r2_sortnregress"]:
        a = auc_by_sg(m, rn, 1, "raw").mean(); b = auc_by_sg(m, rn, 1, "std").mean()
        o = orient(by_def(m, E[esel(m, rn, 1, "raw") & E.c])); s = orient(by_def(m, E[esel(m, rn, 1, "std") & E.c]))
        rows.append(f"| {rn} | {m} | {a:.3f} | {o[0]:.3f} [{o[1]:.3f}, {o[2]:.3f}] | {b:.3f} | {s[0]:.3f} |")
sec("Secondary analysis: raw scale (scale leakage)", rows)

rows = ["| res | region | d | method | AUROC | acc contemp (n) |", "|---|---|---|---|---|---|"]
for rn in RES:
    for (rg, d) in [(a, b) for a in REGIONS for b in (5, 10)]:
        for m in ["var"] + CAP:
            ids = R[sel(m, rn) & (R.region == rg) & (R.d == d)]
            o = orient(by_def(m, E[esel(m, rn) & E.c & (E.region == rg) & (E.d == d)]))
            rows.append(f"| {rn} | {rg} | {d} | {m} | {ids.auroc.mean():.3f} | {o[0]:.3f} ({o[4]}) |")
sec("Breakdown by region and size (primary analysis)", rows)
P("Noise floor reminder: random-score AUROC and orientation are shown in the primary table.")
open("results.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
