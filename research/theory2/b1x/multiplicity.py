"""B'2: one multiplicity statement over the confirmatory claims of B1, B1-NL, B3 (and B'1 if its analysis exists).

Each claim gets one claim-level p-value, i.e. the p at which that claim's own pre-registered decision rule would
still pass:
- "k of m families significant" claims: the k-th smallest Holm-adjusted p (intersection-union);
- "at least one" claims (H2b, H3): the smallest adjusted p over the qualifying cases, Bonferroni-multiplied by
  their number where the plan had no correction; a swap counts with max(Holm G1, Holm G2);
- tracking: the graph-cluster permutation p from reanalysis_b2 (floor 1/10,001);
- CI-based rules (H4): max of graph-level Wilcoxon p for parts (a) and (c); (b) is an equivalence statement and is
  reported, not tested.
Holm is then applied across all claims (family-wise alpha 0.05).
Run: /usr/bin/python3 multiplicity.py results
"""
import sys, os, json, itertools
import numpy as np
import pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); TH = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(TH, "b1"))
from common import FAMILIES, SC_FAMILIES


def holm(pv):
    pv = np.asarray(pv, float); k = len(pv); o = np.argsort(pv); adj = np.empty(k); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (k - r) * pv[i])); adj[i] = run
    return adj


def wil(x, alt="two-sided"):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2 or np.all(x == 0):
        return 1.0
    return float(stats.wilcoxon(x, zero_method="wilcox", alternative=alt).pvalue)


def swap_claim_p(df, methods, regime=None):
    if regime:
        df = df[df.regime == regime]
    key = ["family", "p", "seed", "T"]
    a = df[df.gen == "G1"].set_index(key).sort_index(); b = df[df.gen == "G2"].set_index(key).loc[a.index]
    tests = []
    for fam in FAMILIES:
        for p in (5, 10):
            for T in (3, 5, 10):
                ms = [m for m in methods if f"{m}_shd" in df.columns and a.loc[(fam, p, slice(None), T), f"{m}_shd"].notna().all()]
                for m1, m2 in itertools.combinations(ms, 2):
                    for g, F in (("G1", a), ("G2", b)):
                        x = (F.loc[(fam, p, slice(None), T), f"{m1}_shd"] - F.loc[(fam, p, slice(None), T), f"{m2}_shd"]).values
                        tests.append(dict(k=(fam, p, T, m1, m2), g=g, mean=float(np.mean(x)), p=wil(x)))
    TT = pd.DataFrame(tests); TT["ph"] = holm(TT.p.values)
    best = 1.0
    for k, grp in TT.groupby("k"):
        if len(grp) == 2 and np.sign(grp["mean"].iloc[0]) * np.sign(grp["mean"].iloc[1]) < 0:
            best = min(best, float(grp.ph.max()))
    return best, len(TT)


def main():
    out = sys.argv[1]
    claims = []
    b1 = json.load(open(os.path.join(TH, "b1", "results", "analysis.json")))
    nl = json.load(open(os.path.join(TH, "b1nl", "results", "analysis.json")))
    b2 = json.load(open(os.path.join(out, "b2_reanalysis.json")))
    b3 = json.load(open(os.path.join(TH, "b3", "results", "results_b3.json")))
    # ---- B1
    ph = sorted(v["p_holm"] for v in b1["H1"]["families"].values())
    claims.append(("B1", "H1 start helps rel-SnR (>= 4/5 non-SC families)", ph[3]))
    claims.append(("B1", "H1 tracking (graph-cluster permutation)", b2["B1_tracking"]["p_cluster_perm"]))
    d = pd.read_csv(os.path.join(TH, "b1", "results", "b1_test.csv.gz"))
    h = d[d.regime == "het"]; key = ["family", "p", "seed", "T"]
    a = h[h.gen == "G1"].set_index(key).sort_index(); b = h[h.gen == "G2"].set_index(key).loc[a.index]
    ps = []
    for fam in SC_FAMILIES:
        for p in (5, 10):
            for T in (3, 5, 10):
                if T < p:
                    x = (a.loc[(fam, p, slice(None), T), "M1_oa"] - b.loc[(fam, p, slice(None), T), "M1_oa"]).values
                    ps.append(wil(x, "greater"))
    claims.append(("B1", "H2b SC gain at T < p (some cell)", min(1.0, min(ps) * len(ps))))
    claims.append(("B1", "H2c symER gain decreases with T", b1["H2"]["H2c_p"]))
    p3, n3 = swap_claim_p(h, ["M1", "M2", "M3", "M4", "M5", "M6"])
    claims.append(("B1", f"H3 >= 1 significant SHD swap (Holm over {n3})", p3))
    # ---- B1-NL
    for cfg in ("NL", "SGN", "SGNU"):
        ph = sorted(v["p_holm"] for v in nl[cfg]["H1"]["families"].values())
        claims.append(("B1-NL", f"H1-{cfg} (>= 4/5 families)", ph[3]))
        claims.append(("B1-NL", f"H1-{cfg} tracking (graph-cluster permutation)", b2[f"NL_{cfg}_tracking"]["p_cluster_perm"]))
    dn = pd.read_csv(os.path.join(TH, "b1nl", "results", "b1nl_test.csv.gz"))
    for cfg in ("NL", "SGN"):
        p3, n3 = swap_claim_p(dn[dn.cfg == cfg], ["M1", "M2", "M3", "M4", "M5"])
        claims.append(("B1-NL", f"H3-{cfg} >= 1 significant SHD swap (Holm over {n3})", p3))
    for cfg in ("NL", "SGN", "SGNU"):
        x = dn[dn.cfg == cfg]
        a = x[x.gen == "G1"].set_index(key).sort_index(); b = x[x.gen == "G2"].set_index(key).loc[a.index]
        gl = lambda s: s.groupby(level=["family", "p", "seed"]).mean()
        pa = wil(gl(a.M3_oa - b.M3_oa).values); pc = wil(gl(a.M1_oa - b.M1_oa).values)
        claims.append(("B1-NL", f"H4-{cfg} var/rel split (parts a, c)", max(pa, pc)))
    # ---- B3
    claims.append(("B3", "H1 theorem beats tie rules on D* (ar_1)", b3["H1_ar_1"]["primary_Dstar"]["tests"]["const_half=wc_rule"]["p_holm"]))
    t2 = b3["H2_scm_5_collider"]["primary_Dstar"]["tests"]
    claims.append(("B3", "H2 collider vs const-1/2 and sensor indicator", max(v["p_holm"] for v in t2.values() if v.get("testable"))))
    # ---- B'1 (if analysed)
    fx = os.path.join(out, "analysis.json")
    if os.path.exists(fx):
        bx = json.load(open(fx))
        r1 = pd.DataFrame(bx["HX1"]["rows"])
        q = r1[r1.v.isin(["DAGMA_raw", "NOTEARS_raw", "GOLEMEV_raw"]) & r1.k.isin(["aoa", "shdn"]) & r1.favourable]
        claims.append(("B'1", "HX1 start moves a raw CO method (favourable, AOA or SHD)", float(q.p_holm.min()) if len(q) else 1.0))
        t = pd.DataFrame(bx["HX2"]["shdn"]["tests"])
        best = 1.0
        for (v, f), g in t.groupby(["v", "fam"]):
            if len(g) == 2 and np.sign(g["mean"].iloc[0]) != np.sign(g["mean"].iloc[1]):
                best = min(best, float(g.p_holm.max()))
        claims.append(("B'1", "HX2 rank change vs FLOP on SHD/|E|", best))
        h3 = bx["HX3"]
        claims.append(("B'1", "HX3 standardisation reduces a CO gain (any method, AOA or SHD)",
                       min([v["p_holm"] for v in h3.values() if v["verdict"] in ("removed", "reduced, not removed")] or [1.0])))
    C = pd.DataFrame(claims, columns=["exp", "claim", "p_claim"])
    C["p_holm_all"] = holm(C.p_claim.values)
    C["survives_FWER_0.05"] = C.p_holm_all < 0.05
    lines = [f"Multiplicity across {len(C)} confirmatory claims (Holm, family-wise alpha 0.05). "
             "Floors: permutation p >= 1e-4 (10,000 perms); B3 sign-flip p >= 1e-5 (Monte Carlo)."]
    for _, r in C.iterrows():
        lines.append(f"  {r.exp:6s} {r.claim:62s} p_claim {r.p_claim:.2e}  Holm {r.p_holm_all:.2e}  "
                     f"{'SURVIVES' if r['survives_FWER_0.05'] else 'does not survive'}")
    print("\n".join(lines))
    open(os.path.join(out, "multiplicity.txt"), "w").write("\n".join(lines) + "\n")
    C.to_csv(os.path.join(out, "multiplicity.csv"), index=False)


if __name__ == "__main__":
    main()
