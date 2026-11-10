"""b1y analysis (decision rules of PLAN.md, fixed before any run). Unit of analysis everywhere: the summary graph
(family, p, seed); the horizons T of one graph are averaged first. Bootstrap: graph-cluster percentile, 10,000
resamples, RNG seed 0 (a fresh default_rng(0) per call). One-sided bootstrap p-values have a floor of 1/10,001.

Usage:
  python analyse_y.py --selftest                 # runs on synthetic dummy results (injected truths); no study data
  python analyse_y.py q1 results/q1.csv.gz       # Q1 verdicts (schema below)
  python analyse_y.py q2 results/q2.csv.gz       # Q2 verdicts
Q1 schema (long): study{main,ndose,rhodose}, family, p, seed, T, n, cap, arm, oa_x1, oa_full, shd_x1n, gap
Q2 schema (long): family, p, seed, T, arm{G1,G2,VM_G1,VM_G2}, method{FLOP,DAGMA,NOTEARS,VarSnR,RelSnR,EMPTY}, shd_x1n, oa_x1
"""
import sys
import numpy as np
import pandas as pd

BOOT = 10000
NONSC = ["chain", "outtree", "intree", "ER1", "ER2"]
SC = ["symER", "ring"]

# ----------------------------------------------------------------------------------------------- thresholds (fixed)
THR = dict(
    negctrl_abs=0.01,                 # |OA(G1rep) - OA(G1)| must stay below this, else STOP (design invalid)
    pattern_ret_hi=0.75, pattern_ret_hi_lo=0.50,   # V2: R >= .75 and CI lower >= .50  -> detectable strength, not population pattern
    pattern_ret_lo=0.35, pattern_ret_lo_hi=0.60,   # V2: R <= .35 and CI upper <= .60  -> population pattern
    dose_frac=0.25,                   # V3: |OA(c=4) - OA(c=.25)| >= 0.25 * (OA(G1) - OA(G2)) and CI excludes 0
    ndose_gain=0.10, ndose_level=0.60,             # V4: Q(25000) - Q(1000) >= .10 (CI > 0) and Q(25000) >= .60
    rho_gain=0.10,                    # V5: Q(cap .95) - Q(cap .5) >= .10 (CI > 0)
    pre_gap=0.10,                     # Q2 precondition: CO advantage gap A(G1) - A(G2) >= .10 with CI > 0
    vm_lo=0.40, vm_lo_hi=0.60,        # Q2 HY1: R_VM <= .40 and CI upper <= .60 -> variance channel
    vm_hi=0.75, vm_hi_lo=0.50,        # Q2 HY1: R_VM >= .75 and CI lower >= .50 -> not variance
    inj=0.60,                         # Q2 injection: R_inj >= .60 -> variance alone sufficient
)


def holm(pvals, alpha=0.05):
    """Holm step-down on a dict name -> p. Returns dict name -> (p_adj, reject)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); out = {}; run = 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p)); out[k] = (run, run <= alpha)
    return out


def _graph_table(df, value, by):
    """graph-level table: mean over T (and over duplicate rows) of `value`, columns = levels of `by`."""
    g = df.groupby(["family", "p", "seed"] + by)[value].mean().unstack(by)
    return g


def _boot_idx(n, rng):
    return rng.integers(0, n, size=(BOOT, n))


def boot_mean(x, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed); idx = _boot_idx(len(x), rng)
    b = x[idx].mean(axis=1)
    return x.mean(), np.percentile(b, [2.5, 97.5]), b


def boot_ratio(num, den, seed=0):
    """Ratio of means with graph-cluster bootstrap; num and den are paired per graph."""
    num = np.asarray(num, float); den = np.asarray(den, float)
    rng = np.random.default_rng(seed); idx = _boot_idx(len(num), rng)
    d = den[idx].mean(axis=1); b = num[idx].mean(axis=1) / np.where(np.abs(d) < 1e-12, np.nan, d)
    return num.mean() / den.mean(), np.nanpercentile(b, [2.5, 97.5]), b


def one_sided_p(b, sign=+1):
    """P(bootstrap statistic on the wrong side of 0) with the +1 correction."""
    return (1 + np.sum(b <= 0 if sign > 0 else b >= 0)) / (BOOT + 1)


# ----------------------------------------------------------------------------------------------- Q1
def q1_verdicts(df, families=NONSC):
    out = {}
    m = df[(df.study == "main") & df.family.isin(families)]
    g = _graph_table(m, "oa_x1", ["arm"])
    gain = (g["G1"] - g["G2"]).values
    # V1 negative control
    d = (g["G1rep"] - g["G1"]).values; mu, ci, _ = boot_mean(d)
    out["V1_negative_control"] = dict(mean=mu, ci=list(ci), valid=bool(abs(mu) < THR["negctrl_abs"]))
    if not out["V1_negative_control"]["valid"]:
        out["STOP"] = "negative control outside +-0.01: design invalid, no further interpretation"
        return out
    gmu, gci, gb = boot_mean(gain)
    out["gain_G1_minus_G2"] = dict(mean=gmu, ci=list(gci))
    # V2 pattern at undetectable strength
    R, ci, _ = boot_ratio((g["CF2_m1"] - g["G2"]).values, gain)
    if R >= THR["pattern_ret_hi"] and ci[0] >= THR["pattern_ret_hi_lo"]:
        v = "detectable strength, not population pattern"
    elif R <= THR["pattern_ret_lo"] and ci[1] <= THR["pattern_ret_lo_hi"]:
        v = "population pattern"
    else:
        v = "mixed"
    out["V2_pattern_retention_CF2_m1"] = dict(R=R, ci=list(ci), verdict=v)
    for arm in ("CF2_m3", "CF2_m10"):
        R, ci, _ = boot_ratio((g[arm] - g["G2"]).values, gain)
        out[f"retention_{arm}"] = dict(R=R, ci=list(ci))
    # V3 strength dose at fixed G1 pattern
    dd = (g["CF1_4"] - g["CF1_0.25"]).values; mu, ci, b = boot_mean(dd)
    ok = (ci[0] > 0 or ci[1] < 0) and abs(mu) >= THR["dose_frac"] * gmu
    out["V3_dose_fixed_pattern"] = dict(delta=mu, ci=list(ci), p_one_sided=one_sided_p(b, +1 if mu > 0 else -1),
                                         verdict="strength matters at fixed population pattern" if ok else "flat")
    # slope of gain on exact gap with family x p fixed effects (partial slope), graph-cluster bootstrap
    gap = m.groupby(["family", "p", "seed"])["gap"].mean().reindex(g.index).values
    fe = pd.Series(["%s_%d" % (f, p) for f, p, s in g.index], index=g.index)
    def partial_slope(ix):
        x = gap[ix]; y = gain[ix]; f = fe.values[ix]
        xs = x.copy(); ys = y.copy()
        for lev in np.unique(f):
            k = f == lev; xs[k] -= x[k].mean(); ys[k] -= y[k].mean()
        den = (xs ** 2).sum(); return (xs * ys).sum() / den if den > 0 else np.nan
    rng = np.random.default_rng(0); n = len(gain)
    sl = partial_slope(np.arange(n)); bs = [partial_slope(rng.integers(0, n, n)) for _ in range(2000)]
    out["slope_gain_on_exact_gap_within_family"] = dict(slope=sl, ci=list(np.nanpercentile(bs, [2.5, 97.5])), boots=2000)
    return out


def q1_dose(df):
    """V4 (n-dose) and V5 (rho-dose): Q = mean OA gain (G1 - G2) / mean exact gap, per level."""
    out = {}
    nd = df[df.study == "ndose"]
    if len(nd):
        g = _graph_table(nd, "oa_x1", ["n", "arm"]); gp = nd.groupby(["family", "p", "seed"])["gap"].mean().reindex(g.index)
        gain = {n: (g[(n, "G1")] - g[(n, "G2s")]).values for n in sorted(nd.n.unique())}
        q = {n: gain[n] / gp.values for n in gain}
        lo, hi = min(gain), max(gain)
        base = 1000 if 1000 in gain else lo
        diff = q[hi] - q[base]; mu, ci, b = boot_mean(diff); lev = boot_mean(q[hi])
        ok = ci[0] > 0 and mu >= THR["ndose_gain"] and lev[0] >= THR["ndose_level"]
        out["V4_n_dose"] = dict(Q={int(n): float(np.mean(v)) for n, v in q.items()}, delta=mu, ci=list(ci),
                                p_one_sided=one_sided_p(b), verdict="converges toward the exact gap" if ok else "not supported")
    rd = df[df.study == "rhodose"]
    if len(rd):
        g = _graph_table(rd, "oa_x1", ["cap", "arm"]); gp = rd.groupby(["family", "p", "seed"])["gap"].mean().reindex(g.index)
        q = {c: (g[(c, "G1")] - g[(c, "G2s")]).values / gp.values for c in sorted(rd.cap.unique())}
        diff = q[max(q)] - q[min(q)]; mu, ci, b = boot_mean(diff)
        ok = ci[0] > 0 and mu >= THR["rho_gain"]
        out["V5_rho_dose"] = dict(Q={float(c): float(np.mean(v)) for c, v in q.items()}, delta=mu, ci=list(ci),
                                  p_one_sided=one_sided_p(b), verdict="correlation strength raises the realised share" if ok else "not supported")
    return out


# ----------------------------------------------------------------------------------------------- Q2
def q2_verdicts(df, co=("DAGMA", "NOTEARS"), families=("intree",)):
    """A(arm) = SHD_x1n(FLOP) - SHD_x1n(CO) (positive = CO better) per graph."""
    out = {}
    for method in co:
        d = df[df.family.isin(families)]
        g = _graph_table(d[d.method.isin([method, "FLOP"])].assign(key=lambda x: x.method + "|" + x.arm), "shd_x1n", ["key"])
        if f"{method}|G1" not in g.columns: continue
        A = {arm: (g[f"FLOP|{arm}"] - g[f"{method}|{arm}"]).values for arm in ("G1", "G2", "VM_G1", "VM_G2") if f"{method}|{arm}" in g.columns}
        gap = A["G1"] - A["G2"]; mu, ci, _ = boot_mean(gap)
        res = dict(A={k: float(v.mean()) for k, v in A.items()}, gap=mu, gap_ci=list(ci))
        res["precondition_met"] = bool(mu >= THR["pre_gap"] and ci[0] > 0)
        if res["precondition_met"]:
            R, rci, _ = boot_ratio(A["VM_G1"] - A["G2"], gap)
            v = ("variance channel" if (R <= THR["vm_lo"] and rci[1] <= THR["vm_lo_hi"]) else
                 "not variance" if (R >= THR["vm_hi"] and rci[0] >= THR["vm_hi_lo"]) else "mixed")
            res["HY1_R_VM"] = dict(R=R, ci=list(rci), verdict=v)
            Ri, ici, _ = boot_ratio(A["VM_G2"] - A["G2"], gap)
            res["HY1_R_inj"] = dict(R=Ri, ci=list(ici), variance_alone_sufficient=bool(Ri >= THR["inj"]))
        else:
            res["note"] = "reversal not reproduced at tuned hyper-parameters: HY1 not evaluated"
        out[method] = res
    return out


# ----------------------------------------------------------------------------------------------- selftest on dummy data
def _dummy_q1(truth, seed=1):
    """truth: 'strength' (CF2 keeps the gain, dose matters) or 'pattern' (CF2 loses the gain, dose flat)."""
    rng = np.random.default_rng(seed); rows = []
    for fam in NONSC:
        for p in (5, 10):
            for s in range(30):
                gap = 0.2 + 0.2 * rng.random(); gl = 0.5 * gap + 0.02 * rng.standard_normal(); base = 0.55 + 0.05 * rng.random()
                for T in (3, 5):
                    oa = {"G2": base, "G1": base + gl, "G1rep": base + gl}
                    if truth == "strength":
                        oa.update({"CF2_m1": base + gl * 0.98, "CF2_m3": base + gl * 0.9, "CF2_m10": base + gl * 0.6,
                                   "CF1_0.25": base + gl * 0.7, "CF1_4": base + gl * 1.2})
                    else:
                        oa.update({"CF2_m1": base + gl * 0.1, "CF2_m3": base + gl * 0.1, "CF2_m10": base + gl * 0.1,
                                   "CF1_0.25": base + gl * 1.0, "CF1_4": base + gl * 1.0})
                    for arm, v in oa.items():
                        rows.append(dict(study="main", family=fam, p=p, seed=s, T=T, n=1000, cap=0.8, arm=arm,
                                         oa_x1=v + 0.01 * rng.standard_normal(), gap=gap))
    return pd.DataFrame(rows)


def _dummy_dose(seed=2, converge=True):
    rng = np.random.default_rng(seed); rows = []
    for fam in ("ring", "chain", "ER2"):
        for p in (5, 10):
            for s in range(20):
                gap = 0.4 + 0.1 * rng.random()
                for n, q in ((200, .4), (1000, .6), (5000, .8), (25000, .95)):
                    q = q if converge else .6
                    for arm, v in (("G1", .6 + q * gap), ("G2s", .6)):
                        rows.append(dict(study="ndose", family=fam, p=p, seed=s, T=3, n=n, cap=0.8, arm=arm, oa_x1=v + 0.01 * rng.standard_normal(), gap=gap))
                for cap, q in ((0.5, .3), (0.8, .6), (0.95, .85)):
                    for arm, v in (("G1", .6 + q * gap), ("G2s", .6)):
                        rows.append(dict(study="rhodose", family=fam, p=p, seed=s, T=3, n=1000, cap=cap, arm=arm, oa_x1=v + 0.01 * rng.standard_normal(), gap=gap))
    return pd.DataFrame(rows)


def _dummy_q2(truth, seed=3):
    rng = np.random.default_rng(seed); rows = []
    for s in range(24):
        for T in (3, 5):
            flop = 0.5 + 0.03 * rng.standard_normal()
            A = {"G1": 0.3, "G2": -0.3, "VM_G1": -0.25 if truth == "variance" else 0.25, "VM_G2": 0.25 if truth == "variance" else -0.25}
            for arm, a in A.items():
                rows.append(dict(family="intree", p=10, seed=s, T=T, arm=arm, method="FLOP", shd_x1n=flop))
                rows.append(dict(family="intree", p=10, seed=s, T=T, arm=arm, method="DAGMA", shd_x1n=flop - a + 0.03 * rng.standard_normal()))
    return pd.DataFrame(rows)


def selftest():
    for truth in ("strength", "pattern"):
        r = q1_verdicts(_dummy_q1(truth)); v2 = r["V2_pattern_retention_CF2_m1"]["verdict"]; v3 = r["V3_dose_fixed_pattern"]["verdict"]
        exp2 = "detectable strength, not population pattern" if truth == "strength" else "population pattern"
        exp3 = "strength matters at fixed population pattern" if truth == "strength" else "flat"
        assert v2 == exp2 and v3 == exp3, (truth, v2, v3)
        print("selftest Q1", truth, "->", v2, "|", v3, "| slope", round(r["slope_gain_on_exact_gap_within_family"]["slope"], 2))
    bad = _dummy_q1("strength"); bad.loc[(bad.arm == "G1rep"), "oa_x1"] += 0.03
    assert "STOP" in q1_verdicts(bad); print("selftest Q1 negative-control stop rule fires")
    d = q1_dose(_dummy_dose(converge=True)); assert d["V4_n_dose"]["verdict"].startswith("converges") and d["V5_rho_dose"]["verdict"].startswith("correlation")
    d2 = q1_dose(_dummy_dose(converge=False)); assert d2["V4_n_dose"]["verdict"] == "not supported"
    print("selftest Q1 dose verdicts", d["V4_n_dose"]["verdict"], "|", d2["V4_n_dose"]["verdict"])
    for truth, exp in (("variance", "variance channel"), ("other", "not variance")):
        r = q2_verdicts(_dummy_q2(truth))["DAGMA"]
        assert r["precondition_met"] and r["HY1_R_VM"]["verdict"] == exp, (truth, r)
        print("selftest Q2", truth, "->", r["HY1_R_VM"]["verdict"], "inj sufficient:", r["HY1_R_inj"]["variance_alone_sufficient"])
    h = holm({"a": 0.001, "b": 0.02, "c": 0.04}); assert abs(h["a"][0] - 0.003) < 1e-12 and abs(h["b"][0] - 0.04) < 1e-12 and h["c"][1]
    print("holm", {k: (round(v[0], 3), v[1]) for k, v in h.items()})
    print("selftest passed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) > 2 and sys.argv[1] == "q1":
        d = pd.read_csv(sys.argv[2]); print(q1_verdicts(d)); print(q1_dose(d))
    elif len(sys.argv) > 2 and sys.argv[1] == "q2":
        print(q2_verdicts(pd.read_csv(sys.argv[2])))
    else:
        print(__doc__)
