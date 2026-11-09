"""B'2: post hoc descriptive re-analysis of the committed B1 and B1-NL data (no new data).

Graph-cluster bootstrap CIs (unit = summary graph, i.e. (family, p, seed); all horizons of a graph resampled
together), unit-level and graph-cluster-permutation tracking, interaction effects (G1 - G2 change of paired
method differences) instead of swap counting, empty-graph reference, SHD/|E|, order accuracy without edges leaving
the first window slice.

Run: OMP_NUM_THREADS=1 /usr/bin/python3 reanalysis_b2.py results
(needs results/orders_x1.csv.gz from reanalysis_orders.py and results/empty_sid.csv.gz, made here if missing)
"""
import sys, os, json, itertools
import numpy as np
import pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "b1")); sys.path.insert(0, os.path.join(HERE, "..", "b1nl"))
from common import FAMILIES, SC_FAMILIES, NONSC_FAMILIES, summary_graph, weights, true_dag, sid

BOOT = 10000
B1CSV = os.path.join(HERE, "..", "b1", "results", "b1_test.csv.gz")
NLCSV = os.path.join(HERE, "..", "b1nl", "results", "b1nl_test.csv.gz")
MNAME = dict(M1="rel-SnR", M2="R2-SnR", M3="Var-SnR", M4="Random-SnR", M5="FLOP", M6="PC", R0="Time-SnR",
             std_M3="Var-SnR(std)", EMPTY="empty graph")
KEY = ["family", "p", "seed", "T"]
out_lines = []; OUT = {}


def P(*a):
    s = " ".join(str(x) for x in a); out_lines.append(s); print(s, flush=True)


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


def cluster_boot(vals, clusters, seed=0, B=BOOT):
    """Mean of vals with a percentile CI from resampling clusters (all members of a cluster together).
    Also returns the naive (data-set) bootstrap CI for comparison."""
    v = np.asarray(vals, float); ok = ~np.isnan(v); v = v[ok]; c = np.asarray(clusters)[ok]
    u, inv = np.unique(c, return_inverse=True)
    sums = np.bincount(inv, weights=v); cnts = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=B).astype(float)
    m = (W @ sums) / (W @ cnts)
    Wn = rng.multinomial(len(v), np.full(len(v), 1 / len(v)), size=B).astype(float)
    mn = Wn @ v / len(v)
    return dict(mean=float(v.mean()), ci=[float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
                ci_naive=[float(np.percentile(mn, 2.5)), float(np.percentile(mn, 97.5))], n=int(len(v)),
                n_clusters=int(len(u)))


def fmt(r, k=3):
    return f"{r['mean']:+.{k}f} [{r['ci'][0]:+.{k}f}, {r['ci'][1]:+.{k}f}]"


def paired(df, regime=None):
    if regime is not None:
        df = df[df.regime == regime]
    a = df[df.gen == "G1"].set_index(KEY).sort_index()
    b = df[df.gen == "G2"].set_index(KEY).loc[a.index]
    return a, b


def gid(idx):
    """graph id (cluster) from a (family, p, seed, T) index."""
    return np.array([f"{f}|{p}|{s}" for f, p, s, _ in idx])


# ----------------------------------------------------------------------------- 1. tab:b1 with cluster CIs
def table_b1(a, b, label):
    P(f"\n== 1. {label}: table numbers with graph-cluster bootstrap 95% CIs (unit = summary graph; "
      f"10,000 resamples). Naive data-set CI in braces for the Delta column. ==")
    res = {}
    for fam in FAMILIES:
        A = a[a.index.get_level_values(0) == fam]; Bq = b[b.index.get_level_values(0) == fam]
        g = gid(A.index)
        r = dict(s_G1=cluster_boot(A.s_exact_G1, g), s_G2=cluster_boot(Bq.s_exact_stat, g),
                 relOA_G1=cluster_boot(A.M1_oa, g), relOA_G2=cluster_boot(Bq.M1_oa, g),
                 dOA=cluster_boot(A.M1_oa.values - Bq.M1_oa.values, g),
                 varOA_G1=cluster_boot(A.M3_oa, g), varOA_G2=cluster_boot(Bq.M3_oa, g),
                 dvarOA=cluster_boot(A.M3_oa.values - Bq.M3_oa.values, g),
                 dgap=cluster_boot(A.s_exact_G1.values - Bq.s_exact_stat.values, g))
        res[fam] = r
        P(f"  {fam:7s} s {r['s_G1']['mean']:.3f}[{r['s_G1']['ci'][0]:.3f},{r['s_G1']['ci'][1]:.3f}] / "
          f"{r['s_G2']['mean']:.3f}[{r['s_G2']['ci'][0]:.3f},{r['s_G2']['ci'][1]:.3f}] | relOA "
          f"{r['relOA_G1']['mean']:.3f}[{r['relOA_G1']['ci'][0]:.3f},{r['relOA_G1']['ci'][1]:.3f}] / "
          f"{r['relOA_G2']['mean']:.3f}[{r['relOA_G2']['ci'][0]:.3f},{r['relOA_G2']['ci'][1]:.3f}] | dOA {fmt(r['dOA'])} "
          f"{{{r['dOA']['ci_naive'][0]:+.3f},{r['dOA']['ci_naive'][1]:+.3f}}} | varOA "
          f"{r['varOA_G1']['mean']:.3f}[{r['varOA_G1']['ci'][0]:.3f},{r['varOA_G1']['ci'][1]:.3f}] / "
          f"{r['varOA_G2']['mean']:.3f}[{r['varOA_G2']['ci'][0]:.3f},{r['varOA_G2']['ci'][1]:.3f}] | dvarOA {fmt(r['dvarOA'])}"
          f" | clusters {r['dOA']['n_clusters']}")
    A = a[a.index.get_level_values(0).isin(NONSC_FAMILIES)]; Bq = b.loc[A.index]
    res["pooled_nonSC_dOA"] = cluster_boot(A.M1_oa.values - Bq.M1_oa.values, gid(A.index))
    P(f"  pooled non-SC dOA {fmt(res['pooled_nonSC_dOA'])} over {res['pooled_nonSC_dOA']['n_clusters']} graphs")
    return res


# ----------------------------------------------------------------------------- 2. tracking
def tracking(a, b, label, nperm=10000):
    P(f"\n== 2. {label}: tracking of the rel-SnR OA gain by the exact gap ==")
    D = pd.DataFrame(dict(d=a.M1_oa.values - b.M1_oa.values,
                          gap=a.s_exact_G1.values - b.s_exact_stat.values), index=a.index).reset_index()
    D["g"] = gid(a.index)
    C = D.groupby(["family", "p", "T"]).agg(d=("d", "mean"), gap=("gap", "mean")).reset_index()
    rc = stats.spearmanr(C.d, C.gap).statistic
    ru = stats.spearmanr(D.d, D.gap).statistic
    slope = np.polyfit(D.gap, D.d, 1)[0]
    # cluster bootstrap of cell-level and unit-level Spearman
    graphs = D.g.unique(); gi = {g: k for k, g in enumerate(graphs)}
    D["gi"] = D.g.map(gi)
    rows_of = [np.flatnonzero(D.gi.values == k) for k in range(len(graphs))]
    fam_of = np.array([g.split("|")[0] + "|" + g.split("|")[1] for g in graphs])
    rng = np.random.default_rng(0)
    bc, bu, bs = [], [], []
    for _ in range(2000):
        pick = []
        for fp in np.unique(fam_of):          # stratified: resample graphs within each (family, p)
            pool = np.flatnonzero(fam_of == fp)
            pick.append(rng.choice(pool, len(pool)))
        idx = np.concatenate([rows_of[k] for k in np.concatenate(pick)])
        Db = D.iloc[idx]
        Cb = Db.groupby(["family", "p", "T"]).agg(d=("d", "mean"), gap=("gap", "mean"))
        bc.append(stats.spearmanr(Cb.d, Cb.gap).statistic)
        bu.append(stats.spearmanr(Db.d, Db.gap).statistic)
        bs.append(np.polyfit(Db.gap, Db.d, 1)[0])
    # graph-cluster permutation: permute whole graphs' d-vectors (aligned by T) across graphs with the same set of T
    piv_d = D.pivot_table(index="g", columns="T", values="d").loc[graphs].values
    piv_g = D.pivot_table(index="g", columns="T", values="gap").loc[graphs].values
    rank_u = lambda x, y: stats.spearmanr(x.ravel(), y.ravel()).statistic
    obs = rank_u(piv_d, piv_g)
    cnt = 0
    for _ in range(nperm):
        pi = rng.permutation(len(graphs))
        cnt += rank_u(piv_d[pi], piv_g) >= obs
    p_perm = (cnt + 1) / (nperm + 1)
    # within-T permutation of graphs (removes the common T trend from the null): permute rows separately per T column
    cnt2 = 0
    for _ in range(nperm):
        pd_ = np.column_stack([piv_d[rng.permutation(len(graphs)), k] for k in range(piv_d.shape[1])])
        cnt2 += rank_u(pd_, piv_g) >= obs
    p_perm_T = (cnt2 + 1) / (nperm + 1)
    r = dict(spearman_cell=float(rc), spearman_cell_ci=[float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))],
             spearman_unit=float(ru), spearman_unit_ci=[float(np.percentile(bu, 2.5)), float(np.percentile(bu, 97.5))],
             slope=float(slope), slope_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
             p_cluster_perm=float(p_perm), p_cluster_perm_withinT=float(p_perm_T), n_units=len(D), n_graphs=len(graphs))
    P(f"  cell-level Spearman {rc:.3f} (graph-cluster bootstrap CI [{r['spearman_cell_ci'][0]:.3f}, {r['spearman_cell_ci'][1]:.3f}])")
    P(f"  unit-level Spearman {ru:.3f} over {len(D)} data-set pairs (CI [{r['spearman_unit_ci'][0]:.3f}, {r['spearman_unit_ci'][1]:.3f}]); "
      f"OLS slope {slope:.3f} [{r['slope_ci'][0]:.3f}, {r['slope_ci'][1]:.3f}]")
    P(f"  graph-cluster permutation p (whole graphs permuted, T aligned; {nperm} perms): {p_perm:.1e}; "
      f"graphs permuted within each T: {p_perm_T:.1e}")
    return r


# ----------------------------------------------------------------------------- 3. interaction effects
def interactions(a, b, label, methods, metrics=("shd", "sidn", "orient"), cells=None):
    P(f"\n== 3. {label}: interaction effects I = [m1 - m2](G1) - [m1 - m2](G2), per cell (unit = graph) and per "
      f"family (pooled over p, T; graph-cluster CI). Wilcoxon on per-graph I, Holm over all pair x cell tests of the metric ==")
    res = {}
    for metric in metrics:
        rows = []
        for fam in FAMILIES:
            for p in (5, 10):
                for T in (3, 5, 10):
                    sel = (a.index.get_level_values(0) == fam) & (a.index.get_level_values(1) == p) & \
                          (a.index.get_level_values(3) == T)
                    A = a[sel]; Bq = b[sel]
                    for m1, m2 in itertools.combinations(methods, 2):
                        c1, c2 = f"{m1}_{metric}", f"{m2}_{metric}"
                        if c1 not in a.columns or c2 not in a.columns or A[c1].isna().all() or A[c2].isna().all():
                            continue
                        I = (A[c1] - A[c2]).values - (Bq[c1] - Bq[c2]).values
                        x = I[~np.isnan(I)]
                        if len(x) < 5:
                            continue
                        rng = np.random.default_rng(0)
                        bm = rng.choice(x, (BOOT, len(x))).mean(1)
                        rows.append(dict(fam=fam, p=p, T=T, m1=m1, m2=m2, I=float(x.mean()),
                                         lo=float(np.percentile(bm, 2.5)), hi=float(np.percentile(bm, 97.5)),
                                         dG1=float(np.nanmean(A[c1] - A[c2])), dG2=float(np.nanmean(Bq[c1] - Bq[c2])),
                                         p_raw=wil(x), n=len(x)))
        R = pd.DataFrame(rows); R["p_holm"] = holm(R.p_raw.values)
        nci = int(((R.lo > 0) | (R.hi < 0)).sum()); nh = int((R.p_holm < 0.05).sum())
        P(f"  [{metric}] pair-cells {len(R)}; interaction CI excludes 0 in {nci}; Holm-significant {nh}")
        top = R[R.p_holm < 0.05].sort_values("p_holm")
        for _, r in top.head(25).iterrows():
            P(f"     {r.fam:7s} p={r.p:2d} T={r['T']:2d} {MNAME[r.m1]:>10s} - {MNAME[r.m2]:<10s} I = {r.I:+8.3f} "
              f"[{r.lo:+.3f}, {r.hi:+.3f}]  (G1 {r.dG1:+.3f}, G2 {r.dG2:+.3f}) Holm {r.p_holm:.1e}")
        # family-level pooled interaction with graph-cluster CI
        fam_rows = []
        for fam in FAMILIES:
            sel = a.index.get_level_values(0) == fam
            A = a[sel]; Bq = b[sel]; g = gid(A.index)
            for m1, m2 in itertools.combinations(methods, 2):
                c1, c2 = f"{m1}_{metric}", f"{m2}_{metric}"
                if c1 not in a.columns or c2 not in a.columns:
                    continue
                I = (A[c1] - A[c2]).values - (Bq[c1] - Bq[c2]).values
                if np.isnan(I).all():
                    continue
                cb = cluster_boot(I, g)
                fam_rows.append(dict(fam=fam, m1=m1, m2=m2, I=cb["mean"], lo=cb["ci"][0], hi=cb["ci"][1],
                                     dG1=float(np.nanmean(A[c1] - A[c2])), dG2=float(np.nanmean(Bq[c1] - Bq[c2])),
                                     n=cb["n"]))
        F = pd.DataFrame(fam_rows)
        res[metric] = dict(cells=R.to_dict(orient="records"), families=F.to_dict(orient="records"),
                           n_ci_excl0=nci, n_holm=nh, n_tests=len(R))
    return res


def focus(res, label, picks):
    P(f"\n  -- {label}: focus contrasts (I = G1-G2 change of m1 - m2; SHD and SID lower is better, orient higher) --")
    out = {}
    for metric in ("shd", "sidn", "orient"):
        if metric not in res:
            continue
        R = pd.DataFrame(res[metric]["cells"]); F = pd.DataFrame(res[metric]["families"])
        for fam, m1, m2 in picks:
            f = F[(F.fam == fam) & (F.m1 == m1) & (F.m2 == m2)]
            if len(f):
                f = f.iloc[0]
                P(f"   [{metric:6s}] {fam:7s} {MNAME[m1]} - {MNAME[m2]}: family-pooled I = {f.I:+.3f} [{f.lo:+.3f}, {f.hi:+.3f}] "
                  f"(G1 {f.dG1:+.3f}, G2 {f.dG2:+.3f})")
                out[f"{metric}|{fam}|{m1}|{m2}|pooled"] = dict(I=f.I, ci=[f.lo, f.hi], dG1=f.dG1, dG2=f.dG2)
            c = R[(R.fam == fam) & (R.m1 == m1) & (R.m2 == m2)]
            for _, r in c.iterrows():
                P(f"        p={r.p:2d} T={r['T']:2d}: I = {r.I:+8.3f} [{r.lo:+.3f}, {r.hi:+.3f}] (G1 {r.dG1:+.3f}, G2 {r.dG2:+.3f}) "
                  f"Holm {r.p_holm:.2g}")
                out[f"{metric}|{fam}|{m1}|{m2}|{r.p}|{r['T']}"] = dict(I=r.I, ci=[r.lo, r.hi], dG1=r.dG1, dG2=r.dG2,
                                                                        p_holm=r.p_holm)
    return out


# ----------------------------------------------------------------------------- 4. empty graph and SHD/|E|
def empty_sid_table(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    rows = []
    from common_nl import params
    for fam in FAMILIES:
        for p in (5, 10):
            for s in range(1000, 1030):
                rng_g = np.random.default_rng([FAMILIES.index(fam), p, s]); S = summary_graph(fam, p, rng_g)
                for T in (3, 5, 10):
                    G = true_dag(S, T); sn, sc = sid(G, np.zeros_like(G))
                    rows.append(dict(exp="B1", cfg="LG", family=fam, p=p, seed=s, T=T, empty_sidn=sn, empty_sid=sc))
            for cfg, seeds in (("NL", range(2000, 2025)), ("SGN", range(2000, 2025)), ("SGNU", range(2000, 2010))):
                for s in seeds:
                    S = params(cfg, fam, p, s)["S"]
                    for T in (3, 5, 10):
                        G = true_dag(S, T); sn, sc = sid(G, np.zeros_like(G))
                        rows.append(dict(exp="B1NL", cfg=cfg, family=fam, p=p, seed=s, T=T, empty_sidn=sn, empty_sid=sc))
    E = pd.DataFrame(rows); E.to_csv(path, index=False)
    return E


def levels(a, b, E, label, methods):
    P(f"\n== 4. {label}: SHD/|E| (empty graph = 1 exactly) and normalised SID (empty-graph SID in last column); "
      f"mean G1 / G2; share of data sets with SHD/|E| > 1 ==")
    Ei = E.set_index(KEY).loc[a.index]
    res = {}
    P("  family  | " + " | ".join(f"{MNAME[m]:>12s}" for m in methods) + " | empty SIDn")
    for fam in FAMILIES + ["ALL"]:
        sel = np.ones(len(a), bool) if fam == "ALL" else (a.index.get_level_values(0) == fam)
        A = a[sel]; Bq = b[sel]; r = {}
        cells = []
        for m in methods:
            if f"{m}_shd" not in a.columns:
                continue
            s1 = (A[f"{m}_shd"] / A.n_true); s2 = (Bq[f"{m}_shd"] / Bq.n_true)
            r[m] = dict(shdn_G1=float(s1.mean()), shdn_G2=float(s2.mean()), frac_gt1_G1=float((s1 > 1).mean()),
                        frac_gt1_G2=float((s2 > 1).mean()), sidn_G1=float(A[f"{m}_sidn"].mean()),
                        sidn_G2=float(Bq[f"{m}_sidn"].mean()))
            cells.append(f"{r[m]['shdn_G1']:.2f}/{r[m]['shdn_G2']:.2f}")
        r["empty_sidn"] = float(Ei[sel].empty_sidn.mean())
        res[fam] = r
        P(f"  {fam:7s} | " + " | ".join(f"{c:>12s}" for c in cells) + f" | {r['empty_sidn']:.2f}")
    P("  share of data sets with SHD/|E| > 1 (all families), G1 / G2: " + ", ".join(
        f"{MNAME[m]} {res['ALL'][m]['frac_gt1_G1']:.2f}/{res['ALL'][m]['frac_gt1_G2']:.2f}" for m in methods if m in res['ALL']))
    P("  normalised SID (all families), G1 / G2: " + ", ".join(
        f"{MNAME[m]} {res['ALL'][m]['sidn_G1']:.2f}/{res['ALL'][m]['sidn_G2']:.2f}" for m in methods if m in res['ALL'])
      + f"; empty graph {res['ALL']['empty_sidn']:.2f}")
    return res


# ----------------------------------------------------------------------------- 5. OA without first-slice edges
def oa_x1(O, a, b, label):
    P(f"\n== 5. {label}: order accuracy G1 - G2 with and without the edges leaving the first window slice "
      f"(graph-cluster CIs) ==")
    a2 = O[O.gen == "G1"].set_index(KEY).sort_index(); b2 = O[O.gen == "G2"].set_index(KEY).loc[a2.index]
    chk = {m: float(np.abs(a2[f"{m}_oa"] - a.loc[a2.index, f"{m}_oa"]).max()) for m in ("M1", "M2", "M3")}
    chk.update({f"{m}_G2": float(np.abs(b2[f"{m}_oa"] - b.loc[a2.index, f"{m}_oa"]).max()) for m in ("M1", "M2", "M3")})
    P(f"  reproduction check, max |recomputed OA - committed OA|: {chk}")
    res = dict(check=chk)
    for fam in FAMILIES + ["nonSC"]:
        sel = a2.index.get_level_values(0).isin(NONSC_FAMILIES if fam == "nonSC" else [fam])
        A = a2[sel]; Bq = b2[sel]; g = gid(A.index); r = {}
        line = f"  {fam:7s}"
        for m in ("M1", "M2", "M3"):
            full = cluster_boot(A[f"{m}_oa"].values - Bq[f"{m}_oa"].values, g)
            x1 = cluster_boot(A[f"{m}_oa_x1"].values - Bq[f"{m}_oa_x1"].values, g)
            r[m] = dict(full=full, x1=x1, G1_x1=float(A[f"{m}_oa_x1"].mean()), G2_x1=float(Bq[f"{m}_oa_x1"].mean()))
            line += f" | {MNAME[m]} all {fmt(full)} ; excl. slice-1 edges {fmt(x1)}"
        res[fam] = r; P(line)
    return res


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    b1 = pd.read_csv(B1CSV); nl = pd.read_csv(NLCSV)
    a, b = paired(b1, "het")
    P("B'2 post hoc re-analysis of committed data (b1/results/b1_test.csv.gz, b1nl/results/b1nl_test.csv.gz). "
      "Descriptive; not pre-registered.")
    OUT["B1_table"] = table_b1(a, b, "B1 (R-het)")
    OUT["B1_tracking"] = tracking(a, b, "B1 (R-het)")
    nlp = {}
    for cfg in ("NL", "SGN", "SGNU"):
        d = nl[nl.cfg == cfg]; d = d.assign(s_exact_G1=d.s_exact_G1, s_exact_stat=d.s_exact_stat)
        nlp[cfg] = paired(d)
        OUT[f"NL_{cfg}_table"] = table_b1(*nlp[cfg], f"B1-NL {cfg}")
        OUT[f"NL_{cfg}_tracking"] = tracking(*nlp[cfg], f"B1-NL {cfg}")
    picks = [("chain", "M1", "M2"), ("intree", "M3", "M5"), ("intree", "M3", "M6"), ("chain", "M3", "M5"),
             ("outtree", "M3", "M5"), ("ring", "M2", "M4")]
    res = interactions(a, b, "B1 (R-het), methods M1-M6", ["M1", "M2", "M3", "M4", "M5", "M6"])
    OUT["B1_interactions_summary"] = {k: dict(n_tests=v["n_tests"], n_ci_excl0=v["n_ci_excl0"], n_holm=v["n_holm"]) for k, v in res.items()}
    OUT["B1_focus"] = focus(res, "B1", picks)
    pd.DataFrame(res["shd"]["cells"]).to_csv(f"{out}/b1_interactions_shd.csv", index=False)
    for cfg in ("NL", "SGN", "SGNU"):
        r = interactions(*nlp[cfg], f"B1-NL {cfg}, methods M1-M5", ["M1", "M2", "M3", "M4", "M5"])
        OUT[f"NL_{cfg}_interactions_summary"] = {k: dict(n_tests=v["n_tests"], n_ci_excl0=v["n_ci_excl0"], n_holm=v["n_holm"]) for k, v in r.items()}
        OUT[f"NL_{cfg}_focus"] = focus(r, f"B1-NL {cfg}", [x for x in picks if "M6" not in x])
    E = empty_sid_table(f"{out}/empty_sid.csv.gz")
    OUT["B1_levels"] = levels(a, b, E[E.exp == "B1"].drop(columns=["exp", "cfg"]), "B1 (R-het)",
                              ["M1", "M2", "M3", "M4", "M5", "M6", "R0"])
    for cfg in ("NL", "SGN", "SGNU"):
        OUT[f"NL_{cfg}_levels"] = levels(*nlp[cfg], E[E.cfg == cfg].drop(columns=["exp", "cfg"]), f"B1-NL {cfg}",
                                         ["M1", "M2", "M3", "M4", "M5", "R0"])
    O = pd.read_csv(f"{out}/orders_x1.csv.gz")
    OUT["B1_oa_x1"] = oa_x1(O[O.exp == "B1"], a, b, "B1 (R-het)")
    for cfg in ("NL", "SGN", "SGNU"):
        OUT[f"NL_{cfg}_oa_x1"] = oa_x1(O[O.cfg == cfg], *nlp[cfg], f"B1-NL {cfg}")
    open(f"{out}/b2_reanalysis.txt", "w").write("\n".join(out_lines) + "\n")
    json.dump(OUT, open(f"{out}/b2_reanalysis.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
