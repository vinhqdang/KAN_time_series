"""B1-NL analysis (PLAN.md Section 2): H1-c, tracking, H3-c, H4-c, sanity S1-S3, levels table, figure; per config.
Run: /usr/bin/python3 analyse_b1nl.py results/b1nl_test.csv.gz results/
"""
import sys, os, json, itertools
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "b1"))
from common import FAMILIES, SC_FAMILIES, NONSC_FAMILIES

BOOT = 10000
METHODS = ["M1", "M2", "M3", "M4", "M5"]
MNAME = dict(M1="rel-SnR", M2="R2-SnR", M3="Var-SnR", M4="Random-SnR", M5="FLOP", R0="Time-SnR (labels)")
CONFIGS = ["NL", "SGN", "SGNU"]
out_lines = []; OUT = {}


def P(*a):
    s = " ".join(str(x) for x in a); out_lines.append(s); print(s)


def boot_ci(x, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (BOOT, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(pv):
    pv = np.asarray(pv, float); k = len(pv); o = np.argsort(pv); adj = np.empty(k); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (k - r) * pv[i])); adj[i] = run
    return adj


def wil(x, alt="two-sided"):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if np.all(x == 0) or len(x) < 2:
        return 1.0
    return float(stats.wilcoxon(x, zero_method="wilcox", alternative=alt).pvalue)


def analyse(df, cfg):
    R = OUT[cfg] = {}
    P(f"\n################ config {cfg}: {len(df)} rows, seeds per (family, p): {df.seed.nunique()}")
    a = df[df.gen == "G1"].set_index(["family", "p", "seed", "T"]).sort_index()
    b = df[df.gen == "G2"].set_index(["family", "p", "seed", "T"]).loc[a.index]
    D = pd.DataFrame(index=a.index)
    D["d"] = a.M1_oa - b.M1_oa
    D["gap"] = a.s_exact - b.s_exact
    D = D.reset_index()

    # ---------------- H1
    P(f"== H1-{cfg}: rel-SnR OA G1 - G2, unit = summary graph (mean over T) ==")
    U = D.groupby(["family", "p", "seed"]).agg(d=("d", "mean"), gap=("gap", "mean")).reset_index()
    rows = []
    for fam in NONSC_FAMILIES:
        x = U[U.family == fam].d.values
        m, lo, hi = boot_ci(x)
        rows.append((fam, m, lo, hi, wil(x, "greater"), wil(x), U[U.family == fam].gap.mean(), len(x)))
    adj = holm([r[4] for r in rows]); adj2 = holm([r[5] for r in rows])
    nsig = 0; h1 = {}; contra = []
    for r, q, q2 in zip(rows, adj, adj2):
        sig = q < 0.05 and r[1] > 0; nsig += sig
        if q2 < 0.05 and r[1] < 0: contra.append(r[0])
        P(f"  {r[0]:8s} n={r[7]} mean d = {r[1]:+.4f} [{r[2]:+.4f}, {r[3]:+.4f}]  exact gap {r[6]:.3f}  "
          f"one-sided p = {r[4]:.2e}, Holm {q:.2e} {'SIG' if sig else ''}; two-sided Holm {q2:.2e}"
          f"{' CONTRADICTS' if (q2 < 0.05 and r[1] < 0) else ''}")
        h1[r[0]] = dict(mean=r[1], ci=[r[2], r[3]], p=r[4], p_holm=q, gap=r[6], n=r[7])
    pooled = boot_ci(U[U.family.isin(NONSC_FAMILIES)].d.values)
    P(f"  pooled non-SC mean d = {pooled[0]:+.4f} [{pooled[1]:+.4f}, {pooled[2]:+.4f}] over "
      f"{U.family.isin(NONSC_FAMILIES).sum()} graphs; Holm-significant families {nsig}/5")
    C = D.groupby(["family", "p", "T"]).agg(d=("d", "mean"), gap=("gap", "mean")).reset_index()
    rs = stats.spearmanr(C.d, C.gap); r_unit = stats.spearmanr(D.d, D.gap)
    slope = np.polyfit(D.gap, D.d, 1)
    p_one = rs.pvalue / 2 if rs.statistic > 0 else 1 - rs.pvalue / 2
    P(f"  tracking: Spearman over {len(C)} cells = {rs.statistic:.3f} (one-sided p = {p_one:.2e}); unit-level "
      f"{r_unit.statistic:.3f}; OLS d = {slope[0]:.3f} * gap {slope[1]:+.3f}; cells with d > 0: {(C.d > 0).sum()}/{len(C)}")
    supported = nsig >= 4 and pooled[0] >= 0.02
    killed = nsig < 3 or pooled[0] < 0.01
    track = rs.statistic >= 0.5 and p_one < 0.05
    verdict = "SUPPORTED" if supported else ("KILLED" if killed else "NOT SUPPORTED (between rules)")
    P(f"  VERDICT H1-{cfg}: {verdict}; contradicting families: {contra or 'none'}; tracking "
      f"{'SUPPORTED' if track else 'NOT SUPPORTED'}")
    R["H1"] = dict(families=h1, pooled=pooled, nsig=int(nsig), verdict=verdict, contradicting=contra,
                   spearman_cells=float(rs.statistic), spearman_p=float(p_one), spearman_unit=float(r_unit.statistic),
                   slope=slope.tolist(), tracking=bool(track), supported=bool(supported), killed=bool(killed))
    R["cells"] = C.to_dict(orient="records")

    # ---------------- SC families (descriptive)
    P(f"== SC families (descriptive), per cell ==")
    sc = {}; h2b = False
    for fam in SC_FAMILIES:
        for p in (5, 10):
            for T in (3, 5, 10):
                x = D[(D.family == fam) & (D.p == p) & (D["T"] == T)]
                m, lo, hi = boot_ci(x.d.values)
                if T < p and m >= 0.03 and lo > 0: h2b = True
                P(f"  {fam:6s} p={p:2d} T={T:2d} mean d = {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  exact gap {x.gap.mean():.3f}")
                sc[f"{fam}_{p}_{T}"] = dict(mean=m, ci=[lo, hi], gap=float(x.gap.mean()))
    P(f"  some SC cell with T < p has d >= 0.03, CI > 0: {h2b}")
    R["SC"] = dict(cells=sc, h2b=h2b)

    # ---------------- H3
    nseed = df.seed.nunique()
    for metric, lower_better in [("shd", True), ("sidn", True), ("orient", False), ("oa", False), ("shdx1", True)]:
        tests = []
        for fam in FAMILIES:
            for p in (5, 10):
                for T in (3, 5, 10):
                    ms = [m for m in METHODS if f"{m}_{metric}" in df.columns
                          and a.loc[(fam, p, slice(None), T), f"{m}_{metric}"].notna().all()
                          and b.loc[(fam, p, slice(None), T), f"{m}_{metric}"].notna().all()]
                    for m1, m2 in itertools.combinations(ms, 2):
                        for g, F in (("G1", a), ("G2", b)):
                            x = (F.loc[(fam, p, slice(None), T), f"{m1}_{metric}"] -
                                 F.loc[(fam, p, slice(None), T), f"{m2}_{metric}"]).values
                            tests.append(dict(fam=fam, p=p, T=T, m1=m1, m2=m2, g=g, mean=float(np.nanmean(x)), p_raw=wil(x)))
        TT = pd.DataFrame(tests); TT["p_holm"] = holm(TT.p_raw.values)
        piv = TT.pivot_table(index=["fam", "p", "T", "m1", "m2"], columns="g", values=["mean", "p_holm"])
        sw_mean = piv[np.sign(piv[("mean", "G1")]) * np.sign(piv[("mean", "G2")]) < 0]
        sw_sig = sw_mean[(sw_mean[("p_holm", "G1")] < 0.05) & (sw_mean[("p_holm", "G2")] < 0.05)]
        P(f"== H3-{cfg} rank swaps on {metric}: tests {len(TT)}; swaps by mean {len(sw_mean)}; significant {len(sw_sig)}")
        for idx, r in sw_sig.iterrows():
            better_g1 = idx[3] if (r[("mean", "G1")] < 0) == lower_better else idx[4]
            better_g2 = idx[3] if (r[("mean", "G2")] < 0) == lower_better else idx[4]
            P(f"    {idx[0]:7s} p={idx[1]:2d} T={idx[2]:2d} {MNAME[idx[3]]:10s} vs {MNAME[idx[4]]:10s}: "
              f"G1 diff {r[('mean', 'G1')]:+.3f} (Holm {r[('p_holm', 'G1')]:.1e}, better {MNAME[better_g1]}), "
              f"G2 diff {r[('mean', 'G2')]:+.3f} (Holm {r[('p_holm', 'G2')]:.1e}, better {MNAME[better_g2]})")
        best_changes = 0; taus = []; ncell = 0
        for fam in FAMILIES:
            for p in (5, 10):
                for T in (3, 5, 10):
                    ms = METHODS if metric != "oa" else ["M1", "M2", "M3", "M4"]
                    v1 = np.array([a.loc[(fam, p, slice(None), T), f"{m}_{metric}"].mean() for m in ms])
                    v2 = np.array([b.loc[(fam, p, slice(None), T), f"{m}_{metric}"].mean() for m in ms])
                    if not lower_better: v1, v2 = -v1, -v2
                    ncell += 1
                    best_changes += ms[int(np.nanargmin(v1))] != ms[int(np.nanargmin(v2))]
                    taus.append(stats.kendalltau(v1, v2).statistic)
        P(f"   best method changes in {best_changes}/{ncell} cells; Kendall tau median {np.nanmedian(taus):.2f}")
        R[f"H3_{metric}"] = dict(n_tests=len(TT), swaps_mean=len(sw_mean), swaps_sig=len(sw_sig),
                                 best_changes=int(best_changes), tau_median=float(np.nanmedian(taus)),
                                 sig_list=[list(map(str, i)) + [float(r[("mean", "G1")]), float(r[("mean", "G2")])]
                                           for i, r in sw_sig.iterrows()])
        if metric == "shd":
            if nseed < 12:
                R["H3_verdict"] = "NOT TESTABLE (too few seeds for Holm over the tests; descriptive only)"
            else:
                R["H3_verdict"] = "SUPPORTED" if len(sw_sig) >= 1 else "NOT SUPPORTED (no ranking change detected)"
            P(f"   VERDICT H3-{cfg} (SHD): {R['H3_verdict']}")

    # ---------------- H4 var vs rel
    P(f"== H4-{cfg}: G1 - G2 differences (pooled over all families, unit = data-set pair) ==")
    VR = {}
    for fam in FAMILIES + ["ALL"]:
        sel = (lambda F: F) if fam == "ALL" else (lambda F, f=fam: F[F.index.get_level_values(0) == f])
        res = {}
        for lab, col in [("OA rel-SnR", "M1_oa"), ("OA Var-SnR raw", "M3_oa"), ("OA Var-SnR std", "std_M3_oa"),
                         ("OA R2-SnR", "M2_oa"), ("est rel-sort", "est_rel_sort"), ("est var-sort raw", "est_var_sort"),
                         ("est var-sort std", "std_est_var_sort"), ("est R2-sort", "est_r2_sort"),
                         ("rel recall", "rel_recall")]:
            res[lab] = boot_ci((sel(a)[col] - sel(b)[col]).values)
        VR[fam] = res
    labs = list(VR["ALL"].keys())
    P("  family  | " + " | ".join(labs))
    for fam, res in VR.items():
        P(f"  {fam:7s} | " + " | ".join(f"{res[l][0]:+.3f}" for l in labs))
    al = VR["ALL"]
    h4a = al["OA Var-SnR raw"][0] >= 0.05 and al["OA Var-SnR raw"][1] > 0
    h4b = abs(al["OA Var-SnR std"][0]) < 0.02
    h4c = al["OA rel-SnR"][1] > 0
    P(f"  (a) Var raw {al['OA Var-SnR raw'][0]:+.3f} [{al['OA Var-SnR raw'][1]:+.3f}, {al['OA Var-SnR raw'][2]:+.3f}] -> {h4a}; "
      f"(b) Var std {al['OA Var-SnR std'][0]:+.4f} -> {h4b}; (c) rel {al['OA rel-SnR'][0]:+.3f} "
      f"[{al['OA rel-SnR'][1]:+.3f}, {al['OA rel-SnR'][2]:+.3f}] -> {h4c}")
    R["H4"] = dict(table=VR, a=bool(h4a), b=bool(h4b), c=bool(h4c), holds=bool(h4a and h4b and h4c))
    P(f"  VERDICT H4-{cfg}: {'HOLDS' if R['H4']['holds'] else 'DOES NOT HOLD'}")

    # ---------------- levels
    P(f"== Levels ({cfg}): mean over p, T and graphs ==")
    lev = {}
    P("  family  | exact s G1/G2 | est rel-sort G1/G2 | recall G1/G2 | OA M1 G1/G2 | OA M2 | OA M3 | SHDn M1 M2 M3 M5 (G1/G2) | SIDn M5")
    for fam in FAMILIES:
        A1 = a[a.index.get_level_values(0) == fam]; B1 = b[b.index.get_level_values(0) == fam]
        r = dict(sG1=A1.s_exact.mean(), sG2=B1.s_exact.mean(), relG1=A1.est_rel_sort.mean(), relG2=B1.est_rel_sort.mean(),
                 recG1=A1.rel_recall.mean(), recG2=B1.rel_recall.mean(), fprG1=A1.rel_fpr.mean(), fprG2=B1.rel_fpr.mean())
        for m in ["M1", "M2", "M3", "M4", "R0"]:
            r[f"{m}_oa_G1"] = A1[f"{m}_oa"].mean(); r[f"{m}_oa_G2"] = B1[f"{m}_oa"].mean()
        for m in ["M1", "M2", "M3", "M4", "M5", "R0"]:
            r[f"{m}_shdn_G1"] = (A1[f"{m}_shd"] / A1.n_true).mean(); r[f"{m}_shdn_G2"] = (B1[f"{m}_shd"] / B1.n_true).mean()
            r[f"{m}_sidn_G1"] = A1[f"{m}_sidn"].mean(); r[f"{m}_sidn_G2"] = B1[f"{m}_sidn"].mean()
        r["dci"] = boot_ci(D[D.family == fam].d.values)
        lev[fam] = r
        P(f"  {fam:7s} | {r['sG1']:.3f}/{r['sG2']:.3f} | {r['relG1']:.3f}/{r['relG2']:.3f} | {r['recG1']:.2f}/{r['recG2']:.2f} | "
          f"{r['M1_oa_G1']:.3f}/{r['M1_oa_G2']:.3f} | {r['M2_oa_G1']:.3f}/{r['M2_oa_G2']:.3f} | {r['M3_oa_G1']:.3f}/{r['M3_oa_G2']:.3f} | "
          + " ".join(f"{r[f'{m}_shdn_G1']:.2f}/{r[f'{m}_shdn_G2']:.2f}" for m in ["M1", "M2", "M3", "M5"])
          + f" | {r['M5_sidn_G1']:.2f}/{r['M5_sidn_G2']:.2f}")
    R["levels"] = lev
    for g, F in (("G1", a), ("G2", b)):
        v = {MNAME[m]: float((F[f"{m}_shd"] / F.n_true).mean()) for m in ["M1", "M2", "M3", "M4", "M5", "R0"]}
        P(f"  {g} SHD/|E|: " + ", ".join(f"{k} {x:.3f}" for k, x in sorted(v.items(), key=lambda kv: kv[1])))
        R[f"rank_{g}"] = v

    # ---------------- sanity
    s1 = {fam: boot_ci((b[b.index.get_level_values(0) == fam].M1_oa - b[b.index.get_level_values(0) == fam].G2L_M1_oa).values)
          for fam in FAMILIES}
    ok1 = all(abs(v[0]) < 0.02 for v in s1.values())
    s2 = {fam: float((a[a.index.get_level_values(0) == fam].est_rel_sort - b[b.index.get_level_values(0) == fam].est_rel_sort).mean())
          for fam in NONSC_FAMILIES}
    ok2 = all(v > 0 for v in s2.values())
    mx = float(df.coupling_err.max()); ok3 = mx < 1e-3
    P("  S1 G2 - G2L rel-SnR OA: " + ", ".join(f"{k} {v[0]:+.4f}" for k, v in s1.items()) + f" -> {'PASS' if ok1 else 'FAIL'}")
    P("  S2 est rel-sort G1 - G2: " + ", ".join(f"{k} {v:+.4f}" for k, v in s2.items()) + f" -> {'PASS' if ok2 else 'FAIL'}")
    P(f"  S3 max coupling error {mx:.2e} -> {'PASS' if ok3 else 'FAIL'}")
    R["sanity"] = dict(S1=s1, S1_pass=ok1, S2=s2, S2_pass=ok2, S3=mx, S3_pass=ok3)
    P(f"  mean seconds per data set {df.secs.mean():.2f}")
    return D


def figure(Ds, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = dict(symER="#1b9e77", ring="#66a61e", chain="#d95f02", outtree="#7570b3", intree="#e7298a",
                ER1="#a6761d", ER2="#666666")
    fig, ax = plt.subplots(1, len(Ds), figsize=(4.0 * len(Ds), 3.6), sharey=True)
    for k, (cfg, D) in enumerate(Ds.items()):
        C = D.groupby(["family", "p", "T"]).agg(d=("d", "mean"), gap=("gap", "mean")).reset_index()
        for fam in FAMILIES:
            c = C[C.family == fam]
            ax[k].scatter(c.gap, c.d, s=12 + 3 * c["T"], color=cols[fam], label=fam,
                          marker="o" if fam not in SC_FAMILIES else "s", alpha=0.85)
        ax[k].plot([0, 0.52], [0, 0.52], "k:", lw=0.8); ax[k].axhline(0, color="k", lw=0.5)
        ax[k].set_xlabel(r"exact gap $s_\rho(G_T)-s_\rho^{\rm stat}$")
        ax[k].set_title(f"{cfg}: Spearman {OUT[cfg]['H1']['spearman_cells']:.2f}", fontsize=9)
    ax[0].set_ylabel("rel-SnR order accuracy, G1 $-$ G2")
    ax[0].legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(f"{out}/fig_b1nl.png", dpi=160); fig.savefig(f"{out}/fig_b1nl.pdf")


def main():
    df = pd.read_csv(sys.argv[1]); out = sys.argv[2]
    Ds = {}
    for cfg in CONFIGS:
        if (df.cfg == cfg).any():
            Ds[cfg] = analyse(df[df.cfg == cfg].copy(), cfg)
    P("\n################ summary")
    for cfg in Ds:
        h = OUT[cfg]
        P(f"  {cfg}: H1 {h['H1']['verdict']} ({h['H1']['nsig']}/5, pooled {h['H1']['pooled'][0]:+.3f}); tracking "
          f"{h['H1']['spearman_cells']:.2f} {'SUPPORTED' if h['H1']['tracking'] else 'NOT SUPPORTED'}; H3 {h['H3_verdict']} "
          f"({h['H3_shd']['swaps_sig']} sig. SHD swaps); H4 {'HOLDS' if h['H4']['holds'] else 'DOES NOT HOLD'}")
    ok = [c for c in ("NL", "SGN") if c in OUT and OUT[c]["H1"]["supported"] and OUT[c]["H1"]["tracking"]]
    rep = "REPLICATES" if len(ok) == 2 else ("PARTIAL" if len(ok) == 1 else "FAILS")
    P(f"  Replication summary (H1 + tracking in NL and SGN): {rep} ({ok})")
    OUT["replication"] = rep
    open(f"{out}/analysis.txt", "w").write("\n".join(out_lines) + "\n")
    json.dump(OUT, open(f"{out}/analysis.json", "w"), indent=1, default=float)
    figure(Ds, out)


if __name__ == "__main__":
    main()
