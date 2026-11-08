"""B1 analysis: hypotheses H1-H3, var-vs-rel separation, sanity rules S1-S3 (PLAN.md Section 2), table and figure.
Run: /usr/bin/python3 analyse_b1.py results/b1_test.csv.gz results/   (writes analysis.txt, analysis.json, fig_b1.png/pdf)
"""
import sys, json, itertools
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import FAMILIES, SC_FAMILIES, NONSC_FAMILIES

BOOT = 10000
METHODS = ["M1", "M2", "M3", "M4", "M5", "M6"]
MNAME = dict(M1="rel-SnR", M2="R2-SnR", M3="Var-SnR", M4="Random-SnR", M5="FLOP", M6="PC", R0="Time-SnR (labels)")
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


def paired(df, gens=("G1", "G2"), regime="het"):
    a = df[(df.regime == regime) & (df.gen == gens[0])].set_index(["family", "p", "seed", "T"])
    b = df[(df.regime == regime) & (df.gen == gens[1])].set_index(["family", "p", "seed", "T"])
    return a, b.loc[a.index]


def main():
    df = pd.read_csv(sys.argv[1]); out = sys.argv[2]
    P(f"rows {len(df)}; data sets per generator/regime:", df.groupby(["gen", "regime"]).size().to_dict())
    a, b = paired(df)
    D = pd.DataFrame(index=a.index)
    D["d"] = a.M1_oa - b.M1_oa
    D["gap"] = a.s_exact - b.s_exact
    D = D.reset_index()

    # ---------------- H1
    P("\n== H1: rel-SnR order accuracy G1 - G2 (R-het), unit = summary graph, mean over T ==")
    U = D.groupby(["family", "p", "seed"]).agg(d=("d", "mean"), gap=("gap", "mean")).reset_index()
    pv = []; rows = []
    for fam in NONSC_FAMILIES:
        x = U[U.family == fam].d.values
        m, lo, hi = boot_ci(x); p1 = wil(x, "greater"); p2 = wil(x)
        pv.append(p1); rows.append((fam, m, lo, hi, p1, p2, U[U.family == fam].gap.mean(), len(x)))
    adj = holm(pv); adj2 = holm([r[5] for r in rows])
    nsig = 0; h1 = {}
    for r, q, q2 in zip(rows, adj, adj2):
        sig = q < 0.05 and r[1] > 0; nsig += sig
        P(f"  {r[0]:8s} n={r[7]} mean d = {r[1]:+.4f} [{r[2]:+.4f}, {r[3]:+.4f}]  exact gap {r[6]:.3f}  "
          f"Wilcoxon one-sided p = {r[4]:.2e}, Holm {q:.2e} {'SIG' if sig else ''}; opposite-direction two-sided Holm {q2:.2e}"
          f"{' CONTRADICTS' if (q2 < 0.05 and r[1] < 0) else ''}")
        h1[r[0]] = dict(mean=r[1], ci=[r[2], r[3]], p=r[4], p_holm=q, gap=r[6])
    pooled = boot_ci(U[U.family.isin(NONSC_FAMILIES)].d.values)
    P(f"  pooled non-SC mean d = {pooled[0]:+.4f} [{pooled[1]:+.4f}, {pooled[2]:+.4f}] over "
      f"{(U.family.isin(NONSC_FAMILIES)).sum()} graphs; Holm-significant families: {nsig}/5")
    C = D.groupby(["family", "p", "T"]).agg(d=("d", "mean"), gap=("gap", "mean")).reset_index()
    rs = stats.spearmanr(C.d, C.gap); r_unit = stats.spearmanr(D.d, D.gap)
    slope = np.polyfit(D.gap, D.d, 1)
    p_one = rs.pvalue / 2 if rs.statistic > 0 else 1 - rs.pvalue / 2
    P(f"  tracking: Spearman(cell mean d, cell mean exact gap) over {len(C)} cells = {rs.statistic:.3f} "
      f"(one-sided p = {p_one:.2e}); unit-level Spearman = {r_unit.statistic:.3f}; OLS d = {slope[0]:.3f} * gap + {slope[1]:+.3f}")
    supported = nsig >= 4 and pooled[0] >= 0.02
    killed = nsig < 3 or pooled[0] < 0.01
    track = rs.statistic >= 0.5 and p_one < 0.05
    P(f"  VERDICT H1: {'SUPPORTED' if supported else ('KILLED' if killed else 'NOT SUPPORTED (between rules)')}; "
      f"tracking {'SUPPORTED' if track else 'NOT SUPPORTED'}")
    OUT["H1"] = dict(families=h1, pooled=pooled, nsig=nsig, spearman_cells=rs.statistic, spearman_p=p_one,
                     spearman_unit=r_unit.statistic, slope=slope.tolist(), supported=bool(supported),
                     killed=bool(killed), tracking=bool(track))
    OUT["cells"] = C.to_dict(orient="records")

    # ---------------- H2
    P("\n== H2: strongly connected families, per cell d = OA_M1(G1) - OA_M1(G2) ==")
    h2 = {}; h2a_ok = True; h2b_ok = False
    for fam in SC_FAMILIES:
        for p in (5, 10):
            for T in (3, 5, 10):
                x = D[(D.family == fam) & (D.p == p) & (D["T"] == T)]
                m, lo, hi = boot_ci(x.d.values)
                tag = "T>=p" if T >= p else "T<p"
                if T >= p and not abs(m) < 0.03: h2a_ok = False
                if T < p and m >= 0.03 and lo > 0: h2b_ok = True
                P(f"  {fam:6s} p={p:2d} T={T:2d} ({tag:4s}) mean d = {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  exact gap {x.gap.mean():.3f}")
                h2[f"{fam}_{p}_{T}"] = dict(mean=m, ci=[lo, hi], gap=float(x.gap.mean()))
    s = D[D.family == "symER"].pivot_table(index=["p", "seed"], columns="T", values="d")
    p2c = wil((s[3] - s[10]).values, "greater")
    P(f"  H2a (|mean d| < 0.03 in all SC cells with T >= p): {'HOLDS' if h2a_ok else 'FAILS'} (predicted to fail)")
    P(f"  H2b (some SC cell with T < p has mean d >= 0.03, CI > 0): {'HOLDS' if h2b_ok else 'FAILS'}")
    P(f"  H2c (symER d(T=3) > d(T=10)): mean {np.mean(s[3] - s[10]):+.4f}, one-sided Wilcoxon p = {p2c:.2e} "
      f"{'HOLDS' if p2c < 0.05 else 'FAILS'}")
    OUT["H2"] = dict(cells=h2, H2a=h2a_ok, H2b=h2b_ok, H2c_p=p2c)

    # ---------------- H3
    for metric, lower_better in [("shd", True), ("sidn", True), ("orient", False), ("oa", False), ("shdx1", True)]:
        P(f"\n== H3 / rank swaps on {metric} (R-het, G1 vs G2) ==")
        tests = []
        for fam in FAMILIES:
            for p in (5, 10):
                for T in (3, 5, 10):
                    ms = [m for m in METHODS if f"{m}_{metric}" in df.columns
                          and a.loc[(fam, p, slice(None), T), f"{m}_{metric}"].notna().all()]
                    for m1, m2 in itertools.combinations(ms, 2):
                        for g, F in (("G1", a), ("G2", b)):
                            x = (F.loc[(fam, p, slice(None), T), f"{m1}_{metric}"] -
                                 F.loc[(fam, p, slice(None), T), f"{m2}_{metric}"]).values
                            tests.append(dict(fam=fam, p=p, T=T, m1=m1, m2=m2, g=g, mean=float(np.nanmean(x)), p_raw=wil(x)))
        if not tests:
            continue
        TT = pd.DataFrame(tests); TT["p_holm"] = holm(TT.p_raw.values)
        piv = TT.pivot_table(index=["fam", "p", "T", "m1", "m2"], columns="g", values=["mean", "p_holm"])
        sw_mean = piv[np.sign(piv[("mean", "G1")]) * np.sign(piv[("mean", "G2")]) < 0]
        sw_sig = sw_mean[(sw_mean[("p_holm", "G1")] < 0.05) & (sw_mean[("p_holm", "G2")] < 0.05)]
        P(f"  tests {len(TT)}; swaps by mean {len(sw_mean)}; significant swaps (Holm < 0.05 under both) {len(sw_sig)}")
        for idx, r in sw_sig.iterrows():
            better_g1 = idx[3] if (r[("mean", "G1")] < 0) == lower_better else idx[4]
            better_g2 = idx[3] if (r[("mean", "G2")] < 0) == lower_better else idx[4]
            P(f"    {idx[0]:7s} p={idx[1]:2d} T={idx[2]:2d} {MNAME[idx[3]]:10s} vs {MNAME[idx[4]]:10s}: "
              f"G1 diff {r[('mean', 'G1')]:+.3f} (Holm {r[('p_holm', 'G1')]:.1e}, better {MNAME[better_g1]}), "
              f"G2 diff {r[('mean', 'G2')]:+.3f} (Holm {r[('p_holm', 'G2')]:.1e}, better {MNAME[better_g2]})")
        # rankings per cell
        best_changes = 0; taus = []; ncell = 0
        for fam in FAMILIES:
            for p in (5, 10):
                for T in (3, 5, 10):
                    ms = [m for m in METHODS if f"{m}_{metric}" in df.columns
                          and a.loc[(fam, p, slice(None), T), f"{m}_{metric}"].notna().all()]
                    v1 = np.array([a.loc[(fam, p, slice(None), T), f"{m}_{metric}"].mean() for m in ms])
                    v2 = np.array([b.loc[(fam, p, slice(None), T), f"{m}_{metric}"].mean() for m in ms])
                    if not lower_better: v1, v2 = -v1, -v2
                    ncell += 1
                    best_changes += ms[int(np.argmin(v1))] != ms[int(np.argmin(v2))]
                    taus.append(stats.kendalltau(v1, v2).statistic)
        P(f"  cells where the best method changes: {best_changes}/{ncell}; Kendall tau(G1 ranking, G2 ranking): "
          f"median {np.nanmedian(taus):.2f}, min {np.nanmin(taus):.2f}")
        OUT[f"H3_{metric}"] = dict(n_tests=len(TT), swaps_mean=len(sw_mean), swaps_sig=len(sw_sig),
                                   best_changes=int(best_changes), n_cells=ncell, tau_median=float(np.nanmedian(taus)),
                                   sig_list=[list(map(str, i)) + [float(r[("mean", "G1")]), float(r[("mean", "G2")])]
                                             for i, r in sw_sig.iterrows()])
        if metric == "shd":
            OUT["H3_supported"] = bool(len(sw_sig) >= 1)
            P(f"  VERDICT H3 (SHD): {'SUPPORTED' if len(sw_sig) >= 1 else 'NOT SUPPORTED (no ranking change detected)'}")

    # ---------------- var versus rel, per regime
    P("\n== var- vs rel-sortability: G1 - G2 differences by regime (mean over all graphs and T, 95% CI) ==")
    a_eq, b_eq = paired(df, regime="eq")
    VR = {}
    for fam in FAMILIES + ["ALL"]:
        sel = (lambda F: F) if fam == "ALL" else (lambda F, f=fam: F[F.index.get_level_values(0) == f])
        line = f"  {fam:7s}"
        res = {}
        for lab, F1, F2, col in [("OA rel-SnR het", a, b, "M1_oa"), ("OA rel-SnR eq", a_eq, b_eq, "M1_oa"),
                                 ("OA Var-SnR het", a, b, "M3_oa"), ("OA Var-SnR eq", a_eq, b_eq, "M3_oa"),
                                 ("OA Var-SnR std", a, b, "std_M3_oa"), ("OA R2-SnR het", a, b, "M2_oa"),
                                 ("est rel-sort het", a, b, "est_rel_sort"), ("est var-sort het", a, b, "est_var_sort"),
                                 ("est var-sort std", a, b, "std_est_var_sort"), ("est R2-sort het", a, b, "est_r2_sort")]:
            x = (sel(F1)[col] - sel(F2)[col]).values
            m, lo, hi = boot_ci(x); res[lab] = [m, lo, hi]
        VR[fam] = res
    labs = list(VR["ALL"].keys())
    P("  " + " | ".join(labs))
    for fam, res in VR.items():
        P(f"  {fam:7s} " + " | ".join(f"{res[l][0]:+.3f}" for l in labs))
    OUT["var_vs_rel"] = VR

    # ---------------- levels table (for the paper)
    P("\n== Levels (R-het): mean over p, T and graphs ==")
    lev = {}
    hdr = "  family  | exact s G1 / G2 | est rel-sort G1 / G2 | OA rel-SnR G1 / G2 | SHD/|E| " + " ".join(
        f"{MNAME[m]} G1/G2" for m in ["M1", "M2", "M3", "M5"])
    P(hdr)
    for fam in FAMILIES:
        A1 = a[a.index.get_level_values(0) == fam]; B1 = b[b.index.get_level_values(0) == fam]
        r = dict(sG1=A1.s_exact.mean(), sG2=B1.s_exact.mean(), relG1=A1.est_rel_sort.mean(), relG2=B1.est_rel_sort.mean(),
                 oaG1=A1.M1_oa.mean(), oaG2=B1.M1_oa.mean())
        for m in ["M1", "M2", "M3", "M4", "M5", "R0"]:
            r[f"{m}_shdn_G1"] = (A1[f"{m}_shd"] / A1.n_true).mean(); r[f"{m}_shdn_G2"] = (B1[f"{m}_shd"] / B1.n_true).mean()
            r[f"{m}_sidn_G1"] = A1[f"{m}_sidn"].mean(); r[f"{m}_sidn_G2"] = B1[f"{m}_sidn"].mean()
        lev[fam] = r
        P(f"  {fam:7s} | {r['sG1']:.3f} / {r['sG2']:.3f} | {r['relG1']:.3f} / {r['relG2']:.3f} | "
          f"{r['oaG1']:.3f} / {r['oaG2']:.3f} | " + " ".join(f"{r[f'{m}_shdn_G1']:.2f}/{r[f'{m}_shdn_G2']:.2f}"
                                                         for m in ["M1", "M2", "M3", "M5"]))
    OUT["levels"] = lev
    P("  method ranking by mean normalised SHD (all cells pooled, R-het):")
    for g, F in (("G1", a), ("G2", b)):
        v = {MNAME[m]: float((F[f"{m}_shd"] / F.n_true).mean()) for m in ["M1", "M2", "M3", "M4", "M5", "R0"]}
        P(f"    {g}: " + ", ".join(f"{k} {x:.3f}" for k, x in sorted(v.items(), key=lambda kv: kv[1])))
        v = {MNAME[m]: float(F[f"{m}_sidn"].mean()) for m in ["M1", "M2", "M3", "M4", "M5", "R0"]}
        P(f"    {g} (SID norm.): " + ", ".join(f"{k} {x:.3f}" for k, x in sorted(v.items(), key=lambda kv: kv[1])))

    # ---------------- sanity
    P("\n== Sanity rules ==")
    a2, b2s = paired(df, gens=("G2", "G2s"))
    s1 = {}
    for fam in FAMILIES:
        x = (a2[a2.index.get_level_values(0) == fam].M1_oa - b2s[b2s.index.get_level_values(0) == fam].M1_oa).values
        s1[fam] = boot_ci(x)
    ok1 = all(abs(v[0]) < 0.02 for v in s1.values())
    P("  S1 G2 - G2s rel-SnR OA: " + ", ".join(f"{k} {v[0]:+.4f}" for k, v in s1.items()) + f" -> {'PASS' if ok1 else 'FAIL'}")
    s2 = {fam: float((a[a.index.get_level_values(0) == fam].est_rel_sort - b[b.index.get_level_values(0) == fam].est_rel_sort).mean())
          for fam in NONSC_FAMILIES}
    ok2 = all(v > 0 for v in s2.values())
    P("  S2 est rel-sort G1 - G2: " + ", ".join(f"{k} {v:+.4f}" for k, v in s2.items()) + f" -> {'PASS' if ok2 else 'FAIL'}")
    mx = df.mixerr.max(); ok3 = mx < 1e-3
    P(f"  S3 max mixing error {mx:.2e} -> {'PASS' if ok3 else 'FAIL'}")
    OUT["sanity"] = dict(S1=s1, S1_pass=ok1, S2=s2, S2_pass=ok2, S3=float(mx), S3_pass=bool(ok3))
    P(f"\nmean seconds per data set {df.secs.mean():.2f}")

    open(f"{out}/analysis.txt", "w").write("\n".join(out_lines) + "\n")
    json.dump(OUT, open(f"{out}/analysis.json", "w"), indent=1, default=float)
    figure(D, a, b, out)


def figure(D, a, b, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = dict(symER="#1b9e77", ring="#66a61e", chain="#d95f02", outtree="#7570b3", intree="#e7298a",
                ER1="#a6761d", ER2="#666666")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    C = D.groupby(["family", "p", "T"]).agg(d=("d", "mean"), gap=("gap", "mean")).reset_index()
    for fam in FAMILIES:
        c = C[C.family == fam]
        ax[0].scatter(c.gap, c.d, s=12 + 3 * c["T"], color=cols[fam], label=fam,
                      marker="o" if fam not in SC_FAMILIES else "s", alpha=0.85)
    lim = [0, 0.52]
    ax[0].plot(lim, lim, "k:", lw=0.8)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xlabel(r"exact gap $s_\rho(G_T)-s_\rho^{\rm stat}$ (Thm 1.5 vs 1.9)")
    ax[0].set_ylabel("rel-SnR order accuracy, G1 $-$ G2")
    ax[0].set_title("(a) per cell (family, p, T); marker size ~ T", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False, ncol=2)
    ms = ["M1", "M2", "M3", "M4", "M5", "R0"]
    x = np.arange(len(FAMILIES)); w = 0.13
    for k, m in enumerate(ms):
        v1 = [(a[a.index.get_level_values(0) == f][f"{m}_sidn"]).mean() for f in FAMILIES]
        v2 = [(b[b.index.get_level_values(0) == f][f"{m}_sidn"]).mean() for f in FAMILIES]
        col = plt.cm.tab10(k)
        ax[1].scatter(x + (k - 2.5) * w, v1, marker="o", color=col, s=18, label=f"{MNAME[m]}")
        ax[1].scatter(x + (k - 2.5) * w, v2, marker="x", color=col, s=22)
        for xi, y1, y2 in zip(x + (k - 2.5) * w, v1, v2):
            ax[1].plot([xi, xi], [y1, y2], color=col, lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(FAMILIES, fontsize=8)
    ax[1].set_ylabel("normalised SID (lower is better)")
    ax[1].set_title("(b) o = G1 (fixed start), x = G2 (burn-in); mean over p, T", fontsize=9)
    ax[1].legend(fontsize=6.5, frameon=False, ncol=3, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{out}/fig_b1.png", dpi=160); fig.savefig(f"{out}/fig_b1.pdf")


if __name__ == "__main__":
    main()
