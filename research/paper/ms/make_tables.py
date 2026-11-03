"""Generate all LaTeX tables (tables/*.tex) from the CSVs under research/.
Usage: python3.13 make_tables.py
Tables whose data do not exist yet are written as clearly marked PENDING placeholders.
"""
import glob
import os

import numpy as np
import pandas as pd

from msdata import (CELLNAME, CELLS, HERE, R, boot_ci, exists, fmt, holm, load_adaptive, load_family,
                    load_replication, wide, wilcoxon_p, write_table)

PEND = r"\textbf{PENDING:} data not yet available when this PDF was built; the script fills this table from \texttt{%s} when it exists."


def booktab(colspec, header, rows, note=None):
    s = "\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}{%s}\n\\toprule\n%s \\\\\n\\midrule\n" % (colspec, " & ".join(header))
    s += "\n".join(" & ".join(r) + " \\\\" for r in rows)
    s += "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
    if note:
        s += "\n\\smallskip\n{\\footnotesize %s}\n" % note
    return s


def pending(name, src, hdr_cols=4):
    write_table(name, "\\fbox{\\parbox{0.9\\linewidth}{" + PEND % src.replace("_", "\\_") + "}}\n")


# ---------------------------------------------------------------- T1 main replication
def t_main(rep):
    cells = [c for c, _ in CELLS]
    p_ssr, p_ssc = {}, {}
    for c in cells[:7]:
        w = wide(rep, "nrmse", c).dropna()
        p_ssr[c] = wilcoxon_p((w["BASIL-W"] - w["SSR"]).values)
        p_ssc[c] = wilcoxon_p((w["BASIL-W"] - w["SSC"]).values)
    h_ssr = dict(zip(cells[:7], holm([p_ssr[c] for c in cells[:7]])))
    h_ssc = dict(zip(cells[:7], holm([p_ssc[c] for c in cells[:7]])))
    rows = []
    for c in cells:
        w = wide(rep, "nrmse", c).dropna()
        if len(w) == 0:
            continue
        ratio_ci = boot_ci((w["BASIL-W"] / w["SSC"]).values)  # per-dataset ratios (cell-level CI of mean ratio)
        diff = (w["BASIL-W"] - w["SSR"]).values
        dci = boot_ci(diff)
        wins = int((diff < 0).sum())
        hp = "--" if c == "C8" else fmt(h_ssr[c], 3)
        hp2 = "--" if c == "C8" else fmt(h_ssc[c], 3)
        rows.append([c, CELLNAME[c], str(len(w)), fmt(w["SSC"].mean()), fmt(w["SSR"].mean()), fmt(w["BASIL-W"].mean()),
                     fmt(w["BASIL-W"].mean() / w["SSC"].mean(), 2), f"{diff.mean():+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}]", f"{wins}/{len(w)}", hp, hp2])
    note = ("Edge-function NRMSE (mean over datasets; lower is better); a missed true edge counts as 1. Ratio = BASIL-W / SSC of the cell means. "
            "Difference = BASIL-W minus refit with 95\\% bootstrap interval over datasets. Holm: two-sided exact Wilcoxon signed-rank p, Holm-adjusted over C1--C7 "
            "(C8 has 5 datasets, minimum attainable p = 0.0625, not testable), for BASIL-W vs refit and BASIL-W vs SSC. "
            "BASIL-W is below SSC in all 75 datasets. Cells share graphs across seeds (C1--C6 use the same 10 DAGs), so they are not independent replications.")
    write_table("tab_main.tex", booktab("llrrrrrrrrr", ["Cell", "Setting", "$N$", "SSC", "refit", "BASIL-W", "ratio", "BASIL-W $-$ refit [95\\% CI]", "wins", "Holm p (refit)", "Holm p (SSC)"], rows, note))


# ---------------------------------------------------------------- T2 structure and calibration
def t_struct(rep):
    rows = []
    for c, _ in CELLS:
        g = rep[rep.cell == c]
        r = [c]
        for col in ["auroc"]:
            r += [fmt(g[g.M == m][col].mean()) for m in ["SSC", "SSR", "BASIL-W"]]
        r += [fmt(g[g.M == m]["orient"].mean(), 2) for m in ["SSC", "SSR", "BASIL-W"]]
        r += ["%s / %s / %s" % tuple(fmt(g[g.M == m]["do_direct"].mean(), 2) for m in ["SSC", "SSR", "BASIL-W"])]
        r += ["%s / %s / %s" % tuple(fmt(g[g.M == m]["do_indirect"].mean(), 2) for m in ["SSC", "SSR", "BASIL-W"])]
        b = g[g.M == "BASIL-W"]
        r += [fmt(b.cover.mean(), 3), fmt(b.ece.mean(), 4)]
        rows.append(r)
    note = ("Means over datasets. do-effect NRMSE for up to 20 direct and 20 indirect true pairs per dataset (SSC / refit / BASIL-W). "
            "Coverage: pointwise coverage of the 90\\% BASIL-W band over true edges, kappa = 1; ECE of the edge probabilities. "
            "BASIL-W AUROC is lower than SSC in 9 or 10 of 10 datasets in every $d=100$ cell.")
    write_table("tab_struct.tex", booktab("lrrrrrrccrr", ["Cell", "AUROC SSC", "refit", "BASIL-W", "orient. SSC", "refit", "BASIL-W", "do direct", "do indirect", "cover.", "ECE"], rows, note))
    # BASIL-W minus SSC AUROC gap
    gaps = []
    for c, _ in CELLS:
        w = wide(rep, "auroc", c).dropna()
        gaps.append((c, (w["BASIL-W"] - w["SSC"]).mean(), int((w["BASIL-W"] < w["SSC"]).sum()), len(w)))
    write_table("auroc_gaps.txt", "\n".join("%s %+.3f %d/%d" % g for g in gaps) + "\n")


# ---------------------------------------------------------------- T3 adaptive arm
def t_adaptive(rep, ada):
    if ada is None:
        pending("tab_adaptive.tex", "research/adaptive/results")
        return
    rows = []
    for c, _ in CELLS:
        a = ada[(ada.cell == c) & (ada.M == "SSA")]
        if len(a) == 0:
            continue
        seeds = sorted(a.seed.unique())
        def sel(df, m):
            return df[(df.cell == c) & (df.M == m) & df.seed.isin(seeds)].set_index("seed")
        ssc, ssr, bw = sel(rep, "SSC"), sel(rep, "SSR"), sel(rep, "BASIL-W")
        sa = a.set_index("seed")
        s = sorted(set(sa.index) & set(bw.index))
        d_bw = (sa.loc[s, "nrmse"] - bw.loc[s, "nrmse"]).values
        d_sr = (sa.loc[s, "nrmse"] - ssr.loc[s, "nrmse"]).values
        rows.append([CELLNAME[c], str(len(s)), fmt(ssc.loc[s, "nrmse"].mean()), fmt(sa.loc[s, "nrmse"].mean()), fmt(ssr.loc[s, "nrmse"].mean()), fmt(bw.loc[s, "nrmse"].mean()),
                     f"{d_sr.mean():+.3f} ({int((d_sr > 0).sum())}/{len(s)})", f"{d_bw.mean():+.3f} ({int((d_bw > 0).sum())}/{len(s)})", fmt(wilcoxon_p(d_bw), 3),
                     fmt(sa.loc[s, "auroc"].mean()), fmt(bw.loc[s, "auroc"].mean())])
    note = ("Adaptive group lasso (SSA: weights $(\\|\\theta\\|+0.01)^{-1}$, same $\\lambda=0.2/d$, $\\gamma$ and $\\epsilon$ not tuned) on the replication seeds; "
            "SSC, refit and BASIL-W values are the replication runs on the same seeds. Differences SSA minus comparator, in parentheses the number of datasets in which SSA is worse. "
            "Wilcoxon p (uncorrected) for SSA vs BASIL-W; with 5 or 3 datasets the minimum attainable p is 0.0625 or 0.25.")
    write_table("tab_adaptive.tex", booktab("lrrrrrrrrrr", ["Cell", "$N$", "SSC", "SSA", "refit", "BASIL-W", "SSA $-$ refit", "SSA $-$ BASIL-W", "p", "AUROC SSA", "AUROC BASIL-W"], rows, note))


# ---------------------------------------------------------------- T4 ablation
ABL = [("base", "BASIL-W (reference)"), ("v1_R2cand", "candidates by marginal $R^2$"), ("v2_cold", "cold start (random order)"),
       ("v3_cold_R2", "cold start + $R^2$ candidates (BASIL-S)"), ("v4a_g_n/4", "$g=n/4$"), ("v4b_g_4n", "$g=4n$"),
       ("v5a_pi_x0.5", "$\\pi \\times 0.5$"), ("v5b_pi_x2", "$\\pi \\times 2$"), ("v6_kmax2", "$k_{max}=2$"), ("v7_4ch_600d", "4 chains, $600d$ iterations")]


def t_ablation():
    p = os.path.join(R, "ablation/results/test/unif_main.csv")
    if not exists(p):
        pending("tab_ablation.tex", "research/ablation/results/test/unif_main.csv")
        return
    m = pd.read_csv(p)
    base = m[m.variant == "base"].set_index("seed")
    rows = []
    for v, lab in ABL:
        g = m[m.variant == v].set_index("seed")
        dn = (g.nrmse - base.nrmse.reindex(g.index))
        do_ = (g.orient - base.orient.reindex(g.index))
        rows.append([lab, fmt(g.nrmse.mean()), (f"{dn.mean():+.3f}" if v != "base" else "--"), fmt(g.orient.mean()), (f"{do_.mean():+.3f}" if v != "base" else "--"),
                     fmt(g.auroc.mean()), fmt(g.do_indirect.mean()), fmt(g.cover.mean()), fmt(g.gap.mean(), 1), fmt(g.time.mean(), 1)])
    note = ("d=100, n=1000, uniform noise, 8 test datasets (seeds 3005--3012), one change at a time. $\\Delta$ = paired mean difference to the reference. "
            "End-score gap: max$-$min over chains of the mean log-score at the end of the chains (nats); time: BASIL fit only, 1 core, seconds. No significance tests.")
    write_table("tab_ablation.tex", booktab("lrrrrrrrrr", ["Variant", "NRMSE", "$\\Delta$", "orient.", "$\\Delta$", "AUROC", "do indirect", "cover.", "gap", "time"], rows, note))


# ---------------------------------------------------------------- T5 bands
def t_bands():
    base = os.path.join(R, "ablation/results")
    srcs = [("uniform (8)", "test/unif_edges.csv"), ("Laplace (5)", "laplace/laplace_edges.csv"), ("hetero (5)", "hetero/hetero_edges.csv")]
    rows = []
    for nm, f in srcs:
        p = os.path.join(base, f)
        if not exists(p):
            continue
        e = pd.read_csv(p, usecols=["variant", "seed", "P", "cov_1.0", "all_1.0", "w_1.0", "cov_1.55", "all_1.55", "w_1.55"])
        e = e[e.variant == "base"]
        for kk in ["1.0", "1.55"]:
            ps = lambda sel: e[sel][f"cov_{kk}"].mean()  # pooled over edges
            r = [nm, "1" if kk == "1.0" else "1.55", fmt(e.groupby("seed")[f"cov_{kk}"].mean().mean()),
                 fmt(ps(e.P > 0.9)), fmt(ps(e.P <= 0.9)), fmt(e.groupby("seed")[f"all_{kk}"].mean().mean(), 2), fmt(e.groupby("seed")[f"w_{kk}"].mean().mean())]
            rows.append(r)
    note = ("Pointwise coverage of the 90\\% band (5th/95th percentile of $m+\\kappa(D-m)$ for posterior draws $D$ and mean $m$) over true edges, mean over datasets (found/missed: pooled over edges). "
            "$\\kappa^*=1.55$ was chosen as the smallest value reaching coverage 0.90 on 5 validation datasets (seeds 3000--3004) and frozen. "
            "`found' = posterior inclusion $P>0.9$; `missed' = $P\\le0.9$; `whole curve' = share of curves entirely inside the band.")
    write_table("tab_bands.tex", booktab("llrrrrr", ["Noise (datasets)", "$\\kappa$", "all edges", "found", "missed", "whole curve", "mean width"], rows, note))


# ---------------------------------------------------------------- T6 Sachs
def t_sachs():
    p = os.path.join(R, "real_data/results/structure.csv")
    q = os.path.join(R, "real_data/results/stability_common.csv")
    if not exists(p):
        pending("tab_sachs.tex", "research/real_data/results/structure.csv")
        return
    s = pd.read_csv(p)
    st = pd.read_csv(q)
    names = [("SSC", "SSC (SPADE-scaled)"), ("SSR_k1.5", "refit"), ("BASILW", "BASIL-W"), ("BASIL", "BASIL (full)"), ("GAM-ridge", "GAM-ridge"),
             ("NOTEARS-linear", "NOTEARS-linear"), ("DAGMA-linear", "DAGMA-linear"), ("var-sortnregress", "var-sortnregress (z-scored)")]
    rows = []
    for k, lab in names:
        g = s[s.method == k]
        ci = boot_ci(g.auroc.values)
        stab = fmt(st[k].mean(), 2) if k in st.columns else "--"
        rows.append([lab, f"{g.auroc.mean():.3f} [{ci[0]:.3f}, {ci[1]:.3f}]", fmt(g.auprc.mean()), fmt(g.skel_auroc.mean(), 3), fmt(g.orient.mean(), 2), fmt(g.hits18.mean(), 1), stab])
    note = ("Sachs, $d=11$, 10 subsamples of 2000 cells (overlapping, so intervals are optimistic), 18 consensus edges (contested; chance AUPRC 0.164, about 2.9 hits at chance). "
            "Stability: mean R$^2_{aff}$ of the edge curves across subsamples on the 24 edges that are in SSC's top 18 in at least 3 subsamples; "
            "an unselected edge counts as a zero curve (R$^2=0$), which is what drives the low values for refit and BASIL-W.")
    write_table("tab_sachs.tex", booktab("lrrrrrr", ["Method", "AUROC [95\\% CI]", "AUPRC", "skel. AUROC", "orient.", "hits in top 18", "stability"], rows, note))


# ---------------------------------------------------------------- T7 negative results (typed from the READMEs; sources in the caption note)
def t_negative():
    rows = [
        ["SPIRE (group-MCP + HSIC term)", "$d=10$, $n=1000$, 30 datasets", "AUROC $-0.021$ [$-0.031,-0.010$] vs SPADE; slope 0.75 (kill line 0.80); HSIC term adds $+0.05$ orientation at best, not significant on validation", "H1 fails, H2 inconclusive"],
        ["BASIL-S cold start", "$d=100$--$500$, 5 datasets (1 at $d=500$)", "chain end-scores differ by hundreds of nats; AUROC $-0.02$ to $-0.03$ vs SPADE-scaled; orientation 0.83 vs 0.94 at $d=200$ (original); $d=500$: orientation 0.71, slope 0.51", "does not scale as specified"],
        ["BASIL-S, cold start (ablation)", "$d=100$, 8 datasets", "NRMSE 0.324 vs 0.177, orientation 0.809 vs 0.943 (8/8 worse)", "warm start essential"],
        ["Full BASIL", "$d=50$", "about 11 min per dataset; no better than BASIL-W on Sachs (AUROC 0.544)", "not scalable"],
        ["SCORE / NoGAM", "$d=10$, 10 datasets", "AUROC 0.858 / 0.895 vs 0.980 (SPADE), 5--10$\\times$ slower", "baseline only"],
        ["Pre-registered BASIL-W do-effects", "$d\\ge100$, 4 of 45 datasets", "indirect do-effect NRMSE up to 185 (coefficients up to $1.4\\times10^7$); fixed post hoc by a ridge map", "instability; fixed, 0/75 on fresh seeds"],
    ]
    write_table("tab_negative.tex", booktab("p{2.9cm}p{3.2cm}p{6.2cm}p{2.6cm}", ["Attempt", "Setting", "Outcome", "Verdict"], rows,
                "Numbers from research/spire (README\\_results, README\\_basil, README\\_scale, README\\_do), research/ablation and research/real\\_data."))


# ---------------------------------------------------------------- independent-DAG replication, adaptive2, semisyn (read whatever exists)
from msdata import load_runs

NOISE_LABEL = {"unif": "uniform", "laplace": "Laplace", "hetero": "heteroscedastic", "boot": "bootstrap residuals", "gauss": "Gaussian"}


def paired(w, a, b):
    """mean difference a-b, CI, wins of a (lower), Wilcoxon p"""
    x = (w[a] - w[b]).dropna().values
    if len(x) < 2:
        return None
    ci = boot_ci(x)
    return x.mean(), ci, int((x < 0).sum()), len(x), wilcoxon_p(x)


def t_rep2():
    d = load_runs("replication2")
    if d is None:
        pending("tab_rep2.tex", "research/replication2/results/d*_s*.csv")
        pending("tab_rep2b.tex", "research/replication2/results/d*_s*.csv")
        return
    rows, rows2 = [], []
    for c in ["unif", "laplace", "hetero"]:
        g = d[d.cell == c]
        if len(g) == 0:
            continue
        w = g.pivot(index="seed", columns="M", values="nrmse")
        full = w.dropna()
        ms = ["SSC", "SSA", "SSR", "BASIL-W"]
        if not set(ms) <= set(full.columns):
            continue
        p_bs = paired(full, "BASIL-W", "SSA")
        p_bc = paired(full, "BASIL-W", "SSC")
        p_br = paired(full, "BASIL-W", "SSR")
        rows.append([NOISE_LABEL[c], str(len(full))] + [fmt(full[m].mean()) for m in ms] +
                    [f"{p_bs[0]:+.3f} [{p_bs[1][0]:+.3f}, {p_bs[1][1]:+.3f}]", f"{p_bs[2]}/{p_bs[3]}", fmt(p_bs[4], 4),
                     f"{p_br[0]:+.3f} ({p_br[2]}/{p_br[3]})", f"{p_bc[2]}/{p_bc[3]}"])
        def mean_of(m, col):
            v = g[g.M == m][col].dropna()
            return fmt(v.mean(), 3) if len(v) else "--"
        def cost(m):
            a = g[g.M == m].set_index("seed").t_cost; b = g[g.M == "SSC"].set_index("seed").t_cost
            r = (a / b).dropna()
            return fmt(r.median(), 2) if len(r) else "--"
        rows2.append([NOISE_LABEL[c]] + ["%s / %s / %s / %s" % tuple(mean_of(m, "auroc") for m in ms)] +
                     ["%s / %s / %s / %s" % tuple(mean_of(m, "do_direct") for m in ms)] +
                     ["%s / %s / %s / %s" % tuple(mean_of(m, "do_indirect") for m in ms)] +
                     [mean_of("BASIL-W", "cover"), mean_of("BASIL-W", "ece"), "%s / %s / %s" % (cost("SSA"), cost("SSR"), cost("BASIL-W"))])
    note = ("Independent-DAG replication (\\texttt{research/replication2}): complete datasets only (all four arms present), edge-function NRMSE, mean over datasets. "
            "BASIL-W $-$ SSA: mean paired difference with 95\\% bootstrap interval, number of datasets in which BASIL-W is lower, uncorrected exact two-sided Wilcoxon p "
            "(the plan's Holm-adjusted results are in research/replication2/results/analysis.md). BASIL-W $-$ refit: mean difference (wins). Last column: datasets where BASIL-W $<$ SSC. "
            "The uniform seeds 3000--3012 coincide with the ablation datasets.")
    write_table("tab_rep2.tex", booktab("lrrrrrrrrrrr", ["Noise", "$N$", "SSC", "SSA", "refit", "BASIL-W", "BASIL-W $-$ SSA [95\\% CI]", "wins", "p", "BASIL-W $-$ refit", "BASIL-W$<$SSC"], rows, note))
    note2 = ("Means over complete datasets (SSC / SSA / refit / BASIL-W). Cost: median over datasets of the per-dataset cost ratio to SSC (SSA / refit / BASIL-W), "
             "excluding metric code, shared loaded machine.")
    write_table("tab_rep2b.tex", booktab("lccccccc", ["Noise", "AUROC", "do direct", "do indirect", "cover. (BASIL-W)", "ECE", "cost ratio"], rows2, note2))


def t_adaptive2(rep):
    d = load_runs("adaptive2", recursive=True)
    if d is None:
        pending("tab_adaptive2.tex", "research/adaptive2/results")
        return
    def parse(m):
        if m == "SSA":
            return (1.0, 0.01)
        if m.startswith("SSA_g"):
            g, e = m[5:].split("_e")
            return (float(g), float(e))
        return None
    d = d.copy()
    d["gam_eps"] = d.method.map(parse)
    body = ""
    # Part A: d = 500
    a = d[d.d == 500]
    if len(a):
        rows = []
        for sd in sorted(a.seed.unique()):
            for m in ["SSC", "SSR", "SSA", "BASIL-W"]:
                g = a[(a.seed == sd) & (a.M == m)]
                if len(g):
                    r = g.iloc[0]
                    rows.append([str(sd), m if m != "SSR" else "refit", fmt(r.nrmse), fmt(r.slope, 2), fmt(r.auroc), fmt(r.orient, 3), fmt(r.get("do_direct", np.nan), 2) if "do_direct" in r else "--",
                                 fmt(r.get("t_cost", np.nan), 0)])
        body += "\\textbf{(a) $d=500$ (individual datasets).}\n\n" + booktab("llrrrrrr", ["Seed", "Method", "NRMSE", "slope", "AUROC", "orient.", "do direct", "time (s)"], rows, "Two datasets: no significance claims.")
    # Part B: grid at d=100
    b = d[(d.d == 100) & d.gam_eps.notna()]
    if len(b):
        seeds = sorted(set(b.seed))
        ssc = b[b.M == "SSC"].groupby("seed").nrmse.mean()
        rows = []
        for ge, g in sorted(b.groupby("gam_eps"), key=lambda kv: kv[0]):
            n = g.seed.nunique()
            rows.append([f"{ge[0]:g}", f"{ge[1]:g}", str(n), fmt(g.nrmse.mean()), fmt(g.slope.mean(), 2), fmt(g.auroc.mean())])
        bw = rep[(rep.cell == "C1") & (rep.M == "BASIL-W") & rep.seed.isin(seeds)].nrmse.mean()
        sc = rep[(rep.cell == "C1") & (rep.M == "SSC") & rep.seed.isin(seeds)].nrmse.mean()
        sr = rep[(rep.cell == "C1") & (rep.M == "SSR") & rep.seed.isin(seeds)].nrmse.mean()
        body += "\n\\bigskip\n\\textbf{(b) $(\\gamma,\\epsilon)$ grid, $d=100$, uniform, seeds %d--%d.}\n\n" % (min(seeds), max(seeds)) + booktab(
            "llrrrr", ["$\\gamma$", "$\\epsilon$", "$N$", "NRMSE", "slope", "AUROC"], rows,
            "Reference on the same seeds (replication): SSC %s, refit %s, BASIL-W %s. The default $(1,0.01)$ is the configuration used for all headline numbers; the grid is a sensitivity analysis and no setting was selected." % (fmt(sc), fmt(sr), fmt(bw)))
    write_table("tab_adaptive2.tex", body if body else "\\fbox{no complete rows yet}\n")


def t_semisyn():
    d = load_runs("semisyn", recursive=True)
    if d is None:
        pending("tab_semisyn.tex", "research/semisyn/results")
        return
    rows = []
    for (dd, no), g in d.groupby(["d", "noise"]):
        w = g.pivot(index="seed", columns="M", values="nrmse")
        ms = [m for m in ["SSC", "SSA", "SSR", "BASIL-W"] if m in w.columns]
        cells = [fmt(w[m].mean()) if m in w else "--" for m in ["SSC", "SSA", "SSR", "BASIL-W"]]
        au = ["%s" % fmt(g[g.M == m].auroc.mean()) if m in set(g.M) else "--" for m in ["SSC", "SSA", "SSR", "BASIL-W"]]
        rows.append([str(dd), NOISE_LABEL.get(no, no), str(g.seed.nunique())] + cells + ["/".join(au)])
    write_table("tab_semisyn.tex", booktab("llrrrrrl", ["$d$", "Noise", "$N$", "SSC", "SSA", "refit", "BASIL-W", "AUROC (same order)"], rows,
                "Semi-synthetic Sachs benchmark: edge-function NRMSE, mean over datasets (research/semisyn). See the text for the construction and caveats."))


def write_status():
    """Macros with the amount of data available at build time (used by the PENDING status paragraphs)."""
    def count(sub, rec=False, key="seed"):
        d = load_runs(sub, recursive=rec)
        return 0 if d is None else d.drop_duplicates(["d", "noise", "seed"]).shape[0]
    r2 = load_runs("replication2")
    n_full = 0
    if r2 is not None:
        w = r2.pivot_table(index=["noise", "seed"], columns="M", values="nrmse")
        n_full = int(w.dropna(subset=[c for c in ["SSC", "SSA", "SSR", "BASIL-W"] if c in w.columns]).shape[0]) if all(c in w.columns for c in ["SSC", "SSA", "SSR", "BASIL-W"]) else 0
    body = "\\newcommand{\\nRepTwoDone}{%d}\n\\newcommand{\\nAdaTwoDone}{%d}\n\\newcommand{\\nSemiDone}{%d}\n" % (n_full, count("adaptive2", True), count("semisyn", True))
    write_table("status.tex", body)


if __name__ == "__main__":
    rep = load_replication()
    ada = load_adaptive()
    t_main(rep)
    t_struct(rep)
    t_adaptive(rep, ada)
    t_ablation()
    t_bands()
    t_sachs()
    t_negative()
    t_rep2()
    t_adaptive2(rep)
    t_semisyn()
    write_status()
    print("tables written:", sorted(os.listdir(os.path.join(HERE, "tables"))))
