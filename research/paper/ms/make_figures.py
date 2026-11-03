"""Generate all manuscript figures (PDF, vector) into figures/. Reads CSVs under research/.
Usage: python3.13 make_figures.py
Figures that depend on experiments that have not finished are drawn from whatever CSVs exist and
carry a visible 'PENDING' note if the data are missing (see SUBMISSION_NOTES.md).
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msdata import (CELLS, COL, HERE, LABEL, MARK, R, SHORT, boot_ci, exists, load_adaptive, load_family,
                    load_replication)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def method_table(rep, ada):
    """long frame: cell, seed, M, nrmse, auroc, do_direct, do_indirect for SSC, SSR, SSA, BASIL-W."""
    keep = ["cell", "seed", "M", "nrmse", "auroc", "orient", "do_direct", "do_indirect"]
    d = rep[keep].copy()
    if ada is not None:
        a = ada[ada.M == "SSA"][keep]
        d = pd.concat([d, a], ignore_index=True)
    return d


# ------------------------------------------------------------------ Figure 1
def fig1(d):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    meths = ["SSC", "SSR", "SSA", "BASIL-W"]
    for cell, _ in CELLS:
        pts = []
        for m in meths:
            g = d[(d.cell == cell) & (d.M == m)]
            if len(g) == 0:
                continue
            pts.append((m, g.auroc.mean(), g.nrmse.mean()))
        ax.plot([p[1] for p in pts if p[0] != "SSA"], [p[2] for p in pts if p[0] != "SSA"], color="0.7", lw=0.6, zorder=1)
    for m in meths:
        g = d[d.M == m].groupby("cell")[["auroc", "nrmse"]].mean()
        ax.scatter(g.auroc, g.nrmse, s=26, color=COL[m], marker=MARK[m], label=LABEL[m], zorder=3, edgecolor="white", lw=0.4)
    ax.set_xlabel("structure: directed AUROC (higher is better)")
    ax.set_ylabel("function: edge-function NRMSE (lower is better)")
    ax.set_title("(a) cell means; grey lines join SSC, refit, BASIL-W")
    ax.legend(loc="upper left", frameon=False)
    # (b) paired bars at d100 uniform, structure vs function side by side
    ax = axs[1]
    c1 = d[d.cell == "C1"]
    x = np.arange(len(meths))
    w = 0.38
    au = [c1[c1.M == m].auroc.mean() for m in meths]
    nr = [c1[c1.M == m].nrmse.mean() for m in meths]
    ax.bar(x - w / 2, au, w, color=[COL[m] for m in meths], alpha=0.95)
    ax.set_ylim(0.9, 1.0)
    ax.set_ylabel("AUROC (solid bars)")
    ax2 = ax.twinx()
    ax2.bar(x + w / 2, nr, w, color=[COL[m] for m in meths], alpha=0.45, hatch="//", edgecolor="white")
    ax2.set_ylim(0, 0.8)
    ax2.set_ylabel("NRMSE (hatched bars)")
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in meths], rotation=20, ha="right")
    ax.set_title("(b) cell C1 (d=100, uniform noise)")
    save(fig, "fig1_structure_vs_function.pdf")


# ------------------------------------------------------------------ Figure 2
def fig2(d, d2=None):
    npan = 2 if d2 is not None else 1
    fig, axs = plt.subplots(1, npan, figsize=(7.0 if npan == 1 else 7.2, 3.0), squeeze=False)
    for ax, dd, title in zip(axs[0], [d, d2], ["(a) fresh seeds 2000-2009 (replication)", "(b) independent DAGs, 20 datasets per cell (replication2)"]):
        if dd is None:
            continue
        meths = ["SSC", "SSR", "SSA", "BASIL-W"]
        off = np.linspace(-0.27, 0.27, len(meths))
        cells = [c for c, _ in CELLS if c in set(dd.cell)]
        for k, m in enumerate(meths):
            xs, ys, lo, hi = [], [], [], []
            for i, c in enumerate(cells):
                g = dd[(dd.cell == c) & (dd.M == m)].nrmse.dropna()
                if len(g) == 0:
                    continue
                ci = boot_ci(g.values)
                xs.append(i + off[k]); ys.append(g.mean()); lo.append(g.mean() - ci[0]); hi.append(ci[1] - g.mean())
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=MARK[m], color=COL[m], ms=4.5, capsize=2, lw=0.9, label=LABEL[m])
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([SHORT[c] for c in cells])
        ax.set_ylabel("edge-function NRMSE")
        ax.set_title(title)
        ax.set_ylim(0, 0.95)
    axs[0][0].legend(loc="upper right", frameon=False, ncol=2)
    if npan == 1:
        axs[0][0].text(0.5, 0.52, "adaptive GL: C1, C2, C3, C8 only (5-10, 3 datasets)", transform=axs[0][0].transAxes,
                       ha="center", fontsize=6.5, color="0.35")
    save(fig, "fig2_edge_nrmse_by_cell.pdf")


# ------------------------------------------------------------------ Figure 3
def fig3():
    p = os.path.join(HERE, "data/example_curves.npz")
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.9), sharex=False)
    fig.subplots_adjust(hspace=0.62, wspace=0.32, top=0.86, bottom=0.12)
    if not exists(p):
        for ax in axs.ravel():
            ax.text(0.5, 0.5, "PENDING\nrun make_example_curves.py", ha="center", va="center", transform=ax.transAxes)
        save(fig, "fig3_example_curves.pdf")
        return
    z = np.load(p)
    for k, ax in enumerate(axs.ravel()):
        x = z[f"e{k}_x"]
        meta = z[f"e{k}_meta"]
        ax.fill_between(x, z[f"e{k}_lo"], z[f"e{k}_hi"], color=COL["BASIL-W"], alpha=0.18, lw=0, label="BASIL-W 90% band (kappa=1)")
        ax.plot(x, z[f"e{k}_truth"], color="k", lw=1.6, label="truth")
        ax.plot(x, z[f"e{k}_ssc"], color=COL["SSC"], lw=1.2, ls="--", label="SPADE-scaled")
        ax.plot(x, z[f"e{k}_basilw"], color=COL["BASIL-W"], lw=1.2, label="BASIL-W")
        ax.set_title(f"x{int(meta[0])} to x{int(meta[1])}\nNRMSE {meta[2]:.2f} (SSC) vs {meta[3]:.2f} (BASIL-W)", fontsize=7)
        if k >= 3:
            ax.set_xlabel("cause (z-scored)")
        if k % 3 == 0:
            ax.set_ylabel("contribution to effect (std. units)")
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"One dataset (d={int(z['d'])}, n={int(z['n'])}, seed {int(z['seed'])}): six true edges drawn at random; "
                 f"\nmean NRMSE over all edges: SSC {float(z['nrmse_ssc_all']):.2f}, BASIL-W {float(z['nrmse_basil_all']):.2f}", fontsize=7, y=0.97)
    save(fig, "fig3_example_curves.pdf")


# ------------------------------------------------------------------ Figure 4
def fig4(d):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    meths = ["SSC", "SSR", "SSA", "BASIL-W"]
    off = np.linspace(-0.27, 0.27, len(meths))
    cells = [c for c, _ in CELLS]
    for ax, col, title in zip(axs, ["do_direct", "do_indirect"], ["(a) direct parents", "(b) indirect (descendant) pairs"]):
        for k, m in enumerate(meths):
            xs, ys, lo, hi = [], [], [], []
            for i, c in enumerate(cells):
                g = d[(d.cell == c) & (d.M == m)][col].dropna()
                if len(g) == 0:
                    continue
                ci = boot_ci(g.values)
                xs.append(i + off[k]); ys.append(g.mean()); lo.append(g.mean() - ci[0]); hi.append(ci[1] - g.mean())
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=MARK[m], color=COL[m], ms=4.5, capsize=2, lw=0.9, label=LABEL[m])
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([SHORT[c] for c in cells])
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
    axs[0].set_ylabel("do-effect NRMSE")
    axs[0].legend(loc="upper right", frameon=False, ncol=2)
    save(fig, "fig4_do_effect.pdf")


# ------------------------------------------------------------------ Figure 5
def fig5():
    s = pd.read_csv(os.path.join(R, "real_data/results/structure.csv"))
    st = pd.read_csv(os.path.join(R, "real_data/results/stability_common.csv"))
    names = {"SSC": "SSC", "SSR_k1.5": "SSR", "BASILW": "BASIL-W", "BASIL": "BASIL", "GAM-ridge": "GAM-ridge", "NOTEARS-linear": "NOTEARS-linear"}
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axs[0]
    ms = list(names)
    for i, m in enumerate(ms):
        v = s[s.method == m].auroc.values
        ci = boot_ci(v)
        ax.errorbar(i, v.mean(), yerr=[[v.mean() - ci[0]], [ci[1] - v.mean()]], fmt=MARK[names[m]], color=COL[names[m]], capsize=2, ms=5)
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.text(len(ms) - 0.6, 0.505, "chance", fontsize=6.5, ha="right")
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels([names[m] for m in ms], rotation=35, ha="right")
    ax.set_ylabel("directed AUROC (18 consensus edges)")
    ax.set_title("(a) structure")
    ax = axs[1]
    sm = ["SSC", "SSR_k1.5", "BASILW", "BASIL", "GAM-ridge"]
    for i, m in enumerate(sm):
        v = st[m].values
        ci = boot_ci(v)
        ax.errorbar(i, v.mean(), yerr=[[v.mean() - ci[0]], [ci[1] - v.mean()]], fmt=MARK[names[m]], color=COL[names[m]], capsize=2, ms=5)
        ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=4, color=COL[names[m]], alpha=0.35, zorder=1)
    ax.set_xticks(range(len(sm)))
    ax.set_xticklabels([names[m] for m in sm], rotation=35, ha="right")
    ax.set_ylabel("curve stability, R$^2_{aff}$")
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) function stability")
    ax = axs[2]
    # (c) per-subsample paired change in AUROC vs SSC
    base = s[s.method == "SSC"].set_index("sub").auroc
    for i, m in enumerate(["SSR_k1.5", "BASILW", "BASIL", "NOTEARS-linear"]):
        v = (s[s.method == m].set_index("sub").auroc - base).values
        ci = boot_ci(v)
        ax.errorbar(i, v.mean(), yerr=[[v.mean() - ci[0]], [ci[1] - v.mean()]], fmt=MARK[names[m]], color=COL[names[m]], capsize=2, ms=5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["SSR", "BASIL-W", "BASIL", "NOTEARS-lin."], rotation=35, ha="right")
    ax.set_ylabel("AUROC minus SSC")
    ax.set_title("(c) paired change vs SSC")
    fig.text(0.5, -0.04, "Sachs, d=11, 10 overlapping subsamples of 2000 cells; intervals: bootstrap over subsamples (optimistic).", ha="center", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig5_sachs.pdf")


# ------------------------------------------------------------------ Figure 6
def fig6():
    base = os.path.join(R, "ablation/results")
    def load(sub, name):
        p = os.path.join(base, sub, name)
        return pd.read_csv(p, usecols=["variant", "seed", "P", "cov_1.0", "cov_1.55"]) if exists(p) else None
    sets = [("uniform", load("test", "unif_edges.csv")), ("Laplace", load("laplace", "laplace_edges.csv")), ("hetero", load("hetero", "hetero_edges.csv"))]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    groups = [("all true edges", lambda e: np.ones(len(e), bool)), ("found (P>0.9)", lambda e: e.P.values > 0.9), ("missed (P<=0.9)", lambda e: e.P.values <= 0.9)]
    w = 0.36
    pos = 0
    ticks, labs = [], []
    for nm, e in sets:
        if e is None:
            continue
        e = e[e.variant == "base"]
        for gi, (gn, sel) in enumerate(groups):
            vals = {}
            for kk in ["cov_1.0", "cov_1.55"]:
                e2 = e[sel(e)]
                vals[kk] = (e2[kk].mean() if gn != "all true edges" else e2.groupby("seed")[kk].mean().mean()) if len(e2) else np.nan
            ax.bar(pos - w / 2, vals["cov_1.0"], w, color="#999999")
            ax.bar(pos + w / 2, vals["cov_1.55"], w, color=COL["BASIL-W"])
            ticks.append(pos); labs.append(f"{nm}\n{gn}")
            pos += 1
        pos += 0.6
    ax.axhline(0.9, color="k", ls=":", lw=0.9)
    ax.text(pos - 0.7, 0.8, "nominal 0.90", fontsize=6.5, ha="right", color="k")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#999999", label="kappa = 1"), Patch(color=COL["BASIL-W"], label="kappa = 1.55 (frozen on validation)")],
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=6)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("pointwise band coverage (test datasets)")
    ax.set_title("(a) before / after calibration")
    ax = axs[1]
    KG = np.round(np.arange(1.0, 3.0001, 0.05), 2)
    pv = os.path.join(base, "val/unif_edges.csv")
    if exists(pv):
        cols = ["variant", "seed"] + [f"cov_{k}" for k in KG if f"cov_{k}" in pd.read_csv(pv, nrows=1).columns or True]
        head = pd.read_csv(pv, nrows=1).columns
        cols = ["variant", "seed"] + [f"cov_{k:g}" if f"cov_{k:g}" in head else f"cov_{k:.1f}" for k in KG]
        cols = [c for c in cols if c in head]
        v = pd.read_csv(pv, usecols=cols)
        v = v[v.variant == "base"]
        ks = [float(c.split("_")[1]) for c in cols[2:]]
        ax.plot(ks, [v.groupby("seed")[c].mean().mean() for c in cols[2:]], color="k", lw=1.4, label="validation (5 datasets)")
    pt = os.path.join(base, "test/unif_edges.csv")
    if exists(pt):
        head = pd.read_csv(pt, nrows=1).columns
        cols = [c for c in ["variant", "seed"] + [f"cov_{k:g}" if f"cov_{k:g}" in head else f"cov_{k:.1f}" for k in KG] if c in head]
        t = pd.read_csv(pt, usecols=cols)
        t = t[t.variant == "base"]
        ks = [float(c.split("_")[1]) for c in cols[2:]]
        ax.plot(ks, [t.groupby("seed")[c].mean().mean() for c in cols[2:]], color=COL["BASIL-W"], lw=1.4, ls="--", label="test, uniform (8, descriptive)")
    ax.axhline(0.9, color="k", ls=":", lw=0.9)
    ax.axvline(1.55, color="0.5", lw=0.8)
    ax.set_xlabel("variance inflation kappa")
    ax.set_ylabel("pointwise coverage")
    ax.set_ylim(0.75, 0.95)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("(b) coverage vs kappa (plateau: missed edges)")
    fig.tight_layout()
    save(fig, "fig6_band_coverage.pdf")


# ------------------------------------------------------------------ Figure 7 (extra): adaptive sensitivity, if present
def fig7():
    a2 = load_family("adaptive2")
    if a2 is None or "method" not in a2.columns:
        return
    # drawn only if the adaptive2 results exist and carry the columns used below; see SUBMISSION_NOTES.md
    return


if __name__ == "__main__":
    rep = load_replication()
    ada = load_adaptive()
    d = method_table(rep, ada)
    rep2 = None
    p2 = os.path.join(R, "replication2/results/all.csv")
    if exists(p2):
        r2 = pd.read_csv(p2)
        if {"cell", "seed", "M", "nrmse"} <= set(r2.columns):
            rep2 = r2.drop_duplicates(["cell", "seed", "M"])
    fig1(d)
    fig2(d, rep2)
    fig3()
    fig4(d)
    fig5()
    fig6()
    fig7()
