"""Figures 7 and A-baselines: learner results under the fixed start (G1) and a burn-in (G2).

Inputs:  ../../b1y/results/q2.csv.gz  (Q2: tuned DAGMA/NOTEARS/GOLEM, FLOP, SortnRegress; arms G1, G2, VM_G1, VM_G2)
         ../../b1y/results/q5.csv.gz  (Q5: label-free and time-labelled reference methods; arms G1, G2)
Metric:  SHD_x1n = structural Hamming distance per true edge, edges that leave slice 1 excluded (lower is better).
Unit:    summary graph (family, p, seed); mean over T within a graph; 95% intervals by graph-cluster bootstrap
         (2000 resamples, seed 11).
Note:    Q2 runs different methods on different (p, T) cells (NOTEARS only where pT <= 30, GOLEM only at pT <= 50).
         To compare methods on the same data sets, fig_learners uses only the cells every plotted method has:
         (p, T) in {(5,3), (5,5), (10,3)}.  The paper's tables pool each method over its own cells; the sign pattern
         is asserted to be the same below.
Outputs: fig_learners.pdf, fig_baselines.pdf
Run:     /home/user/venv312/bin/python fig_learners.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as fs

fs.setup()
RES = os.path.join(HERE, "..", "..", "b1y", "results")
rng = np.random.default_rng(11)
NB = 2000


def cluster_ci(vals):
    """vals: 1-d array over graphs -> (mean, lo, hi)."""
    n = len(vals)
    idx = rng.integers(0, n, size=(NB, n))
    b = vals[idx].mean(1)
    return vals.mean(), *np.percentile(b, [2.5, 97.5])


def table(df, methods, arms, fam=None):
    d = df[df.method.isin(methods) & df.arm.isin(arms)]
    if fam is not None:
        d = d[d.family == fam]
    g = d.groupby(["family", "p", "seed", "method", "arm"]).shd_x1n.mean().reset_index()
    out = {}
    for (m, a), s in g.groupby(["method", "arm"]):
        out[(m, a)] = cluster_ci(s.shd_x1n.to_numpy())
    return out


# ------------------------------------------------------------------------------------------- Q2 figure
Q2 = pd.read_csv(os.path.join(RES, "q2.csv.gz"))
COMMON = {(5, 3), (5, 5), (10, 3)}
Q2c = Q2[[(p, t) in COMMON for p, t in zip(Q2.p, Q2["T"])]]
M2 = [("FLOP", "FLOP", fs.BLUE), ("DAGMA", "DAGMA (tuned)", fs.ORANGE), ("NOTEARS", "NOTEARS (tuned)", fs.ORANGE),
      ("GOLEMNV", "GOLEM-NV", fs.ORANGE), ("GOLEMEV", "GOLEM-EV", fs.ORANGE), ("VarSnR", "Var-SortnRegress", fs.INK2),
      ("RelSnR", "Rel-SortnRegress", fs.INK2)]
FAMS = [("intree", "in-tree"), ("chain", "chain"), ("ring", "directed ring"), ("ER2", "ER, $c=2$")]


def fig_learners():
    fig, axes = plt.subplots(2, 2, figsize=(fs.W_FULL, 4.2), sharex=True,
                             gridspec_kw={"wspace": 0.62, "hspace": 0.32})
    for ax, (fam, title) in zip(axes.ravel(), FAMS):
        res = table(Q2c, [m for m, _, _ in M2], ["G1", "G2", "VM_G1"], fam)
        ys = list(range(len(M2)))[::-1]
        ax.axvline(1.0, color=fs.INK2, lw=0.7, ls=(0, (3, 2)))
        for y, (m, lab, col) in zip(ys, M2):
            g1, g2, vm = res[(m, "G1")], res[(m, "G2")], res[(m, "VM_G1")]
            ax.plot([g1[0], g2[0]], [y, y], color=col, lw=1.0, alpha=0.55, zorder=2)
            for r, mk, filled in ((g1, "o", True), (g2, "s", True), (vm, "D", False)):
                ax.plot([r[1], r[2]], [y, y], color=col, lw=0.9, zorder=3)
                if m in ("FLOP", "RelSnR") and mk == "D":
                    continue  # rescaling leaves correlation-based scores unchanged
                ax.plot([r[0]], [y + (0.19 if mk == "D" else 0)], mk, color=col, mfc=col if filled else "white",
                        mew=1.0, ms=4.3 if mk != "D" else 3.8, zorder=4)
        ax.set_yticks(ys)
        ax.set_yticklabels([lab for _, lab, _ in M2], fontsize=6.6)
        ax.set_title(title, fontsize=7.8, loc="left", pad=3)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(M2) - 0.3)
        ax.set_xlim(-0.05, 2.1)
    for ax in axes[1]:
        ax.set_xlabel(r"$\mathrm{SHD}_{x1}/|E|$ (lower is better)")
    handles = [Line2D([], [], marker="o", color=fs.INK2, ls="none", ms=4.3, label="G1: fixed start"),
               Line2D([], [], marker="s", color=fs.INK2, ls="none", ms=4.3, label="G2: burn-in"),
               Line2D([], [], marker="D", color=fs.INK2, mfc="white", ls="none", ms=3.8,
                      label="G1 data rescaled to G2's variances"),
               Line2D([], [], color=fs.INK2, ls=(0, (3, 2)), lw=0.7, label="empty graph")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, -0.045), columnspacing=1.2,
               handletextpad=0.3)
    fig.savefig(os.path.join(HERE, "fig_learners.pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return res


# sign-pattern check against the committed analysis (in-trees, own cells): tuned DAGMA, NOTEARS, GOLEM-NV beat FLOP under
# G1 and lose under G2 on the common cells too
def check_sign():
    r = table(Q2c, ["FLOP", "DAGMA", "NOTEARS", "GOLEMNV"], ["G1", "G2"], "intree")
    for m in ("DAGMA", "NOTEARS", "GOLEMNV"):
        assert r[(m, "G1")][0] < r[("FLOP", "G1")][0], (m, r)
        assert r[(m, "G2")][0] > r[("FLOP", "G2")][0], (m, r)


# ------------------------------------------------------------------------------------------- Q5 figure
def fig_baselines():
    Q5 = pd.read_csv(os.path.join(RES, "q5.csv.gz"))
    groups = [
        ("label-free", [("FLOP", "FLOP", fs.BLUE), ("BOSS_tetrad", "BOSS (Tetrad)", fs.ORANGE),
                        ("GRaSP_tetrad", "GRaSP (Tetrad)", fs.ORANGE), ("BOSS", "BOSS (causal-learn)", fs.ORANGE),
                        ("GRaSP", "GRaSP (causal-learn)", fs.ORANGE), ("GES", "GES", fs.ORANGE),
                        ("DirectLiNGAM", "DirectLiNGAM", fs.ORANGE), ("VarSnR", "Var-SortnRegress", fs.INK2),
                        ("RelSnR", "Rel-SortnRegress", fs.INK2)]),
        ("uses the time labels (reference)", [("PCMCI", "PCMCI+", fs.AQUA), ("DYNOTEARS", "DYNOTEARS", fs.AQUA),
                                              ("VARLiNGAM_pooled", "VAR-LiNGAM (pooled)", fs.AQUA),
                                              ("VAR_OLS", "VAR, pooled OLS", fs.AQUA)]),
    ]
    allm = [m for _, g in groups for m, _, _ in g]
    res = table(Q5, allm, ["G1", "G2"])
    n = sum(len(g) for _, g in groups)
    fig, ax = plt.subplots(figsize=(fs.W_FULL, 3.5))
    y = n - 1
    ticks, labs = [], []
    ax.axvline(1.0, color=fs.INK2, lw=0.7, ls=(0, (3, 2)))
    for gi, (gname, g) in enumerate(groups):
        ax.text(2.25, y + 0.15, gname, fontsize=6.6, color=fs.INK2, ha="right", va="bottom", style="italic")
        for m, lab, col in g:
            g1, g2 = res[(m, "G1")], res[(m, "G2")]
            ax.plot([g1[0], g2[0]], [y, y], color=col, lw=1.0, alpha=0.55)
            for r, mk in ((g1, "o"), (g2, "s")):
                ax.plot([r[1], r[2]], [y, y], color=col, lw=0.9)
                ax.plot([r[0]], [y], mk, color=col, ms=4.3, zorder=4)
            ticks.append(y)
            labs.append(lab)
            y -= 1
        y -= 0.7
    ax.set_yticks(ticks)
    ax.set_yticklabels(labs, fontsize=6.8)
    ax.set_xlabel(r"$\mathrm{SHD}_{x1}/|E|$ (lower is better), all families, $p\in\{5,10\}$, $T\in\{3,5\}$")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.05, 2.3)
    ax.set_ylim(y + 0.2, n + 0.9)
    handles = [Line2D([], [], marker="o", color=fs.INK2, ls="none", ms=4.3, label="G1: fixed start"),
               Line2D([], [], marker="s", color=fs.INK2, ls="none", ms=4.3, label="G2: burn-in"),
               Line2D([], [], color=fs.INK2, ls=(0, (3, 2)), lw=0.7, label="empty graph")]
    ax.legend(handles=handles, loc="lower right", fontsize=6.5)
    fig.savefig(os.path.join(HERE, "fig_baselines.pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return res


if __name__ == "__main__":
    check_sign()
    fig_learners()
    r = fig_baselines()
    print({k: tuple(round(float(x), 3) for x in v) for k, v in r.items() if k[0] in ("FLOP", "PCMCI", "DYNOTEARS")})
