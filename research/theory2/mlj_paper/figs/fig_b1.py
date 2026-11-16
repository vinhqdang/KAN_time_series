"""B1 and B1-NL appendix figures in the house style (TrueType fonts).

Redraws figs/fig_b1.pdf and figs/fig_b1nl.pdf from the committed result files
  ../../b1/results/b1_test.csv.gz      (B1, regime 'het', generators G1 and G2)
  ../../b1nl/results/b1nl_test.csv.gz  (B1-NL, configurations NL, SGN, SGNU)
with the same statistics as b1/analyse_b1.py and b1nl/analyse_b1nl.py, and
asserts that the per-cell Spearman correlations agree with the committed
analysis.json files. No randomness.
Run: python fig_b1.py   (from mlj_paper/figs)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import figstyle as fs

FAMILIES = ["symER", "ring", "chain", "outtree", "intree", "ER1", "ER2"]
SC_FAMILIES = ["symER", "ring"]
FAMLAB = dict(symER="symmetric ER", ring="ring", chain="chain", outtree="out-tree", intree="in-tree",
              ER1="ER($c{=}1$)", ER2="ER($c{=}2$)")
FAMCOL = dict(symER=fs.AQUA, ring=fs.VIOLET, chain=fs.ORANGE, outtree=fs.BLUE, intree=fs.MAGENTA,
              ER1="#8a6d3b", ER2=fs.INK2)
MNAME = dict(M1="rel-SortnRegress", M2="$R^2$-SortnRegress", M3="Var-SortnRegress", M4="Random-SortnRegress",
             M5="FLOP", R0="Time-SortnRegress (labels)")
MCOL = dict(M1=fs.BLUE, M2=fs.AQUA, M3=fs.ORANGE, M4=fs.INK2, M5=fs.VIOLET, R0=fs.MAGENTA)
XLAB = r"exact gap $s_\rho(G_T)-s_\rho^W$ (Theorems 2 and 5)"
YLAB = "rel-SortnRegress order accuracy,\nfixed start $-$ burn-in"


def cells(a, b):
    D = pd.DataFrame(index=a.index)
    D["d"] = a.M1_oa - b.M1_oa
    D["gap"] = a.s_exact - b.s_exact
    D = D.reset_index()
    return D.groupby(["family", "p", "T"]).agg(d=("d", "mean"), gap=("gap", "mean")).reset_index()


def scatter_cells(ax, C):
    for fam in FAMILIES:
        c = C[C.family == fam]
        ax.scatter(c.gap, c.d, s=6 + 1.6 * c["T"], color=FAMCOL[fam], marker="s" if fam in SC_FAMILIES else "o",
                   alpha=0.9, linewidths=0, zorder=3)
    lim = [0, 0.52]
    ax.plot(lim, lim, color=fs.INK2, lw=0.7, ls=":", zorder=2)
    ax.axhline(0, color=fs.INK2, lw=0.5, zorder=2)
    ax.set_xlim(-0.01, 0.53)


def fam_legend(ax, loc="upper left"):
    h = [Line2D([], [], marker="s" if f in SC_FAMILIES else "o", ls="", color=FAMCOL[f], markersize=4, label=FAMLAB[f])
         for f in FAMILIES]
    ax.legend(handles=h, loc=loc, ncol=1, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25)


def fig_b1():
    df = pd.read_csv(os.path.join(ROOT, "b1", "results", "b1_test.csv.gz"))
    a = df[(df.regime == "het") & (df.gen == "G1")].set_index(["family", "p", "seed", "T"])
    b = df[(df.regime == "het") & (df.gen == "G2")].set_index(["family", "p", "seed", "T"]).loc[a.index]
    C = cells(a, b)
    rho = stats.spearmanr(C.gap, C.d).correlation
    ref = json.load(open(os.path.join(ROOT, "b1", "results", "analysis.json")))["H1"]["spearman_cells"]
    assert abs(rho - ref) < 1e-9, (rho, ref)
    assert (C.d < C.gap + 1e-12).all() and len(C) == 42

    fig, ax = plt.subplots(1, 2, figsize=(fs.W_FULL, 3.0), gridspec_kw=dict(width_ratios=[1, 1.25]))
    scatter_cells(ax[0], C)
    ax[0].set_xlabel(XLAB)
    ax[0].set_ylabel(YLAB)
    ax[0].set_title(f"(a) 42 cells; Spearman {rho:.2f}", loc="left")
    fam_legend(ax[0])

    ms = ["M1", "M2", "M3", "M4", "M5", "R0"]
    x = np.arange(len(FAMILIES))
    w = 0.13
    for k, m in enumerate(ms):
        v1 = [a[a.index.get_level_values(0) == f][f"{m}_sidn"].mean() for f in FAMILIES]
        v2 = [b[b.index.get_level_values(0) == f][f"{m}_sidn"].mean() for f in FAMILIES]
        xs = x + (k - 2.5) * w
        for xi, y1, y2 in zip(xs, v1, v2):
            ax[1].plot([xi, xi], [y1, y2], color=MCOL[m], lw=0.8, zorder=2)
        ax[1].scatter(xs, v1, marker="o", color=MCOL[m], s=9, zorder=3, linewidths=0)
        ax[1].scatter(xs, v2, marker="s", facecolor="white", edgecolor=MCOL[m], s=9, zorder=3, linewidths=0.8)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([FAMLAB[f] for f in FAMILIES], rotation=30, ha="right")
    ax[1].set_ylabel("normalised SID (lower is better)")
    ax[1].grid(axis="x", visible=False)
    h = [Line2D([], [], color=MCOL[m], lw=1.2, label=MNAME[m]) for m in ms]
    h += [Line2D([], [], marker="o", ls="", color=fs.INK2, markersize=3.5, label="fixed start (G1)"),
          Line2D([], [], marker="s", ls="", markerfacecolor="white", markeredgecolor=fs.INK2, markersize=3.5,
                 label="burn-in (G2)")]
    ax[1].set_title("(b) normalised SID by family and method", loc="left")
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=6.3, columnspacing=1.0,
               handlelength=1.4, handletextpad=0.3)
    fig.tight_layout(w_pad=1.2, rect=(0, 0.13, 1, 1))
    fig.savefig(os.path.join(HERE, "fig_b1.pdf"), bbox_inches="tight")
    plt.close(fig)
    return rho


def fig_b1nl():
    df = pd.read_csv(os.path.join(ROOT, "b1nl", "results", "b1nl_test.csv.gz"))
    ref = json.load(open(os.path.join(ROOT, "b1nl", "results", "analysis.json")))
    cfgs = [c for c in ["NL", "SGN", "SGNU"] if (df.cfg == c).any()]
    titles = dict(NL="nonlinear (NL)", SGN="signed weights (SGN)", SGNU="signed, uniform (SGNU)")
    fig, ax = plt.subplots(1, len(cfgs), figsize=(fs.W_FULL, 2.35), sharey=True)
    out = {}
    for k, cfg in enumerate(cfgs):
        d = df[df.cfg == cfg]
        a = d[d.gen == "G1"].set_index(["family", "p", "seed", "T"]).sort_index()
        b = d[d.gen == "G2"].set_index(["family", "p", "seed", "T"]).loc[a.index]
        C = cells(a, b)
        rho = stats.spearmanr(C.gap, C.d).correlation
        assert abs(rho - ref[cfg]["H1"]["spearman_cells"]) < 1e-9, (cfg, rho)
        out[cfg] = rho
        scatter_cells(ax[k], C)
        ax[k].set_title(f"{titles[cfg]}\nSpearman {rho:.2f}", fontsize=7)
        ax[k].set_xlabel(r"exact gap $s_\rho(G_T)-s_\rho^W$")
    ax[0].set_ylabel(YLAB)
    fam_legend(ax[0])
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(HERE, "fig_b1nl.pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    fs.setup()
    print("B1 Spearman", round(fig_b1(), 4))
    print("B1-NL Spearman", {k: round(v, 4) for k, v in fig_b1nl().items()})
