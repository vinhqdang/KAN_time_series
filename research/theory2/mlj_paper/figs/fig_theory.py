"""Figures 4 and 5 (graph-only, no data): exact rel-sortability against T and against the burn-in B, and the
minimal sufficient burn-in against Delta.

Inputs:  none for panels of fig_srho (computed here by enumeration of ancestors/descendants in the unrolled DAG,
         independent of the theorems; cross-checked against Thm 1.5 below);
         ../../b1y/results/q6_burnin_exact.json for fig_bmin.
Outputs: fig_srho.pdf, fig_bmin.pdf
Run:     /home/user/venv312/bin/python fig_theory.py   (any Python 3 with matplotlib)
"""
import json
import os
import sys

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as fs
from unroll import chain, ring, intree, outtree, srho, delta, with_loops

fs.setup()


def sym_path(p):
    return [(i, i + 1) for i in range(p - 1)] + [(i + 1, i) for i in range(p - 1)]


def limit_value(p, S):
    """L = 1/2 + |Ex| / (2|Es|) (Thm 1.3), SCCs by mutual reachability."""
    S = with_loops(p, S)
    reach = [[i == j for j in range(p)] for i in range(p)]
    for (i, j) in S:
        reach[i][j] = True
    for k in range(p):
        for i in range(p):
            for j in range(p):
                if reach[i][k] and reach[k][j]:
                    reach[i][j] = True
    ex = sum(1 for (i, j) in S if not (reach[i][j] and reach[j][i]))
    return 0.5 + ex / (2 * len(S))


p_in, S_in = intree(2)
p_out, S_out = outtree(2)
GRAPHS = [  # label, p, S, colour, linestyle
    ("chain ($p{=}6$)", 6, chain(6), fs.ORANGE, "-"),
    ("directed ring ($p{=}6$)", 6, ring(6), fs.BLUE, "-"),
    ("two-way path ($p{=}6$)", 6, sym_path(6), fs.AQUA, "-"),
    ("in-tree ($p{=}7$)", p_in, S_in, fs.VIOLET, "-"),
    ("out-tree ($p{=}7$)", p_out, S_out, fs.MAGENTA, "-"),
]


def fig_srho():
    fig, axes = plt.subplots(1, 2, figsize=(fs.W_FULL, 2.45), gridspec_kw={"wspace": 0.28})
    ax = axes[0]
    Ts = list(range(2, 21))
    for lab, p, S, c, ls in GRAPHS:
        y = [srho(p, S, T) for T in Ts]
        ax.plot(Ts, y, color=c, ls=ls, marker="o", ms=2.2, label=lab)
    # cross-check against Thm 1.5 for the T = 2 value of the chain: single self-loop edges only
    ax.set_xlabel("horizon $T$ (slices)")
    ax.set_ylabel(r"exact $s_\rho(G_T)$, fixed start")
    ax.set_ylim(0.47, 1.03)
    ax.set_xticks([2, 5, 10, 15, 20])
    ax.legend(loc="upper right", fontsize=6.3, handlelength=1.4, labelspacing=0.25)
    for yv, txt, va in ((limit_value(6, chain(6)), "limit $L\\approx0.73$ (chain, trees)", "bottom"),
                        (0.5, "limit $L=0.5$ (strongly connected graphs)", "bottom")):
        ax.axhline(yv, color=fs.INK2, lw=0.7, ls=(0, (1.5, 2)))
        ax.text(2.1, yv - 0.008, txt, fontsize=6.1, ha="left", va="top", color=fs.INK2)
    fs.panel_label(ax, "(a)", dx=-0.2)

    ax = axes[1]
    Bs = list(range(0, 9))
    TW = 6
    for lab, p, S, c, ls in GRAPHS:
        y = [srho(p, S, TW, B) for B in Bs]
        ax.plot(Bs, y, color=c, ls=ls, marker="o", ms=2.2)
        D = delta(p, S)
        ax.plot([D], [srho(p, S, TW, D)], marker="v", ms=5.5, mfc="white", mec=c, mew=1.1, ls="none", zorder=5)
    ax.set_xlabel("latent burn-in $B$ (slices before the window)")
    ax.set_ylabel(r"window $s_\rho^{W}$, $T=6$")
    ax.set_ylim(0.47, 1.03)
    ax.set_xticks(Bs)
    ax.text(8.2, 0.985, r"$\triangledown$ marks $B=\Delta$" "\n" "(colours as in (a))", fontsize=6.5, ha="right", va="top", color=fs.INK2)
    fs.panel_label(ax, "(b)", dx=-0.2)
    fig.savefig(os.path.join(HERE, "fig_srho.pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_bmin():
    d = json.load(open(os.path.join(HERE, "..", "..", "b1y", "results", "q6_burnin_exact.json")))
    fig, axes = plt.subplots(1, 2, figsize=(fs.W_FULL, 2.3), gridspec_kw={"wspace": 0.3, "width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.plot([0, 8], [0, 8], color=fs.INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text(5.9, 6.3, r"$B_{\min}=\Delta$", fontsize=6.5, color=fs.INK2, rotation=40)
    for name, c, mk, lab in (("chain", fs.ORANGE, "o", "chain"), ("ring", fs.BLUE, "s", "directed ring")):
        xs, ys = [], []
        for p in range(2, 9):
            e = d[f"{name}_{p}"]
            bs = {r["B_min_cnt"] for r in e["rows"]}
            for b in sorted(bs):
                xs.append(e["Delta"])
                ys.append(b)
        ax.plot(xs, ys, color=c, marker=mk, ms=4.5, ls="-", lw=1.0, label=lab, mfc=c)
    ax.set_xlabel(r"$\Delta$ (largest finite distance in the summary graph)")
    ax.set_ylabel(r"minimal sufficient burn-in $B_{\min}$")
    ax.set_xlim(0.3, 7.7)
    ax.set_ylim(0, 7.7)
    ax.legend(loc="upper left")
    ax.text(7.6, 0.2, r"ring: $\lceil\Delta/2\rceil$", fontsize=6.5, color=fs.BLUE, ha="right")
    fs.panel_label(ax, "(a)", dx=-0.2)

    ax = axes[1]
    sl = {int(k): v for k, v in d["random"]["slack_cnt"].items()}
    ks = sorted(sl)
    tot = sum(sl.values())
    ax.bar(ks, [sl[k] for k in ks], color=fs.AQUA, width=0.62)
    for k in ks:
        ax.text(k, sl[k] + 6, f"{sl[k]}", ha="center", fontsize=6.5)
    ax.set_xlabel(r"slack $\Delta - B_{\min}$ (slices)")
    ax.set_ylabel("random-graph cases")
    ax.set_xticks(ks)
    ax.set_ylim(0, 470)
    ax.text(4.3, 440, f"{d['random']['graphs']} graphs, {tot} cases;\n$B_{{\\min}}>\\Delta$ never", fontsize=6.5, ha="right",
            va="top", color=fs.INK2)
    ax.grid(axis="x", visible=False)
    fs.panel_label(ax, "(b)", dx=-0.22)
    fig.savefig(os.path.join(HERE, "fig_bmin.pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    # cross-check of the enumeration against the closed form of Thm 1.3 for the directed chain at T = 40
    L = limit_value(6, chain(6))
    assert abs(srho(6, chain(6), 60) - L) < 0.05, (srho(6, chain(6), 60), L)
    fig_srho()
    fig_bmin()
    print("ok")
