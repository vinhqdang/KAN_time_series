"""Figures for Section 5.2 (controlled tests): fig_controls.pdf (exact gap vs realised gain; arm retention) and
fig_twofactor.pdf (detectability schematic; realised share Q against n and against the spectral-radius cap).

Input:  ../../b1y/results/q1.csv.gz (committed Q1 results; columns study, family, p, seed, T, n, cap, arm, oa_x1, gap)
        ../../b1y/results/analysis_Q1.json (committed verdicts, used as a cross-check of the recomputed numbers)
Method: unit = summary graph (family, p, seed); 95% intervals by graph-cluster bootstrap (2000 resamples, seed 7);
        Q = mean over graphs of gain/gap, as in b1y/analyse_y.py (q1_dose).
Run:    /home/user/venv312/bin/python fig_q1.py
"""
import json
import os
import sys
from math import sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as fs

fs.setup()
RES = os.path.join(HERE, "..", "..", "b1y", "results")
D = pd.read_csv(os.path.join(RES, "q1.csv.gz"))
AJ = json.load(open(os.path.join(RES, "analysis_Q1.json")))
NONSC = ["chain", "outtree", "intree", "ER1", "ER2"]
rng = np.random.default_rng(7)
NB = 2000


def graph_table(df, arms, value="oa_x1"):
    """rows = graphs (family,p,seed), columns = arms: mean over T of `value`; plus the exact gap (mean over T)."""
    d = df[df.arm.isin(arms)]
    t = d.pivot_table(index=["family", "p", "seed"], columns="arm", values=value, aggfunc="mean")
    g = df.groupby(["family", "p", "seed"]).gap.mean()
    return t.join(g)


def boot_ratio(num, den):
    """mean(num)/mean(den) with a graph-cluster bootstrap."""
    n = len(num)
    idx = rng.integers(0, n, size=(NB, n))
    r = num[idx].mean(1) / den[idx].mean(1)
    return num.mean() / den.mean(), np.percentile(r, [2.5, 97.5])


# ------------------------------------------------------------------------------------------- data
main = D[(D.study == "main") & D.family.isin(NONSC)]
arms = ["G1", "G1rep", "G2", "CF1_0.25", "CF1_0.5", "CF1_2", "CF1_4", "CF2_m1", "CF2_m3", "CF2_m10"]
gt = graph_table(main, arms).dropna(subset=["G1", "G2"])
den = (gt["G1"] - gt["G2"]).to_numpy()
ret = {}
for a in arms:
    if a in ("G1", "G2"):
        continue
    ret[a] = boot_ratio((gt[a] - gt["G2"]).to_numpy(), den)
# cross-check against the committed verdicts
Q1 = AJ["Q1_nonSC"]
assert abs(ret["CF2_m1"][0] - Q1["V2_pattern_retention_CF2_m1"]["R"]) < 5e-3
assert abs(ret["CF2_m10"][0] - Q1["retention_CF2_m10"]["R"]) < 5e-3

# cells (family, p, T): exact gap and realised gain at n = 1000
mall = D[(D.study == "main") & D.arm.isin(["G1", "G2"])]
cell = mall.pivot_table(index=["family", "p", "T"], columns="arm", values="oa_x1", aggfunc="mean")
cell = cell.join(D[(D.study == "main") & (D.arm == "G1")].groupby(["family", "p", "T"]).gap.mean()).reset_index()
cell["gain"] = cell["G1"] - cell["G2"]


def q_of(df):
    t = graph_table(df, ["G1", "G2s"]).dropna()
    r = ((t["G1"] - t["G2s"]) / t["gap"]).to_numpy()
    idx = rng.integers(0, len(r), size=(NB, len(r)))
    return r.mean(), np.percentile(r[idx].mean(1), [2.5, 97.5])


nd = D[D.study == "ndose"]
Qn = {int(n): q_of(nd[nd.n == n]) for n in sorted(nd.n.unique())}
rd = D[D.study == "rhodose"]
Qc = {float(c): q_of(rd[rd.cap == c]) for c in sorted(rd.cap.unique())}
Qc[0.8] = Qn[1000]  # the n-dose study at n = 1000 uses the cap 0.8 and the same graphs
V4, V5 = AJ["dose"]["V4_n_dose"]["Q"], AJ["dose"]["V5_rho_dose"]["Q"]
assert abs(Qn[200][0] - V4["200"]) < 5e-3 and abs(Qn[25000][0] - V4["25000"]) < 5e-3
assert abs(Qc[0.5][0] - V5["0.5"]) < 5e-3 and abs(Qc[0.95][0] - V5["0.95"]) < 5e-3


# ------------------------------------------------------------------------------------------- figure A
def fig_controls():
    fig, axes = plt.subplots(1, 2, figsize=(fs.W_FULL, 2.55), gridspec_kw={"wspace": 0.95, "width_ratios": [1, 0.95]})
    ax = axes[0]
    sc = cell.family.isin(["symER", "ring"])
    ax.plot([0, 0.55], [0, 0.55], color=fs.INK2, lw=0.8, ls=(0, (3, 2)))
    ax.scatter(cell.gap[~sc], cell.gain[~sc], s=14, color=fs.BLUE, alpha=0.85, lw=0, label="not strongly connected", zorder=3)
    ax.scatter(cell.gap[sc], cell.gain[sc], s=14, color=fs.ORANGE, alpha=0.85, lw=0, label="strongly connected", zorder=3)
    ax.set_xlabel(r"exact gap $s_\rho(G_T)-s_\rho^{W}$")
    ax.set_ylabel("realised gain in order accuracy\n(fixed start $-$ burn-in), $n=1000$")
    ax.set_xlim(0, 0.55)
    ax.set_ylim(-0.03, 0.55)
    ax.legend(loc="upper left", fontsize=6.3, handletextpad=0.2, borderaxespad=0.1)
    ax.text(0.545, 0.41, "ceiling:\ngain = gap", fontsize=6.3, color=fs.INK2, ha="right", va="top")
    fs.panel_label(ax, "(a)", dx=-0.3)

    ax = axes[1]
    order = ["G1rep", "CF1_0.25", "CF1_0.5", "CF1_2", "CF1_4", "CF2_m1", "CF2_m3", "CF2_m10"]
    labels = {"G1rep": "negative control", "CF1_0.25": r"G1 pattern, strength $\times0.25$",
              "CF1_0.5": r"G1 pattern, strength $\times0.5$", "CF1_2": r"G1 pattern, strength $\times2$",
              "CF1_4": r"G1 pattern, strength $\times4$", "CF2_m1": "burn-in pattern, undetectable",
              "CF2_m3": "burn-in pattern, weak", "CF2_m10": "burn-in pattern, strong"}
    cols = {"G1rep": fs.INK2, "CF1": fs.BLUE, "CF2": fs.ORANGE}
    ys = list(range(len(order)))[::-1]
    ax.axvline(1, color=fs.INK2, lw=0.7, ls=(0, (3, 2)))
    ax.axvline(0, color=fs.INK2, lw=0.7, ls=(0, (3, 2)))
    for y, a in zip(ys, order):
        r, (lo, hi) = ret[a]
        c = cols["G1rep"] if a == "G1rep" else cols[a[:3]]
        ax.plot([lo, hi], [y, y], color=c, lw=1.3)
        ax.plot([r], [y], "o", color=c, ms=4.2)
    ax.set_yticks(ys)
    ax.set_yticklabels([labels[a] for a in order], fontsize=6.4)
    ax.set_xlabel("share of the fixed-start gain retained\n(0 = burn-in level, 1 = fixed-start level)", labelpad=1)
    ax.set_xlim(-0.1, 1.45)
    ax.set_xticks([0, 0.5, 1.0])
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, ys[0] + 0.6)
    fs.panel_label(ax, "(b)", dx=-1.28)
    fig.savefig(os.path.join(HERE, "fig_controls.pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ------------------------------------------------------------------------------------------- figure B
def fig_twofactor():
    fig, axes = plt.subplots(1, 3, figsize=(fs.W_FULL, 2.3), gridspec_kw={"wspace": 0.52, "width_ratios": [1.15, 1, 0.8]})
    ax = axes[0]
    k = np.arange(0, 9)
    ax.plot(k, 0.72 ** k, "o-", color=fs.BLUE, ms=3.2, lw=1.1)
    for n, ls, lab in ((200, (0, (4, 2)), r"$\varepsilon$, $n=200$"), (1000, "-", r"$\varepsilon$, $n=1000$"),
                       (25000, (0, (1, 1.5)), r"$\varepsilon$, $n=25\,000$")):
        tc = student_t.ppf(0.975, n - 2)
        eps = tc / sqrt(n - 2 + tc * tc)
        ax.axhline(eps, color=fs.ORANGE, lw=1.0, ls=ls)
        ax.text(-0.1, eps * 1.1, lab, fontsize=6.2, color=fs.ORANGE, ha="left", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(0.006, 1.3)
    ax.set_xlim(-0.2, 8.2)
    ax.set_xlabel("graph distance between two relatives")
    ax.set_ylabel(r"$|$correlation$|$")
    ax.text(8.1, 0.62, "illustrative:\n$0.72^{\\,\\mathrm{distance}}$", fontsize=6.2, color=fs.BLUE, ha="right", va="top")
    fs.panel_label(ax, "(a)", dx=-0.3)

    ax = axes[1]
    ns = sorted(Qn)
    q = np.array([Qn[n][0] for n in ns])
    lo = np.array([Qn[n][1][0] for n in ns])
    hi = np.array([Qn[n][1][1] for n in ns])
    ax.axhline(1, color=fs.INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text(ns[0], 1.02, "exact gap (ceiling)", fontsize=6.3, color=fs.INK2, va="bottom")
    ax.fill_between(ns, lo, hi, color=fs.BLUE, alpha=0.18, lw=0)
    ax.plot(ns, q, "o-", color=fs.BLUE, ms=3.6)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels(["200", "1000", "5000", "25 000"], fontsize=6.5)
    ax.minorticks_off()
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("sample size $n$")
    ax.set_ylabel(r"realised share $Q$ of the exact gap")
    fs.panel_label(ax, "(b)", dx=-0.3)

    ax = axes[2]
    cs = sorted(Qc)
    q = np.array([Qc[c][0] for c in cs])
    lo = np.array([Qc[c][1][0] for c in cs])
    hi = np.array([Qc[c][1][1] for c in cs])
    ax.axhline(1, color=fs.INK2, lw=0.8, ls=(0, (3, 2)))
    ax.fill_between(cs, lo, hi, color=fs.BLUE, alpha=0.18, lw=0)
    ax.plot(cs, q, "o-", color=fs.BLUE, ms=3.6)
    ax.set_xticks(cs)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("spectral-radius cap")
    ax.set_ylabel(r"$Q$ at $n=1000$")
    fs.panel_label(ax, "(c)", dx=-0.32)
    fig.savefig(os.path.join(HERE, "fig_twofactor.pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    print("retention (non-SC pooled):", {a: round(float(v[0]), 3) for a, v in ret.items()})
    print("Q(n):", {n: round(float(v[0]), 3) for n, v in Qn.items()}, "Q(cap):", {c: round(float(v[0]), 3) for c, v in Qc.items()})
    fig_controls()
    fig_twofactor()
