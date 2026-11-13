"""Figure of Section 5.1 (post hoc oracle): observed gain of the fixed start against the exact gap and against the
detectability oracle, one point per (family, p, T) cell.

Input:  ../../b1x/results/oracle_b3.json (key "cells": family, T, gap, prob, obs)
Output: fig_oracle.pdf
Run:    /home/user/venv312/bin/python make_fig_oracle.py
"""
import json
import os
import sys

import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as fs

fs.setup()
C = json.load(open(os.path.join(HERE, "..", "..", "b1x", "results", "oracle_b3.json")))["cells"]
SC = {"symER", "ring"}
fig, axes = plt.subplots(1, 2, figsize=(fs.W_FULL * 0.8, 2.3), sharey=True, gridspec_kw={"wspace": 0.12})
for k, (pred, lab) in enumerate((("gap", r"exact gap $s_\rho(G_T)-s_\rho^{W}$"),
                                 ("prob", "detectability oracle (expected gain)"))):
    ax = axes[k]
    ax.plot([0, 0.55], [0, 0.55], color=fs.INK2, lw=0.8, ls=(0, (3, 2)))
    for is_sc, col, name in ((False, fs.BLUE, "not strongly connected"), (True, fs.ORANGE, "strongly connected")):
        c = [r for r in C if (r["family"] in SC) == is_sc]
        ax.scatter([r[pred] for r in c], [r["obs"] for r in c], s=[6 + 1.6 * r["T"] for r in c], color=col, alpha=0.85,
                   lw=0, label=name, zorder=3)
    x = [r[pred] for r in C]
    y = [r["obs"] for r in C]
    ax.set_title(f"Spearman {stats.spearmanr(x, y).statistic:.2f}, $R^2$ {stats.pearsonr(x, y).statistic ** 2:.2f}", fontsize=7.5)
    ax.set_xlabel(lab)
    ax.set_xlim(0, 0.55)
    ax.set_ylim(-0.03, 0.55)
axes[0].set_ylabel("observed gain in order accuracy\n(rel-SortnRegress, G1 $-$ G2)")
axes[0].legend(loc="upper left", fontsize=6.3, handletextpad=0.2, borderaxespad=0.1)
fs.panel_label(axes[0], "(a)", dx=-0.22)
fs.panel_label(axes[1], "(b)", dx=-0.03)
fig.savefig(os.path.join(HERE, "fig_oracle.pdf"), bbox_inches="tight", pad_inches=0.03)
print("ok", len(C), "cells")
