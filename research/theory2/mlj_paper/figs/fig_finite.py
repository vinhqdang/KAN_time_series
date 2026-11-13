"""Figure 11 (Section 4): effective sample size under autocorrelation, size of the nominal and plug-in tests, and
recovery of the relatives set.

Input:  ../../finite2/sim_finite2_results.txt (committed output of finite2/sim_finite2.py, fixed seeds)
        panel (a) is the closed form W = (1 + lam^2) / (1 - lam^2) of Prop. 14 for two independent AR(1) series with
        the same coefficient lam (sum_h lam^{2h} over h in Z), so n_eff / n = 1 / W.
Output: fig_finite.pdf
Run:    /home/user/venv312/bin/python fig_finite.py
"""
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as fs

fs.setup()
TXT = open(os.path.join(HERE, "..", "..", "finite2", "sim_finite2_results.txt")).read().splitlines()


def parse_size():
    """rows 'z=1.96 ... lam=(a,a) n=.. W=..: nominal x[..] plugin_M y[..] plugin_K50 z[..] oracle w[..]' for same-sign pairs."""
    out = {}
    pat = re.compile(r"^z=1\.96 .*lam=\(([\d.]+),([\d.]+)\) n=(\d+) W=([\d.]+).*?nominal ([\d.]+)\[.*?plugin_M ([\d.]+)\[.*?"
                     r"plugin_K50 ([\d.]+)\[.*?oracle ([\d.]+)\[")
    for line in TXT:
        m = pat.match(line)
        if m and m.group(1) == m.group(2):
            lam, n = float(m.group(1)), int(m.group(3))
            out[(lam, n)] = dict(W=float(m.group(4)), nominal=float(m.group(5)), plugin_M=float(m.group(6)),
                                 plugin_K50=float(m.group(7)), oracle=float(m.group(8)))
    return out


def parse_recovery(lam="0.9 w=0.3", alpha=0.05):
    out = {}
    on = False
    pat = re.compile(rf"alpha={alpha}\s+n=\s*(\d+): P\(R_hat=R\) nominal ([\d.]+) plugin_M ([\d.]+) plugin_K50 ([\d.]+) oracle ([\d.]+)")
    for line in TXT:
        if line.startswith("lam="):
            on = line.startswith(f"lam={lam}:")
        if on:
            m = pat.search(line)
            if m:
                out[int(m.group(1))] = dict(nominal=float(m.group(2)), plugin=float(m.group(3)), oracle=float(m.group(5)))
    return out


def main():
    size = parse_size()
    rec = parse_recovery()
    fig, axes = plt.subplots(1, 3, figsize=(fs.W_FULL, 2.3), gridspec_kw={"wspace": 0.5, "width_ratios": [1, 1.1, 1]})

    ax = axes[0]
    lam = np.linspace(0, 0.995, 400)
    W = (1 + lam ** 2) / (1 - lam ** 2)
    ax.plot(lam, 1 / W, color=fs.BLUE)
    for l in (0.9, 0.97, 0.99):
        w = [v["W"] for (a, n), v in size.items() if a == l][0]
        assert abs(w - (1 + l * l) / (1 - l * l)) < 0.05 * w, (l, w)
        ax.plot([l], [1 / w], "o", color=fs.BLUE, ms=3.6)
    ax.set_yscale("log")
    ax.set_xlabel("AR(1) coefficient $\\lambda$ of both series")
    ax.set_ylabel(r"$n_{\mathrm{eff}}/n$")
    ax.text(0.02, 0.012, "dots: simulated cases", fontsize=6.3, color=fs.INK2)
    fs.panel_label(ax, "(a)", dx=-0.3)

    ax = axes[1]
    lams = [0.9, 0.97, 0.99]
    pick = {l: max(n for (a, n) in size if a == l) for l in lams}
    x = np.arange(len(lams))
    w = 0.26
    series = (("nominal", fs.ORANGE, "nominal $n$"), ("plugin_K50", fs.BLUE, "plug-in, $K=50$"),
              ("plugin_M", fs.AQUA, r"plug-in, $K_n\propto\sqrt{n}$"))
    for k, (key, col, lab) in enumerate(series):
        ys = [size[(l, pick[l])][key] for l in lams]
        ax.bar(x + (k - 1) * w, ys, w * 0.9, color=col, label=lab)
        for xi, yv in zip(x + (k - 1) * w, ys):
            if yv > 0.08:
                ax.text(xi, yv + 0.015, f"{yv:.2f}", ha="center", fontsize=6, va="bottom")
    ax.axhline(0.05, color=fs.INK2, lw=0.8, ls=(0, (3, 2)), label="target 0.05")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}" for l in lams])
    ax.set_xlabel("AR(1) coefficient $\\lambda$")
    ax.set_ylabel("false-positive rate of one pair")
    ax.set_ylim(0, 1.5)
    ax.legend(loc="upper left", fontsize=5.8, handlelength=0.9, borderaxespad=0.1, labelspacing=0.2)
    ax.grid(axis="x", visible=False)
    fs.panel_label(ax, "(b)", dx=-0.3)

    ax = axes[2]
    ns = sorted(rec)
    for key, col, mk, lab in (("nominal", fs.ORANGE, "o", "nominal $n$"), ("plugin", fs.BLUE, "s", "plug-in"),
                              ("oracle", fs.INK2, "^", "known $W$")):
        ax.plot(ns, [rec[n][key] for n in ns], mk + "-", color=col, ms=3.4, lw=1.0, label=lab)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels(["2000", "8000", "32 000"], fontsize=6.5)
    ax.minorticks_off()
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("sample size $n$")
    ax.set_ylabel(r"$P(\hat R=R)$: relatives set recovered")
    ax.legend(loc="center right", fontsize=6.2, borderaxespad=0.1, labelspacing=0.2)
    fs.panel_label(ax, "(c)", dx=-0.3)
    fig.savefig(os.path.join(HERE, "fig_finite.pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("size picks", pick, "recovery n", ns)


if __name__ == "__main__":
    main()
