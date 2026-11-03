"""Build LaTeX tables and figures for the note from stage0 result files.
Reads ../stage0/results.md and ../stage0/gates.md only (no recomputation)."""
import re, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
S0 = HERE.parent / "stage0"
OUT = HERE / "generated"; OUT.mkdir(exist_ok=True)
res = (S0 / "results.md").read_text()
gates = (S0 / "gates.md").read_text()

def section(text, header):
    i = text.index(header); j = text.find("\n## ", i + 5)
    return text[i:j if j > 0 else None]

def table(sec):
    rows = []
    for ln in sec.splitlines():
        if ln.startswith("|") and not re.match(r"^\|[-| ]+\|$", ln):
            rows.append([c.strip() for c in ln.strip("|").split("|")])
    return rows[1:]

NAMES = {"var": "VAR (lag-only)", "corr": r"$|$correlation$|$", "pcmci+": "PCMCI+",
         "varlingam": "VAR-LiNGAM", "dynotears": "DYNOTEARS (own)",
         "notears_svar2": "NOTEARS-SVAR (ref.)", "var_sortnregress": "var-sortnregress",
         "r2_sortnregress": "R$^2$-sortnregress", "random": "random"}
ORDER = ["var", "corr", "pcmci+", "varlingam", "dynotears", "notears_svar2",
         "var_sortnregress", "r2_sortnregress", "random"]

prim = table(section(res, "## Primary table"))
P = {(r[0], r[1]): r for r in prim}
def ci(s):  # "0.598 [0.575, 0.621]" -> latex
    m = re.match(r"([\d.]+) \[([\d.]+), ([\d.]+)\]", s); return f"{m[1]} [{m[2]}, {m[3]}]"

def tex_main(res_label):
    L = []
    for m in ORDER:
        r = P[(res_label, m)]
        acc = "0.5 (by def.)" if m in ("var", "corr") else ci(r[4])
        L.append(f"{NAMES[m]} & {ci(r[2])} & {acc} \\\\")
        if m == "corr": L.append("\\midrule")
        if m == "notears_svar2": L.append("\\midrule")
    return L
paired = {(r[0], r[1]): r for r in table(section(res, "## Paired comparison"))}

with open(OUT / "table_main.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lcc@{}}\n\\toprule\n")
    for k, lab in (("12h", "12 h"), ("24h", "24 h")):
        n = P[(k, "pcmci+")][6]
        f.write(f"\\multicolumn{{3}}{{@{{}}l}}{{\\textbf{{{lab}}} ({n.replace(' / ', ' contemporaneous edges in ')} subgraphs)}}\\\\\n")
        f.write("Method & Directed AUROC [95\\% CI] & Orientation acc. [95\\% CI] \\\\\n\\midrule\n")
        f.write("\n".join(tex_main(k)) + "\n")
        f.write("\\midrule\n" if k == "12h" else "")
    f.write("\\bottomrule\n\\end{tabular}\n")

# paired table
with open(OUT / "table_paired.tex", "w") as f:
    f.write("\\begin{tabular}{@{}llccc@{}}\n\\toprule\nRes. & Method & $\\Delta$AUROC vs VAR & raw $p$ & Holm $p$ (9) \\\\\n\\midrule\n")
    for k in ("6h", "12h", "24h"):
        for m in ("pcmci+", "varlingam", "dynotears"):
            r = paired[(k, m)]
            f.write(f"{k} & {NAMES[m]} & {r[2]} & {r[3]} & {r[4]} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# raw vs standardised
raw = {(r[0], r[1]): r for r in table(section(res, "## Secondary analysis: raw scale"))}
with open(OUT / "table_raw.tex", "w") as f:
    f.write("\\begin{tabular}{@{}llcc@{}}\n\\toprule\nRes. & Method & Acc. raw [95\\% CI] & Acc. standardised \\\\\n\\midrule\n")
    for k in ("12h", "24h"):
        for m in ("pcmci+", "varlingam", "dynotears", "var_sortnregress", "r2_sortnregress"):
            r = raw[(k, m)]
            f.write(f"{k} & {NAMES[m]} & {ci(r[3])} & {r[5]} \\\\\n")
        f.write("\\midrule\n" if k == "12h" else "")
    f.write("\\bottomrule\n\\end{tabular}\n")

# gates G2 / G1 / sibling tables parsed from gates.md
def gsec(h): return section(gates, h)
g1 = table(gsec("## G1"))
with open(OUT / "table_g1.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lccc@{}}\n\\toprule\nGroup & Edges & G1-consistent & $\\tau_e=0$ \\\\\n\\midrule\n")
    for r in g1:
        if r[0] in ("ALL", "bavaria (both sizes)", "east_germany (both sizes)"):
            nm = {"ALL": "All", "bavaria (both sizes)": "Bavaria", "east_germany (both sizes)": "East Germany"}[r[0]]
            f.write(f"{nm} & {r[1]} & {r[2]} & {r[3]} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
g2 = [r for r in table(gsec("## G2")) if r[0] in ("6h", "12h", "24h")][:3]
with open(OUT / "table_g2.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nRes. & Contemp. frac. & ES $d{=}10$ & BA $d{=}10$ & G2 \\\\\n\\midrule\n")
    for r in g2:
        f.write(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[6]} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
sib = [r for r in table(gsec("## Diagnostic")) if r[0] in ("6h", "12h", "24h")]
with open(OUT / "table_sib.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lccc@{}}\n\\toprule\nRes. & Siblings & Adjacent (true edge) & Other pairs \\\\\n\\midrule\n")
    for r in sib:
        f.write(f"{r[0]} & {r[1].split()[0]} & {r[2].split()[0]} & {r[3].split()[0]} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# figure 1: orientation accuracy with CI at 12h and 24h
def parse_ci(s):
    m = re.match(r"([\d.]+) \[([\d.]+), ([\d.]+)\]", s); return tuple(map(float, m.groups()))
meths = ["pcmci+", "varlingam", "dynotears", "notears_svar2", "r2_sortnregress", "var_sortnregress"]
cols = {"pcmci+": "#1b6ca8", "varlingam": "#d1495b", "dynotears": "#edae49",
        "notears_svar2": "#8d8d8d", "r2_sortnregress": "#2a9d8f", "var_sortnregress": "#6a4c93"}
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
for ax, k in zip(axes, ("12h", "24h")):
    for i, m in enumerate(meths):
        a, lo, hi = parse_ci(P[(k, m)][4])
        ax.errorbar(i, a, yerr=[[a - lo], [hi - a]], fmt="o", color=cols[m], capsize=3)
    ax.axhline(0.5, color="k", lw=0.7, ls=":"); ax.axhline(0.6, color="k", lw=0.7, ls="--")
    ax.set_xticks(range(len(meths)))
    ax.set_xticklabels(["PCMCI+", "VAR-\nLiNGAM", "DYNO-\nTEARS", "NOTEARS\nSVAR", "R2-\nsort", "var-\nsort"], fontsize=7)
    ax.set_title(f"{k} (standardised)", fontsize=9); ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("Orientation accuracy"); axes[1].text(5.4, 0.605, "H1 threshold 0.60", fontsize=6.5, ha="right", va="bottom")
fig.tight_layout(); fig.savefig(OUT / "fig_orient.pdf"); plt.close(fig)

# figure 2: AUROC at 24h, all scores
fig, ax = plt.subplots(figsize=(3.3, 2.7))
am = ["var", "corr", "pcmci+", "varlingam", "dynotears", "notears_svar2", "r2_sortnregress", "random"]
lab = ["VAR", "|corr|", "PCMCI+", "VAR-\nLiNGAM", "DYNO-\nTEARS", "NOTEARS\nSVAR", "R2-\nsort", "random"]
for i, m in enumerate(am):
    a, lo, hi = parse_ci(P[("24h", m)][2])
    ax.errorbar(i, a, yerr=[[a - lo], [hi - a]], fmt="o", color="#1b6ca8" if m not in ("corr", "r2_sortnregress", "random") else "#d1495b", capsize=3)
ax.axhline(0.5, color="k", lw=0.7, ls=":"); ax.set_xticks(range(len(am))); ax.set_xticklabels(lab, fontsize=6)
ax.set_ylabel("Directed AUROC, 24 h"); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig_auroc.pdf"); plt.close(fig)
print("ok")
