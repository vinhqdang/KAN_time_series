"""Builds the LaTeX tables of the extension paper from ../experimental_results (run before compiling main.tex)."""
import os, glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
ER = os.path.join(os.path.dirname(__file__), "..", "experimental_results"); OUT = os.path.dirname(__file__)
def w(name, s): open(os.path.join(OUT, name), "w").write(s)

# T1: AB vs base, fresh seeds
d = pd.read_csv(f"{ER}/x1_final_raw.csv"); f = d[d.part == "fresh"]
s = "\\begin{tabular}{lcccccc}\n\\toprule\n & \\multicolumn{2}{c}{$d{=}6$} & \\multicolumn{2}{c}{$d{=}10$} & \\multicolumn{2}{c}{$d{=}20$} \\\\\n Method & AUROC & AUPRC & AUROC & AUPRC & AUROC & AUPRC \\\\ \\midrule\n"
for m in ["SPADE", "SPADE-AB"]:
    cells = []
    for dd in (6, 10, 20):
        g = f[(f.d == dd) & (f.method == m)]; cells += [f"{g.auroc.mean():.3f}", f"{g.auprc.mean():.3f}"]
    s += f"{m} & " + " & ".join(cells) + " \\\\\n"
s += "\\bottomrule\n\\end{tabular}\n"; w("tab_ab.tex", s)
q = f.pivot_table(index=["d", "seed"], columns="method", values="auroc"); x = q["SPADE-AB"] - q["SPADE"]
w("ab_stats.tex", f"{x.mean():+.3f} AUROC, with {int((x>0).sum())} wins, {int((x==0).sum())} ties and {len(x)-int((x>0).sum())-int((x==0).sum())} losses over {len(x)} runs (Wilcoxon $p={wilcoxon(x[x!=0]).pvalue:.4f}$)")

# T2: Sachs
sc = pd.read_csv(f"{ER}/sachs_raw.csv"); sc = sc[(sc.status == "ok") & (sc.part == "primary")]
g = sc.groupby("method").agg(a=("auroc", "mean"), asd=("auroc", "std"), p=("auprc", "mean"), h=("hits", "mean"), shd=("shd", "mean")).sort_values("a", ascending=False)
s = "\\begin{tabular}{lcccc}\n\\toprule\nMethod & AUROC & AUPRC & top-18 hits & SHD \\\\ \\midrule\n"
for m, r in g.iterrows(): s += f"{m.replace('SPADE-adaptive_boot','SPADE-AB')} & {r.a:.3f}$\\pm${r.asd:.3f} & {r.p:.3f} & {r.h:.1f} & {r.shd:.1f} \\\\\n"
w("tab_sachs.tex", s + "\\bottomrule\n\\end{tabular}\n")

# T3: fidelity
t = pd.read_csv(f"{ER}/x3_fidelity_raw.csv"); t["nl"] = t.linear_oracle_r2 < 0.5
s = "\\begin{tabular}{llcccc}\n\\toprule\n & & \\multicolumn{2}{c}{found edges (all)} & \\multicolumn{2}{c}{found, strongly non-linear} \\\\\n$d$ & Method & $R^2$ & class agr. & $R^2$ & ($n$) \\\\ \\midrule\n"
for dd in (6, 10):
    for m in ["MLP-edge", "SPADE", "SPADE-2stage"]:
        a = t[(t.d == dd) & (t.method == m) & t.found]; b = a[a.nl]
        s += f"{dd} & {m} & {a.r2.mean():.2f} & {a.cls_ok.mean():.2f} & {b.r2.mean():.2f} & ({len(b)}) \\\\\n"
w("tab_fidelity.tex", s + "\\bottomrule\n\\end{tabular}\n")

# T4: scale
rows = [pd.read_csv(p) for dd in (50, 100, 200) for p in sorted(glob.glob(f"{ER}/x4_final_d{dd}*.csv"))]
x4 = pd.concat(rows); s = "\\begin{tabular}{llccc}\n\\toprule\n$d$ & Method & AUROC & AUPRC & seeds \\\\ \\midrule\n"
for dd in sorted(x4.d.unique()):
    for m, gg in x4[x4.d == dd].groupby("method"):
        s += f"{dd} & {m} & {gg.auroc.mean():.3f}$\\pm${gg.auroc.std():.3f} & {gg.auprc.mean():.3f} & {len(gg)} \\\\\n"
w("tab_scale.tex", s.replace("SPADE (lg=0.01)", "SPADE ($\\lambda_g{=}0.01$)").replace("SPADE-scaled (lg=c/d,K=15)", "SPADE-scaled") + "\\bottomrule\n\\end{tabular}\n")
print("tables written")
