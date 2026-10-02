"""Builds LaTeX tables for the non-financial real-data study from the raw result CSVs."""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
ROOT = os.path.join(os.path.dirname(__file__), "..")
ER = os.path.join(ROOT, "experimental_results"); FG = os.path.join(ROOT, "manuscript", "figures")
ORDER = ["SPADE", "DAGMA-nonlinear", "NOTEARS-linear", "SCORE", "NoGAM", "GraN-DAG", "NOTEARS-MLP", "GOLEM", "DAGMA-linear"]


def agg(path):
    d = pd.read_csv(path); ok = d[d.status == "ok"]
    p = ok.pivot(index="resample", columns="method", values="bk_auroc")
    rows = {}
    for m in ORDER:
        if m not in p.columns: continue
        g = ok[ok.method == m]
        pv = ""
        if m != "SPADE" and "SPADE" in p.columns:
            df = (p["SPADE"] - p[m]).dropna()
            pv = f"{wilcoxon(df).pvalue:.3f}" if len(df) >= 5 and (df != 0).any() else "--"
        rows[m] = (g.bk_auroc.mean(), g.bk_auroc.std(), g.n_forbidden.mean(), g.n_edges.mean(), g.n_positive.mean(),
                   g.n_pos_total.iloc[0] if "n_pos_total" in g else 7, g.time_s.mean(), pv, len(g))
    return rows


def table(rows, fh, show_time=True):
    best = max(v[0] for v in rows.values())
    fh.write("\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{l" + "c" * (6 if show_time else 5) + "}\n\\toprule\n")
    fh.write("Method & BK-AUROC & forbidden / edges & recovered & " + ("time (s) & " if show_time else "") + "$p$ vs.\\ SPADE & $n$ \\\\ \\midrule\n")
    for m, (mu, sd, fo, ed, po, pt, tm, pv, n) in rows.items():
        a = f"{mu:.3f}$\\pm${sd:.3f}"; a = f"\\textbf{{{a}}}" if mu == best else a
        nm = f"\\textbf{{{m}}}" if m == "SPADE" else m
        fh.write(f"{nm} & {a} & {fo:.1f} / {ed:.1f} & {po:.1f} / {pt} & " + (f"{tm:.1f} & " if show_time else "") + f"{pv or '--'} & {n} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}}\n")


if __name__ == "__main__":
    targets = {"tab_bike_daily.tex": "realworld_bike_raw.csv",
               "tab_bike_hourly.tex": "realworld_bike_hourly_primary_raw.csv",
               "tab_beijing.tex": "realworld_beijing_primary_raw.csv"}
    for out, src in targets.items():
        if os.path.exists(os.path.join(ER, src)):
            with open(os.path.join(FG, out), "w") as fh: table(agg(os.path.join(ER, src)), fh, show_time=(out == "tab_bike_daily.tex"))
            print("wrote", out)


def scale_table():
    """Full-series scale check (SPADE vs. the cheap baselines), from experimental_results/realworld_large_scale_raw.csv."""
    d = pd.read_csv(os.path.join(ER, "realworld_large_scale_raw.csv"))
    meths = [m for m in ORDER if m in set(d.method)]
    names = {"bike_hourly": "Hourly bike sharing ($n{=}17{,}303$)", "beijing": "Beijing PM2.5 ($n{=}41{,}543$)"}
    with open(os.path.join(FG, "tab_large_scale.tex"), "w") as fh:
        fh.write("\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{l" + "c" * len(meths) + "}\n\\toprule\nDataset & " + " & ".join(meths) + " \\\\ \\midrule\n")
        for ds in ["bike_hourly", "beijing"]:
            g = d[d.dataset == ds]; best = max(g[g.method == m].bk_auroc.mean() for m in meths)
            cells = []
            for m in meths:
                x = g[g.method == m].bk_auroc; c = f"{x.mean():.3f}$\\pm${x.std():.3f}"
                cells.append(f"\\textbf{{{c}}}" if x.mean() == best else c)
            fh.write(names[ds] + " & " + " & ".join(cells) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}}\n")
    print("wrote tab_large_scale.tex")


if __name__ == "__main__" and os.path.exists(os.path.join(ER, "realworld_large_scale_raw.csv")):
    scale_table()
