"""Q3 (PLAN.md section 5): post hoc re-analysis of COMMITTED data with the first-slice-excluded metrics (SHD_x1n = shdx1 / n_true; OA_x1).
No new data. Inputs: ../b1/results/b1_test.csv.gz (regime het), ../b1x/results/b1x_test.csv.gz, ../b1x/results/orders_x1.csv.gz (exp B1, cfg LG).
Unit = summary graph (family, p, seed); the horizons T of one graph are averaged first; percentile bootstrap over graphs, 10,000 resamples, RNG seed 0.
  python q3_reanalysis.py > results/analysis_Q3.txt
"""
import os
import numpy as np
import pandas as pd
from analyse_y import boot_mean, NONSC

HERE = os.path.dirname(os.path.abspath(__file__))
b1 = pd.read_csv(os.path.join(HERE, "..", "b1", "results", "b1_test.csv.gz")); b1 = b1[b1.regime == "het"]
bx = pd.read_csv(os.path.join(HERE, "..", "b1x", "results", "b1x_test.csv.gz"))
ox = pd.read_csv(os.path.join(HERE, "..", "b1x", "results", "orders_x1.csv.gz")); ox = ox[(ox.exp == "B1") & (ox.cfg == "LG")]
NAMES = {"M1": "rel-SnR", "M2": "R2-SnR", "M3": "Var-SnR", "M4": "Random-SnR", "M5": "FLOP", "M6": "PC"}


def g_table(df, col, gens=("G1", "G2")):
    return df.groupby(["family", "p", "seed", "gen"])[col].mean().unstack("gen")[list(gens)].dropna()


def fmt(mu, ci):
    return f"{mu:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"


def level(mu, ci):
    return f"{mu:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


print("Q3 post hoc re-analysis of committed data (descriptive; not a verdict). Unit = summary graph, graph-cluster bootstrap 95% CI.\n")
print("== A. B1 (R-het): order accuracy gain G1 - G2, full-window vs first-slice-excluded (edges leaving slice 1 dropped) ==")
for m in ("M1", "M2", "M3"):
    full = g_table(b1, f"{m}_oa"); x1 = g_table(ox, f"{m}_oa_x1")
    for label, fams in (("non-SC pooled", NONSC), ("symER+ring", ["symER", "ring"])):
        for nm, tab in (("full", full), ("x1  ", x1)):
            t = tab[tab.index.get_level_values("family").isin(fams)]
            mu, ci, _ = boot_mean((t["G1"] - t["G2"]).values)
            print(f"  {NAMES[m]:10s} {label:14s} {nm}: G1 {t['G1'].mean():.3f}  G2 {t['G2'].mean():.3f}  gain {fmt(mu, ci)}  ({len(t)} graphs)")
print("\n== B. B1 (R-het): SHD/|E|, full vs first-slice-excluded, by generator (all families, graph-level means over T) ==")
b1["_n"] = b1["n_true"]
for m in ("M1", "M2", "M3", "M4", "M5", "M6"):
    for nm, col in (("full", f"{m}_shd"), ("x1  ", f"{m}_shdx1")):
        d = b1.dropna(subset=[col]).copy(); d["v"] = d[col] / d["n_true"]
        t = g_table(d, "v")
        mu, ci, _ = boot_mean((t["G2"] - t["G1"]).values)
        print(f"  {NAMES[m]:10s} {nm}: G1 {t['G1'].mean():.3f}  G2 {t['G2'].mean():.3f}  G2-G1 {fmt(mu, ci)}  ({len(t)} graphs)")
print("\n== C. B1x learner extension (B1x design, 12 graphs per family-p): SHD/|E| full vs x1 by generator, raw data ==")
for m, nm_ in (("EMPTY", "empty"), ("M1", "rel-SnR"), ("M3", "Var-SnR"), ("M5", "FLOP"), ("DAGMA_raw", "DAGMA"), ("NOTEARS_raw", "NOTEARS"), ("GOLEMEV_raw", "GOLEM-EV"), ("GOLEMNV_raw", "GOLEM-NV")):
    for nm, suffix in (("full", "_shd"), ("x1  ", "_shdx1")):
        col = f"{m}{suffix}"
        if col not in bx.columns:
            continue
        d = bx.dropna(subset=[col]).copy(); d["v"] = d[col] / d["n_true"]
        t = g_table(d, "v")
        mu, ci, _ = boot_mean((t["G2"] - t["G1"]).values)
        print(f"  {nm_:9s} {nm}: G1 {t['G1'].mean():.3f}  G2 {t['G2'].mean():.3f}  G2-G1 {fmt(mu, ci)}  ({len(t)} graphs)")
print("\n== D. Rank change versus FLOP on in-trees (B1x data): [CO - FLOP] SHD/|E| under G1 and G2, full vs x1; interaction = G1 value - G2 value ==")
d_it = bx[bx.family == "intree"]
for m, nm_ in (("DAGMA_raw", "DAGMA"), ("NOTEARS_raw", "NOTEARS"), ("GOLEMEV_raw", "GOLEM-EV"), ("GOLEMNV_raw", "GOLEM-NV")):
    for nm, suffix in (("full", "_shd"), ("x1  ", "_shdx1")):
        a, b = f"{m}{suffix}", f"M5{suffix}"
        d = d_it.dropna(subset=[a, b]).copy(); d["v"] = (d[a] - d[b]) / d["n_true"]
        if not len(d):
            continue
        t = g_table(d, "v"); mu, ci, _ = boot_mean((t["G1"] - t["G2"]).values)
        print(f"  {nm_:9s} {nm}: [CO-FLOP] G1 {t['G1'].mean():+.3f}  G2 {t['G2'].mean():+.3f}  interaction {fmt(mu, ci)}  ({len(t)} graphs)")
print("\n== E. B1 chain, rel-SnR minus R2-SnR SHD/|E|, cell p = 10, T = 3 (the B1 swap cell): full vs x1, by generator ==")
c = b1[(b1.family == "chain") & (b1.p == 10) & (b1["T"] == 3)]
for nm, suffix in (("full", "_shd"), ("x1  ", "_shdx1")):
    for gen in ("G1", "G2"):
        g = c[c.gen == gen]; v = (g[f"M1{suffix}"] - g[f"M2{suffix}"]) / g["n_true"]
        mu, ci, _ = boot_mean(v.values); print(f"  {nm} {gen}: {fmt(mu, ci)}  ({len(g)} graphs)")
