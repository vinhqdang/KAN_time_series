"""B'1 analysis (PLAN.md §2): HX1, HX2, HX3, descriptive levels, PCMCI+ reference, sanity XS1/XS2.
Run: /usr/bin/python3 analyse_b1x.py results/b1x_test.csv.gz results
"""
import sys, os, json, itertools
import numpy as np
import pandas as pd
from scipy import stats

BOOT = 10000
FAMS = ["ring", "chain", "intree", "ER2"]
CO = ["DAGMA_raw", "DAGMA_std", "NOTEARS_raw", "NOTEARS_std", "GOLEMEV_raw", "GOLEMEV_std", "GOLEMNV_raw", "GOLEMNV_std"]
RAW_MAIN = ["DAGMA_raw", "NOTEARS_raw", "GOLEMEV_raw"]
BASE = ["M1", "M2", "M3", "std_M3", "M5", "EMPTY"]
NAME = dict(M1="rel-SnR", M2="R2-SnR", M3="Var-SnR", std_M3="Var-SnR(std)", M5="FLOP", EMPTY="empty", PCMCI="PCMCI+ (labels)")
for v in CO:
    NAME[v] = v.replace("_", " ").replace("GOLEMEV", "GOLEM-EV").replace("GOLEMNV", "GOLEM-NV")
METRICS = [("shdn", True), ("sidn", True), ("aoa", False)]      # (metric, lower is better)
out_lines = []; OUT = {}


def P(*a):
    s = " ".join(str(x) for x in a); out_lines.append(s); print(s, flush=True)


def holm(pv):
    pv = np.asarray(pv, float); k = len(pv); o = np.argsort(pv); adj = np.empty(k); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (k - r) * pv[i])); adj[i] = run
    return adj


def wil(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2 or np.all(x == 0):
        return 1.0
    return float(stats.wilcoxon(x, zero_method="wilcox").pvalue)


def boot(x, seed=0):
    """x: graph-level values (one per graph) -> mean and percentile CI (= graph-cluster bootstrap)."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return dict(mean=float("nan"), ci=[float("nan")] * 2, n=0)
    m = np.random.default_rng(seed).choice(x, (BOOT, len(x))).mean(1)
    return dict(mean=float(x.mean()), ci=[float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], n=int(len(x)))


def f3(r):
    return f"{r['mean']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"


def main():
    df = pd.read_csv(sys.argv[1]); out = sys.argv[2]
    for m in CO + BASE + ["PCMCI"]:
        if f"{m}_shd" in df.columns:
            df[f"{m}_shdn"] = df[f"{m}_shd"] / df.n_true
            df[f"{m}_shdx1n"] = df[f"{m}_shdx1"] / df.n_true
    key = ["family", "p", "seed", "T"]
    a = df[df.gen == "G1"].set_index(key).sort_index(); b = df[df.gen == "G2"].set_index(key).loc[a.index]
    P(f"B'1 test: {len(df)} data sets; graphs {a.reset_index().groupby(['family','p','seed']).ngroups}")

    def graph_level(series):          # mean over T per graph
        return series.groupby(level=["family", "p", "seed"]).mean()

    # ---------------- XS2: failed fits, cyclic outputs
    P("\n== XS2 / output checks ==")
    for v in CO:
        if f"{v}_shd" not in df.columns:
            continue
        sub = df[df[f"{v}_shd"].notna() | (df.p == 5) | (~v.startswith("GOLEM"))]
        exp = df[(df.p == 5)] if v.startswith("GOLEM") else df
        miss = int(exp[f"{v}_shd"].isna().sum()); cyc = float(1 - exp[f"{v}_dag"].mean())
        P(f"  {NAME[v]:16s}: data sets {len(exp)}, missing {miss}, cyclic output share {cyc:.3f}, SID missing "
          f"{int(exp[f'{v}_sidn'].isna().sum())}")
    OUT["XS2"] = {v: dict(missing=int((df[df.p == 5] if v.startswith('GOLEM') else df)[f"{v}_shd"].isna().sum()))
                  for v in CO if f"{v}_shd" in df.columns}

    # ---------------- XS1: rel-SnR gain replicates
    d = graph_level(a.M1_oa - b.M1_oa); r = boot(d.values)
    P(f"\n== XS1: rel-SnR OA gain G1 - G2 (graph level) {f3(r)}, Wilcoxon p = {wil(d.values):.1e} -> "
      f"{'PASS' if r['ci'][0] > 0 else 'FAIL'}")
    OUT["XS1"] = dict(r, p=wil(d.values), passed=bool(r["ci"][0] > 0))
    for fam in FAMS:
        dd = d[d.index.get_level_values(0) == fam]
        P(f"   {fam:7s} {f3(boot(dd.values))}")

    # ---------------- HX1
    P("\n== HX1: G1 - G2 difference per CO variant and metric (graph level; Wilcoxon two-sided, Holm over 24) ==")
    rows = []
    for v in CO:
        for k, low in METRICS:
            c = f"{v}_{k}"
            dd = graph_level(a[c] - b[c]).dropna()
            r = boot(dd.values)
            rows.append(dict(v=v, k=k, low=low, mean=r["mean"], ci=r["ci"], n=r["n"], p=wil(dd.values)))
    H1 = pd.DataFrame(rows); H1["p_holm"] = holm(H1.p.values)
    H1["favourable"] = np.where(H1.low, H1["mean"] < 0, H1["mean"] > 0)
    for _, r in H1.iterrows():
        sig = r.p_holm < 0.05
        P(f"  {NAME[r.v]:16s} {r.k:5s} d = {r['mean']:+.3f} [{r.ci[0]:+.3f}, {r.ci[1]:+.3f}] (n={r.n}) Holm {r.p_holm:.1e} "
          f"{'SIG ' + ('favourable' if r.favourable else 'unfavourable') if sig else ''}")
    hx1 = bool(((H1.v.isin(RAW_MAIN)) & (H1.k.isin(["aoa", "shdn"])) & (H1.p_holm < 0.05) & H1.favourable).any())
    P(f"  VERDICT HX1: {'SUPPORTED' if hx1 else 'NOT SUPPORTED (no start effect detected on CO methods)'}")
    OUT["HX1"] = dict(rows=H1.to_dict(orient="records"), supported=hx1)

    # ---------------- HX2
    P("\n== HX2: rank relative to FLOP. Delta = metric_v - metric_FLOP per family (graph level), Wilcoxon per generator, "
      "Holm over 64 per metric; rank change = opposite signs, both Holm < 0.05 ==")
    OUT["HX2"] = {}
    for k, low in METRICS:
        rows = []
        for v in CO:
            for fam in FAMS:
                for g, F in (("G1", a), ("G2", b)):
                    Fs = F[F.index.get_level_values(0) == fam]
                    dd = graph_level(Fs[f"{v}_{k}"] - Fs[f"M5_{k}"]).dropna()
                    if len(dd) == 0:
                        continue
                    rows.append(dict(v=v, fam=fam, g=g, mean=float(dd.mean()), p=wil(dd.values), n=len(dd)))
        T2 = pd.DataFrame(rows); T2["p_holm"] = holm(T2.p.values)
        piv = T2.pivot_table(index=["v", "fam"], columns="g", values=["mean", "p_holm"])
        changes = piv[(np.sign(piv[("mean", "G1")]) != np.sign(piv[("mean", "G2")])) &
                      (piv[("p_holm", "G1")] < 0.05) & (piv[("p_holm", "G2")] < 0.05)]
        bymean = piv[np.sign(piv[("mean", "G1")]) != np.sign(piv[("mean", "G2")])]
        P(f"  [{k}] tests {len(T2)}; sign changes by mean {len(bymean)}; significant rank changes {len(changes)}")
        for idx, r in bymean.iterrows():
            P(f"     by mean: {NAME[idx[0]]:16s} {idx[1]:7s} G1 {r[('mean','G1')]:+.3f} (Holm {r[('p_holm','G1')]:.2g}) "
              f"G2 {r[('mean','G2')]:+.3f} (Holm {r[('p_holm','G2')]:.2g})")
        # interactions vs FLOP, vs Var-SnR, vs rel-SnR: per family and pooled
        inter = {}
        for ref in ("M5", "M3", "M1"):
            for v in CO:
                for fam in FAMS + ["ALL"]:
                    sa = a if fam == "ALL" else a[a.index.get_level_values(0) == fam]
                    sb = b.loc[sa.index]
                    I = graph_level((sa[f"{v}_{k}"] - sa[f"{ref}_{k}"]) - (sb[f"{v}_{k}"] - sb[f"{ref}_{k}"])).dropna()
                    if len(I) == 0:
                        continue
                    dG1 = float(graph_level(sa[f"{v}_{k}"] - sa[f"{ref}_{k}"]).mean())
                    dG2 = float(graph_level(sb[f"{v}_{k}"] - sb[f"{ref}_{k}"]).mean())
                    inter[f"{v}|{ref}|{fam}"] = dict(boot(I.values), dG1=dG1, dG2=dG2, p=wil(I.values))
        P(f"  [{k}] interaction I = (v - FLOP)(G1) - (v - FLOP)(G2), graph-cluster CI; columns: " + ", ".join(FAMS + ["ALL"]))
        for v in CO:
            cells = [inter.get(f"{v}|M5|{fam}") for fam in FAMS + ["ALL"]]
            P(f"    {NAME[v]:16s} " + " | ".join("   n/a   " if c is None else
                                                 f"{c['mean']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] (G1 {c['dG1']:+.3f}, G2 {c['dG2']:+.3f})"
                                                 for c in cells))
        OUT["HX2"][k] = dict(tests=T2.to_dict(orient="records"), n_changes=int(len(changes)),
                             changes=[list(i) for i in changes.index], n_bymean=int(len(bymean)), interactions=inter)
        if k == "shdn":
            OUT["HX2"]["supported"] = bool(len(changes) >= 1)
            P(f"  VERDICT HX2 (SHD/|E|): {'SUPPORTED: rank change(s) ' + str([list(i) for i in changes.index]) if len(changes) else 'NOT SUPPORTED (no rank change relative to FLOP)'}")
    # per-cell interactions vs FLOP (descriptive)
    P("\n  per-cell interaction vs FLOP on SHD/|E| (12 graphs per cell; descriptive):")
    cellI = []
    for v in CO:
        for fam in FAMS:
            for p in (5, 10):
                for T in (3, 5):
                    sel = (a.index.get_level_values(0) == fam) & (a.index.get_level_values(1) == p) & (a.index.get_level_values(3) == T)
                    if not sel.any() or a.loc[sel, f"{v}_shdn"].isna().all():
                        continue
                    I = ((a.loc[sel, f"{v}_shdn"] - a.loc[sel, "M5_shdn"]) - (b.loc[sel, f"{v}_shdn"] - b.loc[sel, "M5_shdn"])).values
                    r = boot(I); cellI.append(dict(v=v, fam=fam, p=p, T=T, **r))
    C = pd.DataFrame(cellI)
    nex = int(((C.ci.str[0] > 0) | (C.ci.str[1] < 0)).sum())
    P(f"   {len(C)} cells; CI excludes 0 in {nex} (no multiplicity correction)")
    OUT["HX2_cells_shdn"] = C.to_dict(orient="records")

    # ---------------- HX3
    P("\n== HX3: does standardisation remove the gain? (gain_raw - gain_std, graph level, Holm over 8) ==")
    rows = []
    for m in ("DAGMA", "NOTEARS", "GOLEMEV", "GOLEMNV"):
        for k, low in (("aoa", False), ("shdn", True)):
            gr = graph_level(a[f"{m}_raw_{k}"] - b[f"{m}_raw_{k}"]).dropna()
            gs = graph_level(a[f"{m}_std_{k}"] - b[f"{m}_std_{k}"]).loc[gr.index]
            diff = (gr - gs)
            rows.append(dict(m=m, k=k, low=low, raw=boot(gr.values), std=boot(gs.values), diff=boot(diff.values),
                             p=wil(diff.values)))
    H3 = pd.DataFrame(rows); H3["p_holm"] = holm(H3.p.values)
    res3 = {}
    for _, r in H3.iterrows():
        h1row = H1[(H1.v == f"{r.m}_raw") & (H1.k == r.k)].iloc[0]
        raw_ok = bool(h1row.p_holm < 0.05 and h1row.favourable)
        # favourable direction for the difference of gains: raw gain more favourable than std gain
        diff_sig = r.p_holm < 0.05 and ((r['diff']['mean'] < 0) if r.low else (r['diff']['mean'] > 0))
        std_null = (r["std"]["ci"][0] <= 0 <= r["std"]["ci"][1]) or abs(r["std"]["mean"]) < 0.02
        if not raw_ok:
            verdict = "no gain to remove"
        elif diff_sig and std_null:
            verdict = "removed"
        elif diff_sig:
            verdict = "reduced, not removed"
        else:
            verdict = "not reduced"
        res3[f"{r.m}|{r.k}"] = dict(raw=r.raw, std=r["std"], diff=r["diff"], p_holm=float(r.p_holm), verdict=verdict)
        P(f"  {r.m:8s} {r.k:5s}: gain raw {f3(r.raw)}, std {f3(r['std'])}, raw - std {f3(r['diff'])} Holm {r.p_holm:.1e} -> {verdict}")
    OUT["HX3"] = res3

    # ---------------- levels
    P("\n== Levels: mean over graphs and T (p = 5 and 10 unless noted), G1 / G2 ==")
    meths = ["M5", "M1", "M2", "M3", "std_M3"] + CO + ["EMPTY", "PCMCI"]
    lev = {}
    for k in ("shdn", "sidn", "aoa", "shdx1n"):
        P(f"  [{k}]  " + " | ".join(f"{fam}" for fam in FAMS + ["ALL", "ALL p=5"]))
        for m in meths:
            c = f"{m}_{k}"
            if c not in a.columns:
                continue
            cells = []
            for fam in FAMS + ["ALL", "ALL p=5"]:
                if fam == "ALL":
                    sa, sb = a, b
                elif fam == "ALL p=5":
                    sa = a[a.index.get_level_values(1) == 5]; sb = b.loc[sa.index]
                else:
                    sa = a[a.index.get_level_values(0) == fam]; sb = b.loc[sa.index]
                v1, v2 = float(sa[c].mean()), float(sb[c].mean())
                lev[f"{m}|{k}|{fam}"] = [v1, v2]
                cells.append(f"{v1:.2f}/{v2:.2f}")
            P(f"    {NAME.get(m, m):16s} " + " | ".join(cells))
    OUT["levels"] = lev
    P("\n  best method without labels (lowest mean SHD/|E|) per family, p = 5 (all methods) and p = 10 (no GOLEM):")
    best = {}
    for fam in FAMS:
        for p in (5, 10):
            for g, F in (("G1", a), ("G2", b)):
                Fs = F[(F.index.get_level_values(0) == fam) & (F.index.get_level_values(1) == p)]
                ms = [m for m in ["M5", "M1", "M2", "M3", "std_M3"] + CO if f"{m}_shdn" in Fs and Fs[f"{m}_shdn"].notna().all()]
                vals = {m: Fs[f"{m}_shdn"].mean() for m in ms}
                bm = min(vals, key=vals.get); best[f"{fam}|{p}|{g}"] = [bm, float(vals[bm])]
            P(f"    {fam:7s} p={p:2d}: G1 {NAME[best[f'{fam}|{p}|G1'][0]]} ({best[f'{fam}|{p}|G1'][1]:.2f}), "
              f"G2 {NAME[best[f'{fam}|{p}|G2'][0]]} ({best[f'{fam}|{p}|G2'][1]:.2f})")
    OUT["best"] = best
    # rel vs R2 on chains (B1 mechanism reversal) as interaction
    P("\n  rel-SnR vs R2-SnR on chains, interaction on SHD/|E| and OA (graph-cluster CI):")
    for k in ("shdn", "oa"):
        sa = a[a.index.get_level_values(0) == "chain"]; sb = b.loc[sa.index]
        I = graph_level((sa[f"M1_{k}"] - sa[f"M2_{k}"]) - (sb[f"M1_{k}"] - sb[f"M2_{k}"]))
        r = boot(I.values)
        P(f"    {k}: I = {f3(r)} (G1 {float((sa[f'M1_{k}'] - sa[f'M2_{k}']).mean()):+.3f}, "
          f"G2 {float((sb[f'M1_{k}'] - sb[f'M2_{k}']).mean()):+.3f})")
        OUT[f"chain_rel_vs_r2_{k}"] = r
    P(f"\nmean seconds per data set {df.secs.mean():.1f}; total {df.secs.sum() / 3600:.2f} h")
    open(f"{out}/analysis.txt", "w").write("\n".join(out_lines) + "\n")
    json.dump(OUT, open(f"{out}/analysis.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
