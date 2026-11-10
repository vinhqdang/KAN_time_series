"""Q4 (PLAN.md section 6): run- and station-level inference for the real-data tests, from COMMITTED derived results (no new data).
Part (a) B3 run-level counts; part (b) CausalRivers station-sharing graph, station-disjoint subsets and station bootstrap.
Headline horizon for the river claims, named here before the analysis: 12 h (the 24 h columns are the same records, reported as a robustness row only).
E1 block bootstrap and the K-sensitivity check need raw series and are done by q4_e1_k.py when the data are available (see README).
  python q4_runlevel.py > results/analysis_Q4_runlevel.txt
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B3 = json.load(open(os.path.join(HERE, "..", "b3", "results", "results_b3.json")))
R2 = json.load(open(os.path.join(HERE, "..", "empirical2", "results2.json")))
SG = json.load(open(os.path.join(HERE, "..", "empirical2", "subgraphs_8000.json")))
HEADLINE = "12h"; OTHER = "24h"
print("Q4 run- and station-level inference (derived results only; descriptive, no nominal error rates).\n")

# ------------------------------------------------------------------------------------------------ (a) B3
print("== (a) B3: one row per independent run or set; a run 'agrees' if the theorem has the lower loss on a majority of the units where it disagrees with the competitor and its mean delta is negative ==")
runs = []
for key, label in (("H1_per_experiment_ar_1_uniform_ref", "H1 ar_1_uniform_ref"), ("H1_per_experiment_ar_1_uniform_pol_1_90", "H1 ar_1_uniform_pol_1_90"),
                   ("H1_per_experiment_ar_1_uniform_pol_12_30", "H1 ar_1_uniform_pol_12_30")):
    t = B3[key]["primary_Dstar"]["tests"]["const_half=wc_rule"]
    runs.append((label, t))
for key, label in (("per_set_scm_5_red", "H2 scm_5_red"), ("per_set_scm_5_blue", "H2 scm_5_blue")):
    t = B3[key]["primary_Dstar"]["tests"]
    cand = [v for k, v in t.items() if isinstance(v, dict) and v.get("testable") and "const_half" in k]
    runs.append((label, cand[0] if cand else None))
rows = []
for label, t in runs:
    if t is None:
        print(f"  {label:28s} no testable const_half/wc_rule comparison in the stored results"); continue
    u, b = t["units_with_disagreement"], t["units_theory_better"]
    ok = bool(b > u / 2 and t["delta"] < 0)
    rows.append((label, ok))
    print(f"  {label:28s} units with disagreement {u:3d}, theorem better in {b:3d}; delta {t['delta']:+.3f}  -> run agrees: {ok}")
h1 = [ok for l, ok in rows if l.startswith("H1")]; h2 = [ok for l, ok in rows if l.startswith("H2")]
for nm, v in (("H1", h1), ("H2", h2)):
    if v:
        print(f"  {nm}: {sum(v)} of {len(v)} independent runs agree; smallest attainable exact one-sided sign-test p = 2^-{len(v)} = {2.0 ** -len(v):.3f} (cannot reach 0.05 with fewer than 5 runs)")
print("  The segment-level p-values of the stored B3 analysis (30 units from 3 runs, 20 from 2 sets) are not run-level evidence: units within a run are serially dependent.\n")

# ------------------------------------------------------------------------------------------------ (b) rivers
nodes = [tuple(s["nodes"]) for s in SG]; reg = [s["region"] for s in SG]
stations = sorted({(r, n) for r, ns in zip(reg, nodes) for n in ns})
slots = sum(len(ns) for ns in nodes)
from collections import Counter
use = Counter((r, n) for r, ns in zip(reg, nodes) for n in ns)
print("  [the 900-slot / 457-station / reuse-8 counts quoted in the paper belong to ../empirical/subgraphs_7000.json (the earlier subgraph set); the P4-P6 results analysed here (empirical2/results2.json) use subgraphs_8000.json]")
print(f"== (b) CausalRivers: {len(SG)} subgraphs, {slots} station slots, {len(stations)} distinct stations (station = (region, node id)), max reuse {max(use.values())}; ==")
sets = [set((r, n) for n in ns) for r, ns in zip(reg, nodes)]
adj = [[j for j in range(len(sets)) if j != i and sets[i] & sets[j]] for i in range(len(sets))]
seen = set(); comps = []
for i in range(len(sets)):
    if i in seen: continue
    comp = {i}; stack = [i]
    while stack:
        k = stack.pop()
        for j in adj[k]:
            if j not in comp: comp.add(j); stack.append(j)
    seen |= comp; comps.append(len(comp))
print(f"  station-sharing graph: {sum(1 for a in adj if a)} subgraphs share a station with another; connected components {len(comps)}; sizes (sorted) {sorted(comps, reverse=True)[:10]}")

def acc_stat(per, metric, kind):
    """per: list of per-subgraph results; returns (numerator, denominator) per subgraph for the pooled statistic."""
    out = []
    for e in per:
        if kind == "P5":
            a = e["acc"][metric]
            a = a if isinstance(a, list) else ([a] if a is not None else [])
            out.append((float(np.sum(a)), float(len(a))))
        else:   # P4: share of non-relative pairs declared dependent
            nr = np.array(e["nonrel"], bool); d = np.array(e["dec"][metric], bool)
            out.append((float(d[nr].sum()), float(nr.sum())))
    return np.array(out)

def pooled(arr, w=None):
    w = np.ones(len(arr)) if w is None else w
    den = (w * arr[:, 1]).sum(); return (w * arr[:, 0]).sum() / den if den > 0 else np.nan

rng0 = np.random.default_rng(0)
for h in (HEADLINE, OTHER):
    per = R2["per_subgraph"][h]
    print(f"\n  -- horizon {h} ({'headline' if h == HEADLINE else 'robustness row, same records'}) --")
    for kind, metrics in (("P5", ("cond_neff", "cond_nominal", "marg_nominal")), ("P4", ("marg_nominal", "marg_neff", "cond_nominal", "cond_neff"))):
        for metric in metrics:
            arr = acc_stat(per, metric, kind)
            ok = arr[:, 1] > 0
            full = pooled(arr[ok])
            # (a) station-disjoint subsets: 200 random orderings, greedy maximal subsets, bootstrap over the subgraphs of the subset
            meds, lows, sizes, passes = [], [], [], 0
            for _ in range(200):
                order = rng0.permutation(len(per)); used = set(); chosen = []
                for i in order:
                    if sets[i] & used: continue
                    used |= sets[i]; chosen.append(i)
                a = arr[[i for i in chosen if arr[i, 1] > 0]]
                if len(a) < 5: continue
                est = pooled(a); r = np.random.default_rng(len(chosen)); idx = r.integers(0, len(a), (1000, len(a)))
                bs = (a[idx, 0].sum(1)) / (a[idx, 1].sum(1)); lo = float(np.quantile(bs, 0.025))
                meds.append(est); lows.append(lo); sizes.append(len(chosen)); passes += int(lo > 0.5) if kind == "P5" else 0
            # (b) station bootstrap
            st = {s: k for k, s in enumerate(stations)}; memb = [np.array([st[s] for s in ss]) for ss in sets]
            bs = []
            rb = np.random.default_rng(1)
            for _ in range(2000):
                cnt = np.bincount(rb.integers(0, len(stations), len(stations)), minlength=len(stations))
                w = np.array([cnt[m].sum() / len(m) for m in memb]); bs.append(pooled(arr[ok], w[ok]))
            lo, hi = np.nanquantile(bs, [0.025, 0.975])
            extra = f"; share of orderings with CI lower bound > 0.5: {passes / max(len(meds), 1):.2f}" if kind == "P5" else ""
            print(f"    {kind} {metric:13s} all-subgraph pooled {full:.3f} | station-disjoint subsets: median {np.median(meds):.3f}, median CI lower {np.median(lows):.3f} (median subset size {int(np.median(sizes))} subgraphs, {len(meds)} orderings){extra} | station bootstrap 95% CI [{lo:.3f}, {hi:.3f}]")
