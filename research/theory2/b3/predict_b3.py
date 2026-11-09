"""Graph-only predictions for B3 (no data are read).  For each candidate system: number of source SCCs per weak
component, and on the node sets that the plan fixes (full set + subgraphs, seeds as in PLAN.md), how often each
competitor disagrees with the theorem, overall and on the primary set D* = {theory != wc_rule}.
Usage: python3 -I predict_b3.py  > results/predict_b3_results.txt"""
import os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graphs_b3 as G
from graphs_b3 import SEEDBASE

def nodesets(p, score_edges, j, seedbase, sizes=(5, 8), k=10):
    sets = [("full", tuple(range(p)))]
    for d in sizes:
        for S in G.sample_subgraphs(p, score_edges, d, k, 10 * (seedbase + j) + d):
            sets.append((f"sub{d}", S))
    return sets

def summarise(name, nodes, rel_edges, score_edges, inputs, nunits, seedbase):
    p = len(nodes); R = G.related(p, rel_edges); RD = G.related(p, score_edges)
    comp, srcs = G.n_source_sccs(p, rel_edges)
    compD, srcsD = G.n_source_sccs(p, score_edges)
    print(f"== {name}: {p} nodes, {len(score_edges)} scored edges, {len(rel_edges)} relatives-graph edges")
    print("   source nodes per weak component (relatives graph):",
          {tuple(nodes[v] for v in range(p) if comp[v] == c)[:3] + ("...",): [nodes[v] for v in s] for c, s in srcs.items()})
    print("   source nodes per weak component (G_D):", [[nodes[v] for v in s] for s in srcsD.values()])
    rows = []
    for j in range(nunits):
        for kind, S in nodesets(p, score_edges, j, seedbase):
            indS, pr = G.edge_predictions(S, p, rel_edges, R, inputs, score_edges, RD, comp)
            for e, (a, b) in enumerate(indS):
                rows.append({"unit": j, "kind": kind, "edge": (nodes[a], nodes[b]), **{k: pr[k][e] for k in G.PREDICTORS}})
    n = len(rows); Dstar = [r for r in rows if r["theory"] != r["wc_rule"]]
    print(f"   instances: {n} ({sum(r['kind']=='full' for r in rows)} full-set); D* (theory != wc_rule): {len(Dstar)}"
          f" = {len(Dstar)/n:.3f}; units with D* instances: {len(set(r['unit'] for r in Dstar))}/{nunits}")
    th = np.array([r["theory"] for r in rows])
    print(f"   theory predicted 1/0/half: {np.sum(th==1)}/{np.sum(th==0)}/{np.sum(th==.5)}")
    print("   D* theory values:", {v: sum(1 for r in Dstar if r['theory'] == v) for v in (0, .5, 1)},
          " wc values:", {v: sum(1 for r in Dstar if r['wc_rule'] == v) for v in (0, .5, 1)})
    print("   competitor disagreement with theory: all instances | within D*")
    for c in G.PREDICTORS[1:]:
        a = sum(r["theory"] != r[c] for r in rows); b = sum(r["theory"] != r[c] for r in Dstar)
        ub = len(set(r["unit"] for r in Dstar if r["theory"] != r[c]))
        print(f"     {c:18s} {a:6d} ({a/n:.3f}) | {b:6d} (units {ub})")
    # within-edge switch: colour-source edges with / without a second independent source in S
    sw = [r for r in rows if r["theory"] == .5 and r["edge"][0] in G.COL and r["edge"][1] in G.COLSENS]
    print(f"   colour edges predicted 1/2 by the theory (no second source in S): {len(sw)};"
          f" predicted 1 (second source present): {sum(1 for r in Dstar if r['edge'][0] in G.COL)}")
    print("   theory-1/2 instances where sensor_indicator predicts 1:",
          sum(1 for r in rows if r["theory"] == .5 and r["sensor_indicator"] == 1))
    return rows

if __name__ == "__main__":
    nodes, EL, ED, inp = G.ar1_system()
    summarise("ar_1_uniform_* (3 experiments x 10 units), G_lin", nodes, EL, ED, inp, 30, SEEDBASE["ar_1"])
    summarise("ar_1_uniform_* as if G_D were the relatives graph", nodes, ED, ED, inp, 30, SEEDBASE["ar_1"])
    for exp, sb in list(SEEDBASE.items())[1:]:
        nodes, E, E2, inp = G.scm_system(exp)
        summarise(exp + " (10 units)", nodes, E, E2, inp, 10, sb)
