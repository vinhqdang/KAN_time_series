"""Graph-only predictions for PLAN4 (no data read): per experiment, the theory value and every competitor on
the full node set and on the pre-registered subgraph samples; counts of edge instances where theory and a
competitor disagree.  Output: predict4_results.txt"""
import sys, numpy as np
from graphs4 import *

def summarise(name, nodes, edges, score_edges, inputs, subsets):
    p = len(nodes); R = related(p, edges)
    tot = {k: 0 for k in PREDICTORS}; n_inst = 0; vals = {k: [] for k in PREDICTORS}
    for S in subsets:
        indS, pr, _ = edge_predictions(S, p, edges, R, inputs, score_edges)
        n_inst += len(indS)
        for k in PREDICTORS:
            vals[k] += pr[k]
            tot[k] += sum(1 for a, b in zip(pr["theory"], pr[k]) if a != b)
    th = np.array(vals["theory"])
    print(f"  {name}: {len(subsets)} node sets, {n_inst} edge instances; theory values: "
          f"1:{np.sum(th==1)} 1/2:{np.sum(th==.5)} 0:{np.sum(th==0)}")
    print("    disagreements with theory: " + ", ".join(f"{k}={tot[k]}" for k in PREDICTORS[1:]))

def full_table(nodes, edges, score_edges, inputs):
    p = len(nodes); R = related(p, edges)
    indS, pr, sc = edge_predictions(range(p), p, edges, R, inputs, score_edges)
    print("    edge | " + " ".join(k[:10] for k in PREDICTORS))
    for i, (a, b) in enumerate(indS):
        print(f"    {nodes[a]}->{nodes[b]} | " + " ".join(f"{pr[k][i]:.1f}" for k in PREDICTORS))
    print("    rho_theory: " + ", ".join(f"{nodes[v]}={sc['theory'][v]}" for v in range(p)))

if __name__ == "__main__":
    print("PLAN4 graph-only predictions (predict4.py)\n")
    for exp in ["citris_1", "buchholz_1_obs", "scm_1_reference", "scm_2_reference", "scm_4_reference",
                "scm_5_reference"]:
        for lin in ([True, False] if exp == "citris_1" else [False]):
            nodes, edges, edesign, inputs = lt_system(exp, linear=lin)
            tag = exp + (" [G_lin]" if lin else " [G_D]")
            p = len(nodes)
            print(f"== {tag}: {p} nodes, {len(edges)} relatives-graph edges, {len(edesign)} scored design edges")
            full_table(nodes, edges, edesign, inputs)
            summarise("full set", nodes, edges, edesign, inputs, [tuple(range(p))])
            seeds = {"citris_1": 9300, "buchholz_1_obs": 9400, "scm_1_reference": 9500, "scm_2_reference": 9600,
                     "scm_4_reference": 9700, "scm_5_reference": 9800}[exp]
            nunits = {"citris_1": 10, "buchholz_1_obs": 10, "scm_1_reference": 5, "scm_2_reference": 10,
                      "scm_4_reference": 10, "scm_5_reference": 10}[exp]
            subs = [S for j in range(nunits) for d in (5, 8)
                    for S in sample_subgraphs(p, edesign, d, 10, 10 * (seeds + j) + d)]
            summarise("subgraphs 5,8", nodes, edges, edesign, inputs, subs)
    nodes, edges, _, inputs = wt_regime_system(); p = len(nodes)
    print(f"== WT regime_jumps (hatch constant): {p} nodes, {len(edges)} edges")
    full_table(nodes, edges, edges, inputs)
    summarise("full set", nodes, edges, edges, inputs, [tuple(range(p))])
    subs = [S for j in range(64) for d in (5, 8) for S in sample_subgraphs(p, edges, d, 10, 10 * (9200 + j) + d)]
    summarise("subgraphs 5,8", nodes, edges, edges, inputs, subs)
