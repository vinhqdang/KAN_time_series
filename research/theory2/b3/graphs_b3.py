"""B3 graphs and predictors (graph-only).  Imports the unchanged E1 machinery from ../icml_exp/graphs4.py
(ancestors, related, depth, cmp, node_scores, sample_subgraphs) and adds the competitors required by roadmap B3:
  const_half      every edge 1/2
  wc_rule         weak-component-size rule on the relatives graph: 1/2 iff u and v lie in the same weak
                  component of the full graph, else compare |C(u) & S| with |C(v) & S|
  theory_GD       the THEORY.md Section-6 / Thm 1.9 rule applied to the published design graph G_D
Graph sources (read, not estimated):
  * light-tunnel sensor edges: causalchamber 0.2.8 ground_truth/main.py (_edges_lt_standard; the camera
    configuration adds only the image node 'im', which is not a column of the csv files).
  * lt_camera_walks_v1 ar_1_uniform_*: generator generators/actuators_ar.py: R, G, B, pol_1, pol_2 are five
    independent AR(1) series (separate seeds); no edge among the actuators.
  * lt_camera_v1 scm_4_* / scm_5_*: generators/scm_{4,5}.py.  scm_4: A = R->G->B; scm_5: A = R->G<-B; the
    polarisers are fixed at 0 (so angle_1, angle_2 have no random ancestor and are not nodes).  An experiment
    scm_k_<target> removes the edges into <target> (W[:, idx] = 0)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "icml_exp"))
import graphs4 as G4

COL = ["red", "green", "blue"]
COLSENS = ["ir_1", "vis_1", "ir_2", "vis_2", "ir_3", "vis_3", "current"]
COL_EDGES = [(c, s) for s in COLSENS for c in COL]
MALUS = [("pol_1", "ir_3"), ("pol_2", "ir_3"), ("pol_1", "vis_3"), ("pol_2", "vis_3")]
ANGLE = [("pol_1", "angle_1"), ("pol_2", "angle_2")]

SCM_ACT = {"scm_4": [("red", "green"), ("green", "blue")], "scm_5": [("red", "green"), ("blue", "green")]}

def ar1_system():
    """lt_camera_walks_v1 ar_1_uniform_{ref,pol_1_90,pol_12_30}: 14 nodes.
    Returns nodes, G_lin edges (relatives graph of the primary analysis: G_D minus the 4 Malus edges, PLAN4 /
    THEORY_ADD Lemma B1-B3 by the symmetry of the polariser design), G_D edges (scored edges), input set."""
    inputs = COL + ["pol_1", "pol_2"]
    nodes = inputs + ["ir_1", "vis_1", "ir_2", "vis_2", "ir_3", "vis_3", "current", "angle_1", "angle_2"]
    idx = {v: i for i, v in enumerate(nodes)}
    ED = [(idx[a], idx[b]) for a, b in COL_EDGES + MALUS + ANGLE]
    EL = [(idx[a], idx[b]) for a, b in COL_EDGES + ANGLE]
    return nodes, EL, ED, set(idx[v] for v in inputs)

def scm_system(exp):
    """lt_camera_v1 scm_4_* / scm_5_* (snapshot): 10 nodes (R, G, B and the 7 colour sensors)."""
    base, target = exp.split("_")[0] + "_" + exp.split("_")[1], exp.split("_", 2)[2]
    act = [(a, b) for a, b in SCM_ACT[base] if b != target]          # target 'reference' removes nothing
    nodes = COL + COLSENS
    idx = {v: i for i, v in enumerate(nodes)}
    E = [(idx[a], idx[b]) for a, b in act + COL_EDGES]
    return nodes, E, E, set(idx[v] for v in COL)

def weak_components(p, edges):
    lab = list(range(p))
    def f(x):
        while lab[x] != x:
            lab[x] = lab[lab[x]]; x = lab[x]
        return x
    for a, b in edges:
        ra, rb = f(a), f(b)
        if ra != rb: lab[ra] = rb
    return [f(i) for i in range(p)]

def n_source_sccs(p, edges):
    """number of source SCCs per weak component (DAG part; self-loops irrelevant)"""
    A = G4.ancestors(p, edges)
    comp = weak_components(p, edges)
    srcs = [v for v in range(p) if A[v].sum() == 1 or all(A[u, v] for u in range(p) if A[v, u])]
    out = {}
    for v in srcs: out.setdefault(comp[v], []).append(v)
    return comp, out

# subgraph seed bases (seed = 10 * (base + j) + d, j = unit index within the analysis); fixed in PLAN.md
SEEDBASE = {"ar_1": 9900, "scm_5_red": 9940, "scm_5_blue": 9950, "scm_5_green": 9960, "scm_4_green": 9970,
            "scm_4_blue": 9980, "scm_4_red": 9990}

PREDICTORS = G4.PREDICTORS + ["const_half", "wc_rule", "theory_GD"]

def edge_predictions(S, p, rel_edges, R, inputs, score_edges, R_D, comp):
    """all predictors for the induced copies of score_edges in S.  rel_edges/R: relatives graph used by the theory
    and (as in E1) by the eight naive competitors; R_D: relatives in the design graph; comp: weak components of
    rel_edges."""
    indS, out, sc = G4.edge_predictions(list(S), p, rel_edges, R, inputs, score_edges)
    S = list(S)
    rhoD = {v: int(sum(R_D[v, w] for w in S)) for v in S}
    csz = {v: sum(1 for w in S if comp[w] == comp[v]) for v in S}
    out["const_half"] = [0.5] * len(indS)
    out["wc_rule"] = [0.5 if comp[a] == comp[b] else G4.cmp(csz[a], csz[b]) for a, b in indS]
    out["theory_GD"] = [G4.cmp(rhoD[a], rhoD[b]) for a, b in indS]
    return indS, out

sample_subgraphs = G4.sample_subgraphs
related = G4.related
cmp = G4.cmp
