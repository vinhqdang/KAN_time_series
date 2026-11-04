"""Ground-truth summary graphs for PLAN4 (E1, E2), the THEORY.md Section-6 prediction rule, and the competitor
predictors.  Graph sources (read, not estimated):
  * LT standard / camera edges: causalchamber 0.2.8 ground_truth/main.py (MIT, Causal Chamber GmbH).
  * WT standard edges: same file.
  * Edges among the actuators: the experiment generators (lt_crl_benchmark_v1/generators/citris_1.py,
    .../buchholz_1.py with the sempler convention W[i,j] != 0 <=> i -> j (sempler 0.2.13 lganm.py: X = W^T X + e),
    lt_camera_v1/generators/scm_{1,2,4,5}.py with A[i,j] = 1 <=> i -> j (Z = W^T Z + D e)).
Only random variables of an experiment are nodes; experimenter-set constants are dropped.  Every node carries a
self-loop (THEORY.md setting); self-loops are never scored."""
import numpy as np

LT_SENS_EDGES = [(c, s) for s in ["ir_1", "ir_2", "ir_3", "vis_1", "vis_2", "vis_3", "current"]
                 for c in ["red", "green", "blue"]] + [
    ("pol_1", "ir_3"), ("pol_2", "ir_3"), ("pol_1", "vis_3"), ("pol_2", "vis_3"),
    ("pol_1", "angle_1"), ("pol_2", "angle_2")]
LT_SENSORS = ["ir_1", "vis_1", "ir_2", "vis_2", "ir_3", "vis_3", "current", "angle_1", "angle_2"]
MALUS = [("pol_1", "ir_3"), ("pol_2", "ir_3"), ("pol_1", "vis_3"), ("pol_2", "vis_3")]

WT_EDGES = [("load_in","rpm_in"),("load_in","rpm_out"),("load_in","current_in"),("load_in","current_out"),
 ("load_out","rpm_in"),("load_out","rpm_out"),("load_out","current_out"),("load_out","current_in"),
 ("hatch","rpm_in"),("hatch","rpm_out"),("load_in","pressure_intake"),("hatch","pressure_intake"),
 ("load_out","pressure_intake"),("load_in","pressure_upwind"),("hatch","pressure_upwind"),
 ("load_out","pressure_upwind"),("load_in","pressure_downwind"),("hatch","pressure_downwind"),
 ("load_out","pressure_downwind"),("pot_1","signal_1"),("pot_1","signal_2"),("pot_2","signal_2"),
 ("pot_1","mic"),("load_in","mic"),("load_out","mic"),("hatch","mic")]
WT_SENSORS = ["current_in","current_out","rpm_in","rpm_out","pressure_upwind","pressure_downwind",
              "pressure_ambient","pressure_intake","mic","signal_1","signal_2"]

ACT = {
    # citris_1: red RW; green <- red; blue <- green; pol_1 RW; pol_2 <- pol_1, and sign(blue > red)
    "citris_1": [("red", "green"), ("green", "blue"), ("pol_1", "pol_2"), ("red", "pol_2"), ("blue", "pol_2")],
    # buchholz_1 (sempler, W[i,j] != 0 => i -> j): green->red, green->pol_2, blue->green, blue->pol_2, pol_1->red
    "buchholz_1_obs": [("green", "red"), ("green", "pol_2"), ("blue", "green"), ("blue", "pol_2"), ("pol_1", "red")],
    # scm_1 / scm_2 (A[i,j] = 1 => i -> j): red->green, red->pol_1, green->blue, pol_2->green, pol_2->blue
    "scm_1_reference": [("red", "green"), ("red", "pol_1"), ("green", "blue"), ("pol_2", "green"), ("pol_2", "blue")],
    "scm_2_reference": [("red", "green"), ("red", "pol_1"), ("green", "blue"), ("pol_2", "green"), ("pol_2", "blue")],
    "scm_4_reference": [("red", "green"), ("green", "blue")],      # chain R -> G -> B, polarisers fixed
    "scm_5_reference": [("red", "green"), ("blue", "green")],      # collider R -> G <- B, polarisers fixed
}
ACT_INPUTS = {"citris_1": ["red", "green", "blue", "pol_1", "pol_2"],
              "buchholz_1_obs": ["red", "green", "blue", "pol_1", "pol_2"],
              "scm_1_reference": ["red", "green", "blue", "pol_1", "pol_2"],
              "scm_2_reference": ["red", "green", "blue", "pol_1", "pol_2"],
              "scm_4_reference": ["red", "green", "blue"], "scm_5_reference": ["red", "green", "blue"]}
# edges with identically zero linear cross-covariance at every lag in citris_1 (PLAN4, Lemma L1)
CITRIS_LIN_REMOVED = [("red", "pol_2"), ("blue", "pol_2")] + MALUS

def lt_system(exp, linear=False):
    inputs = ACT_INPUTS[exp]
    E = ACT[exp] + [(a, b) for a, b in LT_SENS_EDGES if a in inputs]
    if linear and exp == "citris_1":
        E = [e for e in E if e not in CITRIS_LIN_REMOVED]
    sens = [s for s in LT_SENSORS if any(b == s for a, b in ACT[exp] + LT_SENS_EDGES if a in inputs)]
    nodes = inputs + sens
    idx = {v: i for i, v in enumerate(nodes)}
    # keep design-graph sensor list even if linear graph isolates nothing
    edges_design = [(idx[a], idx[b]) for a, b in ACT[exp] + LT_SENS_EDGES if a in idx and b in idx]
    edges = [(idx[a], idx[b]) for a, b in E if a in idx and b in idx]
    return nodes, edges, edges_design, set(idx[v] for v in inputs)

def wt_regime_system():
    inputs = ["load_in", "load_out", "pot_1", "pot_2"]            # hatch constant within a regime
    nodes = inputs + WT_SENSORS
    idx = {v: i for i, v in enumerate(nodes)}
    edges = [(idx[a], idx[b]) for a, b in WT_EDGES if a in idx and b in idx]
    return nodes, edges, edges, set(idx[v] for v in inputs)

def ancestors(p, edges):
    A = np.eye(p, dtype=bool)
    for a, b in edges: A[b, a] = True
    for _ in range(p):
        A = A | ((A.astype(int) @ A.astype(int)) > 0)
    return A                                      # A[v, u]: u is an ancestor of v (incl. v)

def related(p, edges):
    A = ancestors(p, edges).astype(int)
    return (A @ A.T) > 0

def depth(p, edges):
    """longest directed path (number of edges) from any node to v, graph acyclic apart from self-loops"""
    d = np.zeros(p, int)
    for _ in range(p):
        for a, b in edges: d[b] = max(d[b], d[a] + 1)
    return d

def cmp(a, b):
    return 1.0 if a < b else (0.5 if a == b else 0.0)

def node_scores(S, p, edges, R, inputs):
    """per-node scores on node set S: theory rho (Section-6 rule) and the competitors."""
    S = list(S); Sset = set(S)
    A = ancestors(p, edges)
    indS = [(a, b) for a, b in edges if a in Sset and b in Sset]
    sc = {
        "theory": {v: int(sum(R[v, w] for w in S)) for v in S},
        "sensor_indicator": {v: 0 if v in inputs else 1 for v in S},
        "n_anc_full": {v: int(A[v].sum()) for v in S},
        "n_anc_S": {v: int(sum(A[v, w] for w in S)) for v in S},
        "indeg_full": {v: sum(1 for a, b in edges if b == v) for v in S},
        "indeg_S": {v: sum(1 for a, b in indS if b == v) for v in S},
        "depth_full": dict(zip(range(p), depth(p, edges))),
        "depth_S": None,
        "all_one": None,
    }
    loc = {v: i for i, v in enumerate(S)}
    dS = depth(len(S), [(loc[a], loc[b]) for a, b in indS])
    sc["depth_S"] = {v: int(dS[loc[v]]) for v in S}
    return sc, indS

PREDICTORS = ["theory", "sensor_indicator", "n_anc_full", "n_anc_S", "indeg_full", "indeg_S",
              "depth_full", "depth_S", "all_one"]

def edge_predictions(S, p, edges, R, inputs, score_edges=None):
    """edges: graph from which relatives (R) and all competitor scores are computed; score_edges: the true causal
    edges whose induced copies in S are scored (default: edges)."""
    sc, _ = node_scores(S, p, edges, R, inputs)
    Sset = set(S)
    indS = [(a, b) for a, b in (score_edges if score_edges is not None else edges) if a in Sset and b in Sset]
    out = {}
    for k in PREDICTORS:
        if k == "all_one":
            out[k] = [1.0] * len(indS)
        else:
            out[k] = [cmp(sc[k][a], sc[k][b]) for a, b in indS]
    return indS, out, sc

def sample_subgraphs(p, edges, d, k, seed):
    rng = np.random.default_rng(seed); out = []; seen = set(); E = set(edges)
    while len(out) < k:
        S = tuple(sorted(rng.choice(p, d, replace=False).tolist()))
        if S in seen: continue
        if any((a, b) in E for a in S for b in S):
            seen.add(S); out.append(S)
    return out
