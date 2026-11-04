"""Ground-truth summary graphs of the Causal Chambers (Gamella, Peters, Buhlmann 2025), restricted to the
variables that are random in a given experiment.  Edges copied from causalchamber 0.2.8,
causalchamber/ground_truth/main.py (MIT licence, Causal Chamber GmbH).  Experimenter-set parameters that are
constant in an experiment are constants, not random variables, and are dropped (an edge from a constant
carries no dependence).  Every node is given a self-loop (THEORY.md setting); self-loops are not scored."""
import itertools
import numpy as np

WT_EDGES = [("load_in","rpm_in"),("res_in","rpm_in"),("load_in","rpm_out"),("load_in","current_in"),
 ("load_in","current_out"),("load_out","rpm_in"),("load_out","rpm_out"),("res_out","rpm_out"),
 ("load_out","current_out"),("load_out","current_in"),("hatch","rpm_in"),("hatch","rpm_out"),
 ("load_in","pressure_intake"),("hatch","pressure_intake"),("load_out","pressure_intake"),
 ("osr_intake","pressure_intake"),("load_in","pressure_upwind"),("hatch","pressure_upwind"),
 ("load_out","pressure_upwind"),("osr_upwind","pressure_upwind"),("load_in","pressure_downwind"),
 ("hatch","pressure_downwind"),("load_out","pressure_downwind"),("osr_downwind","pressure_downwind"),
 ("osr_ambient","pressure_ambient"),("osr_in","current_in"),("v_in","current_in"),("osr_out","current_out"),
 ("v_out","current_out"),("pot_1","signal_1"),("osr_1","signal_1"),("v_1","signal_1"),("pot_1","signal_2"),
 ("pot_2","signal_2"),("osr_2","signal_2"),("v_2","signal_2"),("pot_1","mic"),("load_in","mic"),
 ("load_out","mic"),("hatch","mic"),("osr_mic","mic"),("v_mic","mic")]
WT_SENSORS = ["current_in","current_out","rpm_in","rpm_out","pressure_upwind","pressure_downwind",
              "pressure_ambient","pressure_intake","mic","signal_1","signal_2"]

LT_EDGES = [(c, s) for s in ["ir_1","ir_2","ir_3","vis_1","vis_2","vis_3","current"] for c in ["red","green","blue"]] + [
 ("pol_1","ir_3"),("pol_2","ir_3"),("pol_1","vis_3"),("pol_2","vis_3"),("pol_1","angle_1"),("pol_2","angle_2"),
 ("v_angle_1","angle_1"),("osr_angle_1","angle_1"),("v_angle_2","angle_2"),("osr_angle_2","angle_2"),
 ("v_c","current"),("osr_c","current"),
 ("l_11","ir_1"),("l_12","ir_1"),("l_11","vis_1"),("l_12","vis_1"),("t_ir_1","ir_1"),("diode_ir_1","ir_1"),
 ("t_vis_1","vis_1"),("diode_vis_1","vis_1"),("l_21","ir_2"),("l_22","ir_2"),("l_21","vis_2"),("l_22","vis_2"),
 ("t_ir_2","ir_2"),("diode_ir_2","ir_2"),("t_vis_2","vis_2"),("diode_vis_2","vis_2"),("l_31","ir_3"),
 ("l_32","ir_3"),("l_31","vis_3"),("l_32","vis_3"),("t_ir_3","ir_3"),("diode_ir_3","ir_3"),("t_vis_3","vis_3"),
 ("diode_vis_3","vis_3")]
LT_SENSORS = ["ir_1","vis_1","ir_2","vis_2","ir_3","vis_3","current","angle_1","angle_2"]

# experimenter-set inputs that take more than one value (inspect_design.py, Stage A)
WT_WALK_INPUTS = ["hatch","pot_1","pot_2","load_in","load_out"]
LT_WHITE_INPUTS = ["red","green","blue","pol_1","pol_2","l_11","l_12","l_21","l_22","l_31","l_32"]

def system(chamber):
    if chamber == "wt":
        nodes = WT_WALK_INPUTS + WT_SENSORS; E = WT_EDGES
    else:
        nodes = LT_WHITE_INPUTS + LT_SENSORS; E = LT_EDGES
    idx = {v: i for i, v in enumerate(nodes)}
    edges = [(idx[a], idx[b]) for a, b in E if a in idx and b in idx]
    return nodes, edges

def ancestors(p, edges):
    A = np.eye(p, dtype=bool)
    for a, b in edges: A[b, a] = True            # A[v, u]: u is an ancestor of v
    for _ in range(p):
        A = A | ((A.astype(int) @ A.astype(int)) > 0)
    return A

def related(p, edges):
    A = ancestors(p, edges).astype(int)
    return (A @ A.T) > 0                         # common ancestor (incl. self)

def predicted_edges(S, edges, R):
    """Section-6 rule of THEORY.md: rho(v) = #{w in S : v, w related in the full graph}; value of induced edge
    (u, v) is 1 / 1/2 / 0 for rho(u) <, =, > rho(v)."""
    S = list(S); Sset = set(S)
    rho = {v: int(sum(R[v, w] for w in S)) for v in S}
    ind = [(a, b) for a, b in edges if a in Sset and b in Sset]
    return ind, [1.0 if rho[a] < rho[b] else (0.5 if rho[a] == rho[b] else 0.0) for a, b in ind]

def sample_subgraphs(p, edges, d, k, seed):
    """k distinct d-subsets drawn uniformly among those with >= 1 induced edge (rejection sampling)."""
    rng = np.random.default_rng(seed); out = []; seen = set(); E = set(edges)
    while len(out) < k:
        S = tuple(sorted(rng.choice(p, d, replace=False).tolist()))
        if S in seen: continue
        if any((a, b) in E for a in S for b in S):
            seen.add(S); out.append(S)
    return out
