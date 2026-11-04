"""Post hoc (not pre-registered, no verdicts): which WT relative pairs the primary test detects, and the
estimated relative counts versus the true ones.  Usage: python3 -I explore3.py <this_dir> <wt_dir>"""
import sys, os, itertools, json
import numpy as np
HERE, WT = sys.argv[1:3]; sys.argv = [sys.argv[0], HERE, WT, WT]
sys.path.insert(0, HERE); import cc_graph as G, test3 as T
nodes, edges = G.system("wt"); p = len(nodes); Rel = G.related(p, edges); E = set(edges)
units = [T.std(T.blockdiff(T.load(os.path.join(WT, f"actuators_random_walk_{k}.csv"), nodes)[0], 10)) for k in range(2, 17)]
det = np.zeros((p, p)); rs = []; rho = []
for X in units:
    R, A, NE, _ = T.stats(X, 20); D = T.declare(R, NE, range(p)); det += D; rs.append(R); rho.append(1 + D.sum(1))
det /= len(units); Rm = np.median(np.array(rs), 0)
out = {"relative_pairs": [], "rho_true": {nodes[v]: int(Rel[v].sum()) for v in range(p)},
       "rho_hat_mean": {nodes[v]: float(np.mean([r[v] for r in rho])) for v in range(p)}}
for a, b in itertools.combinations(range(p), 2):
    if Rel[a, b]:
        kind = "edge" if (a, b) in E or (b, a) in E else "trek"
        out["relative_pairs"].append([nodes[a], nodes[b], kind, round(float(det[a, b]), 2), round(float(Rm[a, b]), 3)])
out["nonrel_declared_ge_0.2"] = [[nodes[a], nodes[b], round(float(det[a, b]), 2), round(float(Rm[a, b]), 3)]
                                 for a, b in itertools.combinations(range(p), 2) if not Rel[a, b] and det[a, b] >= 0.2]
json.dump(out, open(os.path.join(HERE, "explore3.json"), "w"), indent=1)
print("rho true vs mean estimated:"); [print(f"  {v:18s} {out['rho_true'][v]:3d} {out['rho_hat_mean'][v]:6.2f}") for v in nodes]
print("edges: detection rate, median r"); [print("  ", r) for r in out["relative_pairs"] if r[2] == "edge"]
tr = [r for r in out["relative_pairs"] if r[2] == "trek"]
print(f"trek (non-adjacent relative) pairs: {len(tr)}, mean detection {np.mean([r[3] for r in tr]):.3f}")
print("non-relatives declared in >= 20% of runs:", out["nonrel_declared_ge_0.2"])
