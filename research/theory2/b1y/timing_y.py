"""Step-0 timing of every learner on VALIDATION seeds (9810 for Q2 cells, 9820 for Q5 cells); no metric is computed or reported.
  OMP_NUM_THREADS=1 python timing_y.py FAMILY P T OUT.json
"""
import sys, json, time
import numpy as np
import common_y as cy
import seeds as sd_
import methods_y as my
import methods_y2 as m2
from methods_x import golem as golem_fn

fam, p, T, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
seed = 9810
rng_g = sd_.graph_rng(fam, p, seed); S = cy.summary_graph(fam, p, rng_g); A, sd = cy.weights_cap(S, rng_g, 0.8)
rng = sd_.data_rng(fam, p, seed, T, cy.ARMS.index("G1"), 1000)
X = cy.simulate_arm("G1", A, sd, T, 1000, rng); d = X.shape[1]
perm = rng.permutation(d); Xp = X[:, perm]; Xs = (Xp - Xp.mean(0)) / Xp.std(0)
res = dict(family=fam, p=p, T=T, d=d)
def tm(name, fn):
    t = time.time()
    try:
        fn(); res[name] = round(time.time() - t, 2)
    except Exception as e:                       # noqa: BLE001
        res[name] = "ERR " + repr(e)[:120]
    print(name, res[name], flush=True)
import flopsearch
tm("FLOP", lambda: flopsearch.flop(np.ascontiguousarray(Xs), 2.0, restarts=10))
tm("DAGMA", lambda: my.dagma_hp(Xp))
tm("NOTEARS", lambda: my.notears_hp(Xp))
tm("GOLEM", lambda: golem_fn(Xp))
import os
if d <= 60:
    if not os.environ.get("SKIP_GES"):
        tm("GES", lambda: my.ges_cpdag(Xp))
    tm("GRaSP", lambda: my.grasp_cpdag(Xp))
    tm("BOSS", lambda: my.boss_cpdag(Xp))
tm("DirectLiNGAM", lambda: my.directlingam_dag(Xp))
tm("PCMCI+", lambda: __import__("methods_x").pcmciplus_window(X, p, T))
tm("VARLiNGAM_pooled", lambda: m2.varlingam_pooled(X, p, T))
tm("DYNOTEARS", lambda: m2.dynotears(X, p, T))
json.dump(res, open(out, "w"))
